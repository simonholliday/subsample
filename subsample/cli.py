"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, the WAV writer, and the MIDI player. Supports two
input modes and two run modes:

  File input   — pass WAV file paths as positional arguments; each file
                 is processed through the detection pipeline and segments
                 saved to the output directory, then picked up by the
                 instrument library loader.

  Live capture — stream from an audio input device (recorder.enabled: true).

  MIDI player  — listen for MIDI input and play instrument samples
                 (player.enabled: true).

Recorder and player run as threads so they can operate concurrently.
The main thread handles KeyboardInterrupt and coordinates shutdown via
a shared threading.Event.

Press Ctrl+C to stop cleanly.
"""

import argparse
import datetime
import logging
import pathlib
import sys
import threading
import typing

import numpy

import subsample.analysis
import subsample.audio
import subsample.buffer
import subsample.config
import subsample.detector
import subsample.library
import subsample.player
import subsample.recorder
import subsample.similarity
import subsample.trim


_log = logging.getLogger(__name__)

# Maps config bit_depth to the NumPy dtype used for sample storage.
# 24-bit audio is stored as int32 (left-shifted by 8) — see audio.unpack_audio().
_AUDIO_DTYPE: dict[int, numpy.dtype] = {
	16: numpy.dtype(numpy.int16),
	24: numpy.dtype(numpy.int32),
	32: numpy.dtype(numpy.int32),
}


def _parse_args () -> argparse.Namespace:

	"""Parse command-line arguments.

	Returns:
		Namespace with a 'files' attribute containing a (possibly empty)
		list of pathlib.Path objects.
	"""

	parser = argparse.ArgumentParser(
		prog="subsample",
		description="Ambient audio sample recorder and analyser",
	)
	parser.add_argument(
		"files",
		nargs="*",
		type=pathlib.Path,
		metavar="FILE",
		help=(
			"WAV files to process through the detection pipeline before "
			"starting live capture. Segments are written to the configured "
			"output directory and named after the source file "
			"(e.g. recording_1.wav, recording_2.wav, …)."
		),
	)
	return parser.parse_args()


def _process_chunk (
	chunk: numpy.ndarray,
	buf: subsample.buffer.CircularBuffer,
	detector: subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
) -> typing.Optional[numpy.ndarray]:

	"""Feed one audio chunk through the detection pipeline.

	Writes the chunk to the circular buffer, runs the level detector, and
	if a recording is completed, retrieves and trims the segment.

	Args:
		chunk:         One audio chunk, shape (chunk_size, channels).
		buf:           Circular buffer receiving the audio stream.
		detector:      Level detector tracking ambient noise and recording state.
		detection_cfg: Detection settings (SNR threshold, trim params, etc.).

	Returns:
		Trimmed audio segment as a numpy array, or None if no recording completed.
	"""

	buf.write(chunk)

	result = detector.process_chunk(chunk, buf.frames_written)

	if result is None:
		return None

	start_frame, end_frame = result
	pre = detection_cfg.trim_pre_samples
	# Read back `pre` extra frames before the detector's start boundary so
	# trim_silence has raw audio available for its fade-in window.
	# trim_silence then uses the same `pre_samples` value to decide how many
	# of those frames to keep and how wide the S-curve fade should be.
	# The two uses are coordinated, not double-counted.
	segment = buf.read_range(max(0, start_frame - pre), end_frame)

	amplitude_threshold = detector.ambient_rms * (
		10.0 ** (detection_cfg.snr_threshold_db / 20.0)
	)

	trimmed = subsample.trim.trim_silence(
		segment,
		amplitude_threshold,
		pre_samples=detection_cfg.trim_pre_samples,
		post_samples=detection_cfg.trim_post_samples,
	)

	if trimmed.size == 0:
		return None

	return trimmed


def _process_input_files (
	files: list[pathlib.Path],
	cfg: subsample.config.Config,
) -> None:

	"""Process audio files through the detection pipeline, writing segments to disk.

	Each file is read with its native sample rate, bit depth, and channel count.
	Detected segments are written to cfg.output.directory with names derived from
	the source filename (e.g. field_recording_1.wav, field_recording_2.wav, …).
	Files that cannot be read are skipped with a warning.

	Args:
		files: Paths to audio files to process.
		cfg:   Application config (detection settings, output directory).
	"""

	for path in files:
		if not path.exists():
			_log.warning("Input file not found, skipping: %s", path)
			continue

		print(f"Processing {path.name}…")

		try:
			file_info = subsample.audio.read_audio_file(path)
		except (OSError, ValueError) as exc:
			_log.warning("Could not read %s: %s — skipping", path.name, exc)
			continue

		audio_dtype = _AUDIO_DTYPE.get(file_info.bit_depth)
		if audio_dtype is None:
			_log.warning(
				"Unsupported bit depth %d in %s — skipping",
				file_info.bit_depth, path.name,
			)
			continue

		max_frames = file_info.sample_rate * cfg.recorder.buffer.max_seconds
		buf = subsample.buffer.CircularBuffer(max_frames, file_info.channels, dtype=audio_dtype)

		detector = subsample.detector.LevelDetector(
			cfg.detection,
			file_info.sample_rate,
			cfg.recorder.audio.chunk_size,
			max_recording_frames=max_frames,
		)

		analysis_params = subsample.analysis.compute_params(file_info.sample_rate)

		writer = subsample.recorder.WavWriter(cfg, analysis_params, on_complete=None)

		segment_index = 1
		chunk_size = cfg.recorder.audio.chunk_size
		n_frames = file_info.audio.shape[0]

		for offset in range(0, n_frames, chunk_size):
			chunk = file_info.audio[offset : offset + chunk_size]

			trimmed = _process_chunk(chunk, buf, detector, cfg.detection)

			if trimmed is not None:
				writer.enqueue(
					trimmed,
					datetime.datetime.now(),
					filename_base=f"{path.stem}_{segment_index}",
					sample_rate=file_info.sample_rate,
					bit_depth=file_info.bit_depth,
				)
				segment_index += 1

		writer.flush()
		writer.shutdown()

		count = segment_index - 1
		print(f"  → {count} segment(s) written from {path.name}")


def _run_recorder (
	cfg: subsample.config.Config,
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	shutdown_event: threading.Event,
	store_audio: bool,
) -> None:

	"""Set up an audio input device and run the real-time capture loop.

	Streams audio from the configured device (or interactively selected device)
	into a circular buffer. Detected recordings are trimmed, queued for WAV
	output, analyzed, and added to the instrument library. Runs until
	shutdown_event is set.

	Args:
		cfg:                Full application config.
		reference_library:  Loaded reference samples, or None if not configured.
		instrument_library: Instrument sample library to update in real time.
		analysis_params:    Pre-computed FFT params matching cfg.recorder.audio.sample_rate.
		similarity_matrix:  Similarity index to update as new samples arrive, or None.
		shutdown_event:     Set this to stop the capture loop cleanly.
		store_audio:        When True, keep PCM data in SampleRecord for playback.
	"""

	pa = subsample.audio.create_pyaudio()

	try:
		if cfg.recorder.audio.device is not None:
			device_index = subsample.audio.find_device_by_name(pa, cfg.recorder.audio.device)
		else:
			devices = subsample.audio.list_input_devices(pa)
			device_index = subsample.audio.select_device(devices)

		reader = subsample.audio.AudioReader(pa, device_index, cfg.recorder.audio)

	except (ValueError, OSError) as exc:
		print(f"Error opening audio device: {exc}", file=sys.stderr)
		pa.terminate()
		return

	audio_dtype = _AUDIO_DTYPE[cfg.recorder.audio.bit_depth]
	max_frames = cfg.recorder.audio.sample_rate * cfg.recorder.buffer.max_seconds
	buf = subsample.buffer.CircularBuffer(max_frames, cfg.recorder.audio.channels, dtype=audio_dtype)

	detector = subsample.detector.LevelDetector(
		cfg.detection,
		cfg.recorder.audio.sample_rate,
		cfg.recorder.audio.chunk_size,
		max_recording_frames=max_frames,
	)

	writer = subsample.recorder.WavWriter(
		cfg,
		analysis_params,
		on_complete=_make_on_complete(
			reference_library, instrument_library, analysis_params, similarity_matrix, store_audio,
		),
	)

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	try:
		while not shutdown_event.is_set():
			chunk = reader.read(timeout=0.5)

			if chunk is None:
				# Timeout — loop back to check shutdown_event.
				continue

			trimmed = _process_chunk(chunk, buf, detector, cfg.detection)

			if trimmed is not None:
				writer.enqueue(trimmed, datetime.datetime.now())

	finally:
		if reader.overflow_count > 0:
			_log.warning(
				"Audio overflows detected during capture: %d — "
				"recordings may contain discontinuities",
				reader.overflow_count,
			)

		reader.stop()
		pa.terminate()
		writer.shutdown()


def _start_player (
	cfg: subsample.config.Config,
	shutdown_event: threading.Event,
) -> None:

	"""Select a MIDI input device, create a MidiPlayer, and run it.

	Resolves the MIDI device from config (substring match) or prompts the
	user to select one interactively. Runs until shutdown_event is set.

	Args:
		cfg:            Full application config (reads cfg.player.midi_device).
		shutdown_event: Set this to stop the player cleanly.
	"""

	try:
		devices = subsample.player.list_midi_input_devices()

		if cfg.player.midi_device is not None:
			device_name = subsample.player.find_midi_device_by_name(cfg.player.midi_device)
		else:
			device_name = subsample.player.select_midi_device(devices)

	except ValueError as exc:
		print(f"Error opening MIDI device: {exc}", file=sys.stderr)
		return

	player = subsample.player.MidiPlayer(device_name, shutdown_event)
	player.run()


def main () -> None:

	"""Run the ambient audio sampler.

	Processes any input files first (if given on the command line), then
	loads libraries and starts the recorder and/or player as configured.
	Both run as threads; the main thread coordinates shutdown on Ctrl+C.
	"""

	logging.basicConfig(
		level=logging.WARNING,
		format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
		datefmt="%H:%M:%S",
	)
	logging.getLogger("subsample").setLevel(logging.DEBUG)

	args = _parse_args()

	cfg = subsample.config.load_config()

	_print_banner(cfg)

	# Load reference library first — before instrument samples — so that similarity
	# comparisons against reference sounds are available as instrument samples load.
	reference_library: typing.Optional[subsample.library.ReferenceLibrary] = None
	if cfg.reference is not None:
		reference_library = subsample.library.load_reference_library(
			pathlib.Path(cfg.reference.directory)
		)
		print(f"  Reference    : {len(reference_library)} sample(s) loaded from {cfg.reference.directory}")

	# Process any input files through the detection pipeline.
	# Segments are written to the output directory (which is typically the same
	# as the instrument directory), so they are picked up by the instrument loader below.
	if args.files:
		_process_input_files(args.files, cfg)

	# Create instrument library. PCM audio is only needed when the player is
	# active — skipping it saves memory when player is disabled.
	max_instrument_bytes = int(cfg.instrument.max_memory_mb * 1024 * 1024)
	if cfg.instrument.directory is not None:
		instrument_library = subsample.library.load_instrument_library(
			pathlib.Path(cfg.instrument.directory),
			max_instrument_bytes,
			load_audio=cfg.player.enabled,
		)
		print(
			f"  Instruments  : {len(instrument_library)} sample(s) loaded"
			f" from {cfg.instrument.directory}"
		)
	else:
		instrument_library = subsample.library.InstrumentLibrary(max_instrument_bytes)

	analysis_params = subsample.analysis.compute_params(cfg.recorder.audio.sample_rate)

	# Build the similarity matrix now that both libraries are populated.
	# bulk_add() is vectorised (single matrix multiply for N × M scores),
	# so loading hundreds of pre-existing instrument samples is fast.
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix] = None
	if reference_library is not None:
		similarity_matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
		if len(instrument_library) > 0:
			similarity_matrix.bulk_add(instrument_library.samples())
		print(f"  Similarity   : {similarity_matrix}")

	# --- Thread-based orchestration ---
	# Both the recorder and player have blocking loops, so each runs on its own
	# thread. The main thread waits on shutdown_event and forwards Ctrl+C.

	shutdown_event = threading.Event()
	threads: list[threading.Thread] = []

	if cfg.recorder.enabled:
		threads.append(threading.Thread(
			target=_run_recorder,
			args=(
				cfg, reference_library, instrument_library,
				analysis_params, similarity_matrix,
				shutdown_event, cfg.player.enabled,
			),
			name="recorder",
		))

	if cfg.player.enabled:
		threads.append(threading.Thread(
			target=_start_player,
			args=(cfg, shutdown_event),
			name="player",
		))

	if not threads:
		print("Neither recorder nor player is enabled. Nothing to do.")
		return

	for t in threads:
		t.start()

	try:
		# Block the main thread without spinning. Event.wait() releases the GIL
		# and responds to KeyboardInterrupt between intervals.
		while not shutdown_event.is_set():
			shutdown_event.wait(timeout=1.0)

	except KeyboardInterrupt:
		print("\nStopping…")
		shutdown_event.set()

	for t in threads:
		t.join(timeout=10.0)

	print("Done.")


def _print_banner (cfg: subsample.config.Config) -> None:

	"""Print the startup summary line."""

	modes = []
	if cfg.recorder.enabled:
		modes.append("recorder")
	if cfg.player.enabled:
		modes.append("player")
	mode_str = " + ".join(modes) if modes else "file-only"

	print(
		f"Subsample  |  {mode_str}  |  "
		f"{cfg.recorder.audio.sample_rate} Hz  "
		f"{cfg.recorder.audio.bit_depth}-bit  "
		f"{cfg.recorder.audio.channels}ch  |  "
		f"buffer {cfg.recorder.buffer.max_seconds}s  |  "
		f"SNR ≥ {cfg.detection.snr_threshold_db} dB  |  "
		f"→ {cfg.output.directory}"
	)


def _make_on_complete (
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	store_audio: bool,
) -> subsample.recorder._OnCompleteCallback:

	"""Return the on_complete callback for the live-capture WavWriter.

	The returned callback runs on the writer thread and must not block.
	It logs the analysis result, adds the recording to the instrument
	library, and updates the similarity matrix.

	Args:
		store_audio: When True, keep PCM data in the SampleRecord. Set to
		             cfg.player.enabled — audio is only needed for playback.
	"""

	def on_complete (
		filepath: pathlib.Path,
		spectral: subsample.analysis.AnalysisResult,
		rhythm: subsample.analysis.RhythmResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		level: subsample.analysis.LevelResult,
		duration: float,
		audio: numpy.ndarray,
	) -> None:

		_log.info(
			"Recorded %s  (%.2fs)\n  %s\n  %s\n  %s\n  %s",
			filepath.name, duration,
			subsample.analysis.format_result(spectral, duration),
			subsample.analysis.format_rhythm_result(rhythm),
			subsample.analysis.format_pitch_result(pitch),
			subsample.analysis.format_level_result(level),
		)

		record = subsample.library.SampleRecord(
			sample_id = subsample.library.allocate_id(),
			name      = filepath.stem,
			spectral  = spectral,
			rhythm    = rhythm,
			pitch     = pitch,
			timbre    = timbre,
			level     = level,
			params    = analysis_params,
			duration  = duration,
			audio     = audio if store_audio else None,
			filepath  = filepath,
		)

		evicted = instrument_library.add(record)

		if similarity_matrix is not None:
			if evicted:
				similarity_matrix.remove(evicted)
			similarity_matrix.add(record)

			scores = similarity_matrix.get_scores(record.sample_id)
			if scores:
				_log.debug(
					"Similarity: %s",
					subsample.similarity.format_similarity_scores(scores),
				)

	return on_complete
