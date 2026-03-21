"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, and the WAV writer. Supports two input modes:

  File input   — pass WAV file paths as positional arguments; each file
                 is processed through the detection pipeline and segments
                 saved to the output directory, then picked up by the
                 instrument library loader.

  Live capture — stream from an audio input device (always runs after
                 file input, if a device is configured).

Press Ctrl+C to stop the live capture loop cleanly.
"""

import argparse
import datetime
import logging
import pathlib
import sys
import typing

import numpy

import subsample.analysis
import subsample.audio
import subsample.buffer
import subsample.config
import subsample.detector
import subsample.library
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

		max_frames = file_info.sample_rate * cfg.buffer.max_seconds
		buf = subsample.buffer.CircularBuffer(max_frames, file_info.channels, dtype=audio_dtype)

		detector = subsample.detector.LevelDetector(
			cfg.detection,
			file_info.sample_rate,
			cfg.audio.chunk_size,
			max_recording_frames=max_frames,
		)

		analysis_params = subsample.analysis.compute_params(file_info.sample_rate)

		writer = subsample.recorder.WavWriter(cfg, analysis_params, on_complete=None)

		segment_index = 1
		chunk_size = cfg.audio.chunk_size
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


def _stream_from_device (
	cfg: subsample.config.Config,
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
) -> None:

	"""Set up an audio input device and run the real-time capture loop.

	Streams audio from the configured device (or interactively selected device)
	into a circular buffer. Detected recordings are trimmed, queued for WAV
	output, analyzed, and added to the instrument library. Press Ctrl+C to stop.

	Args:
		cfg:                Full application config.
		reference_library:  Loaded reference samples, or None if not configured.
		instrument_library: Instrument sample library to update in real time.
		analysis_params:    Pre-computed FFT params matching cfg.audio.sample_rate.
		similarity_matrix:  Similarity index to update as new samples arrive, or None.
	"""

	pa = subsample.audio.create_pyaudio()

	try:
		if cfg.audio.device is not None:
			device_index = subsample.audio.find_device_by_name(pa, cfg.audio.device)
		else:
			devices = subsample.audio.list_input_devices(pa)
			device_index = subsample.audio.select_device(devices)
		reader = subsample.audio.AudioReader(pa, device_index, cfg.audio)
	except (ValueError, OSError) as exc:
		print(f"Error opening audio device: {exc}", file=sys.stderr)
		pa.terminate()
		sys.exit(1)

	audio_dtype = _AUDIO_DTYPE[cfg.audio.bit_depth]
	max_frames = cfg.audio.sample_rate * cfg.buffer.max_seconds
	buf = subsample.buffer.CircularBuffer(max_frames, cfg.audio.channels, dtype=audio_dtype)

	detector = subsample.detector.LevelDetector(
		cfg.detection,
		cfg.audio.sample_rate,
		cfg.audio.chunk_size,
		max_recording_frames=max_frames,
	)

	writer = subsample.recorder.WavWriter(
		cfg,
		analysis_params,
		on_complete=_make_on_complete(
			reference_library, instrument_library, analysis_params, similarity_matrix,
		),
	)

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	try:
		while True:
			chunk = reader.read()

			trimmed = _process_chunk(chunk, buf, detector, cfg.detection)

			if trimmed is not None:
				writer.enqueue(trimmed, datetime.datetime.now())

	except KeyboardInterrupt:
		print("\nStopping…")

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

	print("Done.")


def main () -> None:

	"""Run the ambient audio sampler.

	Processes any input files first (if given on the command line), then
	loads libraries and starts live capture from an audio input device.
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

	# Create instrument library. If a directory is configured, pre-load samples
	# from disk; otherwise start empty. Memory limit is always enforced.
	max_instrument_bytes = int(cfg.instrument.max_memory_mb * 1024 * 1024)
	if cfg.instrument.directory is not None:
		instrument_library = subsample.library.load_instrument_library(
			pathlib.Path(cfg.instrument.directory),
			max_instrument_bytes,
		)
		print(
			f"  Instruments  : {len(instrument_library)} sample(s) loaded"
			f" from {cfg.instrument.directory}"
		)
	else:
		instrument_library = subsample.library.InstrumentLibrary(max_instrument_bytes)

	analysis_params = subsample.analysis.compute_params(cfg.audio.sample_rate)

	# Build the similarity matrix now that both libraries are populated.
	# bulk_add() is vectorised (single matrix multiply for N × M scores),
	# so loading hundreds of pre-existing instrument samples is fast.
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix] = None
	if reference_library is not None:
		similarity_matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
		if len(instrument_library) > 0:
			similarity_matrix.bulk_add(instrument_library.samples())
		print(f"  Similarity   : {similarity_matrix}")

	_stream_from_device(cfg, reference_library, instrument_library, analysis_params, similarity_matrix)


def _print_banner (cfg: subsample.config.Config) -> None:

	"""Print the startup summary line."""

	print(
		f"Subsample  |  "
		f"{cfg.audio.sample_rate} Hz  "
		f"{cfg.audio.bit_depth}-bit  "
		f"{cfg.audio.channels}ch  |  "
		f"buffer {cfg.buffer.max_seconds}s  |  "
		f"SNR ≥ {cfg.detection.snr_threshold_db} dB  |  "
		f"→ {cfg.output.directory}"
	)


def _make_on_complete (
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
) -> subsample.recorder._OnCompleteCallback:

	"""Return the on_complete callback for the live-capture WavWriter.

	The returned callback runs on the writer thread and must not block.
	It logs the analysis result, adds the recording to the instrument
	library, and updates the similarity matrix.
	"""

	def on_complete (
		filepath: pathlib.Path,
		spectral: subsample.analysis.AnalysisResult,
		rhythm: subsample.analysis.RhythmResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		duration: float,
		audio: numpy.ndarray,
	) -> None:

		_log.info(
			"Recorded %s  (%.2fs)\n  %s\n  %s\n  %s",
			filepath.name, duration,
			subsample.analysis.format_result(spectral, duration),
			subsample.analysis.format_rhythm_result(rhythm),
			subsample.analysis.format_pitch_result(pitch),
		)

		record = subsample.library.SampleRecord(
			sample_id = subsample.library.allocate_id(),
			name      = filepath.stem,
			spectral  = spectral,
			rhythm    = rhythm,
			pitch     = pitch,
			timbre    = timbre,
			params    = analysis_params,
			duration  = duration,
			audio     = audio,
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
