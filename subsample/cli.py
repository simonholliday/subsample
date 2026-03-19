"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, and the WAV writer. Press Ctrl+C to stop cleanly.
"""

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


def main () -> None:

	"""Run the ambient audio sampler.

	Loads config, selects an audio input device, then continuously streams
	audio into a circular buffer. When the level detector identifies a
	recording event, the captured segment is queued for WAV file output.
	"""

	logging.basicConfig(
		level=logging.WARNING,
		format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
		datefmt="%H:%M:%S",
	)
	logging.getLogger("subsample").setLevel(logging.DEBUG)

	cfg = subsample.config.load_config()

	_print_banner(cfg)

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

	analysis_params = subsample.analysis.compute_params(cfg.audio.sample_rate)

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

	# Load reference library (if configured) before creating the writer so the
	# on_complete callback can capture it at construction time.
	reference_library: typing.Optional[subsample.library.ReferenceLibrary] = None
	if cfg.reference is not None:
		reference_library = subsample.library.load_reference_library(
			pathlib.Path(cfg.reference.directory)
		)
		print(f"  Reference    : {len(reference_library)} sample(s) loaded from {cfg.reference.directory}")

	writer = subsample.recorder.WavWriter(
		cfg, analysis_params,
		on_complete=_make_on_complete(reference_library, instrument_library, analysis_params),
	)

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	current_frame = 0

	try:
		while True:
			chunk = reader.read()

			buf.write(chunk)

			current_frame += cfg.audio.chunk_size

			result = detector.process_chunk(chunk, current_frame)

			if result is not None:
				start_frame, end_frame = result
				pre = cfg.detection.trim_pre_samples
				audio_segment = buf.read_range(max(0, start_frame - pre), end_frame)

				amplitude_threshold = detector.ambient_rms * (
					10.0 ** (cfg.detection.snr_threshold_db / 20.0)
				)
				audio_segment = subsample.trim.trim_silence(
					audio_segment,
					amplitude_threshold,
					pre_samples=cfg.detection.trim_pre_samples,
					post_samples=cfg.detection.trim_post_samples,
				)

				if audio_segment.size > 0:
					writer.enqueue(audio_segment, datetime.datetime.now())

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


def _make_on_complete (
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
) -> subsample.recorder._OnCompleteCallback:

	"""Return an on_complete callback that logs analysis results and manages samples.

	The returned callback runs on the writer thread after each recording is written
	and analyzed. It:
	  1. Logs the full analysis breakdown (rhythm, spectral, pitch).
	  2. Adds the recording to the instrument library (with its original PCM audio).
	  3. Logs similarity scores against reference samples (if configured).

	Args:
		reference_library: Loaded reference library, or None if not configured.
		instrument_library: Instrument library to add each new recording to.
		analysis_params:    FFT parameters used for analysis (for SampleRecord construction).
	"""

	def _on_complete (
		filepath: pathlib.Path,
		spectral: subsample.analysis.AnalysisResult,
		rhythm:   subsample.analysis.RhythmResult,
		pitch:    subsample.analysis.PitchResult,
		duration: float,
		audio:    numpy.ndarray,
	) -> None:

		_log.debug(
			"  rhythm:   %s\n"
			"  spectral: %s\n"
			"  pitch:    %s",
			subsample.analysis.format_rhythm_result(rhythm),
			subsample.analysis.format_result(spectral, duration),
			subsample.analysis.format_pitch_result(pitch),
		)

		# Build the instrument sample record and add it to the library.
		# The name is the filename stem (e.g. "2026-03-19_14-00-29").
		record = subsample.library.SampleRecord(
			sample_id = subsample.library._allocate_id(),
			name      = filepath.stem,
			spectral  = spectral,
			rhythm    = rhythm,
			pitch     = pitch,
			params    = analysis_params,
			duration  = duration,
			audio     = audio,
			filepath  = filepath,
		)

		evicted = instrument_library.add(record)

		_log.debug(
			"  instrument: #%d %s (%.1f MB used / %.1f MB limit)",
			record.sample_id, record.name,
			instrument_library.memory_used / (1024 * 1024),
			instrument_library.memory_limit / (1024 * 1024),
		)

		if evicted:
			_log.debug("  evicted:    %s", ", ".join(f"#{i}" for i in evicted))

		if reference_library is not None:
			scores = subsample.similarity.score_against_library(spectral, reference_library)
			if scores:
				_log.debug(
					"  similarity: %s",
					subsample.similarity.format_similarity_scores(scores),
				)

	return _on_complete


def _print_banner (cfg: subsample.config.Config) -> None:

	"""Print a startup summary of the active configuration."""

	print("─" * 50)
	print("  Subsample — ambient audio recorder")
	print("─" * 50)
	print(f"  Sample rate : {cfg.audio.sample_rate} Hz / {cfg.audio.bit_depth}-bit")
	print(f"  Channels    : {cfg.audio.channels}")
	print(f"  Buffer      : {cfg.buffer.max_seconds}s")
	print(f"  SNR trigger : {cfg.detection.snr_threshold_db} dB above ambient")
	print(f"  Output dir  : {cfg.output.directory}")
	print("─" * 50)
