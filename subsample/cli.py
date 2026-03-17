"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, and the WAV writer. Press Ctrl+C to stop cleanly.
"""

import datetime
import logging
import sys

import numpy

import subsample.analysis
import subsample.audio
import subsample.buffer
import subsample.config
import subsample.detector
import subsample.recorder
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
		devices = subsample.audio.list_input_devices(pa)
		device_index = subsample.audio.select_device(devices)
		stream = subsample.audio.open_stream(pa, device_index, cfg.audio)
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

	analysis_params = subsample.analysis.compute_params(cfg.audio)
	writer = subsample.recorder.WavWriter(cfg, analysis_params)

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	current_frame = 0

	try:
		while True:
			raw_bytes = stream.read(cfg.audio.chunk_size, exception_on_overflow=False)

			chunk = subsample.audio.unpack_audio(
				raw_bytes, cfg.audio.bit_depth, cfg.audio.channels
			)

			buf.write(chunk)

			current_frame += cfg.audio.chunk_size

			result = detector.process_chunk(chunk, current_frame)

			if result is not None:
				start_frame, end_frame = result
				audio_segment = buf.read_range(start_frame, end_frame)

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
		stream.stop_stream()
		stream.close()
		pa.terminate()
		writer.shutdown()

	print("Done.")


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
