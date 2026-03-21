"""Tests for subsample.recorder."""

import datetime
import logging
import pathlib
import tempfile
import wave

import numpy
import pytest

import subsample.analysis
import subsample.audio
import subsample.config
import subsample.recorder

from .helpers import _make_params


class TestPackInt24:

	def test_zero_samples (self) -> None:
		"""All-zero int32 (left-shifted) should pack to all-zero bytes."""
		audio = numpy.zeros((4, 1), dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert result == bytes(12)  # 4 samples × 3 bytes

	def test_known_positive_value (self) -> None:
		"""Known left-shifted int32 should pack to expected 3-byte little-endian output."""
		# Internal value: 0x01020300 (24-bit value 0x010203 left-shifted by 8)
		# Expected packed bytes: 0x03, 0x02, 0x01 (little-endian 24-bit)
		audio = numpy.array([[0x01020300]], dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert result == bytes([0x03, 0x02, 0x01])

	def test_known_negative_value (self) -> None:
		"""Negative 24-bit value should round-trip correctly."""
		# 24-bit value: -1 = 0xFFFFFF, stored left-shifted as int32: -256 = 0xFFFFFF00
		audio = numpy.array([[-256]], dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert result == bytes([0xFF, 0xFF, 0xFF])

	def test_output_length (self) -> None:
		"""Output should be n_frames × channels × 3 bytes."""
		audio = numpy.zeros((10, 2), dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert len(result) == 10 * 2 * 3

	def test_round_trip_via_unpack (self) -> None:
		"""pack → unpack should recover the original 24-bit sample values."""
		# Create some 24-bit values packed into int32 (left-shifted by 8)
		original_24bit = numpy.array([0, 1000, -1000, 8388607, -8388608], dtype=numpy.int32)
		audio = (original_24bit * 256).reshape(-1, 1)  # simulate left-shift

		packed = subsample.recorder._pack_int24(audio)
		unpacked = subsample.audio.unpack_audio(packed, bit_depth=24, channels=1)

		# Unpacked values should match the original left-shifted representation
		assert numpy.array_equal(unpacked, audio)

	def test_stereo_interleaving (self) -> None:
		"""Stereo samples should be interleaved L/R/L/R in the output bytes."""
		# Frame 0: L=0x010203 << 8, R=0x040506 << 8
		left = 0x01020300
		right = 0x04050600
		audio = numpy.array([[left, right]], dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		# tobytes() on shape (1, 2) flattens row-major: L bytes, then R bytes
		assert result[:3] == bytes([0x03, 0x02, 0x01])
		assert result[3:] == bytes([0x06, 0x05, 0x04])


def _make_config (output_dir: pathlib.Path) -> subsample.config.Config:
	"""Return a minimal Config pointing output at output_dir."""
	return subsample.config.Config(
		streamer=subsample.config.StreamerConfig(
			audio=subsample.config.AudioConfig(
				sample_rate=44100, bit_depth=16, channels=1, chunk_size=512,
			),
			buffer=subsample.config.BufferConfig(max_seconds=10),
		),
		detection=subsample.config.DetectionConfig(
			snr_threshold_db=12.0, ema_alpha=0.1, hold_time=0.5,
			warmup_seconds=0.0, trim_pre_samples=8, trim_post_samples=8,
		),
		output=subsample.config.OutputConfig(
			directory=str(output_dir),
			filename_format="%Y-%m-%d_%H-%M-%S",
		),
	)


class TestWavWriterFilenameBase:

	"""Tests for the filename_base parameter in WavWriter.enqueue()."""

	def test_filename_base_used_as_stem (self) -> None:
		"""When filename_base is provided, the output file should use it as the stem."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)  # 0.1 s at 44100 Hz

		with tempfile.TemporaryDirectory() as tmp:
			out_dir = pathlib.Path(tmp)
			cfg = _make_config(out_dir)
			writer = subsample.recorder.WavWriter(cfg, _make_params())

			writer.enqueue(
				audio,
				datetime.datetime.now(),
				filename_base="my_segment_1",
			)
			writer.flush()
			writer.shutdown()

			files = list(out_dir.glob("*.wav"))

		assert len(files) == 1
		assert files[0].stem == "my_segment_1"

	def test_without_filename_base_uses_timestamp_format (self) -> None:
		"""Without filename_base, the output filename should match the timestamp format."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			out_dir = pathlib.Path(tmp)
			cfg = _make_config(out_dir)
			writer = subsample.recorder.WavWriter(cfg, _make_params())

			writer.enqueue(audio, datetime.datetime(2026, 3, 20, 12, 0, 0))
			writer.flush()
			writer.shutdown()

			files = list(out_dir.glob("*.wav"))

		assert len(files) == 1
		assert files[0].stem == "2026-03-20_12-00-00"

	def test_collision_overwrites (self) -> None:
		"""If filename_base.wav already exists, the second file overwrites it."""
		audio1 = numpy.full((4410, 1), 100, dtype=numpy.int16)
		audio2 = numpy.full((4410, 1), 200, dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			out_dir = pathlib.Path(tmp)
			cfg = _make_config(out_dir)
			writer = subsample.recorder.WavWriter(cfg, _make_params())

			ts = datetime.datetime.now()
			writer.enqueue(audio1, ts, filename_base="segment_1")
			writer.enqueue(audio2, ts, filename_base="segment_1")
			writer.flush()
			writer.shutdown()

			files = list(out_dir.glob("*.wav"))
			assert len(files) == 1
			assert files[0].stem == "segment_1"

			# Verify the *second* write (audio2, value=200) is the one that persisted,
			# not the first. Read back the first PCM sample to confirm.
			with wave.open(str(files[0]), "rb") as wf:
				raw = wf.readframes(1)
			first_sample = numpy.frombuffer(raw, dtype=numpy.int16)[0]

		assert first_sample == 200


class TestWavWriterFlush:

	"""Tests for WavWriter.flush()."""

	def test_flush_blocks_until_queue_empty (self) -> None:
		"""flush() should block until all enqueued items are written."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			out_dir = pathlib.Path(tmp)
			cfg = _make_config(out_dir)
			writer = subsample.recorder.WavWriter(cfg, _make_params())

			for i in range(3):
				writer.enqueue(audio, datetime.datetime.now(), filename_base=f"seg_{i}")

			writer.flush()

			# After flush(), all files must exist on disk
			files = list(out_dir.glob("*.wav"))
			assert len(files) == 3

			writer.shutdown()

	def test_flush_then_shutdown_safe (self) -> None:
		"""flush() followed by shutdown() should not raise or hang."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			cfg = _make_config(pathlib.Path(tmp))
			writer = subsample.recorder.WavWriter(cfg, _make_params())
			writer.enqueue(audio, datetime.datetime.now())
			writer.flush()
			writer.shutdown()  # should not raise


class TestWavWriterQueueDepth:

	"""Tests for the queue_depth property and backlog warning logging."""

	def test_queue_depth_zero_on_creation (self) -> None:
		"""queue_depth should be 0 immediately after construction."""
		with tempfile.TemporaryDirectory() as tmp:
			cfg = _make_config(pathlib.Path(tmp))
			writer = subsample.recorder.WavWriter(cfg, _make_params())

			assert writer.queue_depth == 0

			writer.shutdown()

	def test_queue_depth_zero_after_flush (self) -> None:
		"""queue_depth should be 0 after flush() completes."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			cfg = _make_config(pathlib.Path(tmp))
			writer = subsample.recorder.WavWriter(cfg, _make_params())

			writer.enqueue(audio, datetime.datetime.now())
			writer.flush()

			assert writer.queue_depth == 0

			writer.shutdown()

	def test_backlog_warning_emitted_at_depth_three (
		self, caplog: pytest.LogCaptureFixture
	) -> None:
		"""A WARNING should be logged when queue depth reaches 3."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)
		received: list[subsample.analysis.AnalysisResult] = []

		def on_complete (path, spectral, rhythm, pitch, timbre, duration, raw_audio):
			received.append(spectral)

		with tempfile.TemporaryDirectory() as tmp:
			cfg = _make_config(pathlib.Path(tmp))

			# Pause the writer thread so the queue can build up before processing.
			# Use an Event to gate the first analysis call.
			import threading
			gate = threading.Event()
			original_analyze = subsample.analysis.analyze_all

			call_count = 0

			def gated_analyze (*args, **kwargs):
				nonlocal call_count
				call_count += 1
				if call_count == 1:
					gate.wait()
				return original_analyze(*args, **kwargs)

			writer = subsample.recorder.WavWriter(cfg, _make_params(), on_complete=on_complete)

			with caplog.at_level(logging.WARNING, logger="subsample.recorder"):
				import unittest.mock
				with unittest.mock.patch("subsample.analysis.analyze_all", side_effect=gated_analyze):
					# Enqueue enough items that the queue builds up while the first is held
					for i in range(4):
						writer.enqueue(audio, datetime.datetime.now(), filename_base=f"seg_{i}")

					# Allow processing to continue
					gate.set()
					writer.flush()

			writer.shutdown()

		warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
		assert any("queue depth" in m for m in warning_messages), (
			f"Expected a queue-depth WARNING; got: {warning_messages}"
		)

	def test_drain_info_logged_after_backlog (
		self, caplog: pytest.LogCaptureFixture
	) -> None:
		"""An INFO 'queue drained' log should appear once the backlog clears."""
		audio = numpy.zeros((4410, 1), dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			cfg = _make_config(pathlib.Path(tmp))

			import threading
			import unittest.mock

			gate = threading.Event()
			original_analyze = subsample.analysis.analyze_all
			call_count = 0

			def gated_analyze (*args, **kwargs):
				nonlocal call_count
				call_count += 1
				if call_count == 1:
					gate.wait()
				return original_analyze(*args, **kwargs)

			writer = subsample.recorder.WavWriter(cfg, _make_params())

			with caplog.at_level(logging.INFO, logger="subsample.recorder"):
				with unittest.mock.patch("subsample.analysis.analyze_all", side_effect=gated_analyze):
					for i in range(4):
						writer.enqueue(audio, datetime.datetime.now(), filename_base=f"seg_{i}")
					gate.set()
					writer.flush()

			writer.shutdown()

		info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
		assert any("drained" in m for m in info_messages), (
			f"Expected a 'queue drained' INFO; got: {info_messages}"
		)
