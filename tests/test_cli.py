"""Tests for subsample.cli — argument parsing and detection pipeline helpers."""

import pathlib
import sys

import numpy
import pytest

import subsample.buffer
import subsample.cli
import subsample.config
import subsample.detector


def _make_detection_cfg (
	snr_threshold_db: float = 6.0,
	trim_pre_samples: int = 8,
	trim_post_samples: int = 8,
) -> subsample.config.DetectionConfig:
	"""Return a DetectionConfig suitable for unit tests."""
	return subsample.config.DetectionConfig(
		snr_threshold_db = snr_threshold_db,
		ema_alpha        = 0.1,
		hold_time        = 0.1,
		warmup_seconds   = 0.0,
		trim_pre_samples = trim_pre_samples,
		trim_post_samples = trim_post_samples,
	)


def _make_buffer_and_detector (
	sample_rate: int = 44100,
	chunk_size: int = 512,
	max_seconds: int = 5,
	detection_cfg: subsample.config.DetectionConfig = None,  # type: ignore[assignment]
) -> tuple[subsample.buffer.CircularBuffer, subsample.detector.LevelDetector]:
	"""Return a (CircularBuffer, LevelDetector) pair for testing."""
	if detection_cfg is None:
		detection_cfg = _make_detection_cfg()

	max_frames = sample_rate * max_seconds
	buf = subsample.buffer.CircularBuffer(
		max_frames, channels=1, dtype=numpy.dtype(numpy.int16),
	)
	detector = subsample.detector.LevelDetector(
		detection_cfg, sample_rate, chunk_size, max_recording_frames=max_frames,
	)
	return buf, detector


class TestProcessChunk:

	"""Tests for subsample.cli._process_chunk()."""

	def test_silence_returns_none (self) -> None:
		"""Pure silence should never trigger a recording."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(sample_rate=sample_rate, chunk_size=chunk_size)
		cfg = _make_detection_cfg()

		silent_chunk = numpy.zeros((chunk_size, 1), dtype=numpy.int16)

		# Feed many silent chunks — no recording should be detected
		for _ in range(50):
			result = subsample.cli._process_chunk(silent_chunk, buf, detector, cfg)
			assert result is None

	def test_loud_burst_returns_segment (self) -> None:
		"""A loud burst followed by silence should trigger one detected segment."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(
			sample_rate=sample_rate,
			chunk_size=chunk_size,
			detection_cfg=_make_detection_cfg(snr_threshold_db=3.0),
		)
		cfg = _make_detection_cfg(snr_threshold_db=3.0)

		# Feed low-amplitude chunks to establish an ambient floor before the burst.
		# If the ambient is near zero, the first loud chunk seeds ambient directly
		# to the signal level (giving 0 dB SNR), which would fail to trigger.
		# Amplitude 100 gives ~38 dB headroom against the 8000-amplitude signal.
		ambient = numpy.full((chunk_size, 1), 100, dtype=numpy.int16)
		for _ in range(30):
			subsample.cli._process_chunk(ambient, buf, detector, cfg)

		# Feed loud chunks — well above the established ambient floor (~38 dB SNR)
		loud = numpy.full((chunk_size, 1), 8000, dtype=numpy.int16)
		for _ in range(10):
			subsample.cli._process_chunk(loud, buf, detector, cfg)

		# Feed trailing silence to close the recording (hold_time expires)
		# hold_time=0.1s at 44100 Hz / 512 frames ≈ 9 chunks
		silent = numpy.zeros((chunk_size, 1), dtype=numpy.int16)
		segments = []
		for _ in range(30):
			result = subsample.cli._process_chunk(silent, buf, detector, cfg)
			if result is not None:
				segments.append(result)

		assert len(segments) == 1
		assert segments[0].ndim == 2
		assert segments[0].shape[1] == 1   # mono

	def test_returns_numpy_array (self) -> None:
		"""The returned segment should be a numpy array when a recording is detected."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(
			sample_rate=sample_rate,
			chunk_size=chunk_size,
			detection_cfg=_make_detection_cfg(snr_threshold_db=3.0),
		)
		cfg = _make_detection_cfg(snr_threshold_db=3.0)

		ambient = numpy.full((chunk_size, 1), 100, dtype=numpy.int16)
		loud = numpy.full((chunk_size, 1), 8000, dtype=numpy.int16)
		silent = numpy.zeros((chunk_size, 1), dtype=numpy.int16)

		for _ in range(30):
			subsample.cli._process_chunk(ambient, buf, detector, cfg)
		for _ in range(10):
			subsample.cli._process_chunk(loud, buf, detector, cfg)

		result = None
		for _ in range(30):
			r = subsample.cli._process_chunk(silent, buf, detector, cfg)
			if r is not None:
				result = r
				break

		assert result is not None, "Expected a segment to be detected but got None"
		assert isinstance(result, numpy.ndarray)


class TestParseArgs:

	"""Tests for subsample.cli._parse_args()."""

	def test_no_args_returns_empty_files (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""With no arguments, files should be an empty list."""
		monkeypatch.setattr(sys, "argv", ["subsample"])
		args = subsample.cli._parse_args()
		assert args.files == []

	def test_single_file_arg (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""A single positional argument should appear as a Path in files."""
		monkeypatch.setattr(sys, "argv", ["subsample", "recording.wav"])
		args = subsample.cli._parse_args()
		assert len(args.files) == 1
		assert args.files[0] == pathlib.Path("recording.wav")

	def test_multiple_file_args (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""Multiple positional arguments should each become a Path."""
		monkeypatch.setattr(sys, "argv", ["subsample", "a.wav", "b.wav", "c.wav"])
		args = subsample.cli._parse_args()
		assert len(args.files) == 3
		assert args.files[1] == pathlib.Path("b.wav")

	def test_path_type_returned (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""All entries in files should be pathlib.Path objects."""
		monkeypatch.setattr(sys, "argv", ["subsample", "some/path/recording.wav"])
		args = subsample.cli._parse_args()
		assert all(isinstance(f, pathlib.Path) for f in args.files)
