"""Tests for subsample.detector.LevelDetector."""

import numpy
import pytest

import subsample.config
import subsample.detector


def _make_detection_config (
	snr_threshold_db: float = 6.0,
	hold_time: float = 0.5,
	warmup_seconds: float = 0.0,  # 0 warmup for most tests — skip straight to IDLE
	ema_alpha: float = 0.5,       # High alpha for fast ambient adjustment in tests
) -> subsample.config.DetectionConfig:

	return subsample.config.DetectionConfig(
		snr_threshold_db=snr_threshold_db,
		hold_time=hold_time,
		warmup_seconds=warmup_seconds,
		ema_alpha=ema_alpha,
		trim_pre_samples=0,
		trim_post_samples=0,
	)


def _make_detector (
	snr_threshold_db: float = 6.0,
	hold_time: float = 0.5,
	warmup_seconds: float = 0.0,
	ema_alpha: float = 0.5,
	sample_rate: int = 1000,
	chunk_size: int = 100,
) -> subsample.detector.LevelDetector:

	cfg = _make_detection_config(
		snr_threshold_db=snr_threshold_db,
		hold_time=hold_time,
		warmup_seconds=warmup_seconds,
		ema_alpha=ema_alpha,
	)
	return subsample.detector.LevelDetector(cfg, sample_rate, chunk_size)


def _silent_chunk (n: int = 100) -> numpy.ndarray:
	"""Return a chunk of near-silence (very low amplitude)."""
	return numpy.full(n, 1, dtype=numpy.int16)


def _loud_chunk (n: int = 100, amplitude: int = 10000) -> numpy.ndarray:
	"""Return a chunk at high amplitude."""
	return numpy.full(n, amplitude, dtype=numpy.int16)


class TestWarmup:

	def test_no_trigger_during_warmup (self) -> None:
		# warmup_seconds=0.5, chunk_size=100, sample_rate=1000 → 5 warmup chunks
		detector = _make_detector(warmup_seconds=0.5, sample_rate=1000, chunk_size=100)

		assert detector.state == subsample.detector.DetectorState.WARMUP

		# Even a loud chunk should not trigger during warmup
		for i in range(4):
			result = detector.process_chunk(_loud_chunk(), current_frame=(i + 1) * 100)
			assert result is None
			assert detector.state == subsample.detector.DetectorState.WARMUP

	def test_transitions_to_idle_after_warmup (self) -> None:
		# 5 warmup chunks required (0.5s at 1000 Hz / 100 frames each)
		detector = _make_detector(warmup_seconds=0.5, sample_rate=1000, chunk_size=100)

		for i in range(5):
			detector.process_chunk(_silent_chunk(), current_frame=(i + 1) * 100)

		assert detector.state == subsample.detector.DetectorState.IDLE


class TestIdleToRecording:

	def test_loud_chunk_starts_recording (self) -> None:
		detector = _make_detector()

		# Seed ambient with a quiet chunk first
		detector.process_chunk(_silent_chunk(), current_frame=100)
		assert detector.state == subsample.detector.DetectorState.IDLE

		# Loud chunk should trigger recording
		result = detector.process_chunk(_loud_chunk(), current_frame=200)

		assert result is None  # Recording started but not yet ended
		assert detector.state == subsample.detector.DetectorState.RECORDING

	def test_quiet_chunk_stays_idle (self) -> None:
		detector = _make_detector()

		detector.process_chunk(_silent_chunk(), current_frame=100)
		result = detector.process_chunk(_silent_chunk(), current_frame=200)

		assert result is None
		assert detector.state == subsample.detector.DetectorState.IDLE


class TestHoldTime:

	def test_recording_ends_after_hold_time (self) -> None:
		# hold_time=0.5, chunk_size=100, sample_rate=1000 → 5 hold chunks
		detector = _make_detector(hold_time=0.5, sample_rate=1000, chunk_size=100)

		# Seed ambient
		detector.process_chunk(_silent_chunk(), current_frame=100)

		# Trigger recording
		detector.process_chunk(_loud_chunk(), current_frame=200)
		assert detector.state == subsample.detector.DetectorState.RECORDING

		# Send 5 quiet chunks — hold countdown should expire on the 5th
		result = None
		for i in range(5):
			result = detector.process_chunk(_silent_chunk(), current_frame=300 + i * 100)

		assert result is not None
		assert detector.state == subsample.detector.DetectorState.IDLE

	def test_recording_extends_while_signal_present (self) -> None:
		# hold_time=0.2 → 2 hold chunks; ema_alpha=0.01 so ambient adapts slowly
		detector = _make_detector(hold_time=0.2, sample_rate=1000, chunk_size=100, ema_alpha=0.01)

		# Seed ambient
		detector.process_chunk(_silent_chunk(), current_frame=100)

		# Trigger recording
		detector.process_chunk(_loud_chunk(), current_frame=200)

		# One quiet chunk — still within hold time
		result = detector.process_chunk(_silent_chunk(), current_frame=300)
		assert result is None
		assert detector.state == subsample.detector.DetectorState.RECORDING

		# Loud chunk resets hold time
		result = detector.process_chunk(_loud_chunk(), current_frame=400)
		assert result is None
		assert detector.state == subsample.detector.DetectorState.RECORDING

	def test_recording_boundary_frames_are_correct (self) -> None:
		# hold_time=0.1 → 1 hold chunk at these settings
		detector = _make_detector(hold_time=0.1, sample_rate=1000, chunk_size=100)

		# Seed ambient at frame 100
		detector.process_chunk(_silent_chunk(), current_frame=100)

		# Start recording at frame 200 (chunk ends at 200, started at 100)
		detector.process_chunk(_loud_chunk(), current_frame=200)

		# 1 quiet chunk ends the recording at frame 300
		result = detector.process_chunk(_silent_chunk(), current_frame=300)

		assert result is not None
		start, end = result
		assert start == 100   # recording_start_frame = current_frame - chunk_size = 200 - 100
		assert end == 300


class TestThresholdMath:

	def test_snr_below_threshold_does_not_trigger (self) -> None:
		# 6 dB threshold — need signal to be ~2x ambient RMS
		detector = _make_detector(snr_threshold_db=6.0, ema_alpha=0.99)

		# Large ambient first chunk (ema_alpha=0.99 → ambient ≈ chunk_rms after first chunk)
		big_ambient = numpy.full(100, 5000, dtype=numpy.int16)
		detector.process_chunk(big_ambient, current_frame=100)

		# Signal at 1.5x ambient — SNR ≈ 3.5 dB, below 6 dB threshold
		slightly_louder = numpy.full(100, 7500, dtype=numpy.int16)
		result = detector.process_chunk(slightly_louder, current_frame=200)

		assert result is None
		assert detector.state == subsample.detector.DetectorState.IDLE

	def test_zero_amplitude_does_not_trigger (self) -> None:
		detector = _make_detector()

		detector.process_chunk(_silent_chunk(), current_frame=100)

		zero_chunk = numpy.zeros(100, dtype=numpy.int16)
		result = detector.process_chunk(zero_chunk, current_frame=200)

		assert result is None
		assert detector.state == subsample.detector.DetectorState.IDLE


class TestBufferOverflow:

	def test_force_end_when_max_frames_reached (self) -> None:
		# chunk_size=100, max_recording_frames=500 → force-end after 5 loud chunks
		cfg = _make_detection_config(ema_alpha=0.01)
		detector = subsample.detector.LevelDetector(
			cfg,
			sample_rate=1000,
			chunk_size=100,
			max_recording_frames=500,
		)

		# Seed ambient
		detector.process_chunk(_silent_chunk(), current_frame=100)

		# Trigger recording
		detector.process_chunk(_loud_chunk(), current_frame=200)
		assert detector.state == subsample.detector.DetectorState.RECORDING

		# Keep feeding loud chunks until force-end fires
		result = None
		for i in range(2, 20):
			result = detector.process_chunk(_loud_chunk(), current_frame=100 + i * 100)
			if result is not None:
				break

		# Force-end should have fired before the loop exhausted
		assert result is not None
		assert detector.state == subsample.detector.DetectorState.IDLE

	def test_no_force_end_when_limit_is_zero (self) -> None:
		# max_recording_frames=0 disables the overflow check
		cfg = _make_detection_config(ema_alpha=0.01)
		detector = subsample.detector.LevelDetector(
			cfg,
			sample_rate=1000,
			chunk_size=100,
			max_recording_frames=0,
		)

		detector.process_chunk(_silent_chunk(), current_frame=100)
		detector.process_chunk(_loud_chunk(), current_frame=200)

		# Feed many loud chunks — no force-end should occur
		result = None
		for i in range(2, 20):
			result = detector.process_chunk(_loud_chunk(), current_frame=100 + i * 100)
			assert result is None

		assert detector.state == subsample.detector.DetectorState.RECORDING
