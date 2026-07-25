"""Tests for subsample.detector.LevelDetector."""

import typing

import numpy
import pytest

import subsample.buffer
import subsample.config
import subsample.detector
import subsample.trim


def _make_detection_config (
	threshold_db: float = 6.0,
	hold_seconds: float = 0.5,
	warmup_seconds: float = 0.0,  # 0 warmup for most tests — skip straight to IDLE
	floor_adaptation: float = 0.5,       # High alpha for fast ambient adjustment in tests
	release_threshold_db: typing.Optional[float] = None,
	retrigger_threshold_db: typing.Optional[float] = None,
	fade_out_ms: float = 0.0,
) -> subsample.config.DetectionConfig:

	"""Factory for test DetectionConfig with sensible defaults."""

	return subsample.config.DetectionConfig(
		threshold_db=threshold_db,
		hold_seconds=hold_seconds,
		warmup_seconds=warmup_seconds,
		floor_adaptation=floor_adaptation,
		trim_pre_ms=0,
		trim_post_ms=0,
		release_threshold_db=release_threshold_db,
		retrigger_threshold_db=retrigger_threshold_db,
		fade_out_ms=fade_out_ms,
	)


def _make_detector (
	threshold_db: float = 6.0,
	hold_seconds: float = 0.5,
	warmup_seconds: float = 0.0,
	floor_adaptation: float = 0.5,
	sample_rate: int = 1000,
	chunk_size: int = 100,
	release_threshold_db: typing.Optional[float] = None,
	retrigger_threshold_db: typing.Optional[float] = None,
) -> subsample.detector.LevelDetector:

	"""Factory for test LevelDetector with sensible defaults."""

	cfg = _make_detection_config(
		threshold_db=threshold_db,
		hold_seconds=hold_seconds,
		warmup_seconds=warmup_seconds,
		floor_adaptation=floor_adaptation,
		release_threshold_db=release_threshold_db,
		retrigger_threshold_db=retrigger_threshold_db,
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

	def test_trigger_chunk_does_not_pollute_ambient (self) -> None:
		"""Code-review regression: the triggering chunk must NOT be folded into
		the ambient floor before its own SNR test (which deflated the measured
		SNR and could miss sharp drum hits)."""
		detector = _make_detector(floor_adaptation=0.5)

		for i in range(3):
			detector.process_chunk(_silent_chunk(), current_frame=(i + 1) * 100)

		ambient_before = detector._ambient_rms

		result = detector.process_chunk(_loud_chunk(), current_frame=400)

		assert result is None
		assert detector.state == subsample.detector.DetectorState.RECORDING
		# The loud trigger chunk left the ambient floor untouched.
		assert detector._ambient_rms == ambient_before

	def test_quiet_chunk_stays_idle (self) -> None:
		detector = _make_detector()

		detector.process_chunk(_silent_chunk(), current_frame=100)
		result = detector.process_chunk(_silent_chunk(), current_frame=200)

		assert result is None
		assert detector.state == subsample.detector.DetectorState.IDLE


class TestHoldTime:

	def test_recording_ends_after_hold_seconds (self) -> None:
		# hold_seconds=0.5, chunk_size=100, sample_rate=1000 → 5 hold chunks
		detector = _make_detector(hold_seconds=0.5, sample_rate=1000, chunk_size=100)

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

	def test_ambient_not_updated_during_recording (self) -> None:
		"""Ambient EMA must not track the signal while recording is active.

		floor_adaptation=0.2: without the freeze, ambient drifts toward the signal level
		and SNR drops below 6 dB after ~4 loud chunks, ending the recording early.
		With the freeze, the 14 dB SNR established at trigger is preserved and the
		recording continues for as long as the signal is present.

		Note: floor_adaptation must not be too high (e.g. 0.9), because _update_ambient
		runs on the trigger chunk while still in IDLE — a very high alpha would push
		ambient so close to signal that the threshold is never exceeded.
		"""
		detector = _make_detector(
			threshold_db=6.0,
			hold_seconds=0.5,
			floor_adaptation=0.2,
			sample_rate=1000,
			chunk_size=100,
		)

		# Seed a quiet ambient
		detector.process_chunk(_silent_chunk(), current_frame=100)

		# Trigger recording (ambient updates in IDLE, then state → RECORDING)
		detector.process_chunk(_loud_chunk(), current_frame=200)
		assert detector.state == subsample.detector.DetectorState.RECORDING
		ambient_after_trigger = detector.ambient_rms

		# Feed 20 more loud chunks — ambient must remain frozen during RECORDING
		for i in range(2, 22):
			result = detector.process_chunk(_loud_chunk(), current_frame=100 + i * 100)
			assert result is None, "Recording ended early — ambient EMA is still drifting"
			assert detector.ambient_rms == ambient_after_trigger, (
				"Ambient EMA changed during recording"
			)

		assert detector.state == subsample.detector.DetectorState.RECORDING

	def test_recording_extends_while_signal_present (self) -> None:
		# hold_seconds=0.2 → 2 hold chunks
		detector = _make_detector(hold_seconds=0.2, sample_rate=1000, chunk_size=100, floor_adaptation=0.01)

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
		# hold_seconds=0.1 → 1 hold chunk at these settings
		detector = _make_detector(hold_seconds=0.1, sample_rate=1000, chunk_size=100)

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
		detector = _make_detector(threshold_db=6.0, floor_adaptation=0.99)

		# Large ambient first chunk (floor_adaptation=0.99 → ambient ≈ chunk_rms after first chunk)
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
		cfg = _make_detection_config()
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
		cfg = _make_detection_config()
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

	def test_recording_at_max_frames_is_fully_retrievable (self) -> None:
		"""End-to-end: a recording force-ended at the buffer capacity must still
		be fully readable.

		The whole point of max_recording_frames is to force-end before the
		circular buffer overwrites the recording's own start.  Driving the
		detector and buffer together in cli._process_chunk's write-then-process
		order proves the returned (start, end) span is still entirely in the
		buffer — a short read would mean the start frame was lost.
		"""

		max_frames = 500
		buf = subsample.buffer.CircularBuffer(max_frames=max_frames, channels=1)
		detector = subsample.detector.LevelDetector(
			_make_detection_config(),
			sample_rate=1000,
			chunk_size=100,
			max_recording_frames=max_frames,
		)

		# Seed ambient, then trigger the recording.
		buf.write(_silent_chunk())
		detector.process_chunk(_silent_chunk(), buf.frames_written)

		loud = _loud_chunk()
		buf.write(loud)
		result = detector.process_chunk(loud, buf.frames_written)
		assert detector.state == subsample.detector.DetectorState.RECORDING

		# Hold the recording until the force-end fires.
		for _ in range(20):
			loud = _loud_chunk()
			buf.write(loud)
			result = detector.process_chunk(loud, buf.frames_written)
			if result is not None:
				break

		assert result is not None
		start_frame, end_frame = result

		# read_range clamps to the oldest retained frame, so a full-length read
		# proves the start survived; the span never exceeds the buffer capacity.
		segment = buf.read_range(start_frame, end_frame)
		assert end_frame - start_frame <= max_frames
		assert segment.shape[0] == end_frame - start_frame


def _at_db_over (ambient_amp: int, db: float) -> int:

	"""Amplitude of a constant chunk sitting `db` dB over an ambient amplitude."""

	return int(round(ambient_amp * (10.0 ** (db / 20.0))))


class TestOnsetRefinement:

	"""The recording start is pulled forward from the chunk boundary (where the
	level trigger quantises it) to the actual transient within the chunk, so a
	strike landing late in its detection chunk does not leave a chunk of leading
	near-silence in front of the sample."""

	def test_start_pulled_to_the_strike (self) -> None:
		sr, cs = 48000, 512
		detector = _make_detector(
			threshold_db=10.0, hold_seconds=0.01, sample_rate=sr, chunk_size=cs,
		)
		# First chunk seeds ambient (warmup) to a quiet floor, then IDLE.
		quiet = numpy.full((cs, 1), 10, dtype=numpy.int16)
		detector.process_chunk(quiet, current_frame=cs)

		# Triggering chunk: ~400 samples of near-silence, then the strike.
		transient = numpy.full((cs, 1), 10, dtype=numpy.int16)
		transient[400:] = 8000
		detector.process_chunk(transient, current_frame=2 * cs)

		# Close the recording (hold_seconds ~1 chunk) and read the boundary.
		result = None
		frame = 3 * cs
		for _ in range(5):
			r = detector.process_chunk(quiet, current_frame=frame)
			frame += cs
			if r is not None:
				result = r
				break

		assert result is not None
		start, _end = result
		chunk_start = cs  # the triggering chunk begins at frame 512
		# The strike is at offset 400 in the chunk; start lands just before it,
		# NOT at the chunk boundary (which would leave ~400 samples of silence).
		assert start > chunk_start + 300
		assert start <= chunk_start + 400

	def test_constant_chunk_leaves_boundary_unrefined (self) -> None:
		# A flat chunk (peak from sample 0) has its onset at 0 — no shift, so the
		# refinement never mis-fires on a signal that is loud from the first sample.
		detector = _make_detector(
			threshold_db=6.0, hold_seconds=0.01, sample_rate=48000, chunk_size=512,
		)
		detector.process_chunk(numpy.full((512, 1), 100, dtype=numpy.int16), current_frame=512)
		detector.process_chunk(numpy.full((512, 1), 8000, dtype=numpy.int16), current_frame=1024)
		result = None
		frame = 1536
		for _ in range(5):
			r = detector.process_chunk(numpy.full((512, 1), 100, dtype=numpy.int16), current_frame=frame)
			frame += 512
			if r is not None:
				result = r
				break
		assert result is not None
		start, _end = result
		assert start == 1024 - 512  # exactly the chunk boundary, no shift


class TestReleaseThreshold:

	"""Decoupled CLOSE threshold (release_threshold_db): a recording opens on the
	loud threshold_db attack but ends only once the tail falls below the lower
	release level, so a long decay is preserved instead of cut at the start level."""

	def _open_recording (
		self, detector: subsample.detector.LevelDetector, ambient_amp: int,
	) -> None:
		# Seed ambient, then open with a chunk far above the open threshold.
		detector.process_chunk(_loud_chunk(amplitude=ambient_amp), current_frame=100)
		detector.process_chunk(_loud_chunk(amplitude=ambient_amp * 100), current_frame=200)
		assert detector.state == subsample.detector.DetectorState.RECORDING

	def test_tail_above_release_keeps_recording (self) -> None:
		# snr=20 opens; release=6 keeps recording while the tail is above 6 dB.
		# A tail at 12 dB over ambient is below the open threshold but above release,
		# so with the single-threshold model it would have ended — here it does not.
		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.5,
			release_threshold_db=6.0, sample_rate=1000, chunk_size=100,
		)
		ambient = 100
		self._open_recording(detector, ambient)

		tail = _at_db_over(ambient, 12.0)  # above release (6), below open (20)
		frame = 300
		for _ in range(10):
			result = detector.process_chunk(_loud_chunk(amplitude=tail), current_frame=frame)
			frame += 100
			assert result is None
		assert detector.state == subsample.detector.DetectorState.RECORDING

	def test_tail_below_release_ends_after_hold (self) -> None:
		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.5,
			release_threshold_db=6.0, sample_rate=1000, chunk_size=100,
		)
		ambient = 100
		self._open_recording(detector, ambient)

		# hold_seconds 0.5 s at 10 chunks/s = 5 hold chunks.
		quiet = _at_db_over(ambient, 3.0)  # below release (6)
		result = None
		frame = 300
		for _ in range(6):
			result = detector.process_chunk(_loud_chunk(amplitude=quiet), current_frame=frame)
			frame += 100
			if result is not None:
				break
		assert result is not None
		assert detector.state == subsample.detector.DetectorState.IDLE

	def test_attack_and_tail_amplitude_thresholds (self) -> None:
		# The trim edges use DIFFERENT gates: attack = snr (open), tail = release
		# (close).  Attack > tail whenever release < snr, so a low release preserves
		# a long tail without loosening the attack trim.
		detector = _make_detector(
			threshold_db=20.0, release_threshold_db=6.0,
			sample_rate=1000, chunk_size=100,
		)
		# First chunk seeds the ambient EMA (warmup) to 100, then IDLE.
		detector.process_chunk(_loud_chunk(amplitude=100), current_frame=100)
		assert abs(detector.ambient_rms - 100.0) < 1e-6

		assert detector.attack_amplitude_threshold > detector.tail_amplitude_threshold
		assert abs(detector.attack_amplitude_threshold - 100.0 * 10.0 ** (20.0 / 20.0)) < 0.01
		assert abs(detector.tail_amplitude_threshold - 100.0 * 10.0 ** (6.0 / 20.0)) < 0.01

	def test_attack_threshold_independent_of_release (self) -> None:
		# The whole point: changing release must not change the attack gate.
		high_release = _make_detector(threshold_db=20.0, release_threshold_db=15.0, sample_rate=1000, chunk_size=100)
		low_release = _make_detector(threshold_db=20.0, release_threshold_db=1.0, sample_rate=1000, chunk_size=100)
		for det in (high_release, low_release):
			det.process_chunk(_loud_chunk(amplitude=100), current_frame=100)
		assert high_release.attack_amplitude_threshold == low_release.attack_amplitude_threshold

	def test_snr_only_ends_at_snr_level (self) -> None:
		# Backward-compat: with no release set, the CLOSE reuses threshold_db,
		# so the same 12-dB tail that release=6 preserves ends the recording here.
		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.3, sample_rate=1000, chunk_size=100,
		)
		ambient = 100
		self._open_recording(detector, ambient)

		tail = _at_db_over(ambient, 12.0)  # below snr (20) → counts as silence
		result = None
		frame = 300
		for _ in range(4):  # 3 hold chunks + margin
			result = detector.process_chunk(_loud_chunk(amplitude=tail), current_frame=frame)
			frame += 100
			if result is not None:
				break
		assert result is not None
		assert detector.state == subsample.detector.DetectorState.IDLE


class TestRetrigger:

	"""retrigger_threshold_db: a sharp rise over the decaying tail ends the current
	sample and immediately opens the next, so spaced hits whose tails never reach
	silence are still separated (the 'next hit ends the current' guarantee)."""

	def test_next_hit_splits_into_two_segments (self) -> None:
		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.3,
			release_threshold_db=6.0, retrigger_threshold_db=12.0,
			sample_rate=1000, chunk_size=100,
		)
		ambient = 100
		detector.process_chunk(_loud_chunk(amplitude=ambient), current_frame=100)

		segments = []
		frame = 200
		# hit 1 attack, then a decay that stays above release (never reaches silence)
		envelope = [8000, 4000, 2000, 1200, 900, 800, 750, 720]
		for amp in envelope:
			r = detector.process_chunk(_loud_chunk(amplitude=amp), current_frame=frame)
			frame += 100
			if r is not None:
				segments.append(r)
		# hit 2: a re-strike far above the decaying tail follower
		r = detector.process_chunk(_loud_chunk(amplitude=10000), current_frame=frame)
		frame += 100
		if r is not None:
			segments.append(r)

		assert len(segments) == 1  # the first hit was closed at the second's onset
		assert detector.state == subsample.detector.DetectorState.RECORDING  # hit 2 now open

	def test_attack_peak_does_not_split_a_single_hit (self) -> None:
		# The min-segment guard (and the climbing follower) must stop a hit's own
		# attack peak — a large rise over the quieter onset chunk — from splitting
		# the hit off from its tail.  One strike must yield exactly one segment.
		detector = _make_detector(
			threshold_db=15.0, hold_seconds=0.3,
			release_threshold_db=4.0, retrigger_threshold_db=10.0,
			sample_rate=1000, chunk_size=100,
		)
		ambient = 100
		detector.process_chunk(_loud_chunk(amplitude=ambient), current_frame=100)

		segments = []
		frame = 200
		# a modest onset that opens, then a much louder attack peak, then decay
		# to silence — the classic shape that would false-split without the guard.
		envelope = [700, 8000, 9000, 6000, 3000, 1500, 800, 400, 200, 150, 120, 100, 100, 100]
		for amp in envelope:
			r = detector.process_chunk(_loud_chunk(amplitude=amp), current_frame=frame)
			frame += 100
			if r is not None:
				segments.append(r)

		assert len(segments) == 1
		assert detector.state == subsample.detector.DetectorState.IDLE

	def test_two_stage_attack_needs_a_hold_that_spans_it (self) -> None:
		# The re-trigger guard = max(hold_seconds, 0.1 s).  A two-stage attack — a soft
		# onset that opens the gate, then a louder transient arriving AFTER the guard
		# (breath before a flute note, mallet contact before a bowl, a bow scratch) —
		# reads the transient as a re-strike when hold_seconds is short, over-splitting
		# one gesture into two.  The honest precondition: the whole attack must land
		# within the guard, so slow/two-stage attacks need a longer hold_seconds.  A
		# clean fast-attack ride cymbal (the documented target) is unaffected.
		onset = [700, 700, 700]
		transient_decay = [9000, 6000, 3000, 1500, 800, 400, 200, 150] + [100] * 10

		def run (hold_seconds: float) -> int:
			detector = _make_detector(
				threshold_db=10.0, hold_seconds=hold_seconds,
				release_threshold_db=4.0, retrigger_threshold_db=12.0,
				sample_rate=1000, chunk_size=100,
			)
			detector.process_chunk(_loud_chunk(amplitude=100), current_frame=100)
			segments = []
			frame = 200
			for amp in onset + transient_decay:
				r = detector.process_chunk(_loud_chunk(amplitude=amp), current_frame=frame)
				frame += 100
				if r is not None:
					segments.append(r)
			return len(segments)

		assert run(0.05) >= 2   # short hold → guard too small → the transient false-splits
		assert run(0.5) == 1    # hold spanning the attack → one gesture, one sample

	def test_no_retrigger_when_disabled (self) -> None:
		# With retrigger unset, a second hit before silence just resets the hold —
		# the two hits merge into one recording (historical behaviour).
		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.3,
			release_threshold_db=6.0, sample_rate=1000, chunk_size=100,
		)
		ambient = 100
		detector.process_chunk(_loud_chunk(amplitude=ambient), current_frame=100)

		segments = []
		frame = 200
		envelope = [8000, 2000, 900, 750, 10000, 3000, 900, 750]  # two hits, no silence between
		for amp in envelope:
			r = detector.process_chunk(_loud_chunk(amplitude=amp), current_frame=frame)
			frame += 100
			if r is not None:
				segments.append(r)
		assert segments == []  # still one open recording, never split
		assert detector.state == subsample.detector.DetectorState.RECORDING


class TestFinalize:

	"""finalize() flushes a recording still open at end-of-stream (file input)."""

	def test_finalize_emits_open_recording_and_resets (self) -> None:

		"""A recording still open when the input ends is emitted as
		(start, current_frame), and the detector returns to IDLE — without this,
		file input would drop a final sound running to within hold_seconds of EOF."""

		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.5, sample_rate=1000, chunk_size=100,
		)
		# Seed ambient, then open with a chunk far above the open threshold.
		detector.process_chunk(_loud_chunk(amplitude=100), current_frame=100)
		detector.process_chunk(_loud_chunk(amplitude=100 * 100), current_frame=200)
		assert detector.state == subsample.detector.DetectorState.RECORDING

		bounds = detector.finalize(current_frame=1234)

		assert bounds is not None
		start, end = bounds
		assert end == 1234
		assert start < end
		assert detector.state == subsample.detector.DetectorState.IDLE

	def test_finalize_returns_none_when_not_recording (self) -> None:

		"""With no recording open, finalize is a no-op returning None."""

		detector = _make_detector(
			threshold_db=20.0, hold_seconds=0.5, sample_rate=1000, chunk_size=100,
		)

		assert detector.finalize(current_frame=500) is None


def _noise_chunk (n: int = 100, rms: float = 100.0, seed: int = 0) -> numpy.ndarray:

	"""Return a deterministic Gaussian-noise chunk scaled to approximately `rms`.

	Noise, not a constant.  The whole point of the ambient PEAK EMA is that a noise
	chunk's peak sits a crest factor above its RMS — a relationship a constant chunk
	(peak == RMS) cannot express, which is why the rest of this module's helpers
	cannot exercise it.
	"""

	rng = numpy.random.default_rng(seed)
	raw = rng.standard_normal(n)
	raw *= rms / float(numpy.sqrt(numpy.mean(raw ** 2)))

	return numpy.round(raw).astype(numpy.int16)


class TestAmbientPeak:

	"""The ambient PEAK EMA tracked alongside the ambient RMS EMA.

	It exists so the LEADING trim gate can be compared against per-sample
	magnitudes in the same domain.  See TestFalseTriggerHeadIsTrimmed for the
	defect that motivated it.
	"""

	def test_peak_ema_rises_above_rms_ema_for_noise (self) -> None:

		"""For noise-like ambient the two EMAs must diverge by the crest factor."""

		detector = _make_detector(threshold_db=40.0, sample_rate=1000, chunk_size=100)

		frame = 100
		for seed in range(12):
			detector.process_chunk(_noise_chunk(rms=100.0, seed=seed), current_frame=frame)
			frame += 100

		assert detector.ambient_peak > detector.ambient_rms
		# Gaussian noise over 100 samples peaks ~2.5-3.5 sigma: 8-11 dB of crest.
		crest_db = 20.0 * numpy.log10(detector.ambient_peak / detector.ambient_rms)
		assert 5.0 < crest_db < 15.0

	def test_constant_chunk_keeps_peak_equal_to_rms (self) -> None:

		"""A constant chunk has peak == RMS, so the attack gate is unchanged from
		the historical RMS-derived behaviour.  This is what makes the tracking a
		pure refinement rather than a re-tuning of every existing configuration."""

		detector = _make_detector(threshold_db=20.0, sample_rate=1000, chunk_size=100)
		detector.process_chunk(_loud_chunk(amplitude=100), current_frame=100)

		assert detector.ambient_peak == pytest.approx(detector.ambient_rms)
		assert detector.attack_amplitude_threshold == pytest.approx(
			100.0 * 10.0 ** (20.0 / 20.0)
		)

	def test_attack_gate_is_derived_from_the_peak_ema (self) -> None:

		"""The gate is ambient_peak x 10^(threshold_db/20) — not ambient_rms."""

		detector = _make_detector(threshold_db=12.0, sample_rate=1000, chunk_size=100)

		frame = 100
		for seed in range(12):
			detector.process_chunk(_noise_chunk(rms=100.0, seed=seed), current_frame=frame)
			frame += 100

		assert detector.attack_amplitude_threshold == pytest.approx(
			detector.ambient_peak * 10.0 ** (12.0 / 20.0)
		)
		assert detector.attack_amplitude_threshold > detector.ambient_rms * 10.0 ** (12.0 / 20.0)

	def test_attack_gate_still_dominates_tail_gate_under_noise (self) -> None:

		"""The documented invariant (attack >= tail, so the trimmed onset can never
		precede the tail end) must survive the two gates living in different
		domains — it holds because ambient_peak >= ambient_rms and the open
		threshold is above the close threshold."""

		detector = _make_detector(
			threshold_db=20.0, release_threshold_db=3.0,
			sample_rate=1000, chunk_size=100,
		)

		frame = 100
		for seed in range(12):
			detector.process_chunk(_noise_chunk(rms=100.0, seed=seed), current_frame=frame)
			frame += 100

		assert detector.attack_amplitude_threshold > detector.tail_amplitude_threshold

	def test_ema_pair_stays_in_lockstep_when_ambient_moves (self) -> None:

		"""Both EMAs must advance from the same chunks under the same seeding, or
		ambient_peak could fall below ambient_rms and invert the gate invariant."""

		detector = _make_detector(
			threshold_db=40.0, floor_adaptation=0.5, sample_rate=1000, chunk_size=100,
		)

		frame = 100
		for seed in range(20):
			# Ambient level steps up and down; peak must shadow rms throughout.
			level = 40.0 if seed % 2 else 400.0
			detector.process_chunk(_noise_chunk(rms=level, seed=seed), current_frame=frame)
			frame += 100
			assert detector.ambient_peak >= detector.ambient_rms


class TestFalseTriggerHeadIsTrimmed:

	"""Regression for the 2026-07-25 snare captures.

	In a quiet room the noise floor's chunk-to-chunk wobble can clear a small
	threshold_db, opening a recording on room tone tens of milliseconds before the
	real strike.  When the strike then lands inside hold_seconds the segment is not
	discarded, it is merely prepended with a silent head — and the leading trim,
	the thing that should remove it, could not: an RMS-derived gate compared against
	per-sample magnitudes sits a crest factor too low and finds "signal" in the
	noise at sample 0.

	The precondition is exact: an RMS-derived gate fails only when threshold_db is
	BELOW the ambient's crest factor, because the gate then lands under the noise's
	own peaks.  The captures that exposed this ran threshold_db 9 against room tone
	with ~9-13 dB of crest.  These tests therefore use a threshold_db under the
	crest of the noise they generate — at 12 dB or more the old gate happened to
	clear the noise and the defect is invisible, which is why it survived so long.
	"""

	# Below the ~9-11 dB crest of the Gaussian ambient these tests build.
	_THRESHOLD_DB: float = 6.0

	@staticmethod
	def _segment_with_room_tone_head () -> tuple[numpy.ndarray, int]:

		"""Return (segment, head_samples): room tone, then a loud hit decaying."""

		head = _noise_chunk(n=600, rms=100.0, seed=7)
		hit = _noise_chunk(n=900, rms=20000.0, seed=8)
		# Decay the hit so the tail edge behaves like real material.
		hit = (hit * numpy.linspace(1.0, 0.05, hit.size)).astype(numpy.int16)

		return numpy.concatenate([head, hit]).reshape(-1, 1), head.size

	def test_room_tone_head_is_trimmed_by_the_peak_derived_gate (self) -> None:

		"""The onset lands at the hit, not in the noise that preceded it."""

		detector = _make_detector(
			threshold_db=self._THRESHOLD_DB, sample_rate=1000, chunk_size=100,
		)

		frame = 100
		for seed in range(12):
			detector.process_chunk(_noise_chunk(rms=100.0, seed=seed), current_frame=frame)
			frame += 100

		segment, head_samples = self._segment_with_room_tone_head()

		trimmed = subsample.trim.trim_silence(
			segment,
			detector.tail_amplitude_threshold,
			lead_amplitude_threshold=detector.attack_amplitude_threshold,
		)

		# The head is gone: what remains cannot be longer than the hit itself.
		assert trimmed.shape[0] <= segment.shape[0] - head_samples
		# And nothing of the hit was lost — the peak survives intact.
		assert numpy.max(numpy.abs(trimmed)) == numpy.max(numpy.abs(segment))

	def test_an_rms_derived_gate_would_have_kept_the_head (self) -> None:

		"""Pins the actual cause.  Had the gate stayed in the RMS domain the same
		segment would keep its room-tone head, so this is the difference the peak
		EMA makes — not an artefact of the test's thresholds."""

		detector = _make_detector(
			threshold_db=self._THRESHOLD_DB, sample_rate=1000, chunk_size=100,
		)

		frame = 100
		for seed in range(12):
			detector.process_chunk(_noise_chunk(rms=100.0, seed=seed), current_frame=frame)
			frame += 100

		segment, head_samples = self._segment_with_room_tone_head()
		rms_derived_gate = detector.ambient_rms * 10.0 ** (self._THRESHOLD_DB / 20.0)

		trimmed = subsample.trim.trim_silence(
			segment,
			detector.tail_amplitude_threshold,
			lead_amplitude_threshold=rms_derived_gate,
		)

		assert trimmed.shape[0] > segment.shape[0] - head_samples
