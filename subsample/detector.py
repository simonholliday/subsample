"""Ambient noise tracking and recording trigger logic.

Uses an exponential moving average (EMA) of chunk RMS to model the ambient
noise floor. A signal triggers recording when it exceeds the ambient floor
by the configured SNR threshold. A "hold time" prevents premature cutoff
during brief pauses in the signal.

By default one SNR threshold governs both the start and the end of a recording.
An optional lower ``release_threshold_db`` decouples them (a Schmitt-trigger
hysteresis pair): the recording still starts on the loud ``snr_threshold_db``
attack, but ends only once the tail has decayed to the quieter release level, so
a long decay (cymbal, ride, gong) rings out toward the noise floor instead of
being cut short.  An optional ``retrigger_threshold_db`` ends a recording the
moment the next hit lands (a sharp rise over the decaying tail), so a tail that
background noise holds above the release level cannot merge into the next sample.

State machine:
  WARMUP -> IDLE        after warmup_seconds worth of chunks
  IDLE   -> RECORDING   when the SNR (open) threshold is exceeded
  RECORDING -> IDLE     when signal has been below the CLOSE threshold
                        (release_threshold_db, or snr_threshold_db if unset) for
                        hold_time seconds — returns the (start, end) boundary
  RECORDING -> RECORDING when a re-trigger rise marks the next hit: returns the
                        finished (start, end) boundary and immediately opens a
                        new recording at the onset (only when retrigger is set)
  RECORDING -> IDLE     also force-triggered when recording length reaches
                        max_recording_frames to prevent circular buffer overwrite
"""

import enum
import logging
import math
import typing

import numpy

import subsample.config


_log = logging.getLogger(__name__)


# Minimum ambient RMS before threshold comparisons are meaningful.
# Prevents division-by-zero and spurious triggers in near-silence.
_AMBIENT_FLOOR: float = 1e-6


# EMA smoothing for the decaying-tail follower used by re-trigger detection.
# Faster than the ambient EMA so it hugs the decay tightly — a genuine re-strike
# then stands out as a sharp rise, while slow amplitude modulation (partials
# beating) is tracked out rather than mistaken for a new hit.
_TAIL_FOLLOWER_ALPHA: float = 0.3


# Minimum recording length, in seconds, before re-trigger can fire.  Spans the
# attack transient (whose own peak would otherwise read as a rise over the
# quieter onset chunk) and guarantees a re-triggered split cannot produce a
# sub-perceptual fragment.  The effective guard is max(this, hold_time).
_MIN_RETRIGGER_GUARD_SECONDS: float = 0.1


# Sub-chunk onset refinement.  The level trigger quantises a recording's start to
# the beginning of the chunk that crossed the threshold, which can precede the
# actual strike by up to a whole chunk.  When a recording opens (or re-triggers),
# the start is pulled forward to where the transient rises WITHIN that chunk,
# measured as a fraction of the chunk's OWN peak (robust to the noise floor and to
# a decaying previous sound sharing the chunk).  A small backoff keeps the attack
# foot; the sample-precise leading trim + fade run afterwards as usual.
_ONSET_FRACTION: float = 0.1
_ONSET_BACKOFF_SECONDS: float = 0.0003


class DetectorState (enum.Enum):

	WARMUP = "warmup"
	IDLE = "idle"
	RECORDING = "recording"


class LevelDetector:

	"""Detects audio events by comparing instantaneous level to an adaptive ambient floor.

	The ambient floor is tracked via an EMA so it adjusts slowly over time,
	ensuring that a gradually rising noise floor (e.g., traffic building up)
	is reflected without masking real signals.
	"""

	def __init__ (
		self,
		cfg: subsample.config.DetectionConfig,
		sample_rate: int,
		chunk_size: int,
		max_recording_frames: int = 0,
	) -> None:

		"""Initialise the detector.

		Args:
			cfg:                  Detection configuration (thresholds, timing, EMA alpha).
			sample_rate:          Audio sample rate in Hz, used to convert time to frames.
			chunk_size:           Number of frames per chunk, used for the same conversion.
			max_recording_frames: Force-end any recording that reaches this length, to
			                      prevent the circular buffer from overwriting its own
			                      start. Pass the buffer's total frame capacity here.
			                      0 disables the check (no limit).
		"""

		self._cfg = cfg
		self._chunk_size = chunk_size
		self._max_recording_frames = max_recording_frames

		# Derived frame counts for warmup and hold-time
		chunks_per_second = sample_rate / chunk_size
		self._warmup_chunks_remaining: int = round(cfg.warmup_seconds * chunks_per_second)
		self._hold_chunks_total: int = max(1, round(cfg.hold_time * chunks_per_second))

		# CLOSE threshold: a recording ends when the tail falls below this many dB
		# over ambient (release_threshold_db when set — decoupled from the louder
		# start — otherwise snr_threshold_db, reproducing the historical behaviour).
		self._close_threshold_db: float = (
			cfg.release_threshold_db if cfg.release_threshold_db is not None
			else cfg.snr_threshold_db
		)

		# Re-trigger: precompute the linear rise ratio and the min-segment guard.
		self._retrigger_ratio: typing.Optional[float] = (
			10.0 ** (cfg.retrigger_threshold_db / 20.0)
			if cfg.retrigger_threshold_db is not None else None
		)
		self._retrigger_guard_chunks: int = max(
			self._hold_chunks_total,
			round(_MIN_RETRIGGER_GUARD_SECONDS * chunks_per_second),
		)
		self._onset_backoff_samples: int = round(_ONSET_BACKOFF_SECONDS * sample_rate)

		self._state: DetectorState = DetectorState.WARMUP
		self._ambient_rms: float = 0.0
		self._hold_chunks_remaining: int = 0
		self._recording_start_frame: int = 0

		# Per-recording state for the re-trigger tail follower / min-segment guard.
		self._recording_chunks: int = 0
		self._tail_follower: float = 0.0

	@property
	def state (self) -> DetectorState:

		"""Current detector state."""

		return self._state

	@property
	def ambient_rms (self) -> float:

		"""Current ambient noise RMS estimate."""

		return self._ambient_rms

	@property
	def tail_amplitude_threshold (self) -> float:

		"""Per-sample amplitude gate for the TRAILING (tail) edge of trim_silence.

		Built from the SAME close threshold and (floored) ambient the detector used
		to end the recording — release_threshold_db when set, else snr_threshold_db.
		The single source of truth for the tail so the trimmer cannot re-cut a
		decayed tail back up to a louder level than the detector ended on (which
		would undo the whole point of a decoupled release threshold).
		"""

		ambient = max(self._ambient_rms, _AMBIENT_FLOOR)
		return float(ambient * (10.0 ** (self._close_threshold_db / 20.0)))

	@property
	def attack_amplitude_threshold (self) -> float:

		"""Per-sample amplitude gate for the LEADING (attack) edge of trim_silence.

		Built from snr_threshold_db (the START/open trigger level) and the floored
		ambient — deliberately INDEPENDENT of release_threshold_db.  The attack edge
		must trim tight to the transient regardless of how low the tail threshold is
		set: a low release preserves a long decay, but it must not drag low-level
		room tone into the front of the next sample.  Always >= tail_amplitude_
		threshold (snr >= release), so the trimmed onset never precedes the tail end.
		"""

		ambient = max(self._ambient_rms, _AMBIENT_FLOOR)
		return float(ambient * (10.0 ** (self._cfg.snr_threshold_db / 20.0)))

	def process_chunk (
		self,
		chunk: numpy.ndarray,
		current_frame: int,
	) -> typing.Optional[tuple[int, int]]:

		"""Process one chunk of audio.

		Updates the ambient EMA, advances the state machine, and returns a
		(start_frame, end_frame) pair when a recording segment completes.

		Args:
			chunk:         Integer PCM samples for this chunk (int16 for 16-bit
			               audio, int32 for 24/32-bit audio).
			current_frame: Absolute frame index of the *end* of this chunk.

		Returns:
			(start_frame, end_frame) when a recording ends, otherwise None.
		"""

		chunk_rms = _compute_rms(chunk)

		# Ambient is updated during WARMUP (calibration) and — in IDLE — only
		# for chunks that do NOT trigger, so a loud transient is never folded
		# into the noise floor before its own SNR test (which would deflate the
		# measured SNR and miss sharp hits).  RECORDING never updates ambient.
		if self._state == DetectorState.WARMUP:
			self._update_ambient(chunk_rms)
			return self._handle_warmup()

		if self._state == DetectorState.IDLE:
			return self._handle_idle(chunk, chunk_rms, current_frame)

		if self._state == DetectorState.RECORDING:
			return self._handle_recording(chunk, chunk_rms, current_frame)

		# Exhaustive over DetectorState today; raise (rather than silently
		# returning None = "no recording") if a new state is ever added without
		# a handler here.
		raise AssertionError(f"unhandled detector state {self._state!r}")

	# --- State handlers ---

	def _handle_warmup (self) -> typing.Optional[tuple[int, int]]:

		"""Tick the warmup counter; transition to IDLE when complete."""

		self._warmup_chunks_remaining -= 1

		if self._warmup_chunks_remaining <= 0:
			self._state = DetectorState.IDLE
			_log.info("Ambient calibration complete. Listening…")

		return None

	def _handle_idle (
		self,
		chunk: numpy.ndarray,
		chunk_rms: float,
		current_frame: int,
	) -> typing.Optional[tuple[int, int]]:

		"""Check whether the current chunk exceeds the SNR threshold.

		The SNR test runs against the ambient floor as measured *before* this
		chunk.  Only a non-triggering (genuinely background) chunk then updates
		the ambient EMA — a trigger is excluded so the transient never raises
		the floor it is being compared against.
		"""

		if self._exceeds_threshold(chunk_rms):
			self._state = DetectorState.RECORDING
			chunk_start = current_frame - self._chunk_size
			self._recording_start_frame = max(0, chunk_start + self._onset_offset(chunk))
			self._hold_chunks_remaining = self._hold_chunks_total
			# This triggering chunk is frame 1 of the new recording; seed the tail
			# follower at the attack level so re-trigger measures rises above it.
			self._recording_chunks = 1
			self._tail_follower = chunk_rms
			return None

		self._update_ambient(chunk_rms)
		return None

	def _handle_recording (
		self,
		chunk: numpy.ndarray,
		chunk_rms: float,
		current_frame: int,
	) -> typing.Optional[tuple[int, int]]:

		"""Extend or end the current recording based on signal level and hold time.

		Returns the (start, end) boundary when the recording is finalised.
		Also force-ends if the recording has reached max_recording_frames, to
		prevent the circular buffer from overwriting the beginning of the recording.
		"""

		# Force-end check takes priority — buffer integrity over hold time
		if self._max_recording_frames > 0:
			recording_length = current_frame - self._recording_start_frame
			if recording_length >= self._max_recording_frames:
				end_frame = current_frame
				start_frame = self._recording_start_frame
				self._state = DetectorState.IDLE
				_log.info(
					"Recording force-ended: reached buffer capacity (%d frames)",
					recording_length,
				)
				return (start_frame, end_frame)

		self._recording_chunks += 1

		# Re-trigger: a sharp rise over the decaying tail is the next hit.  Finalise
		# the current sample at this chunk's start (where the new attack begins) and
		# immediately open a fresh recording, staying in RECORDING.  Suppressed until
		# the min-segment guard elapses, which spans the attack so its own peak — a
		# rise over the quieter onset chunk — cannot split the hit off from its tail.
		if (
			self._retrigger_ratio is not None
			and self._recording_chunks > self._retrigger_guard_chunks
			and self._exceeds_retrigger(chunk_rms)
		):
			boundary = max(0, current_frame - self._chunk_size + self._onset_offset(chunk))
			start_frame = self._recording_start_frame
			self._recording_start_frame = boundary
			self._recording_chunks = 1
			self._hold_chunks_remaining = self._hold_chunks_total
			self._tail_follower = chunk_rms
			return (start_frame, boundary)

		# Track the decaying tail — after the re-trigger test, so the current chunk
		# is not folded into the baseline it was just measured against.
		self._update_follower(chunk_rms)

		if self._exceeds_close_threshold(chunk_rms):
			# Signal is still present — reset hold countdown
			self._hold_chunks_remaining = self._hold_chunks_total
			return None

		# Signal has dropped — count down the hold time
		self._hold_chunks_remaining -= 1

		if self._hold_chunks_remaining > 0:
			return None

		# Hold time expired — recording is complete
		end_frame = current_frame
		start_frame = self._recording_start_frame

		self._state = DetectorState.IDLE

		return (start_frame, end_frame)

	# --- Helpers ---

	def _update_ambient (self, chunk_rms: float) -> None:

		"""Update the ambient EMA with the current chunk RMS.

		During WARMUP the EMA is seeded directly to avoid a long ramp from zero.
		"""

		if self._ambient_rms < _AMBIENT_FLOOR:
			# Seed the EMA on the first meaningful chunk rather than smoothing from zero
			self._ambient_rms = max(chunk_rms, _AMBIENT_FLOOR)
		else:
			alpha = self._cfg.ema_alpha
			self._ambient_rms = alpha * chunk_rms + (1.0 - alpha) * self._ambient_rms

	def _onset_offset (self, chunk: numpy.ndarray) -> int:

		"""Sample offset within a triggering chunk of the transient onset.

		The level trigger quantises a recording's start to the chunk boundary, which
		can precede the actual strike by up to a chunk.  This finds where the
		transient rises within the chunk — as a fraction of the chunk's OWN peak, so
		it is robust to the noise floor and to a decaying previous sound sharing the
		chunk — then backs off a little to keep the attack foot.  Returns 0 for a
		flat or degenerate chunk (e.g. a constant test signal), leaving the boundary
		at the chunk start.
		"""

		samples = chunk.astype(numpy.float64)
		magnitude = numpy.abs(samples) if samples.ndim == 1 else numpy.max(numpy.abs(samples), axis=-1)

		if magnitude.size == 0:
			return 0

		peak = float(numpy.max(magnitude))
		if peak <= 0.0:
			return 0

		above = numpy.where(magnitude >= peak * _ONSET_FRACTION)[0]
		if above.size == 0:
			return 0

		return max(0, int(above[0]) - self._onset_backoff_samples)

	def _snr_db_over_ambient (self, chunk_rms: float) -> float:

		"""Level of this chunk in dB relative to the ambient floor.

		Returns -inf for a sub-floor chunk so any finite threshold comparison is
		False without a special case at each call site.
		"""

		if chunk_rms <= _AMBIENT_FLOOR:
			return -math.inf

		ambient = max(self._ambient_rms, _AMBIENT_FLOOR)
		return 20.0 * math.log10(chunk_rms / ambient)

	def _exceeds_threshold (self, chunk_rms: float) -> bool:

		"""True if chunk_rms is at least snr_threshold_db above ambient (OPEN/start)."""

		return self._snr_db_over_ambient(chunk_rms) >= self._cfg.snr_threshold_db

	def _exceeds_close_threshold (self, chunk_rms: float) -> bool:

		"""True while the tail is still above the CLOSE threshold (end/hold test).

		Uses release_threshold_db when configured, otherwise snr_threshold_db — so
		with no release set this is identical to _exceeds_threshold.
		"""

		return self._snr_db_over_ambient(chunk_rms) >= self._close_threshold_db

	def _exceeds_retrigger (self, chunk_rms: float) -> bool:

		"""True if chunk_rms rises retrigger_threshold_db over the decaying tail.

		Compared against the tail follower as it stood *before* this chunk was
		folded in, so a genuine re-strike is measured against the tail it interrupts.
		"""

		if self._retrigger_ratio is None:
			return False

		follower = max(self._tail_follower, _AMBIENT_FLOOR)
		return chunk_rms >= follower * self._retrigger_ratio

	def _update_follower (self, chunk_rms: float) -> None:

		"""Advance the decaying-tail EMA used by re-trigger detection.

		Seeded directly on the first meaningful chunk (as the ambient EMA is) to
		avoid a ramp from zero.
		"""

		if self._tail_follower < _AMBIENT_FLOOR:
			self._tail_follower = max(chunk_rms, _AMBIENT_FLOOR)
		else:
			alpha = _TAIL_FOLLOWER_ALPHA
			self._tail_follower = alpha * chunk_rms + (1.0 - alpha) * self._tail_follower


def _compute_rms (chunk: numpy.ndarray) -> float:

	"""Compute the root-mean-square of an audio chunk.

	Converts to float32 before squaring to avoid integer overflow while using
	half the memory of float64.  int16 values fit float32's 23-bit mantissa
	exactly; int32 (24/32-bit capture) magnitudes near 2**31 exceed it, so the
	cast is lossy there — but the error is far below the SNR threshold's
	resolution, so it is immaterial to the level comparison.
	"""

	samples = chunk.astype(numpy.float32)
	return float(numpy.sqrt(numpy.mean(samples ** 2)))
