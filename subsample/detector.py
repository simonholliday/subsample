"""Ambient noise tracking and recording trigger logic.

Uses an exponential moving average (EMA) of chunk RMS to model the ambient
noise floor. A signal triggers recording when it exceeds the ambient floor
by the configured SNR threshold. A "hold time" prevents premature cutoff
during brief pauses in the signal.

State machine:
  WARMUP -> IDLE        after warmup_seconds worth of chunks
  IDLE   -> RECORDING   when SNR threshold is exceeded
  RECORDING -> IDLE     when signal has been below threshold for hold_time seconds
                        (returns the (start_frame, end_frame) recording boundary)
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

		self._state: DetectorState = DetectorState.WARMUP
		self._ambient_rms: float = 0.0
		self._hold_chunks_remaining: int = 0
		self._recording_start_frame: int = 0

	@property
	def state (self) -> DetectorState:

		"""Current detector state."""

		return self._state

	@property
	def ambient_rms (self) -> float:

		"""Current ambient noise RMS estimate."""

		return self._ambient_rms

	def process_chunk (
		self,
		chunk: numpy.ndarray,
		current_frame: int,
	) -> typing.Optional[tuple[int, int]]:

		"""Process one chunk of audio.

		Updates the ambient EMA, advances the state machine, and returns a
		(start_frame, end_frame) pair when a recording segment completes.

		Args:
			chunk:         Audio samples for this chunk (int16).
			current_frame: Absolute frame index of the *end* of this chunk.

		Returns:
			(start_frame, end_frame) when a recording ends, otherwise None.
		"""

		chunk_rms = _compute_rms(chunk)
		self._update_ambient(chunk_rms)

		if self._state == DetectorState.WARMUP:
			return self._handle_warmup()

		if self._state == DetectorState.IDLE:
			return self._handle_idle(chunk_rms, current_frame)

		if self._state == DetectorState.RECORDING:
			return self._handle_recording(chunk_rms, current_frame)

		return None  # unreachable, satisfies mypy

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
		chunk_rms: float,
		current_frame: int,
	) -> typing.Optional[tuple[int, int]]:

		"""Check whether the current chunk exceeds the SNR threshold."""

		if self._exceeds_threshold(chunk_rms):
			self._state = DetectorState.RECORDING
			self._recording_start_frame = current_frame - self._chunk_size
			self._hold_chunks_remaining = self._hold_chunks_total

		return None

	def _handle_recording (
		self,
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

		if self._exceeds_threshold(chunk_rms):
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

	def _exceeds_threshold (self, chunk_rms: float) -> bool:

		"""True if chunk_rms is more than snr_threshold_db above the ambient floor."""

		ambient = max(self._ambient_rms, _AMBIENT_FLOOR)

		if chunk_rms <= _AMBIENT_FLOOR:
			return False

		snr_db = 20.0 * math.log10(chunk_rms / ambient)

		return float(snr_db) >= self._cfg.snr_threshold_db


def _compute_rms (chunk: numpy.ndarray) -> float:

	"""Compute the root-mean-square of an audio chunk.

	Converts to float64 before squaring to avoid int16 overflow.
	"""

	samples = chunk.astype(numpy.float64)
	return float(numpy.sqrt(numpy.mean(samples ** 2)))
