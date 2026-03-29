"""Sample transform pipeline — derivative audio production and caching.

Produces *derivative* versions of existing samples (pitch-shifted, filtered,
reversed, saturated, time-stretched, etc.) without modifying the originals.
Derivatives are purely in-memory and regenerated on demand — they are never
written to disk.

Architecture overview
---------------------

TransformSpec / TransformKey
    An immutable, hashable description of what transforms to apply to which
    sample.  The composite (sample_id, spec) is the unique identity of one
    derivative.  Steps are stored in *declaration order* — the user controls
    the signal chain via the ``process:`` list in the MIDI map.

TransformResult
    The completed output: float32 audio at the original channel count plus a
    LevelResult so the player can apply the same gain formula it uses for
    originals (no separate gain baking — velocity sensitivity stays dynamic).

TransformCache
    Thread-safe, memory-bounded store.  Eviction strategy: *parent-priority
    FIFO* — when the budget is exceeded, all derivatives of the oldest parent
    are evicted together.  This keeps variant sets intact for recent samples
    rather than fragmenting them.  Cascade eviction (remove_parent) is
    triggered automatically when the instrument library evicts a parent.

TransformProcessor
    ThreadPoolExecutor worker pool.  Mirrors the SampleProcessor pattern:
    enqueue() submits a job and returns immediately; workers convert the
    source PCM to float32, apply the registered transform chain in
    declaration order, compute a LevelResult, and call on_complete.  New
    transform types are registered in TransformProcessor._HANDLERS — no
    other code changes are needed.

TransformManager
    Single coordination point for the player and cli.py.  Handles:
      - on_sample_added()   : auto-enqueue configured transforms for new samples
      - on_parent_evicted() : propagate instrument-library eviction to the cache
      - on_bpm_change()     : invalidate and re-enqueue all time-stretch variants
      - get_pitched()       : player look-up; enqueues on miss, returns None
      - get_at_bpm()        : player look-up for time-stretch variants
      - get_variant()       : generic look-up by arbitrary TransformSpec

Data flow
---------

  SampleRecord added to InstrumentLibrary
      → TransformManager.on_sample_added()
          → TransformProcessor.enqueue(record, spec)   [auto-pitch / auto-BPM]
              → worker: _pcm_to_float32 → handler chain → compute_level
                  → TransformCache.put(result)
                      → on_complete callback

  MIDI note trigger
      → player builds spec via spec_from_process(assignment.process, ...)
      → TransformManager.get_variant(sample_id, spec)
          → hit:  return TransformResult  (player applies gain + pan)
          → miss: enqueue, return None  (player falls back)
      → TransformManager.get_base(sample_id)
          → hit:  return TransformResult  (float32 copy, no DSP)
          → miss: player converts from int PCM on this trigger

  InstrumentLibrary evicts sample_id
      → TransformManager.on_parent_evicted([sample_id])
          → TransformCache.remove_parent(sample_id)

  Target BPM changes
      → TransformManager.on_bpm_change(new_bpm)
          → TransformCache.remove_by_step_type(TimeStretch)
          → filter by tempo_bpm > 0
          → TransformProcessor.enqueue_bpm_change(qualifying_records, new_bpm)

How to add a new transform type
--------------------------------

1.  Define a frozen dataclass for the transform parameters:

        @dataclasses.dataclass(frozen=True)
        class MyTransform:
            my_param: float

2.  Write an apply function matching _ApplyFn:

        def _apply_my_transform (
            audio:       numpy.ndarray,
            sample_rate: int,
            record:      subsample.library.SampleRecord,
            step:        MyTransform,
        ) -> numpy.ndarray:
            # audio is float32, shape (n_frames, channels).
            # Operate on all channels independently.
            # Return float32 array of the same shape.
            ...

3.  Register the handler in TransformProcessor._HANDLERS:

        TransformProcessor._HANDLERS[MyTransform] = _apply_my_transform

4.  Add the new type to the TransformStep union and to the name mapping
    in spec_from_process() so it can be used in MIDI map process: lists.

5.  Add config fields to TransformConfig in config.py if tuning is needed.

6.  Wire auto-enqueue logic in TransformManager.on_sample_added() if the
    transform should fire automatically when new samples arrive.

7.  Update README.md (user-facing) and README-AGENTS.md (agent-facing) to
    document the new transform type.
"""

import collections
import concurrent.futures
import dataclasses
import math
import hashlib
import logging
import os
import pathlib
import struct
import tempfile
import threading
import typing

import librosa
import numpy
import pyrubberband
import scipy.signal

import subsample.analysis
import subsample.library
import subsample.query

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transform step dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PitchShift:

	"""Shift all channels to a target MIDI note.

	target_midi_note: MIDI note number (0–127).  Middle C = 60, A4 = 69.
	The shift in semitones is computed from the parent sample's
	dominant_pitch_hz (via librosa.hz_to_midi) to the target note frequency.
	Processed by _apply_pitch() using Rubber Band (pyrubberband, offline mode).
	"""

	target_midi_note: int



@dataclasses.dataclass(frozen=True)
class TimeStretch:

	"""Time-stretch the audio to match a target tempo without changing pitch.

	Uses beat-quantized non-linear stretching: detected onsets are snapped to
	a grid at target_bpm / resolution, then pyrubberband.timemap_stretch()
	applies the mapping in a single high-quality offline Rubber Band call
	(--fine --smoothing).
	Samples with fewer than 2 onsets receive a simple global time-stretch.

	target_bpm: The desired playback tempo in BPM.
	resolution: Grid subdivision (1=whole, 2=half, 4=quarter, 8=eighth,
	            16=sixteenth).  Higher values give finer onset alignment.
	"""

	target_bpm:  float
	resolution:  int = 16


@dataclasses.dataclass(frozen=True)
class Reverse:

	"""Reverse the audio along the time axis.

	No parameters.  Produces a contiguous copy (negative-stride views from
	numpy slicing can break downstream C-based DSP like pyrubberband).
	"""


@dataclasses.dataclass(frozen=True)
class LowPassFilter:

	"""Low-pass filter — attenuates frequencies above the cutoff.

	freq:          Cutoff frequency in Hz.
	resonance_db:  Peak boost at the cutoff in dB.  0 = flat Butterworth.
	               Higher values create a resonant peak (Chebyshev Type I).
	               Clamped to [0, 24] for stability.
	"""

	freq:          float
	resonance_db:  float = 0.0


@dataclasses.dataclass(frozen=True)
class HighPassFilter:

	"""High-pass filter — attenuates frequencies below the cutoff.

	freq:          Cutoff frequency in Hz.
	resonance_db:  Peak boost at the cutoff in dB.  0 = flat Butterworth.
	"""

	freq:          float
	resonance_db:  float = 0.0


@dataclasses.dataclass(frozen=True)
class BandPassFilter:

	"""Band-pass filter — passes a band centred on freq with width set by Q.

	freq:          Centre frequency in Hz.
	q:             Quality factor controlling bandwidth.  Lower = wider band.
	               Q = 0.707 (1/sqrt(2)) is the Butterworth response (~2 octaves).
	               Q = 1.414 is ~1 octave.  Default 0.7 gives a wide, musical
	               band like a console mid-sweep.
	resonance_db:  Peak boost at the centre in dB.  0 = flat Butterworth.
	"""

	freq:          float
	q:             float = 0.7
	resonance_db:  float = 0.0


@dataclasses.dataclass(frozen=True)
class Saturate:

	"""Soft-clip saturation via tanh with level compensation.

	amount_db:  Drive level in dB.  0 = no effect.  6 = moderate warmth.
	            12+ = heavy distortion.  The output is level-compensated so
	            peak amplitude is preserved.
	"""

	amount_db: float


@dataclasses.dataclass(frozen=True)
class Compress:

	"""Feed-forward dynamic range compressor with soft knee and look-ahead.

	When used without parameters (``compress: true``), threshold, attack, and
	release adapt automatically to each sample's analysis data:

	  - threshold: 6 dB below the sample's peak level (always engages)
	  - attack:    slow for percussive samples (preserves transient snap),
	               fast for gradual onsets (no transient to protect)
	  - release:   short for quick-decay samples, long for sustained sounds

	Set any parameter explicitly to override the auto value for that parameter.

	For the opposite effect (squash transients, raise reverb tails), use a
	fast attack (< 1 ms), high ratio (10+), and low threshold (-30 dB).

	threshold_db:   Level above which compression begins (dBFS).
	                None = auto (6 dB below sample peak).
	ratio:          Compression ratio.  4:1 = moderate, 10:1+ = heavy.
	                1.0 = no compression.
	attack_ms:      How fast gain reduction engages (ms).  Slow = more punch.
	                None = auto (adapts to sample onset character).
	release_ms:     How fast gain recovers after the signal drops (ms).
	                None = auto (adapts to sample decay character).
	knee_db:        Soft knee width in dB.  0 = hard knee.  6 = smooth transition.
	makeup_db:      Output gain boost in dB to compensate for level reduction.
	lookahead_ms:   Delay audio so the gain envelope anticipates transients (ms).
	                0 = no look-ahead.  5+ = catches fast peaks.
	"""

	threshold_db:  typing.Optional[float] = None
	ratio:         float = 4.0
	attack_ms:     typing.Optional[float] = None
	release_ms:    typing.Optional[float] = None
	knee_db:       float = 6.0
	makeup_db:     float = 0.0
	lookahead_ms:  float = 0.0


@dataclasses.dataclass(frozen=True)
class Limit:

	"""Brickwall limiter — shortcut to the compressor with limiter presets.

	Internally uses the same feed-forward compressor with hard knee,
	ratio 100:1, near-instant attack, and look-ahead to catch transients
	before they overshoot.  Only threshold, release, and look-ahead are
	user-adjustable; the remaining parameters are fixed for clean limiting.

	threshold_db:   Maximum output level (dBFS).
	release_ms:     How fast gain recovers after a peak (ms).
	lookahead_ms:   Anticipate peaks by this many ms (0 = none).
	"""

	threshold_db:  float = -1.0
	release_ms:    float = 50.0
	lookahead_ms:  float = 5.0


@dataclasses.dataclass(frozen=True)
class HpssHarmonic:

	"""Keep only the harmonic (tonal/sustained) component via HPSS.

	Removes percussive transients, preserving pitched and sustained content.
	Useful as a pre-filter before repitch to avoid pitch-shifting drum bleed,
	or as a creative effect to extract the "body" of a sound.
	"""


@dataclasses.dataclass(frozen=True)
class HpssPercussive:

	"""Keep only the percussive (transient) component via HPSS.

	Removes harmonic/tonal content, preserving clicks, hits, and transients.
	Useful as a pre-filter before beat_quantize for cleaner grid alignment,
	or as a creative effect to extract the rhythmic skeleton of a sound.
	"""


@dataclasses.dataclass(frozen=True)
class Gate:

	"""Noise gate — silences audio below a threshold.

	When used without parameters (``gate: true``), all settings adapt
	automatically to the sample's analysis data:

	  - threshold: 6 dB above the sample's noise floor (always engages)
	  - attack:    fast for percussive sounds (gate opens instantly),
	               slower for sustained sounds (avoids click)
	  - release:   short for percussive (quick close), long for sustained
	  - hold:      scaled to decay character (prevents chatter on decaying tails)
	  - lookahead: small for percussive (catches transient start), none for sustained

	Set any parameter explicitly to override the auto value.

	threshold_db:   Level below which audio is silenced (dBFS).
	                None = auto (noise_floor + 6 dB).
	attack_ms:      How fast the gate opens when signal exceeds threshold (ms).
	                None = auto (adapts to sample onset speed).
	release_ms:     How fast the gate closes when signal drops below threshold (ms).
	                None = auto (adapts to sample decay character).
	hold_ms:        Minimum time the gate stays open after signal drops (ms).
	                None = auto (adapts to sample decay).
	lookahead_ms:   Delay audio so the gate opens before the transient (ms).
	                None = auto (adapts to sample attack speed).
	"""

	threshold_db:  typing.Optional[float] = None
	attack_ms:     typing.Optional[float] = None
	release_ms:    typing.Optional[float] = None
	hold_ms:       typing.Optional[float] = None
	lookahead_ms:  typing.Optional[float] = None


@dataclasses.dataclass(frozen=True)
class Distort:

	"""Waveshaping distortion — aggressive harmonic saturation.

	Goes beyond the gentle tanh curve of ``saturate`` with multiple shaping
	modes: hard clipping, foldback, bit crushing, and sample-rate reduction.

	When used without parameters (``distort: true``), drive and tone adapt
	to the sample: peaky sounds get less drive (they already clip easily),
	bright sounds get more post-distortion filtering to tame added harmonics.

	mode:              Waveshaping algorithm.
	                   "hard_clip" — flat ceiling at ±1 (classic digital clip).
	                   "fold"      — signal folds back at ±1 (richer harmonics).
	                   "bit_crush" — quantize to fewer amplitude levels (lo-fi).
	                   "downsample" — reduce effective sample rate (aliasing).
	drive_db:          Input gain before waveshaping (dB).
	                   None = auto (adapts to crest factor).
	mix:               Dry/wet blend (0.0 = fully dry, 1.0 = fully wet).
	tone:              Post-distortion low-pass filter as fraction of Nyquist.
	                   0.0 = very dark, 1.0 = no filtering.
	                   None = auto (adapts to spectral rolloff).
	bit_depth:         Target bit depth for bit_crush mode (1–16).
	downsample_factor: Reduction factor for downsample mode (2–64).
	"""

	mode:              str                    = "hard_clip"
	drive_db:          typing.Optional[float] = None
	mix:               float                  = 1.0
	tone:              typing.Optional[float] = None
	bit_depth:         int                    = 8
	downsample_factor: int                    = 4


@dataclasses.dataclass(frozen=True)
class Reshape:

	"""Envelope reshaping — ADSR-style amplitude control.

	Reshapes the amplitude envelope of the sound: tighten a loose kick,
	truncate a reverb tail, add punch to a soft onset.  When used without
	parameters (``reshape: true``), the release time auto-adapts to the
	sample's natural decay, tightening the tail — the single most useful
	default for a sampler ("clean up the capture for playback").

	Parameters set to None mean "preserve the original envelope for this
	phase."  Only the phases you specify are reshaped.

	attack_ms:   Target attack time (ms).  None = preserve original onset.
	hold_ms:     Time to hold at peak before decay begins (ms).
	decay_ms:    Time from peak to sustain level (ms).  None = preserve.
	sustain:     Sustain level as fraction of peak (0.0–1.0).
	release_ms:  Fade-out time at the end of the sound (ms).
	             None = auto (adapts to sample decay character).
	"""

	attack_ms:   typing.Optional[float] = None
	hold_ms:     float                  = 0.0
	decay_ms:    typing.Optional[float] = None
	sustain:     float                  = 1.0
	release_ms:  typing.Optional[float] = None


# Union of all known transform step types.
# Extend this when adding new transforms (see "How to add a new transform type"
# in the module docstring).
TransformStep = typing.Union[
	PitchShift,
	Reverse,
	LowPassFilter,
	HighPassFilter,
	BandPassFilter,
	Saturate,
	Compress,
	Limit,
	HpssHarmonic,
	HpssPercussive,
	TimeStretch,
	Gate,
	Distort,
	Reshape,
]

# ---------------------------------------------------------------------------
# TransformSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TransformSpec:

	"""An ordered, hashable sequence of transforms to apply to a sample.

	Steps are stored in *declaration order* — the user controls the signal
	chain via the ``process:`` list in the MIDI map.  Different orderings
	produce different audio and different cache keys.

	An empty steps tuple is the identity spec (no transform applied).
	"""

	steps: tuple[TransformStep, ...]

# The identity spec: no transform steps.  Used for the base variant — a
# float32, peak-normalised copy of the original at the recorder's sample rate,
# ready for mixing.  Every sample gets one regardless of pitch stability.
_BASE_VARIANT_SPEC: TransformSpec = TransformSpec(steps=())

# ---------------------------------------------------------------------------
# TransformKey
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TransformKey:

	"""The unique identity of one derivative: parent sample + transform spec.

	Frozen and hashable — safe as a dict key and set member.
	"""

	sample_id: int
	spec: TransformSpec

# ---------------------------------------------------------------------------
# TransformResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TransformResult:

	"""A completed transform: derivative audio and its identity.

	audio:
	    float32 NumPy array, shape (n_frames, channels).  The channel count
	    matches the original recording — transforms operate on all channels
	    independently and do NOT mix down to mono.

	level:
	    LevelResult computed from the mono mix of the derivative audio.
	    Stored here so the player can apply the same gain formula it uses for
	    original SampleRecords:
	        norm_gain  = target_rms / level.rms
	        final_gain = min(norm_gain * vel_scale, 1.0 / level.peak)
	    At trigger time the player: multiplies by final_gain, pans to stereo
	    if mono, and appends a Voice.  No int→float conversion needed.

	duration:
	    Length of the derivative in seconds (may differ from the original if
	    time-stretched).
	"""

	key:      TransformKey
	audio:    numpy.ndarray
	duration: float
	level:    subsample.analysis.LevelResult

# ---------------------------------------------------------------------------
# TransformCache
# ---------------------------------------------------------------------------

class TransformCache:

	"""Thread-safe, memory-bounded cache of derivative audio buffers.

	Eviction strategy — parent-priority FIFO
	-----------------------------------------
	When the memory budget is exceeded, all derivatives of the *oldest parent*
	are evicted together before moving on to the next oldest parent.

	Rationale: a half-evicted pitch range (e.g. C3–A3 with B3 missing) is
	useless at runtime.  Evicting a whole parent at once keeps the remaining
	parents' sets complete and playable.

	Parent age is tracked by the first time any derivative for that parent
	was added (_parent_order deque, oldest at index 0).

	Cascade eviction
	-----------------
	When the instrument library evicts a parent sample, all its derivatives
	must be evicted immediately.  Call remove_parent(sample_id) from the
	on_complete callback (see cli.py TransformManager.on_parent_evicted).
	"""

	def __init__ (self, max_memory_bytes: int) -> None:

		self._index:        dict[TransformKey, TransformResult] = {}

		# sample_id → set of TransformKeys for that parent
		self._parent_index: dict[int, set[TransformKey]] = {}

		# Insertion order of parents (sample_id); oldest at left
		self._parent_order: collections.deque[int] = collections.deque()

		self._total_bytes:  int = 0
		self._max_bytes:    int = max_memory_bytes
		self._lock:         threading.Lock = threading.Lock()

	# --- Public API ---

	def put (self, result: TransformResult) -> list[TransformKey]:

		"""Store a derivative, evicting the oldest parent if over budget.

		Returns the list of TransformKeys evicted (empty if none were needed).
		If a single derivative exceeds the entire budget, a warning is logged
		and the result is still stored (newest capture is always kept).
		"""

		sample_id = result.key.sample_id
		new_bytes = result.audio.nbytes
		evicted: list[TransformKey] = []

		with self._lock:

			# Evict whole parent sets until there is room.
			while (
				self._parent_order
				and self._total_bytes + new_bytes > self._max_bytes
			):
				oldest_parent = self._parent_order[0]
				evicted += self._evict_parent_locked(oldest_parent)

			if new_bytes > self._max_bytes:
				_log.warning(
					"Transform variant for sample %d (%d bytes) exceeds budget (%d bytes); "
					"stored anyway — consider raising transform.max_memory_mb",
					sample_id, new_bytes, self._max_bytes,
				)

			# Register parent if this is its first derivative.
			if sample_id not in self._parent_index:
				self._parent_index[sample_id] = set()
				self._parent_order.append(sample_id)

			self._parent_index[sample_id].add(result.key)
			self._index[result.key] = result
			self._total_bytes += new_bytes

		if evicted:
			_log.debug(
				"Transform cache evicted %d variant(s) from parent %d to make room",
				len(evicted), evicted[0].sample_id,
			)

		return evicted

	def get (self, key: TransformKey) -> typing.Optional[TransformResult]:

		"""Return the cached result for a TransformKey, or None if absent."""

		with self._lock:
			return self._index.get(key)

	def get_base (self, sample_id: int) -> typing.Optional[TransformResult]:

		"""Convenience: look up the base variant (identity spec) for a sample.

		The base variant is a float32, peak-normalised copy at the recorder's
		sample rate, with no transform steps applied.  Returns None if not yet
		computed.
		"""

		key = TransformKey(sample_id=sample_id, spec=_BASE_VARIANT_SPEC)

		return self.get(key)

	def get_pitched (
		self,
		sample_id: int,
		midi_note: int,
	) -> typing.Optional[TransformResult]:

		"""Convenience: look up a pitch-only variant by sample ID and MIDI note."""

		key = TransformKey(
			sample_id=sample_id,
			spec=TransformSpec(steps=(PitchShift(target_midi_note=midi_note),)),
		)

		return self.get(key)

	def get_stretched (
		self,
		sample_id:  int,
		target_bpm: float,
		resolution: int = 16,
	) -> typing.Optional[TransformResult]:

		"""Convenience: look up a time-stretch-only variant by sample ID and BPM."""

		key = TransformKey(
			sample_id=sample_id,
			spec=TransformSpec(steps=(TimeStretch(
				target_bpm=target_bpm,
				resolution=resolution,
			),)),
		)

		return self.get(key)

	def has_variants (self, sample_id: int) -> bool:

		"""Return True if any derivatives are currently cached for this parent."""

		with self._lock:
			return bool(self._parent_index.get(sample_id))

	def list_variants (self, sample_id: int) -> list[TransformKey]:

		"""Return a snapshot of all TransformKeys cached for a parent."""

		with self._lock:
			return list(self._parent_index.get(sample_id, set()))

	def remove_parent (self, sample_id: int) -> list[TransformKey]:

		"""Cascade-evict all derivatives of a parent sample.

		Called when the instrument library evicts the parent.  Returns the list
		of TransformKeys removed (empty if none were cached for this parent).
		"""

		with self._lock:
			return self._evict_parent_locked(sample_id)

	def remove_by_step_type (self, step_type: type) -> list[TransformKey]:

		"""Remove all cached variants that include a given transform step type.

		Used to invalidate a class of transforms when their parameters change.
		Example: remove_by_step_type(TimeStretch) when the target BPM changes.

		Returns the list of TransformKeys removed.
		"""

		to_remove: list[TransformKey] = []

		with self._lock:
			for key in list(self._index):
				if any(isinstance(step, step_type) for step in key.spec.steps):
					to_remove.append(key)

			for key in to_remove:
				self._evict_key_locked(key)

		return to_remove

	@property
	def memory_used (self) -> int:
		"""Current total memory used by cached derivatives, in bytes."""
		with self._lock:
			return self._total_bytes

	@property
	def memory_limit (self) -> int:
		"""Maximum memory allowed for cached derivatives, in bytes."""
		return self._max_bytes

	def format_memory (self) -> str:

		"""Return a human-readable memory usage string for logging.

		Example: '23.4 / 50.0 MB, 53% free'
		"""

		used     = self.memory_used   # acquires lock once
		used_mb  = used              / (1024 * 1024)
		limit_mb = self._max_bytes   / (1024 * 1024)
		pct_free = int(100 * (1.0 - used / self._max_bytes)) if self._max_bytes > 0 else 100
		return f"{used_mb:.1f} / {limit_mb:.1f} MB, {pct_free}% free"

	# --- Internal helpers (callers must hold _lock) ---

	def _evict_parent_locked (self, sample_id: int) -> list[TransformKey]:

		"""Remove all derivatives for one parent.  Caller must hold _lock."""

		keys = self._parent_index.pop(sample_id, set())
		evicted: list[TransformKey] = []

		for key in keys:
			result = self._index.pop(key, None)
			if result is not None:
				self._total_bytes -= result.audio.nbytes
				evicted.append(key)

		try:
			self._parent_order.remove(sample_id)
		except ValueError:
			pass

		return evicted

	def _evict_key_locked (self, key: TransformKey) -> None:

		"""Remove one key, updating memory and parent index.  Caller must hold _lock."""

		result = self._index.pop(key, None)

		if result is None:
			return

		self._total_bytes -= result.audio.nbytes

		parent_keys = self._parent_index.get(key.sample_id)
		if parent_keys is not None:
			parent_keys.discard(key)

			if not parent_keys:
				del self._parent_index[key.sample_id]
				try:
					self._parent_order.remove(key.sample_id)
				except ValueError:
					pass

# ---------------------------------------------------------------------------
# VariantDiskCache
# ---------------------------------------------------------------------------

# Binary file format for cached variants:
#   Magic  (4 bytes): b"SSV1"
#   Channels (2 bytes, uint16, little-endian)
#   Sample rate (4 bytes, uint32, little-endian)
#   Frame count (4 bytes, uint32, little-endian)
#   Peak (4 bytes, float32, little-endian)
#   RMS (4 bytes, float32, little-endian)
#   Reserved (10 bytes, zero)
#   Body: n_frames * channels * 4 bytes of float32, little-endian

_VARIANT_MAGIC = b"SSV1"
_VARIANT_HEADER_FORMAT = "<4sHIIff10x"
_VARIANT_HEADER_SIZE = struct.calcsize(_VARIANT_HEADER_FORMAT)  # 32 bytes


def variant_cache_key (
	audio_md5:          str,
	spec:               "TransformSpec",
	output_sample_rate: int,
) -> str:

	"""Compute a deterministic SHA-256 hex digest for a transform variant.

	Includes everything that affects the variant's audio: the parent sample
	content (MD5), the ordered transform chain (spec), the output device
	sample rate, and the analysis algorithm version.  Any change to any of
	these produces a different key — no stale hits.
	"""

	h = hashlib.sha256()
	h.update(audio_md5.encode())
	h.update(subsample.analysis.ANALYSIS_VERSION.encode())
	h.update(str(output_sample_rate).encode())

	for step in spec.steps:
		h.update(repr(step).encode())

	return h.hexdigest()


class VariantDiskCache:

	"""Persistent file-based cache for transform variants.

	Each variant is stored as a single binary file named by its SHA-256 hash
	(no sidecar).  FIFO eviction by modification time keeps total disk usage
	within the configured budget.

	Thread-safe: writes use atomic temp-file + rename; eviction scans are
	independent of reads.  Reads that encounter corrupt files delete them
	and return None.
	"""

	def __init__ (
		self,
		directory:      pathlib.Path,
		max_bytes:      int,
		sample_rate:    int,
	) -> None:

		self._directory   = directory
		self._max_bytes   = max_bytes
		self._sample_rate = sample_rate

		if max_bytes > 0:
			self._directory.mkdir(parents=True, exist_ok=True)

	@property
	def enabled (self) -> bool:
		return self._max_bytes > 0

	def get (
		self,
		audio_md5: str,
		spec:      "TransformSpec",
		key:       "TransformKey",
	) -> typing.Optional["TransformResult"]:

		"""Look up a cached variant on disk.  Returns None on miss or error."""

		if not self.enabled:
			return None

		hex_digest = variant_cache_key(audio_md5, spec, self._sample_rate)
		path = self._directory / f"{hex_digest}.variant"

		if not path.exists():
			return None

		try:
			with open(path, "rb") as f:
				header = f.read(_VARIANT_HEADER_SIZE)

				if len(header) < _VARIANT_HEADER_SIZE:
					_log.warning("Variant cache: corrupt header in %s — deleting", path.name)
					path.unlink(missing_ok=True)
					return None

				magic, channels, sample_rate, n_frames, peak, rms = struct.unpack(
					_VARIANT_HEADER_FORMAT, header,
				)

				if magic != _VARIANT_MAGIC:
					_log.warning("Variant cache: bad magic in %s — deleting", path.name)
					path.unlink(missing_ok=True)
					return None

				if sample_rate != self._sample_rate:
					_log.debug("Variant cache: sample rate mismatch in %s — ignoring", path.name)
					return None

				expected_bytes = n_frames * channels * 4
				body = f.read(expected_bytes)

				if len(body) < expected_bytes:
					_log.warning("Variant cache: truncated body in %s — deleting", path.name)
					path.unlink(missing_ok=True)
					return None

			audio = numpy.frombuffer(body, dtype=numpy.float32).reshape(n_frames, channels)

			# Touch mtime so recently-used files survive FIFO eviction.
			os.utime(path, None)

			level    = subsample.analysis.LevelResult(peak=peak, rms=rms)
			duration = n_frames / sample_rate

			return TransformResult(key=key, audio=audio, duration=duration, level=level)

		except OSError as exc:
			_log.warning("Variant cache: read error for %s: %s", path.name, exc)
			return None

	def put (
		self,
		audio_md5: str,
		spec:      "TransformSpec",
		result:    "TransformResult",
	) -> None:

		"""Write a variant to disk.  Runs FIFO eviction if over budget."""

		if not self.enabled:
			return

		hex_digest = variant_cache_key(audio_md5, spec, self._sample_rate)
		path = self._directory / f"{hex_digest}.variant"

		if path.exists():
			return

		audio = result.audio
		n_frames, channels = audio.shape

		header = struct.pack(
			_VARIANT_HEADER_FORMAT,
			_VARIANT_MAGIC,
			channels,
			self._sample_rate,
			n_frames,
			result.level.peak,
			result.level.rms,
		)

		try:
			fd, tmp_path = tempfile.mkstemp(
				dir=str(self._directory), suffix=".tmp",
			)

			try:
				with os.fdopen(fd, "wb") as f:
					f.write(header)
					f.write(audio.tobytes())
				os.replace(tmp_path, str(path))
			except BaseException:
				try:
					os.unlink(tmp_path)
				except OSError:
					pass
				raise

		except OSError as exc:
			_log.warning("Variant cache: write error for %s: %s", path.name, exc)
			return

		self._evict_if_needed()

	def _evict_if_needed (self) -> None:

		"""Delete oldest variant files until total size is within budget."""

		try:
			entries = []

			for entry in os.scandir(self._directory):
				if entry.name.endswith(".variant") and entry.is_file():
					stat = entry.stat()
					entries.append((stat.st_mtime, stat.st_size, entry.path))

			total = sum(size for _, size, _ in entries)

			if total <= self._max_bytes:
				return

			# Sort oldest first.
			entries.sort()

			for _mtime, size, filepath in entries:
				if total <= self._max_bytes:
					break

				try:
					os.unlink(filepath)
					total -= size
				except OSError:
					pass

			evicted = sum(1 for _ in entries) - sum(
				1 for e in os.scandir(self._directory)
				if e.name.endswith(".variant") and e.is_file()
			)

			if evicted > 0:
				_log.info(
					"Variant disk cache: evicted %d file(s), %.1f MB remaining",
					evicted, total / (1024 * 1024),
				)

		except OSError as exc:
			_log.warning("Variant disk cache eviction error: %s", exc)


# ---------------------------------------------------------------------------
# TransformProcessor
# ---------------------------------------------------------------------------

# Signature for transform apply functions.
# audio:       float32, shape (n_frames, channels) — same channels as original.
# sample_rate: Hz (e.g. 44100).
# record:      Parent SampleRecord — provides source pitch, tempo, etc.
# step:        The specific transform parameters.
# Returns:     float32, shape (n_frames, channels) — may have different n_frames
#              (time-stretch), but channels must be preserved.
_ApplyFn = typing.Callable[
	[numpy.ndarray, int, "subsample.library.SampleRecord", typing.Any],
	numpy.ndarray,
]

# Callback invoked on the worker thread when a transform completes.
_OnTransformComplete = typing.Callable[["TransformResult"], None]


class TransformProcessor:

	"""Background worker pool that applies audio transforms.

	Mirrors the SampleProcessor pattern: enqueue() submits a job and returns
	immediately.  Workers convert the source PCM to float32, apply the
	registered transform chain in priority order, compute a LevelResult, and
	call on_complete.

	Adding a new transform type
	----------------------------
	1. Define a frozen dataclass for the step parameters.
	2. Write an apply function matching _ApplyFn.
	3. Register it in TransformProcessor._HANDLERS:
	       TransformProcessor._HANDLERS[MyTransform] = _apply_my_transform
	   The _execute() loop picks it up automatically — no other changes needed.
	4. Add the type to the TransformStep union and to spec_from_process().
	"""

	# Dispatch table: step type → apply function.
	# Populated at module load for all implemented transforms.
	_HANDLERS: typing.ClassVar[dict[type, _ApplyFn]] = {}

	def __init__ (
		self,
		sample_rate:        int,
		bit_depth:          int,
		output_sample_rate: typing.Optional[int] = None,
		on_complete:        typing.Optional[_OnTransformComplete] = None,
		on_idle:            typing.Optional[typing.Callable[[int], None]] = None,
		disk_cache:         typing.Optional[VariantDiskCache] = None,
	) -> None:

		self._sample_rate        = sample_rate
		self._bit_depth          = bit_depth
		# Output sample rate for the playback device.  If different from the
		# capture rate, _execute() resamples after normalisation so all variants
		# (base and pitch-shifted) arrive at the player pre-converted.
		self._output_sample_rate = output_sample_rate if output_sample_rate is not None else sample_rate
		self._on_complete        = on_complete
		# Called with completed-count when the in-flight set empties.  Used by
		# TransformManager to log cache memory status at queue-idle boundaries.
		self._on_idle            = on_idle
		self._disk_cache         = disk_cache

		n_workers = max(1, ((os.cpu_count() or 1) - 2) // 2)
		_log.info("TransformProcessor: %d worker(s) (cpu_count=%d)", n_workers, os.cpu_count() or 1)

		self._executor = concurrent.futures.ThreadPoolExecutor(
			max_workers=n_workers,
			thread_name_prefix="transform-worker",
		)

		# Set of in-flight TransformKeys — prevents duplicate jobs.
		self._in_flight:      set[TransformKey]  = set()
		self._in_flight_lock: threading.Lock     = threading.Lock()

		# Counters for idle/active boundary logging.
		self._batch_enqueued:  int = 0   # jobs submitted since last idle
		self._batch_completed: int = 0   # jobs finished since last idle

	def enqueue (
		self,
		record: "subsample.library.SampleRecord",
		spec:   TransformSpec,
	) -> None:

		"""Submit a transform job.  Returns immediately.

		Silently skips if:
		  - the record has no audio (nothing to transform)
		  - any step in the spec has no registered handler (transform not yet
		    implemented — see _HANDLERS); submitting would always fail on the worker
		  - an identical job is already in-flight (deduplication)
		"""

		if record.audio is None:
			return

		# Guard: don't submit jobs that will always fail because a handler is
		# not registered.  Once a handler is registered the check passes and
		# jobs run normally.
		if not all(type(step) in self._HANDLERS for step in spec.steps):
			return

		key = TransformKey(sample_id=record.sample_id, spec=spec)

		with self._in_flight_lock:
			if key in self._in_flight:
				return
			was_idle = len(self._in_flight) == 0
			self._in_flight.add(key)
			if was_idle:
				self._batch_enqueued  = 0
				self._batch_completed = 0
			self._batch_enqueued += 1

		if was_idle:
			_log.info("Transform queue active")

		self._executor.submit(self._execute, record, spec, key)

	def enqueue_pitch_range (
		self,
		record:     "subsample.library.SampleRecord",
		midi_notes: list[int],
		process:    typing.Optional[subsample.query.ProcessSpec] = None,
	) -> None:

		"""Submit one pitch-shift job per MIDI note in the list.

		When *process* is provided, builds the full ordered chain via
		spec_from_process() so pre-computed variants include all processors
		(filters, saturate, etc.) alongside the pitch shift.
		"""

		for note in midi_notes:
			if process is not None:
				spec = spec_from_process(process, midi_note=note)
			else:
				spec = TransformSpec(steps=(PitchShift(target_midi_note=note),))

			self.enqueue(record, spec)

	def enqueue_bpm_change (
		self,
		records:    list["subsample.library.SampleRecord"],
		target_bpm: float,
		resolution: int = 16,
	) -> None:

		"""Submit time-stretch jobs for all rhythmic samples.

		Only enqueues for records that have a detected tempo
		(rhythm.tempo_bpm > 0).  Samples with no beat grid are skipped.
		"""

		spec = TransformSpec(steps=(TimeStretch(
			target_bpm=target_bpm,
			resolution=resolution,
		),))

		for record in records:
			if record.rhythm.tempo_bpm > 0.0:
				self.enqueue(record, spec)

	def shutdown (self) -> None:

		"""Wait for all in-flight transforms and stop the worker pool."""

		self._executor.shutdown(wait=True)

	def _execute (
		self,
		record: "subsample.library.SampleRecord",
		spec:   TransformSpec,
		key:    TransformKey,
	) -> None:

		"""Worker method: convert audio, apply transform chain, call on_complete.

		Any exception is caught and logged so a failed transform never kills the
		worker thread.
		"""

		try:
			# enqueue() guards against None audio, but the type system can't see that.
			if record.audio is None:
				return

			# Check disk cache before doing expensive DSP.  This covers the
			# startup pre-computation path (update_assignments → enqueue_pitch_range)
			# which bypasses TransformManager.get_variant().
			if self._disk_cache is not None and spec.steps:
				audio_md5 = hashlib.md5(record.audio.tobytes()).hexdigest()
				disk_hit = self._disk_cache.get(audio_md5, spec, key)

				if disk_hit is not None:
					if self._on_complete is not None:
						self._on_complete(disk_hit)
					return

			# Convert integer PCM to float32 preserving all channels.
			audio = _pcm_to_float32(record.audio, self._bit_depth)

			# Peak-normalise to 0.9 full-scale before the processing chain.
			# In float32 this is lossless, and brings quiet samples up to a
			# consistent level so subsequent DSP stages have good headroom.
			# We measure the actual peak of the float32 audio here rather than
			# using record.level.peak (which was computed on the mono downmix).
			# For stereo sources the true per-channel peak can exceed the mono
			# peak, so using the live measurement is more accurate.
			# TransformResult.level is recomputed after all steps, so
			# _render_float() always applies the correct inverse gain.
			actual_peak = float(numpy.max(numpy.abs(audio)))
			if actual_peak > 0.0:
				audio = audio * (0.9 / actual_peak)

			# Resample to the output device rate if it differs from the capture rate.
			# Done once here, before any transform steps, so pitch shift and all
			# subsequent DSP operate at the correct output rate — the same rate the
			# player's audio stream is opened at.
			# librosa.resample expects time as the last axis: transpose to
			# (channels, n_frames), resample, transpose back.
			if self._output_sample_rate != self._sample_rate:
				audio = librosa.resample(
					audio.T,
					orig_sr=self._sample_rate,
					target_sr=self._output_sample_rate,
				).T.astype(numpy.float32)

			# Apply each step in priority order, dispatching via _HANDLERS.
			# Handlers receive the output sample rate so pitch shift and any future
			# DSP operate correctly at the resampled rate.
			for step in spec.steps:
				handler = self._HANDLERS.get(type(step))

				if handler is None:
					raise NotImplementedError(
						f"No handler registered for {type(step).__name__}. "
						"Add an entry to TransformProcessor._HANDLERS — see the "
						"'How to add a new transform type' guide in transform.py."
					)

				audio = handler(audio, self._output_sample_rate, record, step)

			# Compute level from the mono mix, consistent with how SampleRecord.level
			# was originally computed (analysis.compute_level operates on mono float32).
			level    = subsample.analysis.compute_level(_mix_to_mono(audio))
			duration = audio.shape[0] / self._output_sample_rate

			result = TransformResult(key=key, audio=audio, duration=duration, level=level)

			# Write to disk cache (skip base variants — they're cheap to recompute).
			if (
				self._disk_cache is not None
				and spec.steps
				and record.audio is not None
			):
				audio_md5 = hashlib.md5(record.audio.tobytes()).hexdigest()
				self._disk_cache.put(audio_md5, spec, result)

			if self._on_complete is not None:
				self._on_complete(result)

		except Exception:
			_log.exception(
				"Transform failed for sample %d  spec=%s",
				key.sample_id, key.spec,
			)

		finally:
			with self._in_flight_lock:
				self._in_flight.discard(key)
				self._batch_completed += 1
				now_idle  = len(self._in_flight) == 0
				completed = self._batch_completed

			if now_idle:
				if self._on_idle is not None:
					self._on_idle(completed)
				else:
					_log.info("Transform queue idle — %d variant(s) processed", completed)

# ---------------------------------------------------------------------------
# TransformManager
# ---------------------------------------------------------------------------

class TransformManager:

	"""Coordinates transform requests, caching, and background processing.

	Single point of access for the player and cli.py.  Callers do not need
	to interact with TransformCache or TransformProcessor directly.

	Lifecycle hooks
	---------------
	on_sample_added(record)
	    Called from the on_complete callback after a new SampleRecord is added
	    to InstrumentLibrary.  Phase 2 will auto-enqueue pitch variants here.

	on_parent_evicted(sample_ids)
	    Called when InstrumentLibrary.add() returns evicted IDs.  Cascade-evicts
	    all derivatives of the evicted parents from the cache.

	on_bpm_change(new_bpm)
	    Called when the target playback BPM changes.  Invalidates all cached
	    TimeStretch variants and re-enqueues new ones for rhythmic samples.

	Player look-up pattern
	-----------------------
	    result = manager.get_pitched(sample_id, midi_note)
	    if result is not None:
	        audio = _render_float(result.audio, result.level, velocity)
	        voices.append(Voice(audio=audio))
	    else:
	        # fall back to original via existing _render()
	"""

	def __init__ (
		self,
		cache:              TransformCache,
		processor:          TransformProcessor,
		instrument_library: "subsample.library.InstrumentLibrary",
		cfg:                "subsample.config.TransformConfig",
		disk_cache:         typing.Optional[VariantDiskCache] = None,
	) -> None:

		self._cache              = cache
		self._processor          = processor
		self._instrument_library = instrument_library
		self._cfg                = cfg
		self._disk_cache         = disk_cache

	def get_pitched (
		self,
		sample_id: int,
		midi_note: int,
	) -> typing.Optional[TransformResult]:

		"""Return a cached pitch variant, or None.

		On a cache miss, enqueues the transform for background production and
		returns None.  The caller should fall back to the original sample; the
		variant will be ready on the next trigger.
		"""

		result = self._cache.get_pitched(sample_id, midi_note)

		if result is None:
			record = self._instrument_library.get(sample_id)

			# Only enqueue for samples with a stable, confident pitch.
			# Matches the gate in on_sample_added() — percussive samples have
			# dominant_pitch_hz == 0, which causes log2(0) in librosa.hz_to_midi.
			if record is not None and subsample.analysis.has_stable_pitch(
				record.spectral, record.pitch, record.duration,
			):
				spec = TransformSpec(steps=(PitchShift(target_midi_note=midi_note),))
				self._processor.enqueue(record, spec)

		return result

	def get_at_bpm (
		self,
		sample_id:  int,
		target_bpm: typing.Optional[float] = None,
		resolution: typing.Optional[int]   = None,
	) -> typing.Optional[TransformResult]:

		"""Return a cached time-stretch variant, or None.

		By default reads target_bpm and quantize_resolution from the stored
		config.  Per-assignment overrides can be passed explicitly — this
		supports beat_quantize processors that declare their own BPM/grid.

		On a cache miss, enqueues the transform and returns None so the
		variant is ready on the next trigger.

		Returns None immediately when:
		  - effective target_bpm is 0.0 (disabled)
		  - the sample has no detected rhythm (tempo_bpm <= 0)
		"""

		bpm = target_bpm if target_bpm is not None else self._cfg.target_bpm
		res = resolution if resolution is not None else self._cfg.quantize_resolution

		if bpm <= 0.0:
			return None

		record = self._instrument_library.get(sample_id)

		if record is None:
			return None

		if record.rhythm.tempo_bpm <= 0.0:
			return None

		spec = TransformSpec(steps=(TimeStretch(
			target_bpm=bpm,
			resolution=res,
		),))

		key    = TransformKey(sample_id=sample_id, spec=spec)
		result = self._cache.get(key)

		if result is None:
			self._processor.enqueue(record, spec)

		return result

	def get_variant (
		self,
		sample_id: int,
		spec:      TransformSpec,
	) -> typing.Optional[TransformResult]:

		"""Return a cached variant for an arbitrary spec, or None.

		Generic look-up that works with any TransformSpec — composite chains,
		single-step transforms, or anything built by spec_from_process().

		Look-up order: memory cache → disk cache → enqueue for computation.
		Disk hits are promoted into the memory cache for subsequent look-ups.
		"""

		if not spec.steps:
			return self.get_base(sample_id)

		key    = TransformKey(sample_id=sample_id, spec=spec)
		result = self._cache.get(key)

		if result is not None:
			return result

		# Check disk cache before enqueuing a (possibly expensive) recompute.
		if self._disk_cache is not None:
			record = self._instrument_library.get(sample_id)

			if record is not None and record.audio is not None:
				audio_md5 = hashlib.md5(record.audio.tobytes()).hexdigest()
				disk_hit = self._disk_cache.get(audio_md5, spec, key)

				if disk_hit is not None:
					self._cache.put(disk_hit)
					_log.debug(
						"Disk cache hit for sample %d (%d step(s))",
						sample_id, len(spec.steps),
					)
					return disk_hit

		# Miss — enqueue for background computation.
		if result is None:
			record = self._instrument_library.get(sample_id)
			if record is not None:
				self._processor.enqueue(record, spec)

		return None

	def get_base (self, sample_id: int) -> typing.Optional[TransformResult]:

		"""Return the cached base variant for a sample, or None if not ready.

		The base variant is a float32, peak-normalised copy at the recorder's
		sample rate, with no DSP applied.  Every sample gets one — regardless
		of pitch stability — so the playback path never needs to call
		_pcm_to_float32() at trigger time.

		On a miss, enqueues the base variant for background production and
		returns None.  The caller should fall back to _render() for this
		trigger; the variant will be ready on the next.
		"""

		result = self._cache.get_base(sample_id)

		if result is None:
			record = self._instrument_library.get(sample_id)
			if record is not None:
				self._processor.enqueue(record, _BASE_VARIANT_SPEC)

		return result

	def enqueue_pitch_range (
		self,
		record:     "subsample.library.SampleRecord",
		midi_notes: list[int],
		process:    typing.Optional[subsample.query.ProcessSpec] = None,
	) -> None:

		"""Enqueue pitch-shift variants for a set of MIDI notes.

		Delegates to the underlying TransformProcessor, which deduplicates
		in-flight and already-cached keys — safe to call repeatedly.

		When *process* is provided, each variant is built via
		spec_from_process() so it includes the full ordered chain (filters,
		saturate, etc.) alongside the pitch shift.

		No-ops when auto_pitch is disabled.
		"""

		if not self._cfg.auto_pitch:
			return

		self._processor.enqueue_pitch_range(record, midi_notes, process=process)

	def on_sample_added (self, record: "subsample.library.SampleRecord") -> None:

		"""Auto-enqueue variants when a new SampleRecord enters the library.

		Always enqueues a base variant (identity spec — float32, peak-normalised,
		no DSP) so the playback path has a pre-converted copy ready for every
		sample without calling _pcm_to_float32() at trigger time.

		Pitch and time-stretch variants are NOT enqueued here — they are
		driven by MidiPlayer.update_assignments(), which reads the MIDI map
		to know exactly which samples need which processors and submits the
		precise set.
		"""

		# Base variant: always enqueue regardless of pitch content.
		self._processor.enqueue(record, _BASE_VARIANT_SPEC)

	def on_parent_evicted (self, sample_ids: list[int]) -> None:

		"""Cascade-evict all derivatives of the given parent samples.

		Call this immediately after InstrumentLibrary.add() returns evicted IDs,
		before any new sample is added to the similarity matrix.
		"""

		for sid in sample_ids:
			evicted = self._cache.remove_parent(sid)

			if evicted:
				_log.debug(
					"Cascade-evicted %d transform variant(s) for evicted parent %d",
					len(evicted), sid,
				)

	def on_bpm_change (self, new_bpm: float) -> None:

		"""React to a target BPM change.

		Invalidates all cached TimeStretch variants, then re-enqueues new
		time-stretch jobs for every rhythmic sample in the instrument library.
		Deduplication in TransformProcessor prevents flooding the queue if this
		is called repeatedly in quick succession (e.g. a BPM knob being turned).
		"""

		invalidated = self._cache.remove_by_step_type(TimeStretch)

		_log.info(
			"Target BPM changed to %.1f — invalidated %d time-stretch variant(s)",
			new_bpm, len(invalidated),
		)

		qualifying = [
			r for r in self._instrument_library.samples()
			if r.rhythm.tempo_bpm > 0.0
		]

		self._processor.enqueue_bpm_change(
			qualifying, new_bpm, resolution=self._cfg.quantize_resolution,
		)

	def has_pitch_variant (self, sample_id: int, midi_note: int) -> bool:

		"""Return True if a pitch variant for the given note is currently cached."""

		key = TransformKey(
			sample_id=sample_id,
			spec=TransformSpec(steps=(PitchShift(target_midi_note=midi_note),)),
		)

		return self._cache.get(key) is not None

	def list_variants (self, sample_id: int) -> list[TransformKey]:

		"""Return all TransformKeys currently cached for a parent sample."""

		return self._cache.list_variants(sample_id)

	def shutdown (self) -> None:

		"""Shut down the background processor, waiting for in-flight jobs."""

		self._processor.shutdown()

# ---------------------------------------------------------------------------
# Internal audio helpers
# ---------------------------------------------------------------------------

def _pcm_to_float32 (audio: numpy.ndarray, bit_depth: int) -> numpy.ndarray:

	"""Convert integer PCM to float32, preserving all channels.

	Uses the same normalisation divisor as analysis.to_mono_float() so that
	the resulting float32 values are on the same scale as original LevelResult
	measurements (i.e. peaks near 1.0 for a full-scale recording).

	Unlike to_mono_float(), channels are NOT mixed down — a stereo (n, 2) input
	produces a float32 (n, 2) output.  Transforms operate per-channel.

	Args:
		audio:     Shape (n_frames, channels), dtype int16 or int32.
		           int32 is used for both 24-bit (left-shifted) and native 32-bit.
		bit_depth: 16, 24, or 32.

	Returns:
		Shape (n_frames, channels), dtype float32, values in [-1.0, 1.0].
	"""

	divisor: float = 32768.0 if bit_depth == 16 else 2147483648.0

	return audio.astype(numpy.float32) / divisor


def _mix_to_mono (audio: numpy.ndarray) -> numpy.ndarray:

	"""Average all channels to a 1-D float32 mono signal.

	Used to compute LevelResult from multi-channel derivative audio, matching
	the convention in analysis.compute_level() which expects a 1-D array.

	Args:
		audio: Shape (n_frames, channels), dtype float32.

	Returns:
		Shape (n_frames,), dtype float32.
	"""

	if audio.shape[1] == 1:
		return audio[:, 0]

	return numpy.mean(audio, axis=1, dtype=numpy.float32)  # type: ignore[return-value]


def _apply_pitch (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        PitchShift,
) -> numpy.ndarray:

	"""Shift all channels to a target MIDI note using Rubber Band (offline, finer engine).

	Computes the semitone shift from the parent sample's detected pitch
	(record.pitch.dominant_pitch_hz) to the frequency of step.target_midi_note.
	Uses pyrubberband with --fine --smoothing --formant for Rubber Band v3's
	highest quality offline processing with frequency-domain smoothing and
	formant preservation.  --formant keeps the spectral envelope intact so
	pitched-up vocals/instruments retain their natural timbre rather than
	going chipmunk.  pyrubberband accepts (n_frames, channels) directly — no
	shape transposition is needed.

	Args:
		audio:       float32, shape (n_frames, channels).
		sample_rate: Hz (e.g. 44100).
		record:      Parent SampleRecord — provides source pitch.
		step:        PitchShift with target_midi_note.

	Returns:
		float32, shape (n_frames, channels) — pitch-shifted audio.
	"""

	if not numpy.isfinite(record.pitch.dominant_pitch_hz) or record.pitch.dominant_pitch_hz <= 0.0:
		raise ValueError(
			f"Cannot pitch-shift sample {record.sample_id} ({record.name!r}): "
			f"dominant_pitch_hz is {record.pitch.dominant_pitch_hz!r} — no stable pitch detected"
		)

	source_midi = float(librosa.hz_to_midi(record.pitch.dominant_pitch_hz))
	n_steps     = float(step.target_midi_note) - source_midi

	return pyrubberband.pitch_shift(  # type: ignore[no-any-return]
		audio,
		sample_rate,
		n_steps,
		rbargs={"--fine": "", "--smoothing": "", "--formant": ""},
	)


# ---------------------------------------------------------------------------
# Time-stretch handler — beat-quantized non-linear stretching
# ---------------------------------------------------------------------------

# Pre-onset margin: audio before each onset is included so the transient
# attack is not clipped.  2 ms matches the perceptual attack resolution of
# the human ear.
_PRE_ONSET_SECONDS: float = 0.002

# Short S-curve fade-in applied after cropping to the first attack.
# Prevents clicks from non-zero samples at the crop boundary.  1 ms is
# well below the perceptual attack-shaping threshold.
_CROP_FADE_IN_SECONDS: float = 0.001


def _build_quantize_grid (target_bpm: float, resolution: int, max_seconds: float, min_points: int = 0) -> list[float]:

	"""Build a regularly-spaced time grid at the target BPM and subdivision.

	Args:
		target_bpm:  Target tempo in BPM.
		resolution:  Grid subdivision (1=whole, 2=half, 4=quarter, 8=eighth,
		             16=sixteenth).
		max_seconds: Generate grid points up to at least this time.
		min_points:  Guarantee at least this many grid points.  The greedy
		             onset snapper consumes one point per onset, so pass
		             len(onset_times) + a margin to avoid exhaustion.

	Returns:
		List of grid-point times in seconds, starting at 0.0.
	"""

	grid_interval = 60.0 / target_bpm / (resolution / 4.0)
	count = max(int(max_seconds / grid_interval) + 2, min_points)

	return [i * grid_interval for i in range(count)]


def _snap_onsets_to_grid (onset_times: tuple[float, ...], grid: list[float]) -> list[float]:

	"""Snap each onset to the nearest available grid point.

	Uses greedy left-to-right assignment: each onset takes the closest grid
	point that has not been claimed by an earlier onset.  This guarantees
	monotonically increasing target times and prevents two onsets from
	landing on the same grid point.

	Args:
		onset_times: Source onset positions in seconds (sorted, ascending).
		grid:        Sorted list of grid-point times from _build_quantize_grid().

	Returns:
		List of target times (one per onset), in the same order.
	"""

	assigned: list[float] = []
	next_min_idx = 0

	for onset in onset_times:
		best_idx = next_min_idx
		best_dist = abs(onset - grid[best_idx])

		for j in range(next_min_idx + 1, len(grid)):
			dist = abs(onset - grid[j])

			if dist < best_dist:
				best_idx = j
				best_dist = dist
			elif dist >= best_dist:
				# Grid is sorted — distances will only increase from here.
				break

		best_idx = max(best_idx, next_min_idx)
		assigned.append(grid[best_idx])
		next_min_idx = best_idx + 1

	return assigned


def _build_time_map (
	onset_source_samples: list[int],
	onset_target_samples: list[int],
	source_length:        int,
	target_length:        int,
) -> list[tuple[int, int]]:

	"""Construct the time map for pyrubberband.timemap_stretch().

	Each entry is (source_sample, target_sample).  The first entry anchors
	the start and the last entry anchors the end — required by the API.
	All entries are monotonically increasing and non-negative.

	Args:
		onset_source_samples: Source sample positions for each onset.
		onset_target_samples: Corresponding target sample positions.
		source_length:        Total source audio length in samples.
		target_length:        Desired total target audio length in samples.

	Returns:
		Time map suitable for pyrubberband.timemap_stretch().
	"""

	time_map: list[tuple[int, int]] = [(0, 0)]

	for src, tgt in zip(onset_source_samples, onset_target_samples):

		# Ensure strict monotonicity with previous entry.
		prev_src, prev_tgt = time_map[-1]

		if src > prev_src and tgt > prev_tgt:
			time_map.append((src, tgt))

	# Final anchor: source and target endpoints.
	time_map.append((source_length, target_length))

	return time_map


def _apply_time_stretch (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        TimeStretch,
) -> numpy.ndarray:

	"""Beat-quantized time-stretch using Rubber Band's offline finer engine
	with frequency-domain smoothing to reduce phasing artefacts.

	Uses sample-accurate attack_times (refined from librosa onsets via
	amplitude-envelope analysis) for grid alignment, so percussive hits
	land precisely on the beat.  Falls back to onset_times for pre-v10
	analysis data.  Samples with fewer than 2 attacks receive a simple
	global time-stretch instead.

	Args:
		audio:       float32, shape (n_frames, channels).
		sample_rate: Hz (e.g. 44100).
		record:      Parent SampleRecord — provides rhythm analysis.
		step:        TimeStretch with target_bpm and resolution.

	Returns:
		float32, shape (n_frames_out, channels) — time-stretched audio.
		n_frames_out will differ from n_frames when the tempos differ.
	"""

	source_bpm = record.rhythm.tempo_bpm

	if source_bpm <= 0.0:
		return audio

	# Duration ratio: >1 means output is longer (slower tempo), <1 means shorter.
	# pyrubberband.time_stretch rate is the inverse (speed multiplier).
	duration_ratio = source_bpm / step.target_bpm

	# Prefer sample-accurate attack times for grid alignment — they mark
	# where each transient becomes audible, not where spectral flux peaks.
	# Fall back to onset_times for pre-v10 analysis data.
	attack_times = record.rhythm.attack_times

	if not attack_times:
		attack_times = record.rhythm.onset_times

	# Single-hit or no-onset samples: simple global stretch.
	if len(attack_times) < 2:
		return pyrubberband.time_stretch(  # type: ignore[no-any-return]
			audio, sample_rate, 1.0 / duration_ratio, rbargs={"--fine": "", "--smoothing": ""},
		)

	# ── Crop to first attack ─────────────────────────────────────────────

	first_attack_sec = attack_times[0]
	crop_start_sec   = max(0.0, first_attack_sec - _PRE_ONSET_SECONDS)
	crop_start_frame = int(crop_start_sec * sample_rate)

	audio = audio[crop_start_frame:]

	# S-curve fade-in over the crop boundary to eliminate clicks.
	fade_in_len = int(_CROP_FADE_IN_SECONDS * sample_rate)

	if fade_in_len > 1:
		ramp = (1 - numpy.cos(numpy.linspace(0, numpy.pi, fade_in_len))) / 2
		audio[:fade_in_len] *= ramp[:, numpy.newaxis]

	# Rebase attack times relative to the crop point.
	rebased = [t - crop_start_sec for t in attack_times]

	# ── Build target grid and snap onsets ─────────────────────────────────

	# Generous upper bound: twice the rebased duration or last onset, whichever
	# is larger, ensures the grid extends far enough for any stretch direction.
	# min_points guarantees the greedy snapper never runs out of grid slots
	# even when many tightly-spaced onsets each consume their own point.
	audio_duration_sec = audio.shape[0] / sample_rate
	max_grid_sec = max(rebased[-1], audio_duration_sec) * 2.0

	grid     = _build_quantize_grid(
		step.target_bpm, step.resolution, max_grid_sec,
		min_points=len(rebased) + 2,
	)
	snapped  = _snap_onsets_to_grid(tuple(rebased), grid)

	# ── Build time map (source sample → target sample) ────────────────────

	onset_src = [int(t * sample_rate) for t in rebased]
	onset_tgt = [int(t * sample_rate) for t in snapped]

	# Tail after the last onset: scale by the duration ratio so the decay
	# sounds natural rather than being forced to fill a grid slot.
	tail_src_sec = max(0.0, audio_duration_sec - rebased[-1])
	tail_tgt_sec = tail_src_sec * duration_ratio

	source_length = audio.shape[0]
	target_length = max(
		int((snapped[-1] + tail_tgt_sec) * sample_rate),
		onset_tgt[-1] + 1,
	)

	# The final anchor must be strictly greater than any intermediate entry
	# for pyrubberband to accept the time map.
	if target_length <= onset_tgt[-1]:
		target_length = onset_tgt[-1] + 1

	time_map = _build_time_map(onset_src, onset_tgt, source_length, target_length)

	_log.debug(
		"Time-stretch %s: %.1f → %.1f BPM (res=%d), %d attacks, "
		"%.3fs → %.3fs",
		record.name, source_bpm, step.target_bpm, step.resolution,
		len(attack_times), audio_duration_sec, target_length / sample_rate,
	)

	# ── Apply via Rubber Band's offline finer engine ──────────────────────
	#
	# We provide an explicit time_map so Rubber Band's internal transient
	# detection is bypassed entirely — our two-stage onset pipeline
	# (librosa spectral flux → sample-accurate amplitude-envelope refinement
	# in analysis._refine_onsets_to_attacks) gives ~0.7 ms precision, far
	# tighter than any frame-level detector.  Rubber Band's --detector-perc
	# flag is R2-only and irrelevant here (we use R3 via --fine).

	return pyrubberband.timemap_stretch(  # type: ignore[no-any-return]
		audio, sample_rate, time_map, rbargs={"--fine": "", "--smoothing": ""},
	)


# ---------------------------------------------------------------------------
# Reverse handler
# ---------------------------------------------------------------------------

def _apply_reverse (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Reverse,
) -> numpy.ndarray:

	"""Reverse the audio along the time axis.

	Returns a contiguous copy — the negative-stride view from [::-1] can
	break downstream C-based DSP (pyrubberband, soundfile, etc.).
	"""

	return audio[::-1].copy()


# ---------------------------------------------------------------------------
# Filter handlers
# ---------------------------------------------------------------------------

_MAX_RESONANCE_DB: float = 24.0


def _apply_filter (
	audio:       numpy.ndarray,
	sample_rate: int,
	freq:        float,
	resonance_db: float,
	btype:       str,
	q:           float = 0.7,
) -> numpy.ndarray:

	"""Shared implementation for low-pass, high-pass, and band-pass filters.

	Uses a 2nd-order Butterworth (resonance_db == 0) or Chebyshev Type I
	(resonance_db > 0) filter applied via second-order sections for numerical
	stability.  Band-pass bandwidth is derived from Q (quality factor):
	lower Q = wider band, higher Q = narrower band.
	"""

	nyquist = sample_rate / 2.0

	# Clamp resonance to a safe range.
	resonance_db = max(0.0, min(resonance_db, _MAX_RESONANCE_DB))

	# Clamp Q to a safe range (0.1 = very wide, 20 = very narrow).
	q = max(0.1, min(q, 20.0))

	# Build the Wn parameter.

	if btype == "bandpass":
		# Compute band edges from centre frequency and Q.
		# Q = f0 / bandwidth, so higher Q = narrower band.
		# Exact geometric formula for -3 dB points:
		term = 1.0 / (2.0 * q)
		root = math.sqrt(term ** 2 + 1.0)
		low  = max(1.0, freq * (root - term))
		high = min(nyquist - 1.0, freq * (root + term))

		if low >= high:
			return audio

		wn: typing.Any = [low, high]

	else:
		clamped = max(1.0, min(freq, nyquist - 1.0))

		if clamped <= 1.0:
			return audio

		wn = clamped

	# Design the filter.

	if resonance_db <= 0.0:
		sos = scipy.signal.butter(2, wn, btype=btype, fs=sample_rate, output="sos")  # type: ignore[call-overload]
	else:
		sos = scipy.signal.iirfilter(  # type: ignore[call-overload]
			2, wn,
			rp=resonance_db,
			btype=btype,
			ftype="cheby1",
			fs=sample_rate,
			output="sos",
		)

	return scipy.signal.sosfilt(sos, audio, axis=0).astype(numpy.float32)


def _apply_low_pass (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        LowPassFilter,
) -> numpy.ndarray:

	"""Low-pass filter — attenuates frequencies above the cutoff."""

	return _apply_filter(audio, sample_rate, step.freq, step.resonance_db, "lowpass")


def _apply_high_pass (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        HighPassFilter,
) -> numpy.ndarray:

	"""High-pass filter — attenuates frequencies below the cutoff."""

	return _apply_filter(audio, sample_rate, step.freq, step.resonance_db, "highpass")


def _apply_band_pass (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        BandPassFilter,
) -> numpy.ndarray:

	"""Band-pass filter — passes a band centred on the frequency, width set by Q."""

	return _apply_filter(audio, sample_rate, step.freq, step.resonance_db, "bandpass", q=step.q)


# ---------------------------------------------------------------------------
# Saturate handler
# ---------------------------------------------------------------------------

def _apply_saturate (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Saturate,
) -> numpy.ndarray:

	"""Soft-clip saturation via tanh with level compensation.

	Drive is computed from amount_db.  The output is divided by tanh(drive)
	so that a full-scale input maps back to full-scale, preventing volume
	loss at high drive settings.
	"""

	if step.amount_db <= 0.0:
		return audio

	drive = 10.0 ** (step.amount_db / 20.0)

	saturated = numpy.tanh(audio * drive)
	saturated /= numpy.tanh(numpy.float32(drive))

	result: numpy.ndarray = saturated.astype(numpy.float32)
	return result


# ---------------------------------------------------------------------------
# Compress / Limit handlers
# ---------------------------------------------------------------------------

# Tiny constant to avoid log10(0).
_COMPRESS_EPSILON: float = 1e-10


def _compress (
	audio:        numpy.ndarray,
	sample_rate:  int,
	threshold_db: float,
	ratio:        float,
	attack_ms:    float,
	release_ms:   float,
	knee_db:      float,
	makeup_db:    float,
	lookahead_ms: float,
) -> numpy.ndarray:

	"""Feed-forward dynamic range compressor (Giannoulis et al., JAES 2012).

	Shared back-end for both ``compress`` and ``limit`` processors.

	1. Convert to dB.
	2. Gain computer with soft knee (piecewise quadratic transition).
	3. One-pole ballistics (attack/release smoothing per sample).
	4. Optional look-ahead: audio is delayed so the gain envelope
	   anticipates transients before they arrive.
	5. Apply gain + makeup.

	Multi-channel: the envelope is computed from the max absolute value
	across channels (linked stereo), and the same gain curve is applied to
	all channels to preserve the stereo image.

	All processing is in float64 for numerical precision; the result is
	returned as float32.
	"""

	if ratio <= 1.0:
		# No compression.
		if makeup_db != 0.0:
			result: numpy.ndarray = (audio * 10.0 ** (makeup_db / 20.0)).astype(numpy.float32)
			return result
		return audio

	n_frames = audio.shape[0]

	if n_frames == 0:
		return audio

	# Work in float64 for precision.
	audio_f64 = audio.astype(numpy.float64)

	# Linked envelope: max absolute value across channels per frame.
	if audio_f64.ndim == 1:
		envelope = numpy.abs(audio_f64)
	else:
		envelope = numpy.max(numpy.abs(audio_f64), axis=1)

	# Convert envelope to dB.
	env_db = 20.0 * numpy.log10(envelope + _COMPRESS_EPSILON)

	# ── Gain computer (vectorised) ──────────────────────────────────

	T = threshold_db
	R = ratio
	W = max(knee_db, 0.0)

	gain_db = numpy.zeros(n_frames, dtype=numpy.float64)

	if W > 0.0:
		# Soft knee: three regions.
		below = env_db < (T - W / 2.0)
		above = env_db > (T + W / 2.0)
		knee  = ~below & ~above

		# Below knee: no gain change (gain_db stays 0).
		# Above knee: standard compression.
		gain_db[above] = (T + (env_db[above] - T) / R) - env_db[above]
		# Within knee: quadratic transition.
		diff = env_db[knee] - T + W / 2.0
		gain_db[knee] = ((1.0 / R - 1.0) * diff * diff) / (2.0 * W)
	else:
		# Hard knee: two regions.
		above = env_db > T
		gain_db[above] = (T + (env_db[above] - T) / R) - env_db[above]

	# ── Ballistics (per-sample smoothing) ───────────────────────────

	attack_s  = max(attack_ms / 1000.0, 1e-6)
	release_s = max(release_ms / 1000.0, 1e-6)
	alpha_a = math.exp(-1.0 / (sample_rate * attack_s))
	alpha_r = math.exp(-1.0 / (sample_rate * release_s))

	smoothed = numpy.empty(n_frames, dtype=numpy.float64)
	s = 0.0  # initial state: no gain reduction

	for i in range(n_frames):
		g = gain_db[i]
		if g < s:
			# Signal rising above threshold — engage gain reduction (attack).
			s = alpha_a * s + (1.0 - alpha_a) * g
		else:
			# Signal falling — release gain reduction.
			s = alpha_r * s + (1.0 - alpha_r) * g
		smoothed[i] = s

	# ── Look-ahead ──────────────────────────────────────────────────

	lookahead_samples = int(round(lookahead_ms / 1000.0 * sample_rate))

	if lookahead_samples > 0:
		# Delay the audio relative to the gain curve.  The gain curve was
		# computed from the original signal, so gain reduction begins
		# before the delayed peak arrives.
		pad = numpy.zeros(
			(lookahead_samples,) + audio_f64.shape[1:], dtype=numpy.float64,
		)
		audio_f64 = numpy.concatenate([pad, audio_f64], axis=0)[:n_frames]

	# ── Apply gain + makeup ─────────────────────────────────────────

	linear_gain = numpy.power(10.0, smoothed / 20.0)

	if makeup_db != 0.0:
		linear_gain *= 10.0 ** (makeup_db / 20.0)

	if audio_f64.ndim == 1:
		audio_f64 *= linear_gain
	else:
		audio_f64 *= linear_gain[:, numpy.newaxis]

	return audio_f64.astype(numpy.float32)


def _resolve_compress_params (
	record: "subsample.library.SampleRecord",
	step:   Compress,
) -> tuple[float, float, float]:

	"""Resolve adaptive compression parameters from sample analysis data.

	For each of threshold_db, attack_ms, and release_ms: if the step value
	is None (user did not set it), compute from the sample's analysis.
	If the step value is a float (user set it explicitly), use it as-is.

	Returns:
		(threshold_db, attack_ms, release_ms) — all resolved to floats.
	"""

	# Threshold: 6 dB below the sample's peak level.  This ensures the
	# compressor always engages on the top 6 dB of the signal's dynamic
	# range, regardless of recording level.
	if step.threshold_db is not None:
		threshold_db = step.threshold_db
	else:
		peak_db = 20.0 * math.log10(record.level.peak + _COMPRESS_EPSILON)
		threshold_db = peak_db - 6.0

	# Attack: map from spectral.attack (0 = instant percussive, 1 = gradual).
	# Percussive sounds → slow compressor attack (lets transient punch through).
	# Gradual sounds → fast compressor attack (no transient to protect).
	if step.attack_ms is not None:
		attack_ms = step.attack_ms
	else:
		attack_ms = 1.0 + 29.0 * (1.0 - record.spectral.attack)

	# Release: map from spectral.release (0 = short decay, 1 = long tail).
	# Short-decay sounds → fast release (recovers before the next hit).
	# Sustained sounds → slow release (avoids pumping artefacts).
	if step.release_ms is not None:
		release_ms = step.release_ms
	else:
		release_ms = 30.0 + 270.0 * record.spectral.release

	return threshold_db, attack_ms, release_ms


def _apply_compress (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Compress,
) -> numpy.ndarray:

	"""Dynamic range compressor — adapts to each sample when params are unset."""

	threshold_db, attack_ms, release_ms = _resolve_compress_params(record, step)

	return _compress(
		audio, sample_rate,
		threshold_db = threshold_db,
		ratio        = step.ratio,
		attack_ms    = attack_ms,
		release_ms   = release_ms,
		knee_db      = step.knee_db,
		makeup_db    = step.makeup_db,
		lookahead_ms = step.lookahead_ms,
	)


def _apply_limit (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Limit,
) -> numpy.ndarray:

	"""Brickwall limiter — compressor with limiter presets."""

	return _compress(
		audio, sample_rate,
		threshold_db = step.threshold_db,
		ratio        = 100.0,
		attack_ms    = 0.01,
		release_ms   = step.release_ms,
		knee_db      = 0.0,
		makeup_db    = 0.0,
		lookahead_ms = step.lookahead_ms,
	)


# ---------------------------------------------------------------------------
# HPSS handlers
# ---------------------------------------------------------------------------

def _apply_hpss (
	audio:     numpy.ndarray,
	keep:      str,
) -> numpy.ndarray:

	"""Shared HPSS implementation for harmonic and percussive processors.

	Processes each channel independently via librosa.decompose.hpss on the
	per-channel STFT, then reconstructs the selected component via istft.
	"""

	n_frames, channels = audio.shape
	result = numpy.empty_like(audio)

	for ch in range(channels):
		D = librosa.stft(audio[:, ch])
		harmonic_D, percussive_D = librosa.decompose.hpss(D)

		if keep == "harmonic":
			result[:, ch] = librosa.istft(harmonic_D, length=n_frames)
		else:
			result[:, ch] = librosa.istft(percussive_D, length=n_frames)

	return result.astype(numpy.float32)


def _apply_hpss_harmonic (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        HpssHarmonic,
) -> numpy.ndarray:

	"""Keep only the harmonic (tonal/sustained) component."""

	return _apply_hpss(audio, "harmonic")


def _apply_hpss_percussive (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        HpssPercussive,
) -> numpy.ndarray:

	"""Keep only the percussive (transient) component."""

	return _apply_hpss(audio, "percussive")


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

# Epsilon for dB conversion — prevents log10(0) in gate threshold computation.
_GATE_EPSILON: float = 1e-10


def _resolve_gate_params (
	record: "subsample.library.SampleRecord",
	step:   Gate,
) -> tuple[float, float, float, float, float]:

	"""Resolve adaptive gate parameters from sample analysis data.

	For each Optional field: if None, compute from analysis; if set, use as-is.

	Returns:
		(threshold_db, attack_ms, release_ms, hold_ms, lookahead_ms).
	"""

	# Threshold: 6 dB above the noise floor.
	if step.threshold_db is not None:
		threshold_db = step.threshold_db
	else:
		floor = record.level.noise_floor
		threshold_db = 20.0 * math.log10(floor + _GATE_EPSILON) + 6.0

	# Attack: percussive → fast gate open (1 ms), sustained → slower (5 ms).
	if step.attack_ms is not None:
		attack_ms = step.attack_ms
	else:
		attack_ms = 5.0 - 4.0 * record.spectral.attack

	# Release: short-decay → fast close (20 ms), long-tail → slower (100 ms).
	if step.release_ms is not None:
		release_ms = step.release_ms
	else:
		release_ms = 20.0 + 80.0 * record.spectral.release

	# Hold: longer-decaying sounds need longer hold to avoid chatter.
	if step.hold_ms is not None:
		hold_ms = step.hold_ms
	else:
		hold_ms = 10.0 + 40.0 * record.spectral.release

	# Lookahead: percussive onsets benefit from a small lookahead.
	if step.lookahead_ms is not None:
		lookahead_ms = step.lookahead_ms
	else:
		lookahead_ms = 3.0 * (1.0 - record.spectral.attack)

	return threshold_db, attack_ms, release_ms, hold_ms, lookahead_ms


def _apply_gate (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Gate,
) -> numpy.ndarray:

	"""Apply a noise gate: silence audio below the threshold.

	DSP: envelope detection → dB threshold → hold counter → one-pole
	ballistics → gain multiplication.  Same ballistics model as the
	compressor but with binary gain (open = 1.0, closed = 0.0).
	"""

	threshold_db, attack_ms, release_ms, hold_ms, lookahead_ms = _resolve_gate_params(record, step)

	n_frames, n_channels = audio.shape

	# Work in float64 for gain smoothing precision.
	audio_f64 = audio.astype(numpy.float64)

	# Linked envelope: max absolute value across channels per sample.
	if n_channels == 1:
		envelope = numpy.abs(audio_f64[:, 0])
	else:
		envelope = numpy.max(numpy.abs(audio_f64), axis=1)

	# Convert to dB.
	env_db = 20.0 * numpy.log10(envelope + _GATE_EPSILON)

	# Gate gain: 1.0 where above threshold, 0.0 where below.
	raw_gain = numpy.where(env_db >= threshold_db, 1.0, 0.0)

	# Hold: keep the gate open for hold_samples after it would close.
	hold_samples = int(hold_ms * sample_rate / 1000.0)

	if hold_samples > 0:
		held_gain = numpy.copy(raw_gain)
		hold_counter = 0

		for i in range(n_frames):
			if raw_gain[i] > 0.5:
				hold_counter = hold_samples
				held_gain[i] = 1.0
			elif hold_counter > 0:
				hold_counter -= 1
				held_gain[i] = 1.0
			else:
				held_gain[i] = 0.0

		raw_gain = held_gain

	# One-pole ballistics (same model as compressor).
	alpha_a = 1.0 - math.exp(-1.0 / (attack_ms  * sample_rate / 1000.0)) if attack_ms  > 0.0 else 1.0
	alpha_r = 1.0 - math.exp(-1.0 / (release_ms * sample_rate / 1000.0)) if release_ms > 0.0 else 1.0

	smoothed = numpy.empty(n_frames, dtype=numpy.float64)
	smoothed[0] = raw_gain[0]

	for i in range(1, n_frames):
		if raw_gain[i] > smoothed[i - 1]:
			smoothed[i] = alpha_a * raw_gain[i] + (1.0 - alpha_a) * smoothed[i - 1]
		else:
			smoothed[i] = alpha_r * raw_gain[i] + (1.0 - alpha_r) * smoothed[i - 1]

	# Lookahead: delay the audio relative to the gain envelope.
	lookahead_samples = int(lookahead_ms * sample_rate / 1000.0)

	if lookahead_samples > 0:
		audio_f64 = numpy.pad(audio_f64, ((lookahead_samples, 0), (0, 0)), mode="constant")
		audio_f64 = audio_f64[:n_frames]

	# Apply gain.
	result = audio_f64 * smoothed[:, numpy.newaxis]

	return result.astype(numpy.float32)


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------

def _resolve_distort_params (
	record: "subsample.library.SampleRecord",
	step:   Distort,
) -> tuple[float, float]:

	"""Resolve adaptive distortion parameters from sample analysis data.

	Returns:
		(drive_db, tone) — both resolved to floats.
	"""

	# Drive: peaky sounds (high crest) need less push; compressed sounds need more.
	# Map crest_factor_db [3, 20] → drive [12, 3] dB (inverse relationship).
	if step.drive_db is not None:
		drive_db = step.drive_db
	else:
		cf_db = record.level.crest_factor_db
		# Clamp to useful range and invert.
		t = max(0.0, min(1.0, (cf_db - 3.0) / 17.0))   # 0 = low crest, 1 = high crest
		drive_db = 12.0 - 9.0 * t                        # 12 dB for compressed, 3 dB for peaky

	# Tone: bright sounds get more LPF to tame added harmonics.
	# Map spectral_rolloff [0, 1] → tone [1.0, 0.3] (fraction of Nyquist).
	if step.tone is not None:
		tone = step.tone
	else:
		tone = 1.0 - 0.7 * record.spectral.spectral_rolloff

	return drive_db, tone


def _apply_distort (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Distort,
) -> numpy.ndarray:

	"""Apply waveshaping distortion.

	Modes: hard_clip, fold, bit_crush, downsample.  All modes apply level
	compensation (output peak matches input peak) and optional tone filtering.
	"""

	drive_db, tone = _resolve_distort_params(record, step)

	n_frames = audio.shape[0]
	drive = 10.0 ** (drive_db / 20.0)

	# Store pre-distortion peak for level compensation.
	pre_peak = float(numpy.max(numpy.abs(audio)))

	if pre_peak < 1e-10:
		return audio   # silence → silence

	# Keep dry copy for mix blending.
	dry = audio if step.mix >= 1.0 else audio.copy()

	# Apply drive gain.
	driven = audio * numpy.float32(drive)

	# Waveshaping.
	mode = step.mode

	if mode == "hard_clip":
		wet = numpy.clip(driven, -1.0, 1.0)

	elif mode == "fold":
		# Foldback: signal wraps around ±1 instead of clipping.
		wet = numpy.abs(numpy.mod(driven + 1.0, 4.0) - 2.0) - 1.0

	elif mode == "bit_crush":
		levels = float(2 ** max(1, min(16, step.bit_depth)))
		wet = (numpy.round(driven * levels) / levels).astype(numpy.float32)

	elif mode == "downsample":
		factor = max(2, min(64, step.downsample_factor))
		# Repeat every Nth sample along the time axis.
		wet = numpy.repeat(driven[::factor], factor, axis=0)[:n_frames]

	else:
		_log.warning("Unknown distortion mode %r — returning unchanged", mode)
		return audio

	# Level compensation: restore pre-distortion peak.
	post_peak = float(numpy.max(numpy.abs(wet)))

	if post_peak > 1e-10:
		wet = wet * numpy.float32(pre_peak / post_peak)

	# Tone filter: low-pass to tame high-frequency harmonics added by distortion.
	if tone < 0.99:
		cutoff_hz = max(200.0, tone * sample_rate / 2.0)
		sos = scipy.signal.butter(2, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")

		for ch in range(wet.shape[1]):
			wet[:, ch] = scipy.signal.sosfilt(sos, wet[:, ch]).astype(numpy.float32)

	# Dry/wet blend.
	if step.mix < 1.0:
		wet = dry * numpy.float32(1.0 - step.mix) + wet * numpy.float32(step.mix)

	return wet.astype(numpy.float32)


# ---------------------------------------------------------------------------
# Envelope Reshaping
# ---------------------------------------------------------------------------

def _apply_reshape (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      "subsample.library.SampleRecord",
	step:        Reshape,
) -> numpy.ndarray:

	"""Reshape the amplitude envelope of the sound (ADSR-style).

	Builds a gain curve from the ADSR parameters and multiplies it onto the
	audio.  Phases set to None preserve the original envelope (gain = 1.0).
	"""

	n_frames = audio.shape[0]

	# Resolve auto release.
	if step.release_ms is not None:
		release_ms = step.release_ms
	else:
		# Auto: tighten the tail based on the sample's natural decay character.
		release_ms = 30.0 + 170.0 * record.spectral.release

	# Check if there's anything to do.
	is_noop = (
		step.attack_ms is None
		and step.hold_ms == 0.0
		and step.decay_ms is None
		and step.sustain >= 1.0
		and release_ms is None
	)

	if is_noop:
		return audio

	# Find the onset point (where the sound begins).
	if record.rhythm.attack_times:
		onset_sample = int(record.rhythm.attack_times[0] * sample_rate)
	else:
		# Fallback: first sample exceeding 10% of peak.
		peak_val = float(numpy.max(numpy.abs(audio)))

		if peak_val < 1e-10:
			return audio

		threshold = 0.1 * peak_val
		above = numpy.where(numpy.max(numpy.abs(audio), axis=1) > threshold)[0]
		onset_sample = int(above[0]) if len(above) > 0 else 0

	onset_sample = max(0, min(onset_sample, n_frames - 1))

	# Find the peak sample position (for attack phase endpoint).
	peak_sample = int(numpy.argmax(numpy.max(numpy.abs(audio), axis=1)))
	peak_sample = max(onset_sample, peak_sample)

	# Build the gain curve.
	gain = numpy.ones(n_frames, dtype=numpy.float64)

	cursor = onset_sample

	# Attack phase: ramp from 0 to 1.
	if step.attack_ms is not None:
		attack_samples = max(1, int(step.attack_ms * sample_rate / 1000.0))
		end = min(cursor + attack_samples, n_frames)

		# Zero everything before onset.
		gain[:cursor] = 0.0

		# Linear ramp.
		ramp_len = end - cursor

		if ramp_len > 0:
			gain[cursor:end] = numpy.linspace(0.0, 1.0, ramp_len)

		cursor = end
	else:
		# Preserve original attack — advance cursor to peak.
		cursor = peak_sample

	# Hold phase: keep at 1.0.
	if step.hold_ms > 0.0:
		hold_samples = int(step.hold_ms * sample_rate / 1000.0)
		cursor = min(cursor + hold_samples, n_frames)

	# Decay phase: ramp from 1.0 to sustain level.
	if step.decay_ms is not None:
		decay_samples = max(1, int(step.decay_ms * sample_rate / 1000.0))
		end = min(cursor + decay_samples, n_frames)
		ramp_len = end - cursor

		if ramp_len > 0:
			gain[cursor:end] = numpy.linspace(1.0, step.sustain, ramp_len)

		cursor = end

		# Sustain region: constant at sustain level until release.
		release_samples = max(1, int(release_ms * sample_rate / 1000.0))
		release_start = max(cursor, n_frames - release_samples)
		gain[cursor:release_start] = step.sustain
	else:
		# No explicit decay — sustain level applies from cursor onward.
		if step.sustain < 1.0:
			release_samples = max(1, int(release_ms * sample_rate / 1000.0))
			release_start = max(cursor, n_frames - release_samples)
			gain[cursor:release_start] = step.sustain
		else:
			release_samples = max(1, int(release_ms * sample_rate / 1000.0))
			release_start = max(cursor, n_frames - release_samples)

	# Release phase: fade from sustain level to 0.
	release_samples = max(1, int(release_ms * sample_rate / 1000.0))
	release_start = max(cursor, n_frames - release_samples)

	if release_start < n_frames:
		start_level = gain[release_start] if release_start < n_frames else step.sustain
		gain[release_start:] = numpy.linspace(
			start_level, 0.0, n_frames - release_start,
		)

	# Apply gain curve to all channels.
	result = audio.astype(numpy.float64) * gain[:, numpy.newaxis]

	return result.astype(numpy.float32)


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

TransformProcessor._HANDLERS[PitchShift]      = _apply_pitch
TransformProcessor._HANDLERS[TimeStretch]     = _apply_time_stretch
TransformProcessor._HANDLERS[Reverse]         = _apply_reverse
TransformProcessor._HANDLERS[LowPassFilter]   = _apply_low_pass
TransformProcessor._HANDLERS[HighPassFilter]   = _apply_high_pass
TransformProcessor._HANDLERS[BandPassFilter]  = _apply_band_pass
TransformProcessor._HANDLERS[Saturate]        = _apply_saturate
TransformProcessor._HANDLERS[Compress]        = _apply_compress
TransformProcessor._HANDLERS[Limit]           = _apply_limit
TransformProcessor._HANDLERS[HpssHarmonic]    = _apply_hpss_harmonic
TransformProcessor._HANDLERS[HpssPercussive]  = _apply_hpss_percussive
TransformProcessor._HANDLERS[Gate]            = _apply_gate
TransformProcessor._HANDLERS[Distort]         = _apply_distort
TransformProcessor._HANDLERS[Reshape]         = _apply_reshape


# ---------------------------------------------------------------------------
# Process spec → TransformSpec conversion
# ---------------------------------------------------------------------------

def spec_from_process (
	process:    subsample.query.ProcessSpec,
	midi_note:  typing.Optional[int]   = None,
	target_bpm: typing.Optional[float] = None,
	resolution: int                    = 16,
) -> TransformSpec:

	"""Build an ordered TransformSpec from a MIDI map process pipeline.

	Iterates the process steps in *declaration order*, converting each
	ProcessorStep into the corresponding TransformStep dataclass.  Dynamic
	parameters (midi_note for repitch, target_bpm/resolution for
	beat_quantize) are substituted at the position the user declared them.

	Steps with unresolvable dynamic parameters (e.g. repitch when midi_note
	is None) are silently skipped.  Unknown processor names log a warning
	and are skipped.
	"""

	steps: list[TransformStep] = []

	for proc in process.steps:

		if proc.name == "repitch":
			fixed_note = proc.get("note")

			if fixed_note is not None:
				# Fixed target note — accept int or note name (e.g. "C4").
				if isinstance(fixed_note, int):
					steps.append(PitchShift(target_midi_note=fixed_note))
				else:
					import pymididefs.notes
					steps.append(PitchShift(target_midi_note=pymididefs.notes.name_to_note(str(fixed_note))))

			elif midi_note is not None:
				steps.append(PitchShift(target_midi_note=midi_note))

		elif proc.name == "beat_quantize":
			bpm = float(proc.get("bpm", target_bpm or 0.0))
			grid = int(proc.get("grid", resolution))

			if bpm > 0.0:
				steps.append(TimeStretch(target_bpm=bpm, resolution=grid))

		elif proc.name == "filter_low":
			steps.append(LowPassFilter(
				freq=float(proc.get("freq", 16000.0)),
				resonance_db=float(proc.get("resonance", 0.0)),
			))

		elif proc.name == "filter_high":
			steps.append(HighPassFilter(
				freq=float(proc.get("freq", 80.0)),
				resonance_db=float(proc.get("resonance", 0.0)),
			))

		elif proc.name == "filter_band":
			steps.append(BandPassFilter(
				freq=float(proc.get("freq", 1000.0)),
				q=float(proc.get("q", 0.7)),
				resonance_db=float(proc.get("resonance", 0.0)),
			))

		elif proc.name == "reverse":
			steps.append(Reverse())

		elif proc.name == "saturate":
			steps.append(Saturate(
				amount_db=float(proc.get("amount", 6.0)),
			))

		elif proc.name == "compress":
			# Adaptive fields: None = auto-compute from sample analysis.
			# Explicit YAML values override the auto-computation.
			_threshold_raw = proc.get("threshold")
			_attack_raw    = proc.get("attack")
			_release_raw   = proc.get("release")
			steps.append(Compress(
				threshold_db=float(_threshold_raw) if _threshold_raw is not None else None,
				ratio=float(proc.get("ratio", 4.0)),
				attack_ms=float(_attack_raw) if _attack_raw is not None else None,
				release_ms=float(_release_raw) if _release_raw is not None else None,
				knee_db=float(proc.get("knee", 6.0)),
				makeup_db=float(proc.get("makeup", 0.0)),
				lookahead_ms=float(proc.get("lookahead", 0.0)),
			))

		elif proc.name == "limit":
			steps.append(Limit(
				threshold_db=float(proc.get("threshold", -1.0)),
				release_ms=float(proc.get("release", 50.0)),
				lookahead_ms=float(proc.get("lookahead", 5.0)),
			))

		elif proc.name == "hpss_harmonic":
			steps.append(HpssHarmonic())

		elif proc.name == "hpss_percussive":
			steps.append(HpssPercussive())

		elif proc.name == "gate":
			_threshold_raw = proc.get("threshold")
			_attack_raw    = proc.get("attack")
			_release_raw   = proc.get("release")
			_hold_raw      = proc.get("hold")
			_la_raw        = proc.get("lookahead")
			steps.append(Gate(
				threshold_db=float(_threshold_raw) if _threshold_raw is not None else None,
				attack_ms=float(_attack_raw) if _attack_raw is not None else None,
				release_ms=float(_release_raw) if _release_raw is not None else None,
				hold_ms=float(_hold_raw) if _hold_raw is not None else None,
				lookahead_ms=float(_la_raw) if _la_raw is not None else None,
			))

		elif proc.name == "distort":
			_drive_raw = proc.get("drive")
			_tone_raw  = proc.get("tone")
			steps.append(Distort(
				mode=str(proc.get("mode", "hard_clip")),
				drive_db=float(_drive_raw) if _drive_raw is not None else None,
				mix=float(proc.get("mix", 1.0)),
				tone=float(_tone_raw) if _tone_raw is not None else None,
				bit_depth=int(proc.get("bit_depth", 8)),
				downsample_factor=int(proc.get("downsample_factor", 4)),
			))

		elif proc.name == "reshape":
			_attack_raw  = proc.get("attack")
			_decay_raw   = proc.get("decay")
			_release_raw = proc.get("release")
			steps.append(Reshape(
				attack_ms=float(_attack_raw) if _attack_raw is not None else None,
				hold_ms=float(proc.get("hold", 0.0)),
				decay_ms=float(_decay_raw) if _decay_raw is not None else None,
				sustain=float(proc.get("sustain", 1.0)),
				release_ms=float(_release_raw) if _release_raw is not None else None,
			))

		else:
			_log.warning("Unknown processor %r — skipped", proc.name)

	return TransformSpec(steps=tuple(steps))
