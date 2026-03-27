"""Sample transform pipeline — derivative audio production and caching.

Produces *derivative* versions of existing samples (pitch-shifted, envelope-
adjusted, time-stretched) without modifying the originals. Derivatives are
purely in-memory and regenerated on demand — they are never written to disk.

Architecture overview
---------------------

TransformSpec / TransformKey
    An immutable, hashable description of what transforms to apply to which
    sample.  The composite (sample_id, spec) is the unique identity of one
    derivative.

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
    source PCM to float32, apply the registered transform chain in priority
    order, compute a LevelResult, and call on_complete.  New transform types
    are registered in TransformProcessor._HANDLERS — no other code changes
    are needed.

TransformManager
    Single coordination point for the player and cli.py.  Handles:
      - on_sample_added()   : auto-enqueue configured transforms for new samples
      - on_parent_evicted() : propagate instrument-library eviction to the cache
      - on_bpm_change()     : invalidate and re-enqueue all time-stretch variants
      - get_pitched()       : player look-up; enqueues on miss, returns None
      - get_at_bpm()        : player look-up for time-stretch variants

Data flow
---------

  SampleRecord added to InstrumentLibrary
      → TransformManager.on_sample_added()
          → TransformProcessor.enqueue(record, spec)   [auto-pitch / auto-BPM]
              → worker: _pcm_to_float32 → handler chain → compute_level
                  → TransformCache.put(result)
                      → on_complete callback

  MIDI note trigger
      → TransformManager.get_pitched(sample_id, midi_note)
          → hit:  return TransformResult  (player applies gain + pan, no conversion)
          → miss: try next tier
      → TransformManager.get_at_bpm(sample_id)
          → hit:  return TransformResult  (beat-quantized variant)
          → miss/skip: enqueue if qualifying, return None  (player falls back)
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
            PRIORITY: typing.ClassVar[int] = MY_PRIORITY
            my_param: float

    Choose a PRIORITY integer that places the transform at the correct point
    in the signal chain (lower = earlier; see the PRIORITY constants below).

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

4.  Add any convenience look-up methods to TransformManager if needed
    (e.g. get_at_bpm() pattern).

5.  Add config fields to TransformConfig in config.py if tuning is needed.

6.  Wire auto-enqueue logic in TransformManager.on_sample_added() if the
    transform should fire automatically when new samples arrive.

7.  Update README.md (user-facing) and README-AGENTS.md (agent-facing) to
    document the new transform type.
"""

import collections
import concurrent.futures
import dataclasses
import logging
import os
import threading
import typing

import librosa
import numpy
import pyrubberband

import subsample.analysis
import subsample.library

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

# Transform application order is determined by these integer priorities.
# Lower values execute first.  The rationale for the ordering:
#   Pitch (0)       — changes spectral content; must precede envelope shaping
#   Envelope (1)    — shapes the already-pitched signal
#   TimeStretch (2) — last so it preserves the final pitch and envelope
# New transform types should be assigned a priority that puts them at the
# correct point in the signal chain.

PITCH_PRIORITY:         typing.Final[int] = 0
ENVELOPE_PRIORITY:      typing.Final[int] = 1
TIME_STRETCH_PRIORITY:  typing.Final[int] = 2

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

	PRIORITY: typing.ClassVar[int] = PITCH_PRIORITY

	target_midi_note: int


@dataclasses.dataclass(frozen=True)
class EnvelopeAdjust:

	"""Modify the attack and/or release shape of the audio.

	attack_ms:  Target attack duration in milliseconds.  0.0 = unchanged.
	release_ms: Target release duration in milliseconds.  0.0 = unchanged.
	Unimplemented — handler not yet registered in TransformProcessor._HANDLERS.
	"""

	PRIORITY: typing.ClassVar[int] = ENVELOPE_PRIORITY

	attack_ms:  float
	release_ms: float


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

	PRIORITY: typing.ClassVar[int] = TIME_STRETCH_PRIORITY

	target_bpm:  float
	resolution:  int = 16


# Union of all known transform step types.
# Extend this when adding new transforms (see "How to add a new transform type"
# in the module docstring).
TransformStep = typing.Union[PitchShift, EnvelopeAdjust, TimeStretch]

# ---------------------------------------------------------------------------
# TransformSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TransformSpec:

	"""An ordered, hashable set of transforms to apply to a sample.

	steps is always sorted by each step's PRIORITY so that the same set of
	transforms produces an identical spec regardless of construction order.
	This makes TransformSpec safe to use as a dict key and for deduplication.

	An empty steps tuple is the identity spec (no transform applied).
	"""

	steps: tuple[TransformStep, ...]

	def __post_init__ (self) -> None:

		# Sort by class-level PRIORITY so (PitchShift, TimeStretch) and
		# (TimeStretch, PitchShift) produce the same spec and the same cache key.
		sorted_steps = tuple(
			sorted(self.steps, key=lambda s: type(s).PRIORITY)
		)

		# Frozen dataclasses disallow normal attribute assignment; this is the
		# documented pattern for setting computed fields in __post_init__.
		object.__setattr__(self, "steps", sorted_steps)

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
	1. Define a frozen dataclass with PRIORITY: ClassVar[int].
	2. Write an apply function matching _ApplyFn.
	3. Register it in TransformProcessor._HANDLERS:
	       TransformProcessor._HANDLERS[MyTransform] = _apply_my_transform
	   The _execute() loop picks it up automatically — no other changes needed.
	"""

	# Dispatch table: step type → apply function.
	# PitchShift and TimeStretch are registered at module load.
	# Populate here as further transforms are implemented:
	#   _HANDLERS[EnvelopeAdjust] = _apply_envelope
	_HANDLERS: typing.ClassVar[dict[type, _ApplyFn]] = {}

	def __init__ (
		self,
		sample_rate:        int,
		bit_depth:          int,
		output_sample_rate: typing.Optional[int] = None,
		on_complete:        typing.Optional[_OnTransformComplete] = None,
		on_idle:            typing.Optional[typing.Callable[[int], None]] = None,
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
		# not yet registered (e.g. EnvelopeAdjust).  Once a handler is
		# registered the check passes and jobs run normally.
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
	) -> None:

		"""Submit one pitch-shift job per MIDI note in the list."""

		for note in midi_notes:
			self.enqueue(
				record,
				TransformSpec(steps=(PitchShift(target_midi_note=note),)),
			)

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
	) -> None:

		self._cache              = cache
		self._processor          = processor
		self._instrument_library = instrument_library
		self._cfg                = cfg

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
	) -> None:

		"""Enqueue pitch-shift variants for a set of MIDI notes.

		Delegates to the underlying TransformProcessor, which deduplicates
		in-flight and already-cached keys — safe to call repeatedly.

		Called by MidiPlayer.update_pitched_assignments() when a pitched
		keyboard assignment's best match changes and the new set of variants
		must be pre-computed before the next trigger.

		No-ops when auto_pitch is disabled — consistent with the player's
		overall pitch-variant production policy.

		Args:
			record:     SampleRecord to produce pitch-shifted variants from.
			midi_notes: MIDI note numbers to pre-compute.
		"""

		if not self._cfg.auto_pitch:
			return

		self._processor.enqueue_pitch_range(record, midi_notes)

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
	Uses pyrubberband with --fine --smoothing for Rubber Band v3's highest
	quality offline processing with frequency-domain smoothing to reduce
	phasing artefacts.  pyrubberband accepts (n_frames, channels) directly — no
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
		rbargs={"--fine": "", "--smoothing": ""},
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

	return pyrubberband.timemap_stretch(  # type: ignore[no-any-return]
		audio, sample_rate, time_map, rbargs={"--fine": "", "--smoothing": ""},
	)


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

# Register the pitch-shift handler so TransformProcessor can dispatch to it.
# All enqueue() calls for PitchShift specs are now live; previously they were
# silently dropped because _HANDLERS was empty (Phase 1 scaffold).
TransformProcessor._HANDLERS[PitchShift] = _apply_pitch

# Register the time-stretch handler — beat-quantized non-linear stretching
# via pyrubberband.timemap_stretch() with Rubber Band's offline finer engine
# and frequency-domain smoothing (--fine --smoothing).
TransformProcessor._HANDLERS[TimeStretch] = _apply_time_stretch
