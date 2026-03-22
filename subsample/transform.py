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
          → miss: enqueue for next trigger, return None  (player uses original)

  InstrumentLibrary evicts sample_id
      → TransformManager.on_parent_evicted([sample_id])
          → TransformCache.remove_parent(sample_id)

  Target BPM changes
      → TransformManager.on_bpm_change(new_bpm)
          → TransformCache.remove_by_step_type(TimeStretch)
          → TransformProcessor.enqueue_bpm_change(all_rhythmic_records, new_bpm)

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

	target_bpm: The desired playback tempo in BPM.
	The stretch ratio is computed from the parent sample's rhythm.tempo_bpm
	and target_bpm: ratio = source_bpm / target_bpm.
	Unimplemented — handler not yet registered in TransformProcessor._HANDLERS.
	"""

	PRIORITY: typing.ClassVar[int] = TIME_STRETCH_PRIORITY

	target_bpm: float


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
	) -> typing.Optional[TransformResult]:

		"""Convenience: look up a time-stretch-only variant by sample ID and BPM."""

		key = TransformKey(
			sample_id=sample_id,
			spec=TransformSpec(steps=(TimeStretch(target_bpm=target_bpm),)),
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
	# PitchShift registered at module load (Phase 2, pyrubberband).
	# Populate here as further transforms are implemented:
	#   _HANDLERS[EnvelopeAdjust] = _apply_envelope     (Phase 3)
	#   _HANDLERS[TimeStretch]    = _apply_time_stretch (Phase 3)
	_HANDLERS: typing.ClassVar[dict[type, _ApplyFn]] = {}

	def __init__ (
		self,
		sample_rate: int,
		bit_depth:   int,
		on_complete: typing.Optional[_OnTransformComplete] = None,
	) -> None:

		self._sample_rate = sample_rate
		self._bit_depth   = bit_depth
		self._on_complete = on_complete

		n_workers = max(1, ((os.cpu_count() or 1) - 2) // 2)

		self._executor = concurrent.futures.ThreadPoolExecutor(
			max_workers=n_workers,
			thread_name_prefix="transform-worker",
		)

		# Set of in-flight TransformKeys — prevents duplicate jobs.
		self._in_flight:      set[TransformKey]  = set()
		self._in_flight_lock: threading.Lock     = threading.Lock()

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
		# not yet registered.  In Phase 1 _HANDLERS is empty, so all enqueue
		# calls from get_pitched() / on_sample_added() are silently dropped.
		# Once a handler is registered this check passes and jobs run normally.
		if not all(type(step) in self._HANDLERS for step in spec.steps):
			return

		key = TransformKey(sample_id=record.sample_id, spec=spec)

		with self._in_flight_lock:
			if key in self._in_flight:
				return
			self._in_flight.add(key)

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
	) -> None:

		"""Submit time-stretch jobs for all rhythmic samples.

		Only enqueues for records that have a detected tempo
		(rhythm.tempo_bpm > 0).  Samples with no beat grid are skipped.
		"""

		spec = TransformSpec(steps=(TimeStretch(target_bpm=target_bpm),))

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
			assert record.audio is not None

			# Convert integer PCM to float32 preserving all channels.
			audio = _pcm_to_float32(record.audio, self._bit_depth)

			# Apply each step in priority order, dispatching via _HANDLERS.
			for step in spec.steps:
				handler = self._HANDLERS.get(type(step))

				if handler is None:
					raise NotImplementedError(
						f"No handler registered for {type(step).__name__}. "
						"Add an entry to TransformProcessor._HANDLERS — see the "
						"'How to add a new transform type' guide in transform.py."
					)

				audio = handler(audio, self._sample_rate, record, step)

			# Compute level from the mono mix, consistent with how SampleRecord.level
			# was originally computed (analysis.compute_level operates on mono float32).
			level    = subsample.analysis.compute_level(_mix_to_mono(audio))
			duration = audio.shape[0] / self._sample_rate

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
			if record is not None:
				spec = TransformSpec(steps=(PitchShift(target_midi_note=midi_note),))
				self._processor.enqueue(record, spec)

		return result

	def get_at_bpm (
		self,
		sample_id:  int,
		target_bpm: float,
	) -> typing.Optional[TransformResult]:

		"""Return a cached time-stretch variant, or None.

		On a cache miss, enqueues the transform and returns None.
		"""

		result = self._cache.get_stretched(sample_id, target_bpm)

		if result is None:
			record = self._instrument_library.get(sample_id)
			if record is not None:
				spec = TransformSpec(steps=(TimeStretch(target_bpm=target_bpm),))
				self._processor.enqueue(record, spec)

		return result

	def on_sample_added (self, record: "subsample.library.SampleRecord") -> None:

		"""Auto-enqueue pitch variants when a new SampleRecord enters the library.

		If auto_pitch is enabled and the sample has a stable, confident pitch
		(per has_stable_pitch()), enqueues one PitchShift job per MIDI note in
		[center - pitch_range_semitones, center + pitch_range_semitones], clamped
		to [0, 127].  center is the nearest integer MIDI note to the detected Hz.

		The center-note variant micro-corrects tuning when the original recording
		is slightly off-pitch (e.g. 443 Hz is MIDI 69.12; the A4 variant is tuned
		to exactly 440 Hz).

		If target_bpm > 0 and the sample has detected rhythmic content, a
		TimeStretch variant is also enqueued (requires Phase 3 handler).
		"""

		if self._cfg.auto_pitch and self._cfg.pitch_range_semitones > 0:

			if subsample.analysis.has_stable_pitch(
				record.spectral, record.pitch, record.duration,
			):
				center = int(round(librosa.hz_to_midi(record.pitch.dominant_pitch_hz)))
				low    = max(0,   center - self._cfg.pitch_range_semitones)
				high   = min(127, center + self._cfg.pitch_range_semitones)
				notes  = list(range(low, high + 1))

				self._processor.enqueue_pitch_range(record, notes)

				_log.info(
					"Auto-pitch: sample %d (%s) %.1f Hz (MIDI %d) → %d variant(s) [%d–%d]",
					record.sample_id, record.name,
					record.pitch.dominant_pitch_hz, center, len(notes), low, high,
				)

		# Auto-BPM time-stretch: enqueue if target set and sample has detected rhythm.
		# TimeStretch handler not yet registered (Phase 3) — enqueue() silently skips.
		if self._cfg.target_bpm > 0.0 and record.rhythm.tempo_bpm > 0.0:
			spec = TransformSpec(steps=(TimeStretch(target_bpm=self._cfg.target_bpm),))
			self._processor.enqueue(record, spec)

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

		all_records = self._instrument_library.samples()
		self._processor.enqueue_bpm_change(all_records, new_bpm)

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
	Uses pyrubberband with --fine for Rubber Band v3's highest quality offline
	processing.  pyrubberband accepts (n_frames, channels) directly — no
	shape transposition is needed.

	Args:
		audio:       float32, shape (n_frames, channels).
		sample_rate: Hz (e.g. 44100).
		record:      Parent SampleRecord — provides source pitch.
		step:        PitchShift with target_midi_note.

	Returns:
		float32, shape (n_frames, channels) — pitch-shifted audio.
	"""

	source_midi = float(librosa.hz_to_midi(record.pitch.dominant_pitch_hz))
	n_steps     = float(step.target_midi_note) - source_midi

	return pyrubberband.pitch_shift(  # type: ignore[no-any-return]
		audio,
		sample_rate,
		n_steps,
		rbargs={"--fine": ""},
	)


# Register the pitch-shift handler so TransformProcessor can dispatch to it.
# All enqueue() calls for PitchShift specs are now live; previously they were
# silently dropped because _HANDLERS was empty (Phase 1 scaffold).
TransformProcessor._HANDLERS[PitchShift] = _apply_pitch
