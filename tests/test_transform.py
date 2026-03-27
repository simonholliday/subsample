"""Tests for subsample/transform.py — the sample transform pipeline scaffold."""

import pathlib
import threading
import time
import typing

import numpy
import pytest

import subsample.analysis
import subsample.config
import subsample.library
import subsample.transform

import tests.helpers


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_audio (
	n_frames: int = 4410,
	channels: int = 1,
	dtype: numpy.dtype = numpy.dtype("float32"),
) -> numpy.ndarray:

	"""Return a silent float32 audio array of the requested shape."""

	return numpy.zeros((n_frames, channels), dtype=dtype)


def _make_pcm_audio (
	n_frames: int = 4410,
	channels: int = 1,
) -> numpy.ndarray:

	"""Return a silent int16 PCM array (as stored in SampleRecord.audio)."""

	return numpy.zeros((n_frames, channels), dtype=numpy.int16)


def _make_record (
	sample_id: int = 1,
	audio: typing.Optional[numpy.ndarray] = None,
	tempo_bpm: float = 0.0,
	onset_times: tuple[float, ...] = (),
	attack_times: typing.Optional[tuple[float, ...]] = None,
) -> subsample.library.SampleRecord:

	"""Return a minimal SampleRecord suitable for transform tests."""

	if audio is None:
		audio = _make_pcm_audio()

	# Default attack_times to onset_times if not provided.
	if attack_times is None:
		attack_times = onset_times

	rhythm = subsample.analysis.RhythmResult(
		tempo_bpm        = tempo_bpm,
		beat_times       = (),
		pulse_curve      = numpy.zeros(0, dtype=numpy.float32),
		pulse_peak_times = (),
		onset_times      = onset_times,
		attack_times     = attack_times,
		onset_count      = len(onset_times),
	)

	return subsample.library.SampleRecord(
		sample_id  = sample_id,
		name       = f"test_{sample_id}",
		spectral   = tests.helpers._make_spectral(),
		rhythm     = rhythm,
		pitch      = tests.helpers._make_pitch(),
		timbre     = tests.helpers._make_timbre(),
		level      = tests.helpers._make_level(),
		band_energy = tests.helpers._make_band_energy(),
		params     = tests.helpers._make_params(),
		duration   = float(audio.shape[0]) / 44100.0,
		audio      = audio,
	)


def _make_record_unpitched (
	sample_id: int = 1,
) -> subsample.library.SampleRecord:

	"""Return a SampleRecord that fails has_stable_pitch() (dominant_pitch_hz=0)."""

	pitch = subsample.analysis.PitchResult(
		dominant_pitch_hz    = 0.0,
		pitch_confidence     = 0.0,
		chroma_profile       = tuple(0.0 for _ in range(12)),
		dominant_pitch_class = -1,
		pitch_stability      = 0.0,
		voiced_frame_count   = 0,
	)

	spectral = subsample.analysis.AnalysisResult(
		spectral_flatness  = 0.9,
		attack             = 0.5,
		release            = 0.5,
		spectral_centroid  = 0.5,
		spectral_bandwidth = 0.5,
		zcr                = 0.8,
		harmonic_ratio     = 0.1,
		spectral_contrast  = 0.3,
		voiced_fraction    = 0.1,
		log_attack_time    = 0.5,
		spectral_flux      = 0.5,
	)

	audio = _make_pcm_audio()

	return subsample.library.SampleRecord(
		sample_id  = sample_id,
		name       = f"unpitched_{sample_id}",
		spectral   = spectral,
		rhythm     = subsample.analysis.RhythmResult(
			tempo_bpm=0.0, beat_times=(), pulse_curve=numpy.zeros(0, dtype=numpy.float32),
			pulse_peak_times=(), onset_times=(), attack_times=(), onset_count=0,
		),
		pitch      = pitch,
		timbre     = tests.helpers._make_timbre(),
		level      = tests.helpers._make_level(),
		band_energy = tests.helpers._make_band_energy(),
		params     = tests.helpers._make_params(),
		duration   = float(audio.shape[0]) / 44100.0,
		audio      = audio,
	)


def _make_result (
	sample_id: int = 1,
	spec: typing.Optional[subsample.transform.TransformSpec] = None,
	n_frames: int = 4410,
	channels: int = 1,
) -> subsample.transform.TransformResult:

	"""Return a minimal TransformResult for cache tests."""

	if spec is None:
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)

	key   = subsample.transform.TransformKey(sample_id=sample_id, spec=spec)
	audio = _make_audio(n_frames=n_frames, channels=channels)
	level = subsample.analysis.LevelResult(peak=0.8, rms=0.2)

	return subsample.transform.TransformResult(
		key=key, audio=audio, duration=n_frames / 44100.0, level=level
	)


# ---------------------------------------------------------------------------
# TestTransformSpec
# ---------------------------------------------------------------------------

class TestTransformSpec:

	"""TransformSpec sorts steps by PRIORITY regardless of construction order."""

	def test_empty_spec_is_identity (self) -> None:
		spec = subsample.transform.TransformSpec(steps=())
		assert spec.steps == ()

	def test_single_step_unchanged (self) -> None:
		step = subsample.transform.PitchShift(target_midi_note=60)
		spec = subsample.transform.TransformSpec(steps=(step,))
		assert spec.steps == (step,)

	def test_steps_sorted_by_priority (self) -> None:
		"""Same steps in different order must produce the same spec."""
		pitch   = subsample.transform.PitchShift(target_midi_note=60)
		stretch = subsample.transform.TimeStretch(target_bpm=120.0)

		spec_a = subsample.transform.TransformSpec(steps=(stretch, pitch))
		spec_b = subsample.transform.TransformSpec(steps=(pitch, stretch))

		assert spec_a == spec_b
		assert spec_a.steps[0] == pitch
		assert spec_a.steps[1] == stretch

	def test_all_three_sorted (self) -> None:
		pitch    = subsample.transform.PitchShift(target_midi_note=69)
		envelope = subsample.transform.EnvelopeAdjust(attack_ms=10.0, release_ms=50.0)
		stretch  = subsample.transform.TimeStretch(target_bpm=90.0)

		# Provide in reverse priority order
		spec = subsample.transform.TransformSpec(steps=(stretch, envelope, pitch))

		assert spec.steps == (pitch, envelope, stretch)

	def test_spec_is_hashable (self) -> None:
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		d: dict[subsample.transform.TransformSpec, int] = {spec: 1}
		assert d[spec] == 1


# ---------------------------------------------------------------------------
# TestTransformKey
# ---------------------------------------------------------------------------

class TestTransformKey:

	"""TransformKey is a hashable, equality-comparable composite identity."""

	def test_same_inputs_are_equal (self) -> None:
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		a = subsample.transform.TransformKey(sample_id=1, spec=spec)
		b = subsample.transform.TransformKey(sample_id=1, spec=spec)
		assert a == b

	def test_different_sample_ids_not_equal (self) -> None:
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		a = subsample.transform.TransformKey(sample_id=1, spec=spec)
		b = subsample.transform.TransformKey(sample_id=2, spec=spec)
		assert a != b

	def test_usable_as_dict_key (self) -> None:
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		key = subsample.transform.TransformKey(sample_id=1, spec=spec)
		d: dict[subsample.transform.TransformKey, str] = {key: "hit"}
		assert d[key] == "hit"

	def test_usable_in_set (self) -> None:
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		k1 = subsample.transform.TransformKey(sample_id=1, spec=spec)
		k2 = subsample.transform.TransformKey(sample_id=1, spec=spec)
		k3 = subsample.transform.TransformKey(sample_id=2, spec=spec)

		s = {k1, k2, k3}

		assert len(s) == 2


# ---------------------------------------------------------------------------
# TestTransformCache
# ---------------------------------------------------------------------------

class TestTransformCache:

	"""TransformCache stores derivatives and evicts by parent-priority FIFO."""

	def _make_cache (self, max_mb: float = 10.0) -> subsample.transform.TransformCache:
		return subsample.transform.TransformCache(
			max_memory_bytes=int(max_mb * 1024 * 1024)
		)

	def test_put_and_get_round_trip (self) -> None:
		cache  = self._make_cache()
		result = _make_result(sample_id=1, n_frames=1000)

		evicted = cache.put(result)

		assert evicted == []
		assert cache.get(result.key) is result

	def test_get_returns_none_for_missing_key (self) -> None:
		cache = self._make_cache()
		key   = subsample.transform.TransformKey(
			sample_id=99,
			spec=subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=60),)
			),
		)

		assert cache.get(key) is None

	def test_get_pitched_convenience (self) -> None:
		cache  = self._make_cache()
		result = _make_result(sample_id=5)
		cache.put(result)

		hit = cache.get_pitched(5, 60)
		assert hit is result

	def test_get_pitched_miss (self) -> None:
		cache = self._make_cache()
		assert cache.get_pitched(5, 60) is None

	def test_get_stretched_convenience (self) -> None:
		cache = self._make_cache()
		spec  = subsample.transform.TransformSpec(
			steps=(subsample.transform.TimeStretch(target_bpm=120.0),)
		)
		result = _make_result(sample_id=7, spec=spec)
		cache.put(result)

		hit = cache.get_stretched(7, 120.0)
		assert hit is result

	def test_has_variants_true (self) -> None:
		cache  = self._make_cache()
		result = _make_result(sample_id=1)
		cache.put(result)
		assert cache.has_variants(1) is True

	def test_has_variants_false (self) -> None:
		cache = self._make_cache()
		assert cache.has_variants(99) is False

	def test_list_variants (self) -> None:
		cache = self._make_cache()
		r1    = _make_result(sample_id=3,
			spec=subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=60),)
			))
		r2    = _make_result(sample_id=3,
			spec=subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=62),)
			))
		cache.put(r1)
		cache.put(r2)

		variants = cache.list_variants(3)

		assert len(variants) == 2
		assert r1.key in variants
		assert r2.key in variants

	def test_remove_parent_evicts_all_derivatives (self) -> None:
		cache = self._make_cache()
		r1    = _make_result(sample_id=4,
			spec=subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=60),)
			))
		r2    = _make_result(sample_id=4,
			spec=subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=62),)
			))
		cache.put(r1)
		cache.put(r2)

		evicted = cache.remove_parent(4)

		assert len(evicted) == 2
		assert not cache.has_variants(4)
		assert cache.get(r1.key) is None
		assert cache.get(r2.key) is None

	def test_remove_parent_noop_for_unknown (self) -> None:
		cache   = self._make_cache()
		evicted = cache.remove_parent(999)
		assert evicted == []

	def test_remove_by_step_type (self) -> None:
		cache = self._make_cache()

		pitch_spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		stretch_spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.TimeStretch(target_bpm=120.0),)
		)

		r_pitch   = _make_result(sample_id=1, spec=pitch_spec)
		r_stretch = _make_result(sample_id=2, spec=stretch_spec)
		cache.put(r_pitch)
		cache.put(r_stretch)

		evicted = cache.remove_by_step_type(subsample.transform.TimeStretch)

		assert len(evicted) == 1
		assert evicted[0] == r_stretch.key
		assert cache.get(r_pitch.key) is r_pitch
		assert cache.get(r_stretch.key) is None

	def test_memory_tracking (self) -> None:
		cache = self._make_cache()
		result = _make_result(n_frames=4410, channels=1)  # 4410 * 4 = 17640 bytes

		cache.put(result)

		assert cache.memory_used == result.audio.nbytes

	def test_parent_priority_fifo_eviction (self) -> None:
		"""When over budget, the oldest parent's variants are evicted first."""

		# Use a very small budget: fits ~2 results of 4410 float32 frames
		# Each result: 4410 * 4 bytes = 17640 bytes
		budget = 3 * 17640  # fits 3, so 4th triggers eviction
		cache  = subsample.transform.TransformCache(max_memory_bytes=budget)

		# Add two variants for parent 1 (oldest), one for parent 2
		spec_a = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		spec_b = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=62),)
		)
		spec_c = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)

		r1a = _make_result(sample_id=1, spec=spec_a)
		r1b = _make_result(sample_id=1, spec=spec_b)
		r2  = _make_result(sample_id=2, spec=spec_c)

		cache.put(r1a)
		cache.put(r1b)
		cache.put(r2)

		# Now add a 4th result (parent 3) that pushes over budget.
		# Parent 1 (oldest) should be evicted wholesale.
		spec_d = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)
		r3 = _make_result(sample_id=3, spec=spec_d)

		evicted = cache.put(r3)

		assert len(evicted) == 2
		assert all(k.sample_id == 1 for k in evicted)
		assert not cache.has_variants(1)
		assert cache.has_variants(2)
		assert cache.has_variants(3)

	def test_memory_limit_property (self) -> None:
		budget = 5 * 1024 * 1024
		cache  = subsample.transform.TransformCache(max_memory_bytes=budget)
		assert cache.memory_limit == budget

	def test_cascade_removes_all_on_remove_parent (self) -> None:
		"""All variants for a parent are gone after remove_parent."""
		cache = self._make_cache()
		for note in range(60, 65):
			spec   = subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=note),)
			)
			cache.put(_make_result(sample_id=10, spec=spec))

		assert cache.has_variants(10)
		cache.remove_parent(10)
		assert not cache.has_variants(10)
		assert cache.memory_used == 0


# ---------------------------------------------------------------------------
# TestTransformProcessor
# ---------------------------------------------------------------------------

class TestTransformProcessor:

	"""TransformProcessor deduplicates in-flight jobs; raises on unregistered handlers."""

	def test_enqueue_skips_record_with_no_audio (self) -> None:
		processor = subsample.transform.TransformProcessor(sample_rate=44100, bit_depth=16)
		record    = _make_record(audio=None)
		spec      = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)

		# Should not raise; just a silent no-op
		processor.enqueue(record, spec)
		processor.shutdown()

	def test_enqueue_skips_when_handler_not_registered (self) -> None:
		"""enqueue() is a no-op when no handler is registered for a step.

		Temporarily clears _HANDLERS to simulate a transform type with no
		implementation.  Submitting such jobs is prevented at enqueue() time
		so no errors are logged.
		"""

		completed: list[subsample.transform.TransformResult] = []

		# Temporarily clear all handlers to simulate an unregistered transform.
		original_handlers = dict(subsample.transform.TransformProcessor._HANDLERS)

		try:
			subsample.transform.TransformProcessor._HANDLERS.clear()

			processor = subsample.transform.TransformProcessor(
				sample_rate=44100,
				bit_depth=16,
				on_complete=completed.append,
			)

			record = _make_record(sample_id=1)
			spec   = subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=60),)
			)

			assert subsample.transform.TransformProcessor._HANDLERS == {}

			processor.enqueue(record, spec)
			processor.shutdown()

			# Nothing should have been submitted or completed.
			assert completed == []

		finally:
			subsample.transform.TransformProcessor._HANDLERS.clear()
			subsample.transform.TransformProcessor._HANDLERS.update(original_handlers)

	def test_enqueue_deduplication (self) -> None:
		"""Submitting the same (record, spec) twice should not double-run."""

		completed: list[subsample.transform.TransformResult] = []

		# Register a dummy no-op handler so jobs complete without error
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),)
		)

		def _dummy_handler (
			audio:       numpy.ndarray,
			sample_rate: int,
			record:      subsample.library.SampleRecord,
			step:        subsample.transform.PitchShift,
		) -> numpy.ndarray:
			time.sleep(0.05)  # simulate work so the dedup window stays open
			return audio

		original_handlers = dict(subsample.transform.TransformProcessor._HANDLERS)

		try:
			subsample.transform.TransformProcessor._HANDLERS[
				subsample.transform.PitchShift
			] = _dummy_handler  # type: ignore[assignment]

			processor = subsample.transform.TransformProcessor(
				sample_rate=44100,
				bit_depth=16,
				on_complete=completed.append,
			)

			record = _make_record(sample_id=1)

			# Submit the same job twice in quick succession
			processor.enqueue(record, spec)
			processor.enqueue(record, spec)  # should be de-duplicated

			processor.shutdown()

			assert len(completed) == 1

		finally:
			subsample.transform.TransformProcessor._HANDLERS.clear()
			subsample.transform.TransformProcessor._HANDLERS.update(original_handlers)

	def test_enqueue_pitch_range (self) -> None:
		"""enqueue_pitch_range submits one job per MIDI note."""

		completed: list[subsample.transform.TransformResult] = []

		def _passthrough (
			audio:       numpy.ndarray,
			sample_rate: int,
			record:      subsample.library.SampleRecord,
			step:        subsample.transform.PitchShift,
		) -> numpy.ndarray:
			return audio

		original_handlers = dict(subsample.transform.TransformProcessor._HANDLERS)

		try:
			subsample.transform.TransformProcessor._HANDLERS[
				subsample.transform.PitchShift
			] = _passthrough  # type: ignore[assignment]

			processor = subsample.transform.TransformProcessor(
				sample_rate=44100,
				bit_depth=16,
				on_complete=completed.append,
			)

			record = _make_record(sample_id=2)
			notes  = list(range(60, 65))  # 5 notes

			processor.enqueue_pitch_range(record, notes)
			processor.shutdown()

			assert len(completed) == 5
			result_notes = {
				r.key.spec.steps[0].target_midi_note  # type: ignore[union-attr]
				for r in completed
			}
			assert result_notes == set(notes)

		finally:
			subsample.transform.TransformProcessor._HANDLERS.clear()
			subsample.transform.TransformProcessor._HANDLERS.update(original_handlers)

	def test_enqueue_bpm_change_skips_non_rhythmic (self) -> None:
		"""enqueue_bpm_change only processes records with tempo_bpm > 0."""

		completed: list[subsample.transform.TransformResult] = []

		def _passthrough (
			audio:       numpy.ndarray,
			sample_rate: int,
			record:      subsample.library.SampleRecord,
			step:        subsample.transform.TimeStretch,
		) -> numpy.ndarray:
			return audio

		original_handlers = dict(subsample.transform.TransformProcessor._HANDLERS)

		try:
			subsample.transform.TransformProcessor._HANDLERS[
				subsample.transform.TimeStretch
			] = _passthrough  # type: ignore[assignment]

			processor = subsample.transform.TransformProcessor(
				sample_rate=44100,
				bit_depth=16,
				on_complete=completed.append,
			)

			rhythmic     = _make_record(sample_id=1, tempo_bpm=120.0)
			non_rhythmic = _make_record(sample_id=2, tempo_bpm=0.0)

			processor.enqueue_bpm_change([rhythmic, non_rhythmic], target_bpm=130.0)
			processor.shutdown()

			assert len(completed) == 1
			assert completed[0].key.sample_id == 1

		finally:
			subsample.transform.TransformProcessor._HANDLERS.clear()
			subsample.transform.TransformProcessor._HANDLERS.update(original_handlers)


# ---------------------------------------------------------------------------
# TestTransformManager
# ---------------------------------------------------------------------------

class TestTransformManager:

	"""TransformManager coordinates cache, processor, and library."""

	def _make_manager (
		self,
		max_mb: float = 10.0,
	) -> tuple[
		subsample.transform.TransformManager,
		subsample.transform.TransformCache,
		subsample.library.InstrumentLibrary,
	]:
		lib   = subsample.library.InstrumentLibrary(max_memory_bytes=100 * 1024 * 1024)
		cache = subsample.transform.TransformCache(
			max_memory_bytes=int(max_mb * 1024 * 1024)
		)
		processor = subsample.transform.TransformProcessor(
			sample_rate=44100,
			bit_depth=16,
			on_complete=cache.put,
		)
		cfg = subsample.config.TransformConfig()
		manager = subsample.transform.TransformManager(
			cache=cache,
			processor=processor,
			instrument_library=lib,
			cfg=cfg,
		)
		return manager, cache, lib

	def test_get_pitched_returns_none_when_not_cached (self) -> None:
		manager, _, _ = self._make_manager()
		result = manager.get_pitched(sample_id=1, midi_note=60)
		assert result is None
		manager.shutdown()

	def test_get_pitched_returns_cached_result (self) -> None:
		manager, cache, _ = self._make_manager()
		r = _make_result(sample_id=5)
		cache.put(r)

		result = manager.get_pitched(sample_id=5, midi_note=60)
		assert result is r

		manager.shutdown()

	def test_on_parent_evicted_clears_cache (self) -> None:
		manager, cache, _ = self._make_manager()

		for note in range(60, 63):
			spec = subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=note),)
			)
			cache.put(_make_result(sample_id=10, spec=spec))

		assert cache.has_variants(10)

		manager.on_parent_evicted([10])

		assert not cache.has_variants(10)
		manager.shutdown()

	def test_on_parent_evicted_multiple_ids (self) -> None:
		manager, cache, _ = self._make_manager()

		for sid in [1, 2, 3]:
			spec = subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=60),)
			)
			cache.put(_make_result(sample_id=sid, spec=spec))

		manager.on_parent_evicted([1, 3])

		assert not cache.has_variants(1)
		assert cache.has_variants(2)
		assert not cache.has_variants(3)
		manager.shutdown()

	def test_has_pitch_variant_true (self) -> None:
		manager, cache, _ = self._make_manager()
		cache.put(_make_result(sample_id=7))

		assert manager.has_pitch_variant(7, 60) is True
		manager.shutdown()

	def test_has_pitch_variant_false (self) -> None:
		manager, _, _ = self._make_manager()
		assert manager.has_pitch_variant(99, 60) is False
		manager.shutdown()

	def test_list_variants_delegates_to_cache (self) -> None:
		manager, cache, _ = self._make_manager()

		for note in range(60, 63):
			spec = subsample.transform.TransformSpec(
				steps=(subsample.transform.PitchShift(target_midi_note=note),)
			)
			cache.put(_make_result(sample_id=20, spec=spec))

		variants = manager.list_variants(20)
		assert len(variants) == 3
		manager.shutdown()

	def test_on_sample_added_enqueues_base_variant_only (self) -> None:
		"""on_sample_added enqueues only the base variant, even for tonal samples.

		Pitch variants are driven by MidiPlayer.update_pitched_assignments() which
		reads the MIDI map to determine the exact note range needed — on_sample_added()
		does not apply any semitone cap.
		"""
		manager, cache, lib = self._make_manager()
		record = _make_record(sample_id=1)
		lib.add(record)

		manager.on_sample_added(record)
		manager.shutdown()

		# Only the base variant (identity spec) — no pitch variants from on_sample_added.
		assert cache.has_variants(1)
		assert len(cache.list_variants(1)) == 1
		assert cache.get_base(1) is not None

	def test_on_sample_added_skips_unpitched (self) -> None:
		"""on_sample_added produces a base variant even for samples that fail has_stable_pitch."""
		manager, cache, lib = self._make_manager()
		record = _make_record_unpitched(sample_id=2)
		lib.add(record)

		manager.on_sample_added(record)
		manager.shutdown()

		# Base variant is always created; no pitch variants for unpitched samples.
		assert cache.has_variants(2)
		assert len(cache.list_variants(2)) == 1
		assert cache.get_base(2) is not None

	def test_enqueue_pitch_range_respects_auto_pitch_false (self) -> None:
		"""auto_pitch=False causes enqueue_pitch_range() to be a no-op."""
		lib   = subsample.library.InstrumentLibrary(max_memory_bytes=100 * 1024 * 1024)
		cache = subsample.transform.TransformCache(max_memory_bytes=50 * 1024 * 1024)
		processor = subsample.transform.TransformProcessor(
			sample_rate=44100, bit_depth=16, on_complete=cache.put,
		)
		cfg = subsample.config.TransformConfig(auto_pitch=False)
		manager = subsample.transform.TransformManager(
			cache=cache, processor=processor,
			instrument_library=lib, cfg=cfg,
		)

		record = _make_record(sample_id=3)
		lib.add(record)
		manager.on_sample_added(record)

		# Even when enqueue_pitch_range is called explicitly, auto_pitch=False blocks it.
		manager.enqueue_pitch_range(record, list(range(60, 73)))

		manager.shutdown()

		# Base variant only — enqueue_pitch_range was a no-op.
		assert cache.has_variants(3)
		assert len(cache.list_variants(3)) == 1
		assert cache.get_base(3) is not None


# ---------------------------------------------------------------------------
# TestSampleRateConversion
# ---------------------------------------------------------------------------

class TestSampleRateConversion:

	"""TransformProcessor resamples variants when output_sample_rate differs."""

	def test_base_variant_resampled_to_output_rate (self) -> None:
		"""Base variant audio length reflects the output sample rate, not the recorder rate."""
		completed: list[subsample.transform.TransformResult] = []

		processor = subsample.transform.TransformProcessor(
			sample_rate=44100,
			output_sample_rate=48000,
			bit_depth=16,
			on_complete=completed.append,
		)

		record = _make_record(sample_id=1)
		processor.enqueue(record, subsample.transform._BASE_VARIANT_SPEC)
		processor.shutdown()

		assert len(completed) == 1
		result = completed[0]

		# n_frames / output_rate should match n_frames / input_rate
		# (i.e. duration is preserved, frame count scales with rate).
		original_frames = record.audio.shape[0]   # type: ignore[union-attr]
		expected_frames = int(round(original_frames * 48000 / 44100))
		assert abs(result.audio.shape[0] - expected_frames) <= 2  # allow 1–2 rounding frames

	def test_no_resampling_when_rates_match (self) -> None:
		"""When capture and output rates match, frame count is unchanged."""
		completed: list[subsample.transform.TransformResult] = []

		processor = subsample.transform.TransformProcessor(
			sample_rate=44100,
			output_sample_rate=44100,
			bit_depth=16,
			on_complete=completed.append,
		)

		record = _make_record(sample_id=2)
		processor.enqueue(record, subsample.transform._BASE_VARIANT_SPEC)
		processor.shutdown()

		assert len(completed) == 1
		original_frames = record.audio.shape[0]   # type: ignore[union-attr]
		assert completed[0].audio.shape[0] == original_frames

	def test_duration_uses_output_rate (self) -> None:
		"""TransformResult.duration is computed at the output sample rate."""
		completed: list[subsample.transform.TransformResult] = []

		processor = subsample.transform.TransformProcessor(
			sample_rate=44100,
			output_sample_rate=48000,
			bit_depth=16,
			on_complete=completed.append,
		)

		record = _make_record(sample_id=3)
		processor.enqueue(record, subsample.transform._BASE_VARIANT_SPEC)
		processor.shutdown()

		assert len(completed) == 1
		result = completed[0]

		# Duration should equal n_output_frames / output_rate, which equals the
		# original duration (time-preserved resampling).
		expected_duration = result.audio.shape[0] / 48000
		assert abs(result.duration - expected_duration) < 1e-6


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestAudioHelpers:

	"""Unit tests for the private audio conversion helpers."""

	def test_pcm_to_float32_preserves_channels (self) -> None:
		pcm = numpy.array([[1000, -1000], [2000, -2000]], dtype=numpy.int16)
		out = subsample.transform._pcm_to_float32(pcm, bit_depth=16)
		assert out.shape == (2, 2)
		assert out.dtype == numpy.float32

	def test_pcm_to_float32_normalises_16bit (self) -> None:
		# Full-scale positive int16 → ~1.0
		pcm = numpy.array([[32767]], dtype=numpy.int16)
		out = subsample.transform._pcm_to_float32(pcm, bit_depth=16)
		assert abs(out[0, 0] - 1.0) < 0.001

	def test_mix_to_mono_stereo (self) -> None:
		audio = numpy.array([[1.0, 0.0], [0.5, 0.5]], dtype=numpy.float32)
		mono  = subsample.transform._mix_to_mono(audio)
		assert mono.shape == (2,)
		assert abs(mono[0] - 0.5) < 1e-6
		assert abs(mono[1] - 0.5) < 1e-6

	def test_mix_to_mono_single_channel (self) -> None:
		audio = numpy.array([[0.3], [0.7]], dtype=numpy.float32)
		mono  = subsample.transform._mix_to_mono(audio)
		assert mono.shape == (2,)
		assert abs(mono[0] - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# TestApplyPitch
# ---------------------------------------------------------------------------

class TestApplyPitch:

	"""Tests for the _apply_pitch handler and its Rubber Band integration."""

	def test_returns_same_shape_mono (self) -> None:
		"""Output has the same (n_frames, 1) shape as the mono input."""
		audio  = numpy.random.default_rng(0).standard_normal((4410, 1)).astype(numpy.float32) * 0.1
		record = _make_record(sample_id=1)
		step   = subsample.transform.PitchShift(target_midi_note=72)

		result = subsample.transform._apply_pitch(audio, 44100, record, step)

		assert result.shape == audio.shape
		assert result.dtype == numpy.float32

	def test_returns_same_shape_stereo (self) -> None:
		"""Output has the same (n_frames, 2) shape as a stereo input."""
		audio  = numpy.random.default_rng(1).standard_normal((4410, 2)).astype(numpy.float32) * 0.1
		record = _make_record(sample_id=1)
		step   = subsample.transform.PitchShift(target_midi_note=60)

		result = subsample.transform._apply_pitch(audio, 44100, record, step)

		assert result.shape == audio.shape
		assert result.dtype == numpy.float32

	def test_upward_shift_produces_different_audio (self) -> None:
		"""Shifting a sine wave up by an octave produces distinct output."""
		t     = numpy.linspace(0, 0.1, 4410, endpoint=False, dtype=numpy.float32)
		sine  = numpy.sin(2 * numpy.pi * 440.0 * t)
		audio = sine[:, numpy.newaxis]  # (4410, 1)

		# Default _make_pitch() has dominant_pitch_hz=440.0 (MIDI 69)
		record = _make_record(sample_id=1)
		step   = subsample.transform.PitchShift(target_midi_note=81)  # +12 semitones

		result = subsample.transform._apply_pitch(audio, 44100, record, step)

		assert result.shape == audio.shape
		# The octave-shifted output should differ from the original
		assert not numpy.allclose(result, audio, atol=0.01)


# ---------------------------------------------------------------------------
# TestTransformConfig
# ---------------------------------------------------------------------------

class TestTransformConfig:

	"""Config defaults load correctly and validation fires on bad values."""

	_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml.default"

	def test_default_transform_values (self) -> None:
		cfg = subsample.config.load_config(self._DEFAULT_CONFIG_PATH)
		assert cfg.transform.max_memory_mb       == 50.0
		assert cfg.transform.auto_pitch          is True
		assert cfg.transform.target_bpm          == 0.0
		assert cfg.transform.quantize_resolution == 16

	def test_valid_quantize_resolutions (self) -> None:

		"""All valid subdivision values (1, 2, 4, 8, 16) are accepted."""

		for res in (1, 2, 4, 8, 16):
			cfg = subsample.config.TransformConfig(quantize_resolution=res)
			assert cfg.quantize_resolution == res

	def test_invalid_quantize_resolution_raises (self) -> None:

		"""Values outside {1, 2, 4, 8, 16} are rejected by the parser."""

		# Load the full default config as a raw dict, then inject bad values.
		import yaml
		base_raw: dict[str, typing.Any] = yaml.safe_load(
			self._DEFAULT_CONFIG_PATH.read_text()
		) or {}

		for bad_value in (0, 3, 6, 7, 32):
			raw = dict(base_raw)
			raw["transform"] = {"quantize_resolution": bad_value}

			with pytest.raises(ValueError, match="quantize_resolution"):
				subsample.config._build_config(raw)


# ---------------------------------------------------------------------------
# TestQuantizeGrid
# ---------------------------------------------------------------------------

class TestQuantizeGrid:

	"""Tests for _build_quantize_grid()."""

	def test_quarter_notes_120bpm (self) -> None:

		"""At 120 BPM, resolution=4 → 0.5s grid spacing."""

		grid = subsample.transform._build_quantize_grid(120.0, 4, 2.0)
		assert grid[0] == 0.0
		assert abs(grid[1] - 0.5) < 1e-9
		assert abs(grid[2] - 1.0) < 1e-9

	def test_sixteenth_notes_120bpm (self) -> None:

		"""At 120 BPM, resolution=16 → 0.125s grid spacing."""

		grid = subsample.transform._build_quantize_grid(120.0, 16, 1.0)
		assert abs(grid[1] - 0.125) < 1e-9

	def test_eighth_notes_90bpm (self) -> None:

		"""At 90 BPM, resolution=8 → 1/(1.5*2) ≈ 0.333s grid spacing."""

		grid = subsample.transform._build_quantize_grid(90.0, 8, 1.0)
		expected = 60.0 / 90.0 / 2.0
		assert abs(grid[1] - expected) < 1e-9

	def test_grid_covers_max_time (self) -> None:

		"""Grid extends at least to the requested maximum time."""

		grid = subsample.transform._build_quantize_grid(120.0, 16, 5.0)
		assert grid[-1] >= 5.0


# ---------------------------------------------------------------------------
# TestSnapOnsets
# ---------------------------------------------------------------------------

class TestSnapOnsets:

	"""Tests for _snap_onsets_to_grid()."""

	def test_onsets_snap_to_nearest (self) -> None:

		"""Each onset lands on the closest grid point."""

		grid = subsample.transform._build_quantize_grid(120.0, 16, 2.0)

		# Onsets near 0.12 and 0.63 should snap to 0.125 and 0.625.
		result = subsample.transform._snap_onsets_to_grid((0.12, 0.63), grid)
		assert abs(result[0] - 0.125) < 1e-9
		assert abs(result[1] - 0.625) < 1e-9

	def test_no_two_onsets_on_same_grid_point (self) -> None:

		"""Tightly spaced onsets are pushed to successive grid points."""

		grid = subsample.transform._build_quantize_grid(120.0, 16, 2.0)

		# Three onsets all near 0.125 — first gets 0.125, others are pushed.
		result = subsample.transform._snap_onsets_to_grid((0.10, 0.11, 0.12), grid)
		assert result[0] < result[1] < result[2]

	def test_onsets_already_on_grid (self) -> None:

		"""Onsets exactly on grid points remain unchanged."""

		grid = subsample.transform._build_quantize_grid(120.0, 16, 2.0)
		result = subsample.transform._snap_onsets_to_grid((0.0, 0.125, 0.25), grid)
		assert abs(result[0] - 0.0)   < 1e-9
		assert abs(result[1] - 0.125) < 1e-9
		assert abs(result[2] - 0.25)  < 1e-9

	def test_monotonically_increasing (self) -> None:

		"""Output target times are always strictly increasing."""

		grid = subsample.transform._build_quantize_grid(100.0, 8, 4.0)
		onsets = (0.05, 0.35, 0.65, 0.95, 1.25)
		result = subsample.transform._snap_onsets_to_grid(onsets, grid)

		for i in range(1, len(result)):
			assert result[i] > result[i - 1]

	def test_empty_onsets (self) -> None:

		"""No onsets produces an empty result."""

		grid = subsample.transform._build_quantize_grid(120.0, 16, 2.0)
		result = subsample.transform._snap_onsets_to_grid((), grid)
		assert result == []

	def test_many_tightly_packed_onsets (self) -> None:

		"""Many onsets closer than grid spacing don't exhaust the grid."""

		# 20 onsets all within 0.5s — much denser than the 0.125s grid spacing
		# at 120 BPM / resolution 16.  The grid must have enough points.
		onsets = tuple(i * 0.025 for i in range(20))

		grid = subsample.transform._build_quantize_grid(
			120.0, 16, 1.0, min_points=len(onsets) + 2,
		)
		result = subsample.transform._snap_onsets_to_grid(onsets, grid)

		assert len(result) == 20

		for i in range(1, len(result)):
			assert result[i] > result[i - 1]


# ---------------------------------------------------------------------------
# TestBuildTimeMap
# ---------------------------------------------------------------------------

class TestBuildTimeMap:

	"""Tests for _build_time_map()."""

	def test_includes_start_and_end_anchors (self) -> None:

		"""Time map always starts at (0,0) and ends at (source_len, target_len)."""

		time_map = subsample.transform._build_time_map(
			[1000, 2000], [1100, 2200], 4410, 4800,
		)
		assert time_map[0]  == (0, 0)
		assert time_map[-1] == (4410, 4800)

	def test_monotonically_increasing (self) -> None:

		"""All entries are strictly increasing in both source and target."""

		time_map = subsample.transform._build_time_map(
			[500, 1500, 2500], [600, 1700, 2800], 4410, 5000,
		)

		for i in range(1, len(time_map)):
			assert time_map[i][0] > time_map[i - 1][0]
			assert time_map[i][1] > time_map[i - 1][1]

	def test_skips_non_monotonic_entries (self) -> None:

		"""Entries that would violate monotonicity are dropped."""

		# Second onset has source=0, which is not > previous source=0.
		time_map = subsample.transform._build_time_map(
			[0, 500], [0, 600], 4410, 5000,
		)

		# (0, 0) start anchor + (500, 600) + (4410, 5000) end anchor.
		assert len(time_map) == 3


# ---------------------------------------------------------------------------
# TestTimeStretchHandler
# ---------------------------------------------------------------------------

class TestTimeStretchHandler:

	"""Tests for the _apply_time_stretch handler."""

	def test_handler_registered (self) -> None:

		"""TimeStretch handler is registered in the dispatch table."""

		assert subsample.transform.TimeStretch in subsample.transform.TransformProcessor._HANDLERS

	def test_no_stretch_when_no_rhythm (self) -> None:

		"""Samples with no detected tempo are returned unchanged."""

		audio = _make_audio(n_frames=4410, channels=2)
		record = _make_record(tempo_bpm=0.0)
		step = subsample.transform.TimeStretch(target_bpm=120.0)

		result = subsample.transform._apply_time_stretch(audio, 44100, record, step)
		assert result.shape == audio.shape
		numpy.testing.assert_array_equal(result, audio)

	def test_global_stretch_for_single_onset (self) -> None:

		"""A sample with only 1 onset gets a simple global stretch."""

		sr = 44100
		duration_sec = 1.0
		n_frames = int(duration_sec * sr)

		# Create audio with a click at the onset.
		audio = numpy.zeros((n_frames, 1), dtype=numpy.float32)
		audio[100:110, :] = 0.8

		record = _make_record(
			audio=numpy.zeros((n_frames, 1), dtype=numpy.int16),
			tempo_bpm=120.0,
			onset_times=(0.01,),
		)

		step = subsample.transform.TimeStretch(target_bpm=60.0)
		result = subsample.transform._apply_time_stretch(audio, sr, record, step)

		# 120 → 60 BPM means double the duration (ratio = 120/60 = 2.0).
		# Allow some tolerance for Rubber Band's processing.
		expected_frames = n_frames * 2
		assert abs(result.shape[0] - expected_frames) < sr * 0.1  # within 100ms

	def test_beat_quantized_stretch_preserves_channels (self) -> None:

		"""Stereo audio stays stereo after beat-quantized stretching."""

		sr = 44100
		n_frames = int(1.0 * sr)
		audio = numpy.zeros((n_frames, 2), dtype=numpy.float32)

		# Place clicks at onset positions.
		for onset_sec in (0.0, 0.25, 0.5, 0.75):
			idx = int(onset_sec * sr)
			audio[idx:idx + 50, :] = 0.8

		record = _make_record(
			audio=numpy.zeros((n_frames, 2), dtype=numpy.int16),
			tempo_bpm=120.0,
			onset_times=(0.0, 0.25, 0.5, 0.75),
		)

		step = subsample.transform.TimeStretch(target_bpm=100.0, resolution=8)
		result = subsample.transform._apply_time_stretch(audio, sr, record, step)

		assert result.ndim == 2
		assert result.shape[1] == 2
		assert result.dtype == numpy.float32

	def test_stretch_changes_duration (self) -> None:

		"""Stretching to a slower tempo produces longer audio."""

		sr = 44100
		n_frames = int(1.0 * sr)
		audio = numpy.zeros((n_frames, 1), dtype=numpy.float32)

		for onset_sec in (0.0, 0.25, 0.5, 0.75):
			idx = int(onset_sec * sr)
			audio[idx:idx + 50, :] = 0.8

		record = _make_record(
			audio=numpy.zeros((n_frames, 1), dtype=numpy.int16),
			tempo_bpm=120.0,
			onset_times=(0.0, 0.25, 0.5, 0.75),
		)

		# Slow down: 120 → 90 BPM → longer output.
		step = subsample.transform.TimeStretch(target_bpm=90.0, resolution=16)
		result = subsample.transform._apply_time_stretch(audio, sr, record, step)
		assert result.shape[0] > n_frames

	def test_stretch_to_faster_tempo_shortens (self) -> None:

		"""Stretching to a faster tempo produces shorter audio."""

		sr = 44100
		n_frames = int(1.0 * sr)
		audio = numpy.zeros((n_frames, 1), dtype=numpy.float32)

		for onset_sec in (0.0, 0.25, 0.5, 0.75):
			idx = int(onset_sec * sr)
			audio[idx:idx + 50, :] = 0.8

		record = _make_record(
			audio=numpy.zeros((n_frames, 1), dtype=numpy.int16),
			tempo_bpm=120.0,
			onset_times=(0.0, 0.25, 0.5, 0.75),
		)

		# Speed up: 120 → 160 BPM → shorter output.
		step = subsample.transform.TimeStretch(target_bpm=160.0, resolution=8)
		result = subsample.transform._apply_time_stretch(audio, sr, record, step)
		assert result.shape[0] < n_frames


# ---------------------------------------------------------------------------
# TestOnSampleAdded — no global time-stretch auto-enqueue
# ---------------------------------------------------------------------------

class TestOnSampleAddedNoAutoStretch:

	"""on_sample_added() only enqueues base variant, not time-stretch."""

	def test_no_time_stretch_enqueued (self) -> None:

		"""Even a rhythmic sample should not get auto-stretched at startup."""

		enqueued_specs: list[subsample.transform.TransformSpec] = []

		cfg = subsample.config.TransformConfig(target_bpm=120.0)

		class _FakeProcessor:
			def enqueue (self, record: typing.Any, spec: subsample.transform.TransformSpec) -> None:
				enqueued_specs.append(spec)

		class _FakeCache:
			def put (self, result: typing.Any) -> None:
				pass

		manager = subsample.transform.TransformManager(
			cache=_FakeCache(),  # type: ignore[arg-type]
			processor=_FakeProcessor(),  # type: ignore[arg-type]
			instrument_library=subsample.library.InstrumentLibrary(max_memory_bytes=100_000_000),
			cfg=cfg,
		)

		record = _make_record(tempo_bpm=120.0, onset_times=(0.0, 0.2, 0.4, 0.6, 0.8))
		manager.on_sample_added(record)

		# Only the base variant should be enqueued — no TimeStretch.
		time_stretch_specs = [
			s for s in enqueued_specs
			if any(isinstance(step, subsample.transform.TimeStretch) for step in s.steps)
		]
		assert len(time_stretch_specs) == 0
		assert len(enqueued_specs) == 1  # just the base variant


# ---------------------------------------------------------------------------
# TestGetAtBpm
# ---------------------------------------------------------------------------

class TestGetAtBpm:

	"""TransformManager.get_at_bpm() returns None when disabled or on miss."""

	def test_returns_none_when_disabled (self) -> None:

		"""target_bpm=0.0 means get_at_bpm always returns None."""

		cfg = subsample.config.TransformConfig(target_bpm=0.0)
		cache = subsample.transform.TransformCache(max_memory_bytes=10_000_000)

		class _FakeProcessor:
			def enqueue (self, record: typing.Any, spec: typing.Any) -> None:
				pass

		manager = subsample.transform.TransformManager(
			cache=cache,
			processor=_FakeProcessor(),  # type: ignore[arg-type]
			instrument_library=subsample.library.InstrumentLibrary(max_memory_bytes=100_000_000),
			cfg=cfg,
		)

		assert manager.get_at_bpm(42) is None

	def test_cache_hit_with_matching_resolution (self) -> None:

		"""get_at_bpm() finds a cached variant when resolution matches config."""

		cfg = subsample.config.TransformConfig(
			target_bpm=120.0, quantize_resolution=8,
		)
		cache = subsample.transform.TransformCache(max_memory_bytes=10_000_000)

		# Insert a result keyed with resolution=8.
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.TimeStretch(target_bpm=120.0, resolution=8),)
		)
		result = _make_result(sample_id=7, spec=spec)
		cache.put(result)

		# The library must contain a qualifying record for this sample_id.
		library = subsample.library.InstrumentLibrary(max_memory_bytes=100_000_000)
		record = _make_record(
			sample_id=7, tempo_bpm=120.0,
			onset_times=(0.0, 0.2, 0.4, 0.6, 0.8),
		)
		library.add(record)

		class _FakeProcessor:
			def enqueue (self, record: typing.Any, spec: typing.Any) -> None:
				pass

		manager = subsample.transform.TransformManager(
			cache=cache,
			processor=_FakeProcessor(),  # type: ignore[arg-type]
			instrument_library=library,
			cfg=cfg,
		)

		hit = manager.get_at_bpm(7)
		assert hit is not None
		assert hit.key.sample_id == 7
