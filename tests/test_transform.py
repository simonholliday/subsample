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
) -> subsample.library.SampleRecord:

	"""Return a minimal SampleRecord suitable for transform tests."""

	if audio is None:
		audio = _make_pcm_audio()

	rhythm = subsample.analysis.RhythmResult(
		tempo_bpm        = tempo_bpm,
		beat_times       = (),
		pulse_curve      = numpy.zeros(0, dtype=numpy.float32),
		pulse_peak_times = (),
		onset_times      = (),
		onset_count      = 0,
	)

	return subsample.library.SampleRecord(
		sample_id = sample_id,
		name      = f"test_{sample_id}",
		spectral  = tests.helpers._make_spectral(),
		rhythm    = rhythm,
		pitch     = tests.helpers._make_pitch(),
		timbre    = tests.helpers._make_timbre(),
		level     = tests.helpers._make_level(),
		params    = tests.helpers._make_params(),
		duration  = float(audio.shape[0]) / 44100.0,
		audio     = audio,
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
		sample_id = sample_id,
		name      = f"unpitched_{sample_id}",
		spectral  = spectral,
		rhythm    = subsample.analysis.RhythmResult(
			tempo_bpm=0.0, beat_times=(), pulse_curve=numpy.zeros(0, dtype=numpy.float32),
			pulse_peak_times=(), onset_times=(), onset_count=0,
		),
		pitch     = pitch,
		timbre    = tests.helpers._make_timbre(),
		level     = tests.helpers._make_level(),
		params    = tests.helpers._make_params(),
		duration  = float(audio.shape[0]) / 44100.0,
		audio     = audio,
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

	def test_on_sample_added_enqueues_stable_pitch_variants (self) -> None:
		"""on_sample_added auto-enqueues pitch variants for tonal samples."""
		manager, cache, lib = self._make_manager()
		record = _make_record(sample_id=1)
		lib.add(record)

		manager.on_sample_added(record)
		manager.shutdown()

		# Default pitch helper: 440 Hz = MIDI 69, range 12 → notes 57–81 = 25 pitch
		# variants, plus 1 base variant (identity spec) = 26 total.
		assert cache.has_variants(1)
		assert len(cache.list_variants(1)) == 26
		# Base variant is present alongside the pitch variants.
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

	def test_on_sample_added_respects_auto_pitch_false (self) -> None:
		"""auto_pitch=False suppresses pitch variants but the base variant is always produced."""
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
		manager.shutdown()

		# Base variant only — no pitch variants with auto_pitch=False.
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
		assert cfg.transform.max_memory_mb         == 50.0
		assert cfg.transform.auto_pitch            is True
		assert cfg.transform.pitch_range_semitones == 12
		assert cfg.transform.target_bpm            == 0.0

	def test_invalid_pitch_range_raises (self) -> None:
		with pytest.raises(ValueError, match="pitch_range_semitones"):
			subsample.config._build_config({
				"recorder": {
					"audio": {"sample_rate": 44100, "bit_depth": 16, "channels": 1, "chunk_size": 512},
					"buffer": {"max_seconds": 60},
				},
				"detection": {
					"snr_threshold_db": 12.0, "hold_time": 0.5,
					"warmup_seconds": 1.0, "ema_alpha": 0.1,
				},
				"output": {"directory": "./samples", "filename_format": "%Y"},
				"transform": {"pitch_range_semitones": -1},
			})
