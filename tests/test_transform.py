"""Tests for subsample/transform.py — the sample transform pipeline scaffold."""

import dataclasses
import pathlib
import tempfile
import threading
import time
import typing

import numpy
import pytest
import soundfile

import subsample.analysis
import subsample.config
import subsample.library
import subsample.query
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
		spectral_rolloff   = 0.5,
		spectral_slope     = 0.3,
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

	"""TransformSpec preserves declaration order — different orders are different specs."""

	def test_empty_spec_is_identity (self) -> None:
		spec = subsample.transform.TransformSpec(steps=())
		assert spec.steps == ()

	def test_single_step_unchanged (self) -> None:
		step = subsample.transform.PitchShift(target_midi_note=60)
		spec = subsample.transform.TransformSpec(steps=(step,))
		assert spec.steps == (step,)

	def test_order_preserved (self) -> None:
		"""Different declaration orders produce different specs."""
		pitch   = subsample.transform.PitchShift(target_midi_note=60)
		stretch = subsample.transform.TimeStretch(target_bpm=120.0)

		spec_a = subsample.transform.TransformSpec(steps=(stretch, pitch))
		spec_b = subsample.transform.TransformSpec(steps=(pitch, stretch))

		assert spec_a != spec_b
		assert spec_a.steps == (stretch, pitch)
		assert spec_b.steps == (pitch, stretch)

	def test_declaration_order_is_cache_key (self) -> None:
		"""Steps stay in declaration order — this is the cache key identity."""
		pitch   = subsample.transform.PitchShift(target_midi_note=69)
		filt    = subsample.transform.LowPassFilter(freq=1000.0)
		stretch = subsample.transform.TimeStretch(target_bpm=90.0)

		spec = subsample.transform.TransformSpec(steps=(stretch, filt, pitch))

		assert spec.steps == (stretch, filt, pitch)

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
		# max_memory_mb is derived from auto-detect (35% of global).
		assert cfg.transform.max_memory_mb       > 0
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

	def test_crop_fade_in_applied (self) -> None:

		"""After cropping to first attack, a short S-curve fade-in is applied."""

		sr = 44100
		n_frames = int(1.0 * sr)

		# Constant non-zero audio so any fade is visible as a ramp from 0.
		audio = numpy.full((n_frames, 1), 0.8, dtype=numpy.float32)

		# Place attacks so the crop has a non-trivial start.
		onset_times = (0.05, 0.3, 0.55, 0.8)
		record = _make_record(
			audio=numpy.zeros((n_frames, 1), dtype=numpy.int16),
			tempo_bpm=120.0,
			onset_times=onset_times,
		)

		step = subsample.transform.TimeStretch(target_bpm=120.0, resolution=4)
		result = subsample.transform._apply_time_stretch(audio, sr, record, step)

		# The first sample should be near zero (faded in from silence).
		assert result[0, 0] < 0.01, f"First sample not faded: {result[0, 0]}"

		# After the fade-in window (1ms = 44 samples), the audio should be
		# at full level.
		fade_len = int(subsample.transform._CROP_FADE_IN_SECONDS * sr)
		assert result[fade_len, 0] > 0.5, f"Audio not at full level after fade: {result[fade_len, 0]}"

	def test_beat_quantize_preserves_bformat_inter_channel_ratios (self) -> None:

		"""Beat-quantize on a 4-channel B-format sample preserves inter-channel energy ratios.

		Rubber Band's R3 engine (selected via --fine) processes multichannel
		input phase-coherently, so the directional encoding of an ambisonic
		signal should survive a time-stretch within a small tolerance.  We
		construct a B-format sample where each channel carries the same
		impulse train at fixed relative amplitudes (W: 1.0, Y: 0.6, Z: 0.2,
		X: 0.8), stretch it, and verify the post-stretch energy ratios
		between channels are within 0.5 dB of the pre-stretch ratios.
		"""

		sr = 44100
		n_frames = int(1.0 * sr)
		audio = numpy.zeros((n_frames, 4), dtype=numpy.float32)

		amplitudes = numpy.array([1.0, 0.6, 0.2, 0.8], dtype=numpy.float32)

		for onset_sec in (0.0, 0.25, 0.5, 0.75):
			idx = int(onset_sec * sr)
			audio[idx:idx + 50, :] = amplitudes

		record = _make_record(
			audio=numpy.zeros((n_frames, 4), dtype=numpy.int16),
			tempo_bpm=120.0,
			onset_times=(0.0, 0.25, 0.5, 0.75),
		)

		step   = subsample.transform.TimeStretch(target_bpm=95.0, resolution=16)
		result = subsample.transform._apply_time_stretch(audio, sr, record, step)

		assert result.shape[1] == 4

		original_energy = numpy.sum(audio   ** 2, axis=0)
		stretched_energy = numpy.sum(result ** 2, axis=0)

		# Per-channel energy ratio relative to the W channel should be
		# preserved within 0.5 dB.  W itself is the reference, so its ratio
		# is 1.0 in both pre and post.
		for ch in range(1, 4):
			orig_ratio = stretched_energy[ch] / stretched_energy[0]
			src_ratio  = original_energy[ch]  / original_energy[0]
			delta_db   = 10.0 * numpy.log10(orig_ratio / src_ratio)
			assert abs(delta_db) < 0.5, (
				f"Channel {ch}: inter-channel energy ratio drifted by {delta_db:.2f} dB "
				f"(orig ratio={src_ratio:.3f}, stretched={orig_ratio:.3f})"
			)


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


# ---------------------------------------------------------------------------
# TestReverse
# ---------------------------------------------------------------------------

class TestReverse:

	"""Tests for the _apply_reverse handler."""

	def test_mono_reversed (self) -> None:
		audio = numpy.array([[1.0], [2.0], [3.0], [4.0]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		result = subsample.transform._apply_reverse(audio, 44100, record, subsample.transform.Reverse())

		expected = numpy.array([[4.0], [3.0], [2.0], [1.0]], dtype=numpy.float32)
		numpy.testing.assert_array_equal(result, expected)

	def test_stereo_reversed (self) -> None:
		audio = numpy.array(
			[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
			dtype=numpy.float32,
		)
		record = _make_record(sample_id=1)
		result = subsample.transform._apply_reverse(audio, 44100, record, subsample.transform.Reverse())

		expected = numpy.array(
			[[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]],
			dtype=numpy.float32,
		)
		numpy.testing.assert_array_equal(result, expected)

	def test_output_is_contiguous (self) -> None:
		audio = numpy.arange(20, dtype=numpy.float32).reshape(10, 2)
		record = _make_record(sample_id=1)
		result = subsample.transform._apply_reverse(audio, 44100, record, subsample.transform.Reverse())

		assert result.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# TestLowPassFilter
# ---------------------------------------------------------------------------

class TestLowPassFilter:

	"""Tests for the _apply_low_pass handler."""

	def _make_tone (self, freq: float, sr: int = 44100, duration: float = 0.5) -> numpy.ndarray:
		"""Generate a mono sine wave."""
		t = numpy.arange(int(sr * duration), dtype=numpy.float32) / sr
		return numpy.sin(2 * numpy.pi * freq * t).reshape(-1, 1).astype(numpy.float32)

	def test_attenuates_high_frequencies (self) -> None:
		"""A 5 kHz tone should be significantly attenuated by a 500 Hz low-pass."""
		sr = 44100
		audio = self._make_tone(5000.0, sr)

		record = _make_record(sample_id=1)
		step = subsample.transform.LowPassFilter(freq=500.0, resonance_db=0.0)
		result = subsample.transform._apply_low_pass(audio, sr, record, step)

		# Discard first 10% to skip filter transient.
		discard = len(result) // 10
		rms_before = float(numpy.sqrt(numpy.mean(audio[discard:] ** 2)))
		rms_after  = float(numpy.sqrt(numpy.mean(result[discard:] ** 2)))
		attenuation_db = 20.0 * numpy.log10(max(rms_after, 1e-12) / max(rms_before, 1e-12))
		assert attenuation_db < -20.0

	def test_resonance_changes_response (self) -> None:
		"""Resonance > 0 uses Chebyshev and produces different output than Butterworth."""
		sr = 44100
		audio = self._make_tone(300.0, sr)
		record = _make_record(sample_id=1)

		flat = subsample.transform._apply_low_pass(
			audio, sr, record, subsample.transform.LowPassFilter(freq=500.0, resonance_db=0.0),
		)
		resonant = subsample.transform._apply_low_pass(
			audio, sr, record, subsample.transform.LowPassFilter(freq=500.0, resonance_db=12.0),
		)

		# The resonant version should differ from the flat version.
		assert not numpy.allclose(flat, resonant, atol=1e-6)

	def test_freq_at_nyquist_no_crash (self) -> None:
		"""Cutoff at or above Nyquist should not crash."""
		sr = 44100
		audio = self._make_tone(440.0, sr)
		record = _make_record(sample_id=1)
		step = subsample.transform.LowPassFilter(freq=float(sr), resonance_db=0.0)

		result = subsample.transform._apply_low_pass(audio, sr, record, step)
		assert result.shape == audio.shape

	def test_output_is_float32 (self) -> None:
		audio = self._make_tone(440.0).astype(numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.LowPassFilter(freq=1000.0)
		result = subsample.transform._apply_low_pass(audio, 44100, record, step)
		assert result.dtype == numpy.float32


# ---------------------------------------------------------------------------
# TestHighPassFilter
# ---------------------------------------------------------------------------

class TestHighPassFilter:

	"""Tests for the _apply_high_pass handler."""

	def test_attenuates_low_frequencies (self) -> None:
		sr = 44100
		t = numpy.arange(int(sr * 0.5), dtype=numpy.float32) / sr
		audio = numpy.sin(2 * numpy.pi * 100.0 * t).reshape(-1, 1).astype(numpy.float32)

		record = _make_record(sample_id=1)
		step = subsample.transform.HighPassFilter(freq=2000.0, resonance_db=0.0)
		result = subsample.transform._apply_high_pass(audio, sr, record, step)

		# Discard first 10% to skip filter transient.
		discard = len(result) // 10
		rms_before = float(numpy.sqrt(numpy.mean(audio[discard:] ** 2)))
		rms_after  = float(numpy.sqrt(numpy.mean(result[discard:] ** 2)))
		attenuation_db = 20.0 * numpy.log10(max(rms_after, 1e-12) / max(rms_before, 1e-12))
		assert attenuation_db < -20.0


# ---------------------------------------------------------------------------
# TestBandPassFilter
# ---------------------------------------------------------------------------

class TestBandPassFilter:

	"""Tests for the _apply_band_pass handler."""

	def test_passes_center_attenuates_edges (self) -> None:
		sr = 44100
		t = numpy.arange(int(sr * 0.5), dtype=numpy.float32) / sr
		low  = numpy.sin(2 * numpy.pi * 100.0 * t).reshape(-1, 1).astype(numpy.float32)
		mid  = numpy.sin(2 * numpy.pi * 1000.0 * t).reshape(-1, 1).astype(numpy.float32)
		high = numpy.sin(2 * numpy.pi * 8000.0 * t).reshape(-1, 1).astype(numpy.float32)
		audio = (low + mid + high).astype(numpy.float32)

		record = _make_record(sample_id=1)
		step = subsample.transform.BandPassFilter(freq=1000.0, resonance_db=0.0)
		result = subsample.transform._apply_band_pass(audio, sr, record, step)

		# The center component should dominate; edges attenuated.
		rms_result = float(numpy.sqrt(numpy.mean(result ** 2)))
		rms_mid    = float(numpy.sqrt(numpy.mean(mid ** 2)))

		# Result should be similar to the mid component (within 6 dB).
		ratio_db = 20.0 * numpy.log10(max(rms_result, 1e-12) / max(rms_mid, 1e-12))
		assert ratio_db > -6.0


# ---------------------------------------------------------------------------
# TestSaturate
# ---------------------------------------------------------------------------

class TestSaturate:

	"""Tests for the _apply_saturate handler."""

	def test_zero_drive_unchanged (self) -> None:
		audio = numpy.array([[0.5], [-0.3], [0.8]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Saturate(amount_db=0.0)
		result = subsample.transform._apply_saturate(audio, 44100, record, step)
		numpy.testing.assert_array_equal(result, audio)

	def test_negative_drive_unchanged (self) -> None:
		audio = numpy.array([[0.5], [-0.3]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Saturate(amount_db=-6.0)
		result = subsample.transform._apply_saturate(audio, 44100, record, step)
		numpy.testing.assert_array_equal(result, audio)

	def test_level_compensation (self) -> None:
		"""Full-scale input should remain near full-scale after saturation."""
		audio = numpy.array([[1.0]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Saturate(amount_db=12.0)
		result = subsample.transform._apply_saturate(audio, 44100, record, step)

		assert abs(float(result[0, 0]) - 1.0) < 0.01

	def test_no_values_exceed_one (self) -> None:
		"""Heavy saturation should soft-clip — no output sample exceeds 1.0."""
		numpy.random.seed(42)
		audio = numpy.random.uniform(-0.9, 0.9, (1000, 2)).astype(numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Saturate(amount_db=20.0)
		result = subsample.transform._apply_saturate(audio, 44100, record, step)

		assert numpy.all(numpy.abs(result) <= 1.0)

	def test_silence_stays_silent (self) -> None:
		audio = numpy.zeros((100, 1), dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Saturate(amount_db=12.0)
		result = subsample.transform._apply_saturate(audio, 44100, record, step)
		numpy.testing.assert_array_equal(result, audio)

	def test_output_is_float32 (self) -> None:
		audio = numpy.array([[0.5], [0.3]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Saturate(amount_db=6.0)
		result = subsample.transform._apply_saturate(audio, 44100, record, step)
		assert result.dtype == numpy.float32


# ---------------------------------------------------------------------------
# TestCompress
# ---------------------------------------------------------------------------

class TestCompress:

	"""Tests for the _apply_compress handler and _compress() core."""

	def _make_impulse (self, n_frames: int = 4410, peak: float = 0.9, sr: int = 44100) -> numpy.ndarray:
		"""Create a mono impulse (sharp transient + exponential decay)."""
		audio = numpy.zeros((n_frames, 1), dtype=numpy.float32)
		audio[0, 0] = peak
		decay = numpy.exp(-numpy.arange(n_frames, dtype=numpy.float32) * 10.0 / n_frames)
		audio[:, 0] = peak * decay
		return audio

	def test_below_threshold_unchanged (self) -> None:
		"""Signal entirely below threshold gets no gain reduction."""
		audio = numpy.full((100, 1), 0.01, dtype=numpy.float32)  # ~ -40 dBFS
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress(threshold_db=-10.0, ratio=4.0, knee_db=0.0)
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		numpy.testing.assert_allclose(result, audio, atol=1e-5)

	def test_above_threshold_reduced (self) -> None:
		"""Signal above threshold is attenuated."""
		audio = numpy.full((1000, 1), 0.5, dtype=numpy.float32)  # ~ -6 dBFS
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress(
			threshold_db=-20.0, ratio=4.0, attack_ms=0.01, release_ms=5.0, knee_db=0.0,
		)
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		# After ballistics settle, output should be lower than input.
		assert float(numpy.max(numpy.abs(result[-100:]))) < float(numpy.max(numpy.abs(audio)))

	def test_soft_knee_compresses_at_threshold (self) -> None:
		"""Soft knee applies gain reduction at threshold; hard knee does not."""
		# Signal at exactly -20 dBFS (threshold).  Hard knee: no reduction
		# (signal is not above threshold).  Soft knee with W=12: threshold is
		# at the centre of the knee (-26 to -14), so partial reduction applies.
		level = 10.0 ** (-20.0 / 20.0)  # 0.1
		audio = numpy.full((2000, 1), level, dtype=numpy.float32)
		hard = subsample.transform._compress(audio, 44100, -20.0, 4.0, 0.01, 5.0, 0.0, 0.0, 0.0)
		soft = subsample.transform._compress(audio, 44100, -20.0, 4.0, 0.01, 5.0, 12.0, 0.0, 0.0)
		# Hard knee: signal at threshold → no compression → output ≈ input.
		hard_level = float(numpy.mean(numpy.abs(hard[-200:])))
		numpy.testing.assert_allclose(hard_level, level, atol=1e-3)
		# Soft knee: signal within knee → some reduction → output < input.
		soft_level = float(numpy.mean(numpy.abs(soft[-200:])))
		assert soft_level < level * 0.99

	def test_makeup_gain (self) -> None:
		"""Makeup gain boosts the output level."""
		audio = numpy.full((200, 1), 0.1, dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress(
			threshold_db=-40.0, ratio=1.0, makeup_db=6.0,
		)
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		# ratio=1 means no compression, so only makeup applies.
		expected_gain = 10.0 ** (6.0 / 20.0)
		numpy.testing.assert_allclose(result, audio * expected_gain, atol=1e-5)

	def test_ratio_one_passthrough (self) -> None:
		"""Ratio 1:1 is a passthrough (no compression)."""
		audio = numpy.array([[0.5], [-0.3], [0.8]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress(ratio=1.0)
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		numpy.testing.assert_array_equal(result, audio)

	def test_stereo_linked (self) -> None:
		"""Multi-channel: same gain applied to both channels (preserves stereo image)."""
		audio = numpy.zeros((1000, 2), dtype=numpy.float32)
		audio[:, 0] = 0.8   # left loud
		audio[:, 1] = 0.1   # right quiet
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress(
			threshold_db=-10.0, ratio=4.0, attack_ms=0.01, release_ms=5.0, knee_db=0.0,
		)
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		# Both channels should be reduced proportionally.
		left_ratio = float(result[-1, 0]) / float(audio[-1, 0])
		right_ratio = float(result[-1, 1]) / float(audio[-1, 1])
		numpy.testing.assert_allclose(left_ratio, right_ratio, atol=1e-3)

	def test_output_is_float32 (self) -> None:
		audio = numpy.array([[0.5], [0.3]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress()
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		assert result.dtype == numpy.float32

	def test_empty_audio (self) -> None:
		audio = numpy.zeros((0, 1), dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress()
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		assert result.shape == (0, 1)

	def test_silence_stays_silent (self) -> None:
		audio = numpy.zeros((100, 1), dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Compress(threshold_db=-20.0, ratio=4.0)
		result = subsample.transform._apply_compress(audio, 44100, record, step)
		numpy.testing.assert_allclose(result, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestCompressAdaptive
# ---------------------------------------------------------------------------

class TestCompressAdaptive:

	"""Tests for per-sample adaptive compression defaults."""

	def _record_with (
		self,
		peak: float = 0.85,
		rms: float = 0.25,
		attack: float = 0.2,
		release: float = 0.3,
	) -> subsample.library.SampleRecord:
		"""Create a record with custom level and spectral fields."""
		record = _make_record(sample_id=1)
		spectral = dataclasses.replace(record.spectral, attack=attack, release=release)
		level = subsample.analysis.LevelResult(peak=peak, rms=rms)
		return dataclasses.replace(record, spectral=spectral, level=level)

	def test_auto_threshold_adapts_to_peak (self) -> None:
		"""Auto threshold is relative to the sample's peak — different peaks → different thresholds."""
		loud = self._record_with(peak=0.9)   # ~ -0.9 dBFS
		quiet = self._record_with(peak=0.1)  # ~ -20 dBFS

		audio = numpy.full((2000, 1), 0.5, dtype=numpy.float32)
		step = subsample.transform.Compress()  # all auto

		t_loud, _, _ = subsample.transform._resolve_compress_params(loud, step)
		t_quiet, _, _ = subsample.transform._resolve_compress_params(quiet, step)

		# Loud sample should have a higher (less negative) threshold.
		assert t_loud > t_quiet
		# Both should be 6 dB below their respective peaks.
		import math
		expected_loud = 20.0 * math.log10(0.9 + 1e-10) - 6.0
		expected_quiet = 20.0 * math.log10(0.1 + 1e-10) - 6.0
		assert abs(t_loud - expected_loud) < 0.01
		assert abs(t_quiet - expected_quiet) < 0.01

	def test_auto_attack_percussive_vs_gradual (self) -> None:
		"""Percussive sample (low spectral.attack) → slow compressor attack."""
		percussive = self._record_with(attack=0.0)  # instant onset
		gradual = self._record_with(attack=1.0)      # very slow onset

		step = subsample.transform.Compress()

		_, a_perc, _ = subsample.transform._resolve_compress_params(percussive, step)
		_, a_grad, _ = subsample.transform._resolve_compress_params(gradual, step)

		# Percussive → slow attack (lets transient through).
		assert a_perc > a_grad
		assert abs(a_perc - 30.0) < 0.01  # 1 + 29*(1-0) = 30
		assert abs(a_grad - 1.0) < 0.01   # 1 + 29*(1-1) = 1

	def test_auto_release_short_vs_long (self) -> None:
		"""Short-decay sample → fast compressor release."""
		short = self._record_with(release=0.0)   # instant decay
		long = self._record_with(release=1.0)     # very long tail

		step = subsample.transform.Compress()

		_, _, r_short = subsample.transform._resolve_compress_params(short, step)
		_, _, r_long = subsample.transform._resolve_compress_params(long, step)

		assert r_short < r_long
		assert abs(r_short - 30.0) < 0.01   # 30 + 270*0 = 30
		assert abs(r_long - 300.0) < 0.01   # 30 + 270*1 = 300

	def test_explicit_overrides_auto (self) -> None:
		"""Explicit values override auto; unset values still auto-compute."""
		record = self._record_with(peak=0.5, attack=0.8, release=0.5)
		step = subsample.transform.Compress(threshold_db=-18.0)  # explicit threshold only

		t, a, r = subsample.transform._resolve_compress_params(record, step)

		# Threshold: explicit → -18.0.
		assert t == -18.0
		# Attack: auto → 1 + 29*(1-0.8) = 6.8.
		assert abs(a - 6.8) < 0.01
		# Release: auto → 30 + 270*0.5 = 165.
		assert abs(r - 165.0) < 0.01

	def test_all_explicit_no_auto (self) -> None:
		"""When all three adaptive params are set, no auto-computation occurs."""
		record = self._record_with(peak=0.1, attack=0.0, release=0.0)
		step = subsample.transform.Compress(threshold_db=-12.0, attack_ms=5.0, release_ms=50.0)

		t, a, r = subsample.transform._resolve_compress_params(record, step)

		assert t == -12.0
		assert a == 5.0
		assert r == 50.0

	def test_auto_compress_actually_compresses (self) -> None:
		"""compress: true with auto defaults actually reduces the signal."""
		audio = numpy.full((2000, 1), 0.8, dtype=numpy.float32)
		record = self._record_with(peak=0.85)
		step = subsample.transform.Compress()  # all auto

		result = subsample.transform._apply_compress(audio, 44100, record, step)
		# Auto threshold ≈ -1.4 - 6 ≈ -7.4 dBFS.  Signal at 0.8 is ≈ -1.9 dBFS.
		# Signal is above threshold → should be compressed.
		assert float(numpy.max(numpy.abs(result[-200:]))) < float(numpy.max(numpy.abs(audio)))


# ---------------------------------------------------------------------------
# TestLimit
# ---------------------------------------------------------------------------

class TestLimit:

	"""Tests for the _apply_limit handler (brickwall limiter)."""

	def test_below_threshold_unchanged (self) -> None:
		"""Signal below the limiter threshold passes through."""
		# Use enough frames to exceed the default 5 ms look-ahead (221 samples at 44.1 kHz).
		audio = numpy.full((2000, 1), 0.01, dtype=numpy.float32)  # ~ -40 dBFS
		record = _make_record(sample_id=1)
		step = subsample.transform.Limit(threshold_db=-1.0)
		result = subsample.transform._apply_limit(audio, 44100, record, step)
		# Check the tail (past the look-ahead delay region).
		numpy.testing.assert_allclose(result[300:], audio[300:], atol=1e-4)

	def test_above_threshold_limited (self) -> None:
		"""Signal above threshold is brought down near the ceiling."""
		audio = numpy.full((2000, 1), 0.9, dtype=numpy.float32)  # ~ -0.9 dBFS
		record = _make_record(sample_id=1)
		step = subsample.transform.Limit(threshold_db=-6.0, lookahead_ms=0.0)
		result = subsample.transform._apply_limit(audio, 44100, record, step)
		# After settling, the output should be substantially reduced.
		output_peak = float(numpy.max(numpy.abs(result[-100:])))
		assert output_peak < 0.9

	def test_output_is_float32 (self) -> None:
		audio = numpy.array([[0.5], [0.3]], dtype=numpy.float32)
		record = _make_record(sample_id=1)
		step = subsample.transform.Limit()
		result = subsample.transform._apply_limit(audio, 44100, record, step)
		assert result.dtype == numpy.float32


# ---------------------------------------------------------------------------
# TestSpecFromProcess
# ---------------------------------------------------------------------------

class TestSpecFromProcess:

	"""Tests for the spec_from_process() chain builder."""

	def test_empty_process (self) -> None:
		process = subsample.query.ProcessSpec()
		spec = subsample.transform.spec_from_process(process)
		assert spec.steps == ()

	def test_repitch_with_note (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch"),
		))
		spec = subsample.transform.spec_from_process(process, midi_note=60)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.PitchShift)
		assert spec.steps[0].target_midi_note == 60

	def test_repitch_without_note_skipped (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert spec.steps == ()

	def test_repitch_fixed_note_name (self) -> None:
		"""repitch: { note: C4 } should use the fixed note, ignoring midi_note."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch", params=(("note", "C4"),)),
		))
		spec = subsample.transform.spec_from_process(process, midi_note=36)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.PitchShift)
		assert spec.steps[0].target_midi_note == 60  # C4 = MIDI 60

	def test_repitch_fixed_note_int (self) -> None:
		"""repitch: { note: 72 } should accept an integer directly."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch", params=(("note", 72),)),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		assert spec.steps[0].target_midi_note == 72

	def test_beat_quantize_with_params (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize",
				params=(("tempo", 120), ("grid", 8)),
			),
		))
		spec = subsample.transform.spec_from_process(process, target_bpm=100.0)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.TimeStretch)
		assert spec.steps[0].target_bpm == 120.0  # step params override function arg
		assert spec.steps[0].resolution == 8

	def test_beat_quantize_true_uses_all_defaults (self) -> None:
		"""`stretch_quantize: true` → all defaults, fed by target_bpm."""
		process = subsample.query.parse_process([{"stretch_quantize": True}], "test")
		spec = subsample.transform.spec_from_process(process, target_bpm=120.0)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.TimeStretch)
		assert spec.steps[0].target_bpm == 120.0
		assert spec.steps[0].resolution == 16
		assert spec.steps[0].amount == 1.0

	def test_beat_quantize_true_no_bpm_warns_and_skips (self, caplog: typing.Any) -> None:
		"""`stretch_quantize: true` with no target_bpm logs a warning and skips."""
		# Reset the warn-once set so this test sees the warning fresh.
		subsample.transform._WARN_ONCE_SEEN.discard("stretch_quantize-no-tempo")
		process = subsample.query.parse_process([{"stretch_quantize": True}], "test")
		with caplog.at_level("WARNING"):
			spec = subsample.transform.spec_from_process(process, target_bpm=None)
		assert spec.steps == ()
		assert any("stretch_quantize" in r.message and "tempo" in r.message for r in caplog.records)

	def test_pad_quantize_true_uses_all_defaults (self) -> None:
		"""`pad_quantize: true` → all defaults, fed by target_bpm."""
		process = subsample.query.parse_process([{"pad_quantize": True}], "test")
		spec = subsample.transform.spec_from_process(process, target_bpm=120.0)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.PadQuantize)
		assert spec.steps[0].target_bpm == 120.0
		assert spec.steps[0].resolution == 16
		assert spec.steps[0].amount == 1.0

	def test_pad_quantize_true_no_bpm_warns_and_skips (self, caplog: typing.Any) -> None:
		"""`pad_quantize: true` with no target_bpm logs a warning and skips."""
		subsample.transform._WARN_ONCE_SEEN.discard("pad_quantize-no-tempo")
		process = subsample.query.parse_process([{"pad_quantize": True}], "test")
		with caplog.at_level("WARNING"):
			spec = subsample.transform.spec_from_process(process, target_bpm=None)
		assert spec.steps == ()
		assert any("pad_quantize" in r.message and "tempo" in r.message for r in caplog.records)

	def test_saturate_true_uses_default_drive (self) -> None:
		"""`saturate: true` → 6 dB drive default."""
		process = subsample.query.parse_process([{"saturate": True}], "test")
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.Saturate)
		assert spec.steps[0].amount_db == 6.0

	def test_filter_low (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="filter_low",
				params=(("freq", 800), ("resonance", 6)),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.LowPassFilter)
		assert spec.steps[0].freq == 800.0
		assert spec.steps[0].resonance_db == 6.0

	def test_filter_high_defaults (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="filter_high", params=(("freq", 4000),)),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.HighPassFilter)
		assert spec.steps[0].resonance_db == 0.0

	def test_filter_band (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="filter_band",
				params=(("freq", 1000), ("resonance", 3)),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.BandPassFilter)
		assert spec.steps[0].freq == 1000.0

	def test_reverse (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="reverse"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.Reverse)

	def test_saturate (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="saturate",
				params=(("drive", 8),),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.Saturate)
		assert spec.steps[0].amount_db == 8.0

	def test_compress (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="compress",
				params=(("threshold", -30), ("ratio", 8), ("attack", 1.0), ("release", 50)),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.Compress)
		assert spec.steps[0].threshold_db == -30.0
		assert spec.steps[0].ratio == 8.0
		assert spec.steps[0].attack_ms == 1.0
		assert spec.steps[0].release_ms == 50.0
		# Defaults for unset params:
		assert spec.steps[0].knee_db == 6.0
		assert spec.steps[0].makeup_db == 0.0
		assert spec.steps[0].lookahead_ms == 0.0

	def test_compress_defaults (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="compress"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.Compress)
		# Adaptive fields default to None (auto-compute from sample).
		assert spec.steps[0].threshold_db is None
		assert spec.steps[0].attack_ms is None
		assert spec.steps[0].release_ms is None
		# Fixed fields keep their defaults.
		assert spec.steps[0].ratio == 4.0
		assert spec.steps[0].knee_db == 6.0

	def test_limit (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="limit",
				params=(("threshold", -3), ("release", 30), ("lookahead", 10)),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.Limit)
		assert spec.steps[0].threshold_db == -3.0
		assert spec.steps[0].release_ms == 30.0
		assert spec.steps[0].lookahead_ms == 10.0

	def test_limit_defaults (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="limit"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert isinstance(spec.steps[0], subsample.transform.Limit)
		assert spec.steps[0].threshold_db == -1.0
		assert spec.steps[0].lookahead_ms == 5.0

	def test_declaration_order_preserved (self) -> None:
		"""Steps appear in the spec in the order declared in the process list."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="filter_low", params=(("freq", 800),)),
			subsample.query.ProcessorStep(name="repitch"),
			subsample.query.ProcessorStep(name="saturate", params=(("drive", 6),)),
		))
		spec = subsample.transform.spec_from_process(process, midi_note=60)

		assert len(spec.steps) == 3
		assert isinstance(spec.steps[0], subsample.transform.LowPassFilter)
		assert isinstance(spec.steps[1], subsample.transform.PitchShift)
		assert isinstance(spec.steps[2], subsample.transform.Saturate)

	def test_unknown_processor_skipped (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="unknown_thing"),
			subsample.query.ProcessorStep(name="reverse"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.Reverse)

	def test_composite_chain (self) -> None:
		"""Full chain with multiple processors in user-specified order."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="reverse"),
			subsample.query.ProcessorStep(name="filter_low", params=(("freq", 500), ("resonance", 6))),
			subsample.query.ProcessorStep(name="saturate", params=(("drive", 4),)),
			subsample.query.ProcessorStep(name="repitch"),
			subsample.query.ProcessorStep(name="stretch_quantize", params=(("tempo", 120), ("grid", 16))),
		))
		spec = subsample.transform.spec_from_process(process, midi_note=72, target_bpm=100.0)

		assert len(spec.steps) == 5
		assert isinstance(spec.steps[0], subsample.transform.Reverse)
		assert isinstance(spec.steps[1], subsample.transform.LowPassFilter)
		assert isinstance(spec.steps[2], subsample.transform.Saturate)
		assert isinstance(spec.steps[3], subsample.transform.PitchShift)
		assert spec.steps[3].target_midi_note == 72
		assert isinstance(spec.steps[4], subsample.transform.TimeStretch)
		assert spec.steps[4].target_bpm == 120.0


# ---------------------------------------------------------------------------
# TestCcResolution
# ---------------------------------------------------------------------------

class TestCcResolution:

	"""Tests for CC binding resolution in spec_from_process."""

	def test_cc_binding_resolves_from_state (self) -> None:
		"""CcBinding amount is resolved from cc_omni (omni mode)."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", subsample.query.CcBinding(cc=1))),
			),
		))
		cc_omni = {1: 127}  # CC#1 = 127 (last-write-wins, any channel)
		spec = subsample.transform.spec_from_process(process, cc_omni=cc_omni)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.PadQuantize)
		assert step.amount == 1.0  # CC 127 → max of [0, 1]

	def test_cc_binding_uses_default_when_no_state (self) -> None:
		"""CcBinding falls back to default_value when cc_state is None."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", subsample.query.CcBinding(cc=1, default=0.75))),
			),
		))
		spec = subsample.transform.spec_from_process(process, cc_state=None)
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.PadQuantize)
		assert step.amount == 0.75

	def test_cc_binding_channel_specific (self) -> None:
		"""Channel-specific CcBinding only matches that channel."""
		binding = subsample.query.CcBinding(cc=1, min_val=0.0, max_val=1.0, channel=10)
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", binding)),
			),
		))

		# CC on channel 10 (mido 9): should resolve.
		cc_state_match = {(9, 1): 64}
		spec = subsample.transform.spec_from_process(process, cc_state=cc_state_match)
		assert abs(spec.steps[0].amount - 64.0 / 127.0) < 1e-6

		# CC on channel 1 (mido 0): should NOT match, fall back to default.
		cc_state_wrong = {(0, 1): 127}
		spec2 = subsample.transform.spec_from_process(process, cc_state=cc_state_wrong)
		assert spec2.steps[0].amount == 0.5  # default = midpoint

	def test_cc_binding_omni (self) -> None:
		"""Omni CcBinding (channel=None) uses cc_omni (last-write-wins)."""
		binding = subsample.query.CcBinding(cc=1)
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", binding)),
			),
		))
		cc_omni = {1: 0}  # CC#1 = 0 (last-write-wins)
		spec = subsample.transform.spec_from_process(process, cc_omni=cc_omni)
		assert spec.steps[0].amount == 0.0  # CC 0 → min


# ---------------------------------------------------------------------------
# TestVariantCacheKey
# ---------------------------------------------------------------------------

class TestVariantCacheKey:

	"""Tests for variant_cache_key() determinism and sensitivity."""

	def test_deterministic (self) -> None:
		"""Same inputs produce same hash."""
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		a = subsample.transform.variant_cache_key("abc123", spec, 44100)
		b = subsample.transform.variant_cache_key("abc123", spec, 44100)
		assert a == b

	def test_different_md5 (self) -> None:
		"""Different audio produces different hash."""
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		a = subsample.transform.variant_cache_key("abc123", spec, 44100)
		b = subsample.transform.variant_cache_key("def456", spec, 44100)
		assert a != b

	def test_different_sample_rate (self) -> None:
		"""Different output device produces different hash."""
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		a = subsample.transform.variant_cache_key("abc123", spec, 44100)
		b = subsample.transform.variant_cache_key("abc123", spec, 48000)
		assert a != b

	def test_different_spec (self) -> None:
		"""Different transform chain produces different hash."""
		spec_a = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		spec_b = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=72),),
		)
		a = subsample.transform.variant_cache_key("abc123", spec_a, 44100)
		b = subsample.transform.variant_cache_key("abc123", spec_b, 44100)
		assert a != b

	def test_order_sensitive (self) -> None:
		"""Different step ordering produces different hash."""
		spec_a = subsample.transform.TransformSpec(steps=(
			subsample.transform.LowPassFilter(freq=500.0),
			subsample.transform.Saturate(amount_db=6.0),
		))
		spec_b = subsample.transform.TransformSpec(steps=(
			subsample.transform.Saturate(amount_db=6.0),
			subsample.transform.LowPassFilter(freq=500.0),
		))
		a = subsample.transform.variant_cache_key("abc123", spec_a, 44100)
		b = subsample.transform.variant_cache_key("abc123", spec_b, 44100)
		assert a != b


# ---------------------------------------------------------------------------
# TestVariantDiskCache
# ---------------------------------------------------------------------------

class TestVariantDiskCache:

	"""Tests for VariantDiskCache read/write and FIFO eviction."""

	def _make_result (
		self,
		sample_id: int = 1,
		n_frames: int = 4410,
		channels: int = 2,
		midi_note: int = 60,
	) -> subsample.transform.TransformResult:

		audio = numpy.random.RandomState(42).uniform(-0.5, 0.5, (n_frames, channels)).astype(numpy.float32)
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=midi_note),),
		)
		key = subsample.transform.TransformKey(sample_id=sample_id, spec=spec)
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)
		return subsample.transform.TransformResult(key=key, audio=audio, duration=0.1, level=level)

	def test_roundtrip (self, tmp_path: pathlib.Path) -> None:
		"""Write then read back produces identical audio and metadata."""
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result()
		spec = result.key.spec

		cache.put("test_md5", spec, result)

		loaded = cache.get("test_md5", spec, result.key)
		assert loaded is not None
		numpy.testing.assert_array_equal(loaded.audio, result.audio)
		assert loaded.level.peak == pytest.approx(result.level.peak, abs=1e-5)
		assert loaded.level.rms == pytest.approx(result.level.rms, abs=1e-5)
		assert loaded.duration == pytest.approx(result.duration, abs=1e-3)

	def test_miss_returns_none (self, tmp_path: pathlib.Path) -> None:
		"""Non-existent key returns None."""
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		key = subsample.transform.TransformKey(sample_id=1, spec=spec)

		assert cache.get("nonexistent", spec, key) is None

	def test_disabled_when_zero_budget (self, tmp_path: pathlib.Path) -> None:
		"""max_bytes=0 disables cache — put/get are no-ops."""
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=0, sample_rate=44100,
		)
		assert not cache.enabled

		result = self._make_result()
		cache.put("test_md5", result.key.spec, result)
		assert cache.get("test_md5", result.key.spec, result.key) is None

	def test_different_md5_misses (self, tmp_path: pathlib.Path) -> None:
		"""Same spec but different audio_md5 produces a miss."""
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result()
		cache.put("md5_a", result.key.spec, result)

		assert cache.get("md5_b", result.key.spec, result.key) is None

	def test_different_sample_rate_misses (self, tmp_path: pathlib.Path) -> None:
		"""File written at 44100 is not returned by cache at 48000."""
		cache_44 = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		cache_48 = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=48000,
		)
		result = self._make_result()
		cache_44.put("test_md5", result.key.spec, result)

		# Same directory, different sample rate — file exists but hash differs.
		assert cache_48.get("test_md5", result.key.spec, result.key) is None

	def test_fifo_eviction (self, tmp_path: pathlib.Path) -> None:
		"""Writing more than max_bytes triggers deletion of oldest files."""
		# Each variant is ~35 KB (4410 frames * 2 channels * 4 bytes + header).
		# Set budget to hold ~2 variants.
		variant_bytes = 4410 * 2 * 4 + 32
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=variant_bytes * 2 + 100, sample_rate=44100,
		)

		r1 = self._make_result(midi_note=60)
		r2 = self._make_result(midi_note=72)
		r3 = self._make_result(midi_note=84)

		cache.put("md5", r1.key.spec, r1)
		cache.put("md5", r2.key.spec, r2)

		# Both should be on disk.
		files_before = list(tmp_path.glob("*.variant"))
		assert len(files_before) == 2

		# Writing a third should evict the oldest.
		cache.put("md5", r3.key.spec, r3)

		files_after = list(tmp_path.glob("*.variant"))
		assert len(files_after) <= 2

	def test_corrupt_file_deleted (self, tmp_path: pathlib.Path) -> None:
		"""A file with bad magic bytes is deleted on read."""
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result()
		spec = result.key.spec
		hex_digest = subsample.transform.variant_cache_key("test_md5", spec, 44100)
		bad_path = tmp_path / f"{hex_digest}.variant"

		bad_path.write_bytes(b"BAAD" + b"\x00" * 100)

		loaded = cache.get("test_md5", spec, result.key)
		assert loaded is None
		assert not bad_path.exists()


# ---------------------------------------------------------------------------
# TestHpssHarmonic / TestHpssPercussive
# ---------------------------------------------------------------------------

class TestHpssProcessors:

	"""Tests for the HPSS harmonic and percussive transform handlers."""

	def _make_mixed_signal (self, sr: int = 44100, duration: float = 0.5) -> numpy.ndarray:
		"""Create a stereo signal with a clear sine (harmonic) + click (percussive)."""
		t = numpy.arange(int(sr * duration), dtype=numpy.float32) / sr

		# Harmonic: sustained sine wave
		harmonic = 0.5 * numpy.sin(2 * numpy.pi * 440.0 * t)

		# Percussive: sharp click at 10% of the way through
		percussive = numpy.zeros_like(t)
		click_pos = int(0.1 * len(t))
		percussive[click_pos:click_pos + 50] = 0.8

		mono = (harmonic + percussive).astype(numpy.float32)
		return numpy.column_stack([mono, mono])  # stereo

	def test_harmonic_preserves_shape (self) -> None:
		audio = self._make_mixed_signal()
		record = _make_record(sample_id=1)
		result = subsample.transform._apply_hpss_harmonic(
			audio, 44100, record, subsample.transform.HpssHarmonic(),
		)
		assert result.shape == audio.shape
		assert result.dtype == numpy.float32

	def test_percussive_preserves_shape (self) -> None:
		audio = self._make_mixed_signal()
		record = _make_record(sample_id=1)
		result = subsample.transform._apply_hpss_percussive(
			audio, 44100, record, subsample.transform.HpssPercussive(),
		)
		assert result.shape == audio.shape
		assert result.dtype == numpy.float32

	def test_harmonic_has_less_transient_energy (self) -> None:
		"""Harmonic component should have less energy in the click region."""
		audio = self._make_mixed_signal()
		record = _make_record(sample_id=1)
		harmonic = subsample.transform._apply_hpss_harmonic(
			audio, 44100, record, subsample.transform.HpssHarmonic(),
		)

		# Check the click region (around 10% of the signal)
		click_start = int(0.1 * audio.shape[0])
		click_end = click_start + 50

		original_click_energy = float(numpy.sum(audio[click_start:click_end, 0] ** 2))
		harmonic_click_energy = float(numpy.sum(harmonic[click_start:click_end, 0] ** 2))

		# Harmonic should have significantly less click energy.
		assert harmonic_click_energy < original_click_energy * 0.8

	def test_percussive_has_less_sustained_energy (self) -> None:
		"""Percussive component should have less energy in the sustained region."""
		audio = self._make_mixed_signal()
		record = _make_record(sample_id=1)
		percussive = subsample.transform._apply_hpss_percussive(
			audio, 44100, record, subsample.transform.HpssPercussive(),
		)

		# Check a sustained region (last 30% of signal, well after the click)
		sustained_start = int(0.7 * audio.shape[0])
		original_sustained = float(numpy.sum(audio[sustained_start:, 0] ** 2))
		percussive_sustained = float(numpy.sum(percussive[sustained_start:, 0] ** 2))

		# Percussive should have significantly less sustained energy.
		assert percussive_sustained < original_sustained * 0.5

	def test_spec_from_process_hpss (self) -> None:
		"""hpss_harmonic and hpss_percussive are recognised by spec_from_process."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="hpss_harmonic"),
			subsample.query.ProcessorStep(name="repitch"),
		))
		spec = subsample.transform.spec_from_process(process, midi_note=60)
		assert len(spec.steps) == 2
		assert isinstance(spec.steps[0], subsample.transform.HpssHarmonic)
		assert isinstance(spec.steps[1], subsample.transform.PitchShift)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class TestGate:

	def test_spec_from_process_gate_boolean (self) -> None:
		"""gate: true → Gate with all None auto-fields."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="gate"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.Gate)
		assert spec.steps[0].threshold_db is None
		assert spec.steps[0].attack_ms is None

	def test_spec_from_process_gate_explicit (self) -> None:
		"""gate: {threshold: -40, hold: 20} → explicit params."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="gate", params=(("threshold", -40), ("hold", 20))),
		))
		spec = subsample.transform.spec_from_process(process)
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Gate)
		assert step.threshold_db == -40.0
		assert step.hold_ms == 20.0
		assert step.attack_ms is None  # not set → auto

	def test_signal_above_threshold_passes (self) -> None:
		"""A loud signal with a low threshold passes through mostly unchanged."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		# Fill with a moderate-level signal.
		audio[:, 0] = 0.5

		step = subsample.transform.Gate(threshold_db=-60.0, attack_ms=0.01, release_ms=0.01, hold_ms=0.0, lookahead_ms=0.0)
		result = subsample.transform._apply_gate(audio, 44100, record, step)

		# Signal is well above -60 dBFS, so gate should be open.
		assert result.dtype == numpy.float32
		assert numpy.max(numpy.abs(result)) > 0.4

	def test_signal_below_threshold_gated (self) -> None:
		"""A quiet signal below threshold is silenced."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=44100, channels=1)
		# Very quiet signal.
		audio[:, 0] = 0.0001

		step = subsample.transform.Gate(threshold_db=-20.0, attack_ms=0.01, release_ms=0.01, hold_ms=0.0, lookahead_ms=0.0)
		result = subsample.transform._apply_gate(audio, 44100, record, step)

		# Signal at ~-80 dBFS is well below -20 dBFS threshold.
		assert numpy.max(numpy.abs(result)) < 0.00005

	def test_auto_threshold_from_noise_floor (self) -> None:
		"""Auto threshold is derived from noise_floor + 6 dB."""
		record = _make_record(sample_id=1)
		# Override the level to have a known noise_floor.
		record = dataclasses.replace(record, level=subsample.analysis.LevelResult(
			peak=0.8, rms=0.3, crest_factor=2.67, crest_factor_db=8.5, noise_floor=0.01,
		))

		step = subsample.transform.Gate()  # all auto
		threshold, attack, release, hold, lookahead = subsample.transform._resolve_gate_params(record, step)

		# noise_floor = 0.01 → -40 dBFS → threshold = -34 dBFS
		import math
		expected = 20.0 * math.log10(0.01 + 1e-10) + 6.0
		assert abs(threshold - expected) < 0.1

	def test_explicit_overrides_auto (self) -> None:
		"""Explicit params take precedence over auto values."""
		record = _make_record(sample_id=1)
		step = subsample.transform.Gate(threshold_db=-30.0, attack_ms=2.0, release_ms=50.0, hold_ms=25.0, lookahead_ms=1.0)
		threshold, attack, release, hold, lookahead = subsample.transform._resolve_gate_params(record, step)

		assert threshold == -30.0
		assert attack == 2.0
		assert release == 50.0
		assert hold == 25.0
		assert lookahead == 1.0

	def test_output_dtype (self) -> None:
		"""Gate output is float32."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=2)
		step = subsample.transform.Gate(threshold_db=-60.0, attack_ms=1.0, release_ms=10.0, hold_ms=5.0, lookahead_ms=0.0)
		result = subsample.transform._apply_gate(audio, 44100, record, step)
		assert result.dtype == numpy.float32


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------

class TestDistort:

	def test_spec_from_process_distort_boolean (self) -> None:
		"""distort: true → Distort with default mode and auto drive/tone."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="distort"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Distort)
		assert step.mode == "hard_clip"
		assert step.drive_db is None  # auto
		assert step.tone is None      # auto

	def test_spec_from_process_distort_explicit (self) -> None:
		"""distort: {mode: fold, drive: 12} → explicit params."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="distort", params=(("mode", "fold"), ("drive", 12))),
		))
		spec = subsample.transform.spec_from_process(process)
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Distort)
		assert step.mode == "fold"
		assert step.drive_db == 12.0

	def test_hard_clip_clips (self) -> None:
		"""Hard-clip mode limits output to ±1."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = numpy.linspace(-0.8, 0.8, 4410, dtype=numpy.float32)

		step = subsample.transform.Distort(mode="hard_clip", drive_db=20.0, tone=1.0, mix=1.0)
		result = subsample.transform._apply_distort(audio, 44100, record, step)

		# After level compensation, output should be within ±1.
		assert numpy.all(numpy.abs(result) <= 1.01)

	def test_fold_wraps (self) -> None:
		"""Fold mode produces values within ±1 even with high drive."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = numpy.sin(numpy.linspace(0, 20, 4410)).astype(numpy.float32) * 0.5

		step = subsample.transform.Distort(mode="fold", drive_db=20.0, tone=1.0, mix=1.0)
		result = subsample.transform._apply_distort(audio, 44100, record, step)

		# Fold should keep everything in [-1, 1] before level compensation.
		assert result.dtype == numpy.float32

	def test_bit_crush_quantizes (self) -> None:
		"""Bit-crush mode reduces the number of distinct amplitude levels."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = numpy.linspace(-0.5, 0.5, 4410, dtype=numpy.float32)

		step = subsample.transform.Distort(mode="bit_crush", drive_db=0.0, bit_depth=2, tone=1.0, mix=1.0)
		result = subsample.transform._apply_distort(audio, 44100, record, step)

		# With 2-bit depth (4 levels), there should be very few unique values.
		unique_count = len(numpy.unique(numpy.round(result, decimals=4)))
		assert unique_count < 20  # far fewer than the 4410 input samples

	def test_downsample_reduces_detail (self) -> None:
		"""Downsample mode produces repeated sample groups."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4400, channels=1)
		audio[:, 0] = numpy.sin(numpy.linspace(0, 100, 4400)).astype(numpy.float32) * 0.5

		step = subsample.transform.Distort(mode="downsample", drive_db=0.0, downsample_factor=4, tone=1.0, mix=1.0)
		result = subsample.transform._apply_distort(audio, 44100, record, step)

		# Adjacent samples within each group of 4 should be identical.
		assert result.shape[0] == audio.shape[0]

	def test_mix_blend (self) -> None:
		"""mix=0.5 blends dry and wet signals."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = numpy.sin(numpy.linspace(0, 20, 4410)).astype(numpy.float32) * 0.5

		step_full = subsample.transform.Distort(mode="hard_clip", drive_db=20.0, tone=1.0, mix=1.0)
		step_half = subsample.transform.Distort(mode="hard_clip", drive_db=20.0, tone=1.0, mix=0.5)

		result_full = subsample.transform._apply_distort(audio, 44100, record, step_full)
		result_half = subsample.transform._apply_distort(audio, 44100, record, step_half)

		# Half-mix should be closer to the original than full-wet.
		diff_full = numpy.mean(numpy.abs(result_full - audio))
		diff_half = numpy.mean(numpy.abs(result_half - audio))
		assert diff_half < diff_full

	def test_silence_stays_silent (self) -> None:
		"""Zero input remains zero."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)  # all zeros

		step = subsample.transform.Distort(mode="hard_clip", drive_db=12.0, tone=1.0)
		result = subsample.transform._apply_distort(audio, 44100, record, step)

		assert numpy.max(numpy.abs(result)) == 0.0

	def test_output_dtype (self) -> None:
		"""Distort output is float32."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = 0.5
		step = subsample.transform.Distort(mode="hard_clip", drive_db=6.0, tone=1.0)
		result = subsample.transform._apply_distort(audio, 44100, record, step)
		assert result.dtype == numpy.float32


# ---------------------------------------------------------------------------
# Reshape (Envelope Shaping)
# ---------------------------------------------------------------------------

class TestReshape:

	def test_spec_from_process_reshape_boolean (self) -> None:
		"""reshape: true → Reshape with auto release, all else preserve."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="reshape"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Reshape)
		assert step.attack_ms is None
		assert step.release_ms is None  # auto
		assert step.sustain == 1.0

	def test_spec_from_process_reshape_explicit (self) -> None:
		"""reshape: {attack: 5, release: 100} → explicit params."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="reshape", params=(("attack", 5), ("release", 100))),
		))
		spec = subsample.transform.spec_from_process(process)
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Reshape)
		assert step.attack_ms == 5.0
		assert step.release_ms == 100.0

	def test_release_truncates_tail (self) -> None:
		"""Explicit release causes the audio to fade to zero at the end."""
		record = _make_record(sample_id=1, onset_times=(0.0,), attack_times=(0.0,))
		audio = _make_audio(n_frames=44100, channels=1)
		# Constant-level signal for 1 second.
		audio[:, 0] = 0.5

		step = subsample.transform.Reshape(release_ms=50.0)
		result = subsample.transform._apply_reshape(audio, 44100, record, step)

		# The last few samples should be near zero.
		assert numpy.max(numpy.abs(result[-10:])) < 0.05
		# The beginning should be unchanged.
		assert numpy.mean(numpy.abs(result[:1000])) > 0.4

	def test_sustain_reduces_level (self) -> None:
		"""sustain=0.5 reduces the sustained portion to half level."""
		record = _make_record(sample_id=1, onset_times=(0.0,), attack_times=(0.0,))
		audio = _make_audio(n_frames=44100, channels=1)
		audio[:, 0] = 0.5

		step = subsample.transform.Reshape(sustain=0.5, release_ms=10.0)
		result = subsample.transform._apply_reshape(audio, 44100, record, step)

		# Middle portion should be at roughly half the original level.
		mid = result[10000:20000, 0]
		assert numpy.mean(numpy.abs(mid)) < 0.35

	def test_attack_reshapes_onset (self) -> None:
		"""attack_ms creates a ramp from silence at the onset."""
		record = _make_record(sample_id=1, onset_times=(0.0,), attack_times=(0.0,))
		audio = _make_audio(n_frames=44100, channels=1)
		audio[:, 0] = 0.5

		step = subsample.transform.Reshape(attack_ms=100.0, release_ms=50.0)
		result = subsample.transform._apply_reshape(audio, 44100, record, step)

		# First few samples should be near zero (attack ramp).
		assert numpy.max(numpy.abs(result[:10])) < 0.05
		# After the attack (~100ms = 4410 samples), should be near full level.
		assert numpy.mean(numpy.abs(result[5000:6000])) > 0.4

	def test_auto_release_tightens_tail (self) -> None:
		"""reshape: true with auto release should tighten the tail."""
		record = _make_record(sample_id=1, onset_times=(0.0,), attack_times=(0.0,))
		audio = _make_audio(n_frames=44100, channels=1)
		audio[:, 0] = 0.5

		step = subsample.transform.Reshape()  # all auto
		result = subsample.transform._apply_reshape(audio, 44100, record, step)

		# The last samples should fade toward zero (auto release tightens tail).
		assert numpy.max(numpy.abs(result[-10:])) < 0.1

	def test_output_dtype (self) -> None:
		"""Reshape output is float32."""
		record = _make_record(sample_id=1, onset_times=(0.0,), attack_times=(0.0,))
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = 0.5
		step = subsample.transform.Reshape(release_ms=50.0)
		result = subsample.transform._apply_reshape(audio, 44100, record, step)
		assert result.dtype == numpy.float32


# ---------------------------------------------------------------------------
# PadQuantize (onset-aligned silence padding)
# ---------------------------------------------------------------------------

class TestPadQuantize:

	def test_spec_from_process_pad_quantize (self) -> None:
		"""pad_quantize: {bpm: 120, grid: 8} → PadQuantize with correct params."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("grid", 8)),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.PadQuantize)
		assert step.target_bpm == 120.0
		assert step.resolution == 8
		assert step.amount == 1.0

	def test_spec_from_process_pad_quantize_amount (self) -> None:
		"""pad_quantize: {bpm: 120, amount: 0.5} → PadQuantize with amount."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", 0.5)),
			),
		))
		spec = subsample.transform.spec_from_process(process)
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.PadQuantize)
		assert step.amount == 0.5

	def test_spec_from_process_pad_quantize_boolean (self) -> None:
		"""pad_quantize: true with target_bpm → PadQuantize with defaults."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="pad_quantize"),
		))
		spec = subsample.transform.spec_from_process(process, target_bpm=100.0)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.PadQuantize)
		assert step.target_bpm == 100.0
		assert step.resolution == 16

	def test_two_onsets_inserts_silence (self) -> None:
		"""Two onsets close together are spread apart with silence between them."""
		sr = 44100
		# Two short clicks at 0.0s and 0.05s (50 ms apart).
		onset_a = 0.0
		onset_b = 0.05
		audio = _make_audio(n_frames=int(0.15 * sr), channels=1)
		# Place impulses at the onset positions.
		audio[int(onset_a * sr), 0] = 0.9
		audio[int(onset_b * sr), 0] = 0.9

		record = _make_record(
			sample_id=1,
			onset_times=(onset_a, onset_b),
			attack_times=(onset_a, onset_b),
		)

		# 120 BPM, grid=4 (quarter notes) → grid interval = 0.5s.
		# The two onsets at 0.0s and 0.05s should be snapped to 0.0s and 0.5s.
		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=4)
		result = subsample.transform._apply_pad_quantize(audio, sr, record, step)

		# Output should be longer than input (silence was inserted).
		assert result.shape[0] > audio.shape[0]

	def test_grid_alignment (self) -> None:
		"""Onsets in the output land at expected grid positions."""
		sr = 44100
		# Two 10ms bursts at 0.0s and 0.03s.
		audio = _make_audio(n_frames=int(0.1 * sr), channels=1)
		burst_len = int(0.01 * sr)
		audio[:burst_len, 0] = 0.9
		audio[int(0.03 * sr):int(0.03 * sr) + burst_len, 0] = 0.9

		record = _make_record(
			sample_id=1,
			onset_times=(0.0, 0.03),
			attack_times=(0.0, 0.03),
		)

		# 120 BPM, grid=8 (eighth notes) → grid interval = 0.25s.
		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=8)
		result = subsample.transform._apply_pad_quantize(audio, sr, record, step)

		# Second onset should be at ~0.25s (grid point 1).
		# Check a window past the 1ms fade-in for signal energy.
		target_frame = int(0.25 * sr)
		fade_samples = int(0.001 * sr) + 5  # past the fade-in
		window = result[target_frame + fade_samples:target_frame + fade_samples + 100, 0]
		assert numpy.max(numpy.abs(window)) > 0.3

	def test_single_onset_returns_unchanged (self) -> None:
		"""Single onset — nothing to pad, return audio unchanged."""
		record = _make_record(sample_id=1, onset_times=(0.0,), attack_times=(0.0,))
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = 0.5

		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=16)
		result = subsample.transform._apply_pad_quantize(audio, 44100, record, step)

		# Single onset means no padding — same audio returned.
		assert result.shape == audio.shape

	def test_no_onsets_returns_unchanged (self) -> None:
		"""No onsets — return audio unchanged."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = 0.5

		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=16)
		result = subsample.transform._apply_pad_quantize(audio, 44100, record, step)

		assert result.shape == audio.shape

	def test_stereo_preserved (self) -> None:
		"""Stereo audio stays stereo after pad_quantize."""
		sr = 44100
		audio = _make_audio(n_frames=int(0.1 * sr), channels=2)
		audio[0, :] = 0.9
		audio[int(0.03 * sr), :] = 0.9

		record = _make_record(
			sample_id=1,
			onset_times=(0.0, 0.03),
			attack_times=(0.0, 0.03),
		)

		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=8)
		result = subsample.transform._apply_pad_quantize(audio, sr, record, step)

		assert result.shape[1] == 2

	def test_splice_fades (self) -> None:
		"""Samples near splice boundaries should be near zero (S-curve fades)."""
		sr = 44100
		# Constant-level audio with two onsets.
		audio = _make_audio(n_frames=int(0.2 * sr), channels=1)
		audio[:, 0] = 0.5
		audio[0, 0] = 0.9
		audio[int(0.05 * sr), 0] = 0.9

		record = _make_record(
			sample_id=1,
			onset_times=(0.0, 0.05),
			attack_times=(0.0, 0.05),
		)

		# Wide grid so there's definite silence between segments.
		step = subsample.transform.PadQuantize(target_bpm=60.0, resolution=4)
		result = subsample.transform._apply_pad_quantize(audio, sr, record, step)

		# The end of the first segment should fade out toward zero.
		# Find the silence gap — should have near-zero values.
		first_seg_end = int(0.05 * sr) + 10  # a bit past the first segment
		gap_region = result[first_seg_end:first_seg_end + 100, 0]

		assert numpy.max(numpy.abs(gap_region)) < 0.1

	def test_output_dtype (self) -> None:
		"""PadQuantize output is float32."""
		sr = 44100
		audio = _make_audio(n_frames=int(0.1 * sr), channels=1)
		audio[0, 0] = 0.9
		audio[int(0.03 * sr), 0] = 0.9

		record = _make_record(
			sample_id=1,
			onset_times=(0.0, 0.03),
			attack_times=(0.0, 0.03),
		)

		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=8)
		result = subsample.transform._apply_pad_quantize(audio, sr, record, step)

		assert result.dtype == numpy.float32

	def test_amount_zero_returns_unchanged (self) -> None:
		"""amount=0.0 returns original audio with no grid snapping."""
		sr = 44100
		audio = _make_audio(n_frames=int(0.1 * sr), channels=1)
		audio[:, 0] = 0.5

		record = _make_record(
			sample_id=1,
			onset_times=(0.0, 0.03),
			attack_times=(0.0, 0.03),
		)

		step = subsample.transform.PadQuantize(target_bpm=120.0, resolution=8, amount=0.0)
		result = subsample.transform._apply_pad_quantize(audio, sr, record, step)

		numpy.testing.assert_array_equal(result, audio)

	def test_amount_half_less_displacement (self) -> None:
		"""amount=0.5 moves onsets less than amount=1.0."""
		sr = 44100
		audio = _make_audio(n_frames=int(0.15 * sr), channels=1)
		audio[0, 0] = 0.9
		audio[int(0.05 * sr), 0] = 0.9

		record = _make_record(
			sample_id=1,
			onset_times=(0.0, 0.05),
			attack_times=(0.0, 0.05),
		)

		full = subsample.transform.PadQuantize(target_bpm=120.0, resolution=4, amount=1.0)
		half = subsample.transform.PadQuantize(target_bpm=120.0, resolution=4, amount=0.5)

		result_full = subsample.transform._apply_pad_quantize(audio, sr, record, full)
		result_half = subsample.transform._apply_pad_quantize(audio, sr, record, half)

		# Half-quantize should produce a shorter output (less silence inserted)
		# than full quantize, since the second onset moves less.
		assert result_half.shape[0] < result_full.shape[0]

	def test_amount_clamped (self) -> None:
		"""amount is clamped to [0.0, 1.0] in spec_from_process."""
		process_over = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", 2.0)),
			),
		))
		spec = subsample.transform.spec_from_process(process_over)
		assert spec.steps[0].amount == 1.0

		process_under = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize",
				params=(("tempo", 120), ("strength", -0.5)),
			),
		))
		spec = subsample.transform.spec_from_process(process_under)
		assert spec.steps[0].amount == 0.0


# ---------------------------------------------------------------------------
# Transient (HPSS-based transient enhancement/taming)
# ---------------------------------------------------------------------------

class TestTransient:

	def test_spec_from_process_transient_boolean (self) -> None:
		"""transient: true → Transient with auto amount."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="transient"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Transient)
		assert step.amount_db is None  # auto

	def test_spec_from_process_transient_explicit (self) -> None:
		"""transient: {amount: 6} → explicit amount."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="transient", params=(("gain", 6),)),
		))
		spec = subsample.transform.spec_from_process(process)
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Transient)
		assert step.amount_db == 6.0

	def test_auto_high_crest_tames (self) -> None:
		"""High crest factor (peaky) → auto amount should be negative (taming)."""
		record = _make_record(sample_id=1)
		record = dataclasses.replace(record, level=subsample.analysis.LevelResult(
			peak=0.9, rms=0.1, crest_factor=9.0, crest_factor_db=19.1, noise_floor=0.01,
		))
		step = subsample.transform.Transient()  # auto
		amount = subsample.transform._resolve_transient_params(record, step)
		assert amount < 0.0  # should tame

	def test_auto_low_crest_enhances (self) -> None:
		"""Low crest factor (dull) → auto amount should be positive (enhancing)."""
		record = _make_record(sample_id=1)
		record = dataclasses.replace(record, level=subsample.analysis.LevelResult(
			peak=0.5, rms=0.35, crest_factor=1.43, crest_factor_db=3.1, noise_floor=0.01,
		))
		step = subsample.transform.Transient()  # auto
		amount = subsample.transform._resolve_transient_params(record, step)
		assert amount > 0.0  # should enhance

	def test_enhance_increases_percussive_energy (self) -> None:
		"""Positive amount should increase energy in percussive frequency range."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.5 * sr)
		audio = _make_audio(n_frames=n, channels=1)

		# Create audio with a mix of tonal and percussive content.
		t = numpy.linspace(0, 0.5, n, dtype=numpy.float32)
		# Sine tone (harmonic) + clicks (percussive).
		audio[:, 0] = 0.3 * numpy.sin(2 * numpy.pi * 440 * t)
		audio[0, 0] += 0.5
		audio[sr // 4, 0] += 0.5

		step = subsample.transform.Transient(amount_db=6.0)
		result = subsample.transform._apply_transient(audio, sr, record, step)

		assert result.dtype == numpy.float32
		assert result.shape == audio.shape

	def test_tame_reduces_percussive_energy (self) -> None:
		"""Negative amount should reduce percussive component."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.5 * sr)
		audio = _make_audio(n_frames=n, channels=1)

		t = numpy.linspace(0, 0.5, n, dtype=numpy.float32)
		audio[:, 0] = 0.3 * numpy.sin(2 * numpy.pi * 440 * t)
		audio[0, 0] += 0.5
		audio[sr // 4, 0] += 0.5

		step = subsample.transform.Transient(amount_db=-6.0)
		result = subsample.transform._apply_transient(audio, sr, record, step)

		assert result.dtype == numpy.float32
		assert result.shape == audio.shape

	def test_level_compensation (self) -> None:
		"""Output peak should approximately match input peak."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.5 * sr)
		audio = _make_audio(n_frames=n, channels=1)

		t = numpy.linspace(0, 0.5, n, dtype=numpy.float32)
		audio[:, 0] = 0.5 * numpy.sin(2 * numpy.pi * 440 * t)

		step = subsample.transform.Transient(amount_db=6.0)
		result = subsample.transform._apply_transient(audio, sr, record, step)

		input_peak = float(numpy.max(numpy.abs(audio)))
		output_peak = float(numpy.max(numpy.abs(result)))

		assert abs(output_peak - input_peak) < 0.05

	def test_near_zero_is_passthrough (self) -> None:
		"""Amount near zero (< 0.1 dB) should return audio unchanged."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)
		audio[:, 0] = 0.5

		step = subsample.transform.Transient(amount_db=0.05)
		result = subsample.transform._apply_transient(audio, 44100, record, step)

		numpy.testing.assert_array_equal(result, audio)

	def test_silence_passthrough (self) -> None:
		"""Silent input returns silent output."""
		record = _make_record(sample_id=1)
		audio = _make_audio(n_frames=4410, channels=1)  # all zeros

		step = subsample.transform.Transient(amount_db=6.0)
		result = subsample.transform._apply_transient(audio, 44100, record, step)

		assert numpy.max(numpy.abs(result)) == 0.0

	def test_stereo_preserved (self) -> None:
		"""Stereo audio stays stereo."""
		record = _make_record(sample_id=1)
		sr = 44100
		audio = _make_audio(n_frames=int(0.1 * sr), channels=2)
		audio[:, 0] = 0.5
		audio[:, 1] = 0.3

		step = subsample.transform.Transient(amount_db=3.0)
		result = subsample.transform._apply_transient(audio, sr, record, step)

		assert result.shape[1] == 2


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------

def _write_carrier_wav (path: pathlib.Path, sr: int = 44100, duration: float = 0.5) -> None:

	"""Write a simple carrier WAV file (sawtooth wave) for vocoder tests."""

	n = int(sr * duration)
	t = numpy.linspace(0, duration, n, dtype=numpy.float32)
	# Sawtooth: harmonically rich carrier.
	saw = 2.0 * (t * 220.0 - numpy.floor(t * 220.0 + 0.5)).astype(numpy.float32)
	saw *= 0.5  # scale to [-0.5, 0.5]
	soundfile.write(str(path), saw.reshape(-1, 1), sr, subtype="PCM_16")


class TestVocoder:

	def test_spec_from_process_vocoder (self, tmp_path: pathlib.Path) -> None:
		"""vocoder: {carrier: ...} → Vocoder step with correct path."""
		carrier = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier)

		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="vocoder", params=(("carrier", str(carrier)),)),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Vocoder)
		assert step.carrier_path == str(carrier.resolve())
		assert step.bands == 24
		assert step.depth == 1.0

	def test_spec_from_process_vocoder_reference (self) -> None:
		"""vocoder: {carrier: reference} → Vocoder with reference_path substitution."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="vocoder", params=(("carrier", "reference"),)),
		))
		spec = subsample.transform.spec_from_process(
			process, reference_path="/resolved/ref.wav",
		)
		assert len(spec.steps) == 1
		step = spec.steps[0]
		assert isinstance(step, subsample.transform.Vocoder)
		assert step.carrier_path == "/resolved/ref.wav"

	def test_spec_from_process_vocoder_reference_no_path (self) -> None:
		"""vocoder: {carrier: reference} without reference_path → step skipped."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="vocoder", params=(("carrier", "reference"),)),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 0

	def test_spec_from_process_vocoder_no_carrier (self) -> None:
		"""vocoder: true (no carrier) → step skipped."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="vocoder"),
		))
		spec = subsample.transform.spec_from_process(process)
		assert len(spec.steps) == 0

	def test_vocoder_basic_cross_synthesis (self, tmp_path: pathlib.Path) -> None:
		"""Vocoder output differs from both dry modulator and carrier."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.3 * sr)

		# Modulator: white noise burst (spectrally flat, rhythmic).
		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.5, 0.5, (n, 1)).astype(numpy.float32)

		# Carrier: sawtooth (harmonically rich, steady).
		carrier_path = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.3)

		step = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=16)
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		assert result.shape == modulator.shape
		assert result.dtype == numpy.float32

		# Output should not be silence.
		assert numpy.max(numpy.abs(result)) > 0.01

		# Output should differ from the dry modulator (not identical).
		assert not numpy.allclose(result, modulator, atol=0.01)

	def test_vocoder_depth_zero_returns_dry (self, tmp_path: pathlib.Path) -> None:
		"""depth=0.0 → output equals dry modulator."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.2 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.3, 0.3, (n, 1)).astype(numpy.float32)

		carrier_path = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.2)

		step = subsample.transform.Vocoder(carrier_path=str(carrier_path), depth=0.0)
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		numpy.testing.assert_allclose(result, modulator, atol=1e-5)

	def test_vocoder_band_count_affects_output (self, tmp_path: pathlib.Path) -> None:
		"""Different band counts produce different outputs."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.2 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.5, 0.5, (n, 1)).astype(numpy.float32)

		carrier_path = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.2)

		step_8 = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=8)
		step_24 = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=24)

		result_8 = subsample.transform._apply_vocoder(modulator.copy(), sr, record, step_8)
		result_24 = subsample.transform._apply_vocoder(modulator.copy(), sr, record, step_24)

		assert not numpy.allclose(result_8, result_24, atol=0.01)

	def test_vocoder_carrier_shorter_loops (self, tmp_path: pathlib.Path) -> None:
		"""Carrier shorter than modulator is looped without error."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.5 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.3, 0.3, (n, 1)).astype(numpy.float32)

		# Short carrier (0.1s vs 0.5s modulator).
		carrier_path = tmp_path / "short_carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.1)

		step = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=16)
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		assert result.shape == modulator.shape
		assert numpy.max(numpy.abs(result)) > 0.01

	def test_vocoder_carrier_longer_truncates (self, tmp_path: pathlib.Path) -> None:
		"""Carrier longer than modulator is truncated without error."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.1 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.3, 0.3, (n, 1)).astype(numpy.float32)

		# Long carrier (1.0s vs 0.1s modulator).
		carrier_path = tmp_path / "long_carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=1.0)

		step = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=16)
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		assert result.shape == modulator.shape

	def test_vocoder_formant_shift (self, tmp_path: pathlib.Path) -> None:
		"""Formant shift produces different output from unshifted."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.2 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.5, 0.5, (n, 1)).astype(numpy.float32)

		carrier_path = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.2)

		step_normal = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=16)
		step_shifted = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=16, formant_shift=5)

		result_normal = subsample.transform._apply_vocoder(modulator.copy(), sr, record, step_normal)
		result_shifted = subsample.transform._apply_vocoder(modulator.copy(), sr, record, step_shifted)

		assert not numpy.allclose(result_normal, result_shifted, atol=0.01)

	def test_vocoder_silence_returns_silence (self, tmp_path: pathlib.Path) -> None:
		"""Silent modulator produces silent output."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.1 * sr)

		modulator = numpy.zeros((n, 1), dtype=numpy.float32)

		carrier_path = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.1)

		step = subsample.transform.Vocoder(carrier_path=str(carrier_path))
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		assert numpy.max(numpy.abs(result)) == 0.0

	def test_vocoder_stereo_preserved (self, tmp_path: pathlib.Path) -> None:
		"""Stereo modulator produces stereo output."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.2 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.5, 0.5, (n, 2)).astype(numpy.float32)

		carrier_path = tmp_path / "carrier.wav"
		_write_carrier_wav(carrier_path, sr=sr, duration=0.2)

		step = subsample.transform.Vocoder(carrier_path=str(carrier_path), bands=16)
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		assert result.shape == (n, 2)

	def test_vocoder_missing_carrier_returns_dry (self) -> None:
		"""Missing carrier file → returns dry modulator with warning."""
		record = _make_record(sample_id=1)
		sr = 44100
		n = int(0.1 * sr)

		rng = numpy.random.default_rng(42)
		modulator = rng.uniform(-0.3, 0.3, (n, 1)).astype(numpy.float32)

		step = subsample.transform.Vocoder(carrier_path="/nonexistent/carrier.wav")
		result = subsample.transform._apply_vocoder(modulator, sr, record, step)

		numpy.testing.assert_array_equal(result, modulator)

	def test_carrier_cache_evicts_oldest (self, tmp_path: pathlib.Path) -> None:
		"""Carrier cache evicts the oldest entry when the budget is exceeded."""

		# Write two small carrier files.
		c1 = tmp_path / "c1.wav"
		c2 = tmp_path / "c2.wav"
		_write_carrier_wav(c1, duration=0.01)
		_write_carrier_wav(c2, duration=0.01)

		# Clear the cache and set a tiny budget.
		orig_max = subsample.transform._CARRIER_CACHE_MAX_BYTES

		try:
			subsample.transform._carrier_cache.clear()
			subsample.transform._carrier_cache_order.clear()
			subsample.transform._carrier_cache_bytes = 0
			# Budget fits one carrier (441 samples × 4 bytes = 1764) but not two.
			subsample.transform._CARRIER_CACHE_MAX_BYTES = 1800

			mono1 = subsample.transform._load_carrier(str(c1), 44100)
			key1 = f"{c1}@44100"
			assert key1 in subsample.transform._carrier_cache

			mono2 = subsample.transform._load_carrier(str(c2), 44100)
			key2 = f"{c2}@44100"

			# c2 should be in cache; c1 should have been evicted.
			assert key2 in subsample.transform._carrier_cache
			assert key1 not in subsample.transform._carrier_cache

		finally:
			subsample.transform._CARRIER_CACHE_MAX_BYTES = orig_max
			subsample.transform._carrier_cache.clear()
			subsample.transform._carrier_cache_order.clear()
			subsample.transform._carrier_cache_bytes = 0



class TestSSV2DiskCache:

	"""Tests for SSV2 disk cache format with segment bounds and energy profile."""

	def _make_result (
		self,
		segment_bounds: typing.Optional[tuple[tuple[int, int], ...]] = None,
		energy_profile: typing.Optional[subsample.transform.GridEnergyProfile] = None,
	) -> subsample.transform.TransformResult:
		"""Create a test TransformResult with optional segment bounds and energy profile."""
		audio = numpy.random.RandomState(42).uniform(-0.5, 0.5, (4410, 2)).astype(numpy.float32)
		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		key = subsample.transform.TransformKey(sample_id=1, spec=spec)
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)
		return subsample.transform.TransformResult(
			key=key, audio=audio, duration=0.1, level=level,
			segment_bounds=segment_bounds,
			energy_profile=energy_profile,
		)

	def test_roundtrip_with_bounds (self, tmp_path: pathlib.Path) -> None:
		"""Write then read back preserves segment bounds."""
		bounds = ((0, 1000), (1500, 2500), (3000, 4410))
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result(segment_bounds=bounds)

		cache.put("test_md5", result.key.spec, result)
		loaded = cache.get("test_md5", result.key.spec, result.key)

		assert loaded is not None
		assert loaded.segment_bounds == bounds
		numpy.testing.assert_array_equal(loaded.audio, result.audio)

	def test_roundtrip_without_bounds (self, tmp_path: pathlib.Path) -> None:
		"""Write then read back with no bounds produces None."""
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result(segment_bounds=None)

		cache.put("test_md5_none", result.key.spec, result)
		loaded = cache.get("test_md5_none", result.key.spec, result.key)

		assert loaded is not None
		assert loaded.segment_bounds is None

	def test_reads_legacy_ssv1 (self, tmp_path: pathlib.Path) -> None:
		"""Reader accepts old SSV1 files with segment_bounds = None."""
		import struct

		audio = numpy.random.RandomState(42).uniform(-0.5, 0.5, (100, 1)).astype(numpy.float32)
		header = struct.pack("<4sHIIff10x", b"SSV1", 1, 44100, 100, 0.5, 0.2)

		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		key = subsample.transform.TransformKey(sample_id=1, spec=spec)

		hex_digest = subsample.transform.variant_cache_key("legacy_md5", spec, 44100)
		path = tmp_path / f"{hex_digest}.variant"
		path.write_bytes(header + audio.tobytes())

		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		loaded = cache.get("legacy_md5", spec, key)

		assert loaded is not None
		assert loaded.segment_bounds is None
		numpy.testing.assert_array_equal(loaded.audio, audio)

	def test_bounds_values_are_correct (self, tmp_path: pathlib.Path) -> None:
		"""Each bound (start, end) has start < end and is within range."""
		bounds = ((0, 1000), (1500, 2500), (3000, 4410))
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result(segment_bounds=bounds)
		cache.put("validate_md5", result.key.spec, result)
		loaded = cache.get("validate_md5", result.key.spec, result.key)

		assert loaded is not None
		assert loaded.segment_bounds is not None

		for start, end in loaded.segment_bounds:
			assert start < end
			assert end <= loaded.audio.shape[0]

	def test_roundtrip_with_energy_profile (self, tmp_path: pathlib.Path) -> None:

		"""Write then read back preserves energy profile alongside segment bounds."""

		bounds = ((0, 1000), (1500, 2500), (3000, 4410))
		profile = subsample.transform.GridEnergyProfile(
			bpm=120.0, resolution=16, energy=(1.0, 0.5, 0.0, 0.3),
		)
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result(segment_bounds=bounds, energy_profile=profile)
		cache.put("ep_md5", result.key.spec, result)
		loaded = cache.get("ep_md5", result.key.spec, result.key)

		assert loaded is not None
		assert loaded.segment_bounds == bounds
		assert loaded.energy_profile is not None
		assert loaded.energy_profile.bpm == pytest.approx(120.0)
		assert loaded.energy_profile.resolution == 16
		assert loaded.energy_profile.energy == pytest.approx((1.0, 0.5, 0.0, 0.3))

	def test_roundtrip_without_energy_profile (self, tmp_path: pathlib.Path) -> None:

		"""Write then read with no profile produces energy_profile = None."""

		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result(energy_profile=None)
		cache.put("no_ep_md5", result.key.spec, result)
		loaded = cache.get("no_ep_md5", result.key.spec, result.key)

		assert loaded is not None
		assert loaded.energy_profile is None

	def test_energy_profile_without_segment_bounds (self, tmp_path: pathlib.Path) -> None:

		"""Energy profile round-trips when segment_bounds is None (seg_count=0 sentinel)."""

		profile = subsample.transform.GridEnergyProfile(
			bpm=90.0, resolution=8, energy=(0.8, 1.0, 0.0, 0.0, 0.6),
		)
		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		result = self._make_result(segment_bounds=None, energy_profile=profile)
		cache.put("ep_no_bounds_md5", result.key.spec, result)
		loaded = cache.get("ep_no_bounds_md5", result.key.spec, result.key)

		assert loaded is not None
		assert loaded.segment_bounds is None
		assert loaded.energy_profile is not None
		assert loaded.energy_profile.bpm == pytest.approx(90.0)
		assert loaded.energy_profile.resolution == 8
		assert loaded.energy_profile.energy == pytest.approx((0.8, 1.0, 0.0, 0.0, 0.6))

	def test_old_format_no_energy_profile (self, tmp_path: pathlib.Path) -> None:

		"""SSV2 files written before energy profile support read back with None."""

		import struct

		audio = numpy.random.RandomState(42).uniform(-0.5, 0.5, (100, 1)).astype(numpy.float32)
		header = struct.pack("<4sHIIff10x", b"SSV2", 1, 44100, 100, 0.5, 0.2)

		# Write segment bounds but no ENRG section (old writer behaviour).
		bounds_footer = struct.pack("<I", 2)                # seg_count = 2
		bounds_footer += struct.pack("<II", 0, 50)          # segment 1
		bounds_footer += struct.pack("<II", 50, 100)        # segment 2

		spec = subsample.transform.TransformSpec(
			steps=(subsample.transform.PitchShift(target_midi_note=60),),
		)
		key = subsample.transform.TransformKey(sample_id=1, spec=spec)

		hex_digest = subsample.transform.variant_cache_key("old_fmt_md5", spec, 44100)
		path = tmp_path / f"{hex_digest}.variant"
		path.write_bytes(header + audio.tobytes() + bounds_footer)

		cache = subsample.transform.VariantDiskCache(
			directory=tmp_path, max_bytes=100_000_000, sample_rate=44100,
		)
		loaded = cache.get("old_fmt_md5", spec, key)

		assert loaded is not None
		assert loaded.segment_bounds == ((0, 50), (50, 100))
		assert loaded.energy_profile is None


# ---------------------------------------------------------------------------
# TestGridEnergyProfile
# ---------------------------------------------------------------------------

class TestGridEnergyProfile:

	"""Tests for _compute_grid_energy_profile()."""

	def test_basic_energy_distribution (self) -> None:

		"""First half loud, second half silent → first slots energised, last zero."""

		sr = 44100
		n_frames = sr  # 1 second

		# First half: sine wave.  Second half: silence.
		t = numpy.linspace(0, numpy.pi * 2 * 440, n_frames // 2, dtype=numpy.float32)
		loud = numpy.sin(t) * 0.5
		silent = numpy.zeros(n_frames - n_frames // 2, dtype=numpy.float32)
		mono = numpy.concatenate([loud, silent])
		audio = mono.reshape(-1, 1)

		profile = subsample.transform._compute_grid_energy_profile(
			audio, sr, bpm=120.0, resolution=8,
		)

		assert profile.bpm == 120.0
		assert profile.resolution == 8

		# At 120 BPM, resolution 8: grid_interval = 0.25s → 4 slots in 1s.
		assert len(profile.energy) == 4

		# Max is 1.0 (normalised).
		assert max(profile.energy) == pytest.approx(1.0)

		# First two slots (0-0.25s, 0.25-0.5s) should be energised.
		assert profile.energy[0] > 0.5
		assert profile.energy[1] > 0.5

		# Last two slots (0.5-0.75s, 0.75-1.0s) should be near zero.
		assert profile.energy[2] < 0.01
		assert profile.energy[3] < 0.01

	def test_single_slot (self) -> None:

		"""Audio shorter than one grid interval → single slot [1.0]."""

		sr = 44100
		# 10ms of audio, grid interval at 120 BPM / res 16 = 31.25ms.
		n_frames = int(sr * 0.01)
		audio = numpy.ones((n_frames, 1), dtype=numpy.float32) * 0.5

		profile = subsample.transform._compute_grid_energy_profile(
			audio, sr, bpm=120.0, resolution=16,
		)

		assert len(profile.energy) == 1
		assert profile.energy[0] == pytest.approx(1.0)

	def test_all_silent (self) -> None:

		"""All-zero audio → all slots are 0.0."""

		sr = 44100
		audio = numpy.zeros((sr, 1), dtype=numpy.float32)

		profile = subsample.transform._compute_grid_energy_profile(
			audio, sr, bpm=120.0, resolution=8,
		)

		for val in profile.energy:
			assert val == pytest.approx(0.0)

	def test_uniform_energy (self) -> None:

		"""Constant-amplitude audio → all slots approximately equal (1.0)."""

		sr = 44100
		audio = numpy.full((sr, 1), 0.3, dtype=numpy.float32)

		profile = subsample.transform._compute_grid_energy_profile(
			audio, sr, bpm=120.0, resolution=8,
		)

		for val in profile.energy:
			assert val == pytest.approx(1.0, abs=0.01)

	def test_stereo_audio (self) -> None:

		"""Multi-channel audio is mixed to mono before computation."""

		sr = 44100
		left  = numpy.ones((sr, 1), dtype=numpy.float32) * 0.4
		right = numpy.ones((sr, 1), dtype=numpy.float32) * 0.2
		audio = numpy.concatenate([left, right], axis=1)

		profile = subsample.transform._compute_grid_energy_profile(
			audio, sr, bpm=120.0, resolution=8,
		)

		# All slots should be uniform after mono mix.
		for val in profile.energy:
			assert val == pytest.approx(1.0, abs=0.01)

	def test_pad_quantize_pattern (self) -> None:

		"""Simulates a pad-quantized pattern: energy at slots 0, 2, 5 only."""

		sr = 44100
		# 120 BPM, resolution 16 → grid_interval = 0.125s = 5512 frames.
		slot_frames = int(0.125 * sr)
		n_slots = 8
		audio = numpy.zeros((slot_frames * n_slots, 1), dtype=numpy.float32)

		# Place energy at slots 0, 2, 5.
		for slot_idx in [0, 2, 5]:
			start = slot_idx * slot_frames
			end = start + slot_frames
			audio[start:end, 0] = 0.5

		profile = subsample.transform._compute_grid_energy_profile(
			audio, sr, bpm=120.0, resolution=16,
		)

		assert len(profile.energy) == 8
		assert profile.energy[0] == pytest.approx(1.0)
		assert profile.energy[1] == pytest.approx(0.0)
		assert profile.energy[2] == pytest.approx(1.0)
		assert profile.energy[3] == pytest.approx(0.0)
		assert profile.energy[4] == pytest.approx(0.0)
		assert profile.energy[5] == pytest.approx(1.0)
		assert profile.energy[6] == pytest.approx(0.0)
		assert profile.energy[7] == pytest.approx(0.0)
