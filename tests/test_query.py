"""Tests for subsample/query.py — sample query engine and MIDI map parser."""

import pathlib
import typing

import numpy
import pytest

import subsample.analysis
import subsample.library
import subsample.query
import subsample.similarity

import tests.helpers


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_record (
	sample_id: int = 1,
	name: str = "test",
	duration: float = 1.0,
	onset_count: int = 0,
	tempo_bpm: float = 0.0,
	dominant_pitch_hz: float = 0.0,
	pitch_confidence: float = 0.0,
	pitch_stability: float = 0.0,
	voiced_fraction: float = 0.0,
	harmonic_ratio: float = 0.0,
	rms: float = 0.1,
) -> subsample.library.SampleRecord:

	"""Build a SampleRecord with controllable fields for query testing."""

	spectral = subsample.analysis.AnalysisResult(
		spectral_flatness  = 0.5,
		attack             = 0.5,
		release            = 0.5,
		spectral_centroid  = 0.5,
		spectral_bandwidth = 0.5,
		zcr                = 0.5,
		harmonic_ratio     = harmonic_ratio,
		spectral_contrast  = 0.5,
		voiced_fraction    = voiced_fraction,
		log_attack_time    = 0.5,
		spectral_flux      = 0.5,
		spectral_rolloff   = 0.5,
		spectral_slope     = 0.5,
	)

	rhythm = subsample.analysis.RhythmResult(
		tempo_bpm        = tempo_bpm,
		beat_times       = (),
		pulse_curve      = numpy.zeros(0, dtype=numpy.float32),
		pulse_peak_times = (),
		onset_times      = tuple(float(i) * 0.1 for i in range(onset_count)),
		attack_times     = tuple(float(i) * 0.1 for i in range(onset_count)),
		onset_count      = onset_count,
	)

	pitch = subsample.analysis.PitchResult(
		dominant_pitch_hz    = dominant_pitch_hz,
		pitch_confidence     = pitch_confidence,
		chroma_profile       = tuple(0.0 for _ in range(12)),
		dominant_pitch_class = -1,
		pitch_stability      = pitch_stability,
		voiced_frame_count   = 10 if dominant_pitch_hz > 0 else 0,
	)

	level = subsample.analysis.LevelResult(peak=0.8, rms=rms)

	return subsample.library.SampleRecord(
		sample_id   = sample_id,
		name        = name,
		spectral    = spectral,
		rhythm      = rhythm,
		pitch       = pitch,
		timbre      = tests.helpers._make_timbre(),
		level       = level,
		band_energy = tests.helpers._make_band_energy(),
		params      = tests.helpers._make_params(),
		duration    = duration,
		audio       = None,
	)


# ---------------------------------------------------------------------------
# WherePredicate.matches
# ---------------------------------------------------------------------------

class TestWherePredicate:

	def test_empty_predicate_matches_all (self) -> None:

		"""A predicate with no filters matches every record."""

		pred = subsample.query.WherePredicate()
		record = _make_record()
		assert pred.matches(record)

	def test_min_duration (self) -> None:
		pred = subsample.query.WherePredicate(duration=subsample.query.Range(gte=2.0))
		assert not pred.matches(_make_record(duration=1.5))
		assert pred.matches(_make_record(duration=2.5))

	def test_max_duration (self) -> None:
		pred = subsample.query.WherePredicate(duration=subsample.query.Range(lte=1.0))
		assert pred.matches(_make_record(duration=0.5))
		assert not pred.matches(_make_record(duration=1.5))

	def test_min_onsets (self) -> None:
		pred = subsample.query.WherePredicate(onsets=subsample.query.Range(gte=4))
		assert not pred.matches(_make_record(onset_count=2))
		assert pred.matches(_make_record(onset_count=5))

	def test_max_onsets (self) -> None:
		pred = subsample.query.WherePredicate(onsets=subsample.query.Range(lte=3))
		assert pred.matches(_make_record(onset_count=2))
		assert not pred.matches(_make_record(onset_count=5))

	def test_pitched_true (self) -> None:

		"""pitched: true requires has_stable_pitch() to pass."""

		pred = subsample.query.WherePredicate(pitched=True)

		# Not pitched: no dominant pitch
		assert not pred.matches(_make_record(dominant_pitch_hz=0.0))

		# Pitched: stable pitch with all criteria met
		assert pred.matches(_make_record(
			dominant_pitch_hz=440.0,
			pitch_confidence=0.8,
			pitch_stability=0.2,
			voiced_fraction=0.8,
			harmonic_ratio=0.6,
			duration=0.5,
		))

	def test_pitched_false (self) -> None:

		"""pitched: false requires has_stable_pitch() to fail."""

		pred = subsample.query.WherePredicate(pitched=False)
		assert pred.matches(_make_record(dominant_pitch_hz=0.0))

	def test_min_tempo (self) -> None:
		pred = subsample.query.WherePredicate(tempo=subsample.query.Range(gte=100.0))
		assert not pred.matches(_make_record(tempo_bpm=80.0))
		assert pred.matches(_make_record(tempo_bpm=120.0))

	def test_max_tempo (self) -> None:
		pred = subsample.query.WherePredicate(tempo=subsample.query.Range(lte=130.0))
		assert pred.matches(_make_record(tempo_bpm=120.0))
		assert not pred.matches(_make_record(tempo_bpm=140.0))

	def test_min_pitch_hz (self) -> None:
		pred = subsample.query.WherePredicate(pitch_hz=subsample.query.Range(gte=300.0))
		assert not pred.matches(_make_record(dominant_pitch_hz=200.0))
		assert pred.matches(_make_record(dominant_pitch_hz=440.0))

	def test_max_pitch_hz (self) -> None:
		pred = subsample.query.WherePredicate(pitch_hz=subsample.query.Range(lte=500.0))
		assert pred.matches(_make_record(dominant_pitch_hz=440.0))
		assert not pred.matches(_make_record(dominant_pitch_hz=880.0))

	def test_name (self) -> None:
		pred = subsample.query.WherePredicate(name="my-kick")
		assert pred.matches(_make_record(name="my-kick"))
		assert not pred.matches(_make_record(name="other"))

	def test_combined_predicates (self) -> None:

		"""Multiple predicates are AND-ed: all must pass."""

		pred = subsample.query.WherePredicate(duration=subsample.query.Range(gte=1.0), onsets=subsample.query.Range(gte=4))
		assert not pred.matches(_make_record(duration=2.0, onset_count=2))
		assert not pred.matches(_make_record(duration=0.5, onset_count=5))
		assert pred.matches(_make_record(duration=2.0, onset_count=5))


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestQuery:

	def _samples (self) -> list[subsample.library.SampleRecord]:

		"""Build a diverse set of test samples."""

		return [
			_make_record(sample_id=1, name="short-quiet",   duration=0.3, onset_count=1, rms=0.05),
			_make_record(sample_id=2, name="long-rhythmic", duration=2.0, onset_count=8, tempo_bpm=120.0, rms=0.2),
			_make_record(sample_id=3, name="medium-tonal",  duration=1.0, onset_count=2, dominant_pitch_hz=440.0, pitch_confidence=0.9, pitch_stability=0.1, voiced_fraction=0.9, harmonic_ratio=0.7, rms=0.15),
			_make_record(sample_id=4, name="long-quiet",    duration=3.0, onset_count=3, rms=0.03),
			_make_record(sample_id=5, name="short-loud",    duration=0.5, onset_count=6, tempo_bpm=140.0, rms=0.4),
		]

	def test_no_filter_returns_all (self) -> None:

		"""Empty where + newest ordering returns all samples, newest first."""

		spec = subsample.query.SelectSpec()
		result = subsample.query.query(spec, self._samples())
		assert len(result) == 5
		assert result[0].sample_id == 5  # newest

	def test_min_duration_filter (self) -> None:
		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(duration=subsample.query.Range(gte=1.0)),
		)
		result = subsample.query.query(spec, self._samples())
		assert all(r.duration >= 1.0 for r in result)
		assert len(result) == 3

	def test_order_by_duration_desc (self) -> None:
		spec = subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="duration", dir="desc"),))
		result = subsample.query.query(spec, self._samples())
		assert result[0].name == "long-quiet"
		assert result[1].name == "long-rhythmic"

	def test_order_by_oldest (self) -> None:
		spec = subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="age", dir="asc"),))
		result = subsample.query.query(spec, self._samples())
		assert result[0].sample_id == 1

	def test_order_by_loudest (self) -> None:
		spec = subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="level", dir="desc"),))
		result = subsample.query.query(spec, self._samples())
		assert result[0].name == "short-loud"

	def test_combined_filter_and_order (self) -> None:

		"""Longer than 1s, ordered by duration descending."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(duration=subsample.query.Range(gte=1.0)),
			order=(subsample.query.OrderClause(by="duration", dir="desc"),),
		)
		result = subsample.query.query(spec, self._samples())
		assert len(result) == 3
		assert result[0].name == "long-quiet"

	def test_pick_nth (self) -> None:

		"""Pick selects the Nth position (1-indexed) after filtering and ordering."""

		spec = subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="duration", dir="desc"),), pick=3)
		result = subsample.query.query(spec, self._samples())

		# pick is handled by the caller, not by query() — query returns the full ranked list.
		# The caller does: result[pick - 1] if pick <= len(result).
		assert len(result) == 5
		assert result[2].name == "medium-tonal"  # 3rd longest

	def test_empty_result (self) -> None:

		"""Filters that match nothing return an empty list."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(duration=subsample.query.Range(gte=100.0)),
		)
		result = subsample.query.query(spec, self._samples())
		assert result == []

	def test_name_filter (self) -> None:
		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(name="medium-tonal"),
		)
		result = subsample.query.query(spec, self._samples())
		assert len(result) == 1
		assert result[0].name == "medium-tonal"


# ---------------------------------------------------------------------------
# quantized_beats filter and ordering
# ---------------------------------------------------------------------------

class TestQuantizedBeats:

	"""Filter and order samples by their quantized beat count.

	The beats_resolver callable maps sample_id -> beats (float) or None.
	Tests use a mock resolver; real resolvers come from TransformManager
	via the GridEnergyProfile on cached variants.
	"""

	def _samples (self) -> list[subsample.library.SampleRecord]:

		"""Four samples with distinct IDs for resolver lookup."""

		return [
			_make_record(sample_id=1, name="two-beats"),
			_make_record(sample_id=2, name="four-beats"),
			_make_record(sample_id=3, name="eight-beats"),
			_make_record(sample_id=4, name="no-variant"),
		]

	def _resolver (self) -> typing.Callable[[int], typing.Optional[float]]:

		"""Return a resolver mapping ids 1/2/3 to beats; id 4 has no variant."""

		mapping = {1: 2.0, 2: 4.0, 3: 8.0}
		return lambda sid: mapping.get(sid)

	def test_min_quantized_beats_filter (self) -> None:

		"""min_quantized_beats excludes samples below the threshold."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(quantized_beats=subsample.query.Range(gte=4.0)),
		)
		result = subsample.query.query(spec, self._samples(), beats_resolver=self._resolver())

		ids = {r.sample_id for r in result}
		assert ids == {2, 3}   # 4.0 and 8.0 pass; 2.0 and None excluded

	def test_max_quantized_beats_filter (self) -> None:

		"""max_quantized_beats excludes samples above the threshold."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(quantized_beats=subsample.query.Range(lte=4.0)),
		)
		result = subsample.query.query(spec, self._samples(), beats_resolver=self._resolver())

		ids = {r.sample_id for r in result}
		assert ids == {1, 2}   # 2.0 and 4.0 pass; 8.0 and None excluded

	def test_quantized_beats_range (self) -> None:

		"""Combined min and max select the middle sample only."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(
				quantized_beats=subsample.query.Range(gte=3.0, lte=5.0),
			),
		)
		result = subsample.query.query(spec, self._samples(), beats_resolver=self._resolver())

		assert [r.sample_id for r in result] == [2]

	def test_quantized_beats_missing_variant_excluded (self) -> None:

		"""Samples whose resolver returns None are excluded from min/max filters."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(quantized_beats=subsample.query.Range(gte=0.0)),
		)
		result = subsample.query.query(spec, self._samples(), beats_resolver=self._resolver())

		ids = {r.sample_id for r in result}
		assert 4 not in ids                 # no-variant excluded
		assert ids == {1, 2, 3}

	def test_quantized_beats_no_resolver_excluded (self) -> None:

		"""Without a resolver, quantized_beats filters exclude everything."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(quantized_beats=subsample.query.Range(gte=1.0)),
		)
		result = subsample.query.query(spec, self._samples(), beats_resolver=None)

		assert result == []

	def test_order_by_quantized_beats_asc (self) -> None:

		"""quantized_beats_asc orders smallest first; None goes to the end."""

		spec = subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="quantized_beats", dir="asc"),))
		result = subsample.query.query(spec, self._samples(), beats_resolver=self._resolver())

		ids = [r.sample_id for r in result]
		assert ids[:3] == [1, 2, 3]         # 2.0, 4.0, 8.0
		assert ids[3] == 4                  # None sorts last

	def test_order_by_quantized_beats_desc (self) -> None:

		"""quantized_beats_desc orders largest first; None still goes to the end."""

		spec = subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="quantized_beats", dir="desc"),))
		result = subsample.query.query(spec, self._samples(), beats_resolver=self._resolver())

		ids = [r.sample_id for r in result]
		assert ids[:3] == [3, 2, 1]         # 8.0, 4.0, 2.0
		assert ids[3] == 4                  # None sorts last, both directions

	def test_non_integer_beats (self) -> None:

		"""Fractional beat counts filter and order correctly."""

		mapping = {1: 3.75, 2: 3.5, 3: 4.25}
		resolver = lambda sid: mapping.get(sid)

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(quantized_beats=subsample.query.Range(gte=3.6)),
			order=(subsample.query.OrderClause(by="quantized_beats", dir="asc"),),
		)
		samples = [
			_make_record(sample_id=1, name="a"),
			_make_record(sample_id=2, name="b"),
			_make_record(sample_id=3, name="c"),
		]
		result = subsample.query.query(spec, samples, beats_resolver=resolver)

		assert [r.sample_id for r in result] == [1, 3]   # 3.75 and 4.25


# ---------------------------------------------------------------------------
# parse_select
# ---------------------------------------------------------------------------

class TestParseSelect:

	def test_single_spec (self) -> None:

		"""A dict is parsed as a single SelectSpec."""

		raw = {"where": {"min_duration": 1.0}, "order_by": "newest", "pick": 2}
		specs = subsample.query.parse_select(raw, "test")
		assert len(specs) == 1
		assert specs[0].where.duration.gte == 1.0
		assert specs[0].order == (subsample.query.OrderClause(by="age", dir="desc"),)
		assert specs[0].pick == 2

	def test_fallback_list (self) -> None:

		"""A list is parsed as a fallback chain of SelectSpecs."""

		raw = [
			{"where": {"name": "my-kick"}},
			{"where": {"reference": "BD0025"}},
		]
		specs = subsample.query.parse_select(raw, "test")
		assert len(specs) == 2
		assert specs[0].where.name == "my-kick"
		assert specs[1].where.reference == "BD0025"

	def test_quantized_beats_yaml_parsed (self) -> None:

		"""min_quantized_beats and max_quantized_beats are parsed as floats."""

		raw = {
			"where": {"min_quantized_beats": 2, "max_quantized_beats": 4.5},
			"order_by": "quantized_beats_desc",
		}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].where.quantized_beats.gte == 2.0
		assert specs[0].where.quantized_beats.lte == 4.5
		assert specs[0].order == (subsample.query.OrderClause(by="quantized_beats", dir="desc"),)

	def test_defaults (self) -> None:

		"""Missing fields get sensible defaults — in particular an empty
		``order`` tuple (query() applies newest-first at evaluation time)."""

		specs = subsample.query.parse_select({}, "test")
		assert specs[0].order == ()
		assert specs[0].pick == 1

	def test_reference_defaults_to_similarity_order (self) -> None:

		"""When reference is in where but no explicit order_by, default to similarity."""

		raw = {"where": {"reference": "BD0025"}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].order == (subsample.query.OrderClause(by="similarity", dir="desc"),)

	def test_path_based_reference_resolves_to_absolute (self) -> None:

		"""Path-based references (containing "/") are resolved to absolute paths."""

		import pathlib
		import tempfile

		with tempfile.TemporaryDirectory() as tmpdir:
			midi_map_dir = pathlib.Path(tmpdir)
			raw = {"where": {"reference": "relative/path/to/ref"}}
			specs = subsample.query.parse_select(raw, "test", midi_map_dir)

			# Should be resolved to an absolute path
			assert specs[0].where.reference.startswith("/")
			assert "relative/path/to/ref" in specs[0].where.reference

	def test_bare_name_reference_preserved (self) -> None:

		"""Bare name references (no "/") are preserved as-is for case-insensitive lookup."""

		raw = {"where": {"reference": "BD0025"}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].where.reference == "BD0025"

	def test_path_based_name_stored_as_stem (self) -> None:

		"""Path-based names store the stem in 'name' and path in 'name_path'."""

		import pathlib
		import tempfile

		with tempfile.TemporaryDirectory() as tmpdir:
			midi_map_dir = pathlib.Path(tmpdir)
			raw = {"where": {"name": "captures/2026-03-27_10-04-07"}}
			specs = subsample.query.parse_select(raw, "test", midi_map_dir)

			# Stem should be extracted
			assert specs[0].where.name == "2026-03-27_10-04-07"
			# Path should be resolved and stored
			assert specs[0].where.name_path is not None
			assert specs[0].where.name_path.startswith("/")
			assert "captures/2026-03-27_10-04-07" in specs[0].where.name_path

	def test_bare_name_has_no_name_path (self) -> None:

		"""Bare names don't set the name_path field."""

		raw = {"where": {"name": "my-sample"}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].where.name == "my-sample"
		assert specs[0].where.name_path is None

	def test_invalid_order_by_raises (self) -> None:

		"""Legacy bare-string form: unknown token must raise with a clear
		message pointing at the offending name."""

		with pytest.raises(ValueError, match="unknown order token"):
			subsample.query.parse_select({"order_by": "bogus"}, "test")

	def test_invalid_order_scorer_raises (self) -> None:

		"""New structured form: unknown 'by' scorer name must raise."""

		with pytest.raises(ValueError, match="unknown order scorer"):
			subsample.query.parse_select(
				{"order": [{"by": "bogus", "dir": "asc"}]}, "test",
			)

	def test_pick_zero_raises (self) -> None:

		with pytest.raises(ValueError, match="pick"):
			subsample.query.parse_select({"pick": 0}, "test")

	def test_invalid_type_raises (self) -> None:

		with pytest.raises(ValueError, match="select"):
			subsample.query.parse_select("not a dict", "test")


# ---------------------------------------------------------------------------
# parse_process
# ---------------------------------------------------------------------------

class TestParseProcess:

	def test_none_returns_empty (self) -> None:
		spec = subsample.query.parse_process(None, "test")
		assert spec.steps == ()

	def test_repitch_true (self) -> None:

		"""repitch: true is a boolean processor with no params."""

		raw = [{"repitch": True}]
		spec = subsample.query.parse_process(raw, "test")
		assert len(spec.steps) == 1
		assert spec.steps[0].name == "repitch"
		assert spec.steps[0].params == ()

	def test_beat_quantize_with_params (self) -> None:
		"""YAML `bpm:` translates to canonical `tempo:` in the step params."""
		raw = [{"stretch_quantize": {"grid": 16, "bpm": 120}}]
		spec = subsample.query.parse_process(raw, "test")
		assert spec.steps[0].name == "stretch_quantize"
		assert spec.steps[0].get("grid") == 16
		assert spec.steps[0].get("tempo") == 120
		assert spec.steps[0].get("bpm") is None  # legacy key was renamed

	def test_multiple_processors (self) -> None:

		raw = [{"stretch_quantize": {"grid": 16}}, {"repitch": True}]
		spec = subsample.query.parse_process(raw, "test")
		assert len(spec.steps) == 2
		assert spec.steps[0].name == "stretch_quantize"
		assert spec.steps[1].name == "repitch"

	def test_has_repitch (self) -> None:
		spec = subsample.query.parse_process([{"repitch": True}], "test")
		assert spec.has_repitch()
		assert not spec.has_stretch_quantize()

	def test_has_beat_quantize (self) -> None:
		spec = subsample.query.parse_process([{"stretch_quantize": {"grid": 8}}], "test")
		assert spec.has_stretch_quantize()
		assert not spec.has_repitch()

	def test_unknown_processor_strict_raises (self) -> None:
		"""Strict mode (default): unknown processor name raises ValueError."""
		subsample.query.set_strict_mode(True)
		try:
			with pytest.raises(ValueError, match="unknown processor 'nonsense'"):
				subsample.query.parse_process([{"nonsense": True}], "test")
		finally:
			subsample.query.set_strict_mode(True)  # restore default

	def test_unknown_processor_lenient_passes (self) -> None:
		"""Lenient mode: unknown processor name parses through; spec_from_process
		will log a warning and skip it at build time."""
		subsample.query.set_strict_mode(False)
		try:
			spec = subsample.query.parse_process([{"nonsense": True}], "test")
			assert len(spec.steps) == 1
			assert spec.steps[0].name == "nonsense"
		finally:
			subsample.query.set_strict_mode(True)

	def test_unknown_processor_bare_string_strict_raises (self) -> None:
		"""Bare-string process entry is also validated."""
		subsample.query.set_strict_mode(True)
		try:
			with pytest.raises(ValueError, match="unknown processor 'nonsense'"):
				subsample.query.parse_process(["nonsense"], "test")
		finally:
			subsample.query.set_strict_mode(True)

	def test_legacy_amount_saturate_renamed_to_drive (self) -> None:
		"""saturate: { amount: 6 } → stored as 'drive' internally."""
		spec = subsample.query.parse_process(
			[{"saturate": {"amount": 6}}], "test",
		)
		assert spec.steps[0].get("drive") == 6
		assert spec.steps[0].get("amount") is None

	def test_legacy_amount_transient_renamed_to_gain (self) -> None:
		"""transient: { amount: -3 } → stored as 'gain' internally."""
		spec = subsample.query.parse_process(
			[{"transient": {"amount": -3}}], "test",
		)
		assert spec.steps[0].get("gain") == -3
		assert spec.steps[0].get("amount") is None

	def test_legacy_amount_beat_quantize_renamed_to_strength (self) -> None:
		spec = subsample.query.parse_process(
			[{"stretch_quantize": {"grid": 16, "amount": 0.5}}], "test",
		)
		assert spec.steps[0].get("strength") == 0.5
		assert spec.steps[0].get("amount") is None

	def test_legacy_amount_pad_quantize_renamed_to_strength (self) -> None:
		spec = subsample.query.parse_process(
			[{"pad_quantize": {"grid": 16, "amount": 0.5}}], "test",
		)
		assert spec.steps[0].get("strength") == 0.5

	def test_legacy_and_new_amount_on_same_step_raises (self) -> None:
		"""Mixing `amount:` and `drive:` on the same saturate step raises —
		same mid-migration guard as the where-predicate legacy shim."""
		with pytest.raises(ValueError, match="duplicate"):
			subsample.query.parse_process(
				[{"saturate": {"amount": 6, "drive": 8}}], "test",
			)

	def test_legacy_bpm_translates_to_tempo (self) -> None:
		"""C2: processor `bpm:` → canonical `tempo:`."""
		spec = subsample.query.parse_process(
			[{"stretch_quantize": {"bpm": 120, "grid": 16}}], "test",
		)
		assert spec.steps[0].get("tempo") == 120
		assert spec.steps[0].get("bpm") is None

	def test_legacy_bpm_and_tempo_on_same_step_raises (self) -> None:
		with pytest.raises(ValueError, match="duplicate"):
			subsample.query.parse_process(
				[{"stretch_quantize": {"bpm": 120, "tempo": 140}}], "test",
			)

	def test_legacy_beat_quantize_name_translates_to_stretch_quantize (self) -> None:
		"""Legacy `beat_quantize:` name is translated to canonical
		`stretch_quantize` at parse time; the resulting step's name is
		the canonical form regardless of which name the user wrote."""
		spec = subsample.query.parse_process(
			[{"beat_quantize": {"grid": 16, "tempo": 120}}], "test",
		)
		assert spec.steps[0].name == "stretch_quantize"
		assert spec.steps[0].get("grid") == 16
		assert spec.steps[0].get("tempo") == 120

	def test_legacy_beat_quantize_bare_string_form (self) -> None:
		"""Bare `- beat_quantize` string also translates."""
		spec = subsample.query.parse_process(["beat_quantize"], "test")
		assert spec.steps[0].name == "stretch_quantize"

	def test_legacy_beat_quantize_true_form (self) -> None:
		"""`beat_quantize: true` also translates."""
		spec = subsample.query.parse_process([{"beat_quantize": True}], "test")
		assert spec.steps[0].name == "stretch_quantize"

	def test_legacy_beat_quantize_chained_param_shim (self) -> None:
		"""Legacy NAME + legacy PARAMS chained through the shim correctly:
		`beat_quantize: { amount: 0.5, bpm: 120 }` →
		`stretch_quantize` with `strength: 0.5, tempo: 120`."""
		spec = subsample.query.parse_process(
			[{"beat_quantize": {"amount": 0.5, "bpm": 120}}], "test",
		)
		assert spec.steps[0].name == "stretch_quantize"
		assert spec.steps[0].get("strength") == 0.5
		assert spec.steps[0].get("tempo") == 120
		assert spec.steps[0].get("amount") is None
		assert spec.steps[0].get("bpm") is None

	def test_legacy_beat_quantize_accepted_under_strict_mode (self) -> None:
		"""Legacy name stays in the strict-mode valid-names whitelist."""
		subsample.query.set_strict_mode(True)
		try:
			spec = subsample.query.parse_process(
				[{"beat_quantize": {"grid": 16}}], "test",
			)
			assert spec.steps[0].name == "stretch_quantize"
		finally:
			subsample.query.set_strict_mode(True)

	def test_has_stretch_quantize_true_for_legacy_yaml (self) -> None:
		"""ProcessSpec.has_stretch_quantize() is True for a spec built
		from legacy YAML (canonicalisation happens at parse time)."""
		spec = subsample.query.parse_process(
			[{"beat_quantize": {"grid": 16}}], "test",
		)
		assert spec.has_stretch_quantize()

	def test_invalid_type_raises (self) -> None:

		with pytest.raises(ValueError, match="process"):
			subsample.query.parse_process("not a list", "test")

	def test_invalid_entry_raises (self) -> None:

		with pytest.raises(ValueError, match="process entry"):
			subsample.query.parse_process([42], "test")


# ---------------------------------------------------------------------------
# ProcessSpec and SelectSpec
# ---------------------------------------------------------------------------

class TestProcessSpec:

	def test_empty_spec (self) -> None:
		spec = subsample.query.ProcessSpec()
		assert not spec.has_repitch()
		assert not spec.has_stretch_quantize()


class TestCcBinding:

	def test_resolve_min (self) -> None:
		"""CC value 0 maps to min_val."""
		b = subsample.query.CcBinding(cc=1, min_val=60.0, max_val=180.0)
		assert b.resolve(0) == 60.0

	def test_resolve_max (self) -> None:
		"""CC value 127 maps to max_val."""
		b = subsample.query.CcBinding(cc=1, min_val=60.0, max_val=180.0)
		assert b.resolve(127) == 180.0

	def test_resolve_midpoint (self) -> None:
		"""CC value 64 maps approximately to midpoint."""
		b = subsample.query.CcBinding(cc=1, min_val=0.0, max_val=1.0)
		assert abs(b.resolve(64) - 64.0 / 127.0) < 1e-6

	def test_default_value_explicit (self) -> None:
		"""Explicit default is returned."""
		b = subsample.query.CcBinding(cc=1, default=0.75)
		assert b.default_value == 0.75

	def test_default_value_midpoint (self) -> None:
		"""No explicit default → midpoint of min/max."""
		b = subsample.query.CcBinding(cc=1, min_val=60.0, max_val=180.0)
		assert b.default_value == 120.0

	def test_defaults (self) -> None:
		"""Default range is 0.0–1.0, omni channel."""
		b = subsample.query.CcBinding(cc=1)
		assert b.min_val == 0.0
		assert b.max_val == 1.0
		assert b.channel is None

	def test_parse_cc_binding_in_process (self) -> None:
		"""A dict param with 'cc' key is parsed as CcBinding."""
		raw = [{"pad_quantize": {"grid": 16, "strength": {"cc": 1, "min": 0.0, "max": 1.0}}}]
		spec = subsample.query.parse_process(raw, "test")
		step = spec.steps[0]
		strength = step.get("strength")
		assert isinstance(strength, subsample.query.CcBinding)
		assert strength.cc == 1
		assert strength.min_val == 0.0
		assert strength.max_val == 1.0

	def test_parse_cc_binding_with_channel (self) -> None:
		"""Channel is parsed when present."""
		raw = [{"pad_quantize": {"strength": {"cc": 1, "channel": 10}}}]
		spec = subsample.query.parse_process(raw, "test")
		strength = spec.steps[0].get("strength")
		assert isinstance(strength, subsample.query.CcBinding)
		assert strength.channel == 10

	def test_parse_scalar_params_unchanged (self) -> None:
		"""Non-dict scalar params remain as plain values (grid is canonical)."""
		raw = [{"pad_quantize": {"grid": 16, "tempo": 120}}]
		spec = subsample.query.parse_process(raw, "test")
		step = spec.steps[0]
		assert step.get("grid") == 16
		assert step.get("tempo") == 120


class TestDirectoryPredicate:

	def test_directory_matches_filepath_under_dir (self) -> None:
		"""Directory predicate matches records whose filepath is under the directory."""
		record = _make_record(name="test")
		import dataclasses, pathlib
		record = dataclasses.replace(record, filepath=pathlib.Path("/samples/captures/test.wav"))
		pred = subsample.query.WherePredicate(directory="/samples/captures")
		assert pred.matches(record)

	def test_directory_rejects_filepath_outside_dir (self) -> None:
		"""Directory predicate rejects records from a different directory."""
		record = _make_record(name="test")
		import dataclasses, pathlib
		record = dataclasses.replace(record, filepath=pathlib.Path("/samples/other/test.wav"))
		pred = subsample.query.WherePredicate(directory="/samples/captures")
		assert not pred.matches(record)

	def test_directory_rejects_prefix_overlap (self) -> None:
		"""Directory predicate must not match dirs that share a prefix (e.g. captures2)."""
		record = _make_record(name="test")
		import dataclasses, pathlib
		record = dataclasses.replace(record, filepath=pathlib.Path("/samples/captures2/test.wav"))
		pred = subsample.query.WherePredicate(directory="/samples/captures")
		assert not pred.matches(record)

	def test_directory_rejects_no_filepath (self) -> None:
		"""Directory predicate rejects records with no filepath."""
		record = _make_record(name="test")
		pred = subsample.query.WherePredicate(directory="/samples/captures")
		assert not pred.matches(record)

	def test_directory_combined_with_other_predicates (self) -> None:
		"""Directory predicate works alongside other predicates."""
		import dataclasses, pathlib
		record = _make_record(name="test", duration=2.0, onset_count=5)
		record = dataclasses.replace(record, filepath=pathlib.Path("/samples/captures/test.wav"))

		# Passes both directory and min_onsets
		pred = subsample.query.WherePredicate(directory="/samples/captures", onsets=subsample.query.Range(gte=3))
		assert pred.matches(record)

		# Passes directory but fails min_onsets
		pred2 = subsample.query.WherePredicate(directory="/samples/captures", onsets=subsample.query.Range(gte=10))
		assert not pred2.matches(record)

	def test_parse_where_directory (self) -> None:
		"""_parse_where correctly parses a directory predicate."""
		import pathlib
		pred = subsample.query._parse_where(
			{"directory": "samples/captures"},
			"test_assignment",
			midi_map_dir=pathlib.Path("/project"),
		)
		assert pred.directory == "/project/samples/captures"

	def test_has_vocoder (self) -> None:
		"""ProcessSpec.has_vocoder() returns True when vocoder step present."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="vocoder", params=(("carrier", "reference"),)),
		))
		assert process.has_vocoder()


class TestSelectSpec:

	def test_defaults (self) -> None:
		spec = subsample.query.SelectSpec()
		# Default is an empty order tuple; query() applies newest-first (age desc)
		# as a fallback when no clauses are given.  Keeping the default empty
		# means "no preference from the user" is distinguishable from an
		# explicitly-written [{by: age, dir: desc}].
		assert spec.order == ()
		assert spec.pick == 1
		assert spec.where == subsample.query.WherePredicate()


# ---------------------------------------------------------------------------
# Range + operator-dict YAML surface
# ---------------------------------------------------------------------------


class TestRange:

	"""Unit tests for the Range helper (AND semantics across all slots)."""

	def test_empty_range_contains_any_value (self) -> None:
		r = subsample.query.Range()
		assert r.is_empty()
		for x in [-1e9, -1.0, 0.0, 1.0, 1e9]:
			assert r.contains(x)

	def test_gte_is_inclusive (self) -> None:
		r = subsample.query.Range(gte=5.0)
		assert     r.contains(5.0)
		assert     r.contains(5.000001)
		assert not r.contains(4.999999)

	def test_lte_is_inclusive (self) -> None:
		r = subsample.query.Range(lte=5.0)
		assert     r.contains(5.0)
		assert     r.contains(4.999999)
		assert not r.contains(5.000001)

	def test_gt_is_strict (self) -> None:
		r = subsample.query.Range(gt=5.0)
		assert not r.contains(5.0)
		assert     r.contains(5.000001)

	def test_lt_is_strict (self) -> None:
		r = subsample.query.Range(lt=5.0)
		assert not r.contains(5.0)
		assert     r.contains(4.999999)

	def test_eq_is_exact (self) -> None:
		r = subsample.query.Range(eq=4.0)
		assert     r.contains(4.0)
		assert not r.contains(3.999)
		assert not r.contains(4.001)

	def test_combined_operators_and_together (self) -> None:
		"""gte + lt composes to a half-open range."""
		r = subsample.query.Range(gte=1.0, lt=5.0)
		assert     r.contains(1.0)
		assert     r.contains(3.0)
		assert not r.contains(0.9)
		assert not r.contains(5.0)

	def test_contradictory_operators_match_nothing (self) -> None:
		"""`{gte: 5, lte: 3}` is intentionally empty — models what was written."""
		r = subsample.query.Range(gte=5.0, lte=3.0)
		assert not r.contains(4.0)
		assert not r.contains(5.0)
		assert not r.contains(3.0)


class TestOperatorDictYaml:

	"""The new per-field `{gte: ..., lte: ...}` YAML form."""

	def test_duration_operator_dict_parsed (self) -> None:
		raw   = {"where": {"duration": {"gte": 1.0, "lt": 5.0}}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].where.duration == subsample.query.Range(gte=1.0, lt=5.0)

	def test_all_five_operators_on_one_field (self) -> None:
		"""All five operators route through the parser together.  (The
		resulting Range would match nothing for most of these combinations;
		the test only checks parser fidelity.)"""
		raw = {"where": {"onsets": {"gte": 1, "lte": 10, "gt": 2, "lt": 9, "eq": 5}}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].where.onsets == subsample.query.Range(
			gte=1.0, lte=10.0, gt=2.0, lt=9.0, eq=5.0,
		)

	def test_scalar_shorthand_is_eq (self) -> None:
		"""Bare scalar under a numeric field is sugar for `{eq: X}`."""
		raw   = {"where": {"quantized_beats": 4}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].where.quantized_beats == subsample.query.Range(eq=4.0)

	def test_pitch_accepts_note_name_under_operator (self) -> None:
		"""String under pitch routes through _note_name_to_hz.  Regression
		guard — the legacy parser supported this but the old test suite
		never covered it."""
		import librosa
		raw   = {"where": {"pitch": {"gte": "C3", "lt": "C6"}}}
		specs = subsample.query.parse_select(raw, "test")
		expected_c3 = float(librosa.midi_to_hz(48))
		expected_c6 = float(librosa.midi_to_hz(84))
		assert specs[0].where.pitch_hz.gte == pytest.approx(expected_c3)
		assert specs[0].where.pitch_hz.lt  == pytest.approx(expected_c6)

	def test_unknown_operator_raises (self) -> None:
		raw = {"where": {"duration": {"between": [1.0, 5.0]}}}
		with pytest.raises(ValueError, match="unknown operator"):
			subsample.query.parse_select(raw, "test")


class TestLegacyWhereShim:

	"""Legacy `min_X`/`max_X` YAML keys keep working."""

	def test_legacy_min_duration_translates_to_gte (self) -> None:
		specs = subsample.query.parse_select(
			{"where": {"min_duration": 1.0}}, "test",
		)
		assert specs[0].where.duration == subsample.query.Range(gte=1.0)

	def test_legacy_max_duration_translates_to_lte (self) -> None:
		specs = subsample.query.parse_select(
			{"where": {"max_duration": 5.0}}, "test",
		)
		assert specs[0].where.duration == subsample.query.Range(lte=5.0)

	def test_legacy_and_new_produce_equal_specs (self) -> None:
		"""Parsing either form must yield an identical SelectSpec — proves
		the shim is a pure translation, not a subtly-different path."""
		legacy = subsample.query.parse_select(
			{"where": {"min_duration": 1.0, "max_duration": 5.0}}, "test",
		)
		new = subsample.query.parse_select(
			{"where": {"duration": {"gte": 1.0, "lte": 5.0}}}, "test",
		)
		assert legacy == new

	def test_legacy_min_pitch_accepts_note_name (self) -> None:
		"""Legacy path still works; regression guard against shim drift."""
		specs = subsample.query.parse_select(
			{"where": {"min_pitch": "A4"}}, "test",
		)
		assert specs[0].where.pitch_hz.gte == pytest.approx(440.0)

	def test_mixing_legacy_and_new_on_same_field_raises (self) -> None:
		"""Mid-migration accident: a user overlays a new-form block on top
		of an old-form block for the same field.  Catch this loudly rather
		than silently picking one form's value."""
		with pytest.raises(ValueError, match="legacy"):
			subsample.query.parse_select(
				{"where": {"min_duration": 1.0, "duration": {"lte": 5.0}}},
				"test",
			)

	def test_mixing_forms_across_different_fields_is_fine (self) -> None:
		"""Gradual migration: legacy on one field, new-form on another.
		This must not raise."""
		specs = subsample.query.parse_select(
			{"where": {"min_duration": 1.0, "onsets": {"gt": 4}}},
			"test",
		)
		assert specs[0].where.duration == subsample.query.Range(gte=1.0)
		assert specs[0].where.onsets   == subsample.query.Range(gt=4.0)


class TestStrictMode:

	"""Strict mode: unknown `where:` keys and processor names raise."""

	def test_unknown_where_key_strict_raises (self) -> None:
		subsample.query.set_strict_mode(True)
		try:
			with pytest.raises(ValueError, match="unknown where-predicate key 'duratoin'"):
				subsample.query.parse_select(
					{"where": {"duratoin": 1.0}}, "test",
				)
		finally:
			subsample.query.set_strict_mode(True)

	def test_unknown_where_key_lenient_warns (self, caplog: typing.Any) -> None:
		subsample.query.set_strict_mode(False)
		try:
			with caplog.at_level("WARNING"):
				specs = subsample.query.parse_select(
					{"where": {"duratoin": 1.0}}, "test",
				)
			assert specs[0].where == subsample.query.WherePredicate()
			assert any("duratoin" in r.message for r in caplog.records)
		finally:
			subsample.query.set_strict_mode(True)

	def test_strict_error_lists_valid_keys (self) -> None:
		subsample.query.set_strict_mode(True)
		try:
			with pytest.raises(ValueError) as exc_info:
				subsample.query.parse_select(
					{"where": {"bogus": 1.0}}, "test",
				)
			msg = str(exc_info.value)
			# A sampling of expected valid keys should appear in the message.
			assert "duration" in msg
			assert "pitched" in msg
			assert "reference" in msg
		finally:
			subsample.query.set_strict_mode(True)

	def test_pitched_must_be_bool (self) -> None:
		"""`pitched:` accepts only True or False literally, not Python-truthy
		strings.  `pitched: yes` in YAML would be truthy today but is rejected."""
		with pytest.raises(ValueError, match="'pitched' must be true or false"):
			subsample.query.parse_select({"where": {"pitched": "yes"}}, "test")

	def test_pitched_int_also_rejected (self) -> None:
		with pytest.raises(ValueError, match="'pitched' must be true or false"):
			subsample.query.parse_select({"where": {"pitched": 1}}, "test")


class TestNamePathSplit:

	"""`where: { path: ... }` — explicit path reference (A8)."""

	def test_explicit_path_populates_name_and_name_path (self, tmp_path: pathlib.Path) -> None:
		"""`path: foo/bar.wav` → name='bar', name_path=<resolved absolute>."""
		specs = subsample.query.parse_select(
			{"where": {"path": "samples/my-kick.wav"}}, "test", tmp_path,
		)
		assert specs[0].where.name == "my-kick"
		assert specs[0].where.name_path == str((tmp_path / "samples/my-kick.wav").resolve())

	def test_legacy_name_as_path_still_works (self, tmp_path: pathlib.Path) -> None:
		"""Legacy `name: path/to/x` auto-detects the path form (indefinite
		back-compat)."""
		specs = subsample.query.parse_select(
			{"where": {"name": "samples/my-kick.wav"}}, "test", tmp_path,
		)
		assert specs[0].where.name == "my-kick"
		assert specs[0].where.name_path == str((tmp_path / "samples/my-kick.wav").resolve())

	def test_bare_name_stays_as_stem (self) -> None:
		"""Bare `name: my-kick` (no slash or leading dot) = stem match."""
		specs = subsample.query.parse_select(
			{"where": {"name": "my-kick"}}, "test",
		)
		assert specs[0].where.name == "my-kick"
		assert specs[0].where.name_path is None

	def test_name_and_path_together_raise (self, tmp_path: pathlib.Path) -> None:
		"""Using both `name:` and `path:` in one where block is rejected."""
		with pytest.raises(ValueError, match="mutually exclusive"):
			subsample.query.parse_select(
				{"where": {"name": "x", "path": "y.wav"}},
				"test", tmp_path,
			)


class TestHpssUnification:

	"""C1: `hpss: { keep: harmonic/percussive }` replaces the two separate
	processor names.  Legacy `hpss_harmonic: true` / `hpss_percussive: true`
	still translate."""

	def test_new_form_harmonic (self) -> None:
		spec = subsample.query.parse_process(
			[{"hpss": {"keep": "harmonic"}}], "test",
		)
		assert spec.steps[0].name == "hpss"
		assert spec.steps[0].get("keep") == "harmonic"

	def test_new_form_percussive (self) -> None:
		spec = subsample.query.parse_process(
			[{"hpss": {"keep": "percussive"}}], "test",
		)
		assert spec.steps[0].name == "hpss"
		assert spec.steps[0].get("keep") == "percussive"

	def test_legacy_hpss_harmonic_translates (self) -> None:
		"""Legacy `hpss_harmonic: true` → canonical `hpss {keep: harmonic}`."""
		spec = subsample.query.parse_process(
			[{"hpss_harmonic": True}], "test",
		)
		assert spec.steps[0].name == "hpss"
		assert spec.steps[0].get("keep") == "harmonic"

	def test_legacy_hpss_percussive_translates (self) -> None:
		spec = subsample.query.parse_process(
			[{"hpss_percussive": True}], "test",
		)
		assert spec.steps[0].name == "hpss"
		assert spec.steps[0].get("keep") == "percussive"

	def test_legacy_bare_string_translates (self) -> None:
		"""Bare-string form `- hpss_harmonic` also translates."""
		spec = subsample.query.parse_process(["hpss_harmonic"], "test")
		assert spec.steps[0].name == "hpss"
		assert spec.steps[0].get("keep") == "harmonic"

	def test_hpss_missing_keep_raises_at_build (self) -> None:
		"""Forgotten `keep:` raises at spec build time, not parse (`hpss:` on
		its own is semantically incomplete but grammatically valid)."""
		import subsample.transform as _transform
		spec = subsample.query.parse_process([{"hpss": {}}], "test")
		assert spec.steps[0].name == "hpss"
		with pytest.raises(ValueError, match="must be 'harmonic' or 'percussive'"):
			_transform.spec_from_process(spec)

	def test_hpss_invalid_keep_raises (self) -> None:
		import subsample.transform as _transform
		spec = subsample.query.parse_process(
			[{"hpss": {"keep": "both"}}], "test",
		)
		with pytest.raises(ValueError, match="must be 'harmonic' or 'percussive'"):
			_transform.spec_from_process(spec)


# ---------------------------------------------------------------------------
# Multi-key ordering, legacy shim, on_missing policy
# ---------------------------------------------------------------------------


class TestMultiKeyOrder:

	"""Composed sort keys — primary wins, secondary breaks ties."""

	def test_primary_and_secondary_compose (self) -> None:

		# Three samples, two share a duration so secondary must break the tie.
		a = _make_record(sample_id=1, duration=2.0, onset_count=5)
		b = _make_record(sample_id=2, duration=2.0, onset_count=3)   # ties a on duration
		c = _make_record(sample_id=3, duration=1.0, onset_count=9)

		spec = subsample.query.SelectSpec(
			order=(
				subsample.query.OrderClause(by="duration", dir="desc"),   # primary: longest first
				subsample.query.OrderClause(by="onsets",   dir="desc"),   # tiebreaker: most onsets first
			),
		)
		ranked = subsample.query.query(spec, [c, b, a])

		# Expected: a and b come first (duration=2.0), tie broken by onsets desc
		# → a (5 onsets) then b (3 onsets); c last (duration=1.0).
		assert [r.sample_id for r in ranked] == [1, 2, 3]

	def test_secondary_only_breaks_ties (self) -> None:

		"""Secondary clause must not re-order records that differ on the primary."""

		a = _make_record(sample_id=1, duration=3.0, onset_count=1)   # would be last on onsets-only
		b = _make_record(sample_id=2, duration=2.0, onset_count=9)

		spec = subsample.query.SelectSpec(
			order=(
				subsample.query.OrderClause(by="duration", dir="desc"),
				subsample.query.OrderClause(by="onsets",   dir="desc"),
			),
		)
		ranked = subsample.query.query(spec, [a, b])

		# Primary wins: a first (longer duration) despite fewer onsets.
		assert [r.sample_id for r in ranked] == [1, 2]


class TestLegacyOrderShim:

	"""Bare-string tokens and the legacy `order_by:` YAML key keep working."""

	def test_bare_string_token_translates_to_clause (self) -> None:

		specs = subsample.query.parse_select({"order_by": "duration_desc"}, "test")
		assert specs[0].order == (
			subsample.query.OrderClause(by="duration", dir="desc"),
		)

	def test_legacy_and_new_key_produce_equal_specs (self) -> None:

		"""Parsing either form must yield an identical SelectSpec — proves
		the shim is a pure translation, not a subtly-different path."""

		legacy = subsample.query.parse_select({"order_by": "loudest"}, "test")
		new    = subsample.query.parse_select(
			{"order": [{"by": "level", "dir": "desc"}]}, "test",
		)
		assert legacy == new

	def test_both_keys_set_raises (self) -> None:

		with pytest.raises(ValueError, match="both 'order' and 'order_by'"):
			subsample.query.parse_select(
				{"order": [{"by": "age", "dir": "desc"}], "order_by": "newest"},
				"test",
			)

	def test_list_can_mix_legacy_and_new_forms (self) -> None:

		"""A user partway through migration may have both token types in
		one list.  Parser accepts both within the same `order:` value."""

		specs = subsample.query.parse_select(
			{"order": ["duration_desc", {"by": "onsets", "dir": "asc"}]},
			"test",
		)
		assert specs[0].order == (
			subsample.query.OrderClause(by="duration", dir="desc"),
			subsample.query.OrderClause(by="onsets",   dir="asc"),
		)


class TestOnMissingPolicy:

	"""Behaviour around scorer.fn returning None — exclude vs sort_last."""

	def test_sort_last_keeps_records_with_none_score_at_end (self) -> None:

		"""quantized_beats is on_missing='sort_last' — non-quantized samples
		keep their place in the result but park at the end regardless of
		direction.  Matches the pre-refactor behaviour."""

		a = _make_record(sample_id=1)
		b = _make_record(sample_id=2)
		c = _make_record(sample_id=3)

		# Resolver returns values for a/b only; c has no quantize profile.
		def _resolver (sample_id: int) -> typing.Optional[float]:
			return {1: 2.0, 2: 4.0}.get(sample_id)

		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(by="quantized_beats", dir="asc"),),
		)
		ranked = subsample.query.query(spec, [c, b, a], beats_resolver=_resolver)
		assert [r.sample_id for r in ranked] == [1, 2, 3]

		spec_desc = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(by="quantized_beats", dir="desc"),),
		)
		ranked_desc = subsample.query.query(spec_desc, [c, b, a], beats_resolver=_resolver)
		# Non-quantized `c` still parks at the end even in desc direction.
		assert [r.sample_id for r in ranked_desc] == [2, 1, 3]

	def test_exclude_drops_records_with_none_score (self) -> None:

		"""A scorer registered with on_missing='exclude' removes None-scoring
		records entirely — this is the policy future quantize_match will
		use.  Register a test-local scorer here so we exercise the exclude
		branch without needing the not-yet-implemented feature."""

		def _only_even (
			record:  subsample.library.SampleRecord,
			_params: tuple[tuple[str, typing.Any], ...],
			_state:  dict[str, typing.Any],
		) -> typing.Optional[float]:
			# Score exists only for even sample_ids; odd ones return None
			# and (under on_missing='exclude') should be filtered out.
			return float(record.sample_id) if record.sample_id % 2 == 0 else None

		try:
			subsample.query._register_scorer("_test_only_even", _only_even, on_missing="exclude")

			a = _make_record(sample_id=1)
			b = _make_record(sample_id=2)
			c = _make_record(sample_id=3)
			d = _make_record(sample_id=4)

			spec = subsample.query.SelectSpec(
				order=(subsample.query.OrderClause(by="_test_only_even", dir="asc"),),
			)
			ranked = subsample.query.query(spec, [a, b, c, d])
			# Only even sample_ids survive the exclude filter.
			assert [r.sample_id for r in ranked] == [2, 4]
		finally:
			subsample.query._SCORERS.pop("_test_only_even", None)


class TestSimilarityClausePosition:

	"""similarity is only supported as the primary clause."""

	def test_similarity_as_secondary_raises (self) -> None:

		a = _make_record(sample_id=1)
		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(reference="foo"),
			order=(
				subsample.query.OrderClause(by="duration",   dir="desc"),
				subsample.query.OrderClause(by="similarity", dir="desc"),
			),
		)
		with pytest.raises(ValueError, match="primary order clause"):
			subsample.query.query(spec, [a])

	def test_similarity_without_reference_raises (self) -> None:

		a = _make_record(sample_id=1)
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(by="similarity", dir="desc"),),
		)
		with pytest.raises(ValueError, match="where.reference"):
			subsample.query.query(spec, [a])


# ---------------------------------------------------------------------------
# beat_match scorer — per-beat energy pattern matching
# ---------------------------------------------------------------------------

class TestBeatMatchHelpers:

	"""Unit tests for the private helpers — no scorer plumbing."""

	def test_downsample_clean_16_slots_at_res_16 (self) -> None:
		"""16 slots @ resolution=16 → 4 beats (4 slots per beat)."""
		energy = [1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 0.0, 0.0,
		          1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 0.0, 0.0]
		beats = subsample.query._downsample_to_beats(energy, resolution=16)
		assert beats == (1.0, 0.0, 1.0, 0.0)

	def test_downsample_8ths_at_res_8 (self) -> None:
		"""8 slots @ resolution=8 → 4 beats (2 slots per beat)."""
		energy = [1.0, 1.0,  0.0, 0.0,  1.0, 1.0,  0.0, 0.0]
		beats = subsample.query._downsample_to_beats(energy, resolution=8)
		assert beats == (1.0, 0.0, 1.0, 0.0)

	def test_downsample_resolution_invariance (self) -> None:
		"""Same musical content at 8ths and 16ths yields same per-beat result."""
		e8  = [1.0, 1.0,  0.0, 0.0,  1.0, 1.0,  0.0, 0.0]
		e16 = [1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 0.0, 0.0,
		       1.0, 1.0, 1.0, 1.0,   0.0, 0.0, 0.0, 0.0]
		assert subsample.query._downsample_to_beats(e8,  resolution=8) \
			== subsample.query._downsample_to_beats(e16, resolution=16)

	def test_downsample_quarter_resolution (self) -> None:
		"""4 slots @ resolution=4 → 4 beats (1 slot per beat, pass-through)."""
		energy = [0.8, 0.2, 0.5, 0.1]
		beats = subsample.query._downsample_to_beats(energy, resolution=4)
		assert beats == (0.8, 0.2, 0.5, 0.1)

	def test_downsample_empty (self) -> None:
		assert subsample.query._downsample_to_beats([], resolution=16) == ()

	def test_cosine_identical_vectors (self) -> None:
		score = subsample.query._cosine_similarity_truncated(
			[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], 4,
		)
		assert score == pytest.approx(1.0)

	def test_cosine_level_insensitive (self) -> None:
		"""Same shape, different scale → cosine = 1.0."""
		score = subsample.query._cosine_similarity_truncated(
			[1.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0], 4,
		)
		assert score == pytest.approx(1.0)

	def test_cosine_orthogonal (self) -> None:
		"""[1,0,1,0] vs [0,1,0,1] are orthogonal → 0.0."""
		score = subsample.query._cosine_similarity_truncated(
			[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], 4,
		)
		assert score == pytest.approx(0.0)

	def test_cosine_zero_magnitude_returns_zero (self) -> None:
		"""Edge case: all-zero input → 0.0 (not NaN)."""
		score = subsample.query._cosine_similarity_truncated(
			[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3,
		)
		assert score == 0.0

	def test_cosine_truncates_to_k (self) -> None:
		"""Only the first k elements are scored."""
		# Identical on first 3, diverges after - with k=3, cosine = 1.0
		score = subsample.query._cosine_similarity_truncated(
			[1.0, 1.0, 1.0, 999.0], [1.0, 1.0, 1.0, 0.0], 3,
		)
		assert score == pytest.approx(1.0)


class TestBeatMatchParseValidation:

	"""Pattern validation at parse time."""

	def test_pattern_accepted_as_list_of_ints (self) -> None:
		specs = subsample.query.parse_select(
			{"order": [{"by": "beat_match", "pattern": [1, 0, 1, 0]}]}, "test",
		)
		# Pattern is coerced to a tuple of floats for hashability.
		clause = specs[0].order[0]
		params = dict(clause.params)
		assert params["pattern"] == (1.0, 0.0, 1.0, 0.0)

	def test_pattern_accepted_as_list_of_floats (self) -> None:
		specs = subsample.query.parse_select(
			{"order": [{"by": "beat_match", "pattern": [1.0, 0.9, 0.8]}]}, "test",
		)
		clause = specs[0].order[0]
		params = dict(clause.params)
		assert params["pattern"] == (1.0, 0.9, 0.8)

	def test_missing_pattern_raises (self) -> None:
		with pytest.raises(ValueError, match="requires a 'pattern'"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match"}]}, "test",
			)

	def test_non_list_pattern_raises (self) -> None:
		with pytest.raises(ValueError, match="must be a list"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match", "pattern": "abc"}]}, "test",
			)

	def test_empty_pattern_raises (self) -> None:
		with pytest.raises(ValueError, match="at least 2 elements"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match", "pattern": []}]}, "test",
			)

	def test_single_element_pattern_raises (self) -> None:
		with pytest.raises(ValueError, match="at least 2 elements"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match", "pattern": [1]}]}, "test",
			)

	def test_non_numeric_pattern_element_raises (self) -> None:
		with pytest.raises(ValueError, match="must be a number"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match", "pattern": [1, "hi", 0]}]}, "test",
			)

	def test_out_of_range_pattern_element_raises (self) -> None:
		with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match", "pattern": [1, 1.5, 0]}]}, "test",
			)

	def test_negative_pattern_element_raises (self) -> None:
		with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
			subsample.query.parse_select(
				{"order": [{"by": "beat_match", "pattern": [1, -0.1]}]}, "test",
			)


class TestBeatMatchScorer:

	"""Scorer behaviour — via query() with synthetic resolver."""

	def _make_resolver (
		self,
		profiles: dict[int, typing.Optional[tuple[float, ...]]],
		resolution: int = 16,
	) -> typing.Callable[[int], typing.Any]:
		"""Build a resolver that returns pre-baked GridEnergyProfiles."""
		def _resolver (sample_id: int) -> typing.Any:
			energy = profiles.get(sample_id)
			if energy is None:
				return None
			import subsample.transform as _t
			return _t.GridEnergyProfile(bpm=120.0, resolution=resolution, energy=energy)
		return _resolver

	def test_binary_pattern_ranks_by_shape (self) -> None:
		"""Pattern [1,0,1,0] → sample matching that shape wins."""
		samples = [
			_make_record(sample_id=1, name="even-beats"),
			_make_record(sample_id=2, name="odd-beats"),
			_make_record(sample_id=3, name="all-loud"),
		]
		# 4 beats @ resolution=4 → 1 slot per beat for clean testing
		resolver = self._make_resolver({
			1: (1.0, 0.0, 1.0, 0.0),  # matches [1,0,1,0]
			2: (0.0, 1.0, 0.0, 1.0),  # matches [0,1,0,1]
			3: (1.0, 1.0, 1.0, 1.0),  # uniform
		}, resolution=4)
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)
		result = subsample.query.query(
			spec, samples, energy_profile_resolver=resolver,
		)
		assert result[0].sample_id == 1  # even-beats wins

	def test_samples_without_profile_excluded (self) -> None:
		"""on_missing='exclude': samples with no profile are dropped."""
		samples = [
			_make_record(sample_id=1, name="has-profile"),
			_make_record(sample_id=2, name="no-profile"),
		]
		resolver = self._make_resolver({
			1: (1.0, 0.0, 1.0, 0.0),
			# sample 2 → resolver returns None → excluded
		}, resolution=4)
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)
		result = subsample.query.query(
			spec, samples, energy_profile_resolver=resolver,
		)
		ids = [r.sample_id for r in result]
		assert ids == [1]  # sample 2 excluded

	def test_no_resolver_excludes_all (self) -> None:
		"""Without an energy_profile_resolver, no sample can score →
		all excluded → empty result."""
		samples = [_make_record(sample_id=1)]
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)
		result = subsample.query.query(spec, samples)
		assert result == []

	def test_dir_asc_puts_worst_match_first (self) -> None:
		samples = [
			_make_record(sample_id=1, name="even-match"),
			_make_record(sample_id=2, name="odd-match"),
		]
		resolver = self._make_resolver({
			1: (1.0, 0.0, 1.0, 0.0),
			2: (0.0, 1.0, 0.0, 1.0),
		}, resolution=4)
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="asc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)
		result = subsample.query.query(
			spec, samples, energy_profile_resolver=resolver,
		)
		# asc: orthogonal (score 0) first, then matching (score 1)
		assert result[0].sample_id == 2

	def test_pattern_longer_than_profile_lhs_truncates (self) -> None:
		"""Pattern of 6 beats, profile of 4 beats → compare first 4."""
		samples = [_make_record(sample_id=1)]
		resolver = self._make_resolver({
			1: (1.0, 0.0, 1.0, 0.0),  # 4 beats
		}, resolution=4)
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)),),
			),),
		)
		result = subsample.query.query(
			spec, samples, energy_profile_resolver=resolver,
		)
		# Sample not excluded — pattern was truncated to 4, matches fully.
		assert len(result) == 1

	def test_pattern_shorter_than_profile_lhs_truncates (self) -> None:
		"""Pattern of 2 beats, profile of 4 beats → compare first 2."""
		samples = [_make_record(sample_id=1)]
		resolver = self._make_resolver({
			1: (1.0, 0.0, 999.0, 999.0),  # first 2 match [1,0]; rest garbage
		}, resolution=4)
		spec = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0)),),
			),),
		)
		result = subsample.query.query(
			spec, samples, energy_profile_resolver=resolver,
		)
		# Only first 2 beats compared → perfect match → sample included
		assert len(result) == 1
