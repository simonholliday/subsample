"""Tests for subsample/query.py — sample query engine and MIDI map parser."""

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
		pred = subsample.query.WherePredicate(min_duration=2.0)
		assert not pred.matches(_make_record(duration=1.5))
		assert pred.matches(_make_record(duration=2.5))

	def test_max_duration (self) -> None:
		pred = subsample.query.WherePredicate(max_duration=1.0)
		assert pred.matches(_make_record(duration=0.5))
		assert not pred.matches(_make_record(duration=1.5))

	def test_min_onsets (self) -> None:
		pred = subsample.query.WherePredicate(min_onsets=4)
		assert not pred.matches(_make_record(onset_count=2))
		assert pred.matches(_make_record(onset_count=5))

	def test_max_onsets (self) -> None:
		pred = subsample.query.WherePredicate(max_onsets=3)
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
		pred = subsample.query.WherePredicate(min_tempo=100.0)
		assert not pred.matches(_make_record(tempo_bpm=80.0))
		assert pred.matches(_make_record(tempo_bpm=120.0))

	def test_max_tempo (self) -> None:
		pred = subsample.query.WherePredicate(max_tempo=130.0)
		assert pred.matches(_make_record(tempo_bpm=120.0))
		assert not pred.matches(_make_record(tempo_bpm=140.0))

	def test_min_pitch_hz (self) -> None:
		pred = subsample.query.WherePredicate(min_pitch_hz=300.0)
		assert not pred.matches(_make_record(dominant_pitch_hz=200.0))
		assert pred.matches(_make_record(dominant_pitch_hz=440.0))

	def test_max_pitch_hz (self) -> None:
		pred = subsample.query.WherePredicate(max_pitch_hz=500.0)
		assert pred.matches(_make_record(dominant_pitch_hz=440.0))
		assert not pred.matches(_make_record(dominant_pitch_hz=880.0))

	def test_name (self) -> None:
		pred = subsample.query.WherePredicate(name="my-kick")
		assert pred.matches(_make_record(name="my-kick"))
		assert not pred.matches(_make_record(name="other"))

	def test_combined_predicates (self) -> None:

		"""Multiple predicates are AND-ed: all must pass."""

		pred = subsample.query.WherePredicate(min_duration=1.0, min_onsets=4)
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
			where=subsample.query.WherePredicate(min_duration=1.0),
		)
		result = subsample.query.query(spec, self._samples())
		assert all(r.duration >= 1.0 for r in result)
		assert len(result) == 3

	def test_order_by_duration_desc (self) -> None:
		spec = subsample.query.SelectSpec(order_by="duration_desc")
		result = subsample.query.query(spec, self._samples())
		assert result[0].name == "long-quiet"
		assert result[1].name == "long-rhythmic"

	def test_order_by_oldest (self) -> None:
		spec = subsample.query.SelectSpec(order_by="oldest")
		result = subsample.query.query(spec, self._samples())
		assert result[0].sample_id == 1

	def test_order_by_loudest (self) -> None:
		spec = subsample.query.SelectSpec(order_by="loudest")
		result = subsample.query.query(spec, self._samples())
		assert result[0].name == "short-loud"

	def test_combined_filter_and_order (self) -> None:

		"""Longer than 1s, ordered by duration descending."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(min_duration=1.0),
			order_by="duration_desc",
		)
		result = subsample.query.query(spec, self._samples())
		assert len(result) == 3
		assert result[0].name == "long-quiet"

	def test_pick_nth (self) -> None:

		"""Pick selects the Nth position (1-indexed) after filtering and ordering."""

		spec = subsample.query.SelectSpec(order_by="duration_desc", pick=3)
		result = subsample.query.query(spec, self._samples())

		# pick is handled by the caller, not by query() — query returns the full ranked list.
		# The caller does: result[pick - 1] if pick <= len(result).
		assert len(result) == 5
		assert result[2].name == "medium-tonal"  # 3rd longest

	def test_empty_result (self) -> None:

		"""Filters that match nothing return an empty list."""

		spec = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(min_duration=100.0),
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
# parse_select
# ---------------------------------------------------------------------------

class TestParseSelect:

	def test_single_spec (self) -> None:

		"""A dict is parsed as a single SelectSpec."""

		raw = {"where": {"min_duration": 1.0}, "order_by": "newest", "pick": 2}
		specs = subsample.query.parse_select(raw, "test")
		assert len(specs) == 1
		assert specs[0].where.min_duration == 1.0
		assert specs[0].order_by == "newest"
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

	def test_defaults (self) -> None:

		"""Missing fields get sensible defaults."""

		specs = subsample.query.parse_select({}, "test")
		assert specs[0].order_by == "newest"
		assert specs[0].pick == 1

	def test_reference_defaults_to_similarity_order (self) -> None:

		"""When reference is in where but no explicit order_by, default to similarity."""

		raw = {"where": {"reference": "BD0025"}}
		specs = subsample.query.parse_select(raw, "test")
		assert specs[0].order_by == "similarity"

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

		with pytest.raises(ValueError, match="order_by"):
			subsample.query.parse_select({"order_by": "bogus"}, "test")

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

		raw = [{"beat_quantize": {"grid": 16, "bpm": 120}}]
		spec = subsample.query.parse_process(raw, "test")
		assert spec.steps[0].name == "beat_quantize"
		assert spec.steps[0].get("grid") == 16
		assert spec.steps[0].get("bpm") == 120

	def test_multiple_processors (self) -> None:

		raw = [{"beat_quantize": {"grid": 16}}, {"repitch": True}]
		spec = subsample.query.parse_process(raw, "test")
		assert len(spec.steps) == 2
		assert spec.steps[0].name == "beat_quantize"
		assert spec.steps[1].name == "repitch"

	def test_has_repitch (self) -> None:
		spec = subsample.query.parse_process([{"repitch": True}], "test")
		assert spec.has_repitch()
		assert not spec.has_beat_quantize()

	def test_has_beat_quantize (self) -> None:
		spec = subsample.query.parse_process([{"beat_quantize": {"grid": 8}}], "test")
		assert spec.has_beat_quantize()
		assert not spec.has_repitch()

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
		assert not spec.has_beat_quantize()


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
		pred = subsample.query.WherePredicate(directory="/samples/captures", min_onsets=3)
		assert pred.matches(record)

		# Passes directory but fails min_onsets
		pred2 = subsample.query.WherePredicate(directory="/samples/captures", min_onsets=10)
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
		assert spec.order_by == "newest"
		assert spec.pick == 1
		assert spec.where == subsample.query.WherePredicate()
