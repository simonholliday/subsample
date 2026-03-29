"""Tests for subsample/similarity.py — composite feature vector similarity scoring."""

import dataclasses
import threading

import numpy
import pytest

import subsample.analysis
import subsample.config
import subsample.library
import subsample.similarity

import tests.helpers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CFG = subsample.config.SimilarityConfig()

# A SimilarityConfig that uses only the spectral group — makes some tests
# easier to reason about by isolating one group at a time.
_SPECTRAL_ONLY_CFG = subsample.config.SimilarityConfig(
	weight_spectral     = 1.0,
	weight_timbre       = 0.0,
	weight_timbre_delta = 0.0,
	weight_timbre_onset = 0.0,
	weight_band_energy  = 0.0,
)

# A SimilarityConfig that uses only the timbre groups.
_TIMBRE_ONLY_CFG = subsample.config.SimilarityConfig(
	weight_spectral     = 0.0,
	weight_timbre       = 1.0,
	weight_timbre_delta = 0.5,
	weight_timbre_onset = 1.0,
	weight_band_energy  = 0.0,
)


def _make_spectral (**overrides: float) -> subsample.analysis.AnalysisResult:

	"""Return an AnalysisResult with all fields set to 0.5 unless overridden."""

	defaults = dict(
		spectral_flatness=0.5, attack=0.5, release=0.5,
		spectral_centroid=0.5, spectral_bandwidth=0.5,
		zcr=0.5, harmonic_ratio=0.5, spectral_contrast=0.5, voiced_fraction=0.5,
		log_attack_time=0.5, spectral_flux=0.5,
	)
	defaults.update(overrides)
	return subsample.analysis.AnalysisResult(**defaults)


def _make_timbre (
	mfcc:        tuple[float, ...] = tuple(1.0 for _ in range(13)),
	mfcc_delta:  tuple[float, ...] = tuple(1.0 for _ in range(13)),
	mfcc_onset:  tuple[float, ...] = tuple(1.0 for _ in range(13)),
) -> subsample.analysis.TimbreResult:

	"""Return a TimbreResult with all coefficients set to 1.0 unless overridden.

	Non-zero defaults are important: zero MFCCs L2-normalise to zero, making
	the timbre groups contribute nothing to similarity scores.
	"""

	return subsample.analysis.TimbreResult(
		mfcc        = mfcc,
		mfcc_delta  = mfcc_delta,
		mfcc_onset  = mfcc_onset,
	)


def _make_record (
	name:     str,
	spectral: subsample.analysis.AnalysisResult,
	timbre:   subsample.analysis.TimbreResult | None = None,
) -> subsample.library.SampleRecord:

	"""Return a minimal SampleRecord wrapping the given spectral (and optional timbre) result."""

	rhythm = subsample.analysis.RhythmResult(
		tempo_bpm=120.0,
		beat_times=(),
		pulse_curve=numpy.array([], dtype=numpy.float32),
		pulse_peak_times=(),
		onset_times=(),
		attack_times=(),
		onset_count=0,
	)
	pitch = subsample.analysis.PitchResult(
		dominant_pitch_hz=0.0,
		pitch_confidence=0.0,
		chroma_profile=tuple(0.0 for _ in range(12)),
		dominant_pitch_class=-1,
		pitch_stability=0.0,
		voiced_frame_count=0,
	)
	params = subsample.analysis.compute_params(44100)

	return subsample.library.SampleRecord(
		sample_id   = subsample.library.allocate_id(),
		name        = name,
		spectral    = spectral,
		rhythm      = rhythm,
		pitch       = pitch,
		timbre      = timbre if timbre is not None else _make_timbre(),
		level       = subsample.analysis.LevelResult(peak=0.5, rms=0.2),
		band_energy = tests.helpers._make_band_energy(),
		params      = params,
		duration    = 1.0,
	)


def _library_with (*records: subsample.library.SampleRecord) -> subsample.library.ReferenceLibrary:

	"""Return a ReferenceLibrary containing the given records."""

	return subsample.library.ReferenceLibrary(list(records))


# ---------------------------------------------------------------------------
# TestAsVector — as_vector() on AnalysisResult is unchanged (9-element contract)
# ---------------------------------------------------------------------------

class TestAsVector:

	def test_as_vector_length (self) -> None:
		result = _make_spectral()
		assert len(result.as_vector()) == 9

	def test_as_vector_dtype (self) -> None:
		result = _make_spectral()
		assert result.as_vector().dtype == numpy.float32

	def test_as_vector_values_match_fields (self) -> None:
		result = _make_spectral(spectral_flatness=0.1, attack=0.9, voiced_fraction=0.3)
		v = result.as_vector()
		assert v[0] == pytest.approx(0.1)
		assert v[1] == pytest.approx(0.9)
		assert v[8] == pytest.approx(0.3)

	def test_as_vector_all_values_in_range (self) -> None:
		result = _make_spectral()
		v = result.as_vector()
		assert numpy.all(v >= 0.0)
		assert numpy.all(v <= 1.0)


# ---------------------------------------------------------------------------
# TestBuildFeatureVector
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:

	def test_full_vector_length (self) -> None:
		# Default config: all 5 groups active → 11 + 12 + 12 + 12 + 8 = 55
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, _DEFAULT_CFG)
		assert len(v) == 55

	def test_spectral_only_length (self) -> None:
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, _SPECTRAL_ONLY_CFG)
		assert len(v) == 11

	def test_timbre_only_length (self) -> None:
		# weight_timbre=1, weight_timbre_delta=0.5, weight_timbre_onset=1 → 12+12+12=36
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, _TIMBRE_ONLY_CFG)
		assert len(v) == 36

	def test_all_weights_zero_returns_empty (self) -> None:
		cfg = subsample.config.SimilarityConfig(
			weight_spectral     = 0.0,
			weight_timbre       = 0.0,
			weight_timbre_delta = 0.0,
			weight_timbre_onset = 0.0,
			weight_band_energy  = 0.0,
		)
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, cfg)
		assert len(v) == 0

	def test_dtype_is_float32 (self) -> None:
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, _DEFAULT_CFG)
		assert v.dtype == numpy.float32

	def test_spectral_includes_log_attack_and_flux (self) -> None:
		# log_attack_time and spectral_flux are NOT in as_vector() but ARE in the
		# spectral group of _build_feature_vector — verify via distinct values
		record = _make_record("X", _make_spectral(
			log_attack_time=0.1, spectral_flux=0.9,
		))
		v = subsample.similarity._build_feature_vector(record, _SPECTRAL_ONLY_CFG)
		# The 11-element spectral group contains the L2-normalised vector;
		# just verify the values are not all the same (i.e. the two new fields are in)
		assert not numpy.all(v == v[0])

	def test_mfcc_coeff_zero_excluded (self) -> None:
		# Coefficient 0 is log-energy and is excluded. If coeff 0 were included
		# each MFCC group would have 13 elements; with it excluded they have 12.
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, _TIMBRE_ONLY_CFG)
		# 3 groups × 12 coeff = 36 (not 39)
		assert len(v) == 36

	def test_zero_mfcc_produces_zero_contribution (self) -> None:
		# All-zero MFCC should L2-normalise to zero, not NaN
		timbre_zero = _make_timbre(
			mfcc       = tuple(0.0 for _ in range(13)),
			mfcc_delta = tuple(0.0 for _ in range(13)),
			mfcc_onset = tuple(0.0 for _ in range(13)),
		)
		record = _make_record("X", _make_spectral(), timbre=timbre_zero)
		v = subsample.similarity._build_feature_vector(record, _TIMBRE_ONLY_CFG)
		assert not numpy.any(numpy.isnan(v))
		assert numpy.all(v == 0.0)

	def test_identical_records_produce_same_vector (self) -> None:
		spectral = _make_spectral(attack=0.3, spectral_flatness=0.7)
		r1 = _make_record("A", spectral)
		r2 = _make_record("B", spectral)
		v1 = subsample.similarity._build_feature_vector(r1, _DEFAULT_CFG)
		v2 = subsample.similarity._build_feature_vector(r2, _DEFAULT_CFG)
		numpy.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# TestCosineInternals
# ---------------------------------------------------------------------------

class TestCosineInternals:

	def test_identical_vectors_return_1 (self) -> None:
		a = numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32)
		assert subsample.similarity._cosine_similarity(a, a) == pytest.approx(1.0)

	def test_orthogonal_vectors_return_0 (self) -> None:
		a = numpy.array([1.0, 0.0], dtype=numpy.float32)
		b = numpy.array([0.0, 1.0], dtype=numpy.float32)
		assert subsample.similarity._cosine_similarity(a, b) == pytest.approx(0.0)

	def test_zero_vector_returns_0 (self) -> None:
		a = numpy.zeros(9, dtype=numpy.float32)
		b = numpy.array([0.5] * 9, dtype=numpy.float32)
		assert subsample.similarity._cosine_similarity(a, b) == pytest.approx(0.0)

	def test_result_is_in_range (self) -> None:
		a = numpy.array([0.2, 0.8, 0.5], dtype=numpy.float32)
		b = numpy.array([0.6, 0.4, 0.9], dtype=numpy.float32)
		score = subsample.similarity._cosine_similarity(a, b)
		assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestL2Normalize
# ---------------------------------------------------------------------------

class TestL2Normalize:

	def test_unit_vector_unchanged (self) -> None:
		v = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
		result = subsample.similarity._l2_normalize(v)
		assert numpy.linalg.norm(result) == pytest.approx(1.0)

	def test_zero_vector_returns_zeros (self) -> None:
		v = numpy.zeros(5, dtype=numpy.float32)
		result = subsample.similarity._l2_normalize(v)
		assert numpy.all(result == 0.0)
		assert not numpy.any(numpy.isnan(result))

	def test_normalised_vector_has_unit_norm (self) -> None:
		v = numpy.array([3.0, 4.0], dtype=numpy.float32)
		result = subsample.similarity._l2_normalize(v)
		assert numpy.linalg.norm(result) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestScoreAgainstLibrary
# ---------------------------------------------------------------------------

class TestScoreAgainstLibrary:

	def test_empty_library_returns_empty (self) -> None:
		lib = _library_with()
		scores = subsample.similarity.score_against_library(
			_make_record("Q", _make_spectral()), lib, _DEFAULT_CFG,
		)
		assert scores == []

	def test_identical_record_scores_1 (self) -> None:
		spectral = _make_spectral()
		timbre   = _make_timbre()
		record   = _make_record("KICK", spectral, timbre)
		lib      = _library_with(record)
		scores   = subsample.similarity.score_against_library(record, lib, _DEFAULT_CFG)
		assert len(scores) == 1
		assert scores[0].score == pytest.approx(1.0)

	def test_partial_match_in_range (self) -> None:
		query = _make_record("Q", _make_spectral(spectral_flatness=0.8, attack=0.2))
		ref   = _make_record("R", _make_spectral(spectral_flatness=0.6, attack=0.4))
		lib   = _library_with(ref)
		scores = subsample.similarity.score_against_library(query, lib, _DEFAULT_CFG)
		assert 0.0 < scores[0].score < 1.0

	def test_returns_sorted_descending (self) -> None:
		query = _make_record("Q",     _make_spectral(spectral_flatness=0.9))
		kick  = _make_record("KICK",  _make_spectral(spectral_flatness=0.9))
		snare = _make_record("SNARE", _make_spectral(spectral_flatness=0.1))
		lib   = _library_with(snare, kick)
		scores = subsample.similarity.score_against_library(query, lib, _DEFAULT_CFG)
		assert scores[0].name == "KICK"
		assert scores[0].score >= scores[1].score

	def test_name_preserved_in_result (self) -> None:
		lib = _library_with(_make_record("BD0025", _make_spectral()))
		scores = subsample.similarity.score_against_library(
			_make_record("Q", _make_spectral()), lib, _DEFAULT_CFG,
		)
		assert scores[0].name == "BD0025"

	def test_score_is_frozen (self) -> None:
		score = subsample.similarity.SimilarityScore(name="KICK", score=0.9)
		with pytest.raises(dataclasses.FrozenInstanceError):
			score.name = "SNARE"  # type: ignore[misc]

	def test_timbre_affects_score (self) -> None:
		# Two records with identical spectral fingerprints but very different MFCCs.
		# Using full config, the timbre difference should lower their mutual score
		# below 1.0 (unlike spectral-only config where they would score 1.0).
		spectral = _make_spectral()
		timbre_a = _make_timbre(mfcc=tuple( 1.0 if i == 1 else 0.0 for i in range(13)))
		timbre_b = _make_timbre(mfcc=tuple(-1.0 if i == 1 else 0.0 for i in range(13)))
		rec_a = _make_record("A", spectral, timbre=timbre_a)
		rec_b = _make_record("B", spectral, timbre=timbre_b)
		lib   = _library_with(rec_a)

		score_full     = subsample.similarity.score_against_library(rec_b, lib, _DEFAULT_CFG)[0].score
		score_spectral = subsample.similarity.score_against_library(rec_b, lib, _SPECTRAL_ONLY_CFG)[0].score

		# Spectral-only: identical fingerprints → 1.0
		assert score_spectral == pytest.approx(1.0)
		# Full: different MFCCs pull the score below 1.0
		assert score_full < 1.0


# ---------------------------------------------------------------------------
# TestFormatSimilarityScores
# ---------------------------------------------------------------------------

class TestFormatSimilarityScores:

	def test_format_empty_returns_empty_string (self) -> None:
		assert subsample.similarity.format_similarity_scores([]) == ""

	def test_format_single_score (self) -> None:
		scores = [subsample.similarity.SimilarityScore(name="KICK", score=0.94)]
		result = subsample.similarity.format_similarity_scores(scores)
		assert "KICK" in result
		assert "0.94" in result

	def test_format_multiple_scores_separated (self) -> None:
		scores = [
			subsample.similarity.SimilarityScore(name="KICK",  score=0.94),
			subsample.similarity.SimilarityScore(name="SNARE", score=0.61),
			subsample.similarity.SimilarityScore(name="HAT",   score=0.22),
		]
		result = subsample.similarity.format_similarity_scores(scores)
		assert "KICK"  in result
		assert "SNARE" in result
		assert "HAT"   in result
		# KICK should appear before SNARE (preserves input order)
		assert result.index("KICK") < result.index("SNARE")


# ---------------------------------------------------------------------------
# TestSimilarityMatrix
# ---------------------------------------------------------------------------

class TestSimilarityMatrix:

	# ----- helpers ----------------------------------------------------------

	def _matrix (self, *ref_names: str) -> subsample.similarity.SimilarityMatrix:
		"""Return a SimilarityMatrix with the given reference sample names (all at 0.5)."""
		refs = [_make_record(n, _make_spectral()) for n in ref_names]
		return subsample.similarity.SimilarityMatrix(_library_with(*refs), _DEFAULT_CFG)

	# ----- construction -----------------------------------------------------

	def test_empty_reference_library (self) -> None:
		matrix = self._matrix()
		assert len(matrix) == 0
		assert matrix.get_match("BD", 0) is None

	def test_repr_reflects_sizes (self) -> None:
		matrix = self._matrix("BD", "SN")
		r = _make_record("A", _make_spectral())
		matrix.add(r)
		assert "2" in repr(matrix)  # 2 refs
		assert "1" in repr(matrix)  # 1 instrument

	# ----- add / get_match --------------------------------------------------

	def test_single_add_is_retrievable (self) -> None:
		matrix = self._matrix("BD")
		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)
		assert matrix.get_match("BD", 0) == inst.sample_id

	def test_get_match_case_insensitive (self) -> None:
		matrix = self._matrix("BD")
		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)
		assert matrix.get_match("bd", 0) == inst.sample_id
		assert matrix.get_match("Bd", 0) == inst.sample_id

	def test_get_match_unknown_reference_returns_none (self) -> None:
		matrix = self._matrix("BD")
		assert matrix.get_match("UNKNOWN", 0) is None

	def test_get_match_out_of_bounds_returns_none (self) -> None:
		matrix = self._matrix("BD")
		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)
		assert matrix.get_match("BD", 1) is None  # only rank 0 exists

	def test_get_match_empty_library_returns_none (self) -> None:
		matrix = self._matrix("BD")
		assert matrix.get_match("BD", 0) is None

	# ----- ranking order ----------------------------------------------------

	def test_ranking_order_most_similar_first (self) -> None:
		# BD ref has high spectral_flatness. i_high matches it; i_low does not.
		bd_ref = _make_record("BD",   _make_spectral(spectral_flatness=0.9))
		i_high = _make_record("HIGH", _make_spectral(spectral_flatness=0.9))
		i_low  = _make_record("LOW",  _make_spectral(spectral_flatness=0.1))
		matrix = subsample.similarity.SimilarityMatrix(
			_library_with(bd_ref), _SPECTRAL_ONLY_CFG,
		)
		matrix.add(i_low)
		matrix.add(i_high)
		assert matrix.get_match("BD", 0) == i_high.sample_id
		assert matrix.get_match("BD", 1) == i_low.sample_id

	def test_identical_fingerprint_scores_highest (self) -> None:
		spectral = _make_spectral(spectral_flatness=0.8, attack=0.2)
		ref   = _make_record("BD", spectral)
		inst  = _make_record("I1", spectral)
		other = _make_record("I2", _make_spectral(spectral_flatness=0.0))
		matrix = subsample.similarity.SimilarityMatrix(
			_library_with(ref), _SPECTRAL_ONLY_CFG,
		)
		matrix.add(other)
		matrix.add(inst)
		assert matrix.get_match("BD", 0) == inst.sample_id

	def test_timbre_affects_ranking (self) -> None:
		# Two instruments identical in spectral shape but different in timbre.
		# The one whose timbre matches the reference should rank higher.
		ref_timbre  = _make_timbre(mfcc=tuple(1.0 if i == 1 else 0.0 for i in range(13)))
		inst_timbre = _make_timbre(mfcc=tuple(1.0 if i == 1 else 0.0 for i in range(13)))
		diff_timbre = _make_timbre(mfcc=tuple(1.0 if i == 2 else 0.0 for i in range(13)))

		spectral = _make_spectral()
		ref   = _make_record("BD",  spectral, timbre=ref_timbre)
		match = _make_record("M",   spectral, timbre=inst_timbre)
		other = _make_record("O",   spectral, timbre=diff_timbre)

		matrix = subsample.similarity.SimilarityMatrix(
			_library_with(ref), _TIMBRE_ONLY_CFG,
		)
		matrix.add(other)
		matrix.add(match)
		assert matrix.get_match("BD", 0) == match.sample_id

	# ----- bulk_add ---------------------------------------------------------

	def test_bulk_add_populates_rankings (self) -> None:
		bd_ref = _make_record("BD", _make_spectral(spectral_flatness=0.9))
		i1 = _make_record("I1", _make_spectral(spectral_flatness=0.9))
		i2 = _make_record("I2", _make_spectral(spectral_flatness=0.1))
		matrix = subsample.similarity.SimilarityMatrix(
			_library_with(bd_ref), _SPECTRAL_ONLY_CFG,
		)
		matrix.bulk_add([i1, i2])
		assert matrix.get_match("BD", 0) == i1.sample_id
		assert matrix.get_match("BD", 1) == i2.sample_id

	def test_bulk_add_empty_list_is_noop (self) -> None:
		matrix = self._matrix("BD")
		matrix.bulk_add([])
		assert len(matrix) == 0

	def test_bulk_add_matches_incremental_add_order (self) -> None:
		# bulk_add and successive add() should produce the same ranking
		bd_ref = _make_record("BD", _make_spectral(spectral_flatness=0.9))
		i1 = _make_record("A", _make_spectral(spectral_flatness=0.9))
		i2 = _make_record("B", _make_spectral(spectral_flatness=0.5))
		i3 = _make_record("C", _make_spectral(spectral_flatness=0.1))

		lib = _library_with(bd_ref)
		m_bulk = subsample.similarity.SimilarityMatrix(lib, _SPECTRAL_ONLY_CFG)
		m_bulk.bulk_add([i1, i2, i3])

		m_incr = subsample.similarity.SimilarityMatrix(lib, _SPECTRAL_ONLY_CFG)
		m_incr.add(i1)
		m_incr.add(i2)
		m_incr.add(i3)

		assert m_bulk.get_match("BD", 0) == m_incr.get_match("BD", 0)
		assert m_bulk.get_match("BD", 1) == m_incr.get_match("BD", 1)
		assert m_bulk.get_match("BD", 2) == m_incr.get_match("BD", 2)

	# ----- remove (eviction) ------------------------------------------------

	def test_remove_clears_from_rankings (self) -> None:
		matrix = self._matrix("BD")
		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)
		matrix.remove([inst.sample_id])
		assert matrix.get_match("BD", 0) is None
		assert len(matrix) == 0

	def test_remove_unknown_id_is_noop (self) -> None:
		matrix = self._matrix("BD")
		matrix.remove([99999])  # should not raise

	def test_remove_updates_rankings_for_all_references (self) -> None:
		refs = [_make_record("BD", _make_spectral()), _make_record("SN", _make_spectral())]
		matrix = subsample.similarity.SimilarityMatrix(_library_with(*refs), _DEFAULT_CFG)
		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)
		matrix.remove([inst.sample_id])
		assert matrix.get_match("BD", 0) is None
		assert matrix.get_match("SN", 0) is None

	def test_remove_preserves_remaining_instruments (self) -> None:
		matrix = self._matrix("BD")
		i1 = _make_record("I1", _make_spectral(spectral_flatness=0.9))
		i2 = _make_record("I2", _make_spectral(spectral_flatness=0.1))
		matrix.add(i1)
		matrix.add(i2)
		matrix.remove([i1.sample_id])
		assert matrix.get_match("BD", 0) == i2.sample_id
		assert matrix.get_match("BD", 1) is None

	# ----- get_matches / get_scores -----------------------------------------

	def test_get_matches_returns_all (self) -> None:
		matrix = self._matrix("BD")
		for i in range(3):
			matrix.add(_make_record(f"I{i}", _make_spectral()))
		assert len(matrix.get_matches("BD")) == 3

	def test_get_matches_limit (self) -> None:
		matrix = self._matrix("BD")
		for i in range(5):
			matrix.add(_make_record(f"I{i}", _make_spectral()))
		assert len(matrix.get_matches("BD", limit=2)) == 2

	def test_get_matches_unknown_reference_returns_empty (self) -> None:
		matrix = self._matrix("BD")
		assert matrix.get_matches("UNKNOWN") == []

	def test_get_scores_returns_scores_for_all_references (self) -> None:
		refs = [_make_record("BD", _make_spectral()), _make_record("SN", _make_spectral())]
		matrix = subsample.similarity.SimilarityMatrix(_library_with(*refs), _DEFAULT_CFG)
		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)
		scores = matrix.get_scores(inst.sample_id)
		assert len(scores) == 2
		names = {s.name for s in scores}
		assert "BD" in names
		assert "SN" in names

	def test_get_scores_unknown_sample_returns_empty (self) -> None:
		matrix = self._matrix("BD")
		assert matrix.get_scores(99999) == []

	def test_get_scores_sorted_descending (self) -> None:
		bd_ref = _make_record("BD", _make_spectral(spectral_flatness=0.9))
		sn_ref = _make_record("SN", _make_spectral(spectral_flatness=0.1))
		# Instrument matches BD much better than SN in spectral shape
		inst = _make_record("I1", _make_spectral(spectral_flatness=0.9))
		matrix = subsample.similarity.SimilarityMatrix(
			_library_with(bd_ref, sn_ref), _SPECTRAL_ONLY_CFG,
		)
		matrix.add(inst)
		scores = matrix.get_scores(inst.sample_id)
		assert scores[0].score >= scores[1].score
		assert scores[0].name == "BD"

	# ----- add_reference (path-based references) ----------------------------

	def test_add_reference_scores_against_existing_instruments (self) -> None:
		"""add_reference() should score a new reference against all existing instruments."""
		bd_ref = _make_record("BD", _make_spectral(spectral_flatness=0.9))
		matrix = subsample.similarity.SimilarityMatrix(
			_library_with(bd_ref), _SPECTRAL_ONLY_CFG,
		)

		# Add some instruments
		i1 = _make_record("I1", _make_spectral(spectral_flatness=0.9))
		i2 = _make_record("I2", _make_spectral(spectral_flatness=0.1))
		matrix.add(i1)
		matrix.add(i2)

		# Now add a new reference with a high flatness (should match i1 better)
		new_ref = _make_record("/tmp/new_ref", _make_spectral(spectral_flatness=0.95))
		matrix.add_reference(new_ref, [i1, i2])

		# The new reference should have i1 at rank 0 and i2 at rank 1
		assert matrix.get_match("/tmp/new_ref".upper(), 0) == i1.sample_id
		assert matrix.get_match("/tmp/new_ref".upper(), 1) == i2.sample_id

	def test_add_reference_idempotent (self) -> None:
		"""add_reference() should be idempotent — calling twice should have no effect."""
		ref = _make_record("REF", _make_spectral())
		matrix = subsample.similarity.SimilarityMatrix(_library_with(ref), _DEFAULT_CFG)

		inst = _make_record("I1", _make_spectral())
		matrix.add(inst)

		new_ref = _make_record("/tmp/new", _make_spectral())
		matrix.add_reference(new_ref, [inst])

		# Call again with the same reference — should be a no-op
		matrix.add_reference(new_ref, [inst])

		# Should still be retrievable
		assert matrix.get_match("/tmp/new".upper(), 0) == inst.sample_id

	def test_add_reference_empty_instruments_creates_empty_ranking (self) -> None:
		"""add_reference() with empty instrument list should create an empty ranking."""
		ref = _make_record("REF", _make_spectral())
		matrix = subsample.similarity.SimilarityMatrix(_library_with(ref), _DEFAULT_CFG)

		new_ref = _make_record("/tmp/empty", _make_spectral())
		matrix.add_reference(new_ref, [])

		# The new reference should exist but have no matches
		assert matrix.get_match("/tmp/empty".upper(), 0) is None

	# ----- thread safety ----------------------------------------------------

	def test_concurrent_add_and_get_match_do_not_raise (self) -> None:
		matrix = self._matrix("BD")
		errors: list[Exception] = []

		def writer () -> None:
			for i in range(50):
				try:
					matrix.add(_make_record(f"W{i}", _make_spectral()))
				except Exception as exc:
					errors.append(exc)

		def reader () -> None:
			for _ in range(100):
				try:
					matrix.get_match("BD", 0)
				except Exception as exc:
					errors.append(exc)

		t1 = threading.Thread(target=writer)
		t2 = threading.Thread(target=reader)
		t1.start()
		t2.start()
		t1.join()
		t2.join()
		assert errors == []


# ---------------------------------------------------------------------------
# TestLevelIndependence — level must NOT affect similarity scores
# ---------------------------------------------------------------------------

class TestLevelIndependence:

	"""Verify that LevelResult data has no influence on similarity scoring.

	Two records identical in every way except level should produce the same
	cosine similarity against any reference.
	"""

	def test_different_levels_same_similarity (self) -> None:
		"""Two records with the same spectral/timbre but different level should score identically."""
		spectral = _make_spectral()
		timbre   = _make_timbre()

		record_quiet = subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = "quiet",
			spectral    = spectral,
			rhythm      = subsample.analysis.RhythmResult(
				tempo_bpm=120.0,
				beat_times=(),
				pulse_curve=numpy.array([], dtype=numpy.float32),
				pulse_peak_times=(),
				onset_times=(),
				attack_times=(),
				onset_count=0,
			),
			pitch       = subsample.analysis.PitchResult(
				dominant_pitch_hz=0.0,
				pitch_confidence=0.0,
				chroma_profile=tuple(0.0 for _ in range(12)),
				dominant_pitch_class=-1,
				pitch_stability=0.0,
				voiced_frame_count=0,
			),
			timbre      = timbre,
			level       = subsample.analysis.LevelResult(peak=0.1, rms=0.03),
			band_energy = tests.helpers._make_band_energy(),
			params      = subsample.analysis.compute_params(44100),
			duration    = 1.0,
		)

		record_loud = subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = "loud",
			spectral    = spectral,
			rhythm      = subsample.analysis.RhythmResult(
				tempo_bpm=120.0,
				beat_times=(),
				pulse_curve=numpy.array([], dtype=numpy.float32),
				pulse_peak_times=(),
				onset_times=(),
				attack_times=(),
				onset_count=0,
			),
			pitch       = subsample.analysis.PitchResult(
				dominant_pitch_hz=0.0,
				pitch_confidence=0.0,
				chroma_profile=tuple(0.0 for _ in range(12)),
				dominant_pitch_class=-1,
				pitch_stability=0.0,
				voiced_frame_count=0,
			),
			timbre      = timbre,
			level       = subsample.analysis.LevelResult(peak=0.95, rms=0.70),
			band_energy = tests.helpers._make_band_energy(),
			params      = subsample.analysis.compute_params(44100),
			duration    = 1.0,
		)

		ref = _make_record("REF", _make_spectral())
		lib = _library_with(ref)
		cfg = _DEFAULT_CFG

		scores_quiet = subsample.similarity.score_against_library(record_quiet, lib, cfg)
		scores_loud  = subsample.similarity.score_against_library(record_loud,  lib, cfg)

		assert len(scores_quiet) == 1
		assert len(scores_loud)  == 1
		assert scores_quiet[0].score == pytest.approx(scores_loud[0].score, abs=1e-6)


# ---------------------------------------------------------------------------
# TestBandEnergyGroup
# ---------------------------------------------------------------------------

class TestBandEnergyGroup:

	"""Tests for the band-energy 5th group in _build_feature_vector()."""

	def _cfg_band_only (self) -> subsample.config.SimilarityConfig:
		"""Config that uses ONLY the band-energy group."""
		return subsample.config.SimilarityConfig(
			weight_spectral     = 0.0,
			weight_timbre       = 0.0,
			weight_timbre_delta = 0.0,
			weight_timbre_onset = 0.0,
			weight_band_energy  = 1.0,
		)

	def test_band_energy_weight_zero_excludes_group (self) -> None:
		"""Setting weight_band_energy=0 should produce a 47-element vector."""
		cfg = subsample.config.SimilarityConfig(weight_band_energy=0.0)
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, cfg)
		assert len(v) == 47

	def test_band_energy_only_has_8_elements (self) -> None:
		"""Band-energy-only config should produce an 8-element vector."""
		record = _make_record("X", _make_spectral())
		v = subsample.similarity._build_feature_vector(record, self._cfg_band_only())
		assert len(v) == 8

	def test_band_energy_kick_vs_hihat (self) -> None:
		"""A kick-like band profile should be more similar to kick ref than to hi-hat ref."""
		kick_profile = subsample.analysis.BandEnergyResult(
			energy_fractions = (0.7, 0.2, 0.05, 0.05),  # sub-bass dominant
			decay_rates      = (0.8, 0.3, 0.1, 0.1),
		)
		hihat_profile = subsample.analysis.BandEnergyResult(
			energy_fractions = (0.05, 0.05, 0.2, 0.7),  # presence dominant
			decay_rates      = (0.1, 0.1, 0.5, 0.9),
		)

		spectral = _make_spectral()

		kick_ref  = _make_record("BD", spectral)
		kick_ref  = dataclasses.replace(kick_ref, band_energy=kick_profile)
		hihat_ref = _make_record("HH", spectral)
		hihat_ref = dataclasses.replace(hihat_ref, band_energy=hihat_profile)

		query = _make_record("Q", spectral)
		query = dataclasses.replace(query, band_energy=kick_profile)

		lib = _library_with(kick_ref, hihat_ref)
		cfg = self._cfg_band_only()
		scores = subsample.similarity.score_against_library(query, lib, cfg)

		assert len(scores) == 2
		kick_score  = next(s for s in scores if s.name == "BD")
		hihat_score = next(s for s in scores if s.name == "HH")
		assert kick_score.score > hihat_score.score
