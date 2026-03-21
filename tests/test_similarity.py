"""Tests for subsample/similarity.py — spectral fingerprint similarity scoring."""

import dataclasses
import threading

import numpy
import pytest

import subsample.analysis
import subsample.library
import subsample.similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_record (name: str, spectral: subsample.analysis.AnalysisResult) -> subsample.library.SampleRecord:

	"""Return a minimal SampleRecord wrapping the given spectral result."""

	rhythm = subsample.analysis.RhythmResult(
		tempo_bpm=120.0,
		beat_times=(),
		pulse_curve=numpy.array([], dtype=numpy.float32),
		pulse_peak_times=(),
		onset_times=(),
		onset_count=0,
	)
	pitch = subsample.analysis.PitchResult(
		dominant_pitch_hz=0.0,
		pitch_confidence=0.0,
		chroma_profile=tuple(0.0 for _ in range(12)),
		dominant_pitch_class=-1,
	)
	timbre = subsample.analysis.TimbreResult(
		mfcc=tuple(0.0 for _ in range(13)),
		mfcc_delta=tuple(0.0 for _ in range(13)),
		mfcc_onset=tuple(0.0 for _ in range(13)),
	)
	params = subsample.analysis.compute_params(44100)

	return subsample.library.SampleRecord(
		sample_id=subsample.library.allocate_id(),
		name=name, spectral=spectral, rhythm=rhythm,
		pitch=pitch, timbre=timbre, params=params, duration=1.0,
	)


def _library_with (*records: subsample.library.SampleRecord) -> subsample.library.ReferenceLibrary:

	"""Return a ReferenceLibrary containing the given records."""

	return subsample.library.ReferenceLibrary(list(records))


# ---------------------------------------------------------------------------
# TestAsVector
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
# TestScoreAgainstLibrary
# ---------------------------------------------------------------------------

class TestScoreAgainstLibrary:

	def test_empty_library_returns_empty (self) -> None:
		lib = _library_with()
		scores = subsample.similarity.score_against_library(_make_spectral(), lib)
		assert scores == []

	def test_identical_fingerprint_scores_1 (self) -> None:
		spectral = _make_spectral()
		lib = _library_with(_make_record("KICK", spectral))
		scores = subsample.similarity.score_against_library(spectral, lib)
		assert len(scores) == 1
		assert scores[0].score == pytest.approx(1.0)

	def test_orthogonal_fingerprint_scores_0 (self) -> None:
		# Query is all-zero in first position, reference is all-zero elsewhere
		query = _make_spectral(
			spectral_flatness=1.0, attack=0.0, release=0.0,
			spectral_centroid=0.0, spectral_bandwidth=0.0,
			zcr=0.0, harmonic_ratio=0.0, spectral_contrast=0.0, voiced_fraction=0.0,
		)
		ref = _make_spectral(
			spectral_flatness=0.0, attack=1.0, release=0.0,
			spectral_centroid=0.0, spectral_bandwidth=0.0,
			zcr=0.0, harmonic_ratio=0.0, spectral_contrast=0.0, voiced_fraction=0.0,
		)
		lib = _library_with(_make_record("REF", ref))
		scores = subsample.similarity.score_against_library(query, lib)
		assert scores[0].score == pytest.approx(0.0)

	def test_partial_match_in_range (self) -> None:
		query = _make_spectral(spectral_flatness=0.8, attack=0.2)
		ref   = _make_spectral(spectral_flatness=0.6, attack=0.4)
		lib = _library_with(_make_record("REF", ref))
		scores = subsample.similarity.score_against_library(query, lib)
		assert 0.0 < scores[0].score < 1.0

	def test_returns_sorted_descending (self) -> None:
		# KICK is identical to query; SNARE is different — KICK should score highest
		query = _make_spectral(spectral_flatness=0.9)
		kick  = _make_spectral(spectral_flatness=0.9)
		snare = _make_spectral(spectral_flatness=0.1)
		lib = _library_with(
			_make_record("SNARE", snare),
			_make_record("KICK", kick),
		)
		scores = subsample.similarity.score_against_library(query, lib)
		assert scores[0].name == "KICK"
		assert scores[0].score >= scores[1].score

	def test_name_preserved_in_result (self) -> None:
		lib = _library_with(_make_record("BD0025", _make_spectral()))
		scores = subsample.similarity.score_against_library(_make_spectral(), lib)
		assert scores[0].name == "BD0025"

	def test_score_is_frozen (self) -> None:
		score = subsample.similarity.SimilarityScore(name="KICK", score=0.9)
		with pytest.raises(dataclasses.FrozenInstanceError):
			score.name = "SNARE"  # type: ignore[misc]


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
		return subsample.similarity.SimilarityMatrix(_library_with(*refs))

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
		bd_ref  = _make_record("BD",    _make_spectral(spectral_flatness=0.9))
		i_high  = _make_record("HIGH",  _make_spectral(spectral_flatness=0.9))
		i_low   = _make_record("LOW",   _make_spectral(spectral_flatness=0.1))
		matrix  = subsample.similarity.SimilarityMatrix(_library_with(bd_ref))
		matrix.add(i_low)
		matrix.add(i_high)
		assert matrix.get_match("BD", 0) == i_high.sample_id
		assert matrix.get_match("BD", 1) == i_low.sample_id

	def test_identical_fingerprint_scores_highest (self) -> None:
		spectral = _make_spectral(spectral_flatness=0.8, attack=0.2)
		ref  = _make_record("BD", spectral)
		inst = _make_record("I1", spectral)
		other = _make_record("I2", _make_spectral(spectral_flatness=0.0))
		matrix = subsample.similarity.SimilarityMatrix(_library_with(ref))
		matrix.add(other)
		matrix.add(inst)
		assert matrix.get_match("BD", 0) == inst.sample_id

	# ----- bulk_add ---------------------------------------------------------

	def test_bulk_add_populates_rankings (self) -> None:
		bd_ref = _make_record("BD", _make_spectral(spectral_flatness=0.9))
		i1 = _make_record("I1", _make_spectral(spectral_flatness=0.9))
		i2 = _make_record("I2", _make_spectral(spectral_flatness=0.1))
		matrix = subsample.similarity.SimilarityMatrix(_library_with(bd_ref))
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
		m_bulk = subsample.similarity.SimilarityMatrix(lib)
		m_bulk.bulk_add([i1, i2, i3])

		m_incr = subsample.similarity.SimilarityMatrix(lib)
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
		matrix = subsample.similarity.SimilarityMatrix(_library_with(*refs))
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
		matrix = subsample.similarity.SimilarityMatrix(_library_with(*refs))
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
		# Instrument matches BD much better than SN
		inst = _make_record("I1", _make_spectral(spectral_flatness=0.9))
		matrix = subsample.similarity.SimilarityMatrix(_library_with(bd_ref, sn_ref))
		matrix.add(inst)
		scores = matrix.get_scores(inst.sample_id)
		assert scores[0].score >= scores[1].score
		assert scores[0].name == "BD"

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
