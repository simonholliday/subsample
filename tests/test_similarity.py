"""Tests for subsample/similarity.py — spectral fingerprint similarity scoring."""

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
		mfcc=tuple(0.0 for _ in range(13)),
	)
	params = subsample.analysis.compute_params(44100)

	return subsample.library.SampleRecord(
		sample_id=subsample.library._allocate_id(),
		name=name, spectral=spectral, rhythm=rhythm,
		pitch=pitch, params=params, duration=1.0,
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
		with pytest.raises(Exception):
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
