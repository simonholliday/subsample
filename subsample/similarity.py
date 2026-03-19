"""Spectral fingerprint similarity scoring for Subsample.

Compares a newly recorded sound against a set of reference samples using
cosine similarity on their spectral fingerprints — the 9-element [0, 1]
vector returned by AnalysisResult.as_vector().

Why cosine similarity?
  Cosine similarity measures the angle between two vectors, ignoring
  magnitude. For timbral comparison the *shape* of the fingerprint matters
  more than its overall scale: a kick drum always looks like a kick drum
  whether it was loud or quiet. Because all values are in [0, 1] (non-
  negative), cosine similarity is guaranteed to lie in [0, 1].

Typical usage:
    scores = score_against_library(spectral_result, reference_library)
    _log.debug("Similarity: %s", format_similarity_scores(scores))
"""

import dataclasses

import numpy

import subsample.analysis
import subsample.library


@dataclasses.dataclass(frozen=True)
class SimilarityScore:

	"""Cosine similarity between a recording and one reference sample.

	Fields:
		name:  Reference sample name (original casing from filename stem).
		score: Cosine similarity in [0.0, 1.0].
		       1.0 = identical spectral fingerprint.
		       0.0 = maximally dissimilar (orthogonal fingerprints).
	"""

	name:  str
	score: float


def score_against_library (
	spectral: subsample.analysis.AnalysisResult,
	library:  subsample.library.ReferenceLibrary,
) -> list[SimilarityScore]:

	"""Compare a spectral fingerprint against every reference sample.

	Computes cosine similarity between the given result's spectral fingerprint
	and each reference sample's fingerprint. Returns scores sorted by
	similarity descending so the best match appears first.

	Args:
		spectral: AnalysisResult from the newly recorded sample.
		library:  Reference library loaded at startup.

	Returns:
		List of SimilarityScore, sorted descending by score.
		Empty if the library is empty.
	"""

	query = spectral.as_vector()

	scores = [
		SimilarityScore(
			name  = record.name,
			score = _cosine_similarity(query, record.spectral.as_vector()),
		)
		for record in library.all()
	]

	# Sort best match first so the first element is always the closest reference
	return sorted(scores, key=lambda s: s.score, reverse=True)


def format_similarity_scores (scores: list[SimilarityScore]) -> str:

	"""Format similarity scores as a single compact log string.

	Example output: "KICK 0.94  SNARE 0.61  HAT 0.22"

	Scores are expected to already be sorted (best match first) by
	score_against_library(). Returns an empty string if scores is empty.
	"""

	if not scores:
		return ""

	return "  ".join(f"{s.name} {s.score:.2f}" for s in scores)


def _cosine_similarity (a: numpy.ndarray, b: numpy.ndarray) -> float:

	"""Cosine similarity between two 1-D non-negative arrays.

	Returns 0.0 if either vector is all-zero (degenerate case — a silent or
	perfectly flat recording produces a zero spectral fingerprint).
	"""

	norm_a = float(numpy.linalg.norm(a))
	norm_b = float(numpy.linalg.norm(b))

	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0

	return float(numpy.dot(a, b) / (norm_a * norm_b))
