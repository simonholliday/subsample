"""Spectral fingerprint similarity scoring for Subsample.

Compares recordings against reference samples using cosine similarity on a
composite feature vector built from five independently-weighted groups:

  Spectral (14)           — 13 AnalysisResult fields + crest factor from LevelResult
  Timbre / sustained (12) — MFCC coefficients 1–12 (mean across duration)
  Timbre / delta (12)     — delta-MFCC coefficients 1–12 (timbre trajectory)
  Timbre / onset (12)     — onset-weighted MFCC coefficients 1–12 (attack)
  Band energy (8)         — 4 per-band energy fractions + 4 per-band decay rates

Each group is L2-normalised independently before being scaled by its
configured weight and concatenated. Cosine similarity on the result is
mathematically equivalent to a weighted average of the per-group cosine
similarities:

  sim = (w₁²·cos_spectral + w₂²·cos_timbre + w₃²·cos_delta + w₄²·cos_onset + w₅²·cos_band)
      / (w₁² + w₂² + w₃² + w₄² + w₅²)

MFCC coefficient 0 is excluded from all groups because it encodes overall
log-energy (loudness). Within a group, including the energy coefficient would
bias similarity toward louder = more similar — cosine similarity on the full
concatenated vector already ignores global magnitude.

Why cosine similarity?
  Cosine similarity measures the angle between two vectors, ignoring
  magnitude. For timbral comparison the *shape* of the fingerprint matters
  more than its overall scale: a kick drum always looks like a kick drum
  whether it was loud or quiet. Because the spectral group values are in
  [0, 1] (non-negative) and the MFCC groups are L2-normalised, the result
  is guaranteed to lie in [0, 1].

Primary usage — per-reference ranked lists:
  matrix = SimilarityMatrix(reference_library, cfg.similarity)
  matrix.bulk_add(instrument_library.samples())     # populate at startup
  matrix.add(new_record)                         # update after each capture
  matrix.remove(evicted_ids)                     # sync with FIFO eviction
  sample_id = matrix.get_match("BD", 0)          # most BD-like instrument
"""

import bisect
import dataclasses
import threading
import typing

import numpy

import subsample.analysis
import subsample.config
import subsample.library


@dataclasses.dataclass(frozen=True)
class RankedMatch:

	"""An instrument sample's cosine similarity to one reference sample.

	Used in SimilarityMatrix ranked lists. Sorted descending by score so the
	best-matching instrument is always at index 0.

	Fields:
		sample_id: Session-unique ID of the instrument sample.
		score:     Cosine similarity in [0.0, 1.0].
	"""

	sample_id: int
	score:     float


@dataclasses.dataclass(frozen=True)
class SimilarityScore:

	"""Cosine similarity between a recording and one reference sample.

	Fields:
		name:  Reference sample name (original casing from filename stem).
		score: Cosine similarity in [0.0, 1.0].
		       1.0 = identical composite fingerprint.
		       0.0 = maximally dissimilar (orthogonal fingerprints).
	"""

	name:  str
	score: float


def score_against_library (
	record:  subsample.library.SampleRecord,
	library: subsample.library.ReferenceLibrary,
	cfg:     subsample.config.SimilarityConfig,
) -> list[SimilarityScore]:

	"""Compare a sample record against every reference sample.

	Computes cosine similarity between the given record's composite feature
	vector and each reference sample's feature vector. Returns scores sorted
	by similarity descending so the best match appears first.

	Args:
		record:  The newly recorded sample (spectral + timbre data used).
		library: Reference library loaded at startup.
		cfg:     Similarity weights controlling each feature group's influence.

	Returns:
		List of SimilarityScore, sorted descending by score.
		Empty if the library is empty.
	"""

	query = _build_feature_vector(record, cfg)

	scores = [
		SimilarityScore(
			name  = ref.name,
			score = _cosine_similarity(query, _build_feature_vector(ref, cfg)),
		)
		for ref in library.samples()
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


class SimilarityMatrix:

	"""Per-reference ranked lists of instrument samples by cosine similarity.

	For every reference sample, maintains a ranked list of instrument samples
	ordered by cosine similarity (most similar first). Pairwise scores are
	cached in _scores so they are never recomputed.

	Thread-safe: the writers (bulk_add(), add(), remove(), add_reference())
	and the readers (get_matches() — the player's hot-path lookup — and
	get_scores()) all hold _lock, so reads see whole, consistent snapshots and
	may be called from any thread.

	Attributes:
		_lock:          Mutex protecting all mutable state.
		_similarity_cfg: Similarity weights used when building feature vectors.
		                 Stored here so all vectors (reference and instrument)
		                 are built consistently with the same weights.
		_ref_vectors:   Precomputed reference vectors {name_upper: array}.
		                Populated at construction; extended by add_reference().
		_rankings:      Per-reference ranked lists {name_upper: [RankedMatch, ...]}.
		                Always sorted descending by score.
		_scores:        Score cache {sample_id: {name_upper: score}}.
		                Also acts as a reverse index for eviction: pop the entry
		                here, then do a linear scan of each ranking list to remove
		                the matching entry. True eviction cost is O(M × N) where
		                M = references, N = instrument samples per list.
	"""

	def __init__ (
		self,
		reference_library: subsample.library.ReferenceLibrary,
		similarity_cfg:    subsample.config.SimilarityConfig,
	) -> None:

		self._lock = threading.Lock()
		self._similarity_cfg = similarity_cfg

		# Precompute reference vectors once — references never change after init
		self._ref_vectors: dict[str, numpy.ndarray] = {
			rec.name.upper(): _build_feature_vector(rec, similarity_cfg)
			for rec in reference_library.samples()
		}
		self._rankings: dict[str, list[RankedMatch]] = {
			name: [] for name in self._ref_vectors
		}

		# Maps sample_id → {ref_name_upper → score}.
		# Doubles as a reverse index: to evict a sample, pop its entry here
		# and remove it from each referenced ranking list.
		self._scores: dict[int, dict[str, float]] = {}

	def bulk_add (self, records: list[subsample.library.SampleRecord]) -> None:

		"""Add many instrument samples using vectorised similarity computation.

		Intended for startup: call once on a fresh (empty) SimilarityMatrix,
		after load_instrument_library() has run. Uses matrix multiplication
		to compute all N × M scores at once.

		IMPORTANT: This method replaces the ranking lists entirely from the
		supplied batch. Calling it on a non-empty matrix (i.e. after add() has
		already run) will desync _scores and _rankings. Always call on a fresh
		matrix before any add() calls.

		Args:
			records: Instrument samples to add (no duplicates expected).
		"""

		if not records or not self._ref_vectors:
			return

		if self._scores:
			raise ValueError(
				"bulk_add() called on a non-empty SimilarityMatrix — "
				"this would desync _scores and _rankings. "
				"Only call bulk_add() on a freshly constructed matrix."
			)

		ref_names = list(self._ref_vectors.keys())
		ref_matrix = numpy.array(
			[self._ref_vectors[n] for n in ref_names], dtype=numpy.float32
		)  # M × D

		inst_matrix = numpy.array(
			[_build_feature_vector(r, self._similarity_cfg) for r in records],
			dtype=numpy.float32,
		)  # N × D

		# Normalise rows; zero-norm vectors stay zero (cosine score is 0.0).
		# numpy.divide(..., where=, out=) keeps the division off the zero
		# rows entirely — a numpy.where(..., a/b, 0.0) form would compute
		# a/b on every row before masking, raising a "divide by zero"
		# RuntimeWarning on each startup that contained a silent sample.
		inst_norms = numpy.linalg.norm(inst_matrix, axis=1, keepdims=True)
		ref_norms  = numpy.linalg.norm(ref_matrix,  axis=1, keepdims=True)

		inst_normed = numpy.zeros_like(inst_matrix)
		numpy.divide(inst_matrix, inst_norms, where=inst_norms > 0, out=inst_normed)

		ref_normed = numpy.zeros_like(ref_matrix)
		numpy.divide(ref_matrix, ref_norms, where=ref_norms > 0, out=ref_normed)

		scores_matrix = (inst_normed @ ref_normed.T).astype(numpy.float64)  # N × M

		with self._lock:

			# Populate score cache for each instrument
			for i, record in enumerate(records):
				self._scores[record.sample_id] = {
					ref_names[j]: float(scores_matrix[i, j])
					for j in range(len(ref_names))
				}

			# Build each reference's ranked list via argsort descending
			for j, ref_name in enumerate(ref_names):
				col   = scores_matrix[:, j]
				order = numpy.argsort(-col)
				self._rankings[ref_name] = [
					RankedMatch(
						sample_id = records[int(k)].sample_id,
						score     = float(col[int(k)]),
					)
					for k in order
				]

	def add (self, record: subsample.library.SampleRecord) -> None:

		"""Add one instrument sample and insert it into each reference's ranked list.

		Called after each live capture. Computes similarity against all M
		references and inserts at the correct position (O(M log N)).

		Args:
			record: The newly captured instrument sample.
		"""

		if not self._ref_vectors:
			return

		vec = _build_feature_vector(record, self._similarity_cfg)
		sid = record.sample_id

		with self._lock:
			# Re-adding an id would bisect.insort a second RankedMatch into every
			# ranking while overwriting the single _scores row, so remove() (which
			# scans by id) would later leave one stale match orphaned — the same
			# desync bulk_add() guards against.  allocate_id is monotonic so this
			# never happens in practice; reject loudly if it ever does.
			if sid in self._scores:
				raise ValueError(
					f"add() called with sample_id {sid} already present — "
					f"sample ids must be unique (allocate_id is monotonic)."
				)

			score_row: dict[str, float] = {}

			for ref_name, ref_vec in self._ref_vectors.items():
				score = _cosine_similarity(vec, ref_vec)
				score_row[ref_name] = score
				bisect.insort(
					self._rankings[ref_name],
					RankedMatch(sample_id=sid, score=score),
					key=lambda m: -m.score,
				)

			self._scores[sid] = score_row

	def add_reference (
		self,
		record: subsample.library.SampleRecord,
		instruments: list[subsample.library.SampleRecord],
	) -> None:

		"""Add a new reference and score it against all current instrument samples.

		Used to add path-based references at MIDI-map load time. Computes similarity
		between the new reference and every instrument in the library, building the
		ranked list and updating score caches.

		Idempotent: calling with a reference whose name is already in the matrix
		is a no-op (the existing reference is not replaced).

		Args:
			record:      The reference sample record (name.upper() becomes the key).
			instruments: All current instrument samples (for scoring).
		"""

		key = record.name.upper()

		with self._lock:
			if key in self._ref_vectors:
				return  # Already present; idempotent

			vec = _build_feature_vector(record, self._similarity_cfg)

			self._ref_vectors[key] = vec
			self._rankings[key] = []

			for inst in instruments:
				inst_vec = _build_feature_vector(inst, self._similarity_cfg)
				score = _cosine_similarity(vec, inst_vec)

				# Add to the ranking for this reference
				bisect.insort(
					self._rankings[key],
					RankedMatch(sample_id=inst.sample_id, score=score),
					key=lambda m: -m.score,
				)

				# Update the per-instrument score cache (for get_scores)
				if inst.sample_id not in self._scores:
					self._scores[inst.sample_id] = {}
				self._scores[inst.sample_id][key] = score

	def remove (self, sample_ids: list[int]) -> None:

		"""Remove evicted instrument samples from all ranked lists.

		Call with the list of IDs returned by InstrumentLibrary.add() to keep
		the matrix consistent with the instrument library.

		Args:
			sample_ids: Instrument sample IDs to remove.
		"""

		if not sample_ids:
			return

		with self._lock:
			for sid in sample_ids:
				ref_scores = self._scores.pop(sid, {})

				for ref_name in ref_scores:
					ranked = self._rankings.get(ref_name)
					if ranked is None:
						continue

					ranked[:] = [m for m in ranked if m.sample_id != sid]

	def get_match (
		self,
		reference_name: str,
		rank: int = 0,
	) -> typing.Optional[int]:

		"""Return the sample_id of the instrument at a given rank for a reference.

		Args:
			reference_name: Reference sample name (case-insensitive).
			rank:           0 = most similar, 1 = second most similar, etc.

		Returns:
			sample_id of the instrument at the given rank, or None if the
			reference name is unknown, rank is out of bounds, or no instrument
			samples have been added yet.
		"""

		with self._lock:
			ranked = self._rankings.get(reference_name.upper())
			if ranked is None or rank >= len(ranked):
				return None
			return ranked[rank].sample_id

	def get_matches (
		self,
		reference_name: str,
		limit: int = 0,
	) -> list[RankedMatch]:

		"""Return the top-ranked instrument matches for a reference.

		Args:
			reference_name: Reference sample name (case-insensitive).
			limit:          Maximum number of results (0 = all).

		Returns:
			Shallow copy of the ranked list, sliced to limit. Empty if the
			reference name is unknown or no instruments have been added.
		"""

		with self._lock:
			ranked = self._rankings.get(reference_name.upper(), [])
			if limit > 0:
				return ranked[:limit]
			return ranked[:]

	def get_scores (self, sample_id: int) -> list[SimilarityScore]:

		"""Return similarity scores of one instrument against all references.

		Uses cached scores — no recomputation. Equivalent to
		score_against_library() but reads from the matrix.

		Args:
			sample_id: Instrument sample ID (must have been added via add() or
			           bulk_add()).

		Returns:
			List of SimilarityScore sorted descending by score.
			Empty if the sample_id is not in the matrix.
		"""

		with self._lock:
			ref_scores = self._scores.get(sample_id, {})

		return sorted(
			[SimilarityScore(name=name, score=score) for name, score in ref_scores.items()],
			key=lambda s: s.score,
			reverse=True,
		)

	def __len__ (self) -> int:

		"""Number of instrument samples currently tracked."""

		with self._lock:
			return len(self._scores)

	def __repr__ (self) -> str:

		with self._lock:
			n_refs = len(self._rankings)
			n_inst = len(self._scores)
		return f"SimilarityMatrix({n_refs} refs × {n_inst} instruments)"


def _build_feature_vector (
	record: subsample.library.SampleRecord,
	cfg:    subsample.config.SimilarityConfig,
) -> numpy.ndarray:

	"""Build the composite similarity feature vector for a sample record.

	Assembles up to five feature groups, each independently L2-normalised and
	scaled by its configured weight, then concatenates them into a single 1-D
	float32 array. Groups with weight 0.0 are omitted entirely.

	Groups:
	  Spectral (14):           all AnalysisResult fields + crest factor, already in [0, 1].
	  Timbre / sustained (12): MFCC coefficients 1–12 (mean over duration).
	  Timbre / delta (12):     delta-MFCC coefficients 1–12 (timbre trajectory).
	  Timbre / onset (12):     onset-weighted MFCC coefficients 1–12 (attack).
	  Band energy (8):         4 energy fractions + 4 decay rates across sub-bass,
	                           low-mid, high-mid, and presence bands.

	MFCC coefficient 0 is excluded from all MFCC groups because it encodes
	overall log-energy. Within a group, coeff 0 would bias similarity toward
	loudness rather than spectral shape; coeff 1–12 carry the timbral information.

	When all five groups are active the result is a 58-element vector. Cosine
	similarity on this vector equals a weighted average of per-group cosine
	similarities (weights squared, renormalised). See module docstring.

	Args:
		record: SampleRecord to convert.
		cfg:    Weight configuration for each feature group.

	Returns:
		1-D float32 array. Empty array if all weights are 0.0.
	"""

	parts: list[numpy.ndarray] = []

	# --- Spectral group (14 values, all [0, 1]) ---
	if cfg.weight_spectral > 0.0:

		# Normalised crest factor: computed from LevelResult (raw engineering
		# values) rather than stored redundantly in AnalysisResult.  The log
		# normalisation spreads the perceptually useful range [sqrt(2), 10]
		# across [0, 1].
		if record.level.rms > 1e-10:
			cf = record.level.peak / record.level.rms
			cf_norm = subsample.analysis.log_normalize(
				cf, subsample.analysis.CREST_MIN, subsample.analysis.CREST_MAX,
			)
		else:
			cf_norm = 0.0

		spectral = numpy.array([
			record.spectral.spectral_flatness,
			record.spectral.attack,
			record.spectral.release,
			record.spectral.spectral_centroid,
			record.spectral.spectral_bandwidth,
			record.spectral.zcr,
			record.spectral.harmonic_ratio,
			record.spectral.spectral_contrast,
			record.spectral.voiced_fraction,
			record.spectral.log_attack_time,
			record.spectral.spectral_flux,
			record.spectral.spectral_rolloff,
			record.spectral.spectral_slope,
			cf_norm,
		], dtype=numpy.float32)
		parts.append(_l2_normalize(spectral) * cfg.weight_spectral)

	# --- Timbre groups (coefficients 1–12 from each MFCC variant) ---
	# Coeff 0 encodes overall log-energy and is excluded; see docstring.
	if cfg.weight_timbre > 0.0:
		mfcc = numpy.array(record.timbre.mfcc[1:], dtype=numpy.float32)
		parts.append(_l2_normalize(mfcc) * cfg.weight_timbre)

	if cfg.weight_timbre_delta > 0.0:
		delta = numpy.array(record.timbre.mfcc_delta[1:], dtype=numpy.float32)
		parts.append(_l2_normalize(delta) * cfg.weight_timbre_delta)

	if cfg.weight_timbre_onset > 0.0:
		onset = numpy.array(record.timbre.mfcc_onset[1:], dtype=numpy.float32)
		parts.append(_l2_normalize(onset) * cfg.weight_timbre_onset)

	# --- Band energy group (8 values: 4 fractions + 4 decay rates) ---
	# Fractions encode WHERE energy lives; decay rates encode HOW FAST it dissipates.
	# Together they directly capture drum-type physical signatures.
	if cfg.weight_band_energy > 0.0:
		band = numpy.array(
			record.band_energy.energy_fractions + record.band_energy.decay_rates,
			dtype=numpy.float32,
		)
		parts.append(_l2_normalize(band) * cfg.weight_band_energy)

	if not parts:
		return numpy.array([], dtype=numpy.float32)

	return numpy.concatenate(parts)


def _l2_normalize (v: numpy.ndarray) -> numpy.ndarray:

	"""Return v divided by its L2 norm, or a zero vector if the norm is negligible.

	A zero input (e.g. silent audio producing all-zero MFCCs) returns a zero
	vector rather than NaN; its cosine contribution will be 0.0.
	"""

	norm = float(numpy.linalg.norm(v))

	if norm < 1e-9:
		return numpy.zeros_like(v)

	return v / norm


def _cosine_similarity (a: numpy.ndarray, b: numpy.ndarray) -> float:

	"""Cosine similarity between two 1-D arrays.

	Returns 0.0 if either vector is all-zero (degenerate case — a silent or
	perfectly flat recording produces a zero spectral fingerprint).
	"""

	norm_a = float(numpy.linalg.norm(a))
	norm_b = float(numpy.linalg.norm(b))

	if norm_a < 1e-9 or norm_b < 1e-9:
		return 0.0

	return float(numpy.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Self-similarity: clustering and ordering a single set of samples
#
# The primary API above ranks instrument samples AGAINST a set of references.
# These functions instead compare a set of samples against ITSELF — the
# all-pairs cosine matrix — which powers curation tools: grouping near-
# duplicates (so days of a running mic collapse to one decision per distinct
# sound) and ordering samples so sonically-alike ones are adjacent (so a
# manual audition compares like with like).  Both build on the same weighted
# feature vector the ranking path uses, so "similar" means the same thing
# everywhere.  See scripts/catalog_samples.py.
# ---------------------------------------------------------------------------


def self_similarity_matrix (
	records: list[subsample.library.SampleRecord],
	cfg:     subsample.config.SimilarityConfig,
) -> numpy.ndarray:

	"""Return the N×N cosine similarity matrix of records against themselves.

	Entry [i, j] is the cosine similarity of records[i] and records[j] on the
	weighted composite feature vector, in [0, 1] (the fingerprint is non-
	negative, so no pair is anti-correlated).  The diagonal is ~1.0 for any
	sample with a non-zero fingerprint, 0.0 for a degenerate (silent) one.

	Memory is O(N²): a few thousand samples is fine (~36 MB at N=3000), but the
	matrix grows quadratically — callers cataloging very large directories
	should expect the cost.

	Args:
		records: Samples to compare.  Order is preserved in the row/column index.
		cfg:     Feature-group weights (typically cfg.similarity).

	Returns:
		float32 array of shape (N, N).  Empty (0, 0) array when records is empty.
	"""

	if not records:
		return numpy.zeros((0, 0), dtype=numpy.float32)

	matrix = numpy.array(
		[_build_feature_vector(r, cfg) for r in records],
		dtype=numpy.float32,
	)  # N × D

	# Row-normalise so the dot product is cosine similarity.  Zero-norm rows
	# (silent samples) stay zero via the where= mask — a plain divide would
	# raise a divide-by-zero warning on every degenerate sample.
	norms  = numpy.linalg.norm(matrix, axis=1, keepdims=True)
	normed = numpy.zeros_like(matrix)
	numpy.divide(matrix, norms, where=norms > 0, out=normed)

	return (normed @ normed.T).astype(numpy.float32)


def _connected_components (n: int, edges: list[tuple[int, int]]) -> list[list[int]]:

	"""Group 0..n-1 into connected components given an edge list (union-find).

	Two nodes joined by any edge — directly or transitively — land in the same
	component.  Returns each component as a sorted list of node indices; the
	components themselves are ordered largest-first, then by smallest member,
	so the biggest duplicate piles surface first and the order is deterministic.
	"""

	parent = list(range(n))

	def find (x: int) -> int:
		# Iterative find with path compression (recursion would risk a deep
		# stack on a long single-linkage chain).
		root = x
		while parent[root] != root:
			root = parent[root]
		while parent[x] != root:
			parent[x], x = root, parent[x]
		return root

	for a, b in edges:
		ra, rb = find(a), find(b)
		if ra != rb:
			parent[ra] = rb

	groups: dict[int, list[int]] = {}
	for node in range(n):
		groups.setdefault(find(node), []).append(node)

	members = [sorted(g) for g in groups.values()]
	members.sort(key=lambda g: (-len(g), g[0]))
	return members


def group_near_duplicates (
	records:   list[subsample.library.SampleRecord],
	cfg:       subsample.config.SimilarityConfig,
	threshold: float,
) -> list[list[int]]:

	"""Cluster records into near-duplicate groups by self-similarity.

	Any two samples whose cosine similarity is >= threshold are placed in the
	same group, transitively (single-linkage / connected components): if A~B and
	B~C, then A, B and C share a group even if A and C fall just below the
	threshold.  At a high threshold (near-identical repeats of one sound) this
	is the intuitive "these are all the same event" grouping; lowering it risks
	chaining distinct sounds together, so it is a tunable.

	Every sample belongs to exactly one group; a sample with no near-duplicate
	forms a group of one.  Groups are returned largest-first (see
	_connected_components), so callers can surface the biggest piles first.

	Args:
		records:   Samples to cluster (indices into this list are returned).
		cfg:       Feature-group weights (typically cfg.similarity).
		threshold: Minimum cosine similarity (0..1) for two samples to group.

	Returns:
		List of groups, each a sorted list of indices into records.
	"""

	n = len(records)

	if n == 0:
		return []

	sim = self_similarity_matrix(records, cfg)

	# Edges are the above-threshold pairs in the strict upper triangle; usually
	# a small fraction of all pairs, so union-find stays cheap even though the
	# matrix itself is dense.
	upper       = numpy.triu(sim >= threshold, k=1)
	edge_coords = numpy.argwhere(upper)
	edges       = [(int(i), int(j)) for i, j in edge_coords]

	return _connected_components(n, edges)


def similarity_order (
	records: list[subsample.library.SampleRecord],
	cfg:     subsample.config.SimilarityConfig,
) -> list[int]:

	"""Return a permutation of record indices that places alike samples adjacent.

	Builds a greedy nearest-neighbour chain: start at the first record, then
	repeatedly step to the most-similar not-yet-visited record.  Auditioning in
	this order means each sample is followed by its closest sonic neighbour, so
	keep/discard becomes a comparison with the sound just heard rather than a
	cold judgement.  Deterministic for a given input order (ties broken by
	lowest index).

	Args:
		records: Samples to order (indices into this list are returned).
		cfg:     Feature-group weights (typically cfg.similarity).

	Returns:
		A list containing each index 0..N-1 exactly once.
	"""

	n = len(records)

	if n <= 1:
		return list(range(n))

	sim = self_similarity_matrix(records, cfg)

	visited = numpy.zeros(n, dtype=bool)
	order   = [0]
	visited[0] = True

	for _ in range(n - 1):
		current   = order[-1]
		neighbours = sim[current].copy()
		neighbours[visited] = -numpy.inf   # never revisit
		order.append(int(numpy.argmax(neighbours)))
		visited[order[-1]] = True

	return order
