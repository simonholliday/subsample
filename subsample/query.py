"""Sample query engine — filter, order, and select samples by analysis metadata.

Provides the select/process pipeline for the MIDI mapping system.  A MIDI map
assignment declares a SelectSpec (which sample to play) and a ProcessSpec
(how to present it).  The query engine evaluates SelectSpec against the
instrument library at trigger time, returning ranked sample IDs.

Architecture
------------

SelectSpec
    Parsed from the ``select:`` block in a MIDI map assignment.  Contains
    filter predicates (``where``), an ordering key (``order_by``), and a pick
    position.

ProcessSpec
    Parsed from the ``process:`` list.  Contains an ordered sequence of
    processor declarations (repitch, beat_quantize, etc.) that map to
    TransformStep subclasses at execution time.

Assignment
    A compiled MIDI map entry combining SelectSpec, ProcessSpec, playback
    flags (one_shot, gain_db), and output routing (pan_weights).

Query evaluation
----------------

  InstrumentLibrary.samples()
      → filter by SelectSpec.where predicates (all must pass)
      → sort by SelectSpec.order_by key
      → return ranked list
      → caller picks Nth position (or per-note rank distribution)

All filtering is a linear scan — with typical library sizes (500-1000 samples)
this is microseconds and needs no secondary indices.
"""

import dataclasses
import logging
import pathlib
import typing

import librosa
import numpy
import pymididefs.notes

import subsample.analysis

if typing.TYPE_CHECKING:
	import subsample.library
	import subsample.similarity

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_path_like (s: str) -> bool:
	"""Return True if the string looks like a filesystem path."""
	return "/" in s or s.startswith(".")


# ---------------------------------------------------------------------------
# Filter predicates — Range + WherePredicate
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Range:

	"""Numeric constraint block for one WherePredicate dimension.

	Each operator is optional; all set operators must pass (AND) for
	``contains()`` to return True.  An empty Range (no operator set)
	matches every value — used as the default for WherePredicate's
	per-field Ranges to mean "no filter on this dimension".

	Operator vocabulary:
	  gte  x >= n   (greater than or equal to)
	  lte  x <= n   (less than or equal to)
	  gt   x >  n   (strictly greater than)
	  lt   x <  n   (strictly less than)
	  eq   x == n   (exactly equal; strict, no epsilon tolerance — fine
	                 for int-valued fields like onset count, care with
	                 non-round-number floats)"""

	gte: typing.Optional[float] = None
	lte: typing.Optional[float] = None
	gt:  typing.Optional[float] = None
	lt:  typing.Optional[float] = None
	eq:  typing.Optional[float] = None

	def contains (self, x: float) -> bool:
		if self.eq  is not None and x != self.eq:  return False
		if self.gte is not None and x <  self.gte: return False
		if self.lte is not None and x >  self.lte: return False
		if self.gt  is not None and x <= self.gt:  return False
		if self.lt  is not None and x >= self.lt:  return False
		return True

	def is_empty (self) -> bool:

		"""True when no operator is set — contains() returns True for
		every value.  Used by matches() to skip the external-state
		resolver lookup for quantized_beats when no constraint is
		active."""

		return (
			self.gte is None and self.lte is None
			and self.gt is None and self.lt is None
			and self.eq is None
		)


# Per-field numeric predicates.  Each operator dict in YAML populates one
# of these; the field name is both the internal attribute and (for pitch,
# with the _hz suffix stripped) the YAML key.
_NUMERIC_FIELDS: tuple[str, ...] = (
	"duration", "onsets", "tempo", "pitch_hz", "quantized_beats",
)

_VALID_OPERATORS: frozenset[str] = frozenset({"gte", "lte", "gt", "lt", "eq"})


# Strict-mode flag.  When True (default), unknown keys in `where:` and
# unknown processor names in `process:` raise ValueError at parse time.
# When False, they are logged as warnings and silently ignored — the
# historical behaviour, retained as an opt-out for users on older maps.
# Toggled via set_strict_mode(); the player reads the config flag
# `player.midi_map.strict` at startup.
_STRICT_MODE: bool = True


def set_strict_mode (strict: bool) -> None:
	"""Enable or disable strict unknown-key / unknown-processor rejection.

	Strict mode (the default) raises ValueError on unknown YAML keys — this
	catches typos that would otherwise silently match every sample.  The
	lenient path is provided only for compatibility with older MIDI map
	files that may carry keys the parser no longer recognises."""
	global _STRICT_MODE
	_STRICT_MODE = strict


# Valid processor names — kept in lockstep with the dispatch ladder in
# subsample.transform.spec_from_process().  Used by parse_process() to
# reject unknown names at parse time when strict mode is enabled.
_VALID_PROCESSOR_NAMES: frozenset[str] = frozenset({
	"repitch",
	"beat_quantize",
	"pad_quantize",
	"filter_low",
	"filter_high",
	"filter_band",
	"reverse",
	"saturate",
	"compress",
	"limit",
	"hpss",
	"hpss_harmonic",    # legacy; translated to hpss {keep: harmonic}
	"hpss_percussive",  # legacy; translated to hpss {keep: percussive}
	"gate",
	"distort",
	"reshape",
	"transient",
	"vocoder",
})


# Per-processor legacy parameter renames.  Shape: (processor_name,
# legacy_param) → new_param.  Applied by parse_process() when building
# the ProcessorStep's params tuple — the spec_from_process() dispatch
# only ever sees the new names.
#
# Each entry is a rename motivated by A1 in the language review:
# `amount` meant four different things depending on the processor.
# The new names are unit-indicative (drive/gain in dB, strength as a
# 0-1 fraction) so `amount` no longer has to be context-disambiguated.
_LEGACY_PROCESSOR_PARAMS: dict[tuple[str, str], str] = {
	("saturate",      "amount"): "drive",     # dB
	("transient",     "amount"): "gain",      # dB (signed)
	("beat_quantize", "amount"): "strength",  # 0-1 fraction
	("pad_quantize",  "amount"): "strength",  # 0-1 fraction
	# `bpm` → `tempo` (C2): property name matches the where-predicate.
	("beat_quantize", "bpm"):    "tempo",
	("pad_quantize",  "bpm"):    "tempo",
}


# Non-range where-predicate keys.  Numeric keys (new-form + legacy) are
# defined later in the file; _valid_where_keys() combines both into one
# frozenset at call time.
_NON_RANGE_WHERE_KEYS: frozenset[str] = frozenset(
	{"pitched", "reference", "name", "path", "directory"}
)


def _valid_where_keys () -> frozenset[str]:
	"""All accepted keys inside a `where:` block, including legacy aliases."""
	return frozenset(
		_NON_RANGE_WHERE_KEYS
		| _NUMERIC_YAML_KEYS.keys()
		| _LEGACY_WHERE_KEYS.keys()
	)


@dataclasses.dataclass(frozen=True)
class WherePredicate:

	"""Filter criteria for sample selection.

	Numeric dimensions (``duration``, ``onsets``, ``tempo``, ``pitch_hz``,
	``quantized_beats``) each carry a Range; an empty Range means "no
	filter on this dimension".  Non-numeric fields (``pitched``,
	``reference``, ``name``, ``name_path``, ``directory``) remain flat
	Optionals — they aren't comparison predicates."""

	duration:        Range = dataclasses.field(default_factory=Range)
	onsets:          Range = dataclasses.field(default_factory=Range)
	tempo:           Range = dataclasses.field(default_factory=Range)
	pitch_hz:        Range = dataclasses.field(default_factory=Range)
	quantized_beats: Range = dataclasses.field(default_factory=Range)

	pitched:   typing.Optional[bool] = None
	reference: typing.Optional[str]  = None
	name:      typing.Optional[str]  = None
	name_path: typing.Optional[str]  = None
	"""Internal field — set by parse_select for path-based name references;
	used by _resolve_path_references to load samples; never evaluated in
	matches() and never exposed in YAML."""
	directory: typing.Optional[str]  = None

	def matches (
		self,
		record: "subsample.library.SampleRecord",
		beats_resolver: typing.Optional[typing.Callable[[int], typing.Optional[float]]] = None,
	) -> bool:

		"""Return True if the record passes all active filter predicates."""

		if not self.duration.contains(record.duration):
			return False

		if not self.onsets.contains(record.rhythm.onset_count):
			return False

		if not self.tempo.contains(record.rhythm.tempo_bpm):
			return False

		if not self.pitch_hz.contains(record.pitch.dominant_pitch_hz):
			return False

		# quantized_beats is the only field that consults external state.
		# We only call the resolver when a constraint is actually active;
		# an empty Range skips the lookup entirely so non-quantized
		# samples aren't excluded from otherwise-unconstrained queries.
		if not self.quantized_beats.is_empty():
			beats = beats_resolver(record.sample_id) if beats_resolver is not None else None
			if beats is None:
				return False
			if not self.quantized_beats.contains(beats):
				return False

		if self.pitched is not None:
			is_pitched = subsample.analysis.has_stable_pitch(
				record.spectral, record.pitch, record.duration,
			)
			if self.pitched != is_pitched:
				return False

		if self.name is not None and record.name != self.name:
			return False

		if self.directory is not None:
			if record.filepath is None:
				return False
			try:
				pathlib.Path(record.filepath).resolve().relative_to(
					pathlib.Path(self.directory).resolve(),
				)
			except ValueError:
				return False

		# reference is handled externally by the query runner (needs SimilarityMatrix).
		# WherePredicate.matches() is a record-level filter; reference scoring
		# requires the full ranked list, so it's applied in query().

		return True


# Legacy ``min_X:`` / ``max_X:`` YAML keys translate into (field, operator)
# pairs.  Kept indefinitely so existing YAML keeps working; not deprecated.
_LEGACY_WHERE_KEYS: dict[str, tuple[str, str]] = {
	"min_duration":        ("duration",        "gte"),
	"max_duration":        ("duration",        "lte"),
	"min_onsets":          ("onsets",          "gte"),
	"max_onsets":          ("onsets",          "lte"),
	"min_tempo":           ("tempo",           "gte"),
	"max_tempo":           ("tempo",           "lte"),
	"min_pitch":           ("pitch_hz",        "gte"),   # value may be a note name
	"max_pitch":           ("pitch_hz",        "lte"),
	"min_quantized_beats": ("quantized_beats", "gte"),
	"max_quantized_beats": ("quantized_beats", "lte"),
}


# YAML keys for the numeric fields — the preferred new-form names.  pitch
# in YAML maps to the internal pitch_hz attribute (the _hz suffix makes
# units explicit in Python, awkward in user-facing YAML).
_NUMERIC_YAML_KEYS: dict[str, str] = {
	"duration":        "duration",
	"onsets":          "onsets",
	"tempo":           "tempo",
	"pitch":           "pitch_hz",
	"quantized_beats": "quantized_beats",
}


# ---------------------------------------------------------------------------
# Ordering — scorer registry + OrderClause
# ---------------------------------------------------------------------------

# External state passed to scorers.  Each scorer opts into whichever keys it
# needs; missing keys simply mean "this scorer can't run in this context"
# and the scorer returns None for all records.
_ExternalState = dict[str, typing.Any]


# A scorer is a pure function: (record, params, state) -> sortable float | None.
# None means "this scorer can't score this record"; the scorer's on_missing
# policy then decides whether the record is excluded from the result or
# sorted to the end.
_ScoreFn = typing.Callable[
	["subsample.library.SampleRecord", tuple[tuple[str, typing.Any], ...], _ExternalState],
	typing.Optional[float],
]


_OnMissing = typing.Literal["exclude", "sort_last"]


@dataclasses.dataclass(frozen=True)
class _ScorerSpec:

	"""Registry entry for a named scorer.

	on_missing:
	  "sort_last" — records whose score is None are kept in the result and
	                placed at the end of the sort, regardless of direction.
	                This matches the historical behaviour of
	                quantized_beats_*, where samples without a grid profile
	                sort last.
	  "exclude"   — records whose score is None are dropped from the result
	                entirely.  Used for scorers where "no score" means "not
	                eligible" (e.g. quantize_match on a non-quantized
	                sample)."""

	fn:         _ScoreFn
	on_missing: _OnMissing = "sort_last"


_SCORERS: dict[str, _ScorerSpec] = {}
"""Registered scorers keyed by their name (the ``by`` value in an
OrderClause).  Populated at module import; see _register_scorer() calls
below.  Module-private: plugin-style registration from user code is not
supported yet but the design allows it if needed later."""


def _register_scorer (
	name: str,
	fn: _ScoreFn,
	*,
	on_missing: _OnMissing = "sort_last",
) -> None:

	"""Register a named scorer for use in OrderClause.by."""

	if name in _SCORERS:
		raise ValueError(f"scorer already registered: {name!r}")
	_SCORERS[name] = _ScorerSpec(fn=fn, on_missing=on_missing)


# Per-sample field scorers — no external state required.
_register_scorer("duration", lambda r, _p, _s: float(r.duration))
_register_scorer("pitch",    lambda r, _p, _s: float(r.pitch.dominant_pitch_hz))
_register_scorer("onsets",   lambda r, _p, _s: float(r.rhythm.onset_count))
_register_scorer("tempo",    lambda r, _p, _s: float(r.rhythm.tempo_bpm))
_register_scorer("level",    lambda r, _p, _s: float(r.level.rms))
_register_scorer("age",      lambda r, _p, _s: float(r.sample_id))


def _beats_scorer (
	record:  "subsample.library.SampleRecord",
	_params: tuple[tuple[str, typing.Any], ...],
	state:   _ExternalState,
) -> typing.Optional[float]:

	"""Scorer for ``quantized_beats`` — reads external_state["beats_resolver"].

	Returns the sample's quantized beat count, or None when no variant /
	profile is available.  on_missing defaults to ``sort_last`` so
	non-quantized samples park at the end of the result rather than being
	dropped — matches the historical behaviour."""

	resolver = state.get("beats_resolver")
	if resolver is None:
		return None
	beats = resolver(record.sample_id)
	return None if beats is None else float(beats)


_register_scorer("quantized_beats", _beats_scorer, on_missing="sort_last")


# Legacy bare-string tokens translate into a single-clause order tuple.  The
# table keeps old YAML files working indefinitely; parse_select accepts these
# verbatim and converts them to OrderClause instances before query().
_LEGACY_ORDER_TOKENS: dict[str, "OrderClause"] = {}   # populated after OrderClause is defined


_VALID_ORDER_NAMES: frozenset[str] = frozenset()  # rebuilt after OrderClause; see below


# ---------------------------------------------------------------------------
# OrderClause + SelectSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OrderClause:

	"""One entry in the ``order:`` list of a MIDI map SelectSpec.

	by:     Scorer name (registered in _SCORERS, or the special token
	        "similarity" which is handled as a fast-path in query()).
	dir:    "asc" or "desc".
	params: Frozen key-value pairs for parameterised scorers.  Empty for
	        the built-in per-sample field scorers; populated for e.g.
	        ``{by: quantize_match, pattern: [1, 0, 1, 0, 1]}``."""

	by:     str
	dir:    typing.Literal["asc", "desc"] = "asc"
	params: tuple[tuple[str, typing.Any], ...] = ()


# Populate legacy token translations now that OrderClause exists.
_LEGACY_ORDER_TOKENS.update({
	"newest":               OrderClause(by="age",             dir="desc"),
	"oldest":               OrderClause(by="age",             dir="asc"),
	"duration_asc":         OrderClause(by="duration",        dir="asc"),
	"duration_desc":        OrderClause(by="duration",        dir="desc"),
	"pitch_asc":            OrderClause(by="pitch",           dir="asc"),
	"pitch_desc":           OrderClause(by="pitch",           dir="desc"),
	"onsets_asc":           OrderClause(by="onsets",          dir="asc"),
	"onsets_desc":          OrderClause(by="onsets",          dir="desc"),
	"tempo_asc":            OrderClause(by="tempo",           dir="asc"),
	"tempo_desc":           OrderClause(by="tempo",           dir="desc"),
	"loudest":              OrderClause(by="level",           dir="desc"),
	"quietest":             OrderClause(by="level",           dir="asc"),
	"similarity":           OrderClause(by="similarity",      dir="desc"),
	"quantized_beats_asc":  OrderClause(by="quantized_beats", dir="asc"),
	"quantized_beats_desc": OrderClause(by="quantized_beats", dir="desc"),
})


def _valid_order_names () -> frozenset[str]:

	"""Return the current set of valid ``by`` names — the registered
	scorers plus the special ``"similarity"`` token (handled as a fast
	path in query())."""

	return frozenset(_SCORERS.keys() | {"similarity"})


@dataclasses.dataclass(frozen=True)
class SelectSpec:

	"""Compiled selection criteria: filter → order → pick position.

	Parsed from the ``select:`` block in a MIDI map assignment.

	``order`` is a tuple of OrderClause (primary at index 0, secondary at
	1, …) — sort keys are composed across the tuple so equal primary
	values break ties on the secondary, and so on.  An empty tuple means
	"no explicit order"; query() defaults to newest-first (``age`` desc).
	"""

	where: WherePredicate              = dataclasses.field(default_factory=WherePredicate)
	order: tuple[OrderClause, ...]     = ()
	pick:  int                         = 1


# ---------------------------------------------------------------------------
# ProcessSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProcessorStep:

	"""A single processor declaration within a process pipeline.

	name:   Processor name (e.g. "repitch", "beat_quantize").
	params: Frozen key-value pairs (e.g. (("grid", 16), ("bpm", 120))).
	        Empty tuple for parameterless processors (e.g. "repitch: true").
	        Stored as a tuple of pairs so the frozen dataclass is truly hashable.
	"""

	name:   str
	params: tuple[tuple[str, typing.Any], ...] = ()

	def get (self, key: str, default: typing.Any = None) -> typing.Any:
		"""Look up a parameter by name, with a default."""
		for k, v in self.params:
			if k == key:
				return v
		return default


@dataclasses.dataclass(frozen=True)
class CcBinding:

	"""Maps a MIDI CC number to a numeric processor parameter.

	When a processor parameter value is a CcBinding (instead of a scalar),
	the actual value is resolved at note-on time from the current CC state.

	cc:       MIDI CC number (0–127).
	min_val:  Output value when CC = 0.
	max_val:  Output value when CC = 127.
	default:  Value before any CC is received. None → midpoint of min/max.
	channel:  MIDI channel (1–16, user-facing). None → omni (any channel).
	"""

	cc:      int
	min_val: float = 0.0
	max_val: float = 1.0
	default: typing.Optional[float] = None
	channel: typing.Optional[int]   = None

	@property
	def default_value (self) -> float:
		"""Return the default, falling back to the midpoint of the range."""
		if self.default is not None:
			return self.default
		return (self.min_val + self.max_val) / 2.0

	def resolve (self, cc_value: int) -> float:
		"""Map a CC value (0–127) to the output range."""
		return self.min_val + (cc_value / 127.0) * (self.max_val - self.min_val)


@dataclasses.dataclass(frozen=True)
class ProcessSpec:

	"""Ordered sequence of processors to apply after sample selection.

	Parsed from the ``process:`` list in a MIDI map assignment.
	An empty steps tuple means no processing — play the base variant.
	"""

	steps: tuple[ProcessorStep, ...] = ()

	def has_repitch (self) -> bool:
		"""True if any step is a repitch processor."""
		return any(s.name == "repitch" for s in self.steps)

	def has_beat_quantize (self) -> bool:
		"""True if any step is a beat_quantize processor."""
		return any(s.name == "beat_quantize" for s in self.steps)

	def has_pad_quantize (self) -> bool:
		"""True if any step is a pad_quantize processor."""
		return any(s.name == "pad_quantize" for s in self.steps)

	def has_vocoder (self) -> bool:
		"""True if any step is a vocoder processor."""
		return any(s.name == "vocoder" for s in self.steps)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Assignment:

	"""A compiled MIDI map entry for one or more notes.

	Combines selection criteria, processing pipeline, and playback/output
	settings.  Stored in the note map keyed by (mido_channel, midi_note).
	"""

	name:      str
	select:    tuple[SelectSpec, ...]
	process:   ProcessSpec        = dataclasses.field(default_factory=ProcessSpec)
	one_shot:  bool               = True
	gain_db:   float              = 0.0
	pan_weights:    typing.Optional[numpy.ndarray]  = None
	output_routing: typing.Optional[tuple[int, ...]] = None
	segment_mode:   typing.Union[str, int]             = ""
	"""Segment playback mode for quantized samples.
	"" = play entire merged audio (default).
	"round_robin" = cycle through segments sequentially.
	"random" = random segment each trigger.
	int (1-indexed) = always play that specific segment."""
	pick:           int                              = 1


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def query (
	select_spec:       SelectSpec,
	samples:           list["subsample.library.SampleRecord"],
	similarity_matrix: typing.Optional["subsample.similarity.SimilarityMatrix"] = None,
	beats_resolver:    typing.Optional[typing.Callable[[int], typing.Optional[float]]] = None,
) -> list["subsample.library.SampleRecord"]:

	"""Evaluate a SelectSpec against a list of samples.

	Applies filter predicates, sorts by the composed order clauses, and
	returns the full ranked list.  The caller picks the Nth position.

	When the *primary* order clause is ``{by: "similarity"}`` and
	``where.reference`` is set, the similarity matrix is consulted directly
	for a ranked list of sample IDs; ``where`` predicates are applied as
	post-filters on that ranked list, preserving similarity order.  Any
	secondary clauses after a primary ``similarity`` are ignored (the
	matrix returns unique scores; ties are not expected).  Using
	``similarity`` at a non-primary position raises ValueError — only the
	primary fast path is supported.

	For all other cases, the sort composes per-clause keys across the
	``order`` tuple.  Each scorer's ``on_missing`` policy determines
	whether records the scorer can't score are dropped from the result
	(``exclude``) or parked at the end (``sort_last``).

	Args:
		select_spec:       The selection criteria to evaluate.
		samples:           All instrument samples (from InstrumentLibrary.samples()).
		similarity_matrix: Required when the primary clause is ``{by: "similarity"}``
		                   with ``where.reference`` set.
		beats_resolver:    Callable returning the quantized beat count for a
		                   given sample_id, or None when no variant/profile is
		                   available.  Required by the ``quantized_beats``
		                   scorer and by ``where.min_quantized_beats`` /
		                   ``where.max_quantized_beats``.

	Returns:
		List of matching SampleRecord objects, ordered by the requested
		clauses.  Empty list if no samples match.
	"""

	where = select_spec.where
	state: _ExternalState = {
		"similarity_matrix": similarity_matrix,
		"beats_resolver":    beats_resolver,
	}

	# Default to newest-first when no explicit order clauses are given.
	clauses: tuple[OrderClause, ...] = select_spec.order
	if not clauses:
		clauses = (OrderClause(by="age", dir="desc"),)

	# Reject similarity at any non-primary position — the similarity matrix
	# returns a pre-ranked list; there is no per-sample score-against-a-
	# reference API to use for secondary ordering.
	for i, clause in enumerate(clauses):
		if clause.by == "similarity" and i > 0:
			raise ValueError(
				f"'similarity' is only supported as the primary order clause "
				f"(found at position {i})"
			)

	primary = clauses[0]

	# Similarity fast-path: primary clause is similarity + reference set.
	if primary.by == "similarity":
		if where.reference is None:
			raise ValueError(
				"'similarity' ordering requires where.reference to be set"
			)
		if similarity_matrix is None:
			return []

		ranked = similarity_matrix.get_matches(where.reference)
		by_id  = {r.sample_id: r for r in samples}

		result: list["subsample.library.SampleRecord"] = []
		for match in ranked:
			record = by_id.get(match.sample_id)
			if record is not None and where.matches(record, beats_resolver):
				result.append(record)

		if primary.dir == "asc":
			result.reverse()

		return result

	# General path: validate scorer names, filter, compose multi-key sort.
	valid_names = _valid_order_names()
	for clause in clauses:
		if clause.by not in valid_names:
			raise ValueError(
				f"Unknown order scorer {clause.by!r}.  "
				f"Valid scorers: {', '.join(sorted(valid_names))}"
			)

	filtered = [r for r in samples if where.matches(r, beats_resolver)]

	# Apply each "exclude"-policy scorer as an additional filter before
	# sorting.  A record failing any exclude-scorer drops from the result.
	for clause in clauses:
		spec = _SCORERS[clause.by]
		if spec.on_missing == "exclude":
			filtered = [
				r for r in filtered
				if spec.fn(r, clause.params, state) is not None
			]

	# Build sort key: tuple of per-clause (missing_flag, signed_value).
	# missing_flag is 0 for scored, 1 for None — 1 always sorts after 0
	# regardless of direction, matching the historical "unknown sorts last"
	# rule (relevant only to sort_last scorers; exclude scorers have
	# already dropped their None records).
	def _compose_key (
		record: "subsample.library.SampleRecord",
	) -> tuple[tuple[int, float], ...]:
		parts: list[tuple[int, float]] = []
		for clause in clauses:
			spec  = _SCORERS[clause.by]
			score = spec.fn(record, clause.params, state)
			if score is None:
				parts.append((1, 0.0))
			else:
				parts.append((0, -float(score) if clause.dir == "desc" else float(score)))
		return tuple(parts)

	filtered.sort(key=_compose_key)
	return filtered


# ---------------------------------------------------------------------------
# YAML parsing — select block
# ---------------------------------------------------------------------------

def _parse_where (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: pathlib.Path = pathlib.Path.cwd(),
) -> WherePredicate:

	"""Parse a ``where:`` dict from a MIDI map assignment into a WherePredicate.

	Accepts both the new per-field operator-dict form and the legacy
	``min_X``/``max_X`` bare-key form.  Within numeric fields, a bare
	scalar is shorthand for ``{eq: X}``.  Legacy and new forms on the
	same field raise ValueError — a cheap guard against mid-migration
	accidents (e.g. someone copying a new-form block on top of an old-
	form one).

	Args:
		raw:                The raw YAML value of the 'where' block.
		assignment_name:    Human-readable name of the assignment (for error messages).
		midi_map_dir:       Directory of the MIDI map file; used to resolve relative paths.
		                    Defaults to current working directory.
	"""

	if raw is None:
		return WherePredicate()

	if not isinstance(raw, dict):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'where' must be a mapping"
		)

	# Accumulated per-field operator kwargs, keyed by internal field name
	# (e.g. "duration", "pitch_hz").  Each value is a dict of
	# {operator_name: float_value} ready to splat into Range(**...).
	range_kwargs: dict[str, dict[str, float]] = {
		field: {} for field in _NUMERIC_FIELDS
	}

	# Fields whose values came from legacy min_X/max_X keys — used to
	# reject collisions with a new-form entry on the same field.
	touched_by_legacy: set[str] = set()
	touched_by_new:    set[str] = set()

	# Non-range predicate kwargs collected separately; constructed directly
	# into WherePredicate at the end.
	other_kwargs: dict[str, typing.Any] = {}

	for key, value in raw.items():

		# Legacy min_X / max_X keys: translate to (field, operator).
		if key in _LEGACY_WHERE_KEYS:
			field, op = _LEGACY_WHERE_KEYS[key]
			if field in touched_by_new:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: field "
					f"{field!r} has both legacy ({key!r}) and new-form "
					f"constraints — use one form or the other, not both."
				)
			range_kwargs[field][op] = _coerce_range_value(
				field, key, value, assignment_name,
			)
			touched_by_legacy.add(field)
			continue

		# New-form numeric field: duration / onsets / tempo / pitch / quantized_beats.
		if key in _NUMERIC_YAML_KEYS:
			field = _NUMERIC_YAML_KEYS[key]
			if field in touched_by_legacy:
				legacy_pair = [
					k for k, (f, _) in _LEGACY_WHERE_KEYS.items() if f == field
				]
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: field "
					f"{key!r} has both new-form and legacy "
					f"({'/'.join(sorted(legacy_pair))}) constraints — use "
					f"one form or the other, not both."
				)

			# Dict → operator block.  Scalar (int/float/str) → eq shorthand.
			if isinstance(value, dict):
				for op, op_value in value.items():
					if op not in _VALID_OPERATORS:
						raise ValueError(
							f"MIDI map assignment {assignment_name!r}: "
							f"unknown operator {op!r} under {key!r}.  "
							f"Valid operators: "
							f"{', '.join(sorted(_VALID_OPERATORS))}"
						)
					range_kwargs[field][op] = _coerce_range_value(
						field, f"{key}.{op}", op_value, assignment_name,
					)
			else:
				# Scalar shorthand for eq.
				range_kwargs[field]["eq"] = _coerce_range_value(
					field, key, value, assignment_name,
				)

			touched_by_new.add(field)
			continue

		# Non-range predicates — unchanged parsing.
		if key == "pitched":
			if not isinstance(value, bool):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: 'pitched' "
					f"must be true or false (got {value!r})"
				)
			other_kwargs["pitched"] = value

		elif key == "reference":
			ref = str(value)
			if is_path_like(ref):
				other_kwargs["reference"] = str((midi_map_dir / ref).resolve())
			else:
				other_kwargs["reference"] = ref

		elif key == "name":
			if "name" in other_kwargs or "name_path" in other_kwargs:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: 'name' and "
					f"'path' are mutually exclusive within a single where "
					f"block — use one, not both."
				)
			raw_name = str(value)
			if is_path_like(raw_name):
				# Legacy behaviour: a path-like `name:` value is treated as
				# an implicit path reference.  Preserved indefinitely; new
				# YAML should use the explicit `path:` key instead.
				other_kwargs["name"]      = pathlib.Path(raw_name).stem
				other_kwargs["name_path"] = str((midi_map_dir / raw_name).resolve())
			else:
				other_kwargs["name"] = raw_name

		elif key == "path":
			# Explicit path reference: load this exact WAV and match only it.
			# Preferred over the legacy `name: path/to/file` form.
			if "name" in other_kwargs or "name_path" in other_kwargs:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: 'name' and "
					f"'path' are mutually exclusive within a single where "
					f"block — use one, not both."
				)
			raw_path = str(value)
			other_kwargs["name"]      = pathlib.Path(raw_path).stem
			other_kwargs["name_path"] = str((midi_map_dir / raw_path).resolve())

		elif key == "directory":
			other_kwargs["directory"] = str((midi_map_dir / str(value)).resolve())

		else:
			if _STRICT_MODE:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: unknown "
					f"where-predicate key {key!r}.  Valid keys: "
					f"{', '.join(sorted(_valid_where_keys()))}."
				)
			_log.warning(
				"MIDI map assignment %r: unknown where predicate %r — ignored",
				assignment_name, key,
			)

	# Build the WherePredicate explicitly so mypy sees the field-name →
	# Range correspondence.  Empty Ranges (no operator set) default via
	# default_factory on the dataclass.
	def _range_for (field: str) -> Range:
		ops = range_kwargs[field]
		return Range(**ops) if ops else Range()

	return WherePredicate(
		duration        = _range_for("duration"),
		onsets          = _range_for("onsets"),
		tempo           = _range_for("tempo"),
		pitch_hz        = _range_for("pitch_hz"),
		quantized_beats = _range_for("quantized_beats"),
		**other_kwargs,
	)


def _coerce_range_value (
	field: str,
	source_key: str,
	value: typing.Any,
	assignment_name: str,
) -> float:

	"""Convert a raw YAML scalar into the float the Range slot expects.

	Handles the pitch-field note-name special case: under ``pitch`` or
	``min_pitch`` / ``max_pitch``, a string value is treated as a note
	name and converted to Hz via _note_name_to_hz.  Other fields require
	a numeric value.
	"""

	if field == "pitch_hz" and isinstance(value, str):
		return _note_name_to_hz(value)

	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: value for "
			f"{source_key!r} must be numeric (got {value!r})"
		) from exc


def _note_name_to_hz (name: str) -> float:

	"""Convert a note name (e.g. 'C3') to Hz for pitch filtering."""

	midi_note = pymididefs.notes.name_to_note(name)
	return float(librosa.midi_to_hz(midi_note))


def _parse_order_clause (
	raw: typing.Any,
	assignment_name: str,
) -> OrderClause:

	"""Parse one ``order:`` entry into an OrderClause.

	Accepts:
	  - A bare string (legacy token: ``duration_desc``, ``loudest``, …)
	    translated via _LEGACY_ORDER_TOKENS.
	  - A mapping with ``by`` (required), ``dir`` (optional, default
	    "asc"), and any extra keys treated as scorer params.
	"""

	if isinstance(raw, str):
		clause = _LEGACY_ORDER_TOKENS.get(raw)
		if clause is None:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: unknown order token "
				f"{raw!r}.  Valid legacy tokens: {', '.join(sorted(_LEGACY_ORDER_TOKENS))}"
			)
		return clause

	if not isinstance(raw, dict):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: order entry must be a "
			f"string (legacy token) or a mapping (got {type(raw).__name__})"
		)

	if "by" not in raw:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: order entry must have a 'by' key"
		)

	by      = str(raw["by"])
	dir_raw = str(raw.get("dir", "asc")).lower()

	if dir_raw not in ("asc", "desc"):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: order entry 'dir' must be "
			f"'asc' or 'desc' (got {dir_raw!r})"
		)

	# Everything except by/dir is a scorer parameter.  Preserve insertion order
	# by iterating the dict directly; values are kept as-is (the scorer decides
	# how to interpret them).
	params: list[tuple[str, typing.Any]] = [
		(str(k), v) for k, v in raw.items() if k not in ("by", "dir")
	]

	return OrderClause(
		by=by,
		dir=typing.cast(typing.Literal["asc", "desc"], dir_raw),
		params=tuple(params),
	)


def _parse_order (
	raw: typing.Any,
	assignment_name: str,
	*,
	key_name: str,
) -> tuple[OrderClause, ...]:

	"""Parse the ``order:`` (or legacy ``order_by:``) value into a tuple.

	Accepts:
	  - A bare string (legacy single-clause form).
	  - A single dict (new single-clause form).
	  - A list of strings and/or dicts (new multi-clause form; clauses
	    from any mix of legacy and new allowed).
	"""

	if isinstance(raw, (str, dict)):
		return (_parse_order_clause(raw, assignment_name),)

	if isinstance(raw, list):
		return tuple(_parse_order_clause(entry, assignment_name) for entry in raw)

	raise ValueError(
		f"MIDI map assignment {assignment_name!r}: {key_name!r} must be a "
		f"string, mapping, or list (got {type(raw).__name__})"
	)


def _parse_select_spec (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: pathlib.Path = pathlib.Path.cwd(),
) -> SelectSpec:

	"""Parse a single ``select:`` dict into a SelectSpec.

	Accepts both the new ``order:`` key (preferred) and the legacy
	``order_by:`` alias; raises ValueError if both are set on the same
	SelectSpec.  Within either key, both bare-string tokens
	(``duration_desc``) and structured clauses (``{by: duration, dir:
	desc}``) are accepted — the parser converts legacy tokens to
	OrderClause via _LEGACY_ORDER_TOKENS.

	Args:
		raw:             The raw YAML value of the 'select' entry.
		assignment_name: Human-readable name of the assignment (for error messages).
		midi_map_dir:    Directory of the MIDI map file; used to resolve relative paths.
		                 Defaults to current working directory.
	"""

	if not isinstance(raw, dict):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'select' entry must be a mapping"
		)

	where = _parse_where(raw.get("where"), assignment_name, midi_map_dir)

	has_order    = "order"    in raw
	has_order_by = "order_by" in raw

	if has_order and has_order_by:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: both 'order' and "
			f"'order_by' keys are set.  Use 'order' (preferred) or the legacy "
			f"'order_by' alias, not both."
		)

	order: tuple[OrderClause, ...]
	if has_order:
		order = _parse_order(raw["order"], assignment_name, key_name="order")
	elif has_order_by:
		order = _parse_order(raw["order_by"], assignment_name, key_name="order_by")
	else:
		# No explicit order.  Default to similarity when a reference is set
		# (preserves the historical auto-default), otherwise leave the tuple
		# empty and let query() apply its newest-first default.
		if where.reference is not None:
			order = (OrderClause(by="similarity", dir="desc"),)
			_log.info(
				"MIDI map assignment %r: auto-selected order "
				"[{by: similarity, dir: desc}] because 'where.reference' "
				"is set and no 'order' was given",
				assignment_name,
			)
		else:
			order = ()

	# Validate every scorer name up-front so errors surface at startup, not
	# at trigger time.  Use _valid_order_names() so newly-registered scorers
	# (e.g. future quantize_match) are recognised automatically.
	valid_names = _valid_order_names()
	for clause in order:
		if clause.by not in valid_names:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: unknown order "
				f"scorer {clause.by!r}.  Valid scorers: "
				f"{', '.join(sorted(valid_names))}"
			)

	pick = int(raw.get("pick", 1))

	if pick < 1:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pick must be >= 1 (got {pick})"
		)

	return SelectSpec(where=where, order=order, pick=pick)


def parse_select (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: pathlib.Path = pathlib.Path.cwd(),
) -> tuple[SelectSpec, ...]:

	"""Parse the ``select:`` block, which can be a single spec or a fallback list.

	Returns a tuple of SelectSpec objects.  At trigger time, each is tried in
	order; the first that returns a non-empty result wins.

	Args:
		raw:             The raw YAML value of the 'select' block.
		assignment_name: Human-readable name of the assignment (for error messages).
		midi_map_dir:    Directory of the MIDI map file; used to resolve relative paths.
		                 Defaults to current working directory.
	"""

	if isinstance(raw, dict):
		return (_parse_select_spec(raw, assignment_name, midi_map_dir),)

	if isinstance(raw, list):
		return tuple(_parse_select_spec(entry, assignment_name, midi_map_dir) for entry in raw)

	raise ValueError(
		f"MIDI map assignment {assignment_name!r}: 'select' must be a mapping or a list of mappings"
	)


# ---------------------------------------------------------------------------
# YAML parsing — process block
# ---------------------------------------------------------------------------

def parse_process (raw: typing.Any, assignment_name: str) -> ProcessSpec:

	"""Parse the ``process:`` block into a ProcessSpec.

	Accepts:
	  - None or missing → empty ProcessSpec (no processing)
	  - A list of processor declarations

	Each declaration is either:
	  - A string "name" (e.g. "repitch") — boolean processor, no params
	  - A dict {name: true} — boolean processor
	  - A dict {name: {param: value, ...}} — processor with params
	"""

	if raw is None:
		return ProcessSpec()

	if not isinstance(raw, list):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'process' must be a list"
		)

	steps: list[ProcessorStep] = []

	def _check_processor_name (name: str) -> None:
		if _STRICT_MODE and name not in _VALID_PROCESSOR_NAMES:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: unknown processor "
				f"{name!r}.  Valid processors: "
				f"{', '.join(sorted(_VALID_PROCESSOR_NAMES))}."
			)

	def _translate_legacy_processor (step: ProcessorStep) -> ProcessorStep:
		"""Canonicalise legacy processor names into the new form.

		Currently handles `hpss_harmonic` / `hpss_percussive` → `hpss` with
		`keep:` param (C1 in the language review).  Other legacy names pass
		through unchanged."""
		if step.name == "hpss_harmonic":
			return ProcessorStep(name="hpss", params=(("keep", "harmonic"),))
		if step.name == "hpss_percussive":
			return ProcessorStep(name="hpss", params=(("keep", "percussive"),))
		return step

	for entry in raw:

		if isinstance(entry, str):
			_check_processor_name(entry)
			steps.append(_translate_legacy_processor(ProcessorStep(name=entry)))

		elif isinstance(entry, dict):

			if len(entry) != 1:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: each process entry "
					f"must have exactly one key (got {list(entry.keys())})"
				)

			proc_name = next(iter(entry))
			proc_value = entry[proc_name]
			_check_processor_name(str(proc_name))

			if isinstance(proc_value, bool) or proc_value is None:
				# e.g. "repitch: true" or "repitch:"
				steps.append(_translate_legacy_processor(ProcessorStep(name=str(proc_name))))

			elif isinstance(proc_value, dict):
				# e.g. "beat_quantize: { grid: 16, bpm: 120 }"
				# Param values that are dicts with a "cc" key become CcBindings.
				resolved_params: list[tuple[str, typing.Any]] = []
				seen_params: set[str] = set()

				for k, v in proc_value.items():
					# Translate legacy param aliases (e.g. saturate.amount → drive).
					k_str = str(k)
					canonical = _LEGACY_PROCESSOR_PARAMS.get(
						(str(proc_name), k_str), k_str,
					)

					if canonical in seen_params:
						# Both "amount" (legacy) and "drive" (new) on one step
						# — reject loudly.  The legacy shim is a pure alias,
						# not a cumulative binding.
						raise ValueError(
							f"MIDI map assignment {assignment_name!r}: "
							f"processor {str(proc_name)!r} has duplicate "
							f"parameter {canonical!r} (possibly from mixing "
							f"legacy and new-form names) — use one, not both."
						)

					seen_params.add(canonical)

					if isinstance(v, dict) and "cc" in v:
						resolved_params.append((canonical, CcBinding(
							cc=int(v["cc"]),
							min_val=float(v.get("min", 0.0)),
							max_val=float(v.get("max", 1.0)),
							default=float(v["default"]) if "default" in v else None,
							channel=int(v["channel"]) if "channel" in v else None,
						)))
					else:
						resolved_params.append((canonical, v))

				frozen_params = tuple(resolved_params)
				steps.append(_translate_legacy_processor(
					ProcessorStep(name=str(proc_name), params=frozen_params),
				))

			else:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: process entry "
					f"{proc_name!r} has unsupported value type {type(proc_value).__name__}"
				)

		else:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: process entry must be "
				f"a string or a dict (got {type(entry).__name__})"
			)

	return ProcessSpec(steps=tuple(steps))
