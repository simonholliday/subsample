"""Sample query engine — filter, order, and select samples by analysis metadata.

Provides the select/process pipeline for the MIDI mapping system.  A MIDI map
assignment declares a SelectSpec (which sample to play) and a ProcessSpec
(how to present it).  The query engine evaluates SelectSpec against the
instrument library at trigger time, returning ranked sample IDs.

Architecture
------------

SelectSpec
    Parsed from the ``select:`` block in a MIDI map assignment.  Contains
    filter predicates (``where``), an ``order`` tuple of OrderClause (composed
    multi-key sort; ``order_by`` is only a legacy YAML alias), and a pick
    position.

ProcessSpec
    Parsed from the ``process:`` list.  Contains an ordered sequence of
    processor declarations (repitch, stretch_quantize, etc.) that map to
    TransformStep subclasses at execution time.

Assignment
    A compiled MIDI map entry combining SelectSpec, ProcessSpec, playback
    flags (one_shot, gain_db), and output routing (pan_weights).

Query evaluation
----------------

  InstrumentLibrary.samples()
      → filter by SelectSpec.where predicates (all must pass)
      → sort by the composed SelectSpec.order clauses
      → return ranked list
      → caller picks Nth position (or per-note rank distribution)

All filtering is a linear scan — with typical library sizes (500-1000 samples)
this is microseconds and needs no secondary indices.
"""

import dataclasses
import fnmatch
import logging
import pathlib
import random
import re
import typing

import librosa
import numpy
import pymididefs.notes

import subsample.analysis

if typing.TYPE_CHECKING:
	import subsample.library
	import subsample.similarity
	import subsample.transform

_log = logging.getLogger(__name__)


# Memo for resolved sample paths.  The ``directory:`` filter compares a
# sample's absolute, symlink-resolved path against the (already-resolved)
# target directory.  ``Path.resolve()`` issues a realpath() — one lstat
# syscall per path component — so resolving every candidate's path on every
# note-on was a syscall storm that dominated MIDI-to-audio latency (hundreds
# of thousands of lstat calls per second under a steady groove).  A sample's
# on-disk path is stable for the life of a session, so the resolution is
# memoised here.  dict get/set are atomic under the GIL and a duplicate
# concurrent resolve yields the same value, so no lock is needed.
#
# Entries for evicted samples are never individually removed (the library
# eviction path doesn't reach in here, and the recorder mints a fresh
# timestamped path per capture), so over a long headless-capture session this
# would otherwise grow with the cumulative number of distinct files seen, not
# the live library size.  A hard cap bounds it: when exceeded the memo is
# cleared wholesale (cheap to repopulate — one realpath per live sample).
_RESOLVED_PATH_CACHE_MAX: typing.Final[int] = 4096
_RESOLVED_PATH_CACHE: dict[pathlib.Path, pathlib.Path] = {}


def _resolved_sample_path (path: pathlib.Path) -> pathlib.Path:

	"""Return ``path`` resolved to absolute, symlink-free form, memoised.

	Keeps the per-note-on ``directory:`` filter off the filesystem — see the
	note on _RESOLVED_PATH_CACHE for why this matters on the trigger path.
	"""

	resolved = _RESOLVED_PATH_CACHE.get(path)

	if resolved is None:
		resolved = path.resolve()

		if len(_RESOLVED_PATH_CACHE) >= _RESOLVED_PATH_CACHE_MAX:
			_RESOLVED_PATH_CACHE.clear()

		_RESOLVED_PATH_CACHE[path] = resolved

	return resolved


# WherePredicate.name_regex stores its pattern as a raw string (so the frozen
# dataclass stays hashable), but matches() runs on the trigger path — so the
# compiled re.Pattern is memoised module-side here rather than recompiling, or
# re-hashing through re's own small wholesale-evicted internal cache, on every
# record.  Same GIL-atomic, cap-and-clear discipline as _RESOLVED_PATH_CACHE.
_COMPILED_REGEX_CACHE_MAX: typing.Final[int] = 256
_COMPILED_REGEX_CACHE: dict[str, "re.Pattern[str]"] = {}


def _compiled_name_regex (pattern: str) -> "re.Pattern[str]":

	"""Return the case-insensitive compiled form of ``pattern``, memoised.

	The pattern's syntax was already validated at parse time, so compilation
	here cannot raise for a parser-built predicate.
	"""

	compiled = _COMPILED_REGEX_CACHE.get(pattern)

	if compiled is None:
		compiled = re.compile(pattern, flags=re.IGNORECASE)

		if len(_COMPILED_REGEX_CACHE) >= _COMPILED_REGEX_CACHE_MAX:
			_COMPILED_REGEX_CACHE.clear()

		_COMPILED_REGEX_CACHE[pattern] = compiled

	return compiled


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

# Operators accepted under the dict form of `where.name:` (matches: glob,
# regex: re-module pattern).  Parallel to _VALID_OPERATORS, kept separate
# because the name predicate's value space is strings, not numbers.
_VALID_NAME_OPERATORS: frozenset[str] = frozenset({"matches", "regex"})


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
	"stretch_quantize",
	"beat_quantize",    # legacy; translated to stretch_quantize
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
	"bit_depth",
	"reshape",
	"transient",
	"vocoder",
})


# Processors that accept a bare scalar value as shorthand for their single
# defining parameter: `bit_depth: 12` ≡ `bit_depth: {bits: 12}`.  Maps
# processor name → parameter name the scalar binds to.
_SCALAR_PROCESSOR_PARAMS: dict[str, str] = {
	"bit_depth": "bits",
}


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
	("saturate",         "amount"): "drive",     # dB
	("transient",        "amount"): "gain",      # dB (signed)
	("stretch_quantize", "amount"): "strength",  # 0-1 fraction
	("pad_quantize",     "amount"): "strength",  # 0-1 fraction
	# `bpm` → `tempo` (C2): property name matches the where-predicate.
	("stretch_quantize", "bpm"):    "tempo",
	("pad_quantize",     "bpm"):    "tempo",
}
# Keyed on canonical processor names.  The parser translates legacy
# processor names (e.g. `beat_quantize` → `stretch_quantize`) before
# looking up param renames, so these entries do not need legacy-name
# duplicates.


def _canonical_processor_name (name: str) -> str:
	"""Translate a legacy processor name to its canonical form.

	Legacy aliases kept in the valid-names whitelist so strict mode
	accepts them; the parser canonicalises before building the
	ProcessorStep so downstream code (spec_from_process, ProcessSpec
	methods) only ever sees the new names.

	- hpss_harmonic / hpss_percussive → hpss (keep: injected at parse)
	- beat_quantize → stretch_quantize (pure name rename)
	"""
	if name in ("hpss_harmonic", "hpss_percussive"):
		return "hpss"
	if name == "beat_quantize":
		return "stretch_quantize"
	return name


def _hpss_keep_for_legacy_name (name: str) -> typing.Optional[str]:
	"""Return the `keep:` value implied by a legacy HPSS processor name.

	`hpss_harmonic` → "harmonic"; `hpss_percussive` → "percussive";
	anything else → None.  Used by parse_process() to inject the
	`keep:` param when a user writes the legacy bare name."""
	if name == "hpss_harmonic":
		return "harmonic"
	if name == "hpss_percussive":
		return "percussive"
	return None


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
	``reference``, ``name``, ``name_list``, ``name_glob``, ``name_regex``,
	``name_path``, ``directory``) remain flat Optionals — they aren't
	comparison predicates.

	The four name-matching forms (``name``, ``name_list``, ``name_glob``,
	``name_regex``) are mutually exclusive at parse time; at most one is
	ever populated for a given WherePredicate."""

	duration:        Range = dataclasses.field(default_factory=Range)
	onsets:          Range = dataclasses.field(default_factory=Range)
	tempo:           Range = dataclasses.field(default_factory=Range)
	pitch_hz:        Range = dataclasses.field(default_factory=Range)
	quantized_beats: Range = dataclasses.field(default_factory=Range)

	pitched:    typing.Optional[bool]              = None
	reference:  typing.Optional[str]               = None
	name:       typing.Optional[str]               = None
	name_list:  typing.Optional[tuple[str, ...]]   = None
	"""Set of bare stems to match against record.name (case-sensitive).
	Populated by `where: { name: [a, b, c] }`.  Tuple keeps the frozen
	dataclass hashable."""
	name_glob:  typing.Optional[str]               = None
	"""fnmatch-style glob pattern matched against record.name, full-string,
	case-insensitive.  Populated by `where: { name: { matches: ... } }`."""
	name_regex: typing.Optional[str]               = None
	"""re-module pattern matched against record.name via re.fullmatch with
	re.IGNORECASE.  Populated by `where: { name: { regex: ... } }`.  The
	raw string is stored (not a compiled Pattern) so the dataclass stays
	hashable; matches() resolves it to a memoised compiled Pattern via
	_compiled_name_regex so the trigger path doesn't recompile per record."""
	name_path: typing.Optional[str]  = None
	"""Internal field — set by parse_select for path-based name references;
	used by _resolve_path_references to load samples; never evaluated in
	matches() and never exposed in YAML."""
	directory: typing.Optional[str]  = None

	def __post_init__ (self) -> None:

		"""Enforce the name-form mutual exclusion the parser guarantees.

		matches() ANDs whichever name forms are populated, so a directly
		constructed predicate setting two of them would silently intersect
		rather than be rejected — guard it at construction instead.
		"""

		name_forms = sum(
			field is not None
			for field in (self.name, self.name_list, self.name_glob, self.name_regex)
		)

		if name_forms > 1:
			raise ValueError(
				"WherePredicate accepts at most one of name / name_list / "
				"name_glob / name_regex (they are mutually exclusive)"
			)

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

		if self.name_list is not None and record.name not in self.name_list:
			return False

		if self.name_glob is not None:
			if not fnmatch.fnmatchcase(record.name.lower(), self.name_glob.lower()):
				return False

		if self.name_regex is not None:
			if _compiled_name_regex(self.name_regex).fullmatch(record.name) is None:
				return False

		if self.directory is not None:
			if record.filepath is None:
				return False

			# self.directory is already absolute + resolved (parse_select stores
			# it that way), so only the sample path needs resolving — and that
			# is memoised, so a steady groove issues no filesystem syscalls here.
			try:
				_resolved_sample_path(record.filepath).relative_to(self.directory)
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


# ---------------------------------------------------------------------------
# beat_match — order by per-beat energy pattern similarity
# ---------------------------------------------------------------------------

def _downsample_to_beats (
	energy:     typing.Sequence[float],
	resolution: int,
) -> tuple[float, ...]:

	"""Collapse a GridEnergyProfile's per-slot energy to per-beat energy.

	`resolution` is the grid's slots-per-bar (4=quarters, 8=eighths, …).
	One beat = quarter-note = `resolution / 4` slots, so each beat's
	energy is the mean of `resolution / 4` consecutive slots.  For
	non-multiple-of-4 resolutions (triplets, 6, 5, …) we fall back to
	`numpy.array_split(energy, round(beat_count))` so buckets differ by
	at most one slot — a best-effort aggregation.

	Resolution-invariant: the same musical content quantized at 8ths vs
	16ths collapses to the same per-beat vector, which is what makes
	cross-grid comparison meaningful."""

	if len(energy) == 0 or resolution <= 0:
		return ()

	# Expected number of beats in the profile.
	beat_count = int(round(len(energy) * 4 / resolution))
	if beat_count <= 0:
		return ()

	arr = numpy.asarray(energy, dtype=numpy.float64)

	if resolution % 4 == 0 and len(arr) % (resolution // 4) == 0:
		# Clean divisibility — reshape + mean along the slot axis.
		slots_per_beat = resolution // 4
		reshaped = arr[: beat_count * slots_per_beat].reshape(beat_count, slots_per_beat)
		return tuple(float(x) for x in reshaped.mean(axis=1))

	# Fallback: best-effort bucket split.
	buckets = numpy.array_split(arr, beat_count)
	return tuple(float(b.mean()) if b.size else 0.0 for b in buckets)


def _cosine_similarity_truncated (
	a: typing.Sequence[float],
	b: typing.Sequence[float],
	k: int,
) -> float:

	"""Cosine similarity between the first `k` elements of `a` and `b`.

	Both inputs are assumed non-negative (pattern values in [0, 1],
	energy values in [0, 1]).  Returns 0.0 when either truncated vector
	has zero magnitude — tied-last behaviour, never NaN."""

	va = numpy.asarray(a[:k], dtype=numpy.float64)
	vb = numpy.asarray(b[:k], dtype=numpy.float64)
	na = float(numpy.linalg.norm(va))
	nb = float(numpy.linalg.norm(vb))
	if na == 0.0 or nb == 0.0:
		return 0.0
	return float(numpy.dot(va, vb) / (na * nb))


def _beat_match_scorer (
	record:  "subsample.library.SampleRecord",
	params:  tuple[tuple[str, typing.Any], ...],
	state:   _ExternalState,
) -> typing.Optional[float]:

	"""Scorer for ``beat_match`` — ranks by per-beat-energy similarity
	to a user-supplied pattern.

	Reads ``external_state["energy_profile_resolver"]`` to fetch the
	quantized variant's ``GridEnergyProfile`` for each sample.  Returns
	None (→ excluded) when no resolver is registered, when the sample
	has no quantized variant yet, or when the profile is empty.

	Metric: cosine similarity between the pattern and the per-beat
	downsampled profile, LHS-aligned over ``min(len(pattern),
	len(beats))`` elements.  Returns a value in ``[0, 1]`` — 1.0 is a
	perfect shape match."""

	resolver = state.get("energy_profile_resolver")
	if resolver is None:
		return None

	profile = resolver(record.sample_id)
	if profile is None or len(profile.energy) == 0:
		return None

	pattern = dict(params).get("pattern")
	if pattern is None:
		# Parser should have rejected this; defence in depth.
		return None

	beats = _downsample_to_beats(profile.energy, profile.resolution)
	k     = min(len(pattern), len(beats))
	if k == 0:
		return None

	return _cosine_similarity_truncated(pattern, beats, k)


_register_scorer("beat_match", _beat_match_scorer, on_missing="exclude")


# Legacy bare-string tokens translate into a single-clause order tuple.  The
# table keeps old YAML files working indefinitely; parse_select accepts these
# verbatim and converts them to OrderClause instances before query().
_LEGACY_ORDER_TOKENS: dict[str, "OrderClause"] = {}   # populated after OrderClause is defined



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
class PickSpec:

	"""How to pick a rank from the ranked match list.

	``pick: 1`` in YAML becomes ``PickSpec(1, 1)`` — always pick the best
	match.  ``pick: [1, 3]`` becomes ``PickSpec(1, 3)`` — draw a fresh
	random rank in [1, 3] inclusive on every note-on, giving a little
	variation across the top matches.  ``pick: {gte: 1, lte: 3}`` is the
	explicit dict form, identical to the list shorthand.

	An end is left *open* with ``null`` in the list (``pick: [2, null]`` —
	"rank 2 to the last match"; ``pick: [null, 5]`` — "the top 5"), with a
	one-sided dict (``pick: {gte: 2}``), or, for the whole list, the
	shortcut ``pick: any``.  An open bound is stored as ``None``: ``lo=None``
	means "from the best match", ``hi=None`` means "to the last match" — so
	there is never a magic sentinel number to stand in for "all of them".

	Bounds are 1-indexed and inclusive.  ``resolve_index`` resolves each
	open bound against the live match count and clamps an over-long upper
	bound down to it, so a range that runs past the end falls back to the
	last available match rather than failing.
	"""

	lo: typing.Optional[int]
	hi: typing.Optional[int]

	def resolve_index (self, ranked_len: int) -> int:

		"""Return a 0-indexed rank, drawn uniformly from [lo, hi].

		An open ``lo`` resolves to the best match (rank 1); an open ``hi``
		resolves to the last match (``ranked_len``).  A concrete upper bound
		past the end is clamped down to ``ranked_len`` so out-of-range picks
		land on the last available match.
		"""

		hi = ranked_len if self.hi is None else min(self.hi, ranked_len)
		lo = 1          if self.lo is None else self.lo
		lo = min(lo, hi)

		return random.randint(lo, hi) - 1


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
	pick:  PickSpec                    = dataclasses.field(default_factory=lambda: PickSpec(1, 1))


# Order/filter keys whose ranking is derived from transform variants, which
# finish baking asynchronously.  A select using any of these can re-rank
# between note-ons as variants become ready, so its result must not be cached
# — see MidiPlayer._rebuild_candidate_cache / _resolve_sample_id.  Every other
# selection key (directory, name, duration, pitch, similarity, age, …) is
# stable until the library itself changes.
_VARIANT_STATE_KEYS: typing.Final[frozenset[str]] = frozenset({
	"quantized_beats", "beat_match",
})


def select_uses_variant_state (select_specs: tuple[SelectSpec, ...]) -> bool:

	"""Return True if any select spec orders or filters by variant-derived state.

	``quantized_beats`` (as a ``where`` filter or an ``order`` key) and
	``beat_match`` (an ``order`` key) read the per-sample beat/energy profile
	of a transform variant.  Those variants are produced in the background, so
	the ranking shifts as they finish — such a select must be evaluated live on
	every trigger rather than pre-computed.  All other selects resolve to the
	same ranked list until the active library changes and are safe to cache.
	"""

	for spec in select_specs:

		if not spec.where.quantized_beats.is_empty():
			return True

		for clause in spec.order:
			if clause.by in _VARIANT_STATE_KEYS:
				return True

	return False


# ---------------------------------------------------------------------------
# ExtractSpec — channel-pattern extraction at playback time
# ---------------------------------------------------------------------------

# The named first/zero-order microphone-pattern extractions.  ``channel`` is
# the literal index escape hatch and is *not* a member of this set — it has
# its own dataclass field.
EXTRACT_KINDS: typing.Final[frozenset[str]] = frozenset({
	"omni", "side", "depth", "height", "left", "right", "front", "back",
})


@dataclasses.dataclass(frozen=True)
class ExtractSpec:

	"""How to extract a 1-channel sub-signal from a multi-channel input.

	An assignment with ``extract: omni`` collapses the input to mono using
	the equal-energy sum (M of M/S for stereo, W for B-format, ITU-like
	downmix for surround), then the existing pan/output routing distributes
	that mono signal across output channels.  Each ``kind`` corresponds to
	a microphone-pattern analogue:

	- ``omni``                       zero-order omnidirectional pickup
	- ``side`` / ``depth`` / ``height`` first-order figure-eight dipoles
	- ``left`` / ``right`` / ``front`` / ``back`` first-order cardioids
	- ``channel`` (with channel_index) literal Nth input channel
	"""

	kind:          str
	channel_index: typing.Optional[int] = None


# ---------------------------------------------------------------------------
# ProcessSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProcessorStep:

	"""A single processor declaration within a process pipeline.

	name:   Processor name (e.g. "repitch", "stretch_quantize").
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

	def has_stretch_quantize (self) -> bool:
		"""True if any step is a stretch_quantize processor."""
		return any(s.name == "stretch_quantize" for s in self.steps)

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
	settings.  Stored in the note map keyed by (mido_channel, midi_note);
	multiple Assignments may share a (channel, note) when each declares a
	distinct ``velocity_trigger`` range (velocity layering), or when they
	opt into ``stack`` to sound together (stacking).
	"""

	name:      str
	select:    tuple[SelectSpec, ...]
	process:   ProcessSpec        = dataclasses.field(default_factory=ProcessSpec)
	one_shot:  bool               = True
	gain_db:   float              = 0.0
	pan_weights:    typing.Optional[numpy.ndarray]  = None
	output_routing: typing.Optional[tuple[int, ...]] = None
	extract:        typing.Optional[ExtractSpec]    = None
	segment_mode:   typing.Union[str, int]             = ""
	"""Segment playback mode for quantized samples.
	"" = play entire merged audio (default).
	"round_robin" = cycle through segments sequentially.
	"random" = random segment each trigger.
	int (1-indexed) = always play that specific segment."""

	velocity_trigger:    tuple[int, int]                       = (0, 127)
	"""Velocity range (inclusive, 0-127) that triggers this assignment.
	Default (0, 127) — every velocity fires the assignment, identical to
	the pre-velocity-layering behaviour.  When two or more Assignments
	share a (channel, note), each must declare a non-overlapping trigger
	range; the player picks the matching layer at note-on by scanning
	the list for the range that contains ``msg.velocity``."""

	velocity_rescale_to: typing.Optional[tuple[int, int]]      = None
	"""Optional output range for in-band velocity rescaling.
	When None (default), the incoming velocity is used unchanged.  When
	set, the velocity is linearly remapped from ``velocity_trigger`` to
	this range before it reaches the gain calculation — so a layer that
	only sees velocities 0-63 can still play through a full 0-127 dynamic
	envelope.  Both bounds inclusive, 0-127."""

	stack: bool = False
	"""Opt in to stacking this sample with others on the same trigger.
	Default False — an assignment whose velocity range overlaps another on
	the same (channel, note) is rejected at load as a copy-paste mistake.
	Set True on *every* overlapping member to instead sound them together:
	at note-on the player fires all stacked layers that cover the incoming
	velocity, so e.g. a kick and a sub-sine play as one composite hit."""


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def query (
	select_spec:             SelectSpec,
	samples:                 list["subsample.library.SampleRecord"],
	similarity_matrix:       typing.Optional["subsample.similarity.SimilarityMatrix"] = None,
	beats_resolver:          typing.Optional[typing.Callable[[int], typing.Optional[float]]] = None,
	energy_profile_resolver: typing.Optional[
		typing.Callable[[int], typing.Optional["subsample.transform.GridEnergyProfile"]]
	] = None,
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
		energy_profile_resolver: Callable returning a sample's
		                   ``GridEnergyProfile`` (or None).  Required by the
		                   ``beat_match`` order scorer, which excludes every
		                   candidate when it is absent.

	Returns:
		List of matching SampleRecord objects, ordered by the requested
		clauses.  Empty list if no samples match.
	"""

	where = select_spec.where
	state: _ExternalState = {
		"similarity_matrix":       similarity_matrix,
		"beats_resolver":          beats_resolver,
		"energy_profile_resolver": energy_profile_resolver,
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

# Names of every WherePredicate field that participates in name matching
# (exact + list + glob + regex) plus the internal name_path used by path
# references.  The parser guards mutex against this set when handling
# either `name:` or `path:`.
_NAME_FORM_KWARGS: typing.Final[frozenset[str]] = frozenset(
	{"name", "name_list", "name_glob", "name_regex", "name_path"}
)


def _has_any_name_form (other_kwargs: dict[str, typing.Any]) -> bool:

	"""True if any of the mutually-exclusive name-form fields has been set
	on the parser's working kwargs dict."""

	return any(k in other_kwargs for k in _NAME_FORM_KWARGS)


def _parse_name_list (
	value: list[typing.Any],
	assignment_name: str,
) -> tuple[str, ...]:

	"""Parse a `where: { name: [a, b, c] }` list into a tuple of bare stems.

	Validates:
	  - Non-empty list.
	  - Every element is a string (not an int / dict / None — usually a typo).
	  - No duplicates (likely a typo).
	  - No path-like elements (call sites should use ``path:`` for path
	    references; the new list form is pure stem matching).
	"""

	if not value:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'name' list must be "
			f"non-empty"
		)

	stems: list[str] = []

	for element in value:
		if not isinstance(element, str):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'name' list "
				f"entries must be strings, got {element!r} "
				f"({type(element).__name__})"
			)

		if is_path_like(element):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'name' list "
				f"entries must be bare stems (no slashes or leading dots); "
				f"got {element!r}.  Use the 'path:' key for individual "
				f"file references."
			)

		if element in stems:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'name' list "
				f"contains duplicate entry {element!r}"
			)

		stems.append(element)

	return tuple(stems)


def _parse_name_operator_dict (
	value: dict[str, typing.Any],
	assignment_name: str,
) -> tuple[str, str]:

	"""Parse a `where: { name: { matches: ... } }` or `{ regex: ... }` dict.

	Returns ``(field_name, raw_pattern)`` where ``field_name`` is either
	``"name_glob"`` or ``"name_regex"`` so the caller can stash the result
	straight into the parser's other_kwargs.

	Validates:
	  - Exactly one operator key (multiple operators would conflict and
	    aren't meaningfully combinable for this predicate).
	  - Operator key in ``_VALID_NAME_OPERATORS``.
	  - Operator value is a non-empty string.
	  - Pattern is not path-like (patterns match the stem only).
	  - For ``regex:``, the pattern compiles (surfaces syntax errors at
	    parse time rather than the first time the assignment is queried).
	"""

	if len(value) == 0:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'name' dict must "
			f"contain exactly one operator "
			f"({', '.join(sorted(_VALID_NAME_OPERATORS))})"
		)

	if len(value) > 1:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'name' dict must "
			f"contain exactly one operator, got "
			f"{sorted(value.keys())}.  Use a fallback chain "
			f"(select: list of specs) if you need multiple patterns."
		)

	op, op_value = next(iter(value.items()))

	if op not in _VALID_NAME_OPERATORS:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: unknown operator "
			f"{op!r} under 'name'.  Valid operators: "
			f"{', '.join(sorted(_VALID_NAME_OPERATORS))}"
		)

	if not isinstance(op_value, str):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'name.{op}' value "
			f"must be a string, got {op_value!r} ({type(op_value).__name__})"
		)

	if op_value == "":
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'name.{op}' pattern "
			f"must be non-empty"
		)

	if is_path_like(op_value):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'name.{op}' pattern "
			f"is path-like ({op_value!r}); patterns match the filename "
			f"stem only.  Use 'directory:' for directory containment or "
			f"'path:' for an individual file."
		)

	if op == "regex":
		try:
			re.compile(op_value)
		except re.error as exc:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'name.regex' "
				f"pattern {op_value!r} is not a valid regular expression: {exc}"
			) from exc

		return ("name_regex", op_value)

	# op == "matches"
	return ("name_glob", op_value)


def _parse_where (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: typing.Optional[pathlib.Path] = None,   # resolved to cwd in _parse_where
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
		                    None (the default) resolves to the current working
		                    directory at call time — not at import, so a later
		                    cwd change is honoured.
	"""

	# Resolve the None sentinel here (per call), so the base directory is never
	# frozen to the cwd captured at module import.
	if midi_map_dir is None:
		midi_map_dir = pathlib.Path.cwd()

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
			if _has_any_name_form(other_kwargs):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: only one "
					f"'name' form (exact, list, matches, regex) or 'path' "
					f"may be used per where block."
				)

			if isinstance(value, list):
				other_kwargs["name_list"] = _parse_name_list(value, assignment_name)
			elif isinstance(value, dict):
				field_name, raw_pattern = _parse_name_operator_dict(value, assignment_name)
				other_kwargs[field_name] = raw_pattern
			else:
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
			if _has_any_name_form(other_kwargs):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: only one "
					f"'name' form (exact, list, matches, regex) or 'path' "
					f"may be used per where block."
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
	# how to interpret them), then any per-scorer validators run below to
	# coerce mutable containers (lists) into hashable form.
	params: list[tuple[str, typing.Any]] = [
		(str(k), v) for k, v in raw.items() if k not in ("by", "dir")
	]

	if by == "beat_match":
		params = _validate_beat_match_params(params, assignment_name)

	return OrderClause(
		by=by,
		dir=typing.cast(typing.Literal["asc", "desc"], dir_raw),
		params=tuple(params),
	)


def _validate_beat_match_params (
	params: list[tuple[str, typing.Any]],
	assignment_name: str,
) -> list[tuple[str, typing.Any]]:

	"""Validate and normalise the `pattern:` param for `beat_match`.

	Returns a new params list where the `pattern` value is a tuple of
	floats (hashable so the OrderClause stays hashable).  Raises
	ValueError at parse time for malformed inputs — empty list, length
	< 2, non-numeric elements, or values outside [0, 1]."""

	out: list[tuple[str, typing.Any]] = []
	found_pattern = False

	for k, v in params:

		if k != "pattern":
			out.append((k, v))
			continue

		found_pattern = True

		if not isinstance(v, list):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: beat_match "
				f"'pattern' must be a list of numbers (got {type(v).__name__})"
			)

		if len(v) < 2:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: beat_match "
				f"'pattern' must have at least 2 elements (got {len(v)})"
			)

		coerced: list[float] = []
		for i, elem in enumerate(v):
			# bool is a subclass of int in Python; accept it (True→1.0, False→0.0)
			# but reject strings and other types explicitly.
			if not isinstance(elem, (int, float)):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: beat_match "
					f"'pattern' element #{i} must be a number in [0, 1] "
					f"(got {elem!r})"
				)
			f = float(elem)
			if f < 0.0 or f > 1.0:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: beat_match "
					f"'pattern' element #{i} ({f}) is outside [0, 1]"
				)
			coerced.append(f)

		# Store as tuple for hashability (OrderClause is frozen and must
		# be hashable; a list inside the tuple would break that).
		out.append(("pattern", tuple(coerced)))

	if not found_pattern:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: beat_match requires "
			f"a 'pattern' parameter (a list of numbers in [0, 1])"
		)

	return out


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


_VALID_PICK_OPERATORS: typing.Final[frozenset[str]] = frozenset({"gte", "lte", "gt", "lt", "eq"})


def _parse_pick (raw: typing.Any, assignment_name: str) -> PickSpec:

	"""Parse the ``pick:`` value from a select spec.

	Accepted forms:
	  - missing / None      → PickSpec(1, 1)      (best match)
	  - "any"               → PickSpec(None, None) (uniform draw across all matches)
	  - int n               → PickSpec(n, n)      (exact rank, n >= 1)
	  - [lo, hi]            → PickSpec(lo, hi)    (random rank in range, both >= 1, lo <= hi)
	  - [lo, null] / [null, hi] → open-ended range (None = best match / last match)
	  - {gte, lte, gt, lt, eq: ...} → equivalent range; a lower-bound-only dict
	    (gte / gt) leaves the upper bound open (hi=None)

	The range form re-rolls on every note-on (see ``PickSpec.resolve_index``).
	"""

	if raw is None:
		return PickSpec(1, 1)

	# "any" is the read-aloud shortcut for the fully-open range [None, None] —
	# a uniform draw across every match, no magic count to write.
	if isinstance(raw, str):
		if raw.strip().lower() == "any":
			return PickSpec(None, None)

		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'pick' string must be "
			f"'any' (got {raw!r})"
		)

	# bool is a subclass of int in Python — reject explicitly so 'pick: true'
	# doesn't silently become PickSpec(1, 1) and 'pick: false' PickSpec(0, 0).
	if isinstance(raw, bool):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'pick' must be a positive "
			f"integer, a 2-element list, or an operator mapping (got bool)"
		)

	if isinstance(raw, int):
		if raw < 1:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pick must be >= 1 (got {raw})"
			)
		return PickSpec(raw, raw)

	if isinstance(raw, list):
		# Each entry is an integer rank or null (an open end).  bool is a
		# subclass of int, so reject it explicitly the same way the scalar
		# form does.
		def _is_bound (x: typing.Any) -> bool:
			return x is None or (isinstance(x, int) and not isinstance(x, bool))

		if len(raw) != 2 or not all(_is_bound(x) for x in raw):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'pick' list must be "
				f"two entries [lo, hi], each a positive integer or null for an "
				f"open end (got {raw!r})"
			)

		lo, hi = raw

		if (lo is not None and lo < 1) or (hi is not None and hi < 1):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pick bounds must be "
				f">= 1 (got [{lo}, {hi}])"
			)

		if lo is not None and hi is not None and lo > hi:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pick 'lo' must be "
				f"<= 'hi' (got [{lo}, {hi}])"
			)

		return PickSpec(lo, hi)

	if isinstance(raw, dict):
		unknown = set(raw.keys()) - _VALID_PICK_OPERATORS

		if unknown and _STRICT_MODE:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: unknown 'pick' "
				f"operator(s) {sorted(unknown)}.  Valid operators: "
				f"{', '.join(sorted(_VALID_PICK_OPERATORS))}"
			)

		# eq pins both bounds; other operators define lo (gte/gt) and hi (lte/lt).
		# A lower-bound-only dict (gte / gt) leaves the upper bound open — "rank
		# N to the last match" — stored as hi=None.
		eq_val  = raw.get("eq")
		gte_val = raw.get("gte")
		gt_val  = raw.get("gt")
		lte_val = raw.get("lte")
		lt_val  = raw.get("lt")

		# An operator-less dict has no bounds to act on.  Reject it so a bare
		# `pick: {}` doesn't silently mean "any" — write `pick: any` for that.
		if all(v is None for v in (eq_val, gte_val, gt_val, lte_val, lt_val)):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'pick' dict must "
				f"include at least one of "
				f"{', '.join(sorted(_VALID_PICK_OPERATORS))}; got {raw!r}"
			)

		if eq_val is not None:
			if not isinstance(eq_val, int) or isinstance(eq_val, bool) or eq_val < 1:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: pick 'eq' must be "
					f"a positive integer (got {eq_val!r})"
				)
			return PickSpec(eq_val, eq_val)

		dict_lo: int = 1

		if gte_val is not None:
			if not isinstance(gte_val, int) or isinstance(gte_val, bool):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: pick 'gte' must "
					f"be an integer (got {gte_val!r})"
				)
			dict_lo = gte_val
		elif gt_val is not None:
			if not isinstance(gt_val, int) or isinstance(gt_val, bool):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: pick 'gt' must "
					f"be an integer (got {gt_val!r})"
				)
			dict_lo = gt_val + 1

		dict_hi: typing.Optional[int] = None

		if lte_val is not None:
			if not isinstance(lte_val, int) or isinstance(lte_val, bool):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: pick 'lte' must "
					f"be an integer (got {lte_val!r})"
				)
			dict_hi = lte_val
		elif lt_val is not None:
			if not isinstance(lt_val, int) or isinstance(lt_val, bool):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: pick 'lt' must "
					f"be an integer (got {lt_val!r})"
				)
			dict_hi = lt_val - 1

		# dict_hi left as None means an open upper bound (lower-bound-only dict).

		if dict_lo < 1 or (dict_hi is not None and dict_hi < 1):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pick bounds must be "
				f">= 1 (got lo={dict_lo}, hi={dict_hi})"
			)

		if dict_hi is not None and dict_lo > dict_hi:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pick 'lo' must be "
				f"<= 'hi' (got lo={dict_lo}, hi={dict_hi})"
			)

		return PickSpec(dict_lo, dict_hi)

	raise ValueError(
		f"MIDI map assignment {assignment_name!r}: 'pick' must be 'any', an "
		f"integer, a 2-element list, or an operator mapping "
		f"(got {type(raw).__name__})"
	)


def _parse_select_spec (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: typing.Optional[pathlib.Path] = None,   # resolved to cwd in _parse_where
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

	pick = _parse_pick(raw.get("pick"), assignment_name)

	return SelectSpec(where=where, order=order, pick=pick)


def parse_select (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: typing.Optional[pathlib.Path] = None,   # resolved to cwd in _parse_where
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
		# An empty fallback chain can never match a sample; reject it loudly
		# rather than letting the assignment silently map nothing.
		if not raw:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'select' list must "
				f"contain at least one spec"
			)

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
	  - A dict {name: scalar} — shorthand for the processor's single
	    defining parameter (only for names in _SCALAR_PROCESSOR_PARAMS,
	    e.g. `bit_depth: 12`)
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

	def _build_parameterless_step (raw_name: str) -> ProcessorStep:
		"""Build a ProcessorStep for a bare / bool processor entry.

		Canonicalises the name and, for legacy HPSS aliases, injects the
		`keep:` param that the canonical `hpss` processor requires."""
		canonical = _canonical_processor_name(raw_name)
		keep = _hpss_keep_for_legacy_name(raw_name)
		if keep is not None:
			return ProcessorStep(name=canonical, params=(("keep", keep),))
		return ProcessorStep(name=canonical)

	for entry in raw:

		if isinstance(entry, str):
			_check_processor_name(entry)
			steps.append(_build_parameterless_step(entry))

		elif isinstance(entry, dict):

			if len(entry) != 1:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: each process entry "
					f"must have exactly one key (got {list(entry.keys())})"
				)

			proc_name = next(iter(entry))
			proc_value = entry[proc_name]
			proc_name_str = str(proc_name)
			_check_processor_name(proc_name_str)

			# Canonicalise the processor name up-front so legacy param
			# lookups (_LEGACY_PROCESSOR_PARAMS) use the new name as key.
			canonical_name = _canonical_processor_name(proc_name_str)

			if isinstance(proc_value, bool) or proc_value is None:
				# e.g. "repitch: true" or "repitch:"
				steps.append(_build_parameterless_step(proc_name_str))

			elif canonical_name in _SCALAR_PROCESSOR_PARAMS and isinstance(proc_value, (int, float)):
				# e.g. "bit_depth: 12" — scalar shorthand for the single
				# defining parameter.  (bool is consumed by the branch above.)
				steps.append(ProcessorStep(
					name=canonical_name,
					params=((_SCALAR_PROCESSOR_PARAMS[canonical_name], proc_value),),
				))

			elif isinstance(proc_value, dict):
				# e.g. "stretch_quantize: { grid: 16, tempo: 120 }"
				# Param values that are dicts with a "cc" key become CcBindings.
				resolved_params: list[tuple[str, typing.Any]] = []
				seen_params: set[str] = set()

				for k, v in proc_value.items():
					# Translate legacy param aliases (e.g. saturate.amount → drive)
					# using the canonical processor name as the lookup key.
					k_str = str(k)
					canonical_param = _LEGACY_PROCESSOR_PARAMS.get(
						(canonical_name, k_str), k_str,
					)

					if canonical_param in seen_params:
						# Both "amount" (legacy) and "drive" (new) on one step
						# — reject loudly.  The legacy shim is a pure alias,
						# not a cumulative binding.
						raise ValueError(
							f"MIDI map assignment {assignment_name!r}: "
							f"processor {proc_name_str!r} has duplicate "
							f"parameter {canonical_param!r} (possibly from "
							f"mixing legacy and new-form names) — use one, "
							f"not both."
						)

					seen_params.add(canonical_param)

					if isinstance(v, dict) and "cc" in v:
						resolved_params.append((canonical_param, CcBinding(
							cc=int(v["cc"]),
							min_val=float(v.get("min", 0.0)),
							max_val=float(v.get("max", 1.0)),
							default=float(v["default"]) if "default" in v else None,
							channel=int(v["channel"]) if "channel" in v else None,
						)))
					else:
						resolved_params.append((canonical_param, v))

				# Inject the HPSS `keep:` param for legacy dict-form entries
				# (e.g. `hpss_harmonic: {}`).  Silently preserves any user-
				# supplied `keep:` value in the rare dict-form legacy case.
				keep = _hpss_keep_for_legacy_name(proc_name_str)
				if keep is not None and "keep" not in seen_params:
					resolved_params.insert(0, ("keep", keep))

				steps.append(ProcessorStep(
					name=canonical_name,
					params=tuple(resolved_params),
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

	# hpss needs a valid `keep:` to build a transform step.  Validate here —
	# at trigger time the same check raises inside the rtmidi handler on
	# EVERY note-on, and aborts the whole variant pre-compute pass; every
	# other config error in the map already surfaces at parse time.
	for step in steps:
		if step.name == "hpss":
			keep = step.get("keep", "")

			if keep not in ("harmonic", "percussive"):
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: hpss requires "
					f"keep: harmonic or keep: percussive (got {keep!r})"
				)

	# bit_depth needs whole-number bits in 1–16 to build a transform step.
	# Validate plain values here, at load, like the hpss check above.  A
	# CcBinding resolves per-trigger and is clamped in spec_from_process;
	# an absent param falls back to the BitDepth default (12).
	for step in steps:
		if step.name == "bit_depth":
			bits = step.get("bits")

			if bits is not None and not isinstance(bits, CcBinding):
				if isinstance(bits, bool) or not isinstance(bits, int) or not (1 <= bits <= 16):
					raise ValueError(
						f"MIDI map assignment {assignment_name!r}: bit_depth "
						f"requires a whole number of bits from 1 to 16 "
						f"(got {bits!r})"
					)

			# dither: bool shorthand (true → triangular) or a named type.
			dither = step.get("dither")

			if dither is not None and not isinstance(dither, bool):
				if not isinstance(dither, str) or dither.lower() not in ("none", "triangular", "rectangular"):
					raise ValueError(
						f"MIDI map assignment {assignment_name!r}: bit_depth "
						f"dither must be true, false, triangular, or "
						f"rectangular (got {dither!r})"
					)

	# A process chain may carry at most ONE beat-aligning step.  Combining
	# stretch_quantize with pad_quantize — or repeating either — is ambiguous
	# (their tempo/grid parameters would fight at trigger time) and almost
	# certainly a mistake; reject it at load.
	quantize_names = [s.name for s in steps if s.name in ("stretch_quantize", "pad_quantize")]

	if len(quantize_names) > 1:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: "
			f"{' and '.join(repr(n) for n in quantize_names[:2])} cannot be "
			f"combined in one process chain — a chain may beat-align a sample "
			f"only once.  Keep a single quantize step."
		)

	return ProcessSpec(steps=tuple(steps))
