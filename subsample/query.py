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
    flags (one_shot, gain_db), and output routing (pan_gains).

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
# Filter predicates
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WherePredicate:

	"""Filter criteria for sample selection.  All fields use None = no filter."""

	min_duration:  typing.Optional[float] = None
	max_duration:  typing.Optional[float] = None
	min_onsets:    typing.Optional[int]   = None
	max_onsets:    typing.Optional[int]   = None
	pitched:       typing.Optional[bool]  = None
	min_tempo:     typing.Optional[float] = None
	max_tempo:     typing.Optional[float] = None
	min_pitch_hz:  typing.Optional[float] = None
	max_pitch_hz:  typing.Optional[float] = None
	reference:     typing.Optional[str]   = None
	name:          typing.Optional[str]   = None
	name_path:     typing.Optional[str]   = None  # Resolved absolute path for path-based name predicates
	directory:     typing.Optional[str]   = None  # Resolved absolute path; filters to samples from this directory

	def matches (self, record: "subsample.library.SampleRecord") -> bool:

		"""Return True if the record passes all active filter predicates."""

		if self.min_duration is not None and record.duration < self.min_duration:
			return False

		if self.max_duration is not None and record.duration > self.max_duration:
			return False

		if self.min_onsets is not None and record.rhythm.onset_count < self.min_onsets:
			return False

		if self.max_onsets is not None and record.rhythm.onset_count > self.max_onsets:
			return False

		if self.pitched is not None:
			is_pitched = subsample.analysis.has_stable_pitch(
				record.spectral, record.pitch, record.duration,
			)

			if self.pitched != is_pitched:
				return False

		if self.min_tempo is not None and record.rhythm.tempo_bpm < self.min_tempo:
			return False

		if self.max_tempo is not None and record.rhythm.tempo_bpm > self.max_tempo:
			return False

		if self.min_pitch_hz is not None and record.pitch.dominant_pitch_hz < self.min_pitch_hz:
			return False

		if self.max_pitch_hz is not None and record.pitch.dominant_pitch_hz > self.max_pitch_hz:
			return False

		if self.name is not None and record.name != self.name:
			return False

		if self.directory is not None:
			if record.filepath is None:
				return False

			try:
				pathlib.Path(record.filepath).resolve().relative_to(pathlib.Path(self.directory).resolve())
			except ValueError:
				return False

		# reference is handled externally by the query runner (needs SimilarityMatrix).
		# WherePredicate.matches() is a record-level filter; reference scoring
		# requires the full ranked list, so it's applied in query().

		return True


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

# Valid order_by values and their sort key + reverse flag.
# Each value maps to (key_function, reverse).
# key_function takes a SampleRecord and returns a sortable value.

_ORDER_BY_KEYS: dict[str, tuple[typing.Callable[["subsample.library.SampleRecord"], typing.Any], bool]] = {
	"newest":        (lambda r: r.sample_id, True),
	"oldest":        (lambda r: r.sample_id, False),
	"duration_asc":  (lambda r: r.duration,  False),
	"duration_desc": (lambda r: r.duration,  True),
	"pitch_asc":     (lambda r: r.pitch.dominant_pitch_hz, False),
	"pitch_desc":    (lambda r: r.pitch.dominant_pitch_hz, True),
	"onsets_asc":    (lambda r: r.rhythm.onset_count, False),
	"onsets_desc":   (lambda r: r.rhythm.onset_count, True),
	"tempo_asc":     (lambda r: r.rhythm.tempo_bpm, False),
	"tempo_desc":    (lambda r: r.rhythm.tempo_bpm, True),
	"loudest":       (lambda r: r.level.rms, True),
	"quietest":      (lambda r: r.level.rms, False),
}

VALID_ORDER_BY: frozenset[str] = frozenset(_ORDER_BY_KEYS.keys() | {"similarity"})


# ---------------------------------------------------------------------------
# SelectSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SelectSpec:

	"""Compiled selection criteria: filter → order → pick position.

	Parsed from the ``select:`` block in a MIDI map assignment.
	"""

	where:    WherePredicate = dataclasses.field(default_factory=WherePredicate)
	order_by: str            = "newest"
	pick:     int            = 1


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
	pan_gains: numpy.ndarray      = dataclasses.field(
		default_factory=lambda: numpy.array([0.7071068, 0.7071068], dtype=numpy.float32),
	)
	pick:      int                = 1


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def query (
	select_spec:       SelectSpec,
	samples:           list["subsample.library.SampleRecord"],
	similarity_matrix: typing.Optional["subsample.similarity.SimilarityMatrix"] = None,
) -> list["subsample.library.SampleRecord"]:

	"""Evaluate a SelectSpec against a list of samples.

	Applies filter predicates, sorts by the requested ordering, and returns
	the full ranked list.  The caller picks the Nth position.

	When the ``where`` clause includes a ``reference`` predicate and
	``order_by`` is ``"similarity"``, the similarity matrix is used to
	produce a ranked list of sample IDs that match the reference.  The
	``where`` predicates are then applied as post-filters on that ranked
	list, preserving similarity order.

	Args:
		select_spec:       The selection criteria to evaluate.
		samples:           All instrument samples (from InstrumentLibrary.samples()).
		similarity_matrix: Required when ``where.reference`` is set and
		                   ``order_by`` is ``"similarity"``.

	Returns:
		List of matching SampleRecord objects, ordered by the requested key.
		Empty list if no samples match.
	"""

	where = select_spec.where

	# Reference + similarity ordering: use the pre-computed ranked list from
	# the similarity matrix, then post-filter by the remaining predicates.
	if where.reference is not None and select_spec.order_by == "similarity":
		if similarity_matrix is None:
			return []

		ranked = similarity_matrix.get_matches(where.reference)

		# Build a sample_id → record lookup from the full list.
		by_id = {r.sample_id: r for r in samples}

		result: list["subsample.library.SampleRecord"] = []

		for match in ranked:
			record = by_id.get(match.sample_id)

			if record is not None and where.matches(record):
				result.append(record)

		return result

	# General case: filter → sort.
	filtered = [r for r in samples if where.matches(r)]

	order_entry = _ORDER_BY_KEYS.get(select_spec.order_by)

	if order_entry is not None:
		key_fn, reverse = order_entry
		filtered.sort(key=key_fn, reverse=reverse)

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

	kwargs: dict[str, typing.Any] = {}

	for key, value in raw.items():

		if key in ("min_duration", "max_duration"):
			kwargs[key] = float(value)

		elif key in ("min_onsets", "max_onsets"):
			kwargs[key] = int(value)

		elif key == "pitched":
			kwargs["pitched"] = bool(value)

		elif key in ("min_tempo", "max_tempo"):
			kwargs[key] = float(value)

		elif key in ("min_pitch", "max_pitch"):
			# Accept Hz (float) or note name (string).
			if isinstance(value, str):
				hz = _note_name_to_hz(value)
				kwargs[key + "_hz"] = hz
			else:
				kwargs[key + "_hz"] = float(value)

		elif key == "reference":
			ref = str(value)
			if is_path_like(ref):
				# Path-based reference: resolve to absolute path (used as matrix key)
				kwargs["reference"] = str((midi_map_dir / ref).resolve())
			else:
				# Bare name: keep as-is; case-insensitive lookup happens at query time
				kwargs["reference"] = ref

		elif key == "name":
			raw_name = str(value)
			if is_path_like(raw_name):
				# Path-based name: store the stem in 'name', resolved path in 'name_path'
				kwargs["name"] = pathlib.Path(raw_name).stem
				kwargs["name_path"] = str((midi_map_dir / raw_name).resolve())
			else:
				kwargs["name"] = raw_name

		elif key == "directory":
			kwargs["directory"] = str((midi_map_dir / str(value)).resolve())

		else:
			_log.warning(
				"MIDI map assignment %r: unknown where predicate %r — ignored",
				assignment_name, key,
			)

	return WherePredicate(**kwargs)


def _note_name_to_hz (name: str) -> float:

	"""Convert a note name (e.g. 'C3') to Hz for pitch filtering."""

	midi_note = pymididefs.notes.name_to_note(name)
	return float(librosa.midi_to_hz(midi_note))


def _parse_select_spec (
	raw: typing.Any,
	assignment_name: str,
	midi_map_dir: pathlib.Path = pathlib.Path.cwd(),
) -> SelectSpec:

	"""Parse a single ``select:`` dict into a SelectSpec.

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

	order_by = str(raw.get("order_by", "newest"))

	if order_by not in VALID_ORDER_BY:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: unknown order_by {order_by!r}. "
			f"Valid values: {', '.join(sorted(VALID_ORDER_BY))}"
		)

	# Default order_by to "similarity" when reference is set and no explicit order.
	if where.reference is not None and "order_by" not in raw:
		order_by = "similarity"

	pick = int(raw.get("pick", 1))

	if pick < 1:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pick must be >= 1 (got {pick})"
		)

	return SelectSpec(where=where, order_by=order_by, pick=pick)


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

	for entry in raw:

		if isinstance(entry, str):
			steps.append(ProcessorStep(name=entry))

		elif isinstance(entry, dict):

			if len(entry) != 1:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: each process entry "
					f"must have exactly one key (got {list(entry.keys())})"
				)

			proc_name = next(iter(entry))
			proc_value = entry[proc_name]

			if isinstance(proc_value, bool) or proc_value is None:
				# e.g. "repitch: true" or "repitch:"
				steps.append(ProcessorStep(name=str(proc_name)))

			elif isinstance(proc_value, dict):
				# e.g. "beat_quantize: { grid: 16, bpm: 120 }"
				# Param values that are dicts with a "cc" key become CcBindings.
				resolved_params: list[tuple[str, typing.Any]] = []

				for k, v in proc_value.items():
					if isinstance(v, dict) and "cc" in v:
						resolved_params.append((str(k), CcBinding(
							cc=int(v["cc"]),
							min_val=float(v.get("min", 0.0)),
							max_val=float(v.get("max", 1.0)),
							default=float(v["default"]) if "default" in v else None,
							channel=int(v["channel"]) if "channel" in v else None,
						)))
					else:
						resolved_params.append((str(k), v))

				frozen_params = tuple(resolved_params)
				steps.append(ProcessorStep(name=str(proc_name), params=frozen_params))

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
