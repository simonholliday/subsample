"""A declared schema of the MIDI map, in the names a map uses.

The MIDI map is the file that tells Subsample which sounds a note plays, how to
process them, and how they sound when they are released.  Its grammar lives in
the parser: subsample.player reads the map's keys and each assignment's, and
subsample.query reads what is inside them.  The processors are declared once in
subsample.processors, and the parser and the step compiler both read that.

This module declares the rest of that grammar a second time, with the forms,
defaults, limits and allowed words a reader needs, and returns the whole map as
one JSON Schema: the grammar from here, and the processors generated from their
own declaration.  Nothing here validates a map or changes what loads.

A second list can drift from the first, so tests/test_midi_map_schema.py is the
load-bearing half of this module.  It fails the suite when a key the parser
accepts is not declared here, when a word declared here is not one the parser
allows, when a limit published here is not enforced, and when a declared default
is not what leaving the key out builds.

The prose is published on subsystem.co as it stands, so it uses British spelling
and never an em dash.
"""

import typing

import subsample.processors


_ABSENT: typing.Final = object()
"""Marks a term whose value is worked out at load time rather than fixed."""


_WHERE_UNITS: typing.Final[dict[str, str]] = {"duration": "s", "tempo": "BPM", "pitch": "Hz"}
"""The unit a measurement is written in, where Subsample has a word for it."""

_SCORER_PARAMETER_TERMS: typing.Final[dict[str, dict[str, typing.Any]]] = {
	"pattern": {
		"description": "",
		"type": "array",
		"items": {"type": "number", "minimum": 0, "maximum": 1},
		"minItems": 2,
	},
}
"""What an ``order:`` entry's own parameters are written as, by name.  The
parameters themselves are declared by the scorer, in subsample.query."""


# ---------------------------------------------------------------------------
# The whole map
# ---------------------------------------------------------------------------

def json_schema () -> dict[str, typing.Any]:

	"""Return the JSON Schema of a MIDI map, in the order a map writes it."""

	# Imported here so reading the schema does not pay for numpy and librosa
	# until someone asks for it.  The parser's own lists give both the words
	# and the order they are published in.
	import subsample.player

	return {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"title": "Subsample MIDI map",
		"description": "",
		"type": "object",
		"properties": _in_order(subsample.player.VALID_MAP_KEYS, _map_terms(), "map key"),
		"additionalProperties": False,
		"$defs": _defs(),
	}


def _map_terms () -> dict[str, typing.Any]:

	"""Every key the map itself may carry."""

	import subsample.bank

	return {
		"definitions": _mounted_definitions(),
		"channel":     {"$ref": "#/$defs/channel"},
		"programs": {
			"description": "",
			"type": "array",
			"items": {"$ref": "#/$defs/program"},
		},
		"program_channel": _number_or_name(0, 16, default=subsample.bank.DEFAULT_BANK_CHANNEL),
		"default_program": {"$ref": "#/$defs/program_number"},
		"templates": {
			"description": "",
			"type": "object",
			"additionalProperties": {"$ref": "#/$defs/template"},
		},
		"assignments": {
			"description": "",
			"type": "array",
			"items": {"$ref": "#/$defs/assignment"},
		},
		"maps": {
			"description": "",
			"type": "array",
			"items": {"$ref": "#/$defs/included_map"},
		},
	}


def _mounted_definitions () -> dict[str, typing.Any]:

	"""The definitions files a map mounts, each under a prefix of its own choosing."""

	import subsample.definitions
	import subsample.player

	return {
		"description": "",
		"type": "object",
		"propertyNames": {
			"pattern": f"^{subsample.definitions.NAME_RE.pattern}$",
			"not": {"enum": list(subsample.player.SYMBOL_NAMESPACES)},
		},
		"additionalProperties": {"type": "string", "minLength": 1},
	}


def _defs () -> dict[str, typing.Any]:

	"""The parts a map writes in more than one place, each declared once."""

	return {
		"assignment":    _assignment(),
		"select_spec":   _select_spec(),
		"order_clause":  _order_clause(),
		"channel":       _number_or_name(1, 16),
		"program_number": _number_or_name(0, 127),
		"controller":    _number_or_name(0, 127),
		"rank":          {"description": "", "type": "integer", "minimum": 1},
		"pan_position":  {"description": "", "type": "number", "minimum": -100, "maximum": 100},
		"release_time":  _release_time(),
		"template":      _template(),
		"program":       _program(),
		"included_map":  _included_map(),
		"notes":         _notes(),
		"note":          _note(),
		"velocity":      _velocity(),
		"velocity_range": _velocity_range(),
		"select":        _select(),
		"where":         _where(),
		"order":         _order(),
		"pick":          _pick(),
		"process_step":  _process_step(),
		"cc_binding":    _cc_binding(),
		"release":       _release(),
		"loop":          _loop(),
		"extract":       _extract(),
		"pan":           _pan(),
		"output":        _output(),
		"silenced_by":   _silenced_by(),
		"defined_name":  _defined_name(),
		"written_number": _written_number(),
	}


# ---------------------------------------------------------------------------
# An assignment: one instrument on one channel
# ---------------------------------------------------------------------------

def _assignment () -> dict[str, typing.Any]:

	"""One assignment: which notes play which sounds, and how they sound."""

	import subsample.player

	return {
		"description": "",
		"type": "object",
		"properties": _in_order(
			subsample.player.VALID_ASSIGNMENT_KEYS, _assignment_terms(), "assignment key",
		),
		"required": ["notes", "select"],
		"additionalProperties": False,
	}


def _template () -> dict[str, typing.Any]:

	"""A named set of assignment fields that an assignment may start from.

	A template carries the same fields as an assignment, except that it may not
	name a template of its own, and it requires nothing: an assignment it is
	merged into supplies the rest."""

	import subsample.player

	fields = {
		name: term for name, term in _assignment_terms().items()
		if name != "template"
	}

	return {
		"description": "",
		"type": "object",
		"properties": _in_order(
			tuple(name for name in subsample.player.VALID_ASSIGNMENT_KEYS if name != "template"),
			fields,
			"template key",
		),
		"additionalProperties": False,
	}


def _assignment_terms () -> dict[str, typing.Any]:

	"""Every key one assignment may carry."""

	import subsample.query

	return {
		"name": {
			"description": "",
			"type": "string",
			"default": "<unnamed>",
		},
		"template": {
			"description": "",
			"anyOf": [
				{"type": "string"},
				{"type": "array", "items": {"type": "string"}, "minItems": 1},
			],
		},
		"channel":     {"$ref": "#/$defs/channel"},
		"notes":       {"$ref": "#/$defs/notes"},
		"velocity":    {"$ref": "#/$defs/velocity"},
		"select":      {"$ref": "#/$defs/select"},
		"process": {
			"description": "",
			"type": "array",
			"items": {"$ref": "#/$defs/process_step"},
			"contains": _beat_aligning_step(),
			"maxContains": 1,
		},
		"mode": {
			"description": "",
			"oneOf": [_word(mode, "", "") for mode in subsample.query.VALID_MODES],
			"default": "one_shot",
		},
		"loop":        {"$ref": "#/$defs/loop"},
		"release":     {"$ref": "#/$defs/release"},
		"extract":     {"$ref": "#/$defs/extract"},
		"gain": {
			"description": "",
			"type": "number",
			"default": 0.0,
			"x-unit": "dB",
		},
		"pan":         {"$ref": "#/$defs/pan"},
		"output":      {"$ref": "#/$defs/output"},
		"stack": {
			"description": "",
			"type": "boolean",
			"default": False,
		},
		"silenced_by": {"$ref": "#/$defs/silenced_by"},
	}


def _beat_aligning_step () -> dict[str, typing.Any]:

	"""A process entry that aligns the sample to the beat, in any of its forms."""

	import subsample.query

	names = subsample.query.BEAT_ALIGNING_PROCESSORS

	return {
		"anyOf": [
			{"enum": list(names)},
			{"type": "object", "anyOf": [{"required": [name]} for name in names]},
		],
	}


# ---------------------------------------------------------------------------
# What a note plays
# ---------------------------------------------------------------------------

def _notes () -> dict[str, typing.Any]:

	"""The notes an assignment answers to."""

	return {
		"description": "",
		"anyOf": [
			{"$ref": "#/$defs/note"},
			{"type": "array", "items": {"$ref": "#/$defs/note"}, "minItems": 1},
			_zone_tuned(),
		],
	}


def _note () -> dict[str, typing.Any]:

	"""One note: its number, its name, or a name a definitions file gives it.

	A string also carries the two other forms this position accepts: a range
	such as ``36..60`` or ``C2..C4``, and a number written as text."""

	return {
		"description": "",
		"anyOf": [
			{"type": "integer", "minimum": 0, "maximum": 127},
			{"type": "string", "minLength": 1},
		],
	}


def _zone_tuned () -> dict[str, typing.Any]:

	"""A range of notes tuned from one sound, which repitch follows."""

	import subsample.player

	sentinel = subsample.player.ZONE_TUNED_SENTINEL

	terms = {
		"mode": {"description": "", "const": sentinel},
		"range": {
			"description": "",
			"type": "array",
			"items": {"$ref": "#/$defs/note"},
			"minItems": 2,
			"maxItems": 2,
			"default": [0, 127],
		},
	}

	return {
		"anyOf": [
			{"const": sentinel},
			{
				"type": "object",
				"properties": _in_order(
					subsample.player.VALID_NOTES_INNER_KEYS, terms, "notes key",
				),
				"required": ["mode"],
				"additionalProperties": False,
			},
		],
	}


def _velocity () -> dict[str, typing.Any]:

	"""Which velocities an assignment answers to, and what they become."""

	import subsample.player

	terms = {
		"trigger":  {"$ref": "#/$defs/velocity_range"},
		"rescale": {
			"description": "",
			"anyOf": [
				{"type": "boolean"},
				{"$ref": "#/$defs/velocity_range"},
			],
			"default": False,
		},
	}

	return {
		"description": "",
		"default": [0, 127],
		"anyOf": [
			{"$ref": "#/$defs/velocity_range"},
			{
				"type": "object",
				"properties": _in_order(
					subsample.player.VELOCITY_INNER_KEYS, terms, "velocity key",
				),
				"required": ["trigger"],
				"additionalProperties": False,
			},
		],
	}


def _velocity_range () -> dict[str, typing.Any]:

	"""A pair of MIDI velocities, low then high."""

	return {
		"description": "",
		"type": "array",
		"items": {"type": "integer", "minimum": 0, "maximum": 127},
		"minItems": 2,
		"maxItems": 2,
	}


def _silenced_by () -> dict[str, typing.Any]:

	"""The notes that cut this one short, as a closed hi-hat cuts an open one."""

	choke = {
		"anyOf": [
			{"const": "self"},
			{"$ref": "#/$defs/note"},
		],
	}

	return {
		"description": "",
		"anyOf": [
			choke,
			{"type": "array", "items": choke},
			{"const": False},
		],
	}


# ---------------------------------------------------------------------------
# Which sounds a note plays: select, where, order and pick
# ---------------------------------------------------------------------------

def _select () -> dict[str, typing.Any]:

	"""How an assignment chooses a sound: the whole library filtered, ranked and picked from.

	A list of these is a fallback chain, tried in turn until one finds a sound."""

	import subsample.query

	return {
		"description": "",
		"anyOf": [
			{"$ref": "#/$defs/select_spec"},
			{"type": "array", "items": {"$ref": "#/$defs/select_spec"}, "minItems": 1},
		],
	}


def _select_spec () -> dict[str, typing.Any]:

	"""One way of choosing a sound: a filter, a ranking and a pick."""

	import subsample.query

	terms = {
		"where":    {"$ref": "#/$defs/where"},
		"order":    {"$ref": "#/$defs/order"},
		"order_by": _deprecated({"$ref": "#/$defs/order"}),
		"pick":     {"$ref": "#/$defs/pick"},
	}

	return {
		"description": "",
		"type": "object",
		"properties": _in_order(subsample.query.VALID_SELECT_KEYS, terms, "select key"),
		"not": {"required": ["order", "order_by"]},
		"additionalProperties": False,
	}


def _where () -> dict[str, typing.Any]:

	"""What a sound must be for this assignment to play it."""

	import subsample.query

	terms: dict[str, typing.Any] = {
		"name": _name_term(),
		"path":      {"description": "", "type": "string", "minLength": 1},
		"directory": {"description": "", "type": "string", "minLength": 1},
		"reference": {"description": "", "type": "string", "minLength": 1},
		"pitched":   {"description": "", "type": "boolean"},
		"loopable":  {"description": "", "type": "boolean"},
	}

	for key in subsample.query.NUMERIC_YAML_KEYS:
		terms[key] = _measurement(key)

	for key, (field, _operator) in subsample.query.LEGACY_WHERE_KEYS.items():
		terms[key] = _deprecated({"description": "", **_measured_value(field)})

	accepted = (
		*subsample.query.NON_RANGE_WHERE_KEYS,
		*subsample.query.NUMERIC_YAML_KEYS,
		*subsample.query.LEGACY_WHERE_KEYS,
	)

	return {
		"description": "",
		"type": "object",
		"properties": _in_order(accepted, terms, "where key"),
		"additionalProperties": False,
	}


def _measurement (key: str) -> dict[str, typing.Any]:

	"""One measured quality of a sound: a value it must equal, or bounds it must lie within."""

	import subsample.query

	value = _measured_value(subsample.query.NUMERIC_YAML_KEYS[key])

	term: dict[str, typing.Any] = {
		"description": "",
		"anyOf": [
			value,
			{
				"type": "object",
				"properties": {
					operator: {"description": "", **value}
					for operator in subsample.query.VALID_OPERATORS
				},
				"minProperties": 1,
				"additionalProperties": False,
			},
		],
	}

	if key in _WHERE_UNITS:
		term["x-unit"] = _WHERE_UNITS[key]

	return term


def _measured_value (field: str) -> dict[str, typing.Any]:

	"""What one measurement is written as: a number, or a note name where a pitch is asked for."""

	if field == "pitch_hz":
		return {"anyOf": [{"type": "number"}, {"type": "string", "minLength": 1}]}

	return {"type": "number"}


def _name_term () -> dict[str, typing.Any]:

	"""A sound named outright, named among several, or matched by a pattern."""

	import subsample.query

	pattern = {"type": "string", "minLength": 1}

	return {
		"description": "",
		"anyOf": [
			{"type": "string", "minLength": 1},
			{"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "uniqueItems": True},
			{
				"type": "object",
				"properties": {
					operator: {"description": "", **pattern}
					for operator in subsample.query.VALID_NAME_OPERATORS
				},
				"minProperties": 1,
				"maxProperties": 1,
				"additionalProperties": False,
			},
		],
	}


def _order () -> dict[str, typing.Any]:

	"""How the sounds that matched are ranked, best first."""

	return {
		"description": "",
		"anyOf": [
			{"$ref": "#/$defs/order_clause"},
			{"type": "array", "items": {"$ref": "#/$defs/order_clause"}, "minItems": 1},
		],
	}


def _order_clause () -> dict[str, typing.Any]:

	"""One ranking: what the sounds are ranked by, and which way round."""

	import subsample.query

	parameters: dict[str, typing.Any] = {}
	wanted:     list[dict[str, typing.Any]] = []

	for scorer, names in subsample.query.SCORER_PARAMETERS.items():
		for name in names:
			parameters[name] = {
				**_SCORER_PARAMETER_TERMS[name],
				"x-applies-when": [{"by": [scorer]}],
			}

		wanted.append({
			"if":   {"properties": {"by": {"const": scorer}}, "required": ["by"]},
			"then": {"required": list(names)},
		})

	return {
		"description": "",
		"anyOf": [
			{"oneOf": [
				_deprecated(_word(token, "", ""))
				for token in subsample.query.LEGACY_ORDER_TOKENS
			]},
			{
				"type": "object",
				"properties": {
					"by":  {"description": "", "oneOf": [
						_word(name, "", "") for name in subsample.query.valid_order_names()
					]},
					"dir": {"description": "", "oneOf": [
						_word(direction, "", "") for direction in ("asc", "desc")
					]},
					**parameters,
				},
				"required": ["by"],
				"allOf": wanted,
				"additionalProperties": False,
			},
		],
	}


def _pick () -> dict[str, typing.Any]:

	"""Which of the ranked sounds a note actually plays."""

	import subsample.query

	rank: dict[str, typing.Any] = {"$ref": "#/$defs/rank"}

	velocity_terms = {
		"mode":      {"description": "", "const": "velocity"},
		"variation": {"description": "", "type": "integer", "minimum": 0, "maximum": 127, "default": 0},
		"curve": {
			"description": "",
			"oneOf": [_word(curve, "", "") for curve in subsample.query.VALID_PICK_CURVES],
			"default": "linear",
		},
		"spacing": {
			"description": "",
			"oneOf": [_word(spacing, "", "") for spacing in subsample.query.VALID_PICK_SPACINGS],
			"default": "rank",
		},
	}

	return {
		"description": "",
		"default": 1,
		"anyOf": [
			{"oneOf": [_word("any", "", ""), _word("velocity", "", "")]},
			dict(rank),
			{
				"type": "array",
				"items": {"anyOf": [dict(rank), {"type": "null"}]},
				"minItems": 2,
				"maxItems": 2,
			},
			{
				"type": "object",
				"properties": {
					operator: {"description": "", **rank}
					for operator in subsample.query.VALID_PICK_OPERATORS
				},
				"minProperties": 1,
				"additionalProperties": False,
			},
			{
				"type": "object",
				"properties": _in_order(
					subsample.query.VALID_VELOCITY_PICK_KEYS, velocity_terms, "velocity pick key",
				),
				"required": ["mode"],
				"additionalProperties": False,
			},
		],
	}


# ---------------------------------------------------------------------------
# How a sound plays: release, loop, extract, pan and output
# ---------------------------------------------------------------------------

def _release () -> dict[str, typing.Any]:

	"""What the sound does when the key comes up."""

	import subsample.player
	import subsample.query

	curve = {
		"description": "",
		"oneOf": [_word(shape, "", "") for shape in subsample.query.VALID_RELEASE_CURVES],
		"default": "cosine",
	}

	time = {"$ref": "#/$defs/release_time"}

	spelt_out = {
		"type": "object",
		"properties": _in_order(
			subsample.player.RELEASE_INNER_KEYS, {"time": time, "curve": curve}, "release key",
		),
		"additionalProperties": False,
	}

	# The knob written where the time itself would go, which is the same
	# binding with the fade's shape beside it.
	bound = {
		"type": "object",
		"properties": {
			**typing.cast(dict[str, typing.Any], _cc_binding()["properties"]),
			"curve": curve,
		},
		"required": ["cc"],
		"additionalProperties": False,
	}

	return {
		"description": "",
		"anyOf": [
			{"type": "boolean"},
			_word("full", "", ""),
			{"$ref": "#/$defs/release_time"},
			spelt_out,
			bound,
		],
	}


def _release_time () -> dict[str, typing.Any]:

	"""How long the sound takes to fade once the key comes up, or the knob that sets it."""

	import subsample.player

	return {
		"description": "",
		"anyOf": [
			{"type": "number", "minimum": 0},
			{"$ref": "#/$defs/cc_binding"},
		],
		"x-unit": "ms",
		"x-sweep": list(subsample.player.RELEASE_CC_SWEEP_MS),
	}


def _loop () -> dict[str, typing.Any]:

	"""Where the sound loops while the key is held."""

	import subsample.player

	terms = {
		"start":     {"description": "", "type": "number", "minimum": 0, "x-unit": "s"},
		"end":       {"description": "", "type": "number", "minimum": 0, "x-unit": "s"},
		"crossfade": {"description": "", "type": "number", "minimum": 0, "x-unit": "ms"},
	}

	return {
		"description": "",
		"type": "object",
		"properties": _in_order(subsample.player.LOOP_INNER_KEYS, terms, "loop key"),
		"additionalProperties": False,
	}


def _extract () -> dict[str, typing.Any]:

	"""Which part of a multi-channel recording this assignment plays."""

	import subsample.player
	import subsample.query

	blend = {
		"description": "",
		"type": "array",
		"items": {"type": "number"},
		"minItems": 1,
	}

	return {
		"description": "",
		"anyOf": [
			{"oneOf": [_word(kind, "", "") for kind in subsample.query.EXTRACT_KINDS]},
			{"type": "string", "pattern": r"^channel\.[0-9]+$"},
			{
				"type": "object",
				"properties": _in_order(
					subsample.player.VALID_EXTRACT_KEYS, {"blend": blend}, "extract key",
				),
				"required": ["blend"],
				"additionalProperties": False,
			},
		],
	}


def _pan () -> dict[str, typing.Any]:

	"""Where the sound sits across the speakers."""

	import subsample.channel
	import subsample.query

	position: dict[str, typing.Any] = {"$ref": "#/$defs/pan_position"}

	terms = {
		"gte":       dict(position),
		"lte":       dict(position),
		"position":  dict(position),
		"variation": {"description": "", "type": "number", "minimum": 0, "maximum": 200, "default": 0},
	}

	return {
		"description": "",
		"anyOf": [
			dict(position),
			{
				"type": "array",
				"items": {"type": "number", "minimum": 0},
				"anyOf": [
					{"minItems": count, "maxItems": count}
					for count in sorted(subsample.channel.STANDARD_LAYOUTS)
				],
			},
			_word("any", "", ""),
			{
				"type": "object",
				"properties": _in_order(subsample.query.VALID_PAN_KEYS, terms, "pan key"),
				"minProperties": 1,
				"additionalProperties": False,
			},
		],
	}


def _output () -> dict[str, typing.Any]:

	"""The device channels this assignment plays out of, counting from 1."""

	return {
		"description": "",
		"type": "array",
		"items": {"type": "integer", "minimum": 1},
		"minItems": 1,
		"uniqueItems": True,
	}


# ---------------------------------------------------------------------------
# Programs and ensembles
# ---------------------------------------------------------------------------

def _program () -> dict[str, typing.Any]:

	"""One program: the sounds a Program Change message switches to."""

	import subsample.bank

	terms = {
		"name":      {"description": "", "type": "string", "minLength": 1},
		"program":   {"$ref": "#/$defs/program_number"},
		"directory": {"description": "", "type": "string", "minLength": 1},
		"map":       {"description": "", "type": "string", "minLength": 1},
	}

	return {
		"description": "",
		"type": "object",
		"properties": _in_order(subsample.bank.VALID_PROGRAM_KEYS, terms, "program key"),
		"required": ["name"],
		"oneOf": [{"required": ["directory"]}, {"required": ["map"]}],
		"additionalProperties": False,
	}


def _included_map () -> dict[str, typing.Any]:

	"""One map an ensemble plays, and the channel it answers on."""

	import subsample.ensemble

	terms = {
		"map":     {"description": "", "type": "string", "minLength": 1},
		"channel": {"$ref": "#/$defs/channel"},
	}

	return {
		"description": "",
		"anyOf": [
			{"type": "string", "minLength": 1},
			{
				"type": "object",
				"properties": _in_order(
					subsample.ensemble.VALID_INCLUDE_KEYS, terms, "included map key",
				),
				"required": ["map"],
				"additionalProperties": False,
			},
		],
	}


# ---------------------------------------------------------------------------
# Values a map may write in more than one place
# ---------------------------------------------------------------------------

def _defined_name () -> dict[str, typing.Any]:

	"""A name from a mounted definitions file, such as ``drum.kick_1``."""

	import subsample.definitions

	return {
		"description": "",
		"type": "string",
		"pattern": f"^{subsample.definitions.SYMBOL_RE.pattern}$",
	}


def _written_number () -> dict[str, typing.Any]:

	"""A whole number written as text, which a map may do anywhere a number is asked for."""

	return {
		"description": "",
		"type": "string",
		"pattern": r"^\s*[+-]?[0-9]+\s*$",
	}


def _number_or_name (
	minimum: int,
	maximum: int,
	default: typing.Any = _ABSENT,
) -> dict[str, typing.Any]:

	"""A whole number, or a name a definitions file gives that number."""

	term: dict[str, typing.Any] = {
		"description": "",
		"anyOf": [
			{"type": "integer", "minimum": minimum, "maximum": maximum},
			{"$ref": "#/$defs/defined_name"},
			{"$ref": "#/$defs/written_number"},
		],
	}

	if default is not _ABSENT:
		term["default"] = default

	return term


def _in_order (
	accepted: typing.Sequence[str],
	declared: typing.Mapping[str, typing.Any],
	what:     str,
) -> dict[str, typing.Any]:

	"""The declared terms, in the order the parser's own list gives them.

	Reading the parser's list is what keeps the two together: a key added to one
	and not the other stops the schema being built at all, rather than going out
	as a reference that quietly omits it."""

	undeclared = [name for name in accepted if name not in declared]
	unaccepted = [name for name in declared if name not in accepted]

	if undeclared or unaccepted:
		raise KeyError(
			f"midi_map_schema and the parser disagree about the {what}s: "
			f"the parser accepts {undeclared} which are not declared, and "
			f"{unaccepted} are declared but not accepted"
		)

	return {name: declared[name] for name in accepted}


# ---------------------------------------------------------------------------
# A CC binding, which any number a knob can drive accepts in place of a value
# ---------------------------------------------------------------------------

def _cc_binding () -> dict[str, typing.Any]:

	"""A MIDI controller bound to a value, in place of the value itself."""

	import subsample.query

	fields: dict[str, typing.Any] = {
		"cc":      {"$ref": "#/$defs/controller"},
		"channel": {"$ref": "#/$defs/channel"},
		"min":     {"description": "", "type": "number"},
		"max":     {"description": "", "type": "number"},
		"default": {"description": "", "type": "number"},
	}

	return {
		"description": "",
		"type": "object",
		"properties": {key: fields[key] for key in subsample.query.CC_BINDING_KEYS},
		"required": ["cc"],
		"additionalProperties": False,
	}


# ---------------------------------------------------------------------------
# The processors, generated from subsample.processors
# ---------------------------------------------------------------------------

def _process_step () -> dict[str, typing.Any]:

	"""One entry of a `process:` list: a processor named on its own, or named with what it takes."""

	return {
		"description": "",
		"anyOf": [
			{"oneOf": _processor_words()},
			{
				"type": "object",
				"minProperties": 1,
				"maxProperties": 1,
				"properties": _processor_properties(),
				"additionalProperties": False,
			},
		],
	}


def _processor_words () -> list[dict[str, typing.Any]]:

	"""Every processor name a map may write on its own, current names then legacy ones."""

	words: list[dict[str, typing.Any]] = []

	for processor in subsample.processors.PROCESSORS.values():
		words.append(_word(processor.name, processor.title, processor.description))

		for legacy in processor.legacy_names:
			words.append(_deprecated(_word(legacy.name, "", "")))

	return words


def _processor_properties () -> dict[str, typing.Any]:

	"""Each processor name as a key, and what a map may write against it."""

	properties: dict[str, typing.Any] = {}

	for processor in subsample.processors.PROCESSORS.values():
		properties[processor.name] = _processor_value(processor)

		for legacy in processor.legacy_names:
			properties[legacy.name] = _deprecated(_processor_value(processor, tuple(legacy.implies)))

	return properties


def _processor_value (
	processor: subsample.processors.Processor,
	implied:   tuple[str, ...] = (),
) -> dict[str, typing.Any]:

	"""What a map may write against a processor's name: nothing, its shorthand, or its parameters.

	``implied`` names the parameters an older spelling of the name supplies by
	itself, such as ``hpss_harmonic``'s ``keep``, which the map need not give."""

	forms: list[dict[str, typing.Any]] = [{"type": "boolean"}, {"type": "null"}]

	if processor.shorthand is not None:
		forms.extend(_number_forms(processor.parameter(processor.shorthand)))

	forms.append(_parameters_object(processor, implied))

	return {
		"title": processor.title,
		"description": processor.description,
		"anyOf": forms,
	}


def _parameters_object (
	processor: subsample.processors.Processor,
	implied:   tuple[str, ...] = (),
) -> dict[str, typing.Any]:

	"""The mapping of parameters a processor takes, with the names it no longer prefers."""

	properties: dict[str, typing.Any] = {}
	value: dict[str, typing.Any] = {"type": "object"}

	for parameter in processor.parameters:
		properties[parameter.name] = _parameter(parameter)

		for legacy in parameter.legacy_names:
			properties[legacy] = _deprecated(_parameter(parameter))

	value["properties"] = properties
	value.update(_requirements(processor, implied))
	value["additionalProperties"] = False

	return value


def _requirements (
	processor: subsample.processors.Processor,
	implied:   tuple[str, ...],
) -> dict[str, typing.Any]:

	"""What a map must give, in whichever spelling it gives it.

	A parameter the map must set is required by name.  Where an older spelling
	would satisfy the parser, either name does, so the requirement is written as
	the choice the parser actually makes."""

	wanted = [
		parameter for parameter in processor.parameters
		if parameter.required and parameter.name not in implied
	]

	if not wanted:
		return {}

	if all(not parameter.legacy_names for parameter in wanted):
		return {"required": [parameter.name for parameter in wanted]}

	return {"allOf": [
		{"anyOf": [
			{"required": [name]}
			for name in (parameter.name, *parameter.legacy_names)
		]}
		for parameter in wanted
	]}


def _parameter (parameter: subsample.processors.Parameter) -> dict[str, typing.Any]:

	"""One processor parameter: what it accepts, where it rests, and where it has any effect."""

	entry: dict[str, typing.Any] = {
		"title": parameter.title,
		"description": parameter.description,
		"anyOf": _parameter_forms(parameter),
	}

	if parameter.default is not None:
		entry["default"] = parameter.default

	if parameter.automatic is not None:
		entry["x-automatic"] = parameter.automatic

	if parameter.unit is not None:
		entry["x-unit"] = parameter.unit

	if parameter.sweep is not None:
		entry["x-sweep"] = list(parameter.sweep)

	if parameter.taper != "linear":
		entry["x-taper"] = parameter.taper

	if parameter.limits_when:
		entry["x-limits-when"] = [
			{"when": _condition(limit_when.when), **_limit_keywords(limit_when.limit)}
			for limit_when in parameter.limits_when
		]

	if parameter.applies_when:
		entry["x-applies-when"] = [_condition(condition) for condition in parameter.applies_when]

	return entry


def _parameter_forms (parameter: subsample.processors.Parameter) -> list[dict[str, typing.Any]]:

	"""Every form a parameter's value may take, in the order the declaration lists them."""

	forms: list[dict[str, typing.Any]] = []

	for form in parameter.forms:

		if form in ("number", "integer"):
			forms.append({"type": form, **_limit_keywords(parameter.limit)})

		elif form == "boolean":
			forms.append({"type": "boolean"})

		elif form == "choice":
			forms.append({"oneOf": [
				_word(choice.value, choice.title, choice.description)
				for choice in parameter.choices
			]})

		elif form == "note_name":
			forms.append({"type": "string"})

		elif form == "path":
			forms.append({"type": "string", "minLength": 1})

	if parameter.bindable:
		forms.append({"$ref": "#/$defs/cc_binding"})

	return forms


def _number_forms (parameter: subsample.processors.Parameter) -> list[dict[str, typing.Any]]:

	"""The numeric forms of a parameter, which is all a processor's shorthand takes."""

	return [
		form for form in _parameter_forms(parameter)
		if form.get("type") in ("number", "integer")
	]


def _limit_keywords (limit: subsample.processors.Limit) -> dict[str, typing.Any]:

	"""A declared limit as the JSON Schema keywords that refuse the same values."""

	keywords: dict[str, typing.Any] = {}

	if limit.minimum is not None:
		keywords["minimum"] = limit.minimum

	if limit.maximum is not None:
		keywords["maximum"] = limit.maximum

	if limit.exclusive_minimum is not None:
		keywords["exclusiveMinimum"] = limit.exclusive_minimum

	if limit.exclusive_maximum is not None:
		keywords["exclusiveMaximum"] = limit.exclusive_maximum

	return keywords


def _condition (condition: typing.Mapping[str, tuple[str, ...]]) -> dict[str, list[str]]:

	"""One condition on sibling parameters: every key must hold for it to hold."""

	return {name: list(values) for name, values in condition.items()}


def _word (value: str, title: str, description: str) -> dict[str, typing.Any]:

	"""One allowed word, with the label and prose that belong to it."""

	return {"const": value, "title": title, "description": description}


def _deprecated (entry: dict[str, typing.Any]) -> dict[str, typing.Any]:

	"""The same term under a name Subsample still accepts but no longer documents."""

	return {**entry, "deprecated": True}

