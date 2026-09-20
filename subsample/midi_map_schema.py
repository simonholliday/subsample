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
		"description": "The beat pattern to match: one number for each beat, of which only the shape counts, not the level.",
		"type": "array",
		"items": {"type": "number", "minimum": 0, "maximum": 1},
		"minItems": 2,
		"examples": [[1, 0, 0.5, 0]],
	},
}
"""What an ``order:`` entry's own parameters are written as, by name.  The
parameters themselves are declared by the scorer, in subsample.query."""


# ---------------------------------------------------------------------------
# The words a map writes, each with the title and description a reader sees
# ---------------------------------------------------------------------------
#
# Each list is keyed by the word the parser accepts.  _words() publishes them in
# the parser's order and refuses a word the parser does not accept, or one it
# accepts that has no prose here.

_MODES: typing.Final[dict[str, tuple[str, str]]] = {
	"one_shot": ("One shot", "Plays to the end of the sound and ignores note-off."),
	"gated":    ("Gated", "Plays while the key is held, and releases at note-off."),
	"loop":     ("Loop", "Loops while the key is held, and releases past the loop at note-off."),
}

_MEASUREMENTS: typing.Final[dict[str, str]] = {
	"duration":        "The sample's length, kept within bounds. A length is measured from the audio, so it is written as a range: one value would keep only a sample measured at exactly that, which almost none is.",
	"duration_beats":  "The sample's length in beats at the session tempo, where a beat is a quarter note, kept within bounds. A map that uses it needs a session tempo. It is worked out from a measured length, so it is written as a range rather than as one value.",
	"onsets":          "How many hits Subsample detected in the sample. A number keeps exactly that count, and bounds keep the counts within them.",
	"tempo":           "The tempo Subsample detected in the sample, kept within bounds. A detected tempo is measured, so it is written as a range: one value would keep only a sample measured at exactly that, which almost none is.",
	"pitch":           "The pitch Subsample detected in the sample, as a frequency or as a note name such as `C3`, kept within bounds. A detected pitch is measured, so it is written as a range rather than as one pitch.",
	"quantized_beats": "The sample's length in beats once the assignment's quantise processor has run. A sample not yet quantised does not qualify. A quantised length comes out of the grid whole, so a number keeps exactly that many beats, and bounds keep the lengths within them.",
}

_MEASUREMENT_EXAMPLES: typing.Final[dict[str, list[typing.Any]]] = {
	"duration":        [{"lte": 0.5}],
	"duration_beats":  [{"gte": 4}],
	"onsets":          [1, {"gte": 4}],
	"tempo":           [{"gte": 118, "lte": 122}],
	"pitch":           [{"gte": "C2", "lte": "C3"}, {"lte": 200}],
	"quantized_beats": [4, {"gte": 4}],
}
"""What a map writes to keep a sample by one of its measurements.  A bare number
is an exact match, which only a whole-number measurement such as `onsets` is
ever written with."""

_MEASUREMENT_BOUNDS: typing.Final[dict[str, str]] = {
	"gte": "At least this value.",
	"lte": "At most this value.",
	"gt":  "More than this value.",
	"lt":  "Less than this value.",
	"eq":  "Exactly this value.",
}

_NAME_MATCHES: typing.Final[dict[str, tuple[str, str]]] = {
	"matches": ("Wildcards the whole name must match, where `*` stands for any run of characters and `?` for any one character.", "kick*"),
	"regex":   ("A regular expression the whole name must match.", "^(kick|bd)_[0-9]+$"),
}

_ORDER_BY: typing.Final[dict[str, tuple[str, str]]] = {
	"duration":        ("Duration", "The sample's length."),
	"pitch":           ("Pitch", "The pitch Subsample detected in the sample."),
	"onsets":          ("Hits", "How many hits Subsample detected in the sample."),
	"tempo":           ("Tempo", "The tempo Subsample detected in the sample."),
	"level":           ("Level", "How loud the sample is. A velocity pick needs this as the first ranking."),
	"age":             ("Age", "When Subsample took the sample in, so `desc` puts the newest first."),
	"quantized_beats": ("Quantised beats", "The sample's length in beats once quantised. A sample not yet quantised comes last, whichever way round."),
	"beat_match":      ("Beat match", "How closely the sample's energy on each beat follows the beat pattern in `pattern`. It needs a quantise processor, and leaves out any sample not yet quantised."),
	"similarity":      ("Similarity", "How closely the sample resembles the `reference` in `where`. It must be the first ranking."),
}

_ORDER_DIRECTIONS: typing.Final[dict[str, tuple[str, str]]] = {
	"asc":  ("Lowest first", "The lowest value comes first."),
	"desc": ("Highest first", "The highest value comes first."),
}

_OLDER_ORDER_TITLES: typing.Final[dict[str, str]] = {
	"newest":               "Newest first",
	"oldest":               "Oldest first",
	"duration_asc":         "Shortest first",
	"duration_desc":        "Longest first",
	"pitch_asc":            "Lowest pitch first",
	"pitch_desc":           "Highest pitch first",
	"onsets_asc":           "Fewest hits first",
	"onsets_desc":          "Most hits first",
	"tempo_asc":            "Slowest first",
	"tempo_desc":           "Fastest first",
	"loudest":              "Loudest first",
	"quietest":             "Quietest first",
	"similarity":           "Most similar first",
	"quantized_beats_asc":  "Fewest quantised beats first",
	"quantized_beats_desc": "Most quantised beats first",
}

_PICK_WORDS: typing.Final[dict[str, tuple[str, str]]] = {
	"any":      ("Any", "A match at random on every note, each as likely as the next."),
	"velocity": ("Velocity", "The match that suits how hard the note is struck, from the quiet end of the ranking to the loud end. The ranking must be by `level`."),
}

_RANK_BOUNDS: typing.Final[dict[str, str]] = {
	"gte": "The first rank drawn from.",
	"lte": "The last rank drawn from.",
	"gt":  "Draws from the ranks after this one.",
	"lt":  "Draws from the ranks before this one.",
	"eq":  "Always this rank.",
}

_PICK_CURVES: typing.Final[dict[str, tuple[str, str]]] = {
	"linear":      ("Linear", "Velocity spreads evenly across the ranking."),
	"logarithmic": ("Logarithmic", "Soft notes spread across more of the ranking, for more distinct quiet tones."),
	"exponential": ("Exponential", "Hard notes spread across more of the ranking, for a finer choice among the loud samples."),
}

_PICK_SPACINGS: typing.Final[dict[str, tuple[str, str]]] = {
	"rank":     ("Rank", "Each sample takes an equal share of the velocities, by its place in the ranking."),
	"loudness": ("Loudness", "Each sample sits at its own measured level, so the layout follows the real dynamics of the samples."),
}

_RELEASE_CURVES: typing.Final[dict[str, tuple[str, str]]] = {
	"cosine":      ("Cosine", "Eases out of the sound and into silence smoothly."),
	"exponential": ("Exponential", "Drops quickly, then trails off slowly, as a damped string does."),
}

_EXTRACT_PARTS: typing.Final[dict[str, tuple[str, str]]] = {
	"omni":   ("Omni", "Every direction equally: the sum of the audio channels."),
	"side":   ("Side", "A figure of eight facing left and right: the difference between them."),
	"depth":  ("Depth", "A figure of eight facing front and back, from a recording that carries both."),
	"height": ("Height", "A figure of eight facing up and down, from an ambisonic recording."),
	"left":   ("Left", "A cardioid facing left."),
	"right":  ("Right", "A cardioid facing right."),
	"front":  ("Front", "A cardioid facing forward. On a stereo recording it is the same as `omni`, with a warning."),
	"back":   ("Back", "A cardioid facing back, from a recording that carries front and back."),
}


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
		"description": "The file that tells Subsample which sounds each MIDI note plays, how they are processed, and how they play. It holds assignments, and may add templates, programs, other maps to play at the same time, and definitions files that name notes, controllers, MIDI channels and programs.",
		"type": "object",
		"properties": in_order(subsample.player.VALID_MAP_KEYS, _map_terms(), "map key"),
		"additionalProperties": False,
		"$defs": _defs(),
	}


def _map_terms () -> dict[str, typing.Any]:

	"""Every key the map itself may carry."""

	import subsample.bank

	return {
		"definitions": _mounted_definitions(),
		"channel": {
			"description": "The MIDI channel an assignment answers on when it names none of its own, so a map can be played on whatever MIDI channel a project gives it without editing it. A map with no `channel` needs one on every assignment.",
			"$ref": "#/$defs/channel",
			"examples": [10],
		},
		"programs": {
			"description": "Instrument sets that a MIDI Program Change switches between, all loaded at startup so that a switch is instant. Without it, Subsample plays the library its configuration names.",
			"type": "array",
			"items": {"$ref": "#/$defs/program"},
			"examples": [[
				{"name": "Acoustic kit", "directory": "kits/acoustic"},
				{"name": "Electronic kit", "directory": "kits/electronic"},
			]],
		},
		"program_channel": _number_or_name(
			0, 16,
			"The MIDI channel that Program Change messages are read on, where 0 reads every MIDI channel.",
			default=subsample.bank.DEFAULT_BANK_CHANNEL,
			examples=[0],
		),
		"default_program": {
			"description": "The program active at startup, which must be one the list declares. Left out, the first program in the list.",
			"$ref": "#/$defs/program_number",
			"examples": [1, "my.brushes"],
		},
		"templates": {
			"description": "Named sets of assignment fields that assignments may start from, so that a kit writes its shared MIDI channel, processing or selection once.",
			"type": "object",
			"additionalProperties": {"$ref": "#/$defs/template"},
			"examples": [{"kit": {
				"channel": 10,
				"process": [{"compress": {"threshold": -20.0, "ratio": 8.0}}],
			}}],
		},
		"assignments": {
			"description": "Each instrument the map plays: the notes it answers to, the sound it chooses for them, and how that sound plays. A map whose programs are all `map:` presets, or that plays other maps through `maps`, may leave it out.",
			"type": "array",
			"items": {"$ref": "#/$defs/assignment"},
			"examples": [[{
				"name":    "Kick",
				"channel": 10,
				"notes":   "drum.kick_1",
				"select":  {"where": {"reference": "GM36_BassDrum1"}},
			}]],
		},
		"maps": {
			"description": "Other maps to play at the same time, each on its own MIDI channel, which makes this map an ensemble. A map included here may not include maps of its own.",
			"type": "array",
			"items": {"$ref": "#/$defs/included_map"},
			"examples": [["drums.yaml", {"map": "bass.yaml", "channel": 2}]],
		},
	}


def _mounted_definitions () -> dict[str, typing.Any]:

	"""The definitions files a map mounts, each under a prefix of its own choosing."""

	import subsample.definitions
	import subsample.player

	return {
		"description": "Definitions files to take names from, each under a prefix the map chooses: `{my: project.yaml}` lets the map write `my.kick` for a note the file names. A path is relative to the map. The `drum` prefix is reserved for the built-in General MIDI drum names.",
		"type": "object",
		"propertyNames": {
			"pattern": f"^{subsample.definitions.NAME_RE.pattern}$",
			"not": {"enum": list(subsample.player.SYMBOL_NAMESPACES)},
		},
		"additionalProperties": {"type": "string", "minLength": 1},
		"examples": [{"my": "project.yaml"}],
	}


def _defs () -> dict[str, typing.Any]:

	"""The parts a map writes in more than one place, each declared once."""

	return {
		"assignment":    _assignment(),
		"select_spec":   _select_spec(),
		"order_clause":  _order_clause(),
		"channel": _number_or_name(
			1, 16,
			"A MIDI channel, counted from 1, or a name a definitions file gives one.",
		),
		"program_number": _number_or_name(
			0, 127,
			"A MIDI program number, or a name a definitions file gives one.",
		),
		"controller": _number_or_name(
			0, 127,
			"A MIDI controller number, or a name a definitions file gives one.",
		),
		"rank": {
			"description": "A place in the ranking, counted from 1 for the best match.",
			"type": "integer",
			"minimum": 1,
		},
		"pan_position": {
			"description": "A position across the stereo field: -100 is hard left, 0 the centre, and 100 hard right.",
			"type": "number",
			"minimum": -100,
			"maximum": 100,
		},
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
		"description": "One instrument: the notes it answers to, the sound it chooses for them, how that sound is processed, and how it plays.",
		"type": "object",
		"properties": in_order(
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
		"description": "A named set of assignment fields. An assignment that names the template starts from its fields, and its own fields win: a field it sets replaces the template's whole, with nothing merged inside it. A template may not name a template of its own.",
		"type": "object",
		"properties": in_order(
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
			"description": "A name for the assignment, which log lines and error messages use.",
			"type": "string",
			"default": "<unnamed>",
			"examples": ["Kick"],
		},
		"template": {
			"description": "The template, or the templates in order, that the assignment starts from. A later template overrides an earlier one, and the assignment's own fields override them all.",
			"anyOf": [
				{"type": "string"},
				{"type": "array", "items": {"type": "string"}, "minItems": 1},
			],
			"examples": ["kit", ["kit", "room"]],
		},
		"channel": {
			"description": "The MIDI channel the assignment answers on. Left out, the map's `channel`, or the MIDI channel an ensemble plays the map on.",
			"$ref": "#/$defs/channel",
			"examples": [10, "my.kit"],
		},
		"notes":       {"$ref": "#/$defs/notes"},
		"velocity":    {"$ref": "#/$defs/velocity"},
		"select":      {"$ref": "#/$defs/select"},
		"process": {
			"description": "The processors the sound passes through, in order, rendered ahead of time. A processor is named alone for its defaults, or given its parameters. At most one of them may align the sound to the beat.",
			"type": "array",
			"items": {"$ref": "#/$defs/process_step"},
			"contains": _beat_aligning_step(),
			"maxContains": 1,
			"examples": [[
				"reverse",
				{"filter_low": {"freq": 800.0, "resonance": 6.0}},
				{"saturate": {"drive": 4.0}},
			]],
		},
		"mode": {
			"description": "How the sound answers the key: played to its end, played while the key is held, or looped while the key is held. Writing `loop:` sets the mode to `loop`.",
			"oneOf": _words(subsample.query.VALID_MODES, _MODES, "mode"),
			"default": "one_shot",
		},
		"loop":        {"$ref": "#/$defs/loop"},
		"release":     {"$ref": "#/$defs/release"},
		"extract":     {"$ref": "#/$defs/extract"},
		"gain": {
			"description": "Gain applied to the sound as it plays.",
			"type": "number",
			"default": 0.0,
			"x-unit": "dB",
			"examples": [-3.0],
		},
		"pan":         {"$ref": "#/$defs/pan"},
		"output":      {"$ref": "#/$defs/output"},
		"stack": {
			"description": "Lets the assignment sound together with another on the same note and velocity. Every assignment that overlaps must set it, or the map is refused.",
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
		"description": "The notes the assignment answers to: one note, a range written as `C2..C4` or `36..60`, a list of notes, or `zone-tuned`, which shares a range of the keyboard out among pitched samples. Without a `pick`, each note of a list plays the next rank, starting with the best match, unless the assignment repitches.",
		"anyOf": [
			{"$ref": "#/$defs/note"},
			{"type": "array", "items": {"$ref": "#/$defs/note"}, "minItems": 1},
			_zone_tuned(),
		],
		"examples": [
			36,
			"C2..C4",
			["drum.kick_1", "drum.kick_2"],
			{"mode": "zone-tuned", "range": ["C2", "C4"]},
		],
	}


def _note () -> dict[str, typing.Any]:

	"""One note: its number, its name, or a name a definitions file gives it.

	A string also carries the two other forms this position accepts: a range
	such as ``36..60`` or ``C2..C4``, and a number written as text."""

	return {
		"description": "One note: its MIDI number, its name, where `C4` is note 60, or a name from a definitions file or from the built-in General MIDI drum names, such as `drum.kick_1`.",
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
		"mode": {
			"description": "Shares the range out among the pitched samples that qualify, each on the notes nearest its own pitch. The assignment must repitch, and may neither stack nor be silenced.",
			"const": sentinel,
		},
		"range": {
			"description": "The part of the keyboard the zone covers, low note then high note.",
			"type": "array",
			"items": {"$ref": "#/$defs/note"},
			"minItems": 2,
			"maxItems": 2,
			"default": [0, 127],
			"examples": [["C2", "C4"]],
		},
	}

	return {
		"anyOf": [
			{"const": sentinel},
			{
				"type": "object",
				"properties": in_order(
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
		"trigger": {
			"description": "The velocities that play the assignment.",
			"$ref": "#/$defs/velocity_range",
			"examples": [[64, 127]],
		},
		"rescale": {
			"description": "Stretches the trigger range over a wider range of loudness, so a layer that only hears soft notes still plays through its whole dynamic range. `true` stretches it over every velocity, and a pair names the range. Left out, a velocity plays as it arrives.",
			"anyOf": [
				{"type": "boolean"},
				{"$ref": "#/$defs/velocity_range"},
			],
			"default": False,
			"examples": [[40, 127]],
		},
	}

	return {
		"description": "The velocities the assignment answers to, so several assignments can share a note as velocity layers: a pair, low then high, or `trigger` with `rescale`.",
		"default": [0, 127],
		"anyOf": [
			{"$ref": "#/$defs/velocity_range"},
			{
				"type": "object",
				"properties": in_order(
					subsample.player.VELOCITY_INNER_KEYS, terms, "velocity key",
				),
				"required": ["trigger"],
				"additionalProperties": False,
			},
		],
		"examples": [[0, 63], {"trigger": [64, 127], "rescale": True}],
	}


def _velocity_range () -> dict[str, typing.Any]:

	"""A pair of MIDI velocities, low then high."""

	return {
		"description": "A range of MIDI velocities, low then high, both included.",
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
		"description": "The notes on the same MIDI channel that cut this sound short, as closing a hi-hat stops its open ring. `self` means a new strike of this sound cuts the last. The cut is a quick damp that overrides any release. `false`, or an empty list, means nothing cuts it, which blanks a template's.",
		"anyOf": [
			choke,
			{"type": "array", "items": choke},
			{"const": False},
		],
		"examples": [
			"self",
			"drum.hi_hat_closed",
			["drum.hi_hat_closed", "drum.hi_hat_pedal"],
		],
	}


# ---------------------------------------------------------------------------
# Which sounds a note plays: select, where, order and pick
# ---------------------------------------------------------------------------

def _select () -> dict[str, typing.Any]:

	"""How an assignment chooses a sound: the whole library filtered, ranked and picked from.

	A list of these is a fallback chain, tried in turn until one finds a sound."""

	return {
		"description": "How the assignment chooses a sound from the library. A list is a fallback chain: each is tried in turn until one finds a sound, and the first `pick` in the chain governs the whole chain.",
		"anyOf": [
			{"$ref": "#/$defs/select_spec"},
			{"type": "array", "items": {"$ref": "#/$defs/select_spec"}, "minItems": 1},
		],
		"examples": [
			{"where": {"reference": "GM36_BassDrum1"}},
			[
				{"where": {"name": "my-favourite-kick"}},
				{"where": {"reference": "GM36_BassDrum1"}},
			],
		],
	}


def _select_spec () -> dict[str, typing.Any]:

	"""One way of choosing a sound: a filter, a ranking and a pick."""

	import subsample.query

	terms = {
		"where":    {"$ref": "#/$defs/where"},
		"order":    {"$ref": "#/$defs/order"},
		"order_by": _deprecated({
			"description": "The older spelling of `order`.",
			"$ref": "#/$defs/order",
		}),
		"pick":     {"$ref": "#/$defs/pick"},
	}

	return {
		"description": "One way of choosing a sound: which samples qualify, how they are ranked, and which of them plays.",
		"type": "object",
		"properties": in_order(subsample.query.VALID_SELECT_KEYS, terms, "select key"),
		"not": {"required": ["order", "order_by"]},
		"additionalProperties": False,
	}


def _where () -> dict[str, typing.Any]:

	"""What a sound must be for this assignment to play it."""

	import subsample.query

	terms: dict[str, typing.Any] = {
		"name": _name_term(),
		"path": {
			"description": "One audio file, by its path relative to the map. It may not be combined with `name`.",
			"type": "string",
			"minLength": 1,
			"examples": ["kicks/909-kick.wav"],
		},
		"directory": {
			"description": "The samples inside this directory and the directories within it, relative to the map. Subsample loads them at startup.",
			"type": "string",
			"minLength": 1,
			"examples": ["kits/acoustic"],
		},
		"reference": {
			"description": "Ranks the samples by how closely they resemble a reference: a built-in reference by name, such as `GM36_BassDrum1`, or an audio file by its path relative to the map. An assignment whose reference name is unknown is left out, with a warning.",
			"type": "string",
			"minLength": 1,
			"examples": ["GM36_BassDrum1", "references/my-kick.wav"],
		},
		"pitched": {
			"description": "`true` keeps only samples with a stable pitch, and `false` only samples without one.",
			"type": "boolean",
		},
		"loopable": {
			"description": "`true` keeps only samples with a steady part worth looping, and `false` only samples without one.",
			"type": "boolean",
		},
	}

	for key in subsample.query.NUMERIC_YAML_KEYS:
		terms[key] = _measurement(key)

	yaml_key = {field: key for key, field in subsample.query.NUMERIC_YAML_KEYS.items()}

	# An older spelling says only which current spelling it stands for, and
	# says it from the parser's own table, so the two cannot disagree.
	for key, (field, operator) in subsample.query.LEGACY_WHERE_KEYS.items():
		terms[key] = _deprecated({
			"description": f"The older spelling of `{operator}` under `{yaml_key[field]}`.",
			**_measured_value(field),
		})

	accepted = (
		*subsample.query.NON_RANGE_WHERE_KEYS,
		*subsample.query.NUMERIC_YAML_KEYS,
		*subsample.query.LEGACY_WHERE_KEYS,
	)

	return {
		"description": "Which samples qualify. Every condition written must hold. Left out, every sample qualifies.",
		"type": "object",
		"properties": in_order(accepted, terms, "where key"),
		"additionalProperties": False,
		"examples": [{"pitched": True, "duration": {"gte": 1.0}}],
	}


def _measurement (key: str) -> dict[str, typing.Any]:

	"""One measured quality of a sound: bounds it lies within, or the one value a count has."""

	import subsample.query

	value = _measured_value(subsample.query.NUMERIC_YAML_KEYS[key])
	exact = key in subsample.query.EXACT_WHERE_KEYS

	operators = tuple(
		operator for operator in subsample.query.VALID_OPERATORS
		if exact or operator != "eq"
	)

	bounds: dict[str, typing.Any] = {
		"type": "object",
		"properties": {
			operator: {"description": description, **value}
			for operator, description in in_order(
				operators,
				{name: prose for name, prose in _MEASUREMENT_BOUNDS.items() if name in operators},
				"measurement operator",
			).items()
		},
		"minProperties": 1,
		"additionalProperties": False,
	}

	term: dict[str, typing.Any] = {"description": _MEASUREMENTS[key]}

	# Only a whole count is a value a sample really has, so everything else
	# here publishes bounds and nothing else (#3018).
	term.update({"anyOf": [value, bounds]} if exact else bounds)

	if key in _WHERE_UNITS:
		term["x-unit"] = _WHERE_UNITS[key]

	term["examples"] = _MEASUREMENT_EXAMPLES[key]

	return term


def _measured_value (field: str) -> dict[str, typing.Any]:

	"""What one measurement is written as: a number, or a note name where a pitch is asked for."""

	if field == "pitch_hz":
		return {"anyOf": [{"type": "number"}, {"type": "string", "minLength": 1}]}

	return {"type": "number"}


def _name_term () -> dict[str, typing.Any]:

	"""A sound named outright, named among several, or matched by a wildcard."""

	import subsample.query

	return {
		"description": "Samples by file name, without the folder or the extension: one name exactly, any of a list of names, or a name matched by wildcards or a regular expression. Names are not unique, so a name may match several samples. An exact name matches letter case, and a wildcard or a regular expression does not.",
		"anyOf": [
			{"type": "string", "minLength": 1},
			{"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "uniqueItems": True},
			{
				"type": "object",
				"properties": {
					operator: {
						"description": description,
						"type": "string",
						"minLength": 1,
						"examples": [example],
					}
					for operator, (description, example) in in_order(
						subsample.query.VALID_NAME_OPERATORS, _NAME_MATCHES, "name operator",
					).items()
				},
				"minProperties": 1,
				"maxProperties": 1,
				"additionalProperties": False,
			},
		],
		"examples": ["909-kick", ["909-kick", "808-kick"], {"matches": "kick*"}],
	}


def _order () -> dict[str, typing.Any]:

	"""How the sounds that matched are ranked, best first."""

	return {
		"description": "How the qualifying samples are ranked: one ranking, or a list in which each ranking breaks the ties the one before it leaves. Left out, the newest sample comes first, or the closest match where `where` names a `reference`.",
		"anyOf": [
			{"$ref": "#/$defs/order_clause"},
			{"type": "array", "items": {"$ref": "#/$defs/order_clause"}, "minItems": 1},
		],
		"examples": [
			{"by": "level", "dir": "desc"},
			[{"by": "duration", "dir": "asc"}, {"by": "level", "dir": "desc"}],
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

	# An older word says only which ranking it stands for, from the parser's
	# own table, so the two cannot disagree.
	older: list[dict[str, typing.Any]] = []

	titles = in_order(tuple(subsample.query.LEGACY_ORDER_TOKENS), _OLDER_ORDER_TITLES, "older order word")

	for token, title in titles.items():
		clause = subsample.query.LEGACY_ORDER_TOKENS[token]
		older.append(_deprecated(_word(
			token, title, f"The older spelling of `{{by: {clause.by}, dir: {clause.dir}}}`.",
		)))

	return {
		"description": "One ranking: what the samples are ranked by, and which way round.",
		"anyOf": [
			{"oneOf": older},
			{
				"type": "object",
				"properties": {
					"by": {
						"description": "What the samples are ranked by.",
						"oneOf": _words(subsample.query.valid_order_names(), _ORDER_BY, "order name"),
					},
					"dir": {
						"description": "Which end of the ranking comes first. Left out, the best match comes first for `beat_match` and `similarity`, and the lowest value for everything else.",
						"oneOf": _words(("asc", "desc"), _ORDER_DIRECTIONS, "order direction"),
					},
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
		"mode": {
			"description": "Chooses by velocity, with the settings beside it.",
			"const": "velocity",
		},
		"variation": {
			"description": "How far the choice may stray from the velocity played, in velocity values across both directions: 10 strays up to 5 either way. The loudness still follows the velocity played.",
			"type": "integer",
			"minimum": 0,
			"maximum": 127,
			"default": 0,
			"examples": [10],
		},
		"curve": {
			"description": "How the velocity played maps across the ranking.",
			"oneOf": _words(subsample.query.VALID_PICK_CURVES, _PICK_CURVES, "pick curve"),
			"default": "linear",
		},
		"spacing": {
			"description": "How the ranked samples are laid out across the range of velocities.",
			"oneOf": _words(subsample.query.VALID_PICK_SPACINGS, _PICK_SPACINGS, "pick spacing"),
			"default": "rank",
		},
	}

	return {
		"description": "Which of the ranked samples plays: a rank, a range of ranks drawn from at random on every note, `any` for any match at random, or `velocity` to choose by how hard the note is struck. Left out, a single note plays the best match, and each note of a list plays the next rank unless the assignment repitches.",
		"anyOf": [
			{"oneOf": _words(("any", "velocity"), _PICK_WORDS, "pick word")},
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
					operator: {"description": description, **rank}
					for operator, description in in_order(
						subsample.query.VALID_PICK_OPERATORS, _RANK_BOUNDS, "pick operator",
					).items()
				},
				"minProperties": 1,
				"additionalProperties": False,
			},
			{
				"type": "object",
				"properties": in_order(
					subsample.query.VALID_VELOCITY_PICK_KEYS, velocity_terms, "velocity pick key",
				),
				"required": ["mode"],
				"additionalProperties": False,
			},
		],
		"examples": [2, [1, 4], {"gte": 3}, {"mode": "velocity", "variation": 10}],
	}


# ---------------------------------------------------------------------------
# How a sound plays: release, loop, extract, pan and output
# ---------------------------------------------------------------------------

def _release () -> dict[str, typing.Any]:

	"""What the sound does when the key comes up."""

	import subsample.player
	import subsample.query

	curve = {
		"description": "The shape of the fade.",
		"oneOf": _words(subsample.query.VALID_RELEASE_CURVES, _RELEASE_CURVES, "release curve"),
		"default": "cosine",
	}

	time = {"$ref": "#/$defs/release_time"}

	spelt_out = {
		"type": "object",
		"properties": in_order(
			subsample.player.RELEASE_INNER_KEYS, {"time": time, "curve": curve}, "release key",
		),
		"additionalProperties": False,
	}

	# The knob written where the time itself would go, which is the same
	# binding with the fade's shape beside it.  Its examples are the fade's own
	# times, because the binding's are a filter's frequencies.
	times: dict[str, list[typing.Any]] = {
		"cc": [72], "min": [20], "max": [3000], "default": [400],
	}

	fields = {
		name: ({**field, "examples": times[name]} if name in times else field)
		for name, field in typing.cast(
			dict[str, typing.Any], _cc_binding()["properties"],
		).items()
	}

	bound = {
		"type": "object",
		"properties": {**fields, "curve": curve},
		"required": ["cc"],
		"additionalProperties": False,
	}

	return {
		"description": "What the sound does after note-off. Left out, a gated sound stops with a short fade that avoids a click, and a looping one fades with the adaptive tail. `true` is the adaptive tail, shaped from the sample. A number is the fade time. `full` lets the sound play on to its end with no fade. A one-shot sound never receives note-off, so a release has no effect on it.",
		"anyOf": [
			{"type": "boolean"},
			_word("full", "Full", "Plays on to the end of the sound with no fade. A looping sound stops looping and rings out its natural tail."),
			{"$ref": "#/$defs/release_time"},
			spelt_out,
			bound,
		],
		"examples": [
			250,
			{"time": 250, "curve": "exponential"},
			{"cc": 72, "max": 3000, "curve": "exponential"},
		],
	}


def _release_time () -> dict[str, typing.Any]:

	"""How long the sound takes to fade once the key comes up, or the knob that sets it."""

	import subsample.player

	return {
		"description": "How long the fade after note-off lasts. A knob's value is read as each note is struck, so it shapes the notes played next. Bound to a knob with no `default:`, it keeps the adaptive tail until the knob first moves.",
		"anyOf": [
			{"type": "number", "minimum": 0},
			{"$ref": "#/$defs/cc_binding"},
		],
		"x-unit": "ms",
		"x-sweep": list(subsample.player.RELEASE_CC_SWEEP_MS),
		"examples": [250],
	}


def _loop () -> dict[str, typing.Any]:

	"""Where the sound loops while the key is held."""

	import subsample.player

	terms = {
		"start": {
			"description": "Where the loop begins, from the start of the sample.",
			"type": "number",
			"minimum": 0,
			"x-unit": "s",
			"examples": [0.5],
		},
		"end": {
			"description": "Where the loop ends, from the start of the sample. It must come after `start`.",
			"type": "number",
			"minimum": 0,
			"x-unit": "s",
			"examples": [2.5],
		},
		"crossfade": {
			"description": "How long the join is blended over, so the loop repeats without a click.",
			"type": "number",
			"minimum": 0,
			"x-unit": "ms",
			"examples": [20],
		},
	}

	return {
		"description": "Where the sound loops while the key is held. A point left out is found automatically, and writing `loop:` at all sets the mode to `loop`. A sample with no clean loop plays gated instead, with a note in the log.",
		"type": "object",
		"properties": in_order(subsample.player.LOOP_INNER_KEYS, terms, "loop key"),
		"additionalProperties": False,
		"examples": [{"start": 0.5, "end": 2.5, "crossfade": 20}],
	}


def _extract () -> dict[str, typing.Any]:

	"""Which part of a multi-channel recording this assignment plays."""

	import subsample.player
	import subsample.query

	blend = {
		"description": "A mix of the audio channels into mono: one weight for each, where a negative weight flips the polarity of its audio channel. The weights are scaled to add up to one, so only their balance matters.",
		"type": "array",
		"items": {"type": "number"},
		"minItems": 1,
		"examples": [[1, -1]],
	}

	return {
		"description": "Plays one part of a multi-channel recording as mono, as a microphone facing a chosen way would hear it, before `pan` and `output` place it. `channel.2` plays the second audio channel alone. The same sample plays whole wherever another assignment does not extract it. A part the recording cannot give is refused when the map loads.",
		"anyOf": [
			{"oneOf": _words(subsample.query.EXTRACT_KINDS, _EXTRACT_PARTS, "extract kind")},
			{"type": "string", "pattern": r"^channel\.[0-9]+$"},
			{
				"type": "object",
				"properties": in_order(
					subsample.player.VALID_EXTRACT_KEYS, {"blend": blend}, "extract key",
				),
				"required": ["blend"],
				"additionalProperties": False,
			},
		],
		"examples": ["channel.2", {"blend": [1, -1]}],
	}


def _pan () -> dict[str, typing.Any]:

	"""Where the sound sits across the speakers."""

	import subsample.channel
	import subsample.query

	position: dict[str, typing.Any] = {"$ref": "#/$defs/pan_position"}

	terms = {
		"gte": {
			"description": "The leftmost position a random pan may land on.",
			**position,
			"examples": [-60],
		},
		"lte": {
			"description": "The rightmost position a random pan may land on.",
			**position,
			"examples": [60],
		},
		"position": {
			"description": "The centre a random pan lands around.",
			**position,
			"examples": [-20],
		},
		"variation": {
			"description": "How widely a random pan spreads around `position`: 40 lands up to 20 either side.",
			"type": "number",
			"minimum": 0,
			"maximum": 200,
			"default": 0,
			"examples": [40],
		},
	}

	return {
		"description": "Where the sound sits across the outputs: a position from hard left to hard right, a list of relative weights with one for each audio channel of a standard layout, `any` for a new random position on every note, or bounds to draw one from. A random position keeps the sound at the same loudness wherever it lands.",
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
			_word("any", "Any", "A new random position on every note, anywhere from hard left to hard right."),
			{
				"type": "object",
				"properties": in_order(subsample.query.VALID_PAN_KEYS, terms, "pan key"),
				"minProperties": 1,
				"additionalProperties": False,
			},
		],
		"examples": [25, [50, 87], {"gte": -60, "lte": 60}, {"position": -20, "variation": 40}],
	}


def _output () -> dict[str, typing.Any]:

	"""The device channels this assignment plays out of, counting from 1."""

	return {
		"description": "The outputs of the audio device the sound plays through, counted from 1 as the hardware labels them. With a list of pan weights, give one output for each weight, and with a random pan, give two. Left out, the first outputs.",
		"type": "array",
		"items": {"type": "integer", "minimum": 1},
		"minItems": 1,
		"uniqueItems": True,
		"examples": [[3, 4]],
	}


# ---------------------------------------------------------------------------
# Programs and ensembles
# ---------------------------------------------------------------------------

def _program () -> dict[str, typing.Any]:

	"""One program: the sounds a Program Change message switches to."""

	import subsample.bank

	terms = {
		"name": {
			"description": "A name for the program, which log lines use.",
			"type": "string",
			"minLength": 1,
			"examples": ["Acoustic kit"],
		},
		"program": {
			"description": "The Program Change number that selects the program. Left out, its place in the list, counted from 0.",
			"$ref": "#/$defs/program_number",
			"examples": [1],
		},
		"directory": {
			"description": "A directory of samples for the map's own assignments to choose from while the program is active, relative to where Subsample runs.",
			"type": "string",
			"minLength": 1,
			"examples": ["kits/acoustic"],
		},
		"map": {
			"description": "A whole map, with its own assignments and samples, relative to this map. It may not declare programs of its own.",
			"type": "string",
			"minLength": 1,
			"examples": ["kits/808-kit.yaml"],
		},
	}

	return {
		"description": "One program: the instrument set a Program Change message switches to. It names either a directory of samples for this map's assignments, or a whole map of its own.",
		"type": "object",
		"properties": in_order(subsample.bank.VALID_PROGRAM_KEYS, terms, "program key"),
		"required": ["name"],
		"oneOf": [{"required": ["directory"]}, {"required": ["map"]}],
		"additionalProperties": False,
	}


def _included_map () -> dict[str, typing.Any]:

	"""One map an ensemble plays, and the channel it answers on."""

	import subsample.ensemble

	terms = {
		"map": {
			"description": "The map to play, by its path relative to this map.",
			"type": "string",
			"minLength": 1,
			"examples": ["drums.yaml"],
		},
		"channel": {
			"description": "The MIDI channel to play the map on, in place of the one it declares. An assignment that names its own MIDI channel keeps it.",
			"$ref": "#/$defs/channel",
			"examples": [2],
		},
	}

	return {
		"description": "One map an ensemble plays: its path alone, which keeps the MIDI channel the map declares, or its path with a MIDI channel to play it on.",
		"anyOf": [
			{"type": "string", "minLength": 1},
			{
				"type": "object",
				"properties": in_order(
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
		"description": "A name from a mounted definitions file, written as its prefix and its name, such as `my.kick`.",
		"type": "string",
		"pattern": f"^{subsample.definitions.SYMBOL_RE.pattern}$",
	}


def _written_number () -> dict[str, typing.Any]:

	"""A whole number written as text, which a map may do anywhere a number is asked for."""

	return {
		"description": "A whole number written as text, which the map reads as that number.",
		"type": "string",
		"pattern": r"^\s*[+-]?[0-9]+\s*$",
	}


def _number_or_name (
	minimum:     int,
	maximum:     int,
	description: str,
	default:     typing.Any = _ABSENT,
	examples:    typing.Optional[list[typing.Any]] = None,
) -> dict[str, typing.Any]:

	"""A whole number, or a name a definitions file gives that number."""

	term: dict[str, typing.Any] = {
		"description": description,
		"anyOf": [
			{"type": "integer", "minimum": minimum, "maximum": maximum},
			{"$ref": "#/$defs/defined_name"},
			{"$ref": "#/$defs/written_number"},
		],
	}

	if default is not _ABSENT:
		term["default"] = default

	if examples:
		term["examples"] = examples

	return term


def _words (
	accepted: typing.Sequence[str],
	prose:    typing.Mapping[str, tuple[str, str]],
	what:     str,
) -> list[dict[str, typing.Any]]:

	"""The words the parser accepts, in its order, each with its title and description."""

	return [
		_word(value, title, description)
		for value, (title, description) in in_order(accepted, prose, what).items()
	]


def in_order (
	accepted: typing.Sequence[str],
	declared: typing.Mapping[str, typing.Any],
	what:     str,
) -> dict[str, typing.Any]:

	"""The declared terms, in the order the parser's own list gives them.

	Reading the parser's list is what keeps the two together: a key added to one
	and not the other stops the schema being built at all, rather than going out
	as a reference that quietly omits it.

	subsample.definitions_schema declares a second file format the same way and
	calls this, rather than keeping a guard of its own that could differ."""

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
		"cc": {
			"description": "The MIDI controller that sets the value.",
			"$ref": "#/$defs/controller",
			"examples": [74, "my.sampler_release"],
		},
		"channel": {
			"description": "The only MIDI channel the controller is read on. Left out, every MIDI channel.",
			"$ref": "#/$defs/channel",
			"examples": [2],
		},
		"min": {
			"description": "The value at the bottom of the knob's travel. Left out, the bottom of the parameter's own range. A `min` above `max` turns the knob round.",
			"type": "number",
			"examples": [200],
		},
		"max": {
			"description": "The value at the top of the knob's travel. Left out, the top of the parameter's own range.",
			"type": "number",
			"examples": [8000],
		},
		"default": {
			"description": "The value until the controller first moves. Left out, the value the parameter has without the knob, or the middle of the knob's travel where that value lies outside it.",
			"type": "number",
			"examples": [1000],
		},
	}

	return {
		"description": "A MIDI controller in place of a fixed value, so a knob or a fader sets it. A value outside what the parameter allows is refused when the map loads, and so is a binding on a parameter that takes a word.",
		"type": "object",
		"properties": {key: fields[key] for key in subsample.query.CC_BINDING_KEYS},
		"required": ["cc"],
		"additionalProperties": False,
		"examples": [
			{"cc": 74},
			{"cc": 74, "channel": 2, "min": 200, "max": 8000, "default": 1000},
		],
	}


# ---------------------------------------------------------------------------
# The processors, generated from subsample.processors
# ---------------------------------------------------------------------------

def _process_step () -> dict[str, typing.Any]:

	"""One entry of a `process:` list: a processor named on its own, or named with what it takes."""

	return {
		"description": "One processor: its name alone, for its defaults, or its name with its parameters.",
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

		# An older name is the same processor, so it carries the same words.
		for legacy in processor.legacy_names:
			words.append(_deprecated(_word(legacy.name, processor.title, processor.description)))

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

	value: dict[str, typing.Any] = {
		"title": processor.title,
		"description": processor.description,
		"anyOf": forms,
	}

	if processor.examples:
		value["examples"] = list(processor.examples)

	return value


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

	if parameter.examples:
		entry["examples"] = list(parameter.examples)

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

	"""The same term under a name Subsample still accepts but no longer documents.

	Nothing under it carries an example: an example is there to be copied, and
	nothing should copy a spelling that is on its way out."""

	return {**_unexampled(entry), "deprecated": True}


def _unexampled (node: typing.Any) -> typing.Any:

	"""The same schema with every example taken out of it, however deep it sits."""

	if isinstance(node, dict):
		return {
			key: _unexampled(value) for key, value in node.items()
			if key != "examples"
		}

	if isinstance(node, list):
		return [_unexampled(item) for item in node]

	return node

