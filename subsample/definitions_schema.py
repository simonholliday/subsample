"""A declared schema of the definitions file, in the names the file uses.

A definitions file is the small file a music project keeps its own names in, so
a MIDI map can write ``my.kick`` where it would otherwise write 36.  Subsample
reads four sections of it and leaves the rest alone, because the file belongs to
the project rather than to this app: a sequencer may keep its own names beside
Subsample's.

subsample.definitions is what actually reads one.  This module declares that
format a second time, with the ranges, the name rule and the prose a reader
needs, and returns it as one JSON Schema.  Nothing here validates a file or
changes what loads.

A second list can drift from the first, so tests/test_definitions_schema.py is
the load-bearing half: it loads real definitions files, and fails when a section
published here is not one Subsample reads, when a range published here is not
enforced, or when an example does not load.

The prose is published on subsystem.co as it stands, so it uses British spelling
and never an em dash.
"""

import typing

import subsample.definitions
import subsample.midi_map_schema


_SECTIONS: typing.Final[dict[str, tuple[str, str]]] = {
	"notes": (
		"The notes the project has names for, so a map can answer to `my.kick` instead of to 36. A name works anywhere a map takes a note, and the built-in General MIDI drum names under `drum` are the same idea with the names already filled in.",
		"The MIDI note the name stands for, where 60 is `C4`.",
	),
	"cc": (
		"The controllers the project has names for, so a map can put a parameter on `my.cutoff` instead of on 74, and moving the knob to another controller is one edit here.",
		"The MIDI controller the name stands for.",
	),
	"channels": (
		"The MIDI channels the project has names for, so a map can answer on `my.kit` instead of on 10, and a project that moves its drums to another MIDI channel changes one line.",
		"The MIDI channel the name stands for, counted from 1.",
	),
	"programs": (
		"The programs the project has names for, so a map can say which instrument set a Program Change message switches to by name.",
		"The Program Change number the name stands for.",
	),
}
"""What each section a map may take names from holds, and what one entry means."""

_EXAMPLES: typing.Final[dict[str, dict[str, int]]] = {
	"notes":    {"kick": 36, "snare": 38},
	"cc":       {"cutoff": 74, "sampler_release": 21},
	"channels": {"kit": 10, "bass": 2},
	"programs": {"brushes": 1},
}
"""What a project writes in each section, one name it would really give."""


def json_schema () -> dict[str, typing.Any]:

	"""Return the JSON Schema of a definitions file, in the order a reference lists it."""

	return {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"title": "Subsample definitions file",
		"description": "The file a music project keeps its own names in: the notes, controllers, MIDI channels and programs it works with, each given its number once. A MIDI map mounts the file under a prefix it chooses, with `definitions: {my: project.yaml}`, and then writes `my.kick` wherever it would write the number. Subsample reads the four sections below and leaves every other section alone, so the same file can name things for other tools without either tool knowing about the other.",
		"type": "object",
		"properties": subsample.midi_map_schema.in_order(
			subsample.definitions.CONSUMED_SECTIONS,
			{name: _section(name) for name in _SECTIONS},
			"definitions section",
		),

		# The file belongs to the project, so a section Subsample does not read
		# is another tool's business rather than a mistake.
		"additionalProperties": True,
		"examples": [{"notes": {"kick": 36, "snare": 38}, "channels": {"kit": 10}}],
	}


def _section (name: str) -> dict[str, typing.Any]:

	"""One section: the names a project gives one kind of number."""

	description, value = _SECTIONS[name]
	low, high = subsample.definitions.SECTION_RANGES[name]

	return {
		"description": description,
		"type": "object",
		"propertyNames": {
			"description": "A name is lowercase letters, digits and underscores, starting with a letter. A map may write it in any case.",
			"pattern": f"^{subsample.definitions.NAME_RE.pattern}$",
		},
		"additionalProperties": {
			"description": value,
			"type": "integer",
			"minimum": low,
			"maximum": high,
		},
		"examples": [_EXAMPLES[name]],
	}
