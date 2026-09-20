"""Tests for subsample.midi_map_schema — the declared map agrees with the parser.

The parser is the map's real grammar.  The schema declares it a second time so
the documentation and Superconductor's catalogue can be generated from it, and a
second list can drift from the first.  These tests are what stop it: they load
real maps through subsample.player, so a word, a limit or a default published
here is one Subsample actually keeps.

When one fails, the fix is to change the schema to match the parser, unless the
parser is the half that is wrong.
"""

import dataclasses
import json
import logging
import pathlib
import typing

import pytest
import yaml

import subsample.bank
import subsample.channel
import subsample.definitions
import subsample.ensemble
import subsample.midi_map_schema
import subsample.player
import subsample.processors
import subsample.query


_SCHEMA = subsample.midi_map_schema.json_schema()

_LIMIT_KEYWORDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


# ---------------------------------------------------------------------------
# Reading the schema
# ---------------------------------------------------------------------------

def _subschemas (
	node: dict[str, typing.Any],
	path: str = "",
) -> typing.Iterator[tuple[str, dict[str, typing.Any]]]:

	"""Every schema inside this one, with the path that reaches it."""

	yield path, node

	for keyword in ("items", "not", "if", "then", "contains", "propertyNames", "additionalProperties"):
		child = node.get(keyword)

		if isinstance(child, dict):
			yield from _subschemas(child, f"{path}/{keyword}")

	for keyword in ("anyOf", "oneOf", "allOf"):
		for index, child in enumerate(node.get(keyword, ())):
			yield from _subschemas(child, f"{path}/{keyword}/{index}")

	for container in ("properties", "$defs"):
		for name, child in node.get(container, {}).items():
			yield from _subschemas(child, f"{path}/{container}/{name}")


def _grammar () -> list[tuple[str, dict[str, typing.Any]]]:

	"""Every schema this module declares by hand, leaving out the generated processors."""

	return [
		(path, node) for path, node in _subschemas(_SCHEMA)
		if "$defs/process_step" not in path
	]


def _limits () -> list[typing.Any]:

	"""One case per limit the grammar publishes."""

	cases = []

	for path, node in _grammar():
		for keyword in _LIMIT_KEYWORDS:
			if keyword in node:
				cases.append(pytest.param(path.lstrip("/"), keyword, node[keyword], id=f"{path.lstrip('/')}-{keyword}"))

	return cases


def _defaults () -> list[typing.Any]:

	"""One case per default the grammar publishes."""

	return [
		pytest.param(path.lstrip("/"), node["default"], id=path.lstrip("/"))
		for path, node in _grammar() if "default" in node
	]


def _words (path: str) -> list[str]:

	"""The words a term allows, in the order it declares them."""

	node = _at(path)

	return [option["const"] for option in node["oneOf"]]


def _at (path: str) -> dict[str, typing.Any]:

	"""The schema at this path."""

	for found, node in _subschemas(_SCHEMA):
		if found == path:
			return node

	raise AssertionError(f"no schema at {path}")


# ---------------------------------------------------------------------------
# Loading real maps
# ---------------------------------------------------------------------------

def _assignment (**fields: typing.Any) -> dict[str, typing.Any]:

	"""The smallest assignment that loads, with these fields written into it."""

	entry: dict[str, typing.Any] = {
		"name":    "Probe",
		"channel": 1,
		"notes":   60,
		"select":  {"where": {"pitched": True}},
	}
	entry.update(fields)

	return entry


def _map (**keys: typing.Any) -> dict[str, typing.Any]:

	"""The smallest map that loads, with these keys written into it."""

	mapping: dict[str, typing.Any] = {"assignments": [_assignment()]}
	mapping.update(keys)

	return mapping


def _load (
	tmp_path:   pathlib.Path,
	mapping:    dict[str, typing.Any],
	references: typing.Optional[list[str]] = None,
) -> subsample.player.MidiMapResult:

	"""Load a map as Subsample loads one, refusing what a real map would be refused for."""

	path = tmp_path / "midi-map.yaml"
	path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

	return subsample.player.load_midi_map(path, references or [])


def _first (result: subsample.player.MidiMapResult) -> subsample.query.Assignment:

	"""The assignment the probe map declares."""

	return typing.cast(subsample.query.Assignment, result.note_map[(0, 60)][0][0])


# ---------------------------------------------------------------------------
# Where each published limit and default is written in a real map
# ---------------------------------------------------------------------------

_Probe = typing.Callable[[typing.Any], dict[str, typing.Any]]

_LIMIT_PROBES: dict[str, _Probe] = {
	"properties/program_channel/anyOf/0":
		lambda value: _map(program_channel=value),
	"$defs/order_clause/anyOf/1/properties/pattern/items":
		lambda value: _map(assignments=[_assignment(
			select={"where": {"pitched": True}, "order": [{"by": "beat_match", "pattern": [value, 1]}]},
		)]),
	"$defs/channel/anyOf/0":
		lambda value: _map(assignments=[_assignment(channel=value)]),
	"$defs/program_number/anyOf/0":
		lambda value: _map(programs=[{"name": "Kit", "directory": "sounds", "program": value}]),
	"$defs/controller/anyOf/0":
		lambda value: _map(assignments=[_assignment(process=[{"filter_low": {"freq": {"cc": value}}}])]),
	"$defs/rank":
		lambda value: _map(assignments=[_assignment(select={"where": {"pitched": True}, "pick": value})]),
	"$defs/pan_position":
		lambda value: _map(assignments=[
			_assignment(pan=value),
			_assignment(notes=61, pan={"position": value}),
		]),
	"$defs/release_time/anyOf/0":
		lambda value: _map(assignments=[_assignment(mode="gated", release=value)]),
	"$defs/note/anyOf/0":
		lambda value: _map(assignments=[_assignment(notes=value)]),
	"$defs/velocity_range/items":
		lambda value: _map(assignments=[_assignment(velocity=[value, value])]),
	"$defs/pick/anyOf/4/properties/variation":
		lambda value: _map(assignments=[_assignment(select={
			"where": {"pitched": True},
			"order": "loudest",
			"pick":  {"mode": "velocity", "variation": value},
		})]),
	"$defs/loop/properties/start":
		lambda value: _map(assignments=[_assignment(loop={"start": value})]),
	"$defs/loop/properties/end":
		lambda value: _map(assignments=[_assignment(loop={"end": value})]),
	"$defs/loop/properties/crossfade":
		lambda value: _map(assignments=[_assignment(loop={"crossfade": value})]),
	"$defs/pan/anyOf/1/items":
		lambda value: _map(assignments=[_assignment(pan=[value, 1.0])]),
	"$defs/pan/anyOf/3/properties/variation":
		lambda value: _map(assignments=[_assignment(pan={"position": 0, "variation": value})]),
	"$defs/output/items":
		lambda value: _map(assignments=[_assignment(output=[value])]),
}
"""A map that writes one value where a published limit governs it, by the path
of the term that publishes the limit.  A limit with no probe fails the suite:
what a schema says Subsample refuses, Subsample has to refuse."""


_Reading = typing.Callable[[subsample.player.MidiMapResult], typing.Any]

_DEFAULT_PROBES: dict[str, tuple[dict[str, typing.Any], _Reading]] = {
	"properties/program_channel": (
		_map(), lambda result: result.bank_channel,
	),
	"$defs/assignment/properties/name": (
		_map(assignments=[{k: v for k, v in _assignment().items() if k != "name"}]),
		lambda result: _first(result).name,
	),
	"$defs/assignment/properties/mode": (
		_map(), lambda result: _first(result).mode,
	),
	"$defs/assignment/properties/gain": (
		_map(), lambda result: _first(result).gain_db,
	),
	"$defs/assignment/properties/stack": (
		_map(), lambda result: _first(result).stack,
	),
	"$defs/template/properties/name": (
		_map(templates={"base": {}}, assignments=[
			{k: v for k, v in _assignment(template="base").items() if k != "name"},
		]),
		lambda result: _first(result).name,
	),
	"$defs/template/properties/mode": (
		_map(templates={"base": {}}, assignments=[_assignment(template="base")]),
		lambda result: _first(result).mode,
	),
	"$defs/template/properties/gain": (
		_map(templates={"base": {}}, assignments=[_assignment(template="base")]),
		lambda result: _first(result).gain_db,
	),
	"$defs/template/properties/stack": (
		_map(templates={"base": {}}, assignments=[_assignment(template="base")]),
		lambda result: _first(result).stack,
	),
	"$defs/notes/anyOf/2/anyOf/1/properties/range": (
		_map(assignments=[_assignment(notes="zone-tuned", process=[{"repitch": True}])]),
		lambda result: list(result.zone_templates[0].keyboard_range),
	),
	"$defs/velocity": (
		_map(), lambda result: list(_first(result).velocity_trigger),
	),
	"$defs/velocity/anyOf/1/properties/rescale": (
		_map(assignments=[_assignment(velocity={"trigger": [0, 127]})]),
		lambda result: _first(result).velocity_rescale_to is not None,
	),
	"$defs/pick/anyOf/4/properties/variation": (
		_map(assignments=[_assignment(select={
			"where": {"pitched": True}, "order": "loudest", "pick": {"mode": "velocity"},
		})]),
		lambda result: _first(result).select[0].pick.variation,
	),
	"$defs/pick/anyOf/4/properties/curve": (
		_map(assignments=[_assignment(select={
			"where": {"pitched": True}, "order": "loudest", "pick": {"mode": "velocity"},
		})]),
		lambda result: _first(result).select[0].pick.curve,
	),
	"$defs/pick/anyOf/4/properties/spacing": (
		_map(assignments=[_assignment(select={
			"where": {"pitched": True}, "order": "loudest", "pick": {"mode": "velocity"},
		})]),
		lambda result: _first(result).select[0].pick.spacing,
	),
	"$defs/release/anyOf/3/properties/curve": (
		_map(assignments=[_assignment(mode="gated", release={"time": 100})]),
		lambda result: _first(result).release.curve,
	),
	"$defs/release/anyOf/4/properties/curve": (
		_map(assignments=[_assignment(mode="gated", release={"cc": 74})]),
		lambda result: _first(result).release.curve,
	),
	"$defs/pan/anyOf/3/properties/variation": (
		_map(assignments=[_assignment(pan={"position": 0})]),
		lambda result: _first(result).pan_spec.hi - _first(result).pan_spec.lo,
	),
}
"""A map that leaves one term out, and what the loaded map then holds, by the
path of the term that publishes the default."""


# ---------------------------------------------------------------------------
# Where each published example is written in a real map
# ---------------------------------------------------------------------------

_DEFINITIONS: typing.Final[dict[str, typing.Any]] = {
	"notes":    {"kick": 36},
	"cc":       {"sampler_release": 21},
	"channels": {"kit": 10},
	"programs": {"brushes": 1},
}
"""A definitions file the probes mount as `my`, naming one of each kind, so an
example written as `my.kit` has something to resolve to."""


@dataclasses.dataclass(frozen=True)
class _Case:

	"""A map with one example written where it belongs, and what it needs around it."""

	mapping:     dict[str, typing.Any]
	references:  tuple[str, ...] = ()
	files:       dict[str, typing.Any] = dataclasses.field(default_factory=dict)
	ensemble:    bool = False


_Probe = typing.Callable[[typing.Any], _Case]


def _defined (mapping: dict[str, typing.Any]) -> _Case:

	"""The same map with the probe definitions file mounted under `my`."""

	return _Case(
		{"definitions": {"my": "project.yaml"}, **mapping},
		files={"project.yaml": _DEFINITIONS},
	)


def _played (**fields: typing.Any) -> _Case:

	"""A map whose one assignment carries these fields, with the definitions file mounted."""

	return _defined(_map(assignments=[_assignment(**fields)]))


def _chosen (**select: typing.Any) -> _Case:

	"""A map whose one assignment chooses its sound this way."""

	return _Case(_map(assignments=[_assignment(select=select)]), references=("GM36_BassDrum1",))


def _measured (key: str, value: typing.Any, **fields: typing.Any) -> _Case:

	"""A map that keeps a sample by one of its measurements."""

	return _Case(_map(assignments=[_assignment(select={"where": {key: value}}, **fields)]))


def _in_a_template (key: str, value: typing.Any) -> _Case:

	"""A map whose template carries the term, and whose assignment leaves it to the template."""

	fields = {name: field for name, field in _assignment().items() if name != key}

	return _defined(_map(
		templates={"kit": {key: value}},
		assignments=[{**fields, "template": "kit"}],
	))


def _stepped (step: dict[str, typing.Any]) -> _Case:

	"""A map whose one assignment passes its sound through one processor."""

	return _defined(_map(assignments=[_assignment(process=[step])]))


_QUANTISED: typing.Final[dict[str, typing.Any]] = {"stretch_quantize": {"grid": 16}}
"""The step a beat pattern and a quantised length are measured against."""


_EXAMPLE_PROBES: dict[str, _Probe] = {
	"properties/definitions":
		lambda value: _Case(_map(definitions=value), files={"project.yaml": _DEFINITIONS}),
	"properties/channel":
		lambda value: _Case(_map(
			channel=value,
			assignments=[{k: v for k, v in _assignment().items() if k != "channel"}],
		)),
	"properties/programs":
		lambda value: _Case(_map(programs=value)),
	"properties/program_channel":
		lambda value: _Case(_map(program_channel=value)),
	"properties/default_program":
		lambda value: _defined(_map(
			programs=[{"name": "Brushes", "program": 1, "directory": "kits/brushes"}],
			default_program=value,
		)),
	"properties/templates":
		lambda value: _Case(_map(
			templates=value,
			assignments=[_assignment(template=next(iter(value)))],
		)),
	"properties/assignments":
		lambda value: _Case(_map(assignments=value), references=("GM36_BassDrum1",)),
	"properties/maps":
		lambda value: _Case(
			{"maps": value},
			files={
				"drums.yaml": _map(),
				"bass.yaml":  {"assignments": [
					{k: v for k, v in _assignment(notes=48).items() if k != "channel"},
				]},
			},
			ensemble=True,
		),

	"$defs/assignment/properties/name":     lambda value: _played(name=value),
	"$defs/assignment/properties/channel":  lambda value: _played(channel=value),
	"$defs/assignment/properties/process":  lambda value: _played(process=value),
	"$defs/assignment/properties/gain":     lambda value: _played(gain=value),
	"$defs/assignment/properties/template":
		lambda value: _defined(_map(
			templates={"kit": {}, "room": {}},
			assignments=[_assignment(template=value)],
		)),

	"$defs/template/properties/name":     lambda value: _in_a_template("name", value),
	"$defs/template/properties/channel":  lambda value: _in_a_template("channel", value),
	"$defs/template/properties/process":  lambda value: _in_a_template("process", value),
	"$defs/template/properties/gain":     lambda value: _in_a_template("gain", value),

	"$defs/program/properties/name":
		lambda value: _Case(_map(programs=[{"name": value, "directory": "kits/acoustic"}])),
	"$defs/program/properties/program":
		lambda value: _Case(_map(programs=[{"name": "Brushes", "program": value, "directory": "kits/brushes"}])),
	"$defs/program/properties/directory":
		lambda value: _Case(_map(programs=[{"name": "Acoustic kit", "directory": value}])),
	"$defs/program/properties/map":
		lambda value: _Case(
			_map(programs=[{"name": "808 kit", "map": value}]),
			files={value: _map()},
		),

	"$defs/included_map/anyOf/1/properties/map":
		lambda value: _Case(
			{"maps": [{"map": value}]},
			files={value: _map()},
			ensemble=True,
		),
	"$defs/included_map/anyOf/1/properties/channel":
		lambda value: _Case(
			{"maps": [{"map": "drums.yaml", "channel": value}]},
			files={"drums.yaml": {"assignments": [
				{k: v for k, v in _assignment().items() if k != "channel"},
			]}},
			ensemble=True,
		),

	"$defs/notes":
		lambda value: _played(notes=value, process=[{"repitch": True}]),
	"$defs/notes/anyOf/2/anyOf/1/properties/range":
		lambda value: _played(
			notes={"mode": "zone-tuned", "range": value}, process=[{"repitch": True}],
		),

	"$defs/velocity":
		lambda value: _played(velocity=value),
	"$defs/velocity/anyOf/1/properties/trigger":
		lambda value: _played(velocity={"trigger": value}),
	"$defs/velocity/anyOf/1/properties/rescale":
		lambda value: _played(velocity={"trigger": [0, 63], "rescale": value}),

	"$defs/silenced_by":
		lambda value: _Case(_map(assignments=[
			_assignment(silenced_by=value),
			_assignment(notes=["drum.hi_hat_closed", "drum.hi_hat_pedal"]),
		])),

	"$defs/select":
		lambda value: _Case(_map(assignments=[_assignment(select=value)]), references=("GM36_BassDrum1",)),
	"$defs/where":
		lambda value: _chosen(where=value),
	"$defs/where/properties/name":
		lambda value: _chosen(where={"name": value}),
	"$defs/where/properties/name/anyOf/2/properties/matches":
		lambda value: _chosen(where={"name": {"matches": value}}),
	"$defs/where/properties/name/anyOf/2/properties/regex":
		lambda value: _chosen(where={"name": {"regex": value}}),
	"$defs/where/properties/path":
		lambda value: _chosen(where={"path": value}),
	"$defs/where/properties/directory":
		lambda value: _chosen(where={"directory": value}),
	"$defs/where/properties/reference":
		lambda value: _chosen(where={"reference": value}),
	"$defs/where/properties/duration":
		lambda value: _measured("duration", value),
	"$defs/where/properties/duration_beats":
		lambda value: _measured("duration_beats", value),
	"$defs/where/properties/onsets":
		lambda value: _measured("onsets", value),
	"$defs/where/properties/tempo":
		lambda value: _measured("tempo", value),
	"$defs/where/properties/pitch":
		lambda value: _measured("pitch", value),
	"$defs/where/properties/quantized_beats":
		lambda value: _measured("quantized_beats", value, process=[_QUANTISED]),

	"$defs/order":
		lambda value: _chosen(where={"pitched": True}, order=value),
	"$defs/order_clause/anyOf/1/properties/pattern":
		lambda value: _Case(_map(assignments=[_assignment(
			select={"where": {"pitched": True}, "order": [{"by": "beat_match", "pattern": value}]},
			process=[_QUANTISED],
		)])),

	"$defs/pick":
		lambda value: _chosen(where={"pitched": True}, order={"by": "level"}, pick=value),
	"$defs/pick/anyOf/4/properties/variation":
		lambda value: _chosen(
			where={"pitched": True},
			order={"by": "level"},
			pick={"mode": "velocity", "variation": value},
		),

	"$defs/cc_binding":
		lambda value: _stepped({"filter_low": {"freq": value}}),
	"$defs/cc_binding/properties/cc":
		lambda value: _stepped({"filter_low": {"freq": {"cc": value}}}),
	"$defs/cc_binding/properties/channel":
		lambda value: _stepped({"filter_low": {"freq": {"cc": 74, "channel": value}}}),
	"$defs/cc_binding/properties/min":
		lambda value: _stepped({"filter_low": {"freq": {"cc": 74, "min": value}}}),
	"$defs/cc_binding/properties/max":
		lambda value: _stepped({"filter_low": {"freq": {"cc": 74, "max": value}}}),
	"$defs/cc_binding/properties/default":
		lambda value: _stepped({"filter_low": {"freq": {"cc": 74, "default": value}}}),

	"$defs/release":
		lambda value: _played(mode="gated", release=value),
	"$defs/release_time":
		lambda value: _played(mode="gated", release={"time": value}),
	"$defs/release/anyOf/4/properties/cc":
		lambda value: _played(mode="gated", release={"cc": value}),
	"$defs/release/anyOf/4/properties/channel":
		lambda value: _played(mode="gated", release={"cc": 72, "channel": value}),
	"$defs/release/anyOf/4/properties/min":
		lambda value: _played(mode="gated", release={"cc": 72, "min": value}),
	"$defs/release/anyOf/4/properties/max":
		lambda value: _played(mode="gated", release={"cc": 72, "max": value}),
	"$defs/release/anyOf/4/properties/default":
		lambda value: _played(mode="gated", release={"cc": 72, "default": value}),

	"$defs/loop":
		lambda value: _played(loop=value),
	"$defs/loop/properties/start":
		lambda value: _played(loop={"start": value}),
	"$defs/loop/properties/end":
		lambda value: _played(loop={"end": value}),
	"$defs/loop/properties/crossfade":
		lambda value: _played(loop={"crossfade": value}),

	"$defs/extract":
		lambda value: _played(extract=value),
	"$defs/extract/anyOf/2/properties/blend":
		lambda value: _played(extract={"blend": value}),

	"$defs/pan":
		lambda value: _played(pan=value),
	"$defs/pan/anyOf/3/properties/gte":
		lambda value: _played(pan={"gte": value, "lte": 60}),
	"$defs/pan/anyOf/3/properties/lte":
		lambda value: _played(pan={"gte": -60, "lte": value}),
	"$defs/pan/anyOf/3/properties/position":
		lambda value: _played(pan={"position": value}),
	"$defs/pan/anyOf/3/properties/variation":
		lambda value: _played(pan={"position": 0, "variation": value}),

	"$defs/output":
		lambda value: _played(output=value, pan=[50, 50]),
}
"""A map that writes one published example where it belongs, by the path of the
term that publishes it.  The processors are generated below, because every one
of them is written into a `process:` list the same way."""


def _processor_probes () -> dict[str, _Probe]:

	"""Where each processor's own example, and each parameter's, is written."""

	probes: dict[str, _Probe] = {}
	entries = "$defs/process_step/anyOf/1/properties"

	for processor in subsample.processors.PROCESSORS.values():

		if processor.examples:
			probes[f"{entries}/{processor.name}"] = (
				lambda value, name=processor.name: _stepped({name: value})
			)

		for index, form in enumerate(_processor_entry(processor.name)["anyOf"]):
			if form.get("type") != "object":
				continue

			for parameter in processor.parameters:

				if not parameter.examples:
					continue

				probes[f"{entries}/{processor.name}/anyOf/{index}/properties/{parameter.name}"] = (
					lambda value, processor=processor, parameter=parameter:
						_stepped({processor.name: {
							parameter.name: value, **_beside(processor, parameter),
						}})
				)

	return probes


def _beside (
	processor: subsample.processors.Processor,
	parameter: subsample.processors.Parameter,
) -> dict[str, typing.Any]:

	"""The siblings a parameter needs to do anything: what the processor must be
	told, and what makes this parameter apply at all."""

	context = {
		other.name: other.choice_values[0]
		for other in processor.parameters
		if other.required and other.name != parameter.name
	}

	for condition in parameter.applies_when[:1]:
		for name, words in condition.items():
			context[name] = words[0]

	return context


def _examples () -> list[typing.Any]:

	"""One case per example the schema publishes, named by where it sits."""

	cases = []

	for path, node in _subschemas(_SCHEMA):
		for index, value in enumerate(node.get("examples", ())):
			cases.append(pytest.param(path.lstrip("/"), value, id=f"{path.lstrip('/')}-{index}"))

	return cases


def _load_case (tmp_path: pathlib.Path, case: _Case) -> None:

	"""Write the map and whatever it names beside it, and load it as Subsample does."""

	for name, content in case.files.items():
		beside = tmp_path / name
		beside.parent.mkdir(parents=True, exist_ok=True)
		beside.write_text(yaml.safe_dump(content), encoding="utf-8")

	path = tmp_path / "midi-map.yaml"
	path.write_text(yaml.safe_dump(case.mapping), encoding="utf-8")

	if case.ensemble:
		subsample.player.load_ensemble(path, list(case.references))
	else:
		subsample.player.load_midi_map(path, list(case.references))


def _a_sentence (text: str) -> bool:

	"""True when prose starts a sentence and ends one."""

	return bool(text) and (text[0].isupper() or text[0] in "`-0123456789") and text.endswith(".")


def _a_label (text: str) -> bool:

	"""True when a title reads as a label: a capital first, and no full stop."""

	return bool(text) and text[0].isupper() and not text.endswith(".")


def _outside (keyword: str, bound: typing.Any) -> typing.Any:

	"""A value the published limit refuses."""

	return {
		"minimum":          lambda: bound - 1,
		"maximum":          lambda: bound + 1,
		"exclusiveMinimum": lambda: bound,
		"exclusiveMaximum": lambda: bound,
	}[keyword]()


def _edge (keyword: str, bound: typing.Any) -> typing.Any:

	"""The last value the published limit still admits."""

	return {
		"minimum":          lambda: bound,
		"maximum":          lambda: bound,
		"exclusiveMinimum": lambda: bound + 1,
		"exclusiveMaximum": lambda: bound - 1,
	}[keyword]()


# ---------------------------------------------------------------------------
# The words a map may write
# ---------------------------------------------------------------------------

def _word_lists () -> list[typing.Any]:

	"""Each list of words the schema publishes, beside the parser's own list."""

	return [
		pytest.param("/$defs/assignment/properties/mode", subsample.query.VALID_MODES, id="mode"),
		pytest.param("/$defs/order_clause/anyOf/1/properties/by", subsample.query.valid_order_names(), id="order-by"),
		pytest.param("/$defs/order_clause/anyOf/0", tuple(subsample.query.LEGACY_ORDER_TOKENS), id="order-token"),
		pytest.param("/$defs/pick/anyOf/4/properties/curve", subsample.query.VALID_PICK_CURVES, id="pick-curve"),
		pytest.param("/$defs/pick/anyOf/4/properties/spacing", subsample.query.VALID_PICK_SPACINGS, id="pick-spacing"),
		pytest.param("/$defs/release/anyOf/3/properties/curve", subsample.query.VALID_RELEASE_CURVES, id="release-curve"),
		pytest.param("/$defs/extract/anyOf/0", subsample.query.EXTRACT_KINDS, id="extract-kind"),
	]


class TestWordsComeFromTheParser:

	@pytest.mark.parametrize(("path", "accepted"), _word_lists())
	def test_the_schema_publishes_the_parsers_own_words (
		self, path: str, accepted: tuple[str, ...],
	) -> None:

		"""A word list written out here a second time would drift; these come from the parser."""

		assert tuple(_words(path)) == tuple(accepted)

	@pytest.mark.parametrize("mode", subsample.query.VALID_MODES)
	def test_every_mode_loads (self, tmp_path: pathlib.Path, mode: str) -> None:

		"""Every playback mode the schema publishes is one a map may write."""

		assert _load(tmp_path, _map(assignments=[_assignment(mode=mode)])).note_map

	@pytest.mark.parametrize("curve", subsample.query.VALID_RELEASE_CURVES)
	def test_every_release_curve_loads (self, tmp_path: pathlib.Path, curve: str) -> None:

		"""Every fade shape the schema publishes is one a map may write."""

		result = _load(tmp_path, _map(assignments=[
			_assignment(mode="gated", release={"time": 100, "curve": curve}),
		]))

		assert _first(result).release is not None
		assert _first(result).release.curve == curve

	@pytest.mark.parametrize("curve", subsample.query.VALID_PICK_CURVES)
	def test_every_pick_curve_loads (self, tmp_path: pathlib.Path, curve: str) -> None:

		"""Every velocity curve the schema publishes is one a map may write."""

		result = _load(tmp_path, _map(assignments=[_assignment(select={
			"where": {"pitched": True},
			"order": "loudest",
			"pick":  {"mode": "velocity", "curve": curve},
		})]))

		assert _first(result).select[0].pick.curve == curve

	@pytest.mark.parametrize("spacing", subsample.query.VALID_PICK_SPACINGS)
	def test_every_pick_spacing_loads (self, tmp_path: pathlib.Path, spacing: str) -> None:

		"""Every velocity spacing the schema publishes is one a map may write."""

		result = _load(tmp_path, _map(assignments=[_assignment(select={
			"where": {"pitched": True},
			"order": "loudest",
			"pick":  {"mode": "velocity", "spacing": spacing},
		})]))

		assert _first(result).select[0].pick.spacing == spacing

	@pytest.mark.parametrize("kind", subsample.query.EXTRACT_KINDS)
	def test_every_extract_kind_loads (self, tmp_path: pathlib.Path, kind: str) -> None:

		"""Every part of a multi-channel recording the schema names is one a map may ask for."""

		result = _load(tmp_path, _map(assignments=[_assignment(extract=kind)]))

		assert _first(result).extract is not None
		assert _first(result).extract.kind == kind

	@pytest.mark.parametrize("name", subsample.query.valid_order_names())
	def test_every_order_name_loads (self, tmp_path: pathlib.Path, name: str) -> None:

		"""Every ranking the schema names is one a map may order by."""

		clause: dict[str, typing.Any] = {"by": name}

		if name == "beat_match":
			clause["pattern"] = [1, 0]

		where = {"reference": "BD0025"} if name == "similarity" else {"pitched": True}

		result = _load(
			tmp_path,
			_map(assignments=[_assignment(select={"where": where, "order": [clause]})]),
			references=["BD0025"],
		)

		assert _first(result).select[0].order[0].by == name

	@pytest.mark.parametrize("token", tuple(subsample.query.LEGACY_ORDER_TOKENS))
	def test_every_older_order_word_loads (self, tmp_path: pathlib.Path, token: str) -> None:

		"""Every older ranking word the schema still publishes is one a map may write."""

		where = {"reference": "BD0025"} if token == "similarity" else {"pitched": True}

		result = _load(
			tmp_path,
			_map(assignments=[_assignment(select={"where": where, "order": token})]),
			references=["BD0025"],
		)

		assert _first(result).select[0].order == (subsample.query.LEGACY_ORDER_TOKENS[token],)


# ---------------------------------------------------------------------------
# The limits and defaults a map is held to
# ---------------------------------------------------------------------------

class TestLimitsAreEnforced:

	def test_every_published_limit_has_a_probe (self) -> None:

		"""A limit nobody proves is one the schema may be publishing wrongly."""

		published = {case.values[0] for case in _limits()}

		assert published == set(_LIMIT_PROBES)

	@pytest.mark.parametrize(("path", "keyword", "bound"), _limits())
	def test_a_value_outside_a_published_limit_is_refused (
		self, tmp_path: pathlib.Path, path: str, keyword: str, bound: typing.Any,
	) -> None:

		"""A limit the schema publishes is one the map loader enforces."""

		with pytest.raises(ValueError):
			_load(tmp_path, _LIMIT_PROBES[path](_outside(keyword, bound)))

	@pytest.mark.parametrize(("path", "keyword", "bound"), _limits())
	def test_the_edge_of_a_published_limit_loads (
		self, tmp_path: pathlib.Path, path: str, keyword: str, bound: typing.Any,
	) -> None:

		"""A limit the schema publishes is not one the loader draws more tightly."""

		assert _load(tmp_path, _LIMIT_PROBES[path](_edge(keyword, bound))) is not None


class TestDefaultsAreWhatLoads:

	def test_every_published_default_has_a_probe (self) -> None:

		"""A default nobody proves is one the schema may be publishing wrongly."""

		published = {case.values[0] for case in _defaults()}

		assert published == set(_DEFAULT_PROBES)

	@pytest.mark.parametrize(("path", "declared"), _defaults())
	def test_leaving_a_term_out_gives_its_published_default (
		self, tmp_path: pathlib.Path, path: str, declared: typing.Any,
	) -> None:

		"""What the schema says a map gets for free is what a map without it gets."""

		mapping, reading = _DEFAULT_PROBES[path]

		assert reading(_load(tmp_path, mapping)) == declared


class TestUndeclaredKeysAreRefused:

	def test_a_map_key_the_schema_does_not_declare_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""The schema closes the map to keys it does not name, as the loader does."""

		assert _SCHEMA["additionalProperties"] is False

		with pytest.raises(ValueError):
			_load(tmp_path, _map(tempo=120))

	def test_an_assignment_key_the_schema_does_not_declare_is_refused (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""An assignment is closed the same way: a misspelt key is refused, not ignored."""

		assert _at("/$defs/assignment")["additionalProperties"] is False

		with pytest.raises(ValueError):
			_load(tmp_path, _map(assignments=[_assignment(realease=100)]))


# ---------------------------------------------------------------------------
# The processors, which are generated rather than written out here
# ---------------------------------------------------------------------------

def _processor_entry (name: str) -> dict[str, typing.Any]:

	"""What a map may write against one processor's name."""

	return typing.cast(
		dict[str, typing.Any],
		_at("/$defs/process_step/anyOf/1")["properties"][name],
	)


def _parameter_entry (processor: str, parameter: str) -> dict[str, typing.Any]:

	"""What a map may write against one processor parameter."""

	for form in _processor_entry(processor)["anyOf"]:
		if form.get("type") == "object":
			return typing.cast(dict[str, typing.Any], form["properties"][parameter])

	raise AssertionError(f"{processor} takes no parameters")


def _declared_parameters () -> list[typing.Any]:

	"""Every processor parameter Subsample declares."""

	return [
		pytest.param(processor.name, parameter.name, id=f"{processor.name}.{parameter.name}")
		for processor in subsample.processors.PROCESSORS.values()
		for parameter in processor.parameters
	]


class TestProcessorsFollowTheirDeclaration:

	def test_every_processor_is_a_word_and_a_key (self) -> None:

		"""A processor may be named on its own or given what it takes, so it appears as both."""

		words = [option["const"] for option in _at("/$defs/process_step/anyOf/0")["oneOf"]]
		keys  = list(_at("/$defs/process_step/anyOf/1")["properties"])

		assert words == keys
		assert set(subsample.processors.PROCESSORS) <= set(words)

	def test_an_older_processor_name_is_marked_as_one (self) -> None:

		"""A name Subsample still accepts but no longer documents says so."""

		older = {
			legacy.name
			for processor in subsample.processors.PROCESSORS.values()
			for legacy in processor.legacy_names
		}

		assert older
		assert all(_processor_entry(name).get("deprecated") is True for name in older)
		assert all(
			_processor_entry(name).get("deprecated") is None
			for name in subsample.processors.PROCESSORS
		)

	@pytest.mark.parametrize(("processor", "parameter"), _declared_parameters())
	def test_each_parameter_carries_its_declaration (self, processor: str, parameter: str) -> None:

		"""Every fact the declaration holds about a parameter reaches the schema unchanged."""

		declared = subsample.processors.PROCESSORS[processor].parameter(parameter)
		entry    = _parameter_entry(processor, parameter)

		assert entry.get("x-unit") == declared.unit
		assert entry.get("default") == declared.default
		assert entry.get("x-automatic") == declared.automatic
		assert entry.get("x-sweep") == (list(declared.sweep) if declared.sweep else None)
		assert entry.get("x-taper") == (declared.taper if declared.taper != "linear" else None)
		assert len(entry.get("x-applies-when", ())) == len(declared.applies_when)
		assert len(entry.get("x-limits-when", ())) == len(declared.limits_when)

	@pytest.mark.parametrize(("processor", "parameter"), _declared_parameters())
	def test_each_parameter_offers_the_forms_it_takes (self, processor: str, parameter: str) -> None:

		"""A parameter accepts a knob exactly when Subsample lets a knob drive it."""

		declared = subsample.processors.PROCESSORS[processor].parameter(parameter)
		forms    = _parameter_entry(processor, parameter)["anyOf"]

		bound = {"$ref": "#/$defs/cc_binding"} in forms

		assert bound == declared.bindable

		for form in forms:
			if form.get("type") in ("number", "integer"):
				for keyword, bound_value in (
					("minimum", declared.limit.minimum),
					("maximum", declared.limit.maximum),
					("exclusiveMinimum", declared.limit.exclusive_minimum),
					("exclusiveMaximum", declared.limit.exclusive_maximum),
				):
					assert form.get(keyword) == bound_value

			if "oneOf" in form:
				assert [option["const"] for option in form["oneOf"]] == list(declared.choice_values)

	def test_a_processor_that_must_be_told_something_says_so (self) -> None:

		"""A parameter with no default and no automatic value is required by name."""

		for processor in subsample.processors.PROCESSORS.values():
			wanted = [parameter.name for parameter in processor.parameters if parameter.required]

			for form in _processor_entry(processor.name)["anyOf"]:
				if form.get("type") == "object":
					assert form.get("required", []) == wanted

	def test_a_chain_may_align_to_the_beat_only_once (self) -> None:

		"""The one rule about a chain as a whole, published where a reader can act on it."""

		process = _at("/$defs/assignment/properties/process")

		assert process["maxContains"] == 1
		assert process["contains"]["anyOf"][0]["enum"] == list(subsample.query.BEAT_ALIGNING_PROCESSORS)


# ---------------------------------------------------------------------------
# The examples a reader copies
# ---------------------------------------------------------------------------

_PROBES: typing.Final[dict[str, _Probe]] = {**_EXAMPLE_PROBES, **_processor_probes()}


def _older_spellings () -> tuple[str, ...]:

	"""The path of every term the schema marks as a name it no longer documents."""

	return tuple(path for path, node in _subschemas(_SCHEMA) if node.get("deprecated"))


def _parts_a_key_points_at () -> set[str]:

	"""The shared parts a key names with nothing of its own, so the part carries that key's prose."""

	named = set()

	for _path, node in _subschemas(_SCHEMA):
		for child in node.get("properties", {}).values():
			if set(child) == {"$ref"}:
				named.add(child["$ref"].rsplit("/", 1)[-1])

	return named


def _values_are_listed (node: dict[str, typing.Any]) -> bool:

	"""True when the schema names every value a term may take, so an example would only repeat one."""

	if "const" in node or node.get("type") in ("boolean", "null"):
		return True

	if "oneOf" in node and all("const" in option for option in node["oneOf"]):
		return True

	# A processor that takes nothing is written as its name and no more.
	if node.get("type") == "object" and not node.get("properties"):
		return True

	if "anyOf" in node:
		return all(_values_are_listed(option) for option in node["anyOf"])

	return False


def _is_a_bound (path: str) -> bool:

	"""True when a term is one operator of a bounds block, which its own key shows in place."""

	operator = path.rsplit("/", 1)[-1]

	if path.startswith("/$defs/where/properties/") and "/anyOf/1/properties/" in path:
		return operator in subsample.query.VALID_OPERATORS

	if path.startswith("/$defs/pick/anyOf/3/properties/"):
		return operator in subsample.query.VALID_PICK_OPERATORS

	return False


def _owed_an_example (path: str, node: dict[str, typing.Any], parts: set[str]) -> bool:

	"""True when a term is one a reader would copy, so the schema owes it an example."""

	if not path or "description" not in node or "const" in node:
		return False

	if any(path.startswith(older) for older in _older_spellings()):
		return False

	# A shared part stands in for a key only where the key points at it with
	# nothing of its own; one reached from a list or a choice is shown whole by
	# the key that holds it.
	if path.count("/") == 2 and path.startswith("/$defs/"):
		return path.rsplit("/", 1)[-1] in parts

	return not (_is_a_bound(path) or _values_are_listed(node))


class TestExamplesAreWhatAMapWrites:

	"""The examples subsystem.co publishes, each proved against the parser."""

	def test_every_term_a_reader_would_copy_shows_what_to_write (self) -> None:

		"""A term with no example leaves a reader to guess, and nothing would catch a wrong guess."""

		parts = _parts_a_key_points_at()

		unshown = [
			path for path, node in _subschemas(_SCHEMA)
			if _owed_an_example(path, node, parts) and not node.get("examples")
		]

		assert not unshown

	def test_an_older_spelling_shows_nothing_to_copy (self) -> None:

		"""An example is there to be copied, and nothing should copy a spelling on its way out."""

		for older in _older_spellings():
			for path, node in _subschemas(_at(older), older):
				assert "examples" not in node, path

	def test_a_knob_is_shown_where_the_binding_is_declared_and_nowhere_else (self) -> None:

		"""Simon's decision, 2026-09-20: one entry shows the form, and the 56 parameters that take
		a knob each show only their own number."""

		assert _at("/$defs/cc_binding")["examples"]

		bound = [
			path for path, node in _subschemas(_SCHEMA)
			for value in node.get("examples", ())
			if isinstance(value, dict) and "cc" in value
		]

		# Release is the one term with a shape of its own here: the knob with
		# the fade's curve beside it, which is written nowhere else.
		assert bound == ["/$defs/cc_binding", "/$defs/cc_binding", "/$defs/release"]

	def test_every_example_has_somewhere_it_is_written (self) -> None:

		"""An example nobody loads is one the schema may be publishing wrongly."""

		published = {case.values[0] for case in _examples()}

		assert published == set(_PROBES)

	@pytest.mark.parametrize(("path", "value"), _examples())
	def test_every_example_loads_in_a_real_map (
		self, tmp_path: pathlib.Path, path: str, value: typing.Any,
	) -> None:

		"""What the reference offers to be copied is what Subsample accepts."""

		_load_case(tmp_path, _PROBES[path](value))

	@pytest.mark.parametrize(("path", "value"), _examples())
	def test_no_example_loads_with_a_complaint (
		self,
		tmp_path: pathlib.Path,
		caplog:   pytest.LogCaptureFixture,
		path:     str,
		value:    typing.Any,
	) -> None:

		"""A map that loads and warns has written something that does nothing, which no example should teach."""

		caplog.set_level(logging.WARNING)

		_load_case(tmp_path, _PROBES[path](value))

		assert [record.getMessage() for record in caplog.records] == []


# ---------------------------------------------------------------------------
# Fit to publish
# ---------------------------------------------------------------------------

class TestSchemaIsPublishable:

	def test_schema_is_json (self) -> None:

		"""The schema survives a JSON round trip, which is how a documentation build reads it."""

		assert json.loads(json.dumps(_SCHEMA)) == _SCHEMA

	def test_every_reference_resolves_and_every_part_is_used (self) -> None:

		"""A dangling reference would leave a reader with nothing to render."""

		referenced = {
			node["$ref"] for _path, node in _subschemas(_SCHEMA) if "$ref" in node
		}
		declared = {f"#/$defs/{name}" for name in _SCHEMA["$defs"]}

		assert referenced - declared == set()
		assert declared - referenced == set()

	def test_every_term_has_somewhere_to_say_what_it_is (self) -> None:

		"""Each term carries a description, or takes the one its shared part carries."""

		missing = [
			f"{path}/{name}"
			for path, node in _subschemas(_SCHEMA)
			for name, child in node.get("properties", {}).items()
			if "description" not in child and "$ref" not in child
			and "/if/" not in f"{path}/" and "/then/" not in f"{path}/"
		]

		assert not missing

	def test_every_description_is_written_in_whole_sentences (self) -> None:

		"""A reference entry is read alone at its anchor, so its prose starts and ends a sentence."""

		unwritten = [
			path for path, node in _subschemas(_SCHEMA)
			if "description" in node and not _a_sentence(node["description"])
		]

		assert not unwritten

	def test_every_word_has_a_label_and_a_description (self) -> None:

		"""A word is offered as a choice, so it needs a label to show and prose to explain it."""

		unwritten = [
			path for path, node in _subschemas(_SCHEMA)
			if "const" in node and "title" in node
			and not (_a_label(node["title"]) and _a_sentence(node["description"]))
		]

		assert not unwritten

	def test_every_title_is_a_label (self) -> None:

		"""A title starts with a capital and is not a sentence."""

		unlabelled = [
			path for path, node in _subschemas(_SCHEMA)
			if "title" in node and not _a_label(node["title"])
		]

		assert not unlabelled

	def test_no_prose_contains_an_em_dash (self) -> None:

		"""The documentation site refuses to publish an em dash."""

		dashed = [
			path for path, node in _subschemas(_SCHEMA)
			if "—" in node.get("description", "") or "—" in node.get("title", "")
		]

		assert not dashed


class TestPublishedRulesHold:

	"""The rules the schema states about a map as a whole, rather than about one value."""

	def test_a_ranking_may_be_written_under_one_key_or_the_other (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""The schema refuses both spellings at once, as the loader does."""

		assert _at("/$defs/select_spec")["not"] == {"required": ["order", "order_by"]}

		with pytest.raises(ValueError):
			_load(tmp_path, _map(assignments=[_assignment(select={
				"where": {"pitched": True}, "order": "loudest", "order_by": "loudest",
			})]))

	def test_a_chain_is_refused_a_second_beat_aligning_step (self, tmp_path: pathlib.Path) -> None:

		"""Two steps that both align to the beat would fight, and the loader refuses them."""

		with pytest.raises(ValueError):
			_load(tmp_path, _map(assignments=[_assignment(
				process=[{"stretch_quantize": {"grid": 16}}, {"pad_quantize": {"grid": 16}}],
			)]))

	def test_a_program_names_a_directory_or_a_map_but_not_both (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""The schema publishes the choice as a choice, and the loader enforces it."""

		assert _at("/$defs/program")["oneOf"] == [
			{"required": ["directory"]}, {"required": ["map"]},
		]

		with pytest.raises(ValueError):
			_load(tmp_path, _map(programs=[{"name": "Kit", "directory": "sounds", "map": "other.yaml"}]))

		with pytest.raises(ValueError):
			_load(tmp_path, _map(programs=[{"name": "Kit"}]))

	def test_a_processor_that_must_be_told_something_is_refused_without_it (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""What the schema marks as required is what the loader insists on."""

		with pytest.raises(ValueError):
			_load(tmp_path, _map(assignments=[_assignment(process=[{"vocoder": {}}])]))

	def test_an_assignment_says_which_notes_it_answers_to_and_what_it_plays (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""notes and select are the two an assignment cannot leave out."""

		assert _at("/$defs/assignment")["required"] == ["notes", "select"]

		for missing in ("notes", "select"):
			fields = {key: value for key, value in _assignment().items() if key != missing}

			with pytest.raises(ValueError):
				_load(tmp_path, _map(assignments=[fields]))


class TestEachNoteOfAListPlaysTheNextRank:

	"""What a pick left out means, which depends on the notes, so the schema says it in prose."""

	def test_each_note_of_a_list_plays_the_next_rank (self, tmp_path: pathlib.Path) -> None:

		"""Without a pick, the first note plays the best match and each after it the next."""

		result = _load(tmp_path, _map(assignments=[_assignment(notes=[60, 61, 62])]))

		ranks = [result.note_map[(0, note)][0][1] for note in (60, 61, 62)]

		assert ranks == [subsample.query.PickSpec(rank, rank) for rank in (1, 2, 3)]

	def test_a_repitched_list_plays_the_best_match_on_every_note (self, tmp_path: pathlib.Path) -> None:

		"""Repitching plays one sound across the notes, so every note takes the best match."""

		result = _load(tmp_path, _map(assignments=[
			_assignment(notes=[60, 61, 62], process=[{"repitch": True}]),
		]))

		assert {result.note_map[(0, note)][0][1] for note in (60, 61, 62)} == {subsample.query.PickSpec(1, 1)}

	def test_a_pick_the_map_writes_applies_to_every_note (self, tmp_path: pathlib.Path) -> None:

		"""A written pick is not shared out: every note draws the same way."""

		result = _load(tmp_path, _map(assignments=[
			_assignment(notes=[60, 61], select={"where": {"pitched": True}, "pick": "any"}),
		]))

		assert {result.note_map[(0, note)][0][1] for note in (60, 61)} == {subsample.query.PickSpec(None, None)}

	def test_the_schema_publishes_no_single_default_for_pick (self) -> None:

		"""A default of the best match would be wrong for every note of a list but the first."""

		assert "default" not in _at("/$defs/pick")


class TestAnEmptyListIsRefusedWhereItCanOnlyBeAMistake:

	"""#2693 decision 16: an empty list means nothing, and only a choke may say so."""

	def test_a_ranking_written_as_an_empty_list_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""`order: []` used to choose newest first, an order the map never wrote."""

		with pytest.raises(ValueError, match="empty list"):
			_load(tmp_path, _map(assignments=[_assignment(
				select={"where": {"pitched": True}, "order": []},
			)]))

	def test_an_empty_list_of_templates_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""`template: []` inherits nothing, which is what leaving the key out says."""

		with pytest.raises(ValueError, match="empty list"):
			_load(tmp_path, _map(
				templates={"base": {}},
				assignments=[_assignment(template=[])],
			))

	def test_a_choke_may_still_be_blanked (self, tmp_path: pathlib.Path) -> None:

		"""`silenced_by: []` blanks a template's default, which is what it is for."""

		result = _load(tmp_path, _map(
			templates={"choked": {"silenced_by": 62}},
			assignments=[_assignment(template="choked", silenced_by=[])],
		))

		assert _first(result).silenced_by is None


def _in_another_case () -> list[typing.Any]:

	"""A map that writes one published word in another letter case."""

	velocity_pick = {"where": {"pitched": True}, "order": "loudest"}

	cases: dict[str, dict[str, typing.Any]] = {
		"mode":           _map(assignments=[_assignment(mode="ONE_SHOT")]),
		"order-dir":      _map(assignments=[_assignment(
			select={"where": {"pitched": True}, "order": [{"by": "level", "dir": "DESC"}]},
		)]),
		"pick-any":       _map(assignments=[_assignment(
			select={"where": {"pitched": True}, "pick": "ANY"},
		)]),
		"pick-velocity":  _map(assignments=[_assignment(select={**velocity_pick, "pick": "VELOCITY"})]),
		"pick-mode":      _map(assignments=[_assignment(
			select={**velocity_pick, "pick": {"mode": "VELOCITY"}},
		)]),
		"pick-curve":     _map(assignments=[_assignment(
			select={**velocity_pick, "pick": {"mode": "velocity", "curve": "LINEAR"}},
		)]),
		"pick-spacing":   _map(assignments=[_assignment(
			select={**velocity_pick, "pick": {"mode": "velocity", "spacing": "RANK"}},
		)]),
		"release-curve":  _map(assignments=[_assignment(
			mode="gated", release={"time": 100, "curve": "COSINE"},
		)]),
		"extract-kind":   _map(assignments=[_assignment(extract="OMNI")]),
		"pan-any":        _map(assignments=[_assignment(pan="ANY")]),
		"silenced-self":  _map(assignments=[_assignment(silenced_by="Self")]),
	}

	return [pytest.param(mapping, id=name) for name, mapping in cases.items()]


class TestAWordIsWrittenExactly:

	"""#2693 decision 12: one spelling for every word in the map, as the processors have."""

	@pytest.mark.parametrize("mapping", _in_another_case())
	def test_a_word_in_another_letter_case_is_refused (
		self, tmp_path: pathlib.Path, mapping: dict[str, typing.Any],
	) -> None:

		"""A word the schema publishes in one spelling is the only spelling that loads."""

		with pytest.raises(ValueError):
			_load(tmp_path, mapping)


class TestValuesAreWrittenAsTheirType:

	"""#2693 decisions 11, 15 and 17: what a value is, checked where the map says it."""

	def test_a_processor_written_as_false_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""`{reverse: false}` used to add the step, so a map said one thing and played another."""

		with pytest.raises(ValueError, match="written as false"):
			_load(tmp_path, _map(assignments=[_assignment(process=[{"reverse": False}])]))

	def test_a_processor_written_as_true_still_runs (self, tmp_path: pathlib.Path) -> None:

		"""The way to ask for a step with its defaults is unchanged."""

		result = _load(tmp_path, _map(assignments=[_assignment(process=[{"reverse": True}])]))

		assert [step.name for step in _first(result).process.steps] == ["reverse"]

	def test_a_quoted_gain_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""A number in quotes is not a number, as a processor's parameter already has it."""

		with pytest.raises(ValueError, match="'gain' must be a number"):
			_load(tmp_path, _map(assignments=[_assignment(gain="3")]))

	def test_a_name_that_is_not_text_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""A bare number for a name reads as a note or a channel in every log line about it."""

		with pytest.raises(ValueError, match="'name' must be text"):
			_load(tmp_path, _map(assignments=[_assignment(name=808)]))

	def test_a_pattern_that_is_not_a_number_is_refused (self, tmp_path: pathlib.Path) -> None:

		"""Not-a-number passes a 0 to 1 test and then ranks the pool arbitrarily."""

		with pytest.raises(ValueError, match="finite number"):
			_load(tmp_path, _map(assignments=[_assignment(select={
				"where": {"pitched": True},
				"order": [{"by": "beat_match", "pattern": [float("nan"), 1]}],
			})]))


class TestUnitsComeFromTheClosedSet:

	def test_every_unit_the_map_declares_is_a_known_word (self) -> None:

		"""Superconductor may act on these words, so the map declares no other (#2435)."""

		declared = {
			node["x-unit"] for _path, node in _subschemas(_SCHEMA) if "x-unit" in node
		}

		assert declared <= set(subsample.processors.UNITS)

	def test_every_known_word_is_used_somewhere_in_the_map (self) -> None:

		"""A unit word nothing is measured in would be a word nobody can act on."""

		declared = {
			node["x-unit"] for _path, node in _subschemas(_SCHEMA) if "x-unit" in node
		}

		assert declared == set(subsample.processors.UNITS)
