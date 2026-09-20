"""Tests that every example Subsample publishes is one its own schemas accept.

The other schema tests load an example through the parser, which proves that
Subsample accepts the value.  These prove the other half: that the schema
declares the form the example is written in.  A reference generated from the
schema can then be trusted to describe the file the README teaches.

Every YAML block in README.md says what it is an example of, in the word after
the language on its opening fence:

	```yaml map.assignments
	- name: Kick
	  ...
	```

Readers never see the marker, because GitHub, PyPI and subsystem.co all take
only the first word as the language.  A block without one fails here, so no
example escapes the check.  The markers are:

	map, definitions, config    that kind of file, whole or in part
	assignment, select_spec …   a part of a map, by its name in the schema
	<either>.<key>.<key>        the value at that path, as `assignment.process`
	unchecked                   a block no test can check, marked as one

A block is a fragment of a file rather than a whole one, so what it must carry
is not checked; what it writes is.  A block that writes one key several times
over, to show the spellings it accepts, is checked as several examples, one per
spelling.
"""

import itertools
import json
import pathlib
import re
import typing

import jsonschema
import pytest
import yaml

import subsample.config_schema
import subsample.definitions_schema
import subsample.midi_map_schema


_README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

_SHIPPED = pathlib.Path(subsample.__file__).resolve().parent / "data"

_SCHEMAS: dict[str, dict[str, typing.Any]] = {
	"map":         subsample.midi_map_schema.json_schema(),
	"definitions": subsample.definitions_schema.json_schema(),
	"config":      subsample.config_schema.json_schema(),
}

_ENTRY = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):")


# ---------------------------------------------------------------------------
# Reading the README's examples
# ---------------------------------------------------------------------------

def _fenced () -> list[tuple[int, str, list[str]]]:

	"""Every YAML block in the README: where it starts, what it says it is, and its lines."""

	blocks: list[tuple[int, str, list[str]]] = []
	opened: typing.Optional[tuple[int, str]] = None
	body: list[str] = []

	for number, line in enumerate(_README.read_text(encoding="utf-8").split("\n"), start=1):

		if opened is None:
			if line.startswith("```yaml"):
				opened, body = (number, line[len("```yaml"):].strip()), []
			continue

		if line.startswith("```"):
			blocks.append((opened[0], opened[1], body))
			opened = None
			continue

		body.append(line)

	return blocks


def _examples () -> list[typing.Any]:

	"""One case per README block, named by the line it starts on."""

	return [
		pytest.param(number, marker, body, id=f"line-{number}")
		for number, marker, body in _fenced()
	]


def _siblings (lines: list[str]) -> list[tuple[str, list[str]]]:

	"""Each mapping key at these lines' outermost indent, with the lines it covers."""

	indents = [
		len(line) - len(line.lstrip())
		for line in lines if line.strip() and not line.lstrip().startswith("#")
	]

	if not indents:
		return []

	outer = min(indents)
	entries: list[tuple[str, list[str]]] = []

	for line in lines:
		stripped = line.strip()
		match = _ENTRY.match(stripped)
		starts = bool(match) and len(line) - len(line.lstrip()) == outer

		if starts and match is not None:
			entries.append((match.group(1), [line]))

		elif entries:
			entries[-1][1].append(line)

	return entries


def _spellings (lines: list[str]) -> list[list[str]]:

	"""Every example a block holds, one for each spelling of a key it writes more than once.

	A block that shows `notes:` six ways is six examples, not one: read as a
	single document only the last spelling would survive, and the other five
	would go out unchecked."""

	entries = _siblings(lines)

	if not entries:
		return [lines]

	written = [key for key, _body in entries]
	repeated = [key for key in dict.fromkeys(written) if written.count(key) > 1]

	if repeated:
		spellings: list[list[str]] = []

		for chosen, (key, _body) in enumerate(entries):
			if key != repeated[0]:
				continue

			kept = [
				entry for index, entry in enumerate(entries)
				if entry[0] != repeated[0] or index == chosen
			]
			spellings += _spellings([line for _key, body in kept for line in body])

		return spellings

	# No key is written twice here, so the only spellings left are inside one.
	inside = [_spellings(body[1:]) or [[]] for _key, body in entries]

	return [
		[line for (key, body), chosen in zip(entries, taken) for line in [body[0], *chosen]]
		for taken in itertools.product(*inside)
	]


# ---------------------------------------------------------------------------
# Finding the schema a marker names
# ---------------------------------------------------------------------------

def _resolved (node: dict[str, typing.Any]) -> dict[str, typing.Any]:

	"""A schema with its reference to a shared part followed."""

	reference = node.get("$ref")

	if isinstance(reference, str):
		return typing.cast(
			dict[str, typing.Any], _SCHEMAS["map"]["$defs"][reference.rsplit("/", 1)[-1]],
		)

	return node


def _inside (node: dict[str, typing.Any], key: str) -> typing.Optional[dict[str, typing.Any]]:

	"""The schema of one key written inside this one, through a shared part or a choice of forms."""

	node = _resolved(node)

	if key in node.get("properties", {}):
		return typing.cast(dict[str, typing.Any], node["properties"][key])

	for option in node.get("anyOf", []):
		found = _inside(option, key)

		if found is not None:
			return found

	return None


def _marked (marker: str) -> dict[str, typing.Any]:

	"""The schema a marker names: a kind of file, a part of a map, or a value inside one."""

	first, _dot, rest = marker.partition(".")

	if first in _SCHEMAS:
		node = _SCHEMAS[first]

	elif first in _SCHEMAS["map"]["$defs"]:
		node = _SCHEMAS["map"]["$defs"][first]

	else:
		raise AssertionError(f"{marker!r} names neither a kind of file nor a part of a map")

	for step in rest.split(".") if rest else []:
		found = _inside(node, step)

		assert found is not None, f"{marker!r}: nothing is written as {step!r} there"
		node = found

	return node


def _fragment (node: typing.Any) -> typing.Any:

	"""The same schema without what a file must carry, at any depth.

	A README block shows one part of a file: an assignment written to teach
	`channel:` says nothing about which sound it plays, and should not have to.

	A rule that reads what a file carries, rather than demanding it, is left
	alone, because dropping what it names turns it into something else: a
	choice of forms is how one form is told from another (a program names a
	directory or a map, never neither), a `not` would come to refuse
	everything, and a rule that holds in one case would come to hold in all."""

	untouched = ("anyOf", "oneOf", "allOf", "not", "if", "then", "else")

	if isinstance(node, dict):
		return {
			key: (value if key in untouched else _fragment(value))
			for key, value in node.items()
			if key != "required"
		}

	if isinstance(node, list):
		return [_fragment(item) for item in node]

	return node


def _document (node: dict[str, typing.Any], complete: bool) -> dict[str, typing.Any]:

	"""The schema as a document a validator can resolve on its own."""

	document = {**_resolved(node), "$defs": _SCHEMAS["map"]["$defs"]}

	return document if complete else typing.cast(dict[str, typing.Any], _fragment(document))


def _valid (
	instance: typing.Any, node: dict[str, typing.Any], complete: bool = True,
) -> list[str]:

	"""Everything a schema has against an example, in the words a validator gives."""

	validator = jsonschema.Draft202012Validator(_document(node, complete))

	return [
		f"{'.'.join(str(part) for part in error.absolute_path) or 'the document'}: {error.message}"
		for error in validator.iter_errors(instance)
	]


# ---------------------------------------------------------------------------
# The examples themselves
# ---------------------------------------------------------------------------

class TestEveryReadmeExampleSaysWhatItIs:

	def test_the_readme_holds_examples_to_check (self) -> None:

		"""A change that stopped finding the blocks would pass every test below in silence."""

		assert len(_fenced()) > 50

	@pytest.mark.parametrize(("number", "marker", "body"), _examples())
	def test_a_block_says_what_it_is_an_example_of (
		self, number: int, marker: str, body: list[str],
	) -> None:

		"""An unmarked block is one nothing checks, which is how a wrong example survives."""

		assert marker, f"README.md:{number}: the fence says nothing about what this is an example of"

		if marker != "unchecked":
			assert _marked(marker)


class TestEveryReadmeExampleIsOneTheSchemaAccepts:

	@pytest.mark.parametrize(("number", "marker", "body"), _examples())
	def test_a_block_is_written_as_the_schema_says (
		self, number: int, marker: str, body: list[str],
	) -> None:

		"""What the README teaches is what the reference declares."""

		if marker == "unchecked":
			return

		node = _marked(marker)

		for spelling in _spellings(body):
			text = "\n".join(spelling)
			instance = yaml.safe_load(text)

			if instance is None:
				continue

			problems = _valid(instance, node, complete=False)

			assert not problems, f"README.md:{number} ({marker}): {problems}\n{text}"


class TestEveryShippedFileIsOneTheSchemaAccepts:

	@pytest.mark.parametrize("name", ["midi-map.yaml.default", "midi-map-gm-drums.yaml"])
	def test_a_shipped_map_is_written_as_the_schema_says (self, name: str) -> None:

		"""Subsample's own maps are the largest examples it publishes."""

		mapping = yaml.safe_load((_SHIPPED / name).read_text(encoding="utf-8"))

		assert not _valid(mapping, _SCHEMAS["map"])

	def test_the_shipped_configuration_is_written_as_the_schema_says (self) -> None:

		"""The file a new installation starts from, against the schema its reference comes from."""

		configuration = yaml.safe_load((_SHIPPED / "config.yaml.default").read_text(encoding="utf-8"))

		assert not _valid(configuration, _SCHEMAS["config"])


def _published () -> list[typing.Any]:

	"""Every example the schemas publish, with the schema that publishes it."""

	cases = []

	for kind, schema in (("map", _SCHEMAS["map"]), ("definitions", _SCHEMAS["definitions"])):
		for path, node in _walk(schema):
			for index, value in enumerate(node.get("examples", ())):
				cases.append(pytest.param(node, value, id=f"{kind}{path}-{index}"))

	return cases


def _walk (
	node: dict[str, typing.Any], path: str = "",
) -> typing.Iterator[tuple[str, dict[str, typing.Any]]]:

	"""Every schema inside this one, with the path that reaches it."""

	yield path, node

	for keyword in ("items", "contains", "propertyNames", "additionalProperties"):
		child = node.get(keyword)

		if isinstance(child, dict):
			yield from _walk(child, f"{path}/{keyword}")

	for keyword in ("anyOf", "oneOf", "allOf"):
		for index, child in enumerate(node.get(keyword, ())):
			yield from _walk(child, f"{path}/{keyword}/{index}")

	for container in ("properties", "$defs"):
		for name, child in node.get(container, {}).items():
			yield from _walk(child, f"{path}/{container}/{name}")


class TestEveryPublishedExampleIsValidWhereItSits:

	def test_every_example_has_a_case (self) -> None:

		"""A change that stopped finding the examples would pass the test below in silence."""

		assert len(_published()) > 150

	@pytest.mark.parametrize(("node", "value"), _published())
	def test_an_example_is_written_as_its_own_term_declares (
		self, node: dict[str, typing.Any], value: typing.Any,
	) -> None:

		"""Loading an example proves Subsample takes it; this proves the schema declares its form."""

		assert not _valid(value, node)

	def test_the_schemas_are_valid_schemas (self) -> None:

		"""A schema a validator refuses is one no reader can generate a reference from."""

		for schema in _SCHEMAS.values():
			jsonschema.Draft202012Validator.check_schema(schema)

			assert json.loads(json.dumps(schema)) == schema
