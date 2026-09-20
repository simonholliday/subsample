"""Tests for subsample.definitions_schema — the declared file agrees with the loader.

subsample.definitions is the definitions file's real grammar.  The schema
declares it a second time so the documentation can be generated from it, and a
second list can drift from the first.  These tests are what stop it: they load
real definitions files through subsample.definitions, so a section, a range or
a name rule published here is one Subsample actually keeps.

When one fails, the fix is to change the schema to match the loader, unless the
loader is the half that is wrong.
"""

import json
import pathlib
import typing

import pytest
import yaml

import subsample.definitions
import subsample.definitions_schema


_SCHEMA = subsample.definitions_schema.json_schema()


def _load (tmp_path: pathlib.Path, file: dict[str, typing.Any]) -> subsample.definitions.Definitions:

	"""Mount one definitions file as a map does, and load it."""

	(tmp_path / "project.yaml").write_text(yaml.safe_dump(file), encoding="utf-8")

	return subsample.definitions.load_definitions({"my": "project.yaml"}, tmp_path)


def _section (name: str) -> dict[str, typing.Any]:

	"""What the schema publishes for one section."""

	return typing.cast(dict[str, typing.Any], _SCHEMA["properties"][name])


def _value (name: str) -> dict[str, typing.Any]:

	"""What the schema publishes for one entry of a section."""

	return typing.cast(dict[str, typing.Any], _section(name)["additionalProperties"])


def _a_sentence (text: str) -> bool:

	"""True when prose starts a sentence and ends one."""

	return bool(text) and (text[0].isupper() or text[0] == "`") and text.endswith(".")


class TestSectionsComeFromTheLoader:

	def test_the_schema_publishes_the_sections_subsample_reads (self) -> None:

		"""A section written out here a second time would drift; these come from the loader."""

		assert tuple(_SCHEMA["properties"]) == subsample.definitions.CONSUMED_SECTIONS

	def test_a_section_subsample_does_not_read_is_left_alone (self, tmp_path: pathlib.Path) -> None:

		"""The file belongs to the project, so another tool's section is not an error."""

		assert _SCHEMA["additionalProperties"] is True

		definitions = _load(tmp_path, {"notes": {"kick": 36}, "nrpn": {"filter_sweep": 1002}})

		assert definitions.lookup("notes", "my", "kick", "test") == 36

	@pytest.mark.parametrize("section", subsample.definitions.CONSUMED_SECTIONS)
	def test_every_section_loads (self, tmp_path: pathlib.Path, section: str) -> None:

		"""Every section the schema publishes is one a project may write."""

		low, _high = subsample.definitions.SECTION_RANGES[section]

		definitions = _load(tmp_path, {section: {"thing": low}})

		assert definitions.lookup(section, "my", "thing", "test") == low


class TestRangesAreEnforced:

	@pytest.mark.parametrize("section", subsample.definitions.CONSUMED_SECTIONS)
	def test_the_published_range_is_the_loaders_own (self, section: str) -> None:

		"""What the reference says a number may be is what the loader checks."""

		low, high = subsample.definitions.SECTION_RANGES[section]
		value = _value(section)

		assert (value["minimum"], value["maximum"]) == (low, high)

	@pytest.mark.parametrize("section", subsample.definitions.CONSUMED_SECTIONS)
	def test_a_number_outside_the_published_range_is_refused (
		self, tmp_path: pathlib.Path, section: str,
	) -> None:

		"""A range the schema publishes is one the loader enforces, at both ends."""

		value = _value(section)

		for outside in (value["minimum"] - 1, value["maximum"] + 1):
			with pytest.raises(ValueError, match="outside"):
				_load(tmp_path, {section: {"thing": outside}})

	@pytest.mark.parametrize("section", subsample.definitions.CONSUMED_SECTIONS)
	def test_both_edges_of_the_published_range_load (
		self, tmp_path: pathlib.Path, section: str,
	) -> None:

		"""A range the schema publishes is not one the loader draws more tightly."""

		value = _value(section)

		for edge in (value["minimum"], value["maximum"]):
			assert _load(tmp_path, {section: {"thing": edge}}).tables

	@pytest.mark.parametrize("written", [True, 1.5, "36", None])
	def test_a_value_that_is_not_a_whole_number_is_refused (
		self, tmp_path: pathlib.Path, written: typing.Any,
	) -> None:

		"""The schema says integer, and `true` would otherwise become 1 without a word."""

		assert _value("notes")["type"] == "integer"

		with pytest.raises(ValueError, match="whole number"):
			_load(tmp_path, {"notes": {"kick": written}})


class TestNamesFollowTheLoadersRule:

	def test_the_published_pattern_is_the_loaders_own (self) -> None:

		"""A name rule written out twice would drift; this one comes from the loader."""

		pattern = _section("notes")["propertyNames"]["pattern"]

		assert pattern == f"^{subsample.definitions.NAME_RE.pattern}$"

	@pytest.mark.parametrize("name", ["Kick", "kick.1", "1kick", "kick-1", "_kick"])
	def test_a_name_the_pattern_refuses_is_refused (
		self, tmp_path: pathlib.Path, name: str,
	) -> None:

		"""What the reference says a name may be is what the loader insists on."""

		with pytest.raises(ValueError, match="must match"):
			_load(tmp_path, {"notes": {name: 36}})

	def test_a_name_is_matched_whatever_case_a_map_writes (self, tmp_path: pathlib.Path) -> None:

		"""The prose says a map may write it in any case, so a map that does still resolves."""

		definitions = _load(tmp_path, {"notes": {"kick": 36}})

		assert definitions.lookup("notes", "MY", "Kick", "test") == 36


class TestExamplesAreWhatAFileHolds:

	def test_the_whole_file_example_loads (self, tmp_path: pathlib.Path) -> None:

		"""What the reference offers to be copied is what Subsample reads."""

		for example in _SCHEMA["examples"]:
			assert _load(tmp_path, example).tables["my"]

	@pytest.mark.parametrize("section", subsample.definitions.CONSUMED_SECTIONS)
	def test_every_section_example_loads (self, tmp_path: pathlib.Path, section: str) -> None:

		"""Each section shows what to write there, proved by loading it."""

		for example in _section(section)["examples"]:
			definitions = _load(tmp_path, {section: example})

			for name, number in example.items():
				assert definitions.lookup(section, "my", name, "test") == number


class TestSchemaIsPublishable:

	def test_schema_is_json (self) -> None:

		"""The schema survives a JSON round trip, which is how a documentation build reads it."""

		assert json.loads(json.dumps(_SCHEMA)) == _SCHEMA

	def test_every_term_says_what_it_is_and_shows_what_to_write (self) -> None:

		"""A reference entry is read alone at its anchor, so each carries prose and an example."""

		assert _a_sentence(_SCHEMA["description"])
		assert _SCHEMA["examples"]

		for section in subsample.definitions.CONSUMED_SECTIONS:
			assert _a_sentence(_section(section)["description"])
			assert _a_sentence(_value(section)["description"])
			assert _a_sentence(_section(section)["propertyNames"]["description"])
			assert _section(section)["examples"]

	def test_no_prose_contains_an_em_dash (self) -> None:

		"""The documentation site refuses to publish an em dash."""

		assert "—" not in json.dumps(_SCHEMA)
