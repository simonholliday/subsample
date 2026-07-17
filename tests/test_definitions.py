"""Tests for subsample.definitions — the per-project definitions file.

Covers the file loader (sections, validation, tolerance of foreign sections),
the mount parser (prefixes, paths), note_namespaces(), and resolve_scalar()
(symbolic vs fall-through behaviour and the targeted error messages).
"""

import pathlib
import typing

import pytest

import subsample.definitions


def _write (tmp_path: pathlib.Path, name: str, content: str) -> pathlib.Path:
	p = tmp_path / name
	p.write_text(content, encoding="utf-8")
	return p


def _load (
	tmp_path: pathlib.Path,
	content: str,
	prefix: str = "my",
	reserved: frozenset[str] = frozenset({"drum"}),
) -> subsample.definitions.Definitions:

	"""Write one file and mount it under `prefix`."""

	_write(tmp_path, "defs.yaml", content)
	return subsample.definitions.load_definitions(
		{prefix: "defs.yaml"}, tmp_path, reserved_prefixes=reserved,
	)


class TestLoadDefinitionsFile:

	def test_all_four_sections_load (self, tmp_path: pathlib.Path) -> None:
		defs = _load(tmp_path, """
notes:    { ride_edge_soft: 53, dawn_chorus_pheasant: 60 }
cc:       { sampler_release: 21 }
channels: { kit: 10, birds: 3 }
programs: { brushes: 1 }
""")
		assert defs.tables["my"]["notes"]["dawn_chorus_pheasant"] == 60
		assert defs.tables["my"]["cc"]["sampler_release"] == 21
		assert defs.tables["my"]["channels"]["birds"] == 3
		assert defs.tables["my"]["programs"]["brushes"] == 1

	def test_unknown_sections_ignored (self, tmp_path: pathlib.Path) -> None:

		"""Foreign sections belong to other tools (the sequencer) — no error,
		and they do not appear in the tables."""

		defs = _load(tmp_path, """
notes: { kick_alt: 35 }
nrpn:  { filter_env: 1042 }
mixer: { monitor_bus: 7 }
""")
		assert defs.tables["my"]["notes"]["kick_alt"] == 35
		assert "nrpn" not in defs.tables["my"]
		assert "mixer" not in defs.tables["my"]

	def test_empty_file_ok (self, tmp_path: pathlib.Path) -> None:
		defs = _load(tmp_path, "")
		assert defs.tables["my"] == {}

	def test_absent_and_null_sections_ok (self, tmp_path: pathlib.Path) -> None:
		defs = _load(tmp_path, """
notes:
cc: { a: 1 }
""")
		assert "notes" not in defs.tables["my"]   # null section = absent
		assert defs.tables["my"]["cc"]["a"] == 1

	def test_non_mapping_top_level_raises (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="top level must be a mapping"):
			_load(tmp_path, "- a\n- b\n")

	def test_consumed_section_not_mapping_raises (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="section 'notes' must be a mapping"):
			_load(tmp_path, "notes: [a, b]\n")

	def test_bool_value_raises (self, tmp_path: pathlib.Path) -> None:

		"""bool is an int subclass — it must fail loudly, not become 1."""

		with pytest.raises(ValueError, match="whole number"):
			_load(tmp_path, "notes: { x: true }\n")

	def test_non_int_value_raises (self, tmp_path: pathlib.Path) -> None:
		for bad in ("x: hat", "x: 1.5", "x:"):
			with pytest.raises(ValueError, match="whole number"):
				_load(tmp_path, f"notes: {{ {bad} }}\n")

	def test_note_and_cc_value_range (self, tmp_path: pathlib.Path) -> None:
		for section in ("notes", "cc"):
			with pytest.raises(ValueError, match=r"outside \[0, 127\]"):
				_load(tmp_path, f"{section}: {{ x: -1 }}\n")
			with pytest.raises(ValueError, match=r"outside \[0, 127\]"):
				_load(tmp_path, f"{section}: {{ x: 128 }}\n")
			ok = _load(tmp_path, f"{section}: {{ lo: 0, hi: 127 }}\n")
			assert ok.tables["my"][section] == {"lo": 0, "hi": 127}

	def test_channel_value_range (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match=r"outside \[1, 16\]"):
			_load(tmp_path, "channels: { x: 0 }\n")
		with pytest.raises(ValueError, match=r"outside \[1, 16\]"):
			_load(tmp_path, "channels: { x: 17 }\n")
		ok = _load(tmp_path, "channels: { lo: 1, hi: 16 }\n")
		assert ok.tables["my"]["channels"] == {"lo": 1, "hi": 16}

	def test_program_value_range (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match=r"outside \[0, 127\]"):
			_load(tmp_path, "programs: { x: 128 }\n")

	def test_bad_name_pattern_raises (self, tmp_path: pathlib.Path) -> None:
		for bad in ("Dawn_Chorus", "1st", "a.b", "a-b"):
			with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
				_load(tmp_path, f'notes: {{ "{bad}": 60 }}\n')

	def test_missing_file_raises (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="not found.*nope.yaml"):
			subsample.definitions.load_definitions(
				{"my": "nope.yaml"}, tmp_path,
			)

	def test_unparseable_yaml_raises_valueerror (self, tmp_path: pathlib.Path) -> None:

		"""yaml.YAMLError is wrapped — callers only ever handle ValueError."""

		with pytest.raises(ValueError, match="could not be read"):
			_load(tmp_path, "notes: {broken: [\n")


class TestLoadDefinitionsMount:

	def test_none_returns_empty (self, tmp_path: pathlib.Path) -> None:
		defs = subsample.definitions.load_definitions(None, tmp_path)
		assert defs.tables == {}
		assert defs.paths == ()

	def test_mount_not_mapping_raises (self, tmp_path: pathlib.Path) -> None:

		"""The bare-string form is deliberately rejected — one explicit shape,
		and the prefix is always the user's choice."""

		with pytest.raises(ValueError, match="mapping of prefix"):
			subsample.definitions.load_definitions("project.yaml", tmp_path)

	def test_prefix_bad_pattern_raises (self, tmp_path: pathlib.Path) -> None:
		_write(tmp_path, "defs.yaml", "notes: { a: 1 }\n")
		with pytest.raises(ValueError, match="prefix 'My'"):
			subsample.definitions.load_definitions({"My": "defs.yaml"}, tmp_path)

	def test_reserved_prefix_raises (self, tmp_path: pathlib.Path) -> None:
		_write(tmp_path, "defs.yaml", "notes: { a: 1 }\n")
		with pytest.raises(ValueError, match="reserved"):
			subsample.definitions.load_definitions(
				{"drum": "defs.yaml"}, tmp_path,
				reserved_prefixes=frozenset({"drum"}),
			)

	def test_relative_and_absolute_paths (self, tmp_path: pathlib.Path) -> None:
		sub = tmp_path / "shared"
		sub.mkdir()
		_write(sub, "a.yaml", "notes: { a: 1 }\n")
		_write(tmp_path, "b.yaml", "notes: { b: 2 }\n")

		defs = subsample.definitions.load_definitions(
			{"rel": "shared/a.yaml", "abs": str(tmp_path / "b.yaml")}, tmp_path,
		)
		assert defs.tables["rel"]["notes"]["a"] == 1
		assert defs.tables["abs"]["notes"]["b"] == 2

	def test_multiple_prefixes_and_paths_recorded (self, tmp_path: pathlib.Path) -> None:
		_write(tmp_path, "one.yaml", "notes: { a: 1 }\n")
		_write(tmp_path, "two.yaml", "cc: { b: 2 }\n")

		defs = subsample.definitions.load_definitions(
			{"one": "one.yaml", "two": "two.yaml"}, tmp_path,
		)
		assert set(defs.tables) == {"one", "two"}
		assert defs.paths == (
			(tmp_path / "one.yaml").resolve(),
			(tmp_path / "two.yaml").resolve(),
		)


class TestNoteNamespaces:

	def test_prefix_without_notes_section_present (self, tmp_path: pathlib.Path) -> None:

		"""A mounted prefix with no notes: section still claims its namespace,
		so `my.typo` gets the targeted unknown-symbol error instead of falling
		through to the note-name grammar."""

		defs = _load(tmp_path, "cc: { a: 1 }\n")
		spaces = defs.note_namespaces()
		assert spaces == {"my": {}}

	def test_notes_table_exposed (self, tmp_path: pathlib.Path) -> None:
		defs = _load(tmp_path, "notes: { kick_alt: 35 }\n")
		assert defs.note_namespaces() == {"my": {"kick_alt": 35}}


class TestResolveScalar:

	def _defs (self, tmp_path: pathlib.Path) -> subsample.definitions.Definitions:
		return _load(tmp_path, """
notes:    { ride_edge_soft: 53 }
cc:       { sampler_release: 21 }
channels: { kit: 10 }
programs: { brushes: 1 }
""")

	def test_int_and_numeric_string_fall_through (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		assert subsample.definitions.resolve_scalar(defs, "cc", 21, "t") == 21
		assert subsample.definitions.resolve_scalar(defs, "cc", "21", "t") == 21
		assert subsample.definitions.resolve_scalar(None, "cc", 21, "t") == 21

	def test_symbolic_resolves_each_section (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		assert subsample.definitions.resolve_scalar(defs, "cc", "my.sampler_release", "t") == 21
		assert subsample.definitions.resolve_scalar(defs, "channels", "my.kit", "t") == 10
		assert subsample.definitions.resolve_scalar(defs, "programs", "my.brushes", "t") == 1

	def test_use_site_case_insensitive (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		assert subsample.definitions.resolve_scalar(defs, "channels", "My.KIT", "t") == 10

	def test_unknown_prefix_lists_mounts (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		with pytest.raises(ValueError, match="mounted prefixes: my"):
			subsample.definitions.resolve_scalar(defs, "cc", "your.thing", "t")

	def test_no_mounts_message (self, tmp_path: pathlib.Path) -> None:
		for empty in (None, subsample.definitions.Definitions(tables={})):
			with pytest.raises(ValueError, match="mounts no 'definitions:'"):
				subsample.definitions.resolve_scalar(empty, "cc", "my.x", "t")

	def test_drum_at_scalar_site_hint (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		with pytest.raises(ValueError, match="NOTE names"):
			subsample.definitions.resolve_scalar(defs, "cc", "drum.kick", "t")

	def test_unknown_name_lists_valid (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		with pytest.raises(ValueError, match="valid: sampler_release"):
			subsample.definitions.resolve_scalar(defs, "cc", "my.nope", "t")

	def test_cross_section_hint (self, tmp_path: pathlib.Path) -> None:

		"""A notes name used at a channels site hints where the name lives."""

		defs = self._defs(tmp_path)
		with pytest.raises(ValueError, match="defined in section 'notes'"):
			subsample.definitions.resolve_scalar(defs, "channels", "my.ride_edge_soft", "t")

	def test_non_symbolic_dotted_string_keeps_int_error (self, tmp_path: pathlib.Path) -> None:

		"""'1.5' is not symbol-shaped — it falls through to int() and keeps
		the exact invalid-literal error it produced before this feature."""

		defs = self._defs(tmp_path)
		with pytest.raises(ValueError, match="invalid literal"):
			subsample.definitions.resolve_scalar(defs, "cc", "1.5", "t")

	def test_context_in_messages (self, tmp_path: pathlib.Path) -> None:
		defs = self._defs(tmp_path)
		with pytest.raises(ValueError, match="assignment 'Pheasant' 'channel'"):
			subsample.definitions.resolve_scalar(
				defs, "channels", "my.nope", "assignment 'Pheasant' 'channel'",
			)
