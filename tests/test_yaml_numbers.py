"""A number in a YAML file means what it looks like (#3048).

Subsample read its YAML with PyYAML, which implements YAML **1.1** - where a
leading zero means octal and a colon means sexagesimal.  A drum map lining its
note numbers up in a column read ``kick: 036`` as **30**, a different drum,
and nothing said so; a cue written ``1:30`` became **90**.

Decision 18 of subsequence#2991: YAML 1.2 number rules, in Subsample and
Subsequence alike, so the definitions file they share cannot name one drum
here and another there.  Subsequence's half is commit ``982975b``.

These cover the loader itself, the promise that nothing else about YAML moved,
and every file Subsample reads - because the loader being right is worth
nothing if a call site still reaches for ``yaml.safe_load``.
"""

import pathlib
import typing

import pytest
import yaml

import subsample.config
import subsample.definitions
import subsample.player
import subsample.yaml_numbers


_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "yaml_numbers.yaml"


def _write (tmp_path: pathlib.Path, name: str, content: str) -> pathlib.Path:

	"""Write one file and hand back its path."""

	path = tmp_path / name
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")

	return path


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

class TestANumberMeansWhatItLooksLike:

	@pytest.mark.parametrize(
		("written", "means"),
		[
			("036",    36),		# 1.1 read this as 30 - octal
			("042",    42),		# 1.1 read this as 34
			("012345", 12345),	# 1.1 read this as 5349
			("0o42",   34),		# 1.1 left this the string '0o42'
			("0x2A",   42),
			("38",     38),
			("-7",     -7),
			("0",      0),
			("-0x10",  -16),	# a deliberate superset of 1.2: see the module
		],
	)
	def test_an_integer_reads_as_written (self, written: str, means: int) -> None:

		"""The heart of it: a leading zero is how people line a column up."""

		loaded = subsample.yaml_numbers.load(f"value: {written}")

		assert loaded["value"] == means
		assert isinstance(loaded["value"], int), (
			f"{written} came back as {type(loaded['value']).__name__}"
		)

	def test_a_colon_stays_text (self) -> None:

		"""``1:30`` is a time somebody wrote, not 1*60 + 30."""

		assert subsample.yaml_numbers.load("cue: 1:30")["cue"] == "1:30"

	def test_a_real_float_is_still_a_float (self) -> None:

		"""The float resolver was rewritten too, and must not have eaten them."""

		loaded = subsample.yaml_numbers.load("a: 2.5\nb: 1e3\nc: .inf\nd: -0.25")

		assert loaded["a"] == pytest.approx(2.5)
		assert loaded["b"] == pytest.approx(1000.0)
		assert loaded["c"] == float("inf")
		assert loaded["d"] == pytest.approx(-0.25)

	def test_a_plain_integer_is_not_quietly_a_float (self) -> None:

		"""1.2's own float pattern matches a bare "38", which made it 38.0.

		Nothing downstream would have said so: the sections that validate a
		definitions file check range, not type, and a MIDI note number is not
		a float.
		"""

		loaded = subsample.yaml_numbers.load("snare: 38")

		assert isinstance(loaded["snare"], int)
		assert not isinstance(loaded["snare"], bool)

	def test_a_malformed_document_still_raises_yamlerror (self) -> None:

		"""Callers catch yaml.YAMLError, so the loader must keep raising it."""

		with pytest.raises(yaml.YAMLError):
			subsample.yaml_numbers.load("a: [1, 2\nb: 3")


# ---------------------------------------------------------------------------
# What deliberately did not change
# ---------------------------------------------------------------------------

class TestNothingElseMoved:

	def test_words_that_mean_true_and_false_are_untouched (self) -> None:

		"""Only the number rules changed.

		A config file saying ``enabled: yes`` is not what went wrong, and 1.2's
		core schema would have turned every one of them into a string.
		"""

		loaded = subsample.yaml_numbers.load(
			"a: yes\nb: no\nc: on\nd: off\ne: true\nf: false\ng: null\nh: ~"
		)

		assert loaded["a"] is True
		assert loaded["b"] is False
		assert loaded["c"] is True
		assert loaded["d"] is False
		assert loaded["e"] is True
		assert loaded["f"] is False
		assert loaded["g"] is None
		assert loaded["h"] is None

	def test_the_rest_of_pyyaml_is_left_alone (self) -> None:

		"""The resolvers are copied onto a private loader, not edited in place.

		Editing yaml.SafeLoader would change how every other library in the
		process reads YAML, which is not ours to do.  Checking a value is not
		enough on its own: the octal reading is decided by the constructor, and
		a shared resolver table would leave it looking unchanged either way.
		The sexagesimal reading is resolver-only, so it is the one that moves.
		"""

		assert yaml.safe_load("value: 036")["value"] == 30
		assert yaml.safe_load("cue: 1:30")["cue"] == 90

		assert (
			yaml.SafeLoader.yaml_implicit_resolvers
			is not subsample.yaml_numbers._Yaml12Loader.yaml_implicit_resolvers
		), "the private loader shares PyYAML's own resolver table"


	def test_yaml_11_spellings_1_2_dropped_are_now_text (self) -> None:

		"""Two forms YAML 1.1 read as numbers and 1.2's core schema does not.

		Kept as 1.2 has them rather than added back as a third superset,
		because Subsequence reads the shared definitions file by exactly these
		rules and a divergence is the whole thing this was raised about.  The
		digit separator is the only one anybody would plausibly type, and it
		fails loudly: a definitions file refuses `1_0` by name, and a config
		value survives it anyway, since Python's own float() reads underscores.
		"""

		assert subsample.yaml_numbers.load("a: 1_000")["a"] == "1_000"
		assert subsample.yaml_numbers.load("b: 0b101")["b"] == "0b101"

	def test_a_definitions_file_refuses_a_separator_by_name (self, tmp_path: pathlib.Path) -> None:

		"""The one place the separator mattered, and it says so."""

		_write(tmp_path, "defs.yaml", "notes:\n  kick: 1_0\n")

		with pytest.raises(ValueError, match="'kick' must be a whole number"):
			subsample.definitions.load_definitions(
				{"my": "defs.yaml"}, tmp_path, reserved_prefixes=frozenset({"drum"}),
			)

	def test_an_exponent_without_a_dot_is_now_a_number (self) -> None:

		"""1.1 needed a dot, so `1e3` was the string '1e3'. 1.2 does not."""

		assert subsample.yaml_numbers.load("gain: 1e3")["gain"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# The fixture Subsequence holds a copy of
# ---------------------------------------------------------------------------

class TestTheSharedFixture:

	def test_both_tools_read_the_same_file_the_same_way (self) -> None:

		"""The file that keeps the shared format one format.

		If this ever needs changing, change Subsequence's copy in the same
		breath - a definitions file that names one drum here and another there
		is the whole thing this was raised about.
		"""

		with _FIXTURE.open(encoding="utf-8") as handle:
			loaded = subsample.yaml_numbers.load(handle)

		assert loaded == {
			"octal_looking":      36,
			"also_octal_looking": 42,
			"long_run":           12345,
			"explicit_octal":     34,
			"hexadecimal":        42,
			"colon_pair":         "1:30",
			"plain":              38,
			"negative":           -7,
			"real_float":         1.5,
		}


# ---------------------------------------------------------------------------
# Every file Subsample reads
# ---------------------------------------------------------------------------

class TestEveryFileSubsampleReads:

	def test_a_definitions_file (self, tmp_path: pathlib.Path) -> None:

		"""The format shared with a sequencer, and the reason this was raised."""

		_write(tmp_path, "defs.yaml", "notes:\n  kick: 036\n  snare: 038\n")

		definitions = subsample.definitions.load_definitions(
			{"my": "defs.yaml"}, tmp_path, reserved_prefixes=frozenset({"drum"}),
		)

		assert definitions.tables["my"]["notes"]["kick"] == 36
		assert definitions.tables["my"]["notes"]["snare"] == 38

	def test_a_midi_map (self, tmp_path: pathlib.Path) -> None:

		"""The file a user edits most, and it is nothing but note numbers."""

		path = _write(tmp_path, "midi-map.yaml", """
assignments:
  - name: Kick
    channel: 10
    notes: 036
    select:
      where:
        reference: BD0025
""")

		result = subsample.player.load_midi_map(path, ["BD0025"])

		assert (9, 36) in result.note_map, "036 was read as another drum"
		assert (9, 30) not in result.note_map

	def test_an_ensemble_map (self, tmp_path: pathlib.Path) -> None:

		"""An ensemble binds its sample sets to MIDI channels, also written bare."""

		_write(tmp_path, "kit/midi-map.yaml", """
assignments:
  - name: Kick
    notes: 36
    select:
      where:
        reference: BD0025
""")
		path = _write(tmp_path, "ensemble.yaml", """
maps:
  - map: kit/midi-map.yaml
    channel: 012
""")

		includes = subsample.player._read_ensemble_includes(path, strict=False)

		assert [include.channel for include in includes] == [12], (
			"012 was read as channel 10"
		)

	def test_config_yaml (self, tmp_path: pathlib.Path) -> None:

		"""A port is in range under either reading, so this tests the reading."""

		path = _write(tmp_path, "config.yaml", "osc:\n  send_port: 012000\n")

		config = subsample.config.load_config(path)

		assert config.osc.send_port == 12000, "012000 was read as octal"


# ---------------------------------------------------------------------------
# A key written twice (#3872)
# ---------------------------------------------------------------------------

_FIX: typing.Final[str] = "Write it once, with everything from both."


class TestAKeyWrittenTwice:

	"""YAML says a mapping's keys are unique; PyYAML kept the second and dropped the first."""

	def test_a_second_player_section_in_config_yaml (self, tmp_path: pathlib.Path) -> None:

		"""A player: block added at the end of the file --init wrote lost its
		map line, and the player then would not start for want of a map."""

		path = _write(tmp_path, "config.yaml", (
			"player:\n"
			"  midi_map: midi-map-gm-drums.yaml\n"
			"\n"
			"player:\n"
			"  enabled: true\n"
		))

		with pytest.raises(ValueError) as caught:
			subsample.config.load_config(path)

		assert str(caught.value) == f"Config file {path}: 'player' is written twice, on lines 1 and 4. {_FIX}"

	def test_the_loader_names_the_file_the_key_and_both_lines (self, tmp_path: pathlib.Path) -> None:
		path = _write(tmp_path, "any.yaml", "a: 1\nb: 2\na: 3\n")

		with path.open(encoding="utf-8") as handle:
			with pytest.raises(subsample.yaml_numbers.DuplicateKeyError) as caught:
				subsample.yaml_numbers.load(handle)

		error = caught.value
		assert (error.key, error.first_line, error.second_line) == ("a", 1, 3)
		assert str(error) == f"{path}: 'a' is written twice, on lines 1 and 3. {_FIX}"

	def test_it_is_reported_wherever_a_malformed_file_is (self) -> None:

		"""Every caller catches yaml.YAMLError already, a watched map's reload included."""

		assert issubclass(subsample.yaml_numbers.DuplicateKeyError, yaml.YAMLError)

	def test_a_nested_key_is_named_by_its_path (self) -> None:
		with pytest.raises(subsample.yaml_numbers.DuplicateKeyError) as caught:
			subsample.yaml_numbers.load("player:\n  audio:\n    device: a\n    device: b\n")

		assert caught.value.key == "player.audio.device"

	def test_a_key_in_a_list_item_is_named_from_the_item (self) -> None:
		with pytest.raises(subsample.yaml_numbers.DuplicateKeyError) as caught:
			subsample.yaml_numbers.load("assignments:\n  - name: Kick\n    notes: 36\n    notes: 38\n")

		assert (caught.value.key, caught.value.first_line, caught.value.second_line) == ("notes", 3, 4)

	def test_keys_compare_as_the_values_they_read_as (self) -> None:

		"""036 is thirty-six here, so it is the same note as 36."""

		with pytest.raises(subsample.yaml_numbers.DuplicateKeyError) as caught:
			subsample.yaml_numbers.load("notes:\n  36: kick\n  036: snare\n")

		assert caught.value.key == "notes.036"

	def test_the_first_in_reading_order_is_named (self) -> None:
		with pytest.raises(subsample.yaml_numbers.DuplicateKeyError) as caught:
			subsample.yaml_numbers.load("player:\n  enabled: true\n  enabled: false\nplayer: {}\n")

		assert caught.value.key == "player.enabled"

	def test_the_same_key_in_two_sections_is_not_a_repeat (self) -> None:
		assert subsample.yaml_numbers.load("recorder:\n  enabled: false\nplayer:\n  enabled: true\n") == {
			"recorder": {"enabled": False},
			"player":   {"enabled": True},
		}

	def test_a_merge_may_be_overridden (self) -> None:

		"""The keys a merge brings in are there to be written over."""

		loaded = subsample.yaml_numbers.load(
			"base: &base\n  gain: 1\n  pan: 0\nkick:\n  <<: *base\n  gain: 2\n"
		)

		assert loaded["kick"] == {"gain": 2, "pan": 0}

	def test_one_anchor_used_twice_is_not_a_repeat (self) -> None:
		loaded = subsample.yaml_numbers.load("a: &x\n  k: 1\nb: *x\nc: *x\n")

		assert loaded == {"a": {"k": 1}, "b": {"k": 1}, "c": {"k": 1}}

	def test_a_midi_map (self, tmp_path: pathlib.Path) -> None:

		"""A second assignments: list dropped the first."""

		path = _write(tmp_path, "midi-map.yaml", (
			"assignments:\n"
			"  - name: Kick\n"
			"    notes: 36\n"
			"    select:\n"
			"      where:\n"
			"        reference: BD0025\n"
			"assignments:\n"
			"  - name: Snare\n"
			"    notes: 38\n"
			"    select:\n"
			"      where:\n"
			"        reference: BD0025\n"
		))

		with pytest.raises(subsample.yaml_numbers.DuplicateKeyError) as caught:
			subsample.player.load_midi_map(path, ["BD0025"])

		assert str(caught.value) == f"{path}: 'assignments' is written twice, on lines 1 and 7. {_FIX}"

	def test_an_ensemble_map (self, tmp_path: pathlib.Path) -> None:
		path = _write(tmp_path, "ensemble.yaml", "maps:\n  - map: a.yaml\n    channel: 10\nmaps: []\n")

		with pytest.raises(subsample.yaml_numbers.DuplicateKeyError, match="'maps' is written twice, on lines 1 and 4"):
			subsample.player._read_ensemble_includes(path, strict=False)

	def test_a_definitions_file (self, tmp_path: pathlib.Path) -> None:
		_write(tmp_path, "defs.yaml", "notes:\n  kick: 36\nnotes:\n  snare: 38\n")

		with pytest.raises(ValueError) as caught:
			subsample.definitions.load_definitions(
				{"my": "defs.yaml"}, tmp_path, reserved_prefixes=frozenset({"drum"}),
			)

		assert str(caught.value).endswith(f"(prefix 'my'): 'notes' is written twice, on lines 1 and 3. {_FIX}")
