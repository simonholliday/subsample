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
