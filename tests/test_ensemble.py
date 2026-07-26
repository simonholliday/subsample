"""Tests for subsample.ensemble — parsing an ensemble's `maps:` block.

An ensemble binds sample sets to MIDI channels so several play at once.  This
module covers only the parse; loading and merging live in player.load_ensemble
and are covered by tests/test_player.py::TestEnsemble.
"""

import pathlib

import pytest

import subsample.ensemble
import subsample.definitions


class TestParseMapIncludes:

	"""Both entry forms, and the mistakes that would otherwise bind a set to the
	wrong channel — or to none at all — without saying so."""

	DIR = pathlib.Path("/project")

	def test_none_and_empty_yield_no_includes (self) -> None:
		assert subsample.ensemble.parse_map_includes(None, self.DIR) == []
		assert subsample.ensemble.parse_map_includes([], self.DIR) == []

	def test_bare_string_keeps_the_sets_own_channel (self) -> None:

		"""The natural spelling when each set already knows where it belongs."""

		includes = subsample.ensemble.parse_map_includes(["kit/midi-map.yaml"], self.DIR)

		assert len(includes) == 1
		assert includes[0].channel is None
		assert includes[0].map_path == "/project/kit/midi-map.yaml"

	def test_mapping_form_binds_a_channel (self) -> None:
		includes = subsample.ensemble.parse_map_includes(
			[{"channel": 11, "map": "vocals/live.yaml"}], self.DIR,
		)

		assert includes[0].channel == 11
		assert includes[0].map_path == "/project/vocals/live.yaml"

	def test_paths_resolve_against_the_ensemble (self) -> None:

		"""Same rule as every other path inside a map — it is what lets an
		ensemble and the sets it names travel together."""

		includes = subsample.ensemble.parse_map_includes(
			["../shared/kit/midi-map.yaml"], pathlib.Path("/project/songs"),
		)

		assert includes[0].map_path == "/project/shared/kit/midi-map.yaml"

	def test_absolute_include_path_is_kept (self) -> None:

		"""A set on a shared drive is named absolutely and must not be rebased
		onto the project."""

		includes = subsample.ensemble.parse_map_includes(
			["/mnt/shared/kit/midi-map.yaml"], self.DIR,
		)

		assert includes[0].map_path == "/mnt/shared/kit/midi-map.yaml"

	def test_channel_name_from_definitions (self) -> None:

		"""Bindings resolve through the project vocabulary, like every other
		channel field — the reason to prefer an ensemble file over config.yaml,
		which cannot see definitions."""

		definitions = subsample.definitions.Definitions(
			tables={"my": {"channels": {"kit": 7}}},
		)

		includes = subsample.ensemble.parse_map_includes(
			[{"channel": "my.kit", "map": "kit/midi-map.yaml"}], self.DIR, definitions,
		)

		assert includes[0].channel == 7

	def test_duplicate_channel_names_the_first_holder (self) -> None:

		"""Two sets on one channel would merge into one note map and collide (or
		worse, interleave as velocity layers).  Reject where both can be named."""

		with pytest.raises(ValueError, match="already bound"):
			subsample.ensemble.parse_map_includes(
				[
					{"channel": 10, "map": "a/midi-map.yaml"},
					{"channel": 10, "map": "b/midi-map.yaml"},
				],
				self.DIR,
			)

	def test_duplicate_include_is_rejected (self) -> None:

		"""Always a mistake: with one binding it is a duplicate, with two it
		would need independent copies of one set's samples."""

		with pytest.raises(ValueError, match="more than once"):
			subsample.ensemble.parse_map_includes(
				["kit/midi-map.yaml", "kit/midi-map.yaml"], self.DIR,
			)

	@pytest.mark.parametrize("channel", [0, 17, -1])
	def test_out_of_range_channel (self, channel: int) -> None:
		with pytest.raises(ValueError, match="1-16"):
			subsample.ensemble.parse_map_includes(
				[{"channel": channel, "map": "kit/midi-map.yaml"}], self.DIR,
			)

	def test_unknown_key_is_rejected (self) -> None:

		"""A typo like `chanel:` would silently leave the set on its own channel."""

		with pytest.raises(ValueError, match="unknown key"):
			subsample.ensemble.parse_map_includes(
				[{"chanel": 10, "map": "kit/midi-map.yaml"}], self.DIR,
			)

	def test_missing_map_key (self) -> None:
		with pytest.raises(ValueError, match="missing 'map'"):
			subsample.ensemble.parse_map_includes([{"channel": 10}], self.DIR)

	@pytest.mark.parametrize("entry", [42, None, ["a"]])
	def test_entry_must_be_a_path_or_mapping (self, entry: object) -> None:
		with pytest.raises(ValueError, match="expected a path or a mapping"):
			subsample.ensemble.parse_map_includes([entry], self.DIR)

	def test_maps_must_be_a_list (self) -> None:
		with pytest.raises(ValueError, match="must be a list"):
			subsample.ensemble.parse_map_includes({"10": "kit.yaml"}, self.DIR)

	@pytest.mark.parametrize("value", ["", "   ", 42])
	def test_map_must_be_a_non_empty_path (self, value: object) -> None:
		with pytest.raises(ValueError, match="non-empty path"):
			subsample.ensemble.parse_map_includes(
				[{"channel": 10, "map": value}], self.DIR,
			)
