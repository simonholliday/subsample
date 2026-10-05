"""Tests for pick: round_robin in the player — each note takes the matches in turn (#2674).

The parsing and the rank arithmetic are tested in test_query (TestRoundRobinPick)
and the published words in test_midi_map_schema.  These play notes through a
player and read which sound each note-on chose: the library's ``get`` is asked
for the chosen id and answers None, so the note stops there, before any audio.
"""

import dataclasses
import threading
import unittest.mock

import mido
import numpy
import pytest

import subsample.library
import subsample.player
import subsample.query
import subsample.similarity

import tests.helpers


_ROUND_ROBIN = subsample.query.PickSpec(None, None, "round_robin")


def _record (name: str, sample_id: int) -> subsample.library.SampleRecord:

	"""A sample the select can match, with a little audio so it loads."""

	return subsample.library.SampleRecord(
		sample_id   = sample_id,
		name        = name,
		spectral    = tests.helpers._make_spectral(),
		rhythm      = tests.helpers._make_rhythm(),
		pitch       = tests.helpers._make_pitch(),
		timbre      = tests.helpers._make_timbre(),
		level       = tests.helpers._make_level(),
		band_energy = tests.helpers._make_band_energy(),
		params      = tests.helpers._make_params(),
		duration    = 1.0,
		audio       = numpy.zeros((100, 2), dtype=numpy.int32),
	)


def _assignment (name: str = "Hats") -> subsample.query.Assignment:

	"""An assignment whose select matches every hat."""

	return subsample.query.Assignment(
		name=name,
		select=(subsample.query.SelectSpec(where=subsample.query.WherePredicate(name_glob="hat*")),),
	)


def _player (note_map: subsample.player.NoteMap) -> tuple[subsample.player.MidiPlayer, unittest.mock.MagicMock]:

	"""A player over three hats, whose library finds no audio, so a note-on only chooses."""

	library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
	library.samples.return_value = [_record("hat_a", 1), _record("hat_b", 2), _record("hat_c", 3)]
	library.get.return_value = None

	player = subsample.player.MidiPlayer(
		"Test Device",
		threading.Event(),
		instrument_library=library,
		similarity_matrix=unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix),
		midi_map=note_map,
		sample_rate=44100,
		bit_depth=16,
	)

	return player, library


def _play (player: subsample.player.MidiPlayer, *notes: tuple[int, int]) -> None:

	"""Strike each (channel, note) once, in order."""

	for channel, note in notes:
		player._handle_message(mido.Message("note_on", channel=channel, note=note, velocity=100))


def _chosen (library: unittest.mock.MagicMock) -> list[int]:

	"""The sample ids the note-ons chose, in order."""

	return [call.args[0] for call in library.get.call_args_list]


class TestEachNoteTakesTheMatchesInTurn:

	def test_each_note_on_plays_the_next_match_and_starts_again (self) -> None:

		asgn = _assignment()
		player, library = _player({(0, 42): [(asgn, _ROUND_ROBIN)]})
		ranked = player._candidate_cache[id(asgn)].ids

		_play(player, *[(0, 42)] * 4)

		assert _chosen(library) == [ranked[0], ranked[1], ranked[2], ranked[0]]

	def test_each_note_keeps_its_own_place (self) -> None:

		"""Two notes of a list share the assignment but not the turn, each from its own start."""

		asgn = _assignment()
		player, library = _player({
			(0, 42): [(asgn, _ROUND_ROBIN)],
			(0, 44): [(asgn, dataclasses.replace(_ROUND_ROBIN, start=1))],
		})
		ranked = player._candidate_cache[id(asgn)].ids

		_play(player, (0, 42), (0, 44), (0, 42), (0, 44), (0, 44))

		assert _chosen(library) == [ranked[0], ranked[1], ranked[1], ranked[2], ranked[0]]

	def test_a_note_on_another_channel_has_its_own_turn (self) -> None:

		asgn = _assignment()
		player, library = _player({
			(0, 42): [(asgn, _ROUND_ROBIN)],
			(1, 42): [(asgn, _ROUND_ROBIN)],
		})
		ranked = player._candidate_cache[id(asgn)].ids

		_play(player, (0, 42), (0, 42), (1, 42))

		assert _chosen(library) == [ranked[0], ranked[1], ranked[0]]

	def test_each_layer_of_a_note_keeps_its_own_turn (self) -> None:

		"""Stacked layers are separate assignments, and each steps once a note."""

		upper = dataclasses.replace(_assignment("Upper"), stack=True)
		lower = dataclasses.replace(_assignment("Lower"), stack=True)
		player, library = _player({(0, 42): [(upper, _ROUND_ROBIN), (lower, _ROUND_ROBIN)]})
		ranked = player._candidate_cache[id(upper)].ids

		_play(player, (0, 42), (0, 42))

		assert _chosen(library) == [ranked[0], ranked[0], ranked[1], ranked[1]]

	def test_a_bounded_turn_keeps_to_its_ranks (self) -> None:

		asgn = _assignment()
		player, library = _player({(0, 42): [(asgn, subsample.query.PickSpec(1, 2, "round_robin"))]})
		ranked = player._candidate_cache[id(asgn)].ids

		_play(player, *[(0, 42)] * 3)

		assert _chosen(library) == [ranked[0], ranked[1], ranked[0]]

	def test_a_turn_is_taken_under_the_state_lock (self) -> None:

		"""The read and the write of a turn are one step, so a reload's clear never loses one."""

		asgn = _assignment()
		player, _library = _player({(0, 42): [(asgn, _ROUND_ROBIN)]})
		events: list[str] = []

		class _Tracking:

			def __enter__ (self) -> None:
				events.append("acquire")

			def __exit__ (self, *_exc: object) -> None:
				events.append("release")

		player._state_lock = _Tracking()  # type: ignore[assignment]

		_play(player, (0, 42))

		assert events and events.count("acquire") == events.count("release")
		assert player._pick_turns == {(0, 42, id(asgn)): 1}


class TestATurnStartsAgainWithNewRules:

	def test_a_program_change_starts_every_turn_again (self) -> None:

		asgn = _assignment()
		player, _library = _player({(0, 42): [(asgn, _ROUND_ROBIN)]})

		_play(player, (0, 42), (0, 42))
		player._forget_previous_rules()

		assert player._pick_turns == {}

	def test_a_retired_assignments_turn_is_let_go (self) -> None:

		live = _assignment("Live")
		dead = _assignment("Dead")
		player, _library = _player({(0, 42): [(live, _ROUND_ROBIN)]})
		player._pick_turns[(0, 42, id(live))] = 3
		player._pick_turns[(0, 42, id(dead))] = 5

		player._prune_stale_layer_state()

		assert player._pick_turns == {(0, 42, id(live)): 3}


class TestTheStartupLogNamesTheTurn:

	@pytest.mark.parametrize(("pick", "suffix"), [
		pytest.param(_ROUND_ROBIN, " pick round_robin", id="every-match"),
		pytest.param(subsample.query.PickSpec(1, None, "round_robin"), " pick round_robin", id="from-the-first"),
		pytest.param(subsample.query.PickSpec(1, 4, "round_robin"), " pick round_robin 1-4", id="bounded"),
		pytest.param(subsample.query.PickSpec(2, None, "round_robin"), " pick round_robin 2+", id="open-above"),
	])
	def test_the_suffix_names_the_mode_and_its_bounds (self, pick: subsample.query.PickSpec, suffix: str) -> None:

		assert subsample.player._format_pick_suffix(pick) == suffix

	def test_a_list_with_one_turn_reads_as_one_pick (self, caplog: pytest.LogCaptureFixture) -> None:

		"""Each note's start differs, but the map wrote one pick, so it is not called distributed."""

		asgn = _assignment()
		note_map: subsample.player.NoteMap = {
			(0, note): [(asgn, dataclasses.replace(_ROUND_ROBIN, start=index))]
			for index, note in enumerate((42, 43, 44))
		}

		with caplog.at_level("INFO", logger="subsample.player"):
			_player(note_map)

		text = caplog.text

		assert "pick round_robin" in text
		assert "pick distributed" not in text
