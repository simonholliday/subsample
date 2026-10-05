"""Tests for OSC notes in the player: when they play, how loud, and that they take turns with MIDI (#3610, #603).

The receiver that reads them off the network is tested in test_osc
(TestOscNoteReceiver).  These hand a note straight to MidiPlayer.play_osc_note
as the receiver would, against the hand-moved clock test_note_timing uses.
"""

import logging
import threading
import time
import typing
import unittest.mock

import mido
import numpy
import pytest

import subsample.player
import subsample.query

import tests.test_note_timing as timing


class _Handled:

	"""A stand-in for _handle_message that records what it was given."""

	def __init__ (self) -> None:

		"""Nothing handled yet."""

		self.calls: list[tuple[mido.Message, typing.Optional[float], typing.Optional[float]]] = []

	def __call__ (
		self,
		msg:  mido.Message,
		at:   typing.Optional[float] = None,
		fine: typing.Optional[float] = None,
	) -> None:

		"""Record one message."""

		self.calls.append((msg, at, fine))


def _player_handling () -> tuple[subsample.player.MidiPlayer, _Handled, timing._Clock]:

	"""A player whose handler only records, on a clock at 100 s."""

	clock  = timing._Clock()
	clock.now = 100.0
	player = timing._player(clock)
	handled = _Handled()
	player._handle_message = handled  # type: ignore[method-assign]

	return player, handled, clock


class TestTheTimeANoteIsMeantFor:

	def test_a_bundles_time_becomes_the_players_clock (self, monkeypatch: pytest.MonkeyPatch) -> None:

		"""A note meant for 0.25 s from now is handled as if it arrived 0.25 s from now."""

		player, handled, _clock = _player_handling()
		monkeypatch.setattr(time, "time", lambda: 1000.0)

		player.play_osc_note(True, 9, 36, 0.5, 1000.25)

		assert handled.calls[0][1] == pytest.approx(100.25)

	def test_a_note_on_its_own_is_handled_as_it_arrives (self, monkeypatch: pytest.MonkeyPatch) -> None:

		player, handled, _clock = _player_handling()
		monkeypatch.setattr(time, "time", lambda: 1000.0)

		player.play_osc_note(True, 9, 36, 0.5, 1000.0)

		assert handled.calls[0][1] == pytest.approx(100.0)

	def test_a_timed_note_plays_one_buffer_after_its_time_at_its_own_frame (
		self, monkeypatch: pytest.MonkeyPatch,
	) -> None:

		"""As a MIDI note arriving then would (#4513): sent early, it waits, then lands on its frame."""

		clock  = timing._Clock()
		player = timing._player(clock)
		monkeypatch.setattr(time, "time", lambda: 1000.0 + clock.now)

		def _trigger (
			msg: mido.Message, assignment: typing.Any, pick_spec: typing.Any, effective_velocity: float,
			at: typing.Optional[float] = None, fine: typing.Optional[float] = None,
		) -> None:
			timing._voice(player, starts_at=at, click=True, one_shot=True)

		player._note_map = {(9, 36): [(subsample.query.Assignment(name="Kick", select=()), subsample.query.PickSpec(1, 1))]}
		player._trigger_one = _trigger  # type: ignore[method-assign]

		timing._buffer(player, clock, 0.010)

		# Sent at 12 ms, meant for 2.5 ms into the span from 30 to 40 ms, which
		# the callback at 40 ms plays, one buffer on.
		clock.now = 0.012
		player.play_osc_note(True, 9, 36, 1.0, 1000.0 + 0.0325)

		assert not timing._buffer(player, clock, 0.020).any()
		assert not timing._buffer(player, clock, 0.030).any()

		played = timing._buffer(player, clock, 0.040)

		assert numpy.flatnonzero(played).tolist() == [timing._frame(0.0025)]

	def test_a_note_far_ahead_is_warned_about_once_a_minute (
		self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
	) -> None:

		"""A sender with its clock out of step would leave every note waiting in silence."""

		player, handled, _clock = _player_handling()
		monkeypatch.setattr(time, "time", lambda: 1000.0)

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			for _ in range(3):
				player.play_osc_note(True, 9, 36, 0.5, 1030.0)

		assert len(handled.calls) == 3
		assert len([record for record in caplog.records if "ahead" in record.message]) == 1


class TestVelocity:

	@pytest.mark.parametrize(("velocity", "coarse", "fine"), [
		pytest.param(0.5, 64, 63.5, id="half"),
		pytest.param(1.0, 127, 127.0, id="full"),
		pytest.param(0.003, 1, 0.381, id="quiet-but-not-off"),
	])
	def test_the_layer_reads_it_scaled_and_the_rest_in_full (
		self, velocity: float, coarse: int, fine: float,
	) -> None:
		player, handled, _clock = _player_handling()

		player.play_osc_note(True, 9, 36, velocity, time.time())

		msg, _at, given = handled.calls[0]

		assert (msg.type, msg.velocity) == ("note_on", coarse)
		assert given == pytest.approx(fine)

	def test_a_velocity_of_0_is_a_note_on_of_0_as_midi_has_it (self) -> None:

		player, handled, _clock = _player_handling()

		player.play_osc_note(True, 9, 36, 0.0, time.time())

		msg, _at, given = handled.calls[0]

		assert (msg.type, msg.velocity, given) == ("note_on", 0, None)

	def test_a_note_off_is_a_note_off (self) -> None:

		player, handled, _clock = _player_handling()

		player.play_osc_note(False, 9, 36, 0.0, time.time())

		msg, _at, given = handled.calls[0]

		assert (msg.type, msg.channel, msg.note, given) == ("note_off", 9, 36, None)

	def test_the_effective_velocity_keeps_what_7_bits_would_lose (self) -> None:

		"""The layer is chosen by the scaled velocity; the velocity it plays at is unrounded."""

		player, _handled, _clock = _player_handling()
		soft = subsample.query.Assignment(name="Soft", select=(), velocity_trigger=(0, 63))
		hard = subsample.query.Assignment(name="Hard", select=(), velocity_trigger=(64, 127))
		pick = subsample.query.PickSpec(1, 1)

		layers = player._select_velocity_layers([(soft, pick), (hard, pick)], 64, 63.6)

		assert [layer[0].name for layer in layers] == ["Hard"]
		assert layers[0][2] == pytest.approx(63.6)

	def test_a_rescale_reads_the_unrounded_velocity (self) -> None:

		player, _handled, _clock = _player_handling()
		layer = subsample.query.Assignment(name="L", select=(), velocity_trigger=(0, 127), velocity_rescale_to=(0, 63))
		pick  = subsample.query.PickSpec(1, 1)

		layers = player._select_velocity_layers([(layer, pick)], 64, 63.5)

		assert layers[0][2] == pytest.approx(63.5 * 63 / 127)

	def test_a_midi_notes_velocity_is_still_whole (self) -> None:

		player, _handled, _clock = _player_handling()
		layer = subsample.query.Assignment(name="L", select=(), velocity_trigger=(0, 127), velocity_rescale_to=(0, 63))
		pick  = subsample.query.PickSpec(1, 1)

		layers = player._select_velocity_layers([(layer, pick)], 64)

		assert layers[0][2] == 32 and isinstance(layers[0][2], int)

	def test_a_velocity_pick_reads_the_unrounded_velocity (self) -> None:

		clock  = timing._Clock()
		player = timing._player(clock)
		asked: list[float] = []

		def _resolve (
			assignment: typing.Any, pick_spec: typing.Any, eff_library: typing.Any, velocity: float, turn: int = 0,
		) -> typing.Optional[int]:
			asked.append(velocity)

			return None

		player._resolve_sample_id = _resolve  # type: ignore[method-assign]
		player._note_map = {(9, 36): [(subsample.query.Assignment(name="L", select=()), subsample.query.PickSpec(None, None, "velocity"))]}

		player.play_osc_note(True, 9, 36, 0.5, time.time())

		assert asked == [pytest.approx(63.5)]


class TestOscNotesTakeTurnsWithMidi:

	"""_handle_message is written for one thread, and MIDI and OSC notes each have their own."""

	def test_both_ways_in_hold_the_handler_lock (self) -> None:

		player, handled, _clock = _player_handling()
		held: list[bool] = []

		def _record (msg: mido.Message, at: typing.Optional[float] = None, fine: typing.Optional[float] = None) -> None:
			held.append(player._handler_lock.locked())

		player._handle_message = _record  # type: ignore[method-assign]

		player._safe_handle_message(mido.Message("note_on", channel=9, note=36, velocity=100))
		player.play_osc_note(True, 9, 36, 0.5, time.time())

		assert held == [True, True]

	def test_a_midi_note_waits_while_an_osc_note_is_handled (self) -> None:

		player, _handled, _clock = _player_handling()
		inside  = threading.Event()
		release = threading.Event()
		order: list[str] = []

		def _slow (msg: mido.Message, at: typing.Optional[float] = None, fine: typing.Optional[float] = None) -> None:
			if fine is not None:
				order.append("osc begins")
				inside.set()
				release.wait(timeout=5.0)
				order.append("osc ends")
			else:
				order.append("midi")

		player._handle_message = _slow  # type: ignore[method-assign]

		osc = threading.Thread(target=player.play_osc_note, args=(True, 9, 36, 0.5, time.time()))
		osc.start()
		assert inside.wait(timeout=5.0)

		midi = threading.Thread(target=player._safe_handle_message, args=(mido.Message("note_on", channel=9, note=38, velocity=100),))
		midi.start()
		midi.join(timeout=0.2)

		assert midi.is_alive(), "the MIDI note ran while the OSC note was being handled"

		release.set()
		osc.join(timeout=5.0)
		midi.join(timeout=5.0)

		assert order == ["osc begins", "osc ends", "midi"]

	def test_a_midi_note_is_stamped_before_it_waits (self) -> None:

		"""Its time is when it arrived, not when the OSC note let it in."""

		player, handled, clock = _player_handling()
		clock.now = 7.0

		with player._handler_lock:
			midi = threading.Thread(target=player._safe_handle_message, args=(mido.Message("note_on", channel=9, note=38, velocity=100),))
			midi.start()
			midi.join(timeout=0.1)
			clock.now = 8.0

		midi.join(timeout=5.0)

		assert handled.calls[0][1] == 7.0

	def test_a_failing_osc_note_is_logged_not_raised (self, caplog: pytest.LogCaptureFixture) -> None:

		player, _handled, _clock = _player_handling()
		player._handle_message = unittest.mock.MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

		with caplog.at_level(logging.ERROR, logger="subsample.player"):
			player.play_osc_note(True, 9, 36, 0.5, time.time())

		assert any("OSC note handler failed" in record.message for record in caplog.records)
