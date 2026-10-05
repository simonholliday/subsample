"""Tests for note timing: each voice plays one buffer after its message arrived, at its own frame (#600).

A real MidiPlayer's audio callback is run buffer by buffer against a clock the
test moves by hand: at 44.1 kHz a buffer of 441 frames lasts 10 ms, so a
message arriving 2.5 ms into a buffer's span plays from frame 110 of the next.
"""

import threading
import typing
import unittest.mock

import mido
import numpy
import pytest

import subsample.library
import subsample.player
import subsample.similarity


_RATE   = 44100
_FRAMES = 441


class _Clock:

	"""A clock the test moves by hand, standing in for time.perf_counter."""

	def __init__ (self) -> None:

		"""Start at zero."""

		self.now = 0.0

	def __call__ (self) -> float:

		"""The time the test last set."""

		return self.now


def _player (clock: _Clock) -> subsample.player.MidiPlayer:

	"""A player on the test's clock, mixing without its limiter so levels read back exactly."""

	player = subsample.player.MidiPlayer(
		"Test Device",
		threading.Event(),
		instrument_library=unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary),
		similarity_matrix=unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix),
		midi_map={},
		sample_rate=_RATE,
		bit_depth=16,
	)
	player._clock = clock
	player._limiter_enabled = False

	return player


def _buffer (player: subsample.player.MidiPlayer, clock: _Clock, now: float) -> numpy.ndarray:

	"""Run one audio callback at ``now``, and return the first channel of what it played."""

	clock.now = now
	pcm, _flag = player._audio_callback_impl(None, _FRAMES, {}, 0)
	frames = numpy.frombuffer(pcm, dtype=numpy.int16).reshape(-1, player._output_channels)

	return frames[:, 0].astype(numpy.float32) / 32767.0


def _voice (
	player:    subsample.player.MidiPlayer,
	starts_at: typing.Optional[float] = None,
	level:     float = 0.5,
	note:      int = 36,
	length:    int = 4410,
	click:     bool = False,
	one_shot:  bool = False,
) -> subsample.player._Voice:

	"""Add a voice holding ``level`` throughout, or once at its first frame for a click."""

	audio = numpy.zeros((length, player._output_channels), dtype=numpy.float32)

	if click:
		audio[0] = level
	else:
		audio[:] = level

	voice = subsample.player._Voice(audio=audio, note=note, channel=9, one_shot=one_shot, starts_at=starts_at)
	player._voices.append(voice)

	return voice


def _frame (seconds: float) -> int:

	"""The frame a time this far into a buffer's span falls on."""

	return int(seconds * _RATE)


class TestTheSpan:

	"""The span of arrival times each buffer plays moves on by exactly one buffer."""

	def test_the_first_buffer_plays_the_span_just_ended (self) -> None:

		"""The first callback starts the span afresh, ending now."""

		clock = _Clock()
		player = _player(clock)

		_buffer(player, clock, 1.0)

		assert player._span == pytest.approx((0.99, 1.0))

	def test_an_early_or_late_callback_moves_no_note (self) -> None:

		"""Each span is one buffer long and begins where the last ended, but for the slight pull."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		for now in (1.0112, 1.0195, 1.0301, 1.0399):
			previous = player._span
			_buffer(player, clock, now)

			assert player._span is not None and previous is not None
			assert player._span[1] - player._span[0] == pytest.approx(_FRAMES / _RATE)
			assert player._span[0] == pytest.approx(previous[1], abs=0.0001)

	def test_the_span_follows_the_callbacks_over_a_long_set (self) -> None:

		"""An audio clock running 100 parts per million fast is followed, not drifted from."""

		clock = _Clock()
		player = _player(clock)
		interval = (_FRAMES / _RATE) * 1.0001

		for index in range(2000):
			_buffer(player, clock, 1.0 + index * interval)

		assert player._span is not None
		assert player._span[1] == pytest.approx(clock.now, abs=0.0001)

	def test_a_stall_starts_the_span_afresh (self) -> None:

		"""A callback half a second late ends its span at now, rather than far behind it."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		_buffer(player, clock, 1.5)

		assert player._span == pytest.approx((1.49, 1.5))


class TestANoteStartsAtItsOwnFrame:

	def test_a_note_plays_one_buffer_after_it_arrived (self) -> None:

		"""2.5 ms into the span, so frame 110 of the next buffer, and silence before it."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		_voice(player, starts_at=1.0025)
		played = _buffer(player, clock, 1.01)

		start = _frame(0.0025)
		assert numpy.all(played[:start] == 0.0)
		assert played[start:] == pytest.approx(0.5, abs=1e-4)

	def test_notes_keep_the_spacing_they_were_played_with (self) -> None:

		"""Two hits 5 ms apart in one span play 5 ms apart, not together at the buffer's start."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		_voice(player, starts_at=1.001, click=True)
		_voice(player, starts_at=1.006, click=True)
		played = _buffer(player, clock, 1.01)

		assert list(numpy.flatnonzero(played)) == [_frame(0.001), _frame(0.006)]

	def test_a_note_after_the_span_waits_for_the_next_buffer (self) -> None:

		"""One that arrived after this buffer was begun plays in the next, at its own frame."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		voice = _voice(player, starts_at=1.012, click=True)
		first = _buffer(player, clock, 1.01)

		assert numpy.all(first == 0.0)
		assert voice.position == 0 and voice.starts_at == 1.012

		second = _buffer(player, clock, 1.02)

		assert list(numpy.flatnonzero(second)) == [_frame(0.002)]

	def test_a_note_from_before_the_span_starts_at_once (self) -> None:

		"""One handled too late for its own buffer plays at the start of this one, late rather than lost."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		_voice(player, starts_at=0.995, click=True)
		played = _buffer(player, clock, 1.01)

		assert list(numpy.flatnonzero(played)) == [0]

	def test_a_voice_with_no_arrival_time_starts_at_once (self) -> None:

		"""As every voice did before #600."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		_voice(player, click=True)
		played = _buffer(player, clock, 1.01)

		assert list(numpy.flatnonzero(played)) == [0]

	def test_a_note_plays_on_into_the_next_buffer (self) -> None:

		"""A note started at frame 110 plays its remaining frames at the start of the next buffer."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		voice = _voice(player, starts_at=1.0025, length=_FRAMES)
		_buffer(player, clock, 1.01)
		played = _buffer(player, clock, 1.02)

		start = _frame(0.0025)
		assert played[:start] == pytest.approx(0.5, abs=1e-4)
		assert numpy.all(played[start:] == 0.0)
		assert voice not in player._voices


class TestAReleaseAtItsOwnFrame:

	def _sounding (self) -> tuple[subsample.player.MidiPlayer, _Clock, subsample.player._Voice]:

		"""A player with a voice already playing, one buffer in."""

		clock = _Clock()
		player = _player(clock)
		voice = _voice(player)
		_buffer(player, clock, 1.0)

		return player, clock, voice

	def test_a_release_fades_from_the_frame_it_arrived_on (self) -> None:

		"""Unfaded until 5 ms into the buffer, then the 10 ms declick, over into the next buffer."""

		player, clock, voice = self._sounding()

		subsample.player.MidiPlayer._release_held(player, 36, 9, 1.005)
		played = _buffer(player, clock, 1.01)

		turn = _frame(0.005)
		assert played[:turn + 1] == pytest.approx(0.5, abs=1e-4)
		assert numpy.all(numpy.diff(played[turn:]) <= 1e-6)
		assert played[-1] < 0.5

		after = _buffer(player, clock, 1.02)

		fade_end = player._release_fade_frames - (_FRAMES - turn)
		assert numpy.all(after[fade_end:] == 0.0)
		assert voice not in player._voices

	def test_a_release_after_the_span_leaves_the_buffer_unfaded (self) -> None:

		"""It fades in the next buffer instead, from its own frame there."""

		player, clock, voice = self._sounding()

		subsample.player.MidiPlayer._release_held(player, 36, 9, 1.012)
		first = _buffer(player, clock, 1.01)

		assert first == pytest.approx(0.5, abs=1e-4)
		assert voice.releases_at == 1.012 and voice.fade_pos == 0

		second = _buffer(player, clock, 1.02)

		turn = _frame(0.002)
		assert second[:turn + 1] == pytest.approx(0.5, abs=1e-4)
		assert second[turn + 10] < 0.5

	def test_a_short_note_starts_and_ends_in_one_buffer (self) -> None:

		"""A note-on and its note-off 5 ms apart sound for 5 ms, from the note-on's frame."""

		clock = _Clock()
		player = _player(clock)
		_buffer(player, clock, 1.0)

		_voice(player, starts_at=1.001)
		subsample.player.MidiPlayer._release_held(player, 36, 9, 1.006)
		played = _buffer(player, clock, 1.01)

		start, turn = _frame(0.001), _frame(0.006)
		assert numpy.all(played[:start] == 0.0)
		assert played[start:turn + 1] == pytest.approx(0.5, abs=1e-4)
		assert played[turn + 10] < 0.5

	def test_a_restruck_note_hands_over_at_the_new_hit (self) -> None:

		"""The held note fades from the frame where the same note struck again starts."""

		player, clock, _held = self._sounding()

		subsample.player.MidiPlayer._release_held(player, 36, 9, 1.005)
		_voice(player, starts_at=1.005, level=0.25)
		played = _buffer(player, clock, 1.01)

		turn = _frame(0.005)
		assert played[turn - 1] == pytest.approx(0.5, abs=1e-4)
		assert played[turn] == pytest.approx(0.75, abs=1e-4)
		assert played[turn + 100] < 0.75

	def test_a_choke_damps_where_the_choking_hit_starts (self) -> None:

		"""A closed hi-hat cuts the open one at its own frame, not at the buffer's start."""

		clock = _Clock()
		player = _player(clock)
		open_hat = _voice(player, note=46, one_shot=True)
		_buffer(player, clock, 1.0)

		player._choke_map = {(9, 42): frozenset({(9, 46)})}
		subsample.player.MidiPlayer._choke_voices(player, 9, 42, 1.005)
		played = _buffer(player, clock, 1.01)

		turn = _frame(0.005)
		assert open_hat.releasing
		assert played[:turn + 1] == pytest.approx(0.5, abs=1e-4)
		assert played[turn + 10] < 0.5

	def test_a_later_release_does_not_put_off_an_earlier_one (self) -> None:

		"""A choke after a note-off, before either takes effect, keeps the note-off's time."""

		voice = subsample.player._Voice(audio=numpy.zeros((10, 2), dtype=numpy.float32), note=36, channel=9)

		subsample.player._release(voice, 1.003)
		subsample.player._release(voice, 1.008)

		assert voice.releasing and voice.releases_at == 1.003


class TestArrival:

	"""Every message that starts or releases a voice carries when it arrived."""

	def test_a_message_is_stamped_when_it_arrives (self) -> None:

		"""Before the handler does any work, so a slow handler does not make the note late."""

		clock = _Clock()
		clock.now = 3.25
		player = _player(clock)
		player._handle_message = unittest.mock.MagicMock()  # type: ignore[method-assign]
		msg = mido.Message("note_on", channel=9, note=36, velocity=100)

		player._safe_handle_message(msg)

		player._handle_message.assert_called_once_with(msg, 3.25)

	@pytest.mark.parametrize("msg", [
		mido.Message("note_off", channel=9, note=36),
		mido.Message("note_on", channel=9, note=36, velocity=0),
		mido.Message("control_change", channel=9, control=120, value=0),
		mido.Message("control_change", channel=9, control=123, value=0),
	], ids=["note-off", "velocity-0", "all-sound-off", "all-notes-off"])
	def test_each_release_takes_effect_as_of_its_arrival (self, msg: mido.Message) -> None:

		"""A note-off, and both panics, release the voice from the frame they arrived on."""

		player = _player(_Clock())
		voice = _voice(player)

		player._handle_message(msg, 2.5)

		assert voice.releasing and voice.releases_at == 2.5
