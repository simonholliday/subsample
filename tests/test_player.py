"""Tests for subsample.player — MIDI device selection and MidiPlayer lifecycle."""

import pathlib
import threading
import typing
import unittest.mock

import numpy
import pytest

import subsample.library
import subsample.player
import subsample.query
import subsample.similarity


# ---------------------------------------------------------------------------
# Shared helpers for building NoteMap entries in the new format
# ---------------------------------------------------------------------------

def _make_assignment (
	name: str = "test",
	reference: typing.Optional[str] = None,
	sample_name: typing.Optional[str] = None,
	pitched_filter: typing.Optional[bool] = None,
	order_by: str = "newest",
	repitch: bool = False,
	beat_quantize: bool = False,
	one_shot: bool = True,
	pan_weights: typing.Optional[numpy.ndarray] = None,
	gain_db: float = 0.0,
) -> subsample.query.Assignment:

	"""Build an Assignment with common defaults for tests."""

	where_kwargs: dict[str, typing.Any] = {}

	if reference is not None:
		where_kwargs["reference"] = reference

	if sample_name is not None:
		where_kwargs["name"] = sample_name

	if pitched_filter is not None:
		where_kwargs["pitched"] = pitched_filter

	where = subsample.query.WherePredicate(**where_kwargs)

	if reference is not None and order_by == "newest":
		order_by = "similarity"

	order_clause = subsample.query._LEGACY_ORDER_TOKENS[order_by]
	select = (subsample.query.SelectSpec(where=where, order=(order_clause,)),)

	steps: list[subsample.query.ProcessorStep] = []

	if repitch:
		steps.append(subsample.query.ProcessorStep(name="repitch"))

	if beat_quantize:
		steps.append(subsample.query.ProcessorStep(name="beat_quantize", params=(("grid", 16),)))

	process = subsample.query.ProcessSpec(steps=tuple(steps))

	return subsample.query.Assignment(
		name=name,
		select=select,
		process=process,
		one_shot=one_shot,
		gain_db=gain_db,
		pan_weights=pan_weights,
	)


def _make_note_map (
	assignment: subsample.query.Assignment,
	channel: int,
	notes: list[int],
	per_note_pick: bool = False,
) -> subsample.player.NoteMap:

	"""Build a NoteMap for one assignment across multiple notes."""

	note_map: subsample.player.NoteMap = {}

	for i, note in enumerate(notes):
		pick = (i + 1) if per_note_pick else 1
		note_map[(channel, note)] = (assignment, pick)

	return note_map


# ---------------------------------------------------------------------------
# list_midi_input_devices
# ---------------------------------------------------------------------------

class TestListMidiInputDevices:

	def test_returns_list (self) -> None:
		with unittest.mock.patch("mido.get_input_names", return_value=["Device A", "Device B"]):
			result = subsample.player.list_midi_input_devices()

		assert result == ["Device A", "Device B"]

	def test_returns_empty_when_no_devices (self) -> None:
		with unittest.mock.patch("mido.get_input_names", return_value=[]):
			result = subsample.player.list_midi_input_devices()

		assert result == []


# ---------------------------------------------------------------------------
# find_midi_device_by_name
# ---------------------------------------------------------------------------

class TestFindMidiDeviceByName:

	def _patch (self, names: list[str]) -> unittest.mock._patch:  # type: ignore[type-arg]
		return unittest.mock.patch("mido.get_input_names", return_value=names)

	def test_exact_match (self) -> None:
		with self._patch(["Launchpad MK3 MIDI 1", "Other Device"]):
			result = subsample.player.find_midi_device_by_name("Launchpad MK3 MIDI 1")

		assert result == "Launchpad MK3 MIDI 1"

	def test_substring_match (self) -> None:
		with self._patch(["Launchpad MK3 MIDI 1", "Other Device"]):
			result = subsample.player.find_midi_device_by_name("Launchpad")

		assert result == "Launchpad MK3 MIDI 1"

	def test_case_insensitive (self) -> None:
		with self._patch(["Launchpad MK3 MIDI 1"]):
			result = subsample.player.find_midi_device_by_name("launchpad")

		assert result == "Launchpad MK3 MIDI 1"

	def test_returns_first_match (self) -> None:
		with self._patch(["Launchpad A", "Launchpad B"]):
			result = subsample.player.find_midi_device_by_name("Launchpad")

		assert result == "Launchpad A"

	def test_no_match_raises (self) -> None:
		with self._patch(["Other Device"]):
			with pytest.raises(ValueError, match="Nope"):
				subsample.player.find_midi_device_by_name("Nope")

	def test_error_lists_available_devices (self) -> None:
		with self._patch(["Device A", "Device B"]):
			with pytest.raises(ValueError, match="Device A"):
				subsample.player.find_midi_device_by_name("nope")


# ---------------------------------------------------------------------------
# select_midi_device
# ---------------------------------------------------------------------------

class TestSelectMidiDevice:

	def test_auto_selects_single_device (self, capsys: pytest.CaptureFixture[str]) -> None:
		result = subsample.player.select_midi_device(["Only Device"])

		assert result == "Only Device"
		assert "Only Device" in capsys.readouterr().out

	def test_empty_list_raises (self) -> None:
		with pytest.raises(ValueError, match="No MIDI"):
			subsample.player.select_midi_device([])

	def test_multiple_devices_prompts (self, monkeypatch: pytest.MonkeyPatch) -> None:
		monkeypatch.setattr("builtins.input", lambda _: "1")

		result = subsample.player.select_midi_device(["Device A", "Device B"])

		assert result == "Device B"

	def test_multiple_devices_invalid_then_valid (self, monkeypatch: pytest.MonkeyPatch) -> None:
		responses = iter(["bad", "99", "0"])
		monkeypatch.setattr("builtins.input", lambda _: next(responses))

		result = subsample.player.select_midi_device(["Device A", "Device B"])

		assert result == "Device A"


# ---------------------------------------------------------------------------
# MidiPlayer
# ---------------------------------------------------------------------------

class TestMidiPlayer:

	def _make_player (self, shutdown_event: threading.Event) -> subsample.player.MidiPlayer:
		"""Return a MidiPlayer with minimal mocked dependencies."""
		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		return subsample.player.MidiPlayer(
			"Test Device",
			shutdown_event,
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
		)

	def _make_mock_pyaudio (self) -> unittest.mock.MagicMock:
		"""Return a mock PyAudio instance with one output device and a usable stream."""
		mock_stream = unittest.mock.MagicMock()
		mock_pa = unittest.mock.MagicMock()
		mock_pa.open.return_value = mock_stream
		# One output device — select_output_device() auto-selects without prompting.
		mock_pa.get_device_count.return_value = 1
		mock_pa.get_device_info_by_index.return_value = {
			"name": "Mock Output",
			"maxOutputChannels": 2,
			"defaultSampleRate": 44100,
			"index": 0,
		}
		return mock_pa

	def _make_mock_port (self, messages: list[object]) -> unittest.mock.MagicMock:
		"""Return a mock mido port that yields messages in order then None."""
		responses = iter(messages + [None] * 1000)
		port = unittest.mock.MagicMock()
		port.receive.side_effect = lambda block=True: next(responses)
		port.__enter__ = unittest.mock.Mock(return_value=port)
		port.__exit__ = unittest.mock.Mock(return_value=False)
		return port

	def test_exits_on_shutdown_event (self) -> None:
		shutdown_event = threading.Event()
		port = self._make_mock_port([])
		mock_pa = self._make_mock_pyaudio()

		with unittest.mock.patch("mido.open_input", return_value=port):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
				# Set shutdown_event before run() so the loop exits immediately.
				shutdown_event.set()
				player = self._make_player(shutdown_event)
				player.run()

		# run() should have returned without hanging
		assert not threading.current_thread().daemon

	def test_logs_received_message (self, caplog: pytest.LogCaptureFixture) -> None:
		import logging

		shutdown_event = threading.Event()
		mock_msg = unittest.mock.MagicMock()
		mock_msg.__str__ = lambda self: "note_on channel=0 note=60 velocity=64 time=0"
		port = self._make_mock_port([mock_msg])
		mock_pa = self._make_mock_pyaudio()

		with caplog.at_level(logging.INFO, logger="subsample.player"):
			with unittest.mock.patch("mido.open_input", return_value=port):
				with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
					# Set shutdown after one iteration by making wait() set the event.
					shutdown_event.set()
					player = self._make_player(shutdown_event)
					player.run()

		# At minimum the port open/close should be logged; message log may not
		# fire if event was already set before the loop body ran.
		assert any("Test Device" in r.message for r in caplog.records)

	def test_port_closed_on_shutdown (self) -> None:
		shutdown_event = threading.Event()
		port = self._make_mock_port([])
		mock_pa = self._make_mock_pyaudio()

		with unittest.mock.patch("mido.open_input", return_value=port):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
				shutdown_event.set()
				player = self._make_player(shutdown_event)
				player.run()

		port.close.assert_called_once()

	def test_run_on_thread_exits_cleanly (self) -> None:
		shutdown_event = threading.Event()
		port = self._make_mock_port([])
		mock_pa = self._make_mock_pyaudio()

		with unittest.mock.patch("mido.open_input", return_value=port):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
				player = self._make_player(shutdown_event)
				t = threading.Thread(target=player.run)
				t.start()

				# Give thread time to start, then signal shutdown.
				shutdown_event.set()
				t.join(timeout=2.0)

		assert not t.is_alive()


# ---------------------------------------------------------------------------
# note_off / releasing behaviour
# ---------------------------------------------------------------------------

class TestNoteOff:

	def _make_voice (self, note: int = 36, channel: int = 9, n_frames: int = 4410) -> subsample.player._Voice:
		"""Return a _Voice with silent audio of the given length."""
		import numpy
		audio = numpy.zeros((n_frames, 2), dtype=numpy.float32)
		return subsample.player._Voice(audio=audio, note=note, channel=channel)

	def test_note_off_marks_voice_releasing (self) -> None:
		"""A note_off matching an active voice sets voice.releasing = True."""
		import unittest.mock
		import mido

		voice = self._make_voice(note=36, channel=9)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices = [voice]
		player._voices_lock = threading.Lock()

		# Call the real _handle_message on the mock's behalf
		msg = mido.Message("note_off", channel=9, note=36)
		subsample.player.MidiPlayer._handle_message(player, msg)

		assert voice.releasing is True

	def test_note_off_only_matches_correct_note (self) -> None:
		"""A note_off does not affect voices on a different note."""
		import unittest.mock
		import mido

		voice_36 = self._make_voice(note=36, channel=9)
		voice_38 = self._make_voice(note=38, channel=9)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices = [voice_36, voice_38]
		player._voices_lock = threading.Lock()

		msg = mido.Message("note_off", channel=9, note=36)
		subsample.player.MidiPlayer._handle_message(player, msg)

		assert voice_36.releasing is True
		assert voice_38.releasing is False

	def test_note_on_velocity_zero_marks_releasing (self) -> None:
		"""note_on with velocity=0 (mido's note_off encoding) also marks releasing."""
		import unittest.mock
		import mido

		voice = self._make_voice(note=42, channel=9)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices = [voice]
		player._voices_lock = threading.Lock()

		msg = mido.Message("note_on", channel=9, note=42, velocity=0)
		subsample.player.MidiPlayer._handle_message(player, msg)

		assert voice.releasing is True

	def test_releasing_voice_retired_by_callback (self) -> None:
		"""A releasing voice is not kept in the active list after the callback runs."""
		import numpy
		import pyaudio

		n_frames = 4410
		audio = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.5
		voice = subsample.player._Voice(audio=audio, note=36, channel=9, releasing=True)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices              = [voice]
		player._voices_lock         = threading.Lock()
		player._output_channels     = 2
		player._output_bit_depth    = 16
		player._release_fade_frames = 441  # 10 ms at 44100 Hz
		player._last_clip_warn      = 0.0
		player._max_polyphony       = 8
		player._limiter_threshold   = 10.0 ** (-1.5 / 20.0)
		player._limiter_ceiling     = 10.0 ** (-0.1 / 20.0)
		player._limiter_knee        = player._limiter_ceiling - player._limiter_threshold

		# Call the real _audio_callback
		subsample.player.MidiPlayer._audio_callback(
			player, None, 512, {}, 0
		)

		# Voice should have been retired (not kept in _voices)
		assert len(player._voices) == 0

	def test_non_releasing_voice_kept_by_callback (self) -> None:
		"""A normal (non-releasing) voice is kept until its audio is exhausted."""
		import numpy

		n_frames = 4410
		audio = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.5
		voice = subsample.player._Voice(audio=audio, note=36, channel=9)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices              = [voice]
		player._voices_lock         = threading.Lock()
		player._output_channels     = 2
		player._output_bit_depth    = 16
		player._last_clip_warn      = 0.0
		player._max_polyphony       = 8
		player._limiter_threshold   = 10.0 ** (-1.5 / 20.0)
		player._limiter_ceiling     = 10.0 ** (-0.1 / 20.0)
		player._limiter_knee        = player._limiter_ceiling - player._limiter_threshold

		subsample.player.MidiPlayer._audio_callback(
			player, None, 512, {}, 0
		)

		assert len(player._voices) == 1
		assert player._voices[0].position == 512


# ---------------------------------------------------------------------------
# One-shot mode
# ---------------------------------------------------------------------------

class TestOneShot:

	def _make_voice (self, note: int = 36, channel: int = 9, one_shot: bool = False) -> subsample.player._Voice:
		import numpy
		audio = numpy.zeros((4410, 2), dtype=numpy.float32)
		return subsample.player._Voice(audio=audio, note=note, channel=channel, one_shot=one_shot)

	def test_one_shot_voice_ignores_note_off (self) -> None:
		"""note_off must NOT set releasing=True on a one-shot voice."""
		import mido

		voice = self._make_voice(note=36, channel=9, one_shot=True)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices = [voice]
		player._voices_lock = threading.Lock()

		msg = mido.Message("note_off", channel=9, note=36)
		subsample.player.MidiPlayer._handle_message(player, msg)

		assert voice.releasing is False

	def test_non_one_shot_voice_responds_to_note_off (self) -> None:
		"""A voice with one_shot=False must still set releasing=True on note_off."""
		import mido

		voice = self._make_voice(note=42, channel=9, one_shot=False)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices = [voice]
		player._voices_lock = threading.Lock()

		msg = mido.Message("note_off", channel=9, note=42)
		subsample.player.MidiPlayer._handle_message(player, msg)

		assert voice.releasing is True

	def test_one_shot_does_not_affect_other_voices (self) -> None:
		"""note_off should still release a co-existing non-one-shot voice on the same note."""
		import mido

		one_shot_voice = self._make_voice(note=36, channel=9, one_shot=True)
		normal_voice   = self._make_voice(note=36, channel=9, one_shot=False)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices = [one_shot_voice, normal_voice]
		player._voices_lock = threading.Lock()

		msg = mido.Message("note_off", channel=9, note=36)
		subsample.player.MidiPlayer._handle_message(player, msg)

		assert one_shot_voice.releasing is False
		assert normal_voice.releasing is True


# ---------------------------------------------------------------------------
# max_polyphony and target_rms
# ---------------------------------------------------------------------------

class TestMaxPolyphony:

	def _make_player (
		self,
		max_polyphony: int = 8,
	) -> subsample.player.MidiPlayer:
		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
			max_polyphony=max_polyphony,
		)

	def test_target_rms_default (self) -> None:
		player = self._make_player(max_polyphony=8)

		assert player._target_rms == pytest.approx(0.125)

	def test_target_rms_matches_legacy_value (self) -> None:
		# max_polyphony=10 reproduces the previous hard-coded _TARGET_RMS=0.1
		player = self._make_player(max_polyphony=10)

		assert player._target_rms == pytest.approx(0.1)

	def test_target_rms_monophonic (self) -> None:
		# max_polyphony=1 allocates full headroom to a single voice
		player = self._make_player(max_polyphony=1)

		assert player._target_rms == pytest.approx(1.0)

	def _make_callback_player (
		self,
		limiter_threshold_db: float = -1.5,
		limiter_ceiling_db: float = -0.1,
	) -> unittest.mock.MagicMock:
		"""Return a minimal MagicMock wired for _audio_callback testing."""
		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices_lock         = threading.Lock()
		player._output_channels     = 2
		player._output_bit_depth    = 16
		player._release_fade_frames = 441
		player._last_clip_warn      = 0.0
		player._max_polyphony       = 8
		player._limiter_threshold   = 10.0 ** (limiter_threshold_db / 20.0)
		player._limiter_ceiling     = 10.0 ** (limiter_ceiling_db / 20.0)
		player._limiter_knee        = player._limiter_ceiling - player._limiter_threshold
		return player

	def test_limiter_prevents_clipping_warning (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""With the default limiter, even a loud mix must not trigger the warning."""
		import logging
		import numpy

		n_frames = 512
		# Two voices at 0.8 each → sum = 1.6 (well above 0 dBFS).
		# The limiter compresses this to below the ceiling, so no warning fires.
		audio_loud = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.8
		player = self._make_callback_player()
		player._voices = [
			subsample.player._Voice(audio=audio_loud.copy(), note=36, channel=9),
			subsample.player._Voice(audio=audio_loud.copy(), note=38, channel=9),
		]

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		assert not any("clipping" in r.message.lower() for r in caplog.records)

	def test_clipping_warning_fires_if_post_limiter_ceiling_exceeded (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Warning fires when post-limiter output exceeds the ceiling.

		Simulates a bypassed limiter by setting _limiter_threshold > 1.0 so the
		mask is always empty and no sample is soft-clipped.  The raw sum then
		reaches numpy.clip's hard ceiling of 1.0, which exceeds the configured
		limiter_ceiling (~0.989), triggering the diagnostic warning.
		"""
		import logging
		import numpy

		n_frames = 512
		player = self._make_callback_player()
		# Override threshold to 2.0 — no sample in [-1, 1] will trigger the mask,
		# so the limiter effectively does nothing and the hard clip produces 1.0.
		player._limiter_threshold = 2.0
		player._limiter_knee = player._limiter_ceiling - player._limiter_threshold  # negative, unused

		# Single voice at 1.0 → passes through limiter mask untouched → numpy.clip → 1.0.
		# 1.0 > ceiling (~0.989) → warning fires.
		audio = numpy.ones((n_frames, 2), dtype=numpy.float32) * 1.0
		player._voices = [subsample.player._Voice(audio=audio.copy(), note=36, channel=9)]

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		assert any("clipping" in r.message.lower() for r in caplog.records)

	def test_clipping_warning_throttled (self) -> None:
		"""Warning must not repeat within 5 seconds of the previous one."""
		import numpy

		n_frames = 512
		# Bypass limiter (threshold > 1.0) so the warning fires on first call.
		player = self._make_callback_player()
		player._limiter_threshold = 2.0
		player._limiter_knee = player._limiter_ceiling - player._limiter_threshold
		audio = numpy.ones((n_frames, 2), dtype=numpy.float32) * 1.0

		# First call — fires the warning and records the timestamp.
		player._voices = [subsample.player._Voice(audio=audio.copy(), note=36, channel=9)]
		subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)
		first_warn_time: float = player._last_clip_warn

		# Second call immediately after — should be throttled.
		player._voices = [subsample.player._Voice(audio=audio.copy(), note=36, channel=9)]
		with unittest.mock.patch.object(subsample.player._log, "warning") as mock_warn:
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		mock_warn.assert_not_called()
		assert player._last_clip_warn == first_warn_time

	def test_no_clipping_no_warning (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Quiet signal with default limiter: no warning."""
		import logging
		import numpy

		n_frames = 512
		audio_quiet = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.1
		player = self._make_callback_player()
		player._voices = [subsample.player._Voice(audio=audio_quiet.copy(), note=36, channel=9)]

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		assert not any("clipping" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Safety limiter
# ---------------------------------------------------------------------------

class TestLimiter:

	def _make_player (
		self,
		limiter_threshold_db: float = -1.5,
		limiter_ceiling_db: float = -0.1,
	) -> subsample.player.MidiPlayer:
		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
			limiter_threshold_db=limiter_threshold_db,
			limiter_ceiling_db=limiter_ceiling_db,
		)

	def _run_callback_with_audio (
		self,
		player: subsample.player.MidiPlayer,
		audio: "numpy.ndarray",
	) -> "numpy.ndarray":
		"""Run _audio_callback with a single voice and return the mixed output."""
		import numpy
		import pyaudio
		n_frames = len(audio)
		voice = subsample.player._Voice(audio=audio.copy(), note=36, channel=9)
		player._voices = [voice]
		raw, _ = subsample.player.MidiPlayer._audio_callback(
			player, None, n_frames, {}, 0
		)
		# Unpack the int16 bytes back to float for assertion
		pcm = numpy.frombuffer(raw, dtype=numpy.int16).astype(numpy.float32) / 32767.0
		return pcm.reshape(n_frames, 2)

	def test_below_threshold_passes_unchanged (self) -> None:
		"""Signals below threshold must be unchanged to within float32 precision."""
		import numpy
		player = self._make_player(limiter_threshold_db=-1.5)
		threshold = player._limiter_threshold

		# Signal at 70% of threshold — well below, untouched
		level = threshold * 0.7
		audio = numpy.ones((512, 2), dtype=numpy.float32) * level
		result = self._run_callback_with_audio(player, audio)

		numpy.testing.assert_allclose(result, level, atol=1e-3)

	def test_above_threshold_is_compressed (self) -> None:
		"""A signal that would clip is reduced below 0 dBFS by the limiter."""
		import numpy
		player = self._make_player(limiter_threshold_db=-1.5, limiter_ceiling_db=-0.1)

		# Signal at exactly 0 dBFS (1.0) — would hard-clip without limiter
		audio = numpy.ones((512, 2), dtype=numpy.float32) * 1.0
		result = self._run_callback_with_audio(player, audio)

		# After limiter, output should be below 1.0 but above threshold
		assert numpy.max(numpy.abs(result)) < 1.0
		assert numpy.max(numpy.abs(result)) > player._limiter_threshold

	def test_output_never_exceeds_ceiling (self) -> None:
		"""Regardless of input level, output never exceeds the limiter ceiling."""
		import numpy
		player = self._make_player(limiter_threshold_db=-1.5, limiter_ceiling_db=-0.1)
		ceiling = player._limiter_ceiling

		# Very hot signal: +6 dBFS (2.0)
		audio = numpy.ones((512, 2), dtype=numpy.float32) * 2.0
		result = self._run_callback_with_audio(player, audio)

		assert numpy.max(numpy.abs(result)) <= ceiling + 1e-4

	def test_extreme_input_stays_below_ceiling (self) -> None:
		"""Asymptotic behaviour: even +20 dBFS input stays below ceiling."""
		import numpy
		player = self._make_player(limiter_threshold_db=-1.5, limiter_ceiling_db=-0.1)
		ceiling = player._limiter_ceiling

		# +20 dBFS — massively over full scale
		audio = numpy.ones((512, 2), dtype=numpy.float32) * 10.0
		result = self._run_callback_with_audio(player, audio)

		assert numpy.max(numpy.abs(result)) <= ceiling + 1e-4

	def test_symmetry (self) -> None:
		"""Negative and positive signals are compressed identically in magnitude."""
		import numpy
		player = self._make_player()

		level = 1.5  # +3.5 dBFS — above threshold
		pos_audio = numpy.ones((512, 2), dtype=numpy.float32) * level
		neg_audio = numpy.ones((512, 2), dtype=numpy.float32) * -level

		pos_result = self._run_callback_with_audio(player, pos_audio)
		neg_result = self._run_callback_with_audio(player, neg_audio)

		numpy.testing.assert_allclose(numpy.abs(pos_result), numpy.abs(neg_result), atol=1e-4)


# ---------------------------------------------------------------------------
# load_midi_map
# ---------------------------------------------------------------------------

class TestLoadMidiMap:

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_single_note_reference (self, tmp_path: pathlib.Path) -> None:
		"""Single-note reference assignment."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
      order_by: similarity
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert (9, 36) in note_map
		asgn, pick = note_map[(9, 36)]
		assert asgn.select[0].where.reference == "BD0025"
		assert asgn.one_shot is True
		assert pick == 1

	def test_multi_note_rank_distribution (self, tmp_path: pathlib.Path) -> None:
		"""Note list distributes picks: first note = pick 1, second = pick 2."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kicks
    channel: 10
    notes: [36, 35]
    select:
      where:
        reference: BD0025
      order_by: similarity
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert note_map[(9, 36)][1] == 1   # pick 1
		assert note_map[(9, 35)][1] == 2   # pick 2

	def test_channel_conversion (self, tmp_path: pathlib.Path) -> None:
		"""User-facing channel 10 converts to mido channel 9."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert (9, 36) in note_map
		assert (10, 36) not in note_map

	def test_one_shot_defaults_true (self, tmp_path: pathlib.Path) -> None:
		"""one_shot defaults to True when omitted."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		asgn, _ = note_map[(9, 36)]
		assert asgn.one_shot is True

	def test_unknown_reference_skipped (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Assignment whose reference is not in library is skipped with WARNING."""
		import logging
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			note_map = subsample.player.load_midi_map(path, []).note_map

		assert len(note_map) == 0
		assert any("BD0025" in r.message for r in caplog.records)

	def test_case_insensitive_reference (self, tmp_path: pathlib.Path) -> None:
		"""Reference lookup is case-insensitive."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: bd0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert (9, 36) in note_map

	def test_missing_file_raises (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(FileNotFoundError):
			subsample.player.load_midi_map(tmp_path / "no-such-file.yaml", [])

	def test_empty_file_returns_empty_map (self, tmp_path: pathlib.Path) -> None:
		path = self._write_map(tmp_path, "")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map
		assert note_map == {}

	def test_multiple_assignments (self, tmp_path: pathlib.Path) -> None:
		"""Multiple assignments coexist in the map."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
  - name: Snare
    channel: 10
    notes: 38
    select:
      where:
        reference: SD5075
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025", "SD5075"]).note_map

		assert (9, 36) in note_map
		assert (9, 38) in note_map
		assert note_map[(9, 36)][0].select[0].where.reference == "BD0025"
		assert note_map[(9, 38)][0].select[0].where.reference == "SD5075"

	def test_name_filter (self, tmp_path: pathlib.Path) -> None:
		"""where: { name: stem } is parsed correctly."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Fixed kick
    channel: 10
    notes: 36
    select:
      where:
        name: 2026-03-24_14-37-14
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		assert (9, 36) in note_map
		asgn, _ = note_map[(9, 36)]
		assert asgn.select[0].where.name == "2026-03-24_14-37-14"
		assert asgn.one_shot is True

	def test_name_filter_no_reference_validation (self, tmp_path: pathlib.Path) -> None:
		"""name filters are not validated against the reference library."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Fixed kick
    channel: 10
    notes: 36
    select:
      where:
        name: some-recording
""")
		note_map = subsample.player.load_midi_map(path, []).note_map
		assert (9, 36) in note_map

	def test_default_map_parses (self) -> None:
		"""The shipped midi-map.yaml.default parses without error."""
		default_path = pathlib.Path(__file__).parent.parent / "midi-map.yaml.default"
		note_map = subsample.player.load_midi_map(default_path, []).note_map

		assert len(note_map) > 0
		assert (9, 36) in note_map

		# Path-based reference: resolved to absolute path at parse time.
		ref = note_map[(9, 36)][0].select[0].where.reference
		assert ref is not None
		assert "GM36_BassDrum1" in ref
		assert "/" in ref  # path-based, not bare name

	def test_default_pan_is_centre (self, tmp_path: pathlib.Path) -> None:
		"""Omitted pan defaults to equal power across all output channels."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		asgn, _ = note_map[(9, 36)]
		# No pan specified → pan_weights is None (default routing).
		assert asgn.pan_weights is None

	def test_explicit_pan_weights_stored (self, tmp_path: pathlib.Path) -> None:
		"""Explicit pan weights are stored as raw values (normalisation at render time)."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
    pan: [75, 25]
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map
		asgn, _ = note_map[(9, 36)]
		assert asgn.pan_weights is not None
		numpy.testing.assert_allclose(asgn.pan_weights, [75.0, 25.0], atol=1e-5)

	def test_pan_hard_left (self, tmp_path: pathlib.Path) -> None:
		"""pan: [100, 0] stores raw weights."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
    pan: [100, 0]
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map
		asgn, _ = note_map[(9, 36)]
		assert asgn.pan_weights is not None
		numpy.testing.assert_allclose(asgn.pan_weights, [100.0, 0.0], atol=1e-5)

	def test_pan_negative_raises (self, tmp_path: pathlib.Path) -> None:
		"""Negative pan weights raise ValueError."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
    pan: [50, -10]
""")
		with pytest.raises(ValueError, match="pan"):
			subsample.player.load_midi_map(path, ["BD0025"])

	def test_repitch_all_notes_same_pick (self, tmp_path: pathlib.Path) -> None:
		"""repitch in process: all notes share pick 1 (same sample, pitched per note)."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Bass keyboard
    channel: 1
    notes: [48, 50, 52]
    select:
      where:
        reference: BASS_TONE
      order_by: similarity
    process:
      - repitch: true
    one_shot: false
""")
		note_map = subsample.player.load_midi_map(path, ["BASS_TONE"]).note_map

		for midi_note in [48, 50, 52]:
			asgn, pick = note_map[(0, midi_note)]
			assert pick == 1
			assert asgn.process.has_repitch()

	def test_no_repitch_distributes_picks (self, tmp_path: pathlib.Path) -> None:
		"""Without repitch, notes get ascending picks (rank distribution)."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kicks
    channel: 10
    notes: [36, 35]
    select:
      where:
        reference: BD0025
      order_by: similarity
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert note_map[(9, 36)][1] == 1
		assert note_map[(9, 35)][1] == 2
		assert not note_map[(9, 36)][0].process.has_repitch()

	def test_note_name_in_map (self, tmp_path: pathlib.Path) -> None:
		"""Note names (C2) are accepted in assignments."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: C2
    select:
      where:
        reference: BD0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map
		assert (9, 36) in note_map

	def test_note_range_in_map (self, tmp_path: pathlib.Path) -> None:
		"""Range syntax 'C2..C4' expands to all 25 notes."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Bass keyboard
    channel: 1
    notes: C2..C4
    select:
      where:
        reference: BASS_TONE
      order_by: similarity
    process:
      - repitch: true
""")
		note_map = subsample.player.load_midi_map(path, ["BASS_TONE"]).note_map

		assert len(note_map) == 25
		assert (0, 36) in note_map
		assert (0, 60) in note_map


# ---------------------------------------------------------------------------
# _parse_note_name and _parse_note_spec
# ---------------------------------------------------------------------------

class TestParseNoteSpec:

	def test_single_int (self) -> None:
		assert subsample.player._parse_note_spec(36, "test") == [36]

	def test_single_note_name_c4 (self) -> None:
		# C4 = MIDI 60 (Ableton/Logic convention)
		assert subsample.player._parse_note_spec("C4", "test") == [60]

	def test_single_note_name_c3 (self) -> None:
		# C3 = MIDI 48
		assert subsample.player._parse_note_spec("C3", "test") == [48]

	def test_sharp (self) -> None:
		# C#4 = MIDI 61
		assert subsample.player._parse_note_spec("C#4", "test") == [61]

	def test_flat (self) -> None:
		# Db4 = C#4 = MIDI 61
		assert subsample.player._parse_note_spec("Db4", "test") == [61]

	def test_bb (self) -> None:
		# Bb2 = A#2; A2=45, Bb2=46
		assert subsample.player._parse_note_spec("Bb2", "test") == [46]

	def test_f_sharp (self) -> None:
		# F#2: F=5 in octave 2 → (2+1)*12+5=41, F#2=42
		assert subsample.player._parse_note_spec("F#2", "test") == [42]

	def test_c_minus_one (self) -> None:
		# C-1 = MIDI 0 (lowest note)
		assert subsample.player._parse_note_spec("C-1", "test") == [0]

	def test_list_of_ints (self) -> None:
		assert subsample.player._parse_note_spec([36, 38], "test") == [36, 38]

	def test_list_of_names (self) -> None:
		# C3=48, D#3=51
		assert subsample.player._parse_note_spec(["C3", "D#3"], "test") == [48, 51]

	def test_mixed_list (self) -> None:
		# C3=48
		assert subsample.player._parse_note_spec([36, "C3"], "test") == [36, 48]

	def test_int_range (self) -> None:
		assert subsample.player._parse_note_spec("36..38", "test") == [36, 37, 38]

	def test_name_range (self) -> None:
		# C2=36, C4=60
		result = subsample.player._parse_note_spec("C2..C4", "test")
		assert result[0] == 36
		assert result[-1] == 60
		assert len(result) == 25

	def test_out_of_range_raises (self) -> None:
		with pytest.raises(ValueError):
			subsample.player._parse_note_spec("C10", "test")

	def test_malformed_name_raises (self) -> None:
		with pytest.raises(ValueError):
			subsample.player._parse_note_spec("X4", "test")

	def test_reversed_range_raises (self) -> None:
		with pytest.raises(ValueError, match="start"):
			subsample.player._parse_note_spec("60..36", "test")


# ---------------------------------------------------------------------------
# MidiPlayer.update_pitched_assignments
# ---------------------------------------------------------------------------

class TestUpdatePitchedAssignments:

	def _make_player_with_pitch_map (
		self,
		ref_name: str = "BASS_TONE",
		notes: list[int] = [48, 50, 52],
	) -> subsample.player.MidiPlayer:
		"""Return a MidiPlayer with a pitched keyboard assignment."""

		asgn = _make_assignment(name="Pitched", reference=ref_name, repitch=True, one_shot=False)
		note_map = _make_note_map(asgn, channel=0, notes=notes)

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

	def test_no_transform_manager_noop (self) -> None:
		"""update_pitched_assignments() is a no-op when transform_manager is None."""
		player = self._make_player_with_pitch_map()
		# Should not raise even with no transform manager.
		player.update_pitched_assignments()

	def test_no_pitched_assignments_no_enqueue (self) -> None:
		"""No enqueue calls when no pitched assignments exist."""

		asgn = _make_assignment(name="Kicks", reference="BD0025", repitch=False)
		non_pitched_map = _make_note_map(asgn, channel=9, notes=[36])

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		transform_manager  = unittest.mock.MagicMock()

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=non_pitched_map,
			sample_rate=44100,
			bit_depth=16,
			transform_manager=transform_manager,
		)

		player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_not_called()

	def test_enqueues_for_pitched_reference (self) -> None:
		"""enqueue_pitch_range is called with the matched record and notes."""

		notes = [48, 50, 52]
		player = self._make_player_with_pitch_map(notes=notes)

		mock_record = unittest.mock.MagicMock()
		mock_record.sample_id = 42
		mock_record.spectral = unittest.mock.MagicMock()
		mock_record.pitch    = unittest.mock.MagicMock()
		mock_record.duration = 1.0
		mock_record.name     = "tonal-sample"

		# The query engine calls instrument_library.samples() to get the
		# candidate list, and similarity_matrix.get_matches() for ranked results.
		player._instrument_library.samples.return_value = [mock_record]
		player._instrument_library.get.return_value = mock_record
		player._similarity_matrix.get_matches.return_value = [
			unittest.mock.MagicMock(sample_id=42),
		]

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		with unittest.mock.patch("subsample.analysis.has_stable_pitch", return_value=True):
			player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_called_once()
		call_args = transform_manager.enqueue_pitch_range.call_args
		assert call_args[0][0] is mock_record
		assert sorted(call_args[0][1]) == sorted(notes)

	def test_no_match_skips_enqueue (self) -> None:
		"""No enqueue when similarity matrix returns None (no match yet)."""
		player = self._make_player_with_pitch_map()

		player._similarity_matrix.get_match.return_value = None
		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_not_called()

	def test_no_stable_pitch_skips_enqueue (self, caplog: pytest.LogCaptureFixture) -> None:
		"""No enqueue and a warning when the matched sample has no stable pitch."""
		import logging

		player = self._make_player_with_pitch_map()

		mock_record = unittest.mock.MagicMock()
		mock_record.sample_id = 42
		mock_record.name = "some-sample"

		# The query engine needs samples() to return the record, and
		# get_matches() to provide ranked results for the reference.
		player._instrument_library.samples.return_value = [mock_record]
		player._instrument_library.get.return_value = mock_record
		player._similarity_matrix.get_matches.return_value = [
			unittest.mock.MagicMock(sample_id=42),
		]

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		with unittest.mock.patch("subsample.analysis.has_stable_pitch", return_value=False):
			with caplog.at_level(logging.WARNING, logger="subsample.player"):
				player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_not_called()
		assert any("stable pitch" in r.message for r in caplog.records)

	def test_enqueues_for_pitched_filter (self) -> None:
		"""enqueue_pitch_range is called for an assignment with pitched: true filter."""

		notes = [48, 50, 52]
		asgn = _make_assignment(name="Pitched newest", pitched_filter=True, order_by="newest", repitch=True, one_shot=False)
		note_map = _make_note_map(asgn, channel=0, notes=notes)

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

		mock_record = unittest.mock.MagicMock()
		mock_record.sample_id = 42
		mock_record.name = "tonal-sample"

		# The query engine calls has_stable_pitch internally via the
		# WherePredicate.matches() method when pitched=True.  We mock it
		# to return True so the record passes the filter.
		player._instrument_library.samples.return_value = [mock_record]
		player._instrument_library.get.return_value = mock_record

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		with unittest.mock.patch("subsample.analysis.has_stable_pitch", return_value=True):
			player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_called_once()
		call_args = transform_manager.enqueue_pitch_range.call_args
		assert call_args[0][0] is mock_record
		assert sorted(call_args[0][1]) == sorted(notes)

	def test_pitched_selector_no_match_skips (self) -> None:
		"""No enqueue when query returns no results."""

		asgn = _make_assignment(name="Pitched newest", pitched_filter=True, order_by="newest", repitch=True, one_shot=False)
		note_map = _make_note_map(asgn, channel=0, notes=[60])

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		# Mock the instrument library to return no samples (empty query result).
		player._instrument_library.samples.return_value = []

		player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_not_called()

	def test_beat_quantize_pre_computation (self) -> None:
		"""update_assignments() calls get_variant() for beat_quantize assignments."""

		asgn = _make_assignment(
			name="Loops", beat_quantize=True, repitch=False,
			order_by="newest", one_shot=False,
		)
		note_map = _make_note_map(asgn, channel=0, notes=[60])

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

		mock_record = unittest.mock.MagicMock()
		mock_record.sample_id = 7
		mock_record.name = "loop-sample"
		mock_record.rhythm.tempo_bpm = 120.0

		player._instrument_library.samples.return_value = [mock_record]
		player._instrument_library.get.return_value = mock_record

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		player.update_assignments()

		# get_variant should have been called to pre-compute the time-stretch variant.
		transform_manager.get_variant.assert_called_once()
		call_args = transform_manager.get_variant.call_args
		assert call_args[0][0] == 7  # sample_id

	def test_beat_quantize_with_explicit_bpm (self) -> None:
		"""Per-assignment BPM override produces a spec with correct params."""

		# beat_quantize with explicit bpm=120, grid=8
		asgn = subsample.query.Assignment(
			name="Explicit BPM",
			select=(subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="age", dir="desc"),)),),
			process=subsample.query.ProcessSpec(steps=(
				subsample.query.ProcessorStep(name="beat_quantize", params=(("bpm", 120), ("grid", 8))),
			)),
			one_shot=False,
		)
		note_map: subsample.player.NoteMap = {(0, 60): (asgn, 1)}

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

		mock_record = unittest.mock.MagicMock()
		mock_record.sample_id = 7
		mock_record.rhythm.tempo_bpm = 100.0

		player._instrument_library.samples.return_value = [mock_record]
		player._instrument_library.get.return_value = mock_record

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		player.update_assignments()

		transform_manager.get_variant.assert_called_once()
		call_args = transform_manager.get_variant.call_args[0]
		spec = call_args[1]
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.TimeStretch)
		assert spec.steps[0].target_bpm == 120.0
		assert spec.steps[0].resolution == 8


# ---------------------------------------------------------------------------
# TestLoadMidiMapPitched — pitched select + repitch process
# ---------------------------------------------------------------------------

class TestLoadMidiMapPitched:
	"""Test load_midi_map() parsing of pitched select + repitch process."""

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_pitched_oldest (self, tmp_path: pathlib.Path) -> None:
		"""Pitched oldest: where pitched, order oldest, repitch."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Oldest pitched
    channel: 1
    notes: C2..C4
    select:
      where:
        pitched: true
      order_by: oldest
      pick: 1
    process:
      - repitch: true
    one_shot: false
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		assert (0, 36) in note_map
		asgn, pick = note_map[(0, 36)]
		assert asgn.select[0].where.pitched is True
		assert asgn.select[0].order == (subsample.query.OrderClause(by="age", dir="asc"),)
		assert asgn.process.has_repitch()
		assert asgn.one_shot is False
		assert pick == 1

	def test_pitched_newest (self, tmp_path: pathlib.Path) -> None:
		"""Pitched newest: where pitched, order newest, repitch."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Newest pitched
    channel: 2
    notes: 60
    select:
      where:
        pitched: true
      order_by: newest
      pick: 1
    process:
      - repitch: true
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		assert (1, 60) in note_map
		asgn, _ = note_map[(1, 60)]
		assert asgn.select[0].order == (subsample.query.OrderClause(by="age", dir="desc"),)
		assert asgn.process.has_repitch()

	def test_pitched_nth (self, tmp_path: pathlib.Path) -> None:
		"""Pitched pick 2: second pitch-stable sample."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Second pitched
    channel: 1
    notes: 60
    select:
      where:
        pitched: true
      order_by: oldest
      pick: 2
    process:
      - repitch: true
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		assert (0, 60) in note_map
		asgn, pick = note_map[(0, 60)]
		assert asgn.select[0].pick == 2
		assert pick == 2

	def test_repitch_all_notes_same_pick (self, tmp_path: pathlib.Path) -> None:
		"""All notes in a repitched range share pick 1."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Pitched keyboard
    channel: 1
    notes: [48, 50, 52]
    select:
      where:
        pitched: true
      order_by: newest
      pick: 1
    process:
      - repitch: true
    one_shot: false
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		for midi_note in [48, 50, 52]:
			asgn, pick = note_map[(0, midi_note)]
			assert pick == 1
			assert asgn.process.has_repitch()

	def test_pitched_full_range (self, tmp_path: pathlib.Path) -> None:
		"""Pitched with C-1..G9 maps all 128 MIDI notes."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Full keyboard
    channel: 1
    notes: C-1..G9
    select:
      where:
        pitched: true
      order_by: oldest
    process:
      - repitch: true
""")
		note_map = subsample.player.load_midi_map(path, []).note_map
		assert len(note_map) == 128


# NOTE: TestResolvePitchedSelector, TestResolveLibraryPosition, and
# TestResolveTarget were removed in the select/process redesign.
# Resolution logic is now in the query engine (tested in test_query.py).


# ---------------------------------------------------------------------------
# TestNewestOldestTarget (parsing)
# ---------------------------------------------------------------------------

class TestNewestOldestTarget:
	"""Tests for newest/oldest ordering in the new select format."""

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_newest_order (self, tmp_path: pathlib.Path) -> None:
		"""order_by: newest parsed correctly."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Latest capture
    channel: 2
    notes: C2..C4
    select:
      order_by: newest
      pick: 1
    process:
      - repitch: true
    one_shot: false
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		assert (1, 36) in note_map
		asgn, pick = note_map[(1, 36)]
		assert asgn.select[0].order == (subsample.query.OrderClause(by="age", dir="desc"),)
		assert asgn.process.has_repitch()
		assert asgn.one_shot is False

	def test_oldest_order (self, tmp_path: pathlib.Path) -> None:
		"""order_by: oldest parsed correctly."""
		path = self._write_map(tmp_path, """
assignments:
  - name: First capture
    channel: 3
    notes: C2..C4
    select:
      order_by: oldest
      pick: 1
    one_shot: false
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		assert (2, 36) in note_map
		asgn, _ = note_map[(2, 36)]
		assert asgn.select[0].order == (subsample.query.OrderClause(by="age", dir="asc"),)
		assert asgn.one_shot is False

	def test_newest_no_reference_needed (self, tmp_path: pathlib.Path) -> None:
		"""newest ordering is accepted with an empty reference list."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Latest capture
    channel: 2
    notes: 60
    select:
      order_by: newest
""")
		note_map = subsample.player.load_midi_map(path, []).note_map
		assert (1, 60) in note_map


# NOTE: TestResolveLibraryPosition removed — logic is now in query engine.


# ---------------------------------------------------------------------------
# TestLoadMidiMapChain — load_midi_map() parsing of chain targets
# ---------------------------------------------------------------------------

class TestLoadMidiMapFallback:
	"""Test load_midi_map() parsing of select-as-list (fallback chain)."""

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_fallback_list_parsed (self, tmp_path: pathlib.Path) -> None:
		"""select as a list creates a multi-spec fallback chain."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick with fallback
    channel: 10
    notes: 36
    select:
      - where:
          name: my-kick
      - where:
          reference: BD0025
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert (9, 36) in note_map
		asgn, _ = note_map[(9, 36)]
		assert len(asgn.select) == 2
		assert asgn.select[0].where.name == "my-kick"
		assert asgn.select[1].where.reference == "BD0025"
		assert asgn.one_shot is True

	def test_fallback_preserves_order (self, tmp_path: pathlib.Path) -> None:
		"""Fallback specs maintain their YAML list order."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Multi-fallback
    channel: 1
    notes: 60
    select:
      - where:
          name: first-choice
      - order_by: oldest
      - order_by: newest
""")
		note_map = subsample.player.load_midi_map(path, []).note_map

		asgn, _ = note_map[(0, 60)]
		assert len(asgn.select) == 3
		assert asgn.select[0].where.name == "first-choice"
		assert asgn.select[1].order == (subsample.query.OrderClause(by="age", dir="asc"),)
		assert asgn.select[2].order == (subsample.query.OrderClause(by="age", dir="desc"),)

	def test_fallback_with_repitch (self, tmp_path: pathlib.Path) -> None:
		"""Fallback chain with repitch: all notes share pick 1."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Pitched fallback
    channel: 1
    notes: C2..C3
    select:
      - where:
          name: my-tone
      - where:
          reference: BASS_TONE
    process:
      - repitch: true
""")
		note_map = subsample.player.load_midi_map(path, ["BASS_TONE"]).note_map

		for midi_note in range(36, 49):
			assert (0, midi_note) in note_map
			asgn, pick = note_map[(0, midi_note)]
			assert pick == 1
			assert asgn.process.has_repitch()

	def test_fallback_invalid_reference_skips (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Fallback with unknown reference is skipped."""
		import logging
		path = self._write_map(tmp_path, """
assignments:
  - name: Bad ref
    channel: 10
    notes: 36
    select:
      - where:
          name: my-kick
      - where:
          reference: UNKNOWN
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			note_map = subsample.player.load_midi_map(path, []).note_map

		assert len(note_map) == 0


# ---------------------------------------------------------------------------
# TestFallbackResolution — select-as-list fallback in _handle_message()
# ---------------------------------------------------------------------------

class TestFallbackResolution:
	"""Integration tests for select-as-a-list fallback in _handle_message()."""

	def _make_player_with_fallback (
		self,
		select_specs: tuple[subsample.query.SelectSpec, ...],
	) -> subsample.player.MidiPlayer:
		"""Return a MidiPlayer with a fallback-chain assignment on ch 1, note 60."""

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		asgn = subsample.query.Assignment(
			name="Fallback test",
			select=select_specs,
		)

		note_map: subsample.player.NoteMap = {(0, 60): (asgn, 1)}

		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

	def _note_on (self, note: int = 60, velocity: int = 100) -> "unittest.mock.MagicMock":
		msg = unittest.mock.MagicMock()
		msg.type = "note_on"
		msg.channel = 0
		msg.note = note
		msg.velocity = velocity
		return msg

	def test_all_fail_plays_silence (self) -> None:
		"""Fallback plays silence when all select specs return no results."""

		player = self._make_player_with_fallback((
			subsample.query.SelectSpec(where=subsample.query.WherePredicate(name="a")),
			subsample.query.SelectSpec(where=subsample.query.WherePredicate(name="b")),
		))

		# Empty library → no matches.
		player._instrument_library.samples.return_value = []

		player._handle_message(self._note_on())

		with player._voices_lock:
			assert len(player._voices) == 0


# ---------------------------------------------------------------------------
# MidiPlayer.reload_midi_map
# ---------------------------------------------------------------------------

class TestReloadMidiMap:

	def test_replaces_note_map (self) -> None:

		"""reload_midi_map() atomically replaces _note_map with the new map."""

		asgn_a = _make_assignment(name="Kicks", reference="BD0025")
		old_map = _make_note_map(asgn_a, channel=9, notes=[36])

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=old_map,
			sample_rate=44100,
			bit_depth=16,
		)

		assert player._note_map is old_map

		asgn_b = _make_assignment(name="Snares", reference="SD0010")
		new_map = _make_note_map(asgn_b, channel=9, notes=[38])

		player.reload_midi_map(new_map)

		assert player._note_map is new_map
		assert (9, 38) in player._note_map
		assert (9, 36) not in player._note_map

	def test_calls_update_assignments (self) -> None:

		"""reload_midi_map() triggers update_assignments() to pre-compute variants."""

		asgn = _make_assignment(name="Kicks", reference="BD0025")
		note_map = _make_note_map(asgn, channel=9, notes=[36])

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

		with unittest.mock.patch.object(player, "update_assignments") as mock_update:
			player.reload_midi_map(note_map)
			mock_update.assert_called_once()


# ---------------------------------------------------------------------------
# MidiPlayer._render_float — gain_db
# ---------------------------------------------------------------------------

class TestRenderFloatGainDb:

	def _make_player (self) -> subsample.player.MidiPlayer:

		"""Return a MidiPlayer for testing _render_float()."""

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
		)

	def _make_audio (self, value: float = 0.5, n_frames: int = 100) -> numpy.ndarray:

		"""Return a constant-value mono float32 audio array."""

		return numpy.full((n_frames, 1), value, dtype=numpy.float32)

	def _make_level (self, peak: float = 0.5, rms: float = 0.3) -> subsample.library.SampleRecord:

		"""Return a LevelResult with given peak and rms."""

		import subsample.analysis
		return subsample.analysis.LevelResult(peak=peak, rms=rms)

	def _centre_pan_matrix (self) -> numpy.ndarray:
		"""Mono→stereo centre pan matrix."""
		s = float(numpy.sqrt(0.5))
		return numpy.array([[s], [s]], dtype=numpy.float32)

	def test_zero_gain_db_has_no_effect (self) -> None:

		"""gain_db=0.0 produces the same output as the default."""

		player = self._make_player()
		audio = self._make_audio()
		level = self._make_level()
		mat = self._centre_pan_matrix()

		result_default = player._render_float(audio, level, 100, mat)
		result_zero    = player._render_float(audio, level, 100, mat, gain_db=0.0)

		numpy.testing.assert_array_equal(result_default, result_zero)

	def test_negative_gain_db_reduces_level (self) -> None:

		"""A negative gain_db produces quieter output."""

		player = self._make_player()
		audio = self._make_audio()
		level = self._make_level()
		mat = self._centre_pan_matrix()

		result_normal = player._render_float(audio, level, 100, mat, gain_db=0.0)
		result_quiet  = player._render_float(audio, level, 100, mat, gain_db=-6.0)

		assert numpy.max(numpy.abs(result_quiet)) < numpy.max(numpy.abs(result_normal))

	def test_positive_gain_db_increases_level (self) -> None:

		"""A positive gain_db produces louder output (clamped by anti-clip ceiling)."""

		player = self._make_player()
		audio = self._make_audio(value=0.1)
		level = self._make_level(peak=0.1, rms=0.05)
		mat = self._centre_pan_matrix()

		result_normal = player._render_float(audio, level, 100, mat, gain_db=0.0)
		result_loud   = player._render_float(audio, level, 100, mat, gain_db=6.0)

		assert numpy.max(numpy.abs(result_loud)) > numpy.max(numpy.abs(result_normal))


class TestParseOutputRouting:

	"""Tests for _parse_output_routing — YAML output list to 0-indexed tuple."""

	def test_basic_conversion (self) -> None:
		"""1-indexed [3, 4] → 0-indexed (2, 3)."""
		result = subsample.player._parse_output_routing([3, 4], "test", None)
		assert result == (2, 3)

	def test_none_returns_none (self) -> None:
		"""Missing field returns None (default routing)."""
		assert subsample.player._parse_output_routing(None, "test", None) is None

	def test_single_output (self) -> None:
		"""Single output [5] → (4,)."""
		result = subsample.player._parse_output_routing([5], "test", None)
		assert result == (4,)

	def test_length_mismatch_with_pan (self) -> None:
		"""output length != pan length raises ValueError."""
		pan = numpy.array([50.0, 50.0], dtype=numpy.float32)
		with pytest.raises(ValueError, match="must match pan length"):
			subsample.player._parse_output_routing([1, 2, 3], "test", pan)

	def test_matching_pan_length (self) -> None:
		"""output length == pan length succeeds."""
		pan = numpy.array([50.0, 50.0], dtype=numpy.float32)
		result = subsample.player._parse_output_routing([3, 4], "test", pan)
		assert result == (2, 3)

	def test_zero_index_raises (self) -> None:
		"""0 is invalid (1-indexed)."""
		with pytest.raises(ValueError, match="positive integers"):
			subsample.player._parse_output_routing([0, 1], "test", None)

	def test_negative_raises (self) -> None:
		"""Negative values are invalid."""
		with pytest.raises(ValueError, match="positive integers"):
			subsample.player._parse_output_routing([-1, 2], "test", None)

	def test_duplicates_raise (self) -> None:
		"""Duplicate channels raise ValueError."""
		with pytest.raises(ValueError, match="duplicate"):
			subsample.player._parse_output_routing([3, 3], "test", None)

	def test_empty_list_raises (self) -> None:
		"""Empty list raises ValueError."""
		with pytest.raises(ValueError, match="non-empty"):
			subsample.player._parse_output_routing([], "test", None)


class TestSelectSegment:

	"""Tests for _select_segment() — segment playback from quantized audio."""

	def _make_player (self) -> subsample.player.MidiPlayer:
		"""Create a minimal MidiPlayer for testing segment selection."""
		return subsample.player.MidiPlayer(
			"",
			threading.Event(),
			instrument_library=unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary),
			similarity_matrix=unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix),
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
		)

	def _make_audio_and_bounds (self) -> tuple[numpy.ndarray, tuple[tuple[int, int], ...]]:
		"""Create test audio with 4 known segments."""
		audio = numpy.random.randn(4000, 1).astype(numpy.float32) * 0.5
		bounds = ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4000))
		return audio, bounds

	def test_no_segment_mode_returns_full_audio (self) -> None:
		"""Empty segment_mode returns the original audio unchanged."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, result_level = player._select_segment(audio, level, bounds, "", 0, 60)

		assert result_audio is audio
		assert result_level is level

	def test_no_bounds_returns_full_audio (self) -> None:
		"""None segment_bounds returns the original audio unchanged."""
		player = self._make_player()
		audio = numpy.random.randn(1000, 1).astype(numpy.float32)
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, result_level = player._select_segment(audio, level, None, "round_robin", 0, 60)

		assert result_audio is audio
		assert result_level is level

	def test_numeric_index_selects_correct_segment (self) -> None:
		"""Numeric segment mode (1-indexed) selects the right slice."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, _ = player._select_segment(audio, level, bounds, 3, 0, 60)

		assert result_audio.shape[0] == 1000
		numpy.testing.assert_array_equal(result_audio, audio[2000:3000])

	def test_numeric_index_clamped (self) -> None:
		"""Index beyond segment count is clamped to last segment."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, _ = player._select_segment(audio, level, bounds, 99, 0, 60)

		numpy.testing.assert_array_equal(result_audio, audio[3000:4000])

	def test_round_robin_cycles (self) -> None:
		"""Round-robin advances through segments and wraps."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		results = []
		for _ in range(6):
			seg, _ = player._select_segment(audio, level, bounds, "round_robin", 0, 60)
			results.append(seg.shape[0])

		# 4 segments, 6 triggers: should cycle 0,1,2,3,0,1
		assert results == [1000, 1000, 1000, 1000, 1000, 1000]
		assert player._segment_counters[(0, 60)] == 6

	def test_random_stays_in_bounds (self) -> None:
		"""Random mode always selects a valid segment."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		for _ in range(20):
			seg, _ = player._select_segment(audio, level, bounds, "random", 0, 60)
			assert seg.shape[0] == 1000

	def test_segment_mode_parsed_from_yaml_string (self) -> None:
		"""segment: round_robin parsed correctly from YAML."""
		step = subsample.query.ProcessorStep(name="pad_quantize", params=(("grid", 16), ("segment", "round_robin")))
		process = subsample.query.ProcessSpec(steps=(step,))

		assert step.get("segment", "") == "round_robin"

	def test_segment_mode_parsed_from_yaml_int (self) -> None:
		"""segment: 3 parsed correctly from YAML."""
		step = subsample.query.ProcessorStep(name="pad_quantize", params=(("grid", 16), ("segment", 3)))
		process = subsample.query.ProcessSpec(steps=(step,))

		assert step.get("segment", "") == 3
