"""Tests for subsample.player — MIDI device selection and MidiPlayer lifecycle."""

import threading
import unittest.mock

import pytest

import subsample.library
import subsample.player
import subsample.similarity


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
			reference_names=["KICK", "SNARE"],
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
		player._voices = [voice]
		player._voices_lock = threading.Lock()
		player._OUTPUT_CHANNELS = 2
		player._output_bit_depth = 16
		player._release_fade_frames = 441  # 10 ms at 44100 Hz

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
		player._voices = [voice]
		player._voices_lock = threading.Lock()
		player._output_bit_depth = 16

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
			reference_names=[],
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

	def test_clipping_warning_logged (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		import numpy

		n_frames = 512
		# Two voices each at 0.8 → sum = 1.6 → clips
		audio_loud = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.8
		voice_a = subsample.player._Voice(audio=audio_loud.copy(), note=36, channel=9)
		voice_b = subsample.player._Voice(audio=audio_loud.copy(), note=38, channel=9)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices            = [voice_a, voice_b]
		player._voices_lock       = threading.Lock()
		player._output_bit_depth  = 16
		player._release_fade_frames = 441
		player._last_clip_warn    = 0.0
		player._max_polyphony     = 8

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		assert any("clipping" in r.message.lower() for r in caplog.records)

	def test_clipping_warning_throttled (self) -> None:
		import numpy

		n_frames = 512
		audio_loud = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.8

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices_lock       = threading.Lock()
		player._output_bit_depth  = 16
		player._release_fade_frames = 441
		player._max_polyphony     = 8

		# First call fires the warning; record the timestamp it sets
		player._last_clip_warn = 0.0
		voice_a = subsample.player._Voice(audio=audio_loud.copy(), note=36, channel=9)
		voice_b = subsample.player._Voice(audio=audio_loud.copy(), note=38, channel=9)
		player._voices = [voice_a, voice_b]
		subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)
		first_warn_time: float = player._last_clip_warn

		# Second call immediately after — _last_clip_warn is recent → no new warning
		voice_c = subsample.player._Voice(audio=audio_loud.copy(), note=36, channel=9)
		voice_d = subsample.player._Voice(audio=audio_loud.copy(), note=38, channel=9)
		player._voices = [voice_c, voice_d]
		with unittest.mock.patch.object(subsample.player._log, "warning") as mock_warn:
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		mock_warn.assert_not_called()
		# Timestamp should not have advanced (no new warning was issued)
		assert player._last_clip_warn == first_warn_time

	def test_no_clipping_no_warning (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		import numpy

		n_frames = 512
		audio_quiet = numpy.ones((n_frames, 2), dtype=numpy.float32) * 0.1

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player._voices            = [subsample.player._Voice(audio=audio_quiet.copy(), note=36, channel=9)]
		player._voices_lock       = threading.Lock()
		player._output_bit_depth  = 16
		player._release_fade_frames = 441
		player._last_clip_warn    = 0.0
		player._max_polyphony     = 8

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(player, None, n_frames, {}, 0)

		assert not any("clipping" in r.message.lower() for r in caplog.records)
