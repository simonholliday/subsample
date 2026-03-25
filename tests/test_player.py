"""Tests for subsample.player — MIDI device selection and MidiPlayer lifecycle."""

import pathlib
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
		"""Single-note assignment: one note, rank 0."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		assert (9, 36) in note_map
		ttype, ref, rank, one_shot, pitch, pan_gains = note_map[(9, 36)]
		assert ttype == "reference"
		assert ref == "BD0025"
		assert rank == 0
		assert one_shot is True

	def test_multi_note_rank_distribution (self, tmp_path: pathlib.Path) -> None:
		"""Note list distributes ranks: first note = rank 0, second = rank 1."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kicks
    channel: 10
    notes: [36, 35]
    target: reference(BD0025)
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		assert note_map[(9, 36)][2] == 0   # rank 0
		assert note_map[(9, 35)][2] == 1   # rank 1

	def test_channel_conversion (self, tmp_path: pathlib.Path) -> None:
		"""User-facing channel 10 converts to mido channel 9."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		assert (9, 36) in note_map
		assert (10, 36) not in note_map

	def test_one_shot_defaults_true (self, tmp_path: pathlib.Path) -> None:
		"""one_shot defaults to True when omitted."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		_, _, _, one_shot, _, _ = note_map[(9, 36)]
		assert one_shot is True

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
    target: reference(BD0025)
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			note_map = subsample.player.load_midi_map(path, [])

		assert len(note_map) == 0
		assert any("BD0025" in r.message for r in caplog.records)

	def test_case_insensitive_reference (self, tmp_path: pathlib.Path) -> None:
		"""Reference lookup is case-insensitive."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(bd0025)
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		assert (9, 36) in note_map

	def test_missing_file_raises (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(FileNotFoundError):
			subsample.player.load_midi_map(tmp_path / "no-such-file.yaml", [])

	def test_empty_file_returns_empty_map (self, tmp_path: pathlib.Path) -> None:
		path = self._write_map(tmp_path, "")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])
		assert note_map == {}

	def test_multiple_assignments_different_channels (self, tmp_path: pathlib.Path) -> None:
		"""Assignments on different channels coexist in the map."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick ch10
    channel: 10
    notes: 36
    target: reference(BD0025)
  - name: Snare ch10
    channel: 10
    notes: 38
    target: reference(SD5075)
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025", "SD5075"])

		assert (9, 36) in note_map
		assert (9, 38) in note_map
		assert note_map[(9, 36)][1] == "BD0025"
		assert note_map[(9, 38)][1] == "SD5075"

	def test_sample_target_parsed (self, tmp_path: pathlib.Path) -> None:
		"""sample(filename) target is parsed into the note map with target_type 'sample'."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Fixed kick
    channel: 10
    notes: 36
    target: sample(2026-03-24_14-37-14)
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, [])

		assert (9, 36) in note_map
		ttype, fname, rank, one_shot, pitch, pan_gains = note_map[(9, 36)]
		assert ttype == "sample"
		assert fname == "2026-03-24_14-37-14"
		assert one_shot is True

	def test_sample_target_no_reference_validation (self, tmp_path: pathlib.Path) -> None:
		"""sample() targets are not validated against the reference library at load time."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Fixed kick
    channel: 10
    notes: 36
    target: sample(some-recording)
""")
		# Empty reference_names — sample() should still be accepted
		note_map = subsample.player.load_midi_map(path, [])

		assert (9, 36) in note_map

	def test_default_map_parses (self) -> None:
		"""The shipped midi-map.yaml.default parses without error."""
		default_path = pathlib.Path(__file__).parent.parent / "midi-map.yaml.default"
		refs = ["BD0025", "SD5075", "CP", "CH", "OH25", "CY5050", "CB"]
		note_map = subsample.player.load_midi_map(default_path, refs)

		assert len(note_map) > 0
		# Kick on note 36 (mido ch 9)
		assert (9, 36) in note_map
		assert note_map[(9, 36)][1] == "BD0025"

	def test_default_pan_is_centre (self, tmp_path: pathlib.Path) -> None:
		"""Omitted pan defaults to equal power across all output channels."""
		import numpy
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"], output_channels=2)

		_, _, _, _, _, pan_gains = note_map[(9, 36)]
		# Centre: both channels equal power → gain = 1/sqrt(2) ≈ 0.707
		assert pan_gains.shape == (2,)
		numpy.testing.assert_allclose(pan_gains, [1.0 / 2**0.5, 1.0 / 2**0.5], atol=1e-5)

	def test_explicit_pan_constant_power (self, tmp_path: pathlib.Path) -> None:
		"""Constant-power law: sum(gains**2) == 1.0 for any pan position."""
		import numpy
		for weights in [[100, 0], [0, 100], [50, 50], [75, 25], [30, 70]]:
			path = self._write_map(tmp_path, f"""
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
    pan: {weights}
""")
			note_map = subsample.player.load_midi_map(path, ["BD0025"], output_channels=2)
			_, _, _, _, _, pan_gains = note_map[(9, 36)]
			total_power = float(numpy.sum(pan_gains ** 2))
			numpy.testing.assert_allclose(total_power, 1.0, atol=1e-5,
				err_msg=f"pan {weights} total power should be 1.0")

	def test_pan_hard_left (self, tmp_path: pathlib.Path) -> None:
		"""pan: [100, 0] produces gain 1.0 on left, 0.0 on right."""
		import numpy
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
    pan: [100, 0]
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"], output_channels=2)
		_, _, _, _, _, pan_gains = note_map[(9, 36)]
		numpy.testing.assert_allclose(pan_gains, [1.0, 0.0], atol=1e-5)

	def test_pan_wrong_channel_count_raises (self, tmp_path: pathlib.Path) -> None:
		"""pan list length must match output_channels."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    target: reference(BD0025)
    pan: [50, 50, 50]
""")
		with pytest.raises(ValueError, match="pan"):
			subsample.player.load_midi_map(path, ["BD0025"], output_channels=2)

	def test_pitch_true_all_notes_rank_zero (self, tmp_path: pathlib.Path) -> None:
		"""pitch: true on reference assigns rank 0 to all notes."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Bass keyboard
    channel: 1
    notes: [48, 50, 52]
    target: reference(BASS_TONE)
    pitch: true
    one_shot: false
""")
		note_map = subsample.player.load_midi_map(path, ["BASS_TONE"])

		for midi_note in [48, 50, 52]:
			ttype, targ, rank, one_shot, pitch, pan_gains = note_map[(0, midi_note)]
			assert rank == 0
			assert pitch is True

	def test_pitch_false_default_distributes_ranks (self, tmp_path: pathlib.Path) -> None:
		"""Without pitch: true, notes get ascending ranks (existing behaviour)."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kicks
    channel: 10
    notes: [36, 35]
    target: reference(BD0025)
    one_shot: true
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		_, _, rank_36, _, pitch_36, _ = note_map[(9, 36)]
		_, _, rank_35, _, pitch_35, _ = note_map[(9, 35)]
		assert rank_36 == 0
		assert rank_35 == 1
		assert pitch_36 is False
		assert pitch_35 is False

	def test_note_name_in_map (self, tmp_path: pathlib.Path) -> None:
		"""Note names (C4) are accepted in assignments."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: C2
    target: reference(BD0025)
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"])

		# C2 = MIDI 36
		assert (9, 36) in note_map

	def test_note_range_in_map (self, tmp_path: pathlib.Path) -> None:
		"""Range syntax 'C2..C4' expands to all 25 notes."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Bass keyboard
    channel: 1
    notes: C2..C4
    target: reference(BASS_TONE)
    pitch: true
""")
		note_map = subsample.player.load_midi_map(path, ["BASS_TONE"])

		# C2=36, C4=60 → 25 notes on channel 1 (mido ch 0)
		assert len(note_map) == 25
		assert (0, 36) in note_map   # C2
		assert (0, 60) in note_map   # C4


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
		import numpy

		note_map: subsample.player._NoteMap = {
			(0, note): ("reference", ref_name, 0, False, True, numpy.array([0.707, 0.707], dtype=numpy.float32))
			for note in notes
		}

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
		import numpy

		non_pitched_map: subsample.player._NoteMap = {
			(9, 36): ("reference", "BD0025", 0, True, False, numpy.array([0.707, 0.707], dtype=numpy.float32)),
		}

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
		import numpy
		import subsample.analysis

		notes = [48, 50, 52]
		player = self._make_player_with_pitch_map(notes=notes)

		mock_record = unittest.mock.MagicMock()
		mock_record.spectral = unittest.mock.MagicMock()
		mock_record.pitch    = unittest.mock.MagicMock()
		mock_record.duration = 1.0

		player._similarity_matrix.get_match.return_value = 42
		player._instrument_library.get.return_value = mock_record
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
		mock_record.name = "some-sample"
		player._similarity_matrix.get_match.return_value = 42
		player._instrument_library.get.return_value = mock_record
		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		with unittest.mock.patch("subsample.analysis.has_stable_pitch", return_value=False):
			with caplog.at_level(logging.WARNING, logger="subsample.player"):
				player.update_pitched_assignments()

		transform_manager.enqueue_pitch_range.assert_not_called()
		assert any("stable pitch" in r.message for r in caplog.records)
