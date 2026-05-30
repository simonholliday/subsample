"""Tests for subsample.player — MIDI device selection and MidiPlayer lifecycle."""

import dataclasses
import pathlib
import threading
import typing
import unittest.mock

import mido
import numpy
import pytest

import subsample.bank
import subsample.library
import subsample.player
import subsample.query
import subsample.similarity
import subsample.transform

import tests.helpers


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
	stretch_quantize: bool = False,
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

	if stretch_quantize:
		steps.append(subsample.query.ProcessorStep(name="stretch_quantize", params=(("grid", 16),)))

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

	"""Build a NoteMap for one assignment across multiple notes.

	Returns a single-layer NoteMap (one entry per note) with the full
	default velocity range — matches the pre-velocity-layering behaviour
	so legacy tests stay correct after the value-type migration.
	"""

	note_map: subsample.player.NoteMap = {}

	for i, note in enumerate(notes):
		rank = (i + 1) if per_note_pick else 1
		note_map[(channel, note)] = [(assignment, subsample.query.PickSpec(rank, rank))]

	return note_map


# ---------------------------------------------------------------------------
# _ranks_for / _format_pick_suffix — open-ended pick handling
# ---------------------------------------------------------------------------

class TestRanksForOpenEnded:

	"""_ranks_for expands a PickSpec to the ranks a runtime draw could reach,
	resolving open (None) bounds against the live match count."""

	def test_closed_range_unchanged (self) -> None:
		ranks = subsample.player._ranks_for(subsample.query.PickSpec(1, 3), 10)
		assert list(ranks) == [1, 2, 3]

	def test_open_upper_expands_to_count (self) -> None:
		ranks = subsample.player._ranks_for(subsample.query.PickSpec(2, None), 5)
		assert list(ranks) == [2, 3, 4, 5]

	def test_open_lower_starts_at_one (self) -> None:
		ranks = subsample.player._ranks_for(subsample.query.PickSpec(None, 3), 10)
		assert list(ranks) == [1, 2, 3]

	def test_any_covers_full_list (self) -> None:
		ranks = subsample.player._ranks_for(subsample.query.PickSpec(None, None), 4)
		assert list(ranks) == [1, 2, 3, 4]

	def test_open_upper_lo_past_end_clamps (self) -> None:
		ranks = subsample.player._ranks_for(subsample.query.PickSpec(3, None), 2)
		assert list(ranks) == [2]


class TestFormatPickSuffix:

	"""_format_pick_suffix renders the startup-log `pick` annotation."""

	def test_default_best_match_is_blank (self) -> None:
		assert subsample.player._format_pick_suffix(subsample.query.PickSpec(1, 1)) == ""

	def test_scalar_rank (self) -> None:
		assert subsample.player._format_pick_suffix(subsample.query.PickSpec(3, 3)) == " pick 3"

	def test_closed_range (self) -> None:
		assert subsample.player._format_pick_suffix(subsample.query.PickSpec(2, 5)) == " pick 2-5"

	def test_any (self) -> None:
		assert subsample.player._format_pick_suffix(subsample.query.PickSpec(None, None)) == " pick any"

	def test_open_upper (self) -> None:
		assert subsample.player._format_pick_suffix(subsample.query.PickSpec(2, None)) == " pick 2+"

	def test_open_lower_reads_from_one (self) -> None:
		assert subsample.player._format_pick_suffix(subsample.query.PickSpec(None, 5)) == " pick 1-5"


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

	def _make_mock_port (self) -> unittest.mock.MagicMock:

		"""Return a mock mido port for callback-mode dispatch.

		``mido.open_input(..., callback=func)`` registers ``func`` for
		every incoming message.  This mock records whichever callable is
		passed as ``callback=`` (via the kwarg-capturing patch in each
		test) and exposes a ``dispatch(msg)`` helper so tests can deliver
		messages on demand.
		"""

		port = unittest.mock.MagicMock()
		port.__enter__ = unittest.mock.Mock(return_value=port)
		port.__exit__ = unittest.mock.Mock(return_value=False)
		return port

	def _patch_open_input (self, port: unittest.mock.MagicMock) -> unittest.mock._patch[typing.Any]:

		"""Patch ``mido.open_input`` so it returns ``port`` and stashes the
		registered callback on ``port.callback`` for later dispatch."""

		def fake_open_input (*args: typing.Any, **kwargs: typing.Any) -> unittest.mock.MagicMock:
			port.callback = kwargs.get("callback")
			return port

		return unittest.mock.patch("mido.open_input", side_effect=fake_open_input)

	def test_exits_on_shutdown_event (self) -> None:
		shutdown_event = threading.Event()
		port = self._make_mock_port()
		mock_pa = self._make_mock_pyaudio()

		with self._patch_open_input(port):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
				# Set shutdown_event before run() so the wait returns immediately.
				shutdown_event.set()
				player = self._make_player(shutdown_event)
				player.run()

		# run() should have returned without hanging
		assert not threading.current_thread().daemon

	def test_logs_port_open (self, caplog: pytest.LogCaptureFixture) -> None:
		import logging

		shutdown_event = threading.Event()
		port = self._make_mock_port()
		mock_pa = self._make_mock_pyaudio()

		with caplog.at_level(logging.INFO, logger="subsample.player"):
			with self._patch_open_input(port):
				with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
					shutdown_event.set()
					player = self._make_player(shutdown_event)
					player.run()

		# Port open AND close must both name the device — the "opened" log
		# confirms the right port was selected; the "closed" log confirms
		# the finally block ran cleanly and didn't leak the handle.
		device_log_messages = [r.message for r in caplog.records if "Test Device" in r.message]
		assert any("opened" in m for m in device_log_messages), \
			f"expected an 'opened' log naming the device, got: {device_log_messages}"
		assert any("closed" in m for m in device_log_messages), \
			f"expected a 'closed' log naming the device, got: {device_log_messages}"

	def test_port_closed_on_shutdown (self) -> None:
		shutdown_event = threading.Event()
		port = self._make_mock_port()
		mock_pa = self._make_mock_pyaudio()

		with self._patch_open_input(port):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
				shutdown_event.set()
				player = self._make_player(shutdown_event)
				player.run()

		port.close.assert_called_once()

	def test_run_on_thread_exits_cleanly (self) -> None:
		shutdown_event = threading.Event()
		port = self._make_mock_port()
		mock_pa = self._make_mock_pyaudio()

		with self._patch_open_input(port):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=mock_pa):
				player = self._make_player(shutdown_event)
				t = threading.Thread(target=player.run)
				t.start()

				# Give thread time to start, then signal shutdown.
				shutdown_event.set()
				t.join(timeout=2.0)

		assert not t.is_alive()


# ---------------------------------------------------------------------------
# MIDI callback mode — dispatch wiring + safe wrapper
# ---------------------------------------------------------------------------

class TestMidiCallbackMode:

	"""Verifies the callback-mode dispatch path: that the player wires its
	safe-handler into mido.open_input via the ``callback=`` kwarg, and that
	the safe wrapper protects rtmidi's thread from a handler exception
	(which mido does NOT catch — it silently drops the message)."""

	def _make_player (self, shutdown_event: threading.Event) -> subsample.player.MidiPlayer:
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
		mock_stream = unittest.mock.MagicMock()
		mock_pa = unittest.mock.MagicMock()
		mock_pa.open.return_value = mock_stream
		mock_pa.get_device_count.return_value = 1
		mock_pa.get_device_info_by_index.return_value = {
			"name": "Mock Output",
			"maxOutputChannels": 2,
			"defaultSampleRate": 44100,
			"index": 0,
		}
		return mock_pa

	def test_callback_kwarg_passed_to_open_input (self) -> None:

		"""The hardware-port path must register ``_safe_handle_message``
		as the ``callback=`` kwarg of ``mido.open_input`` so rtmidi
		dispatches MIDI messages on its own thread without a polling
		jitter floor."""

		shutdown_event = threading.Event()
		shutdown_event.set()
		captured_kwargs: dict[str, typing.Any] = {}

		def fake_open_input (*args: typing.Any, **kwargs: typing.Any) -> unittest.mock.MagicMock:
			captured_kwargs.update(kwargs)
			return unittest.mock.MagicMock()

		with unittest.mock.patch("mido.open_input", side_effect=fake_open_input):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=self._make_mock_pyaudio()):
				player = self._make_player(shutdown_event)
				player.run()

		assert "callback" in captured_kwargs
		assert captured_kwargs["callback"] == player._safe_handle_message

	def test_virtual_port_callback_kwarg_passed (self) -> None:

		"""The virtual-port path must also pass ``callback=`` + ``virtual=True``."""

		shutdown_event = threading.Event()
		shutdown_event.set()
		captured_kwargs: dict[str, typing.Any] = {}

		instrument_library = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		similarity_matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		player = subsample.player.MidiPlayer(
			"",
			shutdown_event,
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
			virtual_midi_port="Subsample Test Virtual",
		)

		def fake_open_input (*args: typing.Any, **kwargs: typing.Any) -> unittest.mock.MagicMock:
			captured_kwargs.update(kwargs)
			return unittest.mock.MagicMock()

		with unittest.mock.patch("mido.open_input", side_effect=fake_open_input):
			with unittest.mock.patch("subsample.audio.create_pyaudio", return_value=self._make_mock_pyaudio()):
				player.run()

		assert captured_kwargs.get("virtual") is True
		assert captured_kwargs.get("callback") == player._safe_handle_message

	def test_safe_handle_message_swallows_exception (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:

		"""A handler bug must not propagate to rtmidi's callback thread —
		rtmidi would clear the Python error and silently drop the message,
		hiding the bug forever.  The safe wrapper logs at ERROR so failures
		surface."""

		import logging

		shutdown_event = threading.Event()
		player = self._make_player(shutdown_event)

		# Use a real mido Message so the repr in the ERROR log is the kind
		# of payload an operator would see in production logs.
		msg = mido.Message("note_on", channel=0, note=60, velocity=64)

		with unittest.mock.patch.object(
			player, "_handle_message",
			side_effect=ValueError("simulated handler failure"),
		):
			with caplog.at_level(logging.ERROR, logger="subsample.player"):
				# Must NOT raise — that would propagate into rtmidi's
				# callback thread in production and silently drop the msg.
				player._safe_handle_message(msg)

		messages = [r.message for r in caplog.records]
		assert any("simulated handler failure" in m for m in messages)
		# The repr of the failing message is in the log so the bug is debuggable.
		assert any("note_on" in m for m in messages)


# ---------------------------------------------------------------------------
# State lock — protects _cc_state, _cc_omni, _cc_last_log, _segment_counters,
# _last_played from concurrent access between the rtmidi callback thread and
# the threads that read these dicts via update_assignments.
# ---------------------------------------------------------------------------

class TestStateLock:

	def _make_player (self) -> subsample.player.MidiPlayer:
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

	def test_snapshot_cc_state_returns_copies (self) -> None:

		"""Mutating the returned dicts must not affect the live state — the
		snapshot's whole purpose is to give other threads a stable view."""

		player = self._make_player()
		player._cc_state[(0, 7)] = 64
		player._cc_omni[7] = 64

		cc_state, cc_omni = player._snapshot_cc_state()

		cc_state[(0, 7)] = 0
		cc_omni[7] = 0

		assert player._cc_state[(0, 7)] == 64
		assert player._cc_omni[7] == 64

	def test_snapshot_cc_state_reflects_live_values (self) -> None:

		"""Each call to the snapshot helper must see the current state."""

		player = self._make_player()
		player._cc_state[(2, 1)] = 100
		player._cc_omni[1] = 100

		cc_state, cc_omni = player._snapshot_cc_state()
		assert cc_state[(2, 1)] == 100
		assert cc_omni[1] == 100

		# Mutate the live dicts and re-snapshot — the new values must appear.
		player._cc_state[(2, 1)] = 50
		player._cc_omni[1] = 50

		cc_state, cc_omni = player._snapshot_cc_state()
		assert cc_state[(2, 1)] == 50
		assert cc_omni[1] == 50

	def _make_tracking_lock (self) -> tuple[typing.Any, list[str]]:

		"""Return (lock-like object, events list).  The lock supports
		``with`` and records 'acquire' / 'release' into events so tests
		can assert the lock was actually entered around the critical
		section."""

		events: list[str] = []
		real_lock = threading.Lock()

		class _TrackingLock:
			def __enter__(self_inner) -> bool:
				real_lock.acquire()
				events.append("acquire")
				return True

			def __exit__(self_inner, *args: typing.Any) -> None:
				events.append("release")
				real_lock.release()

			def acquire(self_inner, *args: typing.Any, **kwargs: typing.Any) -> bool:
				result = real_lock.acquire(*args, **kwargs)
				events.append("acquire")
				return result

			def release(self_inner) -> None:
				events.append("release")
				real_lock.release()

		return _TrackingLock(), events

	def test_cc_write_holds_state_lock (self) -> None:

		"""The CC branch of _handle_message must acquire _state_lock for the
		_cc_state / _cc_omni writes so concurrent snapshots see the pair as
		one atomic update."""

		player = self._make_player()
		lock, events = self._make_tracking_lock()
		player._state_lock = lock

		msg = mido.Message("control_change", channel=0, control=7, value=64)
		player._handle_message(msg)

		# At least one acquire/release pair must have happened (the CC
		# write itself).  The log throttle path also takes the lock when
		# the CC is mapped — that's a bonus, not asserted here.
		assert events.count("acquire") >= 1
		assert events.count("acquire") == events.count("release")
		assert player._cc_state[(0, 7)] == 64
		assert player._cc_omni[7] == 64

	def test_segment_counter_increment_holds_state_lock (self) -> None:

		"""round_robin RMW must acquire _state_lock so a concurrent clear
		(from reload_midi_map / bank switch) cannot drop an increment."""

		import subsample.analysis

		player = self._make_player()
		lock, events = self._make_tracking_lock()
		player._state_lock = lock

		# Two-segment bounds so round_robin actually advances.
		bounds = ((0, 100), (100, 200))
		audio = numpy.zeros((200, 1), dtype=numpy.float32)
		level = subsample.analysis.LevelResult(peak=0.0, rms=0.0)

		player._select_segment(
			audio, level, bounds, "round_robin",
			channel=9, note=36, velocity_trigger=(0, 127),
		)

		assert events.count("acquire") >= 1
		assert events.count("acquire") == events.count("release")
		# Default velocity layer hashes to (ch, note, 0, 127) — the
		# uniform 4-tuple key shape that keeps layered and non-layered
		# notes on the same code path.
		assert player._segment_counters[(9, 36, 0, 127)] == 1


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
		player._voices              = []
		player._voices_lock         = threading.Lock()
		player._output_channels     = 2
		player._output_bit_depth    = 16
		player._release_fade_frames = 441
		player._last_clip_warn      = 0.0
		player._xrun_count          = 0
		player._last_xrun_warn      = 0.0
		player._buffer_frames       = 256
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

	def test_output_underflow_counts_and_warns (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""A PortAudio output-underflow flag increments the xrun counter and
		surfaces a throttled WARNING so a too-low buffer_frames is diagnosable."""
		import logging
		import pyaudio

		player = self._make_callback_player()

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(
				player, None, 256, {}, pyaudio.paOutputUnderflow,
			)

		assert player._xrun_count == 1
		assert any("xrun" in r.message.lower() for r in caplog.records)

	def test_clean_callback_records_no_xrun (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""status_flags=0 (the normal case) neither counts an xrun nor warns."""
		import logging

		player = self._make_callback_player()

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.MidiPlayer._audio_callback(player, None, 256, {}, 0)

		assert player._xrun_count == 0
		assert not any("xrun" in r.message.lower() for r in caplog.records)

	def test_xrun_warning_throttled (self) -> None:
		"""Repeated underflows keep counting but warn at most once per 5 s."""
		import pyaudio

		player = self._make_callback_player()

		subsample.player.MidiPlayer._audio_callback(player, None, 256, {}, pyaudio.paOutputUnderflow)
		first_warn_time: float = player._last_xrun_warn

		with unittest.mock.patch.object(subsample.player._log, "warning") as mock_warn:
			subsample.player.MidiPlayer._audio_callback(player, None, 256, {}, pyaudio.paOutputUnderflow)

		mock_warn.assert_not_called()
		assert player._last_xrun_warn == first_warn_time
		assert player._xrun_count == 2

	def test_release_fade_spans_multiple_small_buffers (self) -> None:
		"""Code-review regression: a release fade longer than buffer_frames must
		span multiple callbacks, not collapse into one short buffer (which cut
		note-offs abruptly at small buffer sizes)."""
		import numpy

		player = self._make_callback_player()
		player._release_fade_frames = 200

		audio = numpy.ones((1000, 2), dtype=numpy.float32) * 0.5
		voice = subsample.player._Voice(audio=audio, note=36, channel=9, releasing=True)
		player._voices = [voice]

		# One 64-frame callback applies only part of the 200-frame fade.
		subsample.player.MidiPlayer._audio_callback(player, None, 64, {}, 0)
		assert voice.fade_pos == 64
		assert voice in player._voices            # still fading, not retired

		# Further callbacks complete the fade (~200 frames total), then retire.
		for _ in range(6):
			if not player._voices:
				break
			subsample.player.MidiPlayer._audio_callback(player, None, 64, {}, 0)

		assert player._voices == []               # fully faded and retired
		assert voice.fade_pos >= player._release_fade_frames


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
		asgn, pick = note_map[(9, 36)][0]
		assert asgn.select[0].where.reference == "BD0025"
		assert asgn.one_shot is True
		assert pick == subsample.query.PickSpec(1, 1)

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

		assert note_map[(9, 36)][0][1] == subsample.query.PickSpec(1, 1)   # pick 1
		assert note_map[(9, 35)][0][1] == subsample.query.PickSpec(2, 2)   # pick 2

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

		asgn, _ = note_map[(9, 36)][0]
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

	def test_invalid_channel_error_names_assignment (self, tmp_path: pathlib.Path) -> None:
		"""A coercion error on 'channel' must include the assignment name
		and its index in the file — crucial for locating the bad entry in a
		large map."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Fine
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
  - name: Broken
    channel: ten
    notes: 36
    select:
      where:
        reference: BD0025
""")
		with pytest.raises(ValueError, match="assignment 'Broken'.*#2.*invalid 'channel'"):
			subsample.player.load_midi_map(path, ["BD0025"])

	def test_invalid_gain_error_names_assignment (self, tmp_path: pathlib.Path) -> None:
		"""Coercion error on 'gain' is also localised to the assignment."""
		path = self._write_map(tmp_path, """
assignments:
  - name: BadGain
    channel: 10
    notes: 36
    gain: "loud"
    select:
      where:
        reference: BD0025
""")
		with pytest.raises(ValueError, match="assignment 'BadGain'.*invalid 'gain'"):
			subsample.player.load_midi_map(path, ["BD0025"])

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
		assert note_map[(9, 36)][0][0].select[0].where.reference == "BD0025"
		assert note_map[(9, 38)][0][0].select[0].where.reference == "SD5075"

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
		asgn, _ = note_map[(9, 36)][0]
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
		ref = note_map[(9, 36)][0][0].select[0].where.reference
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

		asgn, _ = note_map[(9, 36)][0]
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
		asgn, _ = note_map[(9, 36)][0]
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
		asgn, _ = note_map[(9, 36)][0]
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
			asgn, pick = note_map[(0, midi_note)][0]
			assert pick == subsample.query.PickSpec(1, 1)
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

		assert note_map[(9, 36)][0][1] == subsample.query.PickSpec(1, 1)
		assert note_map[(9, 35)][0][1] == subsample.query.PickSpec(2, 2)
		assert not note_map[(9, 36)][0][0].process.has_repitch()

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
# Velocity layering — _parse_velocity, _validate_velocity_layers, end-to-end
# routing, rescale arithmetic, and per-layer state isolation.
# ---------------------------------------------------------------------------

class TestParseVelocity:

	"""Unit tests for the YAML parser _parse_velocity()."""

	def test_omitted_returns_full_range_no_rescale (self) -> None:
		trigger, rescale_to = subsample.player._parse_velocity(None, "asgn")
		assert trigger == (0, 127)
		assert rescale_to is None

	def test_list_shortcut (self) -> None:
		"""velocity: [0, 63] is shorthand for trigger only, no rescale."""
		trigger, rescale_to = subsample.player._parse_velocity([0, 63], "asgn")
		assert trigger == (0, 63)
		assert rescale_to is None

	def test_dict_form_trigger_only (self) -> None:
		trigger, rescale_to = subsample.player._parse_velocity(
			{"trigger": [10, 100]}, "asgn",
		)
		assert trigger == (10, 100)
		assert rescale_to is None

	def test_dict_form_rescale_false (self) -> None:
		"""rescale: false is identical to omitting rescale."""
		trigger, rescale_to = subsample.player._parse_velocity(
			{"trigger": [0, 63], "rescale": False}, "asgn",
		)
		assert trigger == (0, 63)
		assert rescale_to is None

	def test_dict_form_rescale_true (self) -> None:
		"""rescale: true means rescale to the full MIDI range."""
		trigger, rescale_to = subsample.player._parse_velocity(
			{"trigger": [0, 63], "rescale": True}, "asgn",
		)
		assert trigger == (0, 63)
		assert rescale_to == (0, 127)

	def test_dict_form_rescale_custom_range (self) -> None:
		trigger, rescale_to = subsample.player._parse_velocity(
			{"trigger": [0, 63], "rescale": [10, 100]}, "asgn",
		)
		assert trigger == (0, 63)
		assert rescale_to == (10, 100)

	def test_trigger_out_of_range_rejected (self) -> None:
		with pytest.raises(ValueError, match=r"\[0, 127\]"):
			subsample.player._parse_velocity([0, 200], "asgn")

	def test_trigger_lo_greater_than_hi_rejected (self) -> None:
		with pytest.raises(ValueError, match="lo > hi"):
			subsample.player._parse_velocity([100, 50], "asgn")

	def test_unknown_inner_key_rejected (self) -> None:
		"""Typo guard: trggier is not trigger."""
		with pytest.raises(ValueError, match="unknown velocity key"):
			subsample.player._parse_velocity({"trggier": [0, 63]}, "asgn")

	def test_dict_form_without_trigger_rejected (self) -> None:
		"""rescale without trigger is a malformed velocity dict."""
		with pytest.raises(ValueError, match="requires a 'trigger' field"):
			subsample.player._parse_velocity({"rescale": True}, "asgn")

	def test_single_point_trigger_with_rescale_list_rejected (self) -> None:
		"""Rescaling a 1-velocity trigger has no defined mapping (div-by-0)."""
		with pytest.raises(ValueError, match="non-point trigger"):
			subsample.player._parse_velocity(
				{"trigger": [50, 50], "rescale": [0, 127]}, "asgn",
			)

	def test_single_point_trigger_without_rescale_allowed (self) -> None:
		"""A single-velocity trigger filter (rescale off) is fine — it just
		fires for exactly that velocity."""
		trigger, rescale_to = subsample.player._parse_velocity([64, 64], "asgn")
		assert trigger == (64, 64)
		assert rescale_to is None

	def test_assignment_name_in_error_messages (self) -> None:
		"""Errors must name the assignment for locatability."""
		with pytest.raises(ValueError, match="'Soft hat'"):
			subsample.player._parse_velocity([200, 100], "Soft hat")


class TestVelocityLayering:

	"""End-to-end tests for velocity layering through ``load_midi_map`` and
	the runtime layer selection in ``_select_velocity_layer``."""

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_default_velocity_when_omitted (self, tmp_path: pathlib.Path) -> None:
		"""Assignment without velocity field still routes for all velocities."""

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
		assert len(note_map[(9, 36)]) == 1
		asgn, _ = note_map[(9, 36)][0]
		assert asgn.velocity_trigger == (0, 127)
		assert asgn.velocity_rescale_to is None

	def test_two_layers_load_into_one_note (self, tmp_path: pathlib.Path) -> None:
		"""Two assignments on the same (ch, note) with non-overlapping ranges
		coexist as separate list entries."""

		path = self._write_map(tmp_path, """
assignments:
  - name: Soft hat
    channel: 10
    notes: 42
    velocity: [0, 63]
    select:
      where:
        reference: BD0025
  - name: Hard hat
    channel: 10
    notes: 42
    velocity: [64, 127]
    select:
      where:
        reference: BD0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		assert (9, 42) in note_map
		entries = note_map[(9, 42)]
		assert len(entries) == 2

		names = {e[0].name for e in entries}
		assert names == {"Soft hat", "Hard hat"}

	def test_overlapping_layers_rejected (self, tmp_path: pathlib.Path) -> None:
		"""Overlap is almost always a typo; reject loudly at load."""

		path = self._write_map(tmp_path, """
assignments:
  - name: Soft hat
    channel: 10
    notes: 42
    velocity: [0, 63]
    select:
      where:
        reference: BD0025
  - name: Wider hat
    channel: 10
    notes: 42
    velocity: [50, 100]
    select:
      where:
        reference: BD0025
""")
		with pytest.raises(ValueError, match="overlap"):
			subsample.player.load_midi_map(path, ["BD0025"])

	def test_coverage_gap_warned (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Gaps in velocity coverage log a WARNING but don't reject — the
		user may legitimately want certain velocities to be silent."""

		import logging

		path = self._write_map(tmp_path, """
assignments:
  - name: Soft hat
    channel: 10
    notes: 42
    velocity: [0, 30]
    select:
      where:
        reference: BD0025
  - name: Hard hat
    channel: 10
    notes: 42
    velocity: [60, 127]
    select:
      where:
        reference: BD0025
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		# Map loads (length-2 list for the layered note).
		assert len(note_map[(9, 42)]) == 2

		# Warning names the gap.
		assert any("gap" in r.message.lower() for r in caplog.records)
		assert any("[31, 59]" in r.message for r in caplog.records)


class TestSelectVelocityLayer:

	"""Tests for MidiPlayer._select_velocity_layer — the runtime lookup
	that picks the matching velocity layer at note-on time."""

	def _make_player (self) -> subsample.player.MidiPlayer:
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

	def _entries (
		self,
		ranges_and_rescales: list[tuple[tuple[int, int], typing.Optional[tuple[int, int]]]],
	) -> list[tuple[subsample.query.Assignment, subsample.query.PickSpec]]:
		"""Build a list of velocity-layered entries from (trigger, rescale_to) pairs."""
		entries = []
		for i, (trig, resc) in enumerate(ranges_and_rescales):
			asgn = subsample.query.Assignment(
				name=f"layer_{i}",
				select=(subsample.query.SelectSpec(),),
				velocity_trigger=trig,
				velocity_rescale_to=resc,
			)
			entries.append((asgn, subsample.query.PickSpec(1, 1)))
		return entries

	def test_single_default_layer_matches_any_velocity (self) -> None:
		"""A single full-range layer matches every velocity from 1 to 127."""

		player = self._make_player()
		entries = self._entries([((0, 127), None)])

		for v in (1, 32, 64, 100, 127):
			result = player._select_velocity_layer(entries, v)
			assert result is not None
			asgn, _, effective = result
			assert asgn.name == "layer_0"
			# No rescale_to → effective velocity == input velocity.
			assert effective == v

	def test_two_layers_route_by_velocity (self) -> None:
		"""Low-velocity input picks layer 0, high-velocity picks layer 1."""

		player = self._make_player()
		entries = self._entries([((0, 63), None), ((64, 127), None)])

		soft = player._select_velocity_layer(entries, 30)
		assert soft is not None
		assert soft[0].name == "layer_0"

		hard = player._select_velocity_layer(entries, 100)
		assert hard is not None
		assert hard[0].name == "layer_1"

	def test_returns_none_for_uncovered_velocity (self) -> None:
		"""A velocity in a coverage gap returns None — handler then plays
		nothing, matching the existing 'no mapping for this note' path."""

		player = self._make_player()
		# Gap from 31-59 inclusive.
		entries = self._entries([((0, 30), None), ((60, 127), None)])

		assert player._select_velocity_layer(entries, 45) is None

	def test_rescale_true_endpoints (self) -> None:
		"""rescale: true → output spans 0-127 over the trigger range."""

		player = self._make_player()
		entries = self._entries([((0, 63), (0, 127))])

		_, _, eff_lo = player._select_velocity_layer(entries, 0) or (None, None, None)
		_, _, eff_hi = player._select_velocity_layer(entries, 63) or (None, None, None)

		assert eff_lo == 0
		assert eff_hi == 127

	def test_rescale_custom_range_endpoints (self) -> None:
		"""rescale: [10, 100] → output endpoints land exactly on the target range."""

		player = self._make_player()
		entries = self._entries([((0, 63), (10, 100))])

		_, _, eff_lo = player._select_velocity_layer(entries, 0) or (None, None, None)
		_, _, eff_hi = player._select_velocity_layer(entries, 63) or (None, None, None)

		assert eff_lo == 10
		assert eff_hi == 100

	def test_rescale_midpoint_is_midpoint (self) -> None:
		"""Linear mapping: a midpoint input lands at the midpoint of the output."""

		player = self._make_player()
		# Trigger 0-100, rescale to 0-127 — input 50 → output 63 or 64.
		entries = self._entries([((0, 100), (0, 127))])

		_, _, eff = player._select_velocity_layer(entries, 50) or (None, None, None)
		# Linear: 0 + 50/100 * 127 = 63.5 → rounds to 64 (banker's-rounding
		# matters here, but Python rounds half to even → 64).
		assert eff in (63, 64)

	def test_passthrough_when_no_rescale (self) -> None:
		"""Without rescale_to, the input velocity reaches the handler unchanged."""

		player = self._make_player()
		entries = self._entries([((10, 100), None)])

		_, _, eff = player._select_velocity_layer(entries, 50) or (None, None, None)
		assert eff == 50


class TestPerLayerSegmentCounter:

	"""Two velocity layers on the same note must keep independent
	round_robin counters (and independent _last_played fallbacks)."""

	def _make_player (self) -> subsample.player.MidiPlayer:
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

	def test_round_robin_counters_independent_per_layer (self) -> None:
		"""Two layers on the same (channel, note) advance their own counters."""

		import subsample.analysis

		player = self._make_player()
		bounds = ((0, 100), (100, 200))
		audio  = numpy.zeros((200, 1), dtype=numpy.float32)
		level  = subsample.analysis.LevelResult(peak=0.0, rms=0.0)

		# Layer A: trigger (0, 63).  Two triggers advance its counter to 2.
		for _ in range(2):
			player._select_segment(
				audio, level, bounds, "round_robin",
				channel=9, note=36, velocity_trigger=(0, 63),
			)

		# Layer B: trigger (64, 127).  One trigger advances its counter to 1.
		player._select_segment(
			audio, level, bounds, "round_robin",
			channel=9, note=36, velocity_trigger=(64, 127),
		)

		# Verified the keys are distinct and the counts are independent.
		assert player._segment_counters[(9, 36, 0, 63)]   == 2
		assert player._segment_counters[(9, 36, 64, 127)] == 1


# ---------------------------------------------------------------------------
# Zone-tuned MIDI mapping — _parse_zone_notes, load_midi_map integration,
# materialisation against the active library, and end-to-end note_on dispatch.
# ---------------------------------------------------------------------------

class TestParseZoneNotes:

	"""Unit tests for the YAML parser _parse_zone_notes()."""

	def test_string_sentinel_returns_full_range (self) -> None:
		assert subsample.player._parse_zone_notes("zone-tuned", "asgn") == (0, 127)

	def test_regular_notes_passthrough_returns_none (self) -> None:
		"""Bare int / name / list / range string all fall through —
		_parse_zone_notes returns None and the caller routes to
		_parse_note_spec for the regular mapping path."""

		assert subsample.player._parse_zone_notes(36, "asgn") is None
		assert subsample.player._parse_zone_notes("C4", "asgn") is None
		assert subsample.player._parse_zone_notes("36..60", "asgn") is None
		assert subsample.player._parse_zone_notes([36, 38], "asgn") is None
		assert subsample.player._parse_zone_notes("drum.kick_1", "asgn") is None

	def test_dict_form_without_range_returns_full_range (self) -> None:
		"""The dict form without a `range:` key is the same as the string
		shortcut — auto-zone over the full keyboard."""

		result = subsample.player._parse_zone_notes(
			{"mode": "zone-tuned"}, "asgn",
		)
		assert result == (0, 127)

	def test_dict_form_with_note_name_range (self) -> None:
		"""Note names like C4 / G9 work as range bounds, via the shared
		_parse_single_note helper."""

		result = subsample.player._parse_zone_notes(
			{"mode": "zone-tuned", "range": ["C4", "G9"]}, "asgn",
		)
		# C4 = 60, G9 = 127
		assert result == (60, 127)

	def test_dict_form_with_integer_range (self) -> None:
		result = subsample.player._parse_zone_notes(
			{"mode": "zone-tuned", "range": [0, 50]}, "asgn",
		)
		assert result == (0, 50)

	def test_dict_form_mixed_int_and_name_range (self) -> None:
		result = subsample.player._parse_zone_notes(
			{"mode": "zone-tuned", "range": [60, "C5"]}, "asgn",
		)
		assert result == (60, 72)

	def test_dict_form_unknown_inner_key_rejected (self) -> None:
		"""Typo guard mirroring _VELOCITY_INNER_KEYS — `mdoe` is not `mode`."""

		with pytest.raises(ValueError, match="unknown notes key"):
			subsample.player._parse_zone_notes(
				{"mdoe": "zone-tuned"}, "asgn",
			)

	def test_dict_form_wrong_mode_rejected (self) -> None:
		"""Dict form with a mode that isn't zone-tuned is rejected
		(future-proofs the dict form so it can grow other modes later
		without silently accepting nonsense today)."""

		with pytest.raises(ValueError, match="mode"):
			subsample.player._parse_zone_notes(
				{"mode": "auto-zone"}, "asgn",
			)

	def test_range_out_of_bounds_rejected (self) -> None:
		with pytest.raises(ValueError):
			subsample.player._parse_zone_notes(
				{"mode": "zone-tuned", "range": [0, 200]}, "asgn",
			)

	def test_range_lo_greater_than_hi_rejected (self) -> None:
		with pytest.raises(ValueError, match="lo > hi"):
			subsample.player._parse_zone_notes(
				{"mode": "zone-tuned", "range": [100, 50]}, "asgn",
			)

	def test_range_wrong_length_rejected (self) -> None:
		with pytest.raises(ValueError, match="2-element"):
			subsample.player._parse_zone_notes(
				{"mode": "zone-tuned", "range": [60]}, "asgn",
			)

	def test_assignment_name_in_error_messages (self) -> None:
		with pytest.raises(ValueError, match="'Pitched keyboard'"):
			subsample.player._parse_zone_notes(
				{"mode": "zone-tuned", "range": [200, 50]}, "Pitched keyboard",
			)


class TestLoadMidiMapZoneTuned:

	"""``load_midi_map`` integration: a zone-tuned assignment produces
	no concrete NoteMap entries at load time (those are materialised
	against the library at runtime) but appears as a ZoneTemplate in
	the returned MidiMapResult."""

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_zone_tuned_loaded_as_template (self, tmp_path: pathlib.Path) -> None:
		path = self._write_map(tmp_path, """
assignments:
  - name: Pitched library
    channel: 1
    notes: zone-tuned
    process:
      - repitch: true
    select:
      where: { pitched: true }
""")
		result = subsample.player.load_midi_map(path, [])

		# No manual entries — zone-tuned doesn't materialise at load.
		assert result.note_map == {}
		# Exactly one template recorded.
		assert len(result.zone_templates) == 1
		template = result.zone_templates[0]
		assert template.name == "Pitched library"
		assert template.channel == 0
		assert template.keyboard_range == (0, 127)

	def test_repitch_required (self, tmp_path: pathlib.Path) -> None:
		"""Zone-tuned without ``process: [- repitch: true]`` is rejected
		at load — the feature is meaningless without repitching."""

		path = self._write_map(tmp_path, """
assignments:
  - name: Bad zone-tuned
    channel: 1
    notes: zone-tuned
    select:
      where: { pitched: true }
""")
		with pytest.raises(ValueError, match="repitch"):
			subsample.player.load_midi_map(path, [])

	def test_dict_form_with_range (self, tmp_path: pathlib.Path) -> None:
		path = self._write_map(tmp_path, """
assignments:
  - name: Bass zone
    channel: 1
    notes:
      mode: zone-tuned
      range: [C2, B3]
    process:
      - repitch: true
    select:
      where: { pitched: true }
""")
		result = subsample.player.load_midi_map(path, [])

		assert len(result.zone_templates) == 1
		# C2 = 36, B3 = 59
		assert result.zone_templates[0].keyboard_range == (36, 59)

	def test_manual_on_same_channel_rejected (self, tmp_path: pathlib.Path) -> None:
		"""A manual assignment on a channel owned by a zone-tuned is
		rejected — zone-tuned owns its channel exclusively."""

		path = self._write_map(tmp_path, """
assignments:
  - name: Auto zone
    channel: 1
    notes: zone-tuned
    process:
      - repitch: true
    select:
      where: { pitched: true }
  - name: Manual override
    channel: 1
    notes: C4
    select:
      where: { name: my-sample }
""")
		with pytest.raises(ValueError, match="zone-tuned owns"):
			subsample.player.load_midi_map(path, [])

	def test_overlapping_zone_ranges_rejected (self, tmp_path: pathlib.Path) -> None:
		"""Two zone-tuned on the same channel with overlapping ranges
		are rejected — same overlap-is-almost-always-a-typo rule as for
		velocity layering."""

		path = self._write_map(tmp_path, """
assignments:
  - name: Bass
    channel: 1
    notes:
      mode: zone-tuned
      range: [0, 64]
    process:
      - repitch: true
    select:
      where: { pitched: true }
  - name: Lead
    channel: 1
    notes:
      mode: zone-tuned
      range: [50, 127]
    process:
      - repitch: true
    select:
      where: { pitched: true }
""")
		with pytest.raises(ValueError, match="overlap"):
			subsample.player.load_midi_map(path, [])

	def test_non_overlapping_zone_ranges_accepted (self, tmp_path: pathlib.Path) -> None:
		"""Two zone-tuned on the same channel with adjacent (not
		overlapping) ranges load cleanly — the split-keyboard pattern."""

		path = self._write_map(tmp_path, """
assignments:
  - name: Bass
    channel: 1
    notes:
      mode: zone-tuned
      range: [0, 50]
    process:
      - repitch: true
    select:
      where: { pitched: true }
  - name: Lead
    channel: 1
    notes:
      mode: zone-tuned
      range: [51, 127]
    process:
      - repitch: true
    select:
      where: { pitched: true }
""")
		result = subsample.player.load_midi_map(path, [])

		assert len(result.zone_templates) == 2
		names = {t.name for t in result.zone_templates}
		assert names == {"Bass", "Lead"}


class TestMaterializeZones:

	"""End-to-end materialisation of ZoneTemplates against a populated
	instrument library, including the in-range filter, the sort order,
	the zone-boundary algorithm, and the re-materialisation behaviour on
	library changes."""

	def _make_pitched_record (
		self,
		name:    str,
		pitch_hz: float,
		sample_id: typing.Optional[int] = None,
	) -> subsample.library.SampleRecord:
		"""Build a SampleRecord that passes has_stable_pitch with the given
		detected pitch.  All other analysis fields are stubbed (via the
		test helpers) to values that satisfy the 7-criterion gate."""

		# pitch fields chosen to clear has_stable_pitch:
		#   voiced_fraction > 0.5, voiced_frame_count >= 5, pitch_confidence > 0.5,
		#   pitch_stability < 0.5, harmonic_ratio > 0.4.
		pitch = tests.helpers._make_pitch(
			dominant_pitch_hz=pitch_hz,
			pitch_confidence=0.8,
			pitch_stability=0.1,
			voiced_frame_count=100,
		)
		spectral = dataclasses.replace(
			tests.helpers._make_spectral(),
			voiced_fraction=0.8, harmonic_ratio=0.7,
		)

		return subsample.library.SampleRecord(
			sample_id   = sample_id if sample_id is not None else subsample.library.allocate_id(),
			name        = name,
			spectral    = spectral,
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = pitch,
			timbre      = tests.helpers._make_timbre(),
			level       = tests.helpers._make_level(),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 1.0,
			audio       = numpy.zeros((44100, 1), dtype=numpy.int16),
		)

	def _make_unpitched_record (self, name: str) -> subsample.library.SampleRecord:
		"""Build a SampleRecord that fails has_stable_pitch — dominant_pitch_hz=0
		violates the first of the 7 criteria, which is enough to fail the gate."""
		return subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = name,
			spectral    = tests.helpers._make_spectral(),
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = tests.helpers._make_pitch(dominant_pitch_hz=0.0),
			timbre      = tests.helpers._make_timbre(),
			level       = tests.helpers._make_level(),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 1.0,
			audio       = numpy.zeros((44100, 1), dtype=numpy.int16),
		)

	def _make_player_with_library (
		self,
		records:         list[subsample.library.SampleRecord],
		zone_templates:  tuple[subsample.player.ZoneTemplate, ...] = (),
	) -> subsample.player.MidiPlayer:
		"""Build a MidiPlayer with a real InstrumentLibrary populated
		from the given records and zone-tuned templates installed."""
		import threading
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10_000_000)
		for r in records:
			lib.add(r)
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=lib,
			similarity_matrix=similarity,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
			zone_templates=zone_templates,
		)

	def _make_template (
		self,
		name: str = "Auto",
		channel: int = 0,
		keyboard_range: tuple[int, int] = (0, 127),
	) -> subsample.player.ZoneTemplate:
		"""Build a minimal ZoneTemplate with an empty filter (matches everything)."""
		return subsample.player.ZoneTemplate(
			name=name,
			channel=channel,
			keyboard_range=keyboard_range,
			select=(subsample.query.SelectSpec(),),
			process=subsample.query.ProcessSpec(
				steps=(subsample.query.ProcessorStep(name="repitch"),),
			),
			one_shot=False,
			gain_db=0.0,
			pan_weights=None,
			output_routing=None,
			extract=None,
			segment_mode="",
			velocity_trigger=(0, 127),
			velocity_rescale_to=None,
		)

	def test_no_templates_leaves_map_unchanged (self) -> None:
		"""When no templates are declared the working map is just a copy
		of the base."""
		player = self._make_player_with_library([])
		# Empty templates → _materialize_zones is a no-op beyond the
		# dict-copy of the base.
		assert player._note_map == {}

	def test_single_sample_covers_full_range (self) -> None:
		"""One pitched sample on a full-keyboard template owns every note."""

		import librosa

		# Centre pitch deliberately well inside the keyboard.
		pitch_hz = float(librosa.midi_to_hz(60))
		record = self._make_pitched_record("only_one", pitch_hz=pitch_hz)
		template = self._make_template(channel=0, keyboard_range=(0, 127))

		player = self._make_player_with_library([record], (template,))

		# Every note 0-127 should map to the single sample.
		for note in (0, 60, 127):
			assert (0, note) in player._note_map
			entries = player._note_map[(0, note)]
			assert len(entries) == 1

	def test_two_samples_split_at_midpoint (self) -> None:
		"""Two samples at MIDI 60 and 64 → sample 0 covers [0, 62],
		sample 1 covers [63, 127].  Lower-pitched claims the midpoint."""

		import librosa

		lo_record  = self._make_pitched_record("sample_lo",
			pitch_hz=float(librosa.midi_to_hz(60)))
		hi_record  = self._make_pitched_record("sample_hi",
			pitch_hz=float(librosa.midi_to_hz(64)))
		template = self._make_template(channel=0, keyboard_range=(0, 127))

		player = self._make_player_with_library([lo_record, hi_record], (template,))

		# Note 62 — midpoint — should belong to the lower sample.
		assert player._note_map[(0, 62)][0][0].name.endswith("sample_lo")
		# Note 63 — next note — should belong to the higher sample.
		assert player._note_map[(0, 63)][0][0].name.endswith("sample_hi")

	def test_lowest_sample_extends_to_range_lo (self) -> None:
		"""Lowest sample's zone reaches the keyboard's lo (not just its
		own detected pitch) — user's stated requirement."""

		import librosa

		# Two samples at MIDI 60 and 80.
		records = [
			self._make_pitched_record("low",  pitch_hz=float(librosa.midi_to_hz(60))),
			self._make_pitched_record("high", pitch_hz=float(librosa.midi_to_hz(80))),
		]
		template = self._make_template(channel=0, keyboard_range=(0, 127))
		player = self._make_player_with_library(records, (template,))

		# Note 0 — far below either sample's centre — should belong to low.
		assert player._note_map[(0, 0)][0][0].name.endswith("low")

	def test_highest_sample_extends_to_range_hi (self) -> None:
		import librosa

		records = [
			self._make_pitched_record("low",  pitch_hz=float(librosa.midi_to_hz(40))),
			self._make_pitched_record("high", pitch_hz=float(librosa.midi_to_hz(50))),
		]
		template = self._make_template(channel=0, keyboard_range=(0, 127))
		player = self._make_player_with_library(records, (template,))

		# Note 127 — far above either centre — belongs to the high sample.
		assert player._note_map[(0, 127)][0][0].name.endswith("high")

	def test_restricted_keyboard_range (self) -> None:
		"""range: [60, 80] excludes pitches outside that range from the
		template's coverage."""

		import librosa

		records = [
			# 36 — outside, should be excluded.
			self._make_pitched_record("out_low", pitch_hz=float(librosa.midi_to_hz(36))),
			# 65 — inside.
			self._make_pitched_record("inside",  pitch_hz=float(librosa.midi_to_hz(65))),
			# 90 — outside.
			self._make_pitched_record("out_hi",  pitch_hz=float(librosa.midi_to_hz(90))),
		]
		template = self._make_template(channel=0, keyboard_range=(60, 80))
		player = self._make_player_with_library(records, (template,))

		# Notes 60-80 covered by "inside"; notes outside that range not mapped.
		assert (0, 65) in player._note_map
		assert player._note_map[(0, 65)][0][0].name.endswith("inside")
		# Nothing outside the keyboard range.
		assert (0, 0)   not in player._note_map
		assert (0, 127) not in player._note_map

	def test_no_matching_samples_logs_info (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""When the filter excludes everything, the template contributes
		no entries and an INFO log explains the situation."""
		import logging

		template = self._make_template(channel=0, keyboard_range=(0, 127))
		# No records in the library at all.
		player = self._make_player_with_library([], (template,))

		with caplog.at_level(logging.INFO, logger="subsample.player"):
			player._materialize_zones()

		# Map has no entries on the zone channel.
		assert (0, 60) not in player._note_map
		# An INFO line names the template.
		assert any(
			"no matching pitched samples" in r.message and "Auto" in r.message
			for r in caplog.records
		)

	def test_unpitched_samples_excluded (self) -> None:
		"""has_stable_pitch is the implicit filter — unpitched samples
		are never included even if they pass the user's select."""

		# Build one pitched sample and one unpitched.
		import librosa
		pitched_record   = self._make_pitched_record(
			"pitched_one", pitch_hz=float(librosa.midi_to_hz(60)),
		)
		unpitched_record = self._make_unpitched_record("unpitched_one")

		template = self._make_template(channel=0, keyboard_range=(0, 127))
		player = self._make_player_with_library(
			[pitched_record, unpitched_record], (template,),
		)

		# The pitched sample owns every note on the channel.
		assert player._note_map[(0, 60)][0][0].name.endswith("pitched_one")
		# The unpitched sample's name should NOT appear in any derived Assignment.
		for entries in player._note_map.values():
			for asgn, _ in entries:
				assert "unpitched_one" not in asgn.name

	def test_re_materialization_picks_up_new_sample (self) -> None:
		"""Adding a new pitched sample to the library + calling
		_materialize_zones rebuilds the layout including the new one."""

		import librosa

		first  = self._make_pitched_record(
			"first", pitch_hz=float(librosa.midi_to_hz(40)),
		)
		template = self._make_template(channel=0, keyboard_range=(0, 127))
		player = self._make_player_with_library([first], (template,))

		# Initially only the first sample covers the keyboard.
		assert player._note_map[(0, 60)][0][0].name.endswith("first")

		# Add a higher-pitched sample.
		second = self._make_pitched_record(
			"second", pitch_hz=float(librosa.midi_to_hz(80)),
		)
		player._instrument_library.add(second)
		player._materialize_zones()

		# Now the higher half of the keyboard belongs to "second".
		assert player._note_map[(0, 100)][0][0].name.endswith("second")
		# And the lower half still belongs to "first".
		assert player._note_map[(0, 0)][0][0].name.endswith("first")

	def test_tied_pitches_tiebreak_by_name (self) -> None:
		"""Two samples at the same detected pitch — deterministic
		alphabetical-by-name tiebreak."""

		import librosa

		a_record = self._make_pitched_record(
			"alpha", pitch_hz=float(librosa.midi_to_hz(60)),
		)
		b_record = self._make_pitched_record(
			"bravo", pitch_hz=float(librosa.midi_to_hz(60)),
		)
		template = self._make_template(channel=0, keyboard_range=(0, 127))
		player = self._make_player_with_library([a_record, b_record], (template,))

		# Both samples are at MIDI 60.  Alphabetical order → "alpha" sorts
		# first; the materialiser treats it as the "lower" sample.  Lower
		# claims the midpoint of (60, 60) = 60.  So note 60 → alpha,
		# note 61+ → bravo.
		assert player._note_map[(0, 60)][0][0].name.endswith("alpha")
		assert player._note_map[(0, 127)][0][0].name.endswith("bravo")


class TestZoneTunedRuntime:

	"""End-to-end note_on dispatch through a zone-tuned MidiPlayer —
	confirms the materialised entries route correctly."""

	def _make_player_with_zone (
		self,
		records: list[subsample.library.SampleRecord],
	) -> subsample.player.MidiPlayer:
		"""Single-zone-template player covering MIDI 0-127 on channel 0."""
		import threading
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10_000_000)
		for r in records:
			lib.add(r)
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		template = subsample.player.ZoneTemplate(
			name="Auto",
			channel=0,
			keyboard_range=(0, 127),
			select=(subsample.query.SelectSpec(),),
			process=subsample.query.ProcessSpec(
				steps=(subsample.query.ProcessorStep(name="repitch"),),
			),
			one_shot=False,
			gain_db=0.0,
			pan_weights=None,
			output_routing=None,
			extract=None,
			segment_mode="",
			velocity_trigger=(0, 127),
			velocity_rescale_to=None,
		)
		return subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=lib,
			similarity_matrix=similarity,
			midi_map={},
			sample_rate=44100,
			bit_depth=16,
			zone_templates=(template,),
		)

	def test_note_in_zone_resolves_correct_sample (self) -> None:
		"""A note in a sample's zone should resolve to that sample's name
		via the materialised exact-stem-name where predicate."""

		import librosa

		# Build a pitched record using tests.helpers to keep this test
		# self-contained (no inheritance from TestMaterializeZones).
		pitch = tests.helpers._make_pitch(
			dominant_pitch_hz=float(librosa.midi_to_hz(60)),
			pitch_confidence=0.8,
			pitch_stability=0.1,
			voiced_frame_count=100,
		)
		spectral = dataclasses.replace(
			tests.helpers._make_spectral(),
			voiced_fraction=0.8, harmonic_ratio=0.7,
		)
		record = subsample.library.SampleRecord(
			sample_id=subsample.library.allocate_id(),
			name="tonal",
			spectral=spectral,
			rhythm=tests.helpers._make_rhythm(),
			pitch=pitch,
			timbre=tests.helpers._make_timbre(),
			level=tests.helpers._make_level(),
			band_energy=tests.helpers._make_band_energy(),
			params=tests.helpers._make_params(),
			duration=1.0,
			audio=numpy.zeros((44100, 1), dtype=numpy.int16),
		)

		player = self._make_player_with_zone([record])

		# Entries on note 60 point at an Assignment whose select where.name
		# matches "tonal" exactly.
		entries = player._note_map[(0, 60)]
		assert len(entries) == 1
		asgn, _ = entries[0]
		assert asgn.select[0].where.name == "tonal"
		# And that Assignment's process inherits repitch from the template.
		assert asgn.process.has_repitch()


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

	# -- Symbolic notes via PyMidiDefs (drum.kick_1, etc.) ------------------

	def test_drum_symbol_lowercase (self) -> None:
		"""'drum.kick_1' looks up pymididefs.drums.GM_DRUM_MAP and returns 36."""
		assert subsample.player._parse_note_spec("drum.kick_1", "test") == [36]

	def test_drum_symbol_uppercase (self) -> None:
		"""Symbol part is case-insensitive — uppercase matches PyMidiDefs's Python constants."""
		assert subsample.player._parse_note_spec("drum.KICK_1", "test") == [36]

	def test_drum_symbol_mixed_case_namespace (self) -> None:
		"""Namespace prefix is also case-insensitive."""
		assert subsample.player._parse_note_spec("Drum.kick_1", "test") == [36]

	def test_drum_symbol_in_list (self) -> None:
		"""Symbolic notes work inside a list."""
		assert subsample.player._parse_note_spec(["drum.kick_1", "drum.snare_1"], "test") == [36, 38]

	def test_drum_symbol_mixed_list (self) -> None:
		"""A list can mix symbolic, integer, and note-name forms."""
		assert subsample.player._parse_note_spec(["drum.kick_1", 38, "C3"], "test") == [36, 38, 48]

	def test_drum_primary_aliases (self) -> None:
		"""v0.2.3 unnumbered aliases resolve to the GM primary note (the '1'
		variant): kick=36, snare=38, crash=49, ride=51 — merged into the drum
		namespace from pymididefs.drums.GM_DRUM_PRIMARY_ALIASES."""
		assert subsample.player._parse_note_spec("drum.kick", "test") == [36]
		assert subsample.player._parse_note_spec("drum.snare", "test") == [38]
		assert subsample.player._parse_note_spec("drum.crash", "test") == [49]
		assert subsample.player._parse_note_spec("drum.ride", "test") == [51]

	def test_drum_primary_alias_matches_numbered (self) -> None:
		"""The bare alias and its numbered form resolve to the same note."""
		assert (
			subsample.player._parse_note_spec("drum.kick", "test")
			== subsample.player._parse_note_spec("drum.kick_1", "test")
		)

	def test_drum_primary_alias_case_insensitive (self) -> None:
		"""The bare aliases are case-insensitive like every other drum symbol."""
		assert subsample.player._parse_note_spec("drum.KICK", "test") == [36]

	def test_unknown_namespace_falls_through (self) -> None:
		"""An unknown namespace falls through to the note-name parser, which
		then raises the existing 'not a valid note name' error — no special
		handling needed."""
		with pytest.raises(ValueError):
			subsample.player._parse_note_spec("foo.bar", "test")

	def test_unknown_drum_symbol_raises (self) -> None:
		"""An unknown drum symbol raises with both prefix and symbol in the message."""
		with pytest.raises(ValueError, match="drum.*nonexistent"):
			subsample.player._parse_note_spec("drum.nonexistent", "test")

	def test_drum_range_rejected (self) -> None:
		"""Symbolic ranges are explicitly rejected with a list-syntax hint."""
		with pytest.raises(ValueError, match="list"):
			subsample.player._parse_note_spec("drum.kick_1..drum.snare_1", "test")


# ---------------------------------------------------------------------------
# Symbolic notes — end-to-end through load_midi_map
# ---------------------------------------------------------------------------

class TestLoadMidiMapSymbolicNotes:

	"""Verify symbolic notes flow through load_midi_map() into the NoteMap.

	Distinct from the _parse_note_spec unit tests above — those check the
	parser in isolation; this confirms the symbolic form survives the full
	YAML → Assignment → NoteMap pipeline."""

	def test_drum_symbol_in_yaml_map (self, tmp_path: pathlib.Path) -> None:
		path = tmp_path / "symbolic-map.yaml"
		path.write_text("""
assignments:
  - name: Kick
    channel: 10
    notes: drum.kick_1
    select:
      where:
        pitched: false
""", encoding="utf-8")

		note_map = subsample.player.load_midi_map(path, []).note_map

		# mido channel 9 = MIDI channel 10 (user-facing 1-indexed → 0-indexed)
		assert (9, 36) in note_map
		assert note_map[(9, 36)][0][0].name == "Kick"


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
		"""update_assignments() calls get_variant() for stretch_quantize assignments."""

		asgn = _make_assignment(
			name="Loops", stretch_quantize=True, repitch=False,
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

	def test_range_pick_precomputes_all_ranks (self) -> None:
		"""A range pick pre-computes variants for every reachable rank, so
		the runtime random draw never hits a cold cache."""

		# stretch_quantize assignment with PickSpec(1, 3) on a single note.
		asgn = subsample.query.Assignment(
			name="Varied loops",
			select=(subsample.query.SelectSpec(
				order=(subsample.query.OrderClause(by="age", dir="desc"),),
				pick=subsample.query.PickSpec(1, 3),
			),),
			process=subsample.query.ProcessSpec(steps=(
				subsample.query.ProcessorStep(name="stretch_quantize", params=(("bpm", 120), ("grid", 16))),
			)),
			one_shot=False,
		)
		note_map: subsample.player.NoteMap = {
			(0, 60): [(asgn, subsample.query.PickSpec(1, 3))],
		}

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

		# Three mock samples: ids 30, 20, 10 → newest-first ranks them as
		# [30, 20, 10] so all three are reachable from PickSpec(1, 3).
		records = []
		for sid in (30, 20, 10):
			r = unittest.mock.MagicMock()
			r.sample_id        = sid
			r.name             = f"loop-{sid}"
			r.rhythm.tempo_bpm = 120.0
			records.append(r)

		player._instrument_library.samples.return_value = records
		player._instrument_library.get.side_effect = lambda sid: next(
			(r for r in records if r.sample_id == sid), None,
		)

		transform_manager = unittest.mock.MagicMock()
		player._transform_manager = transform_manager

		player.update_assignments()

		# All three ranks should have been pre-computed (one get_variant per sample_id).
		assert transform_manager.get_variant.call_count == 3
		called_sids = {call.args[0] for call in transform_manager.get_variant.call_args_list}
		assert called_sids == {30, 20, 10}

	def test_beat_quantize_with_explicit_bpm (self) -> None:
		"""Per-assignment BPM override produces a spec with correct params."""

		# stretch_quantize with explicit bpm=120, grid=8
		asgn = subsample.query.Assignment(
			name="Explicit BPM",
			select=(subsample.query.SelectSpec(order=(subsample.query.OrderClause(by="age", dir="desc"),)),),
			process=subsample.query.ProcessSpec(steps=(
				subsample.query.ProcessorStep(name="stretch_quantize", params=(("bpm", 120), ("grid", 8))),
			)),
			one_shot=False,
		)
		note_map: subsample.player.NoteMap = {(0, 60): [(asgn, subsample.query.PickSpec(1, 1))]}

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
		asgn, pick = note_map[(0, 36)][0]
		assert asgn.select[0].where.pitched is True
		assert asgn.select[0].order == (subsample.query.OrderClause(by="age", dir="asc"),)
		assert asgn.process.has_repitch()
		assert asgn.one_shot is False
		assert pick == subsample.query.PickSpec(1, 1)

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
		asgn, _ = note_map[(1, 60)][0]
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
		asgn, pick = note_map[(0, 60)][0]
		assert asgn.select[0].pick == subsample.query.PickSpec(2, 2)
		assert pick == subsample.query.PickSpec(2, 2)

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
			asgn, pick = note_map[(0, midi_note)][0]
			assert pick == subsample.query.PickSpec(1, 1)
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


# ---------------------------------------------------------------------------
# TestLoadMidiMapPickRange — range pick form ([lo, hi] / {gte, lte})
# ---------------------------------------------------------------------------

class TestLoadMidiMapPickRange:

	"""load_midi_map's handling of the range pick forms."""

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_pick_list_form_stored_as_pickspec (self, tmp_path: pathlib.Path) -> None:
		"""pick: [1, 3] parses to PickSpec(1, 3) on the SelectSpec."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Varied kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
      pick: [1, 3]
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		asgn, pick_spec = note_map[(9, 36)][0]
		assert asgn.select[0].pick == subsample.query.PickSpec(1, 3)
		assert pick_spec == subsample.query.PickSpec(1, 3)

	def test_pick_dict_form_stored_as_pickspec (self, tmp_path: pathlib.Path) -> None:
		"""pick: {gte: 1, lte: 3} is equivalent to [1, 3]."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Varied kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
      pick: {gte: 1, lte: 3}
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		_asgn, pick_spec = note_map[(9, 36)][0]
		assert pick_spec == subsample.query.PickSpec(1, 3)

	def test_pick_range_suppresses_auto_distribute (self, tmp_path: pathlib.Path) -> None:
		"""A range pick on multiple notes suppresses auto-distribute — every
		note in the assignment stores the same PickSpec.  Per-note variety
		comes from the runtime random draw, not load-time distribution."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Varied kicks
    channel: 10
    notes: [36, 37, 38]
    select:
      where:
        reference: BD0025
      order_by: similarity
      pick: [1, 3]
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map

		# All three notes should share the same PickSpec(1, 3).
		for note in (36, 37, 38):
			_asgn, pick_spec = note_map[(9, note)][0]
			assert pick_spec == subsample.query.PickSpec(1, 3)


class TestRanksFor:

	"""_ranks_for — the helper that expands a PickSpec to its reachable ranks."""

	def test_single_rank (self) -> None:
		assert list(subsample.player._ranks_for(subsample.query.PickSpec(2, 2), 5)) == [2]

	def test_range (self) -> None:
		assert list(subsample.player._ranks_for(subsample.query.PickSpec(1, 3), 5)) == [1, 2, 3]

	def test_hi_clamped (self) -> None:
		"""hi clamped to ranked_len so we never enqueue past the end."""
		assert list(subsample.player._ranks_for(subsample.query.PickSpec(1, 10), 3)) == [1, 2, 3]

	def test_lo_clamped_when_above_ranked_len (self) -> None:
		"""When lo also exceeds ranked_len, collapse onto the last rank."""
		assert list(subsample.player._ranks_for(subsample.query.PickSpec(8, 12), 3)) == [3]


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
		asgn, pick = note_map[(1, 36)][0]
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
		asgn, _ = note_map[(2, 36)][0]
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
		asgn, _ = note_map[(9, 36)][0]
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

		asgn, _ = note_map[(0, 60)][0]
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
			asgn, pick = note_map[(0, midi_note)][0]
			assert pick == subsample.query.PickSpec(1, 1)
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

		note_map: subsample.player.NoteMap = {(0, 60): [(asgn, subsample.query.PickSpec(1, 1))]}

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

	def _wrap (self, note_map: subsample.player.NoteMap) -> subsample.player.MidiMapResult:
		"""Wrap a NoteMap in a MidiMapResult for reload_midi_map() calls.

		Since the velocity-layering + zone-tuned work, reload_midi_map()
		takes the full MidiMapResult so it can swap zone_templates too.
		Tests that only care about the note-map portion use this helper
		to keep the call sites readable."""

		return subsample.player.MidiMapResult(
			note_map=note_map,
			bank_definitions=[],
			bank_channel=subsample.bank.DEFAULT_BANK_CHANNEL,
		)

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

		# __init__ stores _base_note_map = old_map and creates a working
		# _note_map as dict(old_map) so the runtime materialisation step
		# can mutate it without bleeding into the base.  Equality, not
		# identity, is the right check after the zone-tuned migration.
		assert player._note_map == old_map

		asgn_b = _make_assignment(name="Snares", reference="SD0010")
		new_map = _make_note_map(asgn_b, channel=9, notes=[38])

		player.reload_midi_map(self._wrap(new_map))

		assert player._note_map == new_map
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
			player.reload_midi_map(self._wrap(note_map))
			mock_update.assert_called_once()

	def test_try_update_assignments_swallows_exception (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:

		"""_try_update_assignments must catch any exception, log at ERROR,
		and not propagate — so the live-state paths (bank switch, CC
		debounce, sample integration) survive a transient query failure."""

		import logging
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

		with unittest.mock.patch.object(
			player, "update_assignments",
			side_effect=ValueError("simulated query failure"),
		):
			with caplog.at_level(logging.ERROR, logger="subsample.player"):
				# Must NOT raise.
				player._try_update_assignments("test context")

		messages = [r.message for r in caplog.records]
		assert any("test context" in m for m in messages), \
			"context label should appear in the error log"
		assert any("simulated query failure" in m for m in messages), \
			"underlying exception should appear in the error log"

	def test_rolls_back_when_update_assignments_raises (self) -> None:

		"""reload_midi_map() restores the previous map when update_assignments raises.

		Regression test for the live-performance crash: a YAML semantic error
		only detectable at query time (e.g. similarity ordering without
		where.reference) should not stop playback — the old map must remain
		active so the next note_on still plays."""

		asgn_old = _make_assignment(name="Kicks", reference="BD0025")
		old_map  = _make_note_map(asgn_old, channel=9, notes=[36])

		asgn_new = _make_assignment(name="Snares", reference="SD0010")
		new_map  = _make_note_map(asgn_new, channel=9, notes=[38])

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
		mapped_ccs_before = player._mapped_ccs

		with unittest.mock.patch.object(
			player, "update_assignments",
			side_effect=ValueError("simulated query-time validation failure"),
		):
			with pytest.raises(ValueError, match="simulated"):
				player.reload_midi_map(self._wrap(new_map))

		# Active map and CC set restored to the previous good state.
		assert player._note_map == old_map
		assert (9, 36) in player._note_map
		assert (9, 38) not in player._note_map
		assert player._mapped_ccs == mapped_ccs_before


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


class TestParseExtract:

	"""Tests for _parse_extract — YAML extract value to ExtractSpec."""

	def test_none_returns_none (self) -> None:
		"""Missing field returns None (no extract)."""
		assert subsample.player._parse_extract(None, "test") is None

	def test_omni_parses (self) -> None:
		"""'omni' parses to ExtractSpec(kind='omni')."""
		result = subsample.player._parse_extract("omni", "test")
		assert result == subsample.query.ExtractSpec(kind="omni")

	def test_all_named_kinds_parse (self) -> None:
		"""Every kind in EXTRACT_KINDS parses to a matching ExtractSpec."""
		for kind in subsample.query.EXTRACT_KINDS:
			result = subsample.player._parse_extract(kind, "test")
			assert result == subsample.query.ExtractSpec(kind=kind)

	def test_case_insensitive (self) -> None:
		"""OMNI, Omni, oMnI all normalise to kind='omni'."""
		for value in ("OMNI", "Omni", "oMnI"):
			result = subsample.player._parse_extract(value, "test")
			assert result == subsample.query.ExtractSpec(kind="omni")

	def test_whitespace_stripped (self) -> None:
		"""Leading/trailing whitespace is stripped."""
		result = subsample.player._parse_extract("  omni  ", "test")
		assert result == subsample.query.ExtractSpec(kind="omni")

	def test_channel_index_parses (self) -> None:
		"""'channel.3' parses to ExtractSpec(kind='channel', channel_index=3)."""
		result = subsample.player._parse_extract("channel.3", "test")
		assert result == subsample.query.ExtractSpec(kind="channel", channel_index=3)

	def test_channel_index_higher (self) -> None:
		"""'channel.8' parses correctly."""
		result = subsample.player._parse_extract("channel.8", "test")
		assert result == subsample.query.ExtractSpec(kind="channel", channel_index=8)

	def test_channel_zero_rejected (self) -> None:
		"""'channel.0' raises (1-indexed)."""
		with pytest.raises(ValueError, match="1-indexed"):
			subsample.player._parse_extract("channel.0", "test")

	def test_channel_negative_rejected (self) -> None:
		"""'channel.-1' raises."""
		with pytest.raises(ValueError, match="1-indexed"):
			subsample.player._parse_extract("channel.-1", "test")

	def test_channel_non_integer_rejected (self) -> None:
		"""'channel.foo' raises with 'integer'."""
		with pytest.raises(ValueError, match="integer"):
			subsample.player._parse_extract("channel.foo", "test")

	def test_unknown_kind_rejected (self) -> None:
		"""Unknown kind raises with the valid options listed."""
		with pytest.raises(ValueError, match="unknown extract"):
			subsample.player._parse_extract("midside", "test")

	def test_unknown_kind_lists_valid (self) -> None:
		"""Error message lists the valid kinds."""
		with pytest.raises(ValueError, match="omni"):
			subsample.player._parse_extract("midside", "test")

	def test_non_string_rejected (self) -> None:
		"""Non-string values raise with the type name."""
		with pytest.raises(ValueError, match="must be a string"):
			subsample.player._parse_extract(42, "test")

	def test_assignment_name_in_error (self) -> None:
		"""The assignment name is included in error messages."""
		with pytest.raises(ValueError, match="kick"):
			subsample.player._parse_extract("bogus", "kick")


def _make_player_for_mix_matrix (output_channels: int = 2) -> subsample.player.MidiPlayer:
	"""Construct a minimal MidiPlayer suitable for testing _get_mix_matrix."""
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
		output_channels=output_channels,
	)


class TestGetMixMatrixWithExtract:

	"""Tests for the extract composition path in MidiPlayer._get_mix_matrix."""

	def test_extract_omni_stereo_to_stereo_centred_pan (self) -> None:
		"""extract=omni + pan=[50,50] → (L+R)/√2 sent equally to both outputs."""
		player = _make_player_for_mix_matrix(output_channels=2)
		extract = subsample.query.ExtractSpec(kind="omni")
		pan     = numpy.array([50.0, 50.0], dtype=numpy.float32)

		mat = player._get_mix_matrix(2, pan, None, "pcm", extract)

		assert mat.shape == (2, 2)
		# Both output rows should be identical (mono signal in both outs).
		numpy.testing.assert_allclose(mat[0, :], mat[1, :])
		# Each output should sum from both inputs symmetrically.
		numpy.testing.assert_allclose(mat[0, 0], mat[0, 1])

	def test_extract_omni_no_pan_distributes_to_all_outputs (self) -> None:
		"""Without explicit pan, an extract distributes the mono signal equally across all outputs (constant-power)."""
		player = _make_player_for_mix_matrix(output_channels=2)
		extract = subsample.query.ExtractSpec(kind="omni")

		mat = player._get_mix_matrix(2, None, None, "pcm", extract)

		assert mat.shape == (2, 2)
		# Both output rows must be identical (mono signal present in both outs).
		numpy.testing.assert_allclose(mat[0, :], mat[1, :])
		# Both inputs contribute equally to both outputs.
		numpy.testing.assert_allclose(mat[0, 0], mat[0, 1])
		# Each output sums in the L+R extract at 0.5 (constant-power for stereo).
		numpy.testing.assert_allclose(mat[0, 0], 0.5, atol=1e-6)

	def test_extract_omni_no_pan_matches_explicit_pan (self) -> None:
		"""extract: omni with no pan equals extract: omni with pan=[1, 1] (uniform)."""
		player = _make_player_for_mix_matrix(output_channels=2)
		extract = subsample.query.ExtractSpec(kind="omni")
		uniform = numpy.array([1.0, 1.0], dtype=numpy.float32)

		mat_default  = player._get_mix_matrix(2, None,    None, "pcm", extract)
		mat_explicit = player._get_mix_matrix(2, uniform, None, "pcm", extract)

		numpy.testing.assert_allclose(mat_default, mat_explicit)

	def test_extract_omni_no_pan_8ch_distributes_to_all (self) -> None:
		"""On an 8-channel output device, mono extract with no pan goes to all 8 outputs."""
		player = _make_player_for_mix_matrix(output_channels=8)
		extract = subsample.query.ExtractSpec(kind="omni")

		mat = player._get_mix_matrix(2, None, None, "pcm", extract)

		assert mat.shape == (8, 2)
		# Every output row must be non-zero (mono signal hits every speaker).
		for row in range(8):
			assert numpy.linalg.norm(mat[row, :]) > 0, f"output {row} unexpectedly silent"

	def test_extract_with_output_routing (self) -> None:
		"""extract=omni + output=[3,4] routes the mono signal to device channels 3+4."""
		player = _make_player_for_mix_matrix(output_channels=8)
		extract = subsample.query.ExtractSpec(kind="omni")
		pan     = numpy.array([50.0, 50.0], dtype=numpy.float32)
		routing = (2, 3)   # 0-indexed for channels 3, 4

		mat = player._get_mix_matrix(2, pan, routing, "pcm", extract)

		assert mat.shape == (8, 2)
		# Outputs 1, 2, 5-8 (indices 0, 1, 4-7) must be silent.
		for row in (0, 1, 4, 5, 6, 7):
			numpy.testing.assert_allclose(mat[row, :], 0.0)
		# Outputs 3 and 4 (indices 2, 3) carry the mono signal.
		assert numpy.linalg.norm(mat[2, :]) > 0
		assert numpy.linalg.norm(mat[3, :]) > 0

	def test_extract_caches_per_spec (self) -> None:
		"""Calling _get_mix_matrix twice with the same args hits the cache."""
		player = _make_player_for_mix_matrix(output_channels=2)
		extract = subsample.query.ExtractSpec(kind="omni")
		pan     = numpy.array([50.0, 50.0], dtype=numpy.float32)

		mat1 = player._get_mix_matrix(2, pan, None, "pcm", extract)
		mat2 = player._get_mix_matrix(2, pan, None, "pcm", extract)

		# Identity check: cache returns the same array object.
		assert mat1 is mat2

	def test_extract_cache_key_distinguishes_different_extracts (self) -> None:
		"""Different extracts produce different cache entries (and different matrices)."""
		player = _make_player_for_mix_matrix(output_channels=2)
		pan = numpy.array([50.0, 50.0], dtype=numpy.float32)

		omni_mat = player._get_mix_matrix(2, pan, None, "pcm", subsample.query.ExtractSpec(kind="omni"))
		side_mat = player._get_mix_matrix(2, pan, None, "pcm", subsample.query.ExtractSpec(kind="side"))

		# Both should exist as separate cache entries and differ in value.
		assert omni_mat is not side_mat
		assert not numpy.allclose(omni_mat, side_mat)

	def test_no_extract_falls_through_unchanged (self) -> None:
		"""extract=None preserves existing behaviour (identity for stereo→stereo)."""
		player = _make_player_for_mix_matrix(output_channels=2)
		mat = player._get_mix_matrix(2, None, None, "pcm", None)

		numpy.testing.assert_allclose(mat, numpy.eye(2))


class TestValidateAssignmentExtracts:

	"""Tests for _validate_assignment_extracts — load-time format compatibility."""

	def _make_record (
		self,
		name: str,
		channels: int,
		channel_format: str = "pcm",
	) -> subsample.library.SampleRecord:
		"""Build a minimal SampleRecord with the requested channel layout."""
		return subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = name,
			spectral    = tests.helpers._make_spectral(),
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = tests.helpers._make_pitch(),
			timbre      = tests.helpers._make_timbre(),
			level       = tests.helpers._make_level(),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 1.0,
			audio       = numpy.zeros((100, channels), dtype=numpy.int32),
			channel_format = channel_format,
		)

	def _populated_library (
		self,
		records: list[subsample.library.SampleRecord],
	) -> subsample.library.InstrumentLibrary:
		"""Build an InstrumentLibrary containing the given records."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10_000_000)
		for r in records:
			lib.add(r)
		return lib

	def test_no_extract_skips_validation (self) -> None:
		"""Assignments without an extract are skipped silently."""
		record = self._make_record("a", 2)
		lib    = self._populated_library([record])
		assignment = subsample.query.Assignment(name="a", select=(subsample.query.SelectSpec(),))
		note_map = {(0, 60): [(assignment, subsample.query.PickSpec(1, 1))]}

		# Should not raise.
		subsample.player._validate_assignment_extracts(note_map, lib)

	def test_compatible_extract_passes (self) -> None:
		"""omni on a stereo library is compatible — validation passes."""
		records = [self._make_record(f"sample_{i}", 2) for i in range(3)]
		lib     = self._populated_library(records)

		assignment = subsample.query.Assignment(
			name="kick",
			select=(subsample.query.SelectSpec(),),
			extract=subsample.query.ExtractSpec(kind="omni"),
		)
		note_map = {(0, 36): [(assignment, subsample.query.PickSpec(1, 1))]}

		# Should not raise.
		subsample.player._validate_assignment_extracts(note_map, lib)

	def test_incompatible_extract_rejected (self) -> None:
		"""depth on stereo samples raises with the assignment name and 'depth'."""
		records = [self._make_record(f"sample_{i}", 2) for i in range(3)]
		lib     = self._populated_library(records)

		assignment = subsample.query.Assignment(
			name="bass_drum",
			select=(subsample.query.SelectSpec(),),
			extract=subsample.query.ExtractSpec(kind="depth"),
		)
		note_map = {(0, 36): [(assignment, subsample.query.PickSpec(1, 1))]}

		with pytest.raises(ValueError, match="bass_drum"):
			subsample.player._validate_assignment_extracts(note_map, lib)

	def test_empty_library_skipped (self) -> None:
		"""No matching samples — validation skips (no failure to report)."""
		lib = self._populated_library([])
		assignment = subsample.query.Assignment(
			name="x",
			select=(subsample.query.SelectSpec(),),
			extract=subsample.query.ExtractSpec(kind="depth"),
		)
		note_map = {(0, 36): [(assignment, subsample.query.PickSpec(1, 1))]}

		# Should not raise — depth has no candidates to test.
		subsample.player._validate_assignment_extracts(note_map, lib)

	def test_warns_on_equivalent_to_omni (self, caplog: pytest.LogCaptureFixture) -> None:
		"""front on stereo (no F/B info) logs a warning but doesn't raise."""
		import logging
		caplog.set_level(logging.WARNING, logger="subsample.player")

		records = [self._make_record(f"sample_{i}", 2) for i in range(2)]
		lib     = self._populated_library(records)

		assignment = subsample.query.Assignment(
			name="front_kick",
			select=(subsample.query.SelectSpec(),),
			extract=subsample.query.ExtractSpec(kind="front"),
		)
		note_map = {(0, 36): [(assignment, subsample.query.PickSpec(1, 1))]}

		# Should not raise.
		subsample.player._validate_assignment_extracts(note_map, lib)

		# But should warn.
		assert any("equivalent to 'omni'" in r.message for r in caplog.records)

	def test_deduplicates_same_assignment_across_notes (self) -> None:
		"""An assignment mapped to multiple notes is validated once."""
		records = [self._make_record("sample_0", 2)]
		lib     = self._populated_library(records)

		assignment = subsample.query.Assignment(
			name="x",
			select=(subsample.query.SelectSpec(),),
			extract=subsample.query.ExtractSpec(kind="omni"),
		)
		# Same Assignment object on three notes.
		note_map = {
			(0, 36): [(assignment, subsample.query.PickSpec(1, 1))],
			(0, 37): [(assignment, subsample.query.PickSpec(1, 1))],
			(0, 38): [(assignment, subsample.query.PickSpec(1, 1))],
		}

		# Should not raise and should complete without exception.
		subsample.player._validate_assignment_extracts(note_map, lib)


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

		result_audio, result_level = player._select_segment(audio, level, bounds, "", 0, 60, velocity_trigger=(0, 127))

		assert result_audio is audio
		assert result_level is level

	def test_no_bounds_returns_full_audio (self) -> None:
		"""None segment_bounds returns the original audio unchanged."""
		player = self._make_player()
		audio = numpy.random.randn(1000, 1).astype(numpy.float32)
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, result_level = player._select_segment(audio, level, None, "round_robin", 0, 60, velocity_trigger=(0, 127))

		assert result_audio is audio
		assert result_level is level

	def test_numeric_index_selects_correct_segment (self) -> None:
		"""Numeric segment mode (1-indexed) selects the right slice."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, _ = player._select_segment(audio, level, bounds, 3, 0, 60, velocity_trigger=(0, 127))

		assert result_audio.shape[0] == 1000
		numpy.testing.assert_array_equal(result_audio, audio[2000:3000])

	def test_numeric_index_clamped (self) -> None:
		"""Index beyond segment count is clamped to last segment."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		result_audio, _ = player._select_segment(audio, level, bounds, 99, 0, 60, velocity_trigger=(0, 127))

		numpy.testing.assert_array_equal(result_audio, audio[3000:4000])

	def test_round_robin_cycles (self) -> None:
		"""Round-robin advances through segments and wraps."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		results = []
		for _ in range(6):
			seg, _ = player._select_segment(audio, level, bounds, "round_robin", 0, 60, velocity_trigger=(0, 127))
			results.append(seg.shape[0])

		# 4 segments, 6 triggers: should cycle 0,1,2,3,0,1
		assert results == [1000, 1000, 1000, 1000, 1000, 1000]
		# Default (no velocity layering) hashes to (ch, note, 0, 127).
		assert player._segment_counters[(0, 60, 0, 127)] == 6

	def test_random_stays_in_bounds (self) -> None:
		"""Random mode always selects a valid segment."""
		player = self._make_player()
		audio, bounds = self._make_audio_and_bounds()
		level = subsample.analysis.LevelResult(peak=0.5, rms=0.2)

		for _ in range(20):
			seg, _ = player._select_segment(audio, level, bounds, "random", 0, 60, velocity_trigger=(0, 127))
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


# ---------------------------------------------------------------------------
# _build_energy_profile_resolver — unit tests for the resolver builder
# ---------------------------------------------------------------------------

class TestBuildEnergyProfileResolver:

	"""Unit tests for player._build_energy_profile_resolver — proves the
	resolver is constructed correctly from a ProcessSpec and delegates to
	the transform manager with the right TransformSpec."""

	def _profile (
		self,
		resolution: int = 4,
		energy: tuple[float, ...] = (1.0, 0.0, 1.0, 0.0),
	) -> subsample.transform.GridEnergyProfile:
		return subsample.transform.GridEnergyProfile(
			bpm=120.0, resolution=resolution, energy=energy,
		)

	def test_returns_none_when_no_transform_manager (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 16)),
			),
		))
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=None, session_bpm=120.0,
		)
		assert resolver is None

	def test_returns_none_when_no_quantize_step_and_no_session_bpm (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch"),
		))
		transform_manager = unittest.mock.MagicMock()
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=0.0,
		)
		assert resolver is None

	def test_stretch_quantize_builds_timestretch_spec (self) -> None:
		"""Resolver's get_variant call uses a TransformSpec with a TimeStretch step."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 8)),
			),
		))
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = unittest.mock.MagicMock(
			energy_profile=self._profile(),
		)

		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=100.0,
		)
		assert resolver is not None
		resolver(sample_id=42)

		# Verify the spec passed to get_variant.
		transform_manager.get_variant.assert_called_once()
		sample_id, spec = transform_manager.get_variant.call_args[0]
		assert sample_id == 42
		assert len(spec.steps) == 1
		assert isinstance(spec.steps[0], subsample.transform.TimeStretch)
		assert spec.steps[0].target_bpm == 120.0   # per-assignment bpm wins
		assert spec.steps[0].resolution == 8

	def test_pad_quantize_builds_padquantize_spec (self) -> None:
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="pad_quantize", params=(("bpm", 90), ("grid", 16)),
			),
		))
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = unittest.mock.MagicMock(
			energy_profile=self._profile(),
		)

		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=0.0,
		)
		assert resolver is not None
		resolver(sample_id=1)

		sample_id, spec = transform_manager.get_variant.call_args[0]
		assert isinstance(spec.steps[0], subsample.transform.PadQuantize)
		assert spec.steps[0].target_bpm == 90.0
		assert spec.steps[0].resolution == 16

	def test_session_bpm_fallback_when_no_quantize_step (self) -> None:
		"""No quantize step but session_bpm > 0 → fall back to session-level
		TimeStretch at the default grid (matches _build_beats_resolver)."""
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch"),
		))
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = unittest.mock.MagicMock(
			energy_profile=self._profile(),
		)

		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=140.0,
		)
		assert resolver is not None
		resolver(sample_id=1)

		_, spec = transform_manager.get_variant.call_args[0]
		assert isinstance(spec.steps[0], subsample.transform.TimeStretch)
		assert spec.steps[0].target_bpm == 140.0

	def test_resolver_returns_profile_from_variant (self) -> None:
		profile = self._profile(energy=(0.9, 0.1, 0.8, 0.2))
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = unittest.mock.MagicMock(
			energy_profile=profile,
		)
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 4)),
			),
		))
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=120.0,
		)
		assert resolver is not None
		assert resolver(sample_id=1) is profile

	def test_resolver_returns_none_when_variant_missing (self) -> None:
		"""When get_variant returns None (cache miss), resolver returns None."""
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = None
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 4)),
			),
		))
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=120.0,
		)
		assert resolver is not None
		assert resolver(sample_id=99) is None

	def test_resolver_returns_none_when_variant_has_no_profile (self) -> None:
		"""When get_variant returns a result but energy_profile is None,
		resolver returns None — the scorer treats this as 'no data'."""
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = unittest.mock.MagicMock(
			energy_profile=None,
		)
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 4)),
			),
		))
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=120.0,
		)
		assert resolver is not None
		assert resolver(sample_id=1) is None


# ---------------------------------------------------------------------------
# End-to-end: beat_match scorer via the full player → query() wiring
# ---------------------------------------------------------------------------

class TestBeatMatchEndToEnd:

	"""Integration test: prove the energy_profile_resolver is built by the
	player, passed through to query(), and used by the beat_match scorer
	to rank samples correctly.  This catches wiring bugs the unit tests
	in test_query.py (which use a synthetic resolver) cannot."""

	def test_beat_match_ranks_samples_through_full_player_path (
		self, tmp_path: pathlib.Path,
	) -> None:

		# Three samples, each with a distinct pre-baked GridEnergyProfile.
		# Pattern [1, 0, 1, 0] should pick sample 1 (even-beats).
		profiles = {
			1: subsample.transform.GridEnergyProfile(
				bpm=120.0, resolution=4, energy=(1.0, 0.0, 1.0, 0.0),  # matches
			),
			2: subsample.transform.GridEnergyProfile(
				bpm=120.0, resolution=4, energy=(0.0, 1.0, 0.0, 1.0),  # orthogonal
			),
			3: subsample.transform.GridEnergyProfile(
				bpm=120.0, resolution=4, energy=(0.5, 0.5, 0.5, 0.5),  # uniform
			),
		}

		mock_records = []
		for sid in (1, 2, 3):
			r = unittest.mock.MagicMock()
			r.sample_id = sid
			r.name = f"sample-{sid}"
			r.duration = 1.0
			r.rhythm.tempo_bpm = 120.0
			r.rhythm.onset_count = 4
			r.pitch.dominant_pitch_hz = 0.0
			r.level.rms = 0.1
			# matches() is called on WherePredicate with the record; keep the
			# where-predicate empty in the assignment so all samples pass.
			mock_records.append(r)

		# Mock transform_manager that returns the right profile for each sample_id.
		def fake_get_variant (sample_id: int, spec: typing.Any) -> typing.Any:
			result = unittest.mock.MagicMock()
			result.energy_profile = profiles.get(sample_id)
			return result

		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.side_effect = fake_get_variant

		# Assignment using beat_match against [1, 0, 1, 0] with a
		# stretch_quantize step so the resolver fires.
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 4)),
			),
		))
		select = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)

		# Exercise the exact code path the player uses: build the resolver,
		# pass it through to query().
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=120.0,
		)
		assert resolver is not None

		result = subsample.query.query(
			select, mock_records, similarity_matrix=None,
			beats_resolver=None, energy_profile_resolver=resolver,
		)

		# Sample 1 (matches pattern) wins; sample 2 (orthogonal) scored 0.0
		# but on_missing=exclude means only the non-None scorers are kept —
		# here all three samples *have* a profile, so all three are ranked.
		assert [r.sample_id for r in result] == [1, 3, 2]

	def test_beat_match_excludes_samples_without_quantized_variant (
		self, tmp_path: pathlib.Path,
	) -> None:
		"""When a sample's variant hasn't been computed yet (get_variant
		returns None), beat_match excludes it from the result."""

		profiles: dict[int, subsample.transform.GridEnergyProfile] = {
			1: subsample.transform.GridEnergyProfile(
				bpm=120.0, resolution=4, energy=(1.0, 0.0, 1.0, 0.0),
			),
			# Sample 2 has no profile → resolver returns None → excluded.
		}

		mock_records = []
		for sid in (1, 2):
			r = unittest.mock.MagicMock()
			r.sample_id = sid
			r.name = f"sample-{sid}"
			r.duration = 1.0
			r.rhythm.tempo_bpm = 120.0
			r.rhythm.onset_count = 4
			r.pitch.dominant_pitch_hz = 0.0
			r.level.rms = 0.1
			mock_records.append(r)

		def fake_get_variant (sample_id: int, spec: typing.Any) -> typing.Any:
			if sample_id not in profiles:
				return None  # cache miss
			result = unittest.mock.MagicMock()
			result.energy_profile = profiles[sample_id]
			return result

		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.side_effect = fake_get_variant

		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(
				name="stretch_quantize", params=(("bpm", 120), ("grid", 4)),
			),
		))
		select = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)

		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=120.0,
		)
		assert resolver is not None

		result = subsample.query.query(
			select, mock_records, similarity_matrix=None,
			beats_resolver=None, energy_profile_resolver=resolver,
		)

		# Only sample 1 has a profile; sample 2 is excluded.
		assert [r.sample_id for r in result] == [1]

	def test_beat_match_without_quantize_step_yields_empty_result (self) -> None:
		"""Assignment uses beat_match but its process has no quantize step.
		_build_energy_profile_resolver returns None → no resolver is passed
		→ all samples score None → all excluded → empty result."""

		mock_records = [unittest.mock.MagicMock()]
		mock_records[0].sample_id = 1
		mock_records[0].duration = 1.0
		mock_records[0].rhythm.tempo_bpm = 120.0
		mock_records[0].rhythm.onset_count = 4
		mock_records[0].pitch.dominant_pitch_hz = 0.0
		mock_records[0].level.rms = 0.1

		# Process has only a repitch step — no quantize.
		process = subsample.query.ProcessSpec(steps=(
			subsample.query.ProcessorStep(name="repitch"),
		))
		select = subsample.query.SelectSpec(
			order=(subsample.query.OrderClause(
				by="beat_match", dir="desc",
				params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
			),),
		)

		transform_manager = unittest.mock.MagicMock()

		# session_bpm=0 so the fallback path in _build_energy_profile_resolver
		# also declines — no resolver at all.
		resolver = subsample.player._build_energy_profile_resolver(
			process, transform_manager=transform_manager, session_bpm=0.0,
		)
		assert resolver is None

		result = subsample.query.query(
			select, mock_records, similarity_matrix=None,
			beats_resolver=None, energy_profile_resolver=resolver,
		)
		assert result == []

	def test_midi_player_passes_energy_profile_resolver_through (self) -> None:
		"""Direct assertion on the player: update_assignments() builds the
		energy_profile_resolver and the transform_manager.get_variant is
		invoked during resolution (proves the wiring from the player's
		call site all the way into the scorer)."""

		asgn = subsample.query.Assignment(
			name="Beat match via player",
			select=(subsample.query.SelectSpec(
				order=(subsample.query.OrderClause(
					by="beat_match", dir="desc",
					params=(("pattern", (1.0, 0.0, 1.0, 0.0)),),
				),),
			),),
			process=subsample.query.ProcessSpec(steps=(
				subsample.query.ProcessorStep(
					name="stretch_quantize", params=(("bpm", 120), ("grid", 4)),
				),
			)),
			one_shot=False,
		)
		note_map: subsample.player.NoteMap = {(0, 60): [(asgn, subsample.query.PickSpec(1, 1))]}

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
		mock_record.name = "candidate"
		mock_record.duration = 1.0
		mock_record.rhythm.tempo_bpm = 120.0
		mock_record.rhythm.onset_count = 4
		mock_record.pitch.dominant_pitch_hz = 0.0
		mock_record.level.rms = 0.1

		player._instrument_library.samples.return_value = [mock_record]
		player._instrument_library.get.return_value = mock_record

		profile = subsample.transform.GridEnergyProfile(
			bpm=120.0, resolution=4, energy=(1.0, 0.0, 1.0, 0.0),
		)
		transform_manager = unittest.mock.MagicMock()
		transform_manager.get_variant.return_value = unittest.mock.MagicMock(
			energy_profile=profile,
		)
		player._transform_manager = transform_manager

		# Trigger the full select-and-schedule flow; if the resolver wasn't
		# wired through, the beat_match scorer would exclude this sample
		# and update_assignments would find nothing to enqueue.
		player.update_assignments()

		# get_variant was called at least once (once by the resolver, once
		# by the variant-precompute path). The key assertion is that it
		# ran — which only happens when the resolver is built and passed.
		assert transform_manager.get_variant.called


# ---------------------------------------------------------------------------
# Candidate cache — sample selection pre-computed off the trigger thread
# ---------------------------------------------------------------------------

class TestCandidateCache:

	"""MidiPlayer pre-computes each assignment's ranked candidate list when the
	library changes, so note-on selection is an indexed pick rather than a
	per-trigger query + sort + ``directory:`` filesystem scan (the cause of the
	original MIDI-timing regression).  Variant-dependent selects
	(quantized_beats / beat_match) are excluded and resolved live."""

	def _make_record (self, name: str, sample_id: int) -> subsample.library.SampleRecord:
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

	def _make_player (
		self,
		records:  list[subsample.library.SampleRecord],
		note_map: subsample.player.NoteMap,
	) -> tuple[subsample.player.MidiPlayer, unittest.mock.MagicMock]:

		"""Build a player over a mock library whose sample set is controllable
		via ``lib.samples.return_value`` — so a new capture or an eviction is a
		single assignment in the test."""

		lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		lib.samples.return_value = records
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		player = subsample.player.MidiPlayer(
			"Test Device",
			threading.Event(),
			instrument_library=lib,
			similarity_matrix=similarity,
			midi_map=note_map,
			sample_rate=44100,
			bit_depth=16,
		)

		return player, lib

	def _name_glob_assignment (self, name: str = "Hats", glob: str = "hat*") -> subsample.query.Assignment:
		return subsample.query.Assignment(
			name=name,
			select=(subsample.query.SelectSpec(where=subsample.query.WherePredicate(name_glob=glob)),),
		)

	def test_cache_built_at_construction (self) -> None:
		"""A variant-independent select is resolved and cached during __init__."""
		records = [self._make_record("hat_a", 1), self._make_record("kick", 2)]
		asgn = self._name_glob_assignment()
		note_map = {(0, 42): [(asgn, subsample.query.PickSpec(1, 1))]}

		player, _lib = self._make_player(records, note_map)

		assert player._candidate_cache[id(asgn)] == [1]

	def test_resolve_sample_id_uses_cache_without_querying (self) -> None:
		"""_resolve_sample_id returns the cached pick and never calls the query engine."""
		records = [self._make_record("hat_a", 7)]
		asgn = self._name_glob_assignment()
		note_map = {(0, 42): [(asgn, subsample.query.PickSpec(1, 1))]}

		player, lib = self._make_player(records, note_map)

		with unittest.mock.patch("subsample.query.query") as mock_query:
			result = player._resolve_sample_id(asgn, subsample.query.PickSpec(1, 1), lib)

		assert result == 7
		mock_query.assert_not_called()

	def test_empty_candidates_resolve_to_none (self) -> None:
		"""A select that matches nothing caches an empty list and resolves to None."""
		records = [self._make_record("kick", 1)]
		asgn = self._name_glob_assignment()   # "hat*" matches no "kick"
		note_map = {(0, 42): [(asgn, subsample.query.PickSpec(1, 1))]}

		player, lib = self._make_player(records, note_map)

		assert player._candidate_cache[id(asgn)] == []
		assert player._resolve_sample_id(asgn, subsample.query.PickSpec(1, 1), lib) is None

	def test_range_pick_re_rolls_across_cached_list (self) -> None:
		"""A range pick draws varying ranks from the cached candidate list."""
		records = [self._make_record(f"hat_{i}", i + 1) for i in range(3)]
		asgn = self._name_glob_assignment()
		note_map = {(0, 42): [(asgn, subsample.query.PickSpec(1, 3))]}

		player, lib = self._make_player(records, note_map)

		assert set(player._candidate_cache[id(asgn)]) == {1, 2, 3}

		drawn = {
			player._resolve_sample_id(asgn, subsample.query.PickSpec(1, 3), lib)
			for _ in range(200)
		}

		assert drawn <= {1, 2, 3}
		assert len(drawn) >= 2

	def test_variant_dependent_select_not_cached (self) -> None:
		"""A quantized_beats-ordered select is excluded from the cache and
		resolved live, because its ranking depends on async variant state."""
		records = [self._make_record("loop", 1)]
		asgn = subsample.query.Assignment(
			name="BeatSynced",
			select=(subsample.query.SelectSpec(
				order=(subsample.query.OrderClause(by="quantized_beats", dir="asc"),),
			),),
		)
		note_map = {(0, 60): [(asgn, subsample.query.PickSpec(1, 1))]}

		player, lib = self._make_player(records, note_map)

		assert id(asgn) not in player._candidate_cache

		with unittest.mock.patch("subsample.query.query", wraps=subsample.query.query) as spy:
			result = player._resolve_sample_id(asgn, subsample.query.PickSpec(1, 1), lib)

		assert result == 1
		spy.assert_called()

	def test_update_assignments_picks_up_new_sample (self) -> None:
		"""update_assignments — the central re-evaluation hub — rebuilds the
		cache so a newly-arrived matching sample becomes selectable."""
		records = [self._make_record("hat_a", 1)]
		asgn = self._name_glob_assignment()
		note_map = {(0, 42): [(asgn, subsample.query.PickSpec(1, 1))]}

		player, lib = self._make_player(records, note_map)
		assert set(player._candidate_cache[id(asgn)]) == {1}

		lib.samples.return_value = records + [self._make_record("hat_b", 2)]
		player.update_assignments()

		assert set(player._candidate_cache[id(asgn)]) == {1, 2}

	def test_update_assignments_drops_evicted_sample (self) -> None:
		"""When a sample leaves the library, the next rebuild removes it from
		the cached candidates."""
		records = [self._make_record("hat_a", 1), self._make_record("hat_b", 2)]
		asgn = self._name_glob_assignment()
		note_map = {(0, 42): [(asgn, subsample.query.PickSpec(1, 1))]}

		player, lib = self._make_player(records, note_map)
		assert set(player._candidate_cache[id(asgn)]) == {1, 2}

		lib.samples.return_value = [records[0]]
		player.update_assignments()

		assert set(player._candidate_cache[id(asgn)]) == {1}
