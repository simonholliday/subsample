"""Tests for subsample.osc — OSC sender and receiver."""

import pathlib
import threading
import time
import unittest.mock

import numpy
import pytest

import subsample.analysis
import subsample.config
import subsample.library
import subsample.osc

import tests.helpers


# ---------------------------------------------------------------------------
# OscEventSender tests
# ---------------------------------------------------------------------------

class TestOscEventSender:

	def test_on_complete_sends_captured (self) -> None:
		"""on_complete sends /sample/captured with correct arguments."""

		sender = subsample.osc.OscEventSender("127.0.0.1", 9000)
		sender._client = unittest.mock.MagicMock()

		filepath = pathlib.Path("/tmp/test.wav")
		spectral = tests.helpers._make_spectral()
		rhythm = tests.helpers._make_rhythm()
		pitch = tests.helpers._make_pitch(dominant_pitch_hz=440.0, dominant_pitch_class=9)
		timbre = tests.helpers._make_timbre()
		level = tests.helpers._make_level()
		band_energy = tests.helpers._make_band_energy()
		duration = 2.5
		audio = numpy.zeros((1024, 1), dtype=numpy.int16)

		sender.on_complete(filepath, spectral, rhythm, pitch, timbre, level, band_energy, duration, audio)

		sender._client.send_message.assert_called_once_with(
			"/sample/captured",
			[str(filepath), 2.5, 440.0, 9, 120.0, 2],
		)

	def test_on_sample_loaded_sends_loaded (self) -> None:
		"""on_sample_loaded sends /sample/loaded with correct arguments."""

		sender = subsample.osc.OscEventSender("127.0.0.1", 9000)
		sender._client = unittest.mock.MagicMock()

		record = subsample.library.SampleRecord(
			sample_id   = 1,
			name        = "kick_01",
			spectral    = tests.helpers._make_spectral(),
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = tests.helpers._make_pitch(dominant_pitch_hz=80.0, dominant_pitch_class=4),
			timbre      = tests.helpers._make_timbre(),
			level       = tests.helpers._make_level(),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 0.5,
			audio       = numpy.zeros((512, 1), dtype=numpy.int16),
		)

		sender.on_sample_loaded(record)

		sender._client.send_message.assert_called_once_with(
			"/sample/loaded",
			["kick_01", 0.5, 80.0, 4],
		)

	def test_on_complete_logs_warning_on_send_failure (self) -> None:
		"""A failed send logs a warning but does not raise."""

		sender = subsample.osc.OscEventSender("127.0.0.1", 9000)
		sender._client = unittest.mock.MagicMock()
		sender._client.send_message.side_effect = OSError("send failed")

		filepath = pathlib.Path("/tmp/test.wav")

		# Should not raise.
		sender.on_complete(
			filepath,
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(),
			tests.helpers._make_timbre(),
			tests.helpers._make_level(),
			tests.helpers._make_band_energy(),
			1.0,
			numpy.zeros((1024, 1), dtype=numpy.int16),
		)

	def test_on_sample_loaded_logs_warning_on_send_failure (self) -> None:
		"""A failed send on on_sample_loaded does not raise."""

		sender = subsample.osc.OscEventSender("127.0.0.1", 9000)
		sender._client = unittest.mock.MagicMock()
		sender._client.send_message.side_effect = OSError("send failed")

		record = subsample.library.SampleRecord(
			sample_id   = 1,
			name        = "test",
			spectral    = tests.helpers._make_spectral(),
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = tests.helpers._make_pitch(),
			timbre      = tests.helpers._make_timbre(),
			level       = tests.helpers._make_level(),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 1.0,
		)

		# Should not raise.
		sender.on_sample_loaded(record)

	def test_on_complete_unpitched_sends_minus_one (self) -> None:
		"""Unpitched samples send pitch_class=-1 and pitch_hz=0.0."""

		sender = subsample.osc.OscEventSender("127.0.0.1", 9000)
		sender._client = unittest.mock.MagicMock()

		sender.on_complete(
			pathlib.Path("/tmp/noise.wav"),
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(dominant_pitch_hz=0.0, dominant_pitch_class=-1),
			tests.helpers._make_timbre(),
			tests.helpers._make_level(),
			tests.helpers._make_band_energy(),
			3.0,
			numpy.zeros((1024, 1), dtype=numpy.int16),
		)

		args = sender._client.send_message.call_args[0]
		assert args[1][2] == 0.0   # pitch_hz
		assert args[1][3] == -1    # pitch_class


# ---------------------------------------------------------------------------
# OscReceiver tests
# ---------------------------------------------------------------------------

class TestOscReceiver:

	def test_start_stop_lifecycle (self) -> None:
		"""Receiver starts and stops cleanly."""

		callback = unittest.mock.MagicMock()
		shutdown = threading.Event()

		receiver = subsample.osc.OscReceiver(
			port=19200,
			on_import=callback,
			shutdown_event=shutdown,
		)

		receiver.start()
		assert receiver._thread is not None
		assert receiver._thread.is_alive()

		receiver.stop()
		assert not receiver._thread.is_alive()

	def test_import_message_calls_callback (self) -> None:
		"""A /sample/import message invokes the on_import callback."""

		import pythonosc.udp_client

		callback = unittest.mock.MagicMock()
		shutdown = threading.Event()

		receiver = subsample.osc.OscReceiver(
			port=19201,
			on_import=callback,
			shutdown_event=shutdown,
		)
		receiver.start()

		try:
			# Give the server a moment to bind.
			time.sleep(0.1)

			client = pythonosc.udp_client.SimpleUDPClient("127.0.0.1", 19201)
			client.send_message("/sample/import", ["/tmp/some_audio.wav"])

			# Wait for the message to be processed.
			time.sleep(0.3)

			callback.assert_called_once_with("/tmp/some_audio.wav")

		finally:
			receiver.stop()

	def test_import_no_args_does_not_call_callback (self) -> None:
		"""A /sample/import with no arguments logs a warning, not a crash."""

		import pythonosc.udp_client

		callback = unittest.mock.MagicMock()
		shutdown = threading.Event()

		receiver = subsample.osc.OscReceiver(
			port=19202,
			on_import=callback,
			shutdown_event=shutdown,
		)
		receiver.start()

		try:
			time.sleep(0.1)

			client = pythonosc.udp_client.SimpleUDPClient("127.0.0.1", 19202)
			client.send_message("/sample/import", [])

			time.sleep(0.3)

			callback.assert_not_called()

		finally:
			receiver.stop()

	def test_callback_exception_does_not_crash_server (self) -> None:
		"""An exception in the on_import callback is caught and logged."""

		import pythonosc.udp_client

		callback = unittest.mock.MagicMock(side_effect=RuntimeError("boom"))
		shutdown = threading.Event()

		receiver = subsample.osc.OscReceiver(
			port=19203,
			on_import=callback,
			shutdown_event=shutdown,
		)
		receiver.start()

		try:
			time.sleep(0.1)

			client = pythonosc.udp_client.SimpleUDPClient("127.0.0.1", 19203)
			client.send_message("/sample/import", ["/tmp/fail.wav"])

			time.sleep(0.3)

			# Callback was called (and raised), but server is still alive.
			callback.assert_called_once()
			assert receiver._thread is not None
			assert receiver._thread.is_alive()

		finally:
			receiver.stop()


# ---------------------------------------------------------------------------
# OscConfig loading tests
# ---------------------------------------------------------------------------

class TestOscConfig:

	def test_default_config_has_osc_disabled (self) -> None:
		"""Default config produces OscConfig with enabled=False."""

		cfg = subsample.config.load_config(
			pathlib.Path(__file__).parent.parent / "config.yaml.default"
		)

		assert isinstance(cfg.osc, subsample.config.OscConfig)
		assert cfg.osc.enabled is False
		assert cfg.osc.send_host == "127.0.0.1"
		assert cfg.osc.send_port == 9000
		assert cfg.osc.receive_enabled is False
		assert cfg.osc.receive_port == 9002

	def test_explicit_osc_yaml_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Explicit osc YAML section is parsed correctly."""

		import shutil
		import subsample.config

		# Copy the default config so we have a valid base.
		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		# Append OSC overrides.
		with user_config.open("a") as fh:
			fh.write(
				"\nosc:\n"
				"  enabled: true\n"
				"  send_host: \"192.168.1.10\"\n"
				"  send_port: 8000\n"
				"  receive_enabled: true\n"
				"  receive_port: 9999\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.osc.enabled is True
		assert cfg.osc.send_host == "192.168.1.10"
		assert cfg.osc.send_port == 8000
		assert cfg.osc.receive_enabled is True
		assert cfg.osc.receive_port == 9999
