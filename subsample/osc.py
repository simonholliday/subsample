"""OSC (Open Sound Control) sender and receiver for inter-app communication.

Sends sample events when recordings are captured or library samples are loaded,
and optionally receives /sample/import messages to load audio files into the
in-memory instrument library from other OSC-compatible applications.  Files
are read in place (not copied); the sample is available for playback until
the next restart.

Requires the optional ``python-osc`` dependency::

    pip install subsample[osc]

The module is safe to import without python-osc installed — the classes guard
their constructor bodies with a lazy import so the ImportError surfaces only
when an instance is actually created.
"""

import logging
import pathlib
import threading
import typing

import numpy

import subsample.analysis
import subsample.library


_log = logging.getLogger(__name__)


class OscEventSender:

	"""Send OSC messages when samples are captured or loaded.

	Forwards musically relevant fields (pitch, tempo, duration) to a
	configurable host:port so that sequencers, visualisers, or other
	OSC-compatible applications can react in real time.

	Both callback methods are exception-safe — a failed send logs a
	warning and never raises, since callbacks run on worker threads.
	"""

	def __init__ (self, host: str = "127.0.0.1", port: int = 9000) -> None:

		import pythonosc.udp_client

		self._client = pythonosc.udp_client.SimpleUDPClient(host, port)

	def on_complete (
		self,
		filepath: pathlib.Path,
		spectral: subsample.analysis.AnalysisResult,
		rhythm: subsample.analysis.RhythmResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		level: subsample.analysis.LevelResult,
		band_energy: subsample.analysis.BandEnergyResult,
		duration: float,
		audio: numpy.ndarray,
	) -> None:

		"""Send /sample/captured when a new recording completes analysis.

		Has the same signature as the recorder's _OnCompleteCallback type
		so it can be chained alongside the existing on_complete callback.
		"""

		try:
			self._client.send_message("/sample/captured", [
				str(filepath),
				float(duration),
				float(pitch.dominant_pitch_hz),
				int(pitch.dominant_pitch_class),
				float(rhythm.tempo_bpm),
				int(rhythm.onset_count),
			])
		except Exception:
			_log.warning("OSC send failed for /sample/captured (%s)", filepath.name, exc_info=True)

	def on_sample_loaded (self, record: subsample.library.SampleRecord) -> None:

		"""Send /sample/loaded when a sample is added to the instrument library."""

		try:
			self._client.send_message("/sample/loaded", [
				record.name,
				float(record.duration),
				float(record.pitch.dominant_pitch_hz),
				int(record.pitch.dominant_pitch_class),
			])
		except Exception:
			_log.warning("OSC send failed for /sample/loaded (%s)", record.name, exc_info=True)


class OscReceiver:

	"""Listen for /sample/import OSC messages and trigger file import.

	Runs a threaded UDP server on a daemon thread.  The on_import callback
	is invoked from the server's handler thread with the file path string
	extracted from the OSC message arguments.
	"""

	def __init__ (
		self,
		port: int,
		on_import: typing.Callable[[str], None],
		shutdown_event: threading.Event,
	) -> None:

		import pythonosc.dispatcher
		import pythonosc.osc_server

		self._shutdown_event = shutdown_event

		dispatcher = pythonosc.dispatcher.Dispatcher()
		dispatcher.map("/sample/import", self._handle_import)

		self._on_import = on_import
		self._server = pythonosc.osc_server.ThreadingOSCUDPServer(
			("0.0.0.0", port), dispatcher,
		)
		self._thread: typing.Optional[threading.Thread] = None

	def start (self) -> None:

		"""Launch the OSC server on a daemon thread."""

		self._thread = threading.Thread(
			target=self._server.serve_forever,
			name="osc-receiver",
			daemon=True,
		)
		self._thread.start()
		_log.info("OSC receiver listening on port %d", self._server.server_address[1])

	def stop (self) -> None:

		"""Shut down the OSC server and wait for the thread to exit."""

		self._server.shutdown()

		if self._thread is not None:
			self._thread.join(timeout=5.0)

		_log.debug("OSC receiver stopped")

	def _handle_import (self, address: str, *args: typing.Any) -> None:

		"""Dispatch a /sample/import message to the on_import callback."""

		if not args:
			_log.warning("OSC /sample/import received with no arguments — ignoring")
			return

		file_path = str(args[0])
		_log.info("OSC /sample/import: %s", file_path)

		try:
			self._on_import(file_path)
		except Exception:
			_log.warning("OSC /sample/import handler failed for %s", file_path, exc_info=True)
