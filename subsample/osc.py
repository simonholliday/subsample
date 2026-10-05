"""OSC (Open Sound Control) sender and receiver for inter-app communication.

Sends sample events when recordings are captured or library samples are loaded,
and optionally receives /sample/import messages to load audio files into the
in-memory instrument library from other OSC-compatible applications.  Files
are read in place (not copied); the sample is available for playback until
the next restart.

A second receiver, OscNoteReceiver, takes /note/on and /note/off messages and
hands each note to the player with the time its bundle names (#3610, #603).

Requires the optional ``python-osc`` dependency::

    pip install subsample[osc]

The module is safe to import without python-osc installed — the classes guard
their constructor bodies with a lazy import so the ImportError surfaces only
when an instance is actually created.
"""

import logging
import pathlib
import queue
import socketserver
import threading
import time
import typing

import numpy

import subsample.analysis
import subsample.library
import subsample.loopfind


_log = logging.getLogger(__name__)

# How many /sample/import messages may wait while one is being analysed.  Each
# import analyses a whole file, so a burst beyond this is refused with a warning
# rather than queued without limit.
_IMPORT_QUEUE_LIMIT: int = 64

# How long stop() waits for the import being analysed when it is called.
_STOP_WAIT_SECONDS: float = 10.0


class OscEventSender:

	"""Send OSC messages when samples are captured or loaded.

	Forwards musically relevant fields (pitch, tempo, duration) to a
	configurable host:port so that sequencers, visualisers, or other
	OSC-compatible applications can react in real time.

	The public on_sample_captured / on_sample_loaded callbacks are
	send-exception-safe — a failed UDP send logs a warning and never raises,
	since they run on worker threads.  (The internal event-bus adapters that
	unpack kwargs are not wrapped; they assume the emitter supplies the
	documented keys.)
	"""

	def __init__ (self, host: str = "127.0.0.1", port: int = 9000) -> None:

		"""Send to host:port over UDP; python-osc is needed from here."""

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
		*,
		channel_format: str = "pcm",
		loop: typing.Optional[subsample.loopfind.LoopPoints] = None,
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

	def on_sample_captured_event (self, **kwargs: typing.Any) -> None:

		"""Event handler for sample_captured — unpacks kwargs to on_complete().

		Used as the event subscription target so the event emitter's
		**kwargs dispatch is compatible with on_complete()'s positional
		signature.
		"""

		self.on_complete(
			kwargs["filepath"], kwargs["spectral"], kwargs["rhythm"],
			kwargs["pitch"], kwargs["timbre"], kwargs["level"],
			kwargs["band_energy"], kwargs["duration"], kwargs["audio"],
		)

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

	def on_sample_loaded_event (self, **kwargs: typing.Any) -> None:

		"""Event handler for sample_loaded — unpacks kwargs to on_sample_loaded().

		Used as the event subscription target so the event emitter's
		**kwargs dispatch is compatible with on_sample_loaded()'s positional
		signature.
		"""

		self.on_sample_loaded(kwargs["record"])


class OscReceiver:

	"""Listen for /sample/import OSC messages and import each file in turn.

	A UDP server on one daemon thread takes each message and queues its path;
	one import worker, on another, hands the paths to the on_import callback
	one at a time, in the order they arrived.  A thread per message, as before,
	analysed every file of a burst at once with no limit, and stop() could not
	wait for them.  At most _IMPORT_QUEUE_LIMIT paths wait; one past that is
	refused with a warning.

	SECURITY: ``/sample/import`` reads/loads an arbitrary filesystem path
	supplied by the sender, so the server binds to loopback (127.0.0.1) by
	default — only local senders can reach it.  Pass host="0.0.0.0" (via
	``osc.receive_host`` in config.yaml) to accept messages from other hosts
	on a trusted LAN, understanding that this exposes unauthenticated remote
	file read/load.
	"""

	def __init__ (
		self,
		port: int,
		on_import: typing.Callable[[str], None],
		host: str = "127.0.0.1",
	) -> None:

		"""Bind the UDP socket now, so a busy port fails at construction."""

		import pythonosc.dispatcher
		import pythonosc.osc_server

		dispatcher = pythonosc.dispatcher.Dispatcher()
		dispatcher.map("/sample/import", self._handle_import)

		self._on_import = on_import
		self._server = pythonosc.osc_server.BlockingOSCUDPServer(
			(host, port), dispatcher,
		)

		# None is the worker's signal to stop.
		self._imports: queue.Queue[typing.Optional[str]] = queue.Queue(maxsize=_IMPORT_QUEUE_LIMIT)
		self._thread: typing.Optional[threading.Thread] = None
		self._worker: typing.Optional[threading.Thread] = None

	def start (self) -> None:

		"""Launch the OSC server and the import worker on daemon threads."""

		self._worker = threading.Thread(
			target=self._work,
			name="osc-import",
			daemon=True,
		)
		self._worker.start()

		self._thread = threading.Thread(
			target=self._server.serve_forever,
			name="osc-receiver",
			daemon=True,
		)
		self._thread.start()
		_log.info("OSC receiver listening on port %d", self._server.server_address[1])

	def stop (self) -> None:

		"""Stop listening, drop the imports still waiting, and wait for the one running.

		The wait is bounded by _STOP_WAIT_SECONDS: a first-time analysis of a
		long file can take longer, and shutdown should not hang on a file the
		musician did not ask to keep.  An import still running after that is
		left to its daemon thread, and the caller should not act on it.
		"""

		# Guard against stop() before start(): BaseServer.shutdown() waits on
		# an event only serve_forever() sets, so it would block forever.
		if self._thread is None or self._worker is None:
			self._server.server_close()
			return

		self._server.shutdown()
		self._thread.join(timeout=5.0)

		# shutdown() only stops the serve loop; server_close() releases the
		# bound UDP socket so a later start on the same port can rebind.
		self._server.server_close()

		dropped = 0

		while True:
			try:
				self._imports.get_nowait()
			except queue.Empty:
				break

			dropped += 1

		if dropped:
			_log.info("OSC receiver stopped with %d import(s) still waiting - not imported", dropped)

		self._imports.put(None)
		self._worker.join(timeout=_STOP_WAIT_SECONDS)

		if self._worker.is_alive():
			_log.warning("OSC receiver stopped while an import was still being analysed - it is abandoned")

		_log.debug("OSC receiver stopped")

	def _handle_import (self, address: str, *args: typing.Any) -> None:

		"""Queue a /sample/import message's path for the import worker."""

		if not args:
			_log.warning("OSC /sample/import received with no arguments - ignoring")
			return

		file_path = str(args[0])

		try:
			self._imports.put_nowait(file_path)
		except queue.Full:
			_log.warning(
				"OSC /sample/import: %d imports are already waiting - ignoring %s",
				_IMPORT_QUEUE_LIMIT, file_path,
			)
			return

		_log.info("OSC /sample/import: %s", file_path)

	def _work (self) -> None:

		"""Import each queued path in turn, until stop() sends None."""

		while True:
			file_path = self._imports.get()

			if file_path is None:
				return

			try:
				self._on_import(file_path)
			except Exception:
				_log.warning("OSC /sample/import handler failed for %s", file_path, exc_info=True)


# How often the note receiver repeats a warning about one kind of bad message.
# A sender with a mistake sends it on every note, and once a minute says it.
_NOTE_WARNING_SECONDS: float = 60.0


class _NoteRequestHandler (socketserver.BaseRequestHandler):

	"""Hand one UDP packet to the note receiver that owns the server."""

	def handle (self) -> None:

		"""Pass the packet's bytes on; the receiver reads the notes in it."""

		server = typing.cast(_NoteServer, self.server)
		server.receiver._handle_packet(self.request[0])


class _NoteServer (socketserver.UDPServer):

	"""A UDP server that knows the note receiver it serves."""

	def __init__ (self, address: tuple[str, int], receiver: "OscNoteReceiver") -> None:

		"""Bind the socket now, so a busy port fails at construction."""

		self.receiver = receiver
		super().__init__(address, _NoteRequestHandler)


class OscNoteReceiver:

	"""Listen for /note/on and /note/off, and hand each note on with the time it is meant for.

	#3610 and #603, as #4513 decided them: ``/note/on <channel 1-16> <note
	0-127> <velocity 0.0-1.0>`` and ``/note/off <channel> <note>``.  A message
	in a bundle is meant for the bundle's time; one on its own, or one whose
	time has passed, for when it arrived.  on_note receives (on, channel 0-15,
	note, velocity, when), ``when`` in seconds since the epoch, as a timetag
	reads, and the player places the note one buffer after it, as it would a
	MIDI note arriving then.

	Each packet is read here rather than through python-osc's dispatcher,
	which, for a bundle timed ahead, sleeps on the server thread until the
	time comes and then hands the message on without it.  A note-on's work is
	short, so on_note runs on the server thread and notes keep their order.

	A note can only play a sound, so binding beyond loopback exposes nothing
	more than that, unlike OscReceiver's arbitrary-path import.
	"""

	def __init__ (
		self,
		port:    int,
		on_note: typing.Callable[[bool, int, int, float, float], None],
		host:    str = "127.0.0.1",
	) -> None:

		"""Bind the UDP socket now, so a busy port fails at construction; python-osc is needed from here."""

		import pythonosc.osc_packet

		self._packet      = pythonosc.osc_packet.OscPacket
		self._parse_error = pythonosc.osc_packet.ParseError

		self._on_note = on_note
		self._server  = _NoteServer((host, port), self)
		self._thread: typing.Optional[threading.Thread] = None

		# When each kind of bad message was last warned about.
		self._warned: dict[str, float] = {}

	def start (self) -> None:

		"""Launch the server on a daemon thread."""

		self._thread = threading.Thread(
			target=self._server.serve_forever,
			name="osc-notes",
			daemon=True,
		)
		self._thread.start()
		_log.info("OSC note receiver listening on port %d", self._server.server_address[1])

	def stop (self) -> None:

		"""Stop listening and release the port."""

		# shutdown() waits on an event only serve_forever() sets, so it would
		# block forever before start().
		if self._thread is not None:
			self._server.shutdown()
			self._thread.join(timeout=5.0)

		self._server.server_close()
		_log.debug("OSC note receiver stopped")

	def _handle_packet (self, data: bytes) -> None:

		"""Read every message in one packet, a bundle's in time order, and play each."""

		try:
			packet = self._packet(data)
		except self._parse_error:
			self._warn("packet", "OSC note receiver: a packet that is not OSC - ignored")
			return

		for timed in packet.messages:
			try:
				self._handle(timed.message.address, list(timed.message.params), timed.time)
			except Exception:
				_log.warning("OSC note handler failed for %s", timed.message.address, exc_info=True)

	def _handle (self, address: str, params: list[typing.Any], when: float) -> None:

		"""Check one message's arguments and pass its note on, or say what is wrong with it."""

		if address == "/note/on":
			if len(params) != 3:
				self._warn(address, "OSC /note/on needs a channel, a note and a velocity - ignored")
				return

			channel, note, velocity = _channel(params[0]), _note(params[1]), _velocity(params[2])

			if channel is None or note is None or velocity is None:
				self._warn(
					address,
					f"OSC /note/on {params}: the channel must be 1 to 16, the note 0 to "
					f"127 and the velocity a decimal from 0 to 1 - ignored",
				)
				return

			self._on_note(True, channel, note, velocity, when)
			return

		if address == "/note/off":
			if len(params) != 2:
				self._warn(address, "OSC /note/off needs a channel and a note - ignored")
				return

			channel, note = _channel(params[0]), _note(params[1])

			if channel is None or note is None:
				self._warn(address, f"OSC /note/off {params}: the channel must be 1 to 16 and the note 0 to 127 - ignored")
				return

			self._on_note(False, channel, note, 0.0, when)
			return

		_log.debug("OSC note receiver: %s is not a note address - ignored", address)

	def _warn (self, kind: str, message: str) -> None:

		"""Log a warning about one kind of bad message, at most once a minute."""

		now = time.monotonic()

		if now - self._warned.get(kind, -_NOTE_WARNING_SECONDS) < _NOTE_WARNING_SECONDS:
			return

		self._warned[kind] = now
		_log.warning("%s", message)


def _whole (value: typing.Any, lowest: int, highest: int) -> typing.Optional[int]:

	"""A whole number in [lowest, highest], from an int or a float with nothing after the point, else None."""

	if isinstance(value, bool):
		return None

	if isinstance(value, float) and value.is_integer():
		value = int(value)

	if not isinstance(value, int) or not lowest <= value <= highest:
		return None

	return value


def _channel (value: typing.Any) -> typing.Optional[int]:

	"""A MIDI channel as a message writes it, 1 to 16, as the player numbers it, 0 to 15."""

	channel = _whole(value, 1, 16)

	return None if channel is None else channel - 1


def _note (value: typing.Any) -> typing.Optional[int]:

	"""A MIDI note number, 0 to 127."""

	return _whole(value, 0, 127)


def _velocity (value: typing.Any) -> typing.Optional[float]:

	"""A velocity from 0 to 1, as a decimal or as 0 or 1."""

	if isinstance(value, bool) or not isinstance(value, (int, float)):
		return None

	velocity = float(value)

	return velocity if 0.0 <= velocity <= 1.0 else None
