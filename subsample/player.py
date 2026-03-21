"""MIDI input listener and sample player for Subsample.

Handles MIDI input device selection (replicating the audio device selection
pattern from audio.py) and the MidiPlayer class which listens for MIDI events
and triggers polyphonic audio playback.

Mixing architecture: a PyAudio callback stream requests N frames at regular
intervals. Each triggered note adds a _Voice (pre-rendered stereo float32
audio + playback cursor) to a shared list. The callback sums all active
voices into one output buffer, clips, converts to int16, and returns it.
The MIDI polling loop adds voices under a lock; the callback reads them.

Current implementation: exploratory / hard-coded. Velocities, MIDI channel,
note range, and target RMS are hard-coded constants at the top of this module
and will be moved to config in a future iteration.
"""

import dataclasses
import logging
import threading
import typing

import mido
import numpy
import pyaudio

import subsample.analysis
import subsample.audio
import subsample.library
import subsample.similarity


_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HARD-CODED playback constants — temporary while we explore output levels.
# These will become config fields in a future iteration.
# ---------------------------------------------------------------------------

# MIDI channel to listen on. Mido uses 0-indexed channels; the BeatStep Pro
# sends on channel 16 (user-facing), which mido reports as channel=15.
_MIDI_CHANNEL = 15          # channel 16 in user-facing terms (mido: 0-indexed)

# MIDI note range mapped to reference samples (first note = lowest note number).
_NOTE_FIRST = 36            # MIDI note 36 = first reference sample (alphabetical)
_NOTE_LAST  = 51            # MIDI note 51 = 16th reference (or last if fewer)

# Fixed MIDI velocity used for every trigger. Input velocity is ignored.
_HARD_CODED_VELOCITY = 100  # TODO: replace with msg.velocity from MIDI input

# Target RMS level for playback normalisation (linear, not dBFS).
# 0.1 ≈ -20 dBFS — leaves headroom for mixing multiple simultaneous voices.
_TARGET_RMS = 0.1           # TODO: move to cfg.player.target_rms

# Output always stereo with centre pan (equal L and R).
# TODO: add per-note pan mapping
_OUTPUT_CHANNELS = 2        # stereo; mono signal duplicated to both channels


@dataclasses.dataclass
class _Voice:

	"""A single triggered sample being played back by the mix callback.

	audio:    Pre-rendered stereo float32 array, shape (n_frames, 2), in
	          [-1.0, 1.0]. Gain has already been applied. The callback reads
	          from this array; it is never modified after creation.
	position: Current read cursor in frames. Advances each callback call.
	          Voice is removed when position >= len(audio).
	"""

	audio:    numpy.ndarray
	position: int = 0


def list_midi_input_devices () -> list[str]:

	"""Return the names of all available MIDI input devices.

	Uses mido's default backend (rtmidi). Returns an empty list if no
	MIDI devices are connected or the backend is unavailable.
	"""

	return list(mido.get_input_names())


def find_midi_device_by_name (name: str) -> str:

	"""Find a MIDI input device by a case-insensitive substring of its name.

	Args:
		name: Substring to search for (case-insensitive).

	Returns:
		Full name of the first matching device.

	Raises:
		ValueError: If no device matches, listing all available device names.
	"""

	name_lower = name.lower()

	for device_name in mido.get_input_names():
		if name_lower in str(device_name).lower():
			return str(device_name)

	available = mido.get_input_names()
	available_str = "\n  ".join(available) if available else "(none found)"
	raise ValueError(
		f"No MIDI input device matching {name!r}.\n"
		f"Available devices:\n  {available_str}"
	)


def select_midi_device (devices: list[str]) -> str:

	"""Select a MIDI input device interactively.

	Auto-selects if exactly one device is present. Prints an interactive
	numbered menu when multiple devices are available. Mirrors the behaviour
	of subsample.audio.select_device().

	Args:
		devices: List of MIDI device name strings from list_midi_input_devices().

	Returns:
		Selected device name.

	Raises:
		ValueError: If the devices list is empty.
	"""

	if not devices:
		raise ValueError(
			"No MIDI input devices found. Connect a MIDI device and try again."
		)

	if len(devices) == 1:
		print(f"Using MIDI input: {devices[0]}")
		return devices[0]

	print("Available MIDI input devices:")
	for i, name in enumerate(devices):
		print(f"  [{i}] {name}")

	while True:
		raw = input(f"Select device [0–{len(devices) - 1}]: ").strip()

		try:
			choice = int(raw)
		except ValueError:
			print("  Please enter a number.")
			continue

		if 0 <= choice < len(devices):
			return devices[choice]

		print(f"  Please enter a number between 0 and {len(devices) - 1}.")


class MidiPlayer:

	"""Listens for MIDI messages and plays back instrument samples polyphonically.

	Designed to run on its own thread. Call run() as the thread target;
	it blocks until shutdown_event is set, then closes the MIDI port and
	PyAudio stream and returns cleanly.

	Note → reference mapping is built from the reference library names in
	alphabetical order, starting at MIDI note _NOTE_FIRST.

	Mixing: triggered notes are added as _Voice objects to a shared list.
	A PyAudio callback stream reads from all active voices simultaneously,
	sums them into one output buffer, and returns the mixed audio. This
	runs independently of the MIDI polling loop so notes overlap naturally.
	"""

	def __init__ (
		self,
		device_name: str,
		shutdown_event: threading.Event,
		instrument_library: subsample.library.InstrumentLibrary,
		similarity_matrix: subsample.similarity.SimilarityMatrix,
		reference_names: list[str],
		sample_rate: int,
		bit_depth: int,
		output_device_name: typing.Optional[str] = None,
	) -> None:

		self._device_name        = device_name
		self._shutdown_event     = shutdown_event
		self._instrument_library = instrument_library
		self._similarity_matrix  = similarity_matrix
		self._sample_rate        = sample_rate
		self._bit_depth          = bit_depth
		self._output_device_name = output_device_name

		# Active voices being mixed. The MIDI thread appends; the audio
		# callback reads and removes finished ones. Protected by _voices_lock.
		self._voices:      list[_Voice]  = []
		self._voices_lock: threading.Lock = threading.Lock()

		# Build note → reference name map: note 36 → first name (alphabetical),
		# note 37 → second, etc., up to _NOTE_LAST (16 slots).
		self._note_map: dict[int, str] = {}
		for i, name in enumerate(reference_names):
			note = _NOTE_FIRST + i
			if note > _NOTE_LAST:
				break
			self._note_map[note] = name

		_log.info(
			"MIDI note map (ch %d / mido ch %d, velocity hard-coded to %d):\n  %s",
			_MIDI_CHANNEL + 1, _MIDI_CHANNEL,
			_HARD_CODED_VELOCITY,
			"\n  ".join(f"note {note} → {name}" for note, name in sorted(self._note_map.items())),
		)

	def run (self) -> None:

		"""Open MIDI input and a stereo callback output stream, then dispatch events.

		Blocks until shutdown_event is set. Both the MIDI port and the PyAudio
		stream are closed in the finally block.
		"""

		_log.info("MIDI player opening port: %s", self._device_name)

		pa = subsample.audio.create_pyaudio()

		# Resolve output device — mirrors the input device selection pattern.
		if self._output_device_name is not None:
			output_device_index: typing.Optional[int] = subsample.audio.find_output_device_by_name(
				pa, self._output_device_name,
			)
		else:
			output_devices = subsample.audio.list_output_devices(pa)
			output_device_index = subsample.audio.select_output_device(output_devices)

		# Callback mode: PortAudio pulls audio from _audio_callback on its own
		# high-priority thread. The MIDI loop runs independently and adds voices.
		stream = pa.open(
			format=subsample.audio.get_pyaudio_format(self._bit_depth),
			channels=_OUTPUT_CHANNELS,
			rate=self._sample_rate,
			output=True,
			output_device_index=output_device_index,
			stream_callback=self._audio_callback,
		)

		try:
			with mido.open_input(self._device_name) as port:
				while not self._shutdown_event.is_set():
					msg = port.receive(block=False)

					if msg is not None:
						self._handle_message(msg)
					else:
						# No message — yield briefly before polling again.
						# 10 ms gives ~100 Hz polling rate: responsive yet not busy.
						self._shutdown_event.wait(timeout=0.01)

		finally:
			stream.stop_stream()
			stream.close()
			pa.terminate()
			_log.info("MIDI player closed port: %s", self._device_name)

	def _audio_callback (
		self,
		in_data: typing.Optional[bytes],
		frame_count: int,
		time_info: typing.Any,
		status_flags: int,
	) -> tuple[bytes, int]:

		"""PyAudio output callback — mixes all active voices into one buffer.

		Called by PortAudio on its high-priority audio thread at regular
		intervals. Must return quickly: no logging, no I/O, no blocking.

		Sums all active _Voice arrays into a float32 mix, clips to [-1, 1],
		converts to int16, and returns the bytes. Finished voices (cursor past
		end of audio) are removed from the list.
		"""

		output = numpy.zeros((frame_count, _OUTPUT_CHANNELS), dtype=numpy.float32)

		with self._voices_lock:
			active: list[_Voice] = []

			for voice in self._voices:
				remaining = len(voice.audio) - voice.position
				n = min(frame_count, remaining)

				output[:n] += voice.audio[voice.position : voice.position + n]
				voice.position += n

				if voice.position < len(voice.audio):
					active.append(voice)
				# Voice whose position has reached the end is simply not kept.

			self._voices = active

		mixed = numpy.clip(output, -1.0, 1.0)
		samples = (mixed * 32767.0).astype(numpy.int16)

		return (samples.tobytes(), pyaudio.paContinue)

	def _handle_message (self, msg: mido.Message) -> None:

		"""Dispatch a single MIDI message.

		Only note_on on the configured channel triggers playback. note_off is
		silently discarded (expected). Everything else is logged at DEBUG.
		"""

		# note_off (and note_on with velocity=0, which mido uses for note_off)
		# are expected and silently discarded — no log noise.
		if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
			return

		# Only act on note_on events on the configured channel; log anything else.
		if msg.type != "note_on" or msg.channel != _MIDI_CHANNEL:
			_log.debug("MIDI (ignored): %s", msg)
			return

		ref_name = self._note_map.get(msg.note)
		if ref_name is None:
			_log.debug("MIDI note %d on mido ch%d: no reference mapped", msg.note, _MIDI_CHANNEL)
			return

		sample_id = self._similarity_matrix.get_match(ref_name, rank=0)
		if sample_id is None:
			_log.debug("No instrument match for reference %r — library empty?", ref_name)
			return

		record = self._instrument_library.get(sample_id)
		if record is None or record.audio is None:
			_log.debug("Sample %d not found or audio not loaded", sample_id)
			return

		rendered = self._render(record)
		if rendered is None:
			return

		with self._voices_lock:
			self._voices.append(_Voice(audio=rendered))

		_log.info(
			"note %d → %r → %r  (%.2fs)",
			msg.note, ref_name, record.name, record.duration,
		)

	def _render (
		self,
		record: subsample.library.SampleRecord,
	) -> typing.Optional[numpy.ndarray]:

		"""Convert a SampleRecord to a gain-adjusted stereo float32 array.

		Returns shape (n_frames, 2), values in [-1.0, 1.0]. The gain
		normalisation and centre-pan duplication happen here so the audio
		callback only needs to sum and clip — no per-voice work beyond indexing.

		Returns None if the record has no audio.
		"""

		if record.audio is None:
			return None

		# Convert raw PCM to normalised float32 mono [-1.0, 1.0].
		mono = subsample.analysis.to_mono_float(record.audio, self._bit_depth)

		# --- Gain calculation ---
		# Hard-coded velocity; TODO: replace with msg.velocity from MIDI input.
		vel_scale = (_HARD_CODED_VELOCITY / 127.0) ** 2  # quadratic — more musical than linear

		# Normalise to target RMS so samples recorded at different levels sound
		# balanced. Guard against silence (rms == 0) to avoid division by zero.
		if record.level.rms > 0.0:
			norm_gain = _TARGET_RMS / record.level.rms
		else:
			norm_gain = 1.0

		raw_gain = norm_gain * vel_scale

		# Anti-clip ceiling: ensure gain × peak never exceeds full scale.
		if record.level.peak > 0.0:
			final_gain = min(raw_gain, 1.0 / record.level.peak)
		else:
			final_gain = raw_gain

		_log.debug(
			"gain: norm=%.3f  vel_scale=%.3f  raw=%.3f  final=%.3f  (rms=%.4f peak=%.4f)",
			norm_gain, vel_scale, raw_gain, final_gain,
			record.level.rms, record.level.peak,
		)

		mono = mono * final_gain

		# Centre pan: duplicate mono to both L and R channels.
		# TODO: add per-note pan mapping
		stereo = numpy.column_stack((mono, mono))

		return stereo
