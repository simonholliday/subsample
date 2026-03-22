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
import subsample.transform


_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Playback constants — known deferred items awaiting config migration.
# Tracked in README-AGENTS.md "Extending audio playback" and README.md Planned.
# ---------------------------------------------------------------------------

# MIDI channel to listen on. Mido uses 0-indexed channels; the BeatStep Pro
# sends on channel 16 (user-facing), which mido reports as channel=15.
_MIDI_CHANNEL = 9          # channel 10 in user-facing terms (mido: 0-indexed)

# WIP/exploratory: MIDI channel for pitch-variant testing (channel 1, mido 0-indexed).
# Notes on this channel are mapped directly to cached pitch variants from the transform
# pipeline — see MidiPlayer._build_variant_note_map() and _handle_message().
# This bypasses similarity matching: instrument samples are iterated in insertion order;
# each sample's variants claim their MIDI notes, with later samples overwriting earlier.
_VARIANT_CHANNEL = 0

# MIDI note range mapped to reference samples (first note = lowest note number).
_NOTE_FIRST = 36            # MIDI note 36 = first reference sample (alphabetical)
_NOTE_LAST  = 51            # MIDI note 51 = 16th reference (or last if fewer)

# Fixed MIDI velocity used for every trigger. Input velocity is ignored.
_HARD_CODED_VELOCITY = 127  # TODO: replace with msg.velocity from MIDI input

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
		transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	) -> None:

		self._device_name        = device_name
		self._shutdown_event     = shutdown_event
		self._instrument_library = instrument_library
		self._similarity_matrix  = similarity_matrix
		self._sample_rate        = sample_rate
		self._bit_depth          = bit_depth
		self._output_device_name = output_device_name

		# Optional transform pipeline. When provided, _handle_message() checks
		# for a pre-computed pitched variant before falling back to _render().
		# Pass a TransformManager instance to enable pitched playback;
		# None keeps the existing behaviour (originals only).
		self._transform_manager  = transform_manager

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

		# WIP: variant channel active when transform pipeline is wired in.
		if transform_manager is not None:
			_log.info(
				"WIP variant channel: ch %d (mido ch %d) → pitch variants from transform cache "
				"(rebuilt lazily per trigger from instrument library insertion order)",
				_VARIANT_CHANNEL + 1, _VARIANT_CHANNEL,
			)

	def _build_variant_note_map (self) -> dict[int, int]:

		"""Build a midi_note → sample_id map from all cached pitch variants.

		WIP/exploratory: used by _handle_message() for _VARIANT_CHANNEL triggers.

		Iterates instrument samples in insertion order (oldest first). For each sample,
		all cached PitchShift variants are extracted from the transform pipeline.
		Each variant's target MIDI note is mapped to the sample's ID. Later samples
		overwrite earlier ones, so the most recently added sample wins for any
		given note.

		Returns an empty dict if no transform pipeline is configured.
		"""

		if self._transform_manager is None:
			return {}

		note_map: dict[int, int] = {}

		for record in self._instrument_library.samples():
			for key in self._transform_manager.list_variants(record.sample_id):
				for step in key.spec.steps:
					if isinstance(step, subsample.transform.PitchShift):
						note_map[step.target_midi_note] = record.sample_id

		return note_map

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

		# WIP/exploratory: channel 1 (mido ch 0) plays pitch variants directly.
		# Bypasses similarity matching — maps notes to cached PitchShift variants.
		# See _build_variant_note_map() for the mapping strategy.
		if msg.type == "note_on" and msg.channel == _VARIANT_CHANNEL:
			if self._transform_manager is None:
				_log.debug("MIDI ch%d note %d: variant channel ignored (no transform pipeline)", _VARIANT_CHANNEL, msg.note)
				return

			variant_note_map = self._build_variant_note_map()
			sample_id = variant_note_map.get(msg.note)

			if sample_id is None:
				_log.debug("MIDI ch%d note %d: no variant mapped (variants still computing?)", _VARIANT_CHANNEL, msg.note)
				return

			record = self._instrument_library.get(sample_id)
			if record is None:
				_log.debug("MIDI ch%d note %d: sample %d not in library", _VARIANT_CHANNEL, msg.note, sample_id)
				return

			variant = self._transform_manager.get_pitched(sample_id, msg.note)

			if variant is None:
				_log.debug("MIDI ch%d note %d → sample %d %r: variant cache miss", _VARIANT_CHANNEL, msg.note, sample_id, record.name)
				return

			rendered = self._render_float(variant.audio, variant.level)

			with self._voices_lock:
				self._voices.append(_Voice(audio=rendered))

			_log.info(
				"WIP ch%d note %d → sample %d %r (pitched variant, %.2fs)",
				_VARIANT_CHANNEL, msg.note, sample_id, record.name, variant.duration,
			)
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

		# Check for a pre-computed pitched variant first.
		# If found, use it directly (already float32, no int→float conversion).
		# If not found (None), the manager has enqueued it for next time;
		# fall back to rendering the original at its native pitch.
		if self._transform_manager is not None:
			variant = self._transform_manager.get_pitched(sample_id, msg.note)
			if variant is not None:
				rendered = self._render_float(variant.audio, variant.level)
				with self._voices_lock:
					self._voices.append(_Voice(audio=rendered))
				_log.info(
					"note %d → %r → %r (pitched variant)  (%.2fs)",
					msg.note, ref_name, record.name, variant.duration,
				)
				return

		original: typing.Optional[numpy.ndarray] = self._render(record)
		if original is None:
			return

		with self._voices_lock:
			self._voices.append(_Voice(audio=original))

		_log.info(
			"note %d → %r → %r  (%.2fs)",
			msg.note, ref_name, record.name, record.duration,
		)

	def _render (
		self,
		record: subsample.library.SampleRecord,
	) -> typing.Optional[numpy.ndarray]:

		"""Convert a SampleRecord to a gain-adjusted stereo float32 array.

		Converts int PCM → mono float32 → applies gain → returns stereo.
		Returns None if the record has no audio.

		For transform variants (already float32 multi-channel), use
		_render_float() directly to skip the int→float conversion.
		"""

		if record.audio is None:
			return None

		# Convert raw int PCM to normalised float32 mono.
		mono = subsample.analysis.to_mono_float(record.audio, self._bit_depth)

		# Reshape to (n_frames, 1) so _render_float can handle the panning step.
		mono_2d: numpy.ndarray = mono[:, numpy.newaxis]

		return self._render_float(mono_2d, record.level)

	def _render_float (
		self,
		audio: numpy.ndarray,
		level: subsample.analysis.LevelResult,
	) -> numpy.ndarray:

		"""Apply gain normalisation and pan to stereo float32.

		Shared by both the original _render() path (mono from int PCM) and
		the transform variant path (float32 multi-channel from TransformResult).

		Args:
			audio: float32, shape (n_frames, channels).  Mono or stereo.
			level: LevelResult for this audio (peak + rms), used for gain calc.

		Returns:
			Stereo float32, shape (n_frames, 2), values in [-1.0, 1.0].
		"""

		# --- Gain calculation ---
		# Hard-coded velocity; TODO: replace with msg.velocity from MIDI input.
		vel_scale = (_HARD_CODED_VELOCITY / 127.0) ** 2  # quadratic — more musical than linear

		# Normalise to target RMS so samples recorded at different levels sound
		# balanced. Guard against silence (rms == 0) to avoid division by zero.
		if level.rms > 0.0:
			norm_gain = _TARGET_RMS / level.rms
		else:
			norm_gain = 1.0

		raw_gain = norm_gain * vel_scale

		# Anti-clip ceiling: ensure gain × peak never exceeds full scale.
		if level.peak > 0.0:
			final_gain = min(raw_gain, 1.0 / level.peak)
		else:
			final_gain = raw_gain

		_log.debug(
			"gain: norm=%.3f  vel_scale=%.3f  raw=%.3f  final=%.3f  (rms=%.4f peak=%.4f)",
			norm_gain, vel_scale, raw_gain, final_gain,
			level.rms, level.peak,
		)

		gained = audio * final_gain

		# Pan to stereo.  Multi-channel originals are mixed to mono first;
		# mono originals are duplicated to both channels.
		# TODO: add per-note pan mapping
		if gained.shape[1] == 2:
			stereo = gained
		elif gained.shape[1] == 1:
			stereo = numpy.column_stack((gained[:, 0], gained[:, 0]))
		else:
			# More than 2 channels: mix down to mono then centre pan.
			mono_mix = numpy.mean(gained, axis=1, dtype=numpy.float32)
			stereo   = numpy.column_stack((mono_mix, mono_mix))

		return stereo
