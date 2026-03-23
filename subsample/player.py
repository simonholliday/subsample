"""MIDI input listener and sample player for Subsample.

Handles MIDI input device selection (replicating the audio device selection
pattern from audio.py) and the MidiPlayer class which listens for MIDI events
and triggers polyphonic audio playback.

Mixing architecture: a PyAudio callback stream requests N frames at regular
intervals. Each triggered note adds a _Voice (pre-rendered stereo float32
audio + playback cursor) to a shared list. The callback sums all active
voices into one output buffer, clips, converts to int16, and returns it.
The MIDI polling loop adds voices under a lock; the callback reads them.

Current implementation: exploratory / hard-coded. MIDI channel, note mapping,
and target RMS are hard-coded constants at the top of this module and will be
moved to config in a future iteration.
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

# General MIDI drum note → reference sample mapping.
# When a GM note triggers on the configured MIDI channel, Subsample looks up the
# reference name here and plays the best-matching instrument sample via the
# similarity matrix. Multiple GM notes can map to the same reference (e.g. both
# kick_1 and kick_2 route to "BD0025" so either triggers the best kick sample).
# Hard-coded for the current reference library; will move to config in a future
# iteration alongside the other hard-coded playback constants.
# Each entry is (reference_name, rank, one_shot) where:
#   rank      — selects among instrument samples ordered by similarity to the
#               reference. rank=0 is the closest match, rank=1 the second-closest.
#               Falls back to rank=0 automatically if rank N doesn't exist.
#   one_shot  — when True, note_off events are ignored and the sample plays to
#               completion (natural decay).  When False, note_off triggers a
#               cosine fade-out via the releasing flag.
#
# Percussion behaviour: kicks, snares, and cymbals are one-shot — they should
# ring out naturally regardless of how long the key is held.  Hi-hats are NOT
# one-shot because the closed hi-hat pedal (note 42) conventionally sends
# note_off for the open hi-hat (note 46) to silence it — standard GM behaviour.
_GM_DRUM_NOTE_MAP: dict[int, tuple[str, int, bool]] = {
	36: ("BD0025", 0, True),    # kick_1  → most similar to BD0025
	35: ("BD0025", 1, True),    # kick_2  → second-most similar (falls back to 0 if absent)
	38: ("SD5075", 0, True),    # snare_1 → most similar to SD5075
	40: ("SD5075", 1, True),    # snare_2 → second-most similar
	39: ("CP",     0, True),    # hand_clap
	42: ("CH",     0, True),    # hi_hat_closed
	46: ("OH25",   0, True),    # hi_hat_open
	49: ("CY5050", 0, True),    # crash_1 — rings freely
	57: ("CY5050", 1, True),    # crash_2 → second-most similar to OH25
	55: ("CY5050", 2, True),    # splash_cymbal — rings freely
	56: ("CB",     0, True),    # cowbell
}

# Target RMS level for playback normalisation (linear, not dBFS).
# 0.1 ≈ -20 dBFS — leaves headroom for mixing multiple simultaneous voices.
_TARGET_RMS = 0.1           # TODO: move to cfg.player.target_rms

# Output always stereo with centre pan (equal L and R).
# TODO: add per-note pan mapping
_OUTPUT_CHANNELS = 2        # stereo; mono signal duplicated to both channels

# Cosine fade-out duration applied when a note_off is received.
# Long enough to prevent a click on hard cutoff; short enough to be imperceptible.
# At 44100 Hz, 441 frames = 10 ms.
_RELEASE_FADE_FRAMES = 441


@dataclasses.dataclass
class _Voice:

	"""A single triggered sample being played back by the mix callback.

	audio:     Pre-rendered stereo float32 array, shape (n_frames, 2), in
	           [-1.0, 1.0]. Gain has already been applied. The callback reads
	           from this array; it is never modified after creation.
	note:      MIDI note number that triggered this voice — used to match
	           note_off events in _handle_message().
	channel:   MIDI channel (mido 0-indexed) that triggered this voice.
	position:  Current read cursor in frames. Advances each callback call.
	           Voice is removed when position >= len(audio).
	releasing: Set to True when a note_off arrives for this note+channel
	           (only for non-one-shot voices).  The callback applies a short
	           cosine fade-out over _RELEASE_FADE_FRAMES frames, then retires.
	one_shot:  When True, note_off events are ignored — the sample plays to
	           natural completion.  Kicks, snares, and cymbals are one-shot;
	           hi-hats are not (open hi-hat is silenced by the closed pedal).
	"""

	audio:     numpy.ndarray
	note:      int
	channel:   int
	position:  int  = 0
	releasing: bool = False
	one_shot:  bool = False


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
	available: list[str] = [str(d) for d in mido.get_input_names()]

	for device_name in available:
		if name_lower in device_name.lower():
			return device_name

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

	Note → reference mapping is driven by _GM_DRUM_NOTE_MAP: standard General
	MIDI drum note numbers are mapped to reference sample names. Multiple GM
	notes can map to the same reference (e.g. kick_1 and kick_2 both route to
	"BD0025"). Reference names absent from the loaded library are silently
	filtered out at construction time.

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
		output_bit_depth: typing.Optional[int] = None,
		output_sample_rate: typing.Optional[int] = None,
		transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
		virtual_midi_port: typing.Optional[str] = None,
	) -> None:

		self._device_name        = device_name
		self._shutdown_event     = shutdown_event
		self._instrument_library = instrument_library
		self._similarity_matrix  = similarity_matrix
		self._sample_rate        = sample_rate
		self._bit_depth          = bit_depth
		self._output_device_name = output_device_name

		# Output format for the playback stream.  Both default to the capture
		# format when not overridden.  output_bit_depth drives the final
		# float32→PCM packing in _audio_callback; output_sample_rate informs
		# the transform pipeline so base variants are produced at the correct
		# rate (relevant when input and output sample rates differ).
		self._output_bit_depth   = output_bit_depth   if output_bit_depth   is not None else bit_depth
		self._output_sample_rate = output_sample_rate if output_sample_rate is not None else sample_rate

		# Optional transform pipeline. When provided, _handle_message() checks
		# for a pre-computed pitched variant before falling back to _render().
		# Pass a TransformManager instance to enable pitched playback;
		# None keeps the existing behaviour (originals only).
		self._transform_manager  = transform_manager

		# When set, run() creates a virtual MIDI input port by this name instead
		# of connecting to a hardware device. Overrides device_name for input.
		self._virtual_midi_port  = virtual_midi_port

		# Active voices being mixed. The MIDI thread appends; the audio
		# callback reads and removes finished ones. Protected by _voices_lock.
		self._voices:      list[_Voice]  = []
		self._voices_lock: threading.Lock = threading.Lock()

		# Build note → (reference_name, rank, one_shot) map from the GM drum mapping.
		# Reference names not present in the library are silently omitted so
		# that a sparse reference directory doesn't raise at trigger time.
		reference_set = set(name.upper() for name in reference_names)
		self._note_map: dict[int, tuple[str, int, bool]] = {
			note: (ref, rank, one_shot)
			for note, (ref, rank, one_shot) in _GM_DRUM_NOTE_MAP.items()
			if ref.upper() in reference_set
		}

		_log.info(
			"MIDI note map (ch %d / mido ch %d):\n  %s",
			_MIDI_CHANNEL + 1, _MIDI_CHANNEL,
			"\n  ".join(
				f"note {note} → {ref} (rank {rank}{'  one-shot' if one_shot else ''})"
				for note, (ref, rank, one_shot) in sorted(self._note_map.items())
			),
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

		Input port selection:
		  - virtual_midi_port set → create a named virtual port (other apps connect to it)
		  - otherwise → open the hardware device by device_name
		"""

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
			format=subsample.audio.get_pyaudio_format(self._output_bit_depth),
			channels=_OUTPUT_CHANNELS,
			rate=self._output_sample_rate,
			output=True,
			output_device_index=output_device_index,
			stream_callback=self._audio_callback,
		)

		# Open the MIDI input port — virtual or hardware.
		# Virtual ports are server destinations that external apps connect to;
		# they require virtual=True and do not appear in get_input_names().
		if self._virtual_midi_port is not None:
			port_label = self._virtual_midi_port
			port = mido.open_input(self._virtual_midi_port, virtual=True)
			_log.info("MIDI player opened virtual port: %s", port_label)
		else:
			port_label = self._device_name
			port = mido.open_input(self._device_name)
			_log.info("MIDI player opened hardware port: %s", port_label)

		try:
			while not self._shutdown_event.is_set():
				msg = port.receive(block=False)

				if msg is not None:
					self._handle_message(msg)
				else:
					# No message — yield briefly before polling again.
					# 10 ms gives ~100 Hz polling rate: responsive yet not busy.
					self._shutdown_event.wait(timeout=0.01)

		finally:
			port.close()
			stream.stop_stream()
			stream.close()
			pa.terminate()
			_log.info("MIDI player closed port: %s", port_label)

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

		Releasing voices (note_off received): a cosine fade-out is applied over
		min(remaining, _RELEASE_FADE_FRAMES) frames, then the voice is retired.
		This prevents an audible click on hard cutoff for tonal samples.
		"""

		output = numpy.zeros((frame_count, _OUTPUT_CHANNELS), dtype=numpy.float32)

		with self._voices_lock:
			active: list[_Voice] = []

			for voice in self._voices:
				remaining = len(voice.audio) - voice.position

				if voice.releasing:
					# Fade out over at most _RELEASE_FADE_FRAMES frames, then retire.
					# Also clamped to frame_count — the output buffer is never larger.
					fade_n = min(remaining, _RELEASE_FADE_FRAMES, frame_count)
					if fade_n > 0:
						chunk = voice.audio[voice.position : voice.position + fade_n].copy()
						ramp = ((1.0 + numpy.cos(numpy.linspace(0.0, numpy.pi, fade_n))) / 2.0).astype(numpy.float32)
						output[:fade_n] += chunk * ramp[:, numpy.newaxis]
					# Voice is done — not added to active.

				else:
					n = min(frame_count, remaining)
					output[:n] += voice.audio[voice.position : voice.position + n]
					voice.position += n

					if voice.position < len(voice.audio):
						active.append(voice)
					# Voice whose position has reached the end is simply not kept.

			self._voices = active

		mixed = numpy.clip(output, -1.0, 1.0)

		# Convert to PCM bytes at the stream's declared bit depth.
		# Previously hard-coded to int16 regardless of the stream format,
		# which caused data/format mismatch for 24-bit and 32-bit streams.
		return (subsample.audio.float32_to_pcm_bytes(mixed, self._output_bit_depth), pyaudio.paContinue)

	def _handle_message (self, msg: mido.Message) -> None:

		"""Dispatch a single MIDI message.

		note_off (and note_on with velocity=0) marks matching active voices as
		releasing so the audio callback fades them out over _RELEASE_FADE_FRAMES.
		note_on on the configured channel triggers playback.
		Everything else is logged at DEBUG and ignored.
		"""

		# note_off (and note_on with velocity=0, which mido normalises to note_off)
		# marks matching active voices as releasing so the callback fades them out.
		if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
			with self._voices_lock:
				for voice in self._voices:
					if voice.note == msg.note and voice.channel == msg.channel:
						if not voice.one_shot:
							voice.releasing = True
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

			rendered = self._render_float(variant.audio, variant.level, msg.velocity)

			# Inherit one_shot from the GM drum map if this note is mapped there.
			variant_one_shot = _GM_DRUM_NOTE_MAP.get(msg.note, ("", 0, False))[2]

			with self._voices_lock:
				self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=variant_one_shot))

			_log.info(
				"WIP ch%d note %d (vel %d) → sample %d %r (pitched variant, %.2fs)",
				_VARIANT_CHANNEL, msg.note, msg.velocity, sample_id, record.name, variant.duration,
			)
			return

		# Only act on note_on events on the configured channel; log anything else.
		if msg.type != "note_on" or msg.channel != _MIDI_CHANNEL:
			_log.debug("MIDI (ignored): %s", msg)
			return

		mapping = self._note_map.get(msg.note)
		if mapping is None:
			_log.debug("MIDI note %d on mido ch%d: no reference mapped", msg.note, _MIDI_CHANNEL)
			return

		ref_name, rank, one_shot = mapping

		# Look up the Nth-closest instrument sample for this reference.
		# If rank N doesn't exist yet (e.g. only one hit recorded), fall back
		# to rank 0 so the note always triggers something.
		sample_id = self._similarity_matrix.get_match(ref_name, rank=rank)
		if sample_id is None and rank > 0:
			sample_id = self._similarity_matrix.get_match(ref_name, rank=0)
		if sample_id is None:
			_log.debug("No instrument match for reference %r — library empty?", ref_name)
			return

		record = self._instrument_library.get(sample_id)
		if record is None or record.audio is None:
			_log.debug("Sample %d not found or audio not loaded", sample_id)
			return

		if self._transform_manager is not None:
			# 1. Check for a pitch variant (tonal samples only).
			variant = self._transform_manager.get_pitched(sample_id, msg.note)

			if variant is not None:
				rendered = self._render_float(variant.audio, variant.level, msg.velocity)
				with self._voices_lock:
					self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
				_log.info(
					"note %d (vel %d) → %r → %r (pitched variant)  (%.2fs)",
					msg.note, msg.velocity, ref_name, record.name, variant.duration,
				)
				return

			# 2. Fall back to the base variant (float32, peak-normalised, no DSP).
			# Available for every sample once the background worker has run.
			base = self._transform_manager.get_base(sample_id)

			if base is not None:
				rendered = self._render_float(base.audio, base.level, msg.velocity)
				with self._voices_lock:
					self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
				_log.info(
					"note %d (vel %d) → %r → %r (base variant)  (%.2fs)",
					msg.note, msg.velocity, ref_name, record.name, base.duration,
				)
				return

		# 3. Last resort: convert from int PCM on this trigger.
		# Used only on the first trigger before the base variant is ready,
		# or when no transform manager is configured.
		original: typing.Optional[numpy.ndarray] = self._render(record, msg.velocity)
		if original is None:
			return

		with self._voices_lock:
			self._voices.append(_Voice(audio=original, note=msg.note, channel=msg.channel, one_shot=one_shot))

		_log.info(
			"note %d (vel %d) → %r → %r  (%.2fs)",
			msg.note, msg.velocity, ref_name, record.name, record.duration,
		)

	def _render (
		self,
		record: subsample.library.SampleRecord,
		velocity: int,
	) -> typing.Optional[numpy.ndarray]:

		"""Convert a SampleRecord to a gain-adjusted stereo float32 array.

		Converts int PCM → float32 preserving channel count → applies gain →
		returns stereo.  Stereo recordings play back in stereo; mono recordings
		are centre-panned.  Returns None if the record has no audio.

		For transform variants (already float32 multi-channel), use
		_render_float() directly to skip the int→float conversion.
		"""

		if record.audio is None:
			return None

		# Convert int PCM → float32, preserving all channels so that stereo
		# recordings play back in stereo rather than being mixed to mono.
		float_audio = subsample.transform._pcm_to_float32(record.audio, self._bit_depth)

		return self._render_float(float_audio, record.level, velocity)

	def _render_float (
		self,
		audio: numpy.ndarray,
		level: subsample.analysis.LevelResult,
		velocity: int,
	) -> numpy.ndarray:

		"""Apply gain normalisation and pan to stereo float32.

		Shared by both the original _render() path (mono from int PCM) and
		the transform variant path (float32 multi-channel from TransformResult).

		Args:
			audio:    float32, shape (n_frames, channels).  Mono or stereo.
			level:    LevelResult for this audio (peak + rms), used for gain calc.
			velocity: MIDI velocity (0–127) from the triggering note_on message.

		Returns:
			Stereo float32, shape (n_frames, 2), values in [-1.0, 1.0].
		"""

		# --- Gain calculation ---
		vel_scale = (velocity / 127.0) ** 2  # quadratic — more musical than linear

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
