"""MIDI input listener and sample player for Subsample.

Handles MIDI input device selection (replicating the audio device selection
pattern from audio.py) and the MidiPlayer class which listens for MIDI events
and triggers polyphonic audio playback.

Mixing architecture: a PyAudio callback stream requests N frames at regular
intervals. Each triggered note adds a _Voice (pre-rendered stereo float32
audio + playback cursor) to a shared list. The callback sums all active
voices into one output buffer, clips, converts to PCM bytes at the output
bit depth, and returns them. The MIDI polling loop adds voices under a lock;
the callback reads them.

Per-voice gain is controlled by cfg.player.max_polyphony: each voice's RMS
target is 1.0 / max_polyphony, so N voices at max velocity sum to
approximately full scale. Clipping is detected in the callback and logged at
WARNING (throttled) with guidance to raise max_polyphony.

MIDI routing is loaded from a yaml file at startup via load_midi_map().
The map defines which MIDI notes (on which channels) trigger which samples.
See midi-map.yaml.default for the format specification.
"""

import dataclasses
import logging
import pathlib
import threading
import time
import typing

import mido
import numpy
import pyaudio
import yaml

import pymididefs.notes
import subsample.analysis
import subsample.audio
import subsample.library
import subsample.query
import subsample.similarity
import subsample.transform


_log = logging.getLogger(__name__)

# Cosine fade-out duration applied when a note_off is received.
# Long enough to prevent a click on hard cutoff; short enough to be imperceptible.
# Stored as seconds; converted to frames in MidiPlayer.__init__() using the
# actual output sample rate so the duration is correct regardless of device.
_RELEASE_FADE_SECONDS: float = 0.01  # 10 ms

# Note map: (mido_channel, midi_note) → (Assignment, pick_position).
# pick_position is the 1-indexed rank for this specific note within the
# assignment's note list (used for per-note rank distribution in multi-note
# assignments without explicit pick).  For single-note or explicit-pick
# assignments, this equals Assignment.pick.
NoteMap = dict[tuple[int, int], tuple[subsample.query.Assignment, int]]


def _beat_quantize_params (process: subsample.query.ProcessSpec) -> tuple[typing.Optional[float], int]:

	"""Extract BPM and grid from a beat_quantize processor step.

	Returns (target_bpm, grid) where target_bpm is None when no explicit
	BPM is declared (falling back to the global transform.target_bpm).
	"""

	beat_step = next(s for s in process.steps if s.name == "beat_quantize")
	bpm  = float(beat_step.get("bpm", 0))
	grid = int(beat_step.get("grid", 16))

	return (bpm if bpm > 0 else None, grid)


def _parse_pan_gains (weights_raw: typing.Any, output_channels: int, assignment_name: str) -> numpy.ndarray:

	"""Parse pan weights from config and normalise to constant-power channel gains.

	Pan weights are expressed as a list of non-negative values, one per output channel.
	They are normalised so the summed power across all channels is 1.0:

	    gain[i] = sqrt(weight[i] / sum(weights))

	This gives constant-power panning: a centre pan [50, 50] produces the same
	perceived loudness as a hard-left [100, 0], just distributed differently.

	Args:
		weights_raw:      Raw YAML value (list of numbers, or None for default).
		output_channels:  Number of output channels (pan list length must match).
		assignment_name:  Name for error messages.

	Returns:
		float32 numpy array of shape (output_channels,) with constant-power gains.

	Raises:
		ValueError: If the weights list has the wrong length or any negative value.
	"""

	if weights_raw is None:
		# Default: equal weights across all channels.
		weight_arr = numpy.ones(output_channels, dtype=numpy.float32)
	else:
		weights = list(weights_raw)
		if len(weights) != output_channels:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pan has {len(weights)} value(s) "
				f"but output has {output_channels} channel(s). "
				f"Stereo example: [50, 50]  (L, R weights)"
			)
		weight_arr = numpy.array(weights, dtype=numpy.float32)
		if numpy.any(weight_arr < 0):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pan weights must be >= 0"
			)

	total = float(numpy.sum(weight_arr))
	if total == 0.0:
		# All-zero pan produces silence. Warn so the user can diagnose a mis-configured map.
		_log.warning("Assignment %r: all pan weights are zero — note will be silent", assignment_name)
		return numpy.zeros(output_channels, dtype=numpy.float32)

	# Constant-power normalisation: gain[i] = sqrt(weight[i] / total)
	result: numpy.ndarray = numpy.sqrt(weight_arr / total).astype(numpy.float32)
	return result


# Note name conversion — delegated to pymididefs.
_midi_to_note_name = pymididefs.notes.note_to_name
_parse_note_name = pymididefs.notes.name_to_note


def _parse_note_spec (notes_raw: typing.Any, assignment_name: str) -> list[int]:

	"""Parse the 'notes' field from a MIDI map assignment into MIDI note numbers.

	Accepts:
	  Integer:          36          → [36]
	  Note name:        "C3"        → [48]
	  Numeric range:    "36..60"    → [36, 37, ..., 60]
	  Note name range:  "C2..C4"    → [36, 37, ..., 60]
	  List (mixed):     [36, "C3"]  → [36, 48]

	Args:
		notes_raw:       Raw YAML value for the 'notes' field.
		assignment_name: Assignment name used in error messages.

	Returns:
		Non-empty list of MIDI note numbers.

	Raises:
		ValueError: If any note value is malformed or outside [0, 127].
	"""

	def _single (item: typing.Any) -> int:
		"""Resolve one note value: int, numeric string, or note-name string."""

		if isinstance(item, int):
			if not 0 <= item <= 127:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: note {item} is outside [0, 127]"
				)
			return item

		if isinstance(item, str):
			# Try parsing as a bare integer first ("36"), then as a note name ("C3").
			try:
				n = int(item)
				if not 0 <= n <= 127:
					raise ValueError(
						f"MIDI map assignment {assignment_name!r}: note {n} is outside [0, 127]"
					)
				return n
			except ValueError:
				pass

			try:
				return _parse_note_name(item)
			except ValueError as exc:
				raise ValueError(f"MIDI map assignment {assignment_name!r}: {exc}") from exc

		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: unexpected note value {item!r}"
		)

	# Range syntax: "C2..C4" or "36..60"
	if isinstance(notes_raw, str) and ".." in notes_raw:
		lo_str, hi_str = notes_raw.split("..", 1)
		lo = _single(lo_str.strip())
		hi = _single(hi_str.strip())

		if lo > hi:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: note range {notes_raw!r} — "
				f"start ({lo}) must be <= end ({hi})"
			)

		return list(range(lo, hi + 1))

	# Single value (int or string).
	if isinstance(notes_raw, (int, str)):
		return [_single(notes_raw)]

	# List of mixed values.
	return [_single(item) for item in notes_raw]


def load_midi_map (
	path: pathlib.Path,
	reference_names: list[str],
	output_channels: int = 2,
) -> NoteMap:

	"""Load a MIDI routing map from a YAML file.

	Parses the assignments list using the select/process pipeline format
	and returns a lookup dict keyed by (mido_channel, midi_note).

	Each assignment declares:
	  select:   Which sample to play — filter predicates, ordering, pick position.
	            Can be a single spec or a list (fallback chain, tried in order).
	  process:  How to present it — ordered list of processors (repitch, beat_quantize, etc.).
	  one_shot: Playback behaviour — true (default) ignores note_off.
	  gain:     Level offset in dB (default 0.0).
	  pan:      Stereo weights (default [50, 50] = centre).

	Pan weights are normalised to constant-power gains: gain[i] = sqrt(w[i]/sum(w)).

	reference predicates whose name is not in reference_names are skipped
	with a WARNING — this prevents silent failures when using a map built
	for a different reference library.

	Args:
		path:             Path to the MIDI map YAML file.
		reference_names:  Names from the loaded reference library (case-insensitive).
		output_channels:  Number of output channels (default 2 = stereo).

	Returns:
		NoteMap: dict mapping (mido_channel, midi_note) → (Assignment, pick_position).
		Empty dict if the file is empty or has no valid assignments.

	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError:        If the YAML is malformed or a required field is missing.
	"""

	if not path.exists():
		raise FileNotFoundError(f"MIDI map not found: {path}")

	with path.open(encoding="utf-8") as fh:
		raw = yaml.safe_load(fh)

	if raw is None or "assignments" not in raw:
		_log.warning("MIDI map %s has no assignments — no notes will be mapped", path)
		return {}

	reference_set = {name.upper() for name in reference_names}
	note_map: NoteMap = {}

	for assignment_raw in raw["assignments"]:
		name = assignment_raw.get("name", "<unnamed>")

		# Channel: user-facing 1-16 → mido 0-indexed.
		channel_raw = assignment_raw.get("channel")

		if channel_raw is None:
			raise ValueError(f"MIDI map assignment {name!r}: missing 'channel'")

		mido_channel = int(channel_raw) - 1

		notes_raw = assignment_raw.get("notes")

		if notes_raw is None:
			raise ValueError(f"MIDI map assignment {name!r}: missing 'notes'")

		notes = _parse_note_spec(notes_raw, name)

		# Parse select block (required).
		select_raw = assignment_raw.get("select")

		if select_raw is None:
			raise ValueError(f"MIDI map assignment {name!r}: missing 'select'")

		select_specs = subsample.query.parse_select(select_raw, name)

		# Validate reference predicates against the loaded reference library.
		valid = True

		for spec in select_specs:
			ref = spec.where.reference

			if ref is not None and ref.upper() not in reference_set:
				_log.warning(
					"MIDI map assignment %r: reference %r not in reference library — skipping",
					name, ref,
				)
				valid = False
				break

		if not valid:
			continue

		# Parse process block (optional — defaults to no processing).
		process = subsample.query.parse_process(assignment_raw.get("process"), name)

		one_shot = bool(assignment_raw.get("one_shot", True))
		gain_db  = float(assignment_raw.get("gain", 0.0))
		pan_gains = _parse_pan_gains(assignment_raw.get("pan"), output_channels, name)

		assignment = subsample.query.Assignment(
			name=name,
			select=select_specs,
			process=process,
			one_shot=one_shot,
			gain_db=gain_db,
			pan_gains=pan_gains,
		)

		# Per-note pick distribution:
		# When process includes repitch, all notes share pick=1 (same sample,
		# pitched per note).  Otherwise, each note gets the next pick position
		# so multi-note assignments distribute across ranked matches.
		# An explicit pick in the SelectSpec overrides this default.
		if isinstance(select_raw, dict):
			explicit_pick = "pick" in select_raw
		elif isinstance(select_raw, list) and select_raw:
			explicit_pick = any("pick" in s for s in select_raw if isinstance(s, dict))
		else:
			explicit_pick = False

		for note_idx, note in enumerate(notes):

			if explicit_pick or process.has_repitch() or len(notes) == 1:
				pick = select_specs[0].pick
			else:
				pick = note_idx + 1

			note_map[(mido_channel, int(note))] = (assignment, pick)

	_log.info(
		"MIDI map loaded from %s: %d note(s) across %d assignment(s)",
		path,
		len(note_map),
		len(raw.get("assignments", [])),
	)

	return note_map


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
	           cosine fade-out over self._release_fade_frames frames, then retires.
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

	Note → reference mapping is loaded from a YAML file by the caller via
	load_midi_map() and passed as the `midi_map` parameter.  The map keys
	notes by (mido_channel, midi_note) and stores (ref_name, rank, one_shot,
	pan_gains) per note.  At trigger time, the most similar instrument sample
	to the reference is looked up from the SimilarityMatrix and played back.

	Mixing: triggered notes are added as _Voice objects to a shared list.
	A PyAudio callback stream reads from all active voices simultaneously,
	sums them into one output buffer, applies the safety limiter, and returns
	the mixed audio.  This runs independently of the MIDI polling loop so
	notes overlap naturally.
	"""

	def __init__ (
		self,
		device_name: str,
		shutdown_event: threading.Event,
		instrument_library: subsample.library.InstrumentLibrary,
		similarity_matrix: subsample.similarity.SimilarityMatrix,
		midi_map: NoteMap,
		sample_rate: int,
		bit_depth: int,
		output_device_name: typing.Optional[str] = None,
		output_bit_depth: typing.Optional[int] = None,
		output_sample_rate: typing.Optional[int] = None,
		transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
		virtual_midi_port: typing.Optional[str] = None,
		max_polyphony: int = 8,
		limiter_threshold_db: float = -1.5,
		limiter_ceiling_db: float = -0.1,
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
		self._output_bit_depth    = output_bit_depth   if output_bit_depth   is not None else bit_depth
		self._output_sample_rate  = output_sample_rate if output_sample_rate is not None else sample_rate
		self._release_fade_frames = round(_RELEASE_FADE_SECONDS * self._output_sample_rate)

		# Per-voice RMS target derived from max_polyphony.
		# 1.0 / max_polyphony gives each voice an equal share of headroom:
		# 8 voices → 0.125 RMS per voice ≈ -18 dBFS.  The anti-clip ceiling
		# in _render_float() (1.0 / level.peak) is a separate per-voice guard.
		self._target_rms       = 1.0 / max_polyphony
		self._max_polyphony    = max_polyphony

		# Safety limiter: tanh soft-clipper applied to the mixed output buffer.
		# Pre-computed linear values so the callback does no dB conversions.
		# The knee is the range between threshold and ceiling; the tanh curve
		# maps [0, ∞) to [0, knee) asymptotically, so output never exceeds ceiling.
		self._limiter_threshold = 10.0 ** (limiter_threshold_db / 20.0)
		self._limiter_ceiling   = 10.0 ** (limiter_ceiling_db / 20.0)
		self._limiter_knee      = self._limiter_ceiling - self._limiter_threshold

		# Clipping detection: timestamp of the last warning so we can throttle
		# to at most one log message every 5 seconds during dense passages.
		self._last_clip_warn: float = 0.0

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

		# Number of output channels.  Determines the shape of the mix buffer and
		# must match the pa.open(channels=...) call in run().  Fixed at 2
		# (stereo) for this phase; multichannel support raises this in future.
		self._output_channels: int = 2

		# Note routing map: (mido_channel, midi_note) → (Assignment, pick_position).
		# Loaded from the MIDI map YAML file by the caller (cli.py) via load_midi_map().
		self._note_map: NoteMap = midi_map

		# Group consecutive notes that share the same Assignment into ranges
		# so that a 128-note pitched assignment becomes a single log line.
		groups: list[tuple[int, int, int, subsample.query.Assignment, int]] = []

		for (ch, note), (asgn, pick) in sorted(self._note_map.items()):
			if (
				groups
				and groups[-1][0] == ch
				and groups[-1][2] == note - 1
				and groups[-1][3] is asgn
			):
				groups[-1] = (ch, groups[-1][1], note, asgn, pick)
			else:
				groups.append((ch, note, note, asgn, pick))

		lines: list[str] = []

		for ch, lo, hi, asgn, _pick in groups:
			count = hi - lo + 1

			if count == 1:
				note_str = f"note {_midi_to_note_name(lo)}"
			else:
				note_str = f"notes {_midi_to_note_name(lo)}..{_midi_to_note_name(hi)} ({count})"

			line = f"ch{ch+1} {note_str} → {asgn.name}"

			if asgn.process.has_repitch():
				line += " pitched"

			if asgn.process.has_beat_quantize():
				line += " beat-quantized"

			if asgn.one_shot:
				line += "  one-shot"

			line += f"  pan=[{', '.join(f'{g:.2f}' for g in asgn.pan_gains)}]"
			lines.append(line)

		_log.info(
			"MIDI note map: %d note(s) loaded\n  %s",
			len(self._note_map),
			"\n  ".join(lines),
		)

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
			output_device_index: int = subsample.audio.find_output_device_by_name(
				pa, self._output_device_name,
			)
		else:
			output_devices = subsample.audio.list_output_devices(pa)
			output_device_index = subsample.audio.select_output_device(output_devices)

		# Callback mode: PortAudio pulls audio from _audio_callback on its own
		# high-priority thread. The MIDI loop runs independently and adds voices.
		stream = pa.open(
			format=subsample.audio.get_pyaudio_format(self._output_bit_depth),
			channels=self._output_channels,
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
		min(remaining, self._release_fade_frames) frames, then the voice is retired.
		This prevents an audible click on hard cutoff for tonal samples.
		"""

		output = numpy.zeros((frame_count, self._output_channels), dtype=numpy.float32)

		with self._voices_lock:
			active: list[_Voice] = []

			for voice in self._voices:
				remaining = len(voice.audio) - voice.position

				if voice.releasing:
					# Fade out over at most self._release_fade_frames frames, then retire.
					# Also clamped to frame_count — the output buffer is never larger.
					fade_n = min(remaining, self._release_fade_frames, frame_count)
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

		# Safety limiter: tanh soft-clip above threshold.
		# Operates in-place on samples where abs(output) > threshold.
		# Below threshold: zero cost (mask is False, no computation).
		# Above threshold: smoothly compressed toward ceiling.
		# The hard clip below remains as a final safety net.
		abs_output = numpy.abs(output)
		mask = abs_output > self._limiter_threshold
		if numpy.any(mask):
			sign   = numpy.sign(output[mask])
			excess = abs_output[mask] - self._limiter_threshold
			output[mask] = sign * (
				self._limiter_threshold
				+ self._limiter_knee * numpy.tanh(excess / self._limiter_knee)
			)

		mixed = numpy.clip(output, -1.0, 1.0)

		# Clipping detection: warn only if the post-limiter output still exceeds
		# the ceiling (shouldn't happen — the tanh asymptote guarantees this —
		# but serves as a diagnostic if the limiter is misconfigured or bypassed).
		# Throttled to at most one warning every 5 seconds.
		peak_abs = float(numpy.max(numpy.abs(mixed)))
		if peak_abs > self._limiter_ceiling:
			now = time.monotonic()
			if now - self._last_clip_warn >= 5.0:
				self._last_clip_warn = now
				_log.warning(
					"Audio clipping: post-limiter peak=%.3f (%.1f dBFS) exceeds ceiling %.3f — "
					"raise player.max_polyphony above %d to reduce per-voice level",
					peak_abs,
					20.0 * numpy.log10(peak_abs),
					self._limiter_ceiling,
					self._max_polyphony,
				)

		# Convert to PCM bytes at the stream's declared bit depth.
		# Previously hard-coded to int16 regardless of the stream format,
		# which caused data/format mismatch for 24-bit and 32-bit streams.
		return (subsample.audio.float32_to_pcm_bytes(mixed, self._output_bit_depth), pyaudio.paContinue)

	def _handle_message (self, msg: mido.Message) -> None:

		"""Dispatch a single MIDI message via the select/process pipeline.

		note_off (and note_on with velocity=0) marks matching active voices as
		releasing so the audio callback fades them out over self._release_fade_frames.
		note_on triggers sample selection via the query engine, then looks up
		the appropriate transform variant based on the assignment's ProcessSpec.
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

		# Only act on note_on events; anything else is logged at DEBUG and ignored.
		if msg.type != "note_on":
			_log.debug("MIDI (ignored): %s", msg)
			return

		entry = self._note_map.get((msg.channel, msg.note))

		if entry is None:
			_log.debug("MIDI ch%d note %d: no mapping", msg.channel + 1, msg.note)
			return

		assignment, pick = entry
		pan_gains = assignment.pan_gains
		one_shot  = assignment.one_shot

		# ── Sample selection via query engine ─────────────────────────────

		all_samples = self._instrument_library.samples()
		sample_id: typing.Optional[int] = None

		for select_spec in assignment.select:

			ranked = subsample.query.query(select_spec, all_samples, self._similarity_matrix)

			if ranked:
				# pick is 1-indexed; clamp to available range, fall back to rank 0.
				idx = min(pick - 1, len(ranked) - 1)
				sample_id = ranked[idx].sample_id
				break

		if sample_id is None:
			_log.debug(
				"note %d → %r: no sample matched any select spec",
				msg.note, assignment.name,
			)
			return

		record = self._instrument_library.get(sample_id)

		if record is None or record.audio is None:
			_log.debug("Sample %d not found or audio not loaded", sample_id)
			return

		# ── Variant lookup based on ProcessSpec ───────────────────────────

		if self._transform_manager is not None:

			# 1. Repitch: check for a pitch variant if the process declares repitch.
			if assignment.process.has_repitch():
				variant = self._transform_manager.get_pitched(sample_id, msg.note)

				if variant is not None:
					rendered = self._render_float(variant.audio, variant.level, msg.velocity, pan_gains, assignment.gain_db)
					with self._voices_lock:
						self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
					_log.debug(
						"note %d (vel %d) → %r → %r (pitched variant)  (%.2fs)",
						msg.note, msg.velocity, assignment.name, record.name, variant.duration,
					)
					return

			# 2. Beat-quantize: check for a time-stretch variant.
			if assignment.process.has_beat_quantize():

				if record.rhythm.tempo_bpm <= 0.0:
					_log.warning(
						"beat_quantize %s: sample %r has no detected tempo — "
						"playing without beat-quantizing",
						assignment.name, record.name,
					)

				else:
					bpm, grid = _beat_quantize_params(assignment.process)

					stretched = self._transform_manager.get_at_bpm(
						sample_id,
						target_bpm=bpm,
						resolution=grid,
					)

					if stretched is not None:
						rendered = self._render_float(stretched.audio, stretched.level, msg.velocity, pan_gains, assignment.gain_db)
						with self._voices_lock:
							self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
						_log.debug(
							"note %d (vel %d) → %r → %r (stretched variant)  (%.2fs)",
							msg.note, msg.velocity, assignment.name, record.name, stretched.duration,
						)
						return

			# 3. Fall back to the base variant (float32, peak-normalised, no DSP).
			base = self._transform_manager.get_base(sample_id)

			if base is not None:
				rendered = self._render_float(base.audio, base.level, msg.velocity, pan_gains, assignment.gain_db)
				with self._voices_lock:
					self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
				_log.debug(
					"note %d (vel %d) → %r → %r (base variant)  (%.2fs)",
					msg.note, msg.velocity, assignment.name, record.name, base.duration,
				)
				return

		# 4. Last resort: convert from int PCM on this trigger.
		original: typing.Optional[numpy.ndarray] = self._render(record, msg.velocity, pan_gains, assignment.gain_db)

		if original is None:
			return

		with self._voices_lock:
			self._voices.append(_Voice(audio=original, note=msg.note, channel=msg.channel, one_shot=one_shot))

		_log.debug(
			"note %d (vel %d) → %r → %r  (%.2fs)",
			msg.note, msg.velocity, assignment.name, record.name, record.duration,
		)

	def _render (
		self,
		record: subsample.library.SampleRecord,
		velocity: int,
		pan_gains: numpy.ndarray,
		gain_db: float = 0.0,
	) -> typing.Optional[numpy.ndarray]:

		"""Convert a SampleRecord to a gain-adjusted, panned output array.

		Converts int PCM → float32 preserving channel count → applies gain and
		pan → returns output-channel-count float32 array.  Returns None if the
		record has no audio.

		For transform variants (already float32 multi-channel), use
		_render_float() directly to skip the int→float conversion.
		"""

		if record.audio is None:
			return None

		# Convert int PCM → float32, preserving all channels so that stereo
		# recordings play back in stereo rather than being mixed to mono.
		float_audio = subsample.transform._pcm_to_float32(record.audio, self._bit_depth)

		return self._render_float(float_audio, record.level, velocity, pan_gains, gain_db)

	def update_assignments (self) -> None:

		"""Pre-compute transform variants for all assignments that declare processors.

		Groups notes by Assignment, resolves each to its current sample via the
		query engine, and enqueues the appropriate variants:
		  - repitch: pitch-shift variants for every note in the group
		  - beat_quantize: time-stretch variant with per-assignment BPM/grid params

		The TransformProcessor deduplicates in-flight and cached keys, so
		repeated calls are safe and cheap.

		Call this:
		  - At startup, after the similarity matrix is populated.
		  - In the on_complete callback after a new sample arrives — ensures
		    variants are ready before the next trigger.

		No-ops if no transform manager is configured or no processable
		assignments exist.
		"""

		if self._transform_manager is None:
			return

		# Group notes by Assignment identity (object id) — all notes in the same
		# assignment share the same select/process spec.  Collect (note, pick)
		# pairs so beat_quantize can pre-compute a variant for every pick
		# position (each note may resolve to a different sample).
		groups: dict[int, tuple[subsample.query.Assignment, list[tuple[int, int]]]] = {}

		for (_ch, note), (asgn, pick) in self._note_map.items():

			if asgn.process.has_repitch() or asgn.process.has_beat_quantize():
				group_key = id(asgn)

				if group_key not in groups:
					groups[group_key] = (asgn, [])

				groups[group_key][1].append((note, pick))

		if not groups:
			return

		all_samples = self._instrument_library.samples()

		for asgn, note_picks in groups.values():

			# Resolve the full ranked list via the query engine.
			ranked: list[subsample.library.SampleRecord] = []

			for select_spec in asgn.select:
				ranked = subsample.query.query(select_spec, all_samples, self._similarity_matrix)

				if ranked:
					break

			if not ranked:
				continue

			notes = [n for n, _p in note_picks]

			# Repitch: all notes share pick=1 (same sample, pitched per note).
			if asgn.process.has_repitch():
				record = self._instrument_library.get(ranked[0].sample_id)

				if record is None:
					continue

				if not subsample.analysis.has_stable_pitch(record.spectral, record.pitch, record.duration):
					_log.warning(
						"Pitched %s: best match %r has no stable pitch — skipping pitch variants",
						asgn.name, record.name,
					)

				else:
					self._transform_manager.enqueue_pitch_range(record, notes)

					_log.info(
						"Pitched %s: queued %d variant(s) for %r",
						asgn.name, len(notes), record.name,
					)

			# Beat-quantize: each note may pick a different sample, so enqueue
			# a time-stretch variant for every distinct pick position.
			if asgn.process.has_beat_quantize():
				bpm, grid = _beat_quantize_params(asgn.process)
				enqueued = 0

				# Deduplicate by sample_id — multiple notes with the same pick
				# only need one variant.
				seen_ids: set[int] = set()

				for _note, pick in note_picks:
					idx = min(pick - 1, len(ranked) - 1)
					sid = ranked[idx].sample_id

					if sid in seen_ids:
						continue

					seen_ids.add(sid)
					record = self._instrument_library.get(sid)

					if record is None:
						continue

					if record.rhythm.tempo_bpm <= 0.0:
						_log.warning(
							"beat_quantize %s: sample %r (pick %d) has no detected tempo — "
							"will not be beat-quantized",
							asgn.name, record.name, pick,
						)
						continue

					self._transform_manager.get_at_bpm(
						sid,
						target_bpm=bpm,
						resolution=grid,
					)
					enqueued += 1

				if enqueued > 0:
					_log.info(
						"beat_quantize %s: queued %d variant(s)",
						asgn.name, enqueued,
					)

	# Backward-compatible alias — cli.py and tests call this name.
	update_pitched_assignments = update_assignments

	def reload_midi_map (self, new_map: NoteMap) -> None:

		"""Replace the active note map and re-compute transform variants.

		Thread-safe: dict assignment is atomic under the GIL.  The old map
		remains consistent for any in-flight _handle_message() call; the
		next call sees the new map.

		Args:
			new_map: Parsed NoteMap from load_midi_map().  Must be a complete
			         replacement (not a diff).
		"""

		old_count = len(self._note_map)
		self._note_map = new_map
		self.update_assignments()

		_log.info(
			"MIDI map reloaded: %d note(s) (was %d)",
			len(new_map), old_count,
		)

	def _render_float (
		self,
		audio: numpy.ndarray,
		level: subsample.analysis.LevelResult,
		velocity: int,
		pan_gains: numpy.ndarray,
		gain_db: float = 0.0,
	) -> numpy.ndarray:

		"""Apply gain normalisation and pan to an output-channel-count float32 array.

		Shared by both the original _render() path (mono from int PCM) and
		the transform variant path (float32 multi-channel from TransformResult).

		The input audio is first mixed to mono (all source channel counts are
		supported), then expanded to the output channel count using the
		constant-power pan_gains vector.

		Args:
			audio:     float32, shape (n_frames, in_channels).  Any channel count.
			level:     LevelResult for this audio (peak + rms), used for gain calc.
			velocity:  MIDI velocity (0–127) from the triggering note_on message.
			pan_gains: float32 array, shape (output_channels,).  Pre-computed
			           constant-power gains, one per output channel.  See
			           _parse_pan_gains() in load_midi_map().
			gain_db:   Per-assignment level offset in dB (from Assignment.gain_db).
			           Negative = quieter, positive = louder.

		Returns:
			float32 array, shape (n_frames, output_channels), values in [-1.0, 1.0].
		"""

		# --- Gain calculation ---
		vel_scale = (velocity / 127.0) ** 2  # quadratic — more musical than linear

		# Normalise to target RMS so samples recorded at different levels sound
		# balanced. Guard against silence (rms == 0) to avoid division by zero.
		if level.rms > 0.0:
			norm_gain = self._target_rms / level.rms
		else:
			norm_gain = 1.0

		# Per-assignment level offset from the MIDI map.
		gain_linear = 10.0 ** (gain_db / 20.0) if gain_db != 0.0 else 1.0

		raw_gain = norm_gain * vel_scale * gain_linear

		# Anti-clip ceiling: ensure gain × peak never exceeds full scale.
		if level.peak > 0.0:
			final_gain = min(raw_gain, 1.0 / level.peak)
		else:
			final_gain = raw_gain

		_log.debug(
			"gain: norm=%.3f  vel_scale=%.3f  gain_db=%.1f  raw=%.3f  final=%.3f  (rms=%.4f peak=%.4f)",
			norm_gain, vel_scale, gain_db, raw_gain, final_gain,
			level.rms, level.peak,
		)

		gained = audio * final_gain

		# Mix input to mono.  All source formats (mono, stereo, multichannel)
		# are reduced to a single channel before panning.  This is the correct
		# approach: we pan the *sound*, not individual source channels.
		mono: numpy.ndarray
		if gained.shape[1] == 1:
			mono = gained[:, 0]
		else:
			mono = typing.cast(numpy.ndarray, numpy.mean(gained, axis=1, dtype=numpy.float32))

		# Expand mono to the output channel layout using constant-power pan gains.
		# pan_gains shape: (output_channels,)
		# mono shape:      (n_frames,)
		# result shape:    (n_frames, output_channels)
		result: numpy.ndarray = (mono[:, numpy.newaxis] * pan_gains[numpy.newaxis, :]).astype(numpy.float32)
		return result
