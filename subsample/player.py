"""MIDI input listener and sample player for Subsample.

Handles MIDI input device selection (replicating the audio device selection
pattern from audio.py) and the MidiPlayer class which listens for MIDI events
and triggers polyphonic audio playback.

Mixing architecture: a PyAudio callback stream requests N frames at regular
intervals. Each triggered note adds a _Voice (pre-rendered multichannel float32
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
import random
import threading
import time
import typing

import mido
import numpy
import pyaudio
import soundfile  # type: ignore[import-untyped]
import yaml

import pymididefs.notes
import subsample.analysis
import subsample.audio
import subsample.bank
import subsample.channel
import subsample.cache
import subsample.config
import subsample.events
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


@dataclasses.dataclass(frozen=True)
class MidiMapResult:

	"""Complete result of parsing a MIDI map YAML file.

	Fields:
		note_map:         Note routing table: (mido_channel, midi_note) → (Assignment, pick).
		bank_definitions: Parsed bank declarations from the optional ``banks:`` key.
		                  Empty list when no banks are declared (single-directory mode).
		bank_channel:     MIDI channel for Program Change bank switching (user-facing 1-16,
		                  0 = omni).  Only meaningful when bank_definitions is non-empty.
		default_bank:     MIDI program number of the bank to activate at startup.
		                  None means use the first bank in the list.
	"""

	note_map:         NoteMap
	bank_definitions: list[subsample.bank.BankDefinition]
	bank_channel:     int
	default_bank:     typing.Optional[int] = None


def _quantize_params (
	process: subsample.query.ProcessSpec,
	step_name: str,
	config_bpm: float = 0.0,
) -> tuple[typing.Optional[float], int]:

	"""Extract BPM and grid from a beat_quantize or pad_quantize step.

	Returns (target_bpm, grid). When no explicit BPM is declared in the
	step, falls back to config_bpm (from transform.target_bpm in config).
	CcBinding values are treated as "provided" so the quantize path activates;
	the actual value is resolved later in spec_from_process().
	"""

	step = next(s for s in process.steps if s.name == step_name)
	bpm_raw = step.get("bpm", 0)
	grid_raw = step.get("grid", 16)

	# CcBinding means BPM will be resolved at note-on time — treat as "provided".
	if isinstance(bpm_raw, subsample.query.CcBinding):
		default = bpm_raw.default_value
		bpm = default if default is not None and default > 0 else (config_bpm if config_bpm > 0 else 120.0)
		grid = int(grid_raw) if not isinstance(grid_raw, subsample.query.CcBinding) else 16
		return (bpm, grid)

	bpm = float(bpm_raw)
	grid = int(grid_raw) if not isinstance(grid_raw, subsample.query.CcBinding) else 16

	if bpm <= 0:
		bpm = config_bpm

	return (bpm if bpm > 0 else None, grid)


def _build_beats_resolver (
	process: subsample.query.ProcessSpec,
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	session_bpm: float,
) -> typing.Optional[typing.Callable[[int], typing.Optional[float]]]:

	"""Build a callable that returns the quantized beat count for a sample.

	Inspects the assignment's process spec for a ``beat_quantize`` or
	``pad_quantize`` step and extracts the effective BPM and grid.  The
	returned callable looks up the matching variant via the transform
	manager and reads the ``GridEnergyProfile`` length to derive the
	number of beats as ``len(energy) * 4 / resolution``.

	Returns None (not a callable) when no valid quantize step is present,
	no transform manager is available, or the effective BPM is 0.
	"""

	if transform_manager is None:
		return None

	# Determine which quantize step (if any) applies.
	step: typing.Union[subsample.transform.TimeStretch, subsample.transform.PadQuantize]

	if process.has_beat_quantize():
		bpm, grid = _quantize_params(process, "beat_quantize", session_bpm)
		if bpm is None or bpm <= 0:
			return None
		step = subsample.transform.TimeStretch(target_bpm=float(bpm), resolution=int(grid))
	elif process.has_pad_quantize():
		bpm, grid = _quantize_params(process, "pad_quantize", session_bpm)
		if bpm is None or bpm <= 0:
			return None
		step = subsample.transform.PadQuantize(target_bpm=float(bpm), resolution=int(grid))
	elif session_bpm > 0:
		# Fall back to session-level beat_quantize.
		step = subsample.transform.TimeStretch(target_bpm=float(session_bpm), resolution=16)
	else:
		return None

	spec = subsample.transform.TransformSpec(steps=(step,))

	def _resolver (sample_id: int) -> typing.Optional[float]:
		result = transform_manager.get_variant(sample_id, spec)
		if result is None or result.energy_profile is None:
			return None
		profile = result.energy_profile
		return len(profile.energy) * 4.0 / profile.resolution

	return _resolver


def _parse_pan_weights (weights_raw: typing.Any, assignment_name: str) -> typing.Optional[numpy.ndarray]:

	"""Parse pan weights from YAML into a raw weight array.

	Pan weights define a target channel layout.  Their length determines the
	target (2 = stereo, 6 = 5.1, etc.).  Constant-power normalisation is
	applied later by channel.build_mix_matrix() when the actual output
	channel count is known.

	Args:
		weights_raw:     Raw YAML value (list of numbers, or None for default).
		assignment_name: Name for error messages.

	Returns:
		float32 numpy array of weights, or None if not specified (default routing).

	Raises:
		ValueError: If any weight is negative.
	"""

	if weights_raw is None:
		return None

	weights = list(weights_raw)
	weight_arr = numpy.array(weights, dtype=numpy.float32)

	if numpy.any(weight_arr < 0):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pan weights must be >= 0"
		)

	if float(numpy.sum(weight_arr)) == 0.0:
		_log.warning("Assignment %r: all pan weights are zero — note will be silent", assignment_name)

	return weight_arr


def _parse_output_routing (
	raw: typing.Any,
	assignment_name: str,
	pan_weights: typing.Optional[numpy.ndarray],
) -> typing.Optional[tuple[int, ...]]:

	"""Parse output routing from YAML into a 0-indexed channel tuple.

	The MIDI map uses 1-indexed output numbers (matching hardware labels).
	This function converts to 0-indexed for internal use.  Device-range
	validation is deferred to runtime (the device channel count is not
	known at parse time).

	Args:
		raw:             Raw YAML value (list of ints, or None for default).
		assignment_name: Name for error messages.
		pan_weights:     Parsed pan weights (for length validation).

	Returns:
		Tuple of 0-indexed device channel indices, or None for default routing.

	Raises:
		ValueError: On invalid values, duplicates, or length mismatch with pan.
	"""

	if raw is None:
		return None

	channels = list(raw)

	if not channels:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: output must be a non-empty list"
		)

	for ch in channels:
		if not isinstance(ch, int) or ch < 1:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: output channels must be "
				f"positive integers (1-indexed), got {ch!r}"
			)

	if len(set(channels)) != len(channels):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: output contains duplicate "
			f"channels: {channels}"
		)

	if pan_weights is not None and len(channels) != len(pan_weights):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: output length ({len(channels)}) "
			f"must match pan length ({len(pan_weights)})"
		)

	return tuple(ch - 1 for ch in channels)


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
			except ValueError:
				n = None

			if n is not None:
				if not 0 <= n <= 127:
					raise ValueError(
						f"MIDI map assignment {assignment_name!r}: note {n} is outside [0, 127]"
					)
				return n

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


def _load_reference_from_path (path: pathlib.Path) -> typing.Optional[subsample.library.SampleRecord]:

	"""Load a reference sample from a filesystem path.

	If the analysis sidecar does not exist but the audio file does, the
	sidecar is generated automatically so users can point at any WAV file
	as a reference without running the analysis script first.

	The reference sample's name (key in the similarity matrix) is set to
	the canonical absolute path string so that get_matches(str(path)) works.

	Args:
		path: Absolute path to the WAV file.

	Returns:
		SampleRecord with name=str(path.resolve()), or None on failure.
	"""

	path = pathlib.Path(path).resolve()
	sidecar_path = subsample.cache.cache_path(path)

	# Auto-generate sidecar if missing but the audio file exists.
	if not sidecar_path.exists() and path.exists():
		_log.info("Generating analysis sidecar for reference %s", path.name)

		try:
			data, samplerate = soundfile.read(str(path), always_2d=True, dtype="float32")
			mono: numpy.ndarray = numpy.asarray(numpy.mean(data, axis=1, dtype=numpy.float32))

			params = subsample.analysis.compute_params(samplerate)
			rhythm_cfg = subsample.config.AnalysisConfig()
			spectral, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(
				mono, params, rhythm_cfg,
			)
			duration = len(data) / samplerate

			audio_md5 = subsample.cache.compute_audio_md5(path)
			subsample.cache.save_cache(
				audio_path  = path,
				audio_md5   = audio_md5,
				params      = params,
				spectral    = spectral,
				rhythm      = rhythm,
				pitch       = pitch,
				timbre      = timbre,
				duration    = duration,
				level       = level,
				band_energy = band_energy,
			)
		except Exception as exc:
			# Broad catch: soundfile.read() can raise LibsndfileError (not an
			# OSError subclass), analysis can raise numpy/librosa errors.
			# All are non-fatal — the reference is simply skipped.
			_log.warning("Could not generate sidecar for %s: %s", path.name, exc)
			return None

	if not sidecar_path.exists():
		_log.warning(
			"Reference sample sidecar not found for %s — this reference will be skipped",
			path,
		)
		return None

	result = subsample.cache.load_sidecar(sidecar_path)
	if result is None:
		_log.warning(
			"Failed to load analysis sidecar for %s — this reference will be skipped",
			path,
		)
		return None

	spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result

	return subsample.library.SampleRecord(
		sample_id   = subsample.library.allocate_id(),
		name        = str(path.resolve()),
		spectral    = spectral,
		rhythm      = rhythm,
		pitch       = pitch,
		timbre      = timbre,
		level       = level,
		band_energy = band_energy,
		params      = params,
		duration    = duration,
		audio       = None,
		filepath    = path if path.exists() else None,
	)


def _load_instrument_from_path (
	path: pathlib.Path,
	target_sample_rate: typing.Optional[int] = None,
) -> typing.Optional[subsample.library.SampleRecord]:

	"""Load an instrument sample (WAV + sidecar) from a filesystem path.

	The sample's name is set to the filename stem (e.g. "2026-03-27_09-28-12")
	so that it can be matched by where: { name: ... } predicates.

	Args:
		path:               Absolute path to the WAV file (or sidecar).
		target_sample_rate: When set, resample audio to this rate on load.

	Returns:
		SampleRecord with audio loaded, or None if loading fails.
	"""

	path = pathlib.Path(path).resolve()
	name = path.stem

	if not path.exists():
		_log.warning(
			"Instrument sample WAV not found: %s — this sample will be skipped",
			path,
		)
		return None

	result = subsample.cache.load_or_analyze(path)
	if result is None:
		_log.warning(
			"Failed to load or analyze %s — this sample will be skipped",
			path,
		)
		return None

	spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result

	# Load the audio data, resampling to the output rate if needed.
	audio = subsample.library.load_wav_audio(path, target_sample_rate)
	if audio is None:
		_log.warning(
			"Failed to load audio from %s — this sample will be skipped",
			path,
		)
		return None

	return subsample.library.SampleRecord(
		sample_id   = subsample.library.allocate_id(),
		name        = name,
		spectral    = spectral,
		rhythm      = rhythm,
		pitch       = pitch,
		timbre      = timbre,
		level       = level,
		band_energy = band_energy,
		params      = params,
		duration    = duration,
		audio       = audio,
		filepath    = path,
	)


def _reference_wav_path (assignment: subsample.query.Assignment) -> typing.Optional[str]:

	"""Resolve the reference sample WAV path for an assignment.

	Returns the absolute path string if the assignment's primary select spec
	has a path-based reference, or None otherwise.  Used by the vocoder
	processor to resolve ``carrier: reference``.
	"""

	if not assignment.select:
		return None

	ref = assignment.select[0].where.reference

	if ref is None:
		return None

	if subsample.query.is_path_like(ref):
		resolved = pathlib.Path(ref).resolve()

		if resolved.exists():
			return str(resolved)

	return None


def _resolve_path_references (
	note_map: NoteMap,
	matrices: list[subsample.similarity.SimilarityMatrix],
	instrument_lib: subsample.library.InstrumentLibrary,
	target_sample_rate: typing.Optional[int] = None,
) -> None:

	"""Load path-based references, instruments, and directory samples from the MIDI map.

	Scans all assignments in the note map for:
	  - Path-based references → loaded and added to similarity matrices
	  - Path-based instruments → loaded and added to instrument library
	  - Directory predicates → all samples in the directory loaded into instrument library

	Args:
		note_map:            Note routing table: (mido_channel, midi_note) → (Assignment, pick).
		matrices:            List of SimilarityMatrix (one per bank) to add references to.
		instrument_lib:      InstrumentLibrary to add path-based instruments to.
		target_sample_rate:  When set, resample loaded audio to this rate.
	"""

	# Collect unique paths for references, instruments, and directories
	ref_paths: set[str] = set()
	inst_paths: set[str] = set()
	dir_paths: set[str] = set()

	# Extract unique assignments from the note map
	seen_assignments: set[int] = set()

	for (assignment, pick) in note_map.values():
		assignment_id = id(assignment)
		if assignment_id in seen_assignments:
			continue
		seen_assignments.add(assignment_id)

		for select_spec in assignment.select:
			ref = select_spec.where.reference
			if ref is not None and subsample.query.is_path_like(ref):
				ref_paths.add(ref)

			name_path = select_spec.where.name_path
			if name_path is not None:
				inst_paths.add(name_path)

			if select_spec.where.directory is not None:
				dir_paths.add(select_spec.where.directory)

	# Load samples from directory predicates into the instrument library.
	# This must happen before reference loading so that directory samples
	# are available for similarity scoring.
	for dir_path in sorted(dir_paths):
		directory = pathlib.Path(dir_path)

		if not directory.is_dir():
			_log.warning("MIDI map directory predicate: %s is not a directory — skipped", dir_path)
			continue

		loaded = 0

		try:
			sidecars = sorted(directory.glob("*.analysis.json"))
		except (PermissionError, OSError) as exc:
			_log.warning("Cannot read directory %s: %s — skipped", dir_path, exc)
			continue

		for sidecar in sidecars:
			wav_name = sidecar.name.replace(".analysis.json", "")
			wav_path = directory / wav_name
			name = pathlib.Path(wav_name).stem

			# Skip if already in the library.
			if instrument_lib.find_by_name(name) is not None:
				continue

			record = _load_instrument_from_path(wav_path, target_sample_rate)

			if record is not None:
				instrument_lib.add(record)
				loaded += 1

		if loaded > 0:
			_log.info("Loaded %d sample(s) from directory predicate %s", loaded, dir_path)

	# Load path-based references and add to all matrices
	for ref_path in ref_paths:
		path = pathlib.Path(ref_path)
		record = _load_reference_from_path(path)
		if record is None:
			continue

		# Add to every bank's similarity matrix
		instruments = instrument_lib.samples()
		for matrix in matrices:
			matrix.add_reference(record, instruments)

		_log.debug("Added path-based reference from %s", path)

	# Load path-based instruments and add to library
	for inst_path in inst_paths:
		path = pathlib.Path(inst_path)
		name = path.stem

		# Skip if already in the library
		existing_id = instrument_lib.find_by_name(name)
		if existing_id is not None:
			_log.debug(
				"Instrument sample %s already in library (id %d) — skipping load from %s",
				name, existing_id, path,
			)
			continue

		record = _load_instrument_from_path(path, target_sample_rate)
		if record is None:
			continue

		instrument_lib.add(record)

		_log.debug("Added path-based instrument from %s", path)


def load_midi_map (
	path: pathlib.Path,
	reference_names: list[str],
) -> MidiMapResult:

	"""Load a MIDI routing map from a YAML file.

	Parses the assignments list using the select/process pipeline format
	and returns a MidiMapResult containing the note map, any bank
	definitions, and the bank channel.

	Each assignment declares:
	  select:   Which sample to play — filter predicates, ordering, pick position.
	            Can be a single spec or a list (fallback chain, tried in order).
	  process:  How to present it — ordered list of processors (repitch, beat_quantize, etc.).
	  one_shot: Playback behaviour — true (default) ignores note_off.
	  gain:     Level offset in dB (default 0.0).
	  pan:      Channel weights defining a target layout (e.g. [50, 50] for stereo,
	            [50, 50, 0, 0, 30, 30] for 5.1).  Omit for default routing.

	reference predicates whose name is not in reference_names are skipped
	with a WARNING — this prevents silent failures when using a map built
	for a different reference library.

	Path-based references (containing "/" or starting with ".") are resolved
	relative to the MIDI map file's directory and added to the reference set
	(without validation against reference_names).

	Args:
		path:             Path to the MIDI map YAML file.
		reference_names:  Names from the loaded reference library (case-insensitive).

	Returns:
		MidiMapResult containing the NoteMap, bank definitions, and bank channel.

	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError:        If the YAML is malformed or a required field is missing.
	"""

	if not path.exists():
		raise FileNotFoundError(f"MIDI map not found: {path}")

	midi_map_dir = path.parent

	with path.open(encoding="utf-8") as fh:
		raw = yaml.safe_load(fh)

	if raw is None:
		_log.warning("MIDI map %s is empty — no notes will be mapped", path)
		return MidiMapResult(
			note_map={},
			bank_definitions=[],
			bank_channel=subsample.bank.DEFAULT_BANK_CHANNEL,
		)

	# Parse optional bank definitions.
	bank_definitions = subsample.bank.parse_banks(raw.get("banks"))
	bank_channel = int(raw.get("bank_channel", subsample.bank.DEFAULT_BANK_CHANNEL))
	raw_default_bank = raw.get("default_bank")
	default_bank: typing.Optional[int] = int(raw_default_bank) if raw_default_bank is not None else None

	if "assignments" not in raw:
		_log.warning("MIDI map %s has no assignments — no notes will be mapped", path)
		return MidiMapResult(
			note_map={},
			bank_definitions=bank_definitions,
			bank_channel=bank_channel,
			default_bank=default_bank,
		)

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

		select_specs = subsample.query.parse_select(select_raw, name, midi_map_dir)

		# Validate reference predicates against the loaded reference library.
		# Path-based references (containing "/" or starting with ".") are resolved
		# at parse time and don't need to be in the reference library.
		valid = True

		for spec in select_specs:
			ref = spec.where.reference

			# Skip validation for path-based references (those with "/" in them
			# are absolute paths resolved at parse time)
			if ref is not None and "/" not in ref and ref.upper() not in reference_set:
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
		pan_weights    = _parse_pan_weights(assignment_raw.get("pan"), name)
		output_routing = _parse_output_routing(assignment_raw.get("output"), name, pan_weights)

		# Extract segment playback mode from quantize step parameters.
		segment_mode: typing.Union[str, int] = ""

		for step in process.steps:
			if step.name in ("beat_quantize", "pad_quantize"):
				raw_seg = step.get("segment", "")

				if isinstance(raw_seg, int) and raw_seg > 0:
					segment_mode = raw_seg
				elif isinstance(raw_seg, str) and raw_seg in ("round_robin", "random"):
					segment_mode = raw_seg
				elif raw_seg:
					_log.warning(
						"Assignment %r: invalid segment mode %r — using merged playback",
						name, raw_seg,
					)

				break

		assignment = subsample.query.Assignment(
			name=name,
			select=select_specs,
			process=process,
			one_shot=one_shot,
			gain_db=gain_db,
			pan_weights=pan_weights,
			output_routing=output_routing,
			segment_mode=segment_mode,
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
		"MIDI map loaded from %s: %d note(s) across %d assignment(s)%s",
		path,
		len(note_map),
		len(raw.get("assignments", [])),
		f", {len(bank_definitions)} bank(s)" if bank_definitions else "",
	)

	return MidiMapResult(
		note_map=note_map,
		bank_definitions=bank_definitions,
		bank_channel=bank_channel,
		default_bank=default_bank,
	)


@dataclasses.dataclass
class _Voice:

	"""A single triggered sample being played back by the mix callback.

	audio:     Pre-rendered float32 array, shape (n_frames, output_channels),
	           in [-1.0, 1.0]. Gain has already been applied. The callback
	           reads from this array; it is never modified after creation.
	note:      MIDI note number that triggered this voice — used to match
	           note_off events in _handle_message().
	channel:   MIDI channel (mido 0-indexed) that triggered this voice.
	position:  Current read cursor in frames. Advances each callback call.
	           Voice is removed when position >= len(audio).
	releasing: Set to True when a note_off arrives for this note+channel
	           (only for non-one-shot voices).  The callback applies a short
	           cosine fade-out over the player's _release_fade_frames frames, then retires.
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


_CC_DEBOUNCE_SECONDS: float = 0.2


def _collect_mapped_ccs (note_map: NoteMap) -> set[int]:

	"""Return the set of CC numbers used by CcBinding params in the note map."""

	ccs: set[int] = set()

	for assignment, _ in note_map.values():
		for step in assignment.process.steps:
			for _, value in step.params:
				if isinstance(value, subsample.query.CcBinding):
					ccs.add(value.cc)

	return ccs


class MidiPlayer:

	"""Listens for MIDI messages and plays back instrument samples polyphonically.

	Designed to run on its own thread. Call run() as the thread target;
	it blocks until shutdown_event is set, then closes the MIDI port and
	PyAudio stream and returns cleanly.

	Note routing is loaded from a YAML file by the caller via load_midi_map()
	and passed as the `midi_map` parameter.  The map keys notes by
	(mido_channel, midi_note) and stores (Assignment, pick_position) per note.
	At trigger time, the query engine evaluates the assignment's select chain
	against the active instrument library to find the best-matching sample.

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
		bank_manager: typing.Optional[subsample.bank.BankManager] = None,
		target_bpm: float = 0.0,
		output_channels: typing.Optional[int] = None,
	) -> None:

		self._device_name        = device_name
		self._shutdown_event     = shutdown_event
		self._instrument_library = instrument_library
		self._similarity_matrix  = similarity_matrix
		self._target_bpm         = target_bpm

		# Bank manager: when provided, the player delegates library, similarity,
		# and transform lookups to the active bank.  When None, the player uses
		# the directly-passed instances (single-directory backward compat).
		self._bank_manager       = bank_manager
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

		# Number of output channels.  Determines the shape of the mix buffer
		# and must match the pa.open(channels=...) call in run().  Defaults
		# to 2 (stereo); set via player.audio.channels in config for
		# multi-channel interfaces.
		self._output_channels: int = output_channels if output_channels is not None else 2

		# Note routing map: (mido_channel, midi_note) → (Assignment, pick_position).
		# Loaded from the MIDI map YAML file by the caller (cli.py) via load_midi_map().
		self._note_map: NoteMap = midi_map

		# Most recently played variant per (channel, note).  Used as a fallback
		# during MIDI map transitions: when a new variant is still processing,
		# the old one plays instead of the unprocessed base — giving smooth
		# transitions for gradual BPM or amount changes.
		self._last_played: dict[tuple[int, int], subsample.transform.TransformResult] = {}

		# Event emitter for integrations (Supervisor dashboard, etc.).
		# Currently emits 'cc' on control_change messages.
		self.events = subsample.events.EventEmitter()

		# MIDI CC state: (mido_channel, cc_number) → current value (0–127).
		# Updated on every control_change message; read at note-on time by
		# spec_from_process() to resolve CcBinding parameters.
		self._cc_state: dict[tuple[int, int], int] = {}

		# Omni CC state: cc_number → most recent value from any channel.
		# Used by _resolve_cc for omni CcBindings (channel=None) so the most
		# recent CC update wins regardless of which channel sent it.
		self._cc_omni: dict[int, int] = {}

		# Set of CC numbers that are mapped to processor parameters in the
		# current MIDI map.  Used for O(1) "is this CC relevant?" checks.
		self._mapped_ccs: set[int] = _collect_mapped_ccs(midi_map)

		# Debounce timer for CC-triggered re-evaluation.
		self._cc_debounce_timer: typing.Optional[threading.Timer] = None
		self._cc_debounce_lock: threading.Lock = threading.Lock()

		# Throttle for CC INFO log — at most one per CC number per second.
		self._cc_last_log: dict[int, float] = {}

		# Mix matrix cache: (in_channels, pan_weights_tuple, output_routing) → matrix.
		# Lazily populated by _get_mix_matrix(). Cleared on MIDI map reload.
		_MixCacheKey = tuple[int, typing.Optional[tuple[float, ...]], typing.Optional[tuple[int, ...]]]
		self._mix_matrix_cache: dict[_MixCacheKey, numpy.ndarray] = {}

		# Per-note segment counter for round-robin segment playback.
		# Cleared on MIDI map reload and bank switch.
		self._segment_counters: dict[tuple[int, int], int] = {}

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

			if asgn.process.has_pad_quantize():
				line += " pad-quantized"

			if asgn.one_shot:
				line += "  one-shot"

			if asgn.pan_weights is not None:
				line += f"  pan=[{', '.join(f'{g:.0f}' for g in asgn.pan_weights)}]"
			lines.append(line)

		_log.info(
			"MIDI note map: %d note(s) loaded\n  %s",
			len(self._note_map),
			"\n  ".join(lines),
		)

	# -- Effective delegates -----------------------------------------------
	# When a BankManager is present, the player delegates library, similarity,
	# and transform lookups to the active bank.  When None (single-directory
	# mode), the directly-passed instances are used.

	@property
	def _effective_instrument_library (self) -> subsample.library.InstrumentLibrary:
		if self._bank_manager is not None:
			return self._bank_manager.active_bank.instrument_library
		return self._instrument_library

	@property
	def _effective_similarity_matrix (self) -> subsample.similarity.SimilarityMatrix:
		if self._bank_manager is not None:
			return self._bank_manager.active_bank.similarity_matrix
		return self._similarity_matrix

	@property
	def _effective_transform_manager (self) -> typing.Optional[subsample.transform.TransformManager]:
		if self._bank_manager is not None:
			tm: typing.Any = self._bank_manager.active_bank.transform_manager
			return typing.cast(typing.Optional[subsample.transform.TransformManager], tm)
		return self._transform_manager

	def run (self) -> None:

		"""Open MIDI input and a callback output stream, then dispatch events.

		Blocks until shutdown_event is set. Both the MIDI port and the PyAudio
		stream are closed in the finally block.

		Input port selection:
		  - virtual_midi_port set → create a named virtual port (other apps connect to it)
		  - otherwise → open the hardware device by device_name
		"""

		pa = subsample.audio.create_pyaudio()

		# Resolve output device — mirrors the input device selection pattern.
		output_devices = subsample.audio.list_output_devices(pa)

		if self._output_device_name is not None:
			try:
				output_device_index: int = subsample.audio.find_output_device_by_name(
					pa, self._output_device_name,
				)
			except ValueError:
				_log.warning(
					"Configured audio output device %r not found — prompting for selection",
					self._output_device_name,
				)
				output_device_index = subsample.audio.select_output_device(output_devices)
		else:
			output_device_index = subsample.audio.select_output_device(output_devices)

		# Validate output routing indices against the resolved device channel count.
		# Collect replacements first to avoid mutating the dict during iteration.
		routing_fixes: dict[tuple[int, int], tuple[subsample.query.Assignment, int]] = {}

		for (ch, note), (assignment, _pick) in self._note_map.items():
			routing = assignment.output_routing
			if routing is not None:
				for idx in routing:
					if idx >= self._output_channels:
						_log.warning(
							"Assignment %r (ch %d, note %d): output index %d exceeds "
							"device channel count (%d) — using default routing",
							assignment.name, ch, note, idx + 1, self._output_channels,
						)
						routing_fixes[(ch, note)] = (
							subsample.query.Assignment(
								name=assignment.name,
								select=assignment.select,
								process=assignment.process,
								one_shot=assignment.one_shot,
								gain_db=assignment.gain_db,
								pan_weights=assignment.pan_weights,
								output_routing=None,
								pick=assignment.pick,
							),
							_pick,
						)
						break

		self._note_map.update(routing_fixes)

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
			with self._cc_debounce_lock:
				if self._cc_debounce_timer is not None:
					self._cc_debounce_timer.cancel()

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
		intervals. Must return quickly and avoid blocking. Clipping detection
		is logged at WARNING with per-second throttling.

		Sums all active _Voice arrays into a float32 mix, clips to [-1, 1],
		converts to PCM bytes at the output bit depth, and returns the bytes.
		Finished voices (cursor past end of audio) are removed from the list.

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

		# Program Change: switch the active instrument bank when a BankManager
		# is configured and the message arrives on the designated bank channel.
		if msg.type == "program_change" and self._bank_manager is not None:
			bm = self._bank_manager
			if bm.bank_channel_mido == -1 or msg.channel == bm.bank_channel_mido:
				if bm.switch_to(msg.program):
					self._last_played.clear()
					self._segment_counters.clear()
					self.update_assignments()
			return

		# Control Change: update CC state and debounce re-evaluation for
		# mapped parameters (CcBinding in the process pipeline).
		if msg.type == "control_change":
			self._cc_state[(msg.channel, msg.control)] = msg.value
			self._cc_omni[msg.control] = msg.value
			self.events.emit("cc", channel=msg.channel, cc_number=msg.control, value=msg.value)

			if msg.control in self._mapped_ccs:
				_log.debug(
					"CC ch%d #%d = %d",
					msg.channel + 1, msg.control, msg.value,
				)

				now = time.monotonic()
				last = self._cc_last_log.get(msg.control, 0.0)

				if now - last >= 1.0:
					self._cc_last_log[msg.control] = now
					_log.info(
						"CC ch%d #%d = %d (mapped)",
						msg.channel + 1, msg.control, msg.value,
					)

				with self._cc_debounce_lock:
					if self._cc_debounce_timer is not None:
						self._cc_debounce_timer.cancel()

					self._cc_debounce_timer = threading.Timer(
						_CC_DEBOUNCE_SECONDS, self.update_assignments,
					)
					self._cc_debounce_timer.start()

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
		pan_weights    = assignment.pan_weights
		output_routing = assignment.output_routing
		one_shot       = assignment.one_shot

		# ── Sample selection via query engine ─────────────────────────────

		eff_library    = self._effective_instrument_library
		eff_similarity = self._effective_similarity_matrix
		eff_transform  = self._effective_transform_manager
		all_samples    = eff_library.samples()
		sample_id: typing.Optional[int] = None

		beats_resolver = _build_beats_resolver(
			assignment.process, eff_transform, self._target_bpm,
		)

		for select_spec in assignment.select:

			ranked = subsample.query.query(
				select_spec, all_samples, eff_similarity, beats_resolver,
			)

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

		record = eff_library.get(sample_id)

		if record is None or record.audio is None:
			_log.debug("Sample %d not found or audio not loaded", sample_id)
			return

		# ── Variant lookup based on ProcessSpec ───────────────────────────

		eff_transform = self._effective_transform_manager

		if eff_transform is not None:

			# Build the full ordered transform chain from the process spec.
			# Dynamic parameters (MIDI note, BPM) are substituted at the
			# position the user declared them in the process: list.
			if assignment.process.steps:

				# Validation: skip repitch for unpitched samples.
				midi_note_for_spec: typing.Optional[int] = None

				if assignment.process.has_repitch():
					if subsample.analysis.has_stable_pitch(record.spectral, record.pitch, record.duration):
						midi_note_for_spec = msg.note

				# Validation: skip beat_quantize for samples with no tempo.
				# pad_quantize does NOT need source tempo — only target BPM.
				bpm_for_spec: typing.Optional[float] = None
				grid_for_spec = 16

				if assignment.process.has_beat_quantize():
					if record.rhythm.tempo_bpm > 0.0:
						bpm_for_spec, grid_for_spec = _quantize_params(assignment.process, "beat_quantize", self._target_bpm)
					else:
						_log.warning(
							"beat_quantize %s: sample %r has no detected tempo — "
							"playing without beat-quantizing",
							assignment.name, record.name,
						)

				if assignment.process.has_pad_quantize():
					bpm_for_spec, grid_for_spec = _quantize_params(assignment.process, "pad_quantize", self._target_bpm)

				spec = subsample.transform.spec_from_process(
					assignment.process,
					midi_note=midi_note_for_spec,
					target_bpm=bpm_for_spec,
					resolution=grid_for_spec,
					reference_path=_reference_wav_path(assignment),
					cc_state=self._cc_state,
					cc_omni=self._cc_omni,
				)

				if spec.steps:
					variant = eff_transform.get_variant(sample_id, spec)

					if variant is not None:
						seg_audio, seg_level = self._select_segment(
							variant.audio, variant.level, variant.segment_bounds,
							assignment.segment_mode, msg.channel, msg.note,
						)
						mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing)
						rendered = self._render_float(seg_audio, seg_level, msg.velocity, mix_mat, assignment.gain_db)
						with self._voices_lock:
							self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
						self._last_played[(msg.channel, msg.note)] = variant
						_log.debug(
							"note %d (vel %d) → %r → %r (variant, %d step(s))  (%.2fs)",
							msg.note, msg.velocity, assignment.name, record.name,
							len(spec.steps), variant.duration,
						)
						return

					# New variant not ready — try the previously played variant
					# for this note (smooth transition during gradual param changes).
					prev = self._last_played.get((msg.channel, msg.note))

					if prev is not None and prev.key.sample_id == sample_id:
						seg_audio, seg_level = self._select_segment(
							prev.audio, prev.level, prev.segment_bounds,
							assignment.segment_mode, msg.channel, msg.note,
						)
						mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing)
						rendered = self._render_float(seg_audio, seg_level, msg.velocity, mix_mat, assignment.gain_db)
						with self._voices_lock:
							self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
						_log.debug(
							"note %d (vel %d) → %r → %r (previous variant)  (%.2fs)",
							msg.note, msg.velocity, assignment.name, record.name, prev.duration,
						)
						return

			# Fall back to the base variant (float32, peak-normalised, no DSP).
			base = eff_transform.get_base(sample_id)

			if base is not None:
				seg_audio, seg_level = self._select_segment(
					base.audio, base.level, base.segment_bounds,
					assignment.segment_mode, msg.channel, msg.note,
				)
				mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing)
				rendered = self._render_float(seg_audio, seg_level, msg.velocity, mix_mat, assignment.gain_db)
				with self._voices_lock:
					self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
				_log.debug(
					"note %d (vel %d) → %r → %r (base variant)  (%.2fs)",
					msg.note, msg.velocity, assignment.name, record.name, base.duration,
				)
				return

		# 4. Last resort: convert from int PCM on this trigger.
		mix_mat = self._get_mix_matrix(record.audio.shape[1] if record.audio is not None else 1, pan_weights, output_routing)
		original: typing.Optional[numpy.ndarray] = self._render(record, msg.velocity, mix_mat, assignment.gain_db)

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
		mix_matrix: numpy.ndarray,
		gain_db: float = 0.0,
	) -> typing.Optional[numpy.ndarray]:

		"""Convert a SampleRecord to a gain-adjusted, channel-mapped output array.

		Converts int PCM → float32 preserving all channels → applies gain and
		channel mapping via mix_matrix → returns output-channel-count float32.
		Returns None if the record has no audio.
		"""

		if record.audio is None:
			return None

		float_audio = subsample.transform._pcm_to_float32(record.audio, self._bit_depth)

		return self._render_float(float_audio, record.level, velocity, mix_matrix, gain_db)

	def update_assignments (self) -> None:

		"""Pre-compute transform variants for all assignments that declare processors.

		Groups notes by Assignment, resolves each to its current sample via the
		query engine, and enqueues the appropriate variants.  The full ordered
		process chain (filters, saturate, reverse, etc.) is included alongside
		repitch / beat_quantize via spec_from_process().

		The TransformProcessor deduplicates in-flight and cached keys, so
		repeated calls are safe and cheap.

		Call this:
		  - At startup, after the similarity matrix is populated.
		  - In the on_complete callback after a new sample arrives — ensures
		    variants are ready before the next trigger.

		No-ops if no transform manager is configured or no processable
		assignments exist.
		"""

		eff_transform = self._effective_transform_manager

		if eff_transform is None:
			return

		eff_library    = self._effective_instrument_library
		eff_similarity = self._effective_similarity_matrix

		# Group notes by Assignment identity (object id) — all notes in the same
		# assignment share the same select/process spec.  Collect (note, pick)
		# pairs so beat_quantize can pre-compute a variant for every pick
		# position (each note may resolve to a different sample).
		groups: dict[int, tuple[subsample.query.Assignment, list[tuple[int, int]]]] = {}

		for (_ch, note), (asgn, pick) in self._note_map.items():

			if asgn.process.steps:
				group_key = id(asgn)

				if group_key not in groups:
					groups[group_key] = (asgn, [])

				groups[group_key][1].append((note, pick))

		if not groups:
			return

		all_samples = eff_library.samples()
		_total_assignments = 0
		_total_variants = 0

		for asgn, note_picks in groups.values():

			# Resolve the full ranked list via the query engine.
			ranked: list[subsample.library.SampleRecord] = []

			beats_resolver = _build_beats_resolver(
				asgn.process, eff_transform, self._target_bpm,
			)

			for select_spec in asgn.select:
				ranked = subsample.query.query(
					select_spec, all_samples, eff_similarity, beats_resolver,
				)

				if ranked:
					break

			if not ranked:
				continue

			notes = [n for n, _p in note_picks]

			# Repitch: all notes share pick=1 (same sample, pitched per note).
			# The full process chain is passed so variants include filters, etc.
			if asgn.process.has_repitch():
				record = eff_library.get(ranked[0].sample_id)

				if record is None:
					continue

				if not subsample.analysis.has_stable_pitch(record.spectral, record.pitch, record.duration):
					_log.warning(
						"Pitched %s: best match %r has no stable pitch — skipping pitch variants",
						asgn.name, record.name,
					)

				else:
					eff_transform.enqueue_pitch_range(record, notes, process=asgn.process)
					_total_assignments += 1
					_total_variants += len(notes)

					_log.debug(
						"Pitched %s: queued %d variant(s) for %r",
						asgn.name, len(notes), record.name,
					)

			# Beat-quantize: each note may pick a different sample, so enqueue
			# a variant for every distinct pick position.  The full process
			# chain is included via spec_from_process().
			elif asgn.process.has_beat_quantize():
				bpm, grid = _quantize_params(asgn.process, "beat_quantize", self._target_bpm)
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
					record = eff_library.get(sid)

					if record is None:
						continue

					if record.rhythm.tempo_bpm <= 0.0:
						_log.warning(
							"beat_quantize %s: sample %r (pick %d) has no detected tempo — "
							"will not be beat-quantized",
							asgn.name, record.name, pick,
						)
						continue

					spec = subsample.transform.spec_from_process(
						asgn.process,
						target_bpm=bpm,
						resolution=grid,
						reference_path=_reference_wav_path(asgn),
						cc_state=self._cc_state,
						cc_omni=self._cc_omni,
					)
					eff_transform.get_variant(sid, spec)
					enqueued += 1

				if enqueued > 0:
					_total_assignments += 1
					_total_variants += enqueued

					_log.debug(
						"beat_quantize %s: queued %d variant(s)",
						asgn.name, enqueued,
					)

			# Pad-quantize: same dedup pattern as beat_quantize but no
			# tempo check — pad_quantize only needs onsets, not source tempo.
			elif asgn.process.has_pad_quantize():
				bpm, grid = _quantize_params(asgn.process, "pad_quantize", self._target_bpm)
				enqueued = 0
				seen_ids_pad: set[int] = set()

				for _note, pick in note_picks:
					idx = min(pick - 1, len(ranked) - 1)
					sid = ranked[idx].sample_id

					if sid in seen_ids_pad:
						continue

					seen_ids_pad.add(sid)
					record = eff_library.get(sid)

					if record is None:
						continue

					spec = subsample.transform.spec_from_process(
						asgn.process,
						target_bpm=bpm,
						resolution=grid,
						reference_path=_reference_wav_path(asgn),
						cc_state=self._cc_state,
						cc_omni=self._cc_omni,
					)
					eff_transform.get_variant(sid, spec)
					enqueued += 1

				if enqueued > 0:
					_total_assignments += 1
					_total_variants += enqueued

					_log.debug(
						"pad_quantize %s: queued %d variant(s)",
						asgn.name, enqueued,
					)

			# Process-only (no repitch, no beat/pad_quantize): pre-compute the
			# static chain (filters, saturate, reverse, etc.) once per sample.
			else:
				spec = subsample.transform.spec_from_process(
					asgn.process,
					reference_path=_reference_wav_path(asgn),
					cc_state=self._cc_state,
					cc_omni=self._cc_omni,
				)

				if spec.steps:
					seen_ids_static: set[int] = set()

					for _note, pick in note_picks:
						idx = min(pick - 1, len(ranked) - 1)
						sid = ranked[idx].sample_id

						if sid in seen_ids_static:
							continue

						seen_ids_static.add(sid)
						eff_transform.get_variant(sid, spec)

					_total_assignments += 1
					_total_variants += len(seen_ids_static)

					_log.debug(
						"Process %s: queued %d variant(s) (%d step(s))",
						asgn.name, len(seen_ids_static), len(spec.steps),
					)

		if _total_assignments > 0:
			_log.info(
				"Assignments: %d with process chains, %d variant(s) queued",
				_total_assignments, _total_variants,
			)

	# Backward-compatible alias — cli.py and tests call this name.
	update_pitched_assignments = update_assignments

	def reload_midi_map (self, new_map: NoteMap) -> None:

		"""Replace the active note map and re-compute transform variants.

		Thread-safe: dict assignment is atomic under the GIL.  The old map
		remains consistent for any in-flight _handle_message() call; the
		next call sees the new map.

		Args:
			new_map: Parsed NoteMap from load_midi_map().note_map.  Must be a
			         complete replacement (not a diff).
		"""

		old_count = len(self._note_map)
		self._note_map = new_map
		self._mapped_ccs = _collect_mapped_ccs(new_map)
		self._mix_matrix_cache.clear()
		self._segment_counters.clear()
		self.update_assignments()

		_log.info(
			"MIDI map reloaded: %d note(s) (was %d)",
			len(new_map), old_count,
		)

	def _select_segment (
		self,
		audio: numpy.ndarray,
		level: subsample.analysis.LevelResult,
		segment_bounds: typing.Optional[tuple[tuple[int, int], ...]],
		segment_mode: typing.Union[str, int],
		channel: int,
		note: int,
	) -> tuple[numpy.ndarray, subsample.analysis.LevelResult]:

		"""Select a segment from quantized audio, or return the full audio.

		When segment_mode is active and bounds are available, slices the audio
		to a single segment and recomputes the level.  Otherwise returns the
		original audio and level unchanged.
		"""

		if not segment_mode or segment_bounds is None or not segment_bounds:
			return audio, level

		if isinstance(segment_mode, int):
			idx = max(0, min(segment_mode - 1, len(segment_bounds) - 1))
		elif segment_mode == "round_robin":
			key = (channel, note)
			counter = self._segment_counters.get(key, 0)
			idx = counter % len(segment_bounds)
			self._segment_counters[key] = counter + 1
		elif segment_mode == "random":
			idx = random.randint(0, len(segment_bounds) - 1)
		else:
			return audio, level

		start, end = segment_bounds[idx]
		segment_audio = audio[start:end]
		mono = numpy.asarray(
			numpy.mean(segment_audio, axis=1, dtype=numpy.float32)
			if segment_audio.shape[1] > 1 else segment_audio[:, 0]
		)
		seg_level = subsample.analysis.compute_level(mono)

		return segment_audio, seg_level

	def _get_mix_matrix (
		self,
		in_channels: int,
		pan_weights: typing.Optional[numpy.ndarray],
		output_routing: typing.Optional[tuple[int, ...]] = None,
	) -> numpy.ndarray:

		"""Look up or build a mixing matrix for the given input channel count, pan weights, and output routing.

		Cached by (in_channels, pan_weights_tuple, output_routing) so repeated
		triggers with the same sample format and routing avoid rebuilding.
		"""

		key = (
			in_channels,
			tuple(pan_weights.tolist()) if pan_weights is not None else None,
			output_routing,
		)
		cached = self._mix_matrix_cache.get(key)

		if cached is not None:
			return cached

		if output_routing is not None:
			logical = subsample.channel.build_mix_matrix(in_channels, len(output_routing), pan_weights)
			mat = subsample.channel.route_to_device(logical, self._output_channels, output_routing)
		else:
			mat = subsample.channel.build_mix_matrix(in_channels, self._output_channels, pan_weights)

		self._mix_matrix_cache[key] = mat
		return mat

	def _render_float (
		self,
		audio: numpy.ndarray,
		level: subsample.analysis.LevelResult,
		velocity: int,
		mix_matrix: numpy.ndarray,
		gain_db: float = 0.0,
	) -> numpy.ndarray:

		"""Apply gain normalisation and channel mapping via mixing matrix.

		Maps input channels to output channels in a single matrix multiply,
		preserving the original spatial content (stereo image, surround
		positioning). ITU downmix or conservative upmix is baked into the
		matrix by channel.build_mix_matrix().

		Args:
			audio:      float32, shape (n_frames, in_channels).
			level:      LevelResult for this audio (peak + rms), used for gain calc.
			velocity:   MIDI velocity (0-127) from the triggering note_on message.
			mix_matrix: float32 array, shape (output_channels, in_channels).
			            Built by _get_mix_matrix() from channel.build_mix_matrix().
			gain_db:    Per-assignment level offset in dB (from Assignment.gain_db).

		Returns:
			float32 array, shape (n_frames, output_channels).
		"""

		# --- Gain calculation ---
		vel_scale = (velocity / 127.0) ** 2

		if level.rms > 0.0:
			norm_gain = self._target_rms / level.rms
		else:
			norm_gain = 1.0

		gain_linear = 10.0 ** (gain_db / 20.0) if gain_db != 0.0 else 1.0
		raw_gain = norm_gain * vel_scale * gain_linear

		# Anti-clip ceiling: account for the worst-case row sum of the mix
		# matrix (e.g. a 5.1→stereo downmix sums FL + 0.707*FC + 0.707*BL).
		max_row_sum = float(numpy.max(numpy.sum(numpy.abs(mix_matrix), axis=1)))

		if level.peak > 0.0 and max_row_sum > 0.0:
			final_gain = min(raw_gain, 1.0 / (level.peak * max_row_sum))
		else:
			final_gain = raw_gain

		_log.debug(
			"gain: norm=%.3f  vel_scale=%.3f  gain_db=%.1f  raw=%.3f  final=%.3f  (rms=%.4f peak=%.4f)",
			norm_gain, vel_scale, gain_db, raw_gain, final_gain,
			level.rms, level.peak,
		)

		gained = audio * final_gain

		# Channel mapping: (n_frames, in_ch) @ (in_ch, out_ch) = (n_frames, out_ch)
		result: numpy.ndarray = (gained @ mix_matrix.T).astype(numpy.float32)
		return result
