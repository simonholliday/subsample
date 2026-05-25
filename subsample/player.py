"""MIDI input listener and sample player for Subsample.

Handles MIDI input device selection (replicating the audio device selection
pattern from audio.py) and the MidiPlayer class which listens for MIDI events
and triggers polyphonic audio playback.

Threading model:
  - PortAudio output runs in its own callback thread (_audio_callback).
  - MIDI input runs in callback mode via mido.open_input(callback=...).
    rtmidi dispatches each incoming message to _safe_handle_message on its
    own dedicated thread — no polling loop, no fixed input-latency floor.
    Sub-millisecond MIDI-to-handler latency on a quiet Linux system.
  - The player's run() thread is purely a lifecycle coordinator: open
    ports, wait for shutdown, then close.

Concurrency control:
  - _voices_lock guards _voices (audio callback + handler).
  - _mix_matrix_lock guards _mix_matrix_cache (handler + reload).
  - _cc_debounce_lock guards _cc_debounce_timer (handler + cleanup).
  - _state_lock guards the small mutable dicts (_cc_state, _cc_omni,
    _cc_last_log, _segment_counters, _last_played) that are touched by
    both _handle_message (rtmidi thread) and update_assignments
    (watcher / CC debounce / on-complete threads).  Lock-ordering rule:
    _state_lock is outermost; never acquire it while holding any of the
    others.

Mixing architecture: a PyAudio callback stream requests N frames at regular
intervals. Each triggered note adds a _Voice (pre-rendered multichannel float32
audio + playback cursor) to a shared list. The callback sums all active
voices into one output buffer, clips, converts to PCM bytes at the output
bit depth, and returns them. The MIDI handler adds voices under a lock;
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
import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs
import yaml

import pymididefs.drums
import pymididefs.notes
import subsample.ambisonic
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

# Note map: (mido_channel, midi_note) → (Assignment, PickSpec).
# PickSpec is the per-note rank specification.  For single-note or explicit-pick
# assignments, it equals the SelectSpec's pick (scalar or range).  For multi-note
# assignments without explicit pick, each note gets a single-rank PickSpec(i, i)
# distributing across ranked matches.  The actual rank is drawn at note-on by
# PickSpec.resolve_index() — single ranks are deterministic, ranges re-roll.
NoteMap = dict[tuple[int, int], tuple[subsample.query.Assignment, subsample.query.PickSpec]]


@dataclasses.dataclass(frozen=True)
class MidiMapResult:

	"""Complete result of parsing a MIDI map YAML file.

	Fields:
		note_map:         Note routing table: (mido_channel, midi_note) → (Assignment, PickSpec).
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


def _ranks_for (pick_spec: subsample.query.PickSpec, ranked_len: int) -> range:

	"""Iterable of 1-indexed ranks this PickSpec might draw at trigger time.

	Used by update_assignments() to pre-compute variants for every reachable
	rank: scalar PickSpec(n, n) yields a single rank; range PickSpec(lo, hi)
	yields lo..hi inclusive, clamped to ranked_len so requests past the end
	collapse onto the last rank (mirrors resolve_index's clamping).
	"""

	hi = min(pick_spec.hi, ranked_len)
	lo = min(pick_spec.lo, hi)

	return range(lo, hi + 1)


def _quantize_params (
	process: subsample.query.ProcessSpec,
	step_name: str,
	config_bpm: float = 0.0,
) -> tuple[typing.Optional[float], int]:

	"""Extract BPM and grid from a stretch_quantize or pad_quantize step.

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


def _build_variant_lookup (
	process: subsample.query.ProcessSpec,
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	session_bpm: float,
) -> typing.Optional[typing.Callable[[int], typing.Optional[subsample.transform.TransformResult]]]:

	"""Shared core for the quantize-aware resolvers below.

	Inspects ``process`` for the active quantize step (``stretch_quantize`` /
	``pad_quantize``) and falls back to the session-level stretch_quantize
	when neither is configured but the session has a global ``target_bpm``.
	Returns a callable that, given a sample_id, returns the matching
	``TransformResult`` from the transform manager — or ``None`` when no
	variant exists yet (manager still processing) or when none of the above
	conditions yield a usable BPM.
	"""

	if transform_manager is None:
		return None

	step: typing.Union[subsample.transform.TimeStretch, subsample.transform.PadQuantize]

	if process.has_stretch_quantize():
		bpm, grid = _quantize_params(process, "stretch_quantize", session_bpm)
		if bpm is None or bpm <= 0:
			return None
		step = subsample.transform.TimeStretch(target_bpm=float(bpm), resolution=int(grid))
	elif process.has_pad_quantize():
		bpm, grid = _quantize_params(process, "pad_quantize", session_bpm)
		if bpm is None or bpm <= 0:
			return None
		step = subsample.transform.PadQuantize(target_bpm=float(bpm), resolution=int(grid))
	elif session_bpm > 0:
		# Fall back to session-level stretch_quantize.
		step = subsample.transform.TimeStretch(target_bpm=float(session_bpm), resolution=16)
	else:
		return None

	spec = subsample.transform.TransformSpec(steps=(step,))

	def _lookup (sample_id: int) -> typing.Optional[subsample.transform.TransformResult]:
		return transform_manager.get_variant(sample_id, spec)

	return _lookup


def _build_beats_resolver (
	process: subsample.query.ProcessSpec,
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	session_bpm: float,
) -> typing.Optional[typing.Callable[[int], typing.Optional[float]]]:

	"""Build a callable returning the quantized beat count for a sample.

	Used by the ``quantized_beats`` order scorer.  Beat count is derived
	from the variant's ``GridEnergyProfile`` length as
	``len(energy) * 4 / resolution``.

	Returns None (not a callable) when no valid quantize step is present,
	no transform manager is available, or the effective BPM is 0.
	"""

	lookup = _build_variant_lookup(process, transform_manager, session_bpm)
	if lookup is None:
		return None

	def _resolver (sample_id: int) -> typing.Optional[float]:
		result = lookup(sample_id)
		if result is None or result.energy_profile is None:
			return None
		profile = result.energy_profile
		return len(profile.energy) * 4.0 / profile.resolution

	return _resolver


def _build_energy_profile_resolver (
	process: subsample.query.ProcessSpec,
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	session_bpm: float,
) -> typing.Optional[typing.Callable[[int], typing.Optional[subsample.transform.GridEnergyProfile]]]:

	"""Build a callable returning the full GridEnergyProfile for a sample.

	Twin of ``_build_beats_resolver`` but returns the whole profile (not
	just the beat count).  Used by the ``beat_match`` order scorer, which
	needs per-slot energy data.

	Returns None (not a callable) when no valid quantize step is present,
	no transform manager is available, or the effective BPM is 0.
	"""

	lookup = _build_variant_lookup(process, transform_manager, session_bpm)
	if lookup is None:
		return None

	def _resolver (sample_id: int) -> typing.Optional[subsample.transform.GridEnergyProfile]:
		result = lookup(sample_id)
		if result is None:
			return None
		return result.energy_profile

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


def _parse_extract (raw: typing.Any, assignment_name: str) -> typing.Optional[subsample.query.ExtractSpec]:

	"""Parse the ``extract:`` field from a MIDI map assignment into an ExtractSpec.

	Accepted forms:
	  - One of the named kinds in ``subsample.query.EXTRACT_KINDS`` (omni,
	    side, depth, height, left, right, front, back). Case-insensitive.
	  - ``channel.<N>`` where N is a 1-indexed integer pointing at a literal
	    input channel.
	  - ``None`` (no extract — full multi-channel signal flows through pan/output).

	Validation here is syntactic only.  Semantic compatibility against the
	actual sample's ``channel_format`` is checked at map-load time by
	``_validate_assignment_extracts`` once samples are resolved.

	Args:
		raw:             Raw YAML value (string, or None for no extract).
		assignment_name: Name for error messages.

	Returns:
		ExtractSpec or None.

	Raises:
		ValueError: On unknown kind, malformed channel.N, or wrong type.
	"""

	if raw is None:
		return None

	if not isinstance(raw, str):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: extract must be a string, "
			f"got {type(raw).__name__}"
		)

	value = raw.strip().lower()

	if value in subsample.query.EXTRACT_KINDS:
		return subsample.query.ExtractSpec(kind=value)

	if "." in value:
		prefix, _, rest = value.partition(".")

		if prefix == "channel":
			try:
				n = int(rest)
			except ValueError:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: extract "
					f"'channel.<n>' requires an integer, got {rest!r}"
				)

			if n < 1:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: extract "
					f"'channel.{n}' must be 1-indexed (>= 1)"
				)

			return subsample.query.ExtractSpec(kind="channel", channel_index=n)

	valid = ", ".join(sorted(subsample.query.EXTRACT_KINDS))
	raise ValueError(
		f"MIDI map assignment {assignment_name!r}: unknown extract {raw!r} "
		f"(valid: {valid}, or channel.<n>)"
	)


# Note name conversion — delegated to pymididefs.
_midi_to_note_name = pymididefs.notes.note_to_name
_parse_note_name = pymididefs.notes.name_to_note


# Symbolic note-name namespaces.  A `notes:` value of "drum.kick_1" is looked
# up here: split on the first dot, the prefix selects a PyMidiDefs table, the
# symbol (case-insensitive) is the dict key.  The dict shape is the
# extension point — adding "program" → pymididefs.gm.GM_INSTRUMENT_MAP later
# requires only one entry, no parser changes.
_SYMBOL_NAMESPACES: typing.Final[dict[str, typing.Mapping[str, int]]] = {
	"drum": pymididefs.drums.GM_DRUM_MAP,
}


def _parse_note_spec (notes_raw: typing.Any, assignment_name: str) -> list[int]:

	"""Parse the 'notes' field from a MIDI map assignment into MIDI note numbers.

	Accepts:
	  Integer:          36                  → [36]
	  Note name:        "C3"                → [48]
	  GM drum symbol:   "drum.kick_1"       → [36]   (case-insensitive)
	  Numeric range:    "36..60"            → [36, 37, ..., 60]
	  Note name range:  "C2..C4"            → [36, 37, ..., 60]
	  List (mixed):     [36, "C3", "drum.snare_1"]  → [36, 48, 38]

	Range syntax is intentionally not supported for symbolic notes — drum
	names aren't sequential, use a list instead.

	Args:
		notes_raw:       Raw YAML value for the 'notes' field.
		assignment_name: Assignment name used in error messages.

	Returns:
		Non-empty list of MIDI note numbers.

	Raises:
		ValueError: If any note value is malformed or outside [0, 127].
	"""

	def _single (item: typing.Any) -> int:
		"""Resolve one note value: int, numeric string, note-name string, or
		symbolic form like "drum.kick_1"."""

		if isinstance(item, int):
			if not 0 <= item <= 127:
				raise ValueError(
					f"MIDI map assignment {assignment_name!r}: note {item} is outside [0, 127]"
				)
			return item

		if isinstance(item, str):

			# Symbolic form ("drum.kick_1") — single dot, no range separator.
			# Looked up case-insensitively in _SYMBOL_NAMESPACES.  Unknown
			# namespaces fall through to the int / note-name path so a typo
			# like "C.4" still gets the existing note-name error.
			if "." in item and ".." not in item:
				prefix, _, sym = item.partition(".")
				table = _SYMBOL_NAMESPACES.get(prefix.lower())

				if table is not None:
					try:
						return table[sym.lower()]
					except KeyError:
						valid = sorted(table)
						raise ValueError(
							f"MIDI map assignment {assignment_name!r}: "
							f"unknown {prefix!r} symbol {sym!r} "
							f"(valid: {', '.join(valid[:5])}…)"
						)

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

	# Range syntax: "C2..C4" or "36..60".  Reject symbolic ranges explicitly so
	# the user gets a targeted error pointing at list syntax, rather than the
	# generic "not a valid note name" from the inner _single() call.
	if isinstance(notes_raw, str) and ".." in notes_raw:

		if "." in notes_raw.split("..")[0]:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: "
				f"range syntax (a..b) is not supported for symbolic notes — "
				f"use a list instead, e.g. [drum.kick_1, drum.snare_1]"
			)

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

	spectral, rhythm, pitch, timbre, params, duration, level, band_energy, channel_format = result

	return subsample.library.SampleRecord(
		sample_id      = subsample.library.allocate_id(),
		name           = str(path.resolve()),
		spectral       = spectral,
		rhythm         = rhythm,
		pitch          = pitch,
		timbre         = timbre,
		level          = level,
		band_energy    = band_energy,
		params         = params,
		duration       = duration,
		audio          = None,
		filepath       = path if path.exists() else None,
		channel_format = channel_format,
	)


def _load_instrument_from_path (
	path: pathlib.Path,
	target_sample_rate: typing.Optional[int] = None,
	*,
	with_preview: bool,
) -> typing.Optional[subsample.library.SampleRecord]:

	"""Load an instrument sample (audio + sidecar) from a filesystem path.

	The sample's name is set to the filename stem (e.g. "2026-03-27_09-28-12")
	so that it can be matched by where: { name: ... } predicates.  Goes through
	``cache.ensure_sample_assets`` so the sidecar (and PNG, when previews are
	enabled) are regenerated on the fly if missing — matching the recursive
	library loader's self-healing behaviour for samples found this way.

	Args:
		path:               Absolute path to the audio file.
		target_sample_rate: When set, resample audio to this rate on load.
		with_preview:       Threaded from ``cfg.recorder.previews``; when True,
		                    a missing ``.preview.png`` is regenerated and the
		                    embedded preview block is kept current in the
		                    sidecar.

	Returns:
		SampleRecord with audio loaded, or None if loading fails.
	"""

	path = pathlib.Path(path).resolve()
	name = path.stem

	if not path.exists():
		_log.warning(
			"Instrument sample audio not found: %s — this sample will be skipped",
			path,
		)
		return None

	result = subsample.cache.ensure_sample_assets(path, with_preview=with_preview)
	if result is None:
		_log.warning(
			"Failed to load or analyze %s — this sample will be skipped",
			path,
		)
		return None

	spectral, rhythm, pitch, timbre, params, duration, level, band_energy, channel_format = result

	# Load the audio data, resampling to the output rate if needed.
	audio = subsample.library.load_wav_audio(path, target_sample_rate)
	if audio is None:
		_log.warning(
			"Failed to load audio from %s — this sample will be skipped",
			path,
		)
		return None

	return subsample.library.SampleRecord(
		sample_id      = subsample.library.allocate_id(),
		name           = name,
		spectral       = spectral,
		rhythm         = rhythm,
		pitch          = pitch,
		timbre         = timbre,
		level          = level,
		band_energy    = band_energy,
		params         = params,
		duration       = duration,
		audio          = audio,
		filepath       = path,
		channel_format = channel_format,
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
	*,
	with_preview: bool,
) -> None:

	"""Load path-based references, instruments, and directory samples from the MIDI map.

	Scans all assignments in the note map for:
	  - Path-based references → loaded and added to similarity matrices
	  - Path-based instruments → loaded and added to instrument library
	  - Directory predicates → all audio under the directory (recursively) is
	    loaded into the instrument library

	Directory predicates use the same audio-first, recursive walk as the main
	instrument-library loader so the two paths see the same set of files —
	a sample dropped into a subdirectory of a ``directory:`` predicate is
	picked up just like one in the root.

	Args:
		note_map:            Note routing table: (mido_channel, midi_note) → (Assignment, PickSpec).
		matrices:            List of SimilarityMatrix (one per bank) to add references to.
		instrument_lib:      InstrumentLibrary to add path-based instruments to.
		target_sample_rate:  When set, resample loaded audio to this rate.
		with_preview:        Threaded from ``cfg.recorder.previews``; controls
		                     whether missing PNG sidecars are regenerated for
		                     samples loaded through this path.
	"""

	# Collect unique paths for references, instruments, and directories
	ref_paths: set[str] = set()
	inst_paths: set[str] = set()
	dir_paths: set[str] = set()

	# Extract unique assignments from the note map
	seen_assignments: set[int] = set()

	for (assignment, _pick_spec) in note_map.values():
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
			audio_paths = sorted(
				p for p in directory.rglob("*")
				if p.is_file() and p.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS
			)
		except (PermissionError, OSError) as exc:
			_log.warning("Cannot read directory %s: %s — skipped", dir_path, exc)
			continue

		for audio_path in audio_paths:
			name = audio_path.stem

			# Skip if already in the library — typically because the main
			# instrument loader already picked this audio up (the predicate
			# may point at a subtree of cfg.instrument.directory).
			if instrument_lib.find_by_name(name) is not None:
				continue

			record = _load_instrument_from_path(
				audio_path, target_sample_rate, with_preview=with_preview,
			)

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

		record = _load_instrument_from_path(
			path, target_sample_rate, with_preview=with_preview,
		)
		if record is None:
			continue

		instrument_lib.add(record)

		_log.debug("Added path-based instrument from %s", path)


def _validate_assignment_extracts (
	note_map: NoteMap,
	instrument_lib: subsample.library.InstrumentLibrary,
) -> None:

	"""Reject the MIDI map if any assignment's ``extract`` is incompatible with a candidate sample.

	For every unique Assignment with ``extract != None``, evaluate the
	SelectSpec.where predicates against the populated instrument library
	(without a beats_resolver — samples with active quantized_beats
	predicates are skipped, which is acceptable conservative behaviour).
	For each unique ``(channel_format, in_channels)`` candidate, attempt to
	build the extract matrix.  If any candidate's format would raise, raise
	ValueError listing the assignment name, the extract kind, and the
	offending formats so the user can fix the map.

	When a directional extract (e.g. ``front`` on stereo) degenerates to
	the omni matrix for a given format, log a one-time warning per
	(assignment, format) pair — the assignment still loads.

	Args:
		note_map:       Note routing table after _resolve_path_references has
		                loaded path-based references, instruments, and
		                directory samples.
		instrument_lib: Populated instrument library.

	Raises:
		ValueError: If any extract is incompatible with any matching sample.
	"""

	seen_assignments: set[int] = set()

	for (assignment, _pick_spec) in note_map.values():

		if assignment.extract is None:
			continue

		aid = id(assignment)

		if aid in seen_assignments:
			continue

		seen_assignments.add(aid)

		# Gather the unique (channel_format, in_channels) combinations that
		# any select spec could pick.  We deliberately don't supply a
		# beats_resolver — quantized_beats predicates are skipped (matches()
		# returns False), so they appear as if no sample matched.  This is
		# conservative: validation may miss a few samples in unusual maps,
		# but won't reject correct ones.
		candidates: set[tuple[str, int]] = set()

		for select_spec in assignment.select:

			for record in instrument_lib.samples():

				if not select_spec.where.matches(record):
					continue

				if record.audio is None:
					continue

				ch_count = record.audio.shape[1]
				candidates.add((record.channel_format, ch_count))

		if not candidates:
			continue

		failures:           list[tuple[str, int, str]] = []
		equivalent_to_omni: list[tuple[str, int]]      = []

		for fmt, ch_count in sorted(candidates):

			try:
				ext_mat = subsample.channel.build_extract_matrix(
					assignment.extract, ch_count, fmt,
				)
			except ValueError as exc:
				failures.append((fmt, ch_count, str(exc)))
				continue

			# A directional pattern with no spatial information for this
			# format reduces to the omni matrix (e.g. `front` on stereo —
			# no F/B distinction exists).  Tell the user this is happening
			# but don't reject; the audio result is still useful.
			if assignment.extract.kind not in ("omni", "channel"):
				omni_spec = subsample.query.ExtractSpec(kind="omni")

				try:
					omni_mat = subsample.channel.build_extract_matrix(
						omni_spec, ch_count, fmt,
					)
				except ValueError:
					continue

				if numpy.allclose(ext_mat, omni_mat):
					equivalent_to_omni.append((fmt, ch_count))

		if failures:
			details = "\n".join(
				f"  - {fmt} {ch_count}ch: {msg}"
				for (fmt, ch_count, msg) in failures
			)
			raise ValueError(
				f"MIDI map assignment {assignment.name!r}: extract "
				f"{assignment.extract.kind!r} cannot be applied to "
				f"{len(failures)} matched sample format(s):\n{details}"
			)

		for fmt, ch_count in equivalent_to_omni:
			_log.warning(
				"MIDI map assignment %r: extract %r on %s %dch input is "
				"equivalent to 'omni' — this format carries no spatial "
				"information beyond mono.",
				assignment.name, assignment.extract.kind, fmt, ch_count,
			)


def load_midi_map (
	path: pathlib.Path,
	reference_names: list[str],
	strict: bool = True,
) -> MidiMapResult:

	"""Load a MIDI routing map from a YAML file.

	Parses the assignments list using the select/process pipeline format
	and returns a MidiMapResult containing the note map, any bank
	definitions, and the bank channel.

	Each assignment declares:
	  select:   Which sample to play — filter predicates, ordering, pick position.
	            Can be a single spec or a list (fallback chain, tried in order).
	  process:  How to present it — ordered list of processors (repitch, stretch_quantize, etc.).
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
		strict:           When True (default), unknown where-predicate keys and
		                  unknown processor names raise ValueError at parse time.
		                  When False, they are logged as warnings and ignored.

	Returns:
		MidiMapResult containing the NoteMap, bank definitions, and bank channel.

	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError:        If the YAML is malformed or a required field is missing.
	"""

	if not path.exists():
		raise FileNotFoundError(f"MIDI map not found: {path}")

	subsample.query.set_strict_mode(strict)
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

	for assignment_index, assignment_raw in enumerate(raw["assignments"], start=1):
		name = assignment_raw.get("name", "<unnamed>")

		# Channel: user-facing 1-16 → mido 0-indexed.
		channel_raw = assignment_raw.get("channel")

		if channel_raw is None:
			raise ValueError(f"MIDI map assignment {name!r}: missing 'channel'")

		try:
			mido_channel = int(channel_raw) - 1
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"MIDI map assignment {name!r} (#{assignment_index}): "
				f"invalid 'channel' value {channel_raw!r} — {exc}"
			) from exc

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

		try:
			gain_db = float(assignment_raw.get("gain", 0.0))
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"MIDI map assignment {name!r} (#{assignment_index}): "
				f"invalid 'gain' value {assignment_raw.get('gain')!r} — {exc}"
			) from exc

		pan_weights    = _parse_pan_weights(assignment_raw.get("pan"), name)
		output_routing = _parse_output_routing(assignment_raw.get("output"), name, pan_weights)
		extract        = _parse_extract(assignment_raw.get("extract"), name)

		# Extract segment playback mode from quantize step parameters.
		segment_mode: typing.Union[str, int] = ""

		for step in process.steps:
			if step.name in ("stretch_quantize", "pad_quantize"):
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
			extract=extract,
			segment_mode=segment_mode,
		)

		# Per-note pick distribution:
		# When process includes repitch, all notes share pick=1 (same sample,
		# pitched per note).  Otherwise, each note gets the next pick position
		# so multi-note assignments distribute across ranked matches.
		# An explicit pick in the SelectSpec (scalar or range) overrides this
		# default — range forms re-roll at trigger time rather than being
		# distributed across notes at load time.
		if isinstance(select_raw, dict):
			explicit_pick = "pick" in select_raw
		elif isinstance(select_raw, list) and select_raw:
			explicit_pick = any("pick" in s for s in select_raw if isinstance(s, dict))
		else:
			explicit_pick = False

		for note_idx, note in enumerate(notes):

			if explicit_pick or process.has_repitch() or len(notes) == 1:
				pick_spec = select_specs[0].pick
			else:
				rank = note_idx + 1
				pick_spec = subsample.query.PickSpec(rank, rank)

			note_map[(mido_channel, int(note))] = (assignment, pick_spec)

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
	(mido_channel, midi_note) and stores (Assignment, PickSpec) per note.
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
		ambisonic_config: typing.Optional[subsample.config.AmbisonicConfig] = None,
		buffer_frames: typing.Optional[int] = None,
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

		# Optional PortAudio output buffer size in frames.  When None, the
		# device default is used.  Validated as a power of two in [32, 4096]
		# at config load.  See PlayerAudioConfig.buffer_frames docstring.
		self._buffer_frames: typing.Optional[int] = buffer_frames

		# Note routing map: (mido_channel, midi_note) → (Assignment, PickSpec).
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

		# Project-wide ambisonic decoder/rotation settings.  When None, the
		# player treats any ambisonic-tagged sample the same as raw multichannel
		# PCM (the decoder matrix is not built and the sample plays through the
		# default mix matrix).
		self._ambisonic_config = ambisonic_config

		# Mix matrix cache: (in_channels, pan_weights_tuple, output_routing,
		# channel_format) → matrix.  channel_format is included so ambisonic
		# decode matrices do not collide with raw-PCM routing for the same
		# 4-channel input shape.  Lazily populated by _get_mix_matrix();
		# cleared on MIDI map reload.
		#
		# _mix_matrix_lock serialises the dict across threads:
		#   - the MIDI dispatch thread reads and writes entries in
		#     _get_mix_matrix() during note_on handling;
		#   - the midi-map-watcher thread calls reload_midi_map() which
		#     calls .clear() here.
		# Without the lock a get → build → set sequence on the MIDI thread
		# could interleave with a clear() on the watcher thread.  The
		# failure mode is benign (the cache key is independent of the note
		# map so a surviving entry is still correct for its inputs) but
		# the explicit lock removes the ambiguity and matches the locking
		# discipline used for _voices_lock and _cc_debounce_lock.
		_MixCacheKey = tuple[
			int,
			typing.Optional[tuple[float, ...]],
			typing.Optional[tuple[int, ...]],
			str,
			typing.Optional[tuple[str, typing.Optional[int]]],
		]
		self._mix_matrix_cache: dict[_MixCacheKey, numpy.ndarray] = {}
		self._mix_matrix_lock:  threading.Lock                    = threading.Lock()

		# Per-note segment counter for round-robin segment playback.
		# Cleared on MIDI map reload and bank switch.
		self._segment_counters: dict[tuple[int, int], int] = {}

		# Single lock for the small mutable dicts touched by both
		# _handle_message (rtmidi callback thread) and the threads that run
		# update_assignments (watcher, CC debounce timer, on-complete).
		# Protects: _cc_state, _cc_omni, _cc_last_log, _segment_counters,
		# _last_played.  All critical sections are sub-microsecond — one
		# lock is simpler than per-dict locking and there is no deadlock
		# topology because _state_lock is never acquired alongside any
		# other player lock.
		#
		# Lock ordering rule (enforced by code review, not the runtime):
		# _state_lock is outermost.  Never acquire it while holding
		# _voices_lock, _mix_matrix_lock, or _cc_debounce_lock.
		self._state_lock: threading.Lock = threading.Lock()

		# Group consecutive notes that share the same Assignment into ranges
		# so that a 128-note pitched assignment becomes a single log line.
		# Each group tracks the set of PickSpecs across its notes — a single
		# shared PickSpec means an explicit pick (or single-note assignment);
		# multiple distinct PickSpecs mean auto-distribution kicked in.
		groups: list[tuple[int, int, int, subsample.query.Assignment, set[subsample.query.PickSpec]]] = []

		for (ch, note), (asgn, pick_spec) in sorted(self._note_map.items()):
			if (
				groups
				and groups[-1][0] == ch
				and groups[-1][2] == note - 1
				and groups[-1][3] is asgn
			):
				groups[-1][4].add(pick_spec)
				groups[-1] = (ch, groups[-1][1], note, asgn, groups[-1][4])
			else:
				groups.append((ch, note, note, asgn, {pick_spec}))

		lines: list[str] = []

		for ch, lo, hi, asgn, pick_specs in groups:
			count = hi - lo + 1

			if count == 1:
				note_str = f"note {_midi_to_note_name(lo)}"
			else:
				note_str = f"notes {_midi_to_note_name(lo)}..{_midi_to_note_name(hi)} ({count})"

			line = f"ch{ch+1} {note_str} → {asgn.name}"

			if len(pick_specs) == 1:
				ps = next(iter(pick_specs))
				if ps.lo == ps.hi:
					if ps.lo != 1:
						line += f" pick {ps.lo}"
				else:
					line += f" pick {ps.lo}-{ps.hi}"
			else:
				line += " pick distributed"

			if asgn.process.has_repitch():
				line += " pitched"

			if asgn.process.has_stretch_quantize():
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

		"""Open MIDI input and a callback output stream, then wait for shutdown.

		MIDI input runs in callback mode — ``mido.open_input(callback=…)``
		registers ``_safe_handle_message`` so rtmidi dispatches every
		message on its own dedicated thread the moment it arrives.  There
		is no polling loop; this thread just blocks on the shutdown event
		until it fires, then closes the port and audio stream cleanly.

		Both the MIDI port and the PyAudio stream are closed in the finally
		block — ``port.close()`` internally waits for any in-flight callback
		to return, so no message is silently lost during teardown.

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
		routing_fixes: dict[tuple[int, int], tuple[subsample.query.Assignment, subsample.query.PickSpec]] = {}

		for (ch, note), (assignment, _pick_spec) in self._note_map.items():
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
							dataclasses.replace(assignment, output_routing=None),
							_pick_spec,
						)
						break

		self._note_map.update(routing_fixes)

		# Callback mode: PortAudio pulls audio from _audio_callback on its own
		# high-priority thread. The MIDI thread runs independently and adds voices.
		# When player.audio.buffer_frames is set, pass it as frames_per_buffer
		# to tighten output-side latency.  If the device cannot honour the
		# requested size at open time, fall back to the device default with a
		# clear ERROR log so the user knows their tuning value was ignored.
		open_kwargs: dict[str, typing.Any] = {
			"format":              subsample.audio.get_pyaudio_format(self._output_bit_depth),
			"channels":            self._output_channels,
			"rate":                self._output_sample_rate,
			"output":              True,
			"output_device_index": output_device_index,
			"stream_callback":     self._audio_callback,
		}

		if self._buffer_frames is not None:
			open_kwargs["frames_per_buffer"] = self._buffer_frames
			_log.info("PortAudio output buffer: %d frames", self._buffer_frames)

		try:
			stream = pa.open(**open_kwargs)
		except OSError as exc:
			if "frames_per_buffer" in open_kwargs:
				_log.error(
					"PortAudio rejected buffer_frames=%d (%s) — falling back "
					"to the device default.  Lower or omit player.audio."
					"buffer_frames in config.yaml.",
					self._buffer_frames, exc,
				)
				open_kwargs.pop("frames_per_buffer")
				stream = pa.open(**open_kwargs)
			else:
				raise

		# Open the MIDI input port in callback mode.  rtmidi delivers each
		# message to ``_safe_handle_message`` on its own dedicated thread the
		# moment it arrives — no polling loop, no 10 ms jitter floor.  The
		# kwarg form installs the callback as the last step of port creation,
		# closing the open-then-assign gap that property-style assignment
		# would expose.  Virtual ports are server destinations external apps
		# connect to; they require ``virtual=True`` and do not appear in
		# ``get_input_names()``.
		if self._virtual_midi_port is not None:
			port_label = self._virtual_midi_port
			port = mido.open_input(
				self._virtual_midi_port,
				virtual=True,
				callback=self._safe_handle_message,
			)
			_log.info("MIDI player opened virtual port: %s", port_label)
		else:
			port_label = self._device_name
			port = mido.open_input(
				self._device_name,
				callback=self._safe_handle_message,
			)
			_log.info("MIDI player opened hardware port: %s", port_label)

		try:
			# Block until shutdown is signalled.  All MIDI work happens on
			# rtmidi's callback thread; this thread is now purely a lifecycle
			# coordinator.
			self._shutdown_event.wait()

		finally:
			with self._cc_debounce_lock:
				if self._cc_debounce_timer is not None:
					self._cc_debounce_timer.cancel()

			# port.close() internally clears the callback under rtmidi's lock
			# (waits for any in-flight callback to return), then closes the
			# port.  No separate teardown step needed.
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

	def _snapshot_cc_state (
		self,
	) -> tuple[dict[tuple[int, int], int], dict[int, int]]:

		"""Return shallow copies of the live CC state dicts taken under
		``_state_lock``.

		Pass these to ``spec_from_process`` instead of the live attributes
		so the spec sees a consistent snapshot even if a CC arrives mid-
		call from another thread.  Snapshots are 1-128 entries — copy cost
		is microseconds.

		Required at call sites that run on non-MIDI threads (the CC debounce
		timer, the watcher's ``update_assignments`` path, the on-complete
		integration callback).  At call sites on the rtmidi callback thread
		itself the snapshot is purely defensive symmetry — that thread is
		the single writer for the CC dicts.
		"""

		with self._state_lock:
			return dict(self._cc_state), dict(self._cc_omni)

	def _safe_handle_message (self, msg: mido.Message) -> None:

		"""Defensive wrapper around ``_handle_message`` for rtmidi callback dispatch.

		mido's rtmidi backend invokes user callbacks from a C++ thread.  If a
		Python exception escapes the callback, the binding clears ``PyErr`` and
		the rtmidi thread keeps running — but the message is silently lost and
		the bug never surfaces.  Catch every exception here and log at ERROR
		so handler bugs are visible in the log instead of presenting as
		mysteriously-dropped notes.
		"""

		try:
			self._handle_message(msg)
		except Exception as exc:
			_log.error(
				"MIDI handler failed for %r: %s", msg, exc, exc_info=True,
			)

	def _handle_message (self, msg: mido.Message) -> None:

		"""Dispatch a single MIDI message via the select/process pipeline.

		Runs on rtmidi's dedicated callback thread (see ``run()``).  Must stay
		fast — any slow operation here stalls MIDI dispatch for the lifetime
		of the port.

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
					# Both dicts are guarded by _state_lock — clear them
					# together so any concurrent reader (e.g. _select_segment
					# RMW) sees a consistent post-switch state.
					with self._state_lock:
						self._last_played.clear()
						self._segment_counters.clear()
					# Defensive: a malformed bank map raised at query time
					# (e.g. similarity order without where.reference) must
					# not kill the player thread mid-set.
					self._try_update_assignments(f"bank switch to program {msg.program}")
			return

		# Control Change: update CC state and debounce re-evaluation for
		# mapped parameters (CcBinding in the process pipeline).
		if msg.type == "control_change":
			# Hold _state_lock for the CC-state writes so other threads
			# reading these dicts via _snapshot_cc_state() see a consistent
			# pair (channel-state + omni-state updated together).
			with self._state_lock:
				self._cc_state[(msg.channel, msg.control)] = msg.value
				self._cc_omni[msg.control] = msg.value
			self.events.emit("cc", channel=msg.channel, cc_number=msg.control, value=msg.value)

			if msg.control in self._mapped_ccs:
				_log.debug(
					"CC ch%d #%d = %d",
					msg.channel + 1, msg.control, msg.value,
				)

				now = time.monotonic()
				# Compute "should we log this CC?" under the lock, then run
				# the actual log call outside it — _log.info can be slow
				# (handlers, formatting) and we want to release the lock
				# before any I/O.
				with self._state_lock:
					last = self._cc_last_log.get(msg.control, 0.0)
					if now - last >= 1.0:
						self._cc_last_log[msg.control] = now
						should_log = True
					else:
						should_log = False

				if should_log:
					_log.info(
						"CC ch%d #%d = %d (mapped)",
						msg.channel + 1, msg.control, msg.value,
					)

				with self._cc_debounce_lock:
					if self._cc_debounce_timer is not None:
						self._cc_debounce_timer.cancel()

					# Defensive timer target: a transient query failure here
					# must not silently kill the timer thread and stall every
					# subsequent CC-driven re-evaluation for the session.
					self._cc_debounce_timer = threading.Timer(
						_CC_DEBOUNCE_SECONDS,
						self._try_update_assignments,
						args=(f"CC #{msg.control} debounce",),
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

		assignment, pick_spec = entry
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
		energy_profile_resolver = _build_energy_profile_resolver(
			assignment.process, eff_transform, self._target_bpm,
		)

		for select_spec in assignment.select:

			ranked = subsample.query.query(
				select_spec, all_samples, eff_similarity, beats_resolver,
				energy_profile_resolver=energy_profile_resolver,
			)

			if ranked:
				# PickSpec.resolve_index draws a 0-indexed rank from [lo, hi],
				# clamping hi to len(ranked).  Single-rank PickSpecs are
				# deterministic; range PickSpecs re-roll on every note-on.
				idx = pick_spec.resolve_index(len(ranked))
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

				# Validation: skip stretch_quantize for samples with no tempo.
				# pad_quantize does NOT need source tempo — only target BPM.
				bpm_for_spec: typing.Optional[float] = None
				grid_for_spec = 16

				if assignment.process.has_stretch_quantize():
					if record.rhythm.tempo_bpm > 0.0:
						bpm_for_spec, grid_for_spec = _quantize_params(assignment.process, "stretch_quantize", self._target_bpm)
					else:
						_log.warning(
							"stretch_quantize %s: sample %r has no detected tempo — "
							"playing without beat-quantizing",
							assignment.name, record.name,
						)

				if assignment.process.has_pad_quantize():
					bpm_for_spec, grid_for_spec = _quantize_params(assignment.process, "pad_quantize", self._target_bpm)

				cc_state_snapshot, cc_omni_snapshot = self._snapshot_cc_state()

				spec = subsample.transform.spec_from_process(
					assignment.process,
					midi_note=midi_note_for_spec,
					target_bpm=bpm_for_spec,
					resolution=grid_for_spec,
					reference_path=_reference_wav_path(assignment),
					cc_state=cc_state_snapshot,
					cc_omni=cc_omni_snapshot,
				)

				if spec.steps:
					variant = eff_transform.get_variant(sample_id, spec)

					if variant is not None:
						seg_audio, seg_level = self._select_segment(
							variant.audio, variant.level, variant.segment_bounds,
							assignment.segment_mode, msg.channel, msg.note,
						)
						mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing, record.channel_format, assignment.extract)
						rendered = self._render_float(seg_audio, seg_level, msg.velocity, mix_mat, assignment.gain_db)
						with self._voices_lock:
							self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
						with self._state_lock:
							self._last_played[(msg.channel, msg.note)] = variant
						_log.debug(
							"note %d (vel %d) → %r → %r (variant, %d step(s))  (%.2fs)",
							msg.note, msg.velocity, assignment.name, record.name,
							len(spec.steps), variant.duration,
						)
						return

					# New variant not ready — try the previously played variant
					# for this note (smooth transition during gradual param changes).
					# _state_lock guards against the bank-switch clear.
					with self._state_lock:
						prev = self._last_played.get((msg.channel, msg.note))

					if prev is not None and prev.key.sample_id == sample_id:
						seg_audio, seg_level = self._select_segment(
							prev.audio, prev.level, prev.segment_bounds,
							assignment.segment_mode, msg.channel, msg.note,
						)
						mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing, record.channel_format, assignment.extract)
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
				mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing, record.channel_format, assignment.extract)
				rendered = self._render_float(seg_audio, seg_level, msg.velocity, mix_mat, assignment.gain_db)
				with self._voices_lock:
					self._voices.append(_Voice(audio=rendered, note=msg.note, channel=msg.channel, one_shot=one_shot))
				_log.debug(
					"note %d (vel %d) → %r → %r (base variant)  (%.2fs)",
					msg.note, msg.velocity, assignment.name, record.name, base.duration,
				)
				return

		# 4. Last resort: convert from int PCM on this trigger.
		mix_mat = self._get_mix_matrix(record.audio.shape[1] if record.audio is not None else 1, pan_weights, output_routing, record.channel_format, assignment.extract)
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

	def _enqueue_quantize_variants (
		self,
		asgn: subsample.query.Assignment,
		step_name: str,
		note_picks: list[tuple[int, subsample.query.PickSpec]],
		ranked: list[subsample.library.SampleRecord],
		eff_library: subsample.library.InstrumentLibrary,
		eff_transform: subsample.transform.TransformManager,
		require_tempo: bool,
	) -> int:

		"""Pre-compute one quantize step's variants for every reachable rank.

		Used by ``update_assignments`` for both ``stretch_quantize`` (the
		full beat-quantized variant; ``require_tempo=True`` since the
		algorithm needs a detected source tempo) and ``pad_quantize`` (no
		tempo requirement — only onset positions matter).  Returns the
		number of variants actually enqueued (after dedup + filtering).

		Variants are deduplicated by ``sample_id`` because multiple ranks
		can resolve to the same sample and only one variant is needed per
		(sample, spec) key — the transform processor itself dedups across
		calls, but doing it here avoids per-rank ``spec_from_process``
		work.
		"""

		bpm, grid = _quantize_params(asgn.process, step_name, self._target_bpm)
		enqueued = 0
		seen_ids: set[int] = set()

		# Snapshot CC state once for the whole batch — all variants in this
		# enqueue share the same parameter values.  Required here because
		# this runs on whichever thread invoked update_assignments (watcher,
		# CC debounce, on-complete), not on the rtmidi callback thread.
		cc_state_snapshot, cc_omni_snapshot = self._snapshot_cc_state()

		for _note, pick_spec in note_picks:
			for rank in _ranks_for(pick_spec, len(ranked)):
				sid = ranked[rank - 1].sample_id

				if sid in seen_ids:
					continue

				seen_ids.add(sid)
				record = eff_library.get(sid)

				if record is None:
					continue

				if require_tempo and record.rhythm.tempo_bpm <= 0.0:
					_log.warning(
						"%s %s: sample %r (pick %d) has no detected tempo — "
						"will not be beat-quantized",
						step_name, asgn.name, record.name, rank,
					)
					continue

				spec = subsample.transform.spec_from_process(
					asgn.process,
					target_bpm=bpm,
					resolution=grid,
					reference_path=_reference_wav_path(asgn),
					cc_state=cc_state_snapshot,
					cc_omni=cc_omni_snapshot,
				)
				eff_transform.get_variant(sid, spec)
				enqueued += 1

		return enqueued

	def update_assignments (self) -> None:

		"""Pre-compute transform variants for all assignments that declare processors.

		Groups notes by Assignment, resolves each to its current sample via the
		query engine, and enqueues the appropriate variants.  The full ordered
		process chain (filters, saturate, reverse, etc.) is included alongside
		repitch / stretch_quantize via spec_from_process().

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
		# assignment share the same select/process spec.  Collect (note, PickSpec)
		# pairs so stretch_quantize can pre-compute a variant for every reachable
		# rank (each note may resolve to a different sample, and a range PickSpec
		# may resolve to any rank in [lo, hi] at trigger time).
		groups: dict[int, tuple[subsample.query.Assignment, list[tuple[int, subsample.query.PickSpec]]]] = {}

		for (_ch, note), (asgn, pick_spec) in self._note_map.items():

			if asgn.process.steps:
				group_key = id(asgn)

				if group_key not in groups:
					groups[group_key] = (asgn, [])

				groups[group_key][1].append((note, pick_spec))

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
			energy_profile_resolver = _build_energy_profile_resolver(
				asgn.process, eff_transform, self._target_bpm,
			)

			for select_spec in asgn.select:
				ranked = subsample.query.query(
					select_spec, all_samples, eff_similarity, beats_resolver,
					energy_profile_resolver=energy_profile_resolver,
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

			# Beat-quantize: each note may pick a different sample, and a range
			# PickSpec may resolve to any rank in [lo, hi] at trigger time —
			# enqueue a variant for every reachable rank.  The full process
			# chain is included via spec_from_process().  stretch_quantize
			# additionally requires the source to have a detected tempo;
			# pad_quantize does not (it only needs onsets).
			elif asgn.process.has_stretch_quantize():
				enqueued = self._enqueue_quantize_variants(
					asgn, "stretch_quantize", note_picks, ranked,
					eff_library, eff_transform, require_tempo=True,
				)

				if enqueued > 0:
					_total_assignments += 1
					_total_variants += enqueued

					_log.debug(
						"stretch_quantize %s: queued %d variant(s)",
						asgn.name, enqueued,
					)

			elif asgn.process.has_pad_quantize():
				enqueued = self._enqueue_quantize_variants(
					asgn, "pad_quantize", note_picks, ranked,
					eff_library, eff_transform, require_tempo=False,
				)

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
				cc_state_snapshot, cc_omni_snapshot = self._snapshot_cc_state()
				spec = subsample.transform.spec_from_process(
					asgn.process,
					reference_path=_reference_wav_path(asgn),
					cc_state=cc_state_snapshot,
					cc_omni=cc_omni_snapshot,
				)

				if spec.steps:
					seen_ids_static: set[int] = set()

					for _note, pick_spec in note_picks:
						for rank in _ranks_for(pick_spec, len(ranked)):
							sid = ranked[rank - 1].sample_id

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

	def _try_update_assignments (self, context: str) -> None:

		"""Run ``update_assignments`` defensively, swallowing any exception.

		Used by the live-state paths that should never kill the player on
		a transient failure: bank switches (program_change), CC re-eval
		debounce, and the post-sample integration call in ``cli.py``.  The
		reload path itself does NOT use this — it relies on the raised
		exception to know whether the new map validated.

		``context`` is included in the ERROR log so a stuck behaviour can
		be traced to the trigger.
		"""

		try:
			self.update_assignments()
		except Exception as exc:
			_log.error(
				"update_assignments failed during %s — playback continues with "
				"the previous variant set: %s",
				context, exc,
			)

	def reload_midi_map (self, new_map: NoteMap) -> None:

		"""Replace the active note map and re-compute transform variants.

		Atomic in the sense that matters for live performance: if the new
		map is structurally valid but semantically broken (e.g. an order
		clause like ``similarity`` whose required ``where.reference:`` was
		accidentally commented out), ``update_assignments()`` raises on the
		first offending assignment.  This method catches that, restores the
		previous map, and re-raises so the watcher caller can log and stay
		live under the old map — playback never stops mid-performance for
		a YAML typo.

		Thread-safety: dict rebinds are atomic under the GIL, so any in-
		flight ``_handle_message()`` call sees either the old map or the
		new map, never a half-applied state.

		Args:
			new_map: Parsed NoteMap from load_midi_map().note_map.  Must be a
			         complete replacement (not a diff).

		Raises:
			Exception: whatever update_assignments() surfaces while
			validating the new map.  The active map is restored before
			propagating, so the player remains usable.
		"""

		old_note_map    = self._note_map
		old_mapped_ccs  = self._mapped_ccs
		old_count       = len(self._note_map)

		# Apply the new map first so update_assignments() validates against
		# the configuration the player would actually run with.  This is the
		# canonical validation path — we don't duplicate query logic for a
		# separate dry-run.
		self._note_map   = new_map
		self._mapped_ccs = _collect_mapped_ccs(new_map)

		try:
			self.update_assignments()
		except Exception:
			# Roll back so the caller's catch leaves the player on the
			# previously-good map.  Any variants update_assignments() may
			# have enqueued before raising are harmless — they sit in the
			# transform manager's cache for future calls.
			self._note_map   = old_note_map
			self._mapped_ccs = old_mapped_ccs
			raise

		# Validation succeeded — clear caches whose entries reference the
		# old assignments by identity so the next note_on rebuilds them.
		# _segment_counters clear is under _state_lock so it serialises
		# against the round_robin RMW in _select_segment.
		with self._mix_matrix_lock:
			self._mix_matrix_cache.clear()
		with self._state_lock:
			self._segment_counters.clear()

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
			# RMW under _state_lock so a concurrent clear (from
			# reload_midi_map or bank switch) doesn't drop an increment
			# and two parallel triggers can't lose a step.
			with self._state_lock:
				counter = self._segment_counters.get(key, 0)
				self._segment_counters[key] = counter + 1
			idx = counter % len(segment_bounds)
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
		channel_format: str = "pcm",
		extract: typing.Optional[subsample.query.ExtractSpec] = None,
	) -> numpy.ndarray:

		"""Look up or build a mixing matrix for the given input channel count, pan weights, output routing, and optional extract.

		For raw PCM samples (channel_format="pcm"), the matrix is built by
		channel.build_mix_matrix() as usual.  For ambisonic B-format samples
		(channel_format="b_format_ambix"), the matrix is instead the product
		of the project-wide rotation and decoder — so the same matmul in the
		render path decodes 4-channel B-format to the output layout.

		When ``extract`` is set, the input is first collapsed to 1 channel
		via channel.build_extract_matrix() emulating a named microphone
		pickup pattern (omni, side, etc.), then the existing pan/routing
		logic distributes that mono signal across the output channels.
		Composition: ``final = pan_matrix @ extract_matrix`` — one matmul in
		the render path, just like the non-extract case.

		Cached by (in_channels, pan_weights_tuple, output_routing,
		channel_format, extract).
		"""

		key = (
			in_channels,
			tuple(pan_weights.tolist()) if pan_weights is not None else None,
			output_routing,
			channel_format,
			(extract.kind, extract.channel_index) if extract is not None else None,
		)

		with self._mix_matrix_lock:
			cached = self._mix_matrix_cache.get(key)

			if cached is not None:
				return cached

			if extract is not None:
				# Extract collapses the input to 1 channel emulating a named
				# microphone pickup pattern.  Skip the Ambisonic decoder path
				# entirely — the extract has already produced the desired
				# sub-signal (e.g. W only for `omni` on B-format).
				extract_matrix = subsample.channel.build_extract_matrix(
					extract, in_channels, channel_format,
				)

				# Default pan for the extracted mono signal: equal distribution
				# across all logical outputs (constant-power).  Without this,
				# build_mix_matrix(1, N, None) would apply the conservative
				# upmix rule (mono → front position only), so the user's
				# centred-mono kick would land in the left speaker only.
				# Explicit pan_weights override this default.
				logical_out = len(output_routing) if output_routing is not None else self._output_channels
				effective_pan = pan_weights

				if effective_pan is None:
					effective_pan = numpy.ones(logical_out, dtype=numpy.float32)

				if output_routing is not None:
					pan_matrix = subsample.channel.build_mix_matrix(
						1, len(output_routing), effective_pan,
					)
					logical = pan_matrix @ extract_matrix
					mat = subsample.channel.route_to_device(
						logical, self._output_channels, output_routing,
					)
				else:
					pan_matrix = subsample.channel.build_mix_matrix(
						1, self._output_channels, effective_pan,
					)
					mat = pan_matrix @ extract_matrix

				mat = numpy.asarray(mat, dtype=numpy.float32)
				self._mix_matrix_cache[key] = mat
				return mat

			if channel_format == "b_format_ambix" and self._ambisonic_config is not None:
				logical_out_ch = len(output_routing) if output_routing is not None else self._output_channels

				if logical_out_ch in subsample.ambisonic.SUPPORTED_DECODER_OUT_CHANNELS and in_channels == 4:
					decoder = subsample.ambisonic.combined_decode_matrix(
						order        = 1,
						out_channels = logical_out_ch,
						decoder_type = self._ambisonic_config.decoder,
						yaw_deg      = self._ambisonic_config.yaw_degrees,
						pitch_deg    = self._ambisonic_config.pitch_degrees,
						roll_deg     = self._ambisonic_config.roll_degrees,
					)
					if output_routing is not None:
						mat = subsample.channel.route_to_device(decoder, self._output_channels, output_routing)
					elif logical_out_ch == self._output_channels:
						mat = decoder
					else:
						mat = subsample.channel.route_to_device(decoder, self._output_channels, None)
				else:
					# Unsupported combination (non-standard output channel count
					# or wrong input channel count): fall back to plain routing.
					if output_routing is not None:
						logical = subsample.channel.build_mix_matrix(in_channels, len(output_routing), pan_weights)
						mat = subsample.channel.route_to_device(logical, self._output_channels, output_routing)
					else:
						mat = subsample.channel.build_mix_matrix(in_channels, self._output_channels, pan_weights)

			elif output_routing is not None:
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
