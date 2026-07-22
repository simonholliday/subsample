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
    (watcher / CC debounce / on-complete threads).
  - _rules_lock (reentrant) serialises rule-set re-evaluation:
    update_assignments and _apply_rule_set (hot-reload / program-change
    swap) never interleave, so the swap's install→validate→rollback
    window cannot be observed by a concurrent re-evaluation.
    Lock-ordering rule: _rules_lock is outermost, then _state_lock;
    never acquire either while holding any of the others.

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
See subsample/data/midi-map.yaml.default (scaffolded into projects as
midi-map.yaml by --init) for the format specification.
"""

import collections
import dataclasses
import logging
import math
import pathlib
import random
import threading
import time
import typing

import mido
import numpy
import pyaudio
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
import subsample.definitions
import subsample.events
import subsample.library
import subsample.query
import subsample.similarity
import subsample.transform


_log = logging.getLogger(__name__)

# Cosine fade-out duration applied when a note_off is received and no explicit
# release: is configured.  Long enough to prevent a click on hard cutoff; short
# enough to be imperceptible.  Stored as seconds; converted to frames in
# MidiPlayer.__init__() using the actual output sample rate so the duration is
# correct regardless of device.
_RELEASE_FADE_SECONDS: float = 0.01  # 10 ms

# Decay steepness for the "exponential" release curve.  The ramp is
# (exp(-k*x) - exp(-k)) / (1 - exp(-k)) for x in [0, 1], which starts at 1.0 and
# reaches EXACTLY 0 at x=1 (so the fade is click-free, unlike a raw exp(-k*x)
# that leaves an audible residual).  k=9 gives a musically natural fast-then-slow
# tail; exp(-9) ≈ 1.2e-4.
_RELEASE_EXP_K: float = 9.0

# Note map: (mido_channel, midi_note) → list of (Assignment, PickSpec) layers.
#
# Each list entry is one velocity layer for that note.  The default
# (no ``velocity:`` field in YAML) is a single-entry list with the
# Assignment's ``velocity_trigger`` set to (0, 127) — identical to the
# pre-velocity-layering behaviour.  Multiple entries with non-overlapping
# ``velocity_trigger`` ranges form a velocity-layered note; the player
# scans the list at note-on to find the layer covering ``msg.velocity``.
#
# PickSpec is the per-note rank specification.  For single-note or
# explicit-pick assignments, it equals the SelectSpec's pick (scalar or
# range).  For multi-note assignments without explicit pick, each note
# gets a single-rank PickSpec(i, i) distributing across ranked matches.
# The actual rank is drawn at note-on by PickSpec.resolve_index() — single
# ranks are deterministic, ranges re-roll.
NoteMap = dict[
	tuple[int, int],
	list[tuple[subsample.query.Assignment, subsample.query.PickSpec]],
]


@dataclasses.dataclass(frozen=True)
class _Candidates:

	"""One assignment's pre-resolved trigger pool.

	``ids`` is the ranked list of sample ids (the pick draws an index into it).
	``loudness`` holds the pool's per-sample levels normalised to [0, 1] (loudest
	= 1.0, same order as ``ids``) for a ``spacing: loudness`` velocity pick, or
	None when the pool cannot be loudness-spaced (fewer than two samples or a
	single distinct level).  The two are bundled so the candidate-cache rebind
	stays atomic — an in-flight trigger always sees a matched (ids, loudness)
	pair, never new ids beside stale levels.
	"""

	ids:      list[int]
	loudness: typing.Optional[list[float]]


@dataclasses.dataclass(frozen=True)
class ZoneTemplate:

	"""A declared zone-tuned assignment.

	Held as a template (rather than baked into the NoteMap at load time)
	because the concrete (channel, note) → (Assignment, PickSpec) entries
	must be re-derived whenever the active library changes — new sample
	imports, watcher pickups, library evictions, bank switches, MIDI map
	reloads.  The materialisation step runs the template's ``select`` over
	the active library, filters to pitched samples via
	``has_stable_pitch``, sorts by detected pitch, and lays each sample
	across a contiguous slice of the keyboard centred on its pitch with
	zones meeting at midpoints.

	Every Assignment-style configuration (process, gain, pan, output
	routing, extract, segment_mode, velocity layering) flows through
	verbatim into the materialised Assignments — the template is the
	prototype that's cloned per derived (channel, note) entry.

	Fields:
		name:                Label shown in logs; derived Assignments are
		                     named ``"<template name> → <sample stem>"``.
		channel:             mido 0-indexed channel.  Owned exclusively
		                     by this template + any sibling zone-tuned
		                     templates with non-overlapping keyboard ranges.
		keyboard_range:      (lo, hi) inclusive MIDI note range that this
		                     template's samples will be spread across.
		select:              User's select block — filters which samples
		                     are candidates for zone derivation.
		process:             User's process block — MUST contain repitch.
		mode ... stack:      Inherited verbatim by each derived Assignment.
	"""

	name:                str
	channel:             int
	keyboard_range:      tuple[int, int]
	select:              tuple[subsample.query.SelectSpec, ...]
	process:             subsample.query.ProcessSpec
	mode:                str
	loop:                typing.Optional[subsample.query.LoopSpec]
	gain_db:             float
	pan_weights:         typing.Optional[numpy.ndarray]
	output_routing:      typing.Optional[tuple[int, ...]]
	extract:             typing.Optional[subsample.query.ExtractSpec]
	segment_mode:        typing.Union[str, int]
	velocity_trigger:    tuple[int, int]
	velocity_rescale_to: typing.Optional[tuple[int, int]]
	stack:               bool                                = False
	release:             typing.Optional[subsample.query.ReleaseSpec] = None


@dataclasses.dataclass(frozen=True)
class MidiMapResult:

	"""Complete result of parsing a MIDI map YAML file.

	Fields:
		note_map:         Manual note routing entries — (mido_channel, midi_note) →
		                  list of (Assignment, PickSpec) layers.  Zone-tuned
		                  templates are NOT materialised here; the player builds
		                  the working map by merging this with derived entries
		                  on every (re-)materialisation.
		bank_definitions: Parsed program declarations from the optional ``programs:`` key.
		                  Empty list when no banks are declared (single-directory mode).
		bank_channel:     MIDI channel for Program Change bank switching (user-facing 1-16,
		                  0 = omni).  Only meaningful when bank_definitions is non-empty.
		default_bank:     MIDI program number of the bank to activate at startup.
		                  None means use the first bank in the list.
		zone_templates:   Declared zone-tuned assignments.  Empty tuple when no
		                  ``notes: zone-tuned`` entries are present.
	"""

	note_map:         NoteMap
	bank_definitions: list[subsample.bank.BankDefinition]
	bank_channel:     int
	default_bank:     typing.Optional[int]    = None
	zone_templates:   tuple[ZoneTemplate, ...] = ()


def _loudness_positions (
	records: list[subsample.library.SampleRecord],
) -> typing.Optional[list[float]]:

	"""Normalise a ranked pool's RMS levels to [0, 1] for loudness-spaced picks.

	Returns one fraction per record (quietest -> 0.0, loudest -> 1.0), in the
	same order as ``records``, so ``PickSpec.resolve_index`` can lay the pool
	along the velocity axis by real dynamics.  Returns None when the pool has
	fewer than two samples or a single distinct level — both cases where
	proportional spacing is meaningless and resolve_index falls back to even
	rank spacing.
	"""

	if len(records) < 2:
		return None

	levels = [float(record.level.rms) for record in records]
	lo     = min(levels)
	hi     = max(levels)

	if hi <= lo:
		return None

	span = hi - lo

	return [(level - lo) / span for level in levels]


def _ranks_for (pick_spec: subsample.query.PickSpec, ranked_len: int) -> range:

	"""Iterable of 1-indexed ranks this PickSpec might draw at trigger time.

	Used by update_assignments() to pre-compute variants for every reachable
	rank: scalar PickSpec(n, n) yields a single rank; range PickSpec(lo, hi)
	yields lo..hi inclusive, clamped to ranked_len so requests past the end
	collapse onto the last rank (mirrors resolve_index's clamping).  An open
	bound (None — from ``pick: any`` / ``[2, null]`` / ``{gte: 2}``) resolves
	the same way resolve_index does: open lo → 1, open hi → ranked_len.

	A velocity pick can land on any rank in the pool (velocity chooses the
	index at trigger time), so every rank must be pre-baked — the same full
	span as ``pick: any``.  This is already what the open-bound formula below
	yields for its lo=hi=None, but state it so the intent survives if anyone
	ever sets concrete bounds on a velocity pick.
	"""

	if pick_spec.mode == "velocity":
		return range(1, ranked_len + 1)

	hi = ranked_len if pick_spec.hi is None else min(pick_spec.hi, ranked_len)
	lo = 1          if pick_spec.lo is None else pick_spec.lo
	lo = min(lo, hi)

	return range(lo, hi + 1)


def _format_pick_suffix (pick_spec: subsample.query.PickSpec) -> str:

	"""Human-readable ``pick`` annotation for the startup-log line.

	Returns an empty string for the default best-match pick so unannotated
	notes stay terse; otherwise a leading-space suffix: `` pick 3`` (scalar),
	`` pick 2-5`` (closed range), `` pick 2+`` (open upper), `` pick any``
	(both ends open), or `` pick velocity`` (velocity mode, with the curve and
	± spread appended when non-default).  An open lower bound reads as rank 1.
	"""

	if pick_spec.mode == "velocity":
		# A velocity pick carries lo=hi=None, which would otherwise read as
		# "any"; name the mode instead, with curve/spread when they are set.
		curve = "" if pick_spec.curve == "linear" else f"/{pick_spec.curve}"
		vary  = f" ±{pick_spec.variation // 2}" if pick_spec.variation else ""
		space = " by-loudness" if pick_spec.spacing == "loudness" else ""
		return f" pick velocity{curve}{vary}{space}"

	lo, hi = pick_spec.lo, pick_spec.hi

	if lo is None and hi is None:
		return " pick any"

	if hi is None:
		return f" pick {lo}+"

	lo_eff = 1 if lo is None else lo

	if lo_eff == hi:
		return "" if lo_eff == 1 else f" pick {lo_eff}"

	return f" pick {lo_eff}-{hi}"


# Default grid subdivision for a stretch_quantize / pad_quantize step that
# declares no explicit `grid:`.  Seeded from transform.quantize_resolution via
# set_default_quantize_grid() in cli.main() (mirrors the other set-once-from-
# config wirings); 16 (sixteenth notes) matches the historical hard-coded default
# so an unwired caller — a test — behaves exactly as before.
_DEFAULT_QUANTIZE_GRID: int = 16


def set_default_quantize_grid (grid: int) -> None:

	"""Set the process-wide default quantize grid, from transform.quantize_resolution.

	Called once from cli.main().  Previously this config key was parsed and
	documented but never read; the runtime grid always fell back to a hard-coded
	16.  Wiring it here makes the documented default take effect for every
	stretch_quantize / pad_quantize step that omits an explicit `grid:`."""

	global _DEFAULT_QUANTIZE_GRID
	_DEFAULT_QUANTIZE_GRID = grid


def _quantize_params (
	process: subsample.query.ProcessSpec,
	step_name: str,
	config_bpm: float = 0.0,
) -> tuple[typing.Optional[float], int]:

	"""Extract BPM and grid from a stretch_quantize or pad_quantize step.

	Returns (target_bpm, grid). When no explicit BPM is declared in the
	step, falls back to config_bpm (from tempo.bpm in config).
	The grid falls back to the configured _DEFAULT_QUANTIZE_GRID
	(transform.quantize_resolution).  CcBinding values are treated as "provided"
	so the quantize path activates; the actual value is resolved later in
	spec_from_process().
	"""

	# The parser canonicalises the YAML tempo param (legacy `bpm:` included)
	# to "tempo" — that is the ONLY key parser-built steps ever carry.
	step = next(s for s in process.steps if s.name == step_name)
	bpm_raw = step.get("tempo", 0)
	grid_raw = step.get("grid", _DEFAULT_QUANTIZE_GRID)

	# CcBinding means BPM will be resolved at note-on time — treat as "provided".
	if isinstance(bpm_raw, subsample.query.CcBinding):
		default = bpm_raw.default_value
		bpm = default if default is not None and default > 0 else (config_bpm if config_bpm > 0 else 120.0)
		grid = int(grid_raw) if not isinstance(grid_raw, subsample.query.CcBinding) else _DEFAULT_QUANTIZE_GRID
		return (bpm, grid)

	bpm = float(bpm_raw)
	grid = int(grid_raw) if not isinstance(grid_raw, subsample.query.CcBinding) else _DEFAULT_QUANTIZE_GRID

	if bpm <= 0:
		bpm = config_bpm

	return (bpm if bpm > 0 else None, grid)


# MIDI clock runs at 24 pulses per quarter note, so the span of 24 pulses is
# exactly one beat.
_CLOCK_PULSES_PER_BEAT: typing.Final[int] = 24

# How many beats each tempo measurement spans.  This is the jitter defence, and
# it has to be this wide: clock jitter lands on both ends of the window, so the
# measurement error is roughly bpm^2 * 2*jitter / (60 * beats).  At 125 BPM with
# +/-1 ms of per-pulse jitter a ONE-beat window errs by ~0.5 BPM — enough to
# straddle two whole-BPM values and re-bake every variant on a clock that never
# changed (measured: a steady 125 accepted 125, then 124, then 125).  Four beats
# cuts that to ~0.13 BPM, which rounds stably.  The cost is latency: a tempo
# change is adopted a few beats after it happens, which is the right trade when
# the alternative is a spurious re-bake.
_CLOCK_MEASURE_BEATS: typing.Final[int] = 4

# Decimal places the measured tempo is rounded to before use.  0 = whole BPM.
# This rounding is load-bearing, not cosmetic: the value lands in the variant
# cache key (via TimeStretch/PadQuantize repr), so an unstable last digit would
# re-bake every quantize variant for a clock that never actually changed.
_CLOCK_BPM_DECIMALS: typing.Final[int] = 0

# A measured tempo must repeat for this many consecutive beat measurements
# before it is accepted.  Rejects a momentary glitch (one bad measurement) from
# triggering a re-bake.
_CLOCK_DWELL_BEATS: typing.Final[int] = 2

# How far the measured tempo must move away from the one already in force
# before a switch is even considered.  The dwell alone cannot save a sequencer
# sitting exactly BETWEEN two whole-BPM values: at a true 125.5 the rounding is
# decided by jitter, so it lands 125 or 126 at random and the dwell is satisfied
# by chance every few beats — measured as an endless 125/126/125/126 re-bake
# loop.  Requiring the measurement to move more than half a BPM away from the
# accepted value collapses that to "pick one and stay".  Must stay under 1.0 so
# a genuine one-BPM change is still adopted.
_CLOCK_SWITCH_DEADBAND_BPM: typing.Final[float] = 0.75

# Measurements outside these bounds are discarded as glitches (a resuming
# transport, a dropped pulse) rather than accepted as a tempo.
_CLOCK_MIN_BPM: typing.Final[float] = 20.0
_CLOCK_MAX_BPM: typing.Final[float] = 300.0

# A pulse arriving more than this multiple of the expected spacing after the
# previous one is treated as a clock interruption: the measurement window is
# discarded so a dropped/late pulse cannot inflate it into a phantom tempo.  1.5
# catches a single dropped pulse (a 2x gap) while clearing per-pulse jitter
# (~1.1x) — must stay below 2.0.
_CLOCK_GAP_RESET_FACTOR: typing.Final[float] = 1.5

# How far an incoming clock must sit from tempo.bpm before it is
# worth telling the user their quantize grid does not match their sequence.
_CLOCK_MISMATCH_WARN_BPM: typing.Final[float] = 1.0


class _MidiClockTracker:

	"""Derives a stable tempo from MIDI clock pulses.

	A pure state machine — the caller supplies the timestamp, so this is driven
	entirely by its inputs and is testable without a real clock or a player.

	MIDI clock is 24 PPQN, so a window spanning _CLOCK_MEASURE_BEATS beats gives
	the tempo as 60 * beats / window_seconds; measuring across several beats is
	what averages out per-pulse jitter (see _CLOCK_MEASURE_BEATS — a one-beat
	window is measurably not enough).  The window rolls forward and is measured
	once per beat (not per pulse — it already spans several beats, so measuring
	24x more often would only add work on the MIDI callback thread), rounded,
	and accepted only after it has held for _CLOCK_DWELL_BEATS consecutive
	measurements.  Every accepted change re-bakes every quantize variant, so
	rarity matters far more than latency.
	"""

	def __init__ (self) -> None:

		"""Start with no history and no accepted tempo."""

		self._pulses: collections.deque[float] = collections.deque(
			maxlen=_CLOCK_PULSES_PER_BEAT * _CLOCK_MEASURE_BEATS + 1,
		)
		self._pulse_count: int = 0
		self._candidate: typing.Optional[float] = None
		self._candidate_beats: int = 0
		self._accepted: typing.Optional[float] = None

	@property
	def accepted_bpm (self) -> typing.Optional[float]:

		"""The tempo currently in force, or None before the first acceptance."""

		return self._accepted

	def pulse (self, now: float) -> typing.Optional[float]:

		"""Feed one clock pulse; return the newly accepted BPM, else None.

		A value is returned ONLY on the pulse that accepts a *changed* tempo, so
		the caller can treat a non-None return as "the tempo just changed" and
		arm the expensive re-bake without making the comparison itself.

		Args:
			now: Monotonic timestamp of this pulse, in seconds.

		Returns:
			The newly accepted BPM, or None when nothing changed.
		"""

		# A gap much larger than the expected pulse spacing means the clock was
		# interrupted — a dropped/late pulse, or a brief transport stop/start.
		# The stale timestamps left in the window would otherwise inflate
		# window_seconds for the next ~4 beats and, because the inflated value is
		# roughly constant across them, satisfy the dwell on a PHANTOM tempo (a
		# spurious re-bake, then a re-bake back).  Discard the window and rebuild
		# it; the accepted tempo stays sticky (transport stop keeps the last BPM).
		if self._pulses:
			expected = 60.0 / (self._accepted or 120.0) / _CLOCK_PULSES_PER_BEAT
			if now - self._pulses[-1] > _CLOCK_GAP_RESET_FACTOR * expected:
				self._pulses.clear()
				self._candidate = None
				self._candidate_beats = 0

		self._pulses.append(now)
		self._pulse_count += 1

		# Need the full multi-beat window before a measurement means anything.
		if len(self._pulses) <= _CLOCK_PULSES_PER_BEAT * _CLOCK_MEASURE_BEATS:
			return None

		# Roll the window forward a beat at a time, not a pulse at a time.
		if self._pulse_count % _CLOCK_PULSES_PER_BEAT != 0:
			return None

		window_seconds = now - self._pulses[0]

		if window_seconds <= 0.0:
			return None

		raw_bpm = 60.0 * _CLOCK_MEASURE_BEATS / window_seconds

		if not (_CLOCK_MIN_BPM <= raw_bpm <= _CLOCK_MAX_BPM):
			return None

		# Hysteresis against the tempo already in force — see
		# _CLOCK_SWITCH_DEADBAND_BPM.  Rounding a measurement that has not
		# really moved is what lets a between-values clock re-bake forever.
		if (
			self._accepted is not None
			and abs(raw_bpm - self._accepted) < _CLOCK_SWITCH_DEADBAND_BPM
		):
			self._candidate = None
			self._candidate_beats = 0
			return None

		bpm = round(raw_bpm, _CLOCK_BPM_DECIMALS)

		# Restart the dwell whenever the measurement moves.
		if bpm != self._candidate:
			self._candidate = bpm
			self._candidate_beats = 1
			return None

		self._candidate_beats += 1

		if self._candidate_beats < _CLOCK_DWELL_BEATS:
			return None

		# Held long enough — adopt it, but only report an actual change.
		if bpm == self._accepted:
			return None

		self._accepted = bpm
		return bpm


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
		step = subsample.transform.TimeStretch(target_bpm=float(session_bpm), resolution=_DEFAULT_QUANTIZE_GRID)
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
		weights_raw:     Raw YAML value — a scalar stereo position (-100 hard
		                 left … 0 centre … +100 hard right), a list of
		                 per-channel weights, or None for default routing.
		assignment_name: Name for error messages.

	Returns:
		float32 numpy array of weights, or None if not specified (default routing).

	Raises:
		ValueError: If a position is out of range, a weight is negative, or
		the value is neither a number nor a list of numbers.
	"""

	if weights_raw is None:
		return None

	# bool is an int subclass — reject before the number check, or `pan: true`
	# would silently mean "just right of centre".
	if isinstance(weights_raw, bool):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pan must be a position "
			f"(-100..100) or a list of channel weights (got {weights_raw!r})"
		)

	# A scalar is a stereo pan position in the mixer-familiar convention:
	# -100 hard left, 0 centre, +100 hard right.  Pure sugar for a two-channel
	# weight pair — everything downstream (constant-power normalisation,
	# down/upmix to the output layout) is unchanged.
	if isinstance(weights_raw, (int, float)):
		position = float(weights_raw)

		if not (-100.0 <= position <= 100.0):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pan position must be "
				f"between -100 (hard left) and 100 (hard right); got {weights_raw!r}"
			)

		weights_raw = [(100.0 - position) / 200.0, (100.0 + position) / 200.0]

	if isinstance(weights_raw, str) or not hasattr(weights_raw, "__iter__"):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pan must be a position "
			f"(-100..100) or a list of channel weights (got {weights_raw!r})"
		)

	weights = list(weights_raw)

	# Every weight must be a plain number.  A nested list (`pan: [[0.5, 0.5]]`)
	# would otherwise build a 2-D array that passes the mono-length check below
	# but makes an unhashable mix-matrix cache key on every note-on; a string
	# element would surface numpy's context-free "could not convert" message.
	for w in weights:
		if isinstance(w, bool) or not isinstance(w, (int, float)):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: pan weights must be "
				f"numbers (got {w!r})"
			)

	weight_arr = numpy.array(weights, dtype=numpy.float32)

	# `< 0` is False for NaN, so check finiteness explicitly — a NaN/inf weight
	# would otherwise poison the constant-power normalisation silently.
	if not bool(numpy.all(numpy.isfinite(weight_arr))):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pan weights must be finite numbers"
		)

	if numpy.any(weight_arr < 0):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pan weights must be >= 0"
		)

	# The pan length defines the target channel layout, which build_mix_matrix
	# only supports at standard sizes.  Validate at LOAD so a 3/5/7-element pan
	# fails here with a clear message instead of raising a ValueError on EVERY
	# note-on (which _safe_handle_message swallows to an ERROR log, silently
	# dropping every note of the assignment).
	if int(weight_arr.shape[0]) not in subsample.channel.STANDARD_LAYOUTS:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: pan must have a standard "
			f"layout length "
			f"({', '.join(str(k) for k in sorted(subsample.channel.STANDARD_LAYOUTS))}); "
			f"got {int(weight_arr.shape[0])}"
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

	# A scalar (`output: 3`) must fail the documented ValueError contract,
	# not leak a TypeError past the startup/hot-reload catch sites.
	if isinstance(raw, (str, int, float, bool)) or not hasattr(raw, "__iter__"):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: output must be a list of "
			f"1-indexed channel numbers (got {raw!r})"
		)

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


# Accepted keys inside a release: mapping.  Explicit form is {time, curve};
# the {cc: ...} shorthand additionally allows the CcBinding keys.  Enforced so a
# typo fails loudly rather than silently using a default (as _parse_velocity does).
_RELEASE_INNER_KEYS:        typing.Final[frozenset[str]] = frozenset({"time", "curve"})
_RELEASE_CC_SHORTHAND_KEYS: typing.Final[frozenset[str]] = frozenset({"cc", "channel", "min", "max", "default", "curve"})


def _parse_release (
	raw: typing.Any,
	assignment_name: str,
	definitions: typing.Optional[subsample.definitions.Definitions] = None,
) -> typing.Optional[subsample.query.ReleaseSpec]:

	"""Parse the ``release:`` field from a MIDI map assignment into a ReleaseSpec.

	Sets how a sustained (``mode: gated`` or ``loop``) voice fades to silence after
	note-off.  Accepted forms:
	  - ``None`` / ``false``  — no release declared; the player keeps its
	    built-in fixed declick fade.
	  - ``true``              — adaptive tail (shape derived from the sample).
	  - ``full``              — NO fade; play the remaining audio to its natural
	    end (a loop stops looping and rings out its real tail).
	  - a number (ms)         — fade time, cosine shape.
	  - ``{time: <ms|cc-map>, curve: cosine|exponential}`` — explicit; either
	    inner key may be omitted (time → adaptive, curve → cosine).  ``time``
	    may itself be a ``{cc: N, min:, max:, ...}`` mapping, resolved at note-on.

	Syntactic + range validation only; the mode interaction (a release on a
	mode: one_shot voice is inert) is handled at the assignment level in load_midi_map.

	Args:
		raw:             Raw YAML value.
		assignment_name: Name for error messages.

	Returns:
		ReleaseSpec, or None when no release is declared.

	Raises:
		ValueError: On a negative time, unknown curve, bad cc/channel range, or
		            wrong type.
	"""

	if raw is None or raw is False:
		return None

	if raw is True:
		return subsample.query.ReleaseSpec(time=None, curve="cosine")

	if raw == "full":
		return subsample.query.ReleaseSpec(to_end=True)

	if isinstance(raw, (int, float)) and not isinstance(raw, bool):
		return subsample.query.ReleaseSpec(
			time=_parse_release_time(raw, assignment_name, definitions), curve="cosine",
		)

	if isinstance(raw, dict):
		# curve is read for BOTH the explicit and the {cc: ...} shorthand forms,
		# so `release: { cc: 5, curve: exponential }` keeps its curve.
		curve_raw = raw.get("curve", "cosine")

		# A bare {cc: ...} at the top level is shorthand for {time: {cc: ...}}.
		# Whitelist the accepted keys either way so a typo (curev:, tim:) fails
		# loud rather than silently falling back to a default — matching
		# _parse_velocity / the notes and top-level-map parsers.
		if "cc" in raw and "time" not in raw:
			unknown = set(raw) - _RELEASE_CC_SHORTHAND_KEYS
			allowed = _RELEASE_CC_SHORTHAND_KEYS
			time_raw: typing.Any = raw
		else:
			unknown = set(raw) - _RELEASE_INNER_KEYS
			allowed = _RELEASE_INNER_KEYS
			time_raw = raw.get("time")

		if unknown:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: unknown release key(s) "
				f"{sorted(unknown)} — valid: {sorted(allowed)}"
			)

		time  = _parse_release_time(time_raw, assignment_name, definitions) if time_raw is not None else None
		curve = str(curve_raw).strip().lower()

		if curve not in subsample.query.VALID_RELEASE_CURVES:
			valid = ", ".join(sorted(subsample.query.VALID_RELEASE_CURVES))
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: unknown release curve "
				f"{curve_raw!r} (valid: {valid})"
			)

		return subsample.query.ReleaseSpec(time=time, curve=curve)

	raise ValueError(
		f"MIDI map assignment {assignment_name!r}: release must be a number of "
		f"milliseconds, true, or a mapping — got {type(raw).__name__}"
	)


def _parse_release_time (
	raw:             typing.Any,
	assignment_name: str,
	definitions: typing.Optional[subsample.definitions.Definitions] = None,
) -> typing.Union[float, subsample.query.CcBinding]:

	"""Parse a release ``time`` — a non-negative ms scalar or a ``{cc: ...}`` map."""

	if isinstance(raw, dict) and "cc" in raw:
		# Re-raise numeric-coercion errors with the assignment name and offending
		# mapping, matching _parse_velocity_range / the sibling gain parse.
		cc_context = f"MIDI map assignment {assignment_name!r}: release time"
		try:
			cc_num     = subsample.definitions.resolve_scalar(
				definitions, "cc", raw["cc"], cc_context,
			)
			cc_channel = (
				subsample.definitions.resolve_scalar(
					definitions, "channels", raw["channel"], cc_context,
				)
				if "channel" in raw else None
			)
			min_val    = float(raw.get("min", 0.0))
			max_val    = float(raw.get("max", 1.0))
			default    = float(raw["default"]) if "default" in raw else None
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: malformed release time "
				f"cc mapping {raw!r} — {exc}"
			) from exc

		if not (0 <= cc_num <= 127):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: release time cc "
				f"{cc_num} outside the MIDI range 0-127"
			)

		if cc_channel is not None and not (1 <= cc_channel <= 16):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: release time channel "
				f"{cc_channel} outside the MIDI range 1-16"
			)

		return subsample.query.CcBinding(
			cc=cc_num,
			min_val=min_val,
			max_val=max_val,
			default=default,
			channel=cc_channel,
		)

	if isinstance(raw, (int, float)) and not isinstance(raw, bool):
		# Reject non-finite (.inf/.nan from YAML) here — it would otherwise slip
		# through to note-on and crash round() in _resolve_release.
		if not math.isfinite(raw) or raw < 0.0:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: release time "
				f"{raw} must be a finite value >= 0 ms"
			)
		return float(raw)

	raise ValueError(
		f"MIDI map assignment {assignment_name!r}: release time must be a "
		f"non-negative number of milliseconds or a {{cc: ...}} mapping — "
		f"got {type(raw).__name__}"
	)


_LOOP_INNER_KEYS: typing.Final[frozenset[str]] = frozenset({"start", "end", "crossfade"})

# Shortest loop the player will honour.  Auto-detected loops are far longer
# (loopfind enforces its own minimum), so this only guards a deliberately tiny
# manual loop: {} override, which would otherwise make the callback wrap dozens
# of times per buffer.  Below this it is a buzz, not a loop — play gated instead.
_MIN_LOOP_SECONDS: typing.Final[float] = 0.005


def _parse_loop_float (
	raw:             typing.Any,
	assignment_name: str,
	field:           str,
) -> typing.Optional[float]:

	"""Parse an optional, finite, non-negative loop scalar (seconds or ms), or None."""

	if raw is None:
		return None

	if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0.0:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: loop {field} must be a "
			f"finite number >= 0 — got {raw!r}"
		)

	return float(raw)


def _parse_loop_override (
	raw:             typing.Any,
	assignment_name: str,
) -> typing.Optional[subsample.query.LoopSpec]:

	"""Parse a ``loop: {start, end, crossfade}`` override block, or None.

	start/end are seconds, crossfade is milliseconds; each optional (unset uses
	the sample's auto value).  A present block implies ``mode: loop`` — the
	caller enforces that against any explicit ``mode:``.
	"""

	if raw is None:
		return None

	if not isinstance(raw, dict):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'loop' must be a block with "
			f"start/end/crossfade keys — got {type(raw).__name__}"
		)

	unknown = set(raw) - _LOOP_INNER_KEYS
	if unknown:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: unknown loop key(s) "
			f"{sorted(unknown)} — expected start, end, crossfade"
		)

	start = _parse_loop_float(raw.get("start"),     assignment_name, "start")
	end   = _parse_loop_float(raw.get("end"),       assignment_name, "end")
	xfade = _parse_loop_float(raw.get("crossfade"), assignment_name, "crossfade")

	if start is not None and end is not None and end <= start:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: loop end ({end}s) must be "
			f"greater than start ({start}s)"
		)

	return subsample.query.LoopSpec(start=start, end=end, crossfade=xfade)


def _parse_mode (
	raw:             dict[str, typing.Any],
	assignment_name: str,
	process:         subsample.query.ProcessSpec,
) -> tuple[str, typing.Optional[subsample.query.LoopSpec]]:

	"""Resolve an assignment's playback mode and optional loop override.

	- ``one_shot:`` is a removed alias — its presence is a hard load error
	  (breaking change: migrate ``true`` → ``mode: one_shot``, ``false`` →
	  ``mode: gated``).
	- ``mode:`` must be one of query.VALID_MODES; default one_shot.
	- a ``loop: {...}`` block implies ``mode: loop``; a contradicting explicit
	  ``mode:`` is an error.
	- ``mode: loop`` combined with a timeline-altering step (repitch, time/pad
	  quantize, or reverse) is deferred in v1: warn and fall back to gated,
	  dropping the loop — the stored points are in the sample's own frames and
	  would not survive the transform.
	"""

	if "one_shot" in raw:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'one_shot' is no longer "
			f"supported — use 'mode:'.  'one_shot: true' → 'mode: one_shot' "
			f"(plays to the end, ignores note-off); 'one_shot: false' → "
			f"'mode: gated' (note-off releases).  'mode: loop' holds a seamless "
			f"loop while the key is held."
		)

	loop_override = _parse_loop_override(raw.get("loop"), assignment_name)

	mode_raw = raw.get("mode")
	if mode_raw is None:
		mode = "loop" if loop_override is not None else "one_shot"
	else:
		if not isinstance(mode_raw, str) or mode_raw not in subsample.query.VALID_MODES:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: invalid mode {mode_raw!r} "
				f"— expected one of {sorted(subsample.query.VALID_MODES)}"
			)
		mode = mode_raw
		if loop_override is not None and mode != "loop":
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: a 'loop:' block implies "
				f"'mode: loop', but 'mode: {mode}' was set — remove one of them"
			)

	# The stored loop points live in the sample's own (forward) timeline, so any
	# step that re-times or re-orders that timeline invalidates them — defer such
	# an assignment to gated at load.  This is the COMPLETE set of timeline-
	# altering processors: repitch and time/pad-quantize re-time, reverse mirrors.
	# Every other process step preserves the timeline and loops correctly on its
	# variant, so any future timeline-altering processor MUST be added here.
	if mode == "loop" and (
		process.has_repitch()
		or process.has_stretch_quantize()
		or process.has_pad_quantize()
		or process.has_reverse()
	):
		_log.warning(
			"MIDI map assignment %r: 'mode: loop' with repitch, time/pad-quantize, or "
			"reverse is not supported yet — playing gated (no loop).  The loop points "
			"live in the sample's own timeline and would not survive the transform.",
			assignment_name,
		)
		mode          = "gated"
		loop_override = None

	return mode, loop_override


def _parse_silenced_by (
	raw:             typing.Any,
	assignment_name: str,
	namespaces: typing.Optional[typing.Mapping[str, typing.Mapping[str, int]]] = None,
) -> typing.Optional[subsample.query.ChokeSpec]:

	"""Parse the ``silenced_by:`` field into a ChokeSpec (or None).

	Accepts a single note, a list of notes, the token ``self``, and ``self``
	freely mixed into a list.  Notes reuse the ``notes:`` resolver, so
	``drum.hi_hat_closed`` / ``42`` / ``C3`` all resolve identically and an
	unknown symbol or out-of-range note fails loudly at load.  ``self`` means
	the assignment's own note(s) choke its own voices (single physical
	instrument).  Absent / null / false → None (no choke).
	"""

	if raw is None or raw is False:
		return None

	items = raw if isinstance(raw, list) else [raw]

	is_self = False
	notes:   set[int] = set()

	for item in items:
		# ``self`` sentinel — the assignment's own note(s) choke its own voices.
		if isinstance(item, str) and item.strip().lower() == "self":
			is_self = True
			continue
		# Guard bool BEFORE _parse_single_note: bool is an int subclass, so
		# ``silenced_by: true`` would otherwise resolve to note 1.
		if isinstance(item, bool):
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: 'silenced_by' entries "
				f"must be a note (e.g. drum.hi_hat_closed, 42, C3) or 'self', "
				f"got {item!r}"
			)
		notes.add(_parse_single_note(item, assignment_name, namespaces))

	if not is_self and not notes:
		# An empty list (or a list of nothing but stripped-away tokens) means no
		# choke rather than an error — harmless, and lets a templated default be
		# blanked with ``silenced_by: []``.
		return None

	return subsample.query.ChokeSpec(is_self=is_self, notes=frozenset(notes))


_VELOCITY_INNER_KEYS: typing.Final[frozenset[str]] = frozenset({"trigger", "rescale"})


def _parse_velocity_range (
	raw:             typing.Any,
	assignment_name: str,
	field_label:     str,
) -> tuple[int, int]:

	"""Parse a 2-element [lo, hi] velocity range, used for both ``trigger`` and ``rescale``.

	Returns the validated (lo, hi) tuple.  Raises ValueError when the
	shape, types, or bounds are wrong — the message names the assignment
	and the field so a typo in a 200-line MIDI map is locatable.
	"""

	if not isinstance(raw, (list, tuple)) or len(raw) != 2:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity {field_label} "
			f"must be a 2-element list [lo, hi], got {raw!r}"
		)

	try:
		lo = int(raw[0])
		hi = int(raw[1])
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity {field_label} "
			f"values must be integers, got {raw!r} ({exc})"
		) from exc

	if not (0 <= lo <= 127 and 0 <= hi <= 127):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity {field_label} "
			f"[{lo}, {hi}] is outside the valid MIDI range [0, 127]"
		)

	if lo > hi:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity {field_label} "
			f"[{lo}, {hi}] has lo > hi"
		)

	return (lo, hi)


def _parse_velocity (
	raw:             typing.Any,
	assignment_name: str,
) -> tuple[tuple[int, int], typing.Optional[tuple[int, int]]]:

	"""Parse the ``velocity:`` YAML field into (trigger, rescale_to).

	Accepted forms:
	  - None / omitted         → ((0, 127), None)    — default, all velocities, no rescale
	  - [lo, hi]               → ((lo, hi), None)    — shortcut: filter only
	  - {trigger: [lo, hi]}                                        — explicit dict, no rescale
	  - {trigger: [lo, hi], rescale: false}                        — same; rescale opt-out
	  - {trigger: [lo, hi], rescale: true}            → rescale_to=(0, 127)
	  - {trigger: [lo, hi], rescale: [out_lo, out_hi]} → rescale_to=(out_lo, out_hi)

	Inner-key whitelist (``trigger``, ``rescale``) is enforced regardless
	of strict mode so a typo like ``trggier:`` fails loud rather than
	silently using the default trigger.  Single-point trigger + list
	rescale raises (mapping a single input to a range is undefined).
	"""

	if raw is None:
		return ((0, 127), None)

	# List shortcut: velocity: [0, 63] — filter only, no rescale.
	if isinstance(raw, list):
		return (_parse_velocity_range(raw, assignment_name, "trigger"), None)

	if not isinstance(raw, dict):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity must be a list "
			f"[lo, hi] or a dict with 'trigger' (and optional 'rescale'), got "
			f"{type(raw).__name__}"
		)

	unknown = set(raw) - _VELOCITY_INNER_KEYS
	if unknown:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: unknown velocity key(s) "
			f"{sorted(unknown)!r} (valid: {sorted(_VELOCITY_INNER_KEYS)!r})"
		)

	if "trigger" not in raw:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity dict requires "
			f"a 'trigger' field"
		)

	trigger = _parse_velocity_range(raw["trigger"], assignment_name, "trigger")

	rescale_raw = raw.get("rescale", False)
	rescale_to: typing.Optional[tuple[int, int]]

	if rescale_raw is False or rescale_raw is None:
		rescale_to = None
	elif rescale_raw is True:
		# "true" means rescale to the full MIDI range — equivalent to writing
		# rescale: [0, 127].  Note that an output of 0 is valid (silent) and
		# the user can explicitly write [1, 127] if they want to avoid that.
		rescale_to = (0, 127)
	else:
		rescale_to = _parse_velocity_range(rescale_raw, assignment_name, "rescale")

	# A single-velocity trigger has no defined mapping to a range — the
	# linear formula would divide by zero.  Reject explicitly so the user
	# gets a targeted error rather than a confusing runtime division.
	if rescale_to is not None and trigger[0] == trigger[1]:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: velocity rescale "
			f"requires a non-point trigger (got [{trigger[0]}, {trigger[1]}])"
		)

	return (trigger, rescale_to)


def _validate_velocity_layers (note_map: NoteMap) -> None:

	"""Check the cross-layer invariants on the assembled note map.

	For each (channel, note) with more than one entry:
	  - Overlapping ``velocity_trigger`` ranges raise ValueError naming
	    both assignments — overlap is almost always a copy-paste mistake,
	    and silently picking one would surprise the user mid-performance.
	    The exception is stacking: when *both* overlapping members set
	    ``stack: true`` the overlap is intentional (they sound together),
	    so it is permitted.
	  - Gaps in [0, 127] coverage log WARNING listing the uncovered span.
	    Notes whose velocity falls into a gap play nothing, matching the
	    existing "no mapping for this note" semantics.

	Called once by ``load_midi_map`` after every assignment has been
	appended to its layer list.
	"""

	for (ch, note), entries in note_map.items():
		if len(entries) <= 1:
			continue

		# Sort by trigger lo so the gap walk is linear.  Stable sort means
		# ties preserve YAML order for deterministic error messages.
		sorted_layers = sorted(entries, key=lambda e: e[0].velocity_trigger)

		# Overlap check: no two layers may share a velocity value unless both
		# opt into stacking.  This must compare EVERY pair, not just sorted
		# neighbours: once stacked pairs are allowed to pass, a wide stacked
		# layer can "bridge" past a later non-stacked layer that never
		# overlaps its own immediate neighbour (e.g. [0,100] stacked,
		# [5,6] stacked, [50,60] non-stacked — the only adjacent overlap is
		# the consensual one, yet [0,100] and [50,60] collide).  Layer counts
		# are tiny (≤ ~16 per note), so the O(n²) scan is free at load time.
		for i in range(len(sorted_layers)):
			for j in range(i + 1, len(sorted_layers)):
				asgn_a, _ = sorted_layers[i]
				asgn_b, _ = sorted_layers[j]
				lo_a, hi_a = asgn_a.velocity_trigger
				lo_b, hi_b = asgn_b.velocity_trigger

				# Sorted by lo, so for i < j the pair overlaps iff a's high
				# end reaches b's low end.
				if hi_a >= lo_b and not (asgn_a.stack and asgn_b.stack):
					raise ValueError(
						f"MIDI map ch{ch + 1} note {note}: velocity ranges of "
						f"assignments {asgn_a.name!r} [{lo_a}, {hi_a}] and "
						f"{asgn_b.name!r} [{lo_b}, {hi_b}] overlap — overlapping "
						f"layers create an ambiguous trigger.  Adjust ranges so "
						f"each velocity maps to exactly one layer, or set "
						f"``stack: true`` on both to sound them together."
					)

		# Gap check (warning only).  Walk [0, 127] flagging any velocity
		# value uncovered by the union of trigger ranges.
		gaps: list[tuple[int, int]] = []
		cursor = 0

		for asgn, _ in sorted_layers:
			lo, hi = asgn.velocity_trigger
			if lo > cursor:
				gaps.append((cursor, lo - 1))
			# max(): stacked layers may overlap, so a later layer's hi can be
			# below the running cursor — never let coverage regress.
			cursor = max(cursor, hi + 1)

		if cursor <= 127:
			gaps.append((cursor, 127))

		if gaps:
			_log.warning(
				"MIDI map ch%d note %d: velocity coverage has gap(s) %s — "
				"velocities in these range(s) will not trigger any layer",
				ch + 1, note,
				", ".join(f"[{lo}, {hi}]" for lo, hi in gaps),
			)


def _validate_zone_assignments (
	zone_templates:  list["ZoneTemplate"],
	manual_channels: set[int],
) -> None:

	"""Cross-validate zone-tuned templates after every assignment is parsed.

	Two invariants enforced (raise ValueError on either):
	  - **Channel exclusivity** — a channel that carries a zone-tuned
	    template must not also have any manual ``notes: …`` assignments.
	    Zone-tuned owns its channel so the keyboard layout is fully
	    derived without conflicting fixed-note assignments.
	  - **Range non-overlap** — two zone-tuned templates on the same
	    channel must declare non-overlapping ``keyboard_range`` spans.
	    Mirrors the velocity-layering overlap rule and the same musical
	    rationale: overlap is almost always a copy-paste mistake and
	    silently picking one would surprise the user mid-performance.

	The keyboard_range bounds on each template are already validated
	(0-127 and lo <= hi) by ``_parse_zone_notes``; this function checks
	only the cross-template invariants.
	"""

	zone_channels: set[int] = {t.channel for t in zone_templates}

	# Channel-exclusivity check.
	collisions = zone_channels & manual_channels
	if collisions:
		offenders = sorted(collisions)
		offending_zone_names = [
			t.name for t in zone_templates if t.channel in collisions
		]
		raise ValueError(
			f"MIDI map: channel(s) {[c + 1 for c in offenders]!r} have both "
			f"zone-tuned templates and manual note assignments — zone-tuned "
			f"owns its channel exclusively.  Offending zone-tuned "
			f"assignment(s): {offending_zone_names!r}"
		)

	# Range non-overlap check, per channel.
	by_channel: dict[int, list["ZoneTemplate"]] = {}
	for t in zone_templates:
		by_channel.setdefault(t.channel, []).append(t)

	for ch, templates in by_channel.items():
		if len(templates) <= 1:
			continue

		# Sort by lo so the adjacency walk is linear.
		sorted_templates = sorted(templates, key=lambda t: t.keyboard_range)

		for i in range(len(sorted_templates) - 1):
			a = sorted_templates[i]
			b = sorted_templates[i + 1]
			lo_a, hi_a = a.keyboard_range
			lo_b, hi_b = b.keyboard_range

			if hi_a >= lo_b:
				raise ValueError(
					f"MIDI map ch{ch + 1}: zone-tuned keyboard ranges of "
					f"templates {a.name!r} [{lo_a}, {hi_a}] and "
					f"{b.name!r} [{lo_b}, {hi_b}] overlap — adjust ranges so "
					f"each MIDI note belongs to exactly one zone-tuned template."
				)


def _build_choke_map (
	note_map: "NoteMap",
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:

	"""Compile the ``silenced_by:`` declarations into a note-on choke table.

	Returns ``{(channel, killer_note): frozenset of (channel, victim_note)}`` —
	"when this note fires on this channel, fast-damp every sounding voice on
	those victim keys".  Because choke is resolved to notes (never sample or
	assignment identity), the whole table is a pure function of the finished
	note map and survives zone re-minting / bank swaps unchanged.

	Each assignment declaring ``silenced_by`` contributes: its victim keys are
	the (channel, note) entries it occupies; its killer keys are those same
	keys when ``self`` was listed, plus each explicit choker note on the
	assignment's own channel.  Every killer key maps to the union of victim
	keys it damps (so two assignments naming the same killer both get cut).
	"""

	# One victim-key set + spec per declaring assignment, keyed by identity so a
	# multi-note assignment is gathered once across the (channel, note) entries
	# it occupies.
	victim_keys: dict[int, set[tuple[int, int]]]        = {}
	specs:       dict[int, subsample.query.ChokeSpec]   = {}

	for (ch, note), layers in note_map.items():
		for assignment, _pick in layers:
			spec = assignment.silenced_by
			if spec is None:
				continue
			victim_keys.setdefault(id(assignment), set()).add((ch, note))
			specs[id(assignment)] = spec

	choke_map: dict[tuple[int, int], set[tuple[int, int]]] = {}

	for aid, vkeys in victim_keys.items():
		spec        = specs[aid]
		channels    = {ch for ch, _ in vkeys}          # a single channel in practice
		killer_keys: set[tuple[int, int]] = set()

		if spec.is_self:
			killer_keys |= vkeys                        # own note(s) choke own voices

		for ch in channels:
			for killer_note in spec.notes:
				killer_keys.add((ch, killer_note))

		for kk in killer_keys:
			choke_map.setdefault(kk, set()).update(vkeys)

	return {kk: frozenset(vk) for kk, vk in choke_map.items()}


def _validate_choke_targets (note_map: "NoteMap") -> None:

	"""Warn (never raise) when a declared ``silenced_by`` note has no assignment
	on its channel — the likely-typo signal.

	Not an error: a note that maps to no sound but still damps a ringing voice
	is a legitimate "silent grab" (a dedicated choke key), so we surface the
	suspicion without rejecting it.  De-duplicated per (assignment, channel,
	note) so a multi-note assignment warns at most once per missing target.
	"""

	mapped = set(note_map.keys())
	seen:   set[tuple[int, int, int]] = set()

	for (ch, _note), layers in note_map.items():
		for assignment, _pick in layers:
			spec = assignment.silenced_by
			if spec is None:
				continue
			for killer_note in spec.notes:
				key = (id(assignment), ch, killer_note)
				if key in seen:
					continue
				seen.add(key)
				if (ch, killer_note) not in mapped:
					_log.warning(
						"MIDI map assignment %r: silenced_by note %d has no "
						"assignment on channel %d — the choke will only fire if "
						"that note is played anyway (typo?).",
						assignment.name, killer_note, ch + 1,
					)


# Note name conversion — delegated to pymididefs.
_midi_to_note_name = pymididefs.notes.note_to_name
_parse_note_name = pymididefs.notes.name_to_note


# Symbolic note-name namespaces.  A `notes:` value of "drum.kick_1" is looked
# up here: split on the first dot, the prefix selects a PyMidiDefs table, the
# symbol (case-insensitive) is the dict key.  The dict shape is the
# extension point — adding "program" → pymididefs.gm.GM_INSTRUMENT_MAP later
# requires only one entry, no parser changes.
#
# The drum table merges GM_DRUM_MAP (the canonical one-name-per-note key map)
# with GM_DRUM_PRIMARY_ALIASES (the unnumbered kick / snare / crash / ride
# aliases PyMidiDefs keeps separate), so "drum.kick" resolves to the GM primary
# (Bass Drum 1 = 36) right alongside the explicit "drum.kick_1".  The two maps
# are disjoint by construction, so neither shadows the other.
_SYMBOL_NAMESPACES: typing.Final[dict[str, typing.Mapping[str, int]]] = {
	"drum": {**pymididefs.drums.GM_DRUM_MAP, **pymididefs.drums.GM_DRUM_PRIMARY_ALIASES},
}


def _parse_single_note (
	item: typing.Any,
	assignment_name: str,
	namespaces: typing.Optional[typing.Mapping[str, typing.Mapping[str, int]]] = None,
) -> int:

	"""Resolve one note value: int, numeric string, note-name string, or
	symbolic form like ``drum.kick_1``.

	Extracted from ``_parse_note_spec`` so other parsers (e.g. the
	``range:`` field of ``notes: { mode: zone-tuned, range: [C4, G9] }``)
	can reuse the same accept-anything-then-validate dispatch.

	``namespaces`` is the symbolic-name table view — the module-global
	``_SYMBOL_NAMESPACES`` when None, or that merged with the map's mounted
	``definitions:`` prefixes (load_midi_map threads the merged view here).
	"""

	if namespaces is None:
		namespaces = _SYMBOL_NAMESPACES

	if isinstance(item, int):
		if not 0 <= item <= 127:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: note {item} is outside [0, 127]"
			)
		return item

	if isinstance(item, str):

		# Symbolic form ("drum.kick_1") — single dot, no range separator.
		# Looked up case-insensitively in the namespace tables.  Unknown
		# namespaces fall through to the int / note-name path so a typo
		# like "C.4" still gets the existing note-name error.
		if "." in item and ".." not in item:
			prefix, _, sym = item.partition(".")
			table = namespaces.get(prefix.lower())

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


_ZONE_TUNED_SENTINEL: typing.Final[str] = "zone-tuned"

_VALID_NOTES_INNER_KEYS: typing.Final[frozenset[str]] = frozenset({"mode", "range"})


def _parse_zone_notes (
	notes_raw:       typing.Any,
	assignment_name: str,
	namespaces: typing.Optional[typing.Mapping[str, typing.Mapping[str, int]]] = None,
) -> typing.Optional[tuple[int, int]]:

	"""Detect the ``zone-tuned`` form on the ``notes:`` field.

	Returns:
	  - ``(lo, hi)`` keyboard range when the field is zone-tuned (string
	    sentinel ``"zone-tuned"`` or dict form ``{mode: zone-tuned, …}``).
	  - ``None`` when the field is a regular note spec (int, name, range,
	    list, symbolic); the caller falls through to ``_parse_note_spec``.

	Validation (raises ValueError on failure):
	  - Unknown inner keys under the dict form (typo guard mirroring
	    ``_VELOCITY_INNER_KEYS``).
	  - ``mode`` missing or not equal to ``zone-tuned``.
	  - ``range:`` not a 2-element list, lo > hi, or out of [0, 127].
	"""

	if notes_raw == _ZONE_TUNED_SENTINEL:
		return (0, 127)

	if not isinstance(notes_raw, dict):
		return None

	unknown = set(notes_raw) - _VALID_NOTES_INNER_KEYS
	if unknown:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: unknown notes key(s) "
			f"{sorted(unknown)!r} (valid: {sorted(_VALID_NOTES_INNER_KEYS)!r})"
		)

	mode = notes_raw.get("mode")
	if mode != _ZONE_TUNED_SENTINEL:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: notes dict requires "
			f"mode: {_ZONE_TUNED_SENTINEL!r} (got {mode!r})"
		)

	range_raw = notes_raw.get("range")

	if range_raw is None:
		return (0, 127)

	if not isinstance(range_raw, list) or len(range_raw) != 2:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: notes range must be a "
			f"2-element list [lo, hi], got {range_raw!r}"
		)

	lo = _parse_single_note(range_raw[0], assignment_name, namespaces)
	hi = _parse_single_note(range_raw[1], assignment_name, namespaces)

	if lo > hi:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: notes range [{lo}, {hi}] "
			f"has lo > hi"
		)

	return (lo, hi)


def _parse_note_spec (
	notes_raw: typing.Any,
	assignment_name: str,
	namespaces: typing.Optional[typing.Mapping[str, typing.Mapping[str, int]]] = None,
) -> list[int]:

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

	Does NOT accept the zone-tuned forms (``"zone-tuned"`` or
	``{mode: zone-tuned, …}``); callers must dispatch through
	``_parse_zone_notes`` first to detect those and route to the zone-tuned
	template path.

	Args:
		notes_raw:       Raw YAML value for the 'notes' field.
		assignment_name: Assignment name used in error messages.

	Returns:
		Non-empty list of MIDI note numbers.

	Raises:
		ValueError: If any note value is malformed or outside [0, 127].
	"""

	# Range syntax: "C2..C4" or "36..60".  Reject symbolic ranges explicitly so
	# the user gets a targeted error pointing at list syntax, rather than the
	# generic "not a valid note name" from the inner _parse_single_note() call.
	if isinstance(notes_raw, str) and ".." in notes_raw:

		lo_str, hi_str = notes_raw.split("..", 1)

		# Either end being symbolic makes the range meaningless — drum names
		# are not a musical sequence ("36..drum.kick_1" is as invalid as
		# "drum.kick_1..drum.snare_1").
		if "." in lo_str or "." in hi_str:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: "
				f"range syntax (a..b) is not supported for symbolic notes — "
				f"use a list instead, e.g. [drum.kick_1, drum.snare_1]"
			)
		lo = _parse_single_note(lo_str.strip(), assignment_name, namespaces)
		hi = _parse_single_note(hi_str.strip(), assignment_name, namespaces)

		if lo > hi:
			raise ValueError(
				f"MIDI map assignment {assignment_name!r}: note range {notes_raw!r} — "
				f"start ({lo}) must be <= end ({hi})"
			)

		return list(range(lo, hi + 1))

	# Reject bool explicitly: YAML `yes`/`no` parse to True/False, and bool is
	# an int subclass, so `notes: yes` would otherwise silently map note 1.
	if isinstance(notes_raw, bool):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'notes' value "
			f"{notes_raw!r} is not a note — use a number, note name, "
			f"or drum.<name>"
		)

	# Single value (int or string).
	if isinstance(notes_raw, (int, str)):
		return [_parse_single_note(notes_raw, assignment_name, namespaces)]

	# Anything else must be a list.  A bare non-list scalar (float, mapping,
	# null) is a mistake — reject it with a clear message rather than crashing
	# on iteration below (a float raises "not iterable", and load_midi_map
	# promises FileNotFoundError/ValueError only).
	if not isinstance(notes_raw, list):
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'notes' value "
			f"{notes_raw!r} is not a note or a list of notes"
		)

	# List of mixed values.  An empty list would silently map nothing while
	# still claiming its channel against zone-tuned templates — reject it.
	if not notes_raw:
		raise ValueError(
			f"MIDI map assignment {assignment_name!r}: 'notes' list must "
			f"contain at least one note"
		)

	return [_parse_single_note(item, assignment_name, namespaces) for item in notes_raw]


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

	# Auto-generate the sidecar if missing but the audio file exists.  Route
	# through ensure_sample_assets (the same heal path the library loader uses)
	# rather than hand-rolling the write — so the reference sidecar upholds every
	# writer contract: hash-before-decode, read through read_audio_file (float
	# ceiling honoured), the configured AnalysisConfig, compute_loop (no
	# permanently-unloopable trap), and channel-format-aware analysis.
	if not sidecar_path.exists() and path.exists():
		_log.info("Generating analysis sidecar for reference %s", path.name)

		if subsample.cache.ensure_sample_assets(path, with_preview=False) is None:
			_log.warning(
				"Could not generate sidecar for %s — this reference will be skipped",
				path.name,
			)
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

	return subsample.library.SampleRecord(
		sample_id      = subsample.library.allocate_id(),
		name           = str(path.resolve()),
		spectral       = result.spectral,
		rhythm         = result.rhythm,
		pitch          = result.pitch,
		timbre         = result.timbre,
		level          = result.level,
		band_energy    = result.band_energy,
		params         = result.params,
		duration       = result.duration,
		audio          = None,
		filepath       = path if path.exists() else None,
		channel_format = result.channel_format,
		loop           = result.loop,
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
		spectral       = result.spectral,
		rhythm         = result.rhythm,
		pitch          = result.pitch,
		timbre         = result.timbre,
		level          = result.level,
		band_energy    = result.band_energy,
		params         = result.params,
		duration       = result.duration,
		audio          = audio,
		filepath       = path,
		channel_format = result.channel_format,
		loop           = result.loop,
		audio_sample_rate = target_sample_rate or result.params.sample_rate,
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

	# Extract unique assignments from the note map.  Each (channel, note)
	# now holds a list of velocity layers; iterate the layers and dedupe
	# Assignment identities so an assignment shared across notes (or across
	# layers on the same note) is only processed once.
	seen_assignments: set[int] = set()

	for entries in note_map.values():
		for (assignment, _pick_spec) in entries:
			assignment_id = id(assignment)
			if assignment_id in seen_assignments:
				continue
			seen_assignments.add(assignment_id)

			# All three collections must walk EVERY spec in the assignment's
			# select chain — a fallback chain's primary spec is just as able
			# to carry a path/directory predicate as its last.
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

			# Skip if this exact file is already loaded — typically because the
			# main instrument loader already picked it up (the predicate may
			# point at a subtree of cfg.library.directory).  Keyed by PATH,
			# not stem: two take-folders may each hold "01.wav", and both must
			# load as distinct samples.
			if instrument_lib.find_by_path(audio_path) is not None:
				continue

			record = _load_instrument_from_path(
				audio_path, target_sample_rate, with_preview=with_preview,
			)

			if record is not None:
				instrument_lib.add(record)
				# Rank the new sample against every reference too.  At startup
				# this no-ops (references are added just below, and add() early-
				# returns while the matrix has none) and the add_reference pass
				# ranks it via its instrument snapshot; on a hot-reload the
				# references already exist, so this is what threads the newly
				# loaded samples into their rankings — without it a similarity
				# select silently never sees them until restart.
				for matrix in matrices:
					matrix.add(record)
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

		# Skip if this exact file is already loaded.  Keyed by PATH, not stem,
		# so a same-stem file from a different folder still loads.
		existing_id = instrument_lib.find_by_path(path)
		if existing_id is not None:
			_log.debug(
				"Instrument sample %s already in library (id %d) — skipping load from %s",
				path.stem, existing_id, path,
			)
			continue

		record = _load_instrument_from_path(
			path, target_sample_rate, with_preview=with_preview,
		)
		if record is None:
			continue

		instrument_lib.add(record)
		# Path-pinned instruments load AFTER the reference pass above, so the
		# add_reference snapshot never saw them — rank them into every existing
		# reference here, or a similarity-ordered pool would never include them.
		for matrix in matrices:
			matrix.add(record)

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

	# Dedupe by Assignment identity across both notes and velocity layers
	# so an Assignment that appears on multiple notes (e.g. drum group) or
	# in a velocity-layered note is only validated once.
	seen_assignments: set[int] = set()

	for entries in note_map.values():
		for (assignment, _pick_spec) in entries:

			if assignment.extract is None:
				continue

			aid = id(assignment)

			if aid in seen_assignments:
				continue

			seen_assignments.add(aid)

			# Gather the unique (channel_format, in_channels) combinations that
			# any select spec could pick.  We deliberately don't supply a
			# beats_resolver or bpm — quantized_beats and duration_beats
			# predicates are skipped (matches() returns False), so they appear as
			# if no sample matched.  This is conservative: validation may miss a
			# few samples in unusual maps, but won't reject correct ones.
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


def _parse_templates (templates_raw: typing.Any) -> dict[str, dict[str, typing.Any]]:

	"""Validate and return the optional ``templates:`` section.

	Each entry is a named bundle of assignment fields that assignments can
	inherit via a ``template:`` reference (see
	``_resolve_assignment_inheritance``).  Returns an empty dict when the
	section is absent, so callers treat "no templates" and "empty templates"
	identically.

	Templates are flat: a template body may not itself carry a ``template:``
	key — inheritance is one level deep, which keeps resolution predictable.
	"""

	if templates_raw is None:
		return {}

	if not isinstance(templates_raw, dict):
		raise ValueError(
			f"MIDI map 'templates' must be a mapping of name → assignment "
			f"fields, got {type(templates_raw).__name__}"
		)

	for name, body in templates_raw.items():

		if not isinstance(body, dict):
			raise ValueError(
				f"MIDI map template {name!r}: value must be a mapping of "
				f"assignment fields, got {type(body).__name__}"
			)

		if "template" in body:
			raise ValueError(
				f"MIDI map template {name!r}: templates may not use 'template' "
				f"themselves — inheritance is one level deep"
			)

	return templates_raw


def _resolve_assignment_inheritance (
	assignments_raw: typing.Any,
	templates_raw:   typing.Any,
) -> list[dict[str, typing.Any]]:

	"""Merge each assignment over the template(s) it names via ``template:``.

	A pre-pass run before the per-assignment parsing loop: an assignment with
	``template: percussion`` (or ``template: [a, b]``) inherits the fields of
	those named templates, then its own keys override them.  The merge is
	shallow and top-level — a child ``process`` / ``select`` replaces the
	template's wholesale rather than deep-merging, mirroring YAML ``<<``
	merge-key behaviour.  Multiple templates apply left-to-right (a later one
	overrides an earlier one); the assignment's own keys win over all of them.
	The consumed ``template`` key is stripped from the result.

	Returns a new list of resolved assignment dicts; assignments without a
	``template:`` reference pass through unchanged.
	"""

	if not isinstance(assignments_raw, list):
		raise ValueError(
			f"MIDI map 'assignments' must be a list, got "
			f"{type(assignments_raw).__name__}"
		)

	templates = _parse_templates(templates_raw)

	resolved: list[dict[str, typing.Any]] = []

	for index, assignment_raw in enumerate(assignments_raw, start=1):

		if not isinstance(assignment_raw, dict):
			raise ValueError(
				f"MIDI map assignment #{index}: expected a mapping, got "
				f"{type(assignment_raw).__name__}"
			)

		template_ref = assignment_raw.get("template")

		if template_ref is None:
			resolved.append(assignment_raw)
			continue

		name = assignment_raw.get("name", "<unnamed>")

		# Normalise the scalar-or-list shortcut to an ordered name list.
		if isinstance(template_ref, str):
			names = [template_ref]
		elif isinstance(template_ref, list) and all(isinstance(n, str) for n in template_ref):
			names = template_ref
		else:
			raise ValueError(
				f"MIDI map assignment {name!r}: 'template' must be a template "
				f"name or a list of names, got {template_ref!r}"
			)

		# Build the inherited base, then let the assignment's own keys win.
		merged: dict[str, typing.Any] = {}

		for tname in names:

			if tname not in templates:
				raise ValueError(
					f"MIDI map assignment {name!r}: unknown template {tname!r} "
					f"(defined templates: {sorted(templates)})"
				)

			merged.update(templates[tname])

		merged.update(assignment_raw)
		merged.pop("template", None)

		resolved.append(merged)

	return resolved


# Valid top-level keys in a MIDI map.  Unknown keys raise (typo guard,
# mirroring the velocity/notes inner-key validation) so a misspelt or
# obsolete key fails loudly rather than being silently ignored — e.g. the
# former `banks:`/`bank_channel:`/`default_bank:` keys, renamed to the
# MIDI-correct `programs:`/`program_channel:`/`default_program:` (each entry
# is selected by a Program Change, not by MIDI Bank Select).
_VALID_MAP_KEYS: typing.Final[frozenset[str]] = frozenset({
	"definitions", "programs", "program_channel", "default_program", "assignments",
	"templates",
})

# Every key a single assignment mapping may carry.  A typo (`mdoe:`, `realease:`,
# `gain_db:`) would otherwise be silently ignored and the assignment revert to
# defaults; this whitelist fails such mistakes loudly, matching the top-level and
# inner-block key guards.  `template` is consumed by inheritance resolution
# before the per-assignment loop but is listed so a stray one still validates.
_VALID_ASSIGNMENT_KEYS: typing.Final[frozenset[str]] = frozenset({
	"name", "channel", "notes", "select", "process", "mode", "loop", "release",
	"gain", "pan", "output", "extract", "velocity", "stack", "silenced_by",
	"template",
})


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
	  mode:     Playback — one_shot (default, ignores note_off) | gated (note-off
	            releases) | loop (holds a loop while held); loop: {...} overrides points.
	  gain:     Level offset in dB (default 0.0).
	  pan:      Channel weights defining a target layout (e.g. [50, 50] for stereo,
	            [50, 50, 0, 0, 30, 30] for 5.1).  Omit for default routing.
	  velocity: Optional velocity-layering field.  ``[lo, hi]`` shortcut
	            filters this assignment to that velocity range; the dict
	            form ``{trigger: [lo, hi], rescale: …}`` adds an optional
	            in-band rescale to a target output range so a low-velocity
	            layer can still play through its own full dynamic envelope.
	            Multiple non-overlapping layers on the same (channel, note)
	            form a velocity-switched note.  Omit for the legacy
	            single-assignment-per-note behaviour.
	  notes:    The MIDI note(s) the assignment covers.  In addition to the
	            usual int / name / range / list forms, accepts ``zone-tuned``
	            (string sentinel — auto-zone over MIDI 0-127) or
	            ``{mode: zone-tuned, range: [lo, hi]}`` (auto-zone over a
	            restricted keyboard range).  Zone-tuned assignments require
	            ``process: [- repitch: true]`` and own their channel
	            exclusively.  The actual (channel, note) coverage is derived
	            at materialisation time from each matching pitched sample's
	            detected pitch.

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

	if not isinstance(raw, dict):
		raise ValueError(
			f"MIDI map {path}: top-level YAML must be a mapping, got {type(raw).__name__}"
		)

	unknown_keys = set(raw) - _VALID_MAP_KEYS
	if unknown_keys:
		raise ValueError(
			f"MIDI map {path}: unknown top-level key(s) {sorted(unknown_keys)!r} "
			f"(valid: {sorted(_VALID_MAP_KEYS)!r})"
		)

	# Mount the per-project definitions file(s) — name→number vocabularies
	# shared with other tools (e.g. a sequencer).  Everything below (banks,
	# program_channel, and the per-assignment parsers) may reference the
	# mounted names, so this parses first.
	definitions = subsample.definitions.load_definitions(
		raw.get("definitions"), midi_map_dir,
		reserved_prefixes=frozenset(_SYMBOL_NAMESPACES),
		map_label=f"MIDI map {path}",
	)
	note_namespaces: dict[str, typing.Mapping[str, int]] = {
		**_SYMBOL_NAMESPACES, **definitions.note_namespaces(),
	}

	# Parse optional program definitions — switchable sample libraries, each
	# selected live by a MIDI Program Change on `program_channel`.
	bank_definitions = subsample.bank.parse_banks(raw.get("programs"), definitions)
	bank_channel = subsample.definitions.resolve_scalar(
		definitions, "channels",
		raw.get("program_channel", subsample.bank.DEFAULT_BANK_CHANNEL),
		f"MIDI map {path} 'program_channel'",
	)

	# 0 = respond on any channel; 17+ (or negatives) would silently never
	# match a Program Change message, so reject at load.  (Definitions names
	# are 1-16 by construction; 0 stays literal-only.)
	if not (0 <= bank_channel <= 16):
		raise ValueError(
			f"MIDI map {path}: program_channel must be 1-16, or 0 for any "
			f"channel (got {bank_channel})"
		)
	raw_default_bank = raw.get("default_program")
	default_bank: typing.Optional[int] = (
		subsample.definitions.resolve_scalar(
			definitions, "programs", raw_default_bank,
			f"MIDI map {path} 'default_program'",
		)
		if raw_default_bank is not None else None
	)

	# Validate default_program against the declared programs — an unknown number
	# would otherwise silently start on the first bank, and setting it with no
	# `programs:` block at all would be silently ignored.
	if default_bank is not None:
		if not bank_definitions:
			raise ValueError(
				f"MIDI map {path}: 'default_program' is set but there are no "
				f"'programs:' — remove default_program or add a programs block."
			)
		declared = {d.program for d in bank_definitions}
		if default_bank not in declared:
			raise ValueError(
				f"MIDI map {path}: default_program {default_bank} is not one of the "
				f"declared program numbers {sorted(declared)}."
			)

	# Top-level `assignments:` is required for the no-programs case and
	# whenever any program uses the `directory:` shorthand (those programs
	# reuse the top-level assignments against their swapped pool).  When
	# every program is a `map:` preset, each supplies its own assignments,
	# so the top-level block is optional and a missing one is not a warning.
	directory_programs = [d.name for d in bank_definitions if d.directory is not None]
	needs_top_assignments = (not bank_definitions) or bool(directory_programs)

	if "assignments" not in raw:
		if needs_top_assignments and directory_programs:
			raise ValueError(
				f"MIDI map {path}: 'assignments:' is required because program(s) "
				f"{directory_programs!r} use 'directory:' (they reuse the top-level "
				f"assignments) — add an 'assignments:' block or give those programs "
				f"their own 'map:'"
			)
		if needs_top_assignments:
			_log.warning("MIDI map %s has no assignments — no notes will be mapped", path)
		return MidiMapResult(
			note_map={},
			bank_definitions=bank_definitions,
			bank_channel=bank_channel,
			default_bank=default_bank,
		)

	# Resolve template inheritance before parsing: each assignment that names a
	# template via ``template:`` is merged over it, so the loop below sees fully
	# materialised assignment dicts and needs no inheritance awareness.
	raw["assignments"] = _resolve_assignment_inheritance(
		raw["assignments"], raw.get("templates"),
	)

	reference_set = {name.upper() for name in reference_names}
	note_map: NoteMap = {}
	zone_templates: list[ZoneTemplate] = []
	manual_channels: set[int] = set()

	for assignment_index, assignment_raw in enumerate(raw["assignments"], start=1):
		name = assignment_raw.get("name", "<unnamed>")

		if isinstance(assignment_raw, dict):
			# `one_shot` is a removed alias with its own migration error in
			# _parse_mode — exclude it here so that clearer message fires instead
			# of the generic unknown-key one.
			unknown_keys = set(assignment_raw) - _VALID_ASSIGNMENT_KEYS - {"one_shot"}
			if unknown_keys:
				raise ValueError(
					f"MIDI map assignment {name!r}: unknown key(s) "
					f"{sorted(unknown_keys)}.  Valid keys: "
					f"{sorted(_VALID_ASSIGNMENT_KEYS)}."
				)

		# Channel: user-facing 1-16 → mido 0-indexed.
		channel_raw = assignment_raw.get("channel")

		if channel_raw is None:
			raise ValueError(f"MIDI map assignment {name!r}: missing 'channel'")

		try:
			mido_channel = subsample.definitions.resolve_scalar(
				definitions, "channels", channel_raw, "'channel'",
			) - 1
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"MIDI map assignment {name!r} (#{assignment_index}): "
				f"invalid 'channel' value {channel_raw!r} — {exc}"
			) from exc

		notes_raw = assignment_raw.get("notes")

		if notes_raw is None:
			raise ValueError(f"MIDI map assignment {name!r}: missing 'notes'")

		# Detect zone-tuned mode (string sentinel or dict form).  When detected,
		# the assignment is routed to the ZoneTemplate path instead of producing
		# concrete NoteMap entries; materialisation happens at runtime against
		# the active library.
		zone_range = _parse_zone_notes(notes_raw, name, note_namespaces)

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
		process = subsample.query.parse_process(assignment_raw.get("process"), name, definitions)

		# A where.quantized_beats filter measures the beat-length of the
		# assignment's quantize OUTPUT, so without a stretch_quantize/pad_quantize
		# step the resolver always returns None and the note is permanently silent.
		# Warn at load rather than leave a dead note with no diagnostic.
		if not (process.has_stretch_quantize() or process.has_pad_quantize()):
			if any(not spec.where.quantized_beats.is_empty() for spec in select_specs):
				_log.warning(
					"MIDI map assignment %r: a where.quantized_beats filter has no "
					"stretch_quantize/pad_quantize step to measure against — the note "
					"will never match a sample.  Add a quantize step or drop the filter.",
					name,
				)

		mode, loop_override = _parse_mode(assignment_raw, name, process)

		# release: shapes the note-off fade, and only a voice that RECEIVES a
		# note-off ever releases.  A one_shot voice plays to its natural end and
		# ignores note-off, so a release declared on mode: one_shot can never fire
		# — warn and drop it rather than carry dead state.
		release = _parse_release(assignment_raw.get("release"), name, definitions)

		if release is not None and mode == "one_shot":
			_log.warning(
				"MIDI map assignment %r: 'release' is ignored because mode is "
				"one_shot — a play-to-end voice never receives note-off.  Use "
				"mode: gated or mode: loop to use release.",
				name,
			)
			release = None

		# A loop voice sustains for as long as the key is held, so on note-off it
		# needs a release to fade past the loop; default it to the adaptive tail
		# (shape derived from the sample) when the map does not set one.
		if mode == "loop" and release is None:
			release = subsample.query.ReleaseSpec(time=None, curve="cosine")

		gain_raw = assignment_raw.get("gain", 0.0)

		# Reject bool first (YAML `yes`/`no` → True/False, and float(True) == 1.0
		# would silently mean +1 dB), then require a finite number — a NaN/inf
		# gain would otherwise poison the whole voice render silently.
		if isinstance(gain_raw, bool):
			raise ValueError(
				f"MIDI map assignment {name!r} (#{assignment_index}): "
				f"'gain' must be a number in dB, not a boolean ({gain_raw!r})"
			)

		try:
			gain_db = float(gain_raw)
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"MIDI map assignment {name!r} (#{assignment_index}): "
				f"invalid 'gain' value {gain_raw!r} — {exc}"
			) from exc

		if not math.isfinite(gain_db):
			raise ValueError(
				f"MIDI map assignment {name!r} (#{assignment_index}): "
				f"'gain' must be a finite number ({gain_raw!r})"
			)

		pan_weights    = _parse_pan_weights(assignment_raw.get("pan"), name)
		output_routing = _parse_output_routing(assignment_raw.get("output"), name, pan_weights)
		extract        = _parse_extract(assignment_raw.get("extract"), name)

		velocity_trigger, velocity_rescale_to = _parse_velocity(
			assignment_raw.get("velocity"), name,
		)

		stack = bool(assignment_raw.get("stack", False))

		silenced_by = _parse_silenced_by(assignment_raw.get("silenced_by"), name, note_namespaces)

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

		# Zone-tuned path: emit a ZoneTemplate; concrete NoteMap entries
		# are derived at materialisation time.
		if zone_range is not None:

			if not process.has_repitch():
				raise ValueError(
					f"MIDI map assignment {name!r}: zone-tuned requires "
					f"``process: [- repitch: true]`` so each derived sample "
					f"is pitch-shifted to the note being played.  Add a "
					f"repitch step to the process block."
				)

			# Zone-tuned owns its channel exclusively and derives exactly one
			# sample per note, so stacking can never take effect here — reject
			# rather than silently ignore the flag.
			if stack:
				raise ValueError(
					f"MIDI map assignment {name!r}: ``stack`` is not supported "
					f"on zone-tuned assignments — a zone-tuned channel maps one "
					f"sample per note, so there is nothing to stack with.  Remove "
					f"``stack`` or use manual ``notes:`` assignments to stack."
				)

			# choke is a percussion feature (each note is one physical
			# instrument); a zone-tuned channel is a melodic keyboard layout, so
			# a choke would make the whole range monophonic.  Reject in v1 rather
			# than thread choke identity through the zone-derived assignments.
			if silenced_by is not None:
				raise ValueError(
					f"MIDI map assignment {name!r}: ``silenced_by`` (choke) is not "
					f"supported on zone-tuned assignments — it would make the whole "
					f"keyboard range monophonic.  Use manual ``notes:`` assignments "
					f"for choke groups."
				)

			# A fallback chain on a zone-tuned assignment is silently truncated to
			# its first spec (the materialiser needs a STABLE candidate list — a
			# fallback firing mid-pattern would reshuffle every note's assigned
			# sample).  Warn so the dropped specs aren't a silent surprise.
			if len(select_specs) > 1:
				_log.warning(
					"MIDI map assignment %r: zone-tuned assignments use only the "
					"first select spec — the %d fallback spec(s) are ignored (a "
					"fallback firing mid-pattern would reshuffle the keyboard layout).",
					name, len(select_specs) - 1,
				)

			zone_templates.append(ZoneTemplate(
				name=name,
				channel=mido_channel,
				keyboard_range=zone_range,
				select=select_specs,
				process=process,
				mode=mode,
				loop=loop_override,
				gain_db=gain_db,
				pan_weights=pan_weights,
				output_routing=output_routing,
				extract=extract,
				segment_mode=segment_mode,
				velocity_trigger=velocity_trigger,
				velocity_rescale_to=velocity_rescale_to,
				stack=stack,
				release=release,
			))
			continue

		# Regular path: parse notes and emit one or more NoteMap entries.
		notes = _parse_note_spec(notes_raw, name, note_namespaces)
		manual_channels.add(mido_channel)

		assignment = subsample.query.Assignment(
			name=name,
			select=select_specs,
			process=process,
			mode=mode,
			loop=loop_override,
			release=release,
			gain_db=gain_db,
			pan_weights=pan_weights,
			output_routing=output_routing,
			extract=extract,
			segment_mode=segment_mode,
			velocity_trigger=velocity_trigger,
			velocity_rescale_to=velocity_rescale_to,
			stack=stack,
			silenced_by=silenced_by,
		)

		# Per-note pick distribution:
		# When process includes repitch, all notes share pick=1 (same sample,
		# pitched per note).  Otherwise, each note gets the next pick position
		# so multi-note assignments distribute across ranked matches.
		# An explicit pick anywhere in the select chain (scalar or range)
		# overrides this default — the FIRST spec that declares one governs
		# the whole chain (one pick applies per trigger regardless of which
		# fallback spec matched, so a pick declared only on a fallback spec
		# must not be silently ignored).
		explicit_pick = False
		chain_pick: typing.Optional[subsample.query.PickSpec] = None

		if isinstance(select_raw, dict):
			explicit_pick = "pick" in select_raw
			if explicit_pick:
				chain_pick = select_specs[0].pick
		elif isinstance(select_raw, list) and select_raw:
			for spec_idx, spec_raw in enumerate(select_raw):
				if isinstance(spec_raw, dict) and "pick" in spec_raw:
					explicit_pick = True
					chain_pick = select_specs[spec_idx].pick
					break

		for note_idx, note in enumerate(notes):

			if explicit_pick or process.has_repitch() or len(notes) == 1:
				pick_spec = chain_pick if chain_pick is not None else select_specs[0].pick
			else:
				rank = note_idx + 1
				pick_spec = subsample.query.PickSpec(rank, rank)

			# Append to the layer list so multiple Assignments may coexist on
			# the same (channel, note) when each declares a distinct
			# velocity_trigger range.  Overlap/gap validation runs once below
			# the loop, after every assignment has been added.
			note_map.setdefault((mido_channel, int(note)), []).append(
				(assignment, pick_spec),
			)

	_validate_velocity_layers(note_map)
	_validate_zone_assignments(zone_templates, manual_channels)
	_validate_choke_targets(note_map)

	_log.info(
		"MIDI map loaded from %s: %d note(s) across %d assignment(s)%s%s",
		path,
		len(note_map),
		len(raw.get("assignments", [])),
		f", {len(bank_definitions)} bank(s)" if bank_definitions else "",
		f", {len(zone_templates)} zone-tuned template(s)" if zone_templates else "",
	)

	return MidiMapResult(
		note_map=note_map,
		bank_definitions=bank_definitions,
		bank_channel=bank_channel,
		default_bank=default_bank,
		zone_templates=tuple(zone_templates),
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
	           (only for non-one-shot voices).  The callback then fades the voice
	           out over release_frames with release_curve (the assignment's
	           configured ``release:``, or the global declick default when unset),
	           unless release_to_end is set, and retires it.
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
	fade_pos:  int  = 0
	"""Frames of release fade already applied.  Lets the release fade span
	multiple audio callbacks so small buffer_frames don't truncate it."""

	release_frames: typing.Optional[int] = None
	"""Configured note-off fade length in output-rate frames (from the
	assignment's ``release:``).  None → use the player's global default declick
	length (_release_fade_frames)."""

	release_curve: int = 0
	"""Fade shape as an int so the audio callback does no string work:
	0 = cosine (raised-cosine declick, the default), 1 = exponential."""

	release_total: int = 0
	"""Effective fade length for THIS release, fixed the first callback the
	voice releases (0 until then).  Capped to the audio remaining at note-off so
	the ramp always reaches 0 before the buffer runs out — otherwise a long
	release on a short remainder would retire the voice mid-ramp (a click)."""

	release_to_end: bool = False
	"""``release: full`` — on note-off, apply NO fade: the voice keeps playing
	its remaining audio to the natural end (a looping voice also stops looping).
	When True the note-off handler clears ``looping`` but does NOT set
	``releasing``, so the normal play-to-end branch carries it out."""

	looping:        bool = False
	"""True while this voice should wrap [loop_start, loop_end) — a held
	mode: loop note.  Cleared on note-off (the cursor then runs monotonically
	past loop_end into the tail).  When True the callback fills the whole buffer
	from the loop region instead of ending at len(audio)."""

	loop_start:     int = 0
	loop_end:       int = 0
	"""Loop bounds in this voice's own (output-rate) frames; the wrap jumps from
	loop_end back to loop_start.  Only meaningful while ``looping``."""

	loop_crossfade: int = 0
	"""Crossfade length in frames blended at the wrap (linear), replicating
	loopfind.bake_loop_body so the seam is seamless without baking the body."""

	loop_xfade_in:  typing.Optional[numpy.ndarray] = None
	"""Pre-computed lead-in segment (the ``loop_crossfade`` frames just before
	loop_start) that the wrap fades in against, so the callback does no slicing
	arithmetic beyond an index.  None when not looping or crossfade is 0."""


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


def _collect_mapped_ccs (
	note_map: NoteMap,
	zone_templates: tuple["ZoneTemplate", ...] = (),
) -> set[int]:

	"""Return the set of CC numbers used by CcBinding params in the note map.

	Walks every velocity layer of every note; a CC bound by any layer is
	considered "mapped" for the whole player so that the relevant
	control_change traffic triggers debounced re-evaluation regardless of
	which layer the user is currently triggering.

	Zone-tuned templates are walked too: their concrete entries are created
	later in _materialize_zones (so they are absent from the manual note_map at
	collection time), but a CcBinding in a zone-tuned ``process`` chain must
	still arm the proactive pre-bake.
	"""

	ccs: set[int] = set()

	def _scan_process (process: subsample.query.ProcessSpec) -> None:
		for step in process.steps:
			for _, value in step.params:
				if isinstance(value, subsample.query.CcBinding):
					ccs.add(value.cc)

	for entries in note_map.values():
		for assignment, _ in entries:
			_scan_process(assignment.process)

	for template in zone_templates:
		_scan_process(template.process)

	# NOTE: a CC-bound release: time is deliberately NOT registered here.
	# _mapped_ccs only arms the debounced VARIANT RE-BAKE; the raw CC value is
	# recorded on every control_change regardless (see _handle_message), and the
	# release time is read live from that state at note-on.  Registering it would
	# trigger pointless re-bakes on every release-knob move — release changes no
	# variant.

	return ccs


def _uses_quantize (
	note_map: NoteMap,
	zone_templates: tuple["ZoneTemplate", ...] = (),
) -> bool:

	"""Return True if any assignment declares a quantize processor.

	Mirrors _collect_mapped_ccs's walk — every velocity layer of every note plus
	the zone templates, whose concrete entries do not exist until
	_materialize_zones runs.

	Gates the MIDI-clock tempo work: with no quantize step anywhere in the map
	the session tempo changes nothing, so the clock is not worth tracking and a
	clock/target_bpm mismatch is not worth warning about.
	"""

	def _quantizes (process: subsample.query.ProcessSpec) -> bool:
		return process.has_stretch_quantize() or process.has_pad_quantize()

	for entries in note_map.values():
		for assignment, _ in entries:
			if _quantizes(assignment.process):
				return True

	for template in zone_templates:
		if _quantizes(template.process):
			return True

	return False


def _uses_beat_filter (
	note_map: NoteMap,
	zone_templates: tuple["ZoneTemplate", ...] = (),
) -> bool:

	"""Return True if any assignment's selection filters by ``duration_beats``.

	Mirrors _uses_quantize's walk — every velocity layer of every note plus the
	zone templates.  duration_beats measures a sample's length in beats at the
	session tempo, so like the quantize processors it makes the tempo
	load-bearing: the clock is worth tracking, a clock/tempo mismatch is worth
	warning about, and a map that uses it needs a tempo to load at all.
	"""

	def _filters_by_beats (select: tuple["subsample.query.SelectSpec", ...]) -> bool:
		return any(not spec.where.duration_beats.is_empty() for spec in select)

	for entries in note_map.values():
		for assignment, _ in entries:
			if _filters_by_beats(assignment.select):
				return True

	for template in zone_templates:
		if _filters_by_beats(template.select):
			return True

	return False


_BEAT_FILTER_NO_TEMPO_MESSAGE = (
	"A MIDI map assignment filters by duration_beats, but no session tempo is "
	"set.  Set tempo.bpm in config.yaml — it is the fallback even under "
	"tempo.source: midi, which still needs a tempo before the first clock arrives."
)


def _validate_beat_filter_tempo (
	note_map: NoteMap,
	zone_templates: tuple["ZoneTemplate", ...],
	bpm: float,
) -> None:

	"""Raise if the map filters by ``duration_beats`` but no tempo is set.

	duration_beats measures sample length in beats, so it cannot resolve without
	a session tempo — fail loudly rather than silently emptying every
	beat-filtered pool.  Called at startup (in cli, before the player is built,
	so the ValueError surfaces cleanly alongside the other map checks).
	"""

	if bpm <= 0.0 and _uses_beat_filter(note_map, zone_templates):
		raise ValueError(_BEAT_FILTER_NO_TEMPO_MESSAGE)


class MidiPlayer:

	"""Listens for MIDI messages and plays back instrument samples polyphonically.

	Designed to run on its own thread. Call run() as the thread target;
	it blocks until shutdown_event is set, then closes the MIDI port and
	PyAudio stream and returns cleanly.

	Note routing is loaded from a YAML file by the caller via load_midi_map()
	and passed as the `midi_map` parameter.  The map keys notes by
	(mido_channel, midi_note) and stores a list of (Assignment, PickSpec)
	velocity layers per note.  Default (no ``velocity:`` field) is a single
	layer covering the full 0-127 range; multiple non-overlapping layers
	let velocity-switched libraries (soft/hard hi-hat, piano dynamics) and
	"more triggers per pad" workflows coexist on one MIDI note.

	At trigger time, the player first picks the velocity layer whose
	trigger range covers msg.velocity, then evaluates the layer's select
	chain against the active instrument library to find the best-matching
	sample.

	Mixing: triggered notes are added as _Voice objects to a shared list.
	A PyAudio callback stream reads from all active voices simultaneously,
	sums them into one output buffer, applies the safety limiter, and returns
	the mixed audio.  This runs independently of the rtmidi callback thread
	(which dispatches MIDI messages and appends voices), so notes overlap
	naturally.
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
		tempo_source: str = "config",
		output_channels: typing.Optional[int] = None,
		ambisonic_config: typing.Optional[subsample.config.AmbisonicConfig] = None,
		buffer_frames: typing.Optional[int] = None,
		zone_templates: tuple[ZoneTemplate, ...] = (),
	) -> None:

		self._device_name        = device_name
		self._shutdown_event     = shutdown_event
		self._instrument_library = instrument_library
		self._similarity_matrix  = similarity_matrix
		self._target_bpm         = target_bpm
		self._tempo_source       = tempo_source

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
		# threshold_db = 0.0 disables the soft-clip stage entirely (the hard
		# clip to [-1, 1] in the callback remains): at 0 dB the knee would be
		# negative (ceiling sits below threshold) and the curve would expand
		# rather than compress.
		self._limiter_enabled   = limiter_threshold_db < 0.0
		self._limiter_threshold = 10.0 ** (limiter_threshold_db / 20.0)
		self._limiter_ceiling   = 10.0 ** (limiter_ceiling_db / 20.0)
		self._limiter_knee      = self._limiter_ceiling - self._limiter_threshold

		# Latent guard for direct construction (config validation already
		# enforces this for the YAML surface): an enabled limiter with a
		# non-positive knee would divide by zero or EXPAND in the callback.
		if self._limiter_enabled and self._limiter_knee <= 0.0:
			raise ValueError(
				f"limiter_ceiling_db ({limiter_ceiling_db}) must be greater "
				f"than limiter_threshold_db ({limiter_threshold_db})"
			)

		# Clipping detection: timestamp of the last warning so we can throttle
		# to at most one log message every 5 seconds during dense passages.
		self._last_clip_warn: float = 0.0

		# Output xrun (buffer underflow) detection.  PortAudio reports an
		# underflow via status_flags when the callback can't supply samples in
		# time — audible as a click/dropout.  We count them and log at most
		# once every 5 seconds so the user knows when buffer_frames has been
		# tuned too low for the machine to sustain.
		self._xrun_count:     int   = 0
		self._last_xrun_warn: float = 0.0

		# Audio-callback failure guard: timestamp of the last ERROR so a
		# persistent fault logs once per 5 s instead of per buffer.
		self._last_callback_error_warn: float = 0.0

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

		# Assignment ids already warned that their sample has no usable loop
		# (mode: loop fail-musical) — so the note-on path warns once, not per hit.
		self._loop_unavailable_warned: set[int] = set()

		# (note, channel) pairs already warned that a resolved loop COLLAPSED when
		# clamped to a shorter-than-expected rendered sample (falls back to gated).
		self._loop_collapsed_warned: set[tuple[int, int]] = set()

		# Number of output channels.  Determines the shape of the mix buffer
		# and must match the pa.open(channels=...) call in run().  Defaults
		# to 2 (stereo); set via player.audio.channels in config for
		# multi-channel interfaces.
		self._output_channels: int = output_channels if output_channels is not None else 2

		# Optional PortAudio output buffer size in frames.  When None, the
		# device default is used.  Validated as a power of two in [32, 4096]
		# at config load.  See PlayerAudioConfig.buffer_frames docstring.
		self._buffer_frames: typing.Optional[int] = buffer_frames

		# Note routing map: (mido_channel, midi_note) → list of velocity layers.
		# The "base" map holds the manual entries declared in YAML; the working
		# self._note_map is materialised from base + derived zone-tuned entries
		# at every (re-)materialisation.  At startup we begin with base as the
		# working map; _materialize_zones() below rebuilds it including any
		# zone-tuned entries the YAML declared.
		self._base_note_map:  NoteMap                       = midi_map
		self._zone_templates: tuple[ZoneTemplate, ...]      = zone_templates
		self._note_map:       NoteMap                       = dict(midi_map)

		# Choke table: (channel, killer_note) → victim (channel, note) keys.
		# Derived purely from the base (manual) note map — zone-tuned entries
		# reject silenced_by — so it is rebuilt only when the base map changes
		# (here and in _apply_rule_set_locked), never on zone re-materialisation.
		self._choke_map: dict[tuple[int, int], frozenset[tuple[int, int]]] = (
			_build_choke_map(self._base_note_map)
		)

		# Most recently played variant per layer.  Used as a fallback while a
		# new variant is still processing: the old one plays instead of the
		# unprocessed base — giving smooth transitions for gradual CC/BPM
		# parameter changes, where the Assignment objects (and so the keys)
		# survive the re-evaluation.  Keyed by (channel, note, id(Assignment))
		# so each layer on the same MIDI note keeps independent fallback —
		# including stacked members that share a velocity range, which a
		# velocity-based key could not tell apart.  Identity keys mean a rule
		# swap or zone re-materialisation retires entries rather than carrying
		# them across; _prune_stale_layer_state() sweeps those on every
		# update_assignments so they can't pin evicted variant audio.
		self._last_played: dict[tuple[int, int, int], subsample.transform.TransformResult] = {}

		# Pre-computed sample selection.  An ordinary select (filters + an
		# order such as age / similarity / duration) resolves to the same
		# ranked candidate list on every note-on until the active library
		# changes, so the list is pre-computed off the trigger thread and the
		# trigger only draws a pick index — no per-note query, sort, or
		# filesystem access.  Keyed by id(Assignment); rebuilt by
		# _rebuild_candidate_cache() on every re-evaluation.  Assignments whose
		# select depends on asynchronously-baked variant state (quantized_beats
		# / beat_match) are deliberately absent and resolved live instead.
		self._candidate_cache: dict[int, _Candidates] = {}

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
		# Includes zone-tuned template process chains, whose concrete entries
		# don't exist in midi_map yet.
		self._mapped_ccs: set[int] = _collect_mapped_ccs(midi_map, self._zone_templates)

		# MIDI clock tempo tracking.  Armed only when the map actually quantizes
		# (nothing else consumes the session tempo) or when the user asked to
		# follow the clock — otherwise every clock pulse is dropped on one
		# attribute test.
		#
		# The tracker is driven from the rtmidi callback thread and publishes the
		# OBSERVED tempo into _clock_bpm.  That is deliberately NOT the tempo
		# specs are built from: _update_assignments_locked adopts it into
		# _target_bpm, so the trigger path and the variant pre-compute always
		# read one stable value.  A live-read BPM would let a tempo change land
		# between pre-bake and note-on and turn every pre-baked variant into a
		# cache miss (variant_cache_key hashes target_bpm via the step repr).
		self._map_quantizes: bool = _uses_quantize(midi_map, self._zone_templates)
		self._map_beat_filters: bool = _uses_beat_filter(midi_map, self._zone_templates)
		self._clock_tracker: typing.Optional[_MidiClockTracker] = (
			_MidiClockTracker()
			if self._map_quantizes or self._map_beat_filters or self._tempo_source == "midi"
			else None
		)
		self._clock_bpm: typing.Optional[float] = None
		self._clock_warned_bpm: typing.Optional[float] = None

		# Immutable snapshot of the top-level (global) rules.  `directory:`
		# programs reuse these; a `map:` preset switch overwrites the active
		# _base_note_map, so switching back to a `directory:` program must
		# restore THESE, not whatever preset was last active.  Never mutated
		# after construction.
		self._top_level_note_map:       NoteMap                  = midi_map
		self._top_level_zone_templates: tuple[ZoneTemplate, ...] = zone_templates
		self._top_level_mapped_ccs:     set[int]                 = set(self._mapped_ccs)

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
		# channel_format, extract) → matrix.  channel_format is included so
		# ambisonic decode matrices do not collide with raw-PCM routing for
		# the same 4-channel input shape; extract so per-(kind, channel_index)
		# variants get distinct entries.  Lazily populated by
		# _get_mix_matrix(); cleared on MIDI map reload.
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
		# Keyed by (channel, note, id(Assignment)) so each layer on the same
		# MIDI note advances its own round_robin counter — including stacked
		# members that share a velocity range.
		self._segment_counters: dict[tuple[int, int, int], int] = {}

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
		# _rules_lock is outermost, then _state_lock.  Never acquire
		# _state_lock while holding _voices_lock, _mix_matrix_lock, or
		# _cc_debounce_lock; never acquire _rules_lock while holding any
		# other player lock.
		self._state_lock: threading.Lock = threading.Lock()

		# Serialises rule-set re-evaluation: update_assignments() and
		# _apply_rule_set() run under this lock so a watcher / CC-debounce /
		# on-complete re-evaluation can never interleave with a hot-reload or
		# program-change swap's install→validate→rollback window (which could
		# rebuild _note_map from half-installed rules AFTER the rollback,
		# leaving the player permanently on rolled-back rules).  RLock:
		# _apply_rule_set calls update_assignments() on the same thread.
		self._rules_lock: threading.RLock = threading.RLock()

		# Materialise zone-tuned templates against the active library so
		# the startup log shows the derived per-sample zones rather than
		# an empty NoteMap.  Subsequent re-materialisation happens at the
		# top of update_assignments() — picked up by every re-evaluation
		# path (reload, _integrate_sample, bank switch).
		if self._zone_templates:
			self._materialize_zones()

		# Pre-compute the selection cache so the very first note-on already
		# takes the fast indexed-pick path.  cli also calls
		# update_pitched_assignments() right after construction (which rebuilds
		# this), but building it here keeps the player correct for callers that
		# don't — and against an empty cache the trigger path simply falls back
		# to a live query, so this is an optimisation, never a correctness gate.
		self._rebuild_candidate_cache()

		# Group consecutive notes that share the same Assignment into ranges
		# so that a 128-note pitched assignment becomes a single log line.
		# With velocity layering the inner loop walks every layer; because
		# each velocity layer is a distinct Assignment object, the ``is asgn``
		# identity check naturally separates layers (no extra velocity-range
		# comparison needed in the group condition).  Each group tracks the
		# set of PickSpecs across its notes — a single shared PickSpec means
		# an explicit pick (or single-note assignment); multiple distinct
		# PickSpecs mean auto-distribution kicked in.
		groups: list[tuple[int, int, int, subsample.query.Assignment, set[subsample.query.PickSpec]]] = []

		flat_entries: list[tuple[int, int, subsample.query.Assignment, subsample.query.PickSpec]] = []
		for (ch, note), entries in self._note_map.items():
			for asgn, pick_spec in entries:
				flat_entries.append((ch, note, asgn, pick_spec))

		# Sort by (channel, note, velocity_trigger) so layers of the same
		# note print together in ascending velocity order.
		# Sort velocity range BEFORE note so each layer's consecutive-note run
		# stays adjacent — sorting by note first interleaves the layers of
		# every note and the range-grouping below can never extend a run
		# (a 50-note velocity-split keyboard would print 100 lines).
		flat_entries.sort(key=lambda e: (e[0], e[2].velocity_trigger, e[1]))

		for ch, note, asgn, pick_spec in flat_entries:
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

			# Velocity layer description: omitted when this assignment uses
			# the default full-range trigger (no layering), shown otherwise.
			vel_lo, vel_hi = asgn.velocity_trigger
			if (vel_lo, vel_hi) != (0, 127):
				line += f" vel [{vel_lo},{vel_hi}]"
				if asgn.velocity_rescale_to is not None:
					r_lo, r_hi = asgn.velocity_rescale_to
					if (r_lo, r_hi) == (0, 127):
						line += " rescale"
					else:
						line += f" rescale [{r_lo},{r_hi}]"

			# Confirm to the user that an intentional overlap loaded as a stack
			# rather than being rejected — the same note will list once per
			# stacked sample.
			if asgn.stack:
				line += " (stacked)"

			if len(pick_specs) == 1:
				line += _format_pick_suffix(next(iter(pick_specs)))
			else:
				line += " pick distributed"

			if asgn.process.has_repitch():
				line += " pitched"

			if asgn.process.has_stretch_quantize():
				line += " beat-quantized"

			if asgn.process.has_pad_quantize():
				line += " pad-quantized"

			if asgn.mode != "one_shot":
				line += f"  {asgn.mode}"

			if asgn.pan_weights is not None:
				line += f"  pan=[{', '.join(f'{g:.0f}' for g in asgn.pan_weights)}]"
			lines.append(line)

		total_notes  = len(self._note_map)
		total_layers = sum(len(entries) for entries in self._note_map.values())
		summary      = (
			f"{total_notes} note(s), {total_layers} layer(s)"
			if total_layers > total_notes
			else f"{total_notes} note(s)"
		)

		_log.info(
			"MIDI note map: %s loaded\n  %s",
			summary,
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

		# Resources are initialised to None upfront so the finally block can
		# clean up partial state if any open step raises (e.g. PortAudio
		# rejecting the requested format, or mido failing to bind the MIDI
		# device).  Without this any half-open stream or MIDI port would leak
		# its OS-level handle, blocking subsequent attempts to use the device
		# until the process restarted.
		pa                                            = subsample.audio.create_pyaudio()
		stream:     typing.Optional[typing.Any]       = None
		port:       typing.Optional[mido.ports.BaseInput] = None
		port_label:                          str      = ""

		try:
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

			# Detect the device's real output-channel capability and check the
			# configured count against it.  `player.audio.channels` defaults to
			# stereo when unset (resolved in __init__); the device max is NOT a
			# default — opening every physical output for a stereo set would
			# waste a wider mix per callback and isn't reproducible across rigs.
			# We use the detected max only to log the capability and to turn an
			# over-large request into a clear startup error rather than the
			# cryptic PortAudio "Invalid number of channels" failure.
			device_max_out = subsample.audio.get_output_device_channels(pa, output_device_index)
			device_name    = str(pa.get_device_info_by_index(output_device_index)["name"])
			_log.info(
				"Output device %r supports up to %d channel(s); using %d",
				device_name, device_max_out, self._output_channels,
			)
			if self._output_channels > device_max_out:
				raise ValueError(
					f"player.audio.channels = {self._output_channels} exceeds the output "
					f"device {device_name!r} capability ({device_max_out} channel(s)) — "
					f"lower player.audio.channels, or route within {device_max_out} channels."
				)

			# Validate output routing indices against the resolved device
			# channel count.  The device count is only known here (after device
			# selection), so this fix-up runs at startup.  It must be applied to
			# every rule SOURCE — the top-level map and each `map:` preset —
			# because those are what re-materialisation and Program Change
			# rebuild _note_map from; otherwise the next update_assignments()
			# would revert the strip and a note that worked at startup would
			# start raising in route_to_device (dropped by _safe_handle_message)
			# after the first capture / knob / program switch.  Each source is
			# stripped exactly once (single warning), then the active base is
			# re-pointed at the stripped source for the currently active program.
			# Hold the rule lock across the whole strip → re-point → rematerialise
			# → rebuild sequence: the instrument/midi-map watchers and capture
			# integration are already running by now and can call
			# update_assignments() under _rules_lock, so an unlocked mutation here
			# could interleave and install a torn rule set.  _rules_lock is
			# reentrant, so the _materialize_zones/_rebuild_candidate_cache calls
			# (which may take it again) are fine.
			with self._rules_lock:
				self._top_level_note_map, self._top_level_zone_templates = self._strip_oob_routing_rules(
					self._top_level_note_map, self._top_level_zone_templates,
				)

				if self._bank_manager is not None:
					for _bank in self._bank_manager.all_banks():
						if _bank.note_map is not None:
							_bank.note_map, _bank.zone_templates = self._strip_oob_routing_rules(
								_bank.note_map, _bank.zone_templates or (),
							)

				active_bank = self._bank_manager.active_bank if self._bank_manager is not None else None
				if active_bank is not None and active_bank.note_map is not None:
					self._base_note_map  = active_bank.note_map
					self._zone_templates = active_bank.zone_templates or ()
				else:
					self._base_note_map  = self._top_level_note_map
					self._zone_templates = self._top_level_zone_templates

				# Rebuild the working map and selection cache from the fixed sources.
				self._materialize_zones()
				self._rebuild_candidate_cache()

			# Callback mode: PortAudio pulls audio from _audio_callback on its
			# own high-priority thread.  The rtmidi callback thread runs
			# independently and adds voices.  When player.audio.buffer_frames
			# is set, pass it as frames_per_buffer to tighten output-side
			# latency.  If the device cannot honour the requested size at open
			# time, fall back to the device default with a clear ERROR log so
			# the user knows their tuning value was ignored.
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

					# Clear the stored value too: the stream now runs at the
					# device default, and the xrun warning reports this field —
					# keeping the rejected number would send the user tuning a
					# knob that isn't in effect.
					self._buffer_frames = None

					stream = pa.open(**open_kwargs)
				else:
					raise

			# Report the latency PortAudio actually negotiated for the stream —
			# the floor between a queued voice and sound at the DAC.  This is
			# what a player feels as "delay" relative to a hardware instrument,
			# and is the number to watch when tuning player.audio.buffer_frames.
			# It is usually several times the buffer period (the ALSA backend
			# runs multiple periods), so it stays visible even when the buffer
			# size looks small.
			try:
				_log.info(
					"PortAudio output latency: %.1f ms",
					stream.get_output_latency() * 1000.0,
				)
			except Exception as exc:
				_log.debug("Could not query output latency: %s", exc)

			# Open the MIDI input port in callback mode.  rtmidi delivers each
			# message to ``_safe_handle_message`` on its own dedicated thread
			# the moment it arrives — no polling loop, no 10 ms jitter floor.
			# The kwarg form installs the callback as the last step of port
			# creation, closing the open-then-assign gap that property-style
			# assignment would expose.  Virtual ports are server destinations
			# external apps connect to; they require ``virtual=True`` and do
			# not appear in ``get_input_names()``.
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

			# Block until shutdown is signalled.  All MIDI work happens on
			# rtmidi's callback thread; this thread is now purely a lifecycle
			# coordinator.
			self._shutdown_event.wait()

		finally:
			# Tear down whichever resources were successfully opened.  None
			# checks guard the partial-open paths: a mido failure leaves
			# ``port is None`` but ``stream`` may already be open; an early
			# pa.open failure leaves both None but ``pa`` itself still needs
			# terminate.  port.close() internally clears the callback under
			# rtmidi's lock (waits for any in-flight callback to return),
			# then closes the port.
			if port is not None:
				port.close()

			# Cancel the CC debounce AFTER the port is closed: a CC arriving
			# between a cancel and the close would re-arm a fresh timer that
			# outlives shutdown (and runs update_assignments against a player
			# that is tearing down).
			with self._cc_debounce_lock:
				if self._cc_debounce_timer is not None:
					self._cc_debounce_timer.cancel()

			if stream is not None:
				stream.stop_stream()
				stream.close()

			pa.terminate()

			if port_label:
				_log.info("MIDI player closed port: %s", port_label)

	def _audio_callback (
		self,
		in_data: typing.Optional[bytes],
		frame_count: int,
		time_info: typing.Any,
		status_flags: int,
	) -> tuple[bytes, int]:

		"""Exception-guarded wrapper around the real mix callback.

		An exception escaping a PortAudio callback aborts the stream
		permanently: PortAudio prints to stderr and stops calling back, so
		the player would sit MIDI-alive but audio-dead with nothing in our
		own log.  Mirror of ``_safe_handle_message`` on the MIDI side —
		log the failure (throttled; we are on the audio thread) and return
		one buffer of silence so the stream survives.
		"""

		try:
			return self._audio_callback_impl(in_data, frame_count, time_info, status_flags)
		except Exception:
			now = time.monotonic()

			if now - self._last_callback_error_warn >= 5.0:
				self._last_callback_error_warn = now
				_log.error(
					"Audio callback failed — emitting silence for this buffer",
					exc_info=True,
				)

			silence = b"\x00" * (frame_count * self._output_channels * (self._output_bit_depth // 8))

			return (silence, pyaudio.paContinue)

	def _audio_callback_impl (
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

		Releasing voices (note_off received): the voice fades out over the
		configured release (voice.release_frames / release_curve — the
		assignment's ``release:``, or the global declick default when unset),
		capped to the audio remaining, then retires — unless release_to_end is
		set, in which case it rings out to its natural end with no fade.  The
		default declick prevents an audible click on hard cutoff for tonal samples.
		"""

		# Output underflow (xrun): PortAudio sets this flag when the previous
		# callback didn't return samples in time, so the device played a gap —
		# audible as a click/dropout.  Count every one; log at most every 5 s
		# (we're on the audio thread) so a too-low buffer_frames is visible
		# while tuning latency, without flooding the log.
		if status_flags & pyaudio.paOutputUnderflow:
			self._xrun_count += 1
			now = time.monotonic()

			if now - self._last_xrun_warn >= 5.0:
				self._last_xrun_warn = now
				_log.warning(
					"Audio xrun: %d output underflow(s) — buffer_frames=%s is too low for "
					"this machine to sustain; raise it if you hear clicks.",
					self._xrun_count,
					self._buffer_frames if self._buffer_frames is not None else "device-default",
				)

		output = numpy.zeros((frame_count, self._output_channels), dtype=numpy.float32)

		with self._voices_lock:
			active: list[_Voice] = []

			for voice in self._voices:
				remaining = len(voice.audio) - voice.position

				if voice.releasing:
					# Note-off fade, spread across however many callbacks it takes
					# — voice.fade_pos tracks the progress.  A previous single-
					# callback fade collapsed the whole ramp into one buffer, so at
					# small buffer_frames the note-off cut off abruptly (e.g.
					# ~1.5 ms at 64 frames).
					#
					# Length: the assignment's configured release (voice.release_
					# frames) or the global default declick when unset.  Fixed once
					# (fade_pos == 0) and CAPPED to the audio remaining at note-off,
					# so the ramp always reaches 0 before the buffer runs out — a
					# long release on a short remainder would otherwise retire the
					# voice mid-ramp at high amplitude (an audible click).
					if voice.fade_pos == 0:
						base_frames = (
							voice.release_frames
							if voice.release_frames is not None
							else self._release_fade_frames
						)
						voice.release_total = max(1, min(base_frames, remaining))

					fade_total = voice.release_total
					n = min(frame_count, remaining, fade_total - voice.fade_pos)

					if n > 0:
						idx = numpy.arange(voice.fade_pos, voice.fade_pos + n, dtype=numpy.float32)

						if voice.release_curve == 1:
							# Exponential: normalised so it reaches EXACTLY 0 at the
							# end (a raw exp(-k*x) would leave an audible residual).
							x    = idx / fade_total
							ek   = math.exp(-_RELEASE_EXP_K)
							ramp = ((numpy.exp(-_RELEASE_EXP_K * x) - ek) / (1.0 - ek)).astype(numpy.float32)
						else:
							# Cosine: the raised-cosine declick.  Final value is
							# cos(π(N-1)/N), not exactly 0 — a residual ~1.3e-5 at
							# 441 frames, far below audibility.  Not an off-by-one.
							ramp = ((1.0 + numpy.cos(numpy.pi * idx / fade_total)) / 2.0).astype(numpy.float32)

						output[:n] += voice.audio[voice.position : voice.position + n] * ramp[:, numpy.newaxis]
						voice.position += n
						voice.fade_pos += n

					# Keep fading on subsequent callbacks until the ramp completes
					# (or the audio runs out); otherwise the voice is retired.
					if voice.fade_pos < fade_total and voice.position < len(voice.audio):
						active.append(voice)

				elif voice.looping:
					# Held mode: loop voice.  Play forward, wrapping loop_end→loop_start,
					# and crossfade the last loop_crossfade frames of each lap into the
					# lead-in before loop_start (replicating loopfind.bake_loop_body so
					# the wrap is seamless).  The buffer is the ORIGINAL audio, so the
					# attack plays on the first lap and — on note-off (looping cleared)
					# — the normal/release branch plays straight past loop_end into the
					# real tail with no baking.  (A note-off landing mid-crossfade steps
					# from a blended sample to the raw tail; the release fade masks it,
					# but release: full has no fade — a rare, faint edge.)  Fills the
					# WHOLE buffer: a held loop never runs short.
					xf       = voice.loop_crossfade
					xf_start = voice.loop_end - xf     # first frame of the wrap crossfade
					filled   = 0

					while filled < frame_count:
						n     = min(frame_count - filled, voice.loop_end - voice.position)
						chunk = voice.audio[voice.position : voice.position + n]

						if xf > 0 and voice.loop_xfade_in is not None and voice.position + n > xf_start:
							# Split the chunk at the crossfade window: copy the part
							# before it straight, blend the part inside it (outgoing
							# approach-to-loop_end fading into the pre-loop_start lead-in).
							head = max(0, xf_start - voice.position)
							if head > 0:
								output[filled : filled + head] += chunk[:head]
							k    = (voice.position + head) - xf_start
							m    = n - head
							# k + m <= xf always (n <= loop_end - position bounds it), so
							# loop_xfade_in[k : k + m] never overruns — not an off-by-one.
							ramp = (numpy.arange(k, k + m, dtype=numpy.float32) / numpy.float32(xf))[:, numpy.newaxis]
							output[filled + head : filled + n] += (
								chunk[head:] * (1.0 - ramp)
								+ voice.loop_xfade_in[k : k + m] * ramp
							)
						else:
							output[filled : filled + n] += chunk

						voice.position += n
						filled         += n

						if voice.position >= voice.loop_end:
							voice.position = voice.loop_start

					active.append(voice)   # a held loop is never retired here

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
		# Skipped entirely when disabled (limiter_threshold_db: 0.0).
		if self._limiter_enabled:
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
		# Throttled to at most one warning every 5 seconds.  With the limiter
		# disabled the hard clip legitimately reaches full scale, so the
		# ceiling diagnostic doesn't apply.
		peak_abs = float(numpy.max(numpy.abs(mixed)))
		if self._limiter_enabled and peak_abs > self._limiter_ceiling:
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

	def _sync_clock_tracker (self) -> None:

		"""Create or drop the MIDI clock tracker to match the current rules.

		Called after every rule swap: a map that just gained a quantize step
		needs the clock tracked, one that lost its last quantize step does not
		(and should stop paying for it on the MIDI callback thread).

		An existing tracker is kept as-is rather than rebuilt, so a rule swap
		never discards a tempo already measured and force a re-measure.  The
		observed _clock_bpm is left alone either way — it is sticky, so a map
		that later regains a quantize step can adopt the last known tempo
		immediately instead of waiting for the tracker to re-measure.
		"""

		needed = self._map_quantizes or self._map_beat_filters or self._tempo_source == "midi"

		if needed:
			if self._clock_tracker is None:
				self._clock_tracker = _MidiClockTracker()
		else:
			self._clock_tracker = None

	def _handle_clock (self) -> None:

		"""Fold one MIDI clock pulse into the tempo tracker.

		Runs on the rtmidi callback thread at 24 pulses per beat, so the common
		path stays trivial: feed the tracker and return.  The tracker returns a
		value only when a *changed* tempo has been measured and held (see
		_MidiClockTracker) — a real tempo change, not a pulse.

		Lock discipline: takes _state_lock alone, the same pattern as the CC
		state write, and this thread is its single writer.  It must never take
		_rules_lock — that is held across query and variant work, so blocking
		the MIDI thread on it would stall dispatch for a whole reload.
		"""

		tracker = self._clock_tracker

		if tracker is None:
			return

		accepted = tracker.pulse(time.monotonic())

		if accepted is None:
			return

		with self._state_lock:
			self._clock_bpm = accepted

		if self._tempo_source == "midi":
			self._arm_tempo_rebake(accepted)
			return

		self._warn_clock_mismatch(accepted)

	def _arm_tempo_rebake (self, bpm: float) -> None:

		"""Schedule the re-evaluation that adopts a newly detected tempo.

		Reuses the CC debounce timer rather than adding a second one: both want
		exactly the same update_assignments(), one timer keeps one teardown
		path, and a tempo change landing next to a knob move coalesces into a
		single re-bake instead of two.

		Only ever reached on an accepted tempo CHANGE, never per pulse — each
		rearm starts a Timer thread, which is fine at tempo-change rates and
		would be pathological at 24 PPQN.
		"""

		_log.info(
			"MIDI clock: tempo is now %g BPM — re-baking quantized variants",
			bpm,
		)

		with self._cc_debounce_lock:
			if self._cc_debounce_timer is not None:
				self._cc_debounce_timer.cancel()

			self._cc_debounce_timer = threading.Timer(
				_CC_DEBOUNCE_SECONDS,
				self._try_update_assignments,
				args=(f"MIDI clock {bpm:g} BPM",),
			)
			self._cc_debounce_timer.start()

	def _warn_clock_mismatch (self, bpm: float) -> None:

		"""Warn when the incoming clock disagrees with the configured tempo.

		The papercut this exists for: change the sequencer's tempo, forget to
		change tempo.bpm, and every quantized sample silently snaps
		to a grid that no longer matches the sequence.

		Only fires when the map actually quantizes (nothing else reads the
		session tempo), and only once per distinct offending tempo so a stable
		mismatch is said once.  Silent when target_bpm is 0 — quantizing is
		disabled entirely there, and spec_from_process already reports that.

		Reads and writes _clock_warned_bpm without a lock: only the rtmidi
		callback thread touches it.
		"""

		if not (self._map_quantizes or self._map_beat_filters) or self._target_bpm <= 0.0:
			return

		if abs(bpm - self._target_bpm) < _CLOCK_MISMATCH_WARN_BPM:
			return

		if self._clock_warned_bpm == bpm:
			return

		self._clock_warned_bpm = bpm

		_log.warning(
			"MIDI clock is %g BPM but tempo.bpm is %g — quantized samples "
			"and beat-based selection will not track your sequence.  Set "
			"tempo.source: midi to follow the clock, or update tempo.bpm.",
			bpm, self._target_bpm,
		)

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

	def _select_velocity_layers (
		self,
		entries:  list[tuple[subsample.query.Assignment, subsample.query.PickSpec]],
		velocity: int,
	) -> list[tuple[subsample.query.Assignment, subsample.query.PickSpec, int]]:

		"""Find every velocity layer covering ``velocity`` with its effective velocity.

		Linear scan over ``entries`` (≤ ~16 in practice; cost dwarfed by the
		query-engine and variant-lookup work in the same handler).  Returns an
		empty list when no layer covers the velocity — caller logs DEBUG and
		returns, matching the existing "no mapping for this note" semantics.

		Usually returns at most one layer: non-stacked layers on a note may not
		overlap (enforced at load), so a velocity hits exactly one.  Stacked
		members deliberately overlap, so several layers can cover the same
		velocity — they all fire and sound together.

		The effective velocity equals the input when the layer declares no
		``velocity_rescale_to``; otherwise it is the linear remap from the
		trigger range to the rescale range, rounded to int and clamped to
		[0, 127].  The handler uses this value for gain calculation in
		_render_float; the raw msg.velocity stays in DEBUG logs so both are
		visible when they differ.
		"""

		covering: list[tuple[subsample.query.Assignment, subsample.query.PickSpec, int]] = []

		for asgn, pick in entries:
			lo, hi = asgn.velocity_trigger

			if not (lo <= velocity <= hi):
				continue

			if asgn.velocity_rescale_to is None:
				covering.append((asgn, pick, velocity))
				continue

			out_lo, out_hi = asgn.velocity_rescale_to

			# Linear remap.  trigger_lo < trigger_hi is enforced at parse
			# time when rescale_to is set, so the divisor is always > 0.
			scaled = out_lo + (velocity - lo) / (hi - lo) * (out_hi - out_lo)
			effective = max(0, min(127, int(round(scaled))))

			covering.append((asgn, pick, effective))

		return covering

	def _handle_message (self, msg: mido.Message) -> None:

		"""Dispatch a single MIDI message via the select/process pipeline.

		Runs on rtmidi's dedicated callback thread (see ``run()``).  Must stay
		fast — any slow operation here stalls MIDI dispatch for the lifetime
		of the port.

		note_off (and note_on with velocity=0) marks matching active voices as
		releasing so the audio callback fades them out over self._release_fade_frames.
		note_on triggers sample selection via the query engine, then looks up
		the appropriate transform variant based on the assignment's ProcessSpec.
		Note routing for note_on first picks the velocity layer (the entry in
		the list at note_map[(channel, note)] whose velocity_trigger covers
		msg.velocity), then runs the query/variant lookup against that layer's
		Assignment.  Everything else is logged at DEBUG and ignored.
		"""

		# MIDI clock first: it is by far the highest-rate message (24 per beat,
		# ~50/sec at 125 BPM), and mido's rtmidi backend does not filter timing
		# messages, so they arrive whether or not anything wants them.  Handling
		# them here costs the note path one string compare and saves every clock
		# pulse from falling through to the DEBUG log at the bottom.
		if msg.type == "clock":
			self._handle_clock()
			return

		# note_off (and note_on with velocity=0, which mido normalises to note_off)
		# ends every matching non-one_shot voice — see _release_held for the
		# retirement semantics (stop looping, then fade unless release: full).
		if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
			self._release_held(msg.note, msg.channel)
			return

		# Program Change: switch the active instrument bank when a BankManager
		# is configured and the message arrives on the designated bank channel.
		if msg.type == "program_change" and self._bank_manager is not None:
			bm = self._bank_manager
			if bm.bank_channel_mido == -1 or msg.channel == bm.bank_channel_mido:
				# A redundant Program Change (re-selecting the already-active
				# program — sequencers often re-send it) must not run the full
				# rule swap on the rtmidi dispatch thread: switch_to() returns
				# True even when unchanged, and the swap below clears round-robin
				# / last-played state and re-queries the whole library for nothing
				# (audibly resetting segment round-robin).  Skip when unchanged.
				active = bm.active_bank
				if active is not None and active.program == msg.program:
					return

				# Remember where we came from so a failed rule-apply can roll the
				# POOL back too, not just the rules (see the except below).
				previous_program = active.program if active is not None else None

				if bm.switch_to(msg.program):
					# Both dicts are guarded by _state_lock — clear them
					# together so any concurrent reader (e.g. _select_segment
					# RMW) sees a consistent post-switch state.
					with self._state_lock:
						self._last_played.clear()
						self._segment_counters.clear()

					bank = bm.active_bank

					if bank.note_map is not None:
						# `map:` preset — swap the RULES too (assignments /
						# zones / CCs), not just the pool.
						rule_set = (
							bank.note_map,
							bank.zone_templates or (),
							bank.mapped_ccs or set(),
						)
					else:
						# `directory:` shorthand — reuse the top-level rules.
						# Restore the immutable snapshot (a prior `map:` preset
						# may have replaced the active base) and re-query it
						# against the swapped pool.
						rule_set = (
							self._top_level_note_map,
							self._top_level_zone_templates,
							self._top_level_mapped_ccs,
						)

					# _apply_rule_set runs update_assignments() against the
					# now-active bank's library and rolls back on failure, so a
					# broken kit (or a query that raises) never kills the player
					# thread mid-set.
					try:
						self._apply_rule_set(*rule_set)
					except Exception as exc:
						# The rules rolled back, but switch_to already committed the
						# POOL to the new bank — leaving _effective_* serving the new
						# library under the old rules (largely silence).  Switch the
						# pool back so pool and rules are consistent again.
						if previous_program is not None:
							bm.switch_to(previous_program)
							with self._state_lock:
								self._last_played.clear()
								self._segment_counters.clear()

						_log.error(
							"Program %d (%s) rules failed to apply — staying on the "
							"previous program: %s",
							msg.program, bank.name, exc,
						)
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

			# CC120 (All Sound Off) and CC123 (All Notes Off) are both panics that
			# stop a mode: loop voice looping forever if its note-off is ever lost,
			# but the MIDI spec distinguishes them and so do we.  A separate
			# _voices_lock section (never nested under _state_lock).
			if msg.control == 120:
				# All Sound Off: emergency mute — fast-declick EVERY voice,
				# including one_shots (a ringing one_shot open hi-hat must stop),
				# overriding each voice's own release / release: full / loop tail
				# with the fixed ~10 ms fade, exactly like a choke.
				with self._voices_lock:
					for voice in self._voices:
						voice.looping        = False
						voice.release_to_end = False
						if voice.fade_pos == 0:
							voice.release_frames = self._release_fade_frames
							voice.release_curve  = 0
						voice.releasing = True

			elif msg.control == 123:
				# All Notes Off: like a note-off for every held note — stop looping
				# and fade (unless release: full rings it out).  one_shots ignore
				# note-off, so they play on; CC120 above is the hard mute.
				with self._voices_lock:
					for voice in self._voices:
						if not voice.one_shot:
							voice.looping = False
							if not voice.release_to_end:
								voice.releasing = True

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

		# Choke: a note-on damps every sounding voice that declared itself
		# silenced_by this note (its mute group).  Fired on the raw note-on
		# gesture — BEFORE the mapping lookup (so a silent "grab" note still
		# damps), BEFORE the same-note steal, and BEFORE this hit's own layers
		# are appended (so the fresh voice survives its own sweep).  A choke
		# forces the ~10 ms declick, overriding release, so it must run ahead of
		# the steal (which would otherwise flag a self-choking voice for its own
		# slower release first).
		self._choke_voices(msg.channel, msg.note)

		entries = self._note_map.get((msg.channel, msg.note))

		if not entries:
			_log.debug("MIDI ch%d note %d: no mapping", msg.channel + 1, msg.note)
			return

		# Find every velocity layer covering msg.velocity.  Usually one; more
		# than one only when stacked members deliberately overlap.  Empty means
		# the velocity fell into a coverage gap (already WARNINGed at load) —
		# log DEBUG and stop.
		layers = self._select_velocity_layers(entries, msg.velocity)

		if not layers:
			_log.debug(
				"MIDI ch%d note %d vel %d: no velocity layer covers this velocity",
				msg.channel + 1, msg.note, msg.velocity,
			)
			return

		# Same-note steal: re-striking a note that is already sounding retires the
		# held gated/loop instance first — an implied note-off, so it releases per
		# its own configured release (a finite fade, or ``release: full`` ringing to
		# its natural end) — and the new strike then plays over the top.  Fired once
		# here, BEFORE this note-on's own layers are appended, so it only retires
		# earlier presses, never the (possibly stacked) voices about to be created.
		# Tied to the note-on gesture, not to sample-selection success; one_shot
		# voices are excluded (see _release_held), so overlapping one-shots stack.
		self._release_held(msg.note, msg.channel)

		# Fire every covering layer.  A normal (non-stacked) note loops exactly
		# once; stacked members sound together as one composite hit.  Each call
		# is independent, so one layer finding no sample never silences another.
		for assignment, pick_spec, effective_velocity in layers:
			self._trigger_one(msg, assignment, pick_spec, effective_velocity)

	def _build_trigger_spec (
		self,
		assignment: subsample.query.Assignment,
		record:     subsample.library.SampleRecord,
		midi_note:  int,
	) -> "subsample.transform.TransformSpec":

		"""Build the exact TransformSpec a note-on for this (assignment, record,
		note) produces.

		Single source of truth shared by the trigger path (``_trigger_one``)
		and the variant pre-compute (``update_assignments`` /
		``_enqueue_quantize_variants``).  Any divergence between the two means
		pre-computed cache keys never match trigger-time keys — every
		pre-render is wasted work and the first trigger per note recomputes.

		Dynamic parameters (MIDI note, BPM, CC values, reference path) are
		substituted at the position the user declared them in the
		``process:`` list.
		"""

		# Validation: skip repitch for unpitched samples.
		midi_note_for_spec: typing.Optional[int] = None

		if assignment.process.has_repitch():
			if subsample.analysis.has_stable_pitch(record.spectral, record.pitch, record.duration):
				midi_note_for_spec = midi_note

		# Validation: skip stretch_quantize for samples with no tempo.
		# pad_quantize does NOT need source tempo — only target BPM.
		bpm_for_spec: typing.Optional[float] = None
		grid_for_spec = 16

		if assignment.process.has_stretch_quantize():
			if record.rhythm.tempo_bpm > 0.0:
				bpm_for_spec, grid_for_spec = _quantize_params(assignment.process, "stretch_quantize", self._target_bpm)
			else:
				# DEBUG, not WARNING: on the trigger path this fires on EVERY
				# note-on for a tempo-less sample.  The once-per-rebuild
				# WARNING in update_assignments already surfaces it; an
				# unthrottled per-trigger WARNING would spam the
				# latency-critical path.
				_log.debug(
					"stretch_quantize %s: sample %r has no detected tempo — "
					"playing without beat-quantizing",
					assignment.name, record.name,
				)

		if assignment.process.has_pad_quantize():
			bpm_for_spec, grid_for_spec = _quantize_params(assignment.process, "pad_quantize", self._target_bpm)

		cc_state_snapshot, cc_omni_snapshot = self._snapshot_cc_state()

		return subsample.transform.spec_from_process(
			assignment.process,
			midi_note=midi_note_for_spec,
			target_bpm=bpm_for_spec,
			resolution=grid_for_spec,
			reference_path=_reference_wav_path(assignment),
			cc_state=cc_state_snapshot,
			cc_omni=cc_omni_snapshot,
		)

	def _resolve_release (
		self,
		release: typing.Optional[subsample.query.ReleaseSpec],
		record:  subsample.library.SampleRecord,
	) -> tuple[typing.Optional[int], int, bool]:

		"""Resolve an assignment's ReleaseSpec to (frames, curve_code, to_end) for a voice.

		Returns ``(None, 0, False)`` when no release is configured — the callback
		then uses the player's global default declick length, so behaviour is
		unchanged.  ``release: full`` returns ``(None, 0, True)`` — the third
		element tells the note-off handler to play to the natural end with no
		fade.  Otherwise returns the fade length in output-rate frames and the
		curve code (0=cosine, 1=exponential).

		A CC-bound release time is frozen HERE, at note-on, from a CC snapshot —
		so turning the knob shapes the notes you play next, not the one already
		ringing.  (Reading CC state at note-off would need _state_lock while the
		note-off handler holds _voices_lock, which the lock ordering forbids.)
		The adaptive form (time is None) derives ~30-200 ms from the sample's own
		release character, mirroring ``reshape: true``.
		"""

		if release is None:
			return None, 0, False

		if release.to_end:
			# release: full — no fade; the note-off handler lets the voice play out.
			return None, 0, True

		curve_code = 1 if release.curve == "exponential" else 0

		time_ms = release.time

		if isinstance(time_ms, subsample.query.CcBinding):
			cc_state, cc_omni = self._snapshot_cc_state()
			binding = time_ms

			if binding.channel is not None:
				cc_val = cc_state.get((binding.channel - 1, binding.cc))
			else:
				cc_val = cc_omni.get(binding.cc)

			time_ms = binding.resolve(cc_val) if cc_val is not None else binding.default_value

		if time_ms is None:
			# Adaptive: short for percussive material, longer for sustained,
			# matching the reshape auto-release mapping (30 + 170 * release).
			time_ms = 30.0 + 170.0 * record.spectral.release

		frames = max(1, round(float(time_ms) / 1000.0 * self._output_sample_rate))
		return frames, curve_code, False

	def _resolve_loop (
		self,
		assignment: subsample.query.Assignment,
		record:     subsample.library.SampleRecord,
	) -> typing.Optional[tuple[int, int, int]]:

		"""Resolve a ``mode: loop`` assignment's loop to (start, end, crossfade) in
		OUTPUT-rate frames, or None when it isn't loop mode or has no loop points.

		A manual ``loop: {...}`` override (seconds / ms) wins per field; unset
		fields fall back to the sample's auto-detected points (SampleRecord.loop,
		stored in the sample's own native frames — rescaled here to the output
		rate the voice buffer plays at).  None from a ``mode: loop`` assignment is
		the fail-musical case: the caller plays gated.  Bounds are validated here;
		clamping to the specific rendered buffer happens in _append_voice.
		"""

		if assignment.mode != "loop":
			return None

		sr_out = self._output_sample_rate
		native = record.params.sample_rate
		scale  = (sr_out / native) if native else 1.0

		auto     = record.loop        # loopfind.LoopPoints in native frames, or None
		override = assignment.loop     # query.LoopSpec in seconds/ms, or None

		def resolve_frames (ov_seconds: typing.Optional[float], auto_frames: typing.Optional[int]) -> typing.Optional[int]:
			if ov_seconds is not None:
				return round(ov_seconds * sr_out)
			if auto_frames is not None:
				return round(auto_frames * scale)
			return None

		start = resolve_frames(override.start if override else None, auto.start if auto else None)
		end   = resolve_frames(override.end   if override else None, auto.end   if auto else None)

		if start is None or end is None or end <= start:
			return None

		if end - start < round(_MIN_LOOP_SECONDS * sr_out):
			return None      # too short to be a loop (see _MIN_LOOP_SECONDS)

		if override is not None and override.crossfade is not None:
			crossfade = round(override.crossfade / 1000.0 * sr_out)
		elif auto is not None:
			crossfade = round(auto.crossfade * scale)
		else:
			crossfade = round(0.030 * sr_out)   # 30 ms default (matches loopfind)

		return start, end, crossfade

	def _append_voice (
		self,
		audio:          numpy.ndarray,
		note:           int,
		channel:        int,
		one_shot:       bool,
		release_frames: typing.Optional[int],
		release_curve:  int,
		release_to_end: bool,
		loop_cfg:       typing.Optional[tuple[int, int, int]],
	) -> None:

		"""Build a _Voice for a trigger and enqueue it (under _voices_lock).

		Central voice-construction point so the loop/release fields are threaded
		once, not re-typed at every trigger branch.  loop_cfg is (start, end,
		crossfade) in this buffer's own frames or None; bounds are clamped to the
		rendered length and the crossfade to the available lead-in, and if that
		leaves nothing loopable the voice falls back to a plain one.
		"""

		looping        = False
		loop_start     = 0
		loop_end       = 0
		loop_crossfade = 0
		loop_xfade_in: typing.Optional[numpy.ndarray] = None

		if loop_cfg is not None:
			ls, le, xf = loop_cfg
			n  = len(audio)
			le = min(le, n)
			ls = max(0, min(ls, le - 1))
			xf = max(0, min(xf, ls, le - ls))

			# Re-apply _resolve_loop's minimum-length floor AFTER clamping to the
			# rendered buffer: when the selected sample renders shorter than the
			# resolved loop start, the clamp above can collapse the loop to a few
			# frames — a DC buzz that also makes the callback wrap dozens of times
			# per buffer.  Fall back to a plain (non-looping) voice instead.
			min_loop_frames = max(1, round(_MIN_LOOP_SECONDS * self._output_sample_rate))

			if le - ls >= min_loop_frames:
				looping        = True
				loop_start     = ls
				loop_end       = le
				loop_crossfade = xf
				if xf > 0:
					loop_xfade_in = audio[ls - xf : ls]
			elif (note, channel) not in self._loop_collapsed_warned:
				# The picked sample rendered shorter than its resolved loop start,
				# so clamping collapsed the loop below the minimum length.  Play
				# plain (gated) instead of a DC buzz — but say so once per note,
				# not silently (mirrors the no-usable-loop fail-musical warning).
				self._loop_collapsed_warned.add((note, channel))
				_log.warning(
					"Loop collapsed to nothing after clamping to the rendered "
					"sample on note %d ch %d — the picked sample is shorter than "
					"its loop start; playing gated instead.",
					note, channel + 1,
				)

		voice = _Voice(
			audio=audio, note=note, channel=channel,
			one_shot=one_shot, release_frames=release_frames,
			release_curve=release_curve, release_to_end=release_to_end,
			looping=looping, loop_start=loop_start, loop_end=loop_end,
			loop_crossfade=loop_crossfade, loop_xfade_in=loop_xfade_in,
		)

		with self._voices_lock:
			self._voices.append(voice)

	def _release_held (self, note: int, channel: int) -> None:

		"""Imply a note-off for every currently-sounding non-one_shot voice on
		(note, channel).

		Shared by the real note-off handler and the same-note steal on note-on.
		A looping voice stops looping (its cursor turns monotonic and plays past
		loop_end into the real tail); then, unless ``release: full`` asked to ring
		out unfaded, the callback fades it over the release window.  Clearing
		looping and setting releasing happen in ONE lock section so the callback
		never observes a half-updated voice.  one_shot voices are left untouched.

		Safe to call on a re-strike over voices that may already be retiring:
		re-applying to a voice that is already releasing (or already ringing out
		under ``release: full``) is a no-op, because the release progress lives in
		fade_pos / release_total, which only the callback ever initialises.
		"""

		with self._voices_lock:
			for voice in self._voices:
				if voice.note == note and voice.channel == channel and not voice.one_shot:
					voice.looping = False
					if not voice.release_to_end:
						voice.releasing = True

	def _choke_voices (self, channel: int, note: int) -> None:

		"""Fast-damp every sounding voice choked by a note-on of ``note`` on
		``channel`` (its ``silenced_by`` declarations).

		Unlike _release_held (same-note steal / note-off), this:
		  - cuts one_shot voices too — the whole point (a ringing open hi-hat is
		    a one_shot, and one_shots ignore note-off AND the CC123 all-notes-off
		    panic; only CC120 all-sound-off and a choke cut them);
		  - OVERRIDES the voice's own release / ``release: full`` / loop tail with
		    the fixed ~10 ms declick — a choke is a physical damp, not a note-off.

		Fires on the note-on GESTURE (before sample selection and before this
		hit's own voices are appended), so a silent "grab" note still damps and
		the new hit — appended afterward — is never caught by its own sweep.  One
		``_voices_lock`` section, never nested under _state_lock, so the callback
		never observes a half-updated voice.  Kills EVERY matching voice (one
		physical instrument), not just the newest.
		"""

		victims = self._choke_map.get((channel, note))
		if not victims:
			return

		with self._voices_lock:
			for voice in self._voices:
				if (voice.channel, voice.note) not in victims:
					continue
				voice.looping        = False
				voice.release_to_end = False    # override release: full — must not ring out
				if voice.fade_pos == 0:
					# Force the fast declick, overriding any configured release.
					# Only when not already fading: re-clamping a fade already in
					# progress would jump the ramp and click, so a voice mid-release
					# keeps its current fade (a rare, benign edge — a choke normally
					# hits still-sounding voices, fade_pos == 0).
					voice.release_frames = self._release_fade_frames
					voice.release_curve  = 0
				voice.releasing = True

	def _trigger_one (
		self,
		msg:                mido.Message,
		assignment:         subsample.query.Assignment,
		pick_spec:          subsample.query.PickSpec,
		effective_velocity: int,
	) -> None:

		"""Render one assignment for a note-on and enqueue its voice.

		Called once per covering velocity layer by ``_handle_message`` — once
		for a normal note, several times for a stacked note.  Every early return
		ends only this layer's processing, never the whole handler, so one layer
		finding no sample doesn't silence its stack-mates.

		Resolves the sample, then walks the variant → previous-variant → base →
		int-PCM fallback chain, appending a ``_Voice`` on the first that yields
		audio.  All per-assignment settings (gain, pan, output routing, mode,
		process) come off ``assignment``, so stacked members keep independent
		voices.
		"""

		pan_weights      = assignment.pan_weights
		output_routing   = assignment.output_routing
		# Only one_shot ignores note-off; gated and (for now) loop both release.
		one_shot         = (assignment.mode == "one_shot")

		# State-dict key is keyed by the assignment's identity so every layer —
		# including stacked members that share a velocity range — keeps its own
		# _last_played fallback and round_robin counters.
		state_key = (msg.channel, msg.note, id(assignment))

		# ── Sample selection ──────────────────────────────────────────────
		# eff_transform is captured here (not just where the variant lookup
		# below needs it) so the whole note-on reads one consistent bank.
		# Safety rests on single-threaded dispatch: program_change and note_on
		# are both handled by _handle_message on the one rtmidi thread, so no
		# bank swap can interleave within a handler.  (These are two separate
		# lock-taking property reads, not one atomic snapshot — anyone who adds
		# a concurrent switch_to() caller must revisit this.)

		eff_library   = self._effective_instrument_library
		eff_transform = self._effective_transform_manager

		sample_id = self._resolve_sample_id(assignment, pick_spec, eff_library, msg.velocity)

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

		# Resolve the note-off release once for this trigger (needs `record` for
		# the adaptive form and for the note-on CC snapshot).  Every voice this
		# note-on spawns — variant, previous-variant, base, int-PCM fallback —
		# carries the same resolved values.
		release_frames, release_curve, release_to_end = self._resolve_release(assignment.release, record)

		# Resolve loop points for mode: loop (None otherwise, or fail-musical when
		# no clean loop exists → play gated + warn once).  A loop assignment may
		# carry a time-PRESERVING process (filter, saturate, …) and then loops on
		# the variant buffer, which stays frame-aligned with the stored points;
		# timeline-ALTERING steps (repitch, quantize, reverse) are deferred to
		# gated at load (_parse_mode), so loop_cfg never meets a re-timed buffer.
		loop_cfg = self._resolve_loop(assignment, record)

		if assignment.mode == "loop" and loop_cfg is None and id(assignment) not in self._loop_unavailable_warned:
			self._loop_unavailable_warned.add(id(assignment))
			_log.warning(
				"MIDI map assignment %r: sample %r has no usable loop — playing gated "
				"(held then released) instead of looping.",
				assignment.name, record.name,
			)

		# ── Variant lookup based on ProcessSpec ───────────────────────────
		# eff_transform was captured at the top of the handler alongside
		# eff_library so both read the same bank for this note-on — guaranteed
		# by single-threaded rtmidi dispatch (see the capture site above).

		if eff_transform is not None:

			# Build the full ordered transform chain from the process spec.
			# Dynamic parameters (MIDI note, BPM) are substituted at the
			# position the user declared them in the process: list.
			if assignment.process.steps:

				spec = self._build_trigger_spec(assignment, record, msg.note)

				if spec.steps:
					variant = eff_transform.get_variant(sample_id, spec)

					if variant is not None:
						seg_audio, seg_level = self._select_segment(
							variant.audio, variant.level, variant.segment_bounds,
							assignment.segment_mode, msg.channel, msg.note,
							id(assignment),
						)
						mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing, record.channel_format, assignment.extract)
						rendered = self._render_float(seg_audio, seg_level, effective_velocity, mix_mat, assignment.gain_db)
						self._append_voice(rendered, msg.note, msg.channel, one_shot, release_frames, release_curve, release_to_end, loop_cfg)
						with self._state_lock:
							self._last_played[state_key] = variant
						_log.debug(
							"note %d (vel %d → %d) → %r → %r (variant, %d step(s))  (%.2fs)",
							msg.note, msg.velocity, effective_velocity, assignment.name, record.name,
							len(spec.steps), variant.duration,
						)
						return

					# New variant not ready — try the previously played variant
					# for this layer (smooth transition during gradual param changes).
					# state_key is keyed by the assignment's identity so each layer
					# falls back to its own previous variant, not another layer's.
					with self._state_lock:
						prev = self._last_played.get(state_key)

					if prev is not None and prev.key.sample_id == sample_id:
						seg_audio, seg_level = self._select_segment(
							prev.audio, prev.level, prev.segment_bounds,
							assignment.segment_mode, msg.channel, msg.note,
							id(assignment),
						)
						mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing, record.channel_format, assignment.extract)
						rendered = self._render_float(seg_audio, seg_level, effective_velocity, mix_mat, assignment.gain_db)
						self._append_voice(rendered, msg.note, msg.channel, one_shot, release_frames, release_curve, release_to_end, loop_cfg)
						_log.debug(
							"note %d (vel %d → %d) → %r → %r (previous variant)  (%.2fs)",
							msg.note, msg.velocity, effective_velocity, assignment.name, record.name, prev.duration,
						)
						return

			# Fall back to the base variant (float32, peak-normalised, no DSP).
			base = eff_transform.get_base(sample_id)

			if base is not None:
				seg_audio, seg_level = self._select_segment(
					base.audio, base.level, base.segment_bounds,
					assignment.segment_mode, msg.channel, msg.note,
					id(assignment),
				)
				mix_mat = self._get_mix_matrix(seg_audio.shape[1], pan_weights, output_routing, record.channel_format, assignment.extract)
				rendered = self._render_float(seg_audio, seg_level, effective_velocity, mix_mat, assignment.gain_db)
				self._append_voice(rendered, msg.note, msg.channel, one_shot, release_frames, release_curve, release_to_end, loop_cfg)
				_log.debug(
					"note %d (vel %d → %d) → %r → %r (base variant)  (%.2fs)",
					msg.note, msg.velocity, effective_velocity, assignment.name, record.name, base.duration,
				)
				return

		# 4. Last resort: convert from int PCM on this trigger.
		mix_mat = self._get_mix_matrix(record.audio.shape[1] if record.audio is not None else 1, pan_weights, output_routing, record.channel_format, assignment.extract)
		original: typing.Optional[numpy.ndarray] = self._render(record, effective_velocity, mix_mat, assignment.gain_db)

		if original is None:
			return

		self._append_voice(original, msg.note, msg.channel, one_shot, release_frames, release_curve, release_to_end, loop_cfg)

		_log.debug(
			"note %d (vel %d → %d) → %r → %r  (%.2fs)",
			msg.note, msg.velocity, effective_velocity, assignment.name, record.name, record.duration,
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

		# The divisor must match the ARRAY's dtype, not the configured capture
		# depth: imported files keep their native dtype (int16 for 16-bit,
		# int32 for 24/32-bit), so a 24-bit import under a 16-bit config would
		# otherwise render ~65536x too hot on this fallback path.
		record_depth = 16 if record.audio.dtype == numpy.int16 else 32
		float_audio = subsample.transform._pcm_to_float32(record.audio, record_depth)

		# Last-resort fallback: resample to the output rate when the record's PCM
		# is at a different rate (a live capture at the recorder rate under a
		# differing player.audio.sample_rate).  Disk-loaded samples already store
		# PCM at the output rate, and the common config leaves the two rates
		# equal, so this is a no-op there; without it the very first trigger of a
		# fresh capture — before its resampled variant is ready — would play at
		# the wrong pitch.  The cost falls only on that rare, one-off fallback,
		# never on the pre-rendered variant path.
		src_rate = record.audio_sample_rate
		if (
			isinstance(src_rate, int)
			and src_rate != self._output_sample_rate
			and float_audio.shape[0] > 0
		):
			import librosa
			float_audio = librosa.resample(
				float_audio.T,
				orig_sr=src_rate,
				target_sr=self._output_sample_rate,
				res_type="soxr_hq",
			).T.astype(numpy.float32)

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
		calls, but doing it here avoids per-rank spec-building work.  Specs
		come from ``_build_trigger_spec`` (which snapshots CC state per call)
		so pre-computed keys match trigger-time keys exactly.
		"""

		enqueued = 0
		seen_ids: set[int] = set()

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

				# Same helper as the trigger path so the pre-computed key is
				# byte-identical to the one the note-on will look up.  (The
				# spec is note-independent without repitch — the dedup by
				# sample_id above stays valid.)
				spec = self._build_trigger_spec(asgn, record, _note)

				if spec.steps:
					eff_transform.get_variant(sid, spec)
					enqueued += 1

		return enqueued

	def _materialize_zones (self) -> None:

		"""Rebuild ``self._note_map`` from base entries + zone-derived entries.

		Walks every declared ZoneTemplate, runs its select against the active
		library, filters to pitched samples (``has_stable_pitch``), filters
		to samples whose detected MIDI pitch lies inside the template's
		keyboard range, sorts by pitch, and lays each sample across a
		contiguous keyboard zone centred on its detected pitch.  Each
		derived (channel, note) entry gets an Assignment that is a clone
		of the template with its select replaced by a predicate pinned to
		that sample's ``sample_id`` (not its stem — stems can collide across
		take folders) so the query engine resolves to that specific sample at
		note-on.

		Called from ``__init__`` (so the startup log shows the materialised
		layout) and from the top of ``update_assignments`` (so every
		re-evaluation path — reload, _integrate_sample, bank switch, CC
		debounce — picks up library changes without inventing new plumbing).

		Concurrent calls from different threads produce the same result
		(deterministic from the same input) so the dict-rebind atomicity
		guarantee that the rest of the player relies on still holds.
		"""

		if not self._zone_templates:
			# No templates — the base map IS the working map.  Copy so any
			# subsequent runtime mutation (output-routing fix-up in run())
			# doesn't bleed into the base.
			self._note_map = dict(self._base_note_map)
			return

		# Start from a fresh copy of the manual entries; derived entries
		# are appended per-template below.
		new_map: NoteMap = {k: list(v) for k, v in self._base_note_map.items()}

		eff_library    = self._effective_instrument_library
		eff_similarity = self._effective_similarity_matrix

		# Lazy import: librosa is already a hard dep but the import is
		# slow on cold start; defer until first materialise so module
		# import stays fast.
		import librosa

		for template in self._zone_templates:

			# Run the template's select against the library.  Use the
			# first select spec only — zone-tuned doesn't support
			# fallback chains because the materialiser needs a stable
			# candidate list (a fallback firing mid-pattern would shuffle
			# every note's assigned sample).
			candidates = subsample.query.query(
				template.select[0],
				eff_library.samples(),
				eff_similarity,
				bpm=self._target_bpm,
			)

			# Filter to pitched candidates.  The 7-criterion gate is the
			# same one the player uses elsewhere to decide which samples
			# can be pitch-shifted at all — applying it implicitly here
			# means an unpitched sample never sneaks into a zone where
			# the repitch would produce nonsense.
			pitched: list[tuple[int, subsample.library.SampleRecord]] = []
			lo_note, hi_note = template.keyboard_range

			for record in candidates:

				if not subsample.analysis.has_stable_pitch(
					record.spectral, record.pitch, record.duration,
				):
					continue

				centre_midi = int(round(
					float(librosa.hz_to_midi(record.pitch.dominant_pitch_hz)),
				))

				if lo_note <= centre_midi <= hi_note:
					pitched.append((centre_midi, record))
				else:
					_log.debug(
						"zone-tuned %r: sample %r detected at MIDI %d is "
						"outside range [%d, %d] — excluded",
						template.name, record.name, centre_midi, lo_note, hi_note,
					)

			if not pitched:
				_log.info(
					"zone-tuned %r: no matching pitched samples in range [%d, %d] "
					"— no notes materialised on ch%d",
					template.name, lo_note, hi_note, template.channel + 1,
				)
				continue

			# Sort by (pitch, name) — ties broken alphabetically for a
			# deterministic layout that's reproducible across runs.
			pitched.sort(key=lambda x: (x[0], x[1].name))

			n = len(pitched)

			for i, (centre_midi, record) in enumerate(pitched):

				# Lower-pitched sample claims the floor of any non-integer
				# midpoint (and any integer midpoint too) — consistent
				# rule across both cases keeps the coverage gap-free at
				# the cost of a small asymmetry at integer midpoints
				# (e.g. midi 60+64 → 62, claimed by the lower).
				if i == 0:
					zone_lo = lo_note
				else:
					prev_centre = pitched[i - 1][0]
					zone_lo = (prev_centre + centre_midi) // 2 + 1

				if i == n - 1:
					zone_hi = hi_note
				else:
					next_centre = pitched[i + 1][0]
					zone_hi = (centre_midi + next_centre) // 2

				# Three or more samples sharing the SAME detected pitch leave
				# the middle one(s) with an empty zone (lo > hi) — surface it
				# rather than dropping the sample without a trace.
				if zone_lo > zone_hi:
					_log.warning(
						"Zone-tuned %s: sample %r shares its detected pitch "
						"(MIDI %d) with its neighbours and received no keys — "
						"it will not be playable on this channel",
						template.name, record.name, centre_midi,
					)
					continue

				# Build the derived Assignment for this sample.  The select is
				# replaced with an exact-identity predicate (the sample_id) so
				# the query engine at note-on resolves to THIS sample regardless
				# of how the template's filter would rank things this trigger.
				# Pinning by sample_id (not the stem) is essential: stems can
				# repeat across take-folders, so a name pin would resolve to
				# whichever twin the default order returns and misroute the zone.
				sample_where  = subsample.query.WherePredicate(sample_id=record.sample_id)
				sample_select = (subsample.query.SelectSpec(where=sample_where),)

				derived = subsample.query.Assignment(
					name                = f"{template.name} → {record.name}",
					select              = sample_select,
					process             = template.process,
					mode                = template.mode,
					loop                = template.loop,
					release             = template.release,
					gain_db             = template.gain_db,
					pan_weights         = template.pan_weights,
					output_routing      = template.output_routing,
					extract             = template.extract,
					segment_mode        = template.segment_mode,
					velocity_trigger    = template.velocity_trigger,
					velocity_rescale_to = template.velocity_rescale_to,
					stack               = template.stack,
				)
				pick = subsample.query.PickSpec(1, 1)

				for note in range(zone_lo, zone_hi + 1):
					new_map.setdefault(
						(template.channel, note), [],
					).append((derived, pick))

			_log.debug(
				"zone-tuned %r: %d sample(s) laid out across ch%d notes [%d, %d]",
				template.name, n, template.channel + 1, lo_note, hi_note,
			)

		# Atomic dict rebind: in-flight rtmidi handlers see either the
		# old map or the new map, never a half-applied state.
		self._note_map = new_map

	def _resolve_sample_id (
		self,
		assignment:  subsample.query.Assignment,
		pick_spec:   subsample.query.PickSpec,
		eff_library: subsample.library.InstrumentLibrary,
		velocity:    int,
	) -> typing.Optional[int]:

		"""Resolve an assignment to one concrete sample id for a single trigger.

		Fast path: the assignment's ranked candidate list was pre-computed by
		_rebuild_candidate_cache (it changes only when the active library
		changes), so the trigger just draws a pick index — no query, no sort,
		no filesystem access.  An empty cached list means "nothing matched";
		returns None.

		Slow path (cache miss): the select orders or filters by asynchronously
		baked variant state (quantized_beats / beat_match), whose ranking
		shifts as variants finish, so it is resolved live against the current
		library here.  ``PickSpec.resolve_index`` draws a 0-indexed rank from
		[lo, hi] (clamped to the list length); a range pick re-rolls per call.

		``velocity`` is the RAW incoming MIDI velocity (not the post-rescale gain
		velocity — the rescale compresses loudness, and using it here would make
		the quiet end of the pool unreachable).  It is consulted only by a
		velocity pick; the layer's own velocity_trigger window is forwarded so a
		narrow velocity layer still spans its whole pool.
		"""

		vel_lo, vel_hi = assignment.velocity_trigger

		cached = self._candidate_cache.get(id(assignment))

		if cached is not None:
			if not cached.ids:
				return None

			index = pick_spec.resolve_index(
				len(cached.ids), velocity, vel_lo, vel_hi, cached.loudness,
			)
			return cached.ids[index]

		# Variant-dependent select — resolve live (see _rebuild_candidate_cache).
		eff_similarity = self._effective_similarity_matrix
		eff_transform  = self._effective_transform_manager
		all_samples    = eff_library.samples()

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
				bpm=self._target_bpm,
			)

			if ranked:
				index = pick_spec.resolve_index(
					len(ranked), velocity, vel_lo, vel_hi, _loudness_positions(ranked),
				)
				return ranked[index].sample_id

		return None

	def _rebuild_candidate_cache (self) -> dict[int, list[subsample.library.SampleRecord]]:

		"""Pre-compute each assignment's ranked candidate list off the trigger thread.

		The ranked list an assignment resolves to depends only on the active
		library and similarity index — not on the triggering note, velocity, or
		the per-trigger pick draw — so it is stable between note-ons until the
		library changes.  Resolving it here turns sample selection at note-on
		into a single indexed pick (see _resolve_sample_id), removing the
		per-trigger query, sort, and ``directory:`` filesystem scan from the
		rtmidi callback thread.

		Stores variant-independent results (sample ids, plus the pool's
		normalised levels for a loudness-spaced velocity pick) as _Candidates in
		``self._candidate_cache`` via an atomic rebind.  Selects that order or
		filter by quantized_beats / beat_match are left out of the cache —
		their ranking changes as variants finish baking, so the trigger path
		re-queries them live.

		Returns the full id → ranked-records map (every assignment, cached or
		not) so update_assignments can reuse it for variant pre-computation
		without querying the library a second time.

		Called from update_assignments (after zone materialisation) so every
		re-evaluation path — startup, new sample, eviction, bank switch, MIDI
		map reload, CC debounce — refreshes the cache against the live library.
		"""

		eff_library    = self._effective_instrument_library
		eff_similarity = self._effective_similarity_matrix
		eff_transform  = self._effective_transform_manager
		all_samples    = eff_library.samples()

		ranked_by_assignment: dict[int, list[subsample.library.SampleRecord]] = {}
		new_cache: dict[int, _Candidates] = {}
		seen: set[int] = set()

		for entries in self._note_map.values():
			for assignment, _pick_spec in entries:
				assignment_id = id(assignment)

				if assignment_id in seen:
					continue

				seen.add(assignment_id)

				beats_resolver = _build_beats_resolver(
					assignment.process, eff_transform, self._target_bpm,
				)
				energy_profile_resolver = _build_energy_profile_resolver(
					assignment.process, eff_transform, self._target_bpm,
				)

				ranked: list[subsample.library.SampleRecord] = []

				for select_spec in assignment.select:
					ranked = subsample.query.query(
						select_spec, all_samples, eff_similarity, beats_resolver,
						energy_profile_resolver=energy_profile_resolver,
						bpm=self._target_bpm,
					)

					if ranked:
						break

				ranked_by_assignment[assignment_id] = ranked

				# Variant-dependent selects re-rank as variants bake, so the
				# trigger path must re-query them live — keep them uncached.
				if not subsample.query.select_uses_variant_state(assignment.select):
					new_cache[assignment_id] = _Candidates(
						ids=[record.sample_id for record in ranked],
						loudness=_loudness_positions(ranked),
					)

		# Atomic dict rebind: in-flight rtmidi handlers see either the old or
		# the new cache, never a half-built one.
		self._candidate_cache = new_cache

		return ranked_by_assignment

	def _prune_stale_layer_state (self) -> None:

		"""Drop _last_played / _segment_counters entries for retired assignments.

		Both dicts are keyed by (channel, note, id(Assignment)).  Manual
		assignments keep their identity across re-evaluations, so their
		round-robin position and variant-transition fallback survive; zone-
		derived assignments are re-minted by every _materialize_zones run, and
		rule swaps (reload / preset switch) replace the whole map — keys
		referencing retired objects can never match again, and each orphaned
		_last_played entry pins a full float32 variant in memory.
		"""

		live_ids = {
			id(assignment)
			for entries in self._note_map.values()
			for assignment, _pick_spec in entries
		}

		with self._state_lock:
			for key in [k for k in self._last_played if k[2] not in live_ids]:
				del self._last_played[key]

			for key in [k for k in self._segment_counters if k[2] not in live_ids]:
				del self._segment_counters[key]

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

		Also re-materialises zone-tuned templates before pre-computing
		variants, so any change to the active library (new captures,
		evictions, bank switches) is reflected in the working NoteMap
		before the variant pre-computation walks it.

		No-ops if no transform manager is configured or no processable
		assignments exist.
		"""

		# Serialised against every other re-evaluation AND the rule-swap
		# window in _apply_rule_set (see _rules_lock in __init__).
		with self._rules_lock:
			self._update_assignments_locked()

	def _update_assignments_locked (self) -> None:

		"""Body of update_assignments — caller must hold _rules_lock."""

		# Adopt a clock-detected tempo BEFORE anything reads _target_bpm — the
		# candidate cache below resolves quantized_beats through
		# _build_beats_resolver, and the variant pre-compute builds specs, both
		# off this value.  Doing it here, under _rules_lock, is what keeps the
		# trigger path and the pre-bake agreeing on one BPM: the clock publishes
		# into _clock_bpm continuously, but the tempo that reaches a cache key
		# only ever moves at a re-evaluation.
		#
		# Sticky by design: a stopped transport stops the pulses but leaves
		# _clock_bpm at the last measured tempo, so nothing re-bakes and the
		# grid stays where the sequencer left it.  target_bpm remains the
		# fallback until a clock is ever seen.
		if self._tempo_source == "midi":
			with self._state_lock:
				clock_bpm = self._clock_bpm

			if clock_bpm is not None and clock_bpm != self._target_bpm:
				_log.info(
					"Session tempo %g → %g BPM (from MIDI clock)",
					self._target_bpm, clock_bpm,
				)
				self._target_bpm = clock_bpm

		# A duration_beats filter measures sample length in beats, so it cannot
		# resolve without a tempo.  The effective tempo is now settled (clock
		# adopted just above, or the configured fallback).  On a hot reload this
		# raises inside _apply_rule_set_locked's try and rolls back to the live
		# rules; startup is validated earlier, in cli, before construction.
		if self._map_beat_filters and self._target_bpm <= 0.0:
			raise ValueError(_BEAT_FILTER_NO_TEMPO_MESSAGE)

		# Re-derive zone-tuned entries against the current library before
		# the candidate cache and variant pre-computation walk the NoteMap.
		# Cheap when no templates are declared (early return inside).
		self._materialize_zones()

		# Refresh the hot-path sample-selection cache against the current
		# library, and reuse the ranked lists it computed for the variant
		# pre-computation below — one query per assignment, not two.  This
		# runs even without a transform manager, so selection works when only
		# the player (and no transforms) is configured.
		candidate_records = self._rebuild_candidate_cache()

		# Drop per-layer state keyed by retired Assignment identities — rule
		# swaps and zone re-materialisation mint fresh objects, and stale
		# _last_played entries would otherwise pin full variant audio for the
		# rest of the session.
		self._prune_stale_layer_state()

		eff_transform = self._effective_transform_manager

		if eff_transform is None:
			return

		eff_library = self._effective_instrument_library

		# Group notes by Assignment identity (object id) — all notes in the same
		# assignment share the same select/process spec.  Collect (note, PickSpec)
		# pairs so stretch_quantize can pre-compute a variant for every reachable
		# rank (each note may resolve to a different sample, and a range PickSpec
		# may resolve to any rank in [lo, hi] at trigger time).
		groups: dict[int, tuple[subsample.query.Assignment, list[tuple[int, subsample.query.PickSpec]]]] = {}

		# Walk every velocity layer of every note.  Grouping by Assignment
		# identity dedupes — an Assignment used by multiple layers (or
		# multiple notes) computes its variants once.
		for (_ch, note), entries in self._note_map.items():
			for (asgn, pick_spec) in entries:

				if asgn.process.steps:
					group_key = id(asgn)

					if group_key not in groups:
						groups[group_key] = (asgn, [])

					groups[group_key][1].append((note, pick_spec))

		if not groups:
			return

		_total_assignments = 0
		_total_variants = 0

		for asgn, note_picks in groups.values():

			# Reuse the ranked list _rebuild_candidate_cache already resolved
			# for this assignment — no second query against the library.
			ranked = candidate_records.get(id(asgn), [])

			if not ranked:
				continue

			notes = [n for n, _p in note_picks]

			# Repitch: all notes share pick=1 (same sample, pitched per note).
			# Each note's spec is built by the SAME helper the trigger path
			# uses, so the pre-computed keys are identical to trigger-time
			# keys even for chains that add a vocoder, CC-bound params, or a
			# quantize step alongside the repitch.  get_variant enqueues on
			# miss and dedups against in-flight work.
			if asgn.process.has_repitch():
				# transform.auto_pitch=false disables the note-range fan-out;
				# variants then render lazily on first trigger instead.
				if not eff_transform.auto_pitch_enabled:
					continue

				record = eff_library.get(ranked[0].sample_id)

				if record is None:
					continue

				if not subsample.analysis.has_stable_pitch(record.spectral, record.pitch, record.duration):
					_log.warning(
						"Pitched %s: best match %r has no stable pitch — skipping pitch variants",
						asgn.name, record.name,
					)

				else:
					for note in notes:
						spec = self._build_trigger_spec(asgn, record, note)

						if spec.steps:
							eff_transform.get_variant(record.sample_id, spec)

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

	# Backward-compatible alias — LOAD-BEARING, do not remove in a dead-code
	# sweep: cli.py, scripts/measure_handler_timing.py, and several tests
	# call this name.  Kept until those callers migrate to
	# update_assignments.
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
				"Could not refresh note assignments during %s — playback "
				"continues with the previous set: %s",
				context, exc,
			)

	def _strip_oob_routing_rules (
		self,
		base_note_map:  NoteMap,
		zone_templates: tuple[ZoneTemplate, ...],
	) -> tuple[NoteMap, tuple[ZoneTemplate, ...]]:

		"""Drop output-routing indices that exceed the configured channel count.

		Validates against ``self._output_channels`` (fixed at construction
		from ``player.audio.channels``, default stereo), so it can run both
		at startup in ``run()`` — applied to every rule source, the top-level
		map and each ``map:`` preset — and again in ``reload_midi_map`` for
		hot-edited rules.  Any assignment or zone template whose ``output:``
		names a channel beyond that count falls back to default routing,
		logged once; stripping every source means a later re-materialisation
		or Program Change never revives a stripped index (the bug the strip
		guards against).

		Returns the (possibly rebuilt) note map and zone-template tuple;
		entries without out-of-bounds routing are returned unchanged.
		"""

		def _strip (
			routing: typing.Optional[tuple[int, ...]],
			label:   str,
		) -> typing.Optional[tuple[int, ...]]:

			if routing is None:
				return None

			for idx in routing:
				if idx >= self._output_channels:
					_log.warning(
						"%s: output index %d exceeds device channel count (%d) — "
						"using default routing",
						label, idx + 1, self._output_channels,
					)
					return None

			return routing

		# Memoise the replacement per SOURCE assignment (by identity): a multi-note
		# assignment is ONE object shared across its (ch, note) entries, and
		# identity-keyed logic downstream — e.g. _build_choke_map grouping a
		# multi-note ``silenced_by: self`` by id(assignment) — collapses if the
		# strip hands back a distinct object per note.  Memoising preserves that
		# shared identity (and collapses per-note duplicate warnings to one).
		fixed_base: NoteMap = {}
		replaced:   dict[int, subsample.query.Assignment] = {}

		for (ch, note), entries in base_note_map.items():
			fixed_entries: list[tuple[subsample.query.Assignment, subsample.query.PickSpec]] = []
			for asgn, pick_spec in entries:
				if id(asgn) not in replaced:
					stripped = _strip(
						asgn.output_routing,
						f"Assignment {asgn.name!r} (ch {ch}, note {note})",
					)
					replaced[id(asgn)] = (
						dataclasses.replace(asgn, output_routing=stripped)
						if stripped is not asgn.output_routing
						else asgn
					)
				fixed_entries.append((replaced[id(asgn)], pick_spec))
			fixed_base[(ch, note)] = fixed_entries

		fixed_zones = tuple(
			dataclasses.replace(t, output_routing=stripped)
			if (stripped := _strip(t.output_routing, f"Zone template {t.name!r}")) is not t.output_routing
			else t
			for t in zone_templates
		)

		return fixed_base, fixed_zones

	def _apply_rule_set (
		self,
		base_note_map: NoteMap,
		zone_templates: tuple[ZoneTemplate, ...],
		mapped_ccs: set[int],
	) -> None:

		"""Atomically swap the active rule set and re-materialise, with rollback.

		Shared core of both ``reload_midi_map`` (file-watcher hot-reload) and
		the Program Change handler's preset switch.  Swaps the note map, zone
		templates and CC set, then runs ``update_assignments()`` to validate
		and re-materialise against the *currently active* library/pool.  If
		validation raises, all four fields are restored and the exception is
		re-raised so the caller can stay live under the previous rules.

		Thread-safety: dict and tuple rebinds are atomic under the GIL, so any
		in-flight ``_handle_message()`` READER sees either the old rules or
		the new, never a half-applied state.  WRITERS are serialised by
		``_rules_lock`` (held for the whole install→validate→rollback window,
		reentrantly shared with ``update_assignments``), so a concurrent
		re-evaluation can never rebuild from half-installed rules.  Cache
		clears follow the lock-ordering rule (``_mix_matrix_lock`` and
		``_state_lock`` are never nested).

		Args:
			base_note_map:  Manual note routing to install (NoteMap).
			zone_templates: Zone-tuned templates to install.
			mapped_ccs:     CC numbers referenced by the new rule set.

		Raises:
			Exception: whatever update_assignments() / _materialize_zones()
			surfaces; the previous rules are restored before propagating.
		"""

		with self._rules_lock:
			self._apply_rule_set_locked(base_note_map, zone_templates, mapped_ccs)

	def _apply_rule_set_locked (
		self,
		base_note_map: NoteMap,
		zone_templates: tuple[ZoneTemplate, ...],
		mapped_ccs: set[int],
	) -> None:

		"""Body of _apply_rule_set — caller must hold _rules_lock."""

		old_base_note_map  = self._base_note_map
		old_note_map       = self._note_map
		old_zone_templates = self._zone_templates
		old_mapped_ccs     = self._mapped_ccs
		old_map_quantizes  = self._map_quantizes
		old_map_beat_filters = self._map_beat_filters

		# Apply the new configuration first so update_assignments()
		# validates against what the player would actually run with.  This
		# is the canonical validation path — we don't duplicate query
		# logic for a separate dry-run.
		self._base_note_map  = base_note_map
		self._note_map       = dict(base_note_map)
		self._zone_templates = zone_templates
		self._mapped_ccs     = mapped_ccs

		# Derived from the incoming rules rather than threaded through every
		# caller, so a preset switch back to the top-level rules re-derives it
		# from those rules automatically.
		self._map_quantizes  = _uses_quantize(base_note_map, zone_templates)
		self._map_beat_filters = _uses_beat_filter(base_note_map, zone_templates)
		self._sync_clock_tracker()

		try:
			# update_assignments() calls _materialize_zones() at its top,
			# which exercises every zone template against the active
			# library.  A bad template (e.g. select referencing an unknown
			# scorer) surfaces here.
			self.update_assignments()
		except Exception:
			# Roll back so the caller's catch leaves the player on the
			# previously-good configuration.  Any variants
			# update_assignments() may have enqueued before raising are
			# harmless — they sit in the transform manager's cache for
			# future calls.  The candidate cache is deliberately NOT
			# restored: a stale id misses and falls through to the live
			# query, so the cost is performance-only and self-corrects on
			# the next update_assignments.
			self._base_note_map  = old_base_note_map
			self._note_map       = old_note_map
			self._zone_templates = old_zone_templates
			self._mapped_ccs     = old_mapped_ccs
			self._map_quantizes  = old_map_quantizes
			self._map_beat_filters = old_map_beat_filters
			self._sync_clock_tracker()
			raise

		# Validation succeeded — clear caches whose entries reference the
		# old assignments by identity so the next note_on rebuilds them.
		# _segment_counters clear is under _state_lock so it serialises
		# against the round_robin RMW in _select_segment.
		with self._mix_matrix_lock:
			self._mix_matrix_cache.clear()
		with self._state_lock:
			self._segment_counters.clear()
		# The old Assignment ids the fail-musical-loop warn-dedup keyed on are gone
		# with the swap, so clear it — otherwise it grows unbounded across reloads
		# and a reused id could suppress a genuine warning.  Lock-free: the set is
		# only add/membership-tested on the handler thread and set ops are atomic,
		# so a best-effort clear here can at worst re-emit or drop one cosmetic
		# warning, never corrupt state.
		self._loop_unavailable_warned.clear()
		self._loop_collapsed_warned.clear()

		# Rebuild the choke table from the newly-applied base map.  On the
		# rollback path above we never reached here, and _choke_map was not
		# touched during the try, so it still matches the restored old base map.
		self._choke_map = _build_choke_map(self._base_note_map)

	def reload_midi_map (self, new_result: MidiMapResult) -> None:

		"""Replace the TOP-LEVEL note map (and zone-tuned templates) and re-compute variants.

		Called by the file-watcher when the top-level MIDI map changes.
		Refreshes the immutable top-level snapshot (so a later switch to a
		``directory:`` program picks up the edit), then applies the new rules
		live — UNLESS a ``map:`` preset is currently active, in which case the
		top-level assignments are not the live rule set and applying them would
		clobber the preset; the snapshot refresh still happens so the edit
		takes effect on the next non-preset switch.

		Atomic in the sense that matters for live performance: if the new
		map is structurally valid but semantically broken (e.g. an order
		clause like ``similarity`` whose required ``where.reference:`` was
		accidentally commented out, or a zone-tuned template whose query
		raises), ``update_assignments()`` raises on the first offending
		assignment.  ``_apply_rule_set`` catches that, restores the previous
		map AND zone templates, and re-raises so the watcher caller can log
		and stay live under the old configuration — playback never stops
		mid-performance for a YAML typo.

		Args:
			new_result: Parsed MidiMapResult from load_midi_map().  Must be
			            a complete replacement (not a diff).  Both
			            ``note_map`` (manual entries) and
			            ``zone_templates`` are swapped in.

		Raises:
			Exception: whatever update_assignments() / _materialize_zones()
			surfaces while validating the new map.  The active
			configuration is restored before propagating, so the player
			remains usable.
		"""

		new_ccs = _collect_mapped_ccs(new_result.note_map, new_result.zone_templates)

		# Strip out-of-bounds output routing exactly as run() does at startup.
		# Without this, a hot-edited `output:` beyond the device channel count
		# would survive into the live rules and raise in route_to_device on
		# EVERY note-on (silent note + per-trigger ERROR) instead of degrading
		# gracefully to default routing with one warning here.
		stripped_map, stripped_zones = self._strip_oob_routing_rules(
			new_result.note_map, new_result.zone_templates,
		)

		# A `map:` preset is live — don't clobber it with the top-level rules.
		# Refresh the snapshot so the edit takes effect on the next non-preset
		# switch (whose _apply_rule_set validates it and rolls back on failure);
		# we can't validate here without applying, which would replace the preset.
		if self._bank_manager is not None and self._bank_manager.active_bank.note_map is not None:
			self._top_level_note_map       = stripped_map
			self._top_level_zone_templates = stripped_zones
			self._top_level_mapped_ccs     = new_ccs
			_log.info(
				"MIDI map reloaded (top-level): a map: preset is active, so the "
				"edit applies to directory programs / on the next non-preset switch",
			)
			return

		old_count = len(self._note_map)

		# Apply the new rules FIRST: update_assignments validates them and
		# _apply_rule_set rolls back the live rules + re-raises on failure.  Only
		# refresh the top-level snapshot once they are known good — otherwise a
		# broken hot-edit would become the snapshot a later `directory:` switch
		# restores, even though it never applied cleanly here.
		self._apply_rule_set(stripped_map, stripped_zones, new_ccs)

		self._top_level_note_map       = stripped_map
		self._top_level_zone_templates = stripped_zones
		self._top_level_mapped_ccs     = new_ccs

		_log.info(
			"MIDI map reloaded: %d note(s)%s (was %d)",
			len(self._note_map),
			f", {len(self._zone_templates)} zone-tuned template(s)" if self._zone_templates else "",
			old_count,
		)

	def _select_segment (
		self,
		audio: numpy.ndarray,
		level: subsample.analysis.LevelResult,
		segment_bounds: typing.Optional[tuple[tuple[int, int], ...]],
		segment_mode: typing.Union[str, int],
		channel: int,
		note: int,
		assignment_id: int,
	) -> tuple[numpy.ndarray, subsample.analysis.LevelResult]:

		"""Select a segment from quantized audio, or return the full audio.

		When segment_mode is active and bounds are available, slices the audio
		to a single segment and recomputes the level.  Otherwise returns the
		original audio and level unchanged.

		``assignment_id`` is ``id()`` of the layer's Assignment; it extends the
		round_robin counter key so each layer on the same (channel, note) —
		including stacked members sharing a velocity range — advances its own
		independent counter.
		"""

		if not segment_mode or segment_bounds is None or not segment_bounds:
			return audio, level

		if isinstance(segment_mode, int):
			idx = max(0, min(segment_mode - 1, len(segment_bounds) - 1))
		elif segment_mode == "round_robin":
			key = (channel, note, assignment_id)
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

				if effective_pan is None and logical_out in subsample.channel.STANDARD_LAYOUTS:
					effective_pan = numpy.ones(logical_out, dtype=numpy.float32)
				# For a non-standard output count (3/5/7) leave effective_pan None:
				# build_mix_matrix's default routing spreads a mono extract to the
				# centre front pair rather than raising on every note-on.

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
					else:
						# Without explicit routing, logical_out_ch IS
						# self._output_channels (see its assignment above),
						# so the decoder already matches the device width.
						mat = decoder
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

		# Anti-clip ceiling from the TRUE peak of the buffer being rendered, not
		# the mono-mean level.peak: an anti-phase mic pair or a heavily widened
		# stereo stem can peak well above its mono downmix, so level.peak would
		# under-protect and let the voice render far past full scale into the
		# limiter (audible distortion).  Also accounts for the worst-case row sum
		# of the mix matrix (e.g. a 5.1→stereo downmix sums FL + 0.707*FC + …).
		buffer_peak = float(numpy.max(numpy.abs(audio))) if audio.size else 0.0
		max_row_sum = float(numpy.max(numpy.sum(numpy.abs(mix_matrix), axis=1)))

		if buffer_peak > 0.0 and max_row_sum > 0.0:
			final_gain = min(raw_gain, 1.0 / (buffer_peak * max_row_sum))
		else:
			final_gain = raw_gain

		_log.debug(
			"gain: norm=%.3f  vel_scale=%.3f  gain_db=%.1f  raw=%.3f  final=%.3f  (rms=%.4f buf_peak=%.4f)",
			norm_gain, vel_scale, gain_db, raw_gain, final_gain,
			level.rms, buffer_peak,
		)

		gained = audio * final_gain

		# Channel mapping: (n_frames, in_ch) @ (in_ch, out_ch) = (n_frames, out_ch)
		result: numpy.ndarray = (gained @ mix_matrix.T).astype(numpy.float32)
		return result
