"""Configuration loading and validation for Subsample.

Always loads config.yaml.default (shipped with the package) as the base, then
deep-merges the user's config.yaml on top. The default file is the single source
of truth for all default values; config.yaml only needs to specify overrides.
Exposes typed, frozen dataclasses so every other module gets IDE completion and
mypy coverage.
"""

import dataclasses
import logging
import os
import pathlib
import typing

import yaml


_log = logging.getLogger(__name__)


SUPPORTED_TEMPO_SOURCES: frozenset[str] = frozenset({"config", "midi"})
"""Valid values for tempo.source — where the session tempo comes from.  "config"
reads the fixed tempo.bpm; "midi" follows an incoming MIDI clock on the player's
input (falling back to tempo.bpm until a clock is seen)."""


# ---------------------------------------------------------------------------
# Memory auto-detection
# ---------------------------------------------------------------------------

_AUTO_DETECT_FALLBACK_MB: float = 160.0
"""Fallback total budget when system RAM cannot be detected (96+56+8 at 60/35/5 split)."""


def _auto_detect_memory_mb () -> float:

	"""Return min(25% of total system RAM, 1024) in MB.

	Uses os.sysconf on Linux/macOS.  Falls back to _AUTO_DETECT_FALLBACK_MB
	on platforms where sysconf is unavailable.
	"""

	try:
		total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
		quarter_mb = (total_bytes / (1024 * 1024)) * 0.25
		return min(quarter_mb, 1024.0)

	except (ValueError, OSError, AttributeError):
		return _AUTO_DETECT_FALLBACK_MB


@dataclasses.dataclass(frozen=True)
class AudioConfig:

	"""Audio capture settings for the recorder.

	`channels` may be None (the default), in which case the correct value is
	resolved at startup from the selected input device's reported
	`maxInputChannels`.  Once resolved, a new AudioConfig is constructed via
	`dataclasses.replace` so that the rest of the pipeline always sees a
	concrete integer.

	When set explicitly, `channels` is validated to be > 0 at config-load
	time.  The auto-detect path validates after the device is opened.

	`input` selects which physical input channels to record from on a
	multi-channel interface (0-indexed internally, 1-indexed in YAML).
	When set, the stream opens with enough channels to cover the highest
	index, and only the requested columns are extracted.  When omitted,
	the first `channels` inputs are used.
	"""

	sample_rate: int
	bit_depth: int
	buffer_frames: int
	channels: typing.Optional[int] = None
	"""Number of input channels to capture.  None = auto-detect from device."""
	input: typing.Optional[tuple[int, ...]] = None
	"""Physical input channels to record (0-indexed).  None = first N channels.
	Set via 1-indexed list in config.yaml; converted at config-load time."""
	device: typing.Optional[str] = None
	audio_format: str = "wav"
	"""Output container for captured and imported samples.  "wav" (default)
	writes uncompressed PCM and supports 16/24/32-bit.  "flac" writes
	lossless compressed audio (~40–60% smaller) and supports 16/24-bit;
	per-file fallback to WAV fires transparently when a 32-bit source is
	imported so no precision is lost.  Live capture under audio_format=flac
	is validated at startup — combining it with bit_depth=32 is rejected
	with a clear message rather than failing later on the first capture."""
	ambisonic_format: typing.Optional[str] = None
	"""When set, the four input channels are processed as ambisonic content
	and stored as first-order AmbiX B-format.  Supported values: "a_generic"
	(generic tetrahedral A-format, capsule order FLU/FRD/BLD/BRU, with a
	post-matrix HF shelf), "a_nt_sf1" (Rode NT-SF1 A-format, as a_generic
	plus a per-capsule matching EQ), "b_fuma"
	(pre-encoded FuMA B-format, reordered/renormalised to AmbiX), "b_ambix"
	(pre-encoded AmbiX B-format, stored as-is).  None disables ambisonic
	processing entirely — the default."""
	float_import_ceiling_dbfs: typing.Optional[float] = -1.0
	"""Ceiling, in dBFS, for 32-bit float / 64-bit double audio wherever subsample
	reads it — CLI file-input (`subsample <file>`), library `directory:`/`path:`
	loads, OSC import, and the watcher.  These formats have no hard 0 dBFS limit,
	but subsample's internal integer pipeline does, so a float file whose peaks
	exceed full scale would hard-clip on the way in.  When set, such a source is
	scaled down as a whole (preserving relative dynamics between hits) so its peak
	sits at this ceiling — losslessly, and inaudibly for loudness (playback
	re-normalises per sample; note that `order: loudest`/`quietest` compare the
	absolute stored level, so a scaled-down source can rank below an equally-loud
	unscaled one).  Integer-PCM sources are never touched.  None restores the
	historical hard-clip.  Default -1.0 leaves 1 dB of headroom.

	A hot float sample that was analysed before this setting existed keeps its
	older fingerprint until something re-analyses it — the sidecar is keyed on the
	file's bytes, and those don't change — so it can play a decibel or two quiet in
	the meantime.  Deleting its .analysis.json sidecar forces the refresh."""


@dataclasses.dataclass(frozen=True)
class BufferConfig:

	max_seconds: int


@dataclasses.dataclass(frozen=True)
class RecorderConfig:

	audio: AudioConfig
	buffer: BufferConfig
	enabled: bool = True
	previews: bool = True
	"""Master switch for visual sample previews.  When True (default),
	every captured or imported sample gets a ``.preview.png`` sidecar
	(fixed 1024×256 thumbnail) and an embedded ``preview`` block in its
	``.analysis.json`` sidecar (compact data the Supervisor dashboard
	renders as SVG on demand).  Set False to save ~15-25 KB per PNG and
	~4 KB of JSON per sample if you do not browse the library visually."""
	directory: str = "samples/captures"
	"""Directory where captured recordings are saved.  Absolute, or relative
	to the directory subsample runs from.  Created automatically.  Point
	library.directory at the same path (the default pairing) for a
	persistent library that grows across sessions."""
	filename_format: str = "%Y-%m-%d_%H-%M-%S-%3f"
	"""strftime format for recording filenames (no extension; .wav/.flac is
	added automatically).  %3f is a custom token for zero-padded 3-digit
	milliseconds.  Timestamps are the moment the recording ended."""


@dataclasses.dataclass(frozen=True)
class PlayerAudioConfig:

	device: typing.Optional[str] = None
	bit_depth: typing.Optional[int] = None
	"""Output bit depth for the audio device (16, 24, or 32).
	None (the default) means use the recorder's bit depth, which matches
	the capture quality.  Override here if the output device requires a
	different format."""
	sample_rate: typing.Optional[int] = None
	"""Output sample rate in Hz.
	None (the default) means use the recorder's sample rate.  Set this to
	match the output device's native rate when it differs from the
	recorder (e.g. recorder at 22050 Hz, output at 44100 Hz).  Variants
	are resampled once at production time, not at trigger time."""
	channels: typing.Optional[int] = None
	"""Number of output channels.  None (the default) means stereo (2).
	Set to 6 for 5.1, 8 for 7.1, or any value your device supports.
	Channel ordering follows SMPTE: FL, FR, FC, LFE, BL, BR, SL, SR."""
	buffer_frames: typing.Optional[int] = None
	"""PortAudio output buffer size in frames.  None (the default) lets the
	OS pick the device-preferred size — usually 256-1024 on Linux ALSA.
	Set explicitly to trade latency for stability: smaller values cut
	output-side jitter (e.g. ~5.8 ms at 256 frames @ 44.1 kHz, ~2.9 ms at
	128 frames) but increase the risk of audible underruns under load.
	Must be a power of two in [32, 4096]; values outside this range or
	non-power-of-two are rejected at config load.  If the device cannot
	honour the requested size at stream open, an ERROR is logged and the
	stream falls back to the device default automatically."""


@dataclasses.dataclass(frozen=True)
class PlayerConfig:

	audio: PlayerAudioConfig = dataclasses.field(default_factory=PlayerAudioConfig)
	enabled: bool = False
	midi_device: typing.Optional[str] = None
	"""Name (or substring) of the MIDI input device for triggering playback.
	Case-insensitive substring match. If omitted, auto-selects when only one
	MIDI input device is present, or shows an interactive menu for multiple.
	Ignored when virtual_midi_port is set."""

	virtual_midi_port: typing.Optional[str] = None
	"""Name for a virtual MIDI input port created by Subsample at startup.
	When set, this port is used instead of any hardware MIDI device — midi_device
	is ignored and no device selection menu is shown. External applications (DAWs,
	aconnect, MIDI routing tools) can send MIDI to this port by name.
	Example: "Subsample Virtual MIDI"."""

	max_polyphony: int = 8
	"""Maximum number of simultaneous voices the mix should accommodate.
	Drives per-voice gain: target_rms = 1.0 / max_polyphony.  With the default
	of 8, each voice targets 0.125 RMS (~-18 dBFS), leaving headroom for 8
	simultaneous notes. Raise this if you hear clipping during dense passages
	(more voices, each quieter). Lower it for louder individual notes when
	fewer overlap. Range: 1–64."""

	limiter_threshold_db: float = -1.5
	"""Threshold (dBFS) above which the safety limiter begins soft-clipping.
	Signals below this pass completely untouched. At -1.5 dBFS the limiter
	only engages for genuine near-clip transients and is transparent during
	normal playback. Lower values (e.g. -6.0) give more compression;
	0.0 disables the limiter. Range: -12.0 to 0.0."""

	limiter_ceiling_db: float = -0.1
	"""Maximum output level (dBFS) the limiter allows. The signal is
	smoothly compressed between threshold and ceiling via a tanh curve,
	asymptotically approaching this value. Must be greater than
	limiter_threshold_db. Range: limiter_threshold_db to 0.0."""

	midi_map: typing.Optional[str] = None
	"""Path to a MIDI routing map file (YAML). Defines which MIDI notes
	trigger which samples. Required when the player is enabled — without
	it the player will not start. `subsample --init` scaffolds two maps and
	wires in midi-map-gm-drums.yaml (a complete GM percussion kit); the
	template it copies to midi-map.yaml documents the full format."""

	watch_midi_map: bool = False
	"""When True, monitor the midi_map file at runtime for changes and
	reload assignments without restarting.  Enables a live-coding workflow
	where you edit your MIDI map in a text editor and changes take effect
	on the next trigger.  Debounced to handle editors that write multiple
	intermediate saves.

	Requires midi_map to be set."""

	strict_midi_map: bool = True
	"""When True (default), unknown keys in `where:` blocks and unknown
	processor names in `process:` blocks raise a parse error with the list
	of valid options.  Strict mode catches typos early — e.g. a mistyped
	`duratoin: 1.0` that would otherwise silently match every sample.

	Set to False to restore the historical lenient behaviour where unknown
	keys are logged as warnings and ignored.  Useful when loading older
	MIDI map files that carry keys the parser no longer recognises."""


@dataclasses.dataclass(frozen=True)
class DetectionConfig:

	threshold_db: float
	hold_seconds: float
	warmup_seconds: float
	floor_adaptation: float
	trim_pre_ms: float
	trim_post_ms: float

	release_threshold_db: typing.Optional[float] = None
	"""Separate CLOSE threshold, in dB over the pre-hit ambient floor, at which a
	sounding tail is treated as returned to silence and the recording ends (after
	the hold_seconds debounce).  This decouples the end from the start: threshold_db
	stays the loud OPEN threshold that catches the attack, while a lower
	release_threshold_db lets a long decay (cymbal, ride, gong) ring out toward the
	noise floor instead of being cut ~16 dB above it.  Must be below threshold_db
	(a Schmitt-trigger hysteresis pair: open high, close low).  None (default) makes
	the end reuse threshold_db - identical to the historical single-threshold
	behaviour."""

	retrigger_threshold_db: typing.Optional[float] = None
	"""While already recording, a rise of this many dB over the decaying tail counts
	as the NEXT hit: the current sample is finalised at the new onset and a fresh
	recording begins immediately.  This is the "end on silence OR the next hit,
	whichever comes first" guarantee - and the only thing that ends a tail that
	background noise holds above release_threshold_db.  None (default) disables
	re-triggering; a recording then ends only on silence or the buffer cap.

	Assumes each hit's whole attack lands within max(hold_seconds, 0.1 s): a sound with
	a slow or two-stage attack (a soft onset, then a louder transient a moment later -
	breath before a flute note, mallet contact before a bowl) can otherwise read its
	own late transient as a re-strike and over-split.  Fast-attack percussion (drums,
	cymbals) is unaffected; for slower material, raise hold_seconds so it spans the
	attack.  Typical: 10-15 dB, comfortably above any tail amplitude modulation."""

	fade_out_ms: float = 0.0
	"""Length of a half-cosine fade applied to each segment's trailing edge, in
	milliseconds.  Masks a cut taken mid-tail or at the next hit so it never clicks.
	0.0 (default) keeps the historical ~2 ms declick (trim_post_ms)."""


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:

	start_bpm: float = 120.0
	"""Tempo prior for beat_track — the algorithm's initial BPM estimate.
	Does not constrain the result; just biases the search."""

	tempo_min: float = 30.0
	"""Minimum tempo (BPM) considered by the PLP pulse detector."""

	tempo_max: float = 300.0
	"""Maximum tempo (BPM) considered by the PLP pulse detector."""


@dataclasses.dataclass(frozen=True)
class SimilarityConfig:

	weight_spectral: float = 1.0
	"""Weight applied to the spectral feature group (14 normalised [0, 1] values:
	flatness, attack, release, centroid, bandwidth, ZCR, harmonic ratio, contrast,
	voiced fraction, log-attack time, spectral flux, spectral rolloff, spectral slope,
	and crest factor). Higher weight = spectral shape dominates the comparison.
	Range: 0.0–2.0. Set to 0.0 to disable entirely."""

	weight_timbre: float = 1.0
	"""Weight applied to the sustained-timbre MFCC group (12 mel-frequency cepstral
	coefficients, coeff 1–12). Captures the steady-state timbral character of a sound —
	useful for distinguishing instrument families (pad vs pluck vs brass). Higher weight
	= sustained timbre dominates. Range: 0.0–2.0. Set to 0.0 to disable."""

	weight_timbre_delta: float = 0.5
	"""Weight applied to the delta-MFCC group (12 first-order differences of the MFCCs).
	Encodes how the timbre *changes* over the duration of the sound — useful for sounds
	with an evolving character (attack-to-sustain shift). Secondary signal; default 0.5
	gives it half the influence of the primary timbre group. Range: 0.0–2.0."""

	weight_timbre_onset: float = 1.0
	"""Weight applied to the onset-weighted MFCC group (12 MFCCs weighted toward the
	first ~50 ms of the sound). Captures attack character — critical for percussive
	discrimination (kick vs snare vs hi-hat all have similar sustain but different
	attacks). Higher weight = attack character dominates. Range: 0.0–2.0."""

	weight_band_energy: float = 1.0
	"""Weight applied to the multi-band energy group (8 values: 4 per-band energy
	fractions + 4 per-band decay rates across sub-bass 20-250 Hz, low-mid 250-2k Hz,
	high-mid 2-6k Hz, and presence 6k+ Hz). Directly encodes drum-type physical
	signatures: kick = sub-bass dominant, snare = mid + presence, hi-hat = air.
	Range: 0.0–2.0. Set to 0.0 to disable."""


@dataclasses.dataclass(frozen=True)
class LibraryConfig:

	max_memory_mb: float = 100.0
	"""Maximum audio memory (MB) for in-memory instrument samples.

	When this limit is exceeded the oldest samples are evicted (FIFO) to make
	room. Only in-memory audio is removed; WAV files on disk are never deleted.
	At 44100 Hz 16-bit mono, 100 MB ≈ 19 minutes of audio."""

	directory: str = "samples/captures"
	"""Path to the directory of instrument samples to load at startup.
	Subsample walks this directory recursively, so samples can be organised
	into subdirectories (e.g. `kicks/`, `snares/`) however suits the user.
	Each sample is identified by its audio file; matching `.analysis.json`
	and `.preview.png` sidecars are regenerated on startup if missing, and
	any orphaned sidecars are deleted automatically."""

	watch: bool = False
	"""When True, monitor library.directory at runtime for new audio
	files and hot-load them into the live instrument library without
	restarting.

	Two detection paths run in parallel:

	1. Sidecar path — watches for .analysis.json sidecar files (fastest:
	   the sidecar signals that both the WAV and analysis are ready).
	2. Audio file path — watches for audio files (.wav, .flac, .aiff, .aif,
	   .ogg, .mp3, .mpeg — see ``subsample.cache.AUDIO_EXTENSIONS``) from any
	   source.  After a grace period to let a sidecar arrive (in case the
	   source is another subsample instance), checks file-size stability,
	   runs the full analysis pipeline, writes a sidecar, and loads the
	   sample.

	Works with multi-instance setups (recorder on one machine, player on
	another via a shared directory) and with audio files from any external
	application.

	Requires library.directory to be set and player.enabled to be True."""


@dataclasses.dataclass(frozen=True)
class TempoConfig:

	"""The session tempo — used both by the quantize processors and by the
	``duration_beats`` selection filter, which is why it is a top-level section
	rather than a transform detail."""

	bpm: float = 0.0
	"""Session tempo in BPM.  Used by the ``stretch_quantize`` / ``pad_quantize``
	processors when an assignment does not give its own ``tempo:``, and by the
	``duration_beats`` selection filter (which measures sample length in beats at
	this tempo).  0.0 means unset — those processors are skipped, and a map that
	filters by ``duration_beats`` fails to load until a tempo is provided.  There
	is no auto-stretch-every-rhythmic-sample behaviour.

	Also the fallback when ``source: midi`` is set but no clock has been seen yet
	(cold start, or the sequencer is not sending one).  Leaving it at 0.0 in that
	case means beat-based behaviour waits for the first clock, so set a sensible
	session tempo even when following the clock."""

	source: str = "config"
	"""Where the session tempo comes from.

	``config`` (default) uses ``bpm`` — a fixed session tempo.  ``midi`` follows
	the tempo of an incoming MIDI clock on the player's MIDI input, rounded to
	whole BPM, and falls back to ``bpm`` until a clock is seen.  The detected
	tempo is adopted only when it changes (and only after it holds steady),
	because each change re-bakes every quantize variant.  Once adopted it is
	sticky: a stopped transport keeps the last detected tempo rather than
	reverting.  An assignment's own ``tempo:`` still overrides both."""


@dataclasses.dataclass(frozen=True)
class TransformConfig:

	max_memory_mb: float = 50.0
	"""Maximum memory (MB) for in-memory derivative audio (transform variants).

	Separate from library.max_memory_mb — derivatives are disposable and
	regenerated on demand, so they have their own independent budget.
	Eviction strategy: parent-priority FIFO (all variants of the oldest parent
	are evicted together to keep variant sets intact).
	At 44100 Hz float32 stereo, 50 MB ≈ 150 seconds of derivative audio."""

	auto_pitch: bool = True
	"""When True, pitch-shifted variants are produced for each tonal sample
	(those that pass the has_stable_pitch() test) across the full note range
	defined by the MIDI map assignment (e.g. all notes in C-1..G9).
	Set to False to disable all pitch variant production."""

	quantize_resolution: int = 16
	"""Grid subdivision for beat-quantized time-stretch.
	Determines the note value that onsets are snapped to at the target BPM.
	1 = whole notes, 2 = half, 4 = quarter, 8 = eighth, 16 = sixteenth.
	Higher values give finer onset alignment but stretch more segments."""

	variant_cache_dir: str = "samples/variant-cache"
	"""Directory for persistent disk cache of transform variants.
	Variants are stored as binary files keyed by a SHA-256 hash of the
	parent audio, transform spec, output sample rate, and analysis version.
	Set to empty string or null to disable disk caching."""

	max_disk_mb: float = 500.0
	"""Maximum disk space (MB) for cached variant files.  0 = disabled.
	Least-recently-used files are evicted (a cache read touches mtime, so recently played variants survive) when the budget is
	exceeded.  At 44100 Hz float32 stereo, 500 MB ≈ 1500 seconds."""

	carrier_memory_mb: float = 10.0
	"""Memory budget (MB) for the vocoder carrier file cache.  Always derived
	from the global memory budget (5%) at config load — there is no
	config.yaml key for it, so the dataclass default only matters for
	directly-constructed configs in tests."""


@dataclasses.dataclass(frozen=True)
class OscConfig:

	"""OSC (Open Sound Control) event forwarding configuration.

	When enabled, Subsample sends sample events to OSC-compatible apps
	(sequencers, visualisers, etc.) and optionally receives file import
	requests from other OSC-compatible apps.

	Requires the optional ``python-osc`` dependency:
	``pip install "subsample[osc] @ git+https://github.com/simonholliday/subsample.git"``
	"""

	enabled: bool = False
	"""Master switch for OSC integration.  When False, no sender or
	receiver is created regardless of other settings."""

	send_host: str = "127.0.0.1"
	"""Destination host for outgoing OSC messages."""

	send_port: int = 9000
	"""Destination UDP port for outgoing OSC messages."""

	receive_enabled: bool = False
	"""When True (and enabled is True), start an OSC receiver that
	listens for /sample/import messages to trigger file import."""

	receive_port: int = 9002
	"""UDP port the OSC receiver listens on."""

	receive_host: str = "127.0.0.1"
	"""Interface the OSC receiver binds to.  Defaults to loopback so the
	arbitrary-path /sample/import handler is not exposed to the network.
	Set to "0.0.0.0" to accept messages from other hosts on a trusted LAN
	(unauthenticated remote file read/load — only do this on a network you
	control)."""


@dataclasses.dataclass(frozen=True)
class SupervisorConfig:

	"""Supervisor web dashboard configuration.

	When enabled, Subsample broadcasts its state via WebSocket to the
	Supervisor dashboard.  The dashboard renders live panels showing
	MIDI activity, library contents, and recorder status.

	Requires the optional ``supervisor`` dependency:
	``pip install "subsample[supervisor] @ git+https://github.com/simonholliday/subsample.git"``
	"""

	enabled: bool = False
	"""Master switch for the Supervisor dashboard.  When False, no
	WebSocket server is started."""

	port: int = 9003
	"""WebSocket port the Supervisor server listens on."""


@dataclasses.dataclass(frozen=True)
class AmbisonicConfig:

	"""Project-wide ambisonic decoding and orientation settings.

	Applied at playback time to samples stored as first-order B-format
	AmbiX.  Rotation (yaw/pitch/roll) and decoder type are project-wide —
	all ambisonic samples decode through the same matrix.  Per-sample or
	per-assignment overrides are intentionally out of scope for now.
	"""

	decoder: str = "basic"
	"""Decoder weight mode: "basic" (flat velocity weights, sharp lobes,
	best LF behaviour), "max_re" (narrower front lobe via Max-rE gains,
	best HF localisation), or "inphase" (softest lobes, no anti-phase
	back-radiation — best for listening far from the sweet spot)."""

	yaw_degrees: float = 0.0
	"""Yaw rotation applied to B-format signals before decoding.  Positive
	rotates the sound field counter-clockwise seen from above (i.e. sounds
	originally at +X front move toward +Y left)."""

	pitch_degrees: float = 0.0
	"""Pitch rotation about the +Y (left) axis.  Positive tilts the nose
	downward — a +X front sound moves toward -Z down."""

	roll_degrees: float = 0.0
	"""Roll rotation about the +X (front) axis.  Positive tilts the head
	right — a +Y left sound moves toward +Z up."""

	max_order: int = 1
	"""Reserved for future higher-order support.  Must be 1 — only
	first-order B-format is currently implemented."""


@dataclasses.dataclass(frozen=True)
class Config:

	recorder: RecorderConfig
	detection: DetectionConfig
	max_memory_mb: typing.Optional[float] = None
	"""Total memory budget (MB) for all sample caches.
	None = auto-detect: min(25% of total system RAM, 1024 MB).
	When resolved, the budget is split automatically:
	60% library samples, 35% transform variants, 5% carrier.
	Overridden by explicit per-cache settings in library/transform sections."""
	analysis: AnalysisConfig = dataclasses.field(default_factory=AnalysisConfig)
	library: LibraryConfig = dataclasses.field(default_factory=LibraryConfig)
	similarity: SimilarityConfig = dataclasses.field(default_factory=SimilarityConfig)
	player: PlayerConfig = dataclasses.field(default_factory=PlayerConfig)
	transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)
	tempo: TempoConfig = dataclasses.field(default_factory=TempoConfig)
	osc: OscConfig = dataclasses.field(default_factory=OscConfig)
	supervisor: SupervisorConfig = dataclasses.field(default_factory=SupervisorConfig)
	ambisonic: AmbisonicConfig = dataclasses.field(default_factory=AmbisonicConfig)


def load_config (path: typing.Optional[pathlib.Path] = None) -> Config:

	"""Load configuration, merging config.yaml.default with config.yaml.

	Always loads config.yaml.default as the base. If a user config.yaml exists
	(or an explicit path is given), it is deep-merged on top so user settings
	override defaults while unspecified keys inherit default values.
	"""

	default_path = _locate_default_config()
	base = _read_yaml(default_path)

	user_path = _resolve_user_config_path(path)

	# Avoid loading the same file twice when the caller explicitly passes the
	# default path (e.g. in tests).
	if user_path is not None and user_path.resolve() == default_path.resolve():
		user_path = None

	if user_path is not None:
		user = _read_yaml(user_path)
		raw = _deep_merge(base, user)
		_log.info("Configuration: %s (over built-in defaults)", user_path.resolve())
	else:
		raw = base
		if path is None:
			# INFO on purpose: running from the wrong directory silently gives
			# defaults-only — this line is the "which config am I on?" answer.
			_log.info(
				"Configuration: built-in defaults (no config.yaml in %s)",
				pathlib.Path.cwd(),
			)
		else:
			# Caller explicitly passed the bundled default (test convenience).
			_log.debug("Configuration: built-in defaults (explicit default path)")

	return _build_config(raw)


def data_dir () -> pathlib.Path:

	"""Return the directory of bundled product data (subsample/data/).

	Holds the default config, the shipped MIDI maps, and the GM reference
	sidecars, all declared as package data so editable and regular installs
	resolve identically. (A plain __file__-relative path rather than
	importlib.resources: callers need real filesystem Paths, and subsample can
	never run from a zipped package.)
	"""

	return pathlib.Path(__file__).parent / "data"


def _locate_default_config () -> pathlib.Path:

	"""Return the path to the bundled config.yaml.default.

	Raises FileNotFoundError if the file is missing (broken installation).
	"""

	default = data_dir() / "config.yaml.default"

	if not default.exists():
		raise FileNotFoundError(
			f"Bundled config.yaml.default not found at {default}. "
			"The package installation may be corrupted."
		)

	return default


def _resolve_user_config_path (
	explicit: typing.Optional[pathlib.Path],
) -> typing.Optional[pathlib.Path]:

	"""Return the user's config override path, or None if no user config exists.

	Priority: explicit path argument → ./config.yaml in CWD → None.
	When an explicit path is provided it must exist; no CWD fallback is tried.
	"""

	if explicit is not None:
		explicit = explicit.expanduser()
		if explicit.exists():
			return explicit

		raise FileNotFoundError(f"Config file not found: {explicit}")

	cwd_config = pathlib.Path.cwd() / "config.yaml"
	if cwd_config.exists():
		return cwd_config

	return None


def _deep_merge (
	base: dict[str, typing.Any],
	override: dict[str, typing.Any],
) -> dict[str, typing.Any]:

	"""Recursively merge override onto base, returning a new dict.

	For each key in override: if both values are dicts, recurse; if the base
	value is a dict and the override is None (a section commented out entirely in
	the user file), the base defaults are preserved; otherwise the override value
	wins (including an explicit None on a scalar key — e.g. `channels: null`).
	Keys present in base but absent from override are preserved unchanged.
	Neither input is mutated.
	"""

	result = dict(base)

	for key, override_value in override.items():
		base_value = result.get(key)
		if isinstance(base_value, dict) and isinstance(override_value, dict):
			result[key] = _deep_merge(base_value, override_value)
		elif isinstance(base_value, dict) and override_value is None:
			pass  # empty YAML section (all children commented out) — preserve base defaults
		else:
			result[key] = override_value

	return result


def _read_yaml (path: pathlib.Path) -> dict[str, typing.Any]:

	"""Read and parse a YAML file, returning a plain dict."""

	try:
		with path.open("r", encoding="utf-8") as fh:
			data = yaml.safe_load(fh)
	except yaml.YAMLError as exc:
		raise ValueError(f"Config file {path} contains invalid YAML: {exc}") from exc

	# An empty file, or one whose every line is commented out, parses as None.
	# That means "no overrides", not "malformed" — commenting the whole file out
	# to fall back to defaults is a normal debugging move, and the project's own
	# shipped default is mostly comments.  A non-None non-mapping is still an error.
	if data is None:
		return {}

	if not isinstance(data, dict):
		raise ValueError(f"Config file {path} must contain a YAML mapping at top level")

	return data


def _require (section: dict[str, typing.Any], key: str, section_name: str) -> typing.Any:

	"""Return section[key], raising a clear ValueError if it is absent or empty."""

	if key not in section:
		raise ValueError(
			f"Missing required config key '{section_name}.{key}'. "
			f"Check your config.yaml against config.yaml.default."
		)

	# A key written with no value (`filename_format:`) parses as None, and the
	# string keys then coerced it to the literal "None" — which sent every
	# capture to a file called None.wav, each overwriting the last.  A required
	# key with no value is always a mistake, so say so rather than inventing one.
	if section[key] is None:
		raise ValueError(
			f"Config key '{section_name}.{key}' is present but has no value. "
			f"Give it a value, or remove the line to use the default."
		)

	return section[key]


def _require_bool (
	section:      typing.Mapping[str, typing.Any],
	key:          str,
	default:      bool,
	section_name: str,
) -> bool:

	"""Return an optional boolean config value, rejecting non-boolean YAML.

	``bool(value)`` accepted anything truthy, so a quoted YAML boolean —
	``enabled: "false"``, ``watch: "no"`` — silently meant True, inverting the
	setting.  Quoting a boolean is a common editor habit and the failure is
	invisible.  Unquoted ``true``/``false``/``yes``/``no`` still parse to real
	bools in YAML and are accepted exactly as before.
	"""

	if key not in section or section[key] is None:
		return default

	value = section[key]

	if not isinstance(value, bool):
		raise ValueError(
			f"Config key '{section_name}.{key}' must be true or false "
			f'(got {value!r}) — remove the quotes if you wrote "true" or "false".'
		)

	return value


class _KeyTracker (dict[str, typing.Any]):

	"""A dict that records every key the config builder consults.

	After _build_config has read everything it understands, any key present
	in the YAML but never consulted is a typo or a removed option — warned
	about by name so a misspelt setting doesn't silently fall back to its
	default.  Tracking what the code actually reads means there is no
	hand-maintained schema list to drift out of date.
	"""

	def __init__ (self, raw: dict[str, typing.Any], label: str) -> None:

		super().__init__(raw)
		self.label = label
		self.accessed: set[str] = set()

	def get (self, key: typing.Any, default: typing.Any = None) -> typing.Any:
		self.accessed.add(key)
		return super().get(key, default)

	def __getitem__ (self, key: typing.Any) -> typing.Any:
		self.accessed.add(key)
		return super().__getitem__(key)

	def __contains__ (self, key: typing.Any) -> bool:
		self.accessed.add(key)
		return super().__contains__(key)

	def unknown_keys (self) -> list[str]:
		return sorted(set(self.keys()) - self.accessed)


def _section (
	container: dict[str, typing.Any],
	key: str,
	registry: typing.Optional[list["_KeyTracker"]] = None,
	label: typing.Optional[str] = None,
) -> dict[str, typing.Any]:

	"""Return the named sub-section as a dict, or {} when absent.

	Raises a clear ValueError naming the key when the section is present but
	not a mapping (e.g. a scalar from an indentation typo), instead of letting
	a later ``.get`` fail with an opaque "'str' object has no attribute 'get'".

	When ``registry`` is given, the returned mapping is a _KeyTracker
	registered for the unknown-key sweep at the end of _build_config.
	"""

	value = container.get(key, {})

	# A section header with every child commented out parses as None.  _deep_merge
	# already preserves the defaults for sections that ship UNcommented in
	# config.yaml.default, but only four of them do — the other seven (osc,
	# supervisor, library, similarity, transform, tempo, ambisonic) ship
	# commented, so `osc:` with its children commented reached here as None and
	# hard-failed startup while blaming the user's indentation.  Commenting a
	# subsystem's settings out but keeping the header is the natural way to
	# disable it, and it means "no overrides".
	if value is None:
		value = {}

	if not isinstance(value, dict):
		raise ValueError(
			f"Config section {key!r} must be a mapping (got {type(value).__name__}: {value!r}). "
			"Check your config.yaml indentation."
		)

	if registry is None:
		return value

	tracker = _KeyTracker(value, label or key)
	registry.append(tracker)

	return tracker


def _build_config (raw: dict[str, typing.Any]) -> Config:

	"""Construct the Config dataclass tree from a raw YAML dict."""

	# Every section is wrapped in a _KeyTracker so keys the builder never
	# consults can be warned about (typos / removed options) after the build.
	trackers: list[_KeyTracker] = []
	raw = _KeyTracker(raw, "top-level")
	trackers.append(raw)

	recorder_raw   = _section(raw, "recorder", trackers)
	audio_raw      = _section(recorder_raw, "audio", trackers, "recorder.audio")
	buffer_raw     = _section(recorder_raw, "buffer", trackers, "recorder.buffer")
	detection_raw  = _section(raw, "detection", trackers)
	analysis_raw   = _section(raw, "analysis", trackers)

	# ------------------------------------------------------------------
	# Renamed keys (2026-07) — hard errors naming the exact replacement, so
	# an old config fails loudly instead of silently using defaults.
	# (Same policy as the transform.target_bpm → tempo.bpm migration.)
	# ------------------------------------------------------------------
	if "output" in raw:
		raise ValueError(
			"The `output:` section has moved into `recorder:` — "
			"`output.directory` is now `recorder.directory` and "
			"`output.filename_format` is now `recorder.filename_format`."
		)
	if "instrument" in raw:
		raise ValueError(
			"The `instrument:` section is now called `library:` — rename the "
			"section (its keys are unchanged: library.directory, "
			"library.max_memory_mb, library.watch)."
		)
	for old_key, new_key in (
		("snr_threshold_db", "threshold_db"),
		("ema_alpha", "floor_adaptation"),
		("hold_time", "hold_seconds"),
	):
		if old_key in detection_raw:
			raise ValueError(
				f"`detection.{old_key}` is now `detection.{new_key}` — rename the key."
			)
	for old_key, new_key in (
		("trim_pre_samples", "trim_pre_ms"),
		("trim_post_samples", "trim_post_ms"),
	):
		if old_key in detection_raw:
			raise ValueError(
				f"`detection.{old_key}` is now `detection.{new_key}` and is measured "
				"in milliseconds, not samples — at 44100 Hz divide the old value by "
				"44.1 (the old defaults, 10 and 90 samples, are now 0.25 and 2.0 ms)."
			)
	if "chunk_size" in audio_raw:
		raise ValueError(
			"`recorder.audio.chunk_size` is now `recorder.audio.buffer_frames` — "
			"rename the key."
		)

	device_raw = audio_raw.get("device")
	if device_raw is not None and not isinstance(device_raw, str):
		raise ValueError(
			f"recorder.audio.device must be a string (got {type(device_raw).__name__}: {device_raw!r}). "
			"Check your config.yaml."
		)

	# channels is optional: None means auto-detect from the selected device at startup.
	channels_raw = audio_raw.get("channels")
	channels: typing.Optional[int] = int(channels_raw) if channels_raw is not None else None

	# input is optional: 1-indexed list of physical input channels, converted
	# to 0-indexed tuple.  None means use the first N channels.
	input_raw = audio_raw.get("input")
	input_channels: typing.Optional[tuple[int, ...]] = None

	if input_raw is not None:
		input_list = list(input_raw)

		if not input_list:
			raise ValueError("recorder.audio.input must be a non-empty list")

		for ch in input_list:
			if not isinstance(ch, int) or ch < 1:
				raise ValueError(
					f"recorder.audio.input channels must be positive integers "
					f"(1-indexed), got {ch!r}"
				)

		if len(set(input_list)) != len(input_list):
			raise ValueError(f"recorder.audio.input contains duplicates: {input_list}")

		# Convert 1-indexed → 0-indexed.
		input_channels = tuple(ch - 1 for ch in input_list)

		# Infer channels from input length if not explicitly set.
		if channels is None:
			channels = len(input_channels)
		elif channels != len(input_channels):
			raise ValueError(
				f"recorder.audio.channels ({channels}) does not match "
				f"recorder.audio.input length ({len(input_channels)})"
			)

	ambisonic_format_raw = audio_raw.get("ambisonic_format")
	ambisonic_format: typing.Optional[str]
	if ambisonic_format_raw in (None, ""):
		ambisonic_format = None
	elif isinstance(ambisonic_format_raw, str):
		import subsample.ambisonic
		if ambisonic_format_raw not in subsample.ambisonic.SUPPORTED_AMBISONIC_FORMATS:
			raise ValueError(
				f"recorder.audio.ambisonic_format {ambisonic_format_raw!r} is not supported.  "
				f"Valid values: {sorted(subsample.ambisonic.SUPPORTED_AMBISONIC_FORMATS)} or null/empty string to disable."
			)
		ambisonic_format = ambisonic_format_raw
	else:
		raise ValueError(
			f"recorder.audio.ambisonic_format must be a string or null "
			f"(got {type(ambisonic_format_raw).__name__}: {ambisonic_format_raw!r})"
		)

	audio_format_raw = audio_raw.get("audio_format", "wav")
	if not isinstance(audio_format_raw, str):
		raise ValueError(
			f"recorder.audio.audio_format must be a string "
			f"(got {type(audio_format_raw).__name__}: {audio_format_raw!r})"
		)
	audio_format = audio_format_raw.lower()
	if audio_format not in {"wav", "flac"}:
		raise ValueError(
			f"recorder.audio.audio_format {audio_format_raw!r} is not supported.  "
			f"Valid values: 'wav', 'flac'."
		)

	float_ceiling_raw = audio_raw.get("float_import_ceiling_dbfs", -1.0)
	float_import_ceiling_dbfs = (
		None if float_ceiling_raw is None else float(float_ceiling_raw)
	)

	audio = AudioConfig(
		sample_rate=int(_require(audio_raw, "sample_rate", "recorder.audio")),
		bit_depth=int(_require(audio_raw, "bit_depth", "recorder.audio")),
		buffer_frames=int(_require(audio_raw, "buffer_frames", "recorder.audio")),
		channels=channels,
		input=input_channels,
		device=device_raw,
		audio_format=audio_format,
		ambisonic_format=ambisonic_format,
		float_import_ceiling_dbfs=float_import_ceiling_dbfs,
	)

	# Ambisonic capture requires exactly 4 channels.  Rejecting None (auto-
	# detect) at config-load time is deliberate: if a user's device reported
	# anything other than 4 channels, the failure would otherwise surface
	# deep inside a worker thread on the first capture, with only a stack
	# trace to go on.  Requiring an explicit channels: 4 catches the
	# mismatch at startup with a clear message.
	if ambisonic_format is not None and channels != 4:
		raise ValueError(
			f"recorder.audio.ambisonic_format={ambisonic_format!r} requires "
			f"recorder.audio.channels: 4 (set explicitly; auto-detect is not "
			f"accepted for ambisonic capture).  Got channels={channels}."
		)

	if audio.bit_depth not in {16, 24, 32}:
		raise ValueError(
			f"Unsupported bit_depth {audio.bit_depth}. "
			"Supported values: 16, 24, 32"
		)

	# FLAC only covers 16/24-bit in libsndfile's stable subtypes.  Reject
	# the combination at config-load time so the user sees the mismatch at
	# startup rather than at the first capture.  File imports with a 32-bit
	# source are handled separately by the per-request fallback in
	# recorder._write_audio_file (writes .wav per file).
	if audio.audio_format == "flac" and audio.bit_depth == 32:
		raise ValueError(
			"recorder.audio.audio_format='flac' requires recorder.audio.bit_depth "
			"of 16 or 24.  Set audio_format='wav' for 32-bit live capture."
		)

	if audio.sample_rate <= 0:
		raise ValueError(f"recorder.audio.sample_rate must be > 0 (got {audio.sample_rate})")
	if audio.channels is not None and audio.channels <= 0:
		raise ValueError(f"recorder.audio.channels must be > 0 (got {audio.channels})")
	if audio.buffer_frames <= 0:
		raise ValueError(f"recorder.audio.buffer_frames must be > 0 (got {audio.buffer_frames})")
	if audio.float_import_ceiling_dbfs is not None and audio.float_import_ceiling_dbfs > 0.0:
		raise ValueError(
			"recorder.audio.float_import_ceiling_dbfs must be <= 0 dBFS (a ceiling at or "
			f"below full scale), or null to disable (got {audio.float_import_ceiling_dbfs})"
		)

	buffer = BufferConfig(
		max_seconds=int(_require(buffer_raw, "max_seconds", "recorder.buffer")),
	)

	if buffer.max_seconds <= 0:
		raise ValueError(f"recorder.buffer.max_seconds must be > 0 (got {buffer.max_seconds})")

	recorder = RecorderConfig(
		audio=audio,
		buffer=buffer,
		enabled=_require_bool(recorder_raw, "enabled", True, "recorder"),
		previews=_require_bool(recorder_raw, "previews", True, "recorder"),
		directory=str(_require(recorder_raw, "directory", "recorder")),
		filename_format=str(_require(recorder_raw, "filename_format", "recorder")),
	)

	player_raw      = _section(raw, "player", trackers)
	player_audio_raw = _section(player_raw, "audio", trackers, "player.audio")
	player_device = player_audio_raw.get("device")
	if player_device is not None and not isinstance(player_device, str):
		raise ValueError(
			f"player.audio.device must be a string (got {type(player_device).__name__}: {player_device!r}). "
			"Check your config.yaml."
		)
	player_midi_device = player_raw.get("midi_device")
	if player_midi_device is not None and not isinstance(player_midi_device, str):
		raise ValueError(
			f"player.midi_device must be a string (got {type(player_midi_device).__name__}: {player_midi_device!r}). "
			"Check your config.yaml."
		)

	player_virtual_midi_port = player_raw.get("virtual_midi_port")
	if player_virtual_midi_port is not None and not isinstance(player_virtual_midi_port, str):
		raise ValueError(
			f"player.virtual_midi_port must be a string (got {type(player_virtual_midi_port).__name__}: {player_virtual_midi_port!r}). "
			"Check your config.yaml."
		)

	player_midi_map = player_raw.get("midi_map")
	if player_midi_map is not None and not isinstance(player_midi_map, str):
		raise ValueError(
			f"player.midi_map must be a string path (got {type(player_midi_map).__name__}: {player_midi_map!r}). "
			"Check your config.yaml."
		)

	player_bit_depth_raw = player_audio_raw.get("bit_depth")
	player_bit_depth: typing.Optional[int] = (
		int(player_bit_depth_raw) if player_bit_depth_raw is not None else None
	)
	if player_bit_depth is not None and player_bit_depth not in {16, 24, 32}:
		raise ValueError(
			f"Unsupported player.audio.bit_depth {player_bit_depth}. "
			"Supported values: 16, 24, 32"
		)

	player_sample_rate_raw = player_audio_raw.get("sample_rate")
	player_sample_rate: typing.Optional[int] = (
		int(player_sample_rate_raw) if player_sample_rate_raw is not None else None
	)
	if player_sample_rate is not None and player_sample_rate <= 0:
		raise ValueError(
			f"player.audio.sample_rate must be > 0 (got {player_sample_rate})"
		)

	player_max_polyphony = int(player_raw.get("max_polyphony", 8))
	if player_max_polyphony < 1 or player_max_polyphony > 64:
		raise ValueError(
			f"player.max_polyphony ({player_max_polyphony}) must be in [1, 64]. "
			"Raise it to allow more simultaneous voices; lower it for louder "
			"individual notes."
		)

	player_limiter_threshold_db = float(player_raw.get("limiter_threshold_db", -1.5))
	if player_limiter_threshold_db > 0.0 or player_limiter_threshold_db < -12.0:
		raise ValueError(
			f"player.limiter_threshold_db ({player_limiter_threshold_db}) must be in [-12.0, 0.0]."
		)

	player_limiter_ceiling_db = float(player_raw.get("limiter_ceiling_db", -0.1))
	if player_limiter_ceiling_db > 0.0 or player_limiter_ceiling_db < -12.0:
		raise ValueError(
			f"player.limiter_ceiling_db ({player_limiter_ceiling_db}) must be in [-12.0, 0.0]."
		)
	# threshold 0.0 = limiter disabled; the ceiling is then unused, so the
	# ordering constraint (which 0.0 could never satisfy) doesn't apply.
	if player_limiter_threshold_db < 0.0 and player_limiter_ceiling_db <= player_limiter_threshold_db:
		raise ValueError(
			f"player.limiter_ceiling_db ({player_limiter_ceiling_db}) must be greater than "
			f"player.limiter_threshold_db ({player_limiter_threshold_db})."
		)

	# Validate buffer_frames: must be a power of two in [32, 4096].  Powers
	# of two play nicely with most USB-class audio drivers; the upper bound
	# keeps "lower latency than the default" the intended use of this knob
	# and avoids silently picking a value too large to help.  Range and
	# power-of-two checks are surfaced as distinct messages so a user who
	# tries buffer_frames: 16 (a valid power of two but below the floor) is
	# told the range failed rather than mis-guided to look for a bit-pattern
	# problem.
	player_buffer_frames_raw = player_audio_raw.get("buffer_frames")
	player_buffer_frames: typing.Optional[int]
	if player_buffer_frames_raw is None:
		player_buffer_frames = None
	else:
		player_buffer_frames = int(player_buffer_frames_raw)

		if player_buffer_frames < 32 or player_buffer_frames > 4096:
			raise ValueError(
				f"player.audio.buffer_frames must be in [32, 4096] "
				f"(got {player_buffer_frames})"
			)

		if (player_buffer_frames & (player_buffer_frames - 1)) != 0:
			raise ValueError(
				f"player.audio.buffer_frames must be a power of two "
				f"(got {player_buffer_frames})"
			)

	player = PlayerConfig(
		audio=PlayerAudioConfig(
			device=player_device,
			bit_depth=player_bit_depth,
			sample_rate=player_sample_rate,
			channels=int(player_audio_raw["channels"]) if player_audio_raw.get("channels") is not None else None,
			buffer_frames=player_buffer_frames,
		),
		enabled=_require_bool(player_raw, "enabled", False, "player"),
		midi_device=player_midi_device,
		virtual_midi_port=player_virtual_midi_port,
		max_polyphony=player_max_polyphony,
		limiter_threshold_db=player_limiter_threshold_db,
		limiter_ceiling_db=player_limiter_ceiling_db,
		midi_map=player_midi_map,
		watch_midi_map=_require_bool(player_raw, "watch_midi_map", False, "player"),
		strict_midi_map=_require_bool(player_raw, "strict_midi_map", True, "player"),
	)

	if player.audio.channels is not None and player.audio.channels <= 0:
		raise ValueError(f"player.audio.channels must be > 0 (got {player.audio.channels})")

	release_threshold_raw = detection_raw.get("release_threshold_db", None)
	retrigger_threshold_raw = detection_raw.get("retrigger_threshold_db", None)

	detection = DetectionConfig(
		threshold_db=float(_require(detection_raw, "threshold_db", "detection")),
		hold_seconds=float(_require(detection_raw, "hold_seconds", "detection")),
		warmup_seconds=float(_require(detection_raw, "warmup_seconds", "detection")),
		floor_adaptation=float(_require(detection_raw, "floor_adaptation", "detection")),
		trim_pre_ms=float(detection_raw.get("trim_pre_ms", 0.25)),
		trim_post_ms=float(detection_raw.get("trim_post_ms", 2.0)),
		release_threshold_db=(
			None if release_threshold_raw is None else float(release_threshold_raw)
		),
		retrigger_threshold_db=(
			None if retrigger_threshold_raw is None else float(retrigger_threshold_raw)
		),
		fade_out_ms=float(detection_raw.get("fade_out_ms", 0.0)),
	)

	if not (0.0 < detection.floor_adaptation <= 1.0):
		raise ValueError(
			f"detection.floor_adaptation must be in (0, 1] (got {detection.floor_adaptation})"
		)
	if detection.hold_seconds <= 0:
		raise ValueError(f"detection.hold_seconds must be > 0 (got {detection.hold_seconds})")
	if detection.trim_pre_ms < 0 or detection.trim_post_ms < 0:
		raise ValueError(
			"detection.trim_pre_ms and trim_post_ms must be >= 0 (padding in ms); "
			f"got trim_pre_ms={detection.trim_pre_ms}, trim_post_ms={detection.trim_post_ms}"
		)
	if detection.fade_out_ms < 0:
		raise ValueError(f"detection.fade_out_ms must be >= 0 (got {detection.fade_out_ms})")
	if detection.release_threshold_db is not None and not (
		0.0 < detection.release_threshold_db < detection.threshold_db
	):
		raise ValueError(
			"detection.release_threshold_db must be > 0 and below threshold_db "
			f"(the CLOSE threshold sits under the OPEN threshold); got "
			f"release_threshold_db={detection.release_threshold_db}, "
			f"threshold_db={detection.threshold_db}"
		)
	if detection.retrigger_threshold_db is not None and detection.retrigger_threshold_db <= 0:
		raise ValueError(
			"detection.retrigger_threshold_db must be > 0 (a positive dB rise over the "
			f"decaying tail); got {detection.retrigger_threshold_db}"
		)

	analysis = AnalysisConfig(
		start_bpm=float(analysis_raw.get("start_bpm", 120.0)),
		tempo_min=float(analysis_raw.get("tempo_min", 30.0)),
		tempo_max=float(analysis_raw.get("tempo_max", 300.0)),
	)

	if analysis.tempo_min <= 0 or analysis.tempo_max <= 0:
		raise ValueError(
			f"analysis.tempo_min and tempo_max must be > 0 "
			f"(got {analysis.tempo_min}, {analysis.tempo_max})"
		)
	if analysis.tempo_min >= analysis.tempo_max:
		raise ValueError(
			f"analysis.tempo_min must be < tempo_max "
			f"(got {analysis.tempo_min} >= {analysis.tempo_max})"
		)

	library_raw  = _section(raw, "library", trackers)
	library = LibraryConfig(
		max_memory_mb=float(library_raw.get("max_memory_mb", 100.0)),
		directory=str(library_raw.get("directory", "samples/captures")),
		watch=_require_bool(library_raw, "watch", False, "library"),
	)

	similarity_raw  = _section(raw, "similarity", trackers)
	similarity = SimilarityConfig(
		weight_spectral=float(similarity_raw.get("weight_spectral", 1.0)),
		weight_timbre=float(similarity_raw.get("weight_timbre", 1.0)),
		weight_timbre_delta=float(similarity_raw.get("weight_timbre_delta", 0.5)),
		weight_timbre_onset=float(similarity_raw.get("weight_timbre_onset", 1.0)),
		weight_band_energy=float(similarity_raw.get("weight_band_energy", 1.0)),
	)

	for name, value in [
		("similarity.weight_spectral",      similarity.weight_spectral),
		("similarity.weight_timbre",        similarity.weight_timbre),
		("similarity.weight_timbre_delta",  similarity.weight_timbre_delta),
		("similarity.weight_timbre_onset",  similarity.weight_timbre_onset),
		("similarity.weight_band_energy",   similarity.weight_band_energy),
	]:
		if value < 0.0 or value > 2.0:
			raise ValueError(
				f"{name} must be in [0.0, 2.0] (got {value}). "
				"Set to 0.0 to disable a feature group entirely."
			)

	transform_raw   = _section(raw, "transform", trackers)
	quantize_resolution = int(transform_raw.get("quantize_resolution", 16))

	if quantize_resolution not in {1, 2, 4, 8, 16}:
		raise ValueError(
			f"transform.quantize_resolution must be one of 1, 2, 4, 8, 16 "
			f"(got {quantize_resolution})"
		)

	# The session tempo moved out of the transform section: it now drives sample
	# selection (the duration_beats filter) as well as the quantize processors,
	# so it lives in a top-level `tempo:` section.  The unknown-key sweep only
	# WARNS, so fail loudly here instead of silently ignoring a stale key.
	if "target_bpm" in transform_raw:
		raise ValueError(
			"transform.target_bpm has moved to the top-level 'tempo' section as "
			"tempo.bpm.  Replace `transform:\n  target_bpm: N` in config.yaml "
			"with a top-level `tempo:\n  bpm: N`."
		)

	if "tempo_source" in transform_raw:
		raise ValueError(
			"transform.tempo_source has moved to the top-level 'tempo' section "
			"as tempo.source.  Replace `transform:\n  tempo_source: X` in "
			"config.yaml with a top-level `tempo:\n  source: X`."
		)

	transform = TransformConfig(
		max_memory_mb       = float(transform_raw.get("max_memory_mb", 50.0)),
		auto_pitch          = _require_bool(transform_raw, "auto_pitch", True, "transform"),
		quantize_resolution = quantize_resolution,
		variant_cache_dir   = str(transform_raw.get("variant_cache_dir", "samples/variant-cache") or ""),
		max_disk_mb         = float(transform_raw.get("max_disk_mb",   500.0)),
	)

	tempo_raw = _section(raw, "tempo", trackers)

	tempo_source_raw = tempo_raw.get("source", "config")

	if not isinstance(tempo_source_raw, str):
		raise ValueError(
			f"tempo.source must be a string "
			f"(got {type(tempo_source_raw).__name__}: {tempo_source_raw!r})"
		)

	tempo_source = tempo_source_raw.lower()

	if tempo_source not in SUPPORTED_TEMPO_SOURCES:
		raise ValueError(
			f"tempo.source {tempo_source_raw!r} is not supported.  "
			f"Valid values: {sorted(SUPPORTED_TEMPO_SOURCES)}.  "
			f"'config' uses tempo.bpm; 'midi' follows an incoming MIDI clock."
		)

	tempo = TempoConfig(
		bpm    = float(tempo_raw.get("bpm", 0.0)),
		source = tempo_source,
	)

	# Resolve the memory budget.  Per-cache overrides take precedence for the
	# library and transform memory caches.  The carrier cache (which has no
	# per-cache config key) and the variant disk cache derive from the budget
	# UNCONDITIONALLY — that is the documented 5%/3x contract, so it must hold
	# even when both instrument and transform are given explicit values.
	library_explicit   = "max_memory_mb" in library_raw
	transform_explicit  = "max_memory_mb" in transform_raw
	disk_explicit       = "max_disk_mb" in transform_raw

	global_raw = raw.get("max_memory_mb")
	global_budget: typing.Optional[float] = None

	if global_raw is not None:
		# An explicit global value is both the reported unified budget and the
		# basis for every derivation.
		global_budget    = float(global_raw)
		effective_budget = global_budget
	else:
		# No explicit global: auto-detect the basis.  Report it as the unified
		# budget only when at least one per-cache value actually falls back to
		# it (when both are explicit there is no unifying budget to report, but
		# the carrier/disk caches still need a basis).
		effective_budget = _auto_detect_memory_mb()
		global_budget    = None if (library_explicit and transform_explicit) else effective_budget

	if not library_explicit:
		library = dataclasses.replace(library, max_memory_mb=effective_budget * 0.60)

	if not transform_explicit:
		transform = dataclasses.replace(transform, max_memory_mb=effective_budget * 0.35)

	if not disk_explicit:
		transform = dataclasses.replace(transform, max_disk_mb=effective_budget * 3.0)

	# The carrier cache has no per-cache override path — always 5% of the budget.
	transform = dataclasses.replace(transform, carrier_memory_mb=effective_budget * 0.05)

	# Memory budgets must be positive — a zero or negative value (global or a
	# per-cache override) yields a degenerate cache that evicts everything on the
	# first insert rather than an error the user can see.
	if global_raw is not None and (global_budget is None or global_budget <= 0):
		raise ValueError(f"max_memory_mb must be > 0 (got {global_raw})")
	if library.max_memory_mb <= 0:
		raise ValueError(f"library.max_memory_mb must be > 0 (got {library.max_memory_mb})")
	if transform.max_memory_mb <= 0:
		raise ValueError(f"transform.max_memory_mb must be > 0 (got {transform.max_memory_mb})")

	# --- OSC ---
	osc_raw         = _section(raw, "osc", trackers)
	osc = OscConfig(
		enabled=_require_bool(osc_raw, "enabled", False, "osc"),
		send_host=str(osc_raw.get("send_host", "127.0.0.1")),
		send_port=int(osc_raw.get("send_port", 9000)),
		receive_enabled=_require_bool(osc_raw, "receive_enabled", False, "osc"),
		receive_port=int(osc_raw.get("receive_port", 9002)),
		receive_host=str(osc_raw.get("receive_host", "127.0.0.1")),
	)

	# --- Supervisor ---
	supervisor_raw  = _section(raw, "supervisor", trackers)
	supervisor = SupervisorConfig(
		enabled=_require_bool(supervisor_raw, "enabled", False, "supervisor"),
		port=int(supervisor_raw.get("port", 9003)),
	)

	# --- Ambisonic ---
	ambisonic_raw   = _section(raw, "ambisonic", trackers)
	import subsample.ambisonic
	ambi_decoder = str(ambisonic_raw.get("decoder", "basic"))

	if ambi_decoder not in subsample.ambisonic.SUPPORTED_DECODER_TYPES:
		raise ValueError(
			f"ambisonic.decoder {ambi_decoder!r} is not supported.  "
			f"Valid values: {sorted(subsample.ambisonic.SUPPORTED_DECODER_TYPES)}."
		)

	ambi_max_order = int(ambisonic_raw.get("max_order", 1))

	if ambi_max_order != subsample.ambisonic.AMBISONIC_ORDER_SUPPORTED:
		raise ValueError(
			f"ambisonic.max_order must be 1 (got {ambi_max_order}); higher orders "
			"are reserved for future implementation."
		)

	ambisonic = AmbisonicConfig(
		decoder       = ambi_decoder,
		yaw_degrees   = float(ambisonic_raw.get("yaw_degrees",   0.0)),
		pitch_degrees = float(ambisonic_raw.get("pitch_degrees", 0.0)),
		roll_degrees  = float(ambisonic_raw.get("roll_degrees",  0.0)),
		max_order     = ambi_max_order,
	)

	# Unknown-key sweep: any key present in the YAML but never consulted
	# above is a typo or a removed option — warn by name (never raise, so an
	# old config file keeps working) instead of silently using the default.
	for tracker in trackers:
		unknown = tracker.unknown_keys()

		if unknown:
			_log.warning(
				"config.yaml: unknown key(s) in %s section ignored: %s "
				"— check spelling against config.yaml.default",
				tracker.label, ", ".join(unknown),
			)

	return Config(
		recorder=recorder,
		detection=detection,
		max_memory_mb=global_budget,
		analysis=analysis,
		library=library,
		similarity=similarity,
		player=player,
		transform=transform,
		tempo=tempo,
		osc=osc,
		supervisor=supervisor,
		ambisonic=ambisonic,
	)
