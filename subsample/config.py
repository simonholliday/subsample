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
	chunk_size: int
	channels: typing.Optional[int] = None
	"""Number of input channels to capture.  None = auto-detect from device."""
	input: typing.Optional[tuple[int, ...]] = None
	"""Physical input channels to record (0-indexed).  None = first N channels.
	Set via 1-indexed list in config.yaml; converted at config-load time."""
	device: typing.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BufferConfig:

	max_seconds: int


@dataclasses.dataclass(frozen=True)
class RecorderConfig:

	audio: AudioConfig
	buffer: BufferConfig
	enabled: bool = True


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
	it the player will not start. See midi-map.yaml.default for the format
	specification. midi-map-gm-drums.yaml is a complete GM percussion kit."""

	watch_midi_map: bool = False
	"""When True, monitor the midi_map file at runtime for changes and
	reload assignments without restarting.  Enables a live-coding workflow
	where you edit your MIDI map in a text editor and changes take effect
	on the next trigger.  Debounced to handle editors that write multiple
	intermediate saves.

	Requires midi_map to be set."""


@dataclasses.dataclass(frozen=True)
class DetectionConfig:

	snr_threshold_db: float
	hold_time: float
	warmup_seconds: float
	ema_alpha: float
	trim_pre_samples: int
	trim_post_samples: int


@dataclasses.dataclass(frozen=True)
class OutputConfig:

	directory: str
	filename_format: str


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
class InstrumentConfig:

	max_memory_mb: float = 100.0
	"""Maximum audio memory (MB) for in-memory instrument samples.

	When this limit is exceeded the oldest samples are evicted (FIFO) to make
	room. Only in-memory audio is removed; WAV files on disk are never deleted.
	At 44100 Hz 16-bit mono, 100 MB ≈ 19 minutes of audio."""

	directory: str = "samples/captures"
	"""Path to the directory of instrument samples to load at startup.
	Each sample requires both a WAV file and an .analysis.json sidecar.
	Defaults to the captures directory so newly recorded samples are
	immediately available for playback."""

	clean_orphaned_sidecars: bool = True
	"""When True (default), automatically delete .analysis.json sidecar files
	whose corresponding audio file has been deleted. When False, orphaned
	sidecars are skipped with a warning.

	Note: this only applies to instrument samples. Reference samples are
	intentionally allowed to exist as sidecar-only (no audio required)."""

	watch: bool = False
	"""When True, monitor instrument.directory at runtime for new audio
	files and hot-load them into the live instrument library without
	restarting.

	Two detection paths run in parallel:

	1. Sidecar path — watches for .analysis.json sidecar files (fastest:
	   the sidecar signals that both the WAV and analysis are ready).
	2. Audio file path — watches for audio files (.wav, .flac, .aiff,
	   .ogg, .mp3) from any source.  After a grace period to let a
	   sidecar arrive (in case the source is another subsample instance),
	   checks file-size stability, runs the full analysis pipeline,
	   writes a sidecar, and loads the sample.

	Works with multi-instance setups (recorder on one machine, player on
	another via a shared directory) and with audio files from any external
	application.

	Requires instrument.directory to be set and player.enabled to be True."""


@dataclasses.dataclass(frozen=True)
class TransformConfig:

	max_memory_mb: float = 50.0
	"""Maximum memory (MB) for in-memory derivative audio (transform variants).

	Separate from instrument.max_memory_mb — derivatives are disposable and
	regenerated on demand, so they have their own independent budget.
	Eviction strategy: parent-priority FIFO (all variants of the oldest parent
	are evicted together to keep variant sets intact).
	At 44100 Hz float32 stereo, 50 MB ≈ 150 seconds of derivative audio."""

	auto_pitch: bool = True
	"""When True, pitch-shifted variants are produced for each tonal sample
	(those that pass the has_stable_pitch() test) across the full note range
	defined by the MIDI map assignment (e.g. all notes in C-1..G9).
	Set to False to disable all pitch variant production."""

	target_bpm: float = 0.0
	"""Target BPM for automatic time-stretch variants.  0.0 = disabled.
	When > 0, a time-stretch variant is produced for samples that have a
	detected tempo (rhythm.tempo_bpm > 0)."""

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
	Oldest files (by modification time) are evicted when the budget is
	exceeded.  At 44100 Hz float32 stereo, 500 MB ≈ 1500 seconds."""

	carrier_memory_mb: float = 10.0
	"""Memory budget (MB) for vocoder carrier file cache.  Derived from the
	global memory budget (5%) when not overridden."""


@dataclasses.dataclass(frozen=True)
class Config:

	recorder: RecorderConfig
	detection: DetectionConfig
	output: OutputConfig
	max_memory_mb: typing.Optional[float] = None
	"""Total memory budget (MB) for all sample caches.
	None = auto-detect: min(25% of total system RAM, 1024 MB).
	When resolved, the budget is split automatically:
	60% instruments, 35% transform variants, 5% carrier.
	Overridden by explicit per-cache settings in instrument/transform sections."""
	analysis: AnalysisConfig = dataclasses.field(default_factory=AnalysisConfig)
	instrument: InstrumentConfig = dataclasses.field(default_factory=InstrumentConfig)
	similarity: SimilarityConfig = dataclasses.field(default_factory=SimilarityConfig)
	player: PlayerConfig = dataclasses.field(default_factory=PlayerConfig)
	transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)


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
		_log.debug("Loaded config from %s + %s", default_path.name, user_path.name)
	else:
		raw = base
		_log.debug("Loaded config from %s", default_path.name)

	return _build_config(raw)


def _locate_default_config () -> pathlib.Path:

	"""Return the path to the bundled config.yaml.default.

	Raises FileNotFoundError if the file is missing (broken installation).
	"""

	default = pathlib.Path(__file__).parent.parent / "config.yaml.default"

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

	For each key in override: if both values are dicts, recurse; otherwise the
	override value wins (including explicit None / YAML null). Keys present in
	base but absent from override are preserved unchanged. Neither input is
	mutated.
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

	if not isinstance(data, dict):
		raise ValueError(f"Config file {path} must contain a YAML mapping at top level")

	return data


def _require (section: dict[str, typing.Any], key: str, section_name: str) -> typing.Any:

	"""Return section[key], raising a clear ValueError if the key is absent."""

	if key not in section:
		raise ValueError(
			f"Missing required config key '{section_name}.{key}'. "
			f"Check your config.yaml against config.yaml.default."
		)

	return section[key]


def _build_config (raw: dict[str, typing.Any]) -> Config:

	"""Construct the Config dataclass tree from a raw YAML dict."""

	recorder_raw: dict[str, typing.Any] = raw.get("recorder", {})
	audio_raw: dict[str, typing.Any] = recorder_raw.get("audio", {})
	buffer_raw: dict[str, typing.Any] = recorder_raw.get("buffer", {})
	detection_raw: dict[str, typing.Any] = raw.get("detection", {})
	output_raw: dict[str, typing.Any] = raw.get("output", {})
	analysis_raw: dict[str, typing.Any] = raw.get("analysis", {})

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

	audio = AudioConfig(
		sample_rate=int(_require(audio_raw, "sample_rate", "recorder.audio")),
		bit_depth=int(_require(audio_raw, "bit_depth", "recorder.audio")),
		chunk_size=int(_require(audio_raw, "chunk_size", "recorder.audio")),
		channels=channels,
		input=input_channels,
		device=device_raw,
	)

	if audio.bit_depth not in {16, 24, 32}:
		raise ValueError(
			f"Unsupported bit_depth {audio.bit_depth}. "
			"Supported values: 16, 24, 32"
		)
	if audio.sample_rate <= 0:
		raise ValueError(f"recorder.audio.sample_rate must be > 0 (got {audio.sample_rate})")
	if audio.channels is not None and audio.channels <= 0:
		raise ValueError(f"recorder.audio.channels must be > 0 (got {audio.channels})")
	if audio.chunk_size <= 0:
		raise ValueError(f"recorder.audio.chunk_size must be > 0 (got {audio.chunk_size})")

	buffer = BufferConfig(
		max_seconds=int(_require(buffer_raw, "max_seconds", "recorder.buffer")),
	)

	if buffer.max_seconds <= 0:
		raise ValueError(f"recorder.buffer.max_seconds must be > 0 (got {buffer.max_seconds})")

	recorder = RecorderConfig(
		audio=audio,
		buffer=buffer,
		enabled=bool(recorder_raw.get("enabled", True)),
	)

	player_raw: dict[str, typing.Any] = raw.get("player", {})
	player_audio_raw: dict[str, typing.Any] = player_raw.get("audio", {})
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
	if player_limiter_ceiling_db > 0.0:
		raise ValueError(
			f"player.limiter_ceiling_db ({player_limiter_ceiling_db}) must be ≤ 0.0."
		)
	if player_limiter_ceiling_db <= player_limiter_threshold_db:
		raise ValueError(
			f"player.limiter_ceiling_db ({player_limiter_ceiling_db}) must be greater than "
			f"player.limiter_threshold_db ({player_limiter_threshold_db})."
		)

	player = PlayerConfig(
		audio=PlayerAudioConfig(
			device=player_device,
			bit_depth=player_bit_depth,
			sample_rate=player_sample_rate,
			channels=int(player_audio_raw["channels"]) if player_audio_raw.get("channels") is not None else None,
		),
		enabled=bool(player_raw.get("enabled", False)),
		midi_device=player_midi_device,
		virtual_midi_port=player_virtual_midi_port,
		max_polyphony=player_max_polyphony,
		limiter_threshold_db=player_limiter_threshold_db,
		limiter_ceiling_db=player_limiter_ceiling_db,
		midi_map=player_raw.get("midi_map"),
		watch_midi_map=bool(player_raw.get("watch_midi_map", False)),
	)

	if player.audio.channels is not None and player.audio.channels <= 0:
		raise ValueError(f"player.audio.channels must be > 0 (got {player.audio.channels})")

	detection = DetectionConfig(
		snr_threshold_db=float(_require(detection_raw, "snr_threshold_db", "detection")),
		hold_time=float(_require(detection_raw, "hold_time", "detection")),
		warmup_seconds=float(_require(detection_raw, "warmup_seconds", "detection")),
		ema_alpha=float(_require(detection_raw, "ema_alpha", "detection")),
		trim_pre_samples=int(detection_raw.get("trim_pre_samples", 10)),
		trim_post_samples=int(detection_raw.get("trim_post_samples", 90)),
	)

	if not (0.0 < detection.ema_alpha <= 1.0):
		raise ValueError(
			f"detection.ema_alpha must be in (0, 1] (got {detection.ema_alpha})"
		)
	if detection.hold_time <= 0:
		raise ValueError(f"detection.hold_time must be > 0 (got {detection.hold_time})")

	output = OutputConfig(
		directory=str(_require(output_raw, "directory", "output")),
		filename_format=str(_require(output_raw, "filename_format", "output")),
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

	instrument_raw: dict[str, typing.Any] = raw.get("instrument", {})
	instrument = InstrumentConfig(
		max_memory_mb=float(instrument_raw.get("max_memory_mb", 100.0)),
		directory=str(instrument_raw.get("directory", "samples/captures")),
		clean_orphaned_sidecars=bool(instrument_raw.get("clean_orphaned_sidecars", True)),
		watch=bool(instrument_raw.get("watch", False)),
	)

	similarity_raw: dict[str, typing.Any] = raw.get("similarity", {})
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

	transform_raw: dict[str, typing.Any] = raw.get("transform", {})
	quantize_resolution = int(transform_raw.get("quantize_resolution", 16))

	if quantize_resolution not in {1, 2, 4, 8, 16}:
		raise ValueError(
			f"transform.quantize_resolution must be one of 1, 2, 4, 8, 16 "
			f"(got {quantize_resolution})"
		)

	transform = TransformConfig(
		max_memory_mb       = float(transform_raw.get("max_memory_mb", 50.0)),
		auto_pitch          = bool(transform_raw.get("auto_pitch",     True)),
		target_bpm          = float(transform_raw.get("target_bpm",    0.0)),
		quantize_resolution = quantize_resolution,
		variant_cache_dir   = str(transform_raw.get("variant_cache_dir", "samples/variant-cache") or ""),
		max_disk_mb         = float(transform_raw.get("max_disk_mb",   500.0)),
	)

	# Resolve the unified memory budget.  Per-cache overrides take precedence;
	# when both instrument and transform have explicit values the global budget
	# is unused.
	instrument_explicit = "max_memory_mb" in instrument_raw
	transform_explicit  = "max_memory_mb" in transform_raw
	disk_explicit       = "max_disk_mb" in transform_raw

	global_raw = raw.get("max_memory_mb")
	global_budget: typing.Optional[float] = None

	if not (instrument_explicit and transform_explicit):
		global_budget = float(global_raw) if global_raw is not None else _auto_detect_memory_mb()

		if not instrument_explicit:
			instrument = dataclasses.replace(instrument, max_memory_mb=global_budget * 0.60)

		if not transform_explicit:
			transform = dataclasses.replace(transform, max_memory_mb=global_budget * 0.35)

		if not disk_explicit:
			transform = dataclasses.replace(transform, max_disk_mb=global_budget * 3.0)

		transform = dataclasses.replace(transform, carrier_memory_mb=global_budget * 0.05)

	elif global_raw is not None:
		global_budget = float(global_raw)

	return Config(
		recorder=recorder,
		detection=detection,
		output=output,
		max_memory_mb=global_budget,
		analysis=analysis,
		instrument=instrument,
		similarity=similarity,
		player=player,
		transform=transform,
	)
