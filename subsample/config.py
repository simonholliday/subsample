"""Configuration loading and validation for Subsample.

Reads config.yaml (user-provided) or falls back to config.yaml.default
(shipped with the package). Exposes typed, frozen dataclasses so every
other module gets IDE completion and mypy coverage.
"""

import dataclasses
import pathlib
import typing

import yaml


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
	"""

	sample_rate: int
	bit_depth: int
	chunk_size: int
	channels: typing.Optional[int] = None
	"""Number of input channels to capture.  None = auto-detect from device."""
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
	None (the default) means use the recorder's sample rate.  Never set
	this higher than the recording sample rate — upsampling adds no
	quality and wastes CPU.  Useful when the output device runs at a lower
	rate than the recorder (e.g. recorder at 96 kHz, output at 48 kHz)."""


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
class ReferenceConfig:

	directory: str
	"""Path to the directory containing reference .analysis.json sidecar files."""


@dataclasses.dataclass(frozen=True)
class SimilarityConfig:

	weight_spectral: float = 1.0
	"""Weight applied to the spectral feature group (11 normalised [0, 1] values:
	flatness, attack, release, centroid, bandwidth, ZCR, harmonic ratio, contrast,
	voiced fraction, log-attack time, spectral flux). Higher weight = spectral shape
	dominates the comparison. Range: 0.0–2.0. Set to 0.0 to disable entirely."""

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


@dataclasses.dataclass(frozen=True)
class InstrumentConfig:

	max_memory_mb: float = 100.0
	"""Maximum audio memory (MB) for in-memory instrument samples.

	When this limit is exceeded the oldest samples are evicted (FIFO) to make
	room. Only in-memory audio is removed; WAV files on disk are never deleted.
	At 44100 Hz 16-bit mono, 100 MB ≈ 19 minutes of audio."""

	directory: typing.Optional[str] = None
	"""Optional path to a directory of pre-analyzed instrument samples to load
	at startup. Each sample requires both a WAV file and an .analysis.json
	sidecar. If omitted, the instrument library starts empty."""

	clean_orphaned_sidecars: bool = False
	"""When True, automatically delete .analysis.json sidecar files whose
	corresponding audio file has been deleted. When False (default), orphaned
	sidecars are skipped with a warning and a one-time hint explaining how to
	enable this feature.

	Note: this only applies to instrument samples. Reference samples are
	intentionally allowed to exist as sidecar-only (no audio required)."""


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
	"""When True, automatically create pitch variants for samples that pass the
	has_stable_pitch() test as they arrive.  Variants are produced in the
	background for every MIDI note within pitch_range_semitones of the detected
	pitch.  Set to False to disable automatic pitch variant production."""

	pitch_range_semitones: int = 12
	"""Number of semitones above and below the detected pitch to generate variants.
	12 = one octave each way = 25 variants per sample (center note included).
	The center variant micro-corrects tuning when the original is slightly off-pitch.
	Range: 0–48.  Set to 0 to disable auto-generation (on-demand variants via
	the player still work)."""

	target_bpm: float = 0.0
	"""Target BPM for automatic time-stretch variants.  0.0 = disabled.
	When > 0, a time-stretch variant is produced for every sample that has
	detected rhythmic content (rhythm.tempo_bpm > 0)."""


@dataclasses.dataclass(frozen=True)
class Config:

	recorder: RecorderConfig
	detection: DetectionConfig
	output: OutputConfig
	analysis: AnalysisConfig = dataclasses.field(default_factory=AnalysisConfig)
	instrument: InstrumentConfig = dataclasses.field(default_factory=InstrumentConfig)
	similarity: SimilarityConfig = dataclasses.field(default_factory=SimilarityConfig)
	player: PlayerConfig = dataclasses.field(default_factory=PlayerConfig)
	transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)
	reference: typing.Optional[ReferenceConfig] = None


def load_config (path: typing.Optional[pathlib.Path] = None) -> Config:

	"""Load configuration from a YAML file, falling back to the package default.

	Searches for config.yaml in the given path, the current working directory,
	and finally the bundled config.yaml.default alongside this module.
	"""

	config_path = _resolve_config_path(path)
	raw = _read_yaml(config_path)

	return _build_config(raw)


def _resolve_config_path (explicit: typing.Optional[pathlib.Path]) -> pathlib.Path:

	"""Find the best available config file.

	Preference order: explicit path → ./config.yaml → package default.
	When an explicit path is provided it must exist; no fallback is attempted.
	"""

	if explicit is not None:
		if explicit.exists():
			return explicit

		raise FileNotFoundError(f"Config file not found: {explicit}")

	cwd_config = pathlib.Path.cwd() / "config.yaml"
	if cwd_config.exists():
		return cwd_config

	# Fall back to the default shipped with the package
	default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
	if default.exists():
		return default

	raise FileNotFoundError(
		"No config.yaml found in current directory and config.yaml.default is missing. "
		"Run: cp config.yaml.default config.yaml"
	)


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

	audio = AudioConfig(
		sample_rate=int(_require(audio_raw, "sample_rate", "recorder.audio")),
		bit_depth=int(_require(audio_raw, "bit_depth", "recorder.audio")),
		chunk_size=int(_require(audio_raw, "chunk_size", "recorder.audio")),
		channels=channels,
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

	player = PlayerConfig(
		audio=PlayerAudioConfig(
			device=player_device,
			bit_depth=player_bit_depth,
			sample_rate=player_sample_rate,
		),
		enabled=bool(player_raw.get("enabled", False)),
		midi_device=player_midi_device,
		virtual_midi_port=player_virtual_midi_port,
		max_polyphony=player_max_polyphony,
	)

	detection = DetectionConfig(
		snr_threshold_db=float(_require(detection_raw, "snr_threshold_db", "detection")),
		hold_time=float(_require(detection_raw, "hold_time", "detection")),
		warmup_seconds=float(_require(detection_raw, "warmup_seconds", "detection")),
		ema_alpha=float(_require(detection_raw, "ema_alpha", "detection")),
		trim_pre_samples=int(detection_raw.get("trim_pre_samples", 0)),
		trim_post_samples=int(detection_raw.get("trim_post_samples", 0)),
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
		directory=instrument_raw.get("directory"),
		clean_orphaned_sidecars=bool(instrument_raw.get("clean_orphaned_sidecars", False)),
	)

	similarity_raw: dict[str, typing.Any] = raw.get("similarity", {})
	similarity = SimilarityConfig(
		weight_spectral     = float(similarity_raw.get("weight_spectral",      1.0)),
		weight_timbre       = float(similarity_raw.get("weight_timbre",        1.0)),
		weight_timbre_delta = float(similarity_raw.get("weight_timbre_delta",  0.5)),
		weight_timbre_onset = float(similarity_raw.get("weight_timbre_onset",  1.0)),
	)

	for name, value in [
		("similarity.weight_spectral",      similarity.weight_spectral),
		("similarity.weight_timbre",        similarity.weight_timbre),
		("similarity.weight_timbre_delta",  similarity.weight_timbre_delta),
		("similarity.weight_timbre_onset",  similarity.weight_timbre_onset),
	]:
		if value < 0.0:
			raise ValueError(
				f"{name} must be >= 0.0 (got {value}). "
				"Set to 0.0 to disable a feature group entirely."
			)

	reference_raw: typing.Optional[dict[str, typing.Any]] = raw.get("reference")
	if reference_raw is not None:
		reference: typing.Optional[ReferenceConfig] = ReferenceConfig(
			directory=str(_require(reference_raw, "directory", "reference")),
		)
	else:
		reference = None

	transform_raw: dict[str, typing.Any] = raw.get("transform", {})
	transform = TransformConfig(
		max_memory_mb         = float(transform_raw.get("max_memory_mb",         50.0)),
		auto_pitch            = bool(transform_raw.get("auto_pitch",             True)),
		pitch_range_semitones = int(transform_raw.get("pitch_range_semitones",   12)),
		target_bpm            = float(transform_raw.get("target_bpm",            0.0)),
	)

	if transform.pitch_range_semitones < 0 or transform.pitch_range_semitones > 48:
		raise ValueError(
			f"transform.pitch_range_semitones ({transform.pitch_range_semitones}) must be in [0, 48]. "
			"Values above 48 semitones (±4 octaves) produce too many variants and stress the cache."
		)

	return Config(
		recorder=recorder,
		detection=detection,
		output=output,
		analysis=analysis,
		instrument=instrument,
		similarity=similarity,
		player=player,
		transform=transform,
		reference=reference,
	)
