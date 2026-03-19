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

	sample_rate: int
	bit_depth: int
	channels: int
	chunk_size: int
	device: typing.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BufferConfig:

	max_seconds: int


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
class Config:

	audio: AudioConfig
	buffer: BufferConfig
	detection: DetectionConfig
	output: OutputConfig
	analysis: AnalysisConfig = dataclasses.field(default_factory=AnalysisConfig)
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

	audio_raw: dict[str, typing.Any] = raw.get("audio", {})
	buffer_raw: dict[str, typing.Any] = raw.get("buffer", {})
	detection_raw: dict[str, typing.Any] = raw.get("detection", {})
	output_raw: dict[str, typing.Any] = raw.get("output", {})
	analysis_raw: dict[str, typing.Any] = raw.get("analysis", {})

	audio = AudioConfig(
		sample_rate=int(_require(audio_raw, "sample_rate", "audio")),
		bit_depth=int(_require(audio_raw, "bit_depth", "audio")),
		channels=int(_require(audio_raw, "channels", "audio")),
		chunk_size=int(_require(audio_raw, "chunk_size", "audio")),
		device=audio_raw.get("device"),
	)

	if audio.bit_depth not in {16, 24, 32}:
		raise ValueError(
			f"Unsupported bit_depth {audio.bit_depth}. "
			"Supported values: 16, 24, 32"
		)

	buffer = BufferConfig(
		max_seconds=int(_require(buffer_raw, "max_seconds", "buffer")),
	)

	detection = DetectionConfig(
		snr_threshold_db=float(_require(detection_raw, "snr_threshold_db", "detection")),
		hold_time=float(_require(detection_raw, "hold_time", "detection")),
		warmup_seconds=float(_require(detection_raw, "warmup_seconds", "detection")),
		ema_alpha=float(_require(detection_raw, "ema_alpha", "detection")),
		trim_pre_samples=int(detection_raw.get("trim_pre_samples", 0)),
		trim_post_samples=int(detection_raw.get("trim_post_samples", 0)),
	)

	output = OutputConfig(
		directory=str(_require(output_raw, "directory", "output")),
		filename_format=str(_require(output_raw, "filename_format", "output")),
	)

	analysis = AnalysisConfig(
		start_bpm=float(analysis_raw.get("start_bpm", 120.0)),
		tempo_min=float(analysis_raw.get("tempo_min", 30.0)),
		tempo_max=float(analysis_raw.get("tempo_max", 300.0)),
	)

	reference_raw: typing.Optional[dict[str, typing.Any]] = raw.get("reference")
	if reference_raw is not None:
		reference: typing.Optional[ReferenceConfig] = ReferenceConfig(
			directory=str(_require(reference_raw, "directory", "reference")),
		)
	else:
		reference = None

	return Config(
		audio=audio,
		buffer=buffer,
		detection=detection,
		output=output,
		analysis=analysis,
		reference=reference,
	)
