"""Tests for subsample.config config loading and validation."""

import dataclasses
import logging
import pathlib
import textwrap
import typing
import unittest.mock

import pytest
import yaml

import subsample.config


_DEFAULT_CONFIG_PATH = subsample.config._locate_default_config()


def _load_with (
	tmp_path: pathlib.Path,
	detection: typing.Optional[dict] = None,
	audio: typing.Optional[dict] = None,
) -> subsample.config.Config:

	"""Write a complete config with optional detection/audio overrides and load it."""

	cfg: dict = {
		"recorder": {
			"audio": {"sample_rate": 48000, "bit_depth": 16, "channels": 1, "buffer_frames": 512},
			"buffer": {"max_seconds": 60},
		},
		"detection": {
			"threshold_db": 12.0, "hold_seconds": 0.5,
			"warmup_seconds": 1.0, "floor_adaptation": 0.1,
		},
	}
	if detection:
		cfg["detection"].update(detection)
	if audio:
		cfg["recorder"]["audio"].update(audio)

	path = tmp_path / "config.yaml"
	path.write_text(yaml.safe_dump(cfg))
	return subsample.config.load_config(path)


class TestLoadDefault:

	"""Loads of the SHIPPED subsample/data/config.yaml.default.

	The test_default_* methods pin the shipped default values — changing a
	default in config.yaml.default must update the matching test here (and
	vice versa), so drift between the two is always a test failure.
	"""

	def test_default_config_ships_inside_the_package (self) -> None:
		"""The bundled default must live INSIDE the subsample package directory.

		It ships as package data ([tool.setuptools.package-data] in
		pyproject.toml); a repo-root location would vanish from non-editable
		installs and crash startup. Locks review finding M25's fix in place.
		"""

		default = subsample.config._locate_default_config()

		assert default.parent == subsample.config.data_dir()
		assert subsample.config.data_dir().parent == pathlib.Path(subsample.config.__file__).parent
		assert default.is_file()

	def test_data_dir_ships_the_product_assets (self) -> None:
		"""subsample/data/ holds everything --init scaffolds a project from: both
		MIDI maps and the GM reference sidecars with their credits file.  This
		checks the files exist in the SOURCE tree (which data_dir() resolves to in
		a checkout) — the same files the package-data globs pull into a wheel; it
		does not itself build or inspect a wheel."""

		data = subsample.config.data_dir()

		assert (data / "midi-map-gm-drums.yaml").is_file()
		assert (data / "midi-map.yaml.default").is_file()
		assert (data / "reference" / "CREDITS.md").is_file()
		sidecars = list((data / "reference").glob("*.analysis.json"))
		assert len(sidecars) >= 40

	def test_shipped_reference_sidecars_are_version_current (self) -> None:
		"""Every shipped GM reference sidecar must carry the CURRENT
		ANALYSIS_VERSION.

		A stale sidecar with no WAV beside it cannot self-heal — it would be
		silently skipped, emptying every pip user's reference library. This is
		the release tripwire: after an ANALYSIS_VERSION bump, re-run
		scripts/extract_gm_drums.py to regenerate the shipped copies.
		"""

		import json

		import subsample.analysis

		sidecars = sorted((subsample.config.data_dir() / "reference").glob("*.analysis.json"))
		assert sidecars

		for path in sidecars:
			with path.open(encoding="utf-8") as fh:
				version = json.load(fh)["analysis_version"]
			assert version == subsample.analysis.ANALYSIS_VERSION, (
				f"{path.name} is at analysis_version {version!r}, current is "
				f"{subsample.analysis.ANALYSIS_VERSION!r} — re-run scripts/extract_gm_drums.py"
			)

	def test_loads_default_config (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert isinstance(cfg, subsample.config.Config)
		assert isinstance(cfg.recorder, subsample.config.RecorderConfig)
		assert isinstance(cfg.recorder.audio, subsample.config.AudioConfig)
		assert isinstance(cfg.recorder.buffer, subsample.config.BufferConfig)
		assert isinstance(cfg.player, subsample.config.PlayerConfig)
		assert isinstance(cfg.detection, subsample.config.DetectionConfig)
		assert isinstance(cfg.analysis, subsample.config.AnalysisConfig)

	def test_default_audio_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.audio.sample_rate == 44100
		assert cfg.recorder.audio.bit_depth == 16
		# channels is commented out in the shipped default → None = auto-detect
		# the device's channel count at startup.
		assert cfg.recorder.audio.channels is None
		assert cfg.recorder.audio.input is None
		assert cfg.recorder.audio.buffer_frames == 512

	def test_default_buffer_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.buffer.max_seconds == 60

	def test_default_recorder_enabled (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.enabled is True

	def test_default_player_disabled (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.player.enabled is False
		assert cfg.player.audio.device is None
		assert cfg.player.midi_device is None
		assert cfg.player.virtual_midi_port is None
		assert cfg.player.max_polyphony == 8
		assert cfg.player.limiter_threshold_db == -1.5
		assert cfg.player.limiter_ceiling_db == -0.1
		assert cfg.player.watch_midi_map is False

	def test_default_detection_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.detection.threshold_db == 12.0
		assert cfg.detection.hold_seconds == 0.5
		assert cfg.detection.warmup_seconds == 1.0
		assert cfg.detection.floor_adaptation == 0.1

	def test_default_output_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.directory == "samples/captures"
		assert cfg.recorder.filename_format == "%Y-%m-%d_%H-%M-%S-%3f"

	def test_default_analysis_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.analysis.start_bpm == 120.0
		assert cfg.analysis.tempo_min == 30.0
		assert cfg.analysis.tempo_max == 300.0

	def test_default_instrument_values (self) -> None:
		# TEST DEPENDENCY: config.yaml.default library section defaults
		# library.max_memory_mb is derived from auto-detect (60% of global).
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.library.max_memory_mb > 0
		assert cfg.library.directory == "samples/captures"

	def test_default_similarity_values (self) -> None:
		# Similarity section is commented-out in config.yaml.default so defaults
		# come from SimilarityConfig class defaults.
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.similarity.weight_spectral     == 1.0
		assert cfg.similarity.weight_timbre       == 1.0
		assert cfg.similarity.weight_timbre_delta == 0.5
		assert cfg.similarity.weight_timbre_onset == 1.0

	def test_analysis_defaults_when_section_absent (self, tmp_path: pathlib.Path) -> None:
		"""A config.yaml without an analysis section should use class defaults."""
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 1024
			  buffer:
			    max_seconds: 60
			detection:
			  threshold_db: 6.0
			  hold_seconds: 0.5
			  warmup_seconds: 3.0
			  floor_adaptation: 0.01
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)
		cfg = subsample.config.load_config(config_file)

		assert cfg.analysis.start_bpm == 120.0
		assert cfg.analysis.tempo_min == 30.0
		assert cfg.analysis.tempo_max == 300.0

	def test_default_trim_padding_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.detection.trim_pre_ms == 0.25
		assert cfg.detection.trim_post_ms == 2.0


class TestLoadCustomConfig:

	def test_unknown_keys_warn_by_name (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""A typo'd key must warn with its name and section — previously it
		was silently ignored and the default silently used."""

		config_file = tmp_path / "config.yaml"
		config_file.write_text(textwrap.dedent("""\
			player:
			  enabled: false
			  audio:
			    max_polyphony: 16
			transform:
			  target_bmp: 120.0
		"""))

		with caplog.at_level(logging.WARNING, logger="subsample.config"):
			subsample.config.load_config(config_file)

		messages = " | ".join(r.message for r in caplog.records)
		assert "max_polyphony" in messages and "player.audio" in messages
		assert "target_bmp" in messages and "transform" in messages

	def test_default_config_loads_without_unknown_key_warnings (
		self, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""The shipped default file must never trip the unknown-key sweep."""

		with caplog.at_level(logging.WARNING, logger="subsample.config"):
			subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert not [r for r in caplog.records if "unknown key" in r.message]

	def test_loads_custom_yaml (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  directory: /tmp/my_samples
			  audio:
			    sample_rate: 48000
			    bit_depth: 24
			    channels: 2
			    buffer_frames: 2048
			  buffer:
			    max_seconds: 30
			detection:
			  threshold_db: 10.0
			  hold_seconds: 1.0
			  warmup_seconds: 3.0
			  floor_adaptation: 0.05
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.sample_rate == 48000
		assert cfg.recorder.audio.bit_depth == 24
		assert cfg.recorder.audio.channels == 2
		assert cfg.recorder.buffer.max_seconds == 30
		assert cfg.detection.threshold_db == 10.0
		assert cfg.recorder.directory == "/tmp/my_samples"

	def test_recorder_enabled_flag (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  enabled: false
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.enabled is False

	def test_player_enabled_flag (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			  enabled: true
			  audio:
			    device: "Focusrite Output"
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.player.enabled is True
		assert cfg.player.audio.device == "Focusrite Output"

	def test_player_midi_device_custom (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			  enabled: true
			  midi_device: "Launchpad"
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.player.midi_device == "Launchpad"

	def test_player_midi_device_non_string_raises (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			  midi_device: 42
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		with pytest.raises(ValueError, match="player.midi_device"):
			subsample.config.load_config(config_file)

	def test_player_virtual_midi_port_custom (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			  enabled: true
			  virtual_midi_port: "Subsample Virtual MIDI"
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.player.virtual_midi_port == "Subsample Virtual MIDI"
		assert cfg.player.midi_device is None

	def test_player_virtual_midi_port_non_string_raises (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			  virtual_midi_port: 99
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		with pytest.raises(ValueError, match="player.virtual_midi_port"):
			subsample.config.load_config(config_file)

	def test_config_is_frozen (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		with pytest.raises(dataclasses.FrozenInstanceError):
			cfg.recorder = subsample.config.RecorderConfig(  # type: ignore[misc]
				audio=subsample.config.AudioConfig(
					sample_rate=99, bit_depth=16, channels=1, buffer_frames=1024
				),
				buffer=subsample.config.BufferConfig(max_seconds=60),
			)

	def test_missing_file_raises (self, tmp_path: pathlib.Path) -> None:
		nonexistent = tmp_path / "does_not_exist.yaml"

		with pytest.raises(FileNotFoundError):
			subsample.config.load_config(nonexistent)

	def test_explicit_path_expands_tilde (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""~/... explicit paths work even when the shell didn't expand them
		(supervisor/systemd invocations pass the literal tilde through)."""

		monkeypatch.setenv("HOME", str(tmp_path))
		(tmp_path / "config.yaml").write_text("tempo:\n  bpm: 93.0\n")

		cfg = subsample.config.load_config(pathlib.Path("~/config.yaml"))

		assert cfg.tempo.bpm == 93.0

	def test_similarity_custom_weights (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
			similarity:
			  weight_spectral: 2.0
			  weight_timbre: 0.0
			  weight_timbre_delta: 1.5
			  weight_timbre_onset: 0.0
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)
		cfg = subsample.config.load_config(config_file)

		assert cfg.similarity.weight_spectral     == 2.0
		assert cfg.similarity.weight_timbre       == 0.0
		assert cfg.similarity.weight_timbre_delta == 1.5
		assert cfg.similarity.weight_timbre_onset == 0.0

	def test_similarity_negative_weight_raises (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
			similarity:
			  weight_spectral: -1.0
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		with pytest.raises(ValueError, match="weight_spectral"):
			subsample.config.load_config(config_file)

	def _minimal_yaml (self, channels_line: str = "    channels: 1") -> str:
		"""Return a minimal valid config YAML, with a custom channels line."""
		return textwrap.dedent(f"""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			{channels_line}
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			detection:
			  threshold_db: 6.0
			  hold_seconds: 0.5
			  warmup_seconds: 3.0
			  floor_adaptation: 0.01
		""")

	def test_channels_explicit (self, tmp_path: pathlib.Path) -> None:
		"""An explicit channels value is preserved exactly."""
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._minimal_yaml("    channels: 2"))
		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.channels == 2

	def test_channels_null_yields_none (self, tmp_path: pathlib.Path) -> None:
		"""channels: null in YAML resolves to None (auto-detect at startup)."""
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._minimal_yaml("    channels: null"))
		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.channels is None

	def test_channels_omitted_auto_detects (self, tmp_path: pathlib.Path) -> None:
		"""Omitting channels inherits the shipped default, which is commented
		out → None = auto-detect the device's channel count at startup."""
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._minimal_yaml(""))
		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.channels is None

	def test_channels_zero_raises (self, tmp_path: pathlib.Path) -> None:
		"""channels: 0 should raise ValueError at config-load time."""
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._minimal_yaml("    channels: 0"))

		with pytest.raises(ValueError, match="channels must be > 0"):
			subsample.config.load_config(config_file)

	def test_invalid_bit_depth_raises (self, tmp_path: pathlib.Path) -> None:
		"""Loading a config with unsupported bit_depth should raise ValueError."""
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 8
			    channels: 1
			    buffer_frames: 1024
			  buffer:
			    max_seconds: 60
			detection:
			  threshold_db: 6.0
			  hold_seconds: 0.5
			  warmup_seconds: 2.0
			  floor_adaptation: 0.01
			""")
		config_file = tmp_path / "bad_config.yaml"
		config_file.write_text(yaml_content)

		with pytest.raises(ValueError, match="Unsupported bit_depth"):
			subsample.config.load_config(config_file)

	def _player_yaml (self, player_section: str) -> str:
		"""Return a minimal valid config with a custom player section."""
		return textwrap.dedent(f"""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			{player_section}
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")

	def test_player_max_polyphony_custom (self, tmp_path: pathlib.Path) -> None:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._player_yaml("  max_polyphony: 16"))

		cfg = subsample.config.load_config(config_file)

		assert cfg.player.max_polyphony == 16

	def test_player_max_polyphony_zero_raises (self, tmp_path: pathlib.Path) -> None:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._player_yaml("  max_polyphony: 0"))

		with pytest.raises(ValueError, match="max_polyphony"):
			subsample.config.load_config(config_file)

	def test_player_max_polyphony_too_high_raises (self, tmp_path: pathlib.Path) -> None:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._player_yaml("  max_polyphony: 65"))

		with pytest.raises(ValueError, match="max_polyphony"):
			subsample.config.load_config(config_file)

	def test_player_limiter_ceiling_not_above_zero_raises (self, tmp_path: pathlib.Path) -> None:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._player_yaml("  limiter_ceiling_db: 0.5"))

		with pytest.raises(ValueError, match="limiter_ceiling_db"):
			subsample.config.load_config(config_file)

	def test_player_limiter_ceiling_not_above_threshold_raises (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    buffer_frames: 512
			  buffer:
			    max_seconds: 60
			player:
			  limiter_threshold_db: -1.5
			  limiter_ceiling_db: -3.0
			detection:
			  threshold_db: 12.0
			  hold_seconds: 0.5
			  warmup_seconds: 1.0
			  floor_adaptation: 0.1
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		with pytest.raises(ValueError, match="limiter_ceiling_db"):
			subsample.config.load_config(config_file)

	def test_player_limiter_threshold_out_of_range_raises (self, tmp_path: pathlib.Path) -> None:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._player_yaml("  limiter_threshold_db: -20.0"))

		with pytest.raises(ValueError, match="limiter_threshold_db"):
			subsample.config.load_config(config_file)


class TestDeepMerge:

	def test_flat_override (self) -> None:
		"""Override value replaces base value for the same key."""
		base = {"a": 1, "b": 2}
		override = {"b": 99}
		assert subsample.config._deep_merge(base, override) == {"a": 1, "b": 99}

	def test_nested_override_preserves_unrelated_keys (self) -> None:
		"""Nested dict override only replaces the specified sub-key."""
		base = {"section": {"a": 1, "b": 2}}
		override = {"section": {"b": 99}}
		assert subsample.config._deep_merge(base, override) == {"section": {"a": 1, "b": 99}}

	def test_override_adds_new_key (self) -> None:
		"""Keys in override that are not in base are added to the result."""
		base = {"a": 1}
		override = {"b": 2}
		result = subsample.config._deep_merge(base, override)
		assert result == {"a": 1, "b": 2}

	def test_does_not_mutate_base (self) -> None:
		base: dict[str, object] = {"a": 1, "nested": {"x": 10}}
		override: dict[str, object] = {"a": 2, "nested": {"x": 20}}
		_ = subsample.config._deep_merge(base, override)
		assert base == {"a": 1, "nested": {"x": 10}}

	def test_does_not_mutate_override (self) -> None:
		base: dict[str, object] = {"a": 1}
		override: dict[str, object] = {"b": 2}
		_ = subsample.config._deep_merge(base, override)
		assert override == {"b": 2}

	def test_override_null_wins (self) -> None:
		"""Explicit None in override replaces a non-None scalar base value."""
		base: dict[str, object] = {"channels": 1}
		override: dict[str, object] = {"channels": None}
		result = subsample.config._deep_merge(base, override)
		assert result["channels"] is None

	def test_null_override_of_dict_preserves_base (self) -> None:
		"""A None override for a base dict (empty YAML section) preserves base defaults.

		When all children of a YAML section are commented out, the parser yields
		None for that key.  The merge must treat this as 'no override' rather than
		clobbering the base dict, so that config.yaml.default values still apply.
		"""
		base: dict[str, object] = {"buffer": {"max_seconds": 60}}
		override: dict[str, object] = {"buffer": None}
		result = subsample.config._deep_merge(base, override)
		assert result == {"buffer": {"max_seconds": 60}}

	def test_deeply_nested_merge (self) -> None:
		"""Merge works correctly across multiple levels of nesting."""
		base = {"recorder": {"audio": {"sample_rate": 44100, "bit_depth": 16}}}
		override = {"recorder": {"audio": {"sample_rate": 48000}}}
		result = subsample.config._deep_merge(base, override)
		assert result == {"recorder": {"audio": {"sample_rate": 48000, "bit_depth": 16}}}


class TestConfigCascade:

	def test_minimal_override_loads_successfully (self, tmp_path: pathlib.Path) -> None:
		"""A config.yaml with only a device override loads with all other defaults."""
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    device: "Test Mic"
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.device == "Test Mic"
		assert cfg.recorder.audio.sample_rate == 44100
		assert cfg.recorder.audio.bit_depth == 16
		assert cfg.detection.threshold_db == 12.0
		assert cfg.recorder.directory == "samples/captures"

	def test_partial_section_inherits_sibling_keys (self, tmp_path: pathlib.Path) -> None:
		"""Overriding one key in a section leaves sibling keys at their defaults."""
		yaml_content = textwrap.dedent("""\
			detection:
			  threshold_db: 6.0
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.detection.threshold_db == 6.0
		assert cfg.detection.hold_seconds == 0.5
		assert cfg.detection.warmup_seconds == 1.0
		assert cfg.detection.floor_adaptation == 0.1

	def test_default_path_explicit_no_double_merge (self) -> None:
		"""Passing the default config path explicitly loads correctly without double-merging."""
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)
		assert cfg.recorder.audio.sample_rate == 44100
		assert cfg.detection.trim_pre_ms == 0.25

	def test_channels_null_override_yields_none (self, tmp_path: pathlib.Path) -> None:
		"""Explicitly setting channels: null in user config overrides the default (1)."""
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    channels: null
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.channels is None


class TestInputRouting:

	"""Tests for recorder.audio.input channel routing."""

	def _make_config (self, tmp_path: pathlib.Path, audio_extra: str) -> subsample.config.Config:
		"""Helper: write a config.yaml with the given audio section extras."""
		yaml_content = (
			"recorder:\n"
			"  audio:\n"
			"    sample_rate: 44100\n"
			"    bit_depth: 16\n"
			"    buffer_frames: 512\n"
			f"    {audio_extra}\n"
			"  buffer:\n"
			"    max_seconds: 10\n"
			"detection:\n"
			"  threshold_db: 6\n"
			"  hold_seconds: 0.5\n"
			"  warmup_seconds: 2\n"
			"  floor_adaptation: 0.1\n"
		)
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)
		return subsample.config.load_config(config_file)

	def test_input_parsed_and_converted (self, tmp_path: pathlib.Path) -> None:
		"""1-indexed [3, 4] in YAML becomes 0-indexed (2, 3)."""
		cfg = self._make_config(tmp_path, "input: [3, 4]\n    channels: 2")
		assert cfg.recorder.audio.input == (2, 3)
		assert cfg.recorder.audio.channels == 2

	def test_input_infers_channels (self, tmp_path: pathlib.Path) -> None:
		"""When channels is null, it is inferred from input length."""
		cfg = self._make_config(tmp_path, "input: [1, 2, 5]\n    channels:")
		assert cfg.recorder.audio.channels == 3
		assert cfg.recorder.audio.input == (0, 1, 4)

	def test_input_single_channel (self, tmp_path: pathlib.Path) -> None:
		"""Single input selects one channel."""
		cfg = self._make_config(tmp_path, "input: [5]\n    channels:")
		assert cfg.recorder.audio.channels == 1
		assert cfg.recorder.audio.input == (4,)

	def test_input_none_by_default (self, tmp_path: pathlib.Path) -> None:
		"""No input key → None."""
		cfg = self._make_config(tmp_path, "channels: 1")
		assert cfg.recorder.audio.input is None

	def test_length_mismatch_raises (self, tmp_path: pathlib.Path) -> None:
		"""input length != channels raises ValueError."""
		with pytest.raises(ValueError, match="does not match"):
			self._make_config(tmp_path, "input: [1, 2, 3]\n    channels: 2")

	def test_duplicates_raise (self, tmp_path: pathlib.Path) -> None:
		"""Duplicate channels raise ValueError."""
		with pytest.raises(ValueError, match="duplicates"):
			self._make_config(tmp_path, "input: [1, 1]")

	def test_zero_raises (self, tmp_path: pathlib.Path) -> None:
		"""0 is invalid (1-indexed)."""
		with pytest.raises(ValueError, match="positive integers"):
			self._make_config(tmp_path, "input: [0, 1]")

	def test_negative_raises (self, tmp_path: pathlib.Path) -> None:
		"""Negative values are invalid."""
		with pytest.raises(ValueError, match="positive integers"):
			self._make_config(tmp_path, "input: [-1, 2]")

	def test_empty_raises (self, tmp_path: pathlib.Path) -> None:
		"""Empty list raises ValueError."""
		with pytest.raises(ValueError, match="non-empty"):
			self._make_config(tmp_path, "input: []")


class TestMemoryBudget:

	"""Tests for the unified memory budget and auto-detect logic."""

	def _make_config (self, tmp_path: pathlib.Path, extra: str) -> subsample.config.Config:
		"""Helper: write a config.yaml with the given top-level extras."""
		yaml_content = (
			f"{extra}\n"
			"recorder:\n"
			"  audio:\n"
			"    sample_rate: 44100\n"
			"    bit_depth: 16\n"
			"    buffer_frames: 512\n"
			"    channels: 1\n"
			"  buffer:\n"
			"    max_seconds: 10\n"
			"detection:\n"
			"  threshold_db: 6\n"
			"  hold_seconds: 0.5\n"
			"  warmup_seconds: 2\n"
			"  floor_adaptation: 0.1\n"
		)
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)
		return subsample.config.load_config(config_file)

	def test_explicit_global_splits_correctly (self, tmp_path: pathlib.Path) -> None:
		"""max_memory_mb: 200 → instrument 120, transform 70, carrier 10."""
		cfg = self._make_config(tmp_path, "max_memory_mb: 200")
		assert cfg.library.max_memory_mb == pytest.approx(120.0)
		assert cfg.transform.max_memory_mb == pytest.approx(70.0)
		assert cfg.transform.carrier_memory_mb == pytest.approx(10.0)
		assert cfg.transform.max_disk_mb == pytest.approx(600.0)

	def test_per_cache_overrides_global (self, tmp_path: pathlib.Path) -> None:
		"""Explicit per-cache values take precedence over global."""
		cfg = self._make_config(
			tmp_path,
			"max_memory_mb: 200\n"
			"library:\n"
			"  max_memory_mb: 300\n",
		)
		assert cfg.library.max_memory_mb == 300.0
		assert cfg.transform.max_memory_mb == pytest.approx(70.0)

	def test_both_per_cache_ignores_global (self, tmp_path: pathlib.Path) -> None:
		"""When both per-cache values are set, global budget is not applied."""
		cfg = self._make_config(
			tmp_path,
			"max_memory_mb: 200\n"
			"library:\n"
			"  max_memory_mb: 300\n"
			"transform:\n"
			"  max_memory_mb: 80\n",
		)
		assert cfg.library.max_memory_mb == 300.0
		assert cfg.transform.max_memory_mb == 80.0

	def test_both_per_cache_still_derives_carrier_and_disk_from_global (self, tmp_path: pathlib.Path) -> None:
		"""Code-review regression: carrier (5%) and disk (3x) derive from the
		global budget even when BOTH instrument and transform memory are
		explicit — they used to be silently left at their dataclass defaults."""
		cfg = self._make_config(
			tmp_path,
			"max_memory_mb: 1000\n"
			"library:\n"
			"  max_memory_mb: 300\n"
			"transform:\n"
			"  max_memory_mb: 80\n",
		)
		assert cfg.transform.carrier_memory_mb == pytest.approx(1000.0 * 0.05)   # 50, not the 10 default
		assert cfg.transform.max_disk_mb == pytest.approx(1000.0 * 3.0)          # 3000, not the 500 default

	def test_scalar_section_raises_clear_error (self, tmp_path: pathlib.Path) -> None:
		"""A section given a scalar (indentation typo) raises a keyed ValueError."""
		config_file = tmp_path / "config.yaml"
		config_file.write_text("recorder: oops\n")
		with pytest.raises(ValueError, match="recorder"):
			subsample.config.load_config(config_file)

	def test_disk_override_wins (self, tmp_path: pathlib.Path) -> None:
		"""Explicit max_disk_mb overrides the 3x global default."""
		cfg = self._make_config(
			tmp_path,
			"max_memory_mb: 200\n"
			"transform:\n"
			"  max_disk_mb: 1000\n",
		)
		assert cfg.transform.max_disk_mb == 1000.0
		# Transform memory still comes from global.
		assert cfg.transform.max_memory_mb == pytest.approx(70.0)

	@unittest.mock.patch("subsample.config._auto_detect_memory_mb", return_value=512.0)
	def test_auto_detect_2gb_system (self, mock_detect: unittest.mock.MagicMock, tmp_path: pathlib.Path) -> None:
		"""On a 2 GB system, auto-detect → 512 MB budget."""
		cfg = self._make_config(tmp_path, "")
		assert cfg.library.max_memory_mb == pytest.approx(512.0 * 0.60)
		assert cfg.transform.max_memory_mb == pytest.approx(512.0 * 0.35)
		assert cfg.transform.carrier_memory_mb == pytest.approx(512.0 * 0.05)
		assert cfg.transform.max_disk_mb == pytest.approx(512.0 * 3.0)

	@unittest.mock.patch("subsample.config._auto_detect_memory_mb", return_value=1024.0)
	def test_auto_detect_16gb_system (self, mock_detect: unittest.mock.MagicMock, tmp_path: pathlib.Path) -> None:
		"""On a 16 GB+ system, auto-detect → 1024 MB cap."""
		cfg = self._make_config(tmp_path, "")
		assert cfg.library.max_memory_mb == pytest.approx(1024.0 * 0.60)
		assert cfg.transform.max_memory_mb == pytest.approx(1024.0 * 0.35)

	def test_auto_detect_returns_positive (self) -> None:
		"""_auto_detect_memory_mb returns a positive value on this system."""
		result = subsample.config._auto_detect_memory_mb()
		assert result > 0
		assert result <= 1024.0

	@unittest.mock.patch("os.sysconf", side_effect=AttributeError)
	def test_auto_detect_fallback (self, mock_sysconf: unittest.mock.MagicMock) -> None:
		"""When os.sysconf is unavailable, falls back to 160 MB."""
		result = subsample.config._auto_detect_memory_mb()
		assert result == 160.0


class TestSupervisorConfig:

	def test_default_config_has_supervisor_disabled (self) -> None:
		"""Default config produces SupervisorConfig with enabled=False."""

		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert isinstance(cfg.supervisor, subsample.config.SupervisorConfig)
		assert cfg.supervisor.enabled is False
		assert cfg.supervisor.port == 9003

	def test_explicit_supervisor_yaml_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Explicit supervisor YAML section is parsed correctly."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nsupervisor:\n"
				"  enabled: true\n"
				"  port: 8888\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.supervisor.enabled is True
		assert cfg.supervisor.port == 8888


class TestAmbisonicConfig:

	def test_default_config_has_ambisonic_disabled (self) -> None:
		"""Default config produces AmbisonicConfig with basic decoder, no rotation."""

		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert isinstance(cfg.ambisonic, subsample.config.AmbisonicConfig)
		assert cfg.ambisonic.decoder       == "basic"
		assert cfg.ambisonic.yaw_degrees   == 0.0
		assert cfg.ambisonic.pitch_degrees == 0.0
		assert cfg.ambisonic.roll_degrees  == 0.0
		assert cfg.ambisonic.max_order     == 1

		# Default recorder.audio.ambisonic_format is None.
		assert cfg.recorder.audio.ambisonic_format is None

	def test_explicit_ambisonic_yaml_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Explicit ambisonic section + ambisonic_format field are parsed."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    channels: 4\n"
				"    ambisonic_format: a_nt_sf1\n"
				"\nambisonic:\n"
				"  decoder: max_re\n"
				"  yaw_degrees: 30.0\n"
				"  pitch_degrees: -10.0\n"
				"  roll_degrees: 5.0\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.audio.ambisonic_format == "a_nt_sf1"
		assert cfg.recorder.audio.channels         == 4
		assert cfg.ambisonic.decoder               == "max_re"
		assert cfg.ambisonic.yaw_degrees           == 30.0
		assert cfg.ambisonic.pitch_degrees         == -10.0
		assert cfg.ambisonic.roll_degrees          == 5.0

	def test_invalid_ambisonic_format_rejected (self, tmp_path: pathlib.Path) -> None:
		"""An unknown ambisonic_format raises a clear ValueError."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    channels: 4\n"
				"    ambisonic_format: not_a_format\n"
			)

		with pytest.raises(ValueError, match="ambisonic_format"):
			subsample.config.load_config(user_config)

	def test_ambisonic_format_requires_four_channels (self, tmp_path: pathlib.Path) -> None:
		"""Setting ambisonic_format with channels != 4 raises ValueError."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    channels: 2\n"
				"    ambisonic_format: a_generic\n"
			)

		with pytest.raises(ValueError, match="channels: 4"):
			subsample.config.load_config(user_config)

	def test_ambisonic_format_requires_explicit_channels (self, tmp_path: pathlib.Path) -> None:
		"""Setting ambisonic_format with channels: null (auto-detect) raises
		ValueError — the mismatch must be caught at config-load time, not
		deferred to the first capture on a worker thread.
		"""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    channels: null\n"
				"    ambisonic_format: a_generic\n"
			)

		with pytest.raises(ValueError, match="auto-detect is not accepted"):
			subsample.config.load_config(user_config)

	def test_invalid_decoder_rejected (self, tmp_path: pathlib.Path) -> None:
		"""Unknown ambisonic.decoder value raises ValueError."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nambisonic:\n"
				"  decoder: telepathy\n"
			)

		with pytest.raises(ValueError, match="ambisonic.decoder"):
			subsample.config.load_config(user_config)

	def test_higher_max_order_rejected (self, tmp_path: pathlib.Path) -> None:
		"""max_order > 1 raises ValueError (higher orders not yet implemented)."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nambisonic:\n"
				"  max_order: 2\n"
			)

		with pytest.raises(ValueError, match="max_order must be 1"):
			subsample.config.load_config(user_config)


class TestAudioFormatConfig:

	"""Tests for recorder.audio.audio_format parsing and validation."""

	def test_default_is_wav (self) -> None:
		"""With no override, audio_format resolves to 'wav'."""

		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.audio.audio_format == "wav"

	def test_explicit_flac_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Setting audio_format: flac in YAML is parsed correctly."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    bit_depth: 16\n"
				"    audio_format: flac\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.audio.audio_format == "flac"

	def test_case_insensitive (self, tmp_path: pathlib.Path) -> None:
		"""Uppercase values are normalised to lowercase."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    bit_depth: 16\n"
				"    audio_format: FLAC\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.audio.audio_format == "flac"

	def test_invalid_value_rejected (self, tmp_path: pathlib.Path) -> None:
		"""Anything other than wav/flac raises ValueError."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    audio_format: mp3\n"
			)

		with pytest.raises(ValueError, match="audio_format"):
			subsample.config.load_config(user_config)

	def test_flac_with_32bit_rejected (self, tmp_path: pathlib.Path) -> None:
		"""audio_format: flac combined with bit_depth: 32 is rejected at
		config-load time.  FLAC's stable subtypes only cover 16/24-bit;
		surfacing the mismatch here prevents a worker-thread failure on
		the first capture.
		"""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    bit_depth: 32\n"
				"    audio_format: flac\n"
			)

		with pytest.raises(ValueError, match="bit_depth of 16 or 24"):
			subsample.config.load_config(user_config)

	def test_flac_with_16bit_accepted (self, tmp_path: pathlib.Path) -> None:
		"""audio_format: flac with bit_depth: 16 is a valid combination."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    bit_depth: 16\n"
				"    audio_format: flac\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.audio.audio_format == "flac"
		assert cfg.recorder.audio.bit_depth    == 16

	def test_flac_with_24bit_accepted (self, tmp_path: pathlib.Path) -> None:
		"""audio_format: flac with bit_depth: 24 is a valid combination."""

		import shutil

		default = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    bit_depth: 24\n"
				"    audio_format: flac\n"
			)

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.audio.audio_format == "flac"
		assert cfg.recorder.audio.bit_depth    == 24


class TestPreviewsConfig:

	"""Tests for recorder.previews parsing."""

	def test_default_is_true (self) -> None:
		"""With no override, previews resolves to True — new libraries get
		visual previews out of the box."""

		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.previews is True

	def test_explicit_false_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Setting recorder.previews: false in YAML is parsed correctly."""

		import shutil

		default     = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write("\nrecorder:\n  previews: false\n")

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.previews is False

	def test_explicit_true_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Setting recorder.previews: true is idempotent with the default."""

		import shutil

		default     = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write("\nrecorder:\n  previews: true\n")

		cfg = subsample.config.load_config(user_config)

		assert cfg.recorder.previews is True


# ---------------------------------------------------------------------------
# player.audio.buffer_frames — PortAudio output buffer size knob
# ---------------------------------------------------------------------------

class TestBufferFrames:

	"""``player.audio.buffer_frames`` lets advanced users tighten output-side
	latency at the cost of underrun risk.  Validation rejects non-power-of-
	two values and out-of-range ones at config load — that's our chance to
	complain before PortAudio fails at stream open."""

	def _load_with_buffer_frames (self, tmp_path: pathlib.Path, value: typing.Any) -> "subsample.config.Config":

		import shutil

		default     = subsample.config._locate_default_config()
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(f"\nplayer:\n  audio:\n    buffer_frames: {value}\n")

		return subsample.config.load_config(user_config)

	def test_default_is_none (self) -> None:
		"""Unset → None → PortAudio picks the device default."""
		default = subsample.config._locate_default_config()
		cfg = subsample.config.load_config(default)
		assert cfg.player.audio.buffer_frames is None

	def test_valid_power_of_two_accepted (self, tmp_path: pathlib.Path) -> None:
		for value in (32, 64, 128, 256, 512, 1024, 2048, 4096):
			cfg = self._load_with_buffer_frames(tmp_path, value)
			assert cfg.player.audio.buffer_frames == value

	def test_non_power_of_two_rejected (self, tmp_path: pathlib.Path) -> None:
		for value in (33, 300, 1000, 1500):
			with pytest.raises(ValueError, match="power of two"):
				self._load_with_buffer_frames(tmp_path, value)

	def test_below_range_rejected (self, tmp_path: pathlib.Path) -> None:
		# 16 IS a power of two but below the [32, 4096] floor.
		with pytest.raises(ValueError, match=r"\[32, 4096\]"):
			self._load_with_buffer_frames(tmp_path, 16)

	def test_above_range_rejected (self, tmp_path: pathlib.Path) -> None:
		# 8192 is a power of two but above the [32, 4096] ceiling.
		with pytest.raises(ValueError, match=r"\[32, 4096\]"):
			self._load_with_buffer_frames(tmp_path, 8192)

	def test_out_of_range_message_does_not_blame_power_of_two (
		self, tmp_path: pathlib.Path,
	) -> None:
		"""16 is a valid power of two but below the floor — the error must
		surface the range failure, not a misleading 'power of two' note,
		so the user doesn't waste time looking for a bit-pattern problem."""

		with pytest.raises(ValueError) as exc:
			self._load_with_buffer_frames(tmp_path, 16)

		assert "[32, 4096]" in str(exc.value)
		assert "power of two" not in str(exc.value)


class TestSegmentationConfig:

	"""Decoupled recording end (release/retrigger/fade) + clip-safe float import."""

	def test_defaults_are_no_ops (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		# Segmentation extras default off so behaviour is unchanged for everyone.
		assert cfg.detection.release_threshold_db is None
		assert cfg.detection.retrigger_threshold_db is None
		assert cfg.detection.fade_out_ms == 0.0
		# Float import defaults ON (clip-safe) at -1 dBFS.
		assert cfg.recorder.audio.float_import_ceiling_dbfs == -1.0

	def test_parse_release_retrigger_fade (self, tmp_path: pathlib.Path) -> None:
		cfg = _load_with(tmp_path, detection={
			"threshold_db": 10.5, "release_threshold_db": 4.0,
			"retrigger_threshold_db": 12.0, "fade_out_ms": 30.0,
		})
		assert cfg.detection.release_threshold_db == 4.0
		assert cfg.detection.retrigger_threshold_db == 12.0
		assert cfg.detection.fade_out_ms == 30.0

	def test_parse_float_ceiling_and_null (self, tmp_path: pathlib.Path) -> None:
		assert _load_with(tmp_path, audio={"float_import_ceiling_dbfs": -3.0}) \
			.recorder.audio.float_import_ceiling_dbfs == -3.0
		assert _load_with(tmp_path, audio={"float_import_ceiling_dbfs": None}) \
			.recorder.audio.float_import_ceiling_dbfs is None

	def test_release_must_be_below_snr (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="release_threshold_db"):
			_load_with(tmp_path, detection={"threshold_db": 10.0, "release_threshold_db": 12.0})

	def test_release_must_be_positive (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="release_threshold_db"):
			_load_with(tmp_path, detection={"release_threshold_db": 0.0})

	def test_retrigger_must_be_positive (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="retrigger_threshold_db"):
			_load_with(tmp_path, detection={"retrigger_threshold_db": -1.0})

	def test_fade_out_ms_must_be_non_negative (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="fade_out_ms"):
			_load_with(tmp_path, detection={"fade_out_ms": -5.0})

	def test_float_ceiling_must_be_non_positive (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="float_import_ceiling_dbfs"):
			_load_with(tmp_path, audio={"float_import_ceiling_dbfs": 3.0})


class TestRenamedKeyMigrations:

	"""The 2026-07 rename batch hard-errors with the replacement named, instead
	of silently ignoring an old key (the unknown-key sweep only warns).  One
	test per renamed key/section, matching the migration messages in
	_build_config."""

	def _load (self, tmp_path: pathlib.Path, body: str) -> subsample.config.Config:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(textwrap.dedent(body))
		return subsample.config.load_config(config_file)

	def test_output_section_moved (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="moved into `recorder:`"):
			self._load(tmp_path, """\
				output:
				  directory: /tmp/x
			""")

	def test_instrument_section_renamed (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="now called `library:`"):
			self._load(tmp_path, """\
				instrument:
				  directory: /tmp/x
			""")

	def test_snr_threshold_db_renamed (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="`detection.threshold_db`"):
			self._load(tmp_path, """\
				detection:
				  snr_threshold_db: 9.0
			""")

	def test_ema_alpha_renamed (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="`detection.floor_adaptation`"):
			self._load(tmp_path, """\
				detection:
				  ema_alpha: 0.1
			""")

	def test_hold_time_renamed (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="`detection.hold_seconds`"):
			self._load(tmp_path, """\
				detection:
				  hold_time: 0.5
			""")

	def test_trim_pre_samples_renamed_with_unit_note (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="milliseconds"):
			self._load(tmp_path, """\
				detection:
				  trim_pre_samples: 10
			""")

	def test_trim_post_samples_renamed_with_unit_note (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="`detection.trim_post_ms`"):
			self._load(tmp_path, """\
				detection:
				  trim_post_samples: 90
			""")

	def test_chunk_size_renamed (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="`recorder.audio.buffer_frames`"):
			self._load(tmp_path, """\
				recorder:
				  audio:
				    chunk_size: 512
			""")


class TestEmptyAndCommentedValues:

	"""A key with no value, or a section with everything commented out.

	Both are ordinary things to type, and both used to go wrong silently or
	misleadingly: a bare `filename_format:` became the literal string "None" so
	every capture overwrote samples/captures/None.wav, and a section header
	whose children were all commented out hard-failed startup while blaming the
	user's indentation.
	"""

	def test_required_key_with_no_value_is_rejected (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
		monkeypatch.chdir(tmp_path)
		(tmp_path / "config.yaml").write_text("recorder:\n  filename_format:\n")

		with pytest.raises(ValueError, match="present but has no value"):
			subsample.config.load_config(None)

	def test_commented_out_section_falls_back_to_defaults (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
		"""Only four sections ship uncommented, so this hit the other seven."""
		monkeypatch.chdir(tmp_path)
		(tmp_path / "config.yaml").write_text("osc:\n  # enabled: true\n  # send_port: 9000\n")

		cfg = subsample.config.load_config(None)

		assert cfg.osc.enabled is False

	def test_empty_config_file_means_no_overrides (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
		monkeypatch.chdir(tmp_path)
		(tmp_path / "config.yaml").write_text("")

		assert subsample.config.load_config(None).recorder.enabled is True

	def test_fully_commented_config_file_means_no_overrides (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
		"""Commenting the whole file out to fall back to defaults is a normal move."""
		monkeypatch.chdir(tmp_path)
		(tmp_path / "config.yaml").write_text("# player:\n#   enabled: true\n")

		assert subsample.config.load_config(None).player.enabled is False


class TestBooleanKeysRejectNonBooleans:

	"""`bool(value)` accepted anything truthy, so a QUOTED YAML boolean inverted
	the setting: `enabled: "false"` meant True."""

	@pytest.mark.parametrize("section,key,value", [
		("player",   "enabled", '"false"'),
		("library",  "watch",   "'no'"),
		("recorder", "enabled", '"off"'),
	])
	def test_quoted_boolean_rejected (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		section: str, key: str, value: str,
	) -> None:
		monkeypatch.chdir(tmp_path)
		(tmp_path / "config.yaml").write_text(f"{section}:\n  {key}: {value}\n")

		with pytest.raises(ValueError, match="must be true or false"):
			subsample.config.load_config(None)

	@pytest.mark.parametrize("literal,expected", [("true", True), ("yes", True), ("false", False), ("no", False)])
	def test_unquoted_yaml_booleans_still_work (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		literal: str, expected: bool,
	) -> None:
		monkeypatch.chdir(tmp_path)
		(tmp_path / "config.yaml").write_text(f"player:\n  enabled: {literal}\n")

		assert subsample.config.load_config(None).player.enabled is expected


class TestMinPeakDb:

	"""detection.min_peak_db — the one ABSOLUTE gate among relative thresholds.

	Every other detection threshold is dB over a tracked floor, which is what lets
	the detector work in any room but also means it cannot distinguish a quiet
	room's noise from a quiet sound.  This key is the backstop, so its validation
	has to be strict: a value that silently discards everything is worse than a
	load error.
	"""

	def test_defaults_to_none (self, tmp_path: pathlib.Path) -> None:

		"""Absent means disabled — every segment the detector emits is kept, which
		is the behaviour every existing config relies on."""

		cfg = _load_with(tmp_path)

		assert cfg.detection.min_peak_db is None

	def test_negative_dbfs_is_accepted (self, tmp_path: pathlib.Path) -> None:

		"""A level below full scale is the only meaningful kind of value."""

		cfg = _load_with(tmp_path, detection={"min_peak_db": -45.0})

		assert cfg.detection.min_peak_db == -45.0

	def test_integer_is_coerced_to_float (self, tmp_path: pathlib.Path) -> None:

		"""`min_peak_db: -45` is the natural thing to type and must not be a type
		error downstream."""

		cfg = _load_with(tmp_path, detection={"min_peak_db": -45})

		assert cfg.detection.min_peak_db == pytest.approx(-45.0)

	@pytest.mark.parametrize("value", [0.0, 3.0, float("nan"), float("inf"), float("-inf")])
	def test_non_negative_or_non_finite_raises (
		self, tmp_path: pathlib.Path, value: float,
	) -> None:

		"""0 dBFS is full scale, so any value at or above it discards every
		recording — a total capture failure that would look like a broken
		detector.  Fail the load instead.  NaN fails the comparison and lands
		here too; -inf would disable the gate while appearing to set it."""

		with pytest.raises(ValueError, match="min_peak_db"):
			_load_with(tmp_path, detection={"min_peak_db": value})

	@pytest.mark.parametrize("value", [True, False])
	def test_boolean_raises (self, tmp_path: pathlib.Path, value: bool) -> None:

		"""`min_peak_db: yes` parses as a bool in YAML; float(True) is 1.0 and
		float(False) is 0.0, so both fail the negativity check rather than
		silently configuring a nonsense gate."""

		with pytest.raises(ValueError, match="min_peak_db"):
			_load_with(tmp_path, detection={"min_peak_db": value})


class TestReferenceDirectory:

	"""library.reference_directory — where named `reference:` predicates resolve.

	Defaulting to the packaged set is what lets a MIDI map name a reference
	instead of pointing at one, and that is what makes a sample set shareable:
	a path-based reference has to reach into one project's tree, so a set on a
	shared drive could not use references at all.
	"""

	def test_defaults_to_none (self, tmp_path: pathlib.Path) -> None:

		cfg = _load_with(tmp_path)

		assert cfg.library.reference_directory is None

	def test_custom_directory_is_kept (self, tmp_path: pathlib.Path) -> None:

		cfg = _load_with(tmp_path)
		cfg_file = tmp_path / "config.yaml"
		data = yaml.safe_load(cfg_file.read_text())
		data["library"] = {"reference_directory": "my/refs"}
		cfg_file.write_text(yaml.safe_dump(data))

		cfg = subsample.config.load_config(cfg_file)

		assert cfg.library.reference_directory == "my/refs"


class TestNullableLibraryDirectory:

	"""library.directory: null — load nothing in bulk.

	A project assembled from shared sample sets wants exactly the samples its
	MIDI maps name.  Walking a capture tree of thousands of unrelated samples is
	pure cost, and the tree may not even contain the sets (they can live on a
	shared drive, outside the project entirely).
	"""

	def test_defaults_to_captures (self, tmp_path: pathlib.Path) -> None:

		"""The simple case is untouched — an absent key still means the capture
		directory, so a user who never thinks about this sees no change."""

		cfg = _load_with(tmp_path)

		assert cfg.library.directory == "samples/captures"

	def test_explicit_null_is_kept (self, tmp_path: pathlib.Path) -> None:

		"""`directory: null` must survive as None.  Reading it through
		`.get(key, default)` would turn an explicit null back into the default
		and silently bulk-load anyway."""

		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml.safe_dump({
			"recorder": {
				"audio": {"sample_rate": 48000, "bit_depth": 16, "channels": 1, "buffer_frames": 512},
				"buffer": {"max_seconds": 60},
				"directory": "samples/captures",
			},
			"detection": {
				"threshold_db": 12.0, "hold_seconds": 0.5,
				"warmup_seconds": 1.0, "floor_adaptation": 0.1,
			},
			"library": {"directory": None},
		}))

		cfg = subsample.config.load_config(config_file)

		assert cfg.library.directory is None

	def test_custom_directory_still_works (self, tmp_path: pathlib.Path) -> None:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml.safe_dump({
			"recorder": {
				"audio": {"sample_rate": 48000, "bit_depth": 16, "channels": 1, "buffer_frames": 512},
				"buffer": {"max_seconds": 60},
				"directory": "samples/captures",
			},
			"detection": {
				"threshold_db": 12.0, "hold_seconds": 0.5,
				"warmup_seconds": 1.0, "floor_adaptation": 0.1,
			},
			"library": {"directory": "/shared/sets"},
		}))

		cfg = subsample.config.load_config(config_file)

		assert cfg.library.directory == "/shared/sets"


class TestMidiMaps:

	"""player.midi_maps — sample sets bound to channels in config.yaml.

	The shorthand for an ensemble, for a project that keeps its channel
	assignments beside the rest of its configuration.
	"""

	@staticmethod
	def _load (tmp_path: pathlib.Path, player: dict) -> subsample.config.Config:
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml.safe_dump({
			"recorder": {
				"audio": {"sample_rate": 48000, "bit_depth": 16, "channels": 1, "buffer_frames": 512},
				"buffer": {"max_seconds": 60},
				"directory": "samples/captures",
			},
			"detection": {
				"threshold_db": 12.0, "hold_seconds": 0.5,
				"warmup_seconds": 1.0, "floor_adaptation": 0.1,
			},
			"player": player,
		}))
		return subsample.config.load_config(config_file)

	def test_defaults_to_none (self, tmp_path: pathlib.Path) -> None:
		assert _load_with(tmp_path).player.midi_maps is None

	def test_channel_to_path_mapping (self, tmp_path: pathlib.Path) -> None:
		cfg = self._load(tmp_path, {"midi_maps": {10: "a/midi-map.yaml", 11: "b/kit.yaml"}})

		assert cfg.player.midi_maps == {10: "a/midi-map.yaml", 11: "b/kit.yaml"}

	def test_mutually_exclusive_with_midi_map (self, tmp_path: pathlib.Path) -> None:

		"""Both would leave "which rules are live?" ambiguous, and quietly
		preferring one would make the other's edits look ineffective."""

		with pytest.raises(ValueError, match="mutually exclusive"):
			self._load(tmp_path, {
				"midi_map": "one.yaml",
				"midi_maps": {10: "a/midi-map.yaml"},
			})

	@pytest.mark.parametrize("channel", [0, 17, -1])
	def test_out_of_range_channel_rejected (
		self, tmp_path: pathlib.Path, channel: int,
	) -> None:

		"""Caught here, with the config line in view, rather than producing a set
		that can never be triggered."""

		with pytest.raises(ValueError, match="1-16"):
			self._load(tmp_path, {"midi_maps": {channel: "a/midi-map.yaml"}})

	def test_boolean_key_rejected (self, tmp_path: pathlib.Path) -> None:

		"""A YAML bool is an int subclass, so `true:` would otherwise pass as
		channel 1."""

		with pytest.raises(ValueError, match="not a MIDI channel"):
			self._load(tmp_path, {"midi_maps": {True: "a/midi-map.yaml"}})

	def test_non_string_path_rejected (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="non-empty path"):
			self._load(tmp_path, {"midi_maps": {10: 42}})

	def test_empty_mapping_rejected (self, tmp_path: pathlib.Path) -> None:

		"""An empty block reads as "sets are configured" while configuring none —
		the player would start silent with nothing to explain it."""

		with pytest.raises(ValueError, match="is empty"):
			self._load(tmp_path, {"midi_maps": {}})

	def test_non_mapping_rejected (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(ValueError, match="must be a mapping"):
			self._load(tmp_path, {"midi_maps": ["a/midi-map.yaml"]})
