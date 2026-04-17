"""Tests for subsample.config config loading and validation."""

import dataclasses
import pathlib
import textwrap
import unittest.mock

import pytest

import subsample.config


_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml.default"


class TestLoadDefault:

	def test_loads_default_config (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert isinstance(cfg, subsample.config.Config)
		assert isinstance(cfg.recorder, subsample.config.RecorderConfig)
		assert isinstance(cfg.recorder.audio, subsample.config.AudioConfig)
		assert isinstance(cfg.recorder.buffer, subsample.config.BufferConfig)
		assert isinstance(cfg.player, subsample.config.PlayerConfig)
		assert isinstance(cfg.detection, subsample.config.DetectionConfig)
		assert isinstance(cfg.output, subsample.config.OutputConfig)
		assert isinstance(cfg.analysis, subsample.config.AnalysisConfig)

	def test_default_audio_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.recorder.audio.sample_rate == 44100
		assert cfg.recorder.audio.bit_depth == 16
		assert cfg.recorder.audio.channels == 1
		assert cfg.recorder.audio.input is None
		assert cfg.recorder.audio.chunk_size == 512

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

		assert cfg.detection.snr_threshold_db == 12.0
		assert cfg.detection.hold_time == 0.5
		assert cfg.detection.warmup_seconds == 1.0
		assert cfg.detection.ema_alpha == 0.1

	def test_default_output_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.output.directory == "samples/captures"
		assert cfg.output.filename_format == "%Y-%m-%d_%H-%M-%S-%3f"

	def test_default_analysis_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.analysis.start_bpm == 120.0
		assert cfg.analysis.tempo_min == 30.0
		assert cfg.analysis.tempo_max == 300.0

	def test_default_instrument_values (self) -> None:
		# TEST DEPENDENCY: config.yaml.default instrument section defaults
		# instrument.max_memory_mb is derived from auto-detect (60% of global).
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.instrument.max_memory_mb > 0
		assert cfg.instrument.directory == "samples/captures"

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
			    chunk_size: 1024
			  buffer:
			    max_seconds: 60
			detection:
			  snr_threshold_db: 6.0
			  hold_time: 0.5
			  warmup_seconds: 3.0
			  ema_alpha: 0.01
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)
		cfg = subsample.config.load_config(config_file)

		assert cfg.analysis.start_bpm == 120.0
		assert cfg.analysis.tempo_min == 30.0
		assert cfg.analysis.tempo_max == 300.0

	def test_default_trim_padding_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.detection.trim_pre_samples == 10
		assert cfg.detection.trim_post_samples == 90


class TestLoadCustomConfig:

	def test_loads_custom_yaml (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 48000
			    bit_depth: 24
			    channels: 2
			    chunk_size: 2048
			  buffer:
			    max_seconds: 30
			detection:
			  snr_threshold_db: 10.0
			  hold_time: 1.0
			  warmup_seconds: 3.0
			  ema_alpha: 0.05
			output:
			  directory: /tmp/my_samples
			  filename_format: "%H-%M-%S"
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.sample_rate == 48000
		assert cfg.recorder.audio.bit_depth == 24
		assert cfg.recorder.audio.channels == 2
		assert cfg.recorder.buffer.max_seconds == 30
		assert cfg.detection.snr_threshold_db == 10.0
		assert cfg.output.directory == "/tmp/my_samples"

	def test_recorder_enabled_flag (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  enabled: false
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			  enabled: true
			  audio:
			    device: "Focusrite Output"
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			  enabled: true
			  midi_device: "Launchpad"
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			  midi_device: 42
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			  enabled: true
			  virtual_midi_port: "Subsample Virtual MIDI"
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			  virtual_midi_port: 99
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
					sample_rate=99, bit_depth=16, channels=1, chunk_size=1024
				),
				buffer=subsample.config.BufferConfig(max_seconds=60),
			)

	def test_missing_file_raises (self, tmp_path: pathlib.Path) -> None:
		nonexistent = tmp_path / "does_not_exist.yaml"

		with pytest.raises(FileNotFoundError):
			subsample.config.load_config(nonexistent)

	def test_similarity_custom_weights (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
			recorder:
			  audio:
			    sample_rate: 44100
			    bit_depth: 16
			    channels: 1
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			detection:
			  snr_threshold_db: 6.0
			  hold_time: 0.5
			  warmup_seconds: 3.0
			  ema_alpha: 0.01
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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

	def test_channels_omitted_inherits_default (self, tmp_path: pathlib.Path) -> None:
		"""Omitting channels in user config inherits the default value (1 = mono)."""
		config_file = tmp_path / "config.yaml"
		config_file.write_text(self._minimal_yaml(""))
		cfg = subsample.config.load_config(config_file)

		assert cfg.recorder.audio.channels == 1

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
			    chunk_size: 1024
			  buffer:
			    max_seconds: 60
			detection:
			  snr_threshold_db: 6.0
			  hold_time: 0.5
			  warmup_seconds: 2.0
			  ema_alpha: 0.01
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			{player_section}
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
			    chunk_size: 512
			  buffer:
			    max_seconds: 60
			player:
			  limiter_threshold_db: -1.5
			  limiter_ceiling_db: -3.0
			detection:
			  snr_threshold_db: 12.0
			  hold_time: 0.5
			  warmup_seconds: 1.0
			  ema_alpha: 0.1
			output:
			  directory: ./samples
			  filename_format: "%Y-%m-%d_%H-%M-%S"
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
		assert cfg.detection.snr_threshold_db == 12.0
		assert cfg.output.directory == "samples/captures"

	def test_partial_section_inherits_sibling_keys (self, tmp_path: pathlib.Path) -> None:
		"""Overriding one key in a section leaves sibling keys at their defaults."""
		yaml_content = textwrap.dedent("""\
			detection:
			  snr_threshold_db: 6.0
		""")
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)

		cfg = subsample.config.load_config(config_file)

		assert cfg.detection.snr_threshold_db == 6.0
		assert cfg.detection.hold_time == 0.5
		assert cfg.detection.warmup_seconds == 1.0
		assert cfg.detection.ema_alpha == 0.1

	def test_default_path_explicit_no_double_merge (self) -> None:
		"""Passing the default config path explicitly loads correctly without double-merging."""
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)
		assert cfg.recorder.audio.sample_rate == 44100
		assert cfg.detection.trim_pre_samples == 10

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
			"    chunk_size: 512\n"
			f"    {audio_extra}\n"
			"  buffer:\n"
			"    max_seconds: 10\n"
			"detection:\n"
			"  snr_threshold_db: 6\n"
			"  hold_time: 0.5\n"
			"  warmup_seconds: 2\n"
			"  ema_alpha: 0.1\n"
			"output:\n"
			"  directory: /tmp/test\n"
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
			"    chunk_size: 512\n"
			"    channels: 1\n"
			"  buffer:\n"
			"    max_seconds: 10\n"
			"detection:\n"
			"  snr_threshold_db: 6\n"
			"  hold_time: 0.5\n"
			"  warmup_seconds: 2\n"
			"  ema_alpha: 0.1\n"
			"output:\n"
			"  directory: /tmp/test\n"
		)
		config_file = tmp_path / "config.yaml"
		config_file.write_text(yaml_content)
		return subsample.config.load_config(config_file)

	def test_explicit_global_splits_correctly (self, tmp_path: pathlib.Path) -> None:
		"""max_memory_mb: 200 → instrument 120, transform 70, carrier 10."""
		cfg = self._make_config(tmp_path, "max_memory_mb: 200")
		assert cfg.instrument.max_memory_mb == pytest.approx(120.0)
		assert cfg.transform.max_memory_mb == pytest.approx(70.0)
		assert cfg.transform.carrier_memory_mb == pytest.approx(10.0)
		assert cfg.transform.max_disk_mb == pytest.approx(600.0)

	def test_per_cache_overrides_global (self, tmp_path: pathlib.Path) -> None:
		"""Explicit per-cache values take precedence over global."""
		cfg = self._make_config(
			tmp_path,
			"max_memory_mb: 200\n"
			"instrument:\n"
			"  max_memory_mb: 300\n",
		)
		assert cfg.instrument.max_memory_mb == 300.0
		assert cfg.transform.max_memory_mb == pytest.approx(70.0)

	def test_both_per_cache_ignores_global (self, tmp_path: pathlib.Path) -> None:
		"""When both per-cache values are set, global budget is not applied."""
		cfg = self._make_config(
			tmp_path,
			"max_memory_mb: 200\n"
			"instrument:\n"
			"  max_memory_mb: 300\n"
			"transform:\n"
			"  max_memory_mb: 80\n",
		)
		assert cfg.instrument.max_memory_mb == 300.0
		assert cfg.transform.max_memory_mb == 80.0

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
		assert cfg.instrument.max_memory_mb == pytest.approx(512.0 * 0.60)
		assert cfg.transform.max_memory_mb == pytest.approx(512.0 * 0.35)
		assert cfg.transform.carrier_memory_mb == pytest.approx(512.0 * 0.05)
		assert cfg.transform.max_disk_mb == pytest.approx(512.0 * 3.0)

	@unittest.mock.patch("subsample.config._auto_detect_memory_mb", return_value=1024.0)
	def test_auto_detect_16gb_system (self, mock_detect: unittest.mock.MagicMock, tmp_path: pathlib.Path) -> None:
		"""On a 16 GB+ system, auto-detect → 1024 MB cap."""
		cfg = self._make_config(tmp_path, "")
		assert cfg.instrument.max_memory_mb == pytest.approx(1024.0 * 0.60)
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

		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
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

		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
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

		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
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

		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nrecorder:\n"
				"  audio:\n"
				"    channels: 2\n"
				"    ambisonic_format: a_generic\n"
			)

		with pytest.raises(ValueError, match="requires 4 input channels"):
			subsample.config.load_config(user_config)

	def test_invalid_decoder_rejected (self, tmp_path: pathlib.Path) -> None:
		"""Unknown ambisonic.decoder value raises ValueError."""

		import shutil

		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
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

		default = pathlib.Path(__file__).parent.parent / "config.yaml.default"
		user_config = tmp_path / "config.yaml"
		shutil.copy(default, user_config)

		with user_config.open("a") as fh:
			fh.write(
				"\nambisonic:\n"
				"  max_order: 2\n"
			)

		with pytest.raises(ValueError, match="max_order must be 1"):
			subsample.config.load_config(user_config)
