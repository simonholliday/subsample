"""Tests for subsample.config config loading and validation."""

import dataclasses
import pathlib
import textwrap

import pytest

import subsample.config


_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml.default"


class TestLoadDefault:

	def test_loads_default_config (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert isinstance(cfg, subsample.config.Config)
		assert isinstance(cfg.audio, subsample.config.AudioConfig)
		assert isinstance(cfg.buffer, subsample.config.BufferConfig)
		assert isinstance(cfg.detection, subsample.config.DetectionConfig)
		assert isinstance(cfg.output, subsample.config.OutputConfig)
		assert isinstance(cfg.analysis, subsample.config.AnalysisConfig)

	def test_default_audio_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.audio.sample_rate == 44100
		assert cfg.audio.bit_depth == 16
		assert cfg.audio.channels == 1
		assert cfg.audio.chunk_size == 512

	def test_default_buffer_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.buffer.max_seconds == 60

	def test_default_detection_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.detection.snr_threshold_db == 9.0
		assert cfg.detection.hold_time == 0.5
		assert cfg.detection.warmup_seconds == 3.0
		assert cfg.detection.ema_alpha == 0.01

	def test_default_output_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.output.directory == "./samples"
		assert cfg.output.filename_format == "%Y-%m-%d_%H-%M-%S"

	def test_default_analysis_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.analysis.start_bpm == 120.0
		assert cfg.analysis.tempo_min == 30.0
		assert cfg.analysis.tempo_max == 300.0

	def test_analysis_defaults_when_section_absent (self, tmp_path: pathlib.Path) -> None:
		"""A config.yaml without an analysis section should use class defaults."""
		yaml_content = textwrap.dedent("""\
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
		assert cfg.detection.trim_post_samples == 25


class TestLoadCustomConfig:

	def test_loads_custom_yaml (self, tmp_path: pathlib.Path) -> None:
		yaml_content = textwrap.dedent("""\
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

		assert cfg.audio.sample_rate == 48000
		assert cfg.audio.bit_depth == 24
		assert cfg.audio.channels == 2
		assert cfg.buffer.max_seconds == 30
		assert cfg.detection.snr_threshold_db == 10.0
		assert cfg.output.directory == "/tmp/my_samples"

	def test_config_is_frozen (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		with pytest.raises(dataclasses.FrozenInstanceError):
			cfg.audio = subsample.config.AudioConfig(  # type: ignore[misc]
				sample_rate=99, bit_depth=16, channels=1, chunk_size=1024
			)

	def test_missing_file_raises (self, tmp_path: pathlib.Path) -> None:
		nonexistent = tmp_path / "does_not_exist.yaml"

		with pytest.raises(FileNotFoundError):
			subsample.config.load_config(nonexistent)

	def test_invalid_bit_depth_raises (self, tmp_path: pathlib.Path) -> None:
		"""Loading a config with unsupported bit_depth should raise ValueError."""
		yaml_content = textwrap.dedent("""\
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
