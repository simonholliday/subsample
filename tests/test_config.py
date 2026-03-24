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

	def test_default_detection_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.detection.snr_threshold_db == 12.0
		assert cfg.detection.hold_time == 0.5
		assert cfg.detection.warmup_seconds == 1.0
		assert cfg.detection.ema_alpha == 0.1

	def test_default_output_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.output.directory == "./samples"
		assert cfg.output.filename_format == "%Y-%m-%d_%H-%M-%S"

	def test_default_analysis_values (self) -> None:
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.analysis.start_bpm == 120.0
		assert cfg.analysis.tempo_min == 30.0
		assert cfg.analysis.tempo_max == 300.0

	def test_default_instrument_values (self) -> None:
		# TEST DEPENDENCY: config.yaml.default instrument section defaults
		cfg = subsample.config.load_config(_DEFAULT_CONFIG_PATH)

		assert cfg.instrument.max_memory_mb == 100.0
		assert cfg.instrument.directory is None

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

		assert cfg.detection.trim_pre_samples == 15
		assert cfg.detection.trim_post_samples == 85


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

	def test_channels_omitted_yields_none (self, tmp_path: pathlib.Path) -> None:
		"""Omitting channels entirely also resolves to None (auto-detect)."""
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
