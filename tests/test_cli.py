"""Tests for subsample.cli — argument parsing and detection pipeline helpers."""

import dataclasses
import logging
import pathlib
import sys
import textwrap
import typing
import unittest.mock
import wave

import numpy
import pytest

import subsample.analysis
import subsample.audio
import subsample.buffer
import subsample.cli
import subsample.config
import subsample.detector
import subsample.events
import subsample.library
import subsample.player
import subsample.similarity
import subsample.transform

import tests.helpers


def _make_detection_cfg (
	threshold_db: float = 6.0,
	trim_pre_ms: int = 8,
	trim_post_ms: int = 8,
	hold_seconds: float = 0.1,
	release_threshold_db: typing.Optional[float] = None,
	retrigger_threshold_db: typing.Optional[float] = None,
	fade_out_ms: float = 0.0,
) -> subsample.config.DetectionConfig:
	"""Return a DetectionConfig suitable for unit tests."""
	return subsample.config.DetectionConfig(
		threshold_db = threshold_db,
		floor_adaptation        = 0.1,
		hold_seconds        = hold_seconds,
		warmup_seconds   = 0.0,
		trim_pre_ms = trim_pre_ms,
		trim_post_ms = trim_post_ms,
		release_threshold_db = release_threshold_db,
		retrigger_threshold_db = retrigger_threshold_db,
		fade_out_ms = fade_out_ms,
	)


def _stream_chunks (
	detection_cfg: subsample.config.DetectionConfig,
	blocks: list[tuple[int, int]],
	sample_rate: int = 1000,
	chunk_size: int = 100,
) -> list[numpy.ndarray]:
	"""Feed (amplitude, n_chunks) blocks through _process_chunk; collect segments.

	Exercises the full cli path (buffer + detector + trim) so the close threshold
	and fade wiring are covered end-to-end, not just the detector in isolation.
	"""
	buf, detector = _make_buffer_and_detector(
		sample_rate=sample_rate, chunk_size=chunk_size, max_seconds=10,
		detection_cfg=detection_cfg,
	)
	segments: list[numpy.ndarray] = []
	for amplitude, n_chunks in blocks:
		for _ in range(n_chunks):
			chunk = numpy.full((chunk_size, 1), amplitude, dtype=numpy.int16)
			result = subsample.cli._process_chunk(chunk, buf, detector, detection_cfg, sample_rate)
			if result is not None:
				segments.append(result)
	return segments


def _make_buffer_and_detector (
	sample_rate: int = 44100,
	chunk_size: int = 512,
	max_seconds: int = 5,
	detection_cfg: subsample.config.DetectionConfig = None,  # type: ignore[assignment]
) -> tuple[subsample.buffer.CircularBuffer, subsample.detector.LevelDetector]:
	"""Return a (CircularBuffer, LevelDetector) pair for testing."""
	if detection_cfg is None:
		detection_cfg = _make_detection_cfg()

	max_frames = sample_rate * max_seconds
	buf = subsample.buffer.CircularBuffer(
		max_frames, channels=1, dtype=numpy.dtype(numpy.int16),
	)
	detector = subsample.detector.LevelDetector(
		detection_cfg, sample_rate, chunk_size, max_recording_frames=max_frames,
	)
	return buf, detector


class TestTrimMsConversion:

	"""trim_pre_ms / trim_post_ms are DURATIONS (a Stage-A frames->ms rename);
	the segment builder must convert them to sample counts before trimming."""

	def test_ms_converted_to_samples (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""_build_trimmed_segment converts via round(ms/1000*sr).  A regression
		passing the ms value straight through as a sample count would shrink the
		audible pre/post padding ~44x at 44.1 kHz — this pins the conversion."""

		captured: dict[str, int] = {}

		def fake_trim (
			segment: numpy.ndarray, tail_thr: float, *,
			pre_samples: int, post_samples: int,
			fade_out_samples: int, lead_amplitude_threshold: float,
		) -> numpy.ndarray:
			captured["pre"] = pre_samples
			captured["post"] = post_samples
			return segment

		monkeypatch.setattr("subsample.trim.trim_silence", fake_trim)

		buf = unittest.mock.Mock()
		buf.read_range.return_value = numpy.zeros((1000, 1), dtype=numpy.float32)
		detector = unittest.mock.Mock()
		detector.tail_amplitude_threshold = 0.01
		detector.attack_amplitude_threshold = 0.02
		cfg = _make_detection_cfg(trim_pre_ms=20, trim_post_ms=10)

		subsample.cli._build_trimmed_segment(buf, detector, cfg, 44100, 500, 800)

		assert captured["pre"] == 882    # round(20 / 1000 * 44100)
		assert captured["post"] == 441   # round(10 / 1000 * 44100)


class TestProcessChunk:

	"""Tests for subsample.cli._process_chunk()."""

	def test_silence_returns_none (self) -> None:
		"""Pure silence should never trigger a recording."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(sample_rate=sample_rate, chunk_size=chunk_size)
		cfg = _make_detection_cfg()

		silent_chunk = numpy.zeros((chunk_size, 1), dtype=numpy.int16)

		# Feed many silent chunks — no recording should be detected
		for _ in range(50):
			result = subsample.cli._process_chunk(silent_chunk, buf, detector, cfg, sample_rate)
			assert result is None

	def test_loud_burst_returns_segment (self) -> None:
		"""A loud burst followed by silence should trigger one detected segment."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(
			sample_rate=sample_rate,
			chunk_size=chunk_size,
			detection_cfg=_make_detection_cfg(threshold_db=3.0),
		)
		cfg = _make_detection_cfg(threshold_db=3.0)

		# Feed low-amplitude chunks to establish an ambient floor before the burst.
		# If the ambient is near zero, the first loud chunk seeds ambient directly
		# to the signal level (giving 0 dB SNR), which would fail to trigger.
		# Amplitude 100 gives ~38 dB headroom against the 8000-amplitude signal.
		ambient = numpy.full((chunk_size, 1), 100, dtype=numpy.int16)
		for _ in range(30):
			subsample.cli._process_chunk(ambient, buf, detector, cfg, sample_rate)

		# Feed loud chunks — well above the established ambient floor (~38 dB SNR)
		loud = numpy.full((chunk_size, 1), 8000, dtype=numpy.int16)
		for _ in range(10):
			subsample.cli._process_chunk(loud, buf, detector, cfg, sample_rate)

		# Feed trailing silence to close the recording (hold_seconds expires)
		# hold_seconds=0.1s at 44100 Hz / 512 frames ≈ 9 chunks
		silent = numpy.zeros((chunk_size, 1), dtype=numpy.int16)
		segments = []
		for _ in range(30):
			result = subsample.cli._process_chunk(silent, buf, detector, cfg, sample_rate)
			if result is not None:
				segments.append(result)

		assert len(segments) == 1
		assert segments[0].ndim == 2
		assert segments[0].shape[1] == 1   # mono

	def test_returns_numpy_array (self) -> None:
		"""The returned segment should be a numpy array when a recording is detected."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(
			sample_rate=sample_rate,
			chunk_size=chunk_size,
			detection_cfg=_make_detection_cfg(threshold_db=3.0),
		)
		cfg = _make_detection_cfg(threshold_db=3.0)

		ambient = numpy.full((chunk_size, 1), 100, dtype=numpy.int16)
		loud = numpy.full((chunk_size, 1), 8000, dtype=numpy.int16)
		silent = numpy.zeros((chunk_size, 1), dtype=numpy.int16)

		for _ in range(30):
			subsample.cli._process_chunk(ambient, buf, detector, cfg, sample_rate)
		for _ in range(10):
			subsample.cli._process_chunk(loud, buf, detector, cfg, sample_rate)

		result = None
		for _ in range(30):
			r = subsample.cli._process_chunk(silent, buf, detector, cfg, sample_rate)
			if r is not None:
				result = r
				break

		assert result is not None, "Expected a segment to be detected but got None"
		assert isinstance(result, numpy.ndarray)


class TestSegmentationIntegration:

	"""End-to-end through _process_chunk: the close threshold and fade wiring, which
	the detector- and trim-only tests cannot cover because cli re-derives the close
	threshold independently and feeds BOTH the detector end and the trim gate."""

	def test_release_threshold_preserves_tail_through_trim (self) -> None:
		# snr=20 opens; a sustained tail at 12 dB over ambient sits below the open
		# threshold but above release=6.  With release set, BOTH the detector end
		# AND the trim gate use 6 dB, so the tail survives; with release unset both
		# use 20 dB and the tail is cut at the start level.  A regression that
		# reverted cli's trim gate to threshold_db would fail only here.
		blocks = [(100, 8), (8000, 1), (400, 12), (100, 8)]  # ambient, attack, 12 dB tail, silence

		with_release = _stream_chunks(
			_make_detection_cfg(
				threshold_db=20.0, release_threshold_db=6.0, hold_seconds=0.3,
				trim_pre_ms=0, trim_post_ms=0,
			),
			blocks,
		)
		without = _stream_chunks(
			_make_detection_cfg(
				threshold_db=20.0, hold_seconds=0.3,
				trim_pre_ms=0, trim_post_ms=0,
			),
			blocks,
		)

		len_release = sum(s.shape[0] for s in with_release)
		len_without = sum(s.shape[0] for s in without)
		assert len_release > len_without * 3  # the 12 dB tail is retained, not trimmed off

	def test_fade_out_ms_applies_trailing_fade (self) -> None:
		# fade_out_ms is converted to samples in cli (round(ms/1000 * sample_rate))
		# and only affects the trailing edge, not the segment length.  A dropped
		# /1000 or a sample_rate mix-up would fail here.
		blocks = [(100, 8), (4000, 6), (100, 8)]  # ambient, sustain, silence

		faded = _stream_chunks(
			_make_detection_cfg(
				threshold_db=10.0, hold_seconds=0.3, fade_out_ms=50.0,
				trim_pre_ms=0, trim_post_ms=0,
			),
			blocks,
		)
		plain = _stream_chunks(
			_make_detection_cfg(
				threshold_db=10.0, hold_seconds=0.3, fade_out_ms=0.0,
				trim_pre_ms=0, trim_post_ms=0,
			),
			blocks,
		)

		assert len(faded) == 1 and len(plain) == 1
		f = faded[0][:, 0]
		p = plain[0][:, 0]
		assert f.shape == p.shape                    # fade scales samples, not length
		assert abs(int(f[-1])) < abs(int(p[-1]))     # faded end is attenuated
		assert abs(int(f[-1])) < abs(int(p[-1])) // 4  # ramped well down toward zero

	def test_retrigger_splits_through_process_chunk (self) -> None:
		# The re-trigger split surfaces as two returned segments across successive
		# _process_chunk calls (not just inside the detector).
		blocks = [(100, 8), (8000, 1), (2000, 1), (900, 4), (10000, 1), (2000, 1), (900, 4), (100, 8)]

		segments = _stream_chunks(
			_make_detection_cfg(
				threshold_db=20.0, release_threshold_db=6.0,
				retrigger_threshold_db=12.0, hold_seconds=0.3,
				trim_pre_ms=0, trim_post_ms=0,
			),
			blocks,
		)
		assert len(segments) == 2  # the second hit closed the first, then closed on silence


class TestParseArgs:

	"""Tests for subsample.cli._parse_args()."""

	def test_no_args_returns_empty_files (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""With no arguments, files should be an empty list."""
		monkeypatch.setattr(sys, "argv", ["subsample"])
		args = subsample.cli._parse_args()
		assert args.files == []

	def test_single_file_arg (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""A single positional argument should appear as a Path in files."""
		monkeypatch.setattr(sys, "argv", ["subsample", "recording.wav"])
		args = subsample.cli._parse_args()
		assert len(args.files) == 1
		assert args.files[0] == pathlib.Path("recording.wav")

	def test_multiple_file_args (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""Multiple positional arguments should each become a Path."""
		monkeypatch.setattr(sys, "argv", ["subsample", "a.wav", "b.wav", "c.wav"])
		args = subsample.cli._parse_args()
		assert len(args.files) == 3
		assert args.files[1] == pathlib.Path("b.wav")

	def test_path_type_returned (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""All entries in files should be pathlib.Path objects."""
		monkeypatch.setattr(sys, "argv", ["subsample", "some/path/recording.wav"])
		args = subsample.cli._parse_args()
		assert all(isinstance(f, pathlib.Path) for f in args.files)

	def test_config_flag_parsed_as_path (self) -> None:
		args = subsample.cli._parse_args(["--config", "../shared/config.yaml"])
		assert args.config == pathlib.Path("../shared/config.yaml")

	def test_config_flag_defaults_to_none (self) -> None:
		"""No --config → None → load_config falls back to CWD discovery."""
		args = subsample.cli._parse_args([])
		assert args.config is None

	def test_init_flag (self) -> None:
		assert subsample.cli._parse_args(["--init"]).init is True
		assert subsample.cli._parse_args([]).init is False

	def test_help_lists_tool_commands (self, capsys: pytest.CaptureFixture[str]) -> None:
		"""`subsample --help` names every tool subcommand and the ./ escape rule."""

		with pytest.raises(SystemExit) as excinfo:
			subsample.cli._parse_args(["--help"])

		assert excinfo.value.code == 0
		out = capsys.readouterr().out
		for command in subsample.cli._TOOL_COMMANDS:
			assert command in out
		assert "./import" in out


class TestToolDispatch:

	"""The first CLI argument routes to a tool subcommand before run-mode."""

	def test_every_command_module_has_main (self) -> None:
		"""The dispatch table's targets exist and expose main(argv) -> int."""

		import importlib

		for name, (module_path, _description) in subsample.cli._TOOL_COMMANDS.items():
			module = importlib.import_module(module_path)
			assert callable(module.main), name

	def test_command_routes_to_tool_parser (
		self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
	) -> None:
		"""`subsample catalog --help` reaches the tool's parser, prog-named."""

		monkeypatch.setattr(sys, "argv", ["subsample", "catalog", "--help"])

		with pytest.raises(SystemExit) as excinfo:
			subsample.cli.main()

		assert excinfo.value.code == 0
		assert "subsample catalog" in capsys.readouterr().out

	def test_tool_exit_code_propagates (self, monkeypatch: pytest.MonkeyPatch) -> None:
		"""A tool's failure exit code becomes the process exit code."""

		monkeypatch.setattr(sys, "argv", ["subsample", "analyze"])

		with pytest.raises(SystemExit) as excinfo:
			subsample.cli.main()

		assert excinfo.value.code == 2   # argparse usage error: FILE is required

	def test_path_prefixed_name_is_a_file (self) -> None:
		"""`subsample ./import` is a FILE argument, not the import command."""

		args = subsample.cli._parse_args(["./import"])
		assert args.files == [pathlib.Path("./import")]

	def test_tool_return_code_becomes_exit_code (
		self, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""A tool main() that RETURNS a nonzero int (catalog's missing-directory
		`return 1`, not argparse's own SystemExit) becomes the process exit code
		via the dispatch wrapper — the path the argparse-exit test can't reach."""

		mock_module = unittest.mock.Mock()
		mock_module.main.return_value = 7
		monkeypatch.setattr(
			subsample.cli.importlib, "import_module", lambda name: mock_module,
		)
		monkeypatch.setattr(sys, "argv", ["subsample", "catalog", "whatever"])

		with pytest.raises(SystemExit) as excinfo:
			subsample.cli.main()

		assert excinfo.value.code == 7

	def test_flag_before_subcommand_errors (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""A global flag placed before the subcommand
		(`subsample --config x catalog`) must NOT silently fall into file-input
		mode and exit 0 — it errors with a non-zero code."""

		monkeypatch.chdir(tmp_path)
		cfg = tmp_path / "c.yaml"
		cfg.write_text("recorder:\n  enabled: false\nplayer:\n  enabled: false\n")
		monkeypatch.setattr(
			sys, "argv", ["subsample", "--config", str(cfg), "catalog"],
		)

		with pytest.raises(SystemExit) as excinfo:
			subsample.cli.main()

		assert excinfo.value.code == 2


class TestListDevices:

	"""--list-devices prints the three device sections and exits before any
	config or recorder work."""

	def test_flag_parses (self) -> None:
		assert subsample.cli._parse_args(["--list-devices"]).list_devices is True
		assert subsample.cli._parse_args([]).list_devices is False

	def test_prints_three_sections (
		self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
	) -> None:
		fake_pa = unittest.mock.MagicMock()
		monkeypatch.setattr(subsample.audio, "create_pyaudio", lambda: fake_pa)
		monkeypatch.setattr(
			subsample.audio, "list_input_devices",
			lambda pa: [{"name": "Fake Mic", "defaultSampleRate": 44100.0, "index": 0}],
		)
		monkeypatch.setattr(
			subsample.audio, "list_output_devices",
			lambda pa: [{"name": "Fake Out", "defaultSampleRate": 48000.0, "index": 1}],
		)
		monkeypatch.setattr(
			subsample.player, "list_midi_input_devices",
			lambda: ["Fake Keys 1"],
		)

		subsample.cli._list_devices()

		out = capsys.readouterr().out
		assert "Audio inputs" in out and "Fake Mic" in out and "44100 Hz" in out
		assert "Audio outputs" in out and "Fake Out" in out
		assert "MIDI inputs" in out and "Fake Keys 1" in out
		fake_pa.terminate.assert_called_once()

	def test_list_devices_through_main_runs_before_config (
		self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
	) -> None:
		"""--list-devices is handled in main() BEFORE any config load, so it
		works with no config.yaml present (config load must not run)."""

		monkeypatch.setattr(subsample.audio, "create_pyaudio", lambda: unittest.mock.MagicMock())
		monkeypatch.setattr(subsample.audio, "list_input_devices", lambda pa: [])
		monkeypatch.setattr(subsample.audio, "list_output_devices", lambda pa: [])
		monkeypatch.setattr(subsample.player, "list_midi_input_devices", lambda: [])
		monkeypatch.setattr(
			subsample.config, "load_config",
			unittest.mock.Mock(side_effect=AssertionError("config must not load")),
		)
		monkeypatch.setattr(sys, "argv", ["subsample", "--list-devices"])

		subsample.cli.main()

		assert "Audio inputs" in capsys.readouterr().out


class TestConfigErrorsExitCleanly:

	"""A broken config yields ONE clean error line and exit 1 — no traceback."""

	def test_renamed_key_config_exits_cleanly (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		bad = tmp_path / "old.yaml"
		bad.write_text("detection:\n  snr_threshold_db: 9\n")
		monkeypatch.setattr(sys, "argv", ["subsample", "--config", str(bad)])

		# Safety net: if the migration hard-error were ever removed (the exact
		# regression this guards), load_config would succeed and _main_impl would
		# proceed to real startup — a live PyAudio recorder + an unbounded wait
		# loop, i.e. a hung suite rather than a red test.  Make the device layer
		# raise so a regression fails fast instead of hanging.
		monkeypatch.setattr(
			subsample.audio, "create_pyaudio",
			unittest.mock.Mock(side_effect=RuntimeError("startup must not be reached")),
		)

		with pytest.raises(SystemExit) as excinfo, caplog.at_level(logging.ERROR):
			subsample.cli.main()

		assert excinfo.value.code == 1
		assert any("detection.threshold_db" in r.message for r in caplog.records)


class TestInitConfig:

	"""Tests for subsample.cli._init_config() (the --init project scaffold)."""

	def test_scaffolds_full_project (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		capsys: pytest.CaptureFixture[str],
	) -> None:
		"""--init creates the complete project manifest in CWD."""

		monkeypatch.chdir(tmp_path)

		subsample.cli._init_config()

		assert (tmp_path / "config.yaml").is_file()
		assert (tmp_path / "midi-map-gm-drums.yaml").is_file()
		assert (tmp_path / "midi-map.yaml").is_file()
		assert (tmp_path / ".gitignore").read_text() == "samples/variant-cache/\n"
		assert (tmp_path / "samples" / "captures").is_dir()
		assert (tmp_path / "samples" / "reference" / "CREDITS.md").is_file()
		sidecars = list((tmp_path / "samples" / "reference").glob("*.analysis.json"))
		assert len(sidecars) >= 40
		assert "Created a Subsample project" in capsys.readouterr().out

	def test_scaffolded_config_wires_gm_map_and_keeps_defaults (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""The scaffolded config is the documented defaults with exactly one
		change: the GM kit map wired in (player stays disabled)."""

		monkeypatch.chdir(tmp_path)

		subsample.cli._init_config()

		cfg = subsample.config.load_config(tmp_path / "config.yaml")
		assert cfg.player.midi_map == "midi-map-gm-drums.yaml"
		assert cfg.player.enabled is False

		defaults = subsample.config.load_config(subsample.config._locate_default_config())
		unwired = dataclasses.replace(
			cfg, player=dataclasses.replace(cfg.player, midi_map=defaults.player.midi_map),
		)
		assert unwired == defaults

	def test_refuses_when_any_target_exists (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""All-or-nothing: one existing target (not just config.yaml) aborts
		the whole scaffold and creates nothing."""

		monkeypatch.chdir(tmp_path)
		existing = tmp_path / "midi-map.yaml"
		existing.write_text("assignments: []\n")

		with pytest.raises(SystemExit) as excinfo:
			subsample.cli._init_config()

		assert excinfo.value.code == 1
		assert existing.read_text() == "assignments: []\n"
		assert not (tmp_path / "config.yaml").exists()
		assert not (tmp_path / "midi-map-gm-drums.yaml").exists()
		assert not (tmp_path / "samples").exists()
		assert not (tmp_path / ".gitignore").exists()

	def test_appends_to_existing_gitignore (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""An existing .gitignore (a GitHub-initialised project, say) is not a
		conflict — the scaffold appends its line, preserving what's there."""

		monkeypatch.chdir(tmp_path)
		(tmp_path / ".gitignore").write_text("*.wav\n.venv/\n")

		subsample.cli._init_config()

		lines = (tmp_path / ".gitignore").read_text().splitlines()
		assert lines[:2] == ["*.wav", ".venv/"]
		assert "samples/variant-cache/" in lines

	def test_appends_to_gitignore_without_trailing_newline (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""An existing .gitignore with NO trailing newline gets a separator so the
		appended line is not glued onto the previous one (the separator branch)."""

		monkeypatch.chdir(tmp_path)
		(tmp_path / ".gitignore").write_text("*.wav")   # no trailing newline

		subsample.cli._init_config()

		lines = (tmp_path / ".gitignore").read_text().splitlines()
		assert lines[0] == "*.wav"
		assert "samples/variant-cache/" in lines

	def test_init_through_main_runs_before_config (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""--init is handled in main() BEFORE any config load, so it scaffolds
		even in a directory with no config.yaml (config load must not run)."""

		monkeypatch.chdir(tmp_path)
		monkeypatch.setattr(
			subsample.config, "load_config",
			unittest.mock.Mock(side_effect=AssertionError("config must not load")),
		)
		monkeypatch.setattr(sys, "argv", ["subsample", "--init"])

		subsample.cli.main()

		assert (tmp_path / "config.yaml").is_file()

	def test_gitignore_line_not_duplicated (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		(tmp_path / ".gitignore").write_text("samples/variant-cache/\n")

		subsample.cli._init_config()

		assert (tmp_path / ".gitignore").read_text() == "samples/variant-cache/\n"

	def test_refuses_inside_a_source_checkout (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""The subsample repo itself is the app, not a music project."""

		monkeypatch.chdir(tmp_path)
		(tmp_path / "subsample").mkdir()
		(tmp_path / "subsample" / "__init__.py").write_text("")

		with pytest.raises(SystemExit) as excinfo, caplog.at_level(logging.ERROR):
			subsample.cli._init_config()

		assert excinfo.value.code == 1
		assert any("source repository" in r.message for r in caplog.records)
		assert not (tmp_path / "config.yaml").exists()

	def test_scaffolded_maps_load_with_scaffolded_references (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""The product guarantee: both scaffolded maps load, and the GM kit's
		path-based references resolve from the scaffolded sidecars alone (no
		WAV audio is shipped — the fingerprint is the reference)."""

		monkeypatch.chdir(tmp_path)

		subsample.cli._init_config()

		gm = subsample.player.load_midi_map(tmp_path / "midi-map-gm-drums.yaml", [])
		assert len(gm.note_map) > 0

		template = subsample.player.load_midi_map(tmp_path / "midi-map.yaml", [])
		assert (9, 36) in template.note_map

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.samples.return_value = []

		subsample.player._resolve_path_references(
			gm.note_map, [matrix], instrument_lib, with_preview=False,
		)

		assert matrix.add_reference.call_count >= 40


class TestFormatMmss:

	"""Tests for subsample.cli._format_mmss() — duration formatter used in file-ingest progress logs."""

	def test_zero_seconds (self) -> None:
		assert subsample.cli._format_mmss(0.0) == "00:00"

	def test_sub_minute (self) -> None:
		assert subsample.cli._format_mmss(6.0) == "00:06"

	def test_exactly_one_minute (self) -> None:
		assert subsample.cli._format_mmss(60.0) == "01:00"

	def test_multi_minute (self) -> None:
		assert subsample.cli._format_mmss(92.0) == "01:32"

	def test_rounds_to_nearest_second (self) -> None:
		assert subsample.cli._format_mmss(6.4) == "00:06"
		assert subsample.cli._format_mmss(6.6) == "00:07"

	def test_minutes_not_capped (self) -> None:
		"""Durations over 99 minutes render with three-digit minutes."""
		assert subsample.cli._format_mmss(6000.0) == "100:00"

	def test_negative_clamped_to_zero (self) -> None:
		"""Negative inputs should clamp to 00:00 rather than render '-1:-1'."""
		assert subsample.cli._format_mmss(-5.0) == "00:00"


def _make_record (
	name: str,
	audio: typing.Optional[numpy.ndarray] = None,
) -> subsample.library.SampleRecord:

	"""Build a SampleRecord with default analysis fields and optional audio."""

	return subsample.library.SampleRecord(
		sample_id   = subsample.library.allocate_id(),
		name        = name,
		spectral    = tests.helpers._make_spectral(),
		rhythm      = tests.helpers._make_rhythm(),
		pitch       = tests.helpers._make_pitch(),
		timbre      = tests.helpers._make_timbre(),
		level       = tests.helpers._make_level(),
		band_energy = tests.helpers._make_band_energy(),
		params      = tests.helpers._make_params(),
		duration    = 1.0,
		audio       = audio,
	)


class TestProcessInputFiles:

	"""End-to-end smoke for the file-input mode entry point: a real WAV with
	a clear burst goes through detection and lands as segment file(s) in the
	output directory.  This ~140-line path previously had no test at all."""

	def test_burst_wav_produces_segments (self, tmp_path: pathlib.Path) -> None:
		out_dir = tmp_path / "out"
		out_dir.mkdir()

		# 0.4 s near-silence, 0.8 s loud 440 Hz burst, 0.6 s near-silence.
		sr = 44100
		rng = numpy.random.default_rng(42)
		quiet_a = (rng.standard_normal(int(0.4 * sr)) * 20).astype(numpy.int16)
		t = numpy.arange(int(0.8 * sr)) / sr
		burst = (0.8 * 32767 * numpy.sin(2 * numpy.pi * 440 * t)).astype(numpy.int16)
		quiet_b = (rng.standard_normal(int(0.6 * sr)) * 20).astype(numpy.int16)
		samples = numpy.concatenate([quiet_a, burst, quiet_b])

		wav_path = tmp_path / "field.wav"
		with wave.open(str(wav_path), "wb") as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)
			wf.setframerate(sr)
			wf.writeframes(samples.tobytes())

		config_file = tmp_path / "config.yaml"
		config_file.write_text(textwrap.dedent(f"""\
			recorder:
			  directory: {out_dir}
			detection:
			  threshold_db: 10.0
			  warmup_seconds: 0.0
			  hold_seconds: 0.15
		"""))
		cfg = subsample.config.load_config(config_file)

		subsample.cli._process_input_files([wav_path], cfg)

		segments = sorted(out_dir.glob("field*"))
		assert segments, "no segments written for a clear burst"

	def test_missing_and_unreadable_files_skipped (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
		monkeypatch: pytest.MonkeyPatch,
	) -> None:
		# chdir into the empty tmp dir so load_config(None) sees no stray
		# config.yaml — otherwise this test runs against whatever config sits in
		# the invocation directory (the repo root holds a personal one).
		monkeypatch.chdir(tmp_path)
		bad = tmp_path / "not_audio.wav"
		bad.write_bytes(b"this is not a wav file")

		cfg = subsample.config.load_config(None)

		with caplog.at_level(logging.WARNING, logger="subsample.cli"):
			subsample.cli._process_input_files(
				[tmp_path / "missing.wav", bad], cfg,
			)

		messages = " | ".join(r.message for r in caplog.records)
		assert "not found" in messages
		assert "Could not read" in messages


class TestIntegrateSample:

	"""_integrate_sample is the shared hub that fans a new sample out to the
	library, similarity matrix, transform pipeline, active player, and event
	bus.  It is reached from three callers (capture, watcher, OSC import)."""

	def test_fans_out_to_every_subsystem (self) -> None:
		"""A new sample lands in the library, similarity, transforms, the active
		player's re-evaluation, and the sample_loaded event."""

		library   = subsample.library.InstrumentLibrary(max_memory_bytes=0)
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		similarity.get_scores.return_value = {}
		transform = unittest.mock.MagicMock(spec=subsample.transform.TransformManager)

		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
		player_cell: list[typing.Optional[subsample.player.MidiPlayer]] = [player]

		events = subsample.events.EventEmitter()
		loaded: list[subsample.library.SampleRecord] = []
		events.on("sample_loaded", lambda record: loaded.append(record))

		record = _make_record("kick")
		subsample.cli._integrate_sample(
			record, library, similarity, transform, player_cell, events,
		)

		assert library.get(record.sample_id) is record
		similarity.add.assert_called_once_with(record)
		transform.on_sample_added.assert_called_once_with(record)
		player._try_update_assignments.assert_called_once()
		assert loaded == [record]

	def test_eviction_propagates_to_similarity_and_transforms (self) -> None:
		"""When the library add evicts an old sample, the evicted id is removed
		from the similarity matrix and cascade-evicted from the transform cache."""

		# Budget holds one ~4 KB record; the second add evicts the first.
		library = subsample.library.InstrumentLibrary(max_memory_bytes=5000)
		audio   = numpy.zeros((2000, 1), dtype=numpy.int16)   # 4000 bytes

		first = _make_record("first", audio=audio.copy())
		library.add(first)

		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		similarity.get_scores.return_value = {}
		transform = unittest.mock.MagicMock(spec=subsample.transform.TransformManager)

		second = _make_record("second", audio=audio.copy())
		subsample.cli._integrate_sample(
			second, library, similarity, transform, player_cell=None,
		)

		similarity.remove.assert_called_once_with([first.sample_id])
		transform.on_parent_evicted.assert_called_once_with([first.sample_id])

	def test_cross_file_same_stem_both_integrate (self, tmp_path: pathlib.Path) -> None:
		"""A hot-dropped file whose stem matches a DIFFERENT already-loaded file
		(another take-folder's "kick.wav") integrates as a distinct sample —
		identity is the resolved filepath, not the stem.  Previously this was
		silently skipped."""

		library = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		similarity.get_scores.return_value = {}

		first = _make_record("kick")
		object.__setattr__(first, "filepath", tmp_path / "a" / "kick.wav")
		library.add(first)

		clash = _make_record("kick")
		object.__setattr__(clash, "filepath", tmp_path / "b" / "kick.wav")

		subsample.cli._integrate_sample(
			clash, library, similarity, transform_manager=None, player_cell=None,
		)

		# Both coexist; the same-stem take was integrated, not skipped.
		assert library.get(first.sample_id) is first
		assert library.get(clash.sample_id) is clash
		assert library.find_by_path(tmp_path / "b" / "kick.wav") == clash.sample_id
		similarity.add.assert_called_once_with(clash)

	def test_same_file_reintegration_replaces (self, tmp_path: pathlib.Path) -> None:
		"""Re-integrating the SAME file (re-analysis after an edit) is the
		legitimate replace case and passes through."""

		library = subsample.library.InstrumentLibrary(max_memory_bytes=0)

		path = tmp_path / "kick.wav"
		first = _make_record("kick")
		object.__setattr__(first, "filepath", path)
		library.add(first)

		again = _make_record("kick")
		object.__setattr__(again, "filepath", path)

		subsample.cli._integrate_sample(
			again, library, similarity_matrix=None, transform_manager=None, player_cell=None,
		)

		assert library.get(again.sample_id) is again

	def test_remove_sample_removes_and_cascades (self, tmp_path: pathlib.Path) -> None:
		"""A deleted file's sample is dropped from the library and cascade-cleaned
		from similarity + transforms, and the player is refreshed — kills the
		re-encode ghost."""
		library = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		transform = unittest.mock.MagicMock(spec=subsample.transform.TransformManager)
		player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)

		rec = _make_record("01")
		path = tmp_path / "A" / "01.wav"
		object.__setattr__(rec, "filepath", path)
		library.add(rec)

		subsample.cli._remove_sample(path, library, similarity, transform, [player])

		assert library.get(rec.sample_id) is None
		similarity.remove.assert_called_once_with([rec.sample_id])
		transform.on_parent_evicted.assert_called_once_with([rec.sample_id])
		player._try_update_assignments.assert_called_once()

	def test_remove_sample_missing_is_noop (self, tmp_path: pathlib.Path) -> None:
		"""Removing a path with no loaded sample touches no subsystem."""
		library = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		similarity = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		subsample.cli._remove_sample(
			tmp_path / "nope.wav", library, similarity, None, None,
		)

		similarity.remove.assert_not_called()

	def test_remove_sample_success_with_none_subsystems (self, tmp_path: pathlib.Path) -> None:
		"""A successful removal with no similarity/transform and player_cell=[None]
		(or None) must drop the record without raising on the missing subsystems."""
		library = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)

		rec = _make_record("01")
		path = tmp_path / "01.wav"
		object.__setattr__(rec, "filepath", path)
		library.add(rec)

		subsample.cli._remove_sample(path, library, None, None, [None])
		assert library.get(rec.sample_id) is None

		rec2 = _make_record("02")
		path2 = tmp_path / "02.wav"
		object.__setattr__(rec2, "filepath", path2)
		library.add(rec2)

		subsample.cli._remove_sample(path2, library, None, None, None)
		assert library.get(rec2.sample_id) is None

	def test_tolerates_all_optional_subsystems_none (self) -> None:
		"""With no similarity/transform/player/events wired, the sample is still
		added to the library without error."""

		library = subsample.library.InstrumentLibrary(max_memory_bytes=0)
		record  = _make_record("lonely")

		subsample.cli._integrate_sample(
			record, library,
			similarity_matrix=None,
			transform_manager=None,
			player_cell=None,
			app_events=None,
		)

		assert library.get(record.sample_id) is record


class TestPrintBanner:

	"""_print_banner reports each ACTIVE subsystem's own settings — a player-only
	run must show the player's output, not the (possibly disabled) recorder."""

	_DEFAULT = subsample.config._locate_default_config()

	def _player_only (self) -> subsample.config.Config:
		import dataclasses
		cfg = subsample.config.load_config(self._DEFAULT)
		# Recorder disabled, but its bit_depth (24) is the fallback source for the
		# player's unset output bit depth — pin it so the fallback is deterministic.
		recorder = dataclasses.replace(
			cfg.recorder, enabled=False,
			audio=dataclasses.replace(cfg.recorder.audio, bit_depth=24),
		)
		player   = dataclasses.replace(
			cfg.player,
			enabled=True,
			midi_map="midi-map-gm-drums.yaml",
			audio=dataclasses.replace(cfg.player.audio, sample_rate=48000, channels=8, bit_depth=None),
		)
		library = dataclasses.replace(cfg.library, directory="samples/captures")
		return dataclasses.replace(cfg, recorder=recorder, player=player, library=library)

	def _recorder_only (self) -> subsample.config.Config:
		import dataclasses
		cfg = subsample.config.load_config(self._DEFAULT)
		recorder = dataclasses.replace(
			cfg.recorder, enabled=True,
			audio=dataclasses.replace(cfg.recorder.audio, sample_rate=44100, bit_depth=24, channels=2),
		)
		player = dataclasses.replace(cfg.player, enabled=False)
		return dataclasses.replace(cfg, recorder=recorder, player=player)

	def test_player_only_shows_player_output (self, capsys: pytest.CaptureFixture) -> None:
		subsample.cli._print_banner(self._player_only())
		line = capsys.readouterr().out

		assert "player" in line
		assert "48000 Hz" in line          # player output rate, not recorder 44100
		assert "8ch" in line               # player output channels, not recorder 2
		assert "midi-map-gm-drums.yaml" in line

	def test_player_only_hides_recorder_fields (self, capsys: pytest.CaptureFixture) -> None:
		subsample.cli._print_banner(self._player_only())
		line = capsys.readouterr().out

		# The disabled recorder's rate and capture-only fields must not appear.
		assert "44100" not in line
		assert "trigger ≥" not in line
		assert "buffer" not in line

	def test_player_output_bit_depth_falls_back_to_recorder (self, capsys: pytest.CaptureFixture) -> None:
		# player.audio.bit_depth is None → resolves to the recorder's 24, matching
		# MidiPlayer's own output_bit_depth fallback.
		subsample.cli._print_banner(self._player_only())
		assert "24-bit" in capsys.readouterr().out

	def test_recorder_only_shows_capture_format (self, capsys: pytest.CaptureFixture) -> None:
		subsample.cli._print_banner(self._recorder_only())
		line = capsys.readouterr().out

		assert "recorder" in line
		assert "44100 Hz" in line
		assert "trigger ≥" in line
		assert "buffer" in line
		# No player output segment.
		assert "map " not in line

	def test_both_modes_show_both_segments (self, capsys: pytest.CaptureFixture) -> None:
		import dataclasses
		cfg = self._player_only()
		cfg = dataclasses.replace(cfg, recorder=dataclasses.replace(cfg.recorder, enabled=True))
		subsample.cli._print_banner(cfg)
		line = capsys.readouterr().out

		assert "recorder + player" in line
		assert "48000 Hz" in line          # player output
		assert "trigger ≥" in line     # recorder capture-only field
		assert "||" in line                # two segments joined
