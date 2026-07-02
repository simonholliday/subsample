"""Tests for subsample.cli — argument parsing and detection pipeline helpers."""

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
	snr_threshold_db: float = 6.0,
	trim_pre_samples: int = 8,
	trim_post_samples: int = 8,
) -> subsample.config.DetectionConfig:
	"""Return a DetectionConfig suitable for unit tests."""
	return subsample.config.DetectionConfig(
		snr_threshold_db = snr_threshold_db,
		ema_alpha        = 0.1,
		hold_time        = 0.1,
		warmup_seconds   = 0.0,
		trim_pre_samples = trim_pre_samples,
		trim_post_samples = trim_post_samples,
	)


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
			result = subsample.cli._process_chunk(silent_chunk, buf, detector, cfg)
			assert result is None

	def test_loud_burst_returns_segment (self) -> None:
		"""A loud burst followed by silence should trigger one detected segment."""
		sample_rate = 44100
		chunk_size = 512
		buf, detector = _make_buffer_and_detector(
			sample_rate=sample_rate,
			chunk_size=chunk_size,
			detection_cfg=_make_detection_cfg(snr_threshold_db=3.0),
		)
		cfg = _make_detection_cfg(snr_threshold_db=3.0)

		# Feed low-amplitude chunks to establish an ambient floor before the burst.
		# If the ambient is near zero, the first loud chunk seeds ambient directly
		# to the signal level (giving 0 dB SNR), which would fail to trigger.
		# Amplitude 100 gives ~38 dB headroom against the 8000-amplitude signal.
		ambient = numpy.full((chunk_size, 1), 100, dtype=numpy.int16)
		for _ in range(30):
			subsample.cli._process_chunk(ambient, buf, detector, cfg)

		# Feed loud chunks — well above the established ambient floor (~38 dB SNR)
		loud = numpy.full((chunk_size, 1), 8000, dtype=numpy.int16)
		for _ in range(10):
			subsample.cli._process_chunk(loud, buf, detector, cfg)

		# Feed trailing silence to close the recording (hold_time expires)
		# hold_time=0.1s at 44100 Hz / 512 frames ≈ 9 chunks
		silent = numpy.zeros((chunk_size, 1), dtype=numpy.int16)
		segments = []
		for _ in range(30):
			result = subsample.cli._process_chunk(silent, buf, detector, cfg)
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
			detection_cfg=_make_detection_cfg(snr_threshold_db=3.0),
		)
		cfg = _make_detection_cfg(snr_threshold_db=3.0)

		ambient = numpy.full((chunk_size, 1), 100, dtype=numpy.int16)
		loud = numpy.full((chunk_size, 1), 8000, dtype=numpy.int16)
		silent = numpy.zeros((chunk_size, 1), dtype=numpy.int16)

		for _ in range(30):
			subsample.cli._process_chunk(ambient, buf, detector, cfg)
		for _ in range(10):
			subsample.cli._process_chunk(loud, buf, detector, cfg)

		result = None
		for _ in range(30):
			r = subsample.cli._process_chunk(silent, buf, detector, cfg)
			if r is not None:
				result = r
				break

		assert result is not None, "Expected a segment to be detected but got None"
		assert isinstance(result, numpy.ndarray)


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
			output:
			  directory: {out_dir}
			detection:
			  snr_threshold_db: 10.0
			  warmup_seconds: 0.0
			  hold_time: 0.15
		"""))
		cfg = subsample.config.load_config(config_file)

		subsample.cli._process_input_files([wav_path], cfg)

		segments = sorted(out_dir.glob("field*"))
		assert segments, "no segments written for a clear burst"

	def test_missing_and_unreadable_files_skipped (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
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

	def test_cross_file_stem_collision_skipped (self, tmp_path: pathlib.Path) -> None:
		"""A hot-dropped file whose stem matches a DIFFERENT already-loaded
		file is skipped with a warning — the silent replace would misroute
		MIDI lookups now and crash the next startup (which hard-rejects the
		duplicate pair)."""

		library = subsample.library.InstrumentLibrary(max_memory_bytes=0)
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

		# The original record survives; the clashing one was never integrated.
		assert library.get(first.sample_id) is first
		assert library.get(clash.sample_id) is None
		similarity.add.assert_not_called()

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
