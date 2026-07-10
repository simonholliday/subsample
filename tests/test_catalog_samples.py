"""Tests for scripts/catalog_samples.py — the sample-property catalog tool.

The script lives in scripts/ (not the subsample package), so it is loaded
here via importlib from its file path.
"""

import csv
import importlib.util
import io
import json
import pathlib
import types
import typing

import pytest

import subsample.cache
import subsample.config

import tests.helpers


def _load_script () -> types.ModuleType:

	"""Load scripts/catalog_samples.py as a module from its file path."""

	script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "catalog_samples.py"
	spec   = importlib.util.spec_from_file_location("catalog_samples", script)
	assert spec is not None and spec.loader is not None

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


catalog_samples = _load_script()


def _fix_sidecar_md5 (wav_path: pathlib.Path, sidecar_path: pathlib.Path) -> None:

	"""Replace the helper sidecar's fake MD5 with the WAV's real digest.

	ensure_sample_assets re-analyses on MD5 mismatch; with the real digest it
	serves the sidecar as-is, so tests see the helper values (440 Hz pitch,
	120 BPM, 2 attacks) rather than a re-analysis of the zero-filled WAV.
	"""

	payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
	payload["audio_md5"] = subsample.cache.compute_audio_md5(wav_path)
	sidecar_path.write_text(json.dumps(payload), encoding="utf-8")


def _edit_sidecar (sidecar_path: pathlib.Path, **section_updates: dict) -> None:

	"""Apply per-section key updates to a sidecar JSON payload."""

	payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

	for section, updates in section_updates.items():
		payload[section].update(updates)

	sidecar_path.write_text(json.dumps(payload), encoding="utf-8")


def _make_pitched_sample (directory: pathlib.Path, stem: str) -> pathlib.Path:

	"""Create a WAV + sidecar that passes both capability tests.

	The helper sidecar's values (440 Hz, confidence 0.92, stability 0.1,
	voiced_fraction 0.9, harmonic_ratio 0.7, duration 1.0, 8 voiced frames)
	pass has_stable_pitch, and its 2 attack times pass has_beat_map.
	"""

	wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(directory, stem)
	_fix_sidecar_md5(wav_path, sidecar_path)
	return wav_path


def _make_unpitched_sample (directory: pathlib.Path, stem: str) -> pathlib.Path:

	"""Create a WAV + sidecar that fails both capability tests.

	No detected fundamental (fails has_stable_pitch) and a single attack
	(fails has_beat_map).
	"""

	wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(directory, stem)
	_edit_sidecar(
		sidecar_path,
		pitch  = {"dominant_pitch_hz": 0.0, "dominant_pitch_class": -1},
		rhythm = {"attack_times": [0.05], "onset_times": [0.05], "onset_count": 1},
	)
	_fix_sidecar_md5(wav_path, sidecar_path)
	return wav_path


def _run_csv (
	capsys: pytest.CaptureFixture,
	argv: list[str],
) -> tuple[list[str], list[dict[str, str]]]:

	"""Run main(argv) and parse captured stdout as CSV.

	Returns (header, rows) with each row as a column→value dict.
	"""

	catalog_samples.main(argv)
	reader = csv.reader(io.StringIO(capsys.readouterr().out))
	header = next(reader)
	rows   = [dict(zip(header, row)) for row in reader]
	return header, rows


class TestColumns:

	"""Tests for the column sets."""

	def test_base_includes_capability_indicators (self) -> None:
		"""The default CSV must carry the two capability columns."""
		columns = catalog_samples._columns(full=False)

		assert "pitched" in columns
		assert "quantizable" in columns

	def test_base_includes_all_stable_pitch_inputs (self) -> None:
		"""All seven has_stable_pitch inputs are present, so a failing sample
		can be diagnosed against the documented thresholds."""
		columns = catalog_samples._columns(full=False)

		for name in (
			"pitch_hz", "voiced_fraction", "voiced_frame_count",
			"pitch_confidence", "pitch_stability_st", "harmonic_ratio",
			"duration_s",
		):
			assert name in columns

	def test_full_is_superset_of_base (self) -> None:
		"""--full appends columns; it never drops or reorders base ones."""
		base = catalog_samples._columns(full=False)
		full = catalog_samples._columns(full=True)

		assert full[:len(base)] == base
		assert "mfcc_00" in full
		assert "mfcc_onset_12" in full
		assert "chroma_C" in full

	def test_no_duplicate_columns (self) -> None:
		"""Every column name is unique — duplicates would silently shadow."""
		full = catalog_samples._columns(full=True)

		assert len(full) == len(set(full))


class TestFmt:

	"""Tests for the CSV float formatter."""

	def test_strips_trailing_zeros (self) -> None:
		assert catalog_samples._fmt(440.0) == "440"

	def test_rounds_to_four_places (self) -> None:
		assert catalog_samples._fmt(0.123456) == "0.1235"

	def test_zero (self) -> None:
		assert catalog_samples._fmt(0.0) == "0"

	def test_negative (self) -> None:
		assert catalog_samples._fmt(-12.5) == "-12.5"

	def test_negative_epsilon_rounds_to_zero (self) -> None:
		"""A tiny negative must not render as the bare string "-"."""
		assert catalog_samples._fmt(-0.000001) == "0"


class TestPitchNote:

	"""Tests for the Hz → note-name column."""

	def test_a4 (self) -> None:
		assert catalog_samples._pitch_note(440.0) == "A4"

	def test_near_semitone_boundary_rounds (self) -> None:
		"""451 Hz is still nearer A4 than A#4."""
		assert catalog_samples._pitch_note(451.0) == "A4"

	def test_unpitched_is_blank (self) -> None:
		assert catalog_samples._pitch_note(0.0) == ""

	def test_above_midi_range_is_blank (self) -> None:
		"""A detected fundamental beyond G9 has no MIDI note name."""
		assert catalog_samples._pitch_note(100000.0) == ""


class TestCollectAudioFiles:

	"""Tests for the directory walk."""

	def test_recursive_and_sorted (self, tmp_path: pathlib.Path) -> None:
		"""Same walk as the library loader: recursive, audio extensions only."""
		sub = tmp_path / "kicks"
		sub.mkdir()
		tests.helpers._make_wav(tmp_path / "b.wav")
		tests.helpers._make_wav(sub / "a.wav")
		(tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")

		found = catalog_samples._collect_audio_files(tmp_path)

		assert found == sorted([tmp_path / "b.wav", sub / "a.wav"])

	def test_skips_sidecars (self, tmp_path: pathlib.Path) -> None:
		""".analysis.json files are metadata, not audio."""
		_make_pitched_sample(tmp_path, "tone")

		found = catalog_samples._collect_audio_files(tmp_path)

		assert found == [tmp_path / "tone.wav"]


class TestCsvOutput:

	"""End-to-end CSV tests over a real (sidecar-backed) directory."""

	def test_header_matches_base_columns (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "tone")

		header, _rows = _run_csv(capsys, [str(tmp_path)])

		assert tuple(header) == catalog_samples._columns(full=False)

	def test_pitched_quantizable_sample (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		"""Values come from the sidecar (cache-first), not a re-analysis."""
		_make_pitched_sample(tmp_path, "tone")

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert len(rows) == 1
		row = rows[0]
		assert row["name"] == "tone.wav"
		assert row["path"] == "tone.wav"
		assert row["pitched"] == "yes"
		assert row["quantizable"] == "yes"
		assert row["pitch_note"] == "A4"
		assert row["pitch_hz"] == "440"
		assert row["tempo_bpm"] == "120"
		assert row["attack_count"] == "2"
		assert row["duration_s"] == "1"

	def test_unpitched_sample (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_unpitched_sample(tmp_path, "hit")

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert len(rows) == 1
		row = rows[0]
		assert row["pitched"] == "no"
		assert row["quantizable"] == "no"
		assert row["pitch_note"] == ""
		assert row["pitch_hz"] == ""
		assert row["attack_count"] == "1"

	def test_subdirectory_paths_are_relative (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		sub = tmp_path / "kicks"
		sub.mkdir()
		_make_pitched_sample(sub, "deep")

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert rows[0]["path"] == str(pathlib.Path("kicks") / "deep.wav")

	def test_full_columns (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		"""--full adds the timbre vectors with the sidecar's values."""
		_make_pitched_sample(tmp_path, "tone")

		header, rows = _run_csv(capsys, [str(tmp_path), "--full"])

		assert tuple(header) == catalog_samples._columns(full=True)
		# helpers._make_timbre: mfcc = (0.0, 1.0, ..., 12.0).
		assert rows[0]["mfcc_00"] == "0"
		assert rows[0]["mfcc_12"] == "12"
		assert rows[0]["sample_rate"] == "44100"

	def test_output_file (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		"""-o writes the same CSV to a file instead of stdout."""
		_make_pitched_sample(tmp_path, "tone")
		out_path = tmp_path / "catalog.csv"

		catalog_samples.main([str(tmp_path), "-o", str(out_path)])

		assert capsys.readouterr().out == ""
		reader = csv.reader(io.StringIO(out_path.read_text(encoding="utf-8")))
		header = next(reader)
		assert tuple(header) == catalog_samples._columns(full=False)
		assert len(list(reader)) == 1

	def test_missing_directory_exits (self, tmp_path: pathlib.Path) -> None:
		with pytest.raises(SystemExit) as excinfo:
			catalog_samples.main([str(tmp_path / "nope")])

		assert excinfo.value.code == 1

	def test_empty_directory_emits_header_only (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		header, rows = _run_csv(capsys, [str(tmp_path)])

		assert tuple(header) == catalog_samples._columns(full=False)
		assert rows == []

	def test_unreadable_audio_is_skipped (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		"""A corrupt file is reported to stderr and omitted from the CSV."""
		_make_pitched_sample(tmp_path, "tone")
		(tmp_path / "broken.wav").write_bytes(b"RIFFnot really a wav")

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert len(rows) == 1
		assert rows[0]["name"] == "tone.wav"

	def test_default_directory_from_config (
		self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture,
		monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""With no directory argument, instrument.directory from config is used."""
		_make_pitched_sample(tmp_path, "tone")

		def fake_load_config (path: typing.Optional[pathlib.Path] = None) -> types.SimpleNamespace:
			return types.SimpleNamespace(instrument=types.SimpleNamespace(directory=str(tmp_path)))

		monkeypatch.setattr(subsample.config, "load_config", fake_load_config)

		_header, rows = _run_csv(capsys, [])

		assert len(rows) == 1
		assert rows[0]["name"] == "tone.wav"


class TestPathsMode:

	"""--pitched / --quantizable replace the CSV with matching paths."""

	def test_pitched_filter (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "tone")
		_make_unpitched_sample(tmp_path, "hit")

		catalog_samples.main([str(tmp_path), "--pitched"])

		out_lines = capsys.readouterr().out.splitlines()
		assert out_lines == [str(tmp_path / "tone.wav")]

	def test_quantizable_filter (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "tone")
		_make_unpitched_sample(tmp_path, "hit")

		catalog_samples.main([str(tmp_path), "--quantizable"])

		out_lines = capsys.readouterr().out.splitlines()
		assert out_lines == [str(tmp_path / "tone.wav")]

	def test_both_filters_require_both (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		"""--pitched --quantizable is an AND: quantizable-only samples drop out."""
		_make_pitched_sample(tmp_path, "tone")

		# Quantizable (2 attacks from the helper) but unpitched (no fundamental).
		wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(tmp_path, "loop")
		_edit_sidecar(sidecar_path, pitch={"dominant_pitch_hz": 0.0})
		_fix_sidecar_md5(wav_path, sidecar_path)

		catalog_samples.main([str(tmp_path), "--pitched", "--quantizable"])

		out_lines = capsys.readouterr().out.splitlines()
		assert out_lines == [str(tmp_path / "tone.wav")]

	def test_no_header_in_paths_mode (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		"""Paths mode output is pipeable — no CSV header line."""
		_make_unpitched_sample(tmp_path, "hit")

		catalog_samples.main([str(tmp_path), "--pitched"])

		assert capsys.readouterr().out == ""

	def test_paths_mode_to_file (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "tone")
		out_path = tmp_path / "pitched.txt"

		catalog_samples.main([str(tmp_path), "--pitched", "-o", str(out_path)])

		assert capsys.readouterr().out == ""
		assert out_path.read_text(encoding="utf-8").splitlines() == [str(tmp_path / "tone.wav")]


def _make_distinct_sample (directory: pathlib.Path, stem: str) -> pathlib.Path:

	"""Create a WAV + sidecar whose feature vector differs sharply from the
	_make_pitched_sample default, so grouping keeps it in its own cluster.

	Patches spectral (bright/inharmonic) and timbre (negated MFCCs) — enough to
	drop the composite cosine well below a near-identical pair.
	"""

	wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(directory, stem)
	_edit_sidecar(
		sidecar_path,
		spectral = {
			"spectral_flatness": 0.95, "harmonic_ratio": 0.02,
			"spectral_centroid": 0.98, "zcr": 0.95, "spectral_contrast": 0.02,
		},
		timbre = {
			"mfcc":       [-1.0] * 13,
			"mfcc_delta": [-1.0] * 13,
			"mfcc_onset": [-1.0] * 13,
		},
	)
	_fix_sidecar_md5(wav_path, sidecar_path)
	return wav_path


class TestSnrDb:

	"""Tests for the snr_db helper column."""

	def test_computes_ratio_db (self) -> None:
		# 20*log10(0.85 / 0.0085) = 40 dB.
		assert catalog_samples._snr_db(0.85, 0.0085) == "40"

	def test_zero_noise_floor_blank (self) -> None:
		assert catalog_samples._snr_db(0.85, 0.0) == ""

	def test_zero_peak_blank (self) -> None:
		assert catalog_samples._snr_db(0.0, 0.01) == ""


class TestTriageColumns:

	"""#5 junk-triage columns (snr_db, near_silent, clipping_risk, noisiness)."""

	def test_columns_present_in_base (self) -> None:
		columns = catalog_samples._columns(full=False)
		assert "snr_db" in columns
		assert "near_silent" in columns
		assert "clipping_risk" in columns
		assert "noisiness" in columns

	def test_noisiness_high_for_stationary_unpitched (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		# Static-like: noise_floor near rms (never quiet) and no detected pitch.
		wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(tmp_path, "static")
		_edit_sidecar(
			sidecar_path,
			level    = {"rms": 0.5, "noise_floor": 0.45},
			spectral = {"voiced_fraction": 0.0},
		)
		_fix_sidecar_md5(wav_path, sidecar_path)

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert float(rows[0]["noisiness"]) > 0.8

	def test_noisiness_low_for_pitched (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		# The default helper sidecar is voiced_fraction 0.9 → pitched → low noisiness.
		_make_pitched_sample(tmp_path, "tone")

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert float(rows[0]["noisiness"]) < 0.2

	def test_normal_sample_flags_no (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		# Helper level: peak 0.85, rms 0.25, noise_floor 0.01 — a healthy event.
		_make_pitched_sample(tmp_path, "tone")

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert rows[0]["near_silent"] == "no"
		assert rows[0]["clipping_risk"] == "no"
		assert rows[0]["snr_db"] != ""

	def test_near_silent_flagged (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(tmp_path, "quiet")
		_edit_sidecar(sidecar_path, level={"peak": 0.001})
		_fix_sidecar_md5(wav_path, sidecar_path)

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert rows[0]["near_silent"] == "yes"

	def test_clipping_risk_flagged (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(tmp_path, "hot")
		_edit_sidecar(sidecar_path, level={"peak": 0.999})
		_fix_sidecar_md5(wav_path, sidecar_path)

		_header, rows = _run_csv(capsys, [str(tmp_path)])

		assert rows[0]["clipping_risk"] == "yes"


class TestGroupKeeper:

	"""Unit tests for _group_keeper selection."""

	def _record (self, name: str, peak: float, rms: float) -> subsample.library.SampleRecord:
		import subsample.analysis
		return subsample.library.SampleRecord(
			sample_id   = 0,
			name        = name,
			spectral    = tests.helpers._make_spectral(),
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = tests.helpers._make_pitch(),
			timbre      = tests.helpers._make_timbre(),
			level       = subsample.analysis.LevelResult(peak=peak, rms=rms),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 1.0,
		)

	def test_highest_peak_wins (self) -> None:
		records = [self._record("a", 0.2, 0.1), self._record("b", 0.8, 0.1), self._record("c", 0.5, 0.1)]
		assert catalog_samples._group_keeper(records, [0, 1, 2]) == 1

	def test_rms_breaks_peak_tie (self) -> None:
		records = [self._record("a", 0.5, 0.1), self._record("b", 0.5, 0.3)]
		assert catalog_samples._group_keeper(records, [0, 1]) == 1

	def test_name_breaks_full_tie (self) -> None:
		records = [self._record("z", 0.5, 0.1), self._record("a", 0.5, 0.1)]
		# Same peak and rms — lexicographically lower name ("a") wins.
		assert catalog_samples._group_keeper(records, [0, 1]) == 1


class TestRecordFromAssets:

	"""_record_from_assets wraps a load tuple faithfully."""

	def test_fields_mapped (self, tmp_path: pathlib.Path) -> None:
		wav_path = _make_pitched_sample(tmp_path, "tone")
		assets   = subsample.cache.ensure_sample_assets(wav_path, with_preview=False)
		assert assets is not None

		record = catalog_samples._record_from_assets(7, wav_path, assets)

		assert record.sample_id == 7
		assert record.name == "tone"
		assert record.filepath == wav_path
		assert record.audio is None
		assert record.pitch.dominant_pitch_hz == pytest.approx(440.0)


class TestGroupMode:

	"""#1 --group near-duplicate clustering."""

	def test_group_columns_prepended (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "tone")

		header, _rows = _run_csv(capsys, [str(tmp_path), "--group"])

		assert header[:3] == ["group", "group_size", "group_keeper"]

	def test_identical_samples_grouped_with_one_keeper (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		# Three byte-identical copies → identical sidecars → one group of three.
		for stem in ("a", "b", "c"):
			_make_pitched_sample(tmp_path, stem)

		_header, rows = _run_csv(capsys, [str(tmp_path), "--group", "--similarity-threshold", "0.99"])

		assert len(rows) == 3
		assert {r["group"] for r in rows} == {"1"}
		assert all(r["group_size"] == "3" for r in rows)
		assert sum(1 for r in rows if r["group_keeper"] == "yes") == 1

	def test_distinct_sample_separate_group (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "twin_a")
		_make_pitched_sample(tmp_path, "twin_b")
		_make_distinct_sample(tmp_path, "loner")

		_header, rows = _run_csv(capsys, [str(tmp_path), "--group", "--similarity-threshold", "0.99"])

		by_name = {r["name"]: r for r in rows}
		# The twins share a group; the loner is in its own.
		assert by_name["twin_a.wav"]["group"] == by_name["twin_b.wav"]["group"]
		assert by_name["loner.wav"]["group"] != by_name["twin_a.wav"]["group"]
		assert by_name["loner.wav"]["group_size"] == "1"

	def test_keeper_first_within_group (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		# Two identical-vector copies but different levels: the louder is keeper
		# and must be emitted first in its group block.
		for stem, peak in (("soft", 0.3), ("loud", 0.9)):
			wav_path, sidecar_path = tests.helpers._write_wav_and_sidecar(tmp_path, stem)
			_edit_sidecar(sidecar_path, level={"peak": peak})
			_fix_sidecar_md5(wav_path, sidecar_path)

		_header, rows = _run_csv(capsys, [str(tmp_path), "--group", "--similarity-threshold", "0.99"])

		assert rows[0]["name"] == "loud.wav"
		assert rows[0]["group_keeper"] == "yes"
		assert rows[1]["group_keeper"] == "no"

	def test_largest_group_first (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		# A pile of 3 identical + a single distinct → group 1 is the big pile.
		for stem in ("t1", "t2", "t3"):
			_make_pitched_sample(tmp_path, stem)
		_make_distinct_sample(tmp_path, "solo")

		_header, rows = _run_csv(capsys, [str(tmp_path), "--group", "--similarity-threshold", "0.99"])

		assert rows[0]["group"] == "1"
		assert rows[0]["group_size"] == "3"

	def test_paths_mode_emits_keepers_only (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		for stem in ("a", "b", "c"):
			_make_pitched_sample(tmp_path, stem)

		catalog_samples.main([str(tmp_path), "--group", "--pitched", "--similarity-threshold", "0.99"])

		out_lines = capsys.readouterr().out.splitlines()
		# One group of three pitched copies → exactly one keeper path.
		assert len(out_lines) == 1


class TestOrderSimilarity:

	"""#2 --order similarity nearest-neighbour chain."""

	def test_all_rows_present (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "a")
		_make_distinct_sample(tmp_path, "b")
		_make_pitched_sample(tmp_path, "c")

		_header, rows = _run_csv(capsys, [str(tmp_path), "--order", "similarity"])

		assert {r["name"] for r in rows} == {"a.wav", "b.wav", "c.wav"}

	def test_no_group_columns (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "a")

		header, _rows = _run_csv(capsys, [str(tmp_path), "--order", "similarity"])

		assert "group" not in header

	def test_paths_mode_ordered (self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
		_make_pitched_sample(tmp_path, "a")
		_make_pitched_sample(tmp_path, "b")

		catalog_samples.main([str(tmp_path), "--pitched", "--order", "similarity"])

		out_lines = capsys.readouterr().out.splitlines()
		assert len(out_lines) == 2
