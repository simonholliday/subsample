"""Smoke tests for the subsample.tools subcommands that had no coverage —
`subsample analyze`, `subsample loops`, `subsample similar`.

Each drives the tool's public main(argv) end to end from a clean working
directory (so load_config loads the packaged defaults, not a stray config),
asserting a sane exit code and, where cheap, a line of real output.  These are
deliberately shallow — they guard the argparse spec, the config wiring, and the
happy/empty paths from silently breaking, which is exactly what a rename or a
broken glob loop would do.
"""

import pathlib

import numpy
import pytest
import soundfile

import subsample.tools.analyze_file
import subsample.tools.similarity_report
import subsample.tools.suggest_loops


def _write_tone (path: pathlib.Path, seconds: float = 0.5, sr: int = 44100) -> None:

	"""Write a short mono sine tone the analysis pipeline can fingerprint."""

	t = numpy.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
	tone = (0.4 * numpy.sin(2.0 * numpy.pi * 220.0 * t)).astype(numpy.float32)
	soundfile.write(str(path), tone, sr, subtype="PCM_16")


class TestAnalyzeSmoke:

	def test_analyze_prints_metrics_and_exits_zero (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		capsys: pytest.CaptureFixture[str],
	) -> None:
		monkeypatch.chdir(tmp_path)
		wav = tmp_path / "tone.wav"
		_write_tone(wav)

		rc = subsample.tools.analyze_file.main([str(wav)])

		out = capsys.readouterr().out
		assert rc == 0
		# The metric lines the tool always prints, including the loop line.
		for label in ("rhythm:", "spectral:", "pitch:", "level:", "loop:"):
			assert label in out

	def test_analyze_no_match_glob_returns_one (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		rc = subsample.tools.analyze_file.main(["*.nomatch"])
		assert rc == 1


class TestLoopsSmoke:

	def test_loops_reports_and_exits_zero (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		_write_tone(tmp_path / "pad.wav", seconds=1.0)

		rc = subsample.tools.suggest_loops.main([str(tmp_path)])
		assert rc == 0

	def test_loops_no_files_returns_one (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		rc = subsample.tools.suggest_loops.main([str(tmp_path / "empty")])
		assert rc == 1


class TestSimilarSmoke:

	def test_similar_no_references_returns_one (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		empty_ref = tmp_path / "refs"
		empty_ref.mkdir()

		rc = subsample.tools.similarity_report.main(
			["--reference-dir", str(empty_ref)],
		)
		# No reference fingerprints to compare against — a clean exit 1.
		assert rc == 1
