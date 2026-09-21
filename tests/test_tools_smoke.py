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

import subsample.tools._shared
import subsample.tools.analyze_file
import subsample.tools.similarity_report
import subsample.tools.suggest_loops


def _write_tone (path: pathlib.Path, seconds: float = 0.5, sr: int = 44100) -> None:

	"""Write a short mono sine tone the analysis pipeline can fingerprint."""

	t = numpy.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
	tone = (0.4 * numpy.sin(2.0 * numpy.pi * 220.0 * t)).astype(numpy.float32)
	soundfile.write(str(path), tone, sr, subtype="PCM_16")


def _write_hits (path: pathlib.Path, sr: int = 44100) -> None:

	"""Write a take of four hits, the last one a ghost note well below the rest."""

	buffer = numpy.zeros(int(sr * 2.0), dtype=numpy.float32)
	envelope = numpy.exp(-numpy.linspace(0.0, 12.0, int(0.2 * sr)))
	tone = numpy.sin(2.0 * numpy.pi * 180.0 * numpy.arange(envelope.size) / sr)

	for at, amplitude in ((0.02, 0.9), (0.51, 0.45), (1.00, 0.6), (1.49, 0.08)):
		start = int(at * sr)
		buffer[start:start + envelope.size] += (amplitude * tone * envelope).astype(numpy.float32)

	soundfile.write(str(path), buffer, sr, subtype="PCM_16")


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

	def test_analyze_lists_every_attack_with_its_time_and_level (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		capsys: pytest.CaptureFixture[str],
	) -> None:

		"""A count of onsets does not say whether a take is worth quantizing;
		where each hit lands and how loud it is does."""

		monkeypatch.chdir(tmp_path)
		wav = tmp_path / "hits.wav"
		_write_hits(wav)

		assert subsample.tools.analyze_file.main([str(wav)]) == 0

		attacks = [
			line for line in capsys.readouterr().out.splitlines()
			if line.startswith(" ") and line.strip()[:1].isdigit()
		]

		assert len(attacks) == 4
		assert attacks[0].endswith("0.0dB")     # the hardest hit is the reference
		assert float(attacks[3].split()[-1].removesuffix("dB")) < -10.0    # the ghost note

	def test_analyze_reports_the_same_attacks_from_the_sidecar (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
		capsys: pytest.CaptureFixture[str],
	) -> None:

		"""Per-attack levels are measured from the audio, which a cache hit does
		not load — so the second run has to read it back rather than go quiet."""

		monkeypatch.chdir(tmp_path)
		wav = tmp_path / "hits.wav"
		_write_hits(wav)

		subsample.tools.analyze_file.main([str(wav)])
		cold = capsys.readouterr().out

		subsample.tools.analyze_file.main([str(wav)])
		cached = capsys.readouterr().out

		assert "attacks:  4" in cold
		assert cached == cold

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


class TestNamingFilesTheToolsAccept:

	"""What counts as a file name and what counts as a pattern.

	`glob` reads a bracket as a character class, so `kick [01].wav` matched
	nothing — and because the argument *contained* a pattern character, the
	literal-path fallback was skipped and the tools reported "No files matched"
	for a file sitting right there.  Bracketed names are ordinary in sample
	packs.  Both tools had their own copy of the same block.
	"""

	def test_a_name_with_brackets_is_a_name (self, tmp_path: pathlib.Path) -> None:
		awkward = tmp_path / "kick [01].wav"
		_write_tone(awkward)

		found = subsample.tools._shared.expanded_paths([str(awkward)])

		assert found == [awkward]

	def test_a_pattern_still_expands (self, tmp_path: pathlib.Path) -> None:
		for name in ("a.wav", "b.wav"):
			_write_tone(tmp_path / name)

		found = subsample.tools._shared.expanded_paths([str(tmp_path / "*.wav")])

		assert sorted(path.name for path in found) == ["a.wav", "b.wav"]

	def test_a_pattern_matching_nothing_is_reported (
		self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
	) -> None:
		assert subsample.tools._shared.expanded_paths([str(tmp_path / "*.nope")]) == []
		assert "No files matched" in capsys.readouterr().err

	def test_a_plain_name_that_is_not_there_is_passed_on (self, tmp_path: pathlib.Path) -> None:

		"""So the audio reader gives its own "file not found", not a system error."""

		missing = tmp_path / "absent.wav"

		assert subsample.tools._shared.expanded_paths([str(missing)]) == [missing]


class TestSimilarFindsTheSameReferencesTheAppDoes:

	"""`subsample similar` looked somewhere the app never looks.

	It defaulted to `samples/reference`, which `--init` does not create and
	which the app does not read — the 47 GM fingerprints ship inside the
	package.  So the tool failed with "no reference samples found" in every new
	project, including for the example the README gives.
	"""

	def test_it_uses_the_packaged_set_by_default (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		captures = tmp_path / "samples" / "captures"
		captures.mkdir(parents=True)
		_write_tone(captures / "kick.wav")

		# Exit 0 means it found references to compare against; the old default
		# returned 1 from a clean project, whatever was in the library.
		assert subsample.tools.similarity_report.main([]) == 0

	def test_the_flag_still_overrides_it (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		monkeypatch.chdir(tmp_path)
		empty = tmp_path / "refs"
		empty.mkdir()

		assert subsample.tools.similarity_report.main(["--reference-dir", str(empty)]) == 1
