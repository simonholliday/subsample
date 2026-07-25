"""Tests for subsample.tools.import_samples (`subsample import`) — the bulk
sample-import tool."""

import pathlib

import numpy
import pytest
import soundfile

import subsample.analysis
import subsample.config


import subsample.tools.import_samples

# Short alias — the tool was a standalone script before subsample.tools existed,
# and every test below refers to it by its old bare module name.
import_samples = subsample.tools.import_samples


class TestHotFloatImport:

	"""A hot 32-bit-float source must be scaled to the ceiling, not hard-clipped,
	and the sidecar must describe the audio actually written (H1)."""

	def _write_hot_float (self, path: pathlib.Path) -> None:
		t = numpy.linspace(0, 1, 48000, endpoint=False)
		# +6 dBFS — legal for float, above what 16-bit PCM can hold.
		soundfile.write(str(path), (numpy.sin(2 * numpy.pi * 440 * t) * 2.0).astype(numpy.float32),
		                48000, subtype="FLOAT")

	def test_hot_float_scaled_not_clipped (self, tmp_path: pathlib.Path) -> None:
		src = tmp_path / "hot.wav"
		self._write_hot_float(src)
		out = tmp_path / "out"
		out.mkdir()

		assert import_samples._import_file(
			src, out, force=True, float_ceiling_dbfs=-1.0,
			rhythm_cfg=subsample.config.AnalysisConfig(),
		)

		written = out / "hot.wav"
		data, _ = soundfile.read(str(written), always_2d=True)
		peak = float(numpy.max(numpy.abs(data)))

		# Peak sits at the -1 dBFS ceiling; nothing pinned to the ±full-scale rail.
		assert peak < 0.9995
		assert abs(peak - 10.0 ** (-1.0 / 20.0)) < 0.02
		assert int(numpy.sum(numpy.abs(data) >= 0.9995)) == 0

	def test_none_ceiling_preserves_legacy_clip (self, tmp_path: pathlib.Path) -> None:
		"""float_ceiling_dbfs=None keeps the historical hard-clip behaviour."""
		src = tmp_path / "hot.wav"
		self._write_hot_float(src)
		out = tmp_path / "out"
		out.mkdir()

		assert import_samples._import_file(
			src, out, force=True, float_ceiling_dbfs=None,
			rhythm_cfg=subsample.config.AnalysisConfig(),
		)

		data, _ = soundfile.read(str(out / "hot.wav"), always_2d=True)
		# Without a ceiling the over-unity peaks clip to the rail.
		assert int(numpy.sum(numpy.abs(data) >= 0.9995)) > 0


class TestMainStemCollisions:

	"""Two inputs sharing a stem resolve to one target name.

	Serially the "already exists" guard caught this after the first write.  Under
	the worker pool every worker evaluated that guard BEFORE any of them had
	written, so they all wrote the same path, the last one won, and the summary
	reported every file as imported — silent data loss with a success message.
	"""

	def _make_source (self, path: pathlib.Path, level: float) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		rng   = numpy.random.default_rng(int(level * 100))
		audio = (numpy.exp(-numpy.arange(22050) / 4000.0) * rng.standard_normal(22050)).astype(numpy.float32)
		audio *= level / numpy.max(numpy.abs(audio))
		soundfile.write(str(path), audio, 44100, subtype="PCM_16")

	def test_same_stem_inputs_import_once_and_are_counted_honestly (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
	) -> None:
		monkeypatch.chdir(tmp_path)
		target = tmp_path / "out"
		target.mkdir()

		sources = []

		for index in range(6):
			source = tmp_path / f"pack{index}" / "kick.wav"
			self._make_source(source, 0.2 + 0.1 * index)
			sources.append(str(source))

		rc = subsample.tools.import_samples.main([*sources, "--to", str(target)])

		written = list(target.glob("*.wav")) + list(target.glob("*.flac"))
		assert len(written) == 1, "more than one source claimed the same name"

		out = capsys.readouterr()
		assert "Imported 1 file(s), skipped 5" in out.out
		assert rc == 0

	def test_distinct_stems_all_import (
		self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""The collision guard must not suppress legitimate parallel imports."""
		monkeypatch.chdir(tmp_path)
		target = tmp_path / "out"
		target.mkdir()

		sources = []

		for index in range(6):
			source = tmp_path / "pack" / f"hit{index}.wav"
			self._make_source(source, 0.5)
			sources.append(str(source))

		subsample.tools.import_samples.main([*sources, "--to", str(target)])

		assert len(list(target.glob("*.wav"))) == 6
