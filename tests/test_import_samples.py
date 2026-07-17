"""Tests for scripts/import_samples.py — the bulk sample-import tool.

The script lives in scripts/ (not the subsample package), so it is loaded here
via importlib from its file path.
"""

import importlib.util
import pathlib
import types

import numpy
import soundfile

import subsample.analysis


def _load_script () -> types.ModuleType:

	"""Load scripts/import_samples.py as a module from its file path."""

	script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "import_samples.py"
	spec   = importlib.util.spec_from_file_location("import_samples", script)
	assert spec is not None and spec.loader is not None

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


import_samples = _load_script()


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

		assert import_samples._import_file(src, out, force=True, float_ceiling_dbfs=-1.0)

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

		assert import_samples._import_file(src, out, force=True, float_ceiling_dbfs=None)

		data, _ = soundfile.read(str(out / "hot.wav"), always_2d=True)
		# Without a ceiling the over-unity peaks clip to the rail.
		assert int(numpy.sum(numpy.abs(data) >= 0.9995)) > 0
