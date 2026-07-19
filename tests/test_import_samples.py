"""Tests for subsample.tools.import_samples (`subsample import`) — the bulk
sample-import tool."""

import pathlib

import numpy
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
