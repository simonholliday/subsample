"""Tests for subsample/channel.py — multichannel routing and mixing matrices."""

import numpy
import pytest

import subsample.channel


class TestStandardLayouts:

	def test_all_expected_counts_present (self) -> None:
		"""Standard layouts exist for mono, stereo, quad, 5.1, 7.1."""
		for n in (1, 2, 4, 6, 8):
			assert n in subsample.channel.STANDARD_LAYOUTS
			assert len(subsample.channel.STANDARD_LAYOUTS[n]) == n


class TestBuildMixMatrix:

	# -- Identity (in == out, no pan) --

	def test_stereo_identity (self) -> None:
		"""Stereo in, stereo out, no pan → identity matrix."""
		mat = subsample.channel.build_mix_matrix(2, 2)
		numpy.testing.assert_array_almost_equal(mat, numpy.eye(2))

	def test_quad_identity (self) -> None:
		"""Quad in, quad out, no pan → identity matrix."""
		mat = subsample.channel.build_mix_matrix(4, 4)
		numpy.testing.assert_array_almost_equal(mat, numpy.eye(4))

	# -- Downmix (in > out) --

	def test_5_1_to_stereo_itu (self) -> None:
		"""5.1 to stereo uses ITU-R BS.775 coefficients."""
		mat = subsample.channel.build_mix_matrix(6, 2)
		s = float(numpy.sqrt(0.5))

		# Left output: FL + 0.707*FC + 0.707*BL
		assert mat[0, 0] == pytest.approx(1.0)
		assert mat[0, 2] == pytest.approx(s)
		assert mat[0, 4] == pytest.approx(s)
		assert mat[0, 3] == pytest.approx(0.0)  # LFE discarded

		# Right output: FR + 0.707*FC + 0.707*BR
		assert mat[1, 1] == pytest.approx(1.0)
		assert mat[1, 2] == pytest.approx(s)
		assert mat[1, 5] == pytest.approx(s)

	def test_7_1_to_stereo (self) -> None:
		"""7.1 to stereo folds centre, backs, and sides."""
		mat = subsample.channel.build_mix_matrix(8, 2)
		assert mat.shape == (2, 8)
		s = float(numpy.sqrt(0.5))

		# Left: FL + 0.707*(FC + BL + SL)
		assert mat[0, 0] == pytest.approx(1.0)
		assert mat[0, 6] == pytest.approx(s)  # SL

	def test_quad_to_stereo (self) -> None:
		"""Quad to stereo folds backs at -3 dB."""
		mat = subsample.channel.build_mix_matrix(4, 2)
		s = float(numpy.sqrt(0.5))
		assert mat[0, 2] == pytest.approx(s)  # BL into L
		assert mat[1, 3] == pytest.approx(s)  # BR into R

	def test_7_1_to_5_1_chain (self) -> None:
		"""7.1 to 5.1 folds sides into backs."""
		mat = subsample.channel.build_mix_matrix(8, 6)
		assert mat.shape == (6, 8)
		s = float(numpy.sqrt(0.5))
		assert mat[4, 6] == pytest.approx(s)  # SL folds into BL
		assert mat[5, 7] == pytest.approx(s)  # SR folds into BR

	# -- Upmix (in < out) --

	def test_stereo_to_5_1_conservative (self) -> None:
		"""Stereo to 5.1: FL=L, FR=R, rest silent."""
		mat = subsample.channel.build_mix_matrix(2, 6)
		assert mat.shape == (6, 2)
		assert mat[0, 0] == pytest.approx(1.0)  # FL = L
		assert mat[1, 1] == pytest.approx(1.0)  # FR = R
		assert mat[2, 0] == pytest.approx(0.0)  # FC silent
		assert mat[3, 0] == pytest.approx(0.0)  # LFE silent
		assert mat[4, 0] == pytest.approx(0.0)  # BL silent
		assert mat[5, 0] == pytest.approx(0.0)  # BR silent

	def test_mono_to_stereo_no_pan (self) -> None:
		"""Mono to stereo without pan: maps to FL only (conservative upmix)."""
		mat = subsample.channel.build_mix_matrix(1, 2)
		assert mat.shape == (2, 1)
		# Conservative: mono maps to first output channel only.
		assert mat[0, 0] == pytest.approx(1.0)
		assert mat[1, 0] == pytest.approx(0.0)

	# -- Pan weights --

	def test_mono_to_stereo_centre_pan (self) -> None:
		"""Mono to stereo with [50, 50] pan → equal-power centre."""
		weights = numpy.array([50.0, 50.0], dtype=numpy.float32)
		mat = subsample.channel.build_mix_matrix(1, 2, pan_weights=weights)
		s = float(numpy.sqrt(0.5))
		assert mat[0, 0] == pytest.approx(s, abs=1e-5)
		assert mat[1, 0] == pytest.approx(s, abs=1e-5)

	def test_mono_to_stereo_hard_left (self) -> None:
		"""Mono to stereo with [100, 0] pan → all left."""
		weights = numpy.array([100.0, 0.0], dtype=numpy.float32)
		mat = subsample.channel.build_mix_matrix(1, 2, pan_weights=weights)
		assert mat[0, 0] == pytest.approx(1.0, abs=1e-5)
		assert mat[1, 0] == pytest.approx(0.0, abs=1e-5)

	def test_stereo_with_pan_weights (self) -> None:
		"""Stereo to stereo with pan weights modulates the identity."""
		weights = numpy.array([75.0, 25.0], dtype=numpy.float32)
		mat = subsample.channel.build_mix_matrix(2, 2, pan_weights=weights)
		# Left channel gain > right channel gain (panned left).
		assert mat[0, 0] > mat[1, 1]

	def test_pan_targets_5_1_output_stereo (self) -> None:
		"""Pan weights for 5.1 (6 weights) on a stereo output: auto fold-down."""
		weights = numpy.array([50.0, 50.0, 0.0, 0.0, 30.0, 30.0], dtype=numpy.float32)
		mat = subsample.channel.build_mix_matrix(1, 2, pan_weights=weights)
		assert mat.shape == (2, 1)
		# Both L and R should have signal (front + surround contribution).
		assert mat[0, 0] > 0.0
		assert mat[1, 0] > 0.0

	def test_all_zero_pan_produces_silence (self) -> None:
		"""All-zero pan weights produce a zero matrix."""
		weights = numpy.array([0.0, 0.0], dtype=numpy.float32)
		mat = subsample.channel.build_mix_matrix(1, 2, pan_weights=weights)
		numpy.testing.assert_array_equal(mat, numpy.zeros((2, 1)))

	# -- Matrix shapes --

	def test_matrix_shape (self) -> None:
		"""Output shape is always (out_ch, in_ch)."""
		for in_ch, out_ch in [(1, 2), (2, 6), (6, 2), (8, 2), (2, 8), (4, 4)]:
			mat = subsample.channel.build_mix_matrix(in_ch, out_ch)
			assert mat.shape == (out_ch, in_ch), f"Failed for {in_ch}→{out_ch}"

	def test_matrix_dtype (self) -> None:
		"""Matrices are float32."""
		mat = subsample.channel.build_mix_matrix(6, 2)
		assert mat.dtype == numpy.float32


class TestNormalizeMixMatrix:

	def test_constant_power (self) -> None:
		"""Each row's sum of squared coefficients should be 1.0 after normalisation."""
		mat = numpy.array([[1.0, 0.5], [0.5, 1.0]], dtype=numpy.float32)
		normed = subsample.channel.normalize_mix_matrix(mat)

		for row in range(normed.shape[0]):
			power = float(numpy.sum(normed[row] ** 2))
			assert power == pytest.approx(1.0, abs=1e-5)

	def test_zero_row_unchanged (self) -> None:
		"""All-zero rows are not normalised (avoids division by zero)."""
		mat = numpy.array([[1.0, 0.0], [0.0, 0.0]], dtype=numpy.float32)
		normed = subsample.channel.normalize_mix_matrix(mat)
		numpy.testing.assert_array_equal(normed[1], [0.0, 0.0])


class TestRouteToDevice:

	"""Tests for route_to_device() — physical output routing."""

	def test_none_output_map_passthrough (self) -> None:
		"""None output_map with matching row count returns matrix unchanged."""
		mat = numpy.array([[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32)
		result = subsample.channel.route_to_device(mat, 2, None)
		numpy.testing.assert_array_equal(result, mat)

	def test_none_output_map_pads_with_zeros (self) -> None:
		"""None output_map pads with zero rows to reach device_channels."""
		mat = numpy.array([[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32)
		result = subsample.channel.route_to_device(mat, 8, None)
		assert result.shape == (8, 2)
		numpy.testing.assert_array_equal(result[:2, :], mat)
		numpy.testing.assert_array_equal(result[2:, :], 0.0)

	def test_stereo_to_outputs_3_4 (self) -> None:
		"""Route stereo to device outputs 3-4 (0-indexed: 2, 3)."""
		mat = numpy.array([[0.7, 0.3], [0.3, 0.7]], dtype=numpy.float32)
		result = subsample.channel.route_to_device(mat, 8, (2, 3))
		assert result.shape == (8, 2)
		numpy.testing.assert_array_equal(result[2, :], mat[0, :])
		numpy.testing.assert_array_equal(result[3, :], mat[1, :])
		# All other rows are zero.
		for row in [0, 1, 4, 5, 6, 7]:
			numpy.testing.assert_array_equal(result[row, :], 0.0)

	def test_identity_default_routing (self) -> None:
		"""output_map (0, 1) on a 2-ch device equals the original matrix."""
		mat = numpy.array([[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32)
		result = subsample.channel.route_to_device(mat, 2, (0, 1))
		numpy.testing.assert_array_equal(result, mat)

	def test_out_of_range_raises (self) -> None:
		"""Index >= device_channels raises ValueError."""
		mat = numpy.array([[1.0], [0.0]], dtype=numpy.float32)
		with pytest.raises(ValueError, match="out of range"):
			subsample.channel.route_to_device(mat, 2, (0, 2))

	def test_duplicate_indices_raises (self) -> None:
		"""Duplicate indices raise ValueError."""
		mat = numpy.array([[1.0], [0.0]], dtype=numpy.float32)
		with pytest.raises(ValueError, match="duplicate"):
			subsample.channel.route_to_device(mat, 4, (1, 1))

	def test_length_mismatch_raises (self) -> None:
		"""output_map length != matrix rows raises ValueError."""
		mat = numpy.array([[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32)
		with pytest.raises(ValueError, match="does not match"):
			subsample.channel.route_to_device(mat, 8, (2, 3, 4))
