"""Tests for subsample/channel.py — multichannel routing and mixing matrices."""

import typing

import numpy
import pytest

import subsample.channel
import subsample.query


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
		"""Mono to stereo without pan defaults to CENTRE — equal-power L/R,
		matching `pan: [50, 50]` — not hard-left."""
		mat = subsample.channel.build_mix_matrix(1, 2)
		assert mat.shape == (2, 1)
		assert mat[0, 0] == pytest.approx(numpy.sqrt(0.5))
		assert mat[1, 0] == pytest.approx(numpy.sqrt(0.5))

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


# ---------------------------------------------------------------------------
# Helpers used by TestBuildExtractMatrix
# ---------------------------------------------------------------------------

_SQRT2_INV = float(numpy.sqrt(0.5))
_SQRT3_INV = float(numpy.sqrt(1.0 / 3.0))
_SQRT5_INV = float(numpy.sqrt(1.0 / 5.0))
_SQRT6_INV = float(numpy.sqrt(1.0 / 6.0))
_SQRT7_INV = float(numpy.sqrt(1.0 / 7.0))
_SQRT2     = float(numpy.sqrt(2.0))


def _spec (kind: str, channel_index: typing.Optional[int] = None) -> subsample.query.ExtractSpec:
	"""Test factory — builds an ExtractSpec without a lot of typing."""
	return subsample.query.ExtractSpec(kind=kind, channel_index=channel_index)


class TestBuildExtractMatrix:

	"""Tests for build_extract_matrix() — microphone-pattern extraction."""

	# -- PCM omni: equal-energy sum, LFE excluded on 5.1/7.1 --

	def test_omni_mono (self) -> None:
		"""Omni on mono is identity (no collapse needed)."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 1, "pcm")
		numpy.testing.assert_allclose(mat, [[1.0]])

	def test_omni_stereo (self) -> None:
		"""Omni on stereo is the M of M/S: (L+R)/√2."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 2, "pcm")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, _SQRT2_INV]])

	def test_omni_quad (self) -> None:
		"""Omni on quad sums all 4 channels at 1/2 each."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[0.5, 0.5, 0.5, 0.5]])

	def test_omni_5_1_excludes_lfe (self) -> None:
		"""Omni on 5.1: equal sum across audible channels; LFE index (3) is zero."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 6, "pcm")
		expected = [[_SQRT5_INV, _SQRT5_INV, _SQRT5_INV, 0.0, _SQRT5_INV, _SQRT5_INV]]
		numpy.testing.assert_allclose(mat, expected)

	def test_omni_7_1_excludes_lfe (self) -> None:
		"""Omni on 7.1: equal sum across all 7 audible channels; LFE excluded."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 8, "pcm")
		expected = [[_SQRT7_INV, _SQRT7_INV, _SQRT7_INV, 0.0,
		             _SQRT7_INV, _SQRT7_INV, _SQRT7_INV, _SQRT7_INV]]
		numpy.testing.assert_allclose(mat, expected)

	# -- PCM side: L-R figure-eight dipole --

	def test_side_stereo (self) -> None:
		"""Side on stereo is the S of M/S: (L-R)/√2."""
		mat = subsample.channel.build_extract_matrix(_spec("side"), 2, "pcm")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, -_SQRT2_INV]])

	def test_side_quad (self) -> None:
		"""Side on quad uses both L/R pairs: (FL-FR+BL-BR)/2."""
		mat = subsample.channel.build_extract_matrix(_spec("side"), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[0.5, -0.5, 0.5, -0.5]])

	def test_side_5_1 (self) -> None:
		"""Side on 5.1: FL-FR + BL-BR normalised; centre and LFE excluded."""
		mat = subsample.channel.build_extract_matrix(_spec("side"), 6, "pcm")
		numpy.testing.assert_allclose(mat, [[0.5, -0.5, 0.0, 0.0, 0.5, -0.5]])

	# -- PCM depth: F-B dipole (requires ≥ 4 channels) --

	def test_depth_quad (self) -> None:
		"""Depth on quad: (FL+FR - BL-BR)/2."""
		mat = subsample.channel.build_extract_matrix(_spec("depth"), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[0.5, 0.5, -0.5, -0.5]])

	def test_depth_5_1 (self) -> None:
		"""Depth on 5.1: front group (FL,FR,FC) - back group (BL,BR)."""
		mat = subsample.channel.build_extract_matrix(_spec("depth"), 6, "pcm")
		expected = [[_SQRT6_INV, _SQRT6_INV, _SQRT6_INV * _SQRT2, 0.0,
		             -_SQRT6_INV, -_SQRT6_INV]]
		numpy.testing.assert_allclose(mat, expected)

	# -- PCM left / right cardioids --

	def test_left_stereo (self) -> None:
		"""Left on stereo is just the L channel."""
		mat = subsample.channel.build_extract_matrix(_spec("left"), 2, "pcm")
		numpy.testing.assert_allclose(mat, [[1.0, 0.0]])

	def test_right_stereo (self) -> None:
		"""Right on stereo is just the R channel."""
		mat = subsample.channel.build_extract_matrix(_spec("right"), 2, "pcm")
		numpy.testing.assert_allclose(mat, [[0.0, 1.0]])

	def test_left_5_1 (self) -> None:
		"""Left on 5.1 uses FL + BL (left-side speakers only)."""
		mat = subsample.channel.build_extract_matrix(_spec("left"), 6, "pcm")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, 0.0, 0.0, 0.0, _SQRT2_INV, 0.0]])

	def test_right_7_1 (self) -> None:
		"""Right on 7.1 uses FR + BR + SR."""
		mat = subsample.channel.build_extract_matrix(_spec("right"), 8, "pcm")
		numpy.testing.assert_allclose(mat, [[0.0, _SQRT3_INV, 0.0, 0.0,
		                                    0.0, _SQRT3_INV, 0.0, _SQRT3_INV]])

	# -- PCM front / back cardioids --

	def test_front_stereo_equals_omni (self) -> None:
		"""Front on stereo has no F/B distinction → matrix equals omni."""
		front = subsample.channel.build_extract_matrix(_spec("front"), 2, "pcm")
		omni  = subsample.channel.build_extract_matrix(_spec("omni"),  2, "pcm")
		numpy.testing.assert_allclose(front, omni)

	def test_front_quad (self) -> None:
		"""Front on quad uses FL + FR (no centre channel)."""
		mat = subsample.channel.build_extract_matrix(_spec("front"), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, _SQRT2_INV, 0.0, 0.0]])

	def test_front_5_1_includes_centre (self) -> None:
		"""Front on 5.1 uses FL + FR + FC; FC gets √2 weight to match boosted centre energy."""
		mat = subsample.channel.build_extract_matrix(_spec("front"), 6, "pcm")
		numpy.testing.assert_allclose(mat, [[0.5, 0.5, 0.5 * _SQRT2, 0.0, 0.0, 0.0]])

	def test_back_quad (self) -> None:
		"""Back on quad uses BL + BR."""
		mat = subsample.channel.build_extract_matrix(_spec("back"), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[0.0, 0.0, _SQRT2_INV, _SQRT2_INV]])

	def test_back_7_1 (self) -> None:
		"""Back on 7.1 uses BL + BR + SL + SR."""
		mat = subsample.channel.build_extract_matrix(_spec("back"), 8, "pcm")
		numpy.testing.assert_allclose(mat, [[0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]])

	# -- PCM rejections (no spatial info) --

	def test_depth_on_stereo_rejected (self) -> None:
		"""Depth on stereo: no F/B info — raises with 'stereo' in message."""
		with pytest.raises(ValueError, match="stereo"):
			subsample.channel.build_extract_matrix(_spec("depth"), 2, "pcm")

	def test_depth_on_mono_rejected (self) -> None:
		"""Depth on mono: raises."""
		with pytest.raises(ValueError, match="mono"):
			subsample.channel.build_extract_matrix(_spec("depth"), 1, "pcm")

	def test_height_rejected_on_all_pcm_layouts (self) -> None:
		"""Height requires Ambisonic; rejected on every PCM layout."""
		for n in (1, 2, 4, 6, 8):
			with pytest.raises(ValueError):
				subsample.channel.build_extract_matrix(_spec("height"), n, "pcm")

	def test_back_on_mono_rejected (self) -> None:
		"""Back on mono: no F/B info — rejected."""
		with pytest.raises(ValueError):
			subsample.channel.build_extract_matrix(_spec("back"), 1, "pcm")

	def test_back_on_stereo_rejected (self) -> None:
		"""Back on stereo: no F/B info — rejected."""
		with pytest.raises(ValueError):
			subsample.channel.build_extract_matrix(_spec("back"), 2, "pcm")

	def test_side_on_mono_rejected (self) -> None:
		"""Side on mono: no L/R info — rejected."""
		with pytest.raises(ValueError):
			subsample.channel.build_extract_matrix(_spec("side"), 1, "pcm")

	def test_unsupported_pcm_layout_rejected (self) -> None:
		"""3-channel PCM is not a standard layout — rejected."""
		with pytest.raises(ValueError, match="layout"):
			subsample.channel.build_extract_matrix(_spec("omni"), 3, "pcm")

	# -- Ambisonic B-format extracts --

	def test_ambisonic_omni_is_w_only (self) -> None:
		"""Ambisonic omni is W channel only (ACN index 0)."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[1.0, 0.0, 0.0, 0.0]])

	def test_ambisonic_omni_matches_decoder (self) -> None:
		"""Ambisonic omni matches ambisonic.decoder_matrix(1, 1, 'basic')."""
		import subsample.ambisonic
		mat     = subsample.channel.build_extract_matrix(_spec("omni"), 4, "b_format_ambix")
		decoder = subsample.ambisonic.decoder_matrix(1, 1, "basic")
		numpy.testing.assert_allclose(mat, decoder)

	def test_ambisonic_side_is_y (self) -> None:
		"""Ambisonic side is the Y channel (ACN index 1, L-R dipole)."""
		mat = subsample.channel.build_extract_matrix(_spec("side"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[0.0, 1.0, 0.0, 0.0]])

	def test_ambisonic_height_is_z (self) -> None:
		"""Ambisonic height is the Z channel (ACN index 2, U-D dipole)."""
		mat = subsample.channel.build_extract_matrix(_spec("height"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[0.0, 0.0, 1.0, 0.0]])

	def test_ambisonic_depth_is_x (self) -> None:
		"""Ambisonic depth is the X channel (ACN index 3, F-B dipole)."""
		mat = subsample.channel.build_extract_matrix(_spec("depth"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[0.0, 0.0, 0.0, 1.0]])

	def test_ambisonic_left_is_w_plus_y (self) -> None:
		"""Ambisonic left is (W+Y)/√2 cardioid."""
		mat = subsample.channel.build_extract_matrix(_spec("left"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, _SQRT2_INV, 0.0, 0.0]])

	def test_ambisonic_right_is_w_minus_y (self) -> None:
		"""Ambisonic right is (W-Y)/√2 cardioid."""
		mat = subsample.channel.build_extract_matrix(_spec("right"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, -_SQRT2_INV, 0.0, 0.0]])

	def test_ambisonic_front_is_w_plus_x (self) -> None:
		"""Ambisonic front is (W+X)/√2 cardioid."""
		mat = subsample.channel.build_extract_matrix(_spec("front"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, 0.0, 0.0, _SQRT2_INV]])

	def test_ambisonic_back_is_w_minus_x (self) -> None:
		"""Ambisonic back is (W-X)/√2 cardioid."""
		mat = subsample.channel.build_extract_matrix(_spec("back"), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[_SQRT2_INV, 0.0, 0.0, -_SQRT2_INV]])

	def test_ambisonic_wrong_channel_count_rejected (self) -> None:
		"""B-format requires 4 channels — 2-channel input rejected."""
		with pytest.raises(ValueError, match="4 input channels"):
			subsample.channel.build_extract_matrix(_spec("omni"), 2, "b_format_ambix")

	# -- channel.N literal index --

	def test_channel_index_first (self) -> None:
		"""channel.1 picks the first input channel."""
		mat = subsample.channel.build_extract_matrix(_spec("channel", 1), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[1.0, 0.0, 0.0, 0.0]])

	def test_channel_index_last (self) -> None:
		"""channel.4 picks the fourth input channel on a 4-channel source."""
		mat = subsample.channel.build_extract_matrix(_spec("channel", 4), 4, "pcm")
		numpy.testing.assert_allclose(mat, [[0.0, 0.0, 0.0, 1.0]])

	def test_channel_index_works_for_b_format (self) -> None:
		"""channel.N is format-agnostic: works on B-format too."""
		mat = subsample.channel.build_extract_matrix(_spec("channel", 2), 4, "b_format_ambix")
		numpy.testing.assert_allclose(mat, [[0.0, 1.0, 0.0, 0.0]])

	def test_channel_index_out_of_range_rejected (self) -> None:
		"""channel.5 on a 4-channel input raises ValueError."""
		with pytest.raises(ValueError, match="out of range"):
			subsample.channel.build_extract_matrix(_spec("channel", 5), 4, "pcm")

	def test_channel_index_missing_raises (self) -> None:
		"""kind='channel' with channel_index=None raises (programmer error)."""
		with pytest.raises(ValueError, match="channel_index"):
			subsample.channel.build_extract_matrix(_spec("channel"), 4, "pcm")

	# -- General errors --

	def test_unknown_format_rejected (self) -> None:
		"""Unknown channel_format raises with the bad value in the message."""
		with pytest.raises(ValueError, match="unknown_fmt"):
			subsample.channel.build_extract_matrix(_spec("omni"), 2, "unknown_fmt")

	def test_zero_channels_rejected (self) -> None:
		"""in_channels < 1 raises."""
		with pytest.raises(ValueError, match="must be >= 1"):
			subsample.channel.build_extract_matrix(_spec("omni"), 0, "pcm")

	# -- Output shape and dtype --

	def test_output_shape_is_one_by_in_channels (self) -> None:
		"""Returned matrix always has shape (1, in_channels)."""
		for n in (1, 2, 4, 6, 8):
			mat = subsample.channel.build_extract_matrix(_spec("omni"), n, "pcm")
			assert mat.shape == (1, n)

	def test_output_dtype_float32 (self) -> None:
		"""Returned matrix is float32 (matches build_mix_matrix)."""
		mat = subsample.channel.build_extract_matrix(_spec("omni"), 2, "pcm")
		assert mat.dtype == numpy.float32

	def test_omni_unit_l2_norm (self) -> None:
		"""Omni matrices are normalised to unit L2 length (constant-power)."""
		for n in (1, 2, 4, 6, 8):
			mat = subsample.channel.build_extract_matrix(_spec("omni"), n, "pcm")
			assert numpy.linalg.norm(mat) == pytest.approx(1.0, abs=1e-6)
