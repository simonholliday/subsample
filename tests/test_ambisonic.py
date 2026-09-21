"""Tests for subsample/ambisonic.py — first-order B-format (AmbiX) processing."""

import math

import numpy
import pytest

import subsample.ambisonic


# ---------------------------------------------------------------------------
# Helper: plane-wave B-format signal
# ---------------------------------------------------------------------------

def _plane_wave (azimuth_deg: float, elevation_deg: float, n_frames: int = 64) -> numpy.ndarray:

	"""Unit-magnitude first-order AmbiX signal for a plane wave from a direction.

	Returns shape (n_frames, 4) in AmbiX order (W, Y, Z, X).  The value is
	constant in time so energy analysis is straightforward.
	"""

	az = math.radians(azimuth_deg)
	el = math.radians(elevation_deg)
	x  = math.cos(el) * math.cos(az)
	y  = math.cos(el) * math.sin(az)
	z  = math.sin(el)

	frame = numpy.array([1.0, y, z, x], dtype=numpy.float32)
	return numpy.broadcast_to(frame, (n_frames, 4)).astype(numpy.float32)


# ---------------------------------------------------------------------------
# A → B matrix
# ---------------------------------------------------------------------------

class TestAToBMatrix:

	def test_shape (self) -> None:
		mat = subsample.ambisonic.a_to_b_matrix("generic_tetrahedral")
		assert mat.shape == (4, 4)
		assert mat.dtype == numpy.float32

	def test_w_row_is_equal_sum (self) -> None:
		"""W should be the average of the four capsule signals (unit pressure)."""
		mat = subsample.ambisonic.a_to_b_matrix("generic_tetrahedral")
		numpy.testing.assert_allclose(mat[subsample.ambisonic.ACN_W], [0.5, 0.5, 0.5, 0.5])

	# W carries the pressure scalar and the velocity rows carry √3 with it, so an
	# on-axis plane wave reads X = W as SN3D requires.  See the SN3D test below:
	# these two pin the signs, that one pins the level.
	_PRESSURE = 0.5
	_VELOCITY = 0.5 * math.sqrt(3.0)

	def test_capsule_impulse_flu (self) -> None:
		"""Injecting a unit impulse on the FLU capsule contributes positively to everything.

		FLU direction is (+x, +y, +z) / √3 — so it adds to all three velocity
		components as well as to pressure.
		"""
		mat = subsample.ambisonic.a_to_b_matrix("generic_tetrahedral")
		capsule_impulse = numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
		b = mat @ capsule_impulse
		numpy.testing.assert_allclose(
			b, [self._PRESSURE, self._VELOCITY, self._VELOCITY, self._VELOCITY], rtol=1e-6,
		)

	def test_capsule_impulse_brd_not_in_our_order (self) -> None:
		"""Sanity: the back-left-down (BLD) capsule, col 2, contributes +Y, -Z, -X, +W."""
		mat = subsample.ambisonic.a_to_b_matrix("generic_tetrahedral")
		capsule_impulse = numpy.array([0.0, 0.0, 1.0, 0.0], dtype=numpy.float32)
		b = mat @ capsule_impulse
		# BLD = (-x, +y, -z) / √3 → W +, Y +, Z -, X -
		numpy.testing.assert_allclose(
			b, [self._PRESSURE, self._VELOCITY, -self._VELOCITY, -self._VELOCITY], rtol=1e-6,
		)

	@pytest.mark.parametrize(("axis", "direction", "channel"), [
		("front", (1.0, 0.0, 0.0), 3),
		("left",  (0.0, 1.0, 0.0), 1),
		("up",    (0.0, 0.0, 1.0), 2),
	])
	def test_a_plane_wave_on_an_axis_reads_the_same_as_pressure (
		self, axis: str, direction: tuple[float, float, float], channel: int,
	) -> None:

		"""The SN3D guarantee, and the one nothing checked.

		A matched cardioid responds 0.5 + 0.5·cos, so a plane wave from the front
		reads 0.7887 on the two forward capsules and 0.2113 on the two rear ones.
		With one scalar for every row that gave W = 1 and X = 1/√3: **4.8 dB
		low**, on every A-format capture, storing a flattened and overly
		omnidirectional sound field that no decoder could restore."""

		matrix = subsample.ambisonic.a_to_b_matrix("generic_tetrahedral")
		capsules = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
		source = numpy.array(direction, dtype=numpy.float64)

		a_format = numpy.array([
			0.5 + 0.5 * (numpy.array(capsule, dtype=numpy.float64) / math.sqrt(3.0)).dot(source)
			for capsule in capsules
		], dtype=numpy.float32)

		b_format = matrix @ a_format

		assert b_format[0] == pytest.approx(1.0, abs=1e-5)
		assert b_format[channel] == pytest.approx(b_format[0], abs=1e-5), axis

	def test_nt_sf1_preset_matrix_same_as_generic (self) -> None:
		"""NT-SF1 preset uses the same matrix as generic; correction is via EQ."""
		generic = subsample.ambisonic.a_to_b_matrix("generic_tetrahedral")
		nt_sf1  = subsample.ambisonic.a_to_b_matrix("nt_sf1")
		numpy.testing.assert_allclose(generic, nt_sf1)

	def test_unknown_preset_raises (self) -> None:
		with pytest.raises(ValueError, match="Unknown mic preset"):
			subsample.ambisonic.a_to_b_matrix("not_a_mic")


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class TestRotationMatrix:

	def test_identity_rotation (self) -> None:
		"""0° yaw/pitch/roll returns a 4×4 identity."""
		R = subsample.ambisonic.rotation_matrix(1, 0.0, 0.0, 0.0)
		numpy.testing.assert_allclose(R, numpy.eye(4), atol=1e-6)

	def test_yaw_90_moves_front_to_left (self) -> None:
		"""A plane wave from +X, rotated +90° yaw, becomes a wave from +Y (left)."""
		R = subsample.ambisonic.rotation_matrix(1, 90.0, 0.0, 0.0)
		front_wave = _plane_wave(0.0, 0.0, n_frames=1)[0]       # (W=1, Y=0, Z=0, X=1)
		rotated    = R @ front_wave
		# Expected: (W=1, Y=1, Z=0, X=0)
		numpy.testing.assert_allclose(rotated, [1.0, 1.0, 0.0, 0.0], atol=1e-6)

	def test_yaw_360_is_identity (self) -> None:
		"""360° rotation round-trip is identity within float tolerance."""
		R = subsample.ambisonic.rotation_matrix(1, 360.0, 0.0, 0.0)
		numpy.testing.assert_allclose(R, numpy.eye(4), atol=1e-5)

	def test_w_channel_unchanged (self) -> None:
		"""W row and column form an identity block regardless of rotation."""
		R = subsample.ambisonic.rotation_matrix(1, 37.0, 22.0, -15.0)
		assert R[subsample.ambisonic.ACN_W, subsample.ambisonic.ACN_W] == pytest.approx(1.0)
		# W does not leak into X/Y/Z nor vice versa.
		for acn in (subsample.ambisonic.ACN_Y, subsample.ambisonic.ACN_Z, subsample.ambisonic.ACN_X):
			assert R[subsample.ambisonic.ACN_W, acn] == pytest.approx(0.0, abs=1e-6)
			assert R[acn, subsample.ambisonic.ACN_W] == pytest.approx(0.0, abs=1e-6)

	def test_pitch_90_moves_front_to_up (self) -> None:
		"""Pitch +90° tilts nose down — a +X wave becomes a -Z wave."""
		R = subsample.ambisonic.rotation_matrix(1, 0.0, 90.0, 0.0)
		front = _plane_wave(0.0, 0.0, n_frames=1)[0]
		rotated = R @ front
		# (W=1, Y=0, Z=-1, X=0): pitch +90 rotates +X → -Z.
		numpy.testing.assert_allclose(rotated, [1.0, 0.0, -1.0, 0.0], atol=1e-6)

	def test_unsupported_order_raises (self) -> None:
		with pytest.raises(ValueError, match="not supported"):
			subsample.ambisonic.rotation_matrix(2, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Decoder matrix
# ---------------------------------------------------------------------------

class TestDecoderMatrix:

	@pytest.mark.parametrize("out_ch", [1, 2, 4, 6, 8])
	def test_matrix_shape (self, out_ch: int) -> None:
		"""Decoder matrix has shape (out_ch, 4) for all supported layouts."""
		for decoder_type in ("basic", "max_re", "inphase"):
			D = subsample.ambisonic.decoder_matrix(1, out_ch, decoder_type)
			assert D.shape == (out_ch, 4)
			assert D.dtype == numpy.float32

	@staticmethod
	def _front_plane_wave () -> numpy.ndarray:
		"""A correctly normalised SN3D plane wave from straight ahead."""
		return numpy.array([1.0, 0.0, 0.0, 1.0], dtype=numpy.float32)

	def test_basic_is_a_velocity_decode_with_an_anti_phase_back_lobe (self) -> None:

		"""What "basic" means, and what it did not do.

		Mode matching needs the SN3D-to-N3D factor at both ends, so the velocity
		coefficient is 3 and the pattern is 1 + 3cos.  Without it the pattern was
		1 + cos — a cardioid, the shape "inphase" is supposed to have, with the
		rear speakers in phase and no null anywhere.  Every decoder here was one
		pattern too blunt."""

		gains = subsample.ambisonic.decoder_matrix(1, 4, "basic") @ self._front_plane_wave()

		assert gains[0] > 0.7
		assert gains[2] < -0.1, "a velocity decode puts the rear speakers out of phase"

	def test_inphase_never_puts_a_speaker_out_of_phase (self) -> None:

		"""That is the whole promise of the name, so it is worth a test.

		It is why the weight is 1/3 and not 1/2: with the order gain restored,
		1/2 would give 1 + 1.5cos, which is negative behind."""

		for out_channels in (2, 4, 6, 8):
			matrix = subsample.ambisonic.decoder_matrix(1, out_channels, "inphase")

			for azimuth in range(0, 360, 15):
				radians = math.radians(azimuth)
				wave = numpy.array(
					[1.0, math.sin(radians), 0.0, math.cos(radians)], dtype=numpy.float32,
				)

				assert numpy.all((matrix @ wave) >= -1e-6), f"{out_channels}ch at {azimuth}°"

	def test_each_mode_is_sharper_than_the_next (self) -> None:

		"""basic 1 + 3cos, max_re 1 + √3cos, inphase 1 + cos: three patterns, in order."""

		wave = self._front_plane_wave()
		fronts = [
			(subsample.ambisonic.decoder_matrix(1, 4, kind) @ wave)[0]
			for kind in ("basic", "max_re", "inphase")
		]

		assert fronts[0] > fronts[1] > fronts[2]

	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_the_speaker_gains_still_sum_to_unit_pressure (self, decoder_type: str) -> None:

		"""The velocity terms cancel in the sum whatever their weight, so the
		reconstructed pressure is unity in every mode.  This is what says the
		order gain sharpened the pattern rather than turning the output up."""

		gains = subsample.ambisonic.decoder_matrix(1, 4, decoder_type) @ self._front_plane_wave()

		assert float(gains.sum()) == pytest.approx(1.0, abs=1e-5)

	def test_w_only_input_equal_speaker_output (self) -> None:
		"""Unit W (with X=Y=Z=0) produces the same value on every active speaker."""
		D = subsample.ambisonic.decoder_matrix(1, 4, "basic")
		w_only = numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
		outputs = D @ w_only
		# Quad layout has 4 speakers, all active, equal gain 1/4.
		expected = numpy.full(4, 0.25, dtype=numpy.float32)
		numpy.testing.assert_allclose(outputs, expected)

	def test_front_plane_wave_maxes_front_speakers_basic (self) -> None:
		"""A plane wave from +X produces maximum output on the front-facing speakers.

		For quad (±45°, ±135°), front speakers (FL, FR) should exceed back
		speakers (BL, BR) in amplitude when the decoder is "basic".
		"""
		D = subsample.ambisonic.decoder_matrix(1, 4, "basic")
		front_wave = _plane_wave(0.0, 0.0, n_frames=1)[0]
		outputs = D @ front_wave
		fl, fr, bl, br = outputs
		assert fl > bl
		assert fr > br
		# Symmetry between L and R for a front-facing wave.
		assert fl == pytest.approx(fr)
		assert bl == pytest.approx(br)

	def test_basic_decoder_has_rear_null_for_cardioid_signal (self) -> None:
		"""For out=2 (±45° cardioids) and a wave coming from directly behind,
		the basic decoder shows rear attenuation (outputs are small relative
		to a front wave).
		"""
		D = subsample.ambisonic.decoder_matrix(1, 2, "basic")
		rear_wave  = _plane_wave(180.0, 0.0, n_frames=1)[0]   # from behind
		front_wave = _plane_wave(0.0,   0.0, n_frames=1)[0]   # from front
		rear_outputs  = numpy.abs(D @ rear_wave)
		front_outputs = numpy.abs(D @ front_wave)
		assert float(rear_outputs.sum()) < float(front_outputs.sum())

	def test_left_wave_loudest_on_left_speaker (self) -> None:
		"""For stereo (±45°), a wave from +Y (left) should be louder on the L output."""
		D = subsample.ambisonic.decoder_matrix(1, 2, "basic")
		left_wave = _plane_wave(90.0, 0.0, n_frames=1)[0]
		l_out, r_out = D @ left_wave
		# Basic decoder at ±45° for +Y source: L (at +45°) gets +sin(45°) component,
		# R (at -45°) gets -sin(45°) component.  The magnitudes differ in sign, so
		# L is positive, R is negative.
		assert l_out > r_out

	def test_max_re_has_narrower_lobes_than_basic (self) -> None:
		"""Max-rE trades front energy for a tighter lobe — the front/rear
		amplitude ratio is smaller than basic for an on-axis wave.
		"""
		basic  = subsample.ambisonic.decoder_matrix(1, 4, "basic")
		max_re = subsample.ambisonic.decoder_matrix(1, 4, "max_re")
		front_wave = _plane_wave(0.0, 0.0, n_frames=1)[0]
		basic_out  = basic  @ front_wave
		max_re_out = max_re @ front_wave
		# Front gain magnitude decreases with Max-rE weighting.
		assert abs(basic_out[0]) > abs(max_re_out[0])

	def test_inphase_has_no_back_lobes (self) -> None:
		"""In-phase decoder gives strictly non-negative output everywhere for
		a unit-W + unit-X plane wave (no anti-phase back lobes).
		"""
		D = subsample.ambisonic.decoder_matrix(1, 4, "inphase")
		front_wave = _plane_wave(0.0, 0.0, n_frames=1)[0]
		outputs = D @ front_wave
		assert (outputs >= -1e-6).all()

	def test_mono_decoder_is_w_only (self) -> None:
		D = subsample.ambisonic.decoder_matrix(1, 1, "basic")
		numpy.testing.assert_allclose(D, [[1.0, 0.0, 0.0, 0.0]])

	def test_lfe_slot_gets_w_only (self) -> None:
		"""In 5.1, row index 3 is LFE — it should decode W only."""
		D = subsample.ambisonic.decoder_matrix(1, 6, "basic")
		lfe_row = D[3]
		assert lfe_row[subsample.ambisonic.ACN_W] > 0.0
		# Velocity channels must be zero on the LFE row.
		for acn in (subsample.ambisonic.ACN_Y, subsample.ambisonic.ACN_Z, subsample.ambisonic.ACN_X):
			assert lfe_row[acn] == pytest.approx(0.0)

	def test_unsupported_out_channels_raises (self) -> None:
		with pytest.raises(ValueError, match="Unsupported decoder out_channels"):
			subsample.ambisonic.decoder_matrix(1, 3, "basic")

	def test_unsupported_decoder_type_raises (self) -> None:
		with pytest.raises(ValueError, match="Unknown decoder type"):
			subsample.ambisonic.decoder_matrix(1, 2, "fancy")


# ---------------------------------------------------------------------------
# combined_decode_matrix
# ---------------------------------------------------------------------------

class TestCombinedDecodeMatrix:

	def test_identity_rotation_returns_decoder_alone (self) -> None:
		"""Zero rotation should bypass the matmul and return plain decoder."""
		D_alone    = subsample.ambisonic.decoder_matrix(1, 4, "basic")
		D_combined = subsample.ambisonic.combined_decode_matrix(1, 4, "basic", 0.0, 0.0, 0.0)
		numpy.testing.assert_allclose(D_alone, D_combined)

	def test_yaw_rotates_decoder_speakers (self) -> None:
		"""A 90° yaw rotation should make a +X plane wave decode like it
		came from +Y when heard through the identity decoder.
		"""
		# For a 4-speaker layout, yaw 90° means the signal is rotated 90°
		# before decoding.  A front wave + 90° yaw should look to the
		# decoder like a wave from +Y (left).
		D_rotated = subsample.ambisonic.combined_decode_matrix(1, 4, "basic", 90.0, 0.0, 0.0)
		D_plain   = subsample.ambisonic.decoder_matrix(1, 4, "basic")

		front_wave = _plane_wave(0.0,  0.0, n_frames=1)[0]
		left_wave  = _plane_wave(90.0, 0.0, n_frames=1)[0]

		numpy.testing.assert_allclose(D_rotated @ front_wave, D_plain @ left_wave, atol=1e-6)


# ---------------------------------------------------------------------------
# FuMA → AmbiX
# ---------------------------------------------------------------------------

class TestFumaToAmbix:

	def test_w_gain_sqrt2 (self) -> None:
		"""FuMA W (MaxN) maps to AmbiX W with a √2 boost."""
		fuma = numpy.array([[1.0, 0.0, 0.0, 0.0]], dtype=numpy.float32)   # W=1, rest 0
		ambix = subsample.ambisonic.fuma_to_ambix(fuma)
		assert ambix[0, subsample.ambisonic.ACN_W] == pytest.approx(math.sqrt(2.0))
		assert ambix[0, subsample.ambisonic.ACN_Y] == pytest.approx(0.0)

	def test_channel_reorder (self) -> None:
		"""FuMA (W, X, Y, Z) → AmbiX (W, Y, Z, X)."""
		fuma = numpy.array([[0.0, 1.0, 2.0, 3.0]], dtype=numpy.float32)
		ambix = subsample.ambisonic.fuma_to_ambix(fuma)
		assert ambix[0, subsample.ambisonic.ACN_W] == pytest.approx(0.0)
		assert ambix[0, subsample.ambisonic.ACN_X] == pytest.approx(1.0)
		assert ambix[0, subsample.ambisonic.ACN_Y] == pytest.approx(2.0)
		assert ambix[0, subsample.ambisonic.ACN_Z] == pytest.approx(3.0)

	def test_shape_validation (self) -> None:
		with pytest.raises(ValueError, match="expects shape"):
			subsample.ambisonic.fuma_to_ambix(numpy.zeros((10, 3), dtype=numpy.float32))


# ---------------------------------------------------------------------------
# Biquad EQ
# ---------------------------------------------------------------------------

class TestBiquadAndShelves:

	def test_capsule_eq_none_for_generic (self) -> None:
		assert subsample.ambisonic.capsule_matching_eq("generic_tetrahedral", 48000) is None

	def test_capsule_eq_present_for_nt_sf1 (self) -> None:
		bq = subsample.ambisonic.capsule_matching_eq("nt_sf1", 48000)
		assert bq is not None
		assert bq.a[0] == pytest.approx(1.0)

	def test_hf_shelf_returns_biquad (self) -> None:
		bq = subsample.ambisonic.hf_shelf_correction(1, 48000)
		assert bq.a[0] == pytest.approx(1.0)
		# Shelf is a boost (at Nyquist the gain should be > 1).
		# Nyquist gain = (b0 + b1 + b2) / (a0 + a1 + a2) evaluated at z = -1:
		#   H(-1) = (b0 - b1 + b2) / (a0 - a1 + a2)
		nyq = (bq.b[0] - bq.b[1] + bq.b[2]) / (bq.a[0] - bq.a[1] + bq.a[2])
		assert abs(nyq) > 1.0

	def test_apply_biquad_preserves_shape (self) -> None:
		audio = numpy.random.RandomState(0).randn(1024, 4).astype(numpy.float32)
		bq    = subsample.ambisonic.hf_shelf_correction(1, 48000)
		out   = subsample.ambisonic.apply_biquad(audio, bq, channel_indices=(1, 2, 3))
		assert out.shape == audio.shape
		assert out.dtype == numpy.float32
		# Untouched channel (0) is bit-identical.
		numpy.testing.assert_array_equal(out[:, 0], audio[:, 0])
		# Filtered channels differ from the input.
		assert not numpy.array_equal(out[:, 1], audio[:, 1])


# ---------------------------------------------------------------------------
# Top-level capture pipeline
# ---------------------------------------------------------------------------

class TestProcessCapture:

	def test_b_ambix_is_passthrough (self) -> None:
		audio = numpy.random.RandomState(1).randn(256, 4).astype(numpy.float32)
		out = subsample.ambisonic.process_capture(audio, "b_ambix", sample_rate=48000)
		numpy.testing.assert_array_equal(out, audio)

	def test_b_fuma_converts_to_ambix (self) -> None:
		fuma = numpy.tile([1.0, 2.0, 3.0, 4.0], (8, 1)).astype(numpy.float32)
		ambix = subsample.ambisonic.process_capture(fuma, "b_fuma", sample_rate=48000)
		# W is sqrt(2)-scaled; X maps from FuMA col 1, Y from col 2, Z from col 3.
		assert ambix[0, subsample.ambisonic.ACN_W] == pytest.approx(math.sqrt(2.0))
		assert ambix[0, subsample.ambisonic.ACN_X] == pytest.approx(2.0)
		assert ambix[0, subsample.ambisonic.ACN_Y] == pytest.approx(3.0)
		assert ambix[0, subsample.ambisonic.ACN_Z] == pytest.approx(4.0)

	def test_a_generic_no_capsule_eq (self) -> None:
		"""Generic A-format has no capsule EQ, but the post-matrix HF shelf
		still acts on Y/Z/X — only W stays at the raw matrix output 0.5."""
		# Very short signal so filter warm-up cannot confound — but for
		# 'a_generic' there is no EQ.
		audio = numpy.zeros((1, 4), dtype=numpy.float32)
		audio[0, 0] = 1.0  # FLU capsule impulse
		out = subsample.ambisonic.process_capture(audio, "a_generic", sample_rate=48000)
		# A→B gives (0.5, 0.5, 0.5, 0.5); the post-matrix HF shelf then acts on
		# Y/Z/X — a single-sample impulse is shaped by the biquad's b[0] only,
		# so the first-sample output for filtered channels is 0.5 * bq.b[0].
		# Check W is unfiltered:
		assert out[0, subsample.ambisonic.ACN_W] == pytest.approx(0.5)

	def test_a_nt_sf1_produces_b_format_output (self) -> None:
		"""NT-SF1 capture path runs without errors and preserves shape."""
		audio = numpy.random.RandomState(2).randn(512, 4).astype(numpy.float32) * 0.1
		out = subsample.ambisonic.process_capture(audio, "a_nt_sf1", sample_rate=48000)
		assert out.shape == audio.shape
		assert out.dtype == numpy.float32

	def test_invalid_format_raises (self) -> None:
		audio = numpy.zeros((32, 4), dtype=numpy.float32)
		with pytest.raises(ValueError, match="Unknown ambisonic_format"):
			subsample.ambisonic.process_capture(audio, "quad", sample_rate=48000)

	def test_wrong_channel_count_raises (self) -> None:
		audio = numpy.zeros((32, 2), dtype=numpy.float32)
		with pytest.raises(ValueError, match="expects shape"):
			subsample.ambisonic.process_capture(audio, "b_ambix", sample_rate=48000)
