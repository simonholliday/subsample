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


def _heard (out_channels: int, decoder_type: str, azimuth_deg: float) -> tuple[numpy.ndarray, numpy.ndarray]:

	"""Each speaker's gain for a plane wave from ``azimuth_deg`` at ear level, and each speaker's unit vector (x, y).

	LFE slots are left out of both: they have no direction.
	"""

	angles   = subsample.ambisonic._LAYOUT_ANGLES[out_channels]
	speakers = [index for index, angle in enumerate(angles) if angle is not None]
	towards  = numpy.array([
		[math.cos(math.radians(angle[0])), math.sin(math.radians(angle[0]))]
		for angle in angles if angle is not None
	])

	wave  = _plane_wave(azimuth_deg, 0.0, n_frames=1)[0]
	gains = (subsample.ambisonic.decoder_matrix(1, out_channels, decoder_type) @ wave)[speakers]

	return gains.astype(numpy.float64), towards


def _energy_vector (out_channels: int, decoder_type: str, azimuth_deg: float) -> tuple[float, float]:

	"""Gerzon's energy vector for a plane wave from ``azimuth_deg``: its length, and how far its direction is off, in degrees.

	The length is how sharply the sound is placed (1 is one speaker alone),
	and the direction is where its energy says it is.
	"""

	gains, towards = _heard(out_channels, decoder_type, azimuth_deg)
	vector = (gains ** 2) @ towards / float((gains ** 2).sum())
	off    = math.degrees(math.atan2(vector[1], vector[0])) - azimuth_deg

	return float(numpy.linalg.norm(vector)), abs((off + 180.0) % 360.0 - 180.0)


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

	def test_basic_is_the_sharpest_and_has_an_inverted_lobe_behind (self) -> None:

		"""What "basic" means: 1 + 2cos on the ring, which is negative behind.

		Without the order gain the pattern was 1 + cos, a cardioid, the shape
		"inphase" is supposed to have.  The lobe is small once the ring is
		panned onto four speakers, but it is there."""

		gains = subsample.ambisonic.decoder_matrix(1, 4, "basic") @ self._front_plane_wave()

		assert gains[0] > 0.5
		assert gains[2] < -0.01, "basic puts the rear speakers out of phase"

	def test_basic_does_not_overshoot_on_quad (self) -> None:

		"""The velocity a quad rig rebuilds is no more than the sound's own (#4705).

		With a full sphere's order gain, 3, on a ring of speakers, it was 1.5
		times it: the image pulled outward, with the rear pair at a third of
		the front's level, inverted."""

		gains, towards = _heard(4, "basic", 0.0)
		velocity = gains @ towards / float(gains.sum())

		assert 0.7 < float(velocity[0]) <= 1.0
		assert float(velocity[1]) == pytest.approx(0.0, abs=1e-6)

	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_a_sound_behind_a_stereo_pair_plays_in_phase (self, decoder_type: str) -> None:

		"""It fades as it goes behind, and is never inverted (#4705).

		Decoded straight to two speakers in front, a sound from behind came
		out of both inverted, at a third of a front sound's level with basic."""

		behind, _towards = _heard(2, decoder_type, 180.0)
		ahead, _towards  = _heard(2, decoder_type, 0.0)

		assert numpy.all(behind > 0.0)
		assert numpy.all(behind < ahead)

	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_a_sound_behind_5_1_plays_from_behind (self, decoder_type: str) -> None:

		"""The surrounds carry it, and the front three next to nothing (#4705).

		Each of the three front speakers took a full share of a plain decode,
		so with basic a sound from behind had more energy in front, inverted,
		than behind."""

		gains, _towards = _heard(6, decoder_type, 180.0)
		front, surrounds = gains[:3], gains[3:]

		assert float((front ** 2).sum()) < 0.05 * float((surrounds ** 2).sum())

	@pytest.mark.parametrize("out_channels", [6, 8])
	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_the_energy_points_where_the_sound_is (self, out_channels: int, decoder_type: str) -> None:

		"""On 5.1 and 7.1, a sound's energy is within 30° of it wherever it is (#4705).

		A plain decode suits evenly spaced speakers, and these are not: the
		energy was up to 63° off with max_re, and on the far side with basic."""

		for azimuth in range(0, 360, 5):
			_length, off = _energy_vector(out_channels, decoder_type, float(azimuth))

			assert off <= 30.0, f"{azimuth}° is heard {off:.0f}° off"

	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_a_sound_behind_7_1_keeps_its_direction (self, decoder_type: str) -> None:

		"""7.1's back pair is behind, at ±135° (#4705).

		It sat at 5.1's ±110°, 20° behind its own side pair, so a sound from
		behind was spread over a 140° gap: an energy vector of 0.3."""

		length, off = _energy_vector(8, decoder_type, 180.0)

		assert length > 0.5
		assert off < 1.0

	@pytest.mark.parametrize("out_channels", [2, 4, 6, 8])
	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_the_w_gains_sum_to_one (self, out_channels: int, decoder_type: str) -> None:

		"""A sound from everywhere at once sums to its own pressure, on every layout."""

		angles = subsample.ambisonic._LAYOUT_ANGLES[out_channels]
		matrix = subsample.ambisonic.decoder_matrix(1, out_channels, decoder_type)
		w_sum  = sum(float(matrix[index, subsample.ambisonic.ACN_W]) for index, angle in enumerate(angles) if angle is not None)

		assert w_sum == pytest.approx(1.0, abs=1e-5)

	@pytest.mark.parametrize("out_channels", [2, 4, 6, 8])
	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_left_and_right_mirror_each_other (self, out_channels: int, decoder_type: str) -> None:

		for azimuth in (0.0, 35.0, 90.0, 160.0):
			left, _towards  = _heard(out_channels, decoder_type, azimuth)
			right, _towards = _heard(out_channels, decoder_type, -azimuth)

			# Every layout lists its speakers in left-right pairs, the centre alone.
			mirrored = left.copy()
			pairs    = [(0, 1)] + [(index, index + 1) for index in range(3 if out_channels > 4 else 2, len(left), 2)]

			for one, other in pairs:
				mirrored[one], mirrored[other] = left[other], left[one]

			numpy.testing.assert_allclose(right, mirrored, atol=1e-6)

	def test_every_layout_is_at_ear_level (self) -> None:

		"""The decoder reads azimuths only, so a layout with height would be decoded flat without a word."""

		for angles in subsample.ambisonic._LAYOUT_ANGLES.values():
			assert all(angle[1] == 0.0 for angle in angles if angle is not None)

		for out_channels in (2, 4, 6, 8):
			matrix = subsample.ambisonic.decoder_matrix(1, out_channels, "basic")

			assert numpy.all(matrix[:, subsample.ambisonic.ACN_Z] == 0.0)

	def test_each_call_gets_a_matrix_of_its_own (self) -> None:

		"""The decode is worked out once and kept, so what a caller changes must not reach the next."""

		first = subsample.ambisonic.decoder_matrix(1, 4, "max_re")
		kept  = first.copy()
		first[:] = 99.0

		numpy.testing.assert_array_equal(subsample.ambisonic.decoder_matrix(1, 4, "max_re"), kept)

	def test_inphase_never_puts_a_speaker_out_of_phase (self) -> None:

		"""That is the whole promise of the name, so it is worth a test.

		It is why the ring's weight is 1/2: 1 + cos, a cardioid, is the
		sharpest pattern that is never negative, and panning each virtual
		speaker between the two real ones beside it adds no inversion."""

		for out_channels in (2, 4, 6, 8):
			matrix = subsample.ambisonic.decoder_matrix(1, out_channels, "inphase")

			for azimuth in range(0, 360, 15):
				radians = math.radians(azimuth)
				wave = numpy.array(
					[1.0, math.sin(radians), 0.0, math.cos(radians)], dtype=numpy.float32,
				)

				assert numpy.all((matrix @ wave) >= -1e-6), f"{out_channels}ch at {azimuth}°"

	def test_each_mode_is_sharper_than_the_next (self) -> None:

		"""basic 1 + 2cos, max_re 1 + √2cos, inphase 1 + cos: three patterns, in order."""

		wave = self._front_plane_wave()
		fronts = [
			(subsample.ambisonic.decoder_matrix(1, 4, kind) @ wave)[0]
			for kind in ("basic", "max_re", "inphase")
		]

		assert fronts[0] > fronts[1] > fronts[2]

	@pytest.mark.parametrize("decoder_type", ["basic", "max_re", "inphase"])
	def test_the_speaker_gains_still_sum_to_unit_pressure (self, decoder_type: str) -> None:

		"""The velocity terms cancel in the sum whatever their weight, so the
		reconstructed pressure is unity in every mode.  This is what says a
		weight sharpens the pattern rather than turning the output up."""

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
		# A sound at +Y is beyond the left speaker, so the right one plays
		# next to nothing: with basic, a little of it inverted.
		assert l_out > 0.9
		assert abs(r_out) < 0.1

	def test_max_re_gives_the_nearest_speakers_less_than_basic (self) -> None:
		"""Max-rE is the softer pattern of the two: for a wave from the front
		it gives the front pair less than basic does, and the rear pair no
		inverted share.
		"""
		basic  = subsample.ambisonic.decoder_matrix(1, 4, "basic")
		max_re = subsample.ambisonic.decoder_matrix(1, 4, "max_re")
		front_wave = _plane_wave(0.0, 0.0, n_frames=1)[0]
		basic_out  = basic  @ front_wave
		max_re_out = max_re @ front_wave
		# Front gain magnitude decreases with Max-rE weighting.
		assert abs(basic_out[0]) > abs(max_re_out[0])
		assert basic_out[2] < 0.0 < max_re_out[2]

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


class TestPanned:

	"""A virtual speaker is shared between the two real ones either side of it, at constant power."""

	def test_a_direction_at_a_speaker_is_that_speaker_alone (self) -> None:
		numpy.testing.assert_allclose(subsample.ambisonic._panned((45.0, -45.0, 135.0, -135.0), 135.0), [0.0, 0.0, 1.0, 0.0], atol=1e-12)

	def test_a_direction_between_two_is_shared_at_unit_power (self) -> None:
		gains = subsample.ambisonic._panned((30.0, -30.0, 0.0, 110.0, -110.0), 10.0)

		assert gains[0] > 0.0 and gains[2] > gains[0]
		assert gains[1] == gains[3] == gains[4] == 0.0
		assert float((gains ** 2).sum()) == pytest.approx(1.0)

	def test_a_direction_in_a_gap_of_half_a_circle_fades (self) -> None:

		"""Behind a stereo pair there is no speaker, so a sound there is dropped, and one nearly there is quiet."""

		pair = (45.0, -45.0)

		numpy.testing.assert_allclose(subsample.ambisonic._panned(pair, 180.0), [0.0, 0.0], atol=1e-12)

		beside = subsample.ambisonic._panned(pair, 100.0)
		behind = subsample.ambisonic._panned(pair, 170.0)

		assert beside[0] > behind[0] > 0.0
		assert beside[1] == behind[1] == 0.0


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
