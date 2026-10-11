"""Ambisonic capture, rotation, and decoding — first-order B-format (AmbiX).

Subsample stores ambisonic samples in first-order AmbiX: channel order
follows ACN (W, Y, Z, X) with SN3D normalisation.  Higher orders are not
implemented; the API takes an ``order`` argument so callers that upgrade
later do not need to change shape.

This module is pure — no hidden state, no I/O, no dependencies on other
Subsample modules.  Everything returns numpy.float32 matrices or arrays.

Pipeline stages supplied here:

- A-format → B-format conversion for a 4-capsule tetrahedral mic
  (``a_to_b_matrix``).  "generic_tetrahedral" and "nt_sf1" presets.
- Per-capsule matching EQ (``capsule_matching_eq``) applied pre-matrix,
  and post-matrix HF shelf (``hf_shelf_correction``) on X/Y/Z to
  compensate for capsule-spacing high-frequency loss.
- FuMA → AmbiX conversion (``fuma_to_ambix``) for pre-encoded B-format
  files using the legacy channel order (W, X, Y, Z) and MaxN norm.
- Rotation matrices (``rotation_matrix``) for yaw / pitch / roll; W is
  left unchanged, the (X, Y, Z) block rotates as a 3-vector.
- Decoder matrices (``decoder_matrix``) mapping B-format to a standard
  output layout (mono, stereo, quad, 5.1, 7.1): the field is decoded to a
  regular virtual ring and panned onto the layout's speakers (AllRAD), with
  three weight modes: "basic" (sharpest, with an inverted lobe behind),
  "max_re" (the most energy from a sound's own direction), and "inphase"
  (softest, no inverted lobes).

Axis conventions — AmbiX / ACN / SN3D:
    +X = front, +Y = left, +Z = up.
    Azimuth measured counter-clockwise from +X (so +Y is at +90°).
    Elevation positive = up.
"""

import dataclasses
import functools
import math
import typing

import numpy
import scipy.signal


# ---------------------------------------------------------------------------
# Constants and ACN indices
# ---------------------------------------------------------------------------

AMBISONIC_ORDER_SUPPORTED: int = 1
"""Maximum ambisonic order currently implemented.  Callers must pass 1."""

_AMBI_CHANNELS_FIRST_ORDER: int = 4

# AmbiX / ACN channel indices for first order.
ACN_W: int = 0
ACN_Y: int = 1
ACN_Z: int = 2
ACN_X: int = 3


# ---------------------------------------------------------------------------
# Capsule and mic presets
# ---------------------------------------------------------------------------

SUPPORTED_AMBISONIC_FORMATS: frozenset[str] = frozenset({
	"a_generic",    # generic tetrahedral A-format, capsule order FLU/FRD/BLD/BRU
	"a_nt_sf1",     # Rode NT-SF1: generic matrix + capsule-matching EQ + HF shelf
	"b_fuma",       # pre-encoded B-format, FuMA order (W, X, Y, Z), MaxN norm
	"b_ambix",      # pre-encoded B-format, AmbiX order (W, Y, Z, X), SN3D norm
})


# ---------------------------------------------------------------------------
# Biquad filter record
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Biquad:

	"""Second-order IIR filter coefficients.

	Transfer function (scipy convention):

	              b[0] + b[1] z^-1 + b[2] z^-2
	    H(z)  =  ------------------------------
	              a[0] + a[1] z^-1 + a[2] z^-2

	a[0] is conventionally 1.0 after normalisation.
	"""

	b: tuple[float, float, float]
	a: tuple[float, float, float]


def _high_shelf_biquad (sample_rate: int, f0: float, gain_db: float, q: float = 0.707) -> Biquad:

	"""RBJ Audio EQ cookbook high-shelf.

	f0 is the shelf midpoint frequency; gain_db is the boost (positive) or
	cut (negative) at high frequencies relative to low.  Q controls the
	transition steepness (0.707 is a standard S=1 shelf).
	"""

	a_lin = 10.0 ** (gain_db / 40.0)
	w0    = 2.0 * math.pi * f0 / sample_rate
	cos_w = math.cos(w0)
	sin_w = math.sin(w0)
	alpha = sin_w / (2.0 * q)

	b0 = a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w + 2 * math.sqrt(a_lin) * alpha)
	b1 = -2 * a_lin * ((a_lin - 1) + (a_lin + 1) * cos_w)
	b2 = a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w - 2 * math.sqrt(a_lin) * alpha)
	a0 = (a_lin + 1) - (a_lin - 1) * cos_w + 2 * math.sqrt(a_lin) * alpha
	a1 = 2 * ((a_lin - 1) - (a_lin + 1) * cos_w)
	a2 = (a_lin + 1) - (a_lin - 1) * cos_w - 2 * math.sqrt(a_lin) * alpha

	return Biquad(
		b=(b0 / a0, b1 / a0, b2 / a0),
		a=(1.0,     a1 / a0, a2 / a0),
	)


def apply_biquad (audio: numpy.ndarray, bq: Biquad, channel_indices: tuple[int, ...]) -> numpy.ndarray:

	"""Filter the selected channels of a (n_frames, n_channels) array in place-equivalent.

	Returns a new array; other channels are copied through unmodified.
	"""

	out = audio.astype(numpy.float32, copy=True)

	for idx in channel_indices:
		out[:, idx] = scipy.signal.lfilter(
			numpy.asarray(bq.b, dtype=numpy.float64),
			numpy.asarray(bq.a, dtype=numpy.float64),
			out[:, idx].astype(numpy.float64),
		).astype(numpy.float32)

	return out


# ---------------------------------------------------------------------------
# A-format → B-format (AmbiX) matrices
# ---------------------------------------------------------------------------

# Generic Gerzon A→B matrix for a tetrahedral cardioid array with capsules
# aimed at the vertices (FLU, FRD, BLD, BRU).  Rows follow AmbiX order
# (W, Y, Z, X), columns follow capsule order (FLU, FRD, BLD, BRU).
#
# Direction cosines for each capsule (unit vectors, x=front, y=left, z=up):
#   FLU = (+1, +1, +1) / √3
#   FRD = (+1, −1, −1) / √3
#   BLD = (−1, +1, −1) / √3
#   BRU = (−1, −1, +1) / √3
# Signs in the matrix follow directly from summing each capsule against the
# target component (W is pressure: unit sum; X/Y/Z are velocity components
# along each axis).
#
# W and the velocity components need DIFFERENT scalars, and using one for both
# is what left this matrix wrong until 2026-09-21.  Work it through with the
# model above: a matched cardioid responds 0.5 + 0.5·cos, and for a plane wave
# from the front the four capsules read 0.7887, 0.7887, 0.2113, 0.2113.  A plain
# 0.5 scalar then gives W = 1.0 but X = 0.5·(1.5774 − 0.4226) = 1/√3 = 0.577,
# which is 4.8 dB below where SN3D puts it — every A-format capture was stored
# with a flattened, overly omnidirectional sound field, and no amount of decoder
# tuning downstream could put the energy back.
#
# So W keeps the pressure scalar and X/Y/Z carry √3 with it, which makes an
# on-axis plane wave read X = W = 1 as SN3D requires.  Real-mic calibration
# applies on top.

_A_TO_B_VELOCITY: float = float(numpy.sqrt(3.0))

_A_TO_B_GENERIC: numpy.ndarray = 0.5 * numpy.array([
	# FLU  FRD  BLD  BRU                           → AmbiX channel
	[+1,  +1,  +1,  +1],                          # W  (ACN 0, pressure)
	[+1,  -1,  +1,  -1],                          # Y  (ACN 1, left)
	[+1,  -1,  -1,  +1],                          # Z  (ACN 2, up)
	[+1,  +1,  -1,  -1],                          # X  (ACN 3, front)
], dtype=numpy.float32) * numpy.array(
	[[1.0], [_A_TO_B_VELOCITY], [_A_TO_B_VELOCITY], [_A_TO_B_VELOCITY]], dtype=numpy.float32,
)


def a_to_b_matrix (mic: str) -> numpy.ndarray:

	"""Return the (4, 4) A-format → AmbiX B-format matrix for the given mic preset.

	Supported presets:
	- "generic_tetrahedral": standard Gerzon matrix, capsule order
	  (FLU, FRD, BLD, BRU).
	- "nt_sf1": same matrix as generic (Rode's capsule order matches this
	  convention when the mic is wired per the datasheet).  The NT-SF1
	  preset differs from "generic_tetrahedral" only in that the recorder
	  also applies a ``capsule_matching_eq`` pre-matrix; the
	  ``hf_shelf_correction`` post-matrix is applied to both presets, and
	  the matrix itself is identical.

	Raises ValueError for any other preset.
	"""

	if mic in ("generic_tetrahedral", "nt_sf1"):
		return _A_TO_B_GENERIC.copy()

	raise ValueError(
		f"Unknown mic preset {mic!r}.  Supported: 'generic_tetrahedral', 'nt_sf1'."
	)


# ---------------------------------------------------------------------------
# Per-capsule matching and post-matrix HF shelf correction
# ---------------------------------------------------------------------------

def capsule_matching_eq (mic: str, sample_rate: int) -> typing.Optional[Biquad]:

	"""Return a pre-matrix per-capsule matching filter, or None if not applicable.

	Real tetrahedral mics have small capsule-to-capsule variations and an
	average HF roll-off due to capsule size.  A mild broadband HF shelf
	applied uniformly to all four capsules compensates the bulk of this.
	For a Tier 3 upgrade this would be swapped for per-capsule measured
	FIRs.

	For "generic_tetrahedral" no correction is applied (None).  For
	"nt_sf1" a gentle ~+2 dB shelf above 8 kHz is applied to restore
	on-axis HF content lost to the capsule.
	"""

	if mic == "nt_sf1":
		return _high_shelf_biquad(sample_rate, f0=8000.0, gain_db=2.0, q=0.707)

	if mic == "generic_tetrahedral":
		return None

	raise ValueError(
		f"Unknown mic preset {mic!r}.  Supported: 'generic_tetrahedral', 'nt_sf1'."
	)


def hf_shelf_correction (order: int, sample_rate: int) -> Biquad:

	"""Return the post-matrix HF shelf to apply to X/Y/Z only (W unchanged).

	A finite-aperture tetrahedral array low-passes the velocity components
	above c / (2 * d), where d is the capsule spacing (~25 mm for NT-SF1,
	giving a knee around 4-7 kHz depending on direction).  A high shelf of
	+4 dB from ~4 kHz approximates the needed boost for Tier 2 quality.
	Measured FIR correction is a Tier 3 future extension.

	Args:
		order:       Ambisonic order.  Currently must be 1.
		sample_rate: Hz.

	Returns:
		Biquad to apply to the three first-order channels (ACN indices 1=Y, 2=Z, 3=X).
		Never apply to W — W has no directional bias and does not suffer
		the same capsule-spacing attenuation.
	"""

	if order != AMBISONIC_ORDER_SUPPORTED:
		raise ValueError(f"Ambisonic order {order} not supported (only order 1).")

	return _high_shelf_biquad(sample_rate, f0=4000.0, gain_db=4.0, q=0.707)


# ---------------------------------------------------------------------------
# FuMA → AmbiX conversion
# ---------------------------------------------------------------------------

# FuMA channel order: W, X, Y, Z (legacy Soundfield convention).
# AmbiX channel order: W, Y, Z, X (ACN).
# Normalisation: FuMA uses MaxN (W scaled by 1/√2 relative to SN3D); AmbiX
# uses SN3D.  For first order, only W differs between the two.

_SQRT2: float = math.sqrt(2.0)


def fuma_to_ambix (audio: numpy.ndarray) -> numpy.ndarray:

	"""Convert a first-order FuMA B-format array to AmbiX.

	Input shape: (n_frames, 4) with FuMA channel order (W, X, Y, Z), MaxN.
	Output shape: (n_frames, 4) with AmbiX channel order (W, Y, Z, X), SN3D.
	"""

	if audio.ndim != 2 or audio.shape[1] != _AMBI_CHANNELS_FIRST_ORDER:
		raise ValueError(
			f"fuma_to_ambix expects shape (n, 4), got {audio.shape}"
		)

	out = numpy.empty_like(audio, dtype=numpy.float32)
	out[:, ACN_W] = audio[:, 0] * _SQRT2     # W: MaxN → SN3D gain
	out[:, ACN_X] = audio[:, 1]              # X
	out[:, ACN_Y] = audio[:, 2]              # Y
	out[:, ACN_Z] = audio[:, 3]              # Z

	return out


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotation_matrix (
	order:      int,
	yaw_deg:    float,
	pitch_deg:  float,
	roll_deg:   float,
) -> numpy.ndarray:

	"""Return the ambisonic rotation matrix for the given Tait-Bryan angles.

	Convention: intrinsic Z-Y-X (yaw then pitch then roll), right-handed.
	- yaw   rotates about +Z (up) — positive spins the sound field
	        counter-clockwise seen from above.
	- pitch rotates about +Y (left) — positive tilts the nose down.
	- roll  rotates about +X (front) — positive tilts the head right.

	W passes through unchanged; only the (X, Y, Z) block rotates.  The
	returned matrix is block-diagonal: [[1, 0], [0, R_3x3]] re-indexed into
	AmbiX order (W, Y, Z, X).

	For order > 1 this would need Wigner-D coefficients across higher-order
	channels; not supported here.
	"""

	if order != AMBISONIC_ORDER_SUPPORTED:
		raise ValueError(f"Ambisonic order {order} not supported (only order 1).")

	y, p, r = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
	cy, sy = math.cos(y), math.sin(y)
	cp, sp = math.cos(p), math.sin(p)
	cr, sr = math.cos(r), math.sin(r)

	# Cartesian rotation matrix R in axis order (x, y, z) applying roll, pitch,
	# yaw intrinsically (equivalent to R = R_yaw @ R_pitch @ R_roll).
	# Row/column order: [x, y, z].
	r_cart = numpy.array([
		[ cy * cp,   cy * sp * sr - sy * cr,   cy * sp * cr + sy * sr],
		[ sy * cp,   sy * sp * sr + cy * cr,   sy * sp * cr - cy * sr],
		[-sp,        cp * sr,                  cp * cr               ],
	], dtype=numpy.float32)

	# Reindex into AmbiX (W, Y, Z, X) ordering for the velocity components.
	# Cartesian row/col order is (x=0, y=1, z=2); AmbiX order for those
	# components is (Y=ACN1, Z=ACN2, X=ACN3).
	r_ambix = numpy.zeros((4, 4), dtype=numpy.float32)
	r_ambix[ACN_W, ACN_W] = 1.0

	# Map cart (x, y, z) index → ambix ACN index.
	_cart_to_acn = (ACN_X, ACN_Y, ACN_Z)

	for row_cart in range(3):
		for col_cart in range(3):
			r_ambix[_cart_to_acn[row_cart], _cart_to_acn[col_cart]] = r_cart[row_cart, col_cart]

	return r_ambix


# ---------------------------------------------------------------------------
# Decoder speaker layouts
# ---------------------------------------------------------------------------

# Each entry: tuple of per-output-channel (azimuth, elevation) in degrees,
# or None for LFE slots.  Ambisonic convention: 0° azimuth = front,
# positive azimuth = counter-clockwise (i.e. towards the +Y left axis).
# Stereo and quad are decoded for speakers wider than the physical SMPTE
# layout because ambisonic decoders sound better with more separation;
# 5.1/7.1 use the standard angles to match typical speaker setups: 7.1's
# back pair is at ±135° (ITU-R BS.2051), where until #4705 it sat at 5.1's
# ±110°, 20° behind its own side pair.
#
# Every layout is at ear level, elevation 0, and the decoder reads the
# azimuths only (see _ring_decoder): a layout with height would need the
# decode on a sphere.

_Angle    = tuple[float, float]
_LayoutAngles = tuple[typing.Optional[_Angle], ...]

_LAYOUT_ANGLES: dict[int, _LayoutAngles] = {
	1: ((0.0, 0.0),),
	2: ((45.0, 0.0), (-45.0, 0.0)),
	4: ((45.0, 0.0), (-45.0, 0.0), (135.0, 0.0), (-135.0, 0.0)),
	6: ((30.0, 0.0), (-30.0, 0.0), (0.0, 0.0), None, (110.0, 0.0), (-110.0, 0.0)),
	8: ((30.0, 0.0), (-30.0, 0.0), (0.0, 0.0), None, (135.0, 0.0), (-135.0, 0.0),
	    (90.0, 0.0), (-90.0, 0.0)),
}

SUPPORTED_DECODER_OUT_CHANNELS: frozenset[int] = frozenset(_LAYOUT_ANGLES.keys())

SUPPORTED_DECODER_TYPES: frozenset[str] = frozenset({"basic", "max_re", "inphase"})


# How many virtual speakers the decoder's ring holds: one a degree.  Three sample
# a first-order field exactly; this many make the panning onto the real speakers
# smooth, and every real speaker here sits on one of them.
_RING_SPEAKERS: int = 360


def _ring_weight (decoder_type: str) -> float:

	"""Return the weight g1 on a decoder's first-order terms, for speakers at ear level.

	Each speaker of the virtual ring plays ``1 + (2 · g1) · cos`` of the angle
	between it and the sound.  The 2 is a ring's order gain: what mode matching
	needs on an SN3D signal's X and Y when every speaker is at ear level.  A
	full sphere's is 3, which every layout here was given until #4705: basic's
	velocity then overshot by half on quad, and a sound behind a stereo pair or
	a 5.1 front came out of it inverted.

	- basic:   g1 = 1        → 1 + 2cos.   The sharpest, with an inverted lobe
	                           behind.
	- max_re:  g1 = cos 45°  → 1 + √2cos.  The most of a sound's energy from
	                           its own direction, the standard first-order
	                           Max-rE weight in the plane.
	- inphase: g1 = 1/2      → 1 + cos.    A cardioid, the sharpest pattern
	                           that never goes negative, which is the promise
	                           "in-phase" makes.
	"""

	if decoder_type == "basic":
		return 1.0
	if decoder_type == "max_re":
		return math.cos(math.pi / 4.0)
	if decoder_type == "inphase":
		return 0.5

	raise ValueError(
		f"Unknown decoder type {decoder_type!r}.  "
		f"Supported: {', '.join(sorted(SUPPORTED_DECODER_TYPES))}."
	)


def _panned (azimuths_deg: tuple[float, ...], target_deg: float) -> numpy.ndarray:

	"""Return each speaker's gain for a sound placed at ``target_deg``, at constant power.

	The sound is shared between the two speakers either side of it (pairwise
	amplitude panning: VBAP in the plane), so it never plays inverted and
	never from a speaker across the room.  Where the speakers leave half a
	circle or more empty, as a stereo pair does behind, a speaker that is not
	there stands in the middle of the gap and its share is dropped: a sound
	fades as it moves behind, rather than jumping to the far speaker.
	"""

	placed = sorted(
		((azimuth % 360.0, index) for index, azimuth in enumerate(azimuths_deg)),
		key=lambda speaker: speaker[0],
	)

	# The speakers in order round the circle; None is one that is not there.
	ring: list[tuple[float, typing.Optional[int]]] = []

	for position, (azimuth, index) in enumerate(placed):
		following = placed[(position + 1) % len(placed)][0]
		gap = (following - azimuth) % 360.0 or 360.0

		ring.append((azimuth, index))

		if gap >= 180.0:
			ring.append(((azimuth + gap / 2.0) % 360.0, None))

	gains = numpy.zeros(len(azimuths_deg))

	for position, (start, speaker) in enumerate(ring):
		end, next_speaker = ring[(position + 1) % len(ring)]
		span   = (end - start) % 360.0
		offset = (target_deg - start) % 360.0

		if offset > span:
			continue

		# The pair's gains for a source between them are in the ratio of
		# these sines; scaled to unit power, their common divisor drops out.
		near = math.sin(math.radians(span - offset))
		far  = math.sin(math.radians(offset))
		power = math.hypot(near, far)

		if speaker is not None:
			gains[speaker] = near / power

		if next_speaker is not None:
			gains[next_speaker] = far / power

		break

	return gains


@functools.lru_cache(maxsize=None)
def _ring_decoder (azimuths_deg: tuple[float, ...], decoder_type: str) -> numpy.ndarray:

	"""Return the (speakers, 4) decoder for speakers at ear level at these azimuths.

	AllRAD in the plane (Zotter and Frank, 2012): the field is decoded to a
	regular ring of many virtual speakers, where a first-order decode is exact
	and each weight means what it says, and each virtual speaker is then
	panned onto the real ones.  A plain decode straight to the real speakers
	is right only where they are evenly spaced, and of these layouts only quad
	is: on 5.1 the three front speakers each took a full share, so a sound
	behind was louder in front than behind, and inverted (#4705, M32 of the
	2026-10-07 review; Simon's call was this method for every layout).

	Scaled so the W gains sum to one, as they always have: a sound from
	everywhere at once sums to its own pressure at the listener.  Kept after
	the first call, since the player asks again for every pan position, and
	read-only for that reason.
	"""

	order_gain = 2.0 * _ring_weight(decoder_type)
	matrix = numpy.zeros((len(azimuths_deg), _AMBI_CHANNELS_FIRST_ORDER))

	for step in range(_RING_SPEAKERS):
		azimuth_deg = 360.0 * step / _RING_SPEAKERS
		azimuth = math.radians(azimuth_deg)

		plays = numpy.zeros(_AMBI_CHANNELS_FIRST_ORDER)
		plays[ACN_W] = 1.0
		plays[ACN_Y] = order_gain * math.sin(azimuth)
		plays[ACN_X] = order_gain * math.cos(azimuth)

		matrix += numpy.outer(_panned(azimuths_deg, azimuth_deg), plays)

	matrix /= matrix[:, ACN_W].sum()
	matrix.setflags(write=False)

	return matrix


def decoder_matrix (
	order:         int,
	out_channels:  int,
	decoder_type:  str,
) -> numpy.ndarray:

	"""Return the (out_channels, 4) decoder matrix for the given output layout.

	The matrix multiplies an (n_frames, 4) B-format AmbiX signal to produce
	(n_frames, out_channels) output in SMPTE channel order.  LFE slots
	(position 3 in 5.1/7.1) receive a weighted copy of W to give a usable
	low-frequency channel; a downstream LFE filter in the audio chain is
	recommended but not applied here.

	Every layout is at ear level, so the decode is in the plane (see
	_ring_decoder) and Z plays from no speaker: height reaches the output only
	through a pitch or roll rotation applied first.

	Args:
		order:        Ambisonic order.  Currently must be 1.
		out_channels: 1, 2, 4, 6, or 8.  Any other value raises ValueError.
		decoder_type: "basic", "max_re", or "inphase".

	Returns:
		float32 array of shape (out_channels, 4) with AmbiX column order
		(W, Y, Z, X).
	"""

	if order != AMBISONIC_ORDER_SUPPORTED:
		raise ValueError(f"Ambisonic order {order} not supported (only order 1).")

	if out_channels not in _LAYOUT_ANGLES:
		raise ValueError(
			f"Unsupported decoder out_channels {out_channels}.  "
			f"Supported: {sorted(_LAYOUT_ANGLES.keys())}."
		)

	# Mono: decode to W only (omnidirectional sum) at unity gain.  This is
	# the standard ambisonic mono downmix: directional components cancel on
	# average over the sphere, so W alone is the cleanest neutral mono mix.
	if out_channels == 1:
		matrix = numpy.zeros((1, _AMBI_CHANNELS_FIRST_ORDER), dtype=numpy.float32)
		matrix[0, ACN_W] = 1.0
		return matrix

	angles   = _LAYOUT_ANGLES[out_channels]
	speakers = [index for index, angle in enumerate(angles) if angle is not None]
	azimuths = tuple(angle[0] for angle in angles if angle is not None)

	matrix = numpy.zeros((out_channels, _AMBI_CHANNELS_FIRST_ORDER), dtype=numpy.float32)
	matrix[speakers] = _ring_decoder(azimuths, decoder_type)

	# An LFE slot takes W only, at the speakers' mean W gain, for bass
	# continuity with them.  It is on top of their sum of one, so 5.1's W
	# gains sum to 1.2, and that extra is intentional.
	for index, angle in enumerate(angles):
		if angle is None:
			matrix[index, ACN_W] = 1.0 / len(speakers)

	return matrix


# ---------------------------------------------------------------------------
# Convenience: combined decode matrix (rotation ∘ decoder)
# ---------------------------------------------------------------------------

def combined_decode_matrix (
	order:         int,
	out_channels:  int,
	decoder_type:  str,
	yaw_deg:       float = 0.0,
	pitch_deg:     float = 0.0,
	roll_deg:      float = 0.0,
) -> numpy.ndarray:

	"""Return decoder_matrix @ rotation_matrix, ready to apply to B-format audio.

	The result has shape (out_channels, 4).  Apply as
	``audio_out = bformat_audio @ M.T`` where ``bformat_audio`` is
	(n_frames, 4).  Identity rotation is handled by returning a plain
	decoder matrix without matmul overhead.
	"""

	decoder = decoder_matrix(order, out_channels, decoder_type)

	if yaw_deg == 0.0 and pitch_deg == 0.0 and roll_deg == 0.0:
		return decoder

	rotation = rotation_matrix(order, yaw_deg, pitch_deg, roll_deg)

	combined: numpy.ndarray = (decoder @ rotation).astype(numpy.float32)
	return combined


# ---------------------------------------------------------------------------
# Top-level capture pipeline
# ---------------------------------------------------------------------------

def process_capture (
	audio:              numpy.ndarray,
	ambisonic_format:   str,
	sample_rate:        int,
) -> numpy.ndarray:

	"""Convert a captured 4-channel array to canonical AmbiX B-format.

	Dispatches on ``ambisonic_format``:

	- "a_generic":  A-to-B matrix → post-matrix HF shelf on X/Y/Z.
	- "a_nt_sf1":   capsule matching EQ → A-to-B matrix →
	                post-matrix HF shelf on X/Y/Z.
	- "b_fuma":     reorder + renormalise from FuMA to AmbiX.
	- "b_ambix":    pass-through (ensure float32).

	Args:
		audio:            shape (n_frames, 4), float32 already normalised
		                  to [-1.0, 1.0].  Integer PCM should be normalised
		                  by the caller before invoking this function.
		ambisonic_format: one of SUPPORTED_AMBISONIC_FORMATS.
		sample_rate:      Hz (required for EQ filter design).

	Returns:
		shape (n_frames, 4), float32, AmbiX channel order (W, Y, Z, X).
	"""

	if ambisonic_format not in SUPPORTED_AMBISONIC_FORMATS:
		raise ValueError(
			f"Unknown ambisonic_format {ambisonic_format!r}.  "
			f"Supported: {sorted(SUPPORTED_AMBISONIC_FORMATS)}."
		)

	if audio.ndim != 2 or audio.shape[1] != _AMBI_CHANNELS_FIRST_ORDER:
		raise ValueError(
			f"process_capture expects shape (n, 4), got {audio.shape}"
		)

	work = audio.astype(numpy.float32, copy=False)

	if ambisonic_format == "b_ambix":
		return work.astype(numpy.float32, copy=True)

	if ambisonic_format == "b_fuma":
		return fuma_to_ambix(work)

	# A-format path.
	mic = "nt_sf1" if ambisonic_format == "a_nt_sf1" else "generic_tetrahedral"

	capsule_eq = capsule_matching_eq(mic, sample_rate)

	if capsule_eq is not None:
		# Apply the matching filter to every capsule (all four channels).
		work = apply_biquad(work, capsule_eq, channel_indices=(0, 1, 2, 3))

	b_format = (work @ a_to_b_matrix(mic).T).astype(numpy.float32)

	hf_shelf = hf_shelf_correction(AMBISONIC_ORDER_SUPPORTED, sample_rate)
	b_format = apply_biquad(b_format, hf_shelf, channel_indices=(ACN_Y, ACN_Z, ACN_X))

	return b_format
