"""Multichannel audio routing — SMPTE layouts, ITU downmix, and mixing matrices.

Defines standard channel layouts (mono through 7.1) using SMPTE ordering,
pre-computed ITU-R BS.775 downmix coefficient matrices, and the core
build_mix_matrix() function that produces an (out_ch × in_ch) mixing matrix
for any input/output channel combination.

The mixing matrix is applied as a single numpy matmul in the render path:

    output = gained_audio @ mix_matrix.T

This handles all cases uniformly — stereo passthrough, ITU surround downmix,
conservative upmix, and user-defined pan weights — with no special-casing
by channel count.
"""

import enum
import typing

import numpy


# ---------------------------------------------------------------------------
# SMPTE channel positions (ascending bit order from WAVEFORMATEXTENSIBLE)
# ---------------------------------------------------------------------------

class Channel (enum.IntEnum):

	"""Speaker positions in SMPTE bit order."""

	FL  = 0   # Front Left
	FR  = 1   # Front Right
	FC  = 2   # Front Centre
	LFE = 3   # Low Frequency Effects
	BL  = 4   # Back Left (Rear Left)
	BR  = 5   # Back Right (Rear Right)
	SL  = 6   # Side Left
	SR  = 7   # Side Right


# ---------------------------------------------------------------------------
# Standard layouts — channel count → tuple of Channel members present
# ---------------------------------------------------------------------------

STANDARD_LAYOUTS: dict[int, tuple[Channel, ...]] = {
	1: (Channel.FC,),
	2: (Channel.FL, Channel.FR),
	4: (Channel.FL, Channel.FR, Channel.BL, Channel.BR),
	6: (Channel.FL, Channel.FR, Channel.FC, Channel.LFE, Channel.BL, Channel.BR),
	8: (Channel.FL, Channel.FR, Channel.FC, Channel.LFE, Channel.BL, Channel.BR, Channel.SL, Channel.SR),
}

# Human-readable layout names for logging.
LAYOUT_NAMES: dict[int, str] = {
	1: "mono",
	2: "stereo",
	4: "quad",
	6: "5.1",
	8: "7.1",
}


# ---------------------------------------------------------------------------
# ITU-R BS.775 downmix coefficient matrices
# ---------------------------------------------------------------------------
# Each matrix is (out_ch, in_ch).  Columns follow SMPTE order for the input
# layout; rows follow SMPTE order for the output layout.
#
# Standard coefficients: centre and surrounds at -3 dB (1/sqrt(2) ≈ 0.707).
# LFE is discarded in all downmixes (ITU recommendation).

_SQRT2_INV = float(numpy.sqrt(0.5))  # 0.7071068...

_ITU_DOWNMIX: dict[tuple[int, int], numpy.ndarray] = {

	# 5.1 (FL FR FC LFE BL BR) → stereo (FL FR)
	(6, 2): numpy.array([
		[1.0, 0.0, _SQRT2_INV, 0.0, _SQRT2_INV, 0.0],
		[0.0, 1.0, _SQRT2_INV, 0.0, 0.0, _SQRT2_INV],
	], dtype=numpy.float32),

	# quad (FL FR BL BR) → stereo (FL FR)
	(4, 2): numpy.array([
		[1.0, 0.0, _SQRT2_INV, 0.0],
		[0.0, 1.0, 0.0, _SQRT2_INV],
	], dtype=numpy.float32),

	# 7.1 (FL FR FC LFE BL BR SL SR) → stereo (FL FR)
	(8, 2): numpy.array([
		[1.0, 0.0, _SQRT2_INV, 0.0, _SQRT2_INV, 0.0, _SQRT2_INV, 0.0],
		[0.0, 1.0, _SQRT2_INV, 0.0, 0.0, _SQRT2_INV, 0.0, _SQRT2_INV],
	], dtype=numpy.float32),

	# 7.1 → 5.1: fold sides into backs
	(8, 6): numpy.array([
		[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # FL
		[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # FR
		[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # FC
		[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # LFE
		[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, _SQRT2_INV, 0.0],  # BL + SL
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, _SQRT2_INV],  # BR + SR
	], dtype=numpy.float32),

	# 5.1 → quad: fold centre into L/R, discard LFE
	(6, 4): numpy.array([
		[1.0, 0.0, _SQRT2_INV, 0.0, 0.0, 0.0],  # FL
		[0.0, 1.0, _SQRT2_INV, 0.0, 0.0, 0.0],  # FR
		[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],          # BL
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],          # BR
	], dtype=numpy.float32),

	# stereo → mono: equal sum, -3 dB each
	(2, 1): numpy.array([
		[_SQRT2_INV, _SQRT2_INV],
	], dtype=numpy.float32),
}


def _get_downmix (in_ch: int, out_ch: int) -> typing.Optional[numpy.ndarray]:

	"""Look up or chain ITU downmix matrices.

	Returns an (out_ch, in_ch) matrix, or None if no path exists.
	Tries direct lookup first, then one-step chaining through intermediates.
	"""

	direct = _ITU_DOWNMIX.get((in_ch, out_ch))

	if direct is not None:
		return direct

	# Try chaining through a single intermediate.
	for mid_ch in sorted(STANDARD_LAYOUTS.keys(), reverse=True):
		if mid_ch <= out_ch or mid_ch >= in_ch:
			continue

		a = _ITU_DOWNMIX.get((in_ch, mid_ch))
		b = _ITU_DOWNMIX.get((mid_ch, out_ch))

		if a is not None and b is not None:
			return numpy.asarray((b @ a), dtype=numpy.float32)

	return None


def _build_default_matrix (in_ch: int, out_ch: int) -> numpy.ndarray:

	"""Build a default mix matrix with no user pan weights.

	- in == out: identity (each channel maps 1:1)
	- in > out: ITU downmix if available, else truncate to first out_ch channels
	- in < out: map input to first in_ch output channels, extras silent
	"""

	if in_ch == out_ch:
		return numpy.eye(out_ch, dtype=numpy.float32)

	if in_ch > out_ch:
		# Downmix.
		mat = _get_downmix(in_ch, out_ch)

		if mat is not None:
			return mat

		# Fallback: take the first out_ch input channels.
		return numpy.eye(out_ch, in_ch, dtype=numpy.float32)

	# Upmix (in < out): map input to corresponding front positions, rest silent.
	return numpy.eye(out_ch, in_ch, dtype=numpy.float32)


def normalize_mix_matrix (matrix: numpy.ndarray) -> numpy.ndarray:

	"""Constant-power normalise each output row.

	Scales each row so that the sum of squared coefficients equals 1.0.
	Rows that are all-zero are left unchanged.
	"""

	row_power = numpy.sum(matrix ** 2, axis=1, keepdims=True)
	row_power = numpy.where(row_power > 0.0, row_power, 1.0)

	return (matrix / numpy.sqrt(row_power)).astype(numpy.float32)


def build_mix_matrix (
	in_channels:  int,
	out_channels: int,
	pan_weights:  typing.Optional[numpy.ndarray] = None,
) -> numpy.ndarray:

	"""Build an (out_channels, in_channels) mixing matrix.

	When pan_weights is None, returns the default routing (identity, ITU
	downmix, or conservative upmix depending on in/out counts).

	When pan_weights is provided, its length defines a "target layout".
	The matrix routes input channels to the target layout, then (if needed)
	folds to the actual output channel count via ITU downmix.

	Args:
		in_channels:  Number of channels in the source audio.
		out_channels: Number of channels on the output device.
		pan_weights:  Optional float32 array of non-negative weights, one per
		              channel in the target layout. Length must be a standard
		              layout size (1, 2, 4, 6, 8).

	Returns:
		float32 array, shape (out_channels, in_channels).
	"""

	if pan_weights is None:
		return _build_default_matrix(in_channels, out_channels)

	target_ch = len(pan_weights)

	if target_ch not in STANDARD_LAYOUTS:
		raise ValueError(
			f"pan_weights length {target_ch} does not match a standard layout "
			f"({', '.join(str(k) for k in sorted(STANDARD_LAYOUTS))})"
		)
	total = float(numpy.sum(pan_weights))

	if total == 0.0:
		return numpy.zeros((out_channels, in_channels), dtype=numpy.float32)

	# Constant-power normalised gains from the user weights.
	gains = numpy.sqrt(pan_weights / total).astype(numpy.float32)

	if in_channels == 1:
		# Mono input: pan weights directly define how the single channel
		# distributes across target outputs.  Each row is the gain for
		# that output channel — no base routing needed.
		weighted = gains.reshape(-1, 1).astype(numpy.float32)
	else:
		# Multi-channel input: build the default routing from input to
		# target layout, then modulate each target output row by its gain.
		base = _build_default_matrix(in_channels, target_ch)
		weighted = numpy.diag(gains) @ base

	if target_ch == out_channels:
		return weighted.astype(numpy.float32)

	if target_ch > out_channels:
		# Target is larger than output — fold down to actual output.
		fold = _build_default_matrix(target_ch, out_channels)
		return (fold @ weighted).astype(numpy.float32)

	# Target is smaller than output — expand into front positions.
	result = numpy.zeros((out_channels, in_channels), dtype=numpy.float32)
	result[:target_ch, :] = weighted

	return result


# ---------------------------------------------------------------------------
# Physical output routing
# ---------------------------------------------------------------------------

def route_to_device (
	logical_matrix: numpy.ndarray,
	device_channels: int,
	output_map: typing.Optional[tuple[int, ...]] = None,
) -> numpy.ndarray:

	"""Embed a logical mixing matrix into a device-sized matrix.

	Parameters
	----------
	logical_matrix:  (logical_out, in_ch) mixing matrix from build_mix_matrix().
	device_channels: Total number of output channels on the audio device.
	output_map:      0-indexed device channel indices for each logical output.
	                 Length must equal logical_matrix.shape[0].
	                 None = default routing (first N device channels).

	Returns
	-------
	float32 array of shape (device_channels, in_ch).  Rows not targeted by
	output_map are zero (silence on those physical outputs).
	"""

	logical_out, in_ch = logical_matrix.shape

	if output_map is None:
		if logical_out == device_channels:
			return logical_matrix.astype(numpy.float32)
		if logical_out > device_channels:
			raise ValueError(
				f"Logical output count ({logical_out}) exceeds device channels "
				f"({device_channels}) — cannot route without an explicit output map"
			)
		result = numpy.zeros((device_channels, in_ch), dtype=numpy.float32)
		result[:logical_out, :] = logical_matrix
		return result

	if len(output_map) != logical_out:
		raise ValueError(
			f"output_map length ({len(output_map)}) does not match logical "
			f"output count ({logical_out})"
		)

	if len(set(output_map)) != len(output_map):
		raise ValueError(f"output_map contains duplicate indices: {output_map}")

	for idx in output_map:
		if idx < 0 or idx >= device_channels:
			raise ValueError(
				f"output_map index {idx} is out of range for "
				f"{device_channels}-channel device (valid: 0..{device_channels - 1})"
			)

	result = numpy.zeros((device_channels, in_ch), dtype=numpy.float32)

	for logical_row, device_row in enumerate(output_map):
		result[device_row, :] = logical_matrix[logical_row, :]

	return result
