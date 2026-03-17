"""Tests for subsample.audio.unpack_audio."""

import numpy
import pytest

import subsample.audio
import subsample.recorder


class TestUnpackAudio16Bit:

	def test_matches_frombuffer (self) -> None:
		"""16-bit unpack should be equivalent to numpy.frombuffer."""
		samples = numpy.array([0, 1000, -1000, 32767, -32768], dtype=numpy.int16)
		raw = samples.tobytes()

		result = subsample.audio.unpack_audio(raw, bit_depth=16, channels=1)

		assert result.dtype == numpy.int16
		assert numpy.array_equal(result.flatten(), samples)

	def test_shape_mono (self) -> None:
		samples = numpy.zeros(10, dtype=numpy.int16)
		result = subsample.audio.unpack_audio(samples.tobytes(), bit_depth=16, channels=1)

		assert result.shape == (10, 1)

	def test_shape_stereo (self) -> None:
		samples = numpy.zeros(20, dtype=numpy.int16)
		result = subsample.audio.unpack_audio(samples.tobytes(), bit_depth=16, channels=2)

		assert result.shape == (10, 2)


class TestUnpackAudio24Bit:

	def test_dtype_is_int32 (self) -> None:
		"""24-bit unpacking should produce an int32 array."""
		# 3 bytes per sample; use 3 zero samples
		raw = bytes(9)

		result = subsample.audio.unpack_audio(raw, bit_depth=24, channels=1)

		assert result.dtype == numpy.int32

	def test_zero_samples (self) -> None:
		"""All-zero 24-bit bytes should produce all-zero int32 samples."""
		raw = bytes(3 * 4)  # 4 samples × 3 bytes

		result = subsample.audio.unpack_audio(raw, bit_depth=24, channels=1)

		assert result.shape == (4, 1)
		assert numpy.all(result == 0)

	def test_known_positive_value (self) -> None:
		"""A known 24-bit value should unpack to the expected left-shifted int32."""
		# 24-bit value: 0x010203 = 66051
		# Stored as little-endian bytes: 0x03, 0x02, 0x01
		# Left-shifted by 8: 0x00010203 << 8 = 0x01020300 = 16909056
		raw = bytes([0x03, 0x02, 0x01])

		result = subsample.audio.unpack_audio(raw, bit_depth=24, channels=1)

		assert result[0, 0] == 0x01020300

	def test_round_trip_with_pack_int24 (self) -> None:
		"""unpack_audio → _pack_int24 should recover the original 24-bit values."""
		# Build raw 24-bit bytes for a few known values (little-endian)
		values_24bit = [0, 100, -100, 8388607, -8388608]  # 24-bit range: -2^23 to 2^23-1

		raw_parts = []
		for v in values_24bit:
			# Pack v as signed 24-bit little-endian
			b = v.to_bytes(3, byteorder="little", signed=True)
			raw_parts.append(b)

		raw = b"".join(raw_parts)

		unpacked = subsample.audio.unpack_audio(raw, bit_depth=24, channels=1)
		repacked = subsample.recorder._pack_int24(unpacked)

		assert repacked == raw

	def test_shape_mono (self) -> None:
		raw = bytes(3 * 8)  # 8 samples × 3 bytes
		result = subsample.audio.unpack_audio(raw, bit_depth=24, channels=1)

		assert result.shape == (8, 1)

	def test_shape_stereo (self) -> None:
		raw = bytes(3 * 8)  # 4 stereo frames × 2 channels × 3 bytes
		result = subsample.audio.unpack_audio(raw, bit_depth=24, channels=2)

		assert result.shape == (4, 2)


class TestUnpackAudio32Bit:

	def test_matches_frombuffer (self) -> None:
		samples = numpy.array([0, 100000, -100000], dtype=numpy.int32)
		raw = samples.tobytes()

		result = subsample.audio.unpack_audio(raw, bit_depth=32, channels=1)

		assert result.dtype == numpy.int32
		assert numpy.array_equal(result.flatten(), samples)

	def test_shape_mono (self) -> None:
		samples = numpy.zeros(6, dtype=numpy.int32)
		result = subsample.audio.unpack_audio(samples.tobytes(), bit_depth=32, channels=1)

		assert result.shape == (6, 1)


class TestUnpackAudioErrors:

	def test_unsupported_bit_depth_raises (self) -> None:
		with pytest.raises(ValueError, match="Unsupported bit depth"):
			subsample.audio.unpack_audio(bytes(2), bit_depth=8, channels=1)
