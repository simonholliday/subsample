"""Tests for subsample.recorder._pack_int24."""

import numpy
import pytest

import subsample.audio
import subsample.recorder


class TestPackInt24:

	def test_zero_samples (self) -> None:
		"""All-zero int32 (left-shifted) should pack to all-zero bytes."""
		audio = numpy.zeros((4, 1), dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert result == bytes(12)  # 4 samples × 3 bytes

	def test_known_positive_value (self) -> None:
		"""Known left-shifted int32 should pack to expected 3-byte little-endian output."""
		# Internal value: 0x01020300 (24-bit value 0x010203 left-shifted by 8)
		# Expected packed bytes: 0x03, 0x02, 0x01 (little-endian 24-bit)
		audio = numpy.array([[0x01020300]], dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert result == bytes([0x03, 0x02, 0x01])

	def test_known_negative_value (self) -> None:
		"""Negative 24-bit value should round-trip correctly."""
		# 24-bit value: -1 = 0xFFFFFF, stored left-shifted as int32: -256 = 0xFFFFFF00
		audio = numpy.array([[-256]], dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert result == bytes([0xFF, 0xFF, 0xFF])

	def test_output_length (self) -> None:
		"""Output should be n_frames × channels × 3 bytes."""
		audio = numpy.zeros((10, 2), dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		assert len(result) == 10 * 2 * 3

	def test_round_trip_via_unpack (self) -> None:
		"""pack → unpack should recover the original 24-bit sample values."""
		# Create some 24-bit values packed into int32 (left-shifted by 8)
		original_24bit = numpy.array([0, 1000, -1000, 8388607, -8388608], dtype=numpy.int32)
		audio = (original_24bit * 256).reshape(-1, 1)  # simulate left-shift

		packed = subsample.recorder._pack_int24(audio)
		unpacked = subsample.audio.unpack_audio(packed, bit_depth=24, channels=1)

		# Unpacked values should match the original left-shifted representation
		assert numpy.array_equal(unpacked, audio)

	def test_stereo_interleaving (self) -> None:
		"""Stereo samples should be interleaved L/R/L/R in the output bytes."""
		# Frame 0: L=0x010203 << 8, R=0x040506 << 8
		left = 0x01020300
		right = 0x04050600
		audio = numpy.array([[left, right]], dtype=numpy.int32)

		result = subsample.recorder._pack_int24(audio)

		# tobytes() on shape (1, 2) flattens row-major: L bytes, then R bytes
		assert result[:3] == bytes([0x03, 0x02, 0x01])
		assert result[3:] == bytes([0x06, 0x05, 0x04])
