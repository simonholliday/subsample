"""Tests for subsample.buffer.CircularBuffer."""

import numpy
import pytest

import subsample.buffer


def _make_chunk (value: int, n_frames: int = 10) -> numpy.ndarray:
	"""Return a mono chunk filled with a constant value."""
	return numpy.full(n_frames, value, dtype=numpy.int16)


class TestCircularBufferBasics:

	def test_initial_state (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=100, channels=1)

		assert buf.frames_written == 0
		assert buf.write_head == 0
		assert not buf.is_full

	def test_write_advances_frame_counter (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=100, channels=1)
		chunk = _make_chunk(1, n_frames=10)

		buf.write(chunk)

		assert buf.frames_written == 10
		assert buf.write_head == 10

	def test_is_full_after_capacity_reached (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=20, channels=1)

		buf.write(_make_chunk(1, 20))

		assert buf.is_full

	def test_is_full_not_set_before_capacity (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=20, channels=1)

		buf.write(_make_chunk(1, 19))

		assert not buf.is_full


class TestCircularBufferRead:

	def test_simple_read (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=100, channels=1)
		chunk = _make_chunk(42, n_frames=10)

		buf.write(chunk)
		result = buf.read_range(0, 10)

		assert result.shape == (10, 1)
		assert numpy.all(result == 42)

	def test_read_returns_copy (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=100, channels=1)
		buf.write(_make_chunk(7, 10))

		result = buf.read_range(0, 10)
		result[:] = 0

		# Buffer data must be unchanged after modifying the returned copy
		second_read = buf.read_range(0, 10)
		assert numpy.all(second_read == 7)

	def test_empty_read_returns_empty_array (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=100, channels=1)

		result = buf.read_range(5, 5)

		assert result.shape == (0, 1)

	def test_read_partial_range (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=100, channels=1)
		chunk = numpy.arange(20, dtype=numpy.int16)
		buf.write(chunk)

		result = buf.read_range(5, 10)

		assert result.shape == (5, 1)
		assert numpy.all(result.flatten() == numpy.arange(5, 10, dtype=numpy.int16))


class TestCircularBufferWrapAround:

	def test_write_wraps_around (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=10, channels=1)

		# Fill the buffer exactly
		buf.write(_make_chunk(1, 10))
		assert buf.write_head == 0  # Should wrap back to 0

		# Write again — should overwrite from the beginning
		buf.write(_make_chunk(2, 5))
		assert buf.write_head == 5

	def test_read_wraps_around (self) -> None:
		"""Write past the end and read a range that spans the wrap boundary."""
		buf = subsample.buffer.CircularBuffer(max_frames=10, channels=1)

		# Write frames 0–9 with value 1
		buf.write(_make_chunk(1, 10))
		# Write frames 10–14 with value 2 (overwrites positions 0–4)
		buf.write(_make_chunk(2, 5))

		# Read frames 8–13: positions 8,9 have value 1; positions 10–12 (0–2) have value 2
		result = buf.read_range(8, 13)

		assert result.shape == (5, 1)
		assert numpy.all(result[:2] == 1)
		assert numpy.all(result[2:] == 2)

	def test_overwritten_frames_are_clamped (self) -> None:
		"""Requesting frames older than the buffer capacity returns what's available."""
		buf = subsample.buffer.CircularBuffer(max_frames=10, channels=1)

		# Write 20 frames — first 10 are overwritten
		buf.write(_make_chunk(1, 10))
		buf.write(_make_chunk(2, 10))

		# Requesting from frame 0 should be clamped to frame 10 (oldest available)
		result = buf.read_range(0, 20)

		assert result.shape == (10, 1)
		assert numpy.all(result == 2)


class TestCircularBufferStereo:

	def test_stereo_write_and_read (self) -> None:
		buf = subsample.buffer.CircularBuffer(max_frames=20, channels=2)
		chunk = numpy.ones((10, 2), dtype=numpy.int16) * 99

		buf.write(chunk)
		result = buf.read_range(0, 10)

		assert result.shape == (10, 2)
		assert numpy.all(result == 99)
