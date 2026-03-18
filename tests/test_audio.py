"""Tests for subsample.audio.unpack_audio."""

import typing
import unittest.mock

import numpy
import pytest

import subsample.audio
import subsample.config
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


class TestAudioReader:

	"""Tests for the callback-based audio capture."""

	def _make_audio_cfg (self, chunk_size: int = 16) -> subsample.config.AudioConfig:
		"""Build a minimal AudioConfig for testing."""
		return subsample.config.AudioConfig(
			sample_rate=44100,
			bit_depth=16,
			channels=1,
			chunk_size=chunk_size,
		)

	def _make_reader (
		self,
		chunk_size: int = 16,
	) -> tuple["subsample.audio.AudioReader", typing.Any]:
		"""Return (reader, mock_stream) with pa.open() mocked out."""
		mock_stream = unittest.mock.MagicMock()
		mock_pa = unittest.mock.MagicMock()
		mock_pa.open.return_value = mock_stream

		cfg = self._make_audio_cfg(chunk_size=chunk_size)
		reader = subsample.audio.AudioReader(mock_pa, device_index=0, audio_cfg=cfg)

		return reader, mock_stream

	def test_read_returns_correct_shape (self) -> None:
		"""read() should unpack raw bytes and return shape (chunk_size, channels)."""
		chunk_size = 16
		reader, _ = self._make_reader(chunk_size=chunk_size)

		# Simulate the callback delivering raw int16 bytes
		raw = numpy.zeros(chunk_size, dtype=numpy.int16).tobytes()
		reader._callback(raw, chunk_size, {}, 0)

		chunk = reader.read()

		reader.stop()

		assert chunk.shape == (chunk_size, 1)
		assert chunk.dtype == numpy.int16

	def test_overflow_count_incremented (self) -> None:
		"""Non-zero status_flags should increment overflow_count."""
		reader, _ = self._make_reader()

		assert reader.overflow_count == 0

		raw = numpy.zeros(16, dtype=numpy.int16).tobytes()
		reader._callback(raw, 16, {}, 1)  # status_flags = 1 → overflow
		reader._callback(raw, 16, {}, 1)

		reader.stop()

		assert reader.overflow_count == 2

	def test_stop_closes_stream (self) -> None:
		"""stop() should call stop_stream() and close() on the underlying stream."""
		reader, mock_stream = self._make_reader()

		reader.stop()

		mock_stream.stop_stream.assert_called_once()
		mock_stream.close.assert_called_once()


class TestFindDeviceByName:

	"""Tests for find_device_by_name()."""

	def _make_pa (self, device_names: list[str]) -> unittest.mock.MagicMock:
		"""Return a mock PyAudio exposing the given device names as input devices."""
		mock_pa = unittest.mock.MagicMock()
		mock_pa.get_device_count.return_value = len(device_names)

		def _device_info (i: int) -> dict[str, typing.Union[str, int]]:
			return {"index": i, "name": device_names[i], "maxInputChannels": 1}

		mock_pa.get_device_info_by_index.side_effect = _device_info
		return mock_pa

	def test_exact_match_returns_index (self) -> None:
		pa = self._make_pa(["Built-in Mic", "Samson Go Mic: USB Audio (hw:1,0)"])
		assert subsample.audio.find_device_by_name(pa, "Samson Go Mic: USB Audio (hw:1,0)") == 1

	def test_case_insensitive_match (self) -> None:
		pa = self._make_pa(["Built-in Mic", "Samson Go Mic: USB Audio (hw:1,0)"])
		assert subsample.audio.find_device_by_name(pa, "samson go mic") == 1

	def test_substring_match (self) -> None:
		pa = self._make_pa(["Built-in Mic", "Samson Go Mic: USB Audio (hw:1,0)"])
		assert subsample.audio.find_device_by_name(pa, "Samson") == 1

	def test_first_match_returned_when_multiple (self) -> None:
		pa = self._make_pa(["USB Mic A", "USB Mic B", "Built-in Mic"])
		assert subsample.audio.find_device_by_name(pa, "USB") == 0

	def test_no_match_raises_value_error (self) -> None:
		pa = self._make_pa(["Built-in Mic", "HDMI Output"])
		with pytest.raises(ValueError, match="nonexistent"):
			subsample.audio.find_device_by_name(pa, "nonexistent")

	def test_error_message_lists_available_devices (self) -> None:
		pa = self._make_pa(["Built-in Mic", "USB Audio Device"])
		with pytest.raises(ValueError) as exc_info:
			subsample.audio.find_device_by_name(pa, "Samson")
		msg = str(exc_info.value)
		assert "Built-in Mic" in msg
		assert "USB Audio Device" in msg


class TestUnpackAudioErrors:

	def test_unsupported_bit_depth_raises (self) -> None:
		with pytest.raises(ValueError, match="Unsupported bit depth"):
			subsample.audio.unpack_audio(bytes(2), bit_depth=8, channels=1)
