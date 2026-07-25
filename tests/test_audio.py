"""Tests for subsample.audio."""

import pathlib
import tempfile
import typing
import unittest.mock
import wave

import numpy
import pytest
import soundfile

import subsample.audio
import subsample.config


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

	def _make_audio_cfg (self, buffer_frames: int = 16) -> subsample.config.AudioConfig:
		"""Build a minimal AudioConfig for testing."""
		return subsample.config.AudioConfig(
			sample_rate=44100,
			bit_depth=16,
			channels=1,
			buffer_frames=buffer_frames,
		)

	def _make_reader (
		self,
		buffer_frames: int = 16,
	) -> tuple["subsample.audio.AudioReader", typing.Any]:
		"""Return (reader, mock_stream) with pa.open() mocked out."""
		mock_stream = unittest.mock.MagicMock()
		mock_pa = unittest.mock.MagicMock()
		mock_pa.open.return_value = mock_stream

		cfg = self._make_audio_cfg(buffer_frames=buffer_frames)
		reader = subsample.audio.AudioReader(mock_pa, device_index=0, audio_cfg=cfg)

		return reader, mock_stream

	def test_read_returns_correct_shape (self) -> None:
		"""read() should unpack raw bytes and return shape (buffer_frames, channels)."""
		buffer_frames = 16
		reader, _ = self._make_reader(buffer_frames=buffer_frames)

		# Simulate the callback delivering raw int16 bytes
		raw = numpy.zeros(buffer_frames, dtype=numpy.int16).tobytes()
		reader._callback(raw, buffer_frames, {}, 0)

		chunk = reader.read()

		reader.stop()

		assert chunk.shape == (buffer_frames, 1)
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

	def test_read_returns_none_on_timeout (self) -> None:
		"""read(timeout=...) should return None when no data arrives before the timeout."""
		reader, _ = self._make_reader()

		# Queue is empty — read with a very short timeout should return None.
		result = reader.read(timeout=0.01)

		reader.stop()

		assert result is None

	def test_read_no_timeout_returns_chunk (self) -> None:
		"""read() with no timeout argument should still return a chunk when data is available."""
		buffer_frames = 16
		reader, _ = self._make_reader(buffer_frames=buffer_frames)

		raw = numpy.zeros(buffer_frames, dtype=numpy.int16).tobytes()
		reader._callback(raw, buffer_frames, {}, 0)

		# Should not require a timeout argument — backward-compatible call.
		chunk = reader.read()

		reader.stop()

		assert chunk is not None
		assert chunk.shape == (buffer_frames, 1)


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


class TestReadAudioFile:

	"""Tests for subsample.audio.read_audio_file()."""

	def _write_wav (
		self,
		path: pathlib.Path,
		samples: numpy.ndarray,
		sample_rate: int,
		sample_width: int,
	) -> None:
		"""Write a minimal WAV file containing the given samples."""
		with wave.open(str(path), "wb") as wf:
			wf.setnchannels(1)
			wf.setsampwidth(sample_width)
			wf.setframerate(sample_rate)
			wf.writeframes(samples.tobytes())

	def test_16bit_mono_fields (self) -> None:
		"""read_audio_file() should return correct metadata for a 16-bit mono WAV."""
		samples = numpy.array([0, 100, -100, 32767, -32768], dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			path = pathlib.Path(tmp) / "test.wav"
			self._write_wav(path, samples, sample_rate=44100, sample_width=2)

			info = subsample.audio.read_audio_file(path)

		assert info.sample_rate == 44100
		assert info.bit_depth == 16
		assert info.channels == 1
		assert info.audio.dtype == numpy.int16
		assert info.audio.shape == (len(samples), 1)
		assert numpy.array_equal(info.audio.flatten(), samples)

	def test_8bit_wav_falls_through_to_soundfile (self) -> None:
		"""An 8-bit PCM WAV is readable via soundfile — unpack_audio's
		ValueError from the wave fast path must fall through, not escape as
		'Unsupported bit depth 8'."""

		samples = numpy.array([128, 200, 60, 255, 0], dtype=numpy.uint8)

		with tempfile.TemporaryDirectory() as tmp:
			path = pathlib.Path(tmp) / "eight.wav"
			self._write_wav(path, samples, sample_rate=22050, sample_width=1)

			info = subsample.audio.read_audio_file(path)

		assert info.sample_rate == 22050
		assert info.channels == 1
		assert info.audio.shape[0] == len(samples)
		assert info.audio.dtype in (numpy.int16, numpy.int32)

	def test_stereo_shape (self) -> None:
		"""Stereo WAV should produce channels=2 and correct array shape."""
		samples = numpy.zeros(20, dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			path = pathlib.Path(tmp) / "stereo.wav"
			with wave.open(str(path), "wb") as wf:
				wf.setnchannels(2)
				wf.setsampwidth(2)
				wf.setframerate(48000)
				wf.writeframes(samples.tobytes())

			info = subsample.audio.read_audio_file(path)

		assert info.channels == 2
		assert info.sample_rate == 48000
		assert info.audio.shape == (10, 2)

	def test_audio_matches_unpack_audio (self) -> None:
		"""read_audio_file() audio should match unpack_audio() on the same bytes."""
		samples = numpy.array([1000, -1000, 0, 32767], dtype=numpy.int16)

		with tempfile.TemporaryDirectory() as tmp:
			path = pathlib.Path(tmp) / "match.wav"
			self._write_wav(path, samples, sample_rate=44100, sample_width=2)

			info = subsample.audio.read_audio_file(path)
			expected = subsample.audio.unpack_audio(samples.tobytes(), bit_depth=16, channels=1)

		assert numpy.array_equal(info.audio, expected)

	def test_nonexistent_file_raises (self) -> None:
		"""read_audio_file() should raise OSError for a nonexistent path."""
		with pytest.raises(OSError):
			subsample.audio.read_audio_file(pathlib.Path("/nonexistent/path.wav"))

	def test_non_audio_raises_valueerror (self) -> None:
		"""read_audio_file() should raise ValueError for an unsupported format."""
		with tempfile.TemporaryDirectory() as tmp:
			path = pathlib.Path(tmp) / "notawav.wav"
			path.write_bytes(b"this is not a wav file")

			with pytest.raises(ValueError, match="Unsupported audio format"):
				subsample.audio.read_audio_file(path)


# ---------------------------------------------------------------------------
# float32_to_pcm_bytes round-trip
# ---------------------------------------------------------------------------

class TestFloat32ToPcmBytes:

	def test_round_trip_16bit (self) -> None:
		"""16-bit PCM → unpack → float32_to_pcm_bytes → within ±1 of original."""
		samples = numpy.array([[1000, -1000], [32767, -32768]], dtype=numpy.int16)
		raw = samples.tobytes()
		unpacked = subsample.audio.unpack_audio(raw, 16, 2)
		repacked = subsample.audio.float32_to_pcm_bytes(
			unpacked.astype(numpy.float32) / 32768.0, 16,
		)
		restored = subsample.audio.unpack_audio(repacked, 16, 2)
		# Float32 quantisation may introduce ±1 LSB error.
		numpy.testing.assert_allclose(restored, unpacked, atol=1)

	def test_round_trip_32bit (self) -> None:
		"""32-bit PCM → unpack → float32_to_pcm_bytes → recoverable bytes."""
		samples = numpy.array([[100000, -100000]], dtype=numpy.int32)
		raw = samples.tobytes()
		unpacked = subsample.audio.unpack_audio(raw, 32, 2)
		repacked = subsample.audio.float32_to_pcm_bytes(
			unpacked.astype(numpy.float32) / 2147483648.0, 32,
		)
		restored = subsample.audio.unpack_audio(repacked, 32, 2)
		assert restored.shape == unpacked.shape
		assert restored.dtype == unpacked.dtype

		# float32's 24-bit mantissa can cost the bottom ~8 bits of a 32-bit
		# sample — but the VALUES must round-trip within that tolerance, not
		# merely have the right shape (an all-zeros output previously passed).
		numpy.testing.assert_allclose(restored, unpacked, atol=256)

	def test_output_length (self) -> None:
		"""Output byte length matches expected frames * channels * sample_width."""
		audio = numpy.zeros((10, 2), dtype=numpy.float32)
		result = subsample.audio.float32_to_pcm_bytes(audio, 16)
		assert len(result) == 10 * 2 * 2  # 10 frames, 2 channels, 2 bytes each

	def test_full_scale_does_not_wrap_at_any_depth (self) -> None:
		"""A full-scale +1.0 peak must stay positive at 16/24/32-bit.

		Regression: the 32-bit path scaled by 2147483647 in float32, which
		rounds up to 2**31 and wraps to full-scale NEGATIVE on the int32 cast —
		a loud click on the loudest material.
		"""
		full_scale = numpy.ones((4, 1), dtype=numpy.float32)   # +1.0 peak

		for bit_depth in (16, 24, 32):
			raw = subsample.audio.float32_to_pcm_bytes(full_scale, bit_depth)
			restored = subsample.audio.unpack_audio(raw, bit_depth, 1)

			assert numpy.all(restored > 0), f"{bit_depth}-bit full-scale wrapped to negative"

	def test_negative_full_scale_does_not_wrap (self) -> None:
		"""A full-scale -1.0 trough stays negative at every depth."""
		trough = -numpy.ones((4, 1), dtype=numpy.float32)

		for bit_depth in (16, 24, 32):
			raw = subsample.audio.float32_to_pcm_bytes(trough, bit_depth)
			restored = subsample.audio.unpack_audio(raw, bit_depth, 1)

			assert numpy.all(restored < 0), f"{bit_depth}-bit -full-scale wrapped"

	def test_round_trip_24bit (self) -> None:
		"""24-bit float→PCM→float round-trips within 24-bit resolution (was untested)."""
		floats = numpy.array([[0.1, -0.1], [0.999, -0.999]], dtype=numpy.float32)
		raw = subsample.audio.float32_to_pcm_bytes(floats, 24)
		# unpack_audio returns int32-range values; normalise back to [-1, 1].
		restored = subsample.audio.unpack_audio(raw, 24, 2).astype(numpy.float64) / 2147483648.0
		numpy.testing.assert_allclose(restored, floats, atol=1e-3)


# ---------------------------------------------------------------------------
# get_pyaudio_format
# ---------------------------------------------------------------------------

class TestGetPyaudioFormat:

	def test_known_depths_map_to_constants (self) -> None:
		"""16/24/32 map to the matching PyAudio format constants."""
		import pyaudio
		assert subsample.audio.get_pyaudio_format(16) == pyaudio.paInt16
		assert subsample.audio.get_pyaudio_format(24) == pyaudio.paInt24
		assert subsample.audio.get_pyaudio_format(32) == pyaudio.paInt32

	def test_unsupported_depth_raises (self) -> None:
		"""An unsupported bit depth raises ValueError naming the supported set."""
		with pytest.raises(ValueError, match="Unsupported bit depth"):
			subsample.audio.get_pyaudio_format(8)


# ---------------------------------------------------------------------------
# get_device_channels
# ---------------------------------------------------------------------------

class TestGetDeviceChannels:

	def _make_pa (self, max_input_channels: int) -> unittest.mock.MagicMock:
		"""Return a mock PyAudio instance reporting the given channel count."""
		pa = unittest.mock.MagicMock()
		pa.get_device_info_by_index.return_value = {
			"name": "Mock Device",
			"maxInputChannels": max_input_channels,
		}
		return pa

	def test_returns_channel_count (self) -> None:
		"""Returns the device's maxInputChannels value as an int."""
		pa = self._make_pa(2)

		result = subsample.audio.get_device_channels(pa, 0)

		assert result == 2

	def test_mono_device (self) -> None:
		"""Returns 1 for a mono device."""
		pa = self._make_pa(1)

		result = subsample.audio.get_device_channels(pa, 0)

		assert result == 1

	def test_zero_channels_raises (self) -> None:
		"""Raises ValueError when the device reports no input channels (output-only)."""
		pa = self._make_pa(0)

		with pytest.raises(ValueError, match="no input channels"):
			subsample.audio.get_device_channels(pa, 0)


# ---------------------------------------------------------------------------
# get_output_device_channels
# ---------------------------------------------------------------------------

class TestGetOutputDeviceChannels:

	def _make_pa (self, max_output_channels: int) -> unittest.mock.MagicMock:
		"""Return a mock PyAudio instance reporting the given output channel count."""
		pa = unittest.mock.MagicMock()
		pa.get_device_info_by_index.return_value = {
			"name": "Mock Output Device",
			"maxOutputChannels": max_output_channels,
		}
		return pa

	def test_returns_channel_count (self) -> None:
		"""Returns the device's maxOutputChannels value as an int."""
		pa = self._make_pa(16)

		result = subsample.audio.get_output_device_channels(pa, 0)

		assert result == 16

	def test_stereo_device (self) -> None:
		"""Returns 2 for a stereo device."""
		pa = self._make_pa(2)

		result = subsample.audio.get_output_device_channels(pa, 0)

		assert result == 2

	def test_zero_channels_raises (self) -> None:
		"""Raises ValueError when the device reports no output channels (input-only)."""
		pa = self._make_pa(0)

		with pytest.raises(ValueError, match="no output channels"):
			subsample.audio.get_output_device_channels(pa, 0)


class TestSelectInputChannels:

	"""Tests for select_input_channels() interactive prompt."""

	def test_basic_selection (self) -> None:
		"""'1,2' on a 4-channel device returns (0, 1)."""
		with unittest.mock.patch("builtins.input", return_value="1,2"):
			result = subsample.audio.select_input_channels("Test Device", 4)
		assert result == (0, 1)

	def test_non_contiguous (self) -> None:
		"""'1,3' selects channels 0 and 2."""
		with unittest.mock.patch("builtins.input", return_value="1,3"):
			result = subsample.audio.select_input_channels("Test Device", 8)
		assert result == (0, 2)

	def test_single_channel (self) -> None:
		"""'5' selects channel 4."""
		with unittest.mock.patch("builtins.input", return_value="5"):
			result = subsample.audio.select_input_channels("Test Device", 8)
		assert result == (4,)

	def test_out_of_range_raises (self) -> None:
		"""Channel number > max_channels raises ValueError."""
		with unittest.mock.patch("builtins.input", return_value="5"):
			with pytest.raises(ValueError, match="out of range"):
				subsample.audio.select_input_channels("Test Device", 4)

	def test_empty_input_raises (self) -> None:
		"""Empty string raises ValueError."""
		with unittest.mock.patch("builtins.input", return_value=""):
			with pytest.raises(ValueError, match="No input"):
				subsample.audio.select_input_channels("Test Device", 4)

	def test_duplicate_raises (self) -> None:
		"""Duplicate channel raises ValueError."""
		with unittest.mock.patch("builtins.input", return_value="2,2"):
			with pytest.raises(ValueError, match="Duplicate"):
				subsample.audio.select_input_channels("Test Device", 4)


class TestFlacReadPrecision:

	"""read_audio_file must preserve the native bit depth of FLAC files.

	Regression guard for the soundfile-fallback path: previously it forced
	dtype=int16 unconditionally, which would truncate a 24-bit FLAC's
	lower 8 bits on readback.  Now the dtype and the returned bit_depth
	are selected from soundfile.info().subtype.
	"""

	def test_24bit_flac_preserves_full_precision (self, tmp_path: pathlib.Path) -> None:
		"""Writing 24-bit-worth of int32 samples to FLAC and reading back
		recovers the same values in the upper 24 bits.
		"""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# Three hand-picked 24-bit values covering full positive, full
		# negative, and an off-centre value with non-zero low bits in the
		# 24-bit representation.  Stored int32 = 24-bit value << 8.
		samples_24 = [0x7FFFFF, -0x800000, 0x123456]
		audio = numpy.array([[v << 8] for v in samples_24], dtype=numpy.int32)

		flac_path = tmp_path / "precision.flac"
		soundfile.write(str(flac_path), audio, 44100, subtype="PCM_24", format="FLAC")

		info = subsample.audio.read_audio_file(flac_path)

		assert info.bit_depth == 24
		assert info.audio.dtype == numpy.int32

		# The upper 24 bits should match the original values.  Shift back
		# down and compare; libsndfile's encoder leaves the bottom 8 bits
		# zero on decode.
		recovered = info.audio[:, 0] >> 8
		assert list(recovered) == samples_24

	def test_16bit_flac_returns_int16 (self, tmp_path: pathlib.Path) -> None:
		"""A 16-bit FLAC reads back as int16 with bit_depth 16."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		samples = numpy.array([[1000], [-1000], [32767], [-32768]], dtype=numpy.int16)
		flac_path = tmp_path / "sixteen.flac"
		soundfile.write(str(flac_path), samples, 44100, subtype="PCM_16", format="FLAC")

		info = subsample.audio.read_audio_file(flac_path)

		assert info.bit_depth == 16
		assert info.audio.dtype == numpy.int16
		numpy.testing.assert_array_equal(info.audio, samples)


class TestNonPcmRead:

	"""Non-PCM_16/24/32 subtypes that need special handling in the soundfile
	fallback path: 32-bit FLOAT and 64-bit DOUBLE (must scale float → int32),
	and ALAC_24 / ALAC_32 (must read as int32 to preserve precision).

	Regression guards for two distinct failure modes:
	1. soundfile.read(dtype="int32") on a FLOAT/DOUBLE source direct-casts
	   (1.0 → 1) instead of scaling to full-scale int32; the detection
	   pipeline then saw what looked like silence and produced zero segments.
	2. Falling through to the int16 else-branch for ALAC_24/ALAC_32 silently
	   truncates the upper bits — same family as the original 24-bit FLAC
	   truncation bug."""

	def test_float_wav_scales_to_full_int32_range (self, tmp_path: pathlib.Path) -> None:
		"""A 0.6-amplitude sine in a FLOAT WAV reads back with int32
		magnitudes near 0.6 × 2**31, not as ±1 noise."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		sr = 44100
		t  = numpy.linspace(0, 0.1, int(sr * 0.1), dtype=numpy.float32)
		mono = (numpy.sin(2 * numpy.pi * 440 * t) * 0.6).astype(numpy.float32)
		stereo = numpy.stack([mono, mono], axis=1)

		path = tmp_path / "float.wav"
		soundfile.write(str(path), stereo, sr, subtype="FLOAT")

		info = subsample.audio.read_audio_file(path)

		assert info.bit_depth  == 32
		assert info.audio.dtype == numpy.int32
		assert info.sample_rate == sr
		assert info.channels    == 2

		# Peak should sit close to 0.6 × full-scale int32 (within rounding
		# and a tiny epsilon from sin discretisation).  Before the fix the
		# whole array clamped to ±1.
		expected_peak = 0.6 * (2 ** 31 - 1)
		assert info.audio.max() > expected_peak * 0.95
		assert info.audio.min() < -expected_peak * 0.95

	def test_float_wav_silence_is_zero (self, tmp_path: pathlib.Path) -> None:
		"""A silent FLOAT WAV reads back as all zeros (not ±1 noise)."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		silence = numpy.zeros((1024, 2), dtype=numpy.float32)
		path = tmp_path / "silence.wav"
		soundfile.write(str(path), silence, 44100, subtype="FLOAT")

		info = subsample.audio.read_audio_file(path)

		assert info.bit_depth == 32
		numpy.testing.assert_array_equal(info.audio, numpy.zeros((1024, 2), dtype=numpy.int32))

	def test_float_wav_clipping_safe (self, tmp_path: pathlib.Path) -> None:
		"""A FLOAT WAV with > 1.0 sample doesn't wrap on int32 cast — it clamps."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# Slightly over-scale audio (headroom-mixed files occasionally do this).
		hot = numpy.array([[1.2, -1.2], [0.5, -0.5]], dtype=numpy.float32)
		path = tmp_path / "hot.wav"
		soundfile.write(str(path), hot, 44100, subtype="FLOAT")

		# Explicitly ceiling-free: the scale-to-fit path would pull this peak below
		# full scale and the cast could never over-range, so the clamp being tested
		# here is only reachable with the ceiling off.
		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=None)

		# Over-scale samples must clamp to int32's actual bounds, not wrap to
		# the opposite sign (which would happen on a naked cast of a value
		# slightly above 2**31).  Two's complement is asymmetric: int32 min
		# is -2**31, max is 2**31 - 1.
		assert info.audio[0, 0] == numpy.iinfo(numpy.int32).max
		assert info.audio[0, 1] == numpy.iinfo(numpy.int32).min

	def test_double_wav_scales_to_full_int32_range (self, tmp_path: pathlib.Path) -> None:
		"""A 0.6-amplitude sine in a 64-bit DOUBLE WAV reads back with
		int32 magnitudes near 0.6 × 2**31.  Same failure mode as FLOAT but
		via a different code path (would otherwise fall to the int16
		else-branch and collapse to peaks of ±1)."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		sr = 44100
		t  = numpy.linspace(0, 0.1, int(sr * 0.1), dtype=numpy.float64)
		mono = numpy.sin(2 * numpy.pi * 440 * t) * 0.6
		stereo = numpy.stack([mono, mono], axis=1)

		path = tmp_path / "double.wav"
		soundfile.write(str(path), stereo, sr, subtype="DOUBLE")

		info = subsample.audio.read_audio_file(path)

		assert info.bit_depth   == 32
		assert info.audio.dtype == numpy.int32
		assert info.sample_rate == sr
		assert info.channels    == 2

		expected_peak = 0.6 * (2 ** 31 - 1)
		assert info.audio.max() > expected_peak * 0.95
		assert info.audio.min() < -expected_peak * 0.95

	def test_double_aiff_scales_to_full_int32_range (self, tmp_path: pathlib.Path) -> None:
		"""DOUBLE inside an AIFF container — soundfile normalises the subtype
		string regardless of container, so the same code path serves both."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		sr = 44100
		t  = numpy.linspace(0, 0.05, int(sr * 0.05), dtype=numpy.float64)
		mono = numpy.sin(2 * numpy.pi * 880 * t) * 0.5
		stereo = numpy.stack([mono, mono], axis=1)

		path = tmp_path / "double.aiff"
		soundfile.write(str(path), stereo, sr, subtype="DOUBLE", format="AIFF")

		info = subsample.audio.read_audio_file(path)

		assert info.bit_depth   == 32
		assert info.audio.dtype == numpy.int32

		expected_peak = 0.5 * (2 ** 31 - 1)
		assert info.audio.max() > expected_peak * 0.95

	def test_alac_24_preserves_full_precision (self, tmp_path: pathlib.Path) -> None:
		"""24-bit ALAC (Apple Lossless) in a CAF container reads back as
		int32 with the value in the upper 24 bits, matching how PCM_24 FLAC
		is handled.  Before the fix, ALAC_24 fell to the int16 else-branch
		and silently lost its lower 8 bits."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# Same hand-picked values as the PCM_24 FLAC test, so the same
		# recovery convention applies.
		samples_24 = [0x7FFFFF, -0x800000, 0x123456]
		audio = numpy.array([[v << 8] for v in samples_24], dtype=numpy.int32)

		path = tmp_path / "alac24.caf"
		soundfile.write(str(path), audio, 44100, subtype="ALAC_24", format="CAF")

		info = subsample.audio.read_audio_file(path)

		assert info.bit_depth   == 24
		assert info.audio.dtype == numpy.int32

		recovered = info.audio[:, 0] >> 8
		assert list(recovered) == samples_24

	def test_alac_32_returns_full_int32 (self, tmp_path: pathlib.Path) -> None:
		"""32-bit ALAC reads back as int32 at full scale (not truncated to int16).

		Write via float32 input — libsndfile's ALAC_32 encoder corrupts
		direct int32 input, but the real-world Apple Music / Logic encode
		path (DAW float → ALAC_32) round-trips correctly."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		sr = 44100
		t  = numpy.linspace(0, 0.05, int(sr * 0.05), dtype=numpy.float32)
		mono = (numpy.sin(2 * numpy.pi * 440 * t) * 0.6).astype(numpy.float32)
		stereo = numpy.stack([mono, mono], axis=1)

		path = tmp_path / "alac32.caf"
		soundfile.write(str(path), stereo, sr, subtype="ALAC_32", format="CAF")

		info = subsample.audio.read_audio_file(path)

		assert info.bit_depth   == 32
		assert info.audio.dtype == numpy.int32

		# Peak should sit near 0.6 × full-scale int32, *not* near ±0.6 × 2**15
		# (which is what we'd see if the file fell back to int16 read).
		expected_peak = 0.6 * (2 ** 31 - 1)
		assert info.audio.max() > expected_peak * 0.95
		assert info.audio.min() < -expected_peak * 0.95


class TestFloatImportCeiling:

	"""read_audio_file's float_ceiling_dbfs scales a hot float/double source down
	to fit the integer pipeline instead of hard-clipping peaks above 0 dBFS.
	Integer-PCM sources are never touched."""

	def test_hot_float_scaled_down_not_clipped (self, tmp_path: pathlib.Path) -> None:
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# A source peaking at +6 dBFS (linear 2.0) — float has the headroom.
		hot = numpy.array([[2.0, -2.0], [0.5, -0.5]], dtype=numpy.float32)
		path = tmp_path / "hot.wav"
		soundfile.write(str(path), hot, 44100, subtype="FLOAT")

		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=-1.0)

		# Peak now sits at -1 dBFS (0.891 × full scale), NOT clamped at int32 max.
		expected_peak = (10.0 ** (-1.0 / 20.0)) * (2 ** 31)
		assert info.audio.max() < numpy.iinfo(numpy.int32).max  # not clipped
		assert abs(info.audio.max() - expected_peak) < expected_peak * 0.01
		# The quieter frame is scaled by the same gain (dynamics preserved).
		ratio = info.audio[0, 0] / info.audio[1, 0]
		assert abs(ratio - 4.0) < 0.05  # 2.0/0.5 preserved

	def test_float_below_ceiling_untouched (self, tmp_path: pathlib.Path) -> None:
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# Peak 0.5 is already below the -1 dBFS ceiling — no scaling applied.
		quiet = numpy.array([[0.5, -0.5], [0.25, -0.25]], dtype=numpy.float32)
		path = tmp_path / "quiet.wav"
		soundfile.write(str(path), quiet, 44100, subtype="FLOAT")

		scaled = subsample.audio.read_audio_file(path, float_ceiling_dbfs=-1.0)
		legacy = subsample.audio.read_audio_file(path, float_ceiling_dbfs=None)

		numpy.testing.assert_array_equal(scaled.audio, legacy.audio)
		assert abs(scaled.audio.max() - 0.5 * (2 ** 31)) < 0.5 * (2 ** 31) * 0.01

	def test_ceiling_none_still_clips (self, tmp_path: pathlib.Path) -> None:
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		hot = numpy.array([[2.0, -2.0]], dtype=numpy.float32)
		path = tmp_path / "hot.wav"
		soundfile.write(str(path), hot, 44100, subtype="FLOAT")

		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=None)

		# Legacy behaviour: the +6 dBFS peak hard-clips to the int32 rail.
		assert info.audio[0, 0] == numpy.iinfo(numpy.int32).max
		assert info.audio[0, 1] == numpy.iinfo(numpy.int32).min

	def test_integer_pcm_never_scaled (self, tmp_path: pathlib.Path) -> None:
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# A 16-bit PCM source at a high level must be returned verbatim — the
		# ceiling only guards the float/double conversion, never integer sources.
		pcm = numpy.array([[30000, -30000], [1000, -1000]], dtype=numpy.int16)
		path = tmp_path / "pcm16.wav"
		soundfile.write(str(path), pcm, 44100, subtype="PCM_16")

		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=-1.0)

		assert info.bit_depth == 16
		numpy.testing.assert_array_equal(info.audio, pcm)

	def test_hot_double_scaled_down_not_clipped (self, tmp_path: pathlib.Path) -> None:
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		# The 64-bit DOUBLE path (float64) shares the scaling logic but a distinct
		# read branch — exercise it with a hot source too.
		hot = numpy.array([[2.0, -2.0], [0.5, -0.5]], dtype=numpy.float64)
		path = tmp_path / "hot_double.wav"
		soundfile.write(str(path), hot, 44100, subtype="DOUBLE")

		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=-1.0)

		expected_peak = (10.0 ** (-1.0 / 20.0)) * (2 ** 31)
		assert info.audio.max() < numpy.iinfo(numpy.int32).max
		assert abs(info.audio.max() - expected_peak) < expected_peak * 0.01


@pytest.fixture
def restore_float_ceiling () -> typing.Iterator[None]:

	"""Restore the process-wide float ceiling after a test changes it.

	It is module state, so a leak would quietly alter every later test's audio
	reads and surface as a failure far from its cause."""

	previous = subsample.audio._FLOAT_IMPORT_CEILING_DBFS
	try:
		yield
	finally:
		subsample.audio.set_float_import_ceiling(previous)


def _write_hot_float (path: pathlib.Path) -> pathlib.Path:

	"""Write a 32-bit float source peaking at +6 dBFS (linear 2.0) — legal for
	float, above what the integer pipeline can hold."""

	import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

	hot = numpy.array([[2.0, -2.0], [0.5, -0.5]], dtype=numpy.float32)
	soundfile.write(str(path), hot, 44100, subtype="FLOAT")
	return path


class TestConfiguredFloatImportCeiling:

	"""set_float_import_ceiling() wires the ceiling process-wide so read paths
	that hold no config — the library loader, the analysis cache, the watcher,
	OSC import — treat a hot float source exactly as CLI file-input does."""

	def test_configured_ceiling_applies_when_arg_omitted (
		self, tmp_path: pathlib.Path, restore_float_ceiling: None,
	) -> None:
		path = _write_hot_float(tmp_path / "hot.wav")
		subsample.audio.set_float_import_ceiling(-1.0)

		info = subsample.audio.read_audio_file(path)   # no explicit ceiling

		expected_peak = (10.0 ** (-1.0 / 20.0)) * (2 ** 31)
		assert info.audio.max() < numpy.iinfo(numpy.int32).max
		assert abs(info.audio.max() - expected_peak) < expected_peak * 0.01

	def test_unwired_default_matches_config_default (self, tmp_path: pathlib.Path) -> None:
		"""Nothing wired — a script, a test, direct library use — must still read
		audio the way a default-config player does.  An unwired reader that
		analysed a hot float sample differently would write a sidecar the wired
		player later trusts, leaving it playing audio its fingerprint misdescribes."""
		path = _write_hot_float(tmp_path / "hot.wav")

		assert subsample.audio._FLOAT_IMPORT_CEILING_DBFS \
			== subsample.config.AudioConfig.float_import_ceiling_dbfs

		info = subsample.audio.read_audio_file(path)

		expected_peak = (10.0 ** (-1.0 / 20.0)) * (2 ** 31)
		assert info.audio.max() < numpy.iinfo(numpy.int32).max
		assert abs(info.audio.max() - expected_peak) < expected_peak * 0.01

	def test_explicit_none_overrides_configured (
		self, tmp_path: pathlib.Path, restore_float_ceiling: None,
	) -> None:
		"""None is a caller forcing the historical clip — it must not be confused
		with passing nothing, which follows the configured value."""
		path = _write_hot_float(tmp_path / "hot.wav")
		subsample.audio.set_float_import_ceiling(-1.0)

		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=None)

		assert info.audio.max() == numpy.iinfo(numpy.int32).max

	def test_explicit_value_overrides_configured (
		self, tmp_path: pathlib.Path, restore_float_ceiling: None,
	) -> None:
		path = _write_hot_float(tmp_path / "hot.wav")
		subsample.audio.set_float_import_ceiling(-1.0)

		info = subsample.audio.read_audio_file(path, float_ceiling_dbfs=-6.0)

		expected_peak = (10.0 ** (-6.0 / 20.0)) * (2 ** 31)
		assert abs(info.audio.max() - expected_peak) < expected_peak * 0.01

	def test_configured_none_clips (
		self, tmp_path: pathlib.Path, restore_float_ceiling: None,
	) -> None:
		"""float_import_ceiling_dbfs: null must reach the read paths as a real
		opt-out, not be overridden by a stale earlier value."""
		path = _write_hot_float(tmp_path / "hot.wav")
		subsample.audio.set_float_import_ceiling(-1.0)
		subsample.audio.set_float_import_ceiling(None)

		info = subsample.audio.read_audio_file(path)

		assert info.audio.max() == numpy.iinfo(numpy.int32).max


class TestNonFiniteFloatSamples:

	"""A float source containing NaN or inf must not become a full-scale click.

	numpy.clip passes NaN straight through and .astype(int32) maps it to
	INT32_MIN — a full-scale negative impulse from one bad sample.  An inf was
	worse: it made the ceiling probe inf, so the gain came out 0, the whole file
	went silent, and log10(0) then raised inside the broad handler, which told
	the user their format was unsupported.
	"""

	def _write_float_wav (self, path: pathlib.Path, values: list[float]) -> None:
		soundfile.write(str(path), numpy.array(values, dtype=numpy.float32), 48000, subtype="FLOAT")

	def test_nan_becomes_silence_not_a_full_scale_click (self, tmp_path: pathlib.Path) -> None:
		path = tmp_path / "nan.wav"
		self._write_float_wav(path, [0.5, float("nan"), -0.5])

		subsample.audio.set_float_import_ceiling(-1.0)
		audio = subsample.audio.read_audio_file(path).audio.ravel()

		assert audio[1] == 0
		assert numpy.iinfo(numpy.int32).min not in audio

	def test_inf_does_not_silence_the_file_or_misreport_the_format (self, tmp_path: pathlib.Path) -> None:
		path = tmp_path / "inf.wav"
		self._write_float_wav(path, [0.5, float("inf"), -0.5])

		subsample.audio.set_float_import_ceiling(-1.0)
		audio = subsample.audio.read_audio_file(path).audio.ravel()

		# The finite samples survive with their relative levels intact.
		assert audio[0] > 0 and audio[2] < 0
		assert abs(int(audio[0])) == abs(int(audio[2]))

	def test_nan_handled_with_the_ceiling_disabled_too (self, tmp_path: pathlib.Path) -> None:
		path = tmp_path / "nan_noceiling.wav"
		self._write_float_wav(path, [0.5, float("nan"), -0.5])

		subsample.audio.set_float_import_ceiling(None)
		try:
			audio = subsample.audio.read_audio_file(path).audio.ravel()
		finally:
			subsample.audio.set_float_import_ceiling(-1.0)

		assert audio[1] == 0

	def test_scale_float_to_ceiling_scrubs_non_finite (self) -> None:
		"""The exported helper the import tool uses has the same contract."""

		data   = numpy.array([[0.5], [float("nan")], [2.0]], dtype=numpy.float32)
		scaled = subsample.audio.scale_float_to_ceiling(data, -1.0)

		assert numpy.all(numpy.isfinite(scaled))
		assert float(numpy.max(numpy.abs(scaled))) <= 10.0 ** (-1.0 / 20.0) + 1e-6
