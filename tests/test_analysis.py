"""Tests for subsample.analysis."""

import math

import numpy

import subsample.analysis
import subsample.config


def _make_audio_cfg (sample_rate: int) -> subsample.config.AudioConfig:
	return subsample.config.AudioConfig(
		sample_rate=sample_rate,
		bit_depth=16,
		channels=1,
		chunk_size=1024,
	)


class TestComputeParams:

	def test_44100_hz (self) -> None:
		"""Standard CD rate should yield the classic n_fft=2048."""
		params = subsample.analysis.compute_params(_make_audio_cfg(44100))

		assert params.n_fft == 2048
		assert params.hop_length == 512
		assert params.sample_rate == 44100

	def test_11025_hz (self) -> None:
		"""Quarter rate should quarter the window to n_fft=512."""
		params = subsample.analysis.compute_params(_make_audio_cfg(11025))

		assert params.n_fft == 512
		assert params.hop_length == 128

	def test_22050_hz (self) -> None:
		"""22050 Hz should yield n_fft=1024, hop_length=256."""
		params = subsample.analysis.compute_params(_make_audio_cfg(22050))

		assert params.n_fft == 1024
		assert params.hop_length == 256

	def test_48000_hz (self) -> None:
		"""48 kHz is close enough to 44.1 kHz that n_fft stays at 2048."""
		params = subsample.analysis.compute_params(_make_audio_cfg(48000))

		assert params.n_fft == 2048
		assert params.hop_length == 512

	def test_96000_hz (self) -> None:
		"""96 kHz should yield n_fft=4096, hop_length=1024."""
		params = subsample.analysis.compute_params(_make_audio_cfg(96000))

		assert params.n_fft == 4096
		assert params.hop_length == 1024

	def test_hop_length_is_quarter_n_fft (self) -> None:
		"""hop_length should always equal n_fft // 4."""
		for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
			params = subsample.analysis.compute_params(_make_audio_cfg(rate))
			assert params.hop_length == params.n_fft // 4

	def test_n_fft_is_power_of_two (self) -> None:
		"""n_fft should always be a power of two."""
		for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
			params = subsample.analysis.compute_params(_make_audio_cfg(rate))
			assert params.n_fft > 0 and (params.n_fft & (params.n_fft - 1)) == 0


class TestAnalyze:

	def _params (self, sample_rate: int = 44100) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(_make_audio_cfg(sample_rate))

	def _sine_wave (
		self,
		frequency: float = 440.0,
		duration_seconds: float = 0.5,
		sample_rate: int = 44100,
	) -> numpy.ndarray:
		"""Generate a mono int16 sine wave."""
		n = int(duration_seconds * sample_rate)
		t = numpy.arange(n) / sample_rate
		wave = (numpy.sin(2 * numpy.pi * frequency * t) * 32000).astype(numpy.int16)
		return wave.reshape(-1, 1)

	def _white_noise (
		self,
		duration_seconds: float = 0.5,
		sample_rate: int = 44100,
	) -> numpy.ndarray:
		"""Generate mono int16 white noise."""
		rng = numpy.random.default_rng(seed=42)
		n = int(duration_seconds * sample_rate)
		noise = rng.integers(-32768, 32767, size=n, dtype=numpy.int16)
		return noise.reshape(-1, 1)

	def test_pure_tone_has_low_flatness (self) -> None:
		"""A sine wave is highly tonal; spectral flatness should be low."""
		audio = self._sine_wave()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		assert result.spectral_flatness < 0.3

	def test_white_noise_has_high_flatness (self) -> None:
		"""White noise is spectrally flat; flatness should be higher than a sine wave."""
		audio = self._white_noise()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		# librosa spectral_flatness on random noise lands around 0.5; sine is < 0.1.
		# The key property is that noise >> tone; exact value depends on window size.
		assert result.spectral_flatness > 0.4

	def test_result_is_in_range (self) -> None:
		"""spectral_flatness must be non-negative for any non-silent audio."""
		for audio in [self._sine_wave(), self._white_noise()]:
			result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)
			assert result.spectral_flatness >= 0.0

	def test_stereo_input_does_not_raise (self) -> None:
		"""Stereo audio should be mixed to mono without error."""
		# Use noise rather than silence — silence is degenerate for spectral analysis
		rng = numpy.random.default_rng(seed=99)
		n = int(0.5 * 44100)
		stereo = rng.integers(-32768, 32767, size=(n, 2), dtype=numpy.int16)
		result = subsample.analysis.analyze(stereo, self._params(), bit_depth=16)

		assert isinstance(result.spectral_flatness, float)

	def test_int32_input_does_not_raise (self) -> None:
		"""int32 audio (24-bit or 32-bit) should be handled correctly."""
		n = int(0.5 * 44100)
		# Simulate 24-bit left-shifted int32
		rng = numpy.random.default_rng(seed=7)
		audio = (rng.integers(-8388608, 8388607, size=n, dtype=numpy.int32) * 256).reshape(-1, 1)
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=24)

		assert 0.0 <= result.spectral_flatness <= 1.0

	def test_returns_analysis_result_type (self) -> None:
		"""analyze() should return AnalysisResult with float spectral_flatness."""
		audio = self._sine_wave()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		assert isinstance(result, subsample.analysis.AnalysisResult)
		assert isinstance(result.spectral_flatness, float)
