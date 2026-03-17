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


class TestLogNormalize:

	def test_at_min_ref_returns_zero (self) -> None:
		"""A value equal to min_ref should map to exactly 0.0."""
		assert subsample.analysis._log_normalize(0.001, 0.001, 2.0) == 0.0

	def test_below_min_ref_returns_zero (self) -> None:
		"""Any value below min_ref should clamp to 0.0."""
		assert subsample.analysis._log_normalize(0.0005, 0.001, 2.0) == 0.0

	def test_at_max_ref_returns_one (self) -> None:
		"""A value equal to max_ref should map to exactly 1.0."""
		assert subsample.analysis._log_normalize(2.0, 0.001, 2.0) == 1.0

	def test_above_max_ref_returns_one (self) -> None:
		"""Any value above max_ref should clamp to 1.0."""
		assert subsample.analysis._log_normalize(5.0, 0.001, 2.0) == 1.0

	def test_geometric_midpoint_returns_half (self) -> None:
		"""The geometric mean of min and max should map to 0.5.

		The log midpoint of 0.001 and 2.0 is sqrt(0.001 * 2.0) ≈ 0.04472.
		This is the perceptual halfway point on a log scale.
		"""
		midpoint = math.sqrt(0.001 * 2.0)
		score = subsample.analysis._log_normalize(midpoint, 0.001, 2.0)

		assert abs(score - 0.5) < 1e-6


class TestAnalyze:

	def _params (self, sample_rate: int = 44100) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(_make_audio_cfg(sample_rate))

	def _sine_wave (
		self,
		frequency: float = 440.0,
		duration_seconds: float = 0.5,
		sample_rate: int = 44100,
		amplitude: float = 0.9,
	) -> numpy.ndarray:
		"""Generate a mono int16 sine wave at the given frequency."""
		n = int(duration_seconds * sample_rate)
		t = numpy.arange(n) / sample_rate
		wave = (numpy.sin(2 * numpy.pi * frequency * t) * amplitude * 32000).astype(numpy.int16)
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

	def _percussive_click (
		self,
		duration_seconds: float = 0.5,
		sample_rate: int = 44100,
	) -> numpy.ndarray:
		"""Generate a mono int16 signal with a sharp transient at the start.

		The first 10 samples are at full amplitude; the rest are silence.
		This models the most extreme percussive onset (≈ 0.2 ms at 44.1 kHz).
		"""
		n = int(duration_seconds * sample_rate)
		audio = numpy.zeros(n, dtype=numpy.int16)
		audio[:10] = 32000  # sharp transient at the very start
		return audio.reshape(-1, 1)

	def _gradual_ramp (
		self,
		duration_seconds: float = 1.0,
		sample_rate: int = 44100,
	) -> numpy.ndarray:
		"""Generate a mono int16 signal that ramps from silence to full amplitude.

		The amplitude envelope is linear over the full duration.
		This produces a long, slow attack.
		"""
		n = int(duration_seconds * sample_rate)
		t = numpy.linspace(0.0, 1.0, n)
		# Multiply a sine wave by a linear envelope for a tonal gradual onset
		freq = 440.0
		times = numpy.arange(n) / sample_rate
		wave = numpy.sin(2 * numpy.pi * freq * times) * t * 32000
		return wave.astype(numpy.int16).reshape(-1, 1)

	def _decaying_tone (
		self,
		duration_seconds: float = 1.0,
		sample_rate: int = 44100,
		decay_tau_seconds: float = 0.8,
	) -> numpy.ndarray:
		"""Generate a mono int16 tone that peaks immediately then decays exponentially.

		decay_tau_seconds controls how fast the energy falls (time constant τ).
		Large τ = slow decay (long release); small τ = fast decay (short release).
		"""
		n = int(duration_seconds * sample_rate)
		times = numpy.arange(n) / sample_rate
		envelope = numpy.exp(-times / decay_tau_seconds)
		freq = 440.0
		wave = numpy.sin(2 * numpy.pi * freq * times) * envelope * 32000
		return wave.astype(numpy.int16).reshape(-1, 1)

	# ------------------------------------------------------------------
	# Spectral flatness (existing, preserved)
	# ------------------------------------------------------------------

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

	# ------------------------------------------------------------------
	# Attack
	# ------------------------------------------------------------------

	def test_percussive_attack_is_low (self) -> None:
		"""A signal that peaks immediately should score close to 0."""
		audio = self._percussive_click()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		# A 10-sample transient at 44100 Hz = ~0.23 ms; well below the 1 ms
		# lower reference, so the score should be at or near 0.
		assert result.attack < 0.3

	def test_gradual_attack_scores_higher_than_percussive (self) -> None:
		"""A slow ramp should have a higher attack score than an instant click."""
		click = self._percussive_click()
		ramp = self._gradual_ramp()
		click_result = subsample.analysis.analyze(click, self._params(), bit_depth=16)
		ramp_result = subsample.analysis.analyze(ramp, self._params(), bit_depth=16)

		assert ramp_result.attack > click_result.attack

	def test_attack_is_in_range (self) -> None:
		"""Attack score must always be in [0.0, 1.0]."""
		for audio in [self._percussive_click(), self._gradual_ramp(), self._white_noise()]:
			result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)
			assert 0.0 <= result.attack <= 1.0

	# ------------------------------------------------------------------
	# Release
	# ------------------------------------------------------------------

	def test_long_decay_has_higher_release_than_short (self) -> None:
		"""A slowly decaying tone should score higher release than an instant cutoff.

		_decaying_tone with a large tau keeps energy above threshold for longer,
		producing a higher release score than a brief click that silences immediately.
		"""
		slow_decay = self._decaying_tone(decay_tau_seconds=0.8)
		click = self._percussive_click()

		slow_result = subsample.analysis.analyze(slow_decay, self._params(), bit_depth=16)
		click_result = subsample.analysis.analyze(click, self._params(), bit_depth=16)

		assert slow_result.release > click_result.release

	def test_release_is_in_range (self) -> None:
		"""Release score must always be in [0.0, 1.0]."""
		for audio in [self._percussive_click(), self._decaying_tone(), self._white_noise()]:
			result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)
			assert 0.0 <= result.release <= 1.0

	# ------------------------------------------------------------------
	# Spectral centroid (bassy vs trebly)
	# ------------------------------------------------------------------

	def test_low_frequency_sine_has_low_centroid (self) -> None:
		"""A 100 Hz tone should score low on the centroid scale (bassy)."""
		audio = self._sine_wave(frequency=100.0)
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		assert result.spectral_centroid < 0.4

	def test_high_frequency_sine_has_high_centroid (self) -> None:
		"""A 10 kHz tone should score high on the centroid scale (trebly)."""
		audio = self._sine_wave(frequency=10000.0)
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		assert result.spectral_centroid > 0.6

	def test_higher_frequency_has_higher_centroid (self) -> None:
		"""Centroid score should increase monotonically with tone frequency."""
		low = subsample.analysis.analyze(self._sine_wave(100.0), self._params(), bit_depth=16)
		mid = subsample.analysis.analyze(self._sine_wave(1000.0), self._params(), bit_depth=16)
		high = subsample.analysis.analyze(self._sine_wave(10000.0), self._params(), bit_depth=16)

		assert low.spectral_centroid < mid.spectral_centroid < high.spectral_centroid

	def test_spectral_centroid_is_in_range (self) -> None:
		"""Spectral centroid must always be in [0.0, 1.0]."""
		for audio in [self._sine_wave(100.0), self._sine_wave(10000.0), self._white_noise()]:
			result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)
			assert 0.0 <= result.spectral_centroid <= 1.0

	# ------------------------------------------------------------------
	# Spectral bandwidth (narrow vs wide)
	# ------------------------------------------------------------------

	def test_pure_tone_has_low_bandwidth (self) -> None:
		"""A single sine wave has energy concentrated at one frequency (narrow)."""
		audio = self._sine_wave()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		assert result.spectral_bandwidth < 0.4

	def test_white_noise_has_higher_bandwidth_than_tone (self) -> None:
		"""White noise has energy spread across all frequencies; should be wider."""
		tone = subsample.analysis.analyze(self._sine_wave(), self._params(), bit_depth=16)
		noise = subsample.analysis.analyze(self._white_noise(), self._params(), bit_depth=16)

		assert noise.spectral_bandwidth > tone.spectral_bandwidth

	def test_spectral_bandwidth_is_in_range (self) -> None:
		"""Spectral bandwidth must always be in [0.0, 1.0]."""
		for audio in [self._sine_wave(), self._white_noise()]:
			result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)
			assert 0.0 <= result.spectral_bandwidth <= 1.0

	# ------------------------------------------------------------------
	# Empty audio guard
	# ------------------------------------------------------------------

	def test_empty_audio_returns_zeros (self) -> None:
		"""An empty array should return all-zero metrics without error."""
		empty = numpy.zeros((0, 1), dtype=numpy.int16)
		result = subsample.analysis.analyze(empty, self._params(), bit_depth=16)

		assert result.spectral_flatness == 0.0
		assert result.attack == 0.0
		assert result.release == 0.0
		assert result.spectral_centroid == 0.0
		assert result.spectral_bandwidth == 0.0
