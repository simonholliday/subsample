"""Tests for subsample.analysis."""

import math

import numpy

import warnings

import subsample.analysis
import subsample.config


class TestComputeParams:

	def test_44100_hz (self) -> None:
		"""Standard CD rate should yield the classic n_fft=2048."""
		params = subsample.analysis.compute_params(44100)

		assert params.n_fft == 2048
		assert params.hop_length == 512
		assert params.sample_rate == 44100

	def test_11025_hz (self) -> None:
		"""Quarter rate should quarter the window to n_fft=512."""
		params = subsample.analysis.compute_params(11025)

		assert params.n_fft == 512
		assert params.hop_length == 128

	def test_22050_hz (self) -> None:
		"""22050 Hz should yield n_fft=1024, hop_length=256."""
		params = subsample.analysis.compute_params(22050)

		assert params.n_fft == 1024
		assert params.hop_length == 256

	def test_48000_hz (self) -> None:
		"""48 kHz is close enough to 44.1 kHz that n_fft stays at 2048."""
		params = subsample.analysis.compute_params(48000)

		assert params.n_fft == 2048
		assert params.hop_length == 512

	def test_96000_hz (self) -> None:
		"""96 kHz should yield n_fft=4096, hop_length=1024."""
		params = subsample.analysis.compute_params(96000)

		assert params.n_fft == 4096
		assert params.hop_length == 1024

	def test_hop_length_is_quarter_n_fft (self) -> None:
		"""hop_length should always equal n_fft // 4."""
		for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
			params = subsample.analysis.compute_params(rate)
			assert params.hop_length == params.n_fft // 4

	def test_n_fft_is_power_of_two (self) -> None:
		"""n_fft should always be a power of two."""
		for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
			params = subsample.analysis.compute_params(rate)
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
		return subsample.analysis.compute_params(sample_rate)

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
		"""A sine wave is highly tonal; after log-normalization flatness should be low."""
		audio = self._sine_wave()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		# After log-normalization against _FLATNESS_MIN/_FLATNESS_MAX, a pure sine
		# (raw Wiener entropy near _FLATNESS_MIN) maps to near 0.0.
		assert result.spectral_flatness < 0.5

	def test_white_noise_has_high_flatness (self) -> None:
		"""White noise is spectrally flat; after log-normalization flatness should be high."""
		audio = self._white_noise()
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		# After log-normalization, white noise (raw Wiener entropy ≈ 0.7–0.9) maps
		# close to 1.0. Key property: noise >> tone; exact value depends on window size.
		assert result.spectral_flatness > 0.7

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

	def test_sustained_loud_signal_attack_is_low (self) -> None:
		"""A signal that is loud from sample 0 must not produce the 0.414 artefact.

		Regression: librosa.feature.rms with center=True (default) pads n_fft//2
		zeros before the signal, making the first 2 frames appear weaker than
		frame 2. For any sustained loud signal this gives a constant attack of
		0.414 regardless of the actual onset shape. center=False removes the
		padding so frame 0 already reflects the real signal level.
		"""
		n = int(0.5 * 44100)
		audio = numpy.full((n, 1), 32000, dtype=numpy.int16)
		result = subsample.analysis.analyze(audio, self._params(), bit_depth=16)

		# A signal that is loud from the very first sample should have near-zero attack
		assert result.attack < 0.1

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
		assert result.zcr == 0.0
		assert result.harmonic_ratio == 0.0
		assert result.spectral_contrast == 0.0
		assert result.voiced_fraction == 0.0


class TestAnalyzeMono:

	"""Tests for the float-input entry point used by the file analysis script."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _float_sine (
		self,
		frequency: float = 440.0,
		duration_seconds: float = 0.5,
	) -> numpy.ndarray:
		"""Generate a normalised float32 mono sine wave."""
		n = int(duration_seconds * 44100)
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * frequency * t) * 0.9).astype(numpy.float32)

	def test_returns_analysis_result (self) -> None:
		"""analyze_mono() should return a complete AnalysisResult."""
		result = subsample.analysis.analyze_mono(self._float_sine(), self._params())

		assert isinstance(result, subsample.analysis.AnalysisResult)

	def test_all_metrics_in_range (self) -> None:
		"""All metrics returned by analyze_mono() must be in [0.0, 1.0]."""
		result = subsample.analysis.analyze_mono(self._float_sine(), self._params())

		assert 0.0 <= result.spectral_flatness <= 1.0
		assert 0.0 <= result.attack <= 1.0
		assert 0.0 <= result.release <= 1.0
		assert 0.0 <= result.spectral_centroid <= 1.0
		assert 0.0 <= result.spectral_bandwidth <= 1.0
		assert 0.0 <= result.zcr <= 1.0
		assert 0.0 <= result.harmonic_ratio <= 1.0
		assert 0.0 <= result.spectral_contrast <= 1.0
		assert 0.0 <= result.voiced_fraction <= 1.0
		assert 0.0 <= result.log_attack_time <= 1.0
		assert 0.0 <= result.spectral_flux <= 1.0

	def test_matches_analyze_for_integer_input (self) -> None:
		"""analyze_mono and analyze should agree on the same audio content."""
		# Build equivalent signals: float mono vs int16 mono
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		float_mono = (numpy.sin(2 * numpy.pi * 440 * t) * 0.9).astype(numpy.float32)
		int_audio = (float_mono * 32768.0).astype(numpy.int16).reshape(-1, 1)

		params = self._params()
		result_float = subsample.analysis.analyze_mono(float_mono, params)
		result_int = subsample.analysis.analyze(int_audio, params, bit_depth=16)

		assert abs(result_float.spectral_flatness - result_int.spectral_flatness) < 0.01
		assert abs(result_float.spectral_centroid - result_int.spectral_centroid) < 0.01

	def test_empty_array_returns_zeros (self) -> None:
		"""An empty float32 array should return all-zero metrics without error."""
		empty = numpy.zeros(0, dtype=numpy.float32)
		result = subsample.analysis.analyze_mono(empty, self._params())

		assert result.spectral_flatness == 0.0
		assert result.attack == 0.0
		assert result.release == 0.0
		assert result.spectral_centroid == 0.0
		assert result.spectral_bandwidth == 0.0
		assert result.zcr == 0.0
		assert result.harmonic_ratio == 0.0
		assert result.spectral_contrast == 0.0
		assert result.voiced_fraction == 0.0

	def test_white_noise_all_metrics_in_range (self) -> None:
		"""White noise (spectrally complex) should stay within [0.0, 1.0]."""
		rng = numpy.random.default_rng(seed=42)
		n = int(0.5 * 44100)
		noise = rng.random(n).astype(numpy.float32) * 2.0 - 1.0
		result = subsample.analysis.analyze_mono(noise, self._params())

		assert 0.0 <= result.spectral_flatness <= 1.0
		assert 0.0 <= result.spectral_bandwidth <= 1.0

	def test_short_signal_no_warning (self) -> None:
		"""Signals shorter than n_fft should not emit a UserWarning from librosa."""
		# 606 samples at 44100 Hz is shorter than the default n_fft of 2048
		short = numpy.zeros(606, dtype=numpy.float32)
		short[100:200] = 0.5  # some non-silent content

		import warnings

		with warnings.catch_warnings():
			warnings.simplefilter("error")
			result = subsample.analysis.analyze_mono(short, self._params())

		assert isinstance(result, subsample.analysis.AnalysisResult)


class TestSpectralOnsetFeatures:

	"""Tests for log_attack_time and spectral_flux in AnalysisResult."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _percussive_click (self, n: int = 22050) -> numpy.ndarray:
		"""Sharp transient: single high-amplitude frame then silence."""
		signal = numpy.zeros(n, dtype=numpy.float32)
		signal[0:10] = 0.9
		return signal

	def _sustained_tone (self, n: int = 22050) -> numpy.ndarray:
		"""Sustained sine wave: constant amplitude throughout."""
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * 440 * t) * 0.9).astype(numpy.float32)

	def _gradual_ramp (self, n: int = 22050) -> numpy.ndarray:
		"""Linear fade-in from silence to full amplitude."""
		return numpy.linspace(0.0, 0.9, n).astype(numpy.float32)

	def test_log_attack_time_in_range (self) -> None:
		"""log_attack_time must be in [0.0, 1.0] for all signal types."""
		for signal in [self._percussive_click(), self._sustained_tone(), self._gradual_ramp()]:
			result = subsample.analysis.analyze_mono(signal, self._params())
			assert 0.0 <= result.log_attack_time <= 1.0

	def test_spectral_flux_in_range (self) -> None:
		"""spectral_flux must be in [0.0, 1.0] for all signal types."""
		for signal in [self._percussive_click(), self._sustained_tone(), self._gradual_ramp()]:
			result = subsample.analysis.analyze_mono(signal, self._params())
			assert 0.0 <= result.spectral_flux <= 1.0

	def test_percussive_has_lower_log_attack_than_ramp (self) -> None:
		"""A sharp click should have a shorter log_attack_time than a slow ramp."""
		click_result = subsample.analysis.analyze_mono(self._percussive_click(), self._params())
		ramp_result = subsample.analysis.analyze_mono(self._gradual_ramp(), self._params())

		assert click_result.log_attack_time < ramp_result.log_attack_time

	def test_empty_returns_zero_onset_features (self) -> None:
		"""Empty audio should produce zero log_attack_time and spectral_flux."""
		result = subsample.analysis.analyze_mono(
			numpy.zeros(0, dtype=numpy.float32), self._params()
		)

		assert result.log_attack_time == 0.0
		assert result.spectral_flux == 0.0


class TestFormatResult:

	"""Tests for the shared result formatting function."""

	def _result (self) -> subsample.analysis.AnalysisResult:
		return subsample.analysis.AnalysisResult(
			spectral_flatness=0.017,
			attack=0.414,
			release=0.000,
			spectral_centroid=0.728,
			spectral_bandwidth=0.757,
			zcr=0.312,
			harmonic_ratio=0.845,
			spectral_contrast=0.210,
			voiced_fraction=0.933,
			log_attack_time=0.123,
			spectral_flux=0.456,
		)

	def test_contains_all_field_names (self) -> None:
		"""Output string must include all metric labels."""
		s = subsample.analysis.format_result(self._result(), 0.07)

		assert "duration=" in s
		assert "flatness=" in s
		assert "attack=" in s
		assert "release=" in s
		assert "centroid=" in s
		assert "bandwidth=" in s
		assert "zcr=" in s
		assert "harmonic=" in s
		assert "contrast=" in s
		assert "voiced=" in s
		assert "log_attack=" in s
		assert "flux=" in s

	def test_duration_formatted_correctly (self) -> None:
		"""Duration should appear as e.g. 'duration=0.07s'."""
		s = subsample.analysis.format_result(self._result(), 0.07)

		assert "duration=0.07s" in s

	def test_metric_values_formatted_correctly (self) -> None:
		"""Metric values should be formatted to 3 decimal places."""
		s = subsample.analysis.format_result(self._result(), 0.07)

		assert "flatness=0.017" in s
		assert "attack=0.414" in s
		assert "release=0.000" in s
		assert "centroid=0.728" in s
		assert "bandwidth=0.757" in s
		assert "zcr=0.312" in s
		assert "harmonic=0.845" in s
		assert "contrast=0.210" in s
		assert "voiced=0.933" in s
		assert "log_attack=0.123" in s
		assert "flux=0.456" in s

	def test_returns_single_line (self) -> None:
		"""Output should be a single line with no newlines."""
		s = subsample.analysis.format_result(self._result(), 1.0)

		assert "\n" not in s

	def test_zero_duration (self) -> None:
		"""Zero duration should format correctly without error."""
		s = subsample.analysis.format_result(self._result(), 0.0)

		assert "duration=0.00s" in s


class TestAnalyzeRhythm:

	"""Tests for analyze_rhythm() and format_rhythm_result()."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _cfg (self) -> subsample.config.AnalysisConfig:
		return subsample.config.AnalysisConfig()

	def _click_track (
		self,
		bpm: float = 120.0,
		duration_seconds: float = 2.0,
		sample_rate: int = 44100,
	) -> numpy.ndarray:
		"""Generate a mono float32 click track at a fixed BPM.

		Each beat is a single full-amplitude sample surrounded by silence.
		This is the simplest possible rhythmic signal for beat detection.
		"""
		n = int(duration_seconds * sample_rate)
		audio = numpy.zeros(n, dtype=numpy.float32)
		beat_interval = int(sample_rate * 60.0 / bpm)

		for i in range(0, n, beat_interval):
			audio[i] = 1.0

		return audio

	def _silence (self, duration_seconds: float = 1.0) -> numpy.ndarray:
		"""Generate a silent float32 mono signal."""
		return numpy.zeros(int(duration_seconds * 44100), dtype=numpy.float32)

	# ------------------------------------------------------------------
	# Empty / silence guard
	# ------------------------------------------------------------------

	def test_empty_array_returns_empty_result (self) -> None:
		"""An empty array should return a zeroed RhythmResult without error."""
		empty = numpy.zeros(0, dtype=numpy.float32)
		result = subsample.analysis.analyze_rhythm(empty, self._params(), self._cfg())

		assert result.tempo_bpm == 0.0
		assert result.beat_times == ()
		assert result.pulse_peak_times == ()
		assert result.pulse_curve.shape[0] == 0
		assert result.onset_times == ()
		assert result.onset_count == 0

	def test_very_short_signal_no_warning (self) -> None:
		"""Signals shorter than n_fft should return empty result without UserWarning.

		The early guard in analyze_rhythm() ensures we never call librosa.beat_track
		or librosa.beat.plp on signals too short for meaningful rhythm analysis.
		This prevents librosa's internal stft from emitting n_fft warnings.
		"""
		short = numpy.array([1.0, 0.5, 0.2], dtype=numpy.float32)  # 3 samples

		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")
			result = subsample.analysis.analyze_rhythm(short, self._params(), self._cfg())

		# No UserWarning should be raised
		user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
		assert len(user_warnings) == 0, f"Unexpected UserWarning: {user_warnings}"

		# Result should be empty (early return)
		assert result.tempo_bpm == 0.0
		assert result.beat_times == ()
		assert result.pulse_peak_times == ()
		assert result.pulse_curve.shape[0] == 0

	# ------------------------------------------------------------------
	# Return types
	# ------------------------------------------------------------------

	def test_returns_rhythm_result_type (self) -> None:
		"""analyze_rhythm() should return a RhythmResult instance."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert isinstance(result, subsample.analysis.RhythmResult)

	def test_tempo_bpm_is_float (self) -> None:
		"""tempo_bpm must be a Python float."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert isinstance(result.tempo_bpm, float)

	def test_beat_times_is_tuple_of_floats (self) -> None:
		"""beat_times must be a tuple of floats."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert isinstance(result.beat_times, tuple)
		assert all(isinstance(t, float) for t in result.beat_times)

	def test_pulse_peak_times_is_tuple_of_floats (self) -> None:
		"""pulse_peak_times must be a tuple of floats."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert isinstance(result.pulse_peak_times, tuple)
		assert all(isinstance(t, float) for t in result.pulse_peak_times)

	def test_pulse_curve_is_1d_float32 (self) -> None:
		"""pulse_curve must be a 1-D float32 numpy array."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert result.pulse_curve.ndim == 1
		assert result.pulse_curve.dtype == numpy.float32

	# ------------------------------------------------------------------
	# Click track — should detect a tempo
	# ------------------------------------------------------------------

	def test_click_track_has_nonzero_tempo (self) -> None:
		"""A regular click track should yield a positive BPM estimate."""
		audio = self._click_track(bpm=120.0)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert result.tempo_bpm > 0.0

	def test_click_track_has_beat_times (self) -> None:
		"""A regular click track should produce at least one detected beat."""
		audio = self._click_track(bpm=120.0, duration_seconds=4.0)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert len(result.beat_times) > 0

	def test_click_track_has_pulse_peaks (self) -> None:
		"""A regular click track should produce at least one PLP pulse peak."""
		audio = self._click_track(bpm=120.0, duration_seconds=4.0)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert len(result.pulse_peak_times) > 0

	def test_beat_times_are_within_duration (self) -> None:
		"""All detected beat times should fall within the signal duration."""
		duration = 4.0
		audio = self._click_track(bpm=120.0, duration_seconds=duration)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		for t in result.beat_times:
			assert 0.0 <= t <= duration

	def test_pulse_peak_times_are_within_duration (self) -> None:
		"""All pulse peak times should fall within the signal duration."""
		duration = 4.0
		audio = self._click_track(bpm=120.0, duration_seconds=duration)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		for t in result.pulse_peak_times:
			assert 0.0 <= t <= duration

	# ------------------------------------------------------------------
	# format_rhythm_result
	# ------------------------------------------------------------------

	def test_format_contains_all_labels (self) -> None:
		"""format_rhythm_result() output must include all four labels."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())
		s = subsample.analysis.format_rhythm_result(result)

		assert "tempo=" in s
		assert "beats=" in s
		assert "pulses=" in s
		assert "onsets=" in s

	def test_format_is_single_line (self) -> None:
		"""format_rhythm_result() should return a single line."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())
		s = subsample.analysis.format_rhythm_result(result)

		assert "\n" not in s

	def test_format_empty_result (self) -> None:
		"""format_rhythm_result() on an empty result should not raise."""
		empty = numpy.zeros(0, dtype=numpy.float32)
		result = subsample.analysis.analyze_rhythm(empty, self._params(), self._cfg())
		s = subsample.analysis.format_rhythm_result(result)

		assert "tempo=0.0bpm" in s
		assert "beats=0" in s
		assert "pulses=0" in s
		assert "onsets=0" in s


class TestZeroCrossingRate:

	"""Tests for the zcr metric in AnalysisResult."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _float_sine (self, frequency: float = 440.0) -> numpy.ndarray:
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * frequency * t) * 0.9).astype(numpy.float32)

	def _white_noise (self) -> numpy.ndarray:
		rng = numpy.random.default_rng(seed=7)
		return rng.random(int(0.5 * 44100)).astype(numpy.float32) * 2.0 - 1.0

	def _dc_signal (self) -> numpy.ndarray:
		return numpy.full(int(0.5 * 44100), 0.5, dtype=numpy.float32)

	def test_dc_signal_has_zero_zcr (self) -> None:
		"""A constant (DC) signal never crosses zero — score should be 0."""
		result = subsample.analysis.analyze_mono(self._dc_signal(), self._params())

		assert result.zcr == 0.0

	def test_white_noise_has_high_zcr (self) -> None:
		"""White noise alternates sign frequently — score should be high."""
		result = subsample.analysis.analyze_mono(self._white_noise(), self._params())

		assert result.zcr > 0.5

	def test_noise_higher_zcr_than_sine (self) -> None:
		"""Noise should have higher ZCR than a smooth sine wave."""
		sine = subsample.analysis.analyze_mono(self._float_sine(), self._params())
		noise = subsample.analysis.analyze_mono(self._white_noise(), self._params())

		assert noise.zcr > sine.zcr

	def test_zcr_in_range (self) -> None:
		"""ZCR must always be in [0.0, 1.0]."""
		for audio in [self._float_sine(), self._white_noise(), self._dc_signal()]:
			result = subsample.analysis.analyze_mono(audio, self._params())
			assert 0.0 <= result.zcr <= 1.0


class TestHarmonicRatio:

	"""Tests for the harmonic_ratio metric in AnalysisResult."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _float_sine (self) -> numpy.ndarray:
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * 440 * t) * 0.9).astype(numpy.float32)

	def _percussive_click (self) -> numpy.ndarray:
		"""10 loud samples then silence — highly percussive."""
		n = int(0.5 * 44100)
		audio = numpy.zeros(n, dtype=numpy.float32)
		audio[:10] = 1.0
		return audio

	def test_sine_is_mostly_harmonic (self) -> None:
		"""A sustained sine wave should have a high harmonic ratio."""
		result = subsample.analysis.analyze_mono(self._float_sine(), self._params())

		assert result.harmonic_ratio > 0.5

	def test_harmonic_ratio_in_range (self) -> None:
		"""harmonic_ratio must always be in [0.0, 1.0]."""
		for audio in [self._float_sine(), self._percussive_click()]:
			result = subsample.analysis.analyze_mono(audio, self._params())
			assert 0.0 <= result.harmonic_ratio <= 1.0

	def test_sine_more_harmonic_than_click (self) -> None:
		"""A sustained tone should be more harmonic than a percussive click."""
		sine = subsample.analysis.analyze_mono(self._float_sine(), self._params())
		click = subsample.analysis.analyze_mono(self._percussive_click(), self._params())

		assert sine.harmonic_ratio > click.harmonic_ratio


class TestSpectralContrast:

	"""Tests for the spectral_contrast metric in AnalysisResult."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _float_sine (self) -> numpy.ndarray:
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * 440 * t) * 0.9).astype(numpy.float32)

	def _white_noise (self) -> numpy.ndarray:
		rng = numpy.random.default_rng(seed=11)
		return rng.random(int(0.5 * 44100)).astype(numpy.float32) * 2.0 - 1.0

	def test_spectral_contrast_in_range (self) -> None:
		"""spectral_contrast must always be in [0.0, 1.0]."""
		for audio in [self._float_sine(), self._white_noise()]:
			result = subsample.analysis.analyze_mono(audio, self._params())
			assert 0.0 <= result.spectral_contrast <= 1.0

	def test_sine_has_higher_contrast_than_noise (self) -> None:
		"""A pure tone has sharp spectral peaks; noise has a flat spectrum."""
		sine = subsample.analysis.analyze_mono(self._float_sine(), self._params())
		noise = subsample.analysis.analyze_mono(self._white_noise(), self._params())

		assert sine.spectral_contrast > noise.spectral_contrast


class TestVoicedFraction:

	"""Tests for the voiced_fraction metric in AnalysisResult."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _float_sine (self) -> numpy.ndarray:
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * 440 * t) * 0.9).astype(numpy.float32)

	def _white_noise (self) -> numpy.ndarray:
		rng = numpy.random.default_rng(seed=13)
		return rng.random(int(0.5 * 44100)).astype(numpy.float32) * 2.0 - 1.0

	def test_sine_has_high_voiced_fraction (self) -> None:
		"""A pure 440 Hz tone should be clearly pitched — high voiced fraction."""
		result = subsample.analysis.analyze_mono(self._float_sine(), self._params())

		assert result.voiced_fraction > 0.5

	def test_voiced_fraction_in_range (self) -> None:
		"""voiced_fraction must always be in [0.0, 1.0]."""
		for audio in [self._float_sine(), self._white_noise()]:
			result = subsample.analysis.analyze_mono(audio, self._params())
			assert 0.0 <= result.voiced_fraction <= 1.0

	def test_short_signal_returns_zero (self) -> None:
		"""Signals shorter than pyin's minimum frame requirement should return 0.0.

		pyin requires frame_length > sr / fmin. At 44100 Hz with fmin=65 Hz
		the minimum is ~680 samples. Signals shorter than this cannot be analysed.
		"""
		short = numpy.zeros(10, dtype=numpy.float32)
		result = subsample.analysis.analyze_mono(short, self._params())

		assert result.voiced_fraction == 0.0


class TestAnalyzePitch:

	"""Tests for analyze_pitch() and format_pitch_result()."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _float_sine (self, frequency: float = 440.0) -> numpy.ndarray:
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		return (numpy.sin(2 * numpy.pi * frequency * t) * 0.9).astype(numpy.float32)

	def _white_noise (self) -> numpy.ndarray:
		rng = numpy.random.default_rng(seed=17)
		return rng.random(int(0.5 * 44100)).astype(numpy.float32) * 2.0 - 1.0

	def test_returns_pitch_and_timbre_types (self) -> None:
		"""analyze_pitch() should return a (PitchResult, TimbreResult) tuple."""
		pitch, timbre = subsample.analysis.analyze_pitch(self._float_sine(), self._params())

		assert isinstance(pitch, subsample.analysis.PitchResult)
		assert isinstance(timbre, subsample.analysis.TimbreResult)

	def test_sine_440_detects_pitch_near_440 (self) -> None:
		"""A 440 Hz sine should yield dominant_pitch_hz close to 440."""
		pitch, _ = subsample.analysis.analyze_pitch(self._float_sine(440.0), self._params())

		# pyin has some tolerance; accept within ±10 Hz
		assert abs(pitch.dominant_pitch_hz - 440.0) < 10.0

	def test_sine_dominant_pitch_class_is_a (self) -> None:
		"""A 440 Hz tone is A; dominant_pitch_class should be 9 (A = index 9)."""
		pitch, _ = subsample.analysis.analyze_pitch(self._float_sine(440.0), self._params())

		assert pitch.dominant_pitch_class == 9

	def test_chroma_profile_has_12_elements (self) -> None:
		"""chroma_profile must always have exactly 12 elements (one per pitch class)."""
		pitch, _ = subsample.analysis.analyze_pitch(self._float_sine(), self._params())

		assert len(pitch.chroma_profile) == 12

	def test_mfcc_has_correct_count (self) -> None:
		"""mfcc must have exactly _N_MFCC elements."""
		_, timbre = subsample.analysis.analyze_pitch(self._float_sine(), self._params())

		assert len(timbre.mfcc) == subsample.analysis._N_MFCC

	def test_mfcc_delta_has_correct_count (self) -> None:
		"""mfcc_delta must have exactly _N_MFCC elements."""
		_, timbre = subsample.analysis.analyze_pitch(self._float_sine(), self._params())

		assert len(timbre.mfcc_delta) == subsample.analysis._N_MFCC

	def test_mfcc_onset_has_correct_count (self) -> None:
		"""mfcc_onset must have exactly _N_MFCC elements."""
		_, timbre = subsample.analysis.analyze_pitch(self._float_sine(), self._params())

		assert len(timbre.mfcc_onset) == subsample.analysis._N_MFCC

	def test_mfcc_delta_differs_from_mfcc (self) -> None:
		"""mfcc_delta should be meaningfully different from the plain mfcc vector."""
		_, timbre = subsample.analysis.analyze_pitch(self._float_sine(), self._params())

		# Delta MFCCs encode temporal change, not absolute timbre — they will
		# differ from the mean MFCCs (though both are small for a steady sine)
		assert timbre.mfcc_delta != timbre.mfcc

	def test_mfcc_onset_differs_from_mfcc (self) -> None:
		"""mfcc_onset should differ from plain mfcc for signals with a clear attack."""
		# Use a decaying tone so that onset and tail have distinct timbres
		n = int(0.5 * 44100)
		t = numpy.arange(n) / 44100
		envelope = numpy.exp(-5 * t).astype(numpy.float32)
		decaying = (numpy.sin(2 * numpy.pi * 440 * t) * 0.9 * envelope).astype(numpy.float32)

		_, timbre = subsample.analysis.analyze_pitch(decaying, self._params())

		# Onset-weighted vector weights attack frames more than the mean does
		assert timbre.mfcc_onset != timbre.mfcc

	def test_empty_array_returns_zero_pitch (self) -> None:
		"""An empty array should return zero-filled PitchResult and TimbreResult."""
		empty = numpy.zeros(0, dtype=numpy.float32)
		pitch, timbre = subsample.analysis.analyze_pitch(empty, self._params())

		assert pitch.dominant_pitch_hz == 0.0
		assert pitch.pitch_confidence == 0.0
		assert pitch.dominant_pitch_class == -1
		assert len(pitch.chroma_profile) == 12
		assert len(timbre.mfcc) == subsample.analysis._N_MFCC
		assert len(timbre.mfcc_delta) == subsample.analysis._N_MFCC
		assert len(timbre.mfcc_onset) == subsample.analysis._N_MFCC

	def test_noise_has_low_confidence (self) -> None:
		"""White noise has no clear pitch — confidence should be very low.

		pyin may occasionally detect stray voiced frames in noise (false positives
		at very low confidence), so we cannot assert dominant_pitch_hz == 0.0.
		We instead assert that pitch_confidence is below a meaningful threshold.
		"""
		pitch, _ = subsample.analysis.analyze_pitch(self._white_noise(), self._params())

		assert pitch.pitch_confidence < 0.1

	def test_stable_sine_has_low_pitch_stability (self) -> None:
		"""A sustained constant-frequency sine wave should have near-zero pitch_stability."""
		pitch, _ = subsample.analysis.analyze_pitch(self._float_sine(440.0), self._params())

		# A perfectly stable tone; allow small numerical variation from pyin
		assert pitch.pitch_stability < 0.5

	def test_swept_sine_has_high_pitch_stability (self) -> None:
		"""A linearly swept sine (440 Hz → 880 Hz) should have high pitch_stability."""
		sample_rate = 44100
		duration = 1.0
		t = numpy.linspace(0, duration, int(sample_rate * duration), dtype=numpy.float32)
		# Instantaneous frequency sweeps from 440 to 880 Hz over the duration
		phase = 2.0 * numpy.pi * (440.0 * t + 220.0 * t ** 2 / duration)
		swept = numpy.sin(phase).astype(numpy.float32)

		pitch, _ = subsample.analysis.analyze_pitch(swept, self._params())

		# A full octave sweep covers 12 semitones; std should be well above 0.5
		assert pitch.pitch_stability > 0.5

	def test_empty_audio_has_zero_pitch_stability (self) -> None:
		"""Silence / no voiced frames should yield pitch_stability == 0.0."""
		pitch, _ = subsample.analysis.analyze_pitch(
			numpy.zeros(0, dtype=numpy.float32), self._params()
		)

		assert pitch.pitch_stability == 0.0


class TestOnsetDetection:

	"""Tests for onset_times and onset_count in RhythmResult."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _cfg (self) -> subsample.config.AnalysisConfig:
		return subsample.config.AnalysisConfig()

	def _click_track (self, bpm: float = 120.0, duration_seconds: float = 2.0) -> numpy.ndarray:
		n = int(duration_seconds * 44100)
		audio = numpy.zeros(n, dtype=numpy.float32)
		beat_interval = int(44100 * 60.0 / bpm)
		for i in range(0, n, beat_interval):
			audio[i] = 1.0
		return audio

	def test_silence_has_no_onsets (self) -> None:
		"""A silent signal should produce no detected onsets."""
		silence = numpy.zeros(int(2.0 * 44100), dtype=numpy.float32)
		result = subsample.analysis.analyze_rhythm(silence, self._params(), self._cfg())

		assert result.onset_count == 0
		assert result.onset_times == ()

	def test_click_track_detects_onsets (self) -> None:
		"""A regular click track should produce at least one onset."""
		audio = self._click_track(bpm=120.0, duration_seconds=2.0)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert result.onset_count > 0

	def test_onset_count_matches_len (self) -> None:
		"""onset_count must equal len(onset_times)."""
		audio = self._click_track(bpm=120.0, duration_seconds=2.0)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		assert result.onset_count == len(result.onset_times)

	def test_onset_times_within_duration (self) -> None:
		"""All onset times must fall within the signal duration."""
		duration = 2.0
		audio = self._click_track(bpm=120.0, duration_seconds=duration)
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())

		for t in result.onset_times:
			assert 0.0 <= t <= duration

	def test_empty_array_has_no_onsets (self) -> None:
		"""A too-short array should return empty onset fields from the early guard."""
		empty = numpy.zeros(0, dtype=numpy.float32)
		result = subsample.analysis.analyze_rhythm(empty, self._params(), self._cfg())

		assert result.onset_times == ()
		assert result.onset_count == 0


class TestFormatPitchResult:

	"""Tests for format_pitch_result()."""

	def _pitched_result (self) -> subsample.analysis.PitchResult:
		return subsample.analysis.PitchResult(
			dominant_pitch_hz=440.0,
			pitch_confidence=0.92,
			chroma_profile=tuple(0.0 for _ in range(12)),
			dominant_pitch_class=9,
			pitch_stability=0.12,
			voiced_frame_count=8,
		)

	def _unpitched_result (self) -> subsample.analysis.PitchResult:
		return subsample.analysis.PitchResult(
			dominant_pitch_hz=0.0,
			pitch_confidence=0.0,
			chroma_profile=tuple(0.0 for _ in range(12)),
			dominant_pitch_class=-1,
			pitch_stability=0.0,
			voiced_frame_count=0,
		)

	def test_contains_all_labels (self) -> None:
		"""Output must include all field labels."""
		s = subsample.analysis.format_pitch_result(self._pitched_result())

		assert "pitch=" in s
		assert "pitch_conf=" in s
		assert "chroma=" in s

	def test_pitched_shows_frequency (self) -> None:
		"""A pitched result should show the frequency in Hz."""
		s = subsample.analysis.format_pitch_result(self._pitched_result())

		assert "440.0Hz" in s

	def test_pitched_shows_pitch_class_name (self) -> None:
		"""dominant_pitch_class=9 should format as 'A'."""
		s = subsample.analysis.format_pitch_result(self._pitched_result())

		assert "chroma=A" in s

	def test_unpitched_shows_none (self) -> None:
		"""An unpitched result should show 'none' for pitch and chroma."""
		s = subsample.analysis.format_pitch_result(self._unpitched_result())

		assert "pitch=none" in s
		assert "chroma=none" in s

	def test_is_single_line (self) -> None:
		"""Output should be a single line with no newlines."""
		s = subsample.analysis.format_pitch_result(self._pitched_result())

		assert "\n" not in s


# ---------------------------------------------------------------------------
# TestComputeLevel
# ---------------------------------------------------------------------------

class TestComputeLevel:

	"""Tests for compute_level()."""

	def test_empty_array_returns_zeros (self) -> None:
		"""An empty signal should return peak=0.0 and rms=0.0."""
		mono = numpy.array([], dtype=numpy.float32)
		result = subsample.analysis.compute_level(mono)

		assert result.peak == 0.0
		assert result.rms == 0.0

	def test_silence_returns_zeros (self) -> None:
		"""An all-zero signal should return peak=0.0 and rms=0.0."""
		mono = numpy.zeros(1024, dtype=numpy.float32)
		result = subsample.analysis.compute_level(mono)

		assert result.peak == 0.0
		assert result.rms == 0.0

	def test_full_scale_sine_peak (self) -> None:
		"""A full-scale sine wave should have peak close to 1.0."""
		n = 44100
		t = numpy.linspace(0.0, 1.0, n, endpoint=False)
		mono = numpy.sin(2 * numpy.pi * 440 * t).astype(numpy.float32)

		result = subsample.analysis.compute_level(mono)

		assert abs(result.peak - 1.0) < 0.01

	def test_full_scale_sine_rms (self) -> None:
		"""A full-scale sine wave has theoretical RMS of 1/sqrt(2) ≈ 0.7071."""
		n = 44100
		t = numpy.linspace(0.0, 1.0, n, endpoint=False)
		mono = numpy.sin(2 * numpy.pi * 440 * t).astype(numpy.float32)

		result = subsample.analysis.compute_level(mono)

		assert abs(result.rms - (1.0 / numpy.sqrt(2))) < 0.005

	def test_half_amplitude_sine (self) -> None:
		"""Halving the amplitude should halve both peak and RMS."""
		n = 44100
		t = numpy.linspace(0.0, 1.0, n, endpoint=False)
		full = numpy.sin(2 * numpy.pi * 440 * t).astype(numpy.float32)
		half = (full * 0.5).astype(numpy.float32)

		r_full = subsample.analysis.compute_level(full)
		r_half = subsample.analysis.compute_level(half)

		assert abs(r_half.peak - r_full.peak * 0.5) < 0.01
		assert abs(r_half.rms  - r_full.rms  * 0.5) < 0.005

	def test_peak_is_in_unit_range (self) -> None:
		"""Peak must be in [0.0, 1.0] for a normalised signal."""
		rng = numpy.random.default_rng(seed=0)
		mono = rng.uniform(-1.0, 1.0, 4096).astype(numpy.float32)

		result = subsample.analysis.compute_level(mono)

		assert 0.0 <= result.peak <= 1.0

	def test_rms_never_exceeds_peak (self) -> None:
		"""RMS can never exceed the peak amplitude."""
		rng = numpy.random.default_rng(seed=1)
		mono = rng.uniform(-1.0, 1.0, 4096).astype(numpy.float32)

		result = subsample.analysis.compute_level(mono)

		assert result.rms <= result.peak + 1e-6


# ---------------------------------------------------------------------------
# TestFormatLevelResult
# ---------------------------------------------------------------------------

class TestFormatLevelResult:

	"""Tests for format_level_result()."""

	def test_contains_peak_and_rms_labels (self) -> None:
		"""Output should contain 'peak=' and 'rms=' labels."""
		result = subsample.analysis.LevelResult(peak=0.5, rms=0.25)
		s = subsample.analysis.format_level_result(result)

		assert "peak=" in s
		assert "rms=" in s

	def test_contains_dbfs (self) -> None:
		"""Output should contain dBFS values in parentheses."""
		result = subsample.analysis.LevelResult(peak=0.5, rms=0.25)
		s = subsample.analysis.format_level_result(result)

		assert "dBFS" in s

	def test_silence_shows_inf (self) -> None:
		"""A zero-amplitude signal should show '-inf' for dBFS."""
		result = subsample.analysis.LevelResult(peak=0.0, rms=0.0)
		s = subsample.analysis.format_level_result(result)

		assert "-inf" in s

	def test_is_single_line (self) -> None:
		"""Output should be a single line with no newlines."""
		result = subsample.analysis.LevelResult(peak=0.8, rms=0.3)
		s = subsample.analysis.format_level_result(result)

		assert "\n" not in s


# ---------------------------------------------------------------------------
# TestIsKeyboardCandidate
# ---------------------------------------------------------------------------

class TestHasStablePitch:

	"""Tests for has_stable_pitch()."""

	def _spectral (self, voiced_fraction: float = 0.8, harmonic_ratio: float = 0.7) -> subsample.analysis.AnalysisResult:
		return subsample.analysis.AnalysisResult(
			spectral_flatness=0.0, attack=0.5, release=0.5,
			spectral_centroid=0.5, spectral_bandwidth=0.3,
			zcr=0.1, harmonic_ratio=harmonic_ratio,
			spectral_contrast=0.5, voiced_fraction=voiced_fraction,
			log_attack_time=0.5, spectral_flux=0.2,
		)

	def _pitch (
		self,
		hz: float = 440.0,
		stability: float = 0.1,
		confidence: float = 0.8,
		voiced_frame_count: int = 10,
	) -> subsample.analysis.PitchResult:
		return subsample.analysis.PitchResult(
			dominant_pitch_hz=hz,
			pitch_confidence=confidence,
			chroma_profile=tuple(0.0 for _ in range(12)),
			dominant_pitch_class=9,
			pitch_stability=stability,
			voiced_frame_count=voiced_frame_count,
		)

	def test_stable_tonal_sample_passes (self) -> None:
		"""A pitched, stable, confident, tonal sample should pass."""
		assert subsample.analysis.has_stable_pitch(self._spectral(), self._pitch(), 0.5) is True

	def test_no_pitch_detected_fails (self) -> None:
		"""dominant_pitch_hz == 0 means no pitch was found."""
		assert subsample.analysis.has_stable_pitch(self._spectral(), self._pitch(hz=0.0), 0.5) is False

	def test_low_voiced_fraction_fails (self) -> None:
		"""Less than half the frames pitched — not suitable."""
		assert subsample.analysis.has_stable_pitch(self._spectral(voiced_fraction=0.3), self._pitch(), 0.5) is False

	def test_unstable_pitch_fails (self) -> None:
		"""pitch_stability >= 0.5 semitones (vibrato / bend) should be excluded."""
		assert subsample.analysis.has_stable_pitch(self._spectral(), self._pitch(stability=1.2), 0.5) is False

	def test_percussive_sample_fails (self) -> None:
		"""Low harmonic_ratio means it's percussive rather than tonal."""
		assert subsample.analysis.has_stable_pitch(self._spectral(harmonic_ratio=0.2), self._pitch(), 0.5) is False

	def test_low_pitch_confidence_fails (self) -> None:
		"""Confidence <= 0.5 means pyin fell back to its fmin floor, not a real pitch."""
		assert subsample.analysis.has_stable_pitch(self._spectral(), self._pitch(confidence=0.3), 0.5) is False

	def test_too_few_voiced_frames_fails (self) -> None:
		"""Fewer than 5 voiced frames — not enough data to trust stability."""
		assert subsample.analysis.has_stable_pitch(self._spectral(), self._pitch(voiced_frame_count=3), 0.5) is False

	def test_short_duration_fails (self) -> None:
		"""Duration < 0.1 s — too short to be a useful keyboard sample."""
		assert subsample.analysis.has_stable_pitch(self._spectral(), self._pitch(), 0.05) is False


# ---------------------------------------------------------------------------
# TestAnalyzeBandEnergy
# ---------------------------------------------------------------------------

class TestAnalyzeBandEnergy:

	"""Tests for analyze_band_energy() and format_band_energy_result()."""

	def _params (self) -> subsample.analysis.AnalysisParams:
		return subsample.analysis.compute_params(44100)

	def _sine (self, frequency: float, duration: float = 1.0) -> numpy.ndarray:
		"""Return a float32 mono sine wave at the given frequency."""
		n = int(duration * 44100)
		t = numpy.arange(n, dtype=numpy.float32) / 44100
		return (numpy.sin(2 * numpy.pi * frequency * t) * 0.9).astype(numpy.float32)

	def _white_noise (self, duration: float = 1.0) -> numpy.ndarray:
		rng = numpy.random.default_rng(seed=42)
		n = int(duration * 44100)
		return rng.uniform(-1.0, 1.0, n).astype(numpy.float32)

	def test_returns_band_energy_result_type (self) -> None:
		"""analyze_band_energy() should return a BandEnergyResult."""
		result = subsample.analysis.analyze_band_energy(self._sine(440.0), self._params())
		assert isinstance(result, subsample.analysis.BandEnergyResult)

	def test_energy_fractions_have_four_elements (self) -> None:
		"""energy_fractions must always have exactly 4 elements."""
		result = subsample.analysis.analyze_band_energy(self._sine(440.0), self._params())
		assert len(result.energy_fractions) == 4

	def test_decay_rates_have_four_elements (self) -> None:
		"""decay_rates must always have exactly 4 elements."""
		result = subsample.analysis.analyze_band_energy(self._sine(440.0), self._params())
		assert len(result.decay_rates) == 4

	def test_fractions_sum_to_approximately_one (self) -> None:
		"""Energy fractions across the 4 bands should sum close to 1.0.

		The band definitions top out at 20 kHz; the remaining energy (Nyquist
		gap 20-22.05 kHz) is excluded, so the sum may fall slightly below 1.0.
		"""
		result = subsample.analysis.analyze_band_energy(self._white_noise(), self._params())
		total = sum(result.energy_fractions)
		assert 0.8 <= total <= 1.0

	def test_all_values_in_unit_range (self) -> None:
		"""All fractions and decay rates must be in [0.0, 1.0]."""
		for signal in [self._sine(440.0), self._white_noise()]:
			result = subsample.analysis.analyze_band_energy(signal, self._params())
			for v in result.energy_fractions:
				assert 0.0 <= v <= 1.0
			for v in result.decay_rates:
				assert 0.0 <= v <= 1.0

	def test_empty_array_returns_zeros (self) -> None:
		"""An empty array should return all-zero result without error."""
		empty = numpy.zeros(0, dtype=numpy.float32)
		result = subsample.analysis.analyze_band_energy(empty, self._params())
		assert all(v == 0.0 for v in result.energy_fractions)
		assert all(v == 0.0 for v in result.decay_rates)

	def test_sub_bass_sine_dominates_first_band (self) -> None:
		"""A 100 Hz sine should have most of its energy in the sub-bass band (index 0)."""
		result = subsample.analysis.analyze_band_energy(self._sine(100.0), self._params())
		# Sub-bass (20-250 Hz) should dominate
		assert result.energy_fractions[0] > result.energy_fractions[2]
		assert result.energy_fractions[0] > result.energy_fractions[3]

	def test_high_freq_sine_dominates_upper_bands (self) -> None:
		"""A 10 kHz sine should have most of its energy in the upper bands."""
		result = subsample.analysis.analyze_band_energy(self._sine(10000.0), self._params())
		# Upper bands (high-mid or presence) should dominate
		upper_energy = result.energy_fractions[2] + result.energy_fractions[3]
		lower_energy = result.energy_fractions[0] + result.energy_fractions[1]
		assert upper_energy > lower_energy

	def test_short_signal_does_not_crash (self) -> None:
		"""A very short signal (< n_fft) should return a valid result without error."""
		short = numpy.zeros(100, dtype=numpy.float32)
		short[:10] = 0.5
		result = subsample.analysis.analyze_band_energy(short, self._params())
		assert isinstance(result, subsample.analysis.BandEnergyResult)

	def test_analyze_all_returns_six_tuple (self) -> None:
		"""analyze_all() should return a 6-tuple including BandEnergyResult."""
		import subsample.config
		# analyze_all expects a pre-converted float32 mono array
		mono = numpy.zeros(22050, dtype=numpy.float32)
		params = self._params()
		cfg = subsample.config.AnalysisConfig()
		result = subsample.analysis.analyze_all(mono, params, rhythm_cfg=cfg)
		assert len(result) == 6
		assert isinstance(result[5], subsample.analysis.BandEnergyResult)

	def test_format_band_energy_result_is_single_line (self) -> None:
		"""format_band_energy_result() should return a single line."""
		result = subsample.analysis.BandEnergyResult(
			energy_fractions = (0.4, 0.3, 0.2, 0.1),
			decay_rates      = (0.8, 0.5, 0.3, 0.1),
		)
		s = subsample.analysis.format_band_energy_result(result)
		assert "\n" not in s
		assert len(s) > 0
