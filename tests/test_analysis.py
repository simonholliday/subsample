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


class TestFormatResult:

	"""Tests for the shared result formatting function."""

	def _result (self) -> subsample.analysis.AnalysisResult:
		return subsample.analysis.AnalysisResult(
			spectral_flatness=0.017,
			attack=0.414,
			release=0.000,
			spectral_centroid=0.728,
			spectral_bandwidth=0.757,
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

	def test_very_short_signal_does_not_crash (self) -> None:
		"""Very short non-empty signals should not crash find_peaks.

		3 samples is shorter than n_fft=2048, so librosa will warn about
		the FFT window being too large — that is expected and suppressed here.
		The important assertion is that no exception is raised.
		"""
		short = numpy.array([1.0, 0.5, 0.2], dtype=numpy.float32)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", UserWarning)
			result = subsample.analysis.analyze_rhythm(short, self._params(), self._cfg())

		assert isinstance(result, subsample.analysis.RhythmResult)
		assert result.pulse_curve.ndim == 1

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
		"""format_rhythm_result() output must include all three labels."""
		audio = self._click_track()
		result = subsample.analysis.analyze_rhythm(audio, self._params(), self._cfg())
		s = subsample.analysis.format_rhythm_result(result)

		assert "tempo=" in s
		assert "beats=" in s
		assert "pulses=" in s

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
