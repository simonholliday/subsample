"""Tests for subsample/radio.py — the radio modulation/demodulation DSP.

These re-assert the signatures that the design investigation measured and
audited, so they cannot silently regress: constant-Hz SSB shift (not a pitch
ratio), faithful matched round-trips, the silence-guard on dead cross-demods,
heavy-tailed atmospheric crackle, the stereo source-sharing model, and
byte-identical re-renders.
"""
import warnings

import numpy
import pytest

import subsample.radio


SR = 44100


def _tone (freq: float, n: int = 22050, amp: float = 0.6, channels: int = 2) -> numpy.ndarray:
	t = numpy.arange(n) / SR
	col = (amp * numpy.sin(2.0 * numpy.pi * freq * t)).astype(numpy.float32)
	return numpy.stack([col] * channels, axis=1)


def _peak_freq (x: numpy.ndarray) -> float:
	w = numpy.hanning(len(x))
	mag = numpy.abs(numpy.fft.rfft(x * w))
	freqs = numpy.fft.rfftfreq(len(x), 1.0 / SR)
	return float(freqs[int(numpy.argmax(mag))])


def _kurtosis (x: numpy.ndarray) -> float:
	x = numpy.asarray(x, dtype=numpy.float64)
	m = numpy.mean(x)
	var = numpy.mean((x - m) ** 2)
	if var < 1e-20:
		return 0.0
	return float(numpy.mean((x - m) ** 4) / var ** 2 - 3.0)


def _render (mode: str, **kw: object) -> numpy.ndarray:
	defaults = dict(demod="matched", tune=0.0, signal=0.0, static=0.0, fade=0.0,
	                bandwidth=None, stereo="mono", mix=1.0)
	defaults.update(kw)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		return subsample.radio.render_radio(_tone(700.0), SR, mode=mode, **defaults)   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shape / dtype / length contract
# ---------------------------------------------------------------------------

class TestContract:

	@pytest.mark.parametrize("mode", ["am", "lw", "fm", "ssb"])
	def test_shape_dtype_length_preserved (self, mode: str) -> None:
		x = _tone(700.0)
		out = _render(mode)
		assert out.shape == x.shape          # frame count AND channel count preserved
		assert out.dtype == numpy.float32
		assert numpy.all(numpy.isfinite(out))

	def test_mono_input_one_channel (self) -> None:
		x = _tone(700.0, channels=1)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			out = subsample.radio.render_radio(
				x, SR, mode="am", demod="matched", tune=0.0, signal=0.0,
				static=0.0, fade=0.0, bandwidth=None, stereo="mono", mix=1.0)
		assert out.shape == x.shape

	def test_freqshift_wobble_shape (self) -> None:
		x = _tone(700.0)
		assert subsample.radio.render_freqshift(x, SR, shift_hz=500.0, mix=1.0).shape == x.shape
		assert subsample.radio.render_wobble(x, SR, depth=6.0, rate=0.3, base=0.0, mix=1.0).shape == x.shape


# ---------------------------------------------------------------------------
# Frequency shift — constant Hz, NOT a pitch ratio (the SSB-mistune signature)
# ---------------------------------------------------------------------------

class TestFreqShift:

	def test_constant_hz_shift (self) -> None:
		x = _tone(1000.0, channels=1)
		out = subsample.radio.freq_shift(x[:, 0].astype(numpy.float64), 250.0, SR)
		assert abs(_peak_freq(out) - 1250.0) < 15.0     # +250 Hz exactly, not *1.25

	def test_additive_not_ratio (self) -> None:
		# A 1000 Hz tone shifted +250 -> 1250 (not 1250 for ratio either, but a
		# 500 Hz tone shifted +250 -> 750, proving constant Hz not a 1.25 ratio).
		lo = subsample.radio.freq_shift(_tone(500.0, channels=1)[:, 0].astype(numpy.float64), 250.0, SR)
		assert abs(_peak_freq(lo) - 750.0) < 15.0

	def test_image_suppressed (self) -> None:
		# Single-sideband: the unwanted mirror image is deeply suppressed.
		x = _tone(1000.0, channels=1)[:, 0].astype(numpy.float64)
		out = subsample.radio.freq_shift(x, 250.0, SR)
		w = numpy.hanning(len(out))
		mag = numpy.abs(numpy.fft.rfft(out * w))
		freqs = numpy.fft.rfftfreq(len(out), 1.0 / SR)
		up = mag[numpy.argmin(numpy.abs(freqs - 1250.0))]
		down = mag[numpy.argmin(numpy.abs(freqs - 750.0))]
		assert up > 50.0 * down


# ---------------------------------------------------------------------------
# Matched round-trips recover; wrong demods behave (musical vs silence-guarded)
# ---------------------------------------------------------------------------

class TestModes:

	@pytest.mark.parametrize("mode", ["am", "lw", "fm", "ssb"])
	def test_matched_recovers (self, mode: str) -> None:
		out = _render(mode)
		assert float(numpy.max(numpy.abs(out))) > 0.01     # not muted — signal recovered

	def test_wrong_demod_fm_to_am_is_silence_guarded (self) -> None:
		# AM-detecting a constant-envelope FM signal recovers nothing; the guard
		# must mute it rather than letting normalisation amplify hash.
		out = _render("fm", demod="am")
		assert float(numpy.max(numpy.abs(out))) < 1e-5

	def test_sparse_burst_in_long_buffer_not_muted (self) -> None:
		"""The silence guard must be duty-cycle-robust: a real 30 ms hit with
		a long silent tail must NOT be muted by its own tail (whole-buffer
		RMS would mute it; the loudest-window statistic must not)."""

		for mode in ("am", "fm"):
			n = 8 * SR
			x = numpy.zeros((n, 2), dtype=numpy.float32)
			t = numpy.arange(int(0.03 * SR)) / SR
			burst = (0.9 * numpy.sin(2.0 * numpy.pi * 700.0 * t)).astype(numpy.float32)
			x[:len(burst), 0] = x[:len(burst), 1] = burst

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				out = subsample.radio.render_radio(
					x, SR, mode=mode, demod="matched", tune=0.0, signal=0.0,
					static=0.0, fade=0.0, bandwidth=None, stereo="mono", mix=1.0)

			assert float(numpy.max(numpy.abs(out))) > 0.1, mode

	def test_dead_cross_demod_muted_even_in_long_buffer (self) -> None:
		"""The guard still catches a genuinely dead cross-demod at the same
		buffer length where real content survives."""

		n = 8 * SR
		t = numpy.arange(n) / SR
		x = numpy.stack([(0.9 * numpy.sin(2.0 * numpy.pi * 700.0 * t)).astype(numpy.float32)] * 2, axis=1)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			out = subsample.radio.render_radio(
				x, SR, mode="fm", demod="am", tune=0.0, signal=0.0,
				static=0.0, fade=0.0, bandwidth=None, stereo="mono", mix=1.0)

		assert float(numpy.max(numpy.abs(out))) < 1e-5

	def test_wrong_demod_fm_to_ssb_is_a_real_signal (self) -> None:
		# The musical wrong-demod (FM through an SSB detector) is a real warble.
		out = _render("fm", demod="ssb")
		assert float(numpy.max(numpy.abs(out))) > 0.01

	def test_ssb_tune_shifts_spectrum (self) -> None:
		# A BFO offset is a constant-Hz translation of the recovered audio.
		base = _render("ssb", tune=0.0)
		tuned = _render("ssb", tune=300.0)
		assert _peak_freq(tuned[:, 0]) > _peak_freq(base[:, 0]) + 100.0


# ---------------------------------------------------------------------------
# Atmospheric crackle — genuinely impulsive (heavy-tailed), not Gaussian
# ---------------------------------------------------------------------------

class TestCrackle:

	def test_heavy_tailed_vs_gaussian (self) -> None:
		rng = numpy.random.default_rng(0)
		crk = subsample.radio.crackle(SR, SR, 0.4, 1.0, subsample.radio._CARRIER_HZ, rng)
		gauss = numpy.random.default_rng(1).standard_normal(SR)
		assert _kurtosis(crk) > 20.0           # spiky
		assert abs(_kurtosis(gauss)) < 1.0     # Gaussian baseline ~0

	def test_intensity_increases_density (self) -> None:
		# More intensity -> more energy (denser arrivals), monotone in the mean.
		def energy (amount: float) -> float:
			vals = []
			for s in range(8):
				rng = numpy.random.default_rng(100 + s)
				vals.append(float(numpy.mean(subsample.radio.crackle(
					SR, SR, amount, 1.0, subsample.radio._CARRIER_HZ, rng) ** 2)))
			return float(numpy.median(vals))
		assert energy(0.6) > energy(0.2)

	def test_sparse_intensity_is_bounded (self) -> None:
		"""At the sparsest setting the front-end cap bounds every impulse.
		(Intensity 0.0 still fires ~3 events/s by design — production gates
		the crackle call on static > 0, so 0.0 is never rendered.)"""
		rng = numpy.random.default_rng(0)
		crk = subsample.radio.crackle(SR, SR, 0.0, 1.0, subsample.radio._CARRIER_HZ, rng)
		assert numpy.all(numpy.isfinite(crk))
		# The tanh front-end cap: |crackle| can never exceed 12x carrier_ref.
		assert float(numpy.max(numpy.abs(crk))) <= 12.0


# ---------------------------------------------------------------------------
# Stereo source-sharing: independent hiss, shared crackle, mono collapse
# ---------------------------------------------------------------------------

class TestStereo:

	def test_hiss_is_independent_per_channel (self) -> None:
		a = subsample.radio.gaussian_hiss(20000, 0.5, 1.0, numpy.random.default_rng(11))
		b = subsample.radio.gaussian_hiss(20000, 0.5, 1.0, numpy.random.default_rng(22))
		assert abs(float(numpy.corrcoef(a, b)[0, 1])) < 0.1

	def test_crackle_is_shared_when_seeded_alike (self) -> None:
		a = subsample.radio.crackle(20000, SR, 0.5, 1.0, subsample.radio._CARRIER_HZ, numpy.random.default_rng(7))
		b = subsample.radio.crackle(20000, SR, 0.5, 1.0, subsample.radio._CARRIER_HZ, numpy.random.default_rng(7))
		assert numpy.array_equal(a, b)

	def test_mono_collapse_duplicates_channels (self) -> None:
		out = _render("am", stereo="mono")
		assert numpy.array_equal(out[:, 0], out[:, 1])

	def test_stereo_keeps_channels_distinct (self) -> None:
		# Different L/R content through independent receivers stays distinct.
		t = numpy.arange(22050) / SR
		x = numpy.stack([
			0.6 * numpy.sin(2.0 * numpy.pi * 700.0 * t),
			0.6 * numpy.sin(2.0 * numpy.pi * 1100.0 * t),
		], axis=1).astype(numpy.float32)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			out = subsample.radio.render_radio(
				x, SR, mode="am", demod="matched", tune=0.0, signal=0.3,
				static=0.0, fade=0.0, bandwidth=None, stereo="stereo", mix=1.0)
		assert not numpy.array_equal(out[:, 0], out[:, 1])


# ---------------------------------------------------------------------------
# Determinism — re-renders must be byte-identical (the disk cache depends on it)
# ---------------------------------------------------------------------------

class TestDeterminism:

	def test_byte_identical_re_render (self) -> None:
		a = _render("am", signal=0.4, static=0.5, fade=0.3, stereo="stereo")
		b = _render("am", signal=0.4, static=0.5, fade=0.3, stereo="stereo")
		assert numpy.array_equal(a, b)

	def test_freqshift_deterministic (self) -> None:
		x = _tone(700.0)
		a = subsample.radio.render_freqshift(x, SR, shift_hz=300.0, mix=1.0)
		b = subsample.radio.render_freqshift(x, SR, shift_hz=300.0, mix=1.0)
		assert numpy.array_equal(a, b)


# ---------------------------------------------------------------------------
# Channel filter — per-mode band
# ---------------------------------------------------------------------------

class TestChannelFilter:

	def test_am_keeps_lows_fm_removes_them (self) -> None:
		t = numpy.arange(22050) / SR
		low = numpy.sin(2.0 * numpy.pi * 120.0 * t)      # below the 300 Hz comms edge
		am = subsample.radio.channel_filter(low, "am", None, SR)
		fm = subsample.radio.channel_filter(low, "fm", None, SR)
		assert float(numpy.std(am)) > 0.5                # AM low-pass keeps 120 Hz
		assert float(numpy.std(fm)) < 0.1                # FM band-pass removes it
