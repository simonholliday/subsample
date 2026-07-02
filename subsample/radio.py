"""Radio modulation/demodulation DSP — pure functions for the `radio`,
`freqshift` and `wobble` process effects.

This module is deliberately free of any dependency on the transform pipeline
or player (transform.py imports it, never the reverse).  Every core here was
researched against real radio engineering, rendered on real audio, measured,
and adversarially audited for authenticity (see the project design notes); the
algorithms are genuine radio physics, idealised only where a musical effect
benefits.  Provenance note: the demodulator cores (FM polar discriminator, AM
envelope, Weaver/Hilbert SSB) follow the same standard maths as the sibling
SDR scanner project (`substation`, AGPL, same author), but are reimplemented
fresh here for the offline whole-buffer real-audio domain — the scanner is
demod-only on complex IQ at RF rates with cross-block state and shares no code.

Authenticity choices worth knowing:
  * AM uses a real diode-style detector (rectify + low-pass), NOT the idealised
    abs(hilbert), so cheap-AM character (rectification, fade distortion) is real.
  * FM is narrowband; de-emphasis is 750 us (land-mobile NFM), not 50/75 us.
  * Channel impairments (hiss/static/fade) are injected PRE-demodulation so the
    detector shapes them the way a real receiver does — the un-fakeable payoff.
  * AGC couples weak-signal noise to fading so a fading station swims in rising
    hiss rather than merely getting quieter.

Determinism: all randomness draws from numpy.random.default_rng seeded by a
fixed module constant offset by an integer per component, so a given spec
re-renders byte-identically after cache eviction (the disk cache depends on it).
"""
import logging
import typing

import librosa
import numpy
import scipy.ndimage
import scipy.signal


_log = logging.getLogger(__name__)

# Fixed RNG base seed (cf. transform._DITHER_SEED).  Per-component seeds are
# derived by integer offset so independent-vs-shared noise fields are stable.
_RADIO_SEED: typing.Final[int] = 0x2AD10

_OVERSAMPLE:  typing.Final[int]   = 4         # internal oversample factor
_CARRIER_HZ:  typing.Final[float] = 24000.0   # carrier at the OVERSAMPLED rate
_AM_INDEX:    typing.Final[float] = 0.8       # AM modulation index
_FM_DEV_HZ:   typing.Final[float] = 2500.0    # NFM peak deviation
_DEEMPH_TAU:  typing.Final[float] = 750e-6    # NFM de-emphasis time constant
_CRACKLE_TAU: typing.Final[float] = 0.9e-3    # sferic IF-ring decay

# Valid mode/demod/stereo enum strings are enforced at MIDI-map parse time in
# query.py's radio validation block; by the time render_radio runs they are
# trusted (an unknown demod falls back to the AM detector by construction).


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _t (n: int, sr: float) -> numpy.ndarray:
	return numpy.arange(n, dtype=numpy.float64) / sr


def _analytic (x: numpy.ndarray) -> numpy.ndarray:
	# ⓒ Buffers shorter than ~10 frames make the zero-phase filters below
	# raise (sosfiltfilt padlen).  That is out of contract by design: no
	# capture or pipeline path produces sub-10-frame audio, and guarding
	# every helper would add noise to the hot DSP for an unreachable case.
	return scipy.signal.hilbert(numpy.asarray(x, dtype=numpy.float64))


def _resample (x: numpy.ndarray, orig_sr: float, target_sr: float) -> numpy.ndarray:
	"""1-D resample via the same high-quality SoX path the pipeline uses."""
	if int(orig_sr) == int(target_sr):
		return numpy.asarray(x, dtype=numpy.float64)
	return numpy.asarray(
		librosa.resample(
			numpy.asarray(x, dtype=numpy.float64),
			orig_sr=int(orig_sr),
			target_sr=int(target_sr),
			res_type="soxr_vhq",
		),
		dtype=numpy.float64,
	)


def _fit_length (x: numpy.ndarray, n: int) -> numpy.ndarray:
	"""Pad/trim to exactly n samples (resampling and diff() drift by ±1)."""
	if len(x) == n:
		return x
	if len(x) > n:
		return x[:n]
	return numpy.concatenate([x, numpy.zeros(n - len(x), dtype=x.dtype)])


# ---------------------------------------------------------------------------
# Modulators / demodulators (validated cores)
# ---------------------------------------------------------------------------

def am_mod (msg: numpy.ndarray, fc: float, sr: float, index: float = _AM_INDEX) -> numpy.ndarray:
	"""Double-sideband full-carrier AM: (1 + m*x)*cos(2*pi*fc*t).  msg already
	normalised to [-1, 1] by the caller (shared norm preserves L/R balance)."""
	return (1.0 + index * msg) * numpy.cos(2.0 * numpy.pi * fc * _t(len(msg), sr))


def am_diode_demod (s: numpy.ndarray, sr: float, fcut: float = 5000.0) -> numpy.ndarray:
	"""Diode envelope detector: rectify, then low-pass (1/fc << 1/fcut << 1/fm).
	The authentic cheap-AM detector — rectification + carrier ripple + fade
	distortion — NOT the idealised abs(hilbert) magnitude."""
	sos = scipy.signal.butter(4, fcut, btype="low", fs=sr, output="sos")
	env = scipy.signal.sosfiltfilt(sos, numpy.abs(numpy.asarray(s, dtype=numpy.float64)))
	return env - numpy.mean(env)


def fm_mod (msg: numpy.ndarray, fc: float, sr: float, dev: float = _FM_DEV_HZ) -> numpy.ndarray:
	"""Narrowband FM: cos(2*pi*fc*t + 2*pi*dev*integral(msg))."""
	phase = 2.0 * numpy.pi * fc * _t(len(msg), sr) + 2.0 * numpy.pi * dev * numpy.cumsum(msg) / sr
	return numpy.cos(phase)


def fm_discriminator (s: numpy.ndarray) -> numpy.ndarray:
	"""Ideal polar FM discriminator: instantaneous frequency from the analytic
	signal's phase derivative.  Channel noise through this produces the genuine
	+6 dB/oct triangular spectrum and the threshold clicks."""
	ph = numpy.unwrap(numpy.angle(_analytic(s)))
	f = numpy.diff(ph)
	f = numpy.concatenate([f[:1], f])      # pad first sample to preserve length
	return numpy.asarray(f - numpy.mean(f), dtype=numpy.float64)


def deemphasis (x: numpy.ndarray, sr: float, tau: float = _DEEMPH_TAU) -> numpy.ndarray:
	"""Standard one-pole FM de-emphasis (pole at exp(-1/(tau*sr)), unity DC
	gain).  750 us is the land-mobile NFM constant; applied without TX
	pre-emphasis it is an authentic "lo-fi tilt" that rolls off the
	discriminator's rising hiss and gives the dull two-way-radio timbre.
	(A full pre/de-emphasis pair — flat signal, shaped noise — is a deferred
	refinement; the un-fakeable FM character is the discriminator noise itself.)"""
	x = numpy.asarray(x, dtype=numpy.float64)
	p = float(numpy.exp(-1.0 / (tau * sr)))
	return scipy.signal.lfilter([1.0 - p], [1.0, -p], x)


def ssb_mod (msg: numpy.ndarray, fc: float, sr: float, sideband: str = "usb") -> numpy.ndarray:
	"""Phasing-method SSB, suppressed carrier (USB by default)."""
	z = _analytic(msg)
	t = _t(len(msg), sr)
	if sideband == "lsb":
		return numpy.real(numpy.conj(z) * numpy.exp(1j * 2.0 * numpy.pi * fc * t))
	return numpy.real(z * numpy.exp(1j * 2.0 * numpy.pi * fc * t))


def ssb_demod (s: numpy.ndarray, fc: float, sr: float, lp: float = 4000.0) -> numpy.ndarray:
	"""Coherent product detector with a free-running BFO at fc.  A non-zero
	offset in fc is a constant-Hz frequency shift (the mistuned 'Donald Duck')."""
	t = _t(len(s), sr)
	base = numpy.real(_analytic(s) * numpy.exp(-1j * 2.0 * numpy.pi * fc * t))
	sos = scipy.signal.butter(8, lp, btype="low", fs=sr, output="sos")
	return scipy.signal.sosfiltfilt(sos, base)


def freq_shift (x: numpy.ndarray, shift_hz: float, sr: float) -> numpy.ndarray:
	"""Bode / single-sideband frequency shift: adds a constant Hz to every
	component (harmonic ratios break — NOT a pitch shift).  Same physics as a
	mistuned SSB receiver and the Moog/Bode studio shifter."""
	z = _analytic(x)
	return numpy.real(z * numpy.exp(1j * 2.0 * numpy.pi * shift_hz * _t(len(x), sr)))


def freq_shift_lfo (
	x:     numpy.ndarray,
	sr:    float,
	base:  float,
	rate:  float,
	depth: float,
) -> numpy.ndarray:
	"""Frequency shift whose amount drifts: shift(t) = base + depth*sin(2*pi*rate*t).
	Phase is the INTEGRAL of the instantaneous shift, i.e. a true continuous
	oscillator drift (microphonic / BFO warble), not chopped re-shifting."""
	t = _t(len(x), sr)
	shift_inst = base + depth * numpy.sin(2.0 * numpy.pi * rate * t)
	phase = 2.0 * numpy.pi * numpy.cumsum(shift_inst) / sr
	return numpy.real(_analytic(x) * numpy.exp(1j * phase))


# ---------------------------------------------------------------------------
# Channel: band-limit, hiss, static, fade, AGC
# ---------------------------------------------------------------------------

def channel_filter (
	msg:       numpy.ndarray,
	mode:      str,
	bandwidth: typing.Optional[float],
	sr:        float,
) -> numpy.ndarray:
	"""Per-mode transmit/IF band-limit.  am/lw keep the lows (steep low-pass);
	fm/ssb are the band-passed 'comms' band.  `bandwidth` overrides the top."""
	msg = numpy.asarray(msg, dtype=numpy.float64)

	# Bandwidth is validated at MIDI-map parse time and clamped for CC
	# bindings at build time; the floors here are a last-resort guard for
	# direct construction so the filter design can never raise mid-render.
	if mode in ("am", "lw"):
		hi = bandwidth if bandwidth is not None else (4200.0 if mode == "lw" else 4500.0)
		hi = min(max(hi, 1.0), sr / 2.0 - 100.0)
		if mode == "lw":
			sos = scipy.signal.ellip(8, 0.5, 60.0, hi, btype="low", fs=sr, output="sos")
		else:
			sos = scipy.signal.butter(6, hi, btype="low", fs=sr, output="sos")
		return scipy.signal.sosfiltfilt(sos, msg)

	# fm / ssb: band-pass comms band
	lo = 300.0
	hi = bandwidth if bandwidth is not None else (3000.0 if mode == "fm" else 2700.0)
	hi = min(max(hi, lo + 1.0), sr / 2.0 - 100.0)
	sos = scipy.signal.butter(6, [lo, hi], btype="band", fs=sr, output="sos")
	return scipy.signal.sosfiltfilt(sos, msg)


def gaussian_hiss (
	n:           int,
	signal:      float,
	carrier_ref: float,
	rng:         numpy.random.Generator,
) -> numpy.ndarray:
	"""White thermal hiss scaled by a carrier-to-noise ratio derived from the
	`signal` (weak-signal) amount.  signal 0 -> clean, 1 -> buried."""
	cnr_db = 40.0 - 38.0 * float(numpy.clip(signal, 0.0, 1.0))
	noise_std = carrier_ref * (10.0 ** (-cnr_db / 20.0))
	return numpy.asarray(noise_std * rng.standard_normal(n), dtype=numpy.float64)


def crackle (
	n:           int,
	sr:          float,
	intensity:   float,
	carrier_ref: float,
	carrier_hz:  float,
	rng:         numpy.random.Generator,
) -> numpy.ndarray:
	"""Atmospheric static (QRN): Poisson arrivals x heavy-tailed (log-normal)
	amplitude x damped-sinusoid IF ring, with a soft tanh front-end cap so the
	intensity knob stays monotone.  Density-led: rate spans 3..1200 events/s."""
	intensity = float(numpy.clip(intensity, 0.0, 1.0))
	rate = 3.0 * (1200.0 / 3.0) ** intensity
	amp_scale = 1.0 + 2.0 * max(0.0, intensity - 0.7)
	dur = n / sr
	out = numpy.zeros(n, dtype=numpy.float64)
	count = int(rng.poisson(rate * dur))
	if count > 0:
		pos = rng.integers(0, n, size=count)
		mags = numpy.exp(1.2 * rng.standard_normal(count)) * amp_scale
		mags = mags * rng.choice([-1.0, 1.0], size=count)
		klen = max(1, int(5.0 * _CRACKLE_TAU * sr))
		tk = numpy.arange(klen, dtype=numpy.float64) / sr
		kernel = numpy.exp(-tk / _CRACKLE_TAU) * numpy.sin(2.0 * numpy.pi * carrier_hz * tk)
		for p, a in zip(pos, mags):
			end = min(p + klen, n)
			out[p:end] += a * kernel[: end - p]
	out = out * (0.9 * carrier_ref)
	# soft front-end magnitude cap (a real diode cannot pass unbounded amplitude)
	lim = 12.0 * carrier_ref
	return lim * numpy.tanh(out / (lim + 1e-12))


def _slow_lfo (n: int, sr: float, rate_hz: float, rng: numpy.random.Generator) -> numpy.ndarray:
	"""A unit-scaled, band-limited slow random signal (~rate_hz) for fading."""
	white = rng.standard_normal(n)
	sos = scipy.signal.butter(2, rate_hz, btype="low", fs=sr, output="sos")
	slow = scipy.signal.sosfiltfilt(sos, white)
	return numpy.asarray(slow / (numpy.max(numpy.abs(slow)) + 1e-12), dtype=numpy.float64)


def selective_fade (
	s:      numpy.ndarray,
	sr:     float,
	amount: float,
	rng:    numpy.random.Generator,
) -> numpy.ndarray:
	"""Shortwave selective fading: a moving frequency-selective comb (delayed
	copies with slow random phase walks) plus a slow amplitude fade.  Applied
	to the on-air signal pre-demod, so the demodulator misbehaves authentically
	when the carrier notches deeper than the sidebands."""
	amount = float(numpy.clip(amount, 0.0, 1.0))
	z = _analytic(s)
	n = len(s)
	out = z.copy()
	for delay_ms in (0.7, 1.5):
		d = max(1, int(delay_ms * 1e-3 * sr))
		phase = _slow_lfo(n, sr, 0.3, rng) * (amount * 8.0 * numpy.pi)
		delayed = numpy.concatenate([numpy.zeros(d, dtype=complex), z[:-d]])
		out = out + amount * delayed * numpy.exp(1j * phase)
	amp = 1.0 - amount * 0.5 * (0.5 + 0.5 * _slow_lfo(n, sr, 0.2, rng))
	return numpy.real(out * amp)


def agc (x: numpy.ndarray, sr: float, target: float = 0.3, win_s: float = 0.12) -> numpy.ndarray:
	"""Automatic gain control: hold the recovered level roughly constant so the
	noise floor SWELLS as the carrier fades (a fading station swims in rising
	hiss).  Peak-following with a smoothing window (cf. real receiver AGC)."""
	x = numpy.asarray(x, dtype=numpy.float64)
	w = max(1, int(win_s * sr))
	env = numpy.abs(x)
	peak = scipy.ndimage.maximum_filter1d(env, size=w, mode="nearest")
	smooth = scipy.ndimage.uniform_filter1d(peak, size=w, mode="nearest")
	gain = target / (smooth + 1e-3)
	gain = numpy.minimum(gain, 40.0)       # bound the make-up so silence-guard can still catch dead air
	return x * gain


# ---------------------------------------------------------------------------
# Safety chain (the player does NOT DC-block / NaN-scrub before us)
# ---------------------------------------------------------------------------

def _dc_block (x: numpy.ndarray, sr: float) -> numpy.ndarray:
	sos = scipy.signal.butter(2, 10.0, btype="high", fs=sr, output="sos")
	return scipy.signal.sosfiltfilt(sos, numpy.asarray(x, dtype=numpy.float64))


def _soft_ceiling (x: numpy.ndarray, ceil: float = 0.99) -> numpy.ndarray:
	"""Tanh soft ceiling so no combination emits a full-scale scream."""
	peak = float(numpy.max(numpy.abs(x)))
	if peak <= ceil:
		return x
	return ceil * numpy.tanh(x / ceil)


def _max_windowed_rms (energy: numpy.ndarray, sr: float, win_s: float = 0.05) -> float:
	"""RMS of the loudest `win_s` window, from per-sample energy (x^2).

	O(n) via a cumulative sum; buffers shorter than one window fall back to
	the whole-buffer RMS."""
	w = max(1, int(win_s * sr))

	if len(energy) <= w:
		return float(numpy.sqrt(numpy.mean(energy))) if len(energy) else 0.0

	cum = numpy.cumsum(numpy.concatenate([[0.0], energy]))
	return float(numpy.sqrt(numpy.max(cum[w:] - cum[:-w]) / w))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _modulate (msg: numpy.ndarray, mode: str, sr: float) -> numpy.ndarray:
	if mode == "fm":
		return fm_mod(msg, _CARRIER_HZ, sr)
	if mode == "ssb":
		return ssb_mod(msg, _CARRIER_HZ, sr)
	return am_mod(msg, _CARRIER_HZ, sr)      # am / lw


def _demodulate (s: numpy.ndarray, mode: str, demod: str, tune: float, sr: float) -> numpy.ndarray:
	"""Resolve the receive demodulator.  `matched` uses the mode's natural
	detector; an explicit demod ≠ mode is the deliberate 'wrong demodulator'."""
	use = demod
	if use == "matched":
		use = "fm" if mode == "fm" else ("ssb" if mode == "ssb" else "am")
	if use == "fm":
		# Scale the discriminator (radians/sample) back to message level so all
		# demods land at comparable scale — matched ~message, a dead wrong-demod
		# (e.g. AM-of-FM) stays near-zero, which the silence guard then catches.
		f = fm_discriminator(s) * (sr / (2.0 * numpy.pi * _FM_DEV_HZ))
		return deemphasis(f, sr)
	if use == "ssb":
		# Negate so positive `tune` shifts the recovered audio UP, matching
		# freqshift's sign convention (a BFO offset is a constant-Hz shift).
		return ssb_demod(s, _CARRIER_HZ - tune, sr)
	return am_diode_demod(s, sr)             # am


def _receive_one (
	msg:        numpy.ndarray,
	sr:         float,
	mode:       str,
	demod:      str,
	tune:       float,
	signal:     float,
	static:     float,
	fade:       float,
	bandwidth:  typing.Optional[float],
	hiss_seed:  int,
	shared_seed: int,
) -> numpy.ndarray:
	"""Full single-channel receiver: oversample -> band-limit -> modulate ->
	(pre-demod impairments) -> demodulate -> AGC -> decimate.  Returns audio at
	`sr`, length == len(msg)."""
	n = len(msg)
	os_sr = sr * _OVERSAMPLE
	msg_os = _resample(msg, sr, os_sr)
	n_os = len(msg_os)

	band = channel_filter(msg_os, mode, bandwidth, os_sr)
	onair = _modulate(band, mode, os_sr)

	impaired = (signal > 0.0) or (static > 0.0) or (fade > 0.0)
	if impaired:
		carrier_ref = float(numpy.max(numpy.abs(onair))) + 1e-12
		if fade > 0.0:
			onair = selective_fade(onair, os_sr, fade, numpy.random.default_rng(shared_seed + 7))
		if signal > 0.0:
			onair = onair + gaussian_hiss(n_os, signal, carrier_ref, numpy.random.default_rng(hiss_seed))
		if static > 0.0:
			onair = onair + crackle(
				n_os, os_sr, static, carrier_ref, _CARRIER_HZ,
				numpy.random.default_rng(shared_seed),
			)

	rx = _demodulate(onair, mode, demod, tune, os_sr)

	if (signal > 0.0) or (fade > 0.0):
		rx = agc(rx, os_sr)

	out = _resample(rx, os_sr, sr)
	return _fit_length(out, n)


def render_radio (
	audio:       numpy.ndarray,
	sample_rate: int,
	*,
	mode:      str,
	demod:     str,
	tune:      float,
	signal:    float,
	static:    float,
	fade:      float,
	bandwidth: typing.Optional[float],
	stereo:    str,
	mix:       float,
) -> numpy.ndarray:
	"""Composite radio effect on (n_frames, channels) float32.  Returns the same
	shape.  Default `stereo='mono'` collapses to a single authentic receiver and
	copies it to every channel; `stereo='stereo'` runs each channel as its own
	receiver sharing one sky (independent hiss, shared crackle/fade)."""
	audio = numpy.asarray(audio, dtype=numpy.float64)
	n_frames, channels = audio.shape

	if mix <= 0.0:
		return audio.astype(numpy.float32)

	collapse = (stereo != "stereo") or channels == 1
	if collapse:
		messages = [audio.mean(axis=1)]
	else:
		messages = [audio[:, c] for c in range(channels)]

	# Shared normalisation across channels preserves the L/R balance.  A
	# silent message (e.g. out-of-phase stereo collapsed to mono) recovers
	# silence, so blend the dry per mix exactly like the silence guard below.
	peak = max(float(numpy.max(numpy.abs(m))) for m in messages) + 1e-12
	if peak < 1e-9:
		return (audio * numpy.float32(1.0 - mix)).astype(numpy.float32)

	received = [
		_receive_one(
			m / peak, float(sample_rate), mode, demod, tune,
			signal, static, fade, bandwidth,
			hiss_seed=_RADIO_SEED + 10 + ci,    # independent thermal noise per receiver
			shared_seed=_RADIO_SEED + 1,        # one sky: shared crackle + fade
		)
		for ci, m in enumerate(messages)
	]

	if collapse:
		wet = numpy.repeat(received[0][:, numpy.newaxis], channels, axis=1)
	else:
		wet = numpy.stack(received, axis=1)

	# Silence guard: a near-silent wrong-demod must not be level-matched into
	# hash.  The statistic is the LOUDEST 50 ms window's RMS, not whole-buffer
	# RMS — a real hit followed by a long silent tail must not be muted by its
	# own tail (whole-buffer RMS shrinks with the silent fraction; the loudest
	# window does not).  Measured separation: recovered content ≥ ~0.16 in its
	# loudest window while a dead cross-demod (AM-of-FM, FM-of-AM) leaves only
	# isolated edge clicks ≤ ~0.06, so 0.02 splits the classes with margin.
	guard_rms = _max_windowed_rms(numpy.mean(wet ** 2, axis=1), float(sample_rate))
	if guard_rms < 0.02:
		_log.warning(
			"radio: mode=%r demod=%r recovered no signal (loudest-window rms %.2e) — output muted",
			mode, demod, guard_rms,
		)
		return (audio * numpy.float32(1.0 - mix)).astype(numpy.float32)

	return _finish(wet, audio, float(sample_rate), mix)


def render_freqshift (
	audio: numpy.ndarray, sample_rate: int, *, shift_hz: float, mix: float,
) -> numpy.ndarray:
	"""Standalone Bode/SSB frequency shifter (per channel, shared phase)."""
	audio = numpy.asarray(audio, dtype=numpy.float64)
	if mix <= 0.0 or shift_hz == 0.0:
		return audio.astype(numpy.float32)
	wet = numpy.stack(
		[freq_shift(audio[:, c], shift_hz, float(sample_rate)) for c in range(audio.shape[1])],
		axis=1,
	)
	return _finish(wet, audio, float(sample_rate), mix)


def render_wobble (
	audio: numpy.ndarray, sample_rate: int, *, depth: float, rate: float, base: float, mix: float,
) -> numpy.ndarray:
	"""Standalone oscillator-warble (LFO frequency drift), per channel."""
	audio = numpy.asarray(audio, dtype=numpy.float64)
	if mix <= 0.0 or (depth == 0.0 and base == 0.0):
		return audio.astype(numpy.float32)
	wet = numpy.stack(
		[freq_shift_lfo(audio[:, c], float(sample_rate), base, rate, depth) for c in range(audio.shape[1])],
		axis=1,
	)
	return _finish(wet, audio, float(sample_rate), mix)


def _finish (
	wet: numpy.ndarray, dry: numpy.ndarray, sr: float, mix: float,
) -> numpy.ndarray:
	"""Shared tail: NaN-scrub -> per-channel DC-block -> level-match to dry ->
	wet/dry blend -> soft ceiling.  Returns float32 (n_frames, channels)."""
	wet = numpy.nan_to_num(numpy.asarray(wet, dtype=numpy.float64), nan=0.0, posinf=0.0, neginf=0.0)
	for c in range(wet.shape[1]):
		wet[:, c] = _dc_block(wet[:, c], sr)

	# Match the wet peak to the dry peak so velocity/gain downstream behaves.
	dry_peak = float(numpy.max(numpy.abs(dry)))
	wet_peak = float(numpy.max(numpy.abs(wet)))
	if wet_peak > 1e-10 and dry_peak > 1e-10:
		wet = wet * (dry_peak / wet_peak)

	if mix < 1.0:
		wet = numpy.float64(mix) * wet + numpy.float64(1.0 - mix) * dry

	wet = _soft_ceiling(wet)
	return wet.astype(numpy.float32)
