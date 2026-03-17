"""Audio analysis for Subsample.

Computes perceptual metrics on completed recordings before they are written
to disk. Designed to run in the writer thread so the main capture loop is
never delayed by analysis work.

Analysis parameters (FFT window size, hop length) are derived once from the
audio config at startup and reused for every recording.

All metrics are normalised to [0.0, 1.0]:

  spectral_flatness  — 0 = perfectly tonal (sine wave), 1 = pure noise
  attack             — 0 = instant/percussive onset, 1 = very gradual build
  release            — 0 = instant cutoff, 1 = long sustain/decay tail
  spectral_centroid  — 0 = very bassy, 1 = very trebly
  spectral_bandwidth — 0 = narrow/pure tone, 1 = wide/spectrally complex
"""

import dataclasses
import math

import librosa
import numpy


# ---------------------------------------------------------------------------
# Reference constants for log-scale normalisation
# ---------------------------------------------------------------------------

# Attack / release: time range over which scores are spread.
# 1 ms = imperceptibly fast (e.g. a single sample click at 44 kHz).
# 2 s = extremely slow (ambient pad with gradual fade-in/out).
# Log midpoint ≈ 45 ms — roughly where a medium-speed pluck or strum sits.
_ATTACK_RELEASE_MIN_S: float = 0.001   # 1 ms
_ATTACK_RELEASE_MAX_S: float = 2.0     # 2 s

# Spectral frequency range: 20 Hz (lower limit of human hearing) to Nyquist.
# Using 20 Hz as the minimum means a signal centred at 20 Hz scores 0.0.
_FREQ_MIN_HZ: float = 20.0

# Threshold for onset/decay detection: frames where RMS exceeds this fraction
# of the peak RMS are considered "active". 10 % chosen as a reasonable -20 dB
# equivalent for typical audio signals.
_ACTIVE_THRESHOLD_RATIO: float = 0.1


@dataclasses.dataclass(frozen=True)
class AnalysisParams:

	"""FFT parameters derived from the audio config, computed once at startup."""

	n_fft: int
	hop_length: int
	sample_rate: int


@dataclasses.dataclass(frozen=True)
class AnalysisResult:

	"""Metrics computed from a single recording.

	All values are in [0.0, 1.0]. Extend here as new metrics are added;
	the rest of the pipeline receives a single structured object and
	benefits automatically.
	"""

	spectral_flatness: float
	"""Wiener entropy. 0.0 = perfectly tonal (sine), 1.0 = pure noise."""

	attack: float
	"""Time from onset to peak energy, log-mapped to [0, 1].
	0.0 = instant/percussive (≤ 1 ms), 1.0 = very gradual (≥ 2 s)."""

	release: float
	"""Time from peak energy to decay, log-mapped to [0, 1].
	0.0 = instant cutoff (≤ 1 ms), 1.0 = very long tail (≥ 2 s)."""

	spectral_centroid: float
	"""Centre of mass of the frequency spectrum, log-mapped from 20 Hz to Nyquist.
	0.0 = very bassy, 1.0 = very trebly."""

	spectral_bandwidth: float
	"""Spread of frequency content, log-mapped from 20 Hz to Nyquist.
	0.0 = narrow / pure tone, 1.0 = wide / spectrally complex."""


def compute_params (sample_rate: int) -> AnalysisParams:

	"""Derive FFT analysis parameters for a given sample rate.

	Targets a ~46ms analysis window (the audio analysis standard at 44.1 kHz),
	rounded to the nearest power of two for FFT efficiency. This keeps the
	window duration roughly constant regardless of sample rate.

	Args:
		sample_rate: Audio sample rate in Hz.

	Returns:
		AnalysisParams suitable for passing to analyze() or analyze_mono().

	Examples:
		11025 Hz → n_fft=512,  hop=128
		22050 Hz → n_fft=1024, hop=256
		44100 Hz → n_fft=2048, hop=512
		48000 Hz → n_fft=2048, hop=512
		96000 Hz → n_fft=4096, hop=1024
	"""

	if sample_rate <= 0:
		raise ValueError(f"sample_rate must be positive, got {sample_rate}")

	# Reference window: 2048 samples at 44100 Hz ≈ 0.04644 s
	target_seconds = 2048 / 44100
	n_fft = int(2 ** round(math.log2(target_seconds * sample_rate)))
	hop_length = n_fft // 4

	return AnalysisParams(
		n_fft=n_fft,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)


def analyze (
	audio: numpy.ndarray,
	params: AnalysisParams,
	bit_depth: int,
) -> AnalysisResult:

	"""Compute analysis metrics for a single audio recording.

	Converts the integer PCM array to normalised float32, mixes stereo to mono,
	then delegates to analyze_mono(). The original array is not modified.

	Args:
		audio:     Shape (n_frames, channels), dtype int16 or int32.
		           24-bit audio is stored as int32 left-shifted by 8.
		params:    Pre-computed FFT parameters from compute_params().
		bit_depth: Original capture bit depth (16, 24, or 32).

	Returns:
		AnalysisResult with all computed metrics in [0.0, 1.0].
	"""

	if audio.shape[0] == 0:
		return AnalysisResult(
			spectral_flatness=0.0,
			attack=0.0,
			release=0.0,
			spectral_centroid=0.0,
			spectral_bandwidth=0.0,
		)

	mono = _to_mono_float(audio, bit_depth)

	return analyze_mono(mono, params)


def analyze_mono (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> AnalysisResult:

	"""Compute analysis metrics from a pre-normalised float32 mono array.

	Use this entry point when audio is already normalised to [-1, 1] — for
	example when reading a file via soundfile. For integer PCM from the live
	capture pipeline, use analyze() instead, which handles the conversion.

	Args:
		mono:   Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params: Pre-computed FFT parameters from compute_params().

	Returns:
		AnalysisResult with all computed metrics in [0.0, 1.0].
	"""

	if mono.shape[0] == 0:
		return AnalysisResult(
			spectral_flatness=0.0,
			attack=0.0,
			release=0.0,
			spectral_centroid=0.0,
			spectral_bandwidth=0.0,
		)

	flatness = librosa.feature.spectral_flatness(
		y=mono,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	attack, release = _compute_attack_release(mono, params)
	centroid = _compute_spectral_centroid(mono, params)
	bandwidth = _compute_spectral_bandwidth(mono, params)

	return AnalysisResult(
		spectral_flatness=float(numpy.mean(flatness)),
		attack=attack,
		release=release,
		spectral_centroid=centroid,
		spectral_bandwidth=bandwidth,
	)


def format_result (result: AnalysisResult, duration: float) -> str:

	"""Return a single-line human-readable summary of an analysis result.

	Used by both the WAV writer debug log and the analyze_file script so the
	format is defined once and stays consistent.

	Args:
		result:   Computed analysis metrics.
		duration: Recording length in seconds.

	Returns:
		String of the form:
		"duration=X.XXs  flatness=X.XXX  attack=X.XXX  release=X.XXX  centroid=X.XXX  bandwidth=X.XXX"
	"""

	return (
		f"duration={duration:.2f}s"
		f"  flatness={result.spectral_flatness:.3f}"
		f"  attack={result.attack:.3f}"
		f"  release={result.release:.3f}"
		f"  centroid={result.spectral_centroid:.3f}"
		f"  bandwidth={result.spectral_bandwidth:.3f}"
	)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono_float (audio: numpy.ndarray, bit_depth: int) -> numpy.ndarray:

	"""Convert integer PCM to a normalised float32 mono array.

	Normalises to [-1.0, 1.0] using the full-scale range for the bit depth.
	Stereo (or higher) inputs are mixed down by averaging channels.

	Args:
		audio:     Shape (n_frames, channels), integer dtype.
		bit_depth: 16, 24, or 32.

	Returns:
		Shape (n_frames,), dtype float32.
	"""

	# int32 is used internally for both 24-bit (left-shifted) and native 32-bit;
	# the full-scale divisor is the same in both cases.
	divisor: float = 32768.0 if bit_depth == 16 else 2147483648.0

	float_audio = audio.astype(numpy.float32) / divisor

	if float_audio.shape[1] == 1:
		return float_audio[:, 0]

	# Mix stereo (or multi-channel) to mono
	return numpy.mean(float_audio, axis=1, dtype=numpy.float32)  # type: ignore[return-value]


def _log_normalize (value: float, min_ref: float, max_ref: float) -> float:

	"""Map a value to [0.0, 1.0] using a logarithmic scale.

	Values at or below min_ref return 0.0; values at or above max_ref return
	1.0. The midpoint on the log scale is sqrt(min_ref * max_ref).

	Using log scale for time and frequency reflects human perception: we
	distinguish a 1 ms vs 10 ms attack much more clearly than a 1 s vs 1.01 s
	attack; similarly, an octave is always an octave regardless of register.

	Args:
		value:   The raw value to normalise.
		min_ref: Reference minimum (maps to 0.0).
		max_ref: Reference maximum (maps to 1.0).

	Returns:
		Normalised score in [0.0, 1.0].
	"""

	if value <= min_ref:
		return 0.0

	if value >= max_ref:
		return 1.0

	return math.log(value / min_ref) / math.log(max_ref / min_ref)


def _compute_attack_release (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> tuple[float, float]:

	"""Measure the attack and release durations from the RMS energy envelope.

	Both are mapped to [0, 1] using _log_normalize against
	_ATTACK_RELEASE_MIN_S and _ATTACK_RELEASE_MAX_S.

	How it works:
	  1. Compute the frame-by-frame RMS energy envelope.
	  2. Find the peak RMS frame — this is the energy apex.
	  3. Threshold at _ACTIVE_THRESHOLD_RATIO * peak_rms (≈ -20 dB below peak).
	  4. Attack  = time from first above-threshold frame to peak.
	  5. Release = time from peak to last above-threshold frame.
	  Both are converted to seconds via hop_length / sample_rate.

	Attack and release are independent: a sound can have a fast attack AND a
	long release (e.g. piano), or slow attack AND fast release (e.g. fade-in
	with hard cut), because they measure different ends of the peak.

	Args:
		mono:   Float32 audio, shape (n_frames,), normalised to [-1, 1].
		params: FFT params for consistent frame sizing with other metrics.

	Returns:
		(attack_score, release_score), both in [0.0, 1.0].
	"""

	rms = librosa.feature.rms(
		y=mono,
		frame_length=params.n_fft,
		hop_length=params.hop_length,
	)[0]  # librosa returns shape (1, n_frames); [0] gives (n_frames,)

	peak_idx = int(numpy.argmax(rms))
	peak_rms = float(rms[peak_idx])

	# Guard against silence or near-silence: if the peak is essentially zero
	# there's no meaningful attack or release to measure.
	if peak_rms < 1e-8:
		return (0.0, 0.0)

	threshold = peak_rms * _ACTIVE_THRESHOLD_RATIO

	# Frames where energy is above the threshold
	active_frames = numpy.where(rms >= threshold)[0]

	if active_frames.size == 0:
		return (0.0, 0.0)

	# seconds per frame, used to convert frame counts to wall-clock time
	seconds_per_frame = params.hop_length / params.sample_rate

	# Attack: frames between the first active frame and the peak
	first_active = int(active_frames[0])
	attack_seconds = (peak_idx - first_active) * seconds_per_frame

	# Release: frames between the peak and the last active frame
	last_active = int(active_frames[-1])
	release_seconds = (last_active - peak_idx) * seconds_per_frame

	attack_score = _log_normalize(attack_seconds, _ATTACK_RELEASE_MIN_S, _ATTACK_RELEASE_MAX_S)
	release_score = _log_normalize(release_seconds, _ATTACK_RELEASE_MIN_S, _ATTACK_RELEASE_MAX_S)

	return (attack_score, release_score)


def _compute_spectral_centroid (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> float:

	"""Measure the centre of mass of the frequency spectrum (bassy vs trebly).

	Returns a value in [0, 1] via log-frequency normalisation between 20 Hz
	and Nyquist.  Using log scale matches human pitch perception: an octave
	jump is always an octave regardless of register, so the midpoint of the
	scale (0.5) sits at sqrt(20 * nyquist) Hz — roughly the middle of the
	musical range (~1 kHz at 44.1 kHz sample rate).

	Args:
		mono:   Float32 audio, shape (n_frames,).
		params: FFT params.

	Returns:
		Centroid score in [0.0, 1.0]. 0 = bassy, 1 = trebly.
	"""

	centroid = librosa.feature.spectral_centroid(
		y=mono,
		sr=params.sample_rate,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	# centroid has shape (1, n_frames); mean over frames gives a single Hz value
	mean_hz = float(numpy.mean(centroid))

	nyquist = params.sample_rate / 2.0

	return _log_normalize(mean_hz, _FREQ_MIN_HZ, nyquist)


def _compute_spectral_bandwidth (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> float:

	"""Measure how spread out the frequency content is (narrow vs wide).

	Spectral bandwidth is the weighted standard deviation of the spectrum
	around the spectral centroid — wide means energy is spread across many
	frequencies (complex/noisy), narrow means it is concentrated (tonal/pure).

	Normalised with the same log-frequency scale as spectral_centroid so the
	two metrics are directly comparable in magnitude.

	Args:
		mono:   Float32 audio, shape (n_frames,).
		params: FFT params.

	Returns:
		Bandwidth score in [0.0, 1.0]. 0 = narrow/tonal, 1 = wide/complex.
	"""

	bandwidth = librosa.feature.spectral_bandwidth(
		y=mono,
		sr=params.sample_rate,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	# bandwidth has shape (1, n_frames); mean over frames gives a single Hz value
	mean_hz = float(numpy.mean(bandwidth))

	nyquist = params.sample_rate / 2.0

	return _log_normalize(mean_hz, _FREQ_MIN_HZ, nyquist)
