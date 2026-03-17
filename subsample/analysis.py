"""Audio analysis for Subsample.

Computes perceptual metrics and rhythmic properties on completed recordings
before they are written to disk. Designed to run in the writer thread so the
main capture loop is never delayed by analysis work.

Analysis parameters (FFT window size, hop length) are derived once from the
audio config at startup and reused for every recording.

Spectral metrics (AnalysisResult) — normalised to [0.0, 1.0]:

  spectral_flatness  — 0 = perfectly tonal (sine wave), 1 = pure noise
  attack             — 0 = instant/percussive onset, 1 = very gradual build
  release            — 0 = instant cutoff, 1 = long sustain/decay tail
  spectral_centroid  — 0 = very bassy, 1 = very trebly
  spectral_bandwidth — 0 = narrow/pure tone, 1 = wide/spectrally complex

Rhythm metrics (RhythmResult) — raw values, NOT normalised:

  tempo_bpm          — estimated global tempo in BPM (0.0 if undetected)
  beat_times         — beat positions in seconds from beat_track
  pulse_curve        — frame-by-frame rhythmic salience from PLP
  pulse_peak_times   — local maxima of pulse curve in seconds
"""

import dataclasses
import math

import librosa
import numpy
import scipy.signal

import subsample.config


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


@dataclasses.dataclass(frozen=True)
class RhythmResult:

	"""Rhythmic properties detected from a single recording.

	Unlike AnalysisResult, values are NOT normalised to [0, 1]. They represent
	raw temporal and rhythmic data intended for downstream beat-fitting — mapping
	detected pulses onto a known BPM grid.
	"""

	tempo_bpm: float
	"""Estimated global tempo from beat_track, in beats per minute.
	0.0 if no tempo could be detected (e.g. silence or arrhythmic audio)."""

	beat_times: tuple[float, ...]
	"""Beat positions in seconds, estimated by beat_track using dynamic programming.
	Values are quantised to a regular grid at tempo_bpm.
	Empty tuple if no beats were detected."""

	pulse_curve: numpy.ndarray
	"""Frame-by-frame rhythmic salience from the PLP (Predominant Local Pulse) algorithm.
	Shape (n_frames,), values >= 0. Peaks mark rhythmically salient moments.
	Frame timing: frame i corresponds to i * hop_length / sample_rate seconds.
	Unlike beat_times, this reflects locally varying tempo and is not grid-quantised."""

	pulse_peak_times: tuple[float, ...]
	"""Local maxima of the pulse curve, in seconds.
	Candidate beat locations that do not assume constant tempo.
	Useful when beat_times is empty or the detected grid does not fit the signal well."""


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

	mono = to_mono_float(audio, bit_depth)

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

	# Clamp n_fft to the signal length for short recordings. librosa zero-pads
	# when n_fft > len(signal), but emits a UserWarning. Clamping avoids the
	# warning without sacrificing accuracy: a 606-sample signal has no real
	# spectral information beyond 606 bins regardless of the window size.
	# Also clamp hop_length so it never exceeds n_fft.
	effective_n_fft = min(params.n_fft, len(mono))
	effective_hop = min(params.hop_length, effective_n_fft)
	params = dataclasses.replace(params, n_fft=effective_n_fft, hop_length=effective_hop)

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


def analyze_rhythm (
	mono: numpy.ndarray,
	params: AnalysisParams,
	rhythm_cfg: subsample.config.AnalysisConfig,
) -> RhythmResult:

	"""Detect rhythmic properties from a pre-normalised float32 mono array.

	Runs two complementary algorithms:
	  1. beat_track — dynamic programming to find a consistent beat grid and
	     global tempo. Best when the rhythm is steady (e.g. a clock tick).
	  2. plp (Predominant Local Pulse) — analyses locally dominant periodicities
	     without assuming constant tempo. Better for drifting or irregular rhythms.

	The results are intended for downstream beat-fitting: beat_times provides
	a grid, while pulse_peak_times provides raw candidate locations.

	Args:
		mono:       Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params:     Pre-computed FFT parameters from compute_params().
		rhythm_cfg: Configurable tempo priors from AnalysisConfig.

	Returns:
		RhythmResult with tempo, beat positions, and pulse curve data.
	"""

	if mono.shape[0] == 0:
		return RhythmResult(
			tempo_bpm=0.0,
			beat_times=(),
			pulse_curve=numpy.zeros(0, dtype=numpy.float32),
			pulse_peak_times=(),
		)

	# --- beat_track: global tempo estimation + beat grid ---
	# Returns tempo as a numpy scalar and beat positions in the units specified.
	# start_bpm biases (but does not constrain) the tempo search.
	tempo_raw, beat_frames_raw = librosa.beat.beat_track(
		y=mono,
		sr=params.sample_rate,
		hop_length=params.hop_length,
		start_bpm=rhythm_cfg.start_bpm,
		units='frames',
	)

	# tempo_raw may be a 1-D array of shape (1,) in newer librosa versions
	tempo_bpm = float(numpy.atleast_1d(tempo_raw)[0])

	# Convert beat frame indices to seconds for downstream use
	beat_times: tuple[float, ...] = tuple(
		float(t) for t in librosa.frames_to_time(
			beat_frames_raw,
			sr=params.sample_rate,
			hop_length=params.hop_length,
		)
	)

	# --- plp: frame-by-frame pulse salience, handles varying tempo ---
	# win_length controls the autocorrelation window in frames (librosa default 384).
	# It must not exceed the number of onset envelope frames in the signal, otherwise
	# librosa warns and produces degenerate output. For short recordings we clamp it
	# down — at the cost of coarser tempo resolution, but without errors or warnings.
	n_signal_frames = max(1, len(mono) // params.hop_length)
	plp_win_length = min(384, n_signal_frames)

	# Returns shape (1, n_frames) or (n_frames,) depending on librosa version.
	pulse_raw = librosa.beat.plp(
		y=mono,
		sr=params.sample_rate,
		hop_length=params.hop_length,
		tempo_min=rhythm_cfg.tempo_min,
		tempo_max=rhythm_cfg.tempo_max,
		win_length=plp_win_length,
	)

	# Flatten to 1-D in case librosa returns a leading channel dimension.
	# Use atleast_1d to ensure we get at least (n,), never a 0-D scalar array,
	# which would crash find_peaks.
	pulse_curve: numpy.ndarray = numpy.atleast_1d(pulse_raw.squeeze()).astype(numpy.float32)

	# Extract peak positions: frames where the pulse curve is locally maximal
	peak_indices, _ = scipy.signal.find_peaks(pulse_curve)

	pulse_peak_times: tuple[float, ...] = tuple(
		float(t) for t in librosa.frames_to_time(
			peak_indices,
			sr=params.sample_rate,
			hop_length=params.hop_length,
		)
	)

	return RhythmResult(
		tempo_bpm=tempo_bpm,
		beat_times=beat_times,
		pulse_curve=pulse_curve,
		pulse_peak_times=pulse_peak_times,
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


def format_rhythm_result (result: RhythmResult) -> str:

	"""Return a single-line human-readable summary of a rhythm analysis result.

	Args:
		result: Computed rhythm metrics.

	Returns:
		String of the form:
		"tempo=X.XBpm  beats=N  pulses=N"
	"""

	return (
		f"tempo={result.tempo_bpm:.1f}bpm"
		f"  beats={len(result.beat_times)}"
		f"  pulses={len(result.pulse_peak_times)}"
	)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def to_mono_float (audio: numpy.ndarray, bit_depth: int) -> numpy.ndarray:

	"""Convert integer PCM to a normalised float32 mono array.

	Normalises to [-1.0, 1.0] using the full-scale range for the bit depth.
	Stereo (or higher) inputs are mixed down by averaging channels. The result
	can be passed directly to analyze_mono() and analyze_rhythm().

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
