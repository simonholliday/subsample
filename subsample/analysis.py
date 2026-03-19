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
  zcr                — 0 = smooth/DC, 1 = maximally noisy (zero crossing rate)
  harmonic_ratio     — 0 = purely percussive, 1 = purely harmonic (HPSS energy ratio)
  spectral_contrast  — 0 = flat spectrum, 1 = strong spectral peaks-vs-valleys
  voiced_fraction    — 0 = unpitched/noise, 1 = clearly pitched throughout (pyin)

Rhythm metrics (RhythmResult) — raw values, NOT normalised:

  tempo_bpm          — estimated global tempo in BPM (0.0 if undetected)
  beat_times         — beat positions in seconds from beat_track
  pulse_curve        — frame-by-frame rhythmic salience from PLP
  pulse_peak_times   — local maxima of pulse curve in seconds
  onset_times        — transient onset positions in seconds from onset_detect
  onset_count        — number of detected onsets

Pitch metrics (PitchResult) — raw structured data, NOT normalised:

  dominant_pitch_hz  — median fundamental frequency of voiced frames in Hz (0.0 = unpitched)
  pitch_confidence   — mean pyin voiced probability across voiced frames (0.0–1.0)
  chroma_profile     — 12-element tuple: mean energy per pitch class C through B
  dominant_pitch_class — index 0-11 of strongest pitch class (C=0, C#=1, …, B=11); -1 if unpitched
  mfcc               — 13 mean MFCC coefficients for timbre fingerprinting / similarity
"""

import dataclasses
import math
import warnings

import librosa
import numpy
import scipy.signal

import subsample.config


# Bump this string whenever the analysis algorithm changes in a way that
# would produce different results for the same audio. The cache module uses
# it to detect stale sidecar files and trigger re-analysis.
ANALYSIS_VERSION: str = "1"

# ---------------------------------------------------------------------------
# Reference constants for log-scale normalisation
# ---------------------------------------------------------------------------

# Attack / release: time range over which scores are spread.
# 1 ms = imperceptibly fast (e.g. a single sample click at 44 kHz).
# 2 s = extremely slow (ambient pad with gradual fade-in/out).
# Log midpoint ≈ 45 ms — roughly where a medium-speed pluck or strum sits.
_ATTACK_RELEASE_MIN_S: float = 0.001   # 1 ms
_ATTACK_RELEASE_MAX_S: float = 2.0     # 2 s

# Spectral flatness (Wiener entropy) log-normalization range.
# Raw Wiener entropy from librosa clusters near 0 for all real-world audio —
# even "noisy" sounds. Log scale spreads the useful range.
# Values at or below _FLATNESS_MIN map to 0.0 (very tonal); values at or
# above _FLATNESS_MAX map to 1.0 (noise-like). Calibrate against your source
# material if values still cluster — log the pre-normalization mean.
_FLATNESS_MIN: float = 1e-5    # below the floor of real tonal sounds; maps to 0.0
_FLATNESS_MAX: float = 0.9     # near-white noise

# Spectral frequency range: 20 Hz (lower limit of human hearing) to Nyquist.
# Using 20 Hz as the minimum means a signal centred at 20 Hz scores 0.0.
_FREQ_MIN_HZ: float = 20.0

# Threshold for onset/decay detection: frames where RMS exceeds this fraction
# of the peak RMS are considered "active". 10 % chosen as a reasonable -20 dB
# equivalent for typical audio signals.
_ACTIVE_THRESHOLD_RATIO: float = 0.1

# Zero crossing rate: the theoretical maximum is 0.5 (every consecutive pair
# of samples has opposite sign — alternating +1, -1, +1, ...). A linear map
# from [0, 0.5] to [0, 1] is appropriate since ZCR is already perceptually
# linear (unlike time and frequency, which are logarithmic).
_ZCR_MAX: float = 0.5

# Spectral contrast: mean contrast across all sub-bands in dB. librosa
# spectral_contrast values are always non-negative (they represent the
# difference between spectral peaks and valleys). A linear map against a dB
# ceiling works because contrast is already expressed on a log (dB) scale.
_CONTRAST_MAX_DB: float = 40.0

# pyin (probabilistic YIN) pitch detection frequency search range.
# C2 (65 Hz) covers the lowest notes on a standard guitar and bass.
# C7 (2093 Hz) is well above the highest pitch most instruments produce.
# Narrowing this range speeds up pyin and reduces false detections.
_PYIN_FMIN: float = 65.0     # C2
_PYIN_FMAX: float = 2093.0   # C7

# Number of MFCC coefficients to compute. 13 is the standard for timbre
# fingerprinting; it captures the coarse spectral shape without over-fitting
# individual harmonics. MFCCs are used for similarity comparison (cosine
# distance between vectors), not as individual human-readable metrics.
_N_MFCC: int = 13

# Pitch class names for formatting: index 0=C, 1=C#, …, 11=B.
_PITCH_CLASSES: tuple[str, ...] = (
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
)

# Type alias for the three-array result returned by librosa.pyin.
# (f0_hz, voiced_flag, voiced_probs) — see _run_pyin() in the helpers section.
_PYINResult = tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]


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

	zcr: float
	"""Zero crossing rate, linearly mapped from [0, 0.5] to [0, 1].
	0.0 = pure DC / very smooth signal, 1.0 = maximally noisy (every sample
	alternates sign). Complements spectral_flatness from the time domain:
	percussive transients and noise both produce high ZCR."""

	harmonic_ratio: float
	"""Fraction of total energy in the harmonic component after HPSS
	(Harmonic-Percussive Source Separation). Range [0, 1].
	0.0 = entirely percussive (transient, click, drum hit),
	1.0 = entirely harmonic (sustained tone, pitched instrument).
	More robust than spectral_flatness alone for percussion classification."""

	spectral_contrast: float
	"""Mean spectral contrast across all sub-bands, linearly mapped from
	0–40 dB to [0, 1]. Spectral contrast measures the difference between
	peaks and valleys in each frequency sub-band. A pure tone has a few
	sharp peaks and deep valleys (high contrast); broadband noise or a
	dense mix has a relatively flat spectrum (low contrast)."""

	voiced_fraction: float
	"""Fraction of analysis frames where a fundamental frequency was
	detected by pyin (probabilistic YIN), range [0, 1].
	0.0 = entirely unpitched (noise, percussion, silence),
	1.0 = clearly pitched throughout (sustained tone, singing, instrument).
	Use this to decide whether pitch-related features are meaningful."""

	def as_vector (self) -> numpy.ndarray:

		"""Return the nine spectral metrics as a float32 1-D array.

		This is the **spectral fingerprint** of the sound — a fixed-length
		vector where every element is normalised to [0.0, 1.0]:

		  [spectral_flatness, attack, release, spectral_centroid,
		   spectral_bandwidth, zcr, harmonic_ratio, spectral_contrast,
		   voiced_fraction]

		Use this vector for cosine similarity comparison against reference
		sample fingerprints (see subsample.similarity).
		"""

		return numpy.array([
			self.spectral_flatness, self.attack,          self.release,
			self.spectral_centroid, self.spectral_bandwidth,
			self.zcr, self.harmonic_ratio, self.spectral_contrast, self.voiced_fraction,
		], dtype=numpy.float32)


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

	onset_times: tuple[float, ...]
	"""Transient onset positions in seconds from librosa.onset.onset_detect.
	Marks the start of each audible event (hit, pluck, attack). Unlike beat_times,
	these are not quantised to a grid — they reflect the raw signal structure.
	Useful for slicing multi-hit recordings into individual one-shots."""

	onset_count: int
	"""Number of detected onsets. Convenience field equivalent to len(onset_times)."""


@dataclasses.dataclass(frozen=True)
class PitchResult:

	"""Pitch and timbre properties detected from a single recording.

	Values are NOT normalised to [0, 1]. They represent raw pitch data and
	timbre fingerprints intended for downstream musical use: deciding which
	key a sample belongs to, pitch-shifting to a target note, or finding
	similar-sounding samples via MFCC cosine similarity.

	All fields return sensible defaults (0.0, empty tuples, -1) for unpitched
	or silent audio — check voiced_fraction in AnalysisResult before relying
	on pitch-related fields.
	"""

	dominant_pitch_hz: float
	"""Median fundamental frequency (F0) across voiced frames, in Hz.
	Computed by pyin (probabilistic YIN). 0.0 if no voiced frames were found.
	Use this to determine the base pitch of a tonal sample before pitch-shifting.
	Example: a recorded A4 returns ~440.0 Hz; a drum hit returns 0.0."""

	pitch_confidence: float
	"""Mean pyin voiced probability across voiced frames, in [0, 1].
	Higher values mean pyin is more certain the signal contains a pitched tone.
	0.0 if no voiced frames were found. Not meaningful for percussive audio."""

	chroma_profile: tuple[float, ...]
	"""Mean energy per pitch class over all frames, as a 12-element tuple.
	Index 0 = C, 1 = C#, 2 = D, …, 11 = B. Values are in [0, 1] (normalised
	by librosa). Use this to determine which notes/keys a sample fits.
	For a pure A, index 9 will dominate; for a chord, multiple indices peak."""

	dominant_pitch_class: int
	"""Index of the strongest pitch class in chroma_profile (argmax), 0–11.
	-1 if no chroma energy was detected. Maps to _PITCH_CLASSES for display.
	Example: A = 9, C = 0, F# = 6. Does not encode octave information."""

	mfcc: tuple[float, ...]
	"""13 mean MFCC (Mel-frequency cepstral coefficient) values across all frames.
	MFCCs capture the coarse spectral shape (timbre) of a sound, independent
	of pitch. Use cosine distance between mfcc vectors to find similar-sounding
	samples. Individual coefficients are not humanly interpretable."""


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
			zcr=0.0,
			harmonic_ratio=0.0,
			spectral_contrast=0.0,
			voiced_fraction=0.0,
		)

	mono = to_mono_float(audio, bit_depth)

	return analyze_mono(mono, params)


def analyze_mono (
	mono: numpy.ndarray,
	params: AnalysisParams,
	*,
	_pyin_voiced_flag: numpy.ndarray | None = None,
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
			zcr=0.0,
			harmonic_ratio=0.0,
			spectral_contrast=0.0,
			voiced_fraction=0.0,
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
	zcr = _compute_zcr(mono, params)
	harmonic_ratio = _compute_harmonic_ratio(mono, params)
	contrast = _compute_spectral_contrast(mono, params)
	voiced_fraction = _compute_voiced_fraction(mono, params, pyin_voiced_flag=_pyin_voiced_flag)

	return AnalysisResult(
		spectral_flatness=_log_normalize(
			float(numpy.mean(flatness)), _FLATNESS_MIN, _FLATNESS_MAX
		),
		attack=attack,
		release=release,
		spectral_centroid=centroid,
		spectral_bandwidth=bandwidth,
		zcr=zcr,
		harmonic_ratio=harmonic_ratio,
		spectral_contrast=contrast,
		voiced_fraction=voiced_fraction,
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

	if mono.shape[0] < params.n_fft:
		# Beat analysis requires enough samples to compute onset strength.
		# Signals shorter than n_fft (46ms at 44100 Hz) are too short for meaningful
		# rhythm analysis. librosa.beat.beat_track/plp call stft internally with
		# hardcoded n_fft=2048; for short signals this triggers a UserWarning and
		# produces degenerate output. Just bail out and return silence.
		return RhythmResult(
			tempo_bpm=0.0,
			beat_times=(),
			pulse_curve=numpy.zeros(0, dtype=numpy.float32),
			pulse_peak_times=(),
			onset_times=(),
			onset_count=0,
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

	# --- onset detection: precise transient locations ---
	# onset_detect finds the start of each audible event (attack) in the signal.
	# units='time' returns positions in seconds directly.
	# Unlike beat_times, onsets are not quantised — they mark exact moments
	# where energy rises abruptly, making them useful for slicing multi-hit
	# recordings into individual one-shots.
	onset_times_raw: numpy.ndarray = librosa.onset.onset_detect(
		y=mono,
		sr=params.sample_rate,
		hop_length=params.hop_length,
		units='time',
	)

	onset_times: tuple[float, ...] = tuple(float(t) for t in onset_times_raw)

	return RhythmResult(
		tempo_bpm=tempo_bpm,
		beat_times=beat_times,
		pulse_curve=pulse_curve,
		pulse_peak_times=pulse_peak_times,
		onset_times=onset_times,
		onset_count=len(onset_times),
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
		"duration=X.XXs  flatness=X.XXX  attack=X.XXX  ...  zcr=X.XXX  harmonic=X.XXX  contrast=X.XXX  voiced=X.XXX"
	"""

	return (
		f"duration={duration:.2f}s"
		f"  flatness={result.spectral_flatness:.3f}"
		f"  attack={result.attack:.3f}"
		f"  release={result.release:.3f}"
		f"  centroid={result.spectral_centroid:.3f}"
		f"  bandwidth={result.spectral_bandwidth:.3f}"
		f"  zcr={result.zcr:.3f}"
		f"  harmonic={result.harmonic_ratio:.3f}"
		f"  contrast={result.spectral_contrast:.3f}"
		f"  voiced={result.voiced_fraction:.3f}"
	)


def format_rhythm_result (result: RhythmResult) -> str:

	"""Return a single-line human-readable summary of a rhythm analysis result.

	Args:
		result: Computed rhythm metrics.

	Returns:
		String of the form:
		"tempo=X.XBpm  beats=N  pulses=N  onsets=N"
	"""

	return (
		f"tempo={result.tempo_bpm:.1f}bpm"
		f"  beats={len(result.beat_times)}"
		f"  pulses={len(result.pulse_peak_times)}"
		f"  onsets={result.onset_count}"
	)


def format_pitch_result (result: PitchResult) -> str:

	"""Return a single-line human-readable summary of pitch analysis.

	Args:
		result: Computed pitch metrics.

	Returns:
		String of the form:
		"pitch=440.0Hz  chroma=A  pitch_conf=0.92"
		or "pitch=none  chroma=none  pitch_conf=0.00" for unpitched audio.
	"""

	if result.dominant_pitch_hz > 0.0:
		pitch_str = f"{result.dominant_pitch_hz:.1f}Hz"
	else:
		pitch_str = "none"

	if result.dominant_pitch_class >= 0:
		chroma_str = _PITCH_CLASSES[result.dominant_pitch_class]
	else:
		chroma_str = "none"

	return (
		f"pitch={pitch_str}"
		f"  chroma={chroma_str}"
		f"  pitch_conf={result.pitch_confidence:.2f}"
	)


def analyze_pitch (
	mono: numpy.ndarray,
	params: AnalysisParams,
	*,
	_pyin_result: _PYINResult | None = None,
) -> PitchResult:

	"""Detect pitch and timbre properties from a pre-normalised float32 mono array.

	Runs three analyses:
	  1. pyin — probabilistic YIN for fundamental frequency (F0) per frame.
	     Returns per-frame F0 in Hz, a voiced flag, and a voiced probability.
	  2. chroma_cqt — constant-Q chromagram for pitch class content (C–B).
	     Useful for key detection and identifying which notes a sample contains.
	  3. mfcc — Mel-frequency cepstral coefficients for timbre fingerprinting.
	     Use cosine similarity between mfcc vectors to find similar-sounding samples.

	Use voiced_fraction from AnalysisResult to decide whether pitch-related
	fields are meaningful before relying on dominant_pitch_hz or chroma_profile.

	Args:
		mono:         Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params:       Pre-computed FFT parameters from compute_params().
		_pyin_result: Pre-computed pyin output from _run_pyin(). When provided,
		              the pyin call is skipped (used by analyze_all() to share
		              the computation with analyze_mono()).

	Returns:
		PitchResult with pitch, chroma, and MFCC data.
	"""

	# pyin requires at least one period of fmin to fit in the frame.
	# Use _run_pyin() so this check and the pyin call are in one place.
	pyin = _pyin_result if _pyin_result is not None else _run_pyin(mono, params)

	if pyin is None:
		return PitchResult(
			dominant_pitch_hz=0.0,
			pitch_confidence=0.0,
			chroma_profile=tuple(0.0 for _ in range(12)),
			dominant_pitch_class=-1,
			mfcc=tuple(0.0 for _ in range(_N_MFCC)),
		)

	# --- pyin: probabilistic fundamental frequency estimation ---
	# voiced_flag is True for frames where a pitch was detected with reasonable
	# confidence. f0_hz is NaN for unvoiced frames — we ignore those.
	f0_hz, voiced_flag, voiced_probs = pyin

	voiced_f0 = f0_hz[voiced_flag]
	dominant_pitch_hz = float(numpy.median(voiced_f0)) if voiced_f0.size > 0 else 0.0

	voiced_p = voiced_probs[voiced_flag]
	pitch_confidence = float(numpy.mean(voiced_p)) if voiced_p.size > 0 else 0.0

	# --- chroma_cqt: pitch class energy (constant-Q, more accurate than STFT) ---
	# Returns shape (12, n_frames). Mean across time gives the average energy in
	# each pitch class over the whole recording.
	#
	# chroma_cqt uses multi-rate CQT processing: it internally downsamples the
	# signal by 2x for each lower octave (so a 5847-sample signal becomes
	# ~730 → ~365 → ~183 → ~92 → ~46 at the lowest octaves). Each downsampled
	# level is analysed with an internal n_fft=1024, which triggers librosa's
	# "n_fft too large" UserWarning for levels shorter than 1024 samples. This
	# is an expected, harmless implementation detail — librosa zero-pads
	# internally and produces correct output. Suppress the warning here.
	with warnings.catch_warnings():
		# Suppress only the "n_fft too large" warning from librosa's internal CQT
		# downsampling. Other UserWarnings (e.g. deprecated parameters) remain visible.
		warnings.filterwarnings("ignore", message="n_fft=", category=UserWarning)
		chroma_raw = librosa.feature.chroma_cqt(
			y=mono,
			sr=params.sample_rate,
			hop_length=params.hop_length,
		)

	# Mean over time frames; result is shape (12,)
	chroma_mean = numpy.mean(chroma_raw, axis=1)
	chroma_profile: tuple[float, ...] = tuple(float(v) for v in chroma_mean)

	chroma_sum = float(numpy.sum(chroma_mean))
	dominant_pitch_class = int(numpy.argmax(chroma_mean)) if chroma_sum > 1e-8 else -1

	# --- mfcc: timbre fingerprint ---
	# Returns shape (_N_MFCC, n_frames). Mean across time gives a compact
	# descriptor of the overall timbral character. Not meaningful per-coefficient,
	# but cosine distance between two mfcc vectors is a good timbre similarity measure.
	mfcc_raw = librosa.feature.mfcc(
		y=mono,
		sr=params.sample_rate,
		n_mfcc=_N_MFCC,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	mfcc: tuple[float, ...] = tuple(float(v) for v in numpy.mean(mfcc_raw, axis=1))

	return PitchResult(
		dominant_pitch_hz=dominant_pitch_hz,
		pitch_confidence=pitch_confidence,
		chroma_profile=chroma_profile,
		dominant_pitch_class=dominant_pitch_class,
		mfcc=mfcc,
	)


def analyze_all (
	mono: numpy.ndarray,
	params: AnalysisParams,
	rhythm_cfg: subsample.config.AnalysisConfig,
) -> tuple[AnalysisResult, RhythmResult, PitchResult]:

	"""Run all three analyses (spectral, rhythm, pitch) with shared pyin computation.

	Preferred entry point for the recorder and any code that needs all three
	results. Runs pyin once and passes the result to both analyze_mono() and
	analyze_pitch(), avoiding the ~200–300 ms double computation.

	Args:
		mono:       Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params:     Pre-computed FFT parameters from compute_params().
		rhythm_cfg: Configurable tempo priors from AnalysisConfig.

	Returns:
		(spectral, rhythm, pitch) — a triple of result dataclasses.
	"""

	# Run pyin once; share the result between spectral (voiced_fraction) and pitch.
	pyin = _run_pyin(mono, params)

	rhythm = analyze_rhythm(mono, params, rhythm_cfg)
	spectral = analyze_mono(mono, params, _pyin_voiced_flag=pyin[1] if pyin is not None else None)
	pitch = analyze_pitch(mono, params, _pyin_result=pyin)

	return spectral, rhythm, pitch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pyin_min_frame_length (params: AnalysisParams) -> int:

	"""Minimum signal length (samples) for pyin to run at the configured fmin.

	pyin requires frame_length > sr / fmin so at least one period of the lowest
	frequency fits in the analysis window. At 44100 Hz with fmin=65 Hz (C2)
	this is ≈ 680 samples.
	"""

	return int(math.ceil(params.sample_rate / _PYIN_FMIN)) + 1


def _run_pyin (mono: numpy.ndarray, params: AnalysisParams) -> _PYINResult | None:

	"""Run pyin (probabilistic YIN) pitch detection on a mono float32 signal.

	Returns (f0_hz, voiced_flag, voiced_probs) — the three arrays produced by
	librosa.pyin. voiced_flag is a boolean array (True = pitch detected).
	f0_hz is NaN for unvoiced frames.

	Returns None if the signal is shorter than _pyin_min_frame_length() — in
	that case the caller should treat the signal as unpitched.

	This is the single shared pyin call site. Both analyze_pitch() and
	_compute_voiced_fraction() call this function so pyin is never run twice
	for the same signal.
	"""

	if mono.shape[0] < _pyin_min_frame_length(params):
		return None

	return librosa.pyin(
		mono,
		fmin=_PYIN_FMIN,
		fmax=_PYIN_FMAX,
		sr=params.sample_rate,
		frame_length=params.n_fft,
		hop_length=params.hop_length,
	)


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

	# center=False: frames start at sample 0 with no zero-padding.
	# center=True (librosa default) pads n_fft//2 zeros at the start, which
	# artificially shifts the apparent peak forward by n_fft//2/hop_length
	# frames (2 frames at 44100 Hz with default params). For any sustained
	# signal this produces a constant, meaningless attack value of 0.414.
	rms = librosa.feature.rms(
		y=mono,
		frame_length=params.n_fft,
		hop_length=params.hop_length,
		center=False,
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


def _compute_zcr (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> float:

	"""Measure the zero crossing rate of the signal.

	ZCR is the fraction of consecutive sample pairs that have opposite signs.
	It is a time-domain indicator of noisiness: percussive transients and noise
	produce many sign changes (high ZCR), while smooth tonal signals produce few.

	The theoretical maximum is 0.5 (every sample alternates sign), so a simple
	linear map from [0, 0.5] → [0, 1] is used — no log scale needed.

	Args:
		mono:   Float32 audio, shape (n_frames,).
		params: FFT params (used for consistent frame sizing).

	Returns:
		ZCR score in [0.0, 1.0]. 0 = smooth/DC, 1 = maximally noisy.
	"""

	zcr_frames = librosa.feature.zero_crossing_rate(
		y=mono,
		frame_length=params.n_fft,
		hop_length=params.hop_length,
	)

	# zcr_frames has shape (1, n_frames); mean gives a single rate value
	mean_zcr = float(numpy.mean(zcr_frames))

	return min(mean_zcr / _ZCR_MAX, 1.0)


def _compute_harmonic_ratio (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> float:

	"""Measure the fraction of total energy in the harmonic component.

	Uses HPSS (Harmonic-Percussive Source Separation) to decompose the signal
	into a harmonic part (sustained tones, pitched content) and a percussive part
	(transients, drum hits). The ratio is harmonic_energy / total_energy.

	A pure sine wave scores close to 1.0; a drum hit or click scores close to 0.0.
	More robust than spectral_flatness alone for percussion classification, because
	flatness is sensitive to the spectral shape while HPSS separates by temporal
	structure (sustained vs transient).

	Args:
		mono:   Float32 audio, shape (n_frames,).
		params: FFT params — n_fft is passed to hpss so it matches the clamped
		        window size used by all other helpers (avoids spurious warnings on
		        short signals).

	Returns:
		Harmonic ratio in [0.0, 1.0]. 0 = purely percussive, 1 = purely harmonic.
	"""

	# Pass n_fft so hpss uses the same (possibly clamped) window as the other
	# spectral helpers. Without this, hpss defaults to n_fft=2048 and emits a
	# UserWarning when the signal is shorter than that.
	harmonic, _ = librosa.effects.hpss(mono, n_fft=params.n_fft)

	energy_total = float(numpy.sum(mono ** 2))

	# Guard: avoid division by zero for silent or near-silent signals
	if energy_total < 1e-16:
		return 0.0

	energy_harmonic = float(numpy.sum(harmonic ** 2))

	return min(energy_harmonic / energy_total, 1.0)


def _compute_spectral_contrast (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> float:

	"""Measure the mean spectral contrast across all frequency sub-bands.

	Spectral contrast is the difference (in dB) between peaks and valleys in
	each of librosa's frequency sub-bands. A pure tone has a few sharp peaks and
	deep valleys (high contrast). Broadband noise or a dense mix has a relatively
	flat spectrum (low contrast).

	The mean across all bands and frames is linearly mapped from [0, 40 dB] to
	[0, 1]. No log scale is needed because contrast is already expressed in dB
	(which is itself a log scale).

	Args:
		mono:   Float32 audio, shape (n_frames,).
		params: FFT params.

	Returns:
		Contrast score in [0.0, 1.0]. 0 = flat spectrum, 1 = strong peaks.
	"""

	# spectral_contrast divides the spectrum into n_bands+1=7 sub-bands.
	# Each band needs at least a few FFT bins to be valid. For very short
	# signals where n_fft has been clamped to a small value, the frequency
	# resolution is too coarse to compute meaningful sub-band contrasts.
	# Return 0.0 (no contrast measurable) rather than crashing with an
	# IndexError from librosa's internal band indexing logic.
	if params.n_fft < 64:
		return 0.0

	contrast = librosa.feature.spectral_contrast(
		y=mono,
		sr=params.sample_rate,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	# contrast has shape (n_bands, n_frames); mean across both axes
	mean_db = float(numpy.mean(contrast))

	return min(max(mean_db, 0.0) / _CONTRAST_MAX_DB, 1.0)


def _compute_voiced_fraction (
	mono: numpy.ndarray,
	params: AnalysisParams,
	*,
	pyin_voiced_flag: numpy.ndarray | None = None,
) -> float:

	"""Measure the fraction of frames where a fundamental frequency was detected.

	Uses pyin (probabilistic YIN) to detect voiced frames. A frame is voiced
	when pyin finds a plausible fundamental frequency with sufficient confidence.
	The voiced fraction is the proportion of frames flagged as voiced.

	0.0 = entirely unpitched (noise, percussion, silence).
	1.0 = clearly pitched throughout (sustained tone, singing, instrument).

	This is the [0, 1] companion to PitchResult.dominant_pitch_hz: use it to
	decide whether pitch-related features are meaningful for a given sample.

	Args:
		mono:             Float32 audio, shape (n_frames,).
		params:           FFT params.
		pyin_voiced_flag: Pre-computed voiced_flag array from _run_pyin(). When
		                  provided by analyze_all(), the pyin call is skipped.

	Returns:
		Voiced fraction in [0.0, 1.0].
	"""

	# Fast path: caller already has the pyin result (e.g. from analyze_all).
	if pyin_voiced_flag is not None:
		return float(numpy.mean(pyin_voiced_flag))

	# Slow path: run pyin independently (e.g. when analyze_mono is called alone).
	result = _run_pyin(mono, params)

	if result is None:
		return 0.0

	_f0, voiced_flag, _probs = result

	return float(numpy.mean(voiced_flag))
