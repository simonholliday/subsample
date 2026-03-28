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
  log_attack_time    — 0 = instant spectral onset, 1 = very slow onset (flux-based)
  spectral_flux      — 0 = static spectrum (tone/drone), 1 = rapidly changing spectrum

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

Timbre metrics (TimbreResult) — timbral fingerprints, NOT normalised:

  mfcc               — 13 mean MFCC coefficients for timbre fingerprinting / similarity
  mfcc_delta         — 13 mean delta-MFCC coefficients (first-order temporal difference)
  mfcc_onset         — 13 onset-weighted MFCC coefficients (exponential decay from attack)

Level metrics (LevelResult) — amplitude measurements, NOT used for similarity:

  peak               — peak absolute amplitude in [0, 1] (max(|signal|) on float32 mono)
  rms                — RMS amplitude in [0, 1] (sqrt(mean(signal²)) on float32 mono)
  Used at playback time to normalise levels across samples recorded at different volumes.

Band energy metrics (BandEnergyResult) — per-band energy distribution, used for similarity:

  energy_fractions   — 4-element tuple: fraction of total energy in each band (sums to ~1.0)
                       bands: sub-bass (20-250 Hz), low-mid (250-2k Hz),
                              high-mid (2-6k Hz), presence (6k+ Hz)
  decay_rates        — 4-element tuple: per-band decay rate, log-normalised to [0, 1]
                       0.0 = instant decay (≤ 1 ms), 1.0 = very long decay (≥ 2 s)
  Directly encodes drum-type signatures: kick = sub-bass dominant, snare = mid + presence,
  hi-hat = air. Included in the similarity feature vector as the 5th independent group.
"""

import dataclasses
import math
import warnings

import librosa
import numpy
import scipy.signal

import subsample.config


# Suppress librosa's UserWarning that fires when pyin finds no voiced frames
# (e.g. silent or completely unpitched audio). This is expected and handled
# downstream — pyin returns an all-NaN f0 array which the pitch analysis
# treats as "no pitch detected". The warning would otherwise clutter logs
# every time a near-silent or purely percussive sample is recorded.
# The filter is set at module level so it applies to all threads, including
# the background WavWriter thread that runs analysis on captured recordings.
warnings.filterwarnings(
	"ignore",
	message="Trying to estimate tuning from empty frequency set",
	category=UserWarning,
	module="librosa",
)
# numpy emits these when analysis runs on very short or silent audio (empty
# arrays produced by librosa). They are expected and not actionable.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
# librosa's chroma_cqt uses multi-rate CQT that internally downsamples the signal
# by 2× per octave. Each downsampled level is analysed with an internal n_fft=1024,
# which triggers this UserWarning for short signals (< 1024 samples at that level).
# The computation is correct — librosa zero-pads internally. Python's __warningregistry__
# can cache a first-seen warning before any context-manager filter applies, so a
# module-level filter is the only reliable way to suppress it.
warnings.filterwarnings("ignore", message="n_fft=", category=UserWarning)
# When n_fft is clamped to a very short signal, the mel filterbank can end up
# with more bands than available FFT bins, leaving some filters empty.  The
# MFCC computation still succeeds — affected coefficients are zero — and the
# impact on similarity matching for these edge-case samples is negligible.
warnings.filterwarnings("ignore", message="Empty filters detected in mel frequency basis", category=UserWarning)


# Bump this string whenever the analysis algorithm changes in a way that
# would produce different results for the same audio. The cache module uses
# it to detect stale sidecar files and trigger re-analysis.
ANALYSIS_VERSION: str = "11"

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

# Onset-weighted MFCC: exponential decay time constant in milliseconds.
# Frames near the onset (attack) are weighted by exp(-t / decay), so the
# first ~50 ms contribute most. Percussive identity is concentrated at
# onset; the tail is decay and room, which is less discriminative.
_ONSET_DECAY_MS: float = 50.0

# Spectral flux normalisation range. librosa.onset.onset_strength() output
# is in arbitrary units (roughly proportional to mean absolute spectral
# change per frame). Empirical range for typical audio:
#   0.01 — near-static tone, barely changing spectrum
#   5.0  — busy percussive audio with frequent large spectral jumps
# Log normalisation spreads the useful range; values outside are clamped.
_FLUX_MIN: float = 0.01
_FLUX_MAX: float = 5.0

# Multi-band energy envelope: frequency band boundaries in Hz.
# These four bands directly encode the physical signatures of drum types:
#   sub-bass  (20-250 Hz)    — kick drum fundamentals and body
#   low-mid   (250-2000 Hz)  — snare body, tom resonance
#   high-mid  (2000-6000 Hz) — snare crack, cymbal body, stick attack
#   presence  (6000+ Hz)     — hi-hat sizzle, cymbal shimmer, air
# Fixed constants: changing these boundaries would invalidate all cached analysis.
_BAND_EDGES: tuple[tuple[float, float], ...] = (
	(20.0, 250.0),
	(250.0, 2000.0),
	(2000.0, 6000.0),
	(6000.0, 20000.0),
)
_N_BANDS: int = len(_BAND_EDGES)

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

	log_attack_time: float
	"""Time from first spectral activity to peak spectral flux, log-mapped to [0, 1].
	0.0 = instant spectral onset (≤ 1 ms — a sharp click or drum hit),
	1.0 = very gradual spectral build (≥ 2 s — a slow reverb swell).
	Complements attack (which is RMS-based): spectral flux detects the snare
	wire rattle before the body resonance appears in the energy envelope."""

	spectral_flux: float
	"""Mean spectral flux (onset_strength), log-mapped to [0, 1].
	0.0 = static / barely changing spectrum (drone, sustained tone),
	1.0 = rapidly and continuously evolving spectrum (busy percussion, brushes).
	High values indicate sounds where the spectral shape changes frequently,
	not just at a single transient."""

	def as_vector (self) -> numpy.ndarray:

		"""Return nine of the eleven spectral metrics as a float32 1-D array (excludes log_attack_time and spectral_flux).

		This is the **spectral fingerprint** of the sound — a fixed-length
		vector where every element is normalised to [0.0, 1.0]:

		  [spectral_flatness, attack, release, spectral_centroid,
		   spectral_bandwidth, zcr, harmonic_ratio, spectral_contrast,
		   voiced_fraction]

		Used for display and export (e.g. sidecar JSON, analysis output).
		For similarity scoring, see `_build_feature_vector()` in
		`subsample.similarity`, which uses all 11 AnalysisResult fields
		directly and does not call this method.
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
	Values are quantized to a regular grid at tempo_bpm.
	Empty tuple if no beats were detected."""

	pulse_curve: numpy.ndarray
	"""Frame-by-frame rhythmic salience from the PLP (Predominant Local Pulse) algorithm.
	Shape (n_frames,), values >= 0. Peaks mark rhythmically salient moments.
	Frame timing: frame i corresponds to i * hop_length / sample_rate seconds.
	Unlike beat_times, this reflects locally varying tempo and is not grid-quantized."""

	pulse_peak_times: tuple[float, ...]
	"""Local maxima of the pulse curve, in seconds.
	Candidate beat locations that do not assume constant tempo.
	Useful when beat_times is empty or the detected grid does not fit the signal well."""

	onset_times: tuple[float, ...]
	"""Transient onset positions in seconds from librosa.onset.onset_detect.
	Marks the start of each audible event (hit, pluck, attack). Unlike beat_times,
	these are not quantized to a grid — they reflect the raw signal structure.
	Useful for slicing multi-hit recordings into individual one-shots."""

	attack_times: tuple[float, ...]
	"""Sample-accurate attack start times in seconds, one per onset.
	Refined from onset_times by searching backward in the amplitude envelope
	to find where energy first rises above 10% of the local peak.  These
	align with the perceptual "hit" — the moment a musician would tap — and
	are used by the time-stretch handler for beat-grid alignment.
	Same length as onset_times; each value <= the corresponding onset time."""

	onset_count: int
	"""Number of detected onsets. Convenience field equivalent to len(onset_times)."""


@dataclasses.dataclass(frozen=True)
class PitchResult:

	"""Pitch properties detected from a single recording.

	Values are NOT normalised to [0, 1]. They represent raw pitch data
	intended for downstream musical use: deciding which key a sample belongs
	to, or pitch-shifting to a target note.

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

	pitch_stability: float
	"""Standard deviation of the pyin F0 track across voiced frames, in semitones.

	Measured in semitones (MIDI units) so the value is frequency-independent —
	1 semitone of variation sounds the same at any pitch.

	0.0  — perfectly stable (2+ voiced frames with identical pitch), or ≤1 voiced
	       frame (check voiced_frame_count before trusting this value).
	~0.1 — very stable (synthesiser, piano, held wind note).
	~0.5 — slight natural variation.
	~1.0 — noticeable movement (gentle vibrato, slight bend).
	>2.0 — significant change (strong vibrato, glide, or multiple pitches).

	Unvoiced frames (silence, noise, gaps) are excluded — a tone that drops out
	and returns at the same pitch still registers as stable.

	Only meaningful when voiced_frame_count >= 2."""

	voiced_frame_count: int
	"""Number of pyin frames flagged as voiced (i.e. containing a detected pitch).

	One pyin frame spans hop_length audio samples (default 512 at 44100 Hz ≈ 11.6 ms).
	A value of 0 means pyin found no pitched content at all.

	Use this alongside pitch_stability — when voiced_frame_count == 1, pitch_stability
	is 0.0 by definition (std dev of a single value) but that does not mean the pitch
	is stable; it just means there is not enough data to measure variation.

	Typical range for usable keyboard samples: >= 5 frames (~60 ms of voiced content)."""


@dataclasses.dataclass(frozen=True)
class TimbreResult:

	"""Timbral fingerprints for similarity matching.

	All three fields are 13-element tuples (one value per MFCC coefficient),
	computed from the mel spectrogram of a recording. They are independent of
	pitch — two sounds at different pitches with the same timbral character
	will produce similar MFCC vectors.

	Use cosine distance between vectors of the same type to find similar-sounding
	samples. See subsample.similarity for the comparison infrastructure.
	"""

	mfcc: tuple[float, ...]
	"""13 mean MFCC (Mel-frequency cepstral coefficient) values across all frames.
	Captures the coarse spectral shape (timbre) of a sound, independent of pitch.
	The most stable fingerprint — good for general timbral similarity."""

	mfcc_delta: tuple[float, ...]
	"""13 mean delta-MFCC values (first-order temporal differences).
	Captures how the timbre changes over time rather than its average value.
	For percussive sounds, the attack-to-decay timbre shift is the primary
	identity signal — a snare and a tom can have similar mean MFCCs but very
	different delta-MFCC trajectories."""

	mfcc_onset: tuple[float, ...]
	"""13 onset-weighted MFCC values (exponential decay weighting from attack).
	MFCC frames near the onset contribute more than the decay tail, so the
	vector reflects the timbral character of the attack rather than the
	average. For short percussive sounds, this is usually more discriminative
	than the plain mean MFCCs."""


@dataclasses.dataclass(frozen=True)
class LevelResult:

	"""Peak and RMS amplitude of the recording, measured on the float32 mono signal.

	Both values are in [0.0, 1.0] because the mono signal is normalised to unity
	by to_mono_float() before analysis. Used for per-sample gain normalisation
	during playback — NOT included in similarity scoring.

	At playback time, a normalization gain can be computed as:
	    gain = target_rms / level.rms
	then scaled by MIDI velocity and clamped by level.peak to prevent clipping.
	"""

	peak: float
	"""Peak absolute amplitude: max(abs(signal)).
	0.0 = silence, 1.0 = full digital scale. Used to set the anti-clipping
	ceiling so no combination of normalization gain and velocity can exceed 0 dBFS."""

	rms: float
	"""Root mean square amplitude: sqrt(mean(signal²)).
	0.0 = silence. Represents perceived average loudness and is the primary
	value used to normalise levels across samples with different recording volumes."""


@dataclasses.dataclass(frozen=True)
class BandEnergyResult:

	"""Per-band energy distribution and decay rates for drum-type classification.

	Splits the spectrum into four frequency bands and captures how much energy
	lives in each band (energy_fractions) and how quickly that energy decays
	after the onset peak (decay_rates). Together these directly encode the
	physical signatures of common drum types:

	  kick drum  — sub-bass dominant (fraction[0] high), slow bass decay
	  snare drum — low-mid body + high-mid crack, faster decay
	  hi-hat     — presence dominated (fraction[3] high), very fast decay

	Band order (index 0–3):
	  0: sub-bass  (20–250 Hz)
	  1: low-mid   (250–2000 Hz)
	  2: high-mid  (2000–6000 Hz)
	  3: presence  (6000+ Hz)

	Used in the similarity feature vector as Group 5 (8 values total).
	"""

	energy_fractions: tuple[float, ...]
	"""4-element tuple: fraction of total energy in each frequency band.
	Values are in [0.0, 1.0] and sum to approximately 1.0.
	0.0 = no energy in this band; 1.0 = all energy concentrated here."""

	decay_rates: tuple[float, ...]
	"""4-element tuple: per-band decay rate, log-normalised to [0.0, 1.0].
	Measures time from peak band energy to -20 dB (10 % of peak).
	0.0 = instant decay (≤ 1 ms — sharp transient); 1.0 = very long decay (≥ 2 s)."""


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
			log_attack_time=0.0,
			spectral_flux=0.0,
		)

	mono = to_mono_float(audio, bit_depth)

	return analyze_mono(mono, params)


def analyze_mono (
	mono: numpy.ndarray,
	params: AnalysisParams,
	*,
	_pyin_voiced_flag: numpy.ndarray | None = None,
	_hpss_ratio: float | None = None,
) -> AnalysisResult:

	"""Compute analysis metrics from a pre-normalised float32 mono array.

	Use this entry point when audio is already normalised to [-1, 1] — for
	example when reading a file via soundfile. For integer PCM from the live
	capture pipeline, use analyze() instead, which handles the conversion.

	Args:
		mono:               Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params:             Pre-computed FFT parameters from compute_params().
		_pyin_voiced_flag:  Internal — pre-computed voiced/unvoiced boolean flag
		                    array from pyin, injected by analyze_all() to share the
		                    pyin computation between spectral and pitch analysis.
		                    External callers should omit this argument; it will be
		                    computed automatically when None.
		_hpss_ratio:        Internal — pre-computed harmonic ratio from HPSS,
		                    injected by analyze_all() to avoid redundant HPSS
		                    decomposition.  When None, HPSS is computed internally.

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
			log_attack_time=0.0,
			spectral_flux=0.0,
		)

	# Clamp n_fft to the signal length for short recordings. librosa zero-pads
	# when n_fft > len(signal), but emits a UserWarning. Clamping avoids the
	# warning without sacrificing accuracy: a 606-sample signal has no real
	# spectral information beyond 606 bins regardless of the window size.
	# Also clamp hop_length so it never exceeds n_fft.
	effective_n_fft = min(params.n_fft, len(mono))
	effective_hop = min(params.hop_length, effective_n_fft)
	params = dataclasses.replace(params, n_fft=effective_n_fft, hop_length=effective_hop)

	# Pre-compute the STFT once and share it across all spectral feature helpers.
	# Each helper previously called librosa internals that recomputed the STFT
	# independently, causing 5+ redundant FFT passes per recording.
	# - D: complex STFT, needed for HPSS (to reconstruct harmonic time-domain)
	# - S_magnitude: |D|, needed for centroid, bandwidth, contrast, flatness
	# - S_power: |D|^2, needed for spectral_flatness (expects power spectrogram)
	D = librosa.stft(mono, n_fft=params.n_fft, hop_length=params.hop_length)
	S_magnitude = numpy.abs(D)
	S_power = S_magnitude ** 2

	flatness = librosa.feature.spectral_flatness(S=S_power)

	attack, release = _compute_attack_release(mono, params)
	centroid = _compute_spectral_centroid(params, S_magnitude)
	bandwidth = _compute_spectral_bandwidth(params, S_magnitude)
	zcr = _compute_zcr(mono, params)
	if _hpss_ratio is not None:
		harmonic_ratio = _hpss_ratio
	else:
		harmonic_ratio, _, _ = _compute_hpss(mono, params, D)
	contrast = _compute_spectral_contrast(params, S_magnitude)
	voiced_fraction = _compute_voiced_fraction(mono, params, pyin_voiced_flag=_pyin_voiced_flag)
	log_attack_time, spectral_flux = _compute_spectral_onset_features(params, S_magnitude)

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
		log_attack_time=log_attack_time,
		spectral_flux=spectral_flux,
	)


def _refine_onsets_to_attacks (
	mono:        numpy.ndarray,
	onset_times: tuple[float, ...],
	sample_rate: int,
	hop_length:  int,
) -> tuple[float, ...]:

	"""Refine librosa onset times to sample-accurate attack start positions.

	librosa.onset.onset_detect() returns the frame where spectral flux peaks,
	which is typically 10-30 ms after the percussive transient begins.  This
	function searches the amplitude envelope near each onset to find where
	the energy first rises above the local noise floor — the moment a
	musician would perceive as "the hit".

	This two-stage approach (librosa coarse → amplitude-envelope refinement)
	gives ~0.7 ms precision — far tighter than Rubber Band's internal
	transient detector (--detector-perc, R2-only, frame-level).  The
	time-stretch pipeline uses these refined attack_times to build an
	explicit time map, bypassing Rubber Band's detection entirely.

	Algorithm for each onset:
	  1. Define a search region bounded by the midpoint to the previous onset
	     (prevents bleeding into the prior hit's tail) and a maximum of 50 ms
	     backward (the physical upper bound on STFT lag for the analysis window).
	  2. Find the valley (minimum amplitude) in the first 60% of the region —
	     this is the noise floor between hits.
	  3. Find the peak amplitude near the onset.
	  4. Threshold = valley + 20% of (peak − valley).
	  5. Scan forward from the valley to find the first threshold crossing.

	The 32-sample envelope window (~0.7 ms at 44100 Hz) gives near-sample
	precision without being dominated by individual waveform cycles.  The 20%
	threshold corresponds to ~-14 dB below peak — well above typical noise.

	Args:
		mono:        float32 mono audio, shape (n_samples,).
		onset_times: Onset positions in seconds from onset_detect().
		sample_rate: Hz.
		hop_length:  Analysis hop length (used to bound the peak search).

	Returns:
		Tuple of attack start times in seconds, same length as onset_times.
		Each value is <= the corresponding onset time.
	"""

	if len(onset_times) == 0:
		return ()

	# Short-window amplitude envelope for near-sample precision.
	_ENVELOPE_WINDOW = 32
	abs_audio = numpy.abs(mono)
	cumsum = numpy.concatenate(
		[numpy.zeros(1, dtype=mono.dtype), numpy.cumsum(abs_audio)],
	)
	n_env = len(abs_audio) - _ENVELOPE_WINDOW + 1

	if n_env < 1:
		return onset_times

	envelope = (cumsum[_ENVELOPE_WINDOW:_ENVELOPE_WINDOW + n_env] - cumsum[:n_env]) / _ENVELOPE_WINDOW
	# Pad to match audio length so sample indices correspond directly.
	envelope = numpy.concatenate([envelope, numpy.zeros(_ENVELOPE_WINDOW - 1, dtype=envelope.dtype)])

	max_search = int(0.050 * sample_rate)  # 50 ms
	_THRESHOLD_RATIO = 0.20

	attacks: list[float] = []

	for i, onset_sec in enumerate(onset_times):
		onset_sample = int(onset_sec * sample_rate)

		# Backward limit: midpoint to previous onset, capped at 50 ms.
		if i > 0:
			prev_sample = int(onset_times[i - 1] * sample_rate)
			midpoint = (prev_sample + onset_sample) // 2
			search_start = max(midpoint, onset_sample - max_search)
		else:
			search_start = max(0, onset_sample - max_search)

		search_end = min(len(envelope), onset_sample + hop_length)

		if search_end <= search_start + 1:
			attacks.append(onset_sec)
			continue

		region = envelope[search_start:search_end]

		# Valley: minimum in the first 60% of the region (before the rise).
		valley_zone = max(1, int(len(region) * 0.6))
		valley_idx = int(numpy.argmin(region[:valley_zone]))
		valley_value = float(region[valley_idx])

		# Peak: maximum in the full region.
		peak_value = float(numpy.max(region))

		if peak_value <= valley_value:
			attacks.append(onset_sec)
			continue

		threshold = valley_value + _THRESHOLD_RATIO * (peak_value - valley_value)

		# Scan forward from the valley to find the threshold crossing.
		attack_idx = len(region) - 1

		for s in range(valley_idx, len(region)):
			if region[s] >= threshold:
				attack_idx = s
				break

		attack_sample = min(search_start + attack_idx, onset_sample)
		attacks.append(attack_sample / sample_rate)

	return tuple(attacks)


def analyze_rhythm (
	mono: numpy.ndarray,
	params: AnalysisParams,
	rhythm_cfg: subsample.config.AnalysisConfig,
	*,
	_percussive: numpy.ndarray | None = None,
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
		mono:          Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params:        Pre-computed FFT parameters from compute_params().
		rhythm_cfg:    Configurable tempo priors from AnalysisConfig.
		_percussive:   Internal — HPSS percussive component for cleaner onset
		               detection.  When provided, onset_detect runs on this
		               instead of the full mix, reducing false positives from
		               harmonic energy changes.  Attack refinement still uses
		               the full mono signal.  When None, onset detection runs
		               on the full mix (backward-compatible behaviour).

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
			attack_times=(),
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
	# Unlike beat_times, onsets are not quantized — they mark exact moments
	# where energy rises abruptly, making them useful for slicing multi-hit
	# recordings into individual one-shots.
	#
	# When the HPSS percussive component is available, onset detection runs on
	# it instead of the full mix.  This removes harmonic energy changes (chord
	# transitions, vibrato, note bends) that cause false-positive onsets,
	# giving cleaner transient peaks for beat-quantized time-stretching.
	# Attack refinement (_refine_onsets_to_attacks) still uses the full mono
	# signal so the amplitude-envelope threshold reflects the actual hit.
	onset_source = _percussive if _percussive is not None else mono

	onset_times_raw: numpy.ndarray = librosa.onset.onset_detect(
		y=onset_source,
		sr=params.sample_rate,
		hop_length=params.hop_length,
		units='time',
	)

	onset_times: tuple[float, ...] = tuple(float(t) for t in onset_times_raw)

	# Refine each librosa onset to the sample-accurate attack start — the
	# moment the transient becomes audible.  Used by the time-stretch handler
	# for beat-grid alignment.
	attack_times = _refine_onsets_to_attacks(
		mono, onset_times, params.sample_rate, params.hop_length,
	)

	return RhythmResult(
		tempo_bpm=tempo_bpm,
		beat_times=beat_times,
		pulse_curve=pulse_curve,
		pulse_peak_times=pulse_peak_times,
		onset_times=onset_times,
		attack_times=attack_times,
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
		f"  log_attack={result.log_attack_time:.3f}"
		f"  flux={result.spectral_flux:.3f}"
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
		f"  stability={result.pitch_stability:.3f}st"
		f"  voiced_frames={result.voiced_frame_count}"
	)


def has_stable_pitch (
	spectral: AnalysisResult,
	pitch: PitchResult,
	duration: float,
) -> bool:

	"""Return True if this sample has a single, stable, confident pitch.

	Samples that pass this test are suitable for pitch-shifting across a keyboard
	range — each one maps cleanly to one note without vibrato, bends, or noise.
	Harmonics of the fundamental are fine; a piano note is full of them and still
	maps to a single key.

	This is a compound query over existing analysis fields; thresholds live here
	and can be tightened without re-analysing samples.

	Decision criteria (all seven must hold):

	  dominant_pitch_hz > 0        — pyin found a fundamental at all
	  voiced_fraction   > 0.5      — pitched content in more than half the frames
	  voiced_frame_count >= 5      — at least ~60 ms of voiced content; excludes
	                                 noise bursts where a single pyin frame is
	                                 flagged as voiced (making pitch_stability
	                                 trivially 0.0 — not a sign of stability)
	  pitch_confidence  > 0.5      — pyin was confident about the pitch; rejects
	                                 samples where the 65 Hz fmin floor is returned
	                                 as a fallback rather than a real detection
	  pitch_stability   < 0.5 st   — F0 varies less than half a semitone (std)
	                                 across voiced frames; excludes vibrato,
	                                 pitch bends, and multi-pitch signals
	  harmonic_ratio    > 0.4      — more harmonic than percussive energy
	                                 (HPSS-based); excludes drum hits and noise
	  duration          >= 0.1 s   — at least 100 ms long; sub-100 ms bursts are
	                                 not useful keyboard samples regardless of pitch

	Args:
		spectral: AnalysisResult for the sample (voiced_fraction, harmonic_ratio).
		pitch:    PitchResult for the sample (dominant_pitch_hz, pitch_confidence,
		          voiced_frame_count, pitch_stability).
		duration: Recording length in seconds.

	Returns:
		True if all seven criteria are met, False otherwise.
	"""

	return (
		pitch.dominant_pitch_hz > 0.0
		and spectral.voiced_fraction > 0.5
		and pitch.voiced_frame_count >= 5
		and pitch.pitch_confidence > 0.5
		and pitch.pitch_stability < 0.5
		and spectral.harmonic_ratio > 0.4
		and duration >= 0.1
	)


def compute_level (mono: numpy.ndarray) -> LevelResult:

	"""Compute peak and RMS amplitude from a normalised float32 mono signal.

	Both metrics are derived from the same float32 signal produced by
	to_mono_float(), so values are naturally in [0.0, 1.0]. The computation
	is trivially cheap and always runs alongside the other analyses.

	Args:
		mono: Shape (n_frames,), dtype float32, values in [-1.0, 1.0].

	Returns:
		LevelResult with peak and rms in [0.0, 1.0].
		Both are 0.0 for an empty or silent signal.
	"""

	if mono.size == 0:
		return LevelResult(peak=0.0, rms=0.0)

	peak = float(numpy.max(numpy.abs(mono)))
	rms  = float(numpy.sqrt(numpy.mean(mono.astype(numpy.float64) ** 2)))

	return LevelResult(peak=peak, rms=rms)


def format_level_result (result: LevelResult) -> str:

	"""Return a single-line human-readable summary of the level analysis.

	Shows peak and RMS both as linear [0, 1] values and as dBFS equivalents,
	so the log output is useful to both engineers (dBFS) and code (linear).

	Args:
		result: Computed level metrics.

	Returns:
		String of the form:
		"peak=0.9512 (-0.4dBFS)  rms=0.2345 (-12.6dBFS)"
		or "peak=0.0000 (-infdBFS)  rms=0.0000 (-infdBFS)" for silence.
	"""

	def _dbfs (v: float) -> str:
		if v <= 0.0:
			return "-inf"
		db = 20.0 * math.log10(v)
		return f"{db:.1f}"

	return (
		f"peak={result.peak:.4f} ({_dbfs(result.peak)}dBFS)"
		f"  rms={result.rms:.4f} ({_dbfs(result.rms)}dBFS)"
	)


def analyze_band_energy (
	mono: numpy.ndarray,
	params: AnalysisParams,
) -> BandEnergyResult:

	"""Compute per-band energy fractions and decay rates from a mono signal.

	Splits the spectrum into four frequency bands (sub-bass, low-mid, high-mid,
	presence) and measures how much energy lives in each band and how quickly
	that energy decays after the onset peak.

	The energy fractions sum to approximately 1.0 and directly encode the
	frequency distribution: a kick drum has most energy in sub-bass; a hi-hat
	has most energy in the presence band. The decay rates capture how each
	band behaves over time, further discriminating drum types.

	Args:
		mono:   Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params: Pre-computed FFT parameters from compute_params().

	Returns:
		BandEnergyResult with 4-element energy_fractions and decay_rates tuples.
		All-zero values are returned for empty or silent signals.
	"""

	if mono.shape[0] == 0:
		return BandEnergyResult(
			energy_fractions=(0.0,) * _N_BANDS,
			decay_rates=(0.0,) * _N_BANDS,
		)

	# Clamp to signal length for short recordings, matching analyze_mono().
	effective_n_fft = min(params.n_fft, len(mono))
	effective_hop   = min(params.hop_length, effective_n_fft)

	# Compute STFT once; reuse power spectrogram for all bands.
	D       = librosa.stft(mono, n_fft=effective_n_fft, hop_length=effective_hop)
	S_power = numpy.abs(D) ** 2

	# Map FFT bin indices to their centre frequencies.
	freqs = librosa.fft_frequencies(sr=params.sample_rate, n_fft=effective_n_fft)

	total_energy = float(S_power.sum())

	energy_fractions: list[float] = []
	decay_rates:      list[float] = []

	for band_low, band_high in _BAND_EDGES:

		# Clamp upper edge to Nyquist; band may be empty at low sample rates.
		effective_high = min(band_high, params.sample_rate / 2.0)
		band_mask = (freqs >= band_low) & (freqs < effective_high)

		if not band_mask.any():
			energy_fractions.append(0.0)
			decay_rates.append(0.0)
			continue

		# Sum power across the band's frequency bins per frame, then take RMS.
		band_power_per_frame = S_power[band_mask, :].sum(axis=0)
		band_rms = numpy.sqrt(band_power_per_frame)

		# --- Energy fraction: proportion of total power in this band ---
		band_total = float(band_power_per_frame.sum())
		fraction   = band_total / total_energy if total_energy > 1e-20 else 0.0
		energy_fractions.append(float(numpy.clip(fraction, 0.0, 1.0)))

		# --- Decay rate: time from peak to -20 dB (10 % of peak) ---
		peak_val = float(numpy.max(band_rms))

		if peak_val < 1e-9:
			decay_rates.append(0.0)
			continue

		peak_frame = int(numpy.argmax(band_rms))
		threshold  = peak_val * _ACTIVE_THRESHOLD_RATIO
		post_peak  = band_rms[peak_frame:]

		below = numpy.where(post_peak < threshold)[0]

		if below.size > 0:
			decay_frames = int(below[0])
		else:
			# Energy never drops to threshold — treat as maximum decay duration.
			decay_frames = len(post_peak)

		decay_seconds = decay_frames * effective_hop / params.sample_rate
		decay_rate    = _log_normalize(decay_seconds, _ATTACK_RELEASE_MIN_S, _ATTACK_RELEASE_MAX_S)
		decay_rates.append(decay_rate)

	return BandEnergyResult(
		energy_fractions=tuple(energy_fractions),
		decay_rates=tuple(decay_rates),
	)


def format_band_energy_result (result: BandEnergyResult) -> str:

	"""Return a single-line human-readable summary of band energy analysis.

	Shows energy fraction and decay rate for each of the four bands.

	Args:
		result: Computed band energy metrics.

	Returns:
		String of the form:
		"sub=0.45(d=0.7)  mid=0.30(d=0.4)  hi=0.15(d=0.2)  air=0.10(d=0.1)"
	"""

	labels = ("sub", "mid", "hi", "air")

	parts = [
		f"{label}={frac:.2f}(d={decay:.2f})"
		for label, frac, decay in zip(
			labels, result.energy_fractions, result.decay_rates
		)
	]

	return "  ".join(parts)


def analyze_pitch (
	mono: numpy.ndarray,
	params: AnalysisParams,
	*,
	_pyin_result: _PYINResult | None = None,
) -> tuple[PitchResult, TimbreResult]:

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
		(PitchResult, TimbreResult) — pitch properties and timbral fingerprints.
	"""

	_empty_timbre = TimbreResult(
		mfcc=tuple(0.0 for _ in range(_N_MFCC)),
		mfcc_delta=tuple(0.0 for _ in range(_N_MFCC)),
		mfcc_onset=tuple(0.0 for _ in range(_N_MFCC)),
	)

	# Clamp n_fft to signal length, matching the same guard in analyze_mono().
	# Without this, librosa.feature.mfcc() emits a UserWarning for short
	# recordings when n_fft exceeds the signal length.
	if len(mono) > 0:
		effective_n_fft = min(params.n_fft, len(mono))
		effective_hop = min(params.hop_length, effective_n_fft)
		params = dataclasses.replace(params, n_fft=effective_n_fft, hop_length=effective_hop)

	# pyin requires at least one period of fmin to fit in the frame.
	# Use _run_pyin() so this check and the pyin call are in one place.
	pyin = _pyin_result if _pyin_result is not None else _run_pyin(mono, params)

	if pyin is None:
		return (
			PitchResult(
				dominant_pitch_hz=0.0,
				pitch_confidence=0.0,
				chroma_profile=tuple(0.0 for _ in range(12)),
				dominant_pitch_class=-1,
				pitch_stability=0.0,
				voiced_frame_count=0,
			),
			_empty_timbre,
		)

	# --- pyin: probabilistic fundamental frequency estimation ---
	# voiced_flag is True for frames where a pitch was detected with reasonable
	# confidence. f0_hz is NaN for unvoiced frames — we ignore those.
	f0_hz, voiced_flag, voiced_probs = pyin

	voiced_f0 = f0_hz[voiced_flag]
	dominant_pitch_hz = float(numpy.median(voiced_f0)) if voiced_f0.size > 0 else 0.0

	voiced_p = voiced_probs[voiced_flag]
	pitch_confidence = float(numpy.mean(voiced_p)) if voiced_p.size > 0 else 0.0

	# Pitch stability: std dev of voiced F0 in semitones (MIDI units).
	# Semitone scale is logarithmic and frequency-independent, so 0.5 semitones
	# means the same thing at 100 Hz or 1000 Hz.
	# Only voiced frames are included; unvoiced gaps don't count as instability.
	voiced_frame_count = int(voiced_f0.size)

	if voiced_f0.size > 1:
		pitch_stability = float(numpy.std(librosa.hz_to_midi(voiced_f0)))
	else:
		pitch_stability = 0.0

	# --- chroma_cqt: pitch class energy (constant-Q, more accurate than STFT) ---
	# Returns shape (12, n_frames). Mean across time gives the average energy in
	# each pitch class over the whole recording.
	#
	# chroma_cqt uses multi-rate CQT processing: it internally downsamples the
	# signal by 2x for each lower octave. Each downsampled level is analysed with
	# an internal n_fft=1024, which triggers an "n_fft too large" UserWarning for
	# short signals. This is expected and harmless — suppressed by the module-level
	# warnings.filterwarnings("ignore", message="n_fft=") at the top of this file.
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
	# Returns shape (_N_MFCC, n_frames). Three aggregations are computed:
	#
	#   mfcc       — mean across time: coarse timbral character, stable for
	#                sustained tones; use cosine distance for similarity.
	#
	#   mfcc_delta — mean of first-order temporal differences: captures how
	#                the timbre changes over time rather than its average. A
	#                snare and a tom can have similar mean MFCCs but diverge
	#                strongly in delta-MFCC, which encodes the attack-to-decay
	#                spectral shift.
	#
	#   mfcc_onset — exponential decay weighting from the first frame: onset
	#                frames contribute most; the decay tail is downweighted.
	#                For percussive sounds the first ~50 ms carry the identity;
	#                for sustained tones this converges toward mfcc.
	mfcc_raw = librosa.feature.mfcc(
		y=mono,
		sr=params.sample_rate,
		n_mfcc=_N_MFCC,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	n_mfcc_frames = mfcc_raw.shape[1]

	# Guard: a signal shorter than one MFCC frame produces an empty mfcc_raw.
	# numpy.mean on an empty axis emits "Mean of empty slice"; just return zeros.
	if n_mfcc_frames == 0:
		return (
			PitchResult(
				dominant_pitch_hz=dominant_pitch_hz,
				pitch_confidence=pitch_confidence,
				chroma_profile=chroma_profile,
				dominant_pitch_class=dominant_pitch_class,
				pitch_stability=pitch_stability,
				voiced_frame_count=voiced_frame_count,
			),
			_empty_timbre,
		)

	mfcc: tuple[float, ...] = tuple(float(v) for v in numpy.mean(mfcc_raw, axis=1))

	# Delta MFCCs: frame-to-frame first-order differences, then time-averaged.
	# librosa.feature.delta() requires width ≥ 3 (odd) and ≤ n_frames.
	# Very short recordings can have fewer than 3 MFCC frames — in that case
	# there is no meaningful temporal gradient, so return all zeros.
	if n_mfcc_frames < 3:
		mfcc_delta: tuple[float, ...] = tuple(0.0 for _ in range(_N_MFCC))
	else:
		delta_width = min(9, n_mfcc_frames)
		if delta_width % 2 == 0:
			delta_width -= 1
		mfcc_delta_raw = librosa.feature.delta(mfcc_raw, width=delta_width)
		mfcc_delta = tuple(float(v) for v in numpy.mean(mfcc_delta_raw, axis=1))

	# Onset-weighted MFCCs: exponential decay from frame 0.
	# decay_frames ≈ how many MFCC frames span _ONSET_DECAY_MS milliseconds.
	decay_frames = max(1.0, _ONSET_DECAY_MS * params.sample_rate / (1000.0 * params.hop_length))
	weights = numpy.exp(-numpy.arange(n_mfcc_frames) / decay_frames)
	weights /= weights.sum()
	mfcc_onset: tuple[float, ...] = tuple(
		float(v) for v in (mfcc_raw * weights[numpy.newaxis, :]).sum(axis=1)
	)

	return (
		PitchResult(
			dominant_pitch_hz=dominant_pitch_hz,
			pitch_confidence=pitch_confidence,
			chroma_profile=chroma_profile,
			dominant_pitch_class=dominant_pitch_class,
			pitch_stability=pitch_stability,
			voiced_frame_count=voiced_frame_count,
		),
		TimbreResult(
			mfcc=mfcc,
			mfcc_delta=mfcc_delta,
			mfcc_onset=mfcc_onset,
		),
	)


def analyze_all (
	mono: numpy.ndarray,
	params: AnalysisParams,
	rhythm_cfg: subsample.config.AnalysisConfig,
) -> tuple[AnalysisResult, RhythmResult, PitchResult, TimbreResult, LevelResult, BandEnergyResult]:

	"""Run all analyses with shared STFT, HPSS, and pyin computation.

	Preferred entry point for the recorder and any code that needs all results.
	Computes the STFT and HPSS decomposition once and shares the results:
	  - harmonic audio → pyin for cleaner pitch detection
	  - percussive audio → onset detection for cleaner transient detection
	  - harmonic_ratio → spectral metrics (avoids redundant HPSS in analyze_mono)
	  - pyin → shared between spectral (voiced_fraction) and pitch analysis

	Args:
		mono:       Shape (n_frames,), dtype float32, values in [-1.0, 1.0].
		params:     Pre-computed FFT parameters from compute_params().
		rhythm_cfg: Configurable tempo priors from AnalysisConfig.

	Returns:
		(spectral, rhythm, pitch, timbre, level, band_energy) — six result dataclasses.
	"""

	# Compute HPSS once.  The harmonic component gives pyin a cleaner pitch
	# signal (no percussive interference); the percussive component gives
	# onset_detect cleaner transient peaks (no harmonic false positives).
	# analyze_mono gets the pre-computed ratio so it skips its own HPSS call.
	effective_n_fft = min(params.n_fft, len(mono))
	effective_hop = min(params.hop_length, effective_n_fft)
	hpss_params = dataclasses.replace(params, n_fft=effective_n_fft, hop_length=effective_hop)

	if mono.shape[0] > 0:
		D = librosa.stft(mono, n_fft=hpss_params.n_fft, hop_length=hpss_params.hop_length)
		harmonic_ratio, harmonic_audio, percussive_audio = _compute_hpss(mono, hpss_params, D)
	else:
		harmonic_ratio = 0.0
		harmonic_audio = mono
		percussive_audio = mono

	# Run pyin on the harmonic component for cleaner pitch detection.
	# Percussive energy in the full mix degrades pyin's confidence and can
	# pull the detected F0 away from the true fundamental.
	# Guard: istft can produce NaN for very short signals — fall back to
	# the full mono mix if the harmonic audio is not finite.
	if numpy.all(numpy.isfinite(harmonic_audio)):
		pyin = _run_pyin(harmonic_audio, params)
	else:
		pyin = _run_pyin(mono, params)

	safe_percussive = percussive_audio if numpy.all(numpy.isfinite(percussive_audio)) else None
	rhythm      = analyze_rhythm(mono, params, rhythm_cfg, _percussive=safe_percussive)
	spectral    = analyze_mono(mono, params, _pyin_voiced_flag=pyin[1] if pyin is not None else None, _hpss_ratio=harmonic_ratio)
	pitch, timbre = analyze_pitch(mono, params, _pyin_result=pyin)
	level       = compute_level(mono)
	band_energy = analyze_band_energy(mono, params)

	return spectral, rhythm, pitch, timbre, level, band_energy


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


def _compute_spectral_onset_features (
	params: AnalysisParams,
	S_magnitude: numpy.ndarray,
) -> tuple[float, float]:

	"""Compute log-attack time and mean spectral flux from onset strength.

	Both use librosa.onset.onset_strength() — the spectral flux (half-wave
	rectified frame-to-frame spectral difference) — rather than RMS energy.
	Spectral flux detects the start of spectral change (e.g. the snare wire
	rattle) before that energy peaks in the RMS envelope, making it more
	sensitive to percussive transients.

	log_attack_time: time from first active onset_strength frame to the peak,
	    measured in seconds, log-normalised the same way as attack.
	    0.0 = instantaneous spectral onset (≤ 1 ms), 1.0 = very slow (≥ 2 s).

	spectral_flux: mean onset strength across all frames, log-normalised.
	    0.0 = barely changing spectrum (static tone), 1.0 = rapidly changing
	    spectrum (busy percussive audio with frequent spectral jumps).

	Args:
		params:      FFT params — hop_length and sample_rate control frame timing.
		S_magnitude: Pre-computed magnitude spectrogram |STFT|, passed from
		             analyze_mono() to avoid a redundant STFT computation.

	Returns:
		(log_attack_time, spectral_flux), both in [0.0, 1.0].
	"""

	# Pass the pre-computed spectrogram directly so no STFT is re-run here.
	# onset_strength expects a power spectrogram or mel spectrogram but accepts
	# magnitude; it internally normalises anyway so the result is comparable.
	onset_env = librosa.onset.onset_strength(
		S=S_magnitude,
		sr=params.sample_rate,
	)

	if onset_env.size == 0 or float(numpy.max(onset_env)) < 1e-8:
		return (0.0, 0.0)

	# --- log attack time from spectral flux peak ---
	peak_idx = int(numpy.argmax(onset_env))
	peak_val = float(onset_env[peak_idx])
	threshold = peak_val * _ACTIVE_THRESHOLD_RATIO

	active = numpy.where(onset_env >= threshold)[0]
	seconds_per_frame = params.hop_length / params.sample_rate
	first_active = int(active[0])
	attack_seconds = (peak_idx - first_active) * seconds_per_frame
	log_attack_time = _log_normalize(attack_seconds, _ATTACK_RELEASE_MIN_S, _ATTACK_RELEASE_MAX_S)

	# --- mean spectral flux, log-normalised ---
	spectral_flux = _log_normalize(float(numpy.mean(onset_env)), _FLUX_MIN, _FLUX_MAX)

	return (log_attack_time, spectral_flux)


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
	params: AnalysisParams,
	S_magnitude: numpy.ndarray,
) -> float:

	"""Measure the centre of mass of the frequency spectrum (bassy vs trebly).

	Returns a value in [0, 1] via log-frequency normalisation between 20 Hz
	and Nyquist.  Using log scale matches human pitch perception: an octave
	jump is always an octave regardless of register, so the midpoint of the
	scale (0.5) sits at sqrt(20 * nyquist) Hz — roughly the middle of the
	musical range (~1 kHz at 44.1 kHz sample rate).

	Args:
		params:      FFT params (sample_rate used for Hz normalisation).
		S_magnitude: Pre-computed magnitude spectrogram |STFT|, shape (n_fft//2+1, n_frames).

	Returns:
		Centroid score in [0.0, 1.0]. 0 = bassy, 1 = trebly.
	"""

	centroid = librosa.feature.spectral_centroid(
		S=S_magnitude,
		sr=params.sample_rate,
	)

	# centroid has shape (1, n_frames); mean over frames gives a single Hz value
	mean_hz = float(numpy.mean(centroid))

	nyquist = params.sample_rate / 2.0

	return _log_normalize(mean_hz, _FREQ_MIN_HZ, nyquist)


def _compute_spectral_bandwidth (
	params: AnalysisParams,
	S_magnitude: numpy.ndarray,
) -> float:

	"""Measure how spread out the frequency content is (narrow vs wide).

	Spectral bandwidth is the weighted standard deviation of the spectrum
	around the spectral centroid — wide means energy is spread across many
	frequencies (complex/noisy), narrow means it is concentrated (tonal/pure).

	Normalised with the same log-frequency scale as spectral_centroid so the
	two metrics are directly comparable in magnitude.

	Args:
		params:      FFT params (sample_rate used for Hz normalisation).
		S_magnitude: Pre-computed magnitude spectrogram |STFT|, shape (n_fft//2+1, n_frames).

	Returns:
		Bandwidth score in [0.0, 1.0]. 0 = narrow/tonal, 1 = wide/complex.
	"""

	bandwidth = librosa.feature.spectral_bandwidth(
		S=S_magnitude,
		sr=params.sample_rate,
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


def _compute_hpss (
	mono: numpy.ndarray,
	params: AnalysisParams,
	D: numpy.ndarray,
) -> tuple[float, numpy.ndarray, numpy.ndarray]:

	"""Separate a signal into harmonic and percussive components via HPSS.

	Uses librosa.decompose.hpss on the pre-computed complex STFT to avoid a
	redundant FFT pass.  Returns the harmonic_ratio metric *and* both
	separated time-domain signals so callers can reuse them:
	  - harmonic_audio is used by pyin for cleaner pitch detection
	  - percussive_audio is used by onset_detect for cleaner transient detection

	The two-stage approach (librosa coarse onset on percussive → amplitude-
	envelope refinement on full mono) gives ~0.7 ms precision, far tighter
	than Rubber Band's internal transient detector.

	Args:
		mono:   Float32 audio, shape (n_frames,). Used for total energy and
		        as the length reference for istft reconstruction.
		params: FFT params (hop_length used for istft reconstruction).
		D:      Pre-computed complex STFT, shape (n_fft//2+1, n_frames).

	Returns:
		(harmonic_ratio, harmonic_audio, percussive_audio)
		harmonic_ratio: float in [0.0, 1.0]. 0 = purely percussive, 1 = purely harmonic.
		harmonic_audio: float32, shape (n_frames,). Tonal/sustained content.
		percussive_audio: float32, shape (n_frames,). Transients/clicks/drum hits.
	"""

	harmonic_D, percussive_D = librosa.decompose.hpss(D)

	harmonic = librosa.istft(
		harmonic_D, hop_length=params.hop_length, length=len(mono),
	)
	percussive = librosa.istft(
		percussive_D, hop_length=params.hop_length, length=len(mono),
	)

	energy_total = float(numpy.sum(mono ** 2))

	if energy_total < 1e-16:
		return 0.0, harmonic.astype(numpy.float32), percussive.astype(numpy.float32)

	energy_harmonic = float(numpy.sum(harmonic ** 2))

	if not numpy.isfinite(energy_harmonic):
		return 0.0, harmonic.astype(numpy.float32), percussive.astype(numpy.float32)

	ratio = min(energy_harmonic / energy_total, 1.0)
	return ratio, harmonic.astype(numpy.float32), percussive.astype(numpy.float32)


def _compute_spectral_contrast (
	params: AnalysisParams,
	S_magnitude: numpy.ndarray,
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
		params:      FFT params (sample_rate for sub-band Hz boundaries; n_fft for
		             degenerate-window guard).
		S_magnitude: Pre-computed magnitude spectrogram |STFT|, shape (n_fft//2+1, n_frames).

	Returns:
		Contrast score in [0.0, 1.0]. 0 = flat spectrum, 1 = strong peaks.
	"""

	# Guard against a degenerate n_fft (< 64 bins → frequency resolution too
	# coarse for spectral_contrast's sub-band indexing).
	# Note: the n_fft < signal_length check from the y= path is not needed here
	# because analyze_mono already clamps params.n_fft to len(mono), so the
	# pre-computed S_magnitude always has at least n_fft frames of coverage.
	if params.n_fft < 64:
		return 0.0

	contrast = librosa.feature.spectral_contrast(
		S=S_magnitude,
		sr=params.sample_rate,
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
