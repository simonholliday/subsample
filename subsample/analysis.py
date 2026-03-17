"""Audio analysis for Subsample.

Computes perceptual metrics on completed recordings before they are written
to disk. Designed to run in the writer thread so the main capture loop is
never delayed by analysis work.

Analysis parameters (FFT window size, hop length) are derived once from the
audio config at startup and reused for every recording.
"""

import dataclasses
import math

import librosa
import numpy

import subsample.config


@dataclasses.dataclass(frozen=True)
class AnalysisParams:

	"""FFT parameters derived from the audio config, computed once at startup."""

	n_fft: int
	hop_length: int
	sample_rate: int


@dataclasses.dataclass(frozen=True)
class AnalysisResult:

	"""Metrics computed from a single recording.

	Fields are intentionally minimal for now; extend here as new metrics are
	added so the rest of the pipeline receives a single structured object.
	"""

	spectral_flatness: float
	"""Wiener entropy in [0.0, 1.0]. 0.0 = perfectly tonal, 1.0 = pure noise."""


def compute_params (audio_cfg: subsample.config.AudioConfig) -> AnalysisParams:

	"""Derive FFT analysis parameters from the audio configuration.

	Targets a ~46ms analysis window (the audio analysis standard at 44.1 kHz),
	rounded to the nearest power of two for FFT efficiency. This keeps the
	window duration roughly constant regardless of sample rate.

	Args:
		audio_cfg: The audio section of the loaded config.

	Returns:
		AnalysisParams suitable for passing to analyze() on every recording.

	Examples:
		11025 Hz → n_fft=512,  hop=128
		22050 Hz → n_fft=1024, hop=256
		44100 Hz → n_fft=2048, hop=512
		48000 Hz → n_fft=2048, hop=512
		96000 Hz → n_fft=4096, hop=1024
	"""

	# Reference window: 2048 samples at 44100 Hz ≈ 0.04644 s
	target_seconds = 2048 / 44100
	n_fft = int(2 ** round(math.log2(target_seconds * audio_cfg.sample_rate)))
	hop_length = n_fft // 4

	return AnalysisParams(
		n_fft=n_fft,
		hop_length=hop_length,
		sample_rate=audio_cfg.sample_rate,
	)


def analyze (
	audio: numpy.ndarray,
	params: AnalysisParams,
	bit_depth: int,
) -> AnalysisResult:

	"""Compute analysis metrics for a single audio recording.

	Converts the integer PCM array to normalised float32, mixes stereo to mono,
	then runs librosa analysis. The original array is not modified.

	Args:
		audio:     Shape (n_frames, channels), dtype int16 or int32.
		           24-bit audio is stored as int32 left-shifted by 8.
		params:    Pre-computed FFT parameters from compute_params().
		bit_depth: Original capture bit depth (16, 24, or 32).

	Returns:
		AnalysisResult with all computed metrics.
	"""

	if audio.shape[0] == 0:
		return AnalysisResult(spectral_flatness=0.0)

	mono = _to_mono_float(audio, bit_depth)

	flatness = librosa.feature.spectral_flatness(
		y=mono,
		n_fft=params.n_fft,
		hop_length=params.hop_length,
	)

	return AnalysisResult(spectral_flatness=float(numpy.mean(flatness)))


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
