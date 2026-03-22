"""Shared test helpers for the Subsample test suite.

Plain helper functions used by multiple test modules. Not pytest fixtures —
imported directly by the test files that need them.
"""

import pathlib
import wave

import numpy

import subsample.analysis


def _make_wav (path: pathlib.Path, n_frames: int = 2048, sample_rate: int = 44100) -> None:

	"""Write a minimal mono 16-bit WAV file to path."""

	with wave.open(str(path), "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		data = numpy.zeros(n_frames, dtype=numpy.int16)
		wf.writeframes(data.tobytes())


def _make_spectral () -> subsample.analysis.AnalysisResult:

	"""Return a representative AnalysisResult with distinct per-field values."""

	return subsample.analysis.AnalysisResult(
		spectral_flatness  = 0.1,
		attack             = 0.2,
		release            = 0.3,
		spectral_centroid  = 0.4,
		spectral_bandwidth = 0.5,
		zcr                = 0.6,
		harmonic_ratio     = 0.7,
		spectral_contrast  = 0.8,
		voiced_fraction    = 0.9,
		log_attack_time    = 0.15,
		spectral_flux      = 0.45,
	)


def _make_rhythm () -> subsample.analysis.RhythmResult:

	"""Return a representative RhythmResult with typical field values."""

	return subsample.analysis.RhythmResult(
		tempo_bpm        = 120.0,
		beat_times       = (0.5, 1.0, 1.5),
		pulse_curve      = numpy.array([0.1, 0.2, 0.3, 0.4], dtype=numpy.float32),
		pulse_peak_times = (0.5, 1.5),
		onset_times      = (0.1, 0.6),
		onset_count      = 2,
	)


def _make_pitch () -> subsample.analysis.PitchResult:

	"""Return a representative PitchResult with typical field values."""

	return subsample.analysis.PitchResult(
		dominant_pitch_hz    = 440.0,
		pitch_confidence     = 0.92,
		chroma_profile       = tuple(float(i) / 12.0 for i in range(12)),
		dominant_pitch_class = 9,
		pitch_stability      = 0.1,
		voiced_frame_count   = 8,
	)


def _make_timbre () -> subsample.analysis.TimbreResult:

	"""Return a representative TimbreResult with distinct per-field values."""

	return subsample.analysis.TimbreResult(
		mfcc       = tuple(float(i) for i in range(13)),
		mfcc_delta = tuple(float(i) * 0.1 for i in range(13)),
		mfcc_onset = tuple(float(i) * 0.5 for i in range(13)),
	)


def _make_level () -> subsample.analysis.LevelResult:

	"""Return a representative LevelResult with typical field values."""

	return subsample.analysis.LevelResult(peak=0.85, rms=0.25)


def _make_params (sample_rate: int = 44100) -> subsample.analysis.AnalysisParams:

	"""Return AnalysisParams computed for the given sample rate."""

	return subsample.analysis.compute_params(sample_rate)
