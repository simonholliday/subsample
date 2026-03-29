"""Shared test helpers for the Subsample test suite.

Plain helper functions used by multiple test modules. Not pytest fixtures —
imported directly by the test files that need them.
"""

import json
import pathlib
import typing
import wave

import numpy

import subsample.analysis
import subsample.cache


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
		spectral_rolloff   = 0.55,
		spectral_slope     = 0.35,
	)


def _make_rhythm () -> subsample.analysis.RhythmResult:

	"""Return a representative RhythmResult with typical field values."""

	return subsample.analysis.RhythmResult(
		tempo_bpm        = 120.0,
		beat_times       = (0.5, 1.0, 1.5),
		pulse_curve      = numpy.array([0.1, 0.2, 0.3, 0.4], dtype=numpy.float32),
		pulse_peak_times = (0.5, 1.5),
		onset_times      = (0.1, 0.6),
		attack_times     = (0.08, 0.57),
		onset_count      = 2,
	)


def _make_pitch (
	dominant_pitch_hz:    float = 440.0,
	pitch_confidence:     float = 0.92,
	pitch_stability:      float = 0.1,
	voiced_frame_count:   int   = 8,
	dominant_pitch_class: int   = 9,
) -> subsample.analysis.PitchResult:

	"""Return a representative PitchResult with typical field values.

	All fields evaluated by `has_stable_pitch()` are exposed as keyword
	arguments so tests can exercise boundary conditions without constructing
	PitchResult manually.
	"""

	return subsample.analysis.PitchResult(
		dominant_pitch_hz    = dominant_pitch_hz,
		pitch_confidence     = pitch_confidence,
		chroma_profile       = tuple(float(i) / 12.0 for i in range(12)),
		dominant_pitch_class = dominant_pitch_class,
		pitch_stability      = pitch_stability,
		voiced_frame_count   = voiced_frame_count,
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

	return subsample.analysis.LevelResult(
		peak=0.85,
		rms=0.25,
		crest_factor=3.4,
		crest_factor_db=10.63,
		noise_floor=0.01,
	)


def _make_band_energy () -> subsample.analysis.BandEnergyResult:

	"""Return a representative BandEnergyResult with plausible drum-like values."""

	return subsample.analysis.BandEnergyResult(
		energy_fractions = (0.4, 0.3, 0.2, 0.1),
		decay_rates      = (0.6, 0.4, 0.2, 0.1),
	)


def _make_params (sample_rate: int = 44100) -> subsample.analysis.AnalysisParams:

	"""Return AnalysisParams computed for the given sample rate."""

	return subsample.analysis.compute_params(sample_rate)


def _write_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	audio_ext: str = ".wav",
) -> pathlib.Path:

	"""Write a valid .analysis.json sidecar to directory.

	Does NOT create the audio file — only the sidecar.  Returns the sidecar
	path.  Used by both library and watcher tests.

	The JSON payload mirrors the format in subsample/cache.py.  If the sidecar
	schema changes (new fields, renamed keys), update this helper to match.
	"""

	audio_path   = directory / (audio_stem + audio_ext)
	sidecar_path = subsample.cache.cache_path(audio_path)
	spectral     = _make_spectral()
	rhythm       = _make_rhythm()
	pitch        = _make_pitch()
	timbre       = _make_timbre()
	level        = _make_level()
	band_energy  = _make_band_energy()
	params       = _make_params()

	payload: dict[str, typing.Any] = {
		"analysis_version": subsample.analysis.ANALYSIS_VERSION,
		"audio_md5":        "deadbeef00000000deadbeef00000000",
		"sample_rate":      params.sample_rate,
		"duration":         1.0,
		"params": {
			"n_fft":        params.n_fft,
			"hop_length":   params.hop_length,
			"sample_rate":  params.sample_rate,
		},
		"spectral": {
			"spectral_flatness":  spectral.spectral_flatness,
			"attack":             spectral.attack,
			"release":            spectral.release,
			"spectral_centroid":  spectral.spectral_centroid,
			"spectral_bandwidth": spectral.spectral_bandwidth,
			"zcr":                spectral.zcr,
			"harmonic_ratio":     spectral.harmonic_ratio,
			"spectral_contrast":  spectral.spectral_contrast,
			"voiced_fraction":    spectral.voiced_fraction,
			"log_attack_time":    spectral.log_attack_time,
			"spectral_flux":      spectral.spectral_flux,
			"spectral_rolloff":   spectral.spectral_rolloff,
			"spectral_slope":     spectral.spectral_slope,
		},
		"rhythm": {
			"tempo_bpm":        rhythm.tempo_bpm,
			"beat_times":       list(rhythm.beat_times),
			"pulse_curve":      rhythm.pulse_curve.tolist(),
			"pulse_peak_times": list(rhythm.pulse_peak_times),
			"onset_times":      list(rhythm.onset_times),
			"attack_times":     list(rhythm.attack_times),
			"onset_count":      rhythm.onset_count,
		},
		"pitch": {
			"dominant_pitch_hz":    pitch.dominant_pitch_hz,
			"pitch_confidence":     pitch.pitch_confidence,
			"chroma_profile":       list(pitch.chroma_profile),
			"dominant_pitch_class": pitch.dominant_pitch_class,
			"pitch_stability":      pitch.pitch_stability,
			"voiced_frame_count":   pitch.voiced_frame_count,
		},
		"timbre": {
			"mfcc":       list(timbre.mfcc),
			"mfcc_delta": list(timbre.mfcc_delta),
			"mfcc_onset": list(timbre.mfcc_onset),
		},
		"level": {
			"peak":            level.peak,
			"rms":             level.rms,
			"crest_factor":    level.crest_factor,
			"crest_factor_db": level.crest_factor_db,
			"noise_floor":     level.noise_floor,
		},
		"band_energy": {
			"energy_fractions": list(band_energy.energy_fractions),
			"decay_rates":      list(band_energy.decay_rates),
		},
	}

	sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
	return sidecar_path


def _write_wav_and_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	n_frames: int = 2048,
) -> tuple[pathlib.Path, pathlib.Path]:

	"""Write a WAV file and its sidecar.  Returns (wav_path, sidecar_path)."""

	wav_path     = directory / (audio_stem + ".wav")
	_make_wav(wav_path, n_frames=n_frames)
	sidecar_path = _write_sidecar(directory, audio_stem)
	return wav_path, sidecar_path
