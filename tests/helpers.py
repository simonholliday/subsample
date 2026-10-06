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
import subsample.library
import subsample.player
import subsample.transform


def _make_wav (
	path: pathlib.Path,
	n_frames: int = 2048,
	sample_rate: int = 44100,
	n_channels: int = 1,
) -> None:

	"""Write a minimal 16-bit WAV file to path.

	Defaults to mono; pass n_channels=4 for tests that need an ambisonic-sized
	(W, Y, Z, X) stub with all-zero samples.
	"""

	with wave.open(str(path), "wb") as wf:
		wf.setnchannels(n_channels)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		data = numpy.zeros(n_frames * n_channels, dtype=numpy.int16)
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

	The payload comes from the cache's own serialiser, so it has every field
	a real sidecar has and follows the schema when it changes.
	"""

	audio_path   = directory / (audio_stem + audio_ext)
	sidecar_path = subsample.cache.cache_path(audio_path)

	payload = subsample.cache._serialize(
		# A fake digest is fine here: library/watcher loads go through
		# load_sidecar(), which validates version only — the MD5 is checked
		# by ensure_sample_assets/load_cache paths that regenerate anyway.
		audio_md5   = "deadbeef00000000deadbeef00000000",
		params      = _make_params(),
		spectral    = _make_spectral(),
		rhythm      = _make_rhythm(),
		pitch       = _make_pitch(),
		timbre      = _make_timbre(),
		duration    = 1.0,
		level       = _make_level(),
		band_energy = _make_band_energy(),
	)

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


def _hit_starts (rendered: numpy.ndarray, sample_rate: int) -> list[float]:

	"""Where each hit starts in rendered audio, in seconds.

	A hit starts where the level first reaches 0.05 after at least 100 ms
	below it, which is where the ear hears it for the taps and clicks these
	tests render.
	"""

	level  = numpy.abs(rendered[:, 0])
	quiet  = int(0.1 * sample_rate)
	starts: list[float] = []
	last   = -quiet

	for index in numpy.flatnonzero(level >= 0.05):

		if index - last >= quiet:
			starts.append(index / sample_rate)

		last = int(index)

	return starts


def _render_of (
	audio:          numpy.ndarray,
	level:          subsample.analysis.LevelResult,
	segment_bounds: typing.Optional[tuple[tuple[int, int], ...]] = None,
) -> subsample.transform.TransformResult:

	"""A render built by hand around ``audio``, a test's stand-in for one the transform made.

	It carries none of the render worker's measures, so the player measures
	its true peak and a segment's level itself, as a note-on once always did.
	"""

	return subsample.transform.TransformResult(
		key=subsample.transform.TransformKey(sample_id=1, spec=subsample.transform.TransformSpec(steps=())),
		audio=audio,
		duration=len(audio) / 44100,
		level=level,
		segment_bounds=segment_bounds,
	)


def _played (
	player:   subsample.player.MidiPlayer,
	audio:    numpy.ndarray,
	level:    subsample.analysis.LevelResult,
	velocity: float,
	matrix:   numpy.ndarray,
	gain_db:  float = 0.0,
) -> numpy.ndarray:

	"""The whole of ``audio`` as a voice plays it for a note: its gain set as a note-on sets it, then gained and mixed (#4654)."""

	gain  = player._note_gain(level, subsample.analysis.true_peak(audio), velocity, matrix, gain_db)
	voice = subsample.player._Voice(audio=audio, note=60, channel=0, gain=gain, mix=matrix)

	return voice.frames(0, voice.length)


def _laid_out (audio: numpy.ndarray, period: int) -> tuple[numpy.ndarray, int, int]:

	"""``audio`` laid out whole to loop every ``period`` frames, and its loop: what a voice plays a buffer at a time (#3877, #4654)."""

	voice = subsample.player._Voice(audio=audio, note=60, channel=0, ring_period=period)
	_length, start, end, _passes = subsample.player._ring_layout(len(audio), period)

	return voice.frames(0, voice.length), start, end


def _audio (record: subsample.library.SampleRecord) -> numpy.ndarray:

	"""A record's audio, which every record a test builds with audio has."""

	assert record.audio is not None

	return record.audio


_Step = typing.TypeVar("_Step")


def _first_step (spec: subsample.transform.TransformSpec, kind: type[_Step]) -> _Step:

	"""A compiled spec's first step, checked to be the kind the test expects.

	A spec's steps are typed as a union of every step class, so a field of one
	can be read only once the step is narrowed; the check also fails plainly
	if the parser compiled some other step.
	"""

	step = spec.steps[0]

	assert isinstance(step, kind)

	return step
