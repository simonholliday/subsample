"""Analysis result caching for Subsample.

Saves and loads analysis results as JSON sidecar files alongside audio files,
so that repeated analysis of the same file (e.g. reference sounds loaded at
startup) can be skipped when nothing has changed.

Cache validity is verified in two ways:
  1. MD5 hash of the audio file — detects if the audio content has changed.
  2. ANALYSIS_VERSION string — detects if the analysis algorithm has changed.

If either check fails, the caller is expected to re-analyze and overwrite the
cache. A WARNING is logged in that case because re-analysis can be slow.

Sidecar filename: <audio_file>.analysis.json
Example: kick.wav  →  kick.wav.analysis.json
"""

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import tempfile
import typing

import numpy

import subsample.analysis


_log = logging.getLogger(__name__)

_CACHE_SUFFIX: str = ".analysis.json"
_MD5_CHUNK_BYTES: int = 8192


def cache_path (audio_path: pathlib.Path) -> pathlib.Path:

	"""Return the sidecar JSON path for a given audio file.

	Example: /recordings/kick.wav → /recordings/kick.wav.analysis.json
	"""

	return audio_path.with_name(audio_path.name + _CACHE_SUFFIX)


def compute_audio_md5 (audio_path: pathlib.Path) -> str:

	"""Compute the MD5 hex digest of an audio file.

	Reads the file in small chunks to avoid loading large files into memory.

	Args:
		audio_path: Path to the audio file.

	Returns:
		Lowercase hex MD5 digest string.
	"""

	hasher = hashlib.md5()

	with audio_path.open("rb") as f:
		while True:
			chunk = f.read(_MD5_CHUNK_BYTES)

			if not chunk:
				break

			hasher.update(chunk)

	return hasher.hexdigest()


def save_cache (
	audio_path: pathlib.Path,
	audio_md5: str,
	params: subsample.analysis.AnalysisParams,
	spectral: subsample.analysis.AnalysisResult,
	rhythm: subsample.analysis.RhythmResult,
	pitch: subsample.analysis.PitchResult,
	duration: float,
) -> None:

	"""Write analysis results to a JSON sidecar file.

	Uses an atomic write (temp file + os.replace) so the sidecar is never
	left in a partially-written state if the process is interrupted.

	Args:
		audio_path: Path to the audio file being cached.
		audio_md5:  MD5 hex digest of the audio file (precomputed by caller).
		params:     FFT parameters used during analysis.
		spectral:   Spectral analysis result.
		rhythm:     Rhythm analysis result.
		pitch:      Pitch and timbre analysis result.
		duration:   Recording duration in seconds.
	"""

	payload = _serialize(audio_md5, params, spectral, rhythm, pitch, duration)
	json_str = json.dumps(payload, indent=2)

	sidecar = cache_path(audio_path)

	# Write to a temp file in the same directory, then rename atomically.
	# os.replace() is atomic on POSIX (same filesystem); on Windows it is
	# also atomic for most filesystems since Python 3.3.
	fd, tmp_path = tempfile.mkstemp(
		dir=sidecar.parent,
		prefix=sidecar.name + ".tmp",
	)

	try:
		with os.fdopen(fd, "w", encoding="utf-8") as f:
			f.write(json_str)

		os.replace(tmp_path, sidecar)

	except Exception:
		# Clean up the temp file if anything went wrong
		try:
			os.unlink(tmp_path)
		except OSError:
			pass

		raise


def load_cache (
	audio_path: pathlib.Path,
) -> typing.Optional[tuple[
	subsample.analysis.AnalysisResult,
	subsample.analysis.RhythmResult,
	subsample.analysis.PitchResult,
	subsample.analysis.AnalysisParams,
	float,
]]:

	"""Load cached analysis results if the sidecar is valid.

	Checks analysis version and audio MD5 before returning cached data.
	Returns None (and logs the reason) if:
	  - The sidecar file does not exist
	  - The JSON is malformed or missing expected keys
	  - The analysis version does not match ANALYSIS_VERSION
	  - The audio MD5 does not match the current file content

	When returning None due to a stale cache (version or MD5 mismatch), logs
	a WARNING because re-analysis will cause noticeable delay.

	Args:
		audio_path: Path to the audio file.

	Returns:
		(spectral, rhythm, pitch, params, duration) tuple on success, else None.
	"""

	sidecar = cache_path(audio_path)

	if not sidecar.exists():
		return None

	try:
		with sidecar.open("r", encoding="utf-8") as f:
			payload = json.load(f)

	except (json.JSONDecodeError, OSError) as exc:
		_log.warning("Ignoring malformed cache %s: %s", sidecar.name, exc)
		return None

	# Version check — must come before the MD5 check (cheaper)
	cached_version = payload.get("analysis_version")
	if cached_version != subsample.analysis.ANALYSIS_VERSION:
		_log.warning(
			"Re-analyzing %s (analysis version changed: %s → %s)",
			audio_path.name, cached_version, subsample.analysis.ANALYSIS_VERSION,
		)
		return None

	# MD5 check — reads the audio file; done after version check to skip
	# disk I/O when the version alone invalidates the cache
	cached_md5 = payload.get("audio_md5")
	current_md5 = compute_audio_md5(audio_path)

	if cached_md5 != current_md5:
		_log.warning("Re-analyzing %s (audio file has changed)", audio_path.name)
		return None

	try:
		spectral = _deserialize_spectral(payload["spectral"])
		rhythm   = _deserialize_rhythm(payload["rhythm"])
		pitch    = _deserialize_pitch(payload["pitch"])
		params   = _deserialize_params(payload["params"])
		duration = float(payload["duration"])

	except (KeyError, TypeError, ValueError) as exc:
		_log.warning("Ignoring corrupt cache %s: %s", sidecar.name, exc)
		return None

	return spectral, rhythm, pitch, params, duration


def load_sidecar (
	sidecar_path: pathlib.Path,
) -> typing.Optional[tuple[
	subsample.analysis.AnalysisResult,
	subsample.analysis.RhythmResult,
	subsample.analysis.PitchResult,
	subsample.analysis.AnalysisParams,
	float,
]]:

	"""Load analysis results directly from a sidecar file without audio MD5 check.

	Used for reference (canonical) samples which are treated as trusted static
	assets — the audio file does not need to exist. Only the analysis_version
	is validated.

	Returns None (and logs a warning) if the sidecar is missing, malformed, or
	from an incompatible analysis version.

	Args:
		sidecar_path: Path to the .analysis.json sidecar file directly.

	Returns:
		(spectral, rhythm, pitch, params, duration) tuple on success, else None.
	"""

	if not sidecar_path.exists():
		_log.warning("Sidecar not found: %s", sidecar_path)
		return None

	try:
		with sidecar_path.open("r", encoding="utf-8") as f:
			payload = json.load(f)

	except (json.JSONDecodeError, OSError) as exc:
		_log.warning("Ignoring malformed sidecar %s: %s", sidecar_path.name, exc)
		return None

	# Version check — trust the data if the algorithm hasn't changed
	cached_version = payload.get("analysis_version")
	if cached_version != subsample.analysis.ANALYSIS_VERSION:
		_log.warning(
			"Skipping %s (analysis version mismatch: %s vs current %s)",
			sidecar_path.name, cached_version, subsample.analysis.ANALYSIS_VERSION,
		)
		return None

	try:
		spectral = _deserialize_spectral(payload["spectral"])
		rhythm   = _deserialize_rhythm(payload["rhythm"])
		pitch    = _deserialize_pitch(payload["pitch"])
		params   = _deserialize_params(payload["params"])
		duration = float(payload["duration"])

	except (KeyError, TypeError, ValueError) as exc:
		_log.warning("Ignoring corrupt sidecar %s: %s", sidecar_path.name, exc)
		return None

	return spectral, rhythm, pitch, params, duration


# ---------------------------------------------------------------------------
# Private serialization helpers
# ---------------------------------------------------------------------------

def _serialize (
	audio_md5: str,
	params: subsample.analysis.AnalysisParams,
	spectral: subsample.analysis.AnalysisResult,
	rhythm: subsample.analysis.RhythmResult,
	pitch: subsample.analysis.PitchResult,
	duration: float,
) -> dict[str, typing.Any]:

	"""Build the JSON-serializable dict from analysis results."""

	# dataclasses.asdict() handles all plain-Python-typed fields automatically.
	# RhythmResult contains a numpy ndarray (pulse_curve) which is not
	# JSON-serializable, so we build that section manually.
	spectral_dict = dataclasses.asdict(spectral)
	params_dict   = dataclasses.asdict(params)
	pitch_dict    = dataclasses.asdict(pitch)

	# Convert tuples-within-pitch to plain lists (asdict already does this
	# for tuple fields on Python dataclasses, but be explicit for clarity)
	pitch_dict["chroma_profile"] = list(pitch.chroma_profile)
	pitch_dict["mfcc"]           = list(pitch.mfcc)

	rhythm_dict: dict[str, typing.Any] = {
		"tempo_bpm":       rhythm.tempo_bpm,
		"beat_times":      list(rhythm.beat_times),
		"pulse_curve":     rhythm.pulse_curve.tolist(),
		"pulse_peak_times": list(rhythm.pulse_peak_times),
		"onset_times":     list(rhythm.onset_times),
		"onset_count":     rhythm.onset_count,
	}

	return {
		"analysis_version": subsample.analysis.ANALYSIS_VERSION,
		"audio_md5":        audio_md5,
		"sample_rate":      params.sample_rate,
		"duration":         duration,
		"params":           params_dict,
		"spectral":         spectral_dict,
		"rhythm":           rhythm_dict,
		"pitch":            pitch_dict,
	}


def _deserialize_spectral (data: dict[str, typing.Any]) -> subsample.analysis.AnalysisResult:

	"""Reconstruct an AnalysisResult from a JSON dict.

	Uses .get() with sensible defaults for every field so that sidecar files
	written by older versions of the code (before a field was added) can still
	be loaded without a KeyError. When the ANALYSIS_VERSION is bumped for a
	structural change, old caches are automatically invalidated before reaching
	this point, so defaults here only guard against forward-compatible additions.
	"""

	return subsample.analysis.AnalysisResult(
		spectral_flatness  = float(data.get("spectral_flatness", 0.0)),
		attack             = float(data.get("attack", 0.0)),
		release            = float(data.get("release", 0.0)),
		spectral_centroid  = float(data.get("spectral_centroid", 0.0)),
		spectral_bandwidth = float(data.get("spectral_bandwidth", 0.0)),
		zcr                = float(data.get("zcr", 0.0)),
		harmonic_ratio     = float(data.get("harmonic_ratio", 0.0)),
		spectral_contrast  = float(data.get("spectral_contrast", 0.0)),
		voiced_fraction    = float(data.get("voiced_fraction", 0.0)),
	)


def _deserialize_rhythm (data: dict[str, typing.Any]) -> subsample.analysis.RhythmResult:

	"""Reconstruct a RhythmResult from a JSON dict.

	Converts the pulse_curve list back to a float32 numpy array, and
	beat/pulse/onset lists back to tuples. Uses .get() with empty defaults
	for forward-compatible field additions (see _deserialize_spectral).
	"""

	return subsample.analysis.RhythmResult(
		tempo_bpm        = float(data.get("tempo_bpm", 0.0)),
		beat_times       = tuple(float(t) for t in data.get("beat_times", [])),
		pulse_curve      = numpy.array(data.get("pulse_curve", []), dtype=numpy.float32),
		pulse_peak_times = tuple(float(t) for t in data.get("pulse_peak_times", [])),
		onset_times      = tuple(float(t) for t in data.get("onset_times", [])),
		onset_count      = int(data.get("onset_count", 0)),
	)


def _deserialize_pitch (data: dict[str, typing.Any]) -> subsample.analysis.PitchResult:

	"""Reconstruct a PitchResult from a JSON dict.

	Converts chroma_profile and mfcc lists back to tuples. Uses .get() with
	empty defaults for forward-compatible field additions (see _deserialize_spectral).
	"""

	return subsample.analysis.PitchResult(
		dominant_pitch_hz    = float(data.get("dominant_pitch_hz", 0.0)),
		pitch_confidence     = float(data.get("pitch_confidence", 0.0)),
		chroma_profile       = tuple(float(v) for v in data.get("chroma_profile", [0.0] * 12)),
		dominant_pitch_class = int(data.get("dominant_pitch_class", -1)),
		mfcc                 = tuple(float(v) for v in data.get("mfcc", [0.0] * 13)),
	)


def _deserialize_params (data: dict[str, typing.Any]) -> subsample.analysis.AnalysisParams:

	"""Reconstruct AnalysisParams from a JSON dict. Uses .get() with sensible
	defaults for forward-compatible field additions (see _deserialize_spectral).
	"""

	return subsample.analysis.AnalysisParams(
		n_fft       = int(data.get("n_fft", 2048)),
		hop_length  = int(data.get("hop_length", 512)),
		sample_rate = int(data.get("sample_rate", 44100)),
	)
