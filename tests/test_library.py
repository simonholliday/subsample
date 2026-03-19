"""Tests for subsample/library.py — reference sample library."""

import json
import pathlib
import wave

import numpy
import pytest

import subsample.analysis
import subsample.cache
import subsample.library


# ---------------------------------------------------------------------------
# Fixtures (duplicated from test_cache.py to keep tests self-contained)
# ---------------------------------------------------------------------------

def _make_wav (path: pathlib.Path, n_frames: int = 2048, sample_rate: int = 44100) -> None:

	"""Write a minimal mono 16-bit WAV file to path."""

	with wave.open(str(path), "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		data = numpy.zeros(n_frames, dtype=numpy.int16)
		wf.writeframes(data.tobytes())


def _make_spectral () -> subsample.analysis.AnalysisResult:
	return subsample.analysis.AnalysisResult(
		spectral_flatness=0.1, attack=0.2, release=0.3,
		spectral_centroid=0.4, spectral_bandwidth=0.5,
		zcr=0.6, harmonic_ratio=0.7, spectral_contrast=0.8, voiced_fraction=0.9,
	)


def _make_rhythm () -> subsample.analysis.RhythmResult:
	return subsample.analysis.RhythmResult(
		tempo_bpm=120.0,
		beat_times=(0.5, 1.0),
		pulse_curve=numpy.array([0.1, 0.2], dtype=numpy.float32),
		pulse_peak_times=(0.5,),
		onset_times=(0.1, 0.6),
		onset_count=2,
	)


def _make_pitch () -> subsample.analysis.PitchResult:
	return subsample.analysis.PitchResult(
		dominant_pitch_hz=440.0,
		pitch_confidence=0.92,
		chroma_profile=tuple(float(i) / 12.0 for i in range(12)),
		dominant_pitch_class=9,
		mfcc=tuple(float(i) for i in range(13)),
	)


def _make_params () -> subsample.analysis.AnalysisParams:
	return subsample.analysis.compute_params(44100)


def _write_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	audio_ext: str = ".wav",
) -> pathlib.Path:

	"""Write a valid sidecar file for a (virtual) audio file in directory.

	The audio file itself is NOT created — only the sidecar. This mirrors
	the reference library use case where audio may be absent.
	"""

	audio_filename = audio_stem + audio_ext
	audio_path = directory / audio_filename
	sidecar_path = subsample.cache.cache_path(audio_path)

	spectral = _make_spectral()
	rhythm   = _make_rhythm()
	pitch    = _make_pitch()
	params   = _make_params()

	# Build the JSON payload the same way cache.save_cache() would, but write
	# directly so we don't need the audio file to compute an MD5.
	payload = {
		"analysis_version": subsample.analysis.ANALYSIS_VERSION,
		"audio_md5":        "deadbeef00000000deadbeef00000000",  # placeholder
		"sample_rate":      params.sample_rate,
		"duration":         1.0,
		"params":           {"n_fft": params.n_fft, "hop_length": params.hop_length, "sample_rate": params.sample_rate},
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
		},
		"rhythm": {
			"tempo_bpm":        rhythm.tempo_bpm,
			"beat_times":       list(rhythm.beat_times),
			"pulse_curve":      rhythm.pulse_curve.tolist(),
			"pulse_peak_times": list(rhythm.pulse_peak_times),
			"onset_times":      list(rhythm.onset_times),
			"onset_count":      rhythm.onset_count,
		},
		"pitch": {
			"dominant_pitch_hz":    pitch.dominant_pitch_hz,
			"pitch_confidence":     pitch.pitch_confidence,
			"chroma_profile":       list(pitch.chroma_profile),
			"dominant_pitch_class": pitch.dominant_pitch_class,
			"mfcc":                 list(pitch.mfcc),
		},
	}

	sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

	return sidecar_path


# ---------------------------------------------------------------------------
# TestSampleRecord
# ---------------------------------------------------------------------------

class TestSampleRecord:

	def test_fields_accessible (self) -> None:
		record = subsample.library.SampleRecord(
			name     = "KICK",
			spectral = _make_spectral(),
			rhythm   = _make_rhythm(),
			pitch    = _make_pitch(),
			params   = _make_params(),
			duration = 1.23,
		)
		assert record.name == "KICK"
		assert record.duration == 1.23
		assert record.spectral.attack == pytest.approx(0.2)

	def test_is_frozen (self) -> None:
		record = subsample.library.SampleRecord(
			name="X", spectral=_make_spectral(), rhythm=_make_rhythm(),
			pitch=_make_pitch(), params=_make_params(), duration=1.0,
		)
		with pytest.raises(Exception):
			record.name = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestLoadSidecar
# ---------------------------------------------------------------------------

class TestLoadSidecar:

	def test_valid_sidecar_loads (self, tmp_path: pathlib.Path) -> None:
		sidecar = _write_sidecar(tmp_path, "kick")
		result = subsample.cache.load_sidecar(sidecar)
		assert result is not None
		spectral, rhythm, pitch, params, duration = result
		assert spectral.attack == pytest.approx(0.2)
		assert duration == pytest.approx(1.0)

	def test_audio_file_need_not_exist (self, tmp_path: pathlib.Path) -> None:
		sidecar = _write_sidecar(tmp_path, "kick")
		audio = tmp_path / "kick.wav"
		assert not audio.exists()
		result = subsample.cache.load_sidecar(sidecar)
		assert result is not None

	def test_missing_sidecar_returns_none (self, tmp_path: pathlib.Path) -> None:
		result = subsample.cache.load_sidecar(tmp_path / "ghost.wav.analysis.json")
		assert result is None

	def test_version_mismatch_returns_none (
		self,
		tmp_path: pathlib.Path,
		monkeypatch: pytest.MonkeyPatch,
	) -> None:
		sidecar = _write_sidecar(tmp_path, "kick")
		monkeypatch.setattr(subsample.analysis, "ANALYSIS_VERSION", "999")
		assert subsample.cache.load_sidecar(sidecar) is None

	def test_version_mismatch_logs_warning (
		self,
		tmp_path: pathlib.Path,
		monkeypatch: pytest.MonkeyPatch,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		sidecar = _write_sidecar(tmp_path, "kick")
		monkeypatch.setattr(subsample.analysis, "ANALYSIS_VERSION", "999")
		with caplog.at_level(logging.WARNING, logger="subsample.cache"):
			subsample.cache.load_sidecar(sidecar)
		assert any("mismatch" in r.message.lower() for r in caplog.records)

	def test_malformed_json_returns_none (self, tmp_path: pathlib.Path) -> None:
		sidecar = tmp_path / "kick.wav.analysis.json"
		sidecar.write_text("not json", encoding="utf-8")
		assert subsample.cache.load_sidecar(sidecar) is None


# ---------------------------------------------------------------------------
# TestReferenceLibrary
# ---------------------------------------------------------------------------

class TestReferenceLibrary:

	def _library_with (self, records: list[subsample.library.SampleRecord]) -> subsample.library.ReferenceLibrary:
		return subsample.library.ReferenceLibrary(records)

	def _record (self, name: str) -> subsample.library.SampleRecord:
		return subsample.library.SampleRecord(
			name=name, spectral=_make_spectral(), rhythm=_make_rhythm(),
			pitch=_make_pitch(), params=_make_params(), duration=1.0,
		)

	def test_empty_library (self) -> None:
		lib = self._library_with([])
		assert len(lib) == 0
		assert lib.names() == []
		assert lib.all() == []

	def test_get_by_exact_name (self) -> None:
		lib = self._library_with([self._record("KICK")])
		record = lib.get("KICK")
		assert record is not None
		assert record.name == "KICK"

	def test_get_case_insensitive (self) -> None:
		lib = self._library_with([self._record("KICK")])
		assert lib.get("kick") is not None
		assert lib.get("Kick") is not None
		assert lib.get("KICK") is not None

	def test_get_missing_returns_none (self) -> None:
		lib = self._library_with([self._record("KICK")])
		assert lib.get("SNARE") is None

	def test_names_sorted (self) -> None:
		lib = self._library_with([
			self._record("SNARE"), self._record("KICK"), self._record("HAT"),
		])
		assert lib.names() == ["HAT", "KICK", "SNARE"]

	def test_all_sorted_by_name (self) -> None:
		lib = self._library_with([
			self._record("SNARE"), self._record("KICK"), self._record("HAT"),
		])
		assert [r.name for r in lib.all()] == ["HAT", "KICK", "SNARE"]

	def test_len (self) -> None:
		lib = self._library_with([self._record("A"), self._record("B")])
		assert len(lib) == 2

	def test_repr_contains_count (self) -> None:
		lib = self._library_with([self._record("KICK")])
		assert "1" in repr(lib)


# ---------------------------------------------------------------------------
# TestLoadReferenceLibrary
# ---------------------------------------------------------------------------

class TestLoadReferenceLibrary:

	def test_loads_valid_sidecars (self, tmp_path: pathlib.Path) -> None:
		_write_sidecar(tmp_path, "KICK")
		_write_sidecar(tmp_path, "SNARE")
		lib = subsample.library.load_reference_library(tmp_path)
		assert len(lib) == 2
		assert lib.get("KICK") is not None
		assert lib.get("SNARE") is not None

	def test_name_derived_from_stem (self, tmp_path: pathlib.Path) -> None:
		_write_sidecar(tmp_path, "BD0025", ".WAV")
		lib = subsample.library.load_reference_library(tmp_path)
		assert lib.get("BD0025") is not None

	def test_nonexistent_directory_returns_empty (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		missing = tmp_path / "no_such_dir"
		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			lib = subsample.library.load_reference_library(missing)
		assert len(lib) == 0
		assert any("not found" in r.message.lower() for r in caplog.records)

	def test_empty_directory_returns_empty (self, tmp_path: pathlib.Path) -> None:
		lib = subsample.library.load_reference_library(tmp_path)
		assert len(lib) == 0

	def test_skips_invalid_sidecars (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		# Valid sidecar
		_write_sidecar(tmp_path, "KICK")
		# Invalid sidecar
		bad = tmp_path / "BAD.wav.analysis.json"
		bad.write_text("not json", encoding="utf-8")

		with caplog.at_level(logging.WARNING, logger="subsample.cache"):
			lib = subsample.library.load_reference_library(tmp_path)

		assert len(lib) == 1
		assert lib.get("KICK") is not None

	def test_logs_loaded_count (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		_write_sidecar(tmp_path, "KICK")
		_write_sidecar(tmp_path, "SNARE")

		with caplog.at_level(logging.INFO, logger="subsample.library"):
			subsample.library.load_reference_library(tmp_path)

		assert any("2" in r.message for r in caplog.records)

	def test_does_not_recurse_into_subdirectories (self, tmp_path: pathlib.Path) -> None:
		# Sidecar in a subdirectory should be ignored
		subdir = tmp_path / "Roland TR-808"
		subdir.mkdir()
		_write_sidecar(subdir, "SD0000")
		# Sidecar at top level should load
		_write_sidecar(tmp_path, "KICK")

		lib = subsample.library.load_reference_library(tmp_path)
		assert len(lib) == 1
		assert lib.get("KICK") is not None
