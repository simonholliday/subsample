"""Tests for subsample/cache.py — analysis result caching."""

import json
import pathlib

import numpy
import pytest

import subsample.analysis
import subsample.cache

from .helpers import _make_params, _make_pitch, _make_rhythm, _make_spectral, _make_wav


# ---------------------------------------------------------------------------
# TestCachePath
# ---------------------------------------------------------------------------

class TestCachePath:

	def test_appends_analysis_json_suffix (self) -> None:
		p = pathlib.Path("/recordings/kick.wav")
		assert subsample.cache.cache_path(p) == pathlib.Path("/recordings/kick.wav.analysis.json")

	def test_non_wav_extension (self) -> None:
		p = pathlib.Path("/recordings/snare.flac")
		assert subsample.cache.cache_path(p) == pathlib.Path("/recordings/snare.flac.analysis.json")


# ---------------------------------------------------------------------------
# TestComputeAudioMd5
# ---------------------------------------------------------------------------

class TestComputeAudioMd5:

	def test_returns_hex_string (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "test.wav"
		_make_wav(wav)
		digest = subsample.cache.compute_audio_md5(wav)
		assert isinstance(digest, str)
		assert len(digest) == 32
		assert all(c in "0123456789abcdef" for c in digest)

	def test_same_file_same_digest (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "test.wav"
		_make_wav(wav)
		assert subsample.cache.compute_audio_md5(wav) == subsample.cache.compute_audio_md5(wav)

	def test_different_content_different_digest (self, tmp_path: pathlib.Path) -> None:
		wav1 = tmp_path / "a.wav"
		wav2 = tmp_path / "b.wav"
		_make_wav(wav1, n_frames=1024)
		_make_wav(wav2, n_frames=2048)
		assert subsample.cache.compute_audio_md5(wav1) != subsample.cache.compute_audio_md5(wav2)


# ---------------------------------------------------------------------------
# TestSaveAndLoadRoundTrip
# ---------------------------------------------------------------------------

class TestSaveAndLoadRoundTrip:

	def test_full_roundtrip (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)

		spectral = _make_spectral()
		rhythm   = _make_rhythm()
		pitch    = _make_pitch()
		params   = _make_params()
		duration = 1.23
		md5      = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(wav, md5, params, spectral, rhythm, pitch, duration)
		result = subsample.cache.load_cache(wav)

		assert result is not None
		r_spectral, r_rhythm, r_pitch, r_params, r_duration = result

		assert r_spectral == spectral
		assert r_pitch    == pitch
		assert r_params   == params
		assert abs(r_duration - duration) < 1e-9

	def test_rhythm_fields_survive_roundtrip (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)

		rhythm = _make_rhythm()
		params = _make_params()
		md5    = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(wav, md5, params, _make_spectral(), rhythm, _make_pitch(), 1.0)
		result = subsample.cache.load_cache(wav)
		assert result is not None

		r_rhythm = result[1]
		assert r_rhythm.tempo_bpm        == rhythm.tempo_bpm
		assert r_rhythm.beat_times       == rhythm.beat_times
		assert r_rhythm.pulse_peak_times == rhythm.pulse_peak_times
		assert r_rhythm.onset_times      == rhythm.onset_times
		assert r_rhythm.onset_count      == rhythm.onset_count
		assert numpy.allclose(r_rhythm.pulse_curve, rhythm.pulse_curve)

	def test_sidecar_file_created (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, _make_params(), _make_spectral(), _make_rhythm(), _make_pitch(), 1.0)
		assert subsample.cache.cache_path(wav).exists()


# ---------------------------------------------------------------------------
# TestCacheInvalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:

	def test_missing_cache_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		assert subsample.cache.load_cache(wav) is None

	def test_version_mismatch_returns_none (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, _make_params(), _make_spectral(), _make_rhythm(), _make_pitch(), 1.0)

		# Simulate the analysis algorithm being updated
		monkeypatch.setattr(subsample.analysis, "ANALYSIS_VERSION", "999")
		assert subsample.cache.load_cache(wav) is None

	def test_version_mismatch_logs_warning (
		self,
		tmp_path: pathlib.Path,
		monkeypatch: pytest.MonkeyPatch,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, _make_params(), _make_spectral(), _make_rhythm(), _make_pitch(), 1.0)

		monkeypatch.setattr(subsample.analysis, "ANALYSIS_VERSION", "999")

		import logging
		with caplog.at_level(logging.WARNING, logger="subsample.cache"):
			subsample.cache.load_cache(wav)

		assert any("Re-analyzing" in r.message for r in caplog.records)

	def test_md5_mismatch_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav, n_frames=1024)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, _make_params(), _make_spectral(), _make_rhythm(), _make_pitch(), 1.0)

		# Overwrite the WAV with different content
		_make_wav(wav, n_frames=2048)

		assert subsample.cache.load_cache(wav) is None

	def test_md5_mismatch_logs_warning (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav, n_frames=1024)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, _make_params(), _make_spectral(), _make_rhythm(), _make_pitch(), 1.0)

		_make_wav(wav, n_frames=2048)

		import logging
		with caplog.at_level(logging.WARNING, logger="subsample.cache"):
			subsample.cache.load_cache(wav)

		assert any("Re-analyzing" in r.message for r in caplog.records)

	def test_malformed_json_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		sidecar = subsample.cache.cache_path(wav)
		sidecar.write_text("this is not json", encoding="utf-8")
		assert subsample.cache.load_cache(wav) is None

	def test_missing_key_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		sidecar = subsample.cache.cache_path(wav)
		# Valid JSON but missing required keys
		sidecar.write_text(
			json.dumps({"analysis_version": subsample.analysis.ANALYSIS_VERSION}),
			encoding="utf-8",
		)
		assert subsample.cache.load_cache(wav) is None


# ---------------------------------------------------------------------------
# TestAtomicWrite
# ---------------------------------------------------------------------------

class TestAtomicWrite:

	def test_no_tmp_file_left_behind (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		_make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, _make_params(), _make_spectral(), _make_rhythm(), _make_pitch(), 1.0)

		tmp_files = list(tmp_path.glob("*.tmp*"))
		assert tmp_files == [], f"Temp files left behind: {tmp_files}"
