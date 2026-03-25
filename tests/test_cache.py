"""Tests for subsample/cache.py — analysis result caching."""

import json
import pathlib

import numpy
import pytest

import subsample.analysis
import subsample.cache

import tests.helpers


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
		tests.helpers._make_wav(wav)
		digest = subsample.cache.compute_audio_md5(wav)
		assert isinstance(digest, str)
		assert len(digest) == 32
		assert all(c in "0123456789abcdef" for c in digest)

	def test_same_file_same_digest (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "test.wav"
		tests.helpers._make_wav(wav)
		assert subsample.cache.compute_audio_md5(wav) == subsample.cache.compute_audio_md5(wav)

	def test_different_content_different_digest (self, tmp_path: pathlib.Path) -> None:
		wav1 = tmp_path / "a.wav"
		wav2 = tmp_path / "b.wav"
		tests.helpers._make_wav(wav1, n_frames=1024)
		tests.helpers._make_wav(wav2, n_frames=2048)
		assert subsample.cache.compute_audio_md5(wav1) != subsample.cache.compute_audio_md5(wav2)


# ---------------------------------------------------------------------------
# TestSaveAndLoadRoundTrip
# ---------------------------------------------------------------------------

class TestSaveAndLoadRoundTrip:

	def test_full_roundtrip (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)

		spectral = tests.helpers._make_spectral()
		rhythm   = tests.helpers._make_rhythm()
		pitch    = tests.helpers._make_pitch()
		timbre   = tests.helpers._make_timbre()
		level    = tests.helpers._make_level()
		params   = tests.helpers._make_params()
		duration = 1.23
		md5      = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(wav, md5, params, spectral, rhythm, pitch, timbre, duration, level)
		result = subsample.cache.load_cache(wav)

		assert result is not None
		r_spectral, r_rhythm, r_pitch, r_timbre, r_params, r_duration, r_level, r_band_energy = result

		assert r_spectral == spectral
		assert r_pitch    == pitch
		assert r_timbre   == timbre
		assert r_params   == params
		assert abs(r_duration - duration) < 1e-9
		assert r_level == level

	def test_rhythm_fields_survive_roundtrip (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)

		rhythm = tests.helpers._make_rhythm()
		params = tests.helpers._make_params()
		md5    = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(wav, md5, params, tests.helpers._make_spectral(), rhythm, tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())
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
		tests.helpers._make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, tests.helpers._make_params(), tests.helpers._make_spectral(), tests.helpers._make_rhythm(), tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())
		assert subsample.cache.cache_path(wav).exists()


# ---------------------------------------------------------------------------
# TestCacheInvalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:

	def test_missing_cache_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		assert subsample.cache.load_cache(wav) is None

	def test_version_mismatch_reanalyzes_and_returns_result (
		self,
		tmp_path: pathlib.Path,
		monkeypatch: pytest.MonkeyPatch,
	) -> None:
		"""Version mismatch with audio present should re-analyze and return a result."""
		wav = tmp_path / "kick.wav"
		# Use a long enough WAV for all librosa analysis functions to work
		tests.helpers._make_wav(wav, n_frames=22050)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, tests.helpers._make_params(), tests.helpers._make_spectral(), tests.helpers._make_rhythm(), tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())

		# Simulate the analysis algorithm being updated
		monkeypatch.setattr(subsample.analysis, "ANALYSIS_VERSION", "999")
		result = subsample.cache.load_cache(wav)

		# Re-analysis should succeed and return a valid result tuple
		assert result is not None
		spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result
		assert isinstance(spectral, subsample.analysis.AnalysisResult)

	def test_version_mismatch_logs_info (
		self,
		tmp_path: pathlib.Path,
		monkeypatch: pytest.MonkeyPatch,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Re-analysis triggered by version mismatch should log at INFO, not WARNING."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav, n_frames=22050)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, tests.helpers._make_params(), tests.helpers._make_spectral(), tests.helpers._make_rhythm(), tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())

		monkeypatch.setattr(subsample.analysis, "ANALYSIS_VERSION", "999")

		import logging
		with caplog.at_level(logging.INFO, logger="subsample.cache"):
			subsample.cache.load_cache(wav)

		# Log message must show the old → new version transition.
		assert any(
			r.levelno == logging.INFO
			and "Re-analyzing" in r.message
			and "→" in r.message
			and "999" in r.message
			for r in caplog.records
		)

	def test_md5_mismatch_reanalyzes_and_returns_result (self, tmp_path: pathlib.Path) -> None:
		"""MD5 mismatch (audio changed) should re-analyze and return a result."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav, n_frames=22050)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, tests.helpers._make_params(), tests.helpers._make_spectral(), tests.helpers._make_rhythm(), tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())

		# Overwrite the WAV with different content
		tests.helpers._make_wav(wav, n_frames=22050)
		result = subsample.cache.load_cache(wav)

		assert result is not None
		spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result
		assert isinstance(spectral, subsample.analysis.AnalysisResult)

	def test_md5_mismatch_logs_info (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Re-analysis triggered by MD5 mismatch should log at INFO, not WARNING."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav, n_frames=22050)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, tests.helpers._make_params(), tests.helpers._make_spectral(), tests.helpers._make_rhythm(), tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())

		tests.helpers._make_wav(wav, n_frames=44100)  # different content → different MD5

		import logging
		with caplog.at_level(logging.INFO, logger="subsample.cache"):
			subsample.cache.load_cache(wav)

		assert any(
			r.levelno == logging.INFO and "Re-analyzing" in r.message
			for r in caplog.records
		)

	def test_malformed_json_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		sidecar = subsample.cache.cache_path(wav)
		sidecar.write_text("this is not json", encoding="utf-8")
		assert subsample.cache.load_cache(wav) is None

	def test_missing_key_returns_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
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
		tests.helpers._make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(wav, md5, tests.helpers._make_params(), tests.helpers._make_spectral(), tests.helpers._make_rhythm(), tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())

		tmp_files = list(tmp_path.glob("*.tmp*"))
		assert tmp_files == [], f"Temp files left behind: {tmp_files}"


# ---------------------------------------------------------------------------
# TestAudioMetadata — new fields: bit_depth, channels, captured_at
# ---------------------------------------------------------------------------

class TestAudioMetadata:

	def _save (
		self,
		wav: pathlib.Path,
		bit_depth: int = 16,
		channels: int = 1,
		captured_at: "str | None" = None,
	) -> None:
		"""Helper: write a sidecar with the given metadata fields."""
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(
			wav, md5,
			tests.helpers._make_params(),
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(),
			tests.helpers._make_timbre(),
			1.0,
			tests.helpers._make_level(),
			bit_depth   = bit_depth,
			channels    = channels,
			captured_at = captured_at,
		)

	def test_bit_depth_stored (self, tmp_path: pathlib.Path) -> None:
		"""bit_depth is written to the sidecar JSON."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		self._save(wav, bit_depth=24)

		data = json.loads(subsample.cache.cache_path(wav).read_text())
		assert data["bit_depth"] == 24

	def test_channels_stored (self, tmp_path: pathlib.Path) -> None:
		"""channels is written to the sidecar JSON."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		self._save(wav, channels=2)

		data = json.loads(subsample.cache.cache_path(wav).read_text())
		assert data["channels"] == 2

	def test_captured_at_stored (self, tmp_path: pathlib.Path) -> None:
		"""captured_at ISO timestamp is written to the sidecar JSON."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		self._save(wav, captured_at="2026-03-23T14:30:00")

		data = json.loads(subsample.cache.cache_path(wav).read_text())
		assert data["captured_at"] == "2026-03-23T14:30:00"

	def test_captured_at_null_for_reference_files (self, tmp_path: pathlib.Path) -> None:
		"""captured_at is null when no timestamp is provided (reference/imported files)."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		self._save(wav, captured_at=None)

		data = json.loads(subsample.cache.cache_path(wav).read_text())
		assert data["captured_at"] is None

	def test_default_values (self, tmp_path: pathlib.Path) -> None:
		"""Omitting the new params uses safe defaults (backwards-compatible callers)."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(
			wav, md5,
			tests.helpers._make_params(),
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(),
			tests.helpers._make_timbre(),
			1.0,
			tests.helpers._make_level(),
		)

		data = json.loads(subsample.cache.cache_path(wav).read_text())
		assert data["bit_depth"]   == 16
		assert data["channels"]    == 1
		assert data["captured_at"] is None

	def test_sidecar_missing_new_fields_still_loads (self, tmp_path: pathlib.Path) -> None:
		"""A version-8 sidecar that pre-dates bit_depth/channels/captured_at still loads."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav, n_frames=44100)
		md5 = subsample.cache.compute_audio_md5(wav)

		# Write a sidecar that omits the three new metadata fields.
		subsample.cache.save_cache(
			wav, md5,
			tests.helpers._make_params(),
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(),
			tests.helpers._make_timbre(),
			1.0,
			tests.helpers._make_level(),
		)

		# Remove the new fields to simulate a sidecar from before they were added.
		sidecar = subsample.cache.cache_path(wav)
		data = json.loads(sidecar.read_text())
		for key in ("bit_depth", "channels", "captured_at"):
			data.pop(key, None)
		sidecar.write_text(json.dumps(data), encoding="utf-8")

		# Should still load successfully — _deserialize_payload uses .get() defaults.
		result = subsample.cache.load_cache(wav)
		assert result is not None


# ---------------------------------------------------------------------------
# TestBandEnergyCache
# ---------------------------------------------------------------------------

class TestBandEnergyCache:

	"""Tests for band_energy serialization in save_cache / load_cache."""

	def test_band_energy_round_trip (self, tmp_path: pathlib.Path) -> None:
		"""Band energy fractions and decay rates survive a save/load round-trip."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)

		band_energy = subsample.analysis.BandEnergyResult(
			energy_fractions = (0.6, 0.25, 0.1, 0.05),
			decay_rates      = (0.8, 0.4, 0.2, 0.1),
		)

		md5 = subsample.cache.compute_audio_md5(wav)
		subsample.cache.save_cache(
			wav, md5,
			tests.helpers._make_params(),
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(),
			tests.helpers._make_timbre(),
			1.0,
			tests.helpers._make_level(),
			band_energy = band_energy,
		)

		result = subsample.cache.load_cache(wav)
		assert result is not None
		*_, r_band_energy = result

		assert isinstance(r_band_energy, subsample.analysis.BandEnergyResult)
		assert len(r_band_energy.energy_fractions) == 4
		assert len(r_band_energy.decay_rates) == 4
		for expected, actual in zip(band_energy.energy_fractions, r_band_energy.energy_fractions):
			assert abs(expected - actual) < 1e-9
		for expected, actual in zip(band_energy.decay_rates, r_band_energy.decay_rates):
			assert abs(expected - actual) < 1e-9

	def test_band_energy_missing_defaults_to_zeros (self, tmp_path: pathlib.Path) -> None:
		"""A sidecar without a 'band_energy' key defaults to all-zero values."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav, n_frames=44100)
		md5 = subsample.cache.compute_audio_md5(wav)

		# Write a sidecar without band_energy key (simulates a pre-v9 sidecar).
		subsample.cache.save_cache(
			wav, md5,
			tests.helpers._make_params(),
			tests.helpers._make_spectral(),
			tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(),
			tests.helpers._make_timbre(),
			1.0,
			tests.helpers._make_level(),
		)

		sidecar = subsample.cache.cache_path(wav)
		data = json.loads(sidecar.read_text())
		data.pop("band_energy", None)
		sidecar.write_text(json.dumps(data), encoding="utf-8")

		result = subsample.cache.load_cache(wav)
		assert result is not None
		*_, r_band_energy = result

		assert isinstance(r_band_energy, subsample.analysis.BandEnergyResult)
		assert all(v == 0.0 for v in r_band_energy.energy_fractions)
		assert all(v == 0.0 for v in r_band_energy.decay_rates)
