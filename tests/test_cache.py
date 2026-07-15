"""Tests for subsample/cache.py — analysis result caching."""

import json
import pathlib
import typing

import numpy
import pytest

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.library
import subsample.loopfind

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
		assert result.spectral == spectral
		assert result.pitch    == pitch
		assert result.timbre   == timbre
		assert result.params   == params
		assert abs(result.duration - duration) < 1e-9
		assert result.level == level

	def test_rhythm_fields_survive_roundtrip (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)

		rhythm = tests.helpers._make_rhythm()
		params = tests.helpers._make_params()
		md5    = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(wav, md5, params, tests.helpers._make_spectral(), rhythm, tests.helpers._make_pitch(), tests.helpers._make_timbre(), 1.0, tests.helpers._make_level())
		result = subsample.cache.load_cache(wav)
		assert result is not None

		r_rhythm = result.rhythm
		assert r_rhythm.tempo_bpm        == rhythm.tempo_bpm
		assert r_rhythm.beat_times       == rhythm.beat_times
		assert r_rhythm.pulse_peak_times == rhythm.pulse_peak_times
		assert r_rhythm.onset_times      == rhythm.onset_times
		assert r_rhythm.attack_times     == rhythm.attack_times
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
		assert isinstance(result.spectral, subsample.analysis.AnalysisResult)

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
		assert isinstance(result.spectral, subsample.analysis.AnalysisResult)

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

	def test_missing_audio_md5_triggers_reanalysis (self, tmp_path: pathlib.Path) -> None:

		"""A sidecar with a current analysis_version but no audio_md5 used to
		bypass MD5 validation and return cached data silently — meaning a
		stale sidecar paired with a different audio file would be accepted as
		fresh.  ``load_cache`` now re-analyzes from the audio in that case;
		the audio file is the source of truth."""

		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		sidecar = subsample.cache.cache_path(wav)
		# Valid JSON, current version, but no audio_md5.
		sidecar.write_text(
			json.dumps({"analysis_version": subsample.analysis.ANALYSIS_VERSION}),
			encoding="utf-8",
		)

		result = subsample.cache.load_cache(wav)

		assert result is not None
		# Sidecar was rewritten with a real MD5.
		import json as _json
		payload = _json.loads(sidecar.read_text())
		assert payload["audio_md5"] == subsample.cache.compute_audio_md5(wav)


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
		r_band_energy = result.band_energy

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
		r_band_energy = result.band_energy

		assert isinstance(r_band_energy, subsample.analysis.BandEnergyResult)
		assert all(v == 0.0 for v in r_band_energy.energy_fractions)
		assert all(v == 0.0 for v in r_band_energy.decay_rates)


# ---------------------------------------------------------------------------
# TestEnsureSampleAssets
# ---------------------------------------------------------------------------

class TestEnsureSampleAssets:

	"""Tests for the audio-first orchestrator that drives the recursive
	instrument library load.  Covers the four decision branches: full regen
	(sidecar/PNG cold or stale), PNG-only regen, and the no-op path; plus
	the with_preview=False behaviour and unreadable-audio error path."""

	def _png_path (self, audio_path: pathlib.Path) -> pathlib.Path:
		return audio_path.with_name(audio_path.name + subsample.cache.PREVIEW_PNG_SUFFIX)

	def test_cold_no_sidecar_no_png_regen_both_with_preview_on (
		self,
		tmp_path: pathlib.Path,
	) -> None:
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)

		result = subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		assert result is not None
		sidecar = subsample.cache.cache_path(wav_path)
		assert sidecar.exists()
		# Sidecar carries the preview block when with_preview=True.
		payload = json.loads(sidecar.read_text())
		assert "preview" in payload
		assert self._png_path(wav_path).exists()

	def test_degenerate_audio_skipped_not_raised (self, tmp_path: pathlib.Path) -> None:
		"""A readable-but-degenerate file (a few ms of audio) must warn and
		return None — not let a librosa exception escape and abort the whole
		startup load over one junk file."""

		wav_path = tmp_path / "click.wav"
		tests.helpers._make_wav(wav_path, n_frames=100)

		result = subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		assert result is None

	def test_one_degenerate_file_does_not_abort_library_load (
		self,
		tmp_path: pathlib.Path,
	) -> None:
		"""End-to-end: a directory with one good and one degenerate file loads
		the good one instead of crashing the recursive load."""

		tests.helpers._make_wav(tmp_path / "good.wav")
		tests.helpers._make_wav(tmp_path / "junk.wav", n_frames=100)

		library = subsample.library.load_instrument_library(
			tmp_path, max_memory_bytes=4 * 1024 * 1024, with_preview=False,
		)

		assert library is not None
		names = {r.name for r in library.samples()}
		assert names == {"good"}

	def test_cold_no_sidecar_no_png_with_preview_off (
		self,
		tmp_path: pathlib.Path,
	) -> None:
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)

		result = subsample.cache.ensure_sample_assets(wav_path, with_preview=False)

		assert result is not None
		sidecar = subsample.cache.cache_path(wav_path)
		assert sidecar.exists()
		# No preview block when previews are disabled.
		payload = json.loads(sidecar.read_text())
		assert "preview" not in payload
		assert not self._png_path(wav_path).exists()

	def test_warm_everything_present_is_noop (self, tmp_path: pathlib.Path) -> None:
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)
		# Cold seed produces sidecar + PNG.
		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		sidecar = subsample.cache.cache_path(wav_path)
		png     = self._png_path(wav_path)
		sidecar_mtime_before = sidecar.stat().st_mtime_ns
		png_mtime_before     = png.stat().st_mtime_ns

		# Warm call should not rewrite either file.
		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		assert sidecar.stat().st_mtime_ns == sidecar_mtime_before
		assert png.stat().st_mtime_ns     == png_mtime_before

	def test_png_only_regen_keeps_sidecar (self, tmp_path: pathlib.Path) -> None:
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)
		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		sidecar = subsample.cache.cache_path(wav_path)
		png     = self._png_path(wav_path)
		sidecar_mtime_before = sidecar.stat().st_mtime_ns
		png.unlink()
		assert not png.exists()

		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		# PNG re-rendered from the embedded preview block; sidecar untouched.
		assert png.exists()
		assert sidecar.stat().st_mtime_ns == sidecar_mtime_before

	def test_md5_mismatch_with_previews_on_refreshes_both (
		self,
		tmp_path: pathlib.Path,
	) -> None:

		"""Regression: the preview block was previously silently dropped on
		MD5 mismatch.  ensure_sample_assets must re-embed it."""

		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path, n_frames=2048)
		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		sidecar = subsample.cache.cache_path(wav_path)
		original_payload = json.loads(sidecar.read_text())
		assert "preview" in original_payload

		# Mutate audio bytes — different n_frames → different MD5 → triggers
		# the regen path inside ensure_sample_assets.
		tests.helpers._make_wav(wav_path, n_frames=4096)

		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		updated_payload = json.loads(sidecar.read_text())
		assert "preview" in updated_payload, "Preview block must survive an MD5 regen"
		assert updated_payload["duration"] != original_payload["duration"]
		# And the PNG was rewritten alongside the sidecar.
		assert self._png_path(wav_path).exists()

	def test_unreadable_audio_returns_none (self, tmp_path: pathlib.Path) -> None:
		# Audio file looks valid by extension but contains garbage — the
		# underlying audio reader raises ValueError, which the orchestrator
		# logs and converts to a None return so the library load can skip
		# this sample and continue with the rest.
		wav_path = tmp_path / "broken.wav"
		wav_path.write_bytes(b"not a wav file at all")

		result = subsample.cache.ensure_sample_assets(wav_path, with_preview=False)

		assert result is None

	def test_analysis_honours_configured_float_ceiling (self, tmp_path: pathlib.Path) -> None:
		"""The analysis read scales a hot float source too, not just the playback
		read.  These are two separate reads of the same file, and the sidecar's
		stored level has to describe the audio that actually plays — loudness
		normalisation divides by it."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		wav_path = tmp_path / "hot.wav"
		t = numpy.linspace(0, 0.25, 11025, endpoint=False)
		soundfile.write(
			str(wav_path), (numpy.sin(2 * numpy.pi * 440 * t) * 2.0).astype(numpy.float32),
			44100, subtype="FLOAT",
		)

		previous = subsample.audio._FLOAT_IMPORT_CEILING_DBFS
		subsample.audio.set_float_import_ceiling(-1.0)
		try:
			result = subsample.cache.ensure_sample_assets(wav_path, with_preview=False)
		finally:
			subsample.audio.set_float_import_ceiling(previous)

		assert result is not None
		# The analysed peak sits at the -1 dBFS ceiling, not clipped flat at full scale.
		assert result.level.peak == pytest.approx(10.0 ** (-1.0 / 20.0), abs=0.01)


class TestV11SidecarBackwardCompat:

	"""Sidecars written before v12 lack spectral_rolloff, spectral_slope,
	crest_factor, crest_factor_db, and noise_floor.  Deserialization should
	fall back to 0.0 for all missing fields."""

	def test_missing_spectral_fields_default_to_zero (self, tmp_path: pathlib.Path) -> None:
		"""Spectral fields absent from sidecar deserialize as 0.0."""
		sidecar = tests.helpers._write_sidecar(tmp_path, "old")

		data = json.loads(sidecar.read_text())

		# Simulate a v11 sidecar by removing the new spectral fields
		data["spectral"].pop("spectral_rolloff", None)
		data["spectral"].pop("spectral_slope", None)
		sidecar.write_text(json.dumps(data), encoding="utf-8")

		# Use load_sidecar (skips MD5 check) to test deserialization directly
		result = subsample.cache.load_sidecar(sidecar)
		assert result is not None
		spectral = result.spectral
		assert spectral.spectral_rolloff == 0.0
		assert spectral.spectral_slope == 0.0

	def test_missing_level_fields_default_to_zero (self, tmp_path: pathlib.Path) -> None:
		"""Level fields absent from sidecar deserialize as 0.0."""
		sidecar = tests.helpers._write_sidecar(tmp_path, "old")

		data = json.loads(sidecar.read_text())

		# Simulate a v11 sidecar by removing the new level fields
		data["level"].pop("crest_factor", None)
		data["level"].pop("crest_factor_db", None)
		data["level"].pop("noise_floor", None)
		sidecar.write_text(json.dumps(data), encoding="utf-8")

		# Use load_sidecar (skips MD5 check) to test deserialization directly
		result = subsample.cache.load_sidecar(sidecar)
		assert result is not None
		level = result.level
		assert level.crest_factor == 0.0
		assert level.crest_factor_db == 0.0
		assert level.noise_floor == 0.0


class TestReanalyzePreservesChannelFormat:

	"""Stale-sidecar re-analysis must preserve the original channel_format tag.

	Without this, ambisonic samples would be silently downgraded to "pcm"
	on an ANALYSIS_VERSION bump — the W-channel-only analysis path would
	be skipped, and the player would build a normal mix matrix instead of
	decoding the B-format through the ambisonic decoder matrix.
	"""

	def test_ambisonic_tag_survives_version_mismatch (self, tmp_path: pathlib.Path) -> None:
		"""A b_format_ambix sidecar with a stale analysis_version is re-analysed
		*and* re-tagged b_format_ambix rather than downgraded to pcm.
		"""
		wav = tmp_path / "ambi.wav"
		tests.helpers._make_wav(wav, n_channels=4)
		md5 = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(
			wav, md5, tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
			channels=4, channel_format="b_format_ambix",
		)

		# Corrupt the analysis_version so load_cache triggers re-analysis.
		sidecar = subsample.cache.cache_path(wav)
		data = json.loads(sidecar.read_text())
		data["analysis_version"] = "0"
		sidecar.write_text(json.dumps(data), encoding="utf-8")

		result = subsample.cache.load_cache(wav)
		assert result is not None
		channel_format = result.channel_format
		assert channel_format == "b_format_ambix"

		# New sidecar on disk should also carry the preserved tag.
		new_payload = json.loads(sidecar.read_text())
		assert new_payload["channel_format"] == "b_format_ambix"

	def test_pcm_tag_survives_version_mismatch (self, tmp_path: pathlib.Path) -> None:
		"""Regression guard: non-ambisonic samples still re-analyse as pcm."""
		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)

		subsample.cache.save_cache(
			wav, md5, tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
		)

		sidecar = subsample.cache.cache_path(wav)
		data = json.loads(sidecar.read_text())
		data["analysis_version"] = "0"
		sidecar.write_text(json.dumps(data), encoding="utf-8")

		result = subsample.cache.load_cache(wav)
		assert result is not None
		channel_format = result.channel_format
		assert channel_format == "pcm"


# ---------------------------------------------------------------------------
# TestPreviewRoundtrip
# ---------------------------------------------------------------------------


class TestPreviewRoundtrip:

	"""save_cache accepts a preview_data kwarg; load_preview_data retrieves it."""

	def _preview_data (self) -> "subsample.preview.PreviewData":

		import subsample.preview

		bins = subsample.preview._ENVELOPE_BINS
		n    = subsample.preview._N_BANDS
		return subsample.preview.PreviewData(
			version      = subsample.preview.PREVIEW_VERSION,
			envelope_min = numpy.linspace(-60, 60, bins).astype(numpy.int8),
			envelope_max = numpy.linspace(-40, 80, bins).astype(numpy.int8),
			bands        = tuple(
				numpy.linspace(0, 100 + 5 * b, bins).astype(numpy.int8)
				for b in range(n)
			),
			band_totals  = tuple(1.0 / n for _ in range(n)),
			onset_times  = (0.1, 0.4, 0.7),
			beat_times   = (0.0, 0.5),
			tempo_bpm    = 120.0,
			duration     = 1.0,
			peak         = 0.8,
			rms          = 0.3,
			accent_rgb   = (200, 100, 150),
			pitch_label  = "A3",
			is_rhythmic  = True,
		)

	def test_preview_block_persists_to_sidecar (self, tmp_path: pathlib.Path) -> None:

		import subsample.preview

		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		preview = self._preview_data()

		subsample.cache.save_cache(
			wav, "deadbeef", tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
			preview_data=preview,
		)

		sidecar = subsample.cache.cache_path(wav)
		payload = json.loads(sidecar.read_text())

		assert "preview" in payload
		assert payload["preview"]["version"] == subsample.preview.PREVIEW_VERSION

	def test_no_preview_data_omits_block (self, tmp_path: pathlib.Path) -> None:

		"""save_cache called without preview_data writes no ``preview`` key —
		loaders treat its absence as 'no preview available', no warning."""

		wav = tmp_path / "snare.wav"
		tests.helpers._make_wav(wav)

		subsample.cache.save_cache(
			wav, "deadbeef", tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
		)

		payload = json.loads(subsample.cache.cache_path(wav).read_text())
		assert "preview" not in payload

	def test_load_preview_data_roundtrip (self, tmp_path: pathlib.Path) -> None:

		wav = tmp_path / "hat.wav"
		tests.helpers._make_wav(wav)
		preview = self._preview_data()

		subsample.cache.save_cache(
			wav, "deadbeef", tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
			preview_data=preview,
		)

		loaded = subsample.cache.load_preview_data(wav)
		assert loaded is not None
		assert loaded.version     == preview.version
		assert loaded.pitch_label == preview.pitch_label
		assert loaded.tempo_bpm   == preview.tempo_bpm
		assert (loaded.envelope_min == preview.envelope_min).all()
		assert (loaded.envelope_max == preview.envelope_max).all()
		for b1, b2 in zip(loaded.bands, preview.bands):
			assert (b1 == b2).all()

	def test_load_preview_data_returns_none_when_absent (self, tmp_path: pathlib.Path) -> None:

		"""Sidecars written before the preview feature (no ``preview`` key)
		load normally; load_preview_data just returns None."""

		wav = tmp_path / "legacy.wav"
		tests.helpers._make_wav(wav)

		subsample.cache.save_cache(
			wav, "deadbeef", tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
		)

		assert subsample.cache.load_preview_data(wav) is None

	def test_load_preview_data_returns_none_when_sidecar_missing (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""load_preview_data must tolerate a totally missing sidecar (e.g.
		called before any analysis has run) without raising."""

		wav = tmp_path / "orphan.wav"
		tests.helpers._make_wav(wav)

		assert subsample.cache.load_preview_data(wav) is None

	def test_load_preview_data_returns_none_for_malformed_block (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""A corrupt preview block must not crash the loader — Supervisor
		should just see 'no preview' and move on."""

		wav = tmp_path / "bad.wav"
		tests.helpers._make_wav(wav)
		preview = self._preview_data()

		subsample.cache.save_cache(
			wav, "deadbeef", tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
			preview_data=preview,
		)

		# Corrupt the block: make envelope_min not-base64.
		sidecar = subsample.cache.cache_path(wav)
		payload = json.loads(sidecar.read_text())
		payload["preview"]["envelope_min"] = "not valid base64 at all"
		sidecar.write_text(json.dumps(payload), encoding="utf-8")

		assert subsample.cache.load_preview_data(wav) is None

	def test_main_load_cache_unaffected_by_preview_block (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""The primary load_cache path returns the same 9-tuple whether the
		sidecar carries a preview block or not — preview is a pure bolt-on."""

		wav = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav)
		md5 = subsample.cache.compute_audio_md5(wav)
		preview = self._preview_data()

		subsample.cache.save_cache(
			wav, md5, tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(),
			preview_data=preview,
		)

		result = subsample.cache.load_cache(wav)
		assert result is not None
		# The analysis deserializes intact alongside the preview block.
		assert result.spectral == tests.helpers._make_spectral()
		assert result.duration == pytest.approx(1.0)


class TestLoopPersistence:

	"""Loop points round-trip through the sidecar, and compute_loop's gating."""

	def _loop (self) -> subsample.loopfind.LoopPoints:
		return subsample.loopfind.LoopPoints(start=1000, end=5000, crossfade=200, junction_flux=1.23)

	def _steady_tone (self, seconds: float = 1.5, sr: int = 44100) -> numpy.ndarray:
		"""A sustained harmonic tone — a clean loop candidate for the audio search."""
		t = numpy.arange(int(sr * seconds)) / sr
		x = sum(numpy.sin(2 * numpy.pi * 220.0 * h * t) / h for h in range(1, 6))
		x = x * numpy.minimum(t / 0.01, 1.0)
		return (0.5 * x / numpy.max(numpy.abs(x))).astype(numpy.float32)

	def _save (self, wav: pathlib.Path, loop: typing.Optional[subsample.loopfind.LoopPoints]) -> None:
		tests.helpers._make_wav(wav)
		subsample.cache.save_cache(
			wav, subsample.cache.compute_audio_md5(wav), tests.helpers._make_params(),
			tests.helpers._make_spectral(), tests.helpers._make_rhythm(),
			tests.helpers._make_pitch(), tests.helpers._make_timbre(),
			1.0, tests.helpers._make_level(), loop=loop,
		)

	def test_loop_round_trip (self, tmp_path: pathlib.Path) -> None:
		wav  = tmp_path / "pad.wav"
		loop = self._loop()
		self._save(wav, loop)
		result = subsample.cache.load_cache(wav)
		assert result is not None
		assert result.loop == loop

	def test_no_loop_serializes_null_and_loads_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "hit.wav"
		self._save(wav, None)
		# Stored as an explicit JSON null...
		payload = json.loads(subsample.cache.cache_path(wav).read_text())
		assert "loop" in payload and payload["loop"] is None
		# ...and read back as None.
		result = subsample.cache.load_cache(wav)
		assert result is not None
		assert result.loop is None

	def test_legacy_sidecar_without_loop_key_is_none (self, tmp_path: pathlib.Path) -> None:
		wav = tmp_path / "old.wav"
		self._save(wav, self._loop())
		# Simulate a pre-loop sidecar by dropping the key entirely.
		sidecar = subsample.cache.cache_path(wav)
		payload = json.loads(sidecar.read_text())
		del payload["loop"]
		sidecar.write_text(json.dumps(payload), encoding="utf-8")
		result = subsample.cache.load_sidecar(sidecar)
		assert result is not None
		assert result.loop is None

	def test_compute_loop_none_when_below_duration_floor (self) -> None:
		# Too short for a loop: the is_loopable gate rejects before the audio search.
		loop = subsample.cache.compute_loop(
			self._steady_tone(), 44100,
			tests.helpers._make_spectral(), tests.helpers._make_pitch(),
			tests.helpers._make_level(), 0.3,
		)
		assert loop is None

	def test_compute_loop_finds_points_for_loopable_tone (self) -> None:
		tone = self._steady_tone()
		loop = subsample.cache.compute_loop(
			tone, 44100,
			tests.helpers._make_spectral(), tests.helpers._make_pitch(),
			tests.helpers._make_level(), len(tone) / 44100,
		)
		assert loop is not None
		assert 0 <= loop.start < loop.end <= len(tone)
