"""Tests for subsample/library.py — reference and instrument sample libraries."""

import json
import pathlib

import numpy
import pytest

import subsample.analysis
import subsample.cache
import subsample.library

import tests.helpers


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

	spectral = tests.helpers._make_spectral()
	rhythm   = tests.helpers._make_rhythm()
	pitch    = tests.helpers._make_pitch()
	timbre   = tests.helpers._make_timbre()
	level    = tests.helpers._make_level()
	params   = tests.helpers._make_params()

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
			"log_attack_time":    spectral.log_attack_time,
			"spectral_flux":      spectral.spectral_flux,
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
		},
		"timbre": {
			"mfcc":       list(timbre.mfcc),
			"mfcc_delta": list(timbre.mfcc_delta),
			"mfcc_onset": list(timbre.mfcc_onset),
		},
		"level": {
			"peak": level.peak,
			"rms":  level.rms,
		},
	}

	sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

	return sidecar_path


def _write_wav_and_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	n_frames: int = 2048,
) -> tuple[pathlib.Path, pathlib.Path]:

	"""Write a WAV file and its sidecar for instrument library tests."""

	wav_path = directory / (audio_stem + ".wav")
	tests.helpers._make_wav(wav_path, n_frames=n_frames)
	sidecar_path = _write_sidecar(directory, audio_stem)

	return wav_path, sidecar_path


# ---------------------------------------------------------------------------
# TestSampleRecord
# ---------------------------------------------------------------------------

class TestSampleRecord:

	def test_fields_accessible (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id = 1,
			name      = "KICK",
			spectral  = tests.helpers._make_spectral(),
			rhythm    = tests.helpers._make_rhythm(),
			pitch     = tests.helpers._make_pitch(),
			timbre    = tests.helpers._make_timbre(),
			level     = tests.helpers._make_level(),
			params    = tests.helpers._make_params(),
			duration  = 1.23,
		)
		assert record.sample_id == 1
		assert record.name == "KICK"
		assert record.duration == 1.23
		assert record.spectral.attack == pytest.approx(0.2)

	def test_audio_and_filepath_default_to_none (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(), params=tests.helpers._make_params(), duration=1.0,
		)
		assert record.audio is None
		assert record.filepath is None

	def test_audio_stored_when_provided (self) -> None:
		audio = numpy.zeros((1000, 1), dtype=numpy.int16)
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(), params=tests.helpers._make_params(), duration=1.0,
			audio=audio,
		)
		assert record.audio is not None
		assert record.audio.shape == (1000, 1)

	def test_is_frozen (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(), params=tests.helpers._make_params(), duration=1.0,
		)
		with pytest.raises(Exception):
			record.name = "Y"  # type: ignore[misc]

	def test_as_vector_delegates_to_spectral (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(), params=tests.helpers._make_params(), duration=1.0,
		)
		assert numpy.array_equal(record.as_vector(), record.spectral.as_vector())


# ---------------------------------------------------------------------------
# TestAllocateId
# ---------------------------------------------------------------------------

class TestAllocateId:

	def test_ids_are_unique (self) -> None:
		ids = [subsample.library.allocate_id() for _ in range(10)]
		assert len(set(ids)) == 10

	def test_ids_are_sequential (self) -> None:
		a = subsample.library.allocate_id()
		b = subsample.library.allocate_id()
		assert b == a + 1


# ---------------------------------------------------------------------------
# TestLoadSidecar
# ---------------------------------------------------------------------------

class TestLoadSidecar:

	def test_valid_sidecar_loads (self, tmp_path: pathlib.Path) -> None:
		sidecar = _write_sidecar(tmp_path, "kick")
		result = subsample.cache.load_sidecar(sidecar)
		assert result is not None
		spectral, rhythm, pitch, timbre, params, duration, level = result
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

	def _record (self, name: str) -> subsample.library.SampleRecord:
		return subsample.library.SampleRecord(
			sample_id=subsample.library.allocate_id(),
			name=name, spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(), params=tests.helpers._make_params(), duration=1.0,
		)

	def _library_with (self, records: list[subsample.library.SampleRecord]) -> subsample.library.ReferenceLibrary:
		return subsample.library.ReferenceLibrary(records)

	def test_empty_library (self) -> None:
		lib = self._library_with([])
		assert len(lib) == 0
		assert lib.names() == []
		assert lib.samples() == []

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
		assert [r.name for r in lib.samples()] == ["HAT", "KICK", "SNARE"]

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

	def test_assigns_unique_ids (self, tmp_path: pathlib.Path) -> None:
		_write_sidecar(tmp_path, "KICK")
		_write_sidecar(tmp_path, "SNARE")
		lib = subsample.library.load_reference_library(tmp_path)
		ids = [r.sample_id for r in lib.samples()]
		assert len(set(ids)) == 2

	def test_audio_is_none_for_reference (self, tmp_path: pathlib.Path) -> None:
		_write_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_reference_library(tmp_path)
		assert lib.get("KICK") is not None
		assert lib.get("KICK").audio is None  # type: ignore[union-attr]

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
		_write_sidecar(tmp_path, "KICK")
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
		subdir = tmp_path / "Roland TR-808"
		subdir.mkdir()
		_write_sidecar(subdir, "SD0000")
		_write_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_reference_library(tmp_path)
		assert len(lib) == 1
		assert lib.get("KICK") is not None


# ---------------------------------------------------------------------------
# TestInstrumentLibrary
# ---------------------------------------------------------------------------

def _make_instrument_record (
	name: str,
	n_frames: int = 1000,
	channels: int = 1,
) -> subsample.library.SampleRecord:

	"""Return a SampleRecord with audio data for instrument library tests."""

	audio = numpy.zeros((n_frames, channels), dtype=numpy.int16)
	return subsample.library.SampleRecord(
		sample_id = subsample.library.allocate_id(),
		name      = name,
		spectral  = tests.helpers._make_spectral(),
		rhythm    = tests.helpers._make_rhythm(),
		pitch     = tests.helpers._make_pitch(),
		timbre    = tests.helpers._make_timbre(),
		level     = tests.helpers._make_level(),
		params    = tests.helpers._make_params(),
		duration  = n_frames / 44100.0,
		audio     = audio,
	)


class TestInstrumentLibrary:

	def test_empty_library (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=1024 * 1024)
		assert len(lib) == 0
		assert lib.samples() == []
		assert lib.memory_used == 0

	def test_add_and_get (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=1024 * 1024)
		record = _make_instrument_record("KICK")
		lib.add(record)
		assert lib.get(record.sample_id) is record

	def test_get_missing_returns_none (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=1024 * 1024)
		assert lib.get(99999) is None

	def test_all_returns_insertion_order (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		r1 = _make_instrument_record("A")
		r2 = _make_instrument_record("B")
		r3 = _make_instrument_record("C")
		lib.add(r1)
		lib.add(r2)
		lib.add(r3)
		names = [r.name for r in lib.samples()]
		assert names == ["A", "B", "C"]

	def test_len (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		lib.add(_make_instrument_record("A"))
		lib.add(_make_instrument_record("B"))
		assert len(lib) == 2

	def test_memory_used_reflects_audio_bytes (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		record = _make_instrument_record("KICK", n_frames=1000)
		expected_bytes = record.audio.nbytes  # type: ignore[union-attr]
		lib.add(record)
		assert lib.memory_used == expected_bytes

	def test_fifo_eviction_removes_oldest (self) -> None:
		# Each record is 1000 int16 samples = 2000 bytes; limit = 4000 bytes → fits 2
		limit = 4000
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=limit)
		r1 = _make_instrument_record("A", n_frames=1000)
		r2 = _make_instrument_record("B", n_frames=1000)
		r3 = _make_instrument_record("C", n_frames=1000)
		lib.add(r1)
		lib.add(r2)
		evicted = lib.add(r3)
		# r1 should be evicted (oldest)
		assert r1.sample_id in evicted
		assert lib.get(r1.sample_id) is None
		assert lib.get(r2.sample_id) is r2
		assert lib.get(r3.sample_id) is r3

	def test_add_returns_evicted_ids (self) -> None:
		limit = 2000  # fits one 1000-frame int16 record
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=limit)
		r1 = _make_instrument_record("A", n_frames=1000)
		r2 = _make_instrument_record("B", n_frames=1000)
		lib.add(r1)
		evicted = lib.add(r2)
		assert r1.sample_id in evicted

	def test_no_eviction_when_within_limit (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		evicted = lib.add(_make_instrument_record("A"))
		assert evicted == []

	def test_add_no_audio_no_eviction (self) -> None:
		# Records with audio=None contribute 0 bytes
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=1)
		record = subsample.library.SampleRecord(
			sample_id=subsample.library.allocate_id(),
			name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(), params=tests.helpers._make_params(), duration=1.0,
		)
		evicted = lib.add(record)
		assert evicted == []
		assert len(lib) == 1

	def test_repr_contains_count_and_memory (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=1024 * 1024)
		lib.add(_make_instrument_record("KICK"))
		r = repr(lib)
		assert "1" in r

	def test_memory_limit_property (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=5 * 1024 * 1024)
		assert lib.memory_limit == 5 * 1024 * 1024

	def test_oversized_sample_logs_warning_and_is_still_added (
		self,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		# 1-byte limit, 2000-byte sample — should warn but still be added
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=1)
		record = _make_instrument_record("HUGE", n_frames=1000)
		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			lib.add(record)
		assert any("exceeds" in r.message for r in caplog.records)
		assert lib.get(record.sample_id) is record

	def test_zero_memory_limit_adds_sample (self) -> None:
		# max_memory_bytes=0: the oversized-sample guard (`> 0`) is not triggered,
		# and the eviction loop condition (`> self._max_bytes`) is always True but
		# the queue starts empty, so the first sample is added without eviction.
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=0)
		record = _make_instrument_record("A")
		evicted = lib.add(record)
		assert lib.get(record.sample_id) is record
		assert evicted == []

	def test_find_by_name_returns_id (self) -> None:
		"""find_by_name returns the sample_id for a loaded sample."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		r = _make_instrument_record("my-kick")
		lib.add(r)

		assert lib.find_by_name("my-kick") == r.sample_id

	def test_find_by_name_missing_returns_none (self) -> None:
		"""find_by_name returns None for a name not in the library."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)

		assert lib.find_by_name("no-such-sample") is None

	def test_find_by_name_evicted_returns_none (self) -> None:
		"""find_by_name returns None after a sample has been evicted."""
		r1 = _make_instrument_record("old-kick", n_frames=500)
		r2 = _make_instrument_record("new-kick", n_frames=500)
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=r1.audio.nbytes + 10)
		lib.add(r1)
		lib.add(r2)   # evicts r1

		assert lib.find_by_name("old-kick") is None
		assert lib.find_by_name("new-kick") == r2.sample_id

	def test_single_add_evicts_multiple_old_samples (self) -> None:
		# Three small records: 500 int16 frames × 1 channel = 1000 bytes each (3000 total).
		# One large record: 2000 int16 frames = 4000 bytes.
		# Limit = 4500. After filling with smalls, 3000 + 4000 > 4500 → evict r1,
		# 2000 + 4000 > 4500 → evict r2, 1000 + 4000 > 4500 → evict r3,
		# 0 + 4000 ≤ 4500 → stop. All three evicted in one add() call.
		limit = 4500
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=limit)
		r1 = _make_instrument_record("A", n_frames=500)
		r2 = _make_instrument_record("B", n_frames=500)
		r3 = _make_instrument_record("C", n_frames=500)
		r_large = _make_instrument_record("BIG", n_frames=2000)
		lib.add(r1)
		lib.add(r2)
		lib.add(r3)
		evicted = lib.add(r_large)
		assert set(evicted) == {r1.sample_id, r2.sample_id, r3.sample_id}
		assert lib.get(r_large.sample_id) is r_large
		assert len(lib) == 1


# ---------------------------------------------------------------------------
# TestLoadInstrumentLibrary
# ---------------------------------------------------------------------------

class TestLoadInstrumentLibrary:

	def test_loads_wav_and_sidecar (self, tmp_path: pathlib.Path) -> None:
		_write_wav_and_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_instrument_library(tmp_path, 10 * 1024 * 1024)
		assert len(lib) == 1
		record = lib.samples()[0]
		assert record.name == "KICK"
		assert record.audio is not None

	def test_audio_has_correct_shape (self, tmp_path: pathlib.Path) -> None:
		_write_wav_and_sidecar(tmp_path, "KICK", n_frames=2048)
		lib = subsample.library.load_instrument_library(tmp_path, 10 * 1024 * 1024)
		record = lib.samples()[0]
		assert record.audio is not None
		assert record.audio.shape[0] == 2048  # n_frames
		assert record.audio.shape[1] == 1     # mono

	def test_skips_missing_wav (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		# Write sidecar only — no WAV
		_write_sidecar(tmp_path, "KICK")
		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			lib = subsample.library.load_instrument_library(tmp_path, 10 * 1024 * 1024)
		assert len(lib) == 0
		assert any("not found" in r.message.lower() for r in caplog.records)

	def test_assigns_unique_ids (self, tmp_path: pathlib.Path) -> None:
		_write_wav_and_sidecar(tmp_path, "KICK")
		_write_wav_and_sidecar(tmp_path, "SNARE")
		lib = subsample.library.load_instrument_library(tmp_path, 10 * 1024 * 1024)
		ids = [r.sample_id for r in lib.samples()]
		assert len(set(ids)) == 2

	def test_nonexistent_directory_returns_empty (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		missing = tmp_path / "no_such_dir"
		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			lib = subsample.library.load_instrument_library(missing, 10 * 1024 * 1024)
		assert len(lib) == 0

	def test_filepath_populated (self, tmp_path: pathlib.Path) -> None:
		wav_path, _ = _write_wav_and_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_instrument_library(tmp_path, 10 * 1024 * 1024)
		record = lib.samples()[0]
		assert record.filepath == wav_path

	def test_orphan_disabled_logs_warning_and_hint (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		# Sidecar without WAV — default behaviour (clean_orphaned_sidecars=False)
		sidecar = _write_sidecar(tmp_path, "KICK")
		with caplog.at_level(logging.DEBUG, logger="subsample.library"):
			lib = subsample.library.load_instrument_library(
				tmp_path, 10 * 1024 * 1024, clean_orphaned_sidecars=False,
			)

		assert len(lib) == 0
		assert sidecar.exists(), "Sidecar must NOT be deleted when feature is disabled"
		messages = [r.message for r in caplog.records]
		assert any("not found" in m.lower() for m in messages), "Expected WARNING about missing audio"
		assert any("clean_orphaned_sidecars" in m for m in messages), "Expected one-time hint"

	def test_orphan_disabled_hint_appears_only_once (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		# Multiple orphans — hint should appear only once
		_write_sidecar(tmp_path, "KICK")
		_write_sidecar(tmp_path, "SNARE")
		_write_sidecar(tmp_path, "HAT")
		with caplog.at_level(logging.DEBUG, logger="subsample.library"):
			subsample.library.load_instrument_library(
				tmp_path, 10 * 1024 * 1024, clean_orphaned_sidecars=False,
			)

		hint_count = sum(1 for r in caplog.records if "clean_orphaned_sidecars" in r.message)
		assert hint_count == 1, f"Hint should appear exactly once, got {hint_count}"

	def test_orphan_enabled_deletes_sidecar (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		sidecar = _write_sidecar(tmp_path, "KICK")
		with caplog.at_level(logging.DEBUG, logger="subsample.library"):
			lib = subsample.library.load_instrument_library(
				tmp_path, 10 * 1024 * 1024, clean_orphaned_sidecars=True,
			)

		assert len(lib) == 0
		assert not sidecar.exists(), "Sidecar should have been deleted"
		messages = [r.message for r in caplog.records]
		assert any("orphaned" in m.lower() for m in messages), "Expected INFO about deletion"
		assert not any("clean_orphaned_sidecars" in m for m in messages), "No hint when feature is enabled"

	def test_orphan_enabled_keeps_good_samples (self, tmp_path: pathlib.Path) -> None:
		# One orphan + one valid pair — valid sample loads; orphan is cleaned up
		sidecar_orphan = _write_sidecar(tmp_path, "ORPHAN")
		_write_wav_and_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, clean_orphaned_sidecars=True,
		)
		assert len(lib) == 1
		assert lib.samples()[0].name == "KICK"
		assert not sidecar_orphan.exists()

	def test_orphan_enabled_logs_error_on_permission_failure (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		import unittest.mock
		sidecar = _write_sidecar(tmp_path, "KICK")

		with unittest.mock.patch.object(
			type(sidecar), "unlink",
			side_effect=OSError("Permission denied"),
		):
			with caplog.at_level(logging.DEBUG, logger="subsample.library"):
				subsample.library.load_instrument_library(
					tmp_path, 10 * 1024 * 1024, clean_orphaned_sidecars=True,
				)

		assert sidecar.exists(), "Sidecar must survive when deletion fails"
		assert any(r.levelname == "ERROR" for r in caplog.records), "Expected ERROR log"


# ---------------------------------------------------------------------------
# TestLoadWavAudio
# ---------------------------------------------------------------------------

class TestLoadWavAudio:

	def test_loads_16bit_wav (self, tmp_path: pathlib.Path) -> None:
		path = tmp_path / "test.wav"
		tests.helpers._make_wav(path, n_frames=512)
		audio = subsample.library._load_wav_audio(path)
		assert audio is not None
		assert audio.dtype == numpy.int16
		assert audio.shape == (512, 1)

	def test_missing_file_returns_none (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			result = subsample.library._load_wav_audio(tmp_path / "missing.wav")
		assert result is None
