"""Tests for subsample/library.py — reference and instrument sample libraries."""

import dataclasses
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
	return tests.helpers._write_sidecar(directory, audio_stem, audio_ext)


def _write_wav_and_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	n_frames: int = 2048,
) -> tuple[pathlib.Path, pathlib.Path]:
	return tests.helpers._write_wav_and_sidecar(directory, audio_stem, n_frames)


# ---------------------------------------------------------------------------
# TestSampleRecord
# ---------------------------------------------------------------------------

class TestSampleRecord:

	def test_fields_accessible (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id   = 1,
			name        = "KICK",
			spectral    = tests.helpers._make_spectral(),
			rhythm      = tests.helpers._make_rhythm(),
			pitch       = tests.helpers._make_pitch(),
			timbre      = tests.helpers._make_timbre(),
			level       = tests.helpers._make_level(),
			band_energy = tests.helpers._make_band_energy(),
			params      = tests.helpers._make_params(),
			duration    = 1.23,
		)
		assert record.sample_id == 1
		assert record.name == "KICK"
		assert record.duration == 1.23
		assert record.spectral.attack == pytest.approx(0.2)

	def test_audio_and_filepath_default_to_none (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(),
			band_energy=tests.helpers._make_band_energy(), params=tests.helpers._make_params(), duration=1.0,
		)
		assert record.audio is None
		assert record.filepath is None

	def test_audio_stored_when_provided (self) -> None:
		audio = numpy.zeros((1000, 1), dtype=numpy.int16)
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(),
			band_energy=tests.helpers._make_band_energy(), params=tests.helpers._make_params(), duration=1.0,
			audio=audio,
		)
		assert record.audio is not None
		assert record.audio.shape == (1000, 1)

	def test_is_frozen (self) -> None:
		record = subsample.library.SampleRecord(
			sample_id=1, name="X", spectral=tests.helpers._make_spectral(), rhythm=tests.helpers._make_rhythm(),
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(),
			band_energy=tests.helpers._make_band_energy(), params=tests.helpers._make_params(), duration=1.0,
		)
		with pytest.raises(dataclasses.FrozenInstanceError):
			record.name = "Y"  # type: ignore[misc]


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
		spectral, rhythm, pitch, timbre, params, duration, level, band_energy, channel_format = result
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
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(),
			band_energy=tests.helpers._make_band_energy(), params=tests.helpers._make_params(), duration=1.0,
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
		sample_id   = subsample.library.allocate_id(),
		name        = name,
		spectral    = tests.helpers._make_spectral(),
		rhythm      = tests.helpers._make_rhythm(),
		pitch       = tests.helpers._make_pitch(),
		timbre      = tests.helpers._make_timbre(),
		level       = tests.helpers._make_level(),
		band_energy = tests.helpers._make_band_energy(),
		params      = tests.helpers._make_params(),
		duration    = n_frames / 44100.0,
		audio       = audio,
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

	def test_duplicate_name_readd_stays_findable (self) -> None:
		"""Code-review regression: re-adding a same-name sample (fresh id) must
		replace the prior one and keep find_by_name resolving to the live
		record — not leave a stale duplicate or a dangling name key."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		first  = _make_instrument_record("kick")
		second = _make_instrument_record("kick")   # same name, new sample_id

		assert first.sample_id != second.sample_id

		lib.add(first)
		evicted = lib.add(second)

		# Prior record replaced and reported so callers cascade-clean it.
		assert first.sample_id in evicted
		assert lib.get(first.sample_id) is None
		# Name still resolves — to the live record, not None.
		assert lib.find_by_name("kick") == second.sample_id
		# No stale duplicate lingering.
		assert [r.name for r in lib.samples()].count("kick") == 1

	def test_eviction_after_same_name_readd_keeps_name (self) -> None:
		"""After a same-name re-add, evicting other samples must not delete the
		name key the live record now owns."""
		# limit fits exactly two 2000-byte records.
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=4000)
		lib.add(_make_instrument_record("kick", n_frames=1000))
		live = _make_instrument_record("kick", n_frames=1000)
		lib.add(live)                                   # replaces the first kick
		lib.add(_make_instrument_record("snare", n_frames=1000))   # may evict

		assert lib.find_by_name("kick") == live.sample_id

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
			pitch=tests.helpers._make_pitch(), timbre=tests.helpers._make_timbre(), level=tests.helpers._make_level(),
			band_energy=tests.helpers._make_band_energy(), params=tests.helpers._make_params(), duration=1.0,
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
		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)
		assert len(lib) == 1
		record = lib.samples()[0]
		assert record.name == "KICK"
		assert record.audio is not None

	def test_audio_has_correct_shape (self, tmp_path: pathlib.Path) -> None:
		_write_wav_and_sidecar(tmp_path, "KICK", n_frames=2048)
		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)
		record = lib.samples()[0]
		assert record.audio is not None
		assert record.audio.shape[0] == 2048  # n_frames
		assert record.audio.shape[1] == 1     # mono

	def test_deletes_orphaned_sidecar_unconditionally (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		# Write sidecar only — no WAV.  The orphan sweep is unconditional
		# (no opt-in flag), so the sidecar should be deleted and an INFO
		# line emitted.
		sidecar = _write_sidecar(tmp_path, "KICK")
		with caplog.at_level(logging.INFO, logger="subsample.library"):
			lib = subsample.library.load_instrument_library(
				tmp_path, 10 * 1024 * 1024, with_preview=False,
			)
		assert len(lib) == 0
		assert not sidecar.exists()
		messages = [r.message for r in caplog.records]
		assert any("orphaned" in m.lower() for m in messages)

	def test_assigns_unique_ids (self, tmp_path: pathlib.Path) -> None:
		_write_wav_and_sidecar(tmp_path, "KICK")
		_write_wav_and_sidecar(tmp_path, "SNARE")
		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)
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
			lib = subsample.library.load_instrument_library(
				missing, 10 * 1024 * 1024, with_preview=False,
			)
		assert len(lib) == 0

	def test_filepath_populated (self, tmp_path: pathlib.Path) -> None:
		wav_path, _ = _write_wav_and_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)
		record = lib.samples()[0]
		assert record.filepath == wav_path

	def test_orphan_sweep_keeps_good_samples (self, tmp_path: pathlib.Path) -> None:
		# One orphan sidecar + one valid pair — valid sample loads; orphan is cleaned up.
		sidecar_orphan = _write_sidecar(tmp_path, "ORPHAN")
		_write_wav_and_sidecar(tmp_path, "KICK")
		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)
		assert len(lib) == 1
		assert lib.samples()[0].name == "KICK"
		assert not sidecar_orphan.exists()

	def test_orphan_sweep_logs_error_on_permission_failure (
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
					tmp_path, 10 * 1024 * 1024, with_preview=False,
				)

		assert sidecar.exists(), "Sidecar must survive when deletion fails"
		assert any(r.levelname == "ERROR" for r in caplog.records), "Expected ERROR log"


# ---------------------------------------------------------------------------
# TestLoadInstrumentLibraryRecursive
# ---------------------------------------------------------------------------

class TestLoadInstrumentLibraryRecursive:

	"""Behaviour added by the recursive, audio-first library load:
	subdirectory discovery, automatic sidecar/PNG regeneration on startup,
	the orphan sweep, and stem-collision detection."""

	def _png_path (self, audio_path: pathlib.Path) -> pathlib.Path:

		"""Compound-suffix PNG path for an audio file (mirrors recorder logic)."""

		return audio_path.with_name(audio_path.name + subsample.cache.PREVIEW_PNG_SUFFIX)

	def test_recurses_into_subdirectories (self, tmp_path: pathlib.Path) -> None:
		(tmp_path / "kicks").mkdir()
		(tmp_path / "snares").mkdir()
		_write_wav_and_sidecar(tmp_path / "kicks", "K1")
		_write_wav_and_sidecar(tmp_path / "snares", "S1")

		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		names = {s.name for s in lib.samples()}
		assert names == {"K1", "S1"}

	def test_missing_sidecar_regenerated (self, tmp_path: pathlib.Path) -> None:
		# Audio only — no sidecar — should trigger ensure_sample_assets to
		# analyse and write the sidecar on the way through.
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)
		sidecar_path = subsample.cache.cache_path(wav_path)
		assert not sidecar_path.exists()

		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		assert len(lib) == 1
		assert sidecar_path.exists()

	def test_missing_png_regenerated_with_previews_on (self, tmp_path: pathlib.Path) -> None:
		# Seed a complete sidecar (preview block + PNG) via the orchestrator,
		# delete just the PNG, then confirm the load restores the PNG without
		# rewriting the sidecar.
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)
		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)

		sidecar_path = subsample.cache.cache_path(wav_path)
		png_path     = self._png_path(wav_path)
		sidecar_mtime_before = sidecar_path.stat().st_mtime_ns
		png_path.unlink()

		subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=True,
		)

		assert png_path.exists()
		assert sidecar_path.stat().st_mtime_ns == sidecar_mtime_before

	def test_missing_png_not_regenerated_with_previews_off (
		self,
		tmp_path: pathlib.Path,
	) -> None:
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)
		subsample.cache.ensure_sample_assets(wav_path, with_preview=False)

		png_path = self._png_path(wav_path)
		assert not png_path.exists()

		subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		assert not png_path.exists()

	def test_missing_preview_block_triggers_sidecar_rewrite (
		self,
		tmp_path: pathlib.Path,
	) -> None:
		# _write_wav_and_sidecar produces a sidecar WITHOUT a preview block.
		# To isolate the "missing preview block" trigger, first patch the
		# sidecar's audio_md5 to match the audio (so the load doesn't fall
		# back to MD5-mismatch full regen for unrelated reasons).
		import json
		wav_path, sidecar_path = _write_wav_and_sidecar(tmp_path, "kick")
		payload = json.loads(sidecar_path.read_text())
		payload["audio_md5"] = subsample.cache.compute_audio_md5(wav_path)
		sidecar_path.write_text(json.dumps(payload))
		assert "preview" not in payload

		subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=True,
		)

		updated = json.loads(sidecar_path.read_text())
		assert "preview" in updated, "Sidecar should now embed a preview block"
		assert self._png_path(wav_path).exists()

	def test_md5_mismatch_refreshes_preview_block (self, tmp_path: pathlib.Path) -> None:

		"""Regression: _reanalyze_and_save previously dropped the preview block
		on MD5 mismatch.  ensure_sample_assets must re-embed it so the on-disk
		sidecar stays consistent with the audio it describes."""

		import json
		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path, n_frames=2048)

		# Seed a complete sidecar with the preview block via the orchestrator.
		subsample.cache.ensure_sample_assets(wav_path, with_preview=True)
		sidecar_path = subsample.cache.cache_path(wav_path)
		original = json.loads(sidecar_path.read_text())
		assert "preview" in original

		# Mutate audio bytes (different n_frames → different MD5).
		tests.helpers._make_wav(wav_path, n_frames=4096)

		subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=True,
		)

		updated = json.loads(sidecar_path.read_text())
		assert "preview" in updated, "Preview block must survive an MD5 regen"
		assert updated["duration"] != original["duration"], (
			"Duration field should reflect the longer audio"
		)

	def test_orphan_png_deleted_in_root (self, tmp_path: pathlib.Path) -> None:
		# A PNG named for an audio file that doesn't exist is an orphan —
		# the sweep should delete it regardless of with_preview, since stale
		# PNGs mislead visual auditioning.
		orphan_png = tmp_path / ("missing.wav" + subsample.cache.PREVIEW_PNG_SUFFIX)
		orphan_png.write_bytes(b"not really a png")

		subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		assert not orphan_png.exists()

	def test_orphan_in_nested_directory (self, tmp_path: pathlib.Path) -> None:
		# Orphans deep in the tree should be discovered by the recursive
		# sweep just like top-level ones.
		deep = tmp_path / "a" / "b" / "c"
		deep.mkdir(parents=True)
		sidecar = _write_sidecar(deep, "ghost")
		png = deep / ("ghost.wav" + subsample.cache.PREVIEW_PNG_SUFFIX)
		png.write_bytes(b"x")

		subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		assert not sidecar.exists()
		assert not png.exists()
		assert deep.exists(), "Empty directory is left in place; we never created it"

	def test_directory_with_only_orphans (self, tmp_path: pathlib.Path) -> None:
		_write_sidecar(tmp_path, "ghost1")
		_write_sidecar(tmp_path, "ghost2")

		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		assert len(lib) == 0
		assert tmp_path.exists()
		assert list(tmp_path.iterdir()) == []

	def test_stem_collision_raises (self, tmp_path: pathlib.Path) -> None:
		# Two audio files with the same filename stem in different
		# subdirectories would silently overwrite each other in the stem-
		# keyed _name_index — fail loud instead so the user can fix it.
		(tmp_path / "kicks").mkdir()
		(tmp_path / "snares").mkdir()
		_write_wav_and_sidecar(tmp_path / "kicks", "01")
		_write_wav_and_sidecar(tmp_path / "snares", "01")

		with pytest.raises(subsample.library.InstrumentLibraryError) as excinfo:
			subsample.library.load_instrument_library(
				tmp_path, 10 * 1024 * 1024, with_preview=False,
			)

		msg = str(excinfo.value)
		assert "01" in msg
		assert "kicks" in msg
		assert "snares" in msg

	def test_collision_across_banks_is_allowed (self, tmp_path: pathlib.Path) -> None:
		# Banks are independent InstrumentLibrary instances, so the same
		# stem appearing once in each bank is fine — only within-library
		# collisions are an error.  Verified by loading two separate
		# directories with the same stem and asserting both succeed.
		bank_a = tmp_path / "bank_a"
		bank_b = tmp_path / "bank_b"
		bank_a.mkdir()
		bank_b.mkdir()
		_write_wav_and_sidecar(bank_a, "kick")
		_write_wav_and_sidecar(bank_b, "kick")

		lib_a = subsample.library.load_instrument_library(
			bank_a, 10 * 1024 * 1024, with_preview=False,
		)
		lib_b = subsample.library.load_instrument_library(
			bank_b, 10 * 1024 * 1024, with_preview=False,
		)

		assert len(lib_a) == 1
		assert len(lib_b) == 1
		assert lib_a.samples()[0].name == "kick"
		assert lib_b.samples()[0].name == "kick"


# ---------------------------------------------------------------------------
# TestLoadWavAudio
# ---------------------------------------------------------------------------

class TestLoadWavAudio:

	def test_loads_16bit_wav (self, tmp_path: pathlib.Path) -> None:
		path = tmp_path / "test.wav"
		tests.helpers._make_wav(path, n_frames=512)
		audio = subsample.library.load_wav_audio(path)
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
			result = subsample.library.load_wav_audio(tmp_path / "missing.wav")
		assert result is None
