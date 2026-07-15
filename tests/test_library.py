"""Tests for subsample/library.py — reference and instrument sample libraries."""

import dataclasses
import pathlib
import typing

import numpy
import pytest

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.library
import subsample.query

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
		assert result.spectral.attack == pytest.approx(0.2)
		assert result.duration == pytest.approx(1.0)

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
	filepath: typing.Optional[pathlib.Path] = None,
) -> subsample.library.SampleRecord:

	"""Return a SampleRecord with audio data for instrument library tests.

	``filepath`` defaults to None (a filepath-less in-memory record, which the
	library keys by name); pass one to exercise the resolved-filepath identity
	path (the on-disk / take-folder case).
	"""

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
		filepath    = filepath,
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

	def test_same_path_readd_replaces (self) -> None:
		"""A same-PATH re-add (a re-analysed file, fresh id) replaces the prior
		record and reports it evicted — the recorder/re-analysis overwrite flow,
		now keyed by resolved filepath rather than stem."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		p = pathlib.Path("/kit/01.wav")
		first  = _make_instrument_record("01", filepath=p)
		second = _make_instrument_record("01", filepath=p)   # same path, new id

		lib.add(first)
		evicted = lib.add(second)

		assert first.sample_id in evicted
		assert lib.get(first.sample_id) is None
		assert lib.find_by_path(p) == second.sample_id
		assert len(lib) == 1

	def test_same_stem_different_path_coexist (self) -> None:
		"""Two records sharing a stem but at different paths COEXIST — identity is
		the resolved filepath (the take-folder case, at unit level)."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		a = _make_instrument_record("01", filepath=pathlib.Path("/kit/mid/01.wav"))
		b = _make_instrument_record("01", filepath=pathlib.Path("/kit/outer/01.wav"))

		evicted = lib.add(a) + lib.add(b)

		assert evicted == []                        # neither evicts the other
		assert len(lib) == 2
		assert a.sample_id != b.sample_id
		assert lib.find_by_path(pathlib.Path("/kit/mid/01.wav")) == a.sample_id
		assert lib.find_by_path(pathlib.Path("/kit/outer/01.wav")) == b.sample_id

	def test_find_by_path_missing_returns_none (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		p = pathlib.Path("/kit/01.wav")
		lib.add(_make_instrument_record("01", filepath=p))
		assert lib.find_by_path(pathlib.Path("/kit/02.wav")) is None

	def test_eviction_drops_path_index_key (self) -> None:
		"""FIFO eviction of a filepath-bearing record removes its path-index key
		so a later find_by_path returns None."""
		# ~2000 bytes/record; budget fits two.
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=4000)
		a = _make_instrument_record("01", n_frames=1000, filepath=pathlib.Path("/kit/a/01.wav"))
		lib.add(a)
		lib.add(_make_instrument_record("02", n_frames=1000, filepath=pathlib.Path("/kit/b/02.wav")))
		lib.add(_make_instrument_record("03", n_frames=1000, filepath=pathlib.Path("/kit/c/03.wav")))

		assert lib.get(a.sample_id) is None                          # 'a' FIFO-evicted
		assert lib.find_by_path(pathlib.Path("/kit/a/01.wav")) is None

	def test_remove_by_path_removes_and_returns_id (self) -> None:
		"""remove_by_path drops the record, returns its id (for cascade-clean),
		and frees its indexes — the deleted-file path that kills the ghost."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		p = pathlib.Path("/kit/01.wav")
		r = _make_instrument_record("01", filepath=p)
		lib.add(r)

		assert lib.remove_by_path(p) == r.sample_id
		assert lib.get(r.sample_id) is None
		assert lib.find_by_path(p) is None
		assert len(lib) == 0

	def test_remove_by_path_missing_returns_none (self) -> None:
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		assert lib.remove_by_path(pathlib.Path("/kit/nope.wav")) is None

	def test_remove_by_path_leaves_the_twin (self) -> None:
		"""Removing one take-folder's "01" must not touch its same-stem twin in
		another folder (the re-encode/rename case is path-scoped)."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		a = _make_instrument_record("01", filepath=pathlib.Path("/kit/mid/01.wav"))
		b = _make_instrument_record("01", filepath=pathlib.Path("/kit/outer/01.wav"))
		lib.add(a)
		lib.add(b)

		lib.remove_by_path(pathlib.Path("/kit/mid/01.wav"))

		assert lib.get(a.sample_id) is None
		assert lib.get(b.sample_id) is b            # the twin survives
		assert lib.find_by_path(pathlib.Path("/kit/outer/01.wav")) == b.sample_id

	def test_remove_by_path_frees_memory (self) -> None:
		"""remove_by_path decrements the byte accounting so later eviction math
		stays correct."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		r = _make_instrument_record("01", n_frames=1000, filepath=pathlib.Path("/kit/01.wav"))
		before = lib.memory_used
		lib.add(r)
		assert lib.memory_used == before + r.audio.nbytes

		lib.remove_by_path(pathlib.Path("/kit/01.wav"))
		assert lib.memory_used == before
		assert len(lib) == 0

	def test_mixed_filepath_and_filepathless_same_name_coexist (self) -> None:
		"""A filepath-bearing record and a filepath-less one sharing a name use
		different indexes (path vs name), so they coexist and resolve distinctly."""
		lib = subsample.library.InstrumentLibrary(max_memory_bytes=10 * 1024 * 1024)
		on_disk = _make_instrument_record("01", filepath=pathlib.Path("/kit/01.wav"))
		in_mem  = _make_instrument_record("01", filepath=None)

		lib.add(on_disk)
		lib.add(in_mem)

		assert len(lib) == 2
		assert lib.find_by_path(pathlib.Path("/kit/01.wav")) == on_disk.sample_id
		assert lib.find_by_name("01") == in_mem.sample_id

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

	def test_same_stem_different_folders_coexist (
		self, tmp_path: pathlib.Path,
	) -> None:
		"""Two files sharing a filename stem in different folders load as
		DISTINCT samples — identity is the resolved filepath, not the stem (the
		per-technique take-folder layout where each folder holds "01.wav")."""

		for sub in ("a_kicks", "z_snares"):
			(tmp_path / sub).mkdir()

		_write_wav_and_sidecar(tmp_path / "a_kicks", "01", n_frames=4096)
		_write_wav_and_sidecar(tmp_path / "z_snares", "01", n_frames=4096)

		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		samples = lib.samples()
		assert len(samples) == 2
		assert {s.name for s in samples} == {"01"}                    # same label
		assert len({s.sample_id for s in samples}) == 2               # distinct identity
		assert {s.filepath.parent.name for s in samples} == {"a_kicks", "z_snares"}
		# Each is addressable by its true key — path — and the two disambiguate.
		for s in samples:
			assert lib.find_by_path(s.filepath) == s.sample_id

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
	the orphan sweep, and same-stem-in-different-subdirs coexistence (identity
	is the resolved filepath, not the stem)."""

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

	def test_same_stem_in_subdirs_loads_both (self, tmp_path: pathlib.Path) -> None:
		# Two audio files sharing a filename stem in different subdirectories
		# used to hard-fail (the library was stem-keyed); identity is now the
		# resolved filepath, so both load and find_by_path disambiguates them.
		(tmp_path / "kicks").mkdir()
		(tmp_path / "snares").mkdir()
		_write_wav_and_sidecar(tmp_path / "kicks", "01")
		_write_wav_and_sidecar(tmp_path / "snares", "01")

		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		assert len(lib) == 2
		by_folder = {s.filepath.parent.name: s.sample_id for s in lib.samples()}
		assert set(by_folder) == {"kicks", "snares"}
		assert lib.find_by_path(tmp_path / "kicks" / "01.wav") == by_folder["kicks"]
		assert lib.find_by_path(tmp_path / "snares" / "01.wav") == by_folder["snares"]

	def test_directory_predicate_disambiguates_twins (self, tmp_path: pathlib.Path) -> None:
		"""End to end: a colliding two-folder tree loads both takes, and a
		`directory:` + `name:` select resolves to the intended one (the user's
		ride-take-folder workflow)."""
		(tmp_path / "mid").mkdir()
		(tmp_path / "outer").mkdir()
		_write_wav_and_sidecar(tmp_path / "mid", "01")
		_write_wav_and_sidecar(tmp_path / "outer", "01")

		lib = subsample.library.load_instrument_library(
			tmp_path, 10 * 1024 * 1024, with_preview=False,
		)

		where = subsample.query.WherePredicate(
			name="01", directory=str((tmp_path / "outer").resolve()),
		)
		ranked = subsample.query.query(
			subsample.query.SelectSpec(where=where), lib.samples(), None,
		)

		assert [s.filepath.parent.name for s in ranked] == ["outer"]

	def test_over_budget_multi_take_warns (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""A tree whose combined audio exceeds the memory limit warns about the
		aggregate FIFO eviction (a real risk once multi-take trees can load)."""
		import logging
		_write_wav_and_sidecar(tmp_path, "a", n_frames=8192)
		_write_wav_and_sidecar(tmp_path, "b", n_frames=8192)

		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			subsample.library.load_instrument_library(tmp_path, 4000, with_preview=False)

		assert any("exceed the memory limit" in r.message for r in caplog.records)

	def test_in_budget_load_does_not_aggregate_warn (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""A load that fits the memory limit must not fire the aggregate warning."""
		import logging
		_write_wav_and_sidecar(tmp_path, "a", n_frames=1000)
		_write_wav_and_sidecar(tmp_path, "b", n_frames=1000)

		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			subsample.library.load_instrument_library(tmp_path, 10 * 1024 * 1024, with_preview=False)

		assert not any("exceed the memory limit" in r.message for r in caplog.records)

	def test_single_over_budget_sample_does_not_aggregate_warn (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""One sample that alone exceeds the budget is kept resident (nothing is
		evicted), so the AGGREGATE eviction warning must NOT fire."""
		import logging
		_write_wav_and_sidecar(tmp_path, "big", n_frames=8192)

		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			subsample.library.load_instrument_library(tmp_path, 4000, with_preview=False)

		assert not any("were evicted" in r.message for r in caplog.records)

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

	def test_honours_configured_float_ceiling (self, tmp_path: pathlib.Path) -> None:
		"""A hot 32-bit float sample loaded straight into the library scales to fit
		instead of hard-clipping — the ceiling is not CLI-import-only."""
		import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

		path = tmp_path / "hot.wav"
		soundfile.write(
			str(path), numpy.array([[2.0], [0.5]], dtype=numpy.float32), 44100, subtype="FLOAT",
		)

		previous = subsample.audio._FLOAT_IMPORT_CEILING_DBFS
		subsample.audio.set_float_import_ceiling(-1.0)
		try:
			audio = subsample.library.load_wav_audio(path)
		finally:
			subsample.audio.set_float_import_ceiling(previous)

		assert audio is not None
		assert audio.max() < numpy.iinfo(numpy.int32).max               # not clipped
		expected_peak = (10.0 ** (-1.0 / 20.0)) * (2 ** 31)
		assert abs(audio.max() - expected_peak) < expected_peak * 0.01

	def test_missing_file_returns_none (
		self,
		tmp_path: pathlib.Path,
		caplog: pytest.LogCaptureFixture,
	) -> None:
		import logging
		with caplog.at_level(logging.WARNING, logger="subsample.library"):
			result = subsample.library.load_wav_audio(tmp_path / "missing.wav")
		assert result is None

	def test_24bit_fullscale_resample_no_overflow (self, tmp_path: pathlib.Path) -> None:
		"""A full-scale 24-bit sample that Gibbs-overshoots on resampling must not
		overflow the int32 cast.

		In float32 the int32 ceiling (2147483647) rounds up to 2^31, so a peak
		pushed past 1.0 by resampling ringing used to survive the clip as 2^31 and
		wrap to the full-negative rail on the cast (an audible click), emitting
		"invalid value encountered in cast".  The conversion now promotes to
		float64, where the ceiling is exact.
		"""
		import warnings

		import soundfile

		# A full-scale step (-1 -> +1) at 44100 Hz; band-limited resampling to
		# 48000 rings past 1.0 at the discontinuity.  PCM_24 so the int32 path runs.
		n      = 4096
		signal = numpy.concatenate([
			-numpy.ones(n // 2, dtype=numpy.float32),
			numpy.ones(n // 2, dtype=numpy.float32),
		])
		path = tmp_path / "fullscale24.wav"
		soundfile.write(str(path), signal, 44100, subtype="PCM_24")

		# The bug surfaced as a RuntimeWarning — promote it to an error so the
		# regression fails loudly if the float32 conversion ever returns.
		with warnings.catch_warnings():
			warnings.simplefilter("error", RuntimeWarning)
			audio = subsample.library.load_wav_audio(path, target_sample_rate=48000)

		assert audio is not None
		assert audio.dtype == numpy.int32

		# The overshoot clamps to the positive int32 rail — under the bug it would
		# have wrapped to the negative rail, so the max would be negative/garbage.
		assert int(audio.max()) == 2147483647
