"""Tests for subsample.player path-based reference and instrument loading."""

import pathlib
import typing
import unittest.mock

import pytest

import subsample.library
import subsample.player
import subsample.query
import subsample.similarity

import tests.helpers


# ---------------------------------------------------------------------------
# TestLoadReferenceFromPath
# ---------------------------------------------------------------------------

class TestLoadReferenceFromPath:

	def test_returns_record_when_sidecar_present (self, tmp_path: pathlib.Path) -> None:
		"""Returns a SampleRecord when a valid sidecar exists."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "my_ref")
		record = subsample.player._load_reference_from_path(wav_path)
		assert record is not None
		assert record.name == str(wav_path.resolve())
		assert record.audio is None  # references don't carry audio

	def test_returns_none_when_sidecar_missing (self, tmp_path: pathlib.Path) -> None:
		"""Returns None (not raises) when sidecar is absent."""
		path = tmp_path / "no_sidecar.wav"
		path.write_bytes(b"")  # audio file exists but no sidecar
		assert subsample.player._load_reference_from_path(path) is None

	def test_sets_name_to_absolute_path (self, tmp_path: pathlib.Path) -> None:
		"""Record name is the canonical absolute path string (the matrix key)."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "ref_sample")
		record = subsample.player._load_reference_from_path(wav_path)
		assert record is not None
		assert record.name == str(wav_path.resolve())


# ---------------------------------------------------------------------------
# TestLoadInstrumentFromPath
# ---------------------------------------------------------------------------

class TestLoadInstrumentFromPath:

	def test_returns_record_when_wav_and_sidecar_present (self, tmp_path: pathlib.Path) -> None:
		"""Returns a SampleRecord with audio when WAV + sidecar exist."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "my_inst")
		record = subsample.player._load_instrument_from_path(wav_path, with_preview=False)
		assert record is not None
		assert record.name == "my_inst"
		assert record.audio is not None

	def test_analyzes_and_loads_when_sidecar_missing (self, tmp_path: pathlib.Path) -> None:
		"""Auto-generates sidecar and loads the sample when no sidecar exists."""
		path = tmp_path / "no_sidecar.wav"
		tests.helpers._make_wav(path)
		record = subsample.player._load_instrument_from_path(path, with_preview=False)
		assert record is not None
		assert record.name == "no_sidecar"
		# Sidecar should have been written for next time.
		assert subsample.cache.cache_path(path).exists()

	def test_returns_none_when_wav_missing (self, tmp_path: pathlib.Path) -> None:
		"""Returns None when WAV file is absent (sidecar only)."""
		sidecar = tests.helpers._write_sidecar(tmp_path, "sidecar_only")
		wav_path = sidecar.parent / "sidecar_only.wav"
		assert subsample.player._load_instrument_from_path(wav_path, with_preview=False) is None

	def test_sets_name_to_stem (self, tmp_path: pathlib.Path) -> None:
		"""Record name is the filename stem."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "2026-03-27_09-28-12")
		record = subsample.player._load_instrument_from_path(wav_path, with_preview=False)
		assert record is not None
		assert record.name == "2026-03-27_09-28-12"


# ---------------------------------------------------------------------------
# TestResolvePathReferences
# ---------------------------------------------------------------------------

class TestResolvePathReferences:

	def test_path_reference_added_to_matrix (self, tmp_path: pathlib.Path) -> None:
		"""A path-based reference is added to all provided matrices."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "path_ref")
		ref_key = str(wav_path.resolve())

		where = subsample.query.WherePredicate(reference=ref_key)
		select = subsample.query.SelectSpec(where=where, order=(subsample.query.OrderClause(by="similarity", dir="desc"),))
		assignment = subsample.query.Assignment(
			name="test",
			select=(select,),
		)
		note_map: subsample.player.NoteMap = {
			(9, 36): [(assignment, subsample.query.PickSpec(1, 1))],
		}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.samples.return_value = []

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib, with_preview=False)

		matrix.add_reference.assert_called_once()

	def test_bare_name_reference_skipped (self, tmp_path: pathlib.Path) -> None:
		"""Bare-name references (no path) are not passed to add_reference."""
		where = subsample.query.WherePredicate(reference="GM36_BassDrum1")
		select = subsample.query.SelectSpec(where=where, order=(subsample.query.OrderClause(by="similarity", dir="desc"),))
		assignment = subsample.query.Assignment(name="test", select=(select,))
		note_map: subsample.player.NoteMap = {
			(9, 36): [(assignment, subsample.query.PickSpec(1, 1))],
		}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib, with_preview=False)

		matrix.add_reference.assert_not_called()

	def test_path_instrument_loaded_into_library (self, tmp_path: pathlib.Path) -> None:
		"""A path-based name: predicate loads the instrument into the library."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "captured")

		where = subsample.query.WherePredicate(
			name="captured",
			name_path=str(wav_path.resolve()),
		)
		select = subsample.query.SelectSpec(where=where)
		assignment = subsample.query.Assignment(name="test", select=(select,))
		note_map: subsample.player.NoteMap = {
			(9, 38): [(assignment, subsample.query.PickSpec(1, 1))],
		}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.find_by_path.return_value = None  # not already present

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib, with_preview=False)

		instrument_lib.add.assert_called_once()

	def test_directory_in_first_spec_of_fallback_chain_loaded (
		self,
		tmp_path: pathlib.Path,
	) -> None:

		"""A directory predicate in the FIRST spec of a fallback chain is
		collected — the collection walks every spec, not just the last (the
		loop-indentation regression that silently broke fallback chains)."""

		kick_dir = tmp_path / "Kick"
		kick_dir.mkdir()
		tests.helpers._write_wav_and_sidecar(kick_dir, "k")

		primary  = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(directory=str(kick_dir.resolve())),
		)
		fallback = subsample.query.SelectSpec(
			where=subsample.query.WherePredicate(name="anything"),
		)
		assignment = subsample.query.Assignment(name="chained", select=(primary, fallback))

		note_map: subsample.player.NoteMap = {
			(9, 36): [(assignment, subsample.query.PickSpec(1, 1))],
		}

		library = subsample.library.InstrumentLibrary(max_memory_bytes=4 * 1024 * 1024)
		matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		subsample.player._resolve_path_references(note_map, [matrix], library, with_preview=False)

		# The primary spec's directory loaded even though a fallback follows it.
		assert len(library) == 1

	def test_duplicate_path_reference_skipped (self, tmp_path: pathlib.Path) -> None:
		"""When the same path appears in multiple assignments, load only once."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "shared_ref")
		ref_key = str(wav_path.resolve())

		where = subsample.query.WherePredicate(reference=ref_key)
		select = subsample.query.SelectSpec(where=where, order=(subsample.query.OrderClause(by="similarity", dir="desc"),))
		assignment1 = subsample.query.Assignment(name="test1", select=(select,))
		assignment2 = subsample.query.Assignment(name="test2", select=(select,))

		note_map: subsample.player.NoteMap = {
			(9, 36): [(assignment1, subsample.query.PickSpec(1, 1))],
			(9, 38): [(assignment2, subsample.query.PickSpec(1, 1))],
		}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.samples.return_value = []

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib, with_preview=False)

		# add_reference should be called exactly once despite two assignments
		assert matrix.add_reference.call_count == 1


# ---------------------------------------------------------------------------
# TestPresetSelfContainedLoading — a `map:` preset's own directory predicates
# resolve relative to the preset folder and populate a fresh library.  This is
# the core path-resolution that cli._load_bank relies on for a `map:` program.
# ---------------------------------------------------------------------------

class TestPresetSelfContainedLoading:

	def test_preset_directory_predicate_loads_relative_to_preset (
		self,
		tmp_path: pathlib.Path,
	) -> None:

		"""A self-contained kit folder (preset map + Kick/ samples) loads as a unit."""

		# Build kit/{midi-map.yaml, Kick/k.wav (+ sidecar)}.
		kit_dir = tmp_path / "kit"
		kick_dir = kit_dir / "Kick"
		kick_dir.mkdir(parents=True)
		tests.helpers._write_wav_and_sidecar(kick_dir, "k")

		preset_path = kit_dir / "midi-map.yaml"
		preset_path.write_text(
			"assignments:\n"
			"  - name: Kick\n"
			"    channel: 10\n"
			"    notes: 36\n"
			"    select:\n"
			"      where:\n"
			"        directory: Kick\n",
			encoding="utf-8",
		)

		# Parse the preset map; its `directory: Kick` predicate is stamped with
		# the preset folder as the resolution base (midi_map_dir = preset.parent).
		result = subsample.player.load_midi_map(preset_path, [])
		assert (9, 36) in result.note_map

		# A `map:` preset starts with an EMPTY library; _resolve_path_references
		# fills it from the preset's own directory predicates.
		library = subsample.library.InstrumentLibrary(max_memory_bytes=4 * 1024 * 1024)
		matrix  = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)

		subsample.player._resolve_path_references(
			result.note_map, [matrix], library, with_preview=False,
		)

		# The kit's sample loaded — proving the directory predicate resolved
		# against the preset folder, not the CWD or some external root.
		assert len(library) == 1
		loaded = list(library.samples())
		assert loaded[0].filepath is not None
		assert loaded[0].filepath.resolve() == (kick_dir / "k.wav").resolve()
