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
		record = subsample.player._load_instrument_from_path(wav_path)
		assert record is not None
		assert record.name == "my_inst"
		assert record.audio is not None

	def test_analyzes_and_loads_when_sidecar_missing (self, tmp_path: pathlib.Path) -> None:
		"""Auto-generates sidecar and loads the sample when no sidecar exists."""
		path = tmp_path / "no_sidecar.wav"
		tests.helpers._make_wav(path)
		record = subsample.player._load_instrument_from_path(path)
		assert record is not None
		assert record.name == "no_sidecar"
		# Sidecar should have been written for next time.
		assert subsample.cache.cache_path(path).exists()

	def test_returns_none_when_wav_missing (self, tmp_path: pathlib.Path) -> None:
		"""Returns None when WAV file is absent (sidecar only)."""
		sidecar = tests.helpers._write_sidecar(tmp_path, "sidecar_only")
		wav_path = sidecar.parent / "sidecar_only.wav"
		assert subsample.player._load_instrument_from_path(wav_path) is None

	def test_sets_name_to_stem (self, tmp_path: pathlib.Path) -> None:
		"""Record name is the filename stem."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "2026-03-27_09-28-12")
		record = subsample.player._load_instrument_from_path(wav_path)
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
		select = subsample.query.SelectSpec(where=where, order_by="similarity")
		assignment = subsample.query.Assignment(
			name="test",
			select=(select,),
		)
		note_map: subsample.player.NoteMap = {(9, 36): (assignment, 1)}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.samples.return_value = []

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib)

		matrix.add_reference.assert_called_once()

	def test_bare_name_reference_skipped (self, tmp_path: pathlib.Path) -> None:
		"""Bare-name references (no path) are not passed to add_reference."""
		where = subsample.query.WherePredicate(reference="GM36_BassDrum1")
		select = subsample.query.SelectSpec(where=where, order_by="similarity")
		assignment = subsample.query.Assignment(name="test", select=(select,))
		note_map: subsample.player.NoteMap = {(9, 36): (assignment, 1)}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib)

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
		note_map: subsample.player.NoteMap = {(9, 38): (assignment, 1)}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.find_by_name.return_value = None  # not already present

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib)

		instrument_lib.add.assert_called_once()

	def test_duplicate_path_reference_skipped (self, tmp_path: pathlib.Path) -> None:
		"""When the same path appears in multiple assignments, load only once."""
		wav_path, _ = tests.helpers._write_wav_and_sidecar(tmp_path, "shared_ref")
		ref_key = str(wav_path.resolve())

		where = subsample.query.WherePredicate(reference=ref_key)
		select = subsample.query.SelectSpec(where=where, order_by="similarity")
		assignment1 = subsample.query.Assignment(name="test1", select=(select,))
		assignment2 = subsample.query.Assignment(name="test2", select=(select,))

		note_map: subsample.player.NoteMap = {
			(9, 36): (assignment1, 1),
			(9, 38): (assignment2, 1),
		}

		matrix = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		instrument_lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		instrument_lib.samples.return_value = []

		subsample.player._resolve_path_references(note_map, [matrix], instrument_lib)

		# add_reference should be called exactly once despite two assignments
		assert matrix.add_reference.call_count == 1
