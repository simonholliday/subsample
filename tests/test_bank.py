"""Tests for subsample.bank — MIDI bank switching.

Covers BankDefinition parsing, BankManager lifecycle, and integration with
the MIDI map parser (load_midi_map with banks: key).
"""

import pathlib
import typing

import pytest

import subsample.bank
import subsample.library
import subsample.player
import subsample.similarity

import tests.helpers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bank (
	name: str = "Test Bank",
	program: int = 0,
	directory: str = "/tmp/bank",
) -> subsample.bank.Bank:

	"""Create a minimal Bank for unit testing (empty library, no similarity)."""

	library = subsample.library.InstrumentLibrary(max_memory_bytes=1024 * 1024)

	# SimilarityMatrix requires a ReferenceLibrary; use a minimal stub.
	# For BankManager tests we only need the Bank to exist, not query samples.
	return subsample.bank.Bank(
		name=name,
		directory=pathlib.Path(directory),
		program=program,
		instrument_library=library,
		similarity_matrix=typing.cast(subsample.similarity.SimilarityMatrix, None),
		transform_manager=None,
	)


# ---------------------------------------------------------------------------
# TestParseBanks
# ---------------------------------------------------------------------------

class TestParseBanks:

	def test_none_returns_empty (self) -> None:
		assert subsample.bank.parse_banks(None) == []

	def test_empty_list_returns_empty (self) -> None:
		assert subsample.bank.parse_banks([]) == []

	def test_single_bank (self) -> None:
		raw = [{"name": "Kit A", "directory": "./samples/a"}]
		result = subsample.bank.parse_banks(raw)
		assert len(result) == 1
		assert result[0].name == "Kit A"
		assert result[0].directory == "./samples/a"
		assert result[0].program == 0  # default = list index

	def test_multiple_banks_default_programs (self) -> None:
		raw = [
			{"name": "Kit A", "directory": "./a"},
			{"name": "Kit B", "directory": "./b"},
			{"name": "Kit C", "directory": "./c"},
		]
		result = subsample.bank.parse_banks(raw)
		assert [d.program for d in result] == [0, 1, 2]

	def test_explicit_program_numbers (self) -> None:
		raw = [
			{"name": "Kit A", "directory": "./a", "program": 10},
			{"name": "Kit B", "directory": "./b", "program": 20},
		]
		result = subsample.bank.parse_banks(raw)
		assert result[0].program == 10
		assert result[1].program == 20

	def test_duplicate_program_raises (self) -> None:
		raw = [
			{"name": "Kit A", "directory": "./a", "program": 5},
			{"name": "Kit B", "directory": "./b", "program": 5},
		]
		with pytest.raises(ValueError, match="duplicate program"):
			subsample.bank.parse_banks(raw)

	def test_missing_name_raises (self) -> None:
		raw = [{"directory": "./a"}]
		with pytest.raises(ValueError, match="missing required 'name'"):
			subsample.bank.parse_banks(raw)

	def test_missing_directory_raises (self) -> None:
		raw = [{"name": "Kit A"}]
		with pytest.raises(ValueError, match="missing required 'directory'"):
			subsample.bank.parse_banks(raw)

	def test_program_out_of_range_raises (self) -> None:
		raw = [{"name": "Kit A", "directory": "./a", "program": 128}]
		with pytest.raises(ValueError, match="outside \\[0, 127\\]"):
			subsample.bank.parse_banks(raw)

	def test_non_list_raises (self) -> None:
		with pytest.raises(ValueError, match="must be a list"):
			subsample.bank.parse_banks("not a list")

	def test_non_dict_entry_raises (self) -> None:
		with pytest.raises(ValueError, match="expected a mapping"):
			subsample.bank.parse_banks(["not a dict"])


# ---------------------------------------------------------------------------
# TestBankManager
# ---------------------------------------------------------------------------

class TestBankManager:

	def test_single_bank_active_by_default (self) -> None:
		bank = _make_bank(name="Solo", program=0)
		bm = subsample.bank.BankManager([bank])
		assert bm.active_bank is bank
		assert bm.bank_count == 1

	def test_first_bank_is_default_active (self) -> None:
		a = _make_bank(name="A", program=0)
		b = _make_bank(name="B", program=1)
		bm = subsample.bank.BankManager([a, b])
		assert bm.active_bank is a

	def test_switch_to_valid_program (self) -> None:
		a = _make_bank(name="A", program=0)
		b = _make_bank(name="B", program=1)
		bm = subsample.bank.BankManager([a, b])

		assert bm.switch_to(1) is True
		assert bm.active_bank is b

	def test_switch_to_already_active (self) -> None:
		a = _make_bank(name="A", program=0)
		bm = subsample.bank.BankManager([a])

		assert bm.switch_to(0) is True
		assert bm.active_bank is a

	def test_switch_to_unknown_program (self) -> None:
		a = _make_bank(name="A", program=0)
		bm = subsample.bank.BankManager([a])

		assert bm.switch_to(99) is False
		assert bm.active_bank is a  # unchanged

	def test_empty_banks_raises (self) -> None:
		with pytest.raises(ValueError, match="at least one bank"):
			subsample.bank.BankManager([])

	def test_duplicate_program_raises (self) -> None:
		a = _make_bank(name="A", program=0)
		b = _make_bank(name="B", program=0)
		with pytest.raises(ValueError, match="Duplicate program"):
			subsample.bank.BankManager([a, b])

	def test_all_banks_sorted (self) -> None:
		a = _make_bank(name="A", program=5)
		b = _make_bank(name="B", program=2)
		c = _make_bank(name="C", program=8)
		bm = subsample.bank.BankManager([a, b, c])

		result = bm.all_banks()
		assert [bank.program for bank in result] == [2, 5, 8]

	def test_get_bank (self) -> None:
		a = _make_bank(name="A", program=0)
		bm = subsample.bank.BankManager([a])

		assert bm.get_bank(0) is a
		assert bm.get_bank(99) is None

	def test_bank_channel_mido (self) -> None:
		a = _make_bank(name="A", program=0)

		bm = subsample.bank.BankManager([a], bank_channel=10)
		assert bm.bank_channel_mido == 9  # 10 - 1

		bm_omni = subsample.bank.BankManager([a], bank_channel=0)
		assert bm_omni.bank_channel_mido == -1  # omni

	def test_update_banks_keeps_active (self) -> None:
		a = _make_bank(name="A", program=0)
		b = _make_bank(name="B", program=1)
		bm = subsample.bank.BankManager([a, b])
		bm.switch_to(1)

		# Replace banks — program 1 still exists.
		b2 = _make_bank(name="B v2", program=1)
		c  = _make_bank(name="C", program=2)
		bm.update_banks([b2, c])

		assert bm.active_bank is b2

	def test_update_banks_removes_active_falls_back (self) -> None:
		a = _make_bank(name="A", program=0)
		b = _make_bank(name="B", program=1)
		bm = subsample.bank.BankManager([a, b])
		bm.switch_to(1)

		# Replace — program 1 is gone.
		c = _make_bank(name="C", program=5)
		bm.update_banks([c])

		assert bm.active_bank is c  # falls back to first


# ---------------------------------------------------------------------------
# TestLoadMidiMapBanks — integration with load_midi_map()
# ---------------------------------------------------------------------------

class TestLoadMidiMapBanks:

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_no_banks_key (self, tmp_path: pathlib.Path) -> None:
		"""MIDI map with no banks: key returns empty definitions."""
		path = self._write_map(tmp_path, """
assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        reference: BD0025
""")
		result = subsample.player.load_midi_map(path, ["BD0025"])
		assert result.bank_definitions == []
		assert result.bank_channel == subsample.bank.DEFAULT_BANK_CHANNEL
		assert (9, 36) in result.note_map

	def test_banks_parsed (self, tmp_path: pathlib.Path) -> None:
		"""Banks are extracted from the MIDI map."""
		path = self._write_map(tmp_path, """
banks:
  - name: Acoustic
    directory: ./samples/acoustic
    program: 0
  - name: Electronic
    directory: ./samples/electronic
    program: 1

bank_channel: 10

assignments:
  - name: Kick
    channel: 10
    notes: 36
    select:
      where:
        name: kick
""")
		result = subsample.player.load_midi_map(path, [])
		assert len(result.bank_definitions) == 2
		assert result.bank_definitions[0].name == "Acoustic"
		assert result.bank_definitions[1].directory == "./samples/electronic"
		assert result.bank_channel == 10

	def test_bank_channel_custom (self, tmp_path: pathlib.Path) -> None:
		"""Custom bank_channel is parsed."""
		path = self._write_map(tmp_path, """
banks:
  - name: Kit
    directory: ./kit

bank_channel: 1

assignments: []
""")
		result = subsample.player.load_midi_map(path, [])
		assert result.bank_channel == 1

	def test_banks_with_duplicate_program_raises (self, tmp_path: pathlib.Path) -> None:
		"""Duplicate program numbers in banks raise ValueError."""
		path = self._write_map(tmp_path, """
banks:
  - name: A
    directory: ./a
    program: 5
  - name: B
    directory: ./b
    program: 5

assignments: []
""")
		with pytest.raises(ValueError, match="duplicate program"):
			subsample.player.load_midi_map(path, [])

	def test_empty_yaml_returns_result (self, tmp_path: pathlib.Path) -> None:
		"""An empty YAML file returns a MidiMapResult with empty defaults."""
		path = self._write_map(tmp_path, "")
		result = subsample.player.load_midi_map(path, [])
		assert result.note_map == {}
		assert result.bank_definitions == []
