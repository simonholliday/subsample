"""MIDI bank switching — multiple instrument directories selectable at runtime.

Provides the data types and manager for organising instrument samples into
discrete banks that can be switched via MIDI Program Change messages during
a live performance.

Architecture
------------

BankDefinition (frozen dataclass)
    Parsed from the optional ``banks:`` key in the MIDI map YAML file.
    Describes one bank: human-readable name, directory path, and the MIDI
    program number that activates it.

Bank (dataclass)
    Runtime representation of a loaded bank.  Bundles an InstrumentLibrary,
    SimilarityMatrix, and TransformManager — all the state needed for the
    player to query and render samples from this bank.

BankManager
    Thread-safe coordinator.  Holds all loaded banks keyed by program number,
    tracks the active bank, and provides switch_to() for the player's
    Program Change handler.

Usage flow
----------

1. ``load_midi_map()`` extracts the optional ``banks:`` and ``bank_channel:``
   keys from the MIDI map YAML and returns them in a ``MidiMapResult``.

2. ``cli.py`` calls ``load_bank()`` for each ``BankDefinition``, then
   constructs a ``BankManager`` and passes it to ``MidiPlayer``.

3. On receiving a MIDI Program Change, the player calls
   ``bank_manager.switch_to(program)`` and subsequent note triggers query
   the new active bank.

When no ``banks:`` key is present in the MIDI map, ``cli.py`` wraps the
single ``cfg.instrument.directory`` library in a one-bank ``BankManager``
transparently — the player code path is identical in both cases.
"""

import dataclasses
import logging
import pathlib
import threading
import typing

import subsample.library
import subsample.similarity


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BankDefinition — parsed from YAML
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BankDefinition:

	"""A bank declaration parsed from the MIDI map ``banks:`` list.

	Fields:
		name:      Human-readable label shown in logs and the startup banner.
		directory: Path to the sample directory (WAV + .analysis.json pairs).
		program:   MIDI Program Change number (0-127) that activates this bank.
	"""

	name:      str
	directory: str
	program:   int


def parse_banks (raw: typing.Any) -> list[BankDefinition]:

	"""Parse the ``banks:`` key from MIDI map YAML into BankDefinition objects.

	Each entry must have ``name`` (str) and ``directory`` (str).
	``program`` is optional and defaults to the list index.

	Args:
		raw: The value of the ``banks:`` key from the parsed YAML dict.
		     Expected to be a list of dicts.

	Returns:
		Ordered list of BankDefinition.  Empty list if raw is None or empty.

	Raises:
		ValueError: If any entry is malformed or program numbers are duplicated.
	"""

	if raw is None:
		return []

	if not isinstance(raw, list):
		raise ValueError("MIDI map 'banks' must be a list")

	definitions: list[BankDefinition] = []
	seen_programs: dict[int, str] = {}

	for idx, entry in enumerate(raw):

		if not isinstance(entry, dict):
			raise ValueError(f"MIDI map banks[{idx}]: expected a mapping, got {type(entry).__name__}")

		name = entry.get("name")
		if name is None:
			raise ValueError(f"MIDI map banks[{idx}]: missing required 'name'")

		directory = entry.get("directory")
		if directory is None:
			raise ValueError(f"MIDI map banks[{idx}] ({name!r}): missing required 'directory'")

		program = int(entry.get("program", idx))

		if not 0 <= program <= 127:
			raise ValueError(
				f"MIDI map banks[{idx}] ({name!r}): program {program} is outside [0, 127]"
			)

		if program in seen_programs:
			raise ValueError(
				f"MIDI map banks[{idx}] ({name!r}): duplicate program number {program} "
				f"(already used by {seen_programs[program]!r})"
			)

		seen_programs[program] = name
		definitions.append(BankDefinition(name=name, directory=str(directory), program=program))

	return definitions


# ---------------------------------------------------------------------------
# Bank — runtime loaded bank
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Bank:

	"""A loaded instrument bank ready for playback.

	Bundles all the per-bank state the player needs: the sample library,
	similarity index, and transform manager.  Each bank is independent —
	switching banks swaps all three atomically.

	Fields:
		name:                Human-readable label.
		directory:           Path to the bank's sample directory.
		program:             MIDI Program Change number.
		instrument_library:  Loaded samples for this bank.
		similarity_matrix:   Similarity index for this bank's samples.
		transform_manager:   Transform pipeline for this bank (may be None
		                     when transforms are disabled).
	"""

	name:                str
	directory:           pathlib.Path
	program:             int
	instrument_library:  subsample.library.InstrumentLibrary
	similarity_matrix:   subsample.similarity.SimilarityMatrix
	transform_manager:   typing.Any = None  # subsample.transform.TransformManager (avoid circular import)


# ---------------------------------------------------------------------------
# BankManager — thread-safe active-bank coordinator
# ---------------------------------------------------------------------------

# Default MIDI channel for Program Change bank switching.
# Channel 10 is the GM drum channel — the most natural choice for a
# drum-sample playback tool.  User-facing 1-16; stored internally as
# mido 0-indexed (9).  0 = omni (match any channel).
DEFAULT_BANK_CHANNEL: int = 10


class BankManager:

	"""Thread-safe manager for switching between instrument banks.

	The player holds a reference to the BankManager and delegates library,
	similarity, and transform lookups to the active bank.  On receiving a
	MIDI Program Change, the player calls switch_to() which atomically
	swaps the active bank under a lock.

	Active voices are not affected by a bank switch — they hold direct
	references to their audio buffers.  Only new note triggers query the
	new bank.
	"""

	def __init__ (
		self,
		banks: list[Bank],
		bank_channel: int = DEFAULT_BANK_CHANNEL,
		default_program: typing.Optional[int] = None,
	) -> None:

		"""Initialise with a list of loaded banks.

		Args:
			banks:           Non-empty list of loaded Bank objects.
			bank_channel:    MIDI channel (1-16, user-facing) that listens for
			                 Program Change messages.  0 = omni (any channel).
			default_program: MIDI program number of the bank to activate at
			                 startup.  When None (or not found), the first bank
			                 in the list is used.

		Raises:
			ValueError: If banks is empty or contains duplicate program numbers.
		"""

		if not banks:
			raise ValueError("BankManager requires at least one bank")

		self._banks: dict[int, Bank] = {}

		for bank in banks:
			if bank.program in self._banks:
				raise ValueError(
					f"Duplicate program number {bank.program}: "
					f"{self._banks[bank.program].name!r} and {bank.name!r}"
				)
			self._banks[bank.program] = bank

		self._lock: threading.Lock = threading.Lock()
		self._active: Bank = self._banks.get(default_program, banks[0]) if default_program is not None else banks[0]

		# Store as mido 0-indexed internally.  0 (omni) stays as -1 to
		# distinguish from channel 1 (mido 0).
		self._bank_channel_mido: int = bank_channel - 1 if bank_channel > 0 else -1

	# -- Properties --------------------------------------------------------

	@property
	def active_bank (self) -> Bank:
		"""The currently active bank.  Thread-safe read."""
		with self._lock:
			return self._active

	@property
	def bank_channel_mido (self) -> int:
		"""Mido 0-indexed channel for PC messages, or -1 for omni."""
		return self._bank_channel_mido

	@property
	def bank_count (self) -> int:
		return len(self._banks)

	# -- Switching ---------------------------------------------------------

	def switch_to (self, program: int) -> bool:

		"""Switch the active bank by MIDI program number.

		Args:
			program: MIDI Program Change number (0-127).

		Returns:
			True if the bank was switched (or already active).
			False if the program number is unknown.
		"""

		with self._lock:

			if self._active.program == program:
				_log.debug("Bank switch to program %d — already active (%s)", program, self._active.name)
				return True

			bank = self._banks.get(program)

			if bank is None:
				_log.warning("Bank switch to program %d — no bank mapped to this program", program)
				return False

			old_name = self._active.name
			self._active = bank
			_log.info("Bank switch: %s → %s (program %d)", old_name, bank.name, program)
			return True

	# -- Lookup ------------------------------------------------------------

	def get_bank (self, program: int) -> typing.Optional[Bank]:
		"""Look up a bank by program number.  Returns None if not found."""
		return self._banks.get(program)

	def all_banks (self) -> list[Bank]:
		"""Return all banks in program-number order."""
		return [self._banks[p] for p in sorted(self._banks)]

	# -- Hot-reload --------------------------------------------------------

	def update_banks (self, banks: list[Bank], bank_channel: int = DEFAULT_BANK_CHANNEL) -> None:

		"""Replace the bank set (used during MIDI map hot-reload).

		If the previously active bank's program number still exists in the
		new set, it remains active.  Otherwise the first bank becomes active.

		Args:
			banks:        New list of loaded Bank objects.
			bank_channel: New bank channel (user-facing 1-16, or 0 for omni).
		"""

		new_map: dict[int, Bank] = {}

		for bank in banks:
			new_map[bank.program] = bank

		with self._lock:
			old_program = self._active.program
			self._banks = new_map
			self._bank_channel_mido = bank_channel - 1 if bank_channel > 0 else -1

			if old_program in new_map:
				self._active = new_map[old_program]
			elif new_map:
				first = new_map[min(new_map)]
				self._active = first
				_log.info(
					"Active bank program %d removed during reload — switched to %s (program %d)",
					old_program, first.name, first.program,
				)
