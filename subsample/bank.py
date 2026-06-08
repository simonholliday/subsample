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

1. ``load_midi_map()`` extracts the optional ``programs:``, ``program_channel:``,
   and ``default_program:`` keys from the MIDI map YAML and returns them in a
   ``MidiMapResult``.  ``default_program`` selects which program is active at
   startup (``cli.py`` passes it as ``default_program`` to ``BankManager``).
   (Internally these are still modelled as "banks" — a switchable sample
   library — but the YAML surface uses the MIDI-correct "program" vocabulary
   since each is selected by a Program Change, not by MIDI Bank Select.)

2. ``cli.py`` calls ``_load_bank()`` for each ``BankDefinition``, then
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

	"""A program declaration parsed from the MIDI map ``programs:`` list.

	Each entry carries exactly one of ``directory`` (the shorthand form —
	swap the sample pool, reuse the top-level ``assignments:``) or
	``map_path`` (the preset form — a whole mapper file that brings its own
	assignments and, via its own relative ``directory:`` predicates, its own
	samples).  ``parse_banks`` enforces the XOR.

	Fields:
		name:      Human-readable label shown in logs and the startup banner.
		directory: Path to the sample directory (WAV + .analysis.json pairs),
		           or None when this entry is a ``map:`` preset.
		map_path:  Path to a preset mapper file (resolved relative to the
		           parent map's directory), or None for the ``directory:`` form.
		program:   MIDI Program Change number (0-127) that activates this program.
	"""

	name:      str
	directory: typing.Optional[str] = None
	map_path:  typing.Optional[str] = None
	program:   int                  = 0


def parse_banks (raw: typing.Any) -> list[BankDefinition]:

	"""Parse the ``programs:`` key from MIDI map YAML into BankDefinition objects.

	Each entry must have ``name`` (str) and exactly one of ``directory`` (str —
	swap the pool, reuse top-level assignments) or ``map`` (str — a preset
	mapper file with its own assignments + samples).  ``program`` is optional
	and defaults to the list index.

	Args:
		raw: The value of the ``programs:`` key from the parsed YAML dict.
		     Expected to be a list of dicts.

	Returns:
		Ordered list of BankDefinition.  Empty list if raw is None or empty.

	Raises:
		ValueError: If any entry is malformed or program numbers are duplicated.
	"""

	if raw is None:
		return []

	if not isinstance(raw, list):
		raise ValueError("MIDI map 'programs' must be a list")

	definitions: list[BankDefinition] = []
	seen_programs: dict[int, str] = {}

	for idx, entry in enumerate(raw):

		if not isinstance(entry, dict):
			raise ValueError(f"MIDI map programs[{idx}]: expected a mapping, got {type(entry).__name__}")

		name = entry.get("name")
		if name is None:
			raise ValueError(f"MIDI map programs[{idx}]: missing required 'name'")

		directory = entry.get("directory")
		map_path  = entry.get("map")

		if directory is None and map_path is None:
			raise ValueError(
				f"MIDI map programs[{idx}] ({name!r}): needs exactly one of "
				f"'map' or 'directory'"
			)
		if directory is not None and map_path is not None:
			raise ValueError(
				f"MIDI map programs[{idx}] ({name!r}): 'map' and 'directory' "
				f"are mutually exclusive"
			)

		program = int(entry.get("program", idx))

		if not 0 <= program <= 127:
			raise ValueError(
				f"MIDI map programs[{idx}] ({name!r}): program {program} is outside [0, 127]"
			)

		if program in seen_programs:
			raise ValueError(
				f"MIDI map programs[{idx}] ({name!r}): duplicate program number {program} "
				f"(already used by {seen_programs[program]!r})"
			)

		seen_programs[program] = name
		definitions.append(BankDefinition(
			name=name,
			directory=str(directory) if directory is not None else None,
			map_path=str(map_path) if map_path is not None else None,
			program=program,
		))

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

	A ``map:`` preset additionally carries its own assignment rules
	(``note_map`` / ``zone_templates`` / ``mapped_ccs``).  When these are
	None (the ``directory:`` shorthand form), the player keeps its
	top-level/global rules and only the sample pool swaps.  When they are
	set, a Program Change swaps both the pool AND the rules.

	Fields:
		name:                Human-readable label.
		directory:           Path to the bank's sample directory (or the
		                     preset folder for a ``map:`` program).
		program:             MIDI Program Change number.
		instrument_library:  Loaded samples for this bank.
		similarity_matrix:   Similarity index for this bank's samples.
		transform_manager:   Transform pipeline for this bank (may be None
		                     when transforms are disabled).
		note_map:            This preset's manual note routing (NoteMap), or
		                     None to use the player's top-level rules.
		zone_templates:      This preset's zone-tuned templates, or None.
		mapped_ccs:          This preset's referenced CC numbers, or None.
	"""

	name:                str
	directory:           pathlib.Path
	program:             int
	instrument_library:  subsample.library.InstrumentLibrary
	similarity_matrix:   subsample.similarity.SimilarityMatrix
	transform_manager:   typing.Any              = None  # subsample.transform.TransformManager (avoid circular import)
	note_map:            typing.Optional[typing.Any]        = None  # player.NoteMap (avoid circular import)
	zone_templates:      typing.Optional[tuple[typing.Any, ...]] = None  # tuple[player.ZoneTemplate, ...]
	mapped_ccs:          typing.Optional[set[int]]          = None


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
		"""Number of loaded banks."""
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

		"""Replace the bank set (intended for MIDI map hot-reload).

		NOT wired into the live reload path: bank changes on a watched MIDI map
		are warn-only (cli.py), so this has no production caller today.  Two
		caveats to fix before wiring it up: ``bank_channel`` defaults to
		DEFAULT_BANK_CHANNEL (silently overwriting a configured channel if the
		caller omits it), and there is no ``default_program`` parameter.

		If the previously active bank's program number still exists in the
		new set, it remains active.  Otherwise the first bank becomes active.

		Args:
			banks:        New list of loaded Bank objects.
			bank_channel: New bank channel (user-facing 1-16, or 0 for omni).
		"""

		if not banks:
			raise ValueError("update_banks requires at least one bank")

		new_map: dict[int, Bank] = {}

		for bank in banks:
			if bank.program in new_map:
				raise ValueError(
					f"Duplicate program number {bank.program} in bank update"
				)
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
