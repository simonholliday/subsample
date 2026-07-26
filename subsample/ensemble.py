"""Ensemble — several sample sets playing at once, each on its own channel.

A **sample set** is a directory (plus subdirectories) with one or more MIDI
mapper files at its top level.  Because a map resolves its ``directory:``
predicates relative to itself and names its references rather than pointing at
them, a set is self-contained: it can live anywhere — including a drive shared
between machines — and be used by several projects at once.

An **ensemble** binds sets to channels.  It is not a new kind of file: any map
may declare a ``maps:`` block, so an ensemble is simply a map that happens to
include others, and it may carry its own ``assignments:`` alongside.

	maps:
	  - map: "//server/samples/Home Kit 2026-07/midi-map.yaml"   # its own channel
	  - { channel: 11, map: "//server/samples/vocals/live.yaml" }

Contrast with ``programs:`` (see bank.py), which also loads several maps but as
a SWITCH — one active at a time, selected by Program Change.  An ensemble has
every set live simultaneously, separated by channel.

One level only: a map included via ``maps:`` may not declare ``maps:`` of its
own.  That mirrors the existing rule for ``map:`` presets ("nested presets are
not allowed") and removes any need for cycle detection.

This module owns only parsing.  Loading and merging live in
``player.load_ensemble``, which is where NoteMap is built — the same split
bank.py uses, and it keeps this module free of a player import.
"""

import dataclasses
import logging
import pathlib
import typing

import subsample.definitions


_log = logging.getLogger(__name__)


# Every key one `maps:` entry may carry.  A typo (`chanel:`, `path:`) would
# otherwise be silently ignored and the set bound to the wrong channel — or to
# no channel at all — so this whitelist fails it loudly, matching the guards on
# `programs:` entries and on assignments.
_VALID_INCLUDE_KEYS: typing.Final[frozenset[str]] = frozenset({"channel", "map"})


@dataclasses.dataclass(frozen=True)
class MapInclude:

	"""One sample set bound into an ensemble.

	Fields:
		map_path: Absolute path to the included map file, resolved relative to
		          the ensemble's own directory — the same rule every other
		          path inside a map follows, so an ensemble and the sets it
		          names travel together.
		channel:  User-facing MIDI channel (1-16) the set plays on, or None to
		          use whatever the included map declares for itself.  When set it
		          replaces the included map's top-level ``channel:``; an
		          assignment naming its own channel still wins (load_midi_map
		          warns when that happens, since the binding then moves nothing).
	"""

	map_path: str
	channel:  typing.Optional[int] = None


def parse_map_includes (
	raw:              typing.Any,
	ensemble_dir:  pathlib.Path,
	definitions:      typing.Optional[subsample.definitions.Definitions] = None,
) -> list[MapInclude]:

	"""Parse the ``maps:`` key of a MIDI map into MapInclude objects.

	Each entry is either a bare path string (the set keeps its own channel) or a
	mapping with ``map`` and an optional ``channel``.

	Args:
		raw:             Value of the ``maps:`` key from the parsed YAML.
		ensemble_dir: Directory of the ensemble file; include paths resolve
		                 against it.
		definitions:     The ensemble's mounted definitions, so ``channel:``
		                 may be a name from the file's ``channels:`` section
		                 (``channel: my.kit``).

	Returns:
		Ordered list of MapInclude.  Empty when raw is None or an empty list.

	Raises:
		ValueError: If any entry is malformed, or two entries bind the same
		            channel, or the same map is included twice.
	"""

	if raw is None:
		return []

	if not isinstance(raw, list):
		raise ValueError(
			f"MIDI map 'maps' must be a list of sample sets to include, got "
			f"{type(raw).__name__}"
		)

	includes: list[MapInclude] = []
	seen_channels: dict[int, str] = {}
	seen_paths: set[str] = set()

	for index, entry in enumerate(raw):
		map_raw, channel_raw = _split_entry(entry, index)

		if not isinstance(map_raw, str) or not map_raw.strip():
			raise ValueError(
				f"MIDI map maps[{index}]: 'map' must be a non-empty path to a "
				f"mapper file (got {map_raw!r})"
			)

		channel: typing.Optional[int] = None

		if channel_raw is not None:
			try:
				channel = subsample.definitions.resolve_scalar(
					definitions, "channels", channel_raw, f"maps[{index}] 'channel'",
				)
			except (TypeError, ValueError) as exc:
				raise ValueError(
					f"MIDI map maps[{index}]: invalid 'channel' value "
					f"{channel_raw!r} — {exc}"
				) from exc

			if not (1 <= channel <= 16):
				raise ValueError(
					f"MIDI map maps[{index}]: channel must be 1-16 "
					f"(got {channel_raw!r})"
				)

			# Two sets on one channel would merge into a single note map and
			# either collide note-for-note or, worse, silently interleave as
			# velocity layers.  Reject here, where both offenders can be named.
			if channel in seen_channels:
				raise ValueError(
					f"MIDI map maps[{index}]: channel {channel} is already bound "
					f"to {seen_channels[channel]!r} — each included set needs its "
					f"own channel"
				)

			seen_channels[channel] = map_raw

		resolved = str((ensemble_dir / map_raw).resolve())

		# Including the same file twice is always a mistake: with the same
		# binding it is a duplicate, and with different bindings it would need
		# two independent copies of one set's samples.
		if resolved in seen_paths:
			raise ValueError(
				f"MIDI map maps[{index}]: {map_raw!r} is included more than once"
			)

		seen_paths.add(resolved)
		includes.append(MapInclude(map_path=resolved, channel=channel))

	_log.debug("Ensemble declares %d included map(s)", len(includes))

	return includes


def _split_entry (
	entry: typing.Any,
	index: int,
) -> tuple[typing.Any, typing.Any]:

	"""Return (map, channel) from one `maps:` entry, in either accepted form.

	A bare string is the shorthand for "include this set on whatever channel it
	declares for itself", which is the natural spelling when each set already
	knows where it belongs.
	"""

	if isinstance(entry, str):
		return (entry, None)

	if not isinstance(entry, dict):
		raise ValueError(
			f"MIDI map maps[{index}]: expected a path or a mapping with 'map' "
			f"and optional 'channel', got {type(entry).__name__}"
		)

	unknown = set(entry) - _VALID_INCLUDE_KEYS
	if unknown:
		raise ValueError(
			f"MIDI map maps[{index}]: unknown key(s) {sorted(unknown)} — valid "
			f"keys: {sorted(_VALID_INCLUDE_KEYS)}"
		)

	if "map" not in entry:
		raise ValueError(f"MIDI map maps[{index}]: missing 'map'")

	return (entry["map"], entry.get("channel"))
