"""Per-project definitions file — give your own names to your sounds.

A definitions file is a small YAML file, owned by the music project, that maps
names to MIDI numbers so a map can read ``notes: my.dawn_chorus_pheasant``
instead of ``notes: 60`` — and the same file can name sounds and controls in
other tools (e.g. a sequencer) without either tool depending on the other.

The file has flat name→number sections; subsample consumes exactly four:

	notes:    { ride_edge_soft: 53, dawn_chorus_pheasant: 60 }   # 0-127
	cc:       { sampler_release: 21 }                            # 0-127
	channels: { kit: 10, birds: 3 }                              # 1-16
	programs: { brushes: 1 }                                     # 0-127

Any other top-level section is silently ignored — that is the point of a
shared file: another tool may own e.g. ``nrpn:`` names that mean nothing here.
Validation inside the consumed sections is strict.

A MIDI map mounts a file under a prefix of the user's choosing::

	definitions: { my: project.yaml }    # path relative to the map file

after which ``my.ride_edge_soft`` works wherever the map accepts a note, and
``my.sampler_release`` / ``my.kit`` / ``my.brushes`` wherever it accepts a CC,
channel, or program number.  The field's context selects the section.  Names
are lowercase ``a-z0-9_`` in the file and matched case-insensitively at the
point of use, like the built-in ``drum.*`` GM names (whose prefix is
reserved).

Resolution happens entirely at map-parse time — playback never sees a name.
"""

import dataclasses
import pathlib
import re
import typing

import yaml


# The sections subsample reads.  Everything else in the file belongs to other
# tools and is ignored without comment.
CONSUMED_SECTIONS: typing.Final[frozenset[str]] = frozenset({
	"notes", "cc", "channels", "programs",
})

# Inclusive value range per consumed section.  Channels are user-facing 1-16
# (the map's own `channel:` convention); the rest are raw MIDI data ranges.
_SECTION_RANGES: typing.Final[dict[str, tuple[int, int]]] = {
	"notes":    (0, 127),
	"cc":       (0, 127),
	"channels": (1, 16),
	"programs": (0, 127),
}

# Names and mount prefixes: lowercase identifiers, no dots — the dot is the
# prefix separator at the point of use.
_NAME_RE: typing.Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_]*")

# Shape of a symbolic reference at a scalar (cc / channel / program) site.
# Only a string that full-matches this is treated as a name lookup; anything
# else falls through to the existing int() coercion so the error behaviour
# for non-symbolic garbage ("1.5", "kick") is unchanged.
_SYMBOL_RE: typing.Final[re.Pattern[str]] = re.compile(
	r"[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*"
)


@dataclasses.dataclass(frozen=True)
class Definitions:

	"""Loaded name→number tables from the mounted definitions file(s).

	tables: prefix → section → name → value.  Names are stored lowercase;
	lookups lower their input, matching the ``drum.*`` convention.

	paths: resolved absolute paths of the mounted files, in mount order —
	kept for error reporting (and available should the map watcher ever
	want to watch them).
	"""

	tables: dict[str, dict[str, dict[str, int]]]
	paths:  tuple[pathlib.Path, ...] = ()

	def note_namespaces (self) -> dict[str, typing.Mapping[str, int]]:

		"""Per-prefix ``notes:`` tables, for merging over the symbolic-note
		namespaces.

		EVERY mounted prefix appears — with an empty table when its file has
		no ``notes:`` section — so a miss on a mounted prefix produces the
		targeted unknown-symbol error rather than falling through to the
		note-name grammar.
		"""

		return {
			prefix: sections.get("notes", {})
			for prefix, sections in self.tables.items()
		}

	def lookup (self, section: str, prefix: str, name: str, context: str) -> int:

		"""Resolve ``prefix.name`` in ``section``, raising a targeted error.

		Both parts are lowered (use sites are case-insensitive; files store
		lowercase).  Raises ValueError naming the context, with the mounted
		prefixes on an unknown prefix and the valid names — plus a hint when
		the name exists in a sibling section — on an unknown name.
		"""

		prefix_l = prefix.lower()
		name_l = name.lower()

		sections = self.tables.get(prefix_l)

		if sections is None:
			if prefix_l == "drum":
				raise ValueError(
					f"{context}: 'drum.*' names are GM drum NOTE names and "
					f"cannot name a {section} value — use a mounted "
					f"definitions file instead"
				)
			if not self.tables:
				raise ValueError(
					f"{context}: {prefix}.{name!s} looks like a definitions "
					f"name, but this map mounts no 'definitions:' files"
				)
			raise ValueError(
				f"{context}: unknown definitions prefix {prefix!r} in "
				f"'{prefix}.{name}' — mounted prefixes: "
				f"{', '.join(sorted(self.tables))}"
			)

		table = sections.get(section, {})

		if name_l not in table:
			hint = ""
			for other, other_table in sections.items():
				if other != section and name_l in other_table:
					hint = f" — note: {name_l!r} is defined in section {other!r}"
					break

			valid = sorted(table)
			listing = f" (valid: {', '.join(valid[:5])}…)" if valid else (
				f" (the file mounts no {section!r} section)"
			)
			raise ValueError(
				f"{context}: unknown {section!r} name {name_l!r} under "
				f"prefix {prefix_l!r}{listing}{hint}"
			)

		return table[name_l]


def load_definitions (
	raw:               typing.Any,
	base_dir:          pathlib.Path,
	reserved_prefixes: frozenset[str] = frozenset(),
	map_label:         str = "MIDI map",
) -> Definitions:

	"""Parse a map's ``definitions:`` mount and load the referenced files.

	Args:
		raw:               The map's ``definitions:`` value — None (no mount)
		                   or a mapping of prefix → file path.
		base_dir:          Directory relative paths resolve against (the MIDI
		                   map's own directory, like ``reference:`` paths).
		reserved_prefixes: Prefixes the caller already owns (the built-in
		                   symbolic namespaces, e.g. ``drum``) — passed in
		                   rather than hardcoded here so the two never drift.
		map_label:         Human label for error messages.

	Returns:
		A Definitions holding every mounted file's consumed sections.

	Raises:
		ValueError: On a malformed mount, a reserved or invalid prefix, a
		missing or unreadable file, or any invalid entry in a consumed
		section.  Unknown top-level sections in the file are NOT errors.
	"""

	if raw is None:
		return Definitions(tables={}, paths=())

	if not isinstance(raw, dict):
		raise ValueError(
			f"{map_label}: 'definitions' must be a mapping of prefix to "
			f"file path, e.g. definitions: {{ my: project.yaml }} "
			f"(got {type(raw).__name__}: {raw!r})"
		)

	tables: dict[str, dict[str, dict[str, int]]] = {}
	paths: list[pathlib.Path] = []

	for prefix_raw, path_raw in raw.items():
		prefix = str(prefix_raw)

		if not _NAME_RE.fullmatch(prefix):
			raise ValueError(
				f"{map_label}: definitions prefix {prefix!r} must match "
				f"[a-z][a-z0-9_]* (lowercase letters, digits, underscores)"
			)

		if prefix in reserved_prefixes:
			raise ValueError(
				f"{map_label}: definitions prefix {prefix!r} is reserved "
				f"for the built-in GM drum names — choose another prefix"
			)

		if not isinstance(path_raw, str) or not path_raw.strip():
			raise ValueError(
				f"{map_label}: definitions entry {prefix!r} must be a file "
				f"path string (got {path_raw!r})"
			)

		path = (base_dir / pathlib.Path(path_raw)).resolve()

		if not path.is_file():
			raise ValueError(
				f"{map_label}: definitions file for prefix {prefix!r} not "
				f"found: {path_raw!r} (resolved to {path})"
			)

		tables[prefix] = _load_definitions_file(path, prefix)
		paths.append(path)

	return Definitions(tables=tables, paths=tuple(paths))


def _load_definitions_file (
	path:   pathlib.Path,
	prefix: str,
) -> dict[str, dict[str, int]]:

	"""Load one definitions file and validate its consumed sections."""

	try:
		with path.open(encoding="utf-8") as fh:
			raw = yaml.safe_load(fh)
	except (OSError, yaml.YAMLError) as exc:
		raise ValueError(
			f"definitions file {path} (prefix {prefix!r}) could not be "
			f"read: {exc}"
		) from exc

	if raw is None:
		return {}

	if not isinstance(raw, dict):
		raise ValueError(
			f"definitions file {path}: top level must be a mapping of "
			f"sections (notes:, cc:, …), got {type(raw).__name__}"
		)

	sections: dict[str, dict[str, int]] = {}

	for section in sorted(CONSUMED_SECTIONS):
		section_raw = raw.get(section)

		if section_raw is None:
			continue

		if not isinstance(section_raw, dict):
			raise ValueError(
				f"definitions file {path}: section {section!r} must be a "
				f"mapping of name to number "
				f"(got {type(section_raw).__name__})"
			)

		lo, hi = _SECTION_RANGES[section]
		table: dict[str, int] = {}

		for name_raw, value in section_raw.items():
			name = str(name_raw)

			if not _NAME_RE.fullmatch(name):
				raise ValueError(
					f"definitions file {path}: section {section!r}: name "
					f"{name!r} must match [a-z][a-z0-9_]* (lowercase "
					f"letters, digits, underscores — no dots)"
				)

			# bool is an int subclass — reject it first so `x: true` fails
			# loudly instead of quietly becoming 1.
			if isinstance(value, bool) or not isinstance(value, int):
				raise ValueError(
					f"definitions file {path}: section {section!r}: "
					f"{name!r} must be a whole number (got {value!r})"
				)

			if not lo <= value <= hi:
				raise ValueError(
					f"definitions file {path}: section {section!r}: "
					f"{name!r} = {value} is outside [{lo}, {hi}]"
				)

			table[name] = value

		sections[section] = table

	return sections


def resolve_scalar (
	definitions: typing.Optional[Definitions],
	section:     str,
	raw:         typing.Any,
	context:     str,
) -> int:

	"""Resolve one scalar map value that may be a definitions name.

	A string full-matching ``prefix.name`` is looked up in ``section`` of
	the mounted definitions (strictly — unlike note positions there is no
	other grammar to fall through to).  Anything else takes the verbatim
	``int()`` coercion the call site used before, so error behaviour for
	plain numbers and garbage is unchanged.
	"""

	if isinstance(raw, str) and _SYMBOL_RE.fullmatch(raw.strip()):
		prefix, _, name = raw.strip().partition(".")

		if definitions is None:
			raise ValueError(
				f"{context}: {raw!r} looks like a definitions name, but "
				f"this map mounts no 'definitions:' files"
			)

		return definitions.lookup(section, prefix, name, context)

	return int(raw)
