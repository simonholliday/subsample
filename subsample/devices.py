"""Matching a configured device name against the devices actually present.

Hardware device names carry a number that moves.  On Linux an audio device is
``SC-U: USB Audio (hw:2,0)`` — where the CARD index (2) is assigned by probe
order, so unplugging an unrelated interface renumbers it — and a MIDI port is
``RtMidiIn Client:Subsample Virtual MIDI 129:0``, where the ALSA sequencer
CLIENT id (129) is handed out in registration order and changes run to run.

In both cases the shape is the same: ``<stable text> <VOLATILE>:<stable>``.  The
first number moves; the second (the subdevice, or the port within a client) does
not, and is exactly what tells a multi-port interface's ports apart.

So the configured name is a **glob**:

  - ``*`` matches any run of characters, ``?`` matches exactly one.
  - Matching is case-insensitive, with an implicit ``*`` at each end — a pattern
    with no wildcards is therefore a plain substring, which is what this always
    did before wildcards existed.
  - Nothing else is special.  ``[`` is literal (device names may contain one,
    and treating it as a character class would match for baffling reasons).

The convention to teach: **wildcard the number that moves, keep the one that
does not.**

    device: "SC-U: USB Audio (hw:*,0)"
    device: "*U6MIDI Pro *:0"

Keeping the trailing ``,0`` / ``:0`` matters.  A multi-port interface exposes
one name per port, so ``*U6MIDI Pro*`` matches all of them and forces a prompt
every launch, while ``*U6MIDI Pro *:0`` names one port for good.  Prefer ``*``
to ``?`` for the volatile field: ``hw:?,0`` silently stops matching once the
machine has ten cards.

Resolution is deliberately the same for audio and MIDI (and is specified here so
a sister application can implement it identically):

  0 matches   -> ValueError naming what is available; the caller may fall back
                 to prompting over every device.
  1 match     -> use it.
  2+ matches  -> prompt, over THE MATCHES rather than every device — an
                 ambiguous pattern is a shortlist, not a failure.  Without a
                 terminal there is nobody to prompt, so raise instead of hanging.

Before this existed, matching took the FIRST substring match and used it
silently, which resolved ambiguity by enumeration order — the same class of
surprise the volatile numbering causes in the first place.
"""

import fnmatch
import sys
import typing


# The only characters a pattern may use to mean something other than themselves.
# Kept as data so error messages and docs cannot drift from the implementation.
WILDCARD_CHARACTERS: typing.Final[str] = "*?"


def match_device_names (pattern: str, names: typing.Sequence[str]) -> list[int]:

	"""Return the positions in ``names`` of every device the pattern matches.

	Case-insensitive, with an implicit ``*`` at both ends; ``*`` and ``?`` are
	the only wildcards.  Returns every match, in the order given — the caller
	decides what to do with none, one, or several (see the module docstring).

	Args:
		pattern: The configured device name, possibly containing wildcards.
		names:   Device names as reported by the audio or MIDI backend.

	Returns:
		Indices into ``names``.  Empty when nothing matches.
	"""

	# Escape `[` so a pasted name containing one cannot be read as a character
	# class.  fnmatch would otherwise take "[Pro]" as "any of P, r, o" and match
	# almost everything — a false positive with no visible cause.
	escaped = pattern.replace("[", "[[]")
	compiled = f"*{escaped.lower()}*"

	return [
		index for index, name in enumerate(names)
		if fnmatch.fnmatchcase(name.lower(), compiled)
	]


def has_wildcards (pattern: str) -> bool:

	"""True if the pattern uses any wildcard character.

	Used only to phrase error messages — a pattern that already floats its
	volatile field needs different advice from one that pins it.
	"""

	return any(character in pattern for character in WILDCARD_CHARACTERS)


def can_prompt () -> bool:

	"""True when there is a terminal to ask a question on.

	Subsample is run interactively most of the time, but also from service
	managers, remote sessions and CI, where a prompt is an indefinite hang with
	no output explaining it.  Callers check this before offering a menu and
	raise a message the user can actually read in a log instead.
	"""

	try:
		return sys.stdin is not None and sys.stdin.isatty()
	except (AttributeError, ValueError):
		# ValueError: stdin closed.  Neither is a terminal.
		return False


def format_device_list (names: typing.Sequence[str], indent: str = "  ") -> str:

	"""Render device names one per line for an error message."""

	if not names:
		return f"{indent}(none)"

	return "\n".join(f"{indent}{name}" for name in names)


def ambiguous_pattern_message (
	pattern: str,
	matches: typing.Sequence[str],
	kind:    str,
) -> str:

	"""Explain that a pattern matched several devices, and how to narrow it.

	Used when there is no terminal to prompt on.  Names the matches so the log
	shows what to choose between, and suggests the fix that actually works —
	restoring the stable index rather than adding more of the name.
	"""

	advice = (
		"Add the trailing port or subdevice index to pick one "
		"(e.g. '…(hw:*,0)' or '…*:0')."
		if has_wildcards(pattern) else
		"Narrow it, or use a wildcard for the part that changes "
		"(e.g. '…(hw:*,0)')."
	)

	return (
		f"{len(matches)} {kind} devices match {pattern!r} and there is no "
		f"terminal to choose on:\n{format_device_list(matches)}\n{advice}"
	)
