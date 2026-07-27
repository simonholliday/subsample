"""Tests for subsample.devices — matching a configured device name to a real one.

The problem being solved: an ALSA audio device is named
``SC-U: USB Audio (hw:2,0)`` where the CARD index is assigned by probe order, and
a MIDI port is ``… Subsample Virtual MIDI 129:0`` where the sequencer CLIENT id is
handed out in registration order.  Both move; the second number in each (the
subdevice, or the port within a client) does not.
"""

import typing

import pytest

import subsample.devices


# Real names from the author's machine, plus the shapes that break naive matching.
AUDIO_NAMES = [
	"SC-U: USB Audio (hw:2,0)",
	"U6MIDI Pro: USB Audio (hw:0,0)",
	"HDA Intel PCH: ALC256 Analog (hw:1,0)",
]

MIDI_NAMES = [
	"RtMidiIn Client:Subsample Virtual MIDI 129:0",
	"RtMidiIn Client:U6MIDI Pro Port 1 16:0",
	"RtMidiIn Client:U6MIDI Pro Port 2 16:1",
	"RtMidiIn Client:U6MIDI Pro Port 3 16:2",
	"RtMidiIn Client:Midi Through Port-0 14:0",
]


class TestMatchDeviceNames:

	"""The matching rule itself."""

	@staticmethod
	def _matched (pattern: str, names: typing.Sequence[str]) -> list[str]:
		return [names[i] for i in subsample.devices.match_device_names(pattern, names)]

	def test_plain_substring_still_works (self) -> None:

		"""A pattern with no wildcards must behave exactly as it did before they
		existed — every config in the wild is written this way."""

		assert self._matched("SC-U", AUDIO_NAMES) == ["SC-U: USB Audio (hw:2,0)"]

	def test_matching_is_case_insensitive (self) -> None:
		assert self._matched("sc-u: usb audio", AUDIO_NAMES) == ["SC-U: USB Audio (hw:2,0)"]

	def test_full_name_pinned_exactly (self) -> None:

		"""Pasting the whole name still pins one device — the escape hatch for
		two identical units on different cards."""

		assert self._matched("SC-U: USB Audio (hw:2,0)", AUDIO_NAMES) == [
			"SC-U: USB Audio (hw:2,0)",
		]

	def test_wildcard_covers_a_renumbered_card (self) -> None:

		"""The whole point.  The same pattern must find the device whichever card
		index it lands on — SC-U was card 0 before an unrelated interface was
		plugged in, and card 2 after."""

		pattern = "SC-U: USB Audio (hw:*,0)"

		assert self._matched(pattern, ["SC-U: USB Audio (hw:0,0)"]) != []
		assert self._matched(pattern, ["SC-U: USB Audio (hw:2,0)"]) != []
		assert self._matched(pattern, ["SC-U: USB Audio (hw:11,0)"]) != []

	def test_question_mark_breaks_at_two_digits (self) -> None:

		"""Why the docs recommend `*` and not `?`: a single-character wildcard
		works until the machine has ten cards, then silently stops matching.  The
		failure has no symptom beyond the device "disappearing"."""

		pattern = "SC-U: USB Audio (hw:?,0)"

		assert self._matched(pattern, ["SC-U: USB Audio (hw:2,0)"]) != []
		assert self._matched(pattern, ["SC-U: USB Audio (hw:11,0)"]) == []

	def test_wildcarding_the_stable_index_too_is_ambiguous (self) -> None:

		"""Keeping the trailing index is the advice that matters.  A multi-port
		interface reports one name per port, so dropping it turns one device into
		three and forces a prompt at every launch."""

		assert len(self._matched("*U6MIDI Pro*", MIDI_NAMES)) == 3
		assert self._matched("*U6MIDI Pro *:0", MIDI_NAMES) == [
			"RtMidiIn Client:U6MIDI Pro Port 1 16:0",
		]

	def test_wildcard_covers_a_renumbered_midi_client (self) -> None:

		"""The MIDI half of the same problem: the sequencer client id is assigned
		in registration order, so it differs run to run."""

		pattern = "*Subsample Virtual MIDI *:0"

		for client in (128, 129, 145):
			names = [f"RtMidiIn Client:Subsample Virtual MIDI {client}:0"]
			assert self._matched(pattern, names) == names

	def test_no_match_returns_empty (self) -> None:
		assert subsample.devices.match_device_names("Scarlett", AUDIO_NAMES) == []

	def test_returns_every_match_in_order (self) -> None:

		"""Callers need all of them to detect ambiguity; returning the first is
		what resolved ambiguity by enumeration order before."""

		assert subsample.devices.match_device_names("USB Audio", AUDIO_NAMES) == [0, 1]

	def test_bracket_is_literal_not_a_character_class (self) -> None:

		"""fnmatch would read "[Pro]" as "any of P, r, o" and match almost
		everything — a false positive with no visible cause.  Only `*` and `?`
		are wildcards, so a pasted name containing a bracket behaves."""

		names = ["card [Pro] analog", "has r and o in it"]

		assert self._matched("[Pro]", names) == ["card [Pro] analog"]

	def test_already_wildcarded_pattern_is_not_double_wrapped (self) -> None:

		"""The implicit `*` at each end must not disturb a pattern that already
		has them."""

		assert self._matched("*SC-U*", AUDIO_NAMES) == ["SC-U: USB Audio (hw:2,0)"]

	def test_empty_pattern_matches_everything (self) -> None:

		"""Degenerate but well defined: an empty pattern is `**`.  Callers treat
		"several matches" as ambiguity, so this surfaces as a prompt rather than
		an arbitrary pick."""

		assert len(subsample.devices.match_device_names("", AUDIO_NAMES)) == len(AUDIO_NAMES)

	def test_empty_device_list (self) -> None:
		assert subsample.devices.match_device_names("anything", []) == []


class TestHasWildcards:

	@pytest.mark.parametrize("pattern", ["SC-U*", "hw:?,0", "*"])
	def test_detected (self, pattern: str) -> None:
		assert subsample.devices.has_wildcards(pattern) is True

	@pytest.mark.parametrize("pattern", ["SC-U", "SC-U: USB Audio (hw:2,0)", ""])
	def test_absent (self, pattern: str) -> None:
		assert subsample.devices.has_wildcards(pattern) is False


class TestCanPrompt:

	"""Whether there is a terminal to ask a question on."""

	def test_false_when_stdin_is_not_a_tty (self, monkeypatch: pytest.MonkeyPatch) -> None:

		"""Under a service manager, a remote session or CI, a menu is an
		indefinite hang with nothing in the log to explain it."""

		monkeypatch.setattr("sys.stdin", _FakeStdin(is_tty=False))

		assert subsample.devices.can_prompt() is False

	def test_true_when_stdin_is_a_tty (self, monkeypatch: pytest.MonkeyPatch) -> None:
		monkeypatch.setattr("sys.stdin", _FakeStdin(is_tty=True))

		assert subsample.devices.can_prompt() is True

	def test_closed_stdin_is_not_a_terminal (self, monkeypatch: pytest.MonkeyPatch) -> None:

		"""isatty() on a closed stream raises ValueError; that is still "no
		terminal", not a crash during device selection."""

		monkeypatch.setattr("sys.stdin", _RaisingStdin())

		assert subsample.devices.can_prompt() is False

	def test_absent_stdin_is_not_a_terminal (self, monkeypatch: pytest.MonkeyPatch) -> None:

		"""sys.stdin is None under pythonw and some embedded hosts."""

		monkeypatch.setattr("sys.stdin", None)

		assert subsample.devices.can_prompt() is False


class TestMessages:

	def test_ambiguous_message_names_every_candidate (self) -> None:
		message = subsample.devices.ambiguous_pattern_message(
			"*U6MIDI Pro*", MIDI_NAMES[1:4], "MIDI input",
		)

		assert "3 MIDI input devices" in message
		for name in MIDI_NAMES[1:4]:
			assert name in message

	def test_advice_differs_by_whether_the_pattern_already_floats (self) -> None:

		"""A pattern that already wildcards its volatile field needs different
		advice from one that does not — telling someone to "add a wildcard" when
		they have one is the sort of hint that wastes an afternoon."""

		with_wildcard = subsample.devices.ambiguous_pattern_message(
			"*U6MIDI*", ["a", "b"], "MIDI input",
		)
		without = subsample.devices.ambiguous_pattern_message(
			"U6MIDI", ["a", "b"], "MIDI input",
		)

		assert "trailing port or subdevice index" in with_wildcard
		assert "Narrow it" in without

	def test_format_device_list_handles_empty (self) -> None:
		assert subsample.devices.format_device_list([]) == "  (none)"


class _FakeStdin:

	"""Minimal stand-in for sys.stdin with a controllable isatty()."""

	def __init__ (self, is_tty: bool) -> None:
		self._is_tty = is_tty

	def isatty (self) -> bool:
		return self._is_tty


class _RaisingStdin:

	"""sys.stdin after close(): isatty() raises ValueError."""

	def isatty (self) -> bool:
		raise ValueError("I/O operation on closed file")
