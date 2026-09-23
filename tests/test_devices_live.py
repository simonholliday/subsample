"""Device matching against the names this machine's own backends report (#331).

Every other device test hands the matcher names somebody wrote down, so a
change in how PortAudio or rtmidi formats a name, which is the very thing the
wildcards exist to absorb, would pass them all and fail on the first real
machine.  These ask the backends instead, and go through the same lookups the
app uses.

They check the two promises the device docs make to a user.  **Pasting a
device's whole name selects that device**, whatever else the machine reports
(#398 was exactly this failing: ``default`` is inside ``sysdefault``).  And
**wildcarding the number that moves still finds the device**, and nothing that
is not the same device under another number.

They run wherever the suite runs: enumerating costs a fraction of a second.  A
backend that cannot start, or reports nothing, skips its tests rather than
failing them, so a machine without audio or MIDI loses only the check.
"""

import re
import typing

import pyaudio
import pytest
import rtmidi

import subsample.audio
import subsample.devices
import subsample.player


# An ALSA audio name ends "(hw:<card>,<device>)": the card is handed out by probe
# order and moves; the device within it does not.
_ALSA_CARD = re.compile(r"\(hw:(\d+),(\d+)\)")

# An ALSA sequencer port name ends "<client>:<port>": the client id is handed out
# in registration order and moves; the port within it does not.
_SEQUENCER_CLIENT = re.compile(r" (\d+):(\d+)$")

_FIND_AUDIO: typing.Final[dict[str, typing.Callable[[pyaudio.PyAudio, str], int]]] = {
	"input":  subsample.audio.find_device_by_name,
	"output": subsample.audio.find_output_device_by_name,
}

_LIST_AUDIO: typing.Final[dict[str, typing.Callable[[pyaudio.PyAudio], list[subsample.audio.DeviceInfo]]]] = {
	"input":  subsample.audio.list_input_devices,
	"output": subsample.audio.list_output_devices,
}


@pytest.fixture(scope="module")
def portaudio () -> typing.Iterator[pyaudio.PyAudio]:

	"""One PortAudio session for the module, or a skip where it cannot start."""

	try:
		audio = pyaudio.PyAudio()
	except OSError as exc:
		pytest.skip(f"PortAudio cannot start here: {exc}")

	yield audio

	audio.terminate()


@pytest.fixture(autouse=True)
def _no_terminal (monkeypatch: pytest.MonkeyPatch) -> None:

	"""Never wait on a prompt: an ambiguous name raises instead, naming the matches."""

	monkeypatch.setattr(subsample.devices, "can_prompt", lambda: False)


def _unique (names: typing.Sequence[str]) -> list[str]:

	"""The names reported once, ignoring case.

	Two devices reported under one name are ambiguous by identity, and the app
	is right to ask which is meant, so the full-name promise cannot hold for
	them.
	"""

	lowered = [name.lower() for name in names]

	return [name for name in names if lowered.count(name.lower()) == 1]


def _wildcarded_elsewhere (
	shape:       re.Pattern[str],
	replacement: str,
	names:       typing.Sequence[str],
) -> list[tuple[str, str, list[str]]]:

	"""Each name with the documented shape, its wildcarded pattern, and what that matches."""

	found: list[tuple[str, str, list[str]]] = []

	for name in names:

		if not shape.search(name):
			continue

		pattern = shape.sub(replacement, name)
		matched = [names[index] for index in subsample.devices.match_device_names(pattern, names)]
		found.append((name, pattern, matched))

	return found


def _midi_input_names () -> list[str]:

	"""Every MIDI input rtmidi reports here, or a skip where it cannot start."""

	try:
		names = subsample.player.list_midi_input_devices()
	except (OSError, rtmidi.RtMidiError) as exc:
		pytest.skip(f"rtmidi cannot start here: {exc}")

	if not names:
		pytest.skip("rtmidi reports no MIDI input here")

	return names


class TestAudioDevicesThisMachineReports:

	@pytest.mark.parametrize("direction", ["input", "output"])
	def test_pasting_a_device_s_whole_name_selects_it (
		self,
		portaudio: pyaudio.PyAudio,
		direction: str,
	) -> None:

		"""The escape hatch the docs teach, on every name this machine reports."""

		devices = _LIST_AUDIO[direction](portaudio)

		if not devices:
			pytest.skip(f"PortAudio reports no audio {direction} here")

		names = [str(device["name"]) for device in devices]
		index_of = {str(device["name"]): int(device["index"]) for device in devices}

		for name in _unique(names):

			try:
				found = _FIND_AUDIO[direction](portaudio, name)
			except ValueError as exc:
				pytest.fail(f"pasting the whole name {name!r} did not select it:\n{exc}")

			assert found == index_of[name], f"{name!r} selected device {found}, not {index_of[name]}"

	@pytest.mark.parametrize("direction", ["input", "output"])
	def test_wildcarding_the_card_still_finds_the_device (
		self,
		portaudio: pyaudio.PyAudio,
		direction: str,
	) -> None:

		"""``(hw:*,0)`` finds the device when its card moves, and only that device."""

		names = [str(device["name"]) for device in _LIST_AUDIO[direction](portaudio)]
		shaped = _wildcarded_elsewhere(_ALSA_CARD, r"(hw:*,\2)", names)

		if not shaped:
			pytest.skip(f"no audio {direction} here has an ALSA card number in its name")

		for name, pattern, matched in shaped:

			assert name in matched, f"{pattern!r} no longer finds {name!r}"

			for other in matched:
				assert _ALSA_CARD.sub(r"(hw:*,\2)", other) == pattern, (
					f"{pattern!r} also matches {other!r}, which is not {name!r} on another card"
				)


class TestMidiPortsThisMachineReports:

	def test_pasting_a_port_s_whole_name_selects_it (self) -> None:

		"""The escape hatch the docs teach, on every MIDI input this machine reports."""

		names = _midi_input_names()

		for name in _unique(names):

			try:
				found = subsample.player.find_midi_device_by_name(name)
			except ValueError as exc:
				pytest.fail(f"pasting the whole name {name!r} did not select it:\n{exc}")

			assert found == name

	def test_wildcarding_the_client_still_finds_the_port (self) -> None:

		"""``*:0`` finds the port when its client id moves, and only that port."""

		names = _midi_input_names()
		shaped = _wildcarded_elsewhere(_SEQUENCER_CLIENT, r" *:\2", names)

		if not shaped:
			pytest.skip("no MIDI input here ends in an ALSA sequencer client:port")

		for name, pattern, matched in shaped:

			assert name in matched, f"{pattern!r} no longer finds {name!r}"

			for other in matched:
				assert _SEQUENCER_CLIENT.sub(r" *:\2", other) == pattern, (
					f"{pattern!r} also matches {other!r}, which is not {name!r} under another client id"
				)
