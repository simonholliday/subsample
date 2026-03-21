"""MIDI input listener and sample player for Subsample.

Handles MIDI input device selection (replicating the audio device selection
pattern from audio.py) and the MidiPlayer class which listens for MIDI events
on a background-friendly blocking loop.

Current implementation: skeleton only. Listens for MIDI messages and logs
them at INFO level. Audio playback will be added in a future iteration.
"""

import logging
import threading
import typing

import mido


_log = logging.getLogger(__name__)


def list_midi_input_devices () -> list[str]:

	"""Return the names of all available MIDI input devices.

	Uses mido's default backend (rtmidi). Returns an empty list if no
	MIDI devices are connected or the backend is unavailable.
	"""

	return list(mido.get_input_names())


def find_midi_device_by_name (name: str) -> str:

	"""Find a MIDI input device by a case-insensitive substring of its name.

	Args:
		name: Substring to search for (case-insensitive).

	Returns:
		Full name of the first matching device.

	Raises:
		ValueError: If no device matches, listing all available device names.
	"""

	name_lower = name.lower()

	for device_name in mido.get_input_names():
		if name_lower in str(device_name).lower():
			return str(device_name)

	available = mido.get_input_names()
	available_str = "\n  ".join(available) if available else "(none found)"
	raise ValueError(
		f"No MIDI input device matching {name!r}.\n"
		f"Available devices:\n  {available_str}"
	)


def select_midi_device (devices: list[str]) -> str:

	"""Select a MIDI input device interactively.

	Auto-selects if exactly one device is present. Prints an interactive
	numbered menu when multiple devices are available. Mirrors the behaviour
	of subsample.audio.select_device().

	Args:
		devices: List of MIDI device name strings from list_midi_input_devices().

	Returns:
		Selected device name.

	Raises:
		ValueError: If the devices list is empty.
	"""

	if not devices:
		raise ValueError(
			"No MIDI input devices found. Connect a MIDI device and try again."
		)

	if len(devices) == 1:
		print(f"Using MIDI input: {devices[0]}")
		return devices[0]

	print("Available MIDI input devices:")
	for i, name in enumerate(devices):
		print(f"  [{i}] {name}")

	while True:
		raw = input(f"Select device [0–{len(devices) - 1}]: ").strip()

		try:
			choice = int(raw)
		except ValueError:
			print("  Please enter a number.")
			continue

		if 0 <= choice < len(devices):
			return devices[choice]

		print(f"  Please enter a number between 0 and {len(devices) - 1}.")


class MidiPlayer:

	"""Listens for MIDI messages on a selected input device.

	Designed to run on its own thread. Call run() as the thread target;
	it blocks until shutdown_event is set, then closes the MIDI port and
	returns cleanly.

	This is a skeleton — future iterations will map MIDI notes to instrument
	samples and trigger audio playback.
	"""

	def __init__ (
		self,
		device_name: str,
		shutdown_event: threading.Event,
	) -> None:

		self._device_name = device_name
		self._shutdown_event = shutdown_event

	def run (self) -> None:

		"""Open the MIDI port and listen until shutdown_event is set.

		Uses non-blocking receive() with a short sleep so the shutdown event
		is checked regularly without busy-waiting. Port is always closed in
		the finally block.
		"""

		_log.info("MIDI player opening port: %s", self._device_name)

		with mido.open_input(self._device_name) as port:
			while not self._shutdown_event.is_set():
				msg = port.receive(block=False)

				if msg is not None:
					_log.info("MIDI: %s", msg)
				else:
					# No message available — yield briefly before polling again.
					# 10 ms gives ~100 Hz polling rate: responsive yet not busy.
					self._shutdown_event.wait(timeout=0.01)

		_log.info("MIDI player closed port: %s", self._device_name)
