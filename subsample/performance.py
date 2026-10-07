"""What on this machine can cost the player its timing, and what to change (#4659).

Three things, each read once when the player starts and never from the audio
callback:

  - whether the CPU is set to save power, which can make the audio drop out at
    small buffer sizes (power_saving);
  - whether Linux lets a thread run at real-time priority, so other programs
    cannot delay the audio or a note's handling (promote_this_thread);
  - whether the memory budgets leave the machine room, and whether the
    variant cache is held in memory or short of disk (budget_warnings).

Every message names what to change in the user's own terms.  A check that
cannot read what it needs says nothing rather than guess.
"""

import os
import pathlib
import resource
import shutil
import subprocess
import sys
import typing

import subsample.mounts


_CPUFREQ: typing.Final[pathlib.Path] = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq")

# Under these drivers 'powersave' is the governor's ordinary, self-adjusting
# mode, which Ubuntu runs by default, so the governor's name says nothing.  The
# energy-performance preference says whether it leans to saving power.  Under
# any other driver, the 'powersave' governor holds the CPU at its lowest speed.
_PREFERENCE_DRIVERS: typing.Final[frozenset[str]] = frozenset({"intel_pstate", "amd-pstate-epp"})

_SAVING_PREFERENCES: typing.Final[frozenset[str]] = frozenset({"power", "balance_power"})

_MEMORY_FILESYSTEMS: typing.Final[frozenset[str]] = frozenset({"tmpfs", "ramfs"})

MEMORY_SHARE: typing.Final[float] = 0.75
"""The share of the machine's memory the memory budgets may come to before the
player warns: the rest is left for the system, the desktop and Python itself,
and past it the machine is likely to swap as the caches fill (Simon, #4659)."""

REALTIME_APPLIES: typing.Final[bool] = sys.platform == "linux"
"""Whether a thread asks for real-time priority here.  On macOS, CoreAudio and
CoreMIDI already run their threads at it."""

_MB: typing.Final[int] = 1024 * 1024
_GB: typing.Final[int] = 1024 * _MB


# ---------------------------------------------------------------------------
# Power saving
# ---------------------------------------------------------------------------

def power_saving () -> typing.Optional[str]:

	"""Why this machine's CPU is set to save power, as the warning to give when the player starts, or None."""

	if sys.platform == "linux":
		return _linux_power_saving(_CPUFREQ)

	if sys.platform == "darwin":
		return _macos_power_saving(_pmset())

	return None


def _linux_power_saving (cpufreq: pathlib.Path) -> typing.Optional[str]:

	"""The warning for a Linux CPU set to save power, from its cpufreq folder in sysfs, or None."""

	driver     = _read(cpufreq / "scaling_driver")
	governor   = _read(cpufreq / "scaling_governor")
	preference = _read(cpufreq / "energy_performance_preference")

	if driver in _PREFERENCE_DRIVERS:

		# The 'performance' governor overrides the preference under these drivers.
		if governor == "performance" or preference not in _SAVING_PREFERENCES:
			return None

		return (
			f"The CPU is set to save power (energy preference '{preference}'), which can make "
			"the audio drop out at small buffer sizes.  Switch the power mode to Balanced or "
			"Performance before playing."
		)

	if governor == "powersave":
		return (
			"The CPU's frequency governor is 'powersave', which holds it at its lowest speed "
			"and can make the audio drop out at small buffer sizes.  Choose another governor, "
			"such as 'schedutil' or 'performance', before playing."
		)

	return None


def _macos_power_saving (pmset: typing.Optional[str]) -> typing.Optional[str]:

	"""The warning for a Mac in Low Power Mode, from what ``pmset -g`` printed, or None."""

	if pmset is None:
		return None

	for line in pmset.splitlines():
		fields = line.split()

		if len(fields) >= 2 and fields[0] == "lowpowermode" and fields[1] == "1":
			return (
				"Low Power Mode is on, which slows the CPU and can make the audio drop out at "
				"small buffer sizes.  Turn it off in System Settings, under Battery, before playing."
			)

	return None


def _pmset () -> typing.Optional[str]:

	"""What ``pmset -g`` prints of the Mac's power settings, or None where it cannot be run."""

	try:
		done = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=2.0, check=False)
	except (OSError, subprocess.SubprocessError):
		return None

	return done.stdout if done.returncode == 0 else None


def _read (path: pathlib.Path) -> typing.Optional[str]:

	"""A sysfs file's one value, or None where it is missing or unreadable."""

	try:
		return path.read_text(encoding="ascii", errors="replace").strip()
	except OSError:
		return None


# ---------------------------------------------------------------------------
# Real-time priority
# ---------------------------------------------------------------------------

def promote_this_thread (priority: int) -> typing.Optional[int]:

	"""Ask Linux to run the calling thread at real-time priority, first in, first out.

	Asks for ``priority`` or, where the user's real-time limit (RLIMIT_RTPRIO,
	the ``rtprio`` of limits.conf) is lower but not 0, for that limit.  At a
	limit of 0 it asks anyway, since root or the CAP_SYS_NICE capability may
	still grant it.  Only the calling thread changes, never the process, and
	never a thread it starts later (#4702): those start at ordinary priority.
	A thread inherits the policy of the thread that starts it, so without that
	the MIDI thread's timers ran the player's heaviest Python work, rebuilding
	its assignments, at real-time priority above the recorder's input.

	Returns the priority granted, or None where it is refused, or where it is
	not Linux's to give (REALTIME_APPLIES).
	"""

	if sys.platform != "linux":
		return None

	soft, _hard = resource.getrlimit(resource.RLIMIT_RTPRIO)

	if soft != resource.RLIM_INFINITY and 0 < soft < priority:
		priority = soft

	try:
		os.sched_setscheduler(0, os.SCHED_FIFO | os.SCHED_RESET_ON_FORK, os.sched_param(priority))
	except OSError:
		return None

	return priority


# ---------------------------------------------------------------------------
# Memory and disk budgets
# ---------------------------------------------------------------------------

def budget_warnings (
	library_mb:   float,
	transform_mb: float,
	programs:     int,
	cache_dir:    typing.Optional[str],
	max_disk_mb:  float,
) -> list[str]:

	"""What the memory and variant cache budgets leave this machine short of, as warnings, or none.

	Args:
		library_mb:   library.max_memory_mb.
		transform_mb: transform.max_memory_mb.
		programs:     How many programs the map declares, each with its own
		              library and render cache, or 1.
		cache_dir:    transform.variant_cache_dir as configured, or None.
		max_disk_mb:  transform.max_disk_mb; 0 turns the variant cache off.
	"""

	warnings: list[str] = []

	memory = _physical_memory()
	budget = programs * (library_mb + transform_mb) * _MB

	if memory is not None and budget > MEMORY_SHARE * memory:
		counting = (
			f"counting library.max_memory_mb and transform.max_memory_mb once for each of {programs} programs"
			if programs > 1 else
			"counting library.max_memory_mb and transform.max_memory_mb"
		)
		warnings.append(
			f"Memory budgets come to {budget / _GB:.1f} GB, {round(100 * budget / memory)}% of "
			f"this machine's {memory / _GB:.1f} GB, {counting}.  As the caches fill, the machine "
			"may run short and swap, which interrupts the audio.  Lower those budgets."
		)

	if cache_dir and max_disk_mb > 0:
		warnings.extend(_cache_warnings(cache_dir, max_disk_mb))

	return warnings


def _cache_warnings (cache_dir: str, max_disk_mb: float) -> list[str]:

	"""Whether the variant cache's folder is held in memory, and whether its disk has room for max_disk_mb."""

	warnings: list[str] = []
	folder   = pathlib.Path(cache_dir)
	existing = folder.absolute()

	while not existing.exists() and existing != existing.parent:
		existing = existing.parent

	if subsample.mounts.filesystem(existing) in _MEMORY_FILESYSTEMS:
		warnings.append(
			f"transform.variant_cache_dir ({cache_dir}) is held in memory, not on a disk, so the "
			f"up to {max_disk_mb / 1024:.1f} GB it keeps is taken from memory too.  Put it on a "
			"disk, as the default samples/variant-cache is."
		)

	try:
		free = shutil.disk_usage(existing).free
	except OSError:
		return warnings

	if free + _folder_size(folder) < max_disk_mb * _MB:
		warnings.append(
			f"transform.variant_cache_dir ({cache_dir}) has room for {free / _GB:.1f} GB more, "
			f"short of transform.max_disk_mb's {max_disk_mb / 1024:.1f} GB.  Once that disk is "
			"full, renders are no longer kept for the next session, and other programs using it "
			"can fail.  Lower max_disk_mb, or free some space."
		)

	return warnings


def _folder_size (folder: pathlib.Path) -> int:

	"""The bytes the files directly in folder take, which is all the variant cache keeps; 0 where it does not exist yet."""

	total = 0

	try:
		with os.scandir(folder) as entries:
			for entry in entries:
				try:
					if entry.is_file(follow_symlinks=False):
						total += entry.stat(follow_symlinks=False).st_size
				except OSError:
					continue
	except OSError:
		return 0

	return total


def _physical_memory () -> typing.Optional[int]:

	"""This machine's physical memory in bytes, or None where it cannot be told."""

	try:
		return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
	except (ValueError, OSError):
		return None
