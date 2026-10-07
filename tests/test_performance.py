"""What on this machine can cost the player its timing (#4659): power saving, real-time priority, and the memory and disk budgets."""

import os
import pathlib
import resource
import shutil
import sys
import threading
import types
import typing

import pytest

import subsample.mounts
import subsample.performance


# ---------------------------------------------------------------------------
# Power saving
# ---------------------------------------------------------------------------

def _cpufreq (
	tmp_path:   pathlib.Path,
	driver:     str,
	governor:   str,
	preference: typing.Optional[str] = None,
) -> pathlib.Path:

	"""A cpufreq folder as sysfs shows one, holding the given driver, governor and energy preference."""

	folder = tmp_path / "cpufreq"
	folder.mkdir()
	(folder / "scaling_driver").write_text(driver + "\n")
	(folder / "scaling_governor").write_text(governor + "\n")

	if preference is not None:
		(folder / "energy_performance_preference").write_text(preference + "\n")

	return folder


class TestACpuSetToSavePower:

	"""On Linux by the energy preference where the driver has one, and by the governor where it does not; on macOS by Low Power Mode."""

	@pytest.mark.parametrize("driver", ["intel_pstate", "amd-pstate-epp"])
	@pytest.mark.parametrize("preference", ["power", "balance_power"])
	def test_an_energy_preference_that_saves_power (self, tmp_path: pathlib.Path, driver: str, preference: str) -> None:

		warning = subsample.performance._linux_power_saving(_cpufreq(tmp_path, driver, "powersave", preference))

		assert warning == (
			f"The CPU is set to save power (energy preference '{preference}'), which can make the "
			"audio drop out at small buffer sizes.  Switch the power mode to Balanced or "
			"Performance before playing."
		)

	@pytest.mark.parametrize("preference", ["balance_performance", "performance", "default"])
	def test_a_powersave_governor_under_intel_pstate_is_ordinary (self, tmp_path: pathlib.Path, preference: str) -> None:

		"""Ubuntu's balanced default on an Intel machine: the governor's name says nothing."""

		assert subsample.performance._linux_power_saving(_cpufreq(tmp_path, "intel_pstate", "powersave", preference)) is None

	def test_the_performance_governor_overrides_the_preference (self, tmp_path: pathlib.Path) -> None:

		assert subsample.performance._linux_power_saving(_cpufreq(tmp_path, "intel_pstate", "performance", "power")) is None

	@pytest.mark.parametrize("driver", ["acpi-cpufreq", "intel_cpufreq", "cppc_cpufreq"])
	def test_a_powersave_governor_elsewhere_holds_the_lowest_speed (self, tmp_path: pathlib.Path, driver: str) -> None:

		warning = subsample.performance._linux_power_saving(_cpufreq(tmp_path, driver, "powersave"))

		assert warning == (
			"The CPU's frequency governor is 'powersave', which holds it at its lowest speed and "
			"can make the audio drop out at small buffer sizes.  Choose another governor, such as "
			"'schedutil' or 'performance', before playing."
		)

	@pytest.mark.parametrize("governor", ["schedutil", "ondemand", "performance"])
	def test_any_other_governor_is_fine (self, tmp_path: pathlib.Path, governor: str) -> None:

		assert subsample.performance._linux_power_saving(_cpufreq(tmp_path, "acpi-cpufreq", governor)) is None

	def test_a_machine_without_cpufreq_says_nothing (self, tmp_path: pathlib.Path) -> None:

		"""A virtual machine, as most CI runners are, often has none."""

		assert subsample.performance._linux_power_saving(tmp_path / "missing") is None

	def test_low_power_mode_on_a_mac (self) -> None:

		pmset = (
			"System-wide power settings:\n"
			"Currently in use:\n"
			" standby              1\n"
			" sleep                1 (sleep prevented by coreaudiod)\n"
			" lowpowermode         1\n"
			" displaysleep         10\n"
		)

		assert subsample.performance._macos_power_saving(pmset) == (
			"Low Power Mode is on, which slows the CPU and can make the audio drop out at small "
			"buffer sizes.  Turn it off in System Settings, under Battery, before playing."
		)

	@pytest.mark.parametrize("pmset", [" lowpowermode         0\n", " sleep                1\n", "", None])
	def test_a_mac_not_in_low_power_mode (self, pmset: typing.Optional[str]) -> None:

		assert subsample.performance._macos_power_saving(pmset) is None

	@pytest.mark.skipif(sys.platform != "darwin", reason="pmset is macOS's")
	def test_pmset_can_be_read_on_a_mac (self) -> None:

		"""What the CI runner's own pmset prints, read as the player reads it."""

		pmset = subsample.performance._pmset()

		assert pmset is not None
		assert "Currently in use" in pmset or "power settings" in pmset


# ---------------------------------------------------------------------------
# Real-time priority
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "linux", reason="real-time priority is asked for only on Linux")
class TestAThreadAsksForRealTimePriority:

	def _ask (self, monkeypatch: pytest.MonkeyPatch, limit: int, refuse: bool = False) -> tuple[typing.Optional[int], list[int]]:

		"""Ask for 70 under a real-time limit of ``limit``; what was granted, and the priorities asked for."""

		asked: list[int] = []

		def _set (pid: int, policy: int, param: typing.Any) -> None:

			# SCHED_RESET_ON_FORK: a thread this one starts runs at ordinary
			# priority, not at its real-time one (#4702).  The class skips
			# elsewhere; the check tells the type checker so too.
			if sys.platform == "linux":
				assert (pid, policy) == (0, os.SCHED_FIFO | os.SCHED_RESET_ON_FORK)

			asked.append(param.sched_priority)

			if refuse:
				raise PermissionError(1, "Operation not permitted")

		monkeypatch.setattr(resource, "getrlimit", lambda _which: (limit, limit))
		monkeypatch.setattr(os, "sched_setscheduler", _set)

		return subsample.performance.promote_this_thread(70), asked

	def test_granted_at_the_priority_it_asks_for (self, monkeypatch: pytest.MonkeyPatch) -> None:

		assert self._ask(monkeypatch, 95) == (70, [70])

	def test_no_higher_than_the_users_limit (self, monkeypatch: pytest.MonkeyPatch) -> None:

		assert self._ask(monkeypatch, 40) == (40, [40])

	def test_with_no_limit_at_all (self, monkeypatch: pytest.MonkeyPatch) -> None:

		assert self._ask(monkeypatch, resource.RLIM_INFINITY) == (70, [70])

	def test_at_a_limit_of_nothing_it_still_asks (self, monkeypatch: pytest.MonkeyPatch) -> None:

		"""Root, or the CAP_SYS_NICE capability, may grant it all the same."""

		assert self._ask(monkeypatch, 0) == (70, [70])

	def test_refused (self, monkeypatch: pytest.MonkeyPatch) -> None:

		assert self._ask(monkeypatch, 0, refuse=True) == (None, [70])

	def test_only_the_thread_that_asks_changes (self) -> None:

		"""For real, on a thread of its own: granted or refused, the rest of the process is untouched."""

		# The class skips elsewhere; the check tells the type checker so too.
		if sys.platform == "linux":
			before = os.sched_getscheduler(0)
			thread = threading.Thread(target=subsample.performance.promote_this_thread, args=(70,))
			thread.start()
			thread.join()

			assert os.sched_getscheduler(0) == before


def test_real_time_priority_is_not_asked_for_off_linux (monkeypatch: pytest.MonkeyPatch) -> None:

	"""CoreAudio and CoreMIDI already run their threads at it on macOS."""

	monkeypatch.setattr(sys, "platform", "darwin")

	assert subsample.performance.promote_this_thread(70) is None


# ---------------------------------------------------------------------------
# Memory and disk budgets
# ---------------------------------------------------------------------------

_GB = 1024 * 1024 * 1024


class TestTheMemoryBudgets:

	@pytest.fixture(autouse=True)
	def _thirty_two_gigabytes (self, monkeypatch: pytest.MonkeyPatch) -> None:
		monkeypatch.setattr(subsample.performance, "_physical_memory", lambda: 32 * _GB)

	def test_within_three_quarters_of_the_memory_says_nothing (self) -> None:

		"""16 GB of renders and 1 GB of samples on a 32 GB machine, as Simon's own config has."""

		assert subsample.performance.budget_warnings(1024, 16384, 1, None, 0) == []

	def test_past_three_quarters_for_each_of_several_programs (self) -> None:

		warnings = subsample.performance.budget_warnings(1024, 8192, 3, None, 0)

		assert warnings == [
			"Memory budgets come to 27.0 GB, 84% of this machine's 32.0 GB, counting "
			"library.max_memory_mb and transform.max_memory_mb once for each of 3 programs.  "
			"As the caches fill, the machine may run short and swap, which interrupts the audio.  "
			"Lower those budgets."
		]

	def test_past_three_quarters_with_one_library (self) -> None:

		warnings = subsample.performance.budget_warnings(4096, 24576, 1, None, 0)

		assert warnings == [
			"Memory budgets come to 28.0 GB, 88% of this machine's 32.0 GB, counting "
			"library.max_memory_mb and transform.max_memory_mb.  As the caches fill, the machine "
			"may run short and swap, which interrupts the audio.  Lower those budgets."
		]

	def test_a_machine_whose_memory_cannot_be_told (self, monkeypatch: pytest.MonkeyPatch) -> None:

		monkeypatch.setattr(subsample.performance, "_physical_memory", lambda: None)

		assert subsample.performance.budget_warnings(1_000_000, 1_000_000, 1, None, 0) == []


def test_this_machines_memory_can_be_told () -> None:

	"""On Linux and macOS alike, which CI checks."""

	memory = subsample.performance._physical_memory()

	assert memory is not None
	assert memory > 256 * 1024 * 1024


class TestTheVariantCache:

	@pytest.fixture(autouse=True)
	def _plenty_of_memory (self, monkeypatch: pytest.MonkeyPatch) -> None:
		monkeypatch.setattr(subsample.performance, "_physical_memory", lambda: 1024 * _GB)

	def _free (self, monkeypatch: pytest.MonkeyPatch, gigabytes: float) -> None:
		monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(total=0, used=0, free=int(gigabytes * _GB)))

	def test_held_in_memory (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:

		self._free(monkeypatch, 100.0)
		monkeypatch.setattr(subsample.mounts, "filesystem", lambda _path: "tmpfs")

		cache = str(tmp_path / "variant-cache")

		assert subsample.performance.budget_warnings(100, 50, 1, cache, 32768) == [
			f"transform.variant_cache_dir ({cache}) is held in memory, not on a disk, so the up to "
			"32.0 GB it keeps is taken from memory too.  Put it on a disk, as the default "
			"samples/variant-cache is."
		]

	def test_a_disk_without_room_for_it (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:

		self._free(monkeypatch, 12.0)

		cache = str(tmp_path / "variant-cache")

		assert subsample.performance.budget_warnings(100, 50, 1, cache, 32768) == [
			f"transform.variant_cache_dir ({cache}) has room for 12.0 GB more, short of "
			"transform.max_disk_mb's 32.0 GB.  Once that disk is full, renders are no longer kept "
			"for the next session, and other programs using it can fail.  Lower max_disk_mb, or "
			"free some space."
		]

	def test_what_the_cache_holds_already_counts_as_room (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:

		"""Its own files are room it can reuse: 1 MB free beside 3 MB of renders fits a 4 MB cache."""

		self._free(monkeypatch, 1 / 1024)

		folder = tmp_path / "variant-cache"
		folder.mkdir()
		(folder / "renders.bin").write_bytes(b"\0" * (3 * 1024 * 1024))

		assert subsample.performance.budget_warnings(100, 50, 1, str(folder), 4) == []
		assert subsample.performance.budget_warnings(100, 50, 1, str(folder), 5) != []

	def test_on_an_ordinary_disk_with_room (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:

		self._free(monkeypatch, 100.0)

		assert subsample.performance.budget_warnings(100, 50, 1, str(tmp_path / "variant-cache"), 500) == []

	@pytest.mark.parametrize(("cache", "max_disk_mb"), [(None, 500.0), ("", 500.0), ("anywhere", 0.0)])
	def test_no_cache_is_not_checked (self, monkeypatch: pytest.MonkeyPatch, cache: typing.Optional[str], max_disk_mb: float) -> None:

		self._free(monkeypatch, 0.0)
		monkeypatch.setattr(subsample.mounts, "filesystem", lambda _path: "tmpfs")

		assert subsample.performance.budget_warnings(100, 50, 1, cache, max_disk_mb) == []
