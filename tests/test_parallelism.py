"""Tests for subsample.parallelism — the shared analysis-pool CPU policy."""

import logging
import os
import threading
import typing

import pytest
import threadpoolctl

import subsample.audio
import subsample.cache
import subsample.config
import subsample.parallelism


def _pin_cpus (monkeypatch: pytest.MonkeyPatch, count: int) -> None:

	"""Make usable_cpu_count() report exactly ``count`` CPUs.

	macOS and Windows have no affinity API, so there the test adds one.
	"""

	monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(count)), raising=False)


def test_idle_leaves_headroom (monkeypatch: pytest.MonkeyPatch) -> None:

	"""With nothing playing, analysis uses most of the cores but not every thread."""

	_pin_cpus(monkeypatch, 16)

	# 75% of the cores — fast, but the OS keeps some headroom.
	assert subsample.parallelism.analysis_worker_count(player_active=False) == 12


def test_live_reserves_headroom (monkeypatch: pytest.MonkeyPatch) -> None:

	"""While the player is live, analysis takes only a small share of the cores."""

	_pin_cpus(monkeypatch, 16)

	# One core in four; the rest stay free for the real-time audio thread.
	assert subsample.parallelism.analysis_worker_count(player_active=True) == 4


def test_live_yields_at_least_one_worker (monkeypatch: pytest.MonkeyPatch) -> None:

	"""On a low-core machine the live share still leaves at least one worker."""

	_pin_cpus(monkeypatch, 2)

	assert subsample.parallelism.analysis_worker_count(player_active=True) == 1


def test_idle_never_takes_every_core (monkeypatch: pytest.MonkeyPatch) -> None:

	"""The idle share always leaves a core free above a single-core machine.

	round() used to give 2 of 2 on a dual-core box — every core, defeating the
	headroom the fraction exists to provide — and banker's rounding made 6 cores
	yield 4 rather than 5.
	"""

	for cpus, expected in ((1, 1), (2, 1), (3, 2), (4, 3), (6, 4), (8, 6), (16, 12)):
		_pin_cpus(monkeypatch, cpus)
		assert subsample.parallelism.analysis_worker_count(player_active=False) == expected


def test_worker_count_respects_cpu_affinity (monkeypatch: pytest.MonkeyPatch) -> None:

	"""The pool is sized from the CPUs this process may USE, not the machine's.

	os.cpu_count() reports the host, so under a container CPU quota, a batch
	scheduler, or `taskset -c 0,1` it overcounted badly — 16 workers for a
	2-CPU allowance on this machine.
	"""

	monkeypatch.setattr(os, "cpu_count", lambda: 64)
	_pin_cpus(monkeypatch, 2)

	assert subsample.parallelism.usable_cpu_count() == 2
	assert subsample.parallelism.analysis_worker_count(player_active=False) == 1


def test_unknown_cpu_count_is_safe (monkeypatch: pytest.MonkeyPatch) -> None:

	"""Without an affinity API the machine count is used, and None still yields >= 1."""

	monkeypatch.delattr(os, "sched_getaffinity", raising=False)
	monkeypatch.setattr(os, "cpu_count", lambda: None)

	assert subsample.parallelism.analysis_worker_count(player_active=False) == 1
	assert subsample.parallelism.analysis_worker_count(player_active=True) == 1


def test_cap_blas_threads_pins_to_one () -> None:

	"""Capping pins every loaded math backend to a single thread, repeatably."""

	# numpy must be imported for a BLAS backend to exist at all: without it
	# threadpool_info() is empty and `all([])` is vacuously True, so this test
	# asserted nothing when the file ran on its own.
	import numpy

	# numpy for Apple silicon is built on Apple's Accelerate, which threadpoolctl
	# can neither see nor limit, so with it there is nothing to pin.
	if numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"] == "accelerate":
		pytest.skip("this numpy uses Apple's Accelerate, which threadpoolctl cannot limit")

	subsample.parallelism.cap_blas_threads()
	subsample.parallelism.cap_blas_threads()  # idempotent — must not raise

	info = threadpoolctl.threadpool_info()

	assert info, "no BLAS backend loaded — the assertion below would be vacuous"
	assert all(pool["num_threads"] == 1 for pool in info)


_SETTINGS_UNDER_TEST = (
	subsample.config.AnalysisConfig(tempo_min=72.0, tempo_max=144.0),
	-1.5,
)
"""Non-default analysis settings, to tell a worker that was handed them from one on the defaults."""


@pytest.fixture
def _parent_settings () -> typing.Iterator[None]:

	"""Set the parent's process-wide analysis settings to _SETTINGS_UNDER_TEST, then restore them."""

	previous = (subsample.cache.analysis_config(), subsample.audio.float_import_ceiling())

	subsample.cache.set_analysis_config(_SETTINGS_UNDER_TEST[0])
	subsample.audio.set_float_import_ceiling(_SETTINGS_UNDER_TEST[1])

	yield

	subsample.cache.set_analysis_config(previous[0])
	subsample.audio.set_float_import_ceiling(previous[1])


def test_init_analysis_worker_sets_what_the_parent_set () -> None:

	"""The process-pool initializer installs the settings it is handed."""

	previous = (subsample.cache.analysis_config(), subsample.audio.float_import_ceiling())

	try:
		subsample.parallelism.init_analysis_worker(*_SETTINGS_UNDER_TEST)

		assert subsample.cache.analysis_config() == _SETTINGS_UNDER_TEST[0]
		assert subsample.audio.float_import_ceiling() == _SETTINGS_UNDER_TEST[1]

	finally:
		subsample.cache.set_analysis_config(previous[0])
		subsample.audio.set_float_import_ceiling(previous[1])


def _settings_of (value: int) -> tuple[subsample.config.AnalysisConfig, typing.Optional[float], int]:

	"""Module-level (picklable) worker reporting the settings it analyses with, and its PID."""

	return subsample.cache.analysis_config(), subsample.audio.float_import_ceiling(), os.getpid()


@pytest.mark.usefixtures("_parent_settings")
@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_a_worker_started_fresh_analyses_with_the_parents_settings (
	monkeypatch: pytest.MonkeyPatch, start_method: str,
) -> None:

	"""#390: a worker that does not inherit the parent's memory is handed its settings.

	The library scan reads the analysis config and the float import ceiling
	from module state.  A forked worker inherits them; one started fresh used
	to run on the defaults, analysing every sample differently and saying
	nothing, so a Python without `fork` would have broken the scan silently.
	"""

	monkeypatch.setattr(subsample.parallelism, "_START_METHOD", start_method)
	monkeypatch.setattr(subsample.parallelism, "can_fork_safely", lambda: True)
	_pin_cpus(monkeypatch, 4)

	results = subsample.parallelism.map_analysis(_settings_of, [1, 2], player_active=False)

	for analysis_config, float_ceiling, pid in results:
		assert pid != os.getpid()
		assert analysis_config == _SETTINGS_UNDER_TEST[0]
		assert float_ceiling == _SETTINGS_UNDER_TEST[1]


def _double (value: int) -> int:

	"""Module-level (picklable) worker for the map_analysis tests."""

	return value * 2


def test_map_analysis_preserves_order () -> None:

	"""map_analysis returns results in item order."""

	assert subsample.parallelism.map_analysis(_double, [1, 2, 3, 4], player_active=False) == [2, 4, 6, 8]


def test_map_analysis_empty_items () -> None:

	"""No items → empty list (and no pool is started)."""

	assert subsample.parallelism.map_analysis(_double, [], player_active=False) == []


def _pid_of (value: int) -> int:

	"""Module-level (picklable) worker reporting the PID that ran it."""

	return os.getpid()


def _raise_on_three (value: int) -> int:

	"""Module-level (picklable) worker that fails for exactly one item."""

	if value == 3:
		raise RuntimeError("worker blew up")

	return value * 2


def test_map_analysis_uses_separate_processes (monkeypatch: pytest.MonkeyPatch) -> None:

	"""The fork branch really does run work in other processes.

	Nothing else distinguishes it from the thread fallback, so the branch that
	carries all the risk — pickling, the fork mp_context, the initializer —
	could otherwise never execute in CI and no test would notice.
	"""

	monkeypatch.setattr(subsample.parallelism, "can_fork_safely", lambda: True)
	_pin_cpus(monkeypatch, 4)

	pids = subsample.parallelism.map_analysis(_pid_of, [1, 2, 3, 4], player_active=False)

	assert all(pid != os.getpid() for pid in pids)


def test_map_analysis_isolates_a_failing_item (monkeypatch: pytest.MonkeyPatch) -> None:

	"""One item raising must not discard the rest of the batch.

	executor.map re-raised the first exception in item order and threw away every
	other result, so a single unreadable file aborted a whole library scan.
	"""

	monkeypatch.setattr(subsample.parallelism, "can_fork_safely", lambda: False)
	_pin_cpus(monkeypatch, 4)

	assert subsample.parallelism.map_analysis(
		_raise_on_three, [1, 2, 3, 4], player_active=False,
	) == [2, 4, None, 8]


def test_map_analysis_isolates_failures_across_processes (monkeypatch: pytest.MonkeyPatch) -> None:

	"""Per-item isolation holds on the process branch too."""

	monkeypatch.setattr(subsample.parallelism, "can_fork_safely", lambda: True)
	_pin_cpus(monkeypatch, 4)

	assert subsample.parallelism.map_analysis(
		_raise_on_three, [1, 2, 3, 4], player_active=False,
	) == [2, 4, None, 8]


def test_native_subsystem_makes_forking_unsafe (monkeypatch: pytest.MonkeyPatch) -> None:

	"""Opening a device rules out forking, even though it starts no Python thread.

	PortAudio and rtmidi run their callbacks on threads created in C, which
	threading.active_count() cannot see — so a process holding an open audio
	stream still reported a count of 1 and looked forkable.
	"""

	monkeypatch.setattr(subsample.parallelism, "_native_subsystem_started", False)
	assert subsample.parallelism.can_fork_safely() is True

	subsample.parallelism.note_native_subsystem_started()
	assert subsample.parallelism.can_fork_safely() is False


def test_forking_unsafe_with_a_live_thread () -> None:

	"""A live background thread makes forking a worker pool unsafe."""

	started = threading.Event()
	release = threading.Event()

	def _hold () -> None:
		started.set()
		release.wait(timeout=5.0)

	worker = threading.Thread(target=_hold)
	worker.start()
	started.wait(timeout=5.0)

	try:
		assert subsample.parallelism.can_fork_safely() is False
	finally:
		release.set()
		worker.join(timeout=5.0)


_PARENT = os.getpid()


def _dies_on_item_two (item: int) -> int:

	"""Work that kills its worker on one item, as the OOM killer would.

	Defined at module level so the pool can pickle it, and guarded on the pid so
	the single-threaded retry can finish in this process.
	"""

	if item == 2 and os.getpid() != _PARENT:
		os._exit(1)

	return item * 10


def test_a_dead_worker_does_not_empty_the_batch (caplog: pytest.LogCaptureFixture) -> None:

	"""BrokenProcessPool is a RuntimeError, so the per-item `except Exception`
	caught it first: every remaining item was logged as failed and set to None,
	and the recovery written for exactly this case could never run.  One worker
	killed by the OOM killer therefore lost the whole startup scan or import."""

	if not (subsample.parallelism.analysis_worker_count(player_active=False) > 1
	        and subsample.parallelism.can_fork_safely()):
		pytest.skip("this machine runs the batch in-process, so no worker can die")

	with caplog.at_level(logging.WARNING):
		results = subsample.parallelism.map_analysis(
			_dies_on_item_two, list(range(8)), player_active=False,
		)

	assert results == [0, 10, 20, 30, 40, 50, 60, 70]
	assert any("pool died" in record.message for record in caplog.records), \
		"the pool never died, so this proves nothing"


def _dies_on_item_two_and_fails_on_item_five (item: int) -> int:

	"""As _dies_on_item_two, plus one item that is unreadable wherever it runs."""

	if item == 5:
		raise ValueError("unreadable")

	return _dies_on_item_two(item)


def test_an_item_that_fails_in_the_retry_is_skipped_not_fatal (caplog: pytest.LogCaptureFixture) -> None:

	"""After a worker dies the batch is redone in-process, and one bad file
	there is skipped like anywhere else rather than ending the retry."""

	if not (subsample.parallelism.analysis_worker_count(player_active=False) > 1
	        and subsample.parallelism.can_fork_safely()):
		pytest.skip("this machine runs the batch in-process, so no worker can die")

	with caplog.at_level(logging.WARNING):
		results = subsample.parallelism.map_analysis(
			_dies_on_item_two_and_fails_on_item_five, list(range(8)), player_active=False,
		)

	assert results == [0, 10, 20, 30, 40, None, 60, 70]
	assert any("Analysis failed for item 5 - skipping it" in record.message for record in caplog.records)
