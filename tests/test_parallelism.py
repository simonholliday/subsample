"""Tests for subsample.parallelism — the shared analysis-pool CPU policy."""

import os
import threading

import pytest
import threadpoolctl

import subsample.parallelism


def test_idle_leaves_headroom (monkeypatch: pytest.MonkeyPatch) -> None:

	"""With nothing playing, analysis uses most of the cores but not every thread."""

	monkeypatch.setattr(os, "cpu_count", lambda: 16)

	# 75% of the cores — fast, but the OS keeps some headroom.
	assert subsample.parallelism.analysis_worker_count(player_active=False) == 12


def test_live_reserves_headroom (monkeypatch: pytest.MonkeyPatch) -> None:

	"""While the player is live, analysis takes only a small share of the cores."""

	monkeypatch.setattr(os, "cpu_count", lambda: 16)

	# One core in four; the rest stay free for the real-time audio thread.
	assert subsample.parallelism.analysis_worker_count(player_active=True) == 4


def test_live_yields_at_least_one_worker (monkeypatch: pytest.MonkeyPatch) -> None:

	"""On a low-core machine the live share still leaves at least one worker."""

	monkeypatch.setattr(os, "cpu_count", lambda: 2)

	assert subsample.parallelism.analysis_worker_count(player_active=True) == 1


def test_unknown_cpu_count_is_safe (monkeypatch: pytest.MonkeyPatch) -> None:

	"""os.cpu_count() may return None; the policy must still yield >= 1."""

	monkeypatch.setattr(os, "cpu_count", lambda: None)

	assert subsample.parallelism.analysis_worker_count(player_active=False) == 1
	assert subsample.parallelism.analysis_worker_count(player_active=True) == 1


def test_cap_blas_threads_pins_to_one () -> None:

	"""Capping pins every loaded math backend to a single thread, repeatably."""

	subsample.parallelism.cap_blas_threads()
	subsample.parallelism.cap_blas_threads()  # idempotent — must not raise

	assert all(pool["num_threads"] == 1 for pool in threadpoolctl.threadpool_info())


def test_init_analysis_worker_runs () -> None:

	"""The process-pool initializer runs without error."""

	subsample.parallelism.init_analysis_worker()


def _double (value: int) -> int:

	"""Module-level (picklable) worker for the map_analysis tests."""

	return value * 2


def test_map_analysis_preserves_order () -> None:

	"""map_analysis returns results in item order."""

	assert subsample.parallelism.map_analysis(_double, [1, 2, 3, 4], player_active=False) == [2, 4, 6, 8]


def test_map_analysis_empty_items () -> None:

	"""No items → empty list (and no pool is started)."""

	assert subsample.parallelism.map_analysis(_double, [], player_active=False) == []


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
