"""Shared CPU policy for Subsample's background sample-analysis worker pools.

Fingerprinting a sample — the 58-dimension analysis behind every match — is
heavy, CPU-bound work, and Subsample runs it in a pool of background workers in
two places: the one-off library scan at startup (``subsample.library``) and the
live analyser that fingerprints sounds as they are captured
(``subsample.recorder``).  This module holds the two policy decisions those
pools share, so they behave consistently:

  * **How many workers.**  Before anything is playing — at startup, or an
    offline rebuild from the command-line tools — analysis takes the whole
    machine so the library is ready as fast as possible.  While the player is
    live, it pulls back to a small share of the cores, leaving the rest for the
    real-time audio thread so playback never stutters while new sounds are
    still being fingerprinted in the background.

  * **One math thread per worker.**  Subsample already spreads work across whole
    samples, so letting NumPy's linear-algebra backend open its own thread pool
    inside each worker only piles hundreds of threads onto a few dozen cores —
    slower, not faster, and needlessly jittery next to a live audio thread.
    Each worker is pinned to a single BLAS thread.
"""
import concurrent.futures
import multiprocessing
import os
import threading
import typing
import warnings

import threadpoolctl


# While the player is live, background analysis is limited to roughly this
# fraction of the machine (one core in four), leaving the rest for audio.
# Deliberately generous to the player: a smooth-sounding instrument matters
# more than how quickly the background rebuild finishes.
_LIVE_CORE_DIVISOR = 4

# Fraction of the machine the startup / offline library rebuild uses.  Kept
# below 1.0 so an all-core fingerprinting burst leaves headroom for the OS and
# anything else running, rather than pegging every last thread — a saner default
# on any machine.  (On a power-limited CPU this won't lower the peak core
# temperature — the package draws to its power limit regardless of how many
# cores are busy — but it keeps the machine responsive and cuts load where the
# limit is cooling rather than power.)
_REBUILD_CORE_FRACTION = 0.75

# Holds this process's BLAS thread-limit so it is not garbage-collected:
# threadpoolctl restores the previous (unlimited) setting when the limiter is
# dropped, so we keep a reference for the cap to last the life of the process.
_blas_limiter: typing.Any = None


def analysis_worker_count (player_active: bool) -> int:

	"""Number of background analysis workers to run.

	``player_active`` True means audio is playing right now, so analysis takes
	only a small share of the cores and leaves the rest for the audio thread;
	False (startup, or an offline rebuild) uses most of the machine — a safe
	fraction that stays fast while leaving the OS some headroom.
	"""

	cpu = os.cpu_count() or 1

	if player_active:
		return max(1, cpu // _LIVE_CORE_DIVISOR)

	return max(1, round(cpu * _REBUILD_CORE_FRACTION))


def can_fork_safely () -> bool:

	"""Whether a forked worker pool is safe to start from this process.

	A forked child inherits every lock in whatever state it held at the instant
	of the fork, so forking from a multi-threaded process risks the child
	deadlocking on a mutex another thread happened to hold.  The library scan
	forks only at startup, before any of Subsample's own threads exist; this
	guards that invariant so callers can fall back to threads anywhere a thread
	is already running.
	"""

	return threading.active_count() == 1


def cap_blas_threads () -> None:

	"""Pin this process's math backend (OpenBLAS/MKL/…) to a single thread.

	Subsample parallelises across samples, never within one, so a multi-threaded
	BLAS pool underneath the worker pool only oversubscribes the CPU and can
	disturb the real-time audio thread.  Safe to call more than once; the limit
	holds until the process exits.
	"""

	global _blas_limiter

	_blas_limiter = threadpoolctl.threadpool_limits(limits=1)


def init_analysis_worker () -> None:

	"""Set up one freshly-started analysis worker process.

	Runs once per worker as the process pool's initializer, pinning it to a
	single BLAS thread.
	"""

	cap_blas_threads()


def map_analysis (
	func: typing.Callable[[typing.Any], typing.Any],
	items: typing.Sequence[typing.Any],
	*,
	player_active: bool,
) -> list[typing.Any]:

	"""Run ``func`` over every item in the background analysis worker pool.

	This is the shared engine behind both the startup library scan and bulk
	``import``.  Fingerprinting is GIL-heavy Python work, so it runs in a forked
	process pool — sidestepping the interpreter lock for the multi-core speedup —
	whenever this process can fork safely (it is single-threaded) and there is
	more than one item.  Otherwise it falls back to a thread pool, so the same
	call stays correct from a running session or the test suite.  For the process
	path ``func`` and every item must be picklable (a module-level function and
	picklable arguments).  Results come back in ``items`` order.
	"""

	if not items:
		return []

	# Cap BLAS first: the workers each pin it too, but doing it here also means
	# no live BLAS thread pool exists in this process at fork time.
	cap_blas_threads()

	n_workers = min(analysis_worker_count(player_active), len(items))

	# Python 3.14 emits a blanket "fork() in a multi-threaded process"
	# DeprecationWarning on every worker fork, but we only fork once
	# can_fork_safely() has confirmed no other user thread is running — leaving
	# just the pool's own manager thread, whose fork CPython coordinates.
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message=".*multi-threaded, use of fork.*",
			category=DeprecationWarning,
		)

		executor: concurrent.futures.Executor

		if n_workers > 1 and can_fork_safely():
			executor = concurrent.futures.ProcessPoolExecutor(
				max_workers=n_workers,
				mp_context=multiprocessing.get_context("fork"),
				initializer=init_analysis_worker,
			)
		else:
			executor = concurrent.futures.ThreadPoolExecutor(
				max_workers=n_workers,
				thread_name_prefix="analysis",
			)

		with executor:
			return list(executor.map(func, items))
