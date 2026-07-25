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
import contextlib
import logging
import math
import multiprocessing
import os
import threading
import typing
import warnings

import threadpoolctl


_log = logging.getLogger(__name__)


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

# Keeps the limiter object alive for the life of the process.  threadpoolctl
# applies the cap eagerly in the constructor and only reverts it from an
# explicit __exit__ / restore_original_limits() — there is no __del__ — so
# dropping this reference would NOT un-cap.  Holding it simply keeps the object
# inspectable and makes the process-wide intent explicit.
_blas_limiter: typing.Any = None

# Set once a PortAudio stream or MIDI port has been opened in this process.  See
# note_native_subsystem_started / can_fork_safely.
_native_subsystem_started: bool = False


def usable_cpu_count () -> int:

	"""Number of CPUs this process may actually run on.

	``os.cpu_count()`` reports the machine, not the process's allowance, so it
	overcounts wherever the process is confined: a container with ``--cpus=``, a
	batch scheduler that pins, a ``taskset``ed CI job.  Sizing a pool from it
	produced 16 workers on a 2-CPU allowance — heavy oversubscription and thrash
	on exactly the shared machines that can least afford it.  ``sched_getaffinity``
	is Linux-only, hence the fallback.
	"""

	try:
		return len(os.sched_getaffinity(0))

	except AttributeError:
		# No affinity API (macOS, Windows) — the machine count is the best
		# available answer there.
		return os.cpu_count() or 1


def analysis_worker_count (player_active: bool) -> int:

	"""Number of background analysis workers to run.

	``player_active`` True means audio is playing right now, so analysis takes
	only a small share of the cores and leaves the rest for the audio thread;
	False (startup, or an offline rebuild) uses most of the machine — a safe
	fraction that stays fast while leaving the OS some headroom.
	"""

	cpu = usable_cpu_count()

	if player_active:
		return max(1, cpu // _LIVE_CORE_DIVISOR)

	# floor, not round: rounding gave 2 of 2 cores on a dual-core machine (the
	# whole point of the fraction is to leave something over) and banker's
	# rounding made 6 cores yield 4 rather than 5.  Floor is monotonic and
	# always leaves at least one core free above the single-core case.
	return max(1, math.floor(cpu * _REBUILD_CORE_FRACTION))


def note_native_subsystem_started () -> None:

	"""Record that a subsystem owning native (C-created) threads is now live.

	Called by the code that opens PortAudio or a MIDI port.  Those libraries
	spawn their callback threads inside C, where ``threading.active_count()``
	cannot see them, so this flag is the only reliable signal that forking has
	stopped being safe.  One-way: nothing clears it, because a device closed
	and reopened leaves the same hazard.
	"""

	global _native_subsystem_started

	_native_subsystem_started = True


def can_fork_safely () -> bool:

	"""Whether a forked worker pool is safe to start from this process.

	A forked child inherits every lock in whatever state it held at the instant
	of the fork, so forking from a multi-threaded process risks the child
	deadlocking on a mutex another thread happened to hold.  The library scan
	forks only at startup, before any of Subsample's own threads exist.

	Two conditions, because one test cannot see both hazards.
	``threading.active_count()`` catches Subsample's own Python threads (the
	watcher, OSC, the recorder and player subsystems) but is blind to threads
	created inside C — and those are the dangerous ones: a process holding an
	open PortAudio stream and rtmidi ports still reports ``active_count() == 1``
	while carrying dozens of native threads.  ``note_native_subsystem_started``
	covers that half.

	Not covered, deliberately: OpenBLAS's worker pool, which ``import numpy``
	starts before any of Subsample's code runs.  It installs ``pthread_atfork``
	handlers that reset the pool in the child, so it is survivable — and
	refusing to fork on its account would disable the process pool everywhere.
	"""

	if _native_subsystem_started:
		return False

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
	whenever this process can fork safely (see ``can_fork_safely``) and there is
	more than one item.  Otherwise it falls back to a thread pool, so the same
	call stays correct from a running session or the test suite.  For the process
	path ``func`` and every item must be picklable (a module-level function and
	picklable arguments).  Results come back in ``items`` order.

	**Failures are isolated per item.**  An item whose call raises is logged and
	comes back as ``None``; the rest of the batch still completes.  Callers
	already treat ``None`` as "this one did not load" (both of them skip it), and
	one unreadable file in a thousand-sample library must not abort the scan.
	This also contains ``BrokenProcessPool``, which the process path can raise if
	a worker is killed — most plausibly by the OOM killer, since every worker
	holds a full PCM buffer.
	"""

	if not items:
		return []

	# Cap BLAS first: the workers each pin it too, but doing it here also means
	# no live BLAS thread pool exists in this process at fork time.
	cap_blas_threads()

	n_workers = min(analysis_worker_count(player_active), len(items))

	use_processes = n_workers > 1 and can_fork_safely()

	executor: concurrent.futures.Executor

	if use_processes:
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

	# Say which engine is in use.  The two differ by a large multiple on a
	# multi-core box, and the fallback is silent and easy to trigger without
	# realising — loading a second program, for instance, happens after the first
	# program's transform pool has started threads, so every bank after the first
	# scans GIL-bound while the log said nothing.
	_log.info(
		"Analysing %d item(s) across %d %s",
		len(items), n_workers, "process(es)" if use_processes else "thread(s)",
	)

	results: list[typing.Any] = [None] * len(items)

	# Python 3.14 warns on every fork from a process holding more than one OS
	# thread, which includes the OpenBLAS pool that `import numpy` starts before
	# any of our code runs.  can_fork_safely() has already established that no
	# Subsample thread and no device-owning subsystem is live, and OpenBLAS
	# resets its pool via pthread_atfork, so the warning has nothing left to warn
	# about here.  Scoped to the fork branch only: warnings.catch_warnings mutates
	# a process-global filter list and is not thread-safe, and this branch is
	# reached only when we have just verified the process is single-threaded.
	suppress_fork_warning: typing.ContextManager[typing.Any] = (
		warnings.catch_warnings() if use_processes else contextlib.nullcontext()
	)

	with suppress_fork_warning:
		if use_processes:
			warnings.filterwarnings(
				"ignore",
				message=".*multi-threaded, use of fork.*",
				category=DeprecationWarning,
			)

		return _drain(executor, func, items, results)


def _drain (
	executor: concurrent.futures.Executor,
	func:     typing.Callable[[typing.Any], typing.Any],
	items:    typing.Sequence[typing.Any],
	results:  list[typing.Any],
) -> list[typing.Any]:

	"""Run every item through ``executor``, isolating per-item failures.

	Split out of ``map_analysis`` only so the fork-warning suppression can wrap
	the pool's whole lifetime without indenting the body twice.
	"""

	try:
		with executor:
			# submit + per-future result, not executor.map: map re-raises the
			# first exception in item order and discards every other result, so a
			# single unreadable file aborted the whole scan.
			futures = {executor.submit(func, item): index for index, item in enumerate(items)}

			for future in concurrent.futures.as_completed(futures):
				index = futures[future]

				try:
					results[index] = future.result()

				except Exception:
					_log.exception("Analysis worker failed for item %d — skipping it", index)
					results[index] = None

	except concurrent.futures.process.BrokenProcessPool:
		# A worker died outright (OOM killer is the realistic cause — each holds
		# a full PCM buffer).  Everything still pending is lost, so redo the
		# batch in-process rather than returning a half-empty library.
		_log.warning(
			"Analysis worker pool died — retrying %d item(s) single-threaded",
			len(items),
		)

		for index, item in enumerate(items):
			try:
				results[index] = func(item)

			except Exception:
				_log.exception("Analysis failed for item %d — skipping it", index)
				results[index] = None

	return results
