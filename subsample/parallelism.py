"""Shared CPU policy for Subsample's background worker pools.

Fingerprinting a sample — the 58-dimension analysis behind every match — and
rendering a variant are heavy, CPU-bound work, and Subsample runs them in pools
of background workers: the one-off library scan at startup
(``subsample.library``), the live analyser that fingerprints sounds as they are
captured (``subsample.recorder``), the render pool (``subsample.transform``),
and a file analysed while the player plays (``run_in_analysis_worker``).
This module holds the policy decisions those pools share, so they behave
consistently:

  * **How many workers.**  Before anything is playing — at startup, or an
    offline rebuild from the command-line tools — analysis takes the whole
    machine so the library is ready as fast as possible.  While the player is
    live, it pulls back to a small share of the cores, leaving the rest for the
    real-time audio thread.

  * **Out of the player's process while it plays** (#4667).  Python runs one
    thread's code at a time, so a worker thread in the player's process makes
    the audio callback and the note handlers wait their turn however many cores
    are free: measured at 1024-frame buffers, renders on the player's threads
    cost hundreds of dropouts a minute, and the same renders in a process of
    their own cost none (#4666).  So background work runs in worker processes
    (``BackgroundPool``), and a sound crosses to and from them in a file
    (``write_audio`` / ``read_audio``), which copies without holding the lock
    where pickling would not.  Where worker processes cannot start, the pools
    fall back to threads and say so once.

  * **One math thread per worker.**  Subsample already spreads work across whole
    samples, so letting NumPy's linear-algebra backend open its own thread pool
    inside each worker only piles hundreds of threads onto a few dozen cores —
    slower, not faster, and needlessly jittery next to a live audio thread.
    Each worker is pinned to a single BLAS thread.
"""

import atexit
import concurrent.futures
import contextlib
import dataclasses
import logging
import math
import multiprocessing
import multiprocessing.forkserver
import os
import pathlib
import shutil
import signal
import tempfile
import threading
import traceback
import typing
import warnings

import numpy
import threadpoolctl

import subsample.audio
import subsample.cache
import subsample.config


_log = logging.getLogger(__name__)

# How the analysis pool starts its worker processes.  `fork` starts a worker
# with everything already imported, which is quick.  Nothing a worker needs
# depends on it: init_analysis_worker hands each worker the settings the parent
# configured process-wide, so `forkserver` or `spawn` analyse the same way
# (#390), and a Python that drops `fork` needs only this line changed, and
# can_fork_safely's gate and the fork-warning filter in map_analysis revisited.
_START_METHOD: str = "fork"


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

	# Asked with hasattr rather than by catching AttributeError: mypy checking
	# for macOS knows the function is missing there and refuses a bare call to
	# it, but understands this test.
	if hasattr(os, "sched_getaffinity"):
		return len(os.sched_getaffinity(0))

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
	holds until the process exits.  numpy for Apple silicon is built on Apple's
	Accelerate, which threadpoolctl cannot limit, so there this does nothing.
	"""

	global _blas_limiter

	_blas_limiter = threadpoolctl.threadpool_limits(limits=1)


def init_analysis_worker (
	analysis_config:    subsample.config.AnalysisConfig,
	float_ceiling_dbfs: typing.Optional[float],
) -> None:

	"""Set up one freshly-started analysis worker process.

	Runs once per worker as the process pool's initializer.  It pins the worker
	to a single BLAS thread and gives it the two settings the parent set
	process-wide, which the library scan reads from module state rather than
	being passed them: the analysis config and the float import ceiling.  A
	forked worker inherits both, but one started fresh would run on the
	defaults and say nothing, analysing every sample differently (#390).

	It also leaves the terminal's Ctrl+C and hang-up to the parent (#4702).  A
	terminal sends both to the whole process group, workers included, and the
	parent turns Ctrl+C into a drain that waits for its workers.  A worker that
	took the signal died with it, and the capture or render it held was lost.
	SIGTERM keeps its default, which is how a broken pool ends its workers.
	"""

	signal.signal(signal.SIGINT, signal.SIG_IGN)
	signal.signal(signal.SIGHUP, signal.SIG_IGN)

	cap_blas_threads()

	subsample.cache.set_analysis_config(analysis_config)
	subsample.audio.set_float_import_ceiling(float_ceiling_dbfs)


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
			mp_context=multiprocessing.get_context(_START_METHOD),
			initializer=init_analysis_worker,
			initargs=(
				subsample.cache.analysis_config(),
				subsample.audio.float_import_ceiling(),
			),
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

				except concurrent.futures.process.BrokenProcessPool:
					# Re-raised for the handler below, which redoes the whole
					# batch.  It has to be caught BEFORE the general case:
					# BrokenProcessPool is a RuntimeError, so `except Exception`
					# would swallow it, mark every remaining item as skipped, and
					# leave the recovery below unreachable — one dead worker
					# would empty the batch it was supposed to save.
					raise

				except Exception:
					_log.exception("Analysis worker failed for item %d - skipping it", index)
					results[index] = None

	except concurrent.futures.process.BrokenProcessPool:
		# A worker died outright (OOM killer is the realistic cause — each holds
		# a full PCM buffer).  Everything still pending is lost, so redo the
		# batch in-process rather than returning a half-empty library.
		_log.warning(
			"Analysis worker pool died - retrying %d item(s) single-threaded",
			len(items),
		)

		for index, item in enumerate(items):
			try:
				results[index] = func(item)

			except Exception:
				_log.exception("Analysis failed for item %d - skipping it", index)
				results[index] = None

	return results


# ---------------------------------------------------------------------------
# Background pools: work the player must not wait on (#4667)
# ---------------------------------------------------------------------------

# How a background pool starts its worker processes.  Not `fork`: once
# PortAudio or a MIDI port is open the player holds threads created in C, and a
# forked child inherits their locks in whatever state they were (can_fork_safely).
# A forkserver is a fresh interpreter, started once, that imports the heavy
# modules below and then forks each worker from itself, so a worker starts in
# milliseconds with them loaded and shares their pages with its siblings.
_BACKGROUND_START_METHOD: str = "forkserver"

# What the forkserver imports before it forks a worker: everything a render or
# an analysis reaches, librosa, scipy and Rubber Band's wrapper among it.
_FORKSERVER_PRELOAD: list[str] = ["subsample.transform", "subsample.cache", "subsample.recorder"]

# A log record's attribute naming the once-only key it was logged under, so the
# player logs what workers relay once in the session, not once per worker.
ONCE_KEY: str = "subsample_once_key"

# What run_in_analysis_worker returns: whatever the function it runs returns.
_Result = typing.TypeVar("_Result")

# Why worker processes cannot start here: None when they can, or until asked.
_processes_refused_reason: typing.Optional[str] = None
_processes_checked: bool = False
_processes_lock: threading.Lock = threading.Lock()

# The background pools the whole session shares, by name, made on first use,
# and whether they are stopped at exit yet.
_shared_pools: dict[str, "BackgroundPool"] = {}
_shared_pools_lock: threading.Lock = threading.Lock()
_shared_pools_stopped_at_exit: bool = False

# The folder sounds cross between processes in, made on first use and removed
# at exit.  Its name is the prefix, its run's process ID, and mkdtemp's own
# letters.
_HAND_OFF_PREFIX = "subsample-work-"
_hand_off_folder: typing.Optional[pathlib.Path] = None
_hand_off_lock: threading.Lock = threading.Lock()

# True in a background worker process, where what a job logs is collected and
# handed back to the player (relaying_logs), since nothing logged in a process
# started by the forkserver reaches the player's log otherwise.
_in_background_worker: bool = False


@dataclasses.dataclass(frozen=True)
class AudioFile:

	"""A sound handed between processes in a file: where it is, and how to read it back."""

	path:  str
	shape: tuple[int, ...]
	dtype: str


def hand_off_folder () -> pathlib.Path:

	"""The folder sounds cross between the player and its workers in, made on first use.

	Private to this run and removed when it ends.  It lives in the system's
	temporary folder (TMPDIR), where a worker can write as easily as the player.
	Its name carries this run's process ID, so a later run can tell when it was
	abandoned (_remove_abandoned_hand_off_folders).
	"""

	global _hand_off_folder

	with _hand_off_lock:
		if _hand_off_folder is None:
			_remove_abandoned_hand_off_folders()

			_hand_off_folder = pathlib.Path(tempfile.mkdtemp(prefix=f"{_HAND_OFF_PREFIX}{os.getpid()}-"))
			atexit.register(shutil.rmtree, _hand_off_folder, ignore_errors=True)

		return _hand_off_folder


def _remove_abandoned_hand_off_folders () -> None:

	"""Remove this user's hand-off folders whose run has gone (#4702).

	A run removes its own folder at exit, but one killed outright, by SIGKILL
	or the out-of-memory killer, cannot.  It left the raw sound of every
	capture and render it was handing over, in memory where the temporary
	folder is held there.  A folder whose process ID still runs is left alone,
	even if another program now has that ID; one from v0.6.8, named without
	its process ID, cannot be told apart from a live one, and is left too.
	"""

	for folder in pathlib.Path(tempfile.gettempdir()).glob(f"{_HAND_OFF_PREFIX}*-*"):
		pid = folder.name[len(_HAND_OFF_PREFIX):].split("-", 1)[0]

		try:
			if not pid.isdigit() or folder.stat().st_uid != os.getuid() or _process_runs(int(pid)):
				continue
		except OSError:
			continue

		shutil.rmtree(folder, ignore_errors=True)


def _process_runs (pid: int) -> bool:

	"""Whether a process with this ID exists, as far as this user can tell."""

	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except PermissionError:
		# Another user's: it exists.
		return True

	return True


def write_audio (audio: numpy.ndarray, folder: pathlib.Path) -> AudioFile:

	"""Write a sound to a new file in ``folder`` for another process to read.

	Written unbuffered from the array's own memory, so the copy happens in the
	kernel with the interpreter lock released.  Pickling a sound copies it while
	holding the lock: a 35 MB sound handed over that way cost the audio 157
	dropouts in 20 s, where a file cost none (#4667).
	"""

	audio = numpy.ascontiguousarray(audio)
	fd, path = tempfile.mkstemp(dir=folder, suffix=".pcm")

	try:
		with os.fdopen(fd, "wb", buffering=0) as stream:
			# An empty sound writes an empty file: a view of no bytes cannot be cast.
			view = memoryview(audio).cast("B") if audio.nbytes else memoryview(b"")

			while view:
				written = stream.write(view)
				view    = view[written:]

	except BaseException:
		with contextlib.suppress(OSError):
			os.unlink(path)

		raise

	return AudioFile(path=path, shape=tuple(audio.shape), dtype=audio.dtype.str)


def read_audio (handle: AudioFile, *, remove: bool) -> numpy.ndarray:

	"""Read a handed-over sound into a new array, and remove its file if asked.

	Read straight into the array's memory, so, as in write_audio, the copy
	happens with the interpreter lock released.
	"""

	audio = numpy.empty(handle.shape, dtype=numpy.dtype(handle.dtype))

	with open(handle.path, "rb", buffering=0) as stream:
		view = memoryview(audio).cast("B") if audio.nbytes else memoryview(bytearray())

		while view:
			read = stream.readinto(view)

			if not read:
				raise OSError(f"{handle.path} is shorter than the sound it should hold")

			view = view[read:]

	if remove:
		os.unlink(handle.path)

	return audio


def remove_audio (handle: AudioFile) -> None:

	"""Remove a handed-over sound's file, if it is still there."""

	with contextlib.suppress(OSError):
		os.unlink(handle.path)


class _Collector (logging.Handler):

	"""Keeps what a worker's job logs, formatted, to hand back to the player."""

	def __init__ (self, records: list[logging.LogRecord]) -> None:

		"""A collector that appends to ``records``."""

		super().__init__()
		self._records = records

	def emit (self, record: logging.LogRecord) -> None:

		"""Keep a record in a form that pickles: its message formatted, its traceback as text."""

		record.msg  = record.getMessage()
		record.args = None

		if record.exc_info:
			record.exc_text = logging.Formatter().formatException(record.exc_info)
			record.exc_info = None

		self._records.append(record)


@contextlib.contextmanager
def relaying_logs () -> typing.Iterator[list[logging.LogRecord]]:

	"""Collect what this worker's job logs while the block runs, for the player to log.

	Only in a background worker process.  Elsewhere (a thread, the tests) a
	record is logged where it is made, and the list stays empty.
	"""

	records: list[logging.LogRecord] = []

	if not _in_background_worker:
		yield records
		return

	collector = _Collector(records)
	logger    = logging.getLogger("subsample")
	logger.addHandler(collector)

	try:
		yield records
	finally:
		logger.removeHandler(collector)


def log_relayed (
	records: typing.Sequence[logging.LogRecord],
	once:    typing.Optional[typing.Callable[[str, str], None]] = None,
) -> None:

	"""Log, in the player, what a worker's job logged.

	A record logged under a once-only key goes to ``once`` with its key and
	message instead, so the player logs it once in the session, not once per
	worker that met it.
	"""

	for record in records:
		key = getattr(record, ONCE_KEY, None)

		if key is not None and once is not None:
			once(key, record.getMessage())
			continue

		logger = logging.getLogger(record.name)

		if logger.isEnabledFor(record.levelno):
			logger.handle(record)


class RemoteTraceback (Exception):

	"""Where a worker's failure was raised, attached as the cause of the error the player logs."""

	def __init__ (self, text: str) -> None:

		"""Hold the worker's formatted traceback."""

		super().__init__(text)
		self.text = text

	def __str__ (self) -> str:

		"""The traceback, as the worker formatted it."""

		return self.text


@dataclasses.dataclass(frozen=True)
class Failure:

	"""A worker's job that raised: the error, and where, for the player to log."""

	error:     BaseException
	traceback: str

	@classmethod
	def caught (cls, error: BaseException) -> "Failure":

		"""The error being handled now, with its traceback as text, since a traceback does not pickle."""

		return cls(error=error, traceback="".join(traceback.format_exception(error)))

	def rebuilt (self) -> BaseException:

		"""The error, its worker's traceback attached as its cause, ready to log with exc_info."""

		if self.error.__traceback__ is None:
			self.error.__cause__ = RemoteTraceback(self.traceback)

		return self.error


def init_background_worker (
	analysis_config:    subsample.config.AnalysisConfig,
	float_ceiling_dbfs: typing.Optional[float],
) -> None:

	"""Set up one background worker process: the parent's settings, and its logging.

	What a job logs is collected and handed back (relaying_logs), at every
	level: the player's log_relayed keeps what its own level lets through, so
	the player's level decides, as it is now, not as it was when the pool
	started.  The worker's ``subsample`` logger does not pass records on to a
	root logger that, in a process started by the forkserver, has no handler.
	"""

	global _in_background_worker

	init_analysis_worker(analysis_config, float_ceiling_dbfs)

	_in_background_worker = True

	logger = logging.getLogger("subsample")
	logger.setLevel(logging.DEBUG)
	logger.propagate = False


def processes_refused () -> typing.Optional[str]:

	"""Why worker processes cannot start here, or None when they can.

	Asked once, by starting the forkserver.  A refusal is logged once, and from
	then on every background pool runs in threads instead (Simon's call 11b on
	#4513): the player works, and says background work may make it drop out.
	A long TMPDIR is a real cause, for one: the forkserver's socket lives
	there, and Linux refuses a socket path past 107 characters.
	"""

	global _processes_checked, _processes_refused_reason

	with _processes_lock:
		if _processes_checked:
			return _processes_refused_reason

		_processes_checked = True

		try:
			context = multiprocessing.get_context(_BACKGROUND_START_METHOD)
			context.set_forkserver_preload(_FORKSERVER_PRELOAD)

			# Started with hang-up blocked, as the forkserver, its resource
			# tracker and every worker then are (#4702).  A terminal sends its
			# hang-up to the whole process group: a forkserver that took it
			# died, and the pools took that for every worker dying, and ended
			# them with the captures they held.  Blocked rather than ignored,
			# since a thread other than the main one may be the first to ask.
			blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP})

			try:
				multiprocessing.forkserver.ensure_running()
			finally:
				signal.pthread_sigmask(signal.SIG_SETMASK, blocked)

		except (OSError, ValueError, RuntimeError) as exc:
			_processes_refused_reason = str(exc) or type(exc).__name__

			_log.warning(
				"Rendering and analysis run inside the player, because worker processes "
				"could not start here (%s).  While Subsample renders or analyses sounds, "
				"the audio may drop out and notes may sound late.",
				_processes_refused_reason,
			)

		return _processes_refused_reason


class BackgroundPool:

	"""Workers for one kind of background work: processes of their own, or threads.

	Processes when asked for and the machine allows (processes_refused);
	threads otherwise, and for tests that patch what a job runs or share memory
	with it.  A pool whose worker process died (``BrokenProcessPool``) is
	replaced on the next submit; the jobs it held fail, and their owners deal
	with that as with any failed job.
	"""

	def __init__ (self, name: str, workers: int, *, processes: bool) -> None:

		"""A pool of ``workers`` named ``name``, in processes if ``processes`` and they can start."""

		self.name     = name
		self.workers  = max(1, workers)
		self._lock    = threading.Lock()
		self._processes = processes and processes_refused() is None
		self._executor  = self._make()

	@property
	def in_processes (self) -> bool:

		"""Whether this pool's workers are processes of their own."""

		return self._processes

	def _make (self) -> concurrent.futures.Executor:

		"""A new executor of this pool's kind and size."""

		if not self._processes:
			return concurrent.futures.ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=self.name)

		# The settings are read now, at the pool's first use, after the CLI has
		# wired them, as map_analysis reads them for the library scan.
		return concurrent.futures.ProcessPoolExecutor(
			max_workers=self.workers,
			mp_context=multiprocessing.get_context(_BACKGROUND_START_METHOD),
			initializer=init_background_worker,
			initargs=(
				subsample.cache.analysis_config(),
				subsample.audio.float_import_ceiling(),
			),
		)

	def submit (self, fn: typing.Callable[..., typing.Any], /, *args: typing.Any) -> concurrent.futures.Future[typing.Any]:

		"""Run ``fn(*args)`` on a worker, replacing the pool first if a worker of it died."""

		with self._lock:
			try:
				return self._executor.submit(fn, *args)

			except concurrent.futures.process.BrokenProcessPool:
				_log.warning(
					"One of the %s workers stopped unexpectedly, most often because the "
					"machine ran short of memory, and a new one has started.",
					self.name,
				)

				# A broken pool's manager has already stopped, so this is quick.
				self._executor.shutdown(wait=True, cancel_futures=True)
				self._executor = self._make()

				return self._executor.submit(fn, *args)

	def shutdown (self, wait: bool = True) -> None:

		"""Stop the workers once what they were given is done (or at once, if not ``wait``)."""

		with self._lock:
			self._executor.shutdown(wait=wait, cancel_futures=not wait)


def shared_pool (name: str, workers: int) -> BackgroundPool:

	"""The session's pool of worker processes for one kind of work, made on first use.

	Shared by every owner of that work, so eight programs with a render queue
	each share one set of render workers, not eight.  ``workers`` sizes the
	pool when this call makes it; a later call gets the pool as it was made.
	"""

	global _shared_pools_stopped_at_exit

	with _shared_pools_lock:
		pool = _shared_pools.get(name)

		if pool is None:
			# Stopped at exit, before Python takes its modules apart: a pool
			# collected after that fails in its own clean-up and says so.
			if not _shared_pools_stopped_at_exit:
				atexit.register(shutdown_shared_pools)
				_shared_pools_stopped_at_exit = True

			pool = BackgroundPool(name, workers, processes=True)
			_shared_pools[name] = pool

		return pool


def shutdown_shared_pools () -> None:

	"""Stop every shared pool, waiting for what its workers are doing, then remove the hand-off folder.

	The player calls this on its way out, as a backstop to the exit handler: a
	player that has to end with os._exit runs no exit handlers.
	"""

	global _hand_off_folder

	with _shared_pools_lock:
		pools = list(_shared_pools.values())
		_shared_pools.clear()

	for pool in pools:
		pool.shutdown(wait=True)

	with _hand_off_lock:
		if _hand_off_folder is not None:
			shutil.rmtree(_hand_off_folder, ignore_errors=True)
			_hand_off_folder = None


def _call_relaying (
	fn:     typing.Callable[..., typing.Any],
	args:   tuple[typing.Any, ...],
	kwargs: dict[str, typing.Any],
) -> tuple[typing.Any, typing.Optional[Failure], tuple[logging.LogRecord, ...]]:

	"""Run ``fn`` on a worker, and hand back what it returned or raised, and what it logged."""

	with relaying_logs() as records:
		try:
			value = fn(*args, **kwargs)
		except Exception as exc:
			return None, Failure.caught(exc), tuple(records)

	return value, None, tuple(records)


def run_in_analysis_worker (fn: typing.Callable[..., _Result], /, *args: typing.Any, **kwargs: typing.Any) -> _Result:

	"""Run ``fn`` on the session's analysis worker processes, and wait for it (#4667).

	For analysing a file the player has to have while it plays: one dropped in
	a watched folder, one sent by OSC, or a map's reference.  On the caller's
	thread, analysis makes the audio and the notes wait for Python's lock.
	What ``fn`` logs is logged here, and what it raises is raised here, with
	where it was raised.  ``fn`` and its arguments must pickle.  Where worker
	processes cannot start, ``fn`` runs on the caller's thread, as it did.
	"""

	pool = shared_pool("analysis", analysis_worker_count(player_active=True))

	if not pool.in_processes:
		return fn(*args, **kwargs)

	value, failure, records = pool.submit(_call_relaying, fn, args, kwargs).result()

	log_relayed(records)

	if failure is not None:
		raise failure.rebuilt()

	return typing.cast(_Result, value)
