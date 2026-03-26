"""Hot-loading instrument samples from a watched directory.

Monitors a directory for new instrument samples (WAV + .analysis.json
sidecar pairs) arriving at runtime — for example, from a remote recorder
instance syncing to a shared network drive or Dropbox folder — and loads
them into the live instrument library without restarting.

The sidecar file's arrival is used as the trigger signal: the recorder
always writes the WAV first, then the sidecar, so a sidecar appearing
means both files are present and the analysis is complete.

A short debounce window (DEBOUNCE_SECONDS) absorbs the file-sync delay
typical of network drives and cloud sync tools before attempting to load.
If the sidecar or WAV is still incomplete, the load is retried up to
MAX_RETRIES times before giving up (the sample will be picked up on
next restart).
"""

import logging
import pathlib
import threading
import typing

import watchdog.events
import watchdog.observers

import subsample.audio
import subsample.cache
import subsample.library


_log = logging.getLogger(__name__)

_SIDECAR_SUFFIX: str = ".analysis.json"
_DEBOUNCE_SECONDS: float = 1.0
_MAX_RETRIES: int = 3
_RETRY_DELAY_SECONDS: float = 2.0


class InstrumentWatcher:

	"""Watch a directory for new instrument samples and hot-load them.

	Monitors the given directory for .analysis.json sidecar files appearing
	at runtime. When a new sidecar is detected, waits briefly for the file
	write to complete, then loads the sidecar and its corresponding WAV file
	into a SampleRecord and invokes the on_sample_loaded callback.

	Sidecars present in known_sidecars at construction time are silently
	ignored — they were already loaded at startup.
	"""

	def __init__ (
		self,
		directory: pathlib.Path,
		known_sidecars: set[pathlib.Path],
		on_sample_loaded: typing.Callable[[subsample.library.SampleRecord], None],
	) -> None:

		self._directory = directory
		self._known_sidecars: frozenset[pathlib.Path] = frozenset(known_sidecars)
		self._on_sample_loaded = on_sample_loaded

		# Active debounce timers keyed by resolved sidecar path.
		# Protected by _lock — modified from the watchdog callback thread and
		# the timer threads.
		self._timers: dict[pathlib.Path, threading.Timer] = {}
		self._lock = threading.Lock()

		handler = _SidecarHandler(self._on_sidecar_event)
		# Type annotated as Any because watchdog has no type stubs;
		# ignore_missing_imports silences import errors but leaves the
		# resolved type as Any, which cannot be used as a type annotation.
		self._observer: typing.Any = watchdog.observers.Observer()
		self._observer.schedule(handler, str(directory), recursive=False)

	def start (self) -> None:

		"""Start the background observer thread."""

		self._observer.start()
		_log.info("Instrument watcher started on %s", self._directory)

	def stop (self) -> None:

		"""Stop the observer thread and cancel any pending debounce timers."""

		self._observer.stop()
		self._observer.join()

		with self._lock:
			for timer in self._timers.values():
				timer.cancel()
			self._timers.clear()

		_log.debug("Instrument watcher stopped")

	def _on_sidecar_event (self, sidecar_path: pathlib.Path) -> None:

		"""Schedule (or reschedule) a debounced load for the given sidecar path.

		Called from the watchdog observer thread on every sidecar create/modify
		event. Multiple rapid events for the same file (common during network
		sync) collapse into a single load attempt after the debounce window.
		"""

		path = sidecar_path.resolve()

		if path in self._known_sidecars:
			return

		with self._lock:
			# Cancel any existing timer for this path — file may still be writing.
			existing = self._timers.pop(path, None)
			if existing is not None:
				existing.cancel()

			timer = threading.Timer(
				_DEBOUNCE_SECONDS,
				self._attempt_load,
				args=(path, 0),
			)
			self._timers[path] = timer

			# Start inside the lock so stop() cannot cancel and clear this timer
			# between registration and start.
			timer.start()

	def _attempt_load (self, sidecar_path: pathlib.Path, attempt: int) -> None:

		"""Try to load a sidecar + WAV pair, retrying if files are not yet ready.

		Runs on a timer thread (not the main or watchdog thread). On success,
		invokes on_sample_loaded. On transient failure (file mid-write, WAV not
		yet synced), reschedules itself. Gives up after MAX_RETRIES attempts.
		"""

		with self._lock:
			self._timers.pop(sidecar_path, None)

		result = subsample.cache.load_sidecar(sidecar_path)

		if result is None:
			self._schedule_retry(sidecar_path, attempt, reason="sidecar not readable")
			return

		spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result

		# Derive audio file path: strip .analysis.json suffix to get the WAV name.
		audio_name = sidecar_path.name[: -len(_SIDECAR_SUFFIX)]
		audio_path = sidecar_path.parent / audio_name

		if not audio_path.exists():
			self._schedule_retry(
				sidecar_path, attempt,
				reason=f"WAV not yet present ({audio_name})",
			)
			return

		try:
			file_info = subsample.audio.read_audio_file(audio_path)
		except (OSError, ValueError) as exc:
			self._schedule_retry(
				sidecar_path, attempt,
				reason=f"could not read WAV ({audio_name}): {exc}",
			)
			return

		record = subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = pathlib.Path(audio_name).stem,
			spectral    = spectral,
			rhythm      = rhythm,
			pitch       = pitch,
			timbre      = timbre,
			level       = level,
			band_energy = band_energy,
			params      = params,
			duration    = duration,
			audio       = file_info.audio,
			filepath    = audio_path,
		)

		self._on_sample_loaded(record)

	def _schedule_retry (
		self,
		sidecar_path: pathlib.Path,
		attempt: int,
		reason: str,
	) -> None:

		"""Retry the load after RETRY_DELAY_SECONDS, or give up at MAX_RETRIES."""

		if attempt >= _MAX_RETRIES:
			_log.warning(
				"Watcher: giving up on %s after %d attempts (%s) — "
				"sample will be picked up on next restart",
				sidecar_path.name, _MAX_RETRIES, reason,
			)
			return

		_log.debug(
			"Watcher: %s — retrying in %.1fs (attempt %d/%d)",
			reason, _RETRY_DELAY_SECONDS, attempt + 1, _MAX_RETRIES,
		)

		timer = threading.Timer(
			_RETRY_DELAY_SECONDS,
			self._attempt_load,
			args=(sidecar_path, attempt + 1),
		)

		with self._lock:
			self._timers[sidecar_path] = timer

		timer.start()


class _SidecarHandler (watchdog.events.FileSystemEventHandler):

	"""Watchdog event handler that filters for .analysis.json file events."""

	def __init__ (self, callback: typing.Callable[[pathlib.Path], None]) -> None:

		super().__init__()
		self._callback = callback

	def on_created (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward sidecar creation events to the callback."""

		self._dispatch_if_sidecar(event)

	def on_modified (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward sidecar modification events (e.g. cloud-sync overwrites)."""

		self._dispatch_if_sidecar(event)

	def _dispatch_if_sidecar (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Call the callback if the event is for a sidecar file."""

		if event.is_directory:
			return

		path = pathlib.Path(str(event.src_path))

		if path.name.endswith(_SIDECAR_SUFFIX):
			self._callback(path)
