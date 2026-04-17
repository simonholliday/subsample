"""Hot-loading instrument samples from a watched directory.

Monitors a directory for new instrument samples arriving at runtime and
loads them into the live instrument library without restarting.

Two detection paths run in parallel:

1. **Sidecar path** (for subsample-sourced files) — watches for
   .analysis.json sidecar files.  The recorder always writes the WAV
   first, then the sidecar, so a sidecar appearing means both files are
   present and the analysis is complete.  This is the fastest path: no
   re-analysis needed.

2. **Audio file path** (for files from any source) — watches for audio
   files with recognised extensions (.wav, .flac, .aiff, .ogg, .mp3).
   After an initial debounce, waits a grace period for a sidecar to
   appear (in case the source is another subsample instance).  If none
   appears, checks that the file is no longer being written (file-size
   stability), runs the full analysis pipeline, writes a sidecar, and
   loads the sample.

Both paths invoke the same on_sample_loaded callback.
"""

import logging
import os
import pathlib
import threading
import typing

import watchdog.events
import watchdog.observers

import subsample.cache
import subsample.library


_log = logging.getLogger(__name__)

_SIDECAR_SUFFIX: str = ".analysis.json"
_DEBOUNCE_SECONDS: float = 1.0
_MAX_RETRIES: int = 3
_RETRY_DELAY_SECONDS: float = 2.0

# Audio file detection — recognises formats supported by audio.read_audio_file().
_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
	".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3", ".mpeg",
})

_AUDIO_DEBOUNCE_SECONDS: float = 2.0
"""Initial debounce for audio file events — absorbs editor saves, partial
writes, and rapid filesystem events before starting the sidecar grace period."""

_SIDECAR_GRACE_SECONDS: float = 5.0
"""How long to wait for a sidecar to appear after an audio file is detected.
If the file comes from another subsample instance, the sidecar arrives within
a second or two.  If none appears, the audio path takes over."""

_STABILITY_CHECK_SECONDS: float = 2.0
"""Interval between file-size comparisons to determine whether a file is still
being written by another application."""

_STABILITY_MAX_CHECKS: int = 5
"""Maximum number of size-stability checks before giving up on a file that
keeps changing size."""


class InstrumentWatcher:

	"""Watch a directory for new instrument samples and hot-load them.

	Monitors the given directory for both .analysis.json sidecar files and
	audio files appearing at runtime.  Sidecar events are handled immediately
	(debounced); audio file events wait for a sidecar grace period, then
	check file-size stability before analyzing.

	Files present in known_sidecars / known_audio at construction time are
	silently ignored — they were already loaded at startup.
	"""

	def __init__ (
		self,
		directory: pathlib.Path,
		known_sidecars: set[pathlib.Path],
		on_sample_loaded: typing.Callable[[subsample.library.SampleRecord], None],
		target_sample_rate: typing.Optional[int] = None,
		known_audio: typing.Optional[set[pathlib.Path]] = None,
	) -> None:

		self._directory = directory
		self._known_sidecars: frozenset[pathlib.Path] = frozenset(known_sidecars)
		self._on_sample_loaded = on_sample_loaded
		self._target_sample_rate = target_sample_rate

		# Known audio paths — audio files already loaded at startup.
		# Derived from known_sidecars by stripping the sidecar suffix when
		# no explicit set is provided.
		if known_audio is not None:
			self._known_audio: frozenset[pathlib.Path] = frozenset(known_audio)
		else:
			self._known_audio = frozenset(
				sc.parent / sc.name[: -len(_SIDECAR_SUFFIX)]
				for sc in self._known_sidecars
			)

		# Active debounce timers keyed by resolved path.
		# Protected by _lock — modified from the watchdog callback thread and
		# the timer threads.  Sidecar and audio timers share the same dict
		# and lock since their key spaces (sidecar paths vs audio paths) do
		# not overlap.
		self._timers: dict[pathlib.Path, threading.Timer] = {}
		self._lock = threading.Lock()

		handler = _InstrumentFileHandler(
			sidecar_callback=self._on_sidecar_event,
			audio_callback=self._on_audio_file_event,
		)

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

	# ------------------------------------------------------------------
	# Sidecar path — existing behaviour, unchanged
	# ------------------------------------------------------------------

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

		spectral, rhythm, pitch, timbre, params, duration, level, band_energy, channel_format = result

		# Derive audio file path: strip .analysis.json suffix to get the WAV name.
		audio_name = sidecar_path.name[: -len(_SIDECAR_SUFFIX)]
		audio_path = sidecar_path.parent / audio_name

		if not audio_path.exists():
			self._schedule_retry(
				sidecar_path, attempt,
				reason=f"WAV not yet present ({audio_name})",
			)
			return

		audio = subsample.library.load_wav_audio(audio_path, self._target_sample_rate)

		if audio is None:
			self._schedule_retry(
				sidecar_path, attempt,
				reason=f"could not read WAV ({audio_name})",
			)
			return

		record = subsample.library.SampleRecord(
			sample_id      = subsample.library.allocate_id(),
			name           = pathlib.Path(audio_name).stem,
			spectral       = spectral,
			rhythm         = rhythm,
			pitch          = pitch,
			timbre         = timbre,
			level          = level,
			band_energy    = band_energy,
			params         = params,
			duration       = duration,
			audio          = audio,
			filepath       = audio_path,
			channel_format = channel_format,
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

	# ------------------------------------------------------------------
	# Audio file path — detects files from non-subsample sources
	# ------------------------------------------------------------------

	def _on_audio_file_event (self, audio_path: pathlib.Path) -> None:

		"""Schedule a debounced check for a new audio file.

		Called from the watchdog observer thread when an audio file is created
		or modified.  Starts the debounce → grace → stability → analyze
		pipeline.
		"""

		path = audio_path.resolve()

		if path in self._known_audio:
			return

		with self._lock:

			existing = self._timers.pop(path, None)
			if existing is not None:
				existing.cancel()

			timer = threading.Timer(
				_AUDIO_DEBOUNCE_SECONDS,
				self._check_sidecar_then_load,
				args=(path,),
			)
			self._timers[path] = timer
			timer.start()

	def _check_sidecar_then_load (self, audio_path: pathlib.Path) -> None:

		"""After the initial debounce, check whether a sidecar already exists.

		If a sidecar is present, another subsample instance (or the sidecar
		watcher path) is handling this file — do nothing.  Otherwise, start
		the sidecar grace period to give a remote subsample instance time to
		deliver its sidecar before we analyze locally.
		"""

		with self._lock:
			self._timers.pop(audio_path, None)

		sidecar = subsample.cache.cache_path(audio_path)

		if sidecar.exists():
			_log.debug(
				"Watcher: sidecar already exists for %s — skipping audio path",
				audio_path.name,
			)
			return

		_log.debug(
			"Watcher: no sidecar for %s — waiting %.1fs grace period",
			audio_path.name, _SIDECAR_GRACE_SECONDS,
		)

		# Record current file size for the first stability check.
		try:
			initial_size = audio_path.stat().st_size
		except OSError:
			_log.debug("Watcher: %s disappeared before grace period", audio_path.name)
			return

		timer = threading.Timer(
			_SIDECAR_GRACE_SECONDS,
			self._attempt_audio_load,
			args=(audio_path, initial_size, 0),
		)

		with self._lock:
			self._timers[audio_path] = timer
			timer.start()

	def _attempt_audio_load (
		self,
		audio_path: pathlib.Path,
		prev_size: int,
		stability_checks: int,
	) -> None:

		"""Analyze and load an audio file that has no sidecar.

		Checks:
		1. Sidecar appeared during grace period → skip (sidecar path handles it).
		2. File size changed since last check → retry (file still being written).
		3. File stable → load audio, analyze, write sidecar, create record.
		"""

		with self._lock:
			self._timers.pop(audio_path, None)

		# 1. Sidecar appeared — another path is handling this file.
		sidecar = subsample.cache.cache_path(audio_path)

		if sidecar.exists():
			_log.debug(
				"Watcher: sidecar appeared for %s during grace — skipping",
				audio_path.name,
			)
			return

		# 2. File-size stability check.
		try:
			current_size = audio_path.stat().st_size
		except OSError:
			_log.debug("Watcher: %s disappeared before load", audio_path.name)
			return

		if current_size != prev_size:

			if stability_checks >= _STABILITY_MAX_CHECKS:
				_log.warning(
					"Watcher: %s still changing size after %d checks — giving up",
					audio_path.name, _STABILITY_MAX_CHECKS,
				)
				return

			_log.debug(
				"Watcher: %s size changed (%d → %d) — rechecking in %.1fs "
				"(check %d/%d)",
				audio_path.name, prev_size, current_size,
				_STABILITY_CHECK_SECONDS, stability_checks + 1,
				_STABILITY_MAX_CHECKS,
			)

			timer = threading.Timer(
				_STABILITY_CHECK_SECONDS,
				self._attempt_audio_load,
				args=(audio_path, current_size, stability_checks + 1),
			)

			with self._lock:
				self._timers[audio_path] = timer
				timer.start()

			return

		if current_size == 0:
			_log.debug("Watcher: %s is empty — skipping", audio_path.name)
			return

		# 3. File is stable — analyze and load.
		_log.info(
			"Watcher: new audio file %s (no sidecar) — analyzing",
			audio_path.name,
		)

		result = subsample.cache.load_or_analyze(audio_path)

		if result is None:
			_log.warning(
				"Watcher: could not analyze %s — skipping",
				audio_path.name,
			)
			return

		spectral, rhythm, pitch, timbre, params, duration, level, band_energy, channel_format = result

		audio = subsample.library.load_wav_audio(audio_path, self._target_sample_rate)

		if audio is None:
			_log.warning(
				"Watcher: could not read audio from %s — skipping",
				audio_path.name,
			)
			return

		record = subsample.library.SampleRecord(
			sample_id      = subsample.library.allocate_id(),
			name           = audio_path.stem,
			spectral       = spectral,
			rhythm         = rhythm,
			pitch          = pitch,
			timbre         = timbre,
			level          = level,
			band_energy    = band_energy,
			params         = params,
			duration       = duration,
			audio          = audio,
			filepath       = audio_path,
			channel_format = channel_format,
		)

		self._on_sample_loaded(record)


class _InstrumentFileHandler (watchdog.events.FileSystemEventHandler):

	"""Watchdog event handler that dispatches sidecar and audio file events."""

	def __init__ (
		self,
		sidecar_callback: typing.Callable[[pathlib.Path], None],
		audio_callback: typing.Callable[[pathlib.Path], None],
	) -> None:

		super().__init__()
		self._sidecar_callback = sidecar_callback
		self._audio_callback = audio_callback

	def on_created (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward file creation events."""

		self._dispatch(event)

	def on_modified (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward file modification events (e.g. cloud-sync overwrites)."""

		self._dispatch(event)

	def _dispatch (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Route the event to the appropriate callback based on file type."""

		if event.is_directory:
			return

		path = pathlib.Path(str(event.src_path))

		if path.name.endswith(_SIDECAR_SUFFIX):
			self._sidecar_callback(path)
			return

		suffix = path.suffix.lower()

		if suffix in _AUDIO_EXTENSIONS:
			self._audio_callback(path)


# ---------------------------------------------------------------------------
# MIDI map file watcher
# ---------------------------------------------------------------------------

_MIDI_MAP_DEBOUNCE_SECONDS: float = 0.5


class MidiMapWatcher:

	"""Watch a MIDI map YAML file for changes and invoke a callback on reload.

	Monitors the parent directory of the target file and filters events to
	the target filename only.  A short debounce window absorbs the multiple
	writes that text editors commonly produce during a single save operation.

	The callback receives the path to the changed file.  Parsing, validation,
	and delivery to the player are the caller's responsibility — this class
	handles only filesystem watching and debounce.
	"""

	def __init__ (
		self,
		path: pathlib.Path,
		on_changed: typing.Callable[[pathlib.Path], None],
	) -> None:

		self._path = path.resolve()
		self._on_changed = on_changed

		self._timer: typing.Optional[threading.Timer] = None
		self._lock = threading.Lock()

		handler = _MidiMapFileHandler(self._path.name, self._on_file_event)
		self._observer: typing.Any = watchdog.observers.Observer()
		self._observer.schedule(handler, str(self._path.parent), recursive=False)

	def start (self) -> None:

		"""Start the background observer thread."""

		self._observer.start()
		_log.info("MIDI map watcher started on %s", self._path)

	def stop (self) -> None:

		"""Stop the observer thread and cancel any pending debounce timer."""

		self._observer.stop()
		self._observer.join()

		with self._lock:
			if self._timer is not None:
				self._timer.cancel()
				self._timer = None

		_log.debug("MIDI map watcher stopped")

	def _on_file_event (self, path: pathlib.Path) -> None:

		"""Schedule (or reschedule) a debounced callback for the changed file.

		Called from the watchdog observer thread.  Multiple rapid events
		(common during editor save operations) collapse into a single
		callback after the debounce window.
		"""

		with self._lock:
			if self._timer is not None:
				self._timer.cancel()

			self._timer = threading.Timer(
				_MIDI_MAP_DEBOUNCE_SECONDS,
				self._fire,
			)
			self._timer.start()

	def _fire (self) -> None:

		"""Invoke the callback after the debounce window has elapsed."""

		with self._lock:
			self._timer = None

		self._on_changed(self._path)


class _MidiMapFileHandler (watchdog.events.FileSystemEventHandler):

	"""Watchdog event handler that filters for a specific filename.

	Handles on_modified, on_created (editors that delete + recreate), and
	on_moved (editors that write a temp file then rename into place).
	"""

	def __init__ (
		self,
		target_name: str,
		callback: typing.Callable[[pathlib.Path], None],
	) -> None:

		super().__init__()
		self._target_name = target_name
		self._callback = callback

	def on_modified (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward modification events for the target file."""

		self._dispatch_if_target(event)

	def on_created (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward creation events (editors that delete + recreate)."""

		self._dispatch_if_target(event)

	def on_moved (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Forward rename-into-place events (editors that write to a temp file)."""

		if event.is_directory:
			return

		dest = pathlib.Path(str(getattr(event, "dest_path", "")))

		if dest.name == self._target_name:
			self._callback(dest)

	def _dispatch_if_target (self, event: watchdog.events.FileSystemEvent) -> None:

		"""Call the callback if the event is for the target file."""

		if event.is_directory:
			return

		path = pathlib.Path(str(event.src_path))

		if path.name == self._target_name:
			self._callback(path)
