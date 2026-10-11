"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, the WAV writer, and the MIDI player. Supports two
input modes and two run modes:

  File input   — pass audio file paths as positional arguments, in any
                 format the library loads; each file is processed through
                 the detection pipeline and segments are written to
                 recorder.directory, in recorder.audio.audio_format, with
                 their analysis sidecars.  The process then exits — the
                 chopped segments are picked up at the next subsample
                 startup, when the recursive library load discovers them.

  Live capture — stream from an audio input device (recorder.enabled: true).

  MIDI player  — listen for MIDI input and play instrument samples
                 (player.enabled: true).

Recorder and player run as threads so they can operate concurrently.
The main thread handles KeyboardInterrupt and coordinates shutdown via
a shared threading.Event.

Press Ctrl+C to stop cleanly.
"""

import argparse
import dataclasses
import datetime
import gc
import importlib
import itertools
import logging
import math
import os
import pathlib
import re
import shutil
import signal
import sys
import threading
import time
import typing

import numpy
import yaml

import subsample.analysis
import subsample.audio
import subsample.bank
import subsample.cache
import subsample.buffer
import subsample.config
import subsample.detector
import subsample.events
import subsample.library
import subsample.loopfind
import subsample.osc
import subsample.parallelism
import subsample.performance
import subsample.player
import subsample.recorder
import subsample.similarity
import subsample.transform
import subsample.trim
import subsample.watcher


_log = logging.getLogger(__name__)

# Maps config bit_depth to the NumPy dtype used for sample storage.
# 24-bit audio is stored as int32 (left-shifted by 8) — see audio.unpack_audio().
_AUDIO_DTYPE: dict[int, numpy.dtype] = {
	16: numpy.dtype(numpy.int16),
	24: numpy.dtype(numpy.int32),
	32: numpy.dtype(numpy.int32),
}

# Cadence for in-loop progress logging during file ingestion.
_PROGRESS_LOG_INTERVAL_SEC: typing.Final[float] = 10.0

# How long the shutdown waits for the recorder to queue its last sounds.  Its
# loop wakes every half second and the queueing takes milliseconds, so this only
# bounds a recorder that fails on the way out.
_LAST_SOUNDS_WAIT_SEC: typing.Final[float] = 5.0


def _format_mmss (seconds: float) -> str:

	"""Format a non-negative duration in seconds as mm:ss (zero-padded).

	Minutes are not capped — a 100-minute duration renders as '100:00'.
	"""

	s = int(round(max(0.0, seconds)))
	return f"{s // 60:02d}:{s % 60:02d}"


_TOOL_COMMANDS: dict[str, tuple[str, str]] = {
	"import":  ("subsample.tools.import_samples",    "import pre-trimmed audio files into the sample library"),
	"catalog": ("subsample.tools.catalog_samples",   "CSV catalogue of every sample's detected properties, with curation aids"),
	"analyze": ("subsample.tools.analyze_file",      "analyse audio files and print their detected metrics"),
	"similar": ("subsample.tools.similarity_report", "rank library samples against each reference, or one sound, by similarity"),
	"loops":   ("subsample.tools.suggest_loops",     "find and audition click-free loop points in sustained samples"),
}
"""Subcommand name → (tool module, one-line description).  The first CLI
argument is checked against these BEFORE the run-mode parser, so `subsample
catalog ...` routes to the tool; a FILE that shares a command name needs a path
prefix (`subsample ./import`)."""


def _tool_epilog () -> str:

	"""Return the --help epilog listing the tool subcommands."""

	lines = ["commands:"]
	for name, (_module, description) in _TOOL_COMMANDS.items():
		lines.append(f"  {name:<10}{description}")
	lines.append("")
	lines.append("Run `subsample <command> --help` for a command's own options.")
	lines.append("A FILE named like a command needs a path prefix: `subsample ./import`.")
	return "\n".join(lines)


def parser () -> argparse.ArgumentParser:

	"""Build the parser for `subsample` itself, without parsing anything.

	Building is kept apart from parsing so that subsystem.co can read every
	option and its help without running Subsample, and generate the published
	command-line reference from them (#3020).  Each tool subcommand in
	_TOOL_COMMANDS has a parser() of its own, for the same reason.
	"""

	command = argparse.ArgumentParser(
		prog="subsample",
		# RawDescriptionHelpFormatter (for the epilog's command list) prints
		# this unwrapped, so the line break is written in.
		description=(
			"A sampler that cuts separate sounds out of a live input or a recording,\n"
			"sorts them by how they sound, and plays them from MIDI."
		),
		epilog=_tool_epilog(),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	command.add_argument(
		"files",
		nargs="*",
		type=pathlib.Path,
		metavar="FILE",
		help=(
			"Audio files to cut into samples, written to `recorder.directory` "
			"and named after the source file (e.g. recording_1.wav, "
			"recording_2.wav, …, or .flac files when "
			"`recorder.audio.audio_format` is `flac`). Cutting a file again "
			"replaces the samples it gave before. Samples of that name, in "
			"any letter case, cut from another file are kept, and this file's "
			"take its folder's name too (e.g. day2_recording_1.wav). Subsample "
			"exits once they are done, without starting live capture."
		),
	)
	command.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help=(
			"Path to config.yaml (default: ./config.yaml in the current "
			"directory, falling back to built-in defaults). Relative paths "
			"inside the config still resolve against the current directory, "
			"so one shared config can serve several project folders."
		),
	)
	command.add_argument(
		"--list-devices",
		action="store_true",
		help=(
			"List the audio input, audio output, and MIDI input devices on "
			"this machine, then exit. Any name shown can be used in "
			"config.yaml's device settings, whole or in part, with wildcards "
			"for a number that changes."
		),
	)
	command.add_argument(
		"--init",
		action="store_true",
		help=(
			"Create a complete starter project in the current directory - a "
			"fully commented config.yaml, the ready-to-play GM drum kit map, and "
			"an editable map template - then exit.  The GM reference fingerprints "
			"the kit matches against ship inside Subsample and are named, so "
			"nothing is copied. "
			"Refuses to overwrite existing files."
		),
	)
	return command


def _parse_args (argv: typing.Optional[list[str]] = None) -> argparse.Namespace:

	"""Parse command-line arguments.

	Args:
		argv: Argument list to parse; None (default) reads sys.argv.

	Returns:
		Namespace with 'files' (a possibly empty list of pathlib.Path),
		'config' (an explicit config path or None), and 'list_devices' and
		'init' (both bool).
	"""

	return parser().parse_args(argv)


def _list_devices () -> None:

	"""Print every audio input/output and MIDI input device on this machine.

	Same names and format as the interactive pickers, so a name seen here can
	be pasted (whole, or any substring) straight into recorder.audio.device,
	player.audio.device, or player.midi_device in config.yaml.  Where no MIDI
	system can be opened, it says why in place of the MIDI list.  Where no audio
	system can, it says why in one line and exits 1, as a config problem does:
	it was a traceback (#4703, M9).
	"""

	try:
		pa = subsample.audio.create_pyaudio()
	except OSError as exc:
		_log.error("%s", exc)
		raise SystemExit(1)

	try:
		inputs  = subsample.audio.list_input_devices(pa)
		outputs = subsample.audio.list_output_devices(pa)
	finally:
		pa.terminate()

	print("Audio inputs (recorder.audio.device):")
	for i, device in enumerate(inputs):
		print(f"  [{i}] {device['name']}  (default {int(device['defaultSampleRate'])} Hz)")
	if not inputs:
		print("  (none found)")

	print("Audio outputs (player.audio.device):")
	for i, device in enumerate(outputs):
		print(f"  [{i}] {device['name']}  (default {int(device['defaultSampleRate'])} Hz)")
	if not outputs:
		print("  (none found)")

	print("MIDI inputs (player.midi_device):")

	try:
		midi_names = subsample.player.list_midi_input_devices()
	except subsample.player.MidiUnavailableError as exc:
		# In place of the list, as "(none found)" is for audio, so the check
		# ends as cleanly as it does on a machine with no sound card.
		print(f"  {exc}")
		return

	for i, name in enumerate(midi_names):
		print(f"  [{i}] {name}")
	if not midi_names:
		print("  (none found)")


def _init_config () -> None:

	"""Scaffold a complete Subsample project in the current directory.

	Creates config.yaml (a copy of the documented defaults with the GM kit
	map pre-wired), the two shipped MIDI maps (the instant GM drum kit and an
	editable template), the samples directories, and a .gitignore for the
	regenerable variant cache.  The GM reference fingerprints the kit matches
	against are not copied: they ship with the package and the map names them.
	All-or-nothing: if ANY target file already exists, nothing is created — a
	project's tuned files can never be lost to a stray --init.
	"""

	data = subsample.config.data_dir()
	cwd = pathlib.Path.cwd()

	# Guard against scaffolding into a subsample SOURCE checkout — the repo is
	# the app, not a music project.  (A real project directory never contains
	# the subsample package itself.)
	if (cwd / "subsample" / "__init__.py").exists():
		_log.error(
			"This looks like the subsample source repository (it contains the "
			"subsample/ package) - run --init in your music project directory instead"
		)
		raise SystemExit(1)

	# The GM reference fingerprints are NOT copied.  They ship with the package
	# and the kit map names them (`reference: GM36_BassDrum1`) rather than
	# pointing at a path, so a scaffolded project needs no copy of them — and a
	# map written this way stays portable, which is what lets a sample set be
	# shared between projects or live on a drive shared between machines.
	# Point library.reference_directory at your own set to override them.
	copies = [
		(data / "midi-map-gm-drums.yaml", cwd / "midi-map-gm-drums.yaml"),
		(data / "midi-map.yaml.default", cwd / "midi-map.yaml"),
	]

	# .gitignore is deliberately NOT a conflict target: a real project may well
	# have one already (a GitHub-initialised repo does), and ignore files are
	# additive — the scaffold appends its line instead (below).
	targets = [cwd / "config.yaml"] + [target for _, target in copies]

	existing = [target for target in targets if target.exists()]
	if existing:
		shown = ", ".join(str(p.relative_to(cwd)) for p in existing[:5])
		if len(existing) > 5:
			shown += f" (+{len(existing) - 5} more)"
		_log.error(
			"Refusing to overwrite existing file(s): %s - nothing was created. "
			"Run --init in an empty directory, or move these aside first",
			shown,
		)
		raise SystemExit(1)

	(cwd / "samples" / "captures").mkdir(parents=True, exist_ok=True)

	# The scaffolded config is the documented defaults with exactly one edit:
	# the GM kit map wired in (player stays disabled — the user flips
	# player.enabled when ready to play). The count guard keeps this from ever
	# silently breaking if the default file's comment changes.
	wired = "midi_map: midi-map-gm-drums.yaml"
	default_text = subsample.config._locate_default_config().read_text(encoding="utf-8")
	if default_text.count("# " + wired) != 1:
		raise RuntimeError(
			"Bundled config.yaml.default no longer contains the expected "
			f"'# {wired}' line - cannot wire the starter map. Please report this bug."
		)
	(cwd / "config.yaml").write_text(
		default_text.replace("# " + wired, wired), encoding="utf-8",
	)

	for src, target in copies:
		shutil.copyfile(src, target)

	# Create the ignore file, or append our line to an existing one (exact-line
	# match, so a commented-out mention doesn't count as present).
	gitignore = cwd / ".gitignore"
	ignore_line = "samples/variant-cache/"

	if gitignore.exists():
		existing_text = gitignore.read_text(encoding="utf-8")
		if ignore_line not in existing_text.splitlines():
			separator = "" if existing_text.endswith("\n") or not existing_text else "\n"
			gitignore.write_text(
				existing_text + separator + ignore_line + "\n", encoding="utf-8",
			)
	else:
		gitignore.write_text(ignore_line + "\n", encoding="utf-8")

	n_refs = len(sorted((data / "reference").glob("*.analysis.json")))
	print(f"Created a Subsample project in {cwd}:")
	print("  config.yaml              every setting documented; the GM kit map is pre-wired")
	print("  midi-map-gm-drums.yaml   instant GM drum kit - plays your closest-matching samples")
	print("  midi-map.yaml            a commented template for building your own map")
	print("  samples/captures/        recordings land here, and the library loads from here")
	print()
	print(f"The {n_refs} GM reference fingerprints the kit matches against are built in - the")
	print("map names them, so nothing was copied and the map stays portable.  Set")
	print("library.reference_directory to match against your own references instead.")
	print()
	print("Next: connect a microphone and run `subsample` to start capturing sounds.")
	print("To play the kit, set player.enabled: true in config.yaml (plus your MIDI")
	print("and audio devices), then play MIDI channel 10.")


def _build_trimmed_segment (
	buf: subsample.buffer.CircularBuffer,
	detector: subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
	sample_rate: int,
	start_frame: int,
	end_frame: int,
) -> typing.Optional[numpy.ndarray]:

	"""Read [start_frame, end_frame] from the buffer and trim both edges.

	Shared by the per-chunk completion path and the end-of-file flush so both
	produce identically-trimmed segments — and, being the single choke point every
	captured segment passes through, the place the detection.min_peak_db floor is
	enforced.

	Returns None when the segment trims away to nothing, or when its peak falls
	below detection.min_peak_db.  Both callers already treat None as "no segment".
	"""

	pre = round(detection_cfg.trim_pre_ms / 1000.0 * sample_rate)
	# Read back `pre` extra frames before the detector's start boundary so
	# trim_silence has raw audio available for its fade-in window.
	segment = buf.read_range(max(0, start_frame - pre), end_frame)

	# Tail edge uses the CLOSE threshold (keeps a decayed tail); attack edge uses
	# the open threshold (trims tight to the transient) — both owned by the
	# detector so they cannot drift from what it ended on.
	fade_out_samples = round(detection_cfg.fade_out_ms / 1000.0 * sample_rate)

	trimmed = subsample.trim.trim_silence(
		segment,
		detector.tail_amplitude_threshold,
		pre_samples=pre,
		post_samples=round(detection_cfg.trim_post_ms / 1000.0 * sample_rate),
		fade_out_samples=fade_out_samples,
		lead_amplitude_threshold=detector.attack_amplitude_threshold,
	)

	if trimmed.size == 0:
		return None

	if detection_cfg.min_peak_db is not None:
		peak_dbfs = _peak_dbfs(trimmed)
		if peak_dbfs < detection_cfg.min_peak_db:
			_log.info(
				"Discarded segment: peak %.1f dBFS is below detection.min_peak_db "
				"(%.1f dBFS) - noise, not a hit",
				peak_dbfs, detection_cfg.min_peak_db,
			)
			return None

	return trimmed


def _peak_dbfs (audio: numpy.ndarray) -> float:

	"""Peak level of a captured segment, in dBFS.

	Full scale comes from the array's own dtype, so the value means the same thing
	for 16- and 24/32-bit capture — unlike a raw sample count, which would make any
	absolute threshold silently bit-depth dependent.  Digital silence returns -inf,
	which compares below every finite threshold.

	Capture is always integer PCM (see _AUDIO_DTYPE); a float dtype is treated as
	the conventional +/-1.0 full scale rather than raising, so this stays total.

	The abs() runs in float64 because int32 cannot represent abs(INT32_MIN), which
	appears in 24-bit left-shifted and native 32-bit audio.
	"""

	full_scale = (
		float(numpy.iinfo(audio.dtype).max)
		if numpy.issubdtype(audio.dtype, numpy.integer) else 1.0
	)
	peak = float(numpy.max(numpy.abs(audio.astype(numpy.float64))))

	if peak <= 0.0:
		return -math.inf

	return 20.0 * math.log10(peak / full_scale)


def _reference_directory (cfg: subsample.config.Config) -> pathlib.Path:

	"""Where reference fingerprints are loaded from — see subsample.config."""

	return subsample.config.reference_directory(cfg)


def _preload_midi_map (
	cfg:             subsample.config.Config,
	reference_names: list[str],
) -> typing.Optional[subsample.player.MidiMapResult]:

	"""Load the configured MIDI map early, so bank definitions are known.

	Returns None when there is nothing to load — no player, or no map named.
	Raises SystemExit(1), after logging a tracebackless reason, when a map IS
	named but cannot be loaded.

	Fatal, and fatal HERE.  An enabled player cannot run without its map, and
	this is the earliest point the failure is knowable — everything after it (the
	library load, which can take a minute over thousands of samples; the
	transform queue; the watchers) is wasted work.  Warning and carrying on used
	to defer the failure to MidiMapWatcher.start(), where it surfaced as a raw
	inotify FileNotFoundError naming neither the config key nor the path, long
	after the banner had printed that same bad path as though all were well.

	The reference names must be the REAL ones: a map naming a reference it cannot
	resolve has that assignment skipped with a warning, so pre-loading against an
	empty list would silently produce an empty note map — and _start_player reuses
	this result rather than parsing again.  The reference library is loaded before
	this runs, so the real names are available.

	Anything that raises here is a missing file, malformed YAML, or a schema error
	the later load would hit too.
	"""

	if not cfg.player.enabled:
		return None

	if cfg.player.midi_map is None and cfg.player.midi_maps is None:
		return None

	path = pathlib.Path(cfg.player.midi_map) if cfg.player.midi_map is not None else None

	try:
		return subsample.player.load_configured_map(cfg, reference_names)

	# OSError rather than FileNotFoundError alone: `player.midi_map: samples`
	# names a directory, a plausible mistake, and a map on a drive whose
	# permissions have changed is another.  Both reached the user as a raw
	# traceback, and on the watch-reload path as a traceback from a timer
	# thread instead of "keeping the current map".
	except (OSError, ValueError, yaml.YAMLError) as exc:
		_log.error("Cannot load the MIDI map: %s", exc)

		if path is not None and not path.exists():
			# The resolved path is the whole diagnosis for the commonest mistake:
			# copying the bundled `subsample/data/…` prefix onto a project-local
			# map, which then resolves under a package directory that is not there.
			_log.error(
				"player.midi_map = %r - looked for it at %s.  Relative paths "
				"resolve from the current directory, not from inside the package "
				"(only the maps that ship with Subsample live under subsample/data/).",
				cfg.player.midi_map, path.resolve(),
			)
		elif path is not None:
			_log.error("player.midi_map = %r (%s)", cfg.player.midi_map, path.resolve())

		raise SystemExit(1)


def _start_watcher (watcher: typing.Any, description: str) -> bool:

	"""Start a filesystem watcher, degrading to a warning if it cannot start.

	Watching is a convenience — hot-reloading edited maps and picking up new
	samples — so losing it must never stop the app from playing.  Left
	unguarded, watchdog raises straight out of the startup path: an exhausted
	inotify watch limit (``/proc/sys/fs/inotify/max_user_watches``, a common
	limit to hit on Linux with several directories watched) or a directory
	removed between the check and the call would kill a fully-loaded session
	with a traceback pointing into watchdog internals.

	Returns True when the watcher started and the caller should register it for
	shutdown.  A watcher that failed to start must NOT be registered: watchdog's
	stop()/join() on an observer whose thread never ran raises in turn, during
	teardown, where it is even less welcome.
	"""

	try:
		watcher.start()
	except OSError as exc:
		_log.warning(
			"Could not start the %s watcher (%s) - continuing without it.  "
			"Edits will not be picked up until restart.",
			description, exc,
		)
		return False

	return True


def _process_chunk (
	chunk: numpy.ndarray,
	buf: subsample.buffer.CircularBuffer,
	detector: subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
	sample_rate: int,
) -> typing.Optional[numpy.ndarray]:

	"""Feed one audio chunk through the detection pipeline.

	Writes the chunk to the circular buffer, runs the level detector, and
	if a recording is completed, retrieves and trims the segment.

	Args:
		chunk:         One audio chunk, shape (buffer_frames, channels).
		buf:           Circular buffer receiving the audio stream.
		detector:      Level detector tracking ambient noise and recording state.
		detection_cfg: Detection settings (SNR threshold, trim params, etc.).
		sample_rate:   Stream sample rate, used to convert fade_out_ms to samples.

	Returns:
		Trimmed audio segment as a numpy array, or None if no recording completed.
	"""

	buf.write(chunk)

	result = detector.process_chunk(chunk, buf.frames_written)

	if result is None:
		return None

	start_frame, end_frame = result
	return _build_trimmed_segment(buf, detector, detection_cfg, sample_rate, start_frame, end_frame)


def _queue_the_last_sounds (
	reader:        subsample.audio.AudioReader,
	buf:           subsample.buffer.CircularBuffer,
	detector:      subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
	sample_rate:   int,
	writer:        subsample.recorder.SampleProcessor,
	dropped:       "_DroppedInput",
) -> None:

	"""Cut and queue what a stopped input had already heard, and the sound still sounding when it stopped (#4703, M2).

	The live loop stopped at the shutdown flag and dropped both: the chunks
	still waiting to be read, and the recording still open, so a hit just
	before Ctrl+C was never saved.  File mode already kept a file's last
	sound.  The open one is cut where the input stopped, its tail faded as
	any capture's is.
	"""

	for chunk in reader.drain():
		dropped.before(detector)
		trimmed = _process_chunk(chunk, buf, detector, detection_cfg, sample_rate)
		missing = dropped.owed(detector, trimmed)

		if trimmed is not None:
			writer.enqueue(trimmed, datetime.datetime.now(), missing_seconds=missing)

	dropped.before(detector)
	last = detector.finalize(buf.frames_written)

	if last is None:
		return

	trimmed = _build_trimmed_segment(buf, detector, detection_cfg, sample_rate, last[0], last[1])

	if trimmed is not None:
		writer.enqueue(trimmed, datetime.datetime.now(), missing_seconds=dropped.owed(detector, trimmed))


class _DroppedInput:

	"""Input the recorder dropped for falling behind: said once each time it falls behind, and owed to the capture recorded across it.

	The input queue drops what the reader is too slow to take, so a sound
	recorded across the drop has a splice where it was.  It was neither
	counted nor said (#4704, L23).  Simon's call: the capture is kept, and
	the writer names it with how much it misses.
	"""

	def __init__ (self, reader: subsample.audio.AudioReader, sample_rate: int) -> None:

		"""Follow one reader's drops, at its sample rate."""

		self._reader      = reader
		self._sample_rate = sample_rate
		self._behind:  int = 0   # frames dropped since the recorder fell behind
		self._missing: int = 0   # frames the recording open now misses

	def before (self, detector: subsample.detector.LevelDetector) -> None:

		"""Take what was dropped before the chunk just read: the open recording misses it, and a stretch behind that has ended is said."""

		lost = self._reader.take_dropped()

		if lost:
			self._behind += lost

			if detector.state is subsample.detector.DetectorState.RECORDING:
				self._missing += lost

			return

		if self._behind:
			_log.warning(
				"Recorder fell behind and dropped %.2f s of input audio - the machine was too busy to keep up",
				self._behind / self._sample_rate,
			)
			self._behind = 0

	def owed (
		self,
		detector: subsample.detector.LevelDetector,
		trimmed:  typing.Optional[numpy.ndarray],
	) -> typing.Optional[float]:

		"""The seconds a capture just cut misses, or None; what a recording that was never cut missed is forgotten."""

		if trimmed is not None:
			missing, self._missing = self._missing, 0

			return missing / self._sample_rate if missing else None

		if detector.state is not subsample.detector.DetectorState.RECORDING:
			self._missing = 0

		return None


def _process_input_files (
	files: list[pathlib.Path],
	cfg: subsample.config.Config,
) -> int:

	"""Process audio files through the detection pipeline, writing segments to disk.

	Each file is read with its native sample rate, bit depth, and channel count.
	Detected segments are written to cfg.recorder.directory with names derived from
	the source filename (e.g. field_recording_1.wav, field_recording_2.wav, …).
	Files that cannot be read are skipped with a warning.

	Args:
		files: Paths to audio files to process.
		cfg:   Application config (detection settings, output directory).

	Returns:
		The number of files successfully read and processed — 0 means every input
		was missing or unreadable, which the caller surfaces as a non-zero exit.
	"""

	processed = 0

	for path in files:
		if not path.exists():
			_log.warning("Input file not found, skipping: %s", path)
			continue

		print(f"Processing {path.name}…")

		try:
			file_info = subsample.audio.read_audio_file(
				path, float_ceiling_dbfs=cfg.recorder.audio.float_import_ceiling_dbfs,
			)
		except (OSError, ValueError) as exc:
			_log.warning("Could not read %s: %s - skipping", path.name, exc)
			continue

		file_cfg, ambisonic_note = _file_mode_config(cfg, file_info.channels)
		print(f"  {file_info.sample_rate} Hz  {file_info.bit_depth}-bit  {file_info.channels}ch{ambisonic_note}")

		audio_dtype = _AUDIO_DTYPE.get(file_info.bit_depth)
		if audio_dtype is None:
			_log.warning(
				"Unsupported bit depth %d in %s - skipping",
				file_info.bit_depth, path.name,
			)
			continue

		max_frames = file_info.sample_rate * cfg.recorder.buffer.max_seconds
		buf = subsample.buffer.CircularBuffer(max_frames, file_info.channels, dtype=audio_dtype)

		detector = subsample.detector.LevelDetector(
			cfg.detection,
			file_info.sample_rate,
			cfg.recorder.audio.buffer_frames,
			max_recording_frames=max_frames,
		)

		analysis_params = subsample.analysis.compute_params(file_info.sample_rate)

		writer = subsample.recorder.SampleProcessor(file_cfg, analysis_params, on_complete=None, warn_backlog=False)

		# Its samples' names, and those an earlier cut of it left (M7).
		cut_from      = str(path.resolve())
		base, earlier = _segment_base(path, pathlib.Path(cfg.recorder.directory))

		segment_index = 1
		buffer_frames = cfg.recorder.audio.buffer_frames
		n_frames = file_info.audio.shape[0]
		file_duration_sec = n_frames / file_info.sample_rate

		_log.info("File duration: %s", _format_mmss(file_duration_sec))

		start_time = time.monotonic()
		last_progress_time = start_time

		# try/finally so an error in the read/drain loop still shuts down the
		# worker pool (this is the one SampleProcessor owner without the
		# teardown guarantee its siblings have).
		try:

			# Phase 1 — read pass: stream the file through the detector, dispatching
			# detected segments to the worker pool.  Fast for in-memory files
			# (essentially memory bandwidth + detector cost), so the periodic tick
			# below will only fire for very large files.
			for offset in range(0, n_frames, buffer_frames):
				chunk = file_info.audio[offset : offset + buffer_frames]

				trimmed = _process_chunk(chunk, buf, detector, cfg.detection, file_info.sample_rate)

				if trimmed is not None:
					writer.enqueue(
						trimmed,
						datetime.datetime.now(),
						filename_base=f"{base}_{segment_index}",
						sample_rate=file_info.sample_rate,
						bit_depth=file_info.bit_depth,
						cut_from=cut_from,
					)
					segment_index += 1

				now = time.monotonic()
				if now - last_progress_time >= _PROGRESS_LOG_INTERVAL_SEC:
					position_sec = min(file_duration_sec, (offset + buffer_frames) / file_info.sample_rate)
					pct = 100.0 * position_sec / file_duration_sec if file_duration_sec > 0 else 100.0
					wall_elapsed = now - start_time
					# Extrapolate remaining wall time from the ratio of audio read so far.
					if position_sec > 0:
						eta_wall = wall_elapsed * (file_duration_sec - position_sec) / position_sec
					else:
						eta_wall = 0.0
					_log.info(
						"Reading: %.0f%% - elapsed %s, ETA %s",
						pct, _format_mmss(wall_elapsed), _format_mmss(eta_wall),
					)
					last_progress_time = now

			# End of file: flush a recording still open (its last sound ran to
			# within hold_seconds of EOF) so the final segment is not silently lost.
			flush = detector.finalize(buf.frames_written)
			if flush is not None:
				trimmed = _build_trimmed_segment(
					buf, detector, cfg.detection, file_info.sample_rate, flush[0], flush[1],
				)
				if trimmed is not None:
					writer.enqueue(
						trimmed,
						datetime.datetime.now(),
						filename_base=f"{base}_{segment_index}",
						sample_rate=file_info.sample_rate,
						bit_depth=file_info.bit_depth,
						cut_from=cut_from,
					)
					segment_index += 1

			# Phase 2 — drain: wait for the worker pool to finish analysing and
			# writing every enqueued segment.  For long files this is where most
			# of the wall time lives, so we poll queue_depth and log progress.
			total_segments = segment_index - 1
			if total_segments > 0:
				drain_start = time.monotonic()
				while True:
					depth = writer.queue_depth
					if depth == 0:
						break

					now = time.monotonic()
					if now - last_progress_time >= _PROGRESS_LOG_INTERVAL_SEC:
						completed = total_segments - depth
						wall_elapsed = now - start_time
						drain_elapsed = now - drain_start
						if completed > 0:
							eta_wall = drain_elapsed * depth / completed
						else:
							eta_wall = 0.0
						pct = 100.0 * completed / total_segments
						_log.info(
							"Processing: %d/%d segments (%.1f%%) - elapsed %s, ETA %s",
							completed, total_segments, pct,
							_format_mmss(wall_elapsed), _format_mmss(eta_wall),
						)
						last_progress_time = now

					time.sleep(1.0)

			writer.flush()

			wall_total = time.monotonic() - start_time
			count      = segment_index - 1
			outcomes   = writer.outcomes
			saved      = outcomes.written + outcomes.unanalysed
			print(_cut_summary(count, saved, outcomes.unanalysed, path.name, wall_total))

			# A cut that saved nothing of what it found replaces nothing, and
			# is no success: every segment failing was "processed", exit 0.
			if count == 0 or saved:
				_remove_leftover_segments(earlier, count, path.name)
				processed += 1

		finally:
			writer.shutdown()

	return processed


def _file_mode_config (cfg: subsample.config.Config, channels: int) -> tuple[subsample.config.Config, str]:

	"""The config a file of this many channels is cut under, and what its line says of ambisonics (#4703, M6).

	recorder.audio.ambisonic_format describes the live rig's microphone, and
	was applied to every file: a stereo file failed every segment, and a
	4-channel multitrack file was converted as if it were ambisonic.  It
	applies to 4-channel files only, and the line says when it does.
	"""

	ambisonic_format = cfg.recorder.audio.ambisonic_format

	if ambisonic_format is None:
		return cfg, ""

	if channels == 4:
		return cfg, f", converted to B-format from {ambisonic_format} (recorder.audio.ambisonic_format)"

	audio = dataclasses.replace(cfg.recorder.audio, ambisonic_format=None)

	return (
		dataclasses.replace(cfg, recorder=dataclasses.replace(cfg.recorder, audio=audio)),
		", cut as it is: recorder.audio.ambisonic_format applies to 4-channel files",
	)


def _segment_base (source: pathlib.Path, directory: pathlib.Path) -> tuple[str, list[pathlib.Path]]:

	"""The name this file's samples take before their numbers, and the samples an earlier cut of it left (#4703, M7).

	Its own name, unless samples of that name were cut from another file, or
	note no file at all: then its folder's name before it, and a number after
	while that is taken too.  Two files called ZOOM0001.WAV, from two days'
	folders, wrote the same names, and the second replaced the first's
	samples.  A sample's sidecar notes the file it was cut from.  A name
	that differs only in letter case is the same name (_segments_named).
	"""

	cut_from = str(source.resolve())

	for attempt in itertools.count():
		if attempt == 0:
			base = source.stem
		elif attempt == 1:
			base = f"{source.parent.name}_{source.stem}"
		else:
			base = f"{source.parent.name}_{source.stem}-{attempt}"

		earlier = _segments_named(directory, base)

		if all(subsample.cache.cut_from(sample) == cut_from for sample in earlier):
			return base, earlier

	raise AssertionError("unreachable")


def _segments_named (directory: pathlib.Path, base: str) -> list[pathlib.Path]:

	"""The samples in a folder named as file mode names a cut's: base_1.wav, base_2.flac, and so on.

	In any letter case, on every system (#4912).  Where file names fold
	case, as on macOS, Kick_1.wav and kick_1.wav are one file: a cut of
	day2/kick.wav did not see day1/Kick.wav's samples, and wrote over them.
	Matched the same way everywhere, a library cut on Linux can be copied
	to such a system whole.
	"""

	if not directory.is_dir():
		return []

	pattern = re.compile(re.escape(base) + r"_(\d+)\.(?:wav|flac)", re.IGNORECASE)

	return sorted(sample for sample in directory.iterdir() if pattern.fullmatch(sample.name))


def _remove_leftover_segments (earlier: list[pathlib.Path], count: int, source_name: str) -> None:

	"""Remove the samples an earlier cut of this file left and this one did not replace (#4703, M7).

	One numbered past this cut's, or one this cut saved again in the other
	format.  A re-cut into fewer samples left the old extra ones in the
	library.  Each goes with its sidecar and preview.
	"""

	for sample in earlier:
		index = int(sample.stem.rsplit("_", 1)[1])
		other = sample.with_suffix(".flac" if sample.suffix == ".wav" else ".wav")

		if index <= count and not (other.exists() and other not in earlier):
			continue

		for leftover in (sample, subsample.cache.cache_path(sample), sample.with_name(sample.name + subsample.cache.PREVIEW_PNG_SUFFIX)):
			try:
				leftover.unlink(missing_ok=True)
			except OSError as exc:
				_log.warning("Could not remove %s: %s", leftover.name, exc)

		_log.info("Removed %s, left by an earlier cut of %s", sample.name, source_name)


def _cut_summary (count: int, saved: int, unanalysed: int, source_name: str, seconds: float) -> str:

	"""The line that ends a file's cut: how many segments were written, of how many found (#4703, M6)."""

	segments = f"{count} segment(s)" if saved == count else f"{saved} of {count} segment(s)"
	line     = f"  → {segments} written from {source_name} in {_format_mmss(seconds)}"

	if unanalysed:
		line += f", {unanalysed} without their analysis"

	return line


def _run_recorder (
	cfg: subsample.config.Config,
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	shutdown_event: threading.Event,
	store_audio: bool,
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]] = None,
	app_events: typing.Optional[subsample.events.EventEmitter] = None,
	processor_cell: typing.Optional[list[typing.Optional[subsample.recorder.SampleProcessor]]] = None,
	before_sidecar_write: typing.Optional[typing.Callable[[pathlib.Path], None]] = None,
	last_sounds_queued: typing.Optional[threading.Event] = None,
) -> None:

	"""Set up an audio input device and run the real-time capture loop.

	Streams audio from the configured device (or interactively selected device)
	into a circular buffer. Detected recordings are trimmed, queued for WAV
	output, analyzed, and added to the instrument library. Runs until
	shutdown_event is set.

	Args:
		cfg:                Full application config.
		reference_library:  Loaded reference samples, or None if not configured.
		instrument_library: Instrument sample library to update in real time.
		analysis_params:    Pre-computed FFT params matching cfg.recorder.audio.sample_rate.
		similarity_matrix:  Similarity index to update as new samples arrive, or None.
		shutdown_event:     Set this to stop the capture loop cleanly.
		store_audio:        When True, keep PCM data in SampleRecord for playback.
		transform_manager:  Optional transform pipeline; notified of new and evicted
		                    samples so derivative variants are kept in sync.
		player_cell:        Single-element list holding the active MidiPlayer, or None.
		                    Forwarded to _make_on_complete so pitched assignments are
		                    updated when the best match changes.
		app_events:         Optional event emitter; forwarded to _make_on_complete
		                    so sample_captured and sample_loaded events are emitted.
		processor_cell:     Single-element list this stores its SampleProcessor in, so
		                    the shutdown path can see how many captures are still being
		                    analysed and wait for them rather than killing them.  Set
		                    once the input is open.
		last_sounds_queued: Set once the loop has stopped and its last sounds are
		                    queued, which the shutdown path waits for before it
		                    counts what is still being analysed.
	"""

	try:
		pa = subsample.audio.create_pyaudio()
	except OSError as exc:
		print(f"Error opening audio device: {exc}", file=sys.stderr)
		return

	try:
		devices = subsample.audio.list_input_devices(pa)

		if cfg.recorder.audio.device is not None:
			try:
				device_index = subsample.audio.find_device_by_name(pa, cfg.recorder.audio.device)
			except ValueError:
				_log.warning(
					"Configured audio input device %r not found - prompting for selection",
					cfg.recorder.audio.device,
				)
				device_index = subsample.audio.select_device(devices)
		else:
			device_index = subsample.audio.select_device(devices)

		# Resolve channel count and input routing.
		audio_cfg = cfg.recorder.audio
		detected_channels = subsample.audio.get_device_channels(pa, device_index)

		if audio_cfg.input is not None:
			# Validate input indices against the device's actual channel count.
			for idx in audio_cfg.input:
				if idx >= detected_channels:
					raise ValueError(
						f"recorder.audio.input channel {idx + 1} exceeds device's "
						f"{detected_channels} input channel(s)"
					)

			if audio_cfg.channels is None:
				audio_cfg = dataclasses.replace(audio_cfg, channels=len(audio_cfg.input))

		elif audio_cfg.channels is None:
			# No explicit channels or input routing — auto-detect.
			if detected_channels >= 3:
				# Multi-channel device: prompt user to choose inputs.
				device_info = pa.get_device_info_by_index(device_index)
				selected = subsample.audio.select_input_channels(
					str(device_info["name"]), detected_channels,
				)
				audio_cfg = dataclasses.replace(
					audio_cfg,
					channels=len(selected),
					input=selected,
				)
			else:
				# 1-2 channel device: use all channels.
				audio_cfg = dataclasses.replace(audio_cfg, channels=detected_channels)
				_log.info("Auto-detected %d input channel(s) from device", detected_channels)

	except (ValueError, OSError) as exc:
		print(f"Error opening audio device: {exc}", file=sys.stderr)
		pa.terminate()
		return

	# By this point channels is always resolved (either explicit from config, or
	# auto-detected above).  The assert narrows the type for mypy.
	assert audio_cfg.channels is not None

	audio_dtype = _AUDIO_DTYPE[audio_cfg.bit_depth]
	max_frames = audio_cfg.sample_rate * cfg.recorder.buffer.max_seconds
	buf = subsample.buffer.CircularBuffer(max_frames, audio_cfg.channels, dtype=audio_dtype)

	detector = subsample.detector.LevelDetector(
		cfg.detection,
		audio_cfg.sample_rate,
		audio_cfg.buffer_frames,
		max_recording_frames=max_frames,
	)

	on_complete_callback = _make_on_complete(
		reference_library, instrument_library, analysis_params,
		similarity_matrix, store_audio, transform_manager,
		player_cell=player_cell,
		app_events=app_events,
	)

	# Before the input opens, since it makes recorder.directory: one that
	# could not be made stopped this thread with a traceback and the input
	# left open (#4703, M9).
	try:
		writer = subsample.recorder.SampleProcessor(
			cfg,
			analysis_params,
			on_complete=on_complete_callback,
			reserve_for_player=cfg.player.enabled,
			before_sidecar_write=before_sidecar_write,
		)
	except OSError as exc:
		print(f"Error creating recorder.directory: {exc}", file=sys.stderr)
		pa.terminate()
		return

	try:
		reader = subsample.audio.AudioReader(pa, device_index, audio_cfg)
	except (ValueError, OSError) as exc:
		print(f"Error opening audio device: {exc}", file=sys.stderr)
		writer.shutdown()
		pa.terminate()
		return

	# Hand it to the shutdown path, which waits for whatever is still being
	# analysed: a capture's audio file is written only after its analysis, so a
	# queue killed on the way out is a recording that never existed.
	if processor_cell is not None:
		processor_cell[0] = writer

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	dropped = _DroppedInput(reader, audio_cfg.sample_rate)

	try:
		while not shutdown_event.is_set():
			chunk = reader.read(timeout=0.5)

			if chunk is None:
				# Timeout — loop back to check shutdown_event.
				continue

			dropped.before(detector)
			trimmed = _process_chunk(chunk, buf, detector, cfg.detection, audio_cfg.sample_rate)
			missing = dropped.owed(detector, trimmed)

			if trimmed is not None:
				writer.enqueue(trimmed, datetime.datetime.now(), missing_seconds=missing)

	finally:
		if reader.overflow_count > 0:
			_log.warning(
				"Audio overflows detected during capture: %d - "
				"recordings may contain discontinuities",
				reader.overflow_count,
			)

		if reader.dropped_frames > 0:
			_log.warning(
				"Input audio dropped during capture: %.2f s in all, while the recorder fell behind",
				reader.dropped_frames / audio_cfg.sample_rate,
			)

		# Isolate each teardown step: if reader.stop() raises (e.g. PortAudio
		# error on an unplugged USB device), pa.terminate() and the writer's
		# drain/flush must still run rather than leak the stream and lose
		# in-flight recordings.
		try:
			reader.stop()
		except Exception as exc:
			_log.error("Recorder: reader.stop() failed during shutdown: %s", exc)

		try:
			pa.terminate()
		except Exception as exc:
			_log.error("Recorder: pa.terminate() failed during shutdown: %s", exc)

		try:
			_queue_the_last_sounds(reader, buf, detector, cfg.detection, audio_cfg.sample_rate, writer, dropped)
		except Exception as exc:
			_log.error("Recorder: the last sounds before stopping could not be saved: %s", exc, exc_info=True)

		if last_sounds_queued is not None:
			last_sounds_queued.set()

		writer.shutdown()


def _output_sample_rate (cfg: subsample.config.Config) -> int:

	"""The rate the player's stream runs at: player.audio.sample_rate, or the recorder's when unset.

	Samples are resampled to it as they load, variants are rendered at it, and
	the banner reports it, so each reads it from here rather than working it out
	again (L-24 of the 2026-09-21 review found four copies).
	"""

	if cfg.player.audio.sample_rate is not None:
		return cfg.player.audio.sample_rate

	return cfg.recorder.audio.sample_rate


def _variant_disk_cache (
	cfg:                subsample.config.Config,
	output_sample_rate: int,
) -> typing.Optional[subsample.transform.VariantDiskCache]:

	"""The session's variant disk cache, or None when it is turned off; every program's pipeline shares it.

	One cache for the one folder (#4703, M21): built per program, each kept
	its own budget and trimmed as if alone, so the folder grew to about one
	budget per program.
	"""

	if not cfg.transform.variant_cache_dir or cfg.transform.max_disk_mb <= 0:
		return None

	_log.info(
		"Variant disk cache: %s (max %.0f MB)",
		cfg.transform.variant_cache_dir, cfg.transform.max_disk_mb,
	)

	return subsample.transform.VariantDiskCache(
		directory=pathlib.Path(cfg.transform.variant_cache_dir),
		max_bytes=int(cfg.transform.max_disk_mb * 1024 * 1024),
		sample_rate=output_sample_rate,
	)


def _build_transform_manager (
	cfg:                subsample.config.Config,
	instrument_library: subsample.library.InstrumentLibrary,
	output_sample_rate: int,
	variant_disk_cache: typing.Optional[subsample.transform.VariantDiskCache],
	program:            typing.Optional[str] = None,
) -> subsample.transform.TransformManager:

	"""Build the transform pipeline for one library: its memory cache, its render queue and its manager.

	The workers are running once this returns, so the manager's shutdown is
	registered as a teardown, for a Ctrl+C during the rest of startup.  A
	program's name, when given, labels its idle log line.  The disk cache
	is the session's (_variant_disk_cache), shared by every program.
	"""

	transform_cache = subsample.transform.TransformCache(
		max_memory_bytes=int(cfg.transform.max_memory_mb * 1024 * 1024),
	)
	label = f" [{program}]" if program is not None else ""

	def _on_transform_complete (result: subsample.transform.TransformResult) -> None:

		"""Keep a finished render in the memory cache, unless its sample has left the library.

		Renders are asked for only through the manager, so it is built before
		the first one finishes.
		"""

		transform_manager.keep(result)

	def _on_transform_idle (completed: int) -> None:

		"""Log the cache's size each time the render queue empties."""

		_log.info(
			"Transform queue idle%s - %d variant(s) processed  [cache: %s]",
			label, completed, transform_cache.format_memory(),
		)

	transform_processor = subsample.transform.TransformProcessor(
		sample_rate=cfg.recorder.audio.sample_rate,
		output_sample_rate=output_sample_rate,
		on_complete=_on_transform_complete,
		on_idle=_on_transform_idle,
		disk_cache=variant_disk_cache,
	)

	transform_manager = subsample.transform.TransformManager(
		cache=transform_cache,
		processor=transform_processor,
		instrument_library=instrument_library,
		cfg=cfg.transform,
		disk_cache=variant_disk_cache,
	)

	_register_teardown(lambda: transform_manager.shutdown(drop_waiting=True))

	return transform_manager


def _warn_of_programs_left_out (banks: list[subsample.bank.Bank], max_memory_mb: float) -> None:

	"""Name each program whose samples did not all fit in library.max_memory_mb, and how many it left out.

	A program too large for the limit leaves out the samples it loaded first,
	and nothing loads them again, so it cannot play them.  The check this
	replaces compared the memory used with the limit, which leaving them out
	keeps it under, so it never fired, and it promised a reload on switch
	that does not exist (#4704, L20).
	"""

	for bank in banks:
		left_out = bank.instrument_library.left_out

		if left_out > 0:
			_log.warning(
				"Program %r: %d sample(s) did not fit in library.max_memory_mb "
				"(%.0f MB) and were left out, so it cannot play them - raise "
				"the limit to keep them",
				bank.name, left_out, max_memory_mb,
			)


@dataclasses.dataclass(frozen=True)
class _TopLevelPool:

	"""One pool of samples the top-level map's assignments play from (#4909).

	The library when the map has no programs, or one ``directory:``
	program's, with the matrix and transform manager that go with it.
	``program`` is the program's name, or None without programs.
	"""

	program:    typing.Optional[str]
	library:    subsample.library.InstrumentLibrary
	matrix:     typing.Optional[subsample.similarity.SimilarityMatrix]
	transforms: typing.Optional[subsample.transform.TransformManager]


def _top_level_pools (
	bank_manager:       typing.Optional[subsample.bank.BankManager],
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix:  typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager:  typing.Optional[subsample.transform.TransformManager],
) -> list[_TopLevelPool]:

	"""The pools the top-level map's assignments play from (#4909).

	With no programs, the one library.  With programs, each ``directory:``
	program's: a Program Change to one plays the same assignments from its
	own samples, so each needs what those assignments name by path, folder
	or reference.  A ``map:`` preset plays its own assignments, and loaded
	what they name as it was built (_load_bank), so it is not one.

	The map's samples used to load into the first program listed and no
	other, whichever kind it was.  Every other ``directory:`` program then
	found nothing for such an assignment, and a preset listed first held
	samples its own selects could choose and its map never named.
	"""

	if bank_manager is None:
		return [_TopLevelPool(None, instrument_library, similarity_matrix, transform_manager)]

	return [
		_TopLevelPool(bank.name, bank.instrument_library, bank.similarity_matrix, bank.transform_manager)
		for bank in bank_manager.all_banks()
		if bank.note_map is None
	]


def _load_map_paths (
	note_map:           subsample.player.NoteMap,
	zone_templates:     tuple[typing.Any, ...],
	pools:              list[_TopLevelPool],
	every:              list[subsample.player.PathLoads],
	reference_library:  subsample.library.ReferenceLibrary,
	target_sample_rate: int,
	with_preview:       bool,
) -> None:

	"""Load what the top-level map names into each pool that plays it, each sample held once (#4909).

	``every`` takes what each pool gained and lost, in the pools' order.  It
	is the caller's, so what was loaded before a failure is still known.
	"""

	shared: dict[pathlib.Path, subsample.library.SampleRecord] = {}

	for pool, loads in zip(pools, every):
		subsample.player._resolve_path_references(
			note_map, [pool.matrix] if pool.matrix is not None else [], pool.library,
			target_sample_rate=target_sample_rate,
			with_preview=with_preview,
			reference_library=reference_library,
			transform_manager=pool.transforms,
			zone_templates=zone_templates,
			loads=loads,
			shared=shared,
		)


def _extract_checked_library (
	pools:              list[_TopLevelPool],
	instrument_library: subsample.library.InstrumentLibrary,
) -> subsample.library.InstrumentLibrary:

	"""The library the top-level map's extracts are checked against as it loads.

	The primary library, as before, where it plays the top-level
	assignments; where it is a ``map:`` preset's, the first pool that does.
	A sample in another program that cannot give an extract is left out of
	the assignment as that program is ranked (MidiPlayer._playable_by).
	"""

	if not pools or any(pool.library is instrument_library for pool in pools):
		return instrument_library

	return pools[0].library


def _warn_of_samples_pushed_out (
	pools:         list[_TopLevelPool],
	every:         list[subsample.player.PathLoads],
	max_memory_mb: float,
) -> None:

	"""Name each program that lost samples of its own to those the map names (#4909)."""

	for pool, loads in zip(pools, every):
		if pool.program is not None and loads.evicted:
			_log.warning(
				"Program %r: %d sample(s) were left out to make room for those "
				"the map names, at library.max_memory_mb (%.0f MB), so it cannot "
				"play them - raise the limit to keep them",
				pool.program, len(loads.evicted), max_memory_mb,
			)


def _load_bank (
	defn: subsample.bank.BankDefinition,
	preset: typing.Optional[subsample.player.MidiMapResult],
	reference_library: subsample.library.ReferenceLibrary,
	cfg: subsample.config.Config,
	output_sample_rate: int,
	parent_map_dir: pathlib.Path,
	variant_disk_cache: typing.Optional[subsample.transform.VariantDiskCache],
) -> subsample.bank.Bank:

	"""Load a single program: library, similarity matrix, and transform pipeline.

	Two forms (mutually exclusive on the BankDefinition):

	  - ``directory:`` shorthand — load the directory as the sample pool;
	    the program reuses the player's top-level assignments.  The returned
	    Bank carries ``note_map=None`` (use the global rules).
	  - ``map:`` preset — a whole mapper file (its own assignments +
	    samples), whose rules were loaded with the map.  The library is
	    populated from the preset's own path / ``directory:`` predicates
	    (resolved relative to the preset folder), and the returned Bank
	    carries the preset's note_map / zone_templates / mapped_ccs so a
	    Program Change swaps the rules too.

	Args:
		defn:               Parsed program definition from the MIDI map.
		preset:             The rules a ``map:`` program brings, as
		                    player.load_configured_map loaded them; None for
		                    the ``directory:`` form.
		reference_library:  Shared reference library for similarity scoring.
		cfg:                Full application config (memory limits, transform settings).
		output_sample_rate: Effective player output sample rate for variant resampling.
		parent_map_dir:     Directory of the top-level MIDI map, which a
		                    relative ``map:`` preset path is resolved from.
		variant_disk_cache: The session's variant disk cache, which every
		                    program shares, or None.

	Returns:
		Fully loaded Bank ready for playback.

	Raises:
		ValueError: If an assignment's ``extract:`` does not suit the samples
		            a preset loads, or a preset filters by ``duration_beats``
		            with no session tempo set.
	"""

	max_instrument_bytes = int(cfg.library.max_memory_mb * 1024 * 1024)

	if defn.map_path is not None:
		# `map:` preset — load_configured_map loaded its rules and refused a
		# broken one; the library starts empty and is filled by the preset's
		# own path / directory references below.
		assert preset is not None

		# A preset that filters by duration_beats needs a session tempo, as the
		# top-level map does (_start_player).  It was checked only when a
		# Program Change chose it, so the run started, and the switch was
		# undone mid-performance (#4913).
		try:
			subsample.player._validate_beat_filter_tempo(
				preset.note_map, preset.zone_templates, cfg.tempo.bpm,
			)
		except ValueError as exc:
			raise ValueError(f"Program {defn.name!r}, preset {defn.map_path!r}: {exc}") from exc

		directory = (parent_map_dir / defn.map_path).parent
		instrument_library = subsample.library.InstrumentLibrary(max_instrument_bytes)
	else:
		# `directory:` shorthand — the directory IS the pool.
		directory = pathlib.Path(typing.cast(str, defn.directory))
		instrument_library = subsample.library.load_instrument_library(
			directory,
			max_instrument_bytes,
			load_audio=True,
			with_preview=cfg.recorder.previews,
			target_sample_rate=output_sample_rate,
			reference_directory=subsample.config.reference_directory(cfg),
		)

	# Similarity matrix (per-bank — rankings are relative to each bank's samples).
	# Constructed with NO references preloaded, on purpose: every reference the
	# matrix holds costs a score against every instrument sample added later, so
	# seeding the whole reference library would make a map naming one reference
	# pay for all of them.  _resolve_path_references adds exactly the ones this
	# map names, path-based and named alike.
	similarity_matrix = subsample.similarity.SimilarityMatrix(
		subsample.library.ReferenceLibrary([]), cfg.similarity,
	)
	if len(instrument_library) > 0:
		similarity_matrix.bulk_add(instrument_library.samples())

	# Transform pipeline (per-bank).  One queue and memory cache per bank, each
	# running from here, so each gets its own teardown for a Ctrl+C during
	# startup; the disk cache is the session's, shared by every bank.
	transform_manager = _build_transform_manager(
		cfg, instrument_library, output_sample_rate, variant_disk_cache, defn.name,
	)

	# For a `map:` preset, populate the (empty) library + similarity matrix
	# from the preset's own path / directory references — these resolve
	# relative to the preset folder (load_midi_map stamped the preset's
	# midi_map_dir into each directory predicate), so a self-contained kit
	# folder loads with no extra coupling.  Then validate its extracts.
	if preset is not None:
		subsample.player._resolve_path_references(
			preset.note_map, [similarity_matrix], instrument_library,
			target_sample_rate=output_sample_rate,
			with_preview=cfg.recorder.previews,
			reference_library=reference_library,
			zone_templates=preset.zone_templates,
		)
		subsample.player._validate_assignment_extracts(preset.note_map, instrument_library)

		# A preset with no assignments was reported when its rules loaded.
		if (preset.note_map or preset.zone_templates) and len(instrument_library) == 0:
			_log.warning(
				"Program %r preset %s loaded no samples - check its 'directory:' predicates",
				defn.name, defn.map_path,
			)

	# Auto-enqueue variants for loaded samples.
	if len(instrument_library) > 0:
		for record in instrument_library.samples():
			transform_manager.on_sample_added(record)

	_log.info(
		"Program %r loaded: %d sample(s) from %s  [%s]",
		defn.name, len(instrument_library), directory,
		instrument_library.format_memory(),
	)

	preset_zone_templates: typing.Optional[tuple[typing.Any, ...]] = None
	preset_mapped_ccs:     typing.Optional[set[int]]               = None
	if preset is not None:
		preset_zone_templates = preset.zone_templates
		preset_mapped_ccs     = subsample.player._collect_mapped_ccs(
			preset.note_map, preset.zone_templates,
		)

	return subsample.bank.Bank(
		name=defn.name,
		directory=directory,
		program=defn.program,
		instrument_library=instrument_library,
		similarity_matrix=similarity_matrix,
		transform_manager=transform_manager,
		note_map=preset.note_map if preset is not None else None,
		zone_templates=preset_zone_templates,
		mapped_ccs=preset_mapped_ccs,
	)


def _apply_active_preset_rules (
	player: subsample.player.MidiPlayer,
	bank_manager: typing.Optional[subsample.bank.BankManager],
) -> None:

	"""Install the active program's preset rules at startup, if it carries any.

	The player is constructed with the top-level note_map / zone_templates,
	but the BankManager's active program (e.g. a ``map:`` default_program) may
	carry its own.  Swap them in before playback so the default program's
	rules are live from the first note rather than only after the first
	Program Change.
	"""

	if bank_manager is None:
		return

	active = bank_manager.active_bank
	if active.note_map is None:
		return

	player._apply_rule_set(active.note_map, active.zone_templates or (), active.mapped_ccs or set())


def _warn_of_broken_presets (
	result:          subsample.player.MidiMapResult,
	map_dir:         pathlib.Path,
	reference_names: list[str],
	strict:          bool,
) -> None:

	"""Warn of each ``map:`` preset a restart would stop on, as the map reloads.

	A run keeps the presets it started with, and a preset's own file is not
	watched, so a broken one must not hold up an edit to the top-level map:
	the reload goes ahead, and the preset is named now rather than at the next
	start (#3886).
	"""

	for defn in result.bank_definitions:
		if defn.map_path is None:
			continue

		try:
			subsample.player.load_preset_map(defn, map_dir, reference_names, strict)
		except (OSError, ValueError, yaml.YAMLError) as exc:
			_log.warning(
				"MIDI map reload: a restart would stop on this preset, so fix "
				"it first: %s", exc,
			)


def _start_player (
	cfg: subsample.config.Config,
	shutdown_event: threading.Event,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: subsample.similarity.SimilarityMatrix,
	reference_library: subsample.library.ReferenceLibrary,
	player_cell: list[typing.Optional[subsample.player.MidiPlayer]],
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	bank_manager: typing.Optional[subsample.bank.BankManager] = None,
	preloaded_midi_map_result: typing.Optional[subsample.player.MidiMapResult] = None,
	once_playing: typing.Optional["_ReloadOncePlaying"] = None,
) -> None:

	"""Select a MIDI input device (or create a virtual port), then run the player.

	When cfg.player.virtual_midi_port is set, Subsample creates a named virtual
	MIDI input port and skips hardware device selection entirely. Otherwise, it
	resolves a hardware device from config (whole name, then glob) or prompts the user
	interactively. Runs until shutdown_event is set.

	Args:
		cfg:                Full application config.
		shutdown_event:     Set this to stop the player cleanly.
		instrument_library: Loaded instrument samples (must have audio in memory).
		similarity_matrix:  Similarity index for note → sample lookup.
		reference_library:  Reference library; provides sorted names for note mapping.
		player_cell:        Single-element list; _start_player stores the MidiPlayer
		                    here before calling run() so the on_complete callback can
		                    call update_pitched_assignments() when the best match changes.
		transform_manager:  Optional transform pipeline; enables pitched variant
		                    playback when provided.
		bank_manager:       Optional bank manager for multi-bank switching via MIDI
		                    Program Change.
	"""

	# Load the MIDI routing map.  Requires an explicit path in config —
	# no hidden fallback.  A new user must set player.midi_map to get output.
	if cfg.player.midi_map is None and cfg.player.midi_maps is None:
		print(
			"Player enabled but no MIDI map configured - "
			"set player.midi_map in config.yaml "
			"(e.g. midi_map: \"./midi-map-gm-drums.yaml\"; "
			"`subsample --init` scaffolds a project with that map wired in), "
			"or player.midi_maps to bind several sample sets to channels.",
			file=sys.stderr,
		)
		_log.warning("neither player.midi_map nor player.midi_maps is set - player will not start")
		return

	# Reuse the parse done at startup for bank detection: the parser is pure and
	# it saw the same reference names, so a second parse would only duplicate the
	# work and its INFO logs.
	if preloaded_midi_map_result is not None:
		midi_map_result = preloaded_midi_map_result
	else:
		try:
			midi_map_result = subsample.player.load_configured_map(cfg, reference_library.names())
		except (OSError, ValueError, yaml.YAMLError) as exc:
			# yaml.YAMLError (a plain indentation typo in a hand-edited map) is
			# the commonest live-coding failure and is NOT a ValueError — catch
			# it here too so the player thread reports one clean line instead of
			# dying with a traceback while the recorder keeps running.
			print(f"Error loading MIDI map: {exc}", file=sys.stderr)
			return

	midi_map = midi_map_result.note_map

	# What the map names by path, folder or reference goes to every pool its
	# assignments play from: the library, or each `directory:` program's.
	pools = _top_level_pools(bank_manager, instrument_library, similarity_matrix, transform_manager)
	every = [subsample.player.PathLoads() for _pool in pools]

	effective_output_sr = _output_sample_rate(cfg)

	# Resolve path references and run the post-parse validators inside one
	# guard: each can raise a ValueError (or a file OSError) that would
	# otherwise escape this thread target as a raw traceback and leave the
	# player dead while the rest of the app runs on.  One clean line + return.
	try:
		_load_map_paths(
			midi_map, midi_map_result.zone_templates, pools, every,
			reference_library, effective_output_sr, cfg.recorder.previews,
		)
		_warn_of_samples_pushed_out(pools, every, cfg.library.max_memory_mb)

		# Validate `extract:` directives now that all candidate samples are
		# loaded: any assignment whose extract is incompatible with even one of
		# its matching samples is rejected here, before audio playback begins.
		subsample.player._validate_assignment_extracts(
			midi_map, _extract_checked_library(pools, instrument_library),
		)

		# duration_beats measures sample length in beats, so a map that uses it
		# needs a session tempo to resolve — reject at startup (like the extract
		# check above) rather than silently emptying the pool at the first note.
		subsample.player._validate_beat_filter_tempo(
			midi_map, midi_map_result.zone_templates, cfg.tempo.bpm,
		)
	except (FileNotFoundError, ValueError, OSError) as exc:
		print(f"Error loading MIDI map: {exc}", file=sys.stderr)
		return

	# Virtual port mode: bypass hardware device selection entirely.
	# MidiPlayer.run() will open the named virtual port with virtual=True.
	if cfg.player.virtual_midi_port is not None:
		print(f"  MIDI input   : virtual port \"{cfg.player.virtual_midi_port}\"")
		device_name = ""

	# Hardware port mode: resolve device name from config or interactive menu.
	else:
		try:
			devices = subsample.player.list_midi_input_devices()

			if cfg.player.midi_device is not None:
				try:
					device_name = subsample.player.find_midi_device_by_name(cfg.player.midi_device)
				except ValueError:
					_log.warning(
						"Configured MIDI device %r not found - prompting for selection",
						cfg.player.midi_device,
					)
					device_name = subsample.player.select_midi_device(devices)
			else:
				device_name = subsample.player.select_midi_device(devices)

		except subsample.player.MidiUnavailableError as exc:
			# Worded as a virtual port's failure to open is worded below.
			print(f"\nError starting player: {exc}", file=sys.stderr)
			return

		except ValueError as exc:
			print(f"Error opening MIDI device: {exc}", file=sys.stderr)
			return

	# What on this machine can cost the player its timing, said once here
	# (#4659).  A CPU set to save power is named again if the audio drops out.
	power_saving = subsample.performance.power_saving()

	if power_saving is not None:
		_log.warning("%s", power_saving)

	programs = len(bank_manager.all_banks()) if bank_manager is not None else 1

	for warning in subsample.performance.budget_warnings(
		cfg.library.max_memory_mb, cfg.transform.max_memory_mb, programs,
		cfg.transform.variant_cache_dir, cfg.transform.max_disk_mb,
	):
		_log.warning("%s", warning)

	player = subsample.player.MidiPlayer(
		device_name,
		shutdown_event,
		instrument_library=instrument_library,
		similarity_matrix=similarity_matrix,
		midi_map=midi_map,
		sample_rate=cfg.recorder.audio.sample_rate,
		bit_depth=cfg.recorder.audio.bit_depth,
		output_device_name=cfg.player.audio.device,
		output_bit_depth=cfg.player.audio.bit_depth,
		output_sample_rate=cfg.player.audio.sample_rate,
		transform_manager=transform_manager,
		virtual_midi_port=cfg.player.virtual_midi_port,
		max_polyphony=cfg.player.max_polyphony,
		limiter_threshold_db=cfg.player.limiter_threshold_db,
		limiter_ceiling_db=cfg.player.limiter_ceiling_db,
		bank_manager=bank_manager,
		target_bpm=cfg.tempo.bpm,
		tempo_source=cfg.tempo.source,
		output_channels=cfg.player.audio.channels,
		ambisonic_config=cfg.ambisonic,
		buffer_frames=cfg.player.audio.buffer_frames,
		zone_templates=midi_map_result.zone_templates,
		cpu_saving_power=power_saving is not None,
	)
	player_cell[0] = player

	# A map saved while the player started is reloaded now (#4703, M11).
	if once_playing is not None:
		once_playing.published()

	try:
		# update_pitched_assignments materialises zones and rebuilds the
		# candidate cache; a map error deferred past parse (e.g. `order:
		# similarity` with no reference) surfaces here, so it must sit
		# inside the guard alongside run().
		_apply_active_preset_rules(player, bank_manager)
		player.update_pitched_assignments()
		player.run()
	except (ValueError, OSError) as exc:
		# OSError covers PortAudio rejecting the device/format and mido
		# failing to bind the port — catch it so a device-open failure prints
		# a clean message instead of escaping this thread target as a raw
		# traceback.  Returning ends the thread; the main loop notices all
		# subsystem threads have stopped and exits (see the wait loop) rather
		# than parking on shutdown_event forever.  The player is unpublished,
		# so nothing goes on sending it notes (#4704, L7).
		player_cell[0] = None
		print(f"\nError starting player: {exc}", file=sys.stderr)


# Teardown for subsystems that are already running when startup is interrupted.
# Ctrl+C during the library scan — documented as possibly taking a minute — or
# during the bank or reference load lands before the main loop's own handler
# exists, so it used to print a traceback and leave transform workers running:
# the interpreter then waited on their pool at exit, with nothing on screen
# saying why.  Registered as each is started, run once on the way out.
_STARTED: list[typing.Callable[[], None]] = []


def _register_teardown (stop: typing.Callable[[], None]) -> None:

	"""Remember something that must be stopped if startup is interrupted."""

	_STARTED.append(stop)


def _run_teardowns () -> None:

	"""Stop everything that started, reporting rather than raising."""

	while _STARTED:
		stop = _STARTED.pop()

		try:
			stop()

		except Exception as exc:
			_log.warning("Shutdown step failed: %s", exc)


def main () -> None:

	"""Entry point — tool subcommands first, else the ambient audio sampler."""

	argv = sys.argv[1:]

	if argv and argv[0] in _TOOL_COMMANDS:
		# Lazy import: only the requested tool's module is loaded, and a tool
		# ImportError surfaces as-is rather than being masked as run-mode.
		module = importlib.import_module(_TOOL_COMMANDS[argv[0]][0])
		try:
			rc = module.main(argv[1:])
			# Flush INSIDE the guard: small output fits the pipe buffer, so the
			# write during main() succeeds and BrokenPipe would otherwise surface
			# only at interpreter-shutdown flush ("Exception ignored…" + exit 120).
			# Forcing the flush here raises it where the handler can catch it.
			sys.stdout.flush()
			raise SystemExit(rc)
		except BrokenPipeError:
			# Downstream closed the pipe early (`subsample catalog | head`) —
			# normal pipeline behaviour, not an error.  Redirect stdout to
			# devnull so interpreter shutdown doesn't raise a second time.
			os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
			raise SystemExit(0)
		except KeyboardInterrupt:
			# The tools run long batches (`subsample import` over a sample pack,
			# `subsample catalog` over an un-analysed library), so Ctrl+C here is
			# routine.  Report it the way the main app does instead of unwinding a
			# concurrent.futures traceback into the user's terminal.
			print("\nInterrupted.", file=sys.stderr)
			raise SystemExit(130)

	previous_handlers = _stop_on_signals()

	try:
		_main_impl()

	except KeyboardInterrupt:
		# Startup was interrupted before the main loop's handler existed.  The
		# loads it lands in are the long ones, so this is a normal thing to do.
		# The teardowns wait for renders in flight; another Ctrl+C stops at once.
		_say("\nInterrupted.", sys.stderr)
		previous_handlers.setdefault(signal.SIGINT, _answer_further_ctrl_c(None))
		_run_teardowns()
		raise SystemExit(130) from None

	finally:
		for number, handler in previous_handlers.items():
			signal.signal(number, handler)


def _stop_on_signals () -> dict[int, typing.Any]:

	"""Make SIGTERM and SIGHUP stop Subsample as Ctrl+C does, and return the handlers they had.

	Ctrl+C finishes the captures still being analysed before it exits.  SIGTERM
	(kill, systemctl stop, docker stop) and SIGHUP (a closed terminal or SSH
	session) ended the process at once, losing them (#4702).  Under nohup,
	which ignores SIGHUP, it stays ignored and Subsample keeps running, as
	nohup means.
	"""

	previous: dict[int, typing.Any] = {}

	for number in (signal.SIGTERM, signal.SIGHUP):
		handler = signal.getsignal(number)

		if number == signal.SIGHUP and handler is signal.SIG_IGN:
			continue

		previous[number] = handler
		signal.signal(number, _stop_like_ctrl_c)

	return previous


def _stop_like_ctrl_c (signal_number: int, frame: typing.Any) -> None:

	"""Signal handler: stop as Ctrl+C stops, through the same KeyboardInterrupt."""

	raise KeyboardInterrupt


def _say (text: str, stream: typing.Optional[typing.TextIO] = None) -> None:

	"""Print a line of the shutdown's progress, even where nothing can read it any more.

	After a hang-up the terminal is gone, and writing to it fails.  An error
	here cut the shutdown short and lost the captures it was finishing
	(#4702), so from the first failed line on, output goes nowhere instead.
	"""

	try:
		print(text, file=sys.stdout if stream is None else stream, flush=True)
	except OSError:
		_silence_output()


def _silence_output () -> None:

	"""Send standard output and error nowhere, so later writes to a terminal that has gone succeed."""

	devnull = os.open(os.devnull, os.O_WRONLY)

	for stream in (sys.stdout, sys.stderr):
		try:
			os.dup2(devnull, stream.fileno())
		except (OSError, ValueError):
			pass

	os.close(devnull)


def _main_impl () -> None:

	"""Run the ambient audio sampler.

	Given input files on the command line, cuts them into samples and exits.
	Otherwise loads libraries and starts the recorder and/or player as
	configured.  Both run as threads; the main thread coordinates shutdown on
	Ctrl+C.
	"""

	logging.basicConfig(
		level=logging.WARNING,
		format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
		datefmt="%H:%M:%S",
	)
	logging.getLogger("subsample").setLevel(logging.INFO)

	args = _parse_args()

	if args.init:
		_init_config()
		return

	if args.list_devices:
		_list_devices()
		return

	try:
		cfg = subsample.config.load_config(args.config)
	except (OSError, ValueError) as exc:
		# A config problem — mistyped --config path, a directory or unreadable
		# file at that path (IsADirectoryError/PermissionError, both OSError),
		# malformed YAML, or a renamed key — should read as one clear line, not
		# a traceback.
		_log.error("%s", exc)
		raise SystemExit(1)

	# Wire the carrier cache budget from the resolved config.
	subsample.transform.set_carrier_cache_budget(
		int(cfg.transform.carrier_memory_mb * 1024 * 1024)
	)

	# Wire the float import ceiling so every read path — library, cache, watcher,
	# OSC import — treats a hot float source the same way, not just CLI import.
	subsample.audio.set_float_import_ceiling(
		cfg.recorder.audio.float_import_ceiling_dbfs
	)

	# Wire the analysis tempo priors so the file/library/watcher heal path honours
	# analysis.start_bpm / tempo_min / tempo_max, not just live capture.
	subsample.cache.set_analysis_config(cfg.analysis)

	# Wire the default quantize grid so transform.quantize_resolution actually
	# takes effect (it was documented but previously never read).
	subsample.player.set_default_quantize_grid(cfg.transform.quantize_resolution)

	# Pin BLAS to one thread process-wide.  Subsample parallelises across whole
	# samples, so a multi-threaded math backend only oversubscribes the CPU and
	# can jitter the real-time audio thread.  The background analysis workers
	# reaffirm this in their own processes; setting it here also protects the
	# player's audio callback.
	subsample.parallelism.cap_blas_threads()

	_log.info(
		"Memory budget: instrument %.0f MB, transform %.0f MB, carrier %.0f MB, disk %.0f MB",
		cfg.library.max_memory_mb, cfg.transform.max_memory_mb,
		cfg.transform.carrier_memory_mb, cfg.transform.max_disk_mb,
	)

	# --- Application event emitter ---
	# Integrations (such as the OSC sender) subscribe here
	# instead of being manually wired into each callback chain.
	app_events = subsample.events.EventEmitter()

	if cfg.osc.enabled:
		_start_osc_sender(cfg, app_events)

	# Reference fingerprints a map can name rather than point at.  Defaults to
	# the set bundled with the package, so `reference: GM46_OpenHiHat` resolves
	# anywhere Subsample is installed — which is what lets a sample set on a
	# shared drive use references at all, since a path-based one would have to
	# reach into some particular project's tree.  Path-based references still
	# work and are loaded later by _resolve_path_references() during player
	# startup, which adds them to the similarity matrices dynamically.
	reference_library = subsample.library.load_reference_library(
		_reference_directory(cfg),
	)

	# Process any input files through the detection pipeline, then exit.
	# Segments are written to the output directory and analysed; the recursive
	# library loader will pick them up on the next subsample startup.  This is
	# deliberate — file-input mode is a batch chop, not a hybrid live session.
	if args.files:
		# Catch a global flag placed BEFORE the subcommand (git-style muscle
		# memory): `subsample --config x.yaml catalog` parses `catalog` as an
		# input FILE, not the tool, and would silently "process" it as a missing
		# file and exit 0.  A positional that exactly names a tool and is not an
		# actual file is almost certainly a misordered subcommand.
		for f in args.files:
			if str(f) in _TOOL_COMMANDS and not f.exists():
				_log.error(
					"%r is a subcommand, not an input file - run it first, with its "
					"own flags after it: `subsample %s --config ...`.",
					str(f), str(f),
				)
				raise SystemExit(2)

		# Recorder banner shows the live-capture config; file-input mode reads
		# at each source's native format.  Clarify so a 32-bit float source
		# isn't mistaken for "24-bit" because that's what the config says.
		print("File input mode: reading each file at its native sample rate, bit depth, channel count.")

		# File-input mode returns before the thread wait loop that owns the app's
		# KeyboardInterrupt handling, so Ctrl+C here surfaced as a raw traceback.
		# It is the mode most likely to be interrupted — a long field recording
		# can take minutes — so give it the same clean exit the live app gets.
		try:
			processed = _process_input_files(args.files, cfg)
		except KeyboardInterrupt:
			print("\nInterrupted.", file=sys.stderr)
			raise SystemExit(130) from None

		if processed == 0:
			# Every input was missing or unreadable — exit non-zero so a script
			# or pipeline sees the failure instead of a misleading success.
			_log.error("No input files could be processed.")
			raise SystemExit(1)
		return

	# --- Bank detection ---
	# Pre-load the MIDI map to check for bank definitions before loading
	# instrument libraries.  Banks override cfg.library.directory.
	bank_manager: typing.Optional[subsample.bank.BankManager] = None
	bank_definitions: list[subsample.bank.BankDefinition] = []
	bank_channel: int = subsample.bank.DEFAULT_BANK_CHANNEL
	default_bank: typing.Optional[int] = None
	preloaded_midi_map_result: typing.Optional[subsample.player.MidiMapResult] = None

	preloaded_midi_map_result = _preload_midi_map(cfg, reference_library.names())

	# What the map's files were when it was read, so a save during the long
	# loads below is not missed (#4703, M11).
	map_read_as = _modification_times(preloaded_midi_map_result.source_files) if preloaded_midi_map_result is not None else {}
	once_playing = _ReloadOncePlaying()

	if preloaded_midi_map_result is not None:
		bank_definitions = preloaded_midi_map_result.bank_definitions
		bank_channel = preloaded_midi_map_result.bank_channel
		default_bank = preloaded_midi_map_result.default_bank

	# Now that bank detection has run, print the startup summary (multi-bank mode
	# reports per-program pools rather than the ignored library.directory).
	_print_banner(cfg, multi_bank=bool(bank_definitions))

	output_sample_rate = _output_sample_rate(cfg)

	# Declare shared variables before the bank/single-directory branch.
	max_instrument_bytes = int(cfg.library.max_memory_mb * 1024 * 1024)
	instrument_library:  subsample.library.InstrumentLibrary
	similarity_matrix:   typing.Optional[subsample.similarity.SimilarityMatrix] = None
	transform_manager:   typing.Optional[subsample.transform.TransformManager] = None

	# --- Multi-bank loading ---
	# When the MIDI map declares banks, load each one independently.
	# cfg.library.directory is ignored (banks take precedence).
	if bank_definitions:
		_log.info(
			"MIDI map declares %d program(s) - ignoring library.directory (%s)",
			len(bank_definitions), cfg.library.directory,
		)

		parent_map_dir = pathlib.Path(typing.cast(str, cfg.player.midi_map)).parent

		# The programs came from the pre-loaded map, which loaded every `map:`
		# program's rules with it, so a broken preset has already stopped the
		# run, before any library was loaded.
		assert preloaded_midi_map_result is not None
		presets = preloaded_midi_map_result.presets

		banks: list[subsample.bank.Bank] = []
		variant_disk_cache = _variant_disk_cache(cfg, output_sample_rate)
		try:
			for defn in bank_definitions:
				bank = _load_bank(
					defn, presets.get(defn.program), reference_library, cfg,
					output_sample_rate, parent_map_dir, variant_disk_cache,
				)
				banks.append(bank)
				source = defn.map_path if defn.map_path is not None else defn.directory
				print(
					f"  Program {defn.program:<3d}  : {defn.name!r} - "
					f"{len(bank.instrument_library)} sample(s) from {source}"
				)
		except (OSError, ValueError, yaml.YAMLError) as exc:
			# A program whose samples cannot be read, or do not suit its
			# assignments' extracts, must read as one clean line, not a raw
			# traceback out of the main thread.
			_log.error("Could not load program bank: %s", exc)
			raise SystemExit(1)

		bank_manager = subsample.bank.BankManager(banks, bank_channel, default_program=default_bank)

		# Eager-load memory guards (WARN only — never blocks startup).
		_warn_of_programs_left_out(banks, cfg.library.max_memory_mb)

		# Each program's library is capped independently, so N programs can
		# use up to N× the configured budget in aggregate.
		total_used = sum(b.instrument_library.memory_used for b in banks)
		if total_used > max_instrument_bytes:
			_log.warning(
				"Programs use %.0f MB of audio in total - more than the "
				"library.max_memory_mb budget (%.0f MB); eager-loading every "
				"program multiplies the budget by the program count",
				total_used / (1024 * 1024), cfg.library.max_memory_mb,
			)

		# The primary instrument_library/similarity/transform used by the
		# recorder on_complete callback come from the first program. Captures
		# directed at its directory are also picked up by that program's
		# watcher (see below).
		instrument_library  = banks[0].instrument_library
		similarity_matrix   = banks[0].similarity_matrix
		transform_manager   = banks[0].transform_manager

		print(
			f"  Programs     : {len(banks)} loaded - "
			f"switch via Program Change on ch {bank_channel}"
		)

	# --- Single-directory mode (no banks) ---
	else:
		# Create instrument library. PCM audio is only needed when the player is
		# active — skipping it saves memory when player is disabled.
		#
		# A null library.directory means "load nothing in bulk": every sample
		# then arrives through the MIDI map's own `directory:`/`path:` predicates
		# during player startup (_resolve_path_references).  That is what a
		# project made of shared sample sets wants — the sets say what they need,
		# so there is no reason to walk a whole capture tree that may hold
		# thousands of unrelated samples.
		if cfg.player.enabled and cfg.library.directory is not None:
			instrument_library = subsample.library.load_instrument_library(
				pathlib.Path(cfg.library.directory),
				max_instrument_bytes,
				load_audio=True,
				with_preview=cfg.recorder.previews,
				target_sample_rate=output_sample_rate,
				reference_directory=subsample.config.reference_directory(cfg),
			)
			print(
				f"  Instruments  : {len(instrument_library)} sample(s) loaded"
				f" from {cfg.library.directory}"
			)
			_log.info(
				"Instrument library: %d sample(s)  [%s]",
				len(instrument_library), instrument_library.format_memory(),
			)
		else:
			instrument_library = subsample.library.InstrumentLibrary(max_instrument_bytes)

			if cfg.player.enabled:
				# Say so explicitly.  An empty library here is a configuration
				# choice, but an empty library at the END of startup means every
				# note is silent, and "0 samples" with no explanation is the
				# hardest kind of silence to diagnose.
				print("  Instruments  : library.directory is null - loading only what the MIDI map names")
				_log.info(
					"library.directory is null: the instrument library starts empty and is "
					"populated from the MIDI map's directory:/path: predicates",
				)

		# Build the similarity matrix.  It starts with NO references; the ones the
		# MIDI map actually uses — path-based or named — are added during player
		# startup via _resolve_path_references().  See the note at the bank
		# construction site for why preloading the whole reference library would
		# be a real cost rather than a convenience.
		if cfg.player.enabled:
			similarity_matrix = subsample.similarity.SimilarityMatrix(
				subsample.library.ReferenceLibrary([]), cfg.similarity,
			)
			if len(instrument_library) > 0:
				similarity_matrix.bulk_add(instrument_library.samples())
			print(f"  Similarity   : {similarity_matrix}")

		# --- Transform pipeline ---
		if cfg.player.enabled:
			transform_manager = _build_transform_manager(
				cfg, instrument_library, output_sample_rate, _variant_disk_cache(cfg, output_sample_rate),
			)

			if len(instrument_library) > 0:
				for _record in instrument_library.samples():
					transform_manager.on_sample_added(_record)

	analysis_params = subsample.analysis.compute_params(cfg.recorder.audio.sample_rate)

	# --- Thread-based orchestration ---
	# Both the recorder and player have blocking loops, so each runs on its own
	# thread. The main thread waits on shutdown_event and forwards Ctrl+C.

	shutdown_event = threading.Event()
	threads: list[threading.Thread] = []

	# Shared cell so the on_complete callback can call update_assignments
	# when the best-matching sample changes for a pitched keyboard assignment.
	# _start_player sets this before calling player.run().
	_player_cell: list[typing.Optional[subsample.player.MidiPlayer]] = [None]
	_processor_cell: list[typing.Optional[subsample.recorder.SampleProcessor]] = [None]
	last_sounds_queued = threading.Event()
	# Declared before the subsystem threads because the recorder is handed the
	# helper below, which reads this list when a capture is written — by which
	# time the watchers further down have been added to it.
	instrument_watchers: list[subsample.watcher.InstrumentWatcher] = []

	def _capture_sidecar_is_ours (sidecar: pathlib.Path) -> None:

		"""Tell every watcher a capture's sidecar is ours, before it is written.

		recorder.directory and library.directory are the same by default, so
		with library.watch on, the sidecar landing in the watched directory
		looked like somebody else's file: the capture was integrated a second
		time, its first record evicted by path de-duplication, its variants
		re-baked and sample_loaded fired twice.
		"""

		for watcher in instrument_watchers:
			watcher.note_self_written(sidecar)

	# Subsystem threads are daemons: an interactive device-selection prompt
	# blocks in input(), which Ctrl+C (delivered to the main thread) cannot
	# interrupt.  Without daemon status the interpreter would wait on that
	# stuck thread forever after the main thread prints "Done." and returns.
	# On a clean shutdown the join below still lets a well-behaved thread
	# flush and exit normally; daemon status only matters when a thread is
	# genuinely stuck, where killing it at process exit is exactly right.
	if cfg.recorder.enabled:
		threads.append(threading.Thread(
			target=_run_recorder,
			args=(
				cfg, reference_library, instrument_library,
				analysis_params, similarity_matrix,
				shutdown_event, cfg.player.enabled,
				transform_manager, _player_cell,
				app_events, _processor_cell, _capture_sidecar_is_ours,
				last_sounds_queued,
			),
			name="recorder",
			daemon=True,
		))

	if cfg.player.enabled:
		if similarity_matrix is None:
			print(
				"Player enabled but similarity matrix could not be created.",
				file=sys.stderr,
			)
		else:
			threads.append(threading.Thread(
				target=_start_player,
				args=(
					cfg, shutdown_event, instrument_library,
					similarity_matrix, reference_library, _player_cell,
					transform_manager, bank_manager,
					preloaded_midi_map_result, once_playing,
				),
				name="player",
				daemon=True,
			))

	if not threads:
		print("Neither recorder nor player is enabled. Nothing to do.")
		return

	# --- Directory watchers ---
	# Start after the instrument library and player are configured so the
	# on_watched_sample callback can reference all live subsystems.
	# Only active when player is enabled — the watcher's purpose is to feed
	# new samples into the playback pipeline.

	instrument_watchers.extend(_start_library_watchers(
		cfg, bank_manager, instrument_library, similarity_matrix,
		transform_manager, _player_cell, app_events, output_sample_rate,
	))

	# --- MIDI map file watcher ---
	# Monitors the files the MIDI map is read from, so assignments can be
	# reloaded without restarting — enables live-coding of sample routing.
	midi_map_watcher: typing.Optional[subsample.watcher.MidiMapWatcher] = None

	if (
		cfg.player.watch_midi_map
		and cfg.player.enabled
		and preloaded_midi_map_result is not None
	):
		midi_map_watcher = _start_midi_map_watcher(
			cfg, preloaded_midi_map_result, reference_library, instrument_library,
			similarity_matrix, transform_manager, _player_cell, once_playing, map_read_as,
			bank_manager,
		)

	# --- OSC receivers ---
	# One listens for /sample/import messages and triggers file import; the
	# other plays /note/on and /note/off through the player (#3610, #603).
	osc_receivers: list[typing.Any] = []

	if cfg.osc.enabled and cfg.osc.receive_enabled:
		osc_receivers.append(_start_osc_receiver(
			cfg, instrument_library, similarity_matrix, transform_manager,
			_player_cell, app_events, output_sample_rate, shutdown_event,
		))

	if cfg.osc.enabled and cfg.osc.notes_enabled:
		osc_receivers.append(_start_osc_note_receiver(cfg, _player_cell))

	_start_threads(threads)

	try:
		startup_failed = _wait_for_shutdown(shutdown_event, threads)

		_shut_down(
			threads, _processor_cell[0], last_sounds_queued, instrument_watchers, midi_map_watcher,
			[receiver for receiver in osc_receivers if receiver is not None],
			bank_manager, transform_manager, startup_failed,
		)

	finally:
		# For a process that outlives this run, however its shutdown ended:
		# the tests run many in one.
		gc.unfreeze()
		subsample.audio.release_portaudio()


def _start_threads (threads: list[threading.Thread]) -> None:

	"""Freeze what start-up loaded out of the garbage collector's reach, start PortAudio, then start the recorder and the player.

	The player does the same just before its audio opens (MidiPlayer.run,
	#4679), and its collection then walks only what it loaded itself (#4702).
	Walking everything held Python's lock for 30 to 110 ms while the
	recorder's input was already running.  _main_impl unfreezes once they
	have stopped.
	"""

	gc.collect()
	gc.freeze()

	# PortAudio started here, before the threads that use it, so their own
	# starts hide no stderr (#4704, L27).  After the library scans, which fork
	# their workers only while no native thread is running.
	if threads:
		subsample.audio.hold_portaudio()

	for thread in threads:
		thread.start()


def _wait_for_shutdown (shutdown_event: threading.Event, threads: list[threading.Thread]) -> bool:

	"""Block until asked to stop, or until every subsystem thread has stopped by itself.

	Returns True when the subsystems all stopped without anyone asking, which
	means one failed at startup.
	"""

	startup_failed = False

	try:
		# Block the main thread without spinning. Event.wait() releases the GIL
		# and responds to KeyboardInterrupt between intervals.
		while not shutdown_event.is_set():
			shutdown_event.wait(timeout=1.0)

			# If every subsystem thread has exited on its own — e.g. the only
			# enabled subsystem failed to start (bad map, missing device) and
			# returned without ever entering its run loop — there is nothing left
			# to keep the process alive, so stop waiting instead of parking
			# forever.  A still-alive sibling (both enabled, one failed) keeps the
			# app running.  `threads` holds only the recorder/player subsystems;
			# an empty list (watcher/OSC-only mode) never triggers this.
			if threads and not any(t.is_alive() for t in threads):
				if not shutdown_event.is_set():
					# Nobody asked to stop — a subsystem failed at startup.
					# Exit non-zero so scripts/CI see the failure.
					_log.error("All subsystems have stopped - exiting.")
					startup_failed = True
				break

	except KeyboardInterrupt:
		_say("\nStopping…")
		shutdown_event.set()

	return startup_failed


def _shut_down (
	threads:             list[threading.Thread],
	processor:           typing.Optional[subsample.recorder.SampleProcessor],
	last_sounds_queued:  typing.Optional[threading.Event],
	instrument_watchers: list[subsample.watcher.InstrumentWatcher],
	midi_map_watcher:    typing.Optional[subsample.watcher.MidiMapWatcher],
	osc_receivers:       list[typing.Any],
	bank_manager:        typing.Optional[subsample.bank.BankManager],
	transform_manager:   typing.Optional[subsample.transform.TransformManager],
	startup_failed:      bool,
) -> None:

	"""Stop every subsystem in order, finishing the captures still being analysed first, then exit.

	Raises SystemExit(1) after a startup failure, and hard-exits when a
	subsystem thread is stuck in a device prompt; otherwise returns.  A
	Ctrl+C while it works stops it harder (_answer_further_ctrl_c).
	"""

	previous_ctrl_c = _answer_further_ctrl_c(processor)

	try:
		_stop_everything(
			threads, processor, last_sounds_queued, instrument_watchers, midi_map_watcher,
			osc_receivers, bank_manager, transform_manager, startup_failed,
		)

	finally:
		signal.signal(signal.SIGINT, previous_ctrl_c)


def _answer_further_ctrl_c (processor: typing.Optional[subsample.recorder.SampleProcessor]) -> typing.Any:

	"""Make each Ctrl+C after the first stop harder, rather than raise again, and return the handler it had (#4703, M5).

	The next press saves the captures still being analysed without their
	analysis, which is made when Subsample next loads them, and then the
	shutdown carries on; with none left, it stops at once.  The press after
	that stops at once.  Raised again, each press unwound whatever was
	waiting: a second waited all over again, and a third printed hundreds of
	lines of tracebacks.
	"""

	presses = 0

	def on_ctrl_c (signal_number: int, frame: typing.Any) -> None:

		"""Signal handler: save the waiting captures unanalysed, or stop at once."""

		nonlocal presses
		presses += 1
		waiting = processor.queue_depth if processor is not None else 0

		if presses == 1 and processor is not None and waiting:
			_say(f"Saving {waiting} capture(s) without their analysis, which is made when Subsample next loads them…")
			processor.abandon()
			return

		_say("\nInterrupted.", sys.stderr)

		try:
			sys.stdout.flush()
			sys.stderr.flush()
		except OSError:
			pass

		os._exit(130)

	previous = signal.getsignal(signal.SIGINT)
	signal.signal(signal.SIGINT, on_ctrl_c)

	return previous


def _stop_everything (
	threads:             list[threading.Thread],
	processor:           typing.Optional[subsample.recorder.SampleProcessor],
	last_sounds_queued:  typing.Optional[threading.Event],
	instrument_watchers: list[subsample.watcher.InstrumentWatcher],
	midi_map_watcher:    typing.Optional[subsample.watcher.MidiMapWatcher],
	osc_receivers:       list[typing.Any],
	bank_manager:        typing.Optional[subsample.bank.BankManager],
	transform_manager:   typing.Optional[subsample.transform.TransformManager],
	startup_failed:      bool,
) -> None:

	"""_shut_down's steps, in order."""

	# The recorder queues its last sounds as it stops (#4703, M2): let it, so
	# the drain below counts them, rather than racing its last pass.
	if processor is not None and last_sounds_queued is not None:
		last_sounds_queued.wait(timeout=_LAST_SOUNDS_WAIT_SEC)

	# Wait for captures still being analysed BEFORE joining with a timeout.
	# A capture's audio file is written only after its analysis, its preview and
	# its loop search have run, and pyin alone costs seconds per sample — so a
	# backlog takes longer than the join timeout below, and the hard exit that
	# followed threw those recordings away with no message and exit code 0.
	_drain_captures(processor)

	for t in threads:
		t.join(timeout=10.0)

	for iw in instrument_watchers:
		iw.stop()

	if midi_map_watcher is not None:
		midi_map_watcher.stop()

	for osc_receiver in osc_receivers:
		osc_receiver.stop()

	# Stop the renders: those waiting are dropped, and those the workers hold
	# finish (#4703, M5).
	if bank_manager is not None:
		for bank in bank_manager.all_banks():
			if bank.transform_manager is not None:
				bank.transform_manager.shutdown(drop_waiting=True)
	elif transform_manager is not None:
		transform_manager.shutdown(drop_waiting=True)

	# Every program's renders are done, so the worker processes they shared
	# can stop (#4667).
	subsample.parallelism.shutdown_shared_pools()

	if not startup_failed:
		_say("Done.")

	# A subsystem thread still alive here is wedged in an uninterruptible
	# input() prompt (device selection) that Ctrl+C — delivered only to the
	# main thread — could not reach.  The threads are daemons, so the process
	# CAN exit; but a normal return would let the interpreter abort with a
	# fatal buffered-stdin error while finalizing the wedged daemon.  Flush
	# and hard-exit cleanly instead (all other subsystems are already stopped).
	if any(t.is_alive() for t in threads):
		try:
			sys.stdout.flush()
			sys.stderr.flush()
		except OSError:
			pass

		os._exit(0)

	# A subsystem died at startup without anyone asking to stop — surface it
	# as a non-zero exit (startup_failed implies no live threads, so this never
	# co-occurs with the stuck-daemon path above).
	if startup_failed:
		raise SystemExit(1)


def _start_library_watchers (
	cfg:                subsample.config.Config,
	bank_manager:       typing.Optional[subsample.bank.BankManager],
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix:  typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager:  typing.Optional[subsample.transform.TransformManager],
	player_cell:        list[typing.Optional[subsample.player.MidiPlayer]],
	app_events:         subsample.events.EventEmitter,
	output_sample_rate: int,
) -> list[subsample.watcher.InstrumentWatcher]:

	"""Start the library.watch watchers, and return those that started.

	One per program directory when the map declares programs, else one on
	library.directory.  Each loads a sample that arrives into its library and
	takes out one that goes.
	"""

	watchers: list[subsample.watcher.InstrumentWatcher] = []

	# Without programs, a null library.directory has nothing to watch — the
	# samples come from the MIDI map's own predicates, and those directories are
	# not watched (a set on a shared drive would not deliver events anyway;
	# inotify only sees writes made through the local mount by this machine).
	# With programs, library.directory is ignored, so null is a reasonable
	# setting there and each program's directory is watched whatever it says.
	if cfg.library.watch and cfg.player.enabled:

		# Multi-bank mode: one watcher per bank directory.
		if bank_manager is not None:
			for bank in bank_manager.all_banks():
				_bank = bank  # capture for closure

				loaded_here = _loaded_by_path(_bank.instrument_library)

				def _make_bank_callback (b: subsample.bank.Bank) -> typing.Callable[[subsample.library.SampleRecord], None]:

					"""The watcher's new-sample callback for one program, bound to it here rather than by the loop."""

					def cb (record: subsample.library.SampleRecord) -> None:

						"""Add an arriving sample to this program's library and subsystems."""

						_log.info("Watcher [%s]: new sample - %s (%.2fs)", b.name, record.name, record.duration)
						_integrate_sample(record, b.instrument_library, b.similarity_matrix,
						                  b.transform_manager, player_cell, app_events)

					return cb

				def _make_bank_removal_callback (b: subsample.bank.Bank) -> typing.Callable[[pathlib.Path], None]:

					"""The watcher's removal callback for one program, bound to it here rather than by the loop."""

					def rm (path: pathlib.Path) -> None:

						"""Take a deleted sample out of this program's library and subsystems."""

						_log.info("Watcher [%s]: sample removed - %s", b.name, path.name)
						_remove_sample(path, b.instrument_library, b.similarity_matrix,
						               b.transform_manager, player_cell)

					return rm

				watcher = subsample.watcher.InstrumentWatcher(
					directory=_bank.directory,
					loaded=loaded_here,
					on_sample_loaded=_make_bank_callback(_bank),
					target_sample_rate=output_sample_rate,
					with_preview=cfg.recorder.previews,
					on_sample_removed=_make_bank_removal_callback(_bank),
				)
				if _start_watcher(watcher, f"sample directory {_bank.name!r}"):
					watchers.append(watcher)
					print(f"  Watcher      : monitoring {_bank.directory} ({_bank.name!r})")

		# Single-directory mode.
		elif cfg.library.directory is not None:

			def _on_watched_sample (record: subsample.library.SampleRecord) -> None:

				"""Add an arriving sample to the library and its subsystems."""

				_log.info("Watcher: new sample arrived - %s (%.2fs)", record.name, record.duration)
				_integrate_sample(record, instrument_library, similarity_matrix,
				                  transform_manager, player_cell, app_events)

			def _on_watched_sample_removed (path: pathlib.Path) -> None:

				"""Take a deleted sample out of the library and its subsystems."""

				_log.info("Watcher: sample removed - %s", path.name)
				_remove_sample(path, instrument_library, similarity_matrix,
				               transform_manager, player_cell)

			watcher = subsample.watcher.InstrumentWatcher(
				directory=pathlib.Path(cfg.library.directory),
				loaded=_loaded_by_path(instrument_library),
				on_sample_loaded=_on_watched_sample,
				target_sample_rate=output_sample_rate,
				with_preview=cfg.recorder.previews,
				on_sample_removed=_on_watched_sample_removed,
			)
			if _start_watcher(watcher, "sample directory"):
				watchers.append(watcher)
				print(f"  Watcher      : monitoring {cfg.library.directory} for new samples")

	return watchers


def _loaded_by_path (
	instrument_library: subsample.library.InstrumentLibrary,
) -> dict[pathlib.Path, typing.Optional[str]]:

	"""Each loaded sample's audio path, resolved, with the digest of the audio its analysis came from: what a library watcher starts from (#4703, M23)."""

	return {
		record.filepath.resolve(): record.audio_md5
		for record in instrument_library.samples()
		if record.filepath is not None
	}


class _ReloadOncePlaying:

	"""A map reload asked for before the player was up, made once it is (#4703, M11).

	The map watcher starts before the player thread publishes its player, and
	a reload that found no player was dropped: a map saved while the player
	started played its old version until it was saved again.
	"""

	def __init__ (self) -> None:

		"""Nothing asked for, and no player yet."""

		self._lock      = threading.Lock()
		self._playing   = False
		self._reload: typing.Optional[typing.Callable[[], None]] = None

	def wanted (self, reload: typing.Callable[[], None]) -> bool:

		"""Ask for a reload: True when the player is up and the caller makes it now, False when it waits."""

		with self._lock:
			if self._playing:
				return True

			self._reload = reload

			return False

	def published (self) -> None:

		"""The player is up: make the reload asked for before it was, if one was."""

		with self._lock:
			self._playing = True
			reload, self._reload = self._reload, None

		if reload is None:
			return

		# On the player's thread, before it plays: a failure must not stop it.
		try:
			reload()
		except Exception as exc:
			_log.error("MIDI map reload failed - keeping current map: %s", exc)


def _modification_times (paths: typing.Iterable[pathlib.Path]) -> dict[pathlib.Path, typing.Optional[int]]:

	"""Each file's modification time, in nanoseconds, or None where it cannot be read."""

	times: dict[pathlib.Path, typing.Optional[int]] = {}

	for path in paths:
		try:
			times[path] = path.stat().st_mtime_ns
		except OSError:
			times[path] = None

	return times


def _put_back (
	pools:  list[_TopLevelPool],
	every:  list[subsample.player.PathLoads],
	player: subsample.player.MidiPlayer,
) -> None:

	"""Undo what a refused map edit loaded: take its samples out, and put back those they pushed out (#4908).

	An edit's paths are loaded before it can be checked, since the extract
	check reads the formats of the samples it names.  Refused after that, the
	edit left its folders in the library, and at library.max_memory_mb they
	had pushed out the samples the map being kept plays.  A sample put back
	keeps its id, so it ranks by age where it did.  A capture that arrived
	meanwhile is neither of these, and stays.  With programs, each pool the
	edit loaded into is put back (#4909).
	"""

	removed: set[int] = set()
	put_back = 0

	for pool, loads in zip(pools, every):
		matrices = [pool.matrix] if pool.matrix is not None else []

		for record in loads.added:

			# Loaded again since, by a watcher or an import: no longer the edit's.
			if record.filepath is None or pool.library.find_by_path(record.filepath) != record.sample_id:
				continue

			pool.library.remove_by_path(record.filepath)

			for matrix in matrices:
				matrix.remove([record.sample_id])

			if pool.transforms is not None:
				pool.transforms.on_parent_evicted([record.sample_id])

			removed.add(record.sample_id)

		for record in loads.evicted:
			evicted = pool.library.add(record)

			for matrix in matrices:
				if evicted:
					matrix.remove(evicted)
				matrix.add(record)

			if pool.transforms is not None:
				if evicted:
					pool.transforms.on_parent_evicted(evicted)
				pool.transforms.on_sample_added(record)

			put_back += 1

	if not removed and not put_back:
		return

	_log.info(
		"MIDI map reload: removed %d sample(s) the refused edit loaded%s",
		len(removed),
		f", and put back {put_back} they pushed out" if put_back else "",
	)

	# The notes go back to the samples they had: the player ranked none of
	# this, or ranked the edit's samples for a capture that came meanwhile.
	player._try_update_assignments("the return to the map kept")


def _start_midi_map_watcher (
	cfg:                subsample.config.Config,
	preloaded:          subsample.player.MidiMapResult,
	reference_library:  subsample.library.ReferenceLibrary,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix:  typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager:  typing.Optional[subsample.transform.TransformManager],
	player_cell:        list[typing.Optional[subsample.player.MidiPlayer]],
	once_playing:       typing.Optional["_ReloadOncePlaying"] = None,
	map_read_as:        typing.Optional[dict[pathlib.Path, typing.Optional[int]]] = None,
	bank_manager:       typing.Optional[subsample.bank.BankManager] = None,
) -> typing.Optional[subsample.watcher.MidiMapWatcher]:

	"""Start player.watch_midi_map's watcher, and return it, or None if it could not start.

	Every file the map is read from is watched: the map, each set an ensemble
	includes or player.midi_maps names, and every definitions file they mount
	(MidiMapResult.source_files, #389).  config.yaml itself is not, so a set
	added to or removed from player.midi_maps still takes a restart.  A
	file on a network drive is polled, since a write another machine makes
	there gives no notice (watcher._NetworkDriveObserver, #3972).
	"""

	if cfg.player.midi_maps is not None:
		_log.info(
			"Watching the sets player.midi_maps names; config.yaml is not "
			"watched, so adding or removing a set needs a restart",
		)

	# A map: preset's folder is the top-level map's; the config form has no
	# map file and no programs, so its presets are none.
	_midi_map_dir = (
		pathlib.Path(cfg.player.midi_map).parent
		if cfg.player.midi_map is not None else pathlib.Path.cwd()
	)

	# Snapshot the bank state at startup so the live-reload callback can
	# detect bank-related edits that hot-reload doesn't cover yet.
	_startup_bank_definitions = list(preloaded.bank_definitions)
	_startup_bank_channel     = preloaded.bank_channel
	_startup_default_bank     = preloaded.default_bank

	midi_map_watcher: typing.Optional[subsample.watcher.MidiMapWatcher] = None

	def _on_midi_map_changed (path: pathlib.Path) -> None:

		"""Reload the MIDI map and deliver it to the active player.

		Program-set edits (the `programs:` block, `program_channel:`, and
		`default_program:`) are not hot-reloadable in this version — the
		callback warns and keeps the current program state.  Editing a
		`map:` preset's OWN file is also not watched (it is not among the
		map's source files) and needs a restart, so a broken preset is only
		warned of here: it does not hold up an edit to the map.  Edits to
		the map, a set it includes or a definitions file reload as normal,
		and once the map parses, the watcher follows the files it was
		read from this time (#389).  Asked for before the player is up, the
		reload waits for it (_ReloadOncePlaying, #4703).
		"""

		if once_playing is not None and not once_playing.wanted(lambda: _on_midi_map_changed(path)):
			return

		player = player_cell[0]

		if player is None:
			return

		try:
			# Reload through the same resolver startup used, so editing a
			# ensemble re-merges every set it includes rather than
			# reloading the ensemble file's own assignments alone.
			result = subsample.player.load_configured_map(
				cfg, reference_library.names(), presets=False,
			)
		except (OSError, ValueError, yaml.YAMLError) as exc:
			_log.warning(
				"MIDI map reload failed at parse time - keeping current "
				"map: %s", exc,
			)
			return

		# Follow the files this edit is read from, even if a later check
		# keeps the old map: a set the edit names is the one being worked on.
		if midi_map_watcher is not None:
			midi_map_watcher.watch(result.source_files)

		_warn_of_broken_presets(
			result, _midi_map_dir, reference_library.names(),
			cfg.player.strict_midi_map,
		)

		# Detect program-set changes the live reload can't apply, so the
		# user isn't left wondering why an edit to programs:/program_channel:/
		# default_program: has no audible effect.  The BankDefinition diff
		# also catches a changed map: path or a program retyped between
		# map: and directory: (map_path is part of the frozen equality).
		if (
			result.bank_definitions != _startup_bank_definitions
			or result.bank_channel != _startup_bank_channel
			or result.default_bank != _startup_default_bank
		):
			_log.warning(
				"MIDI map reload: programs, program_channel, or "
				"default_program changed - these only take effect on restart. "
				"Editing a map: preset's own file also needs a restart. "
				"Top-level assignment changes will still apply.",
			)

		# Load any NEW path/directory/reference predicates the edit
		# introduced — startup does this via _load_map_paths, and
		# without it here a live-coded `path:`/`directory:` select would
		# reload "successfully" but silently match nothing until restart.
		# They go where startup puts them: the library, or with programs
		# each `directory:` program's (#4909).  Already-loaded paths are
		# deduped inside.
		#
		# What it loads is noted, so a refusal below can undo it (#4908).
		pools = _top_level_pools(bank_manager, instrument_library, similarity_matrix, transform_manager)
		every = [subsample.player.PathLoads() for _pool in pools]

		try:
			_load_map_paths(
				result.note_map, result.zone_templates, pools, every,
				reference_library, _output_sample_rate(cfg), cfg.recorder.previews,
			)
		except Exception as exc:
			_log.warning(
				"MIDI map reload: could not load new path references - "
				"affected selects may match nothing: %s", exc,
			)

		# Re-run the extract compatibility check that startup does.  Without
		# it, a live edit introducing `extract: depth` on a stereo pool (or
		# `extract: channel.9` on a 2-channel sample) reloads "successfully"
		# and then raises inside build_extract_matrix on EVERY note-on, where
		# _safe_handle_message swallows it into a log line — the assignment
		# just goes silent.  Rejecting the reload keeps the working map live,
		# which is the same trade the parse-time failure above makes.
		try:
			subsample.player._validate_assignment_extracts(
				result.note_map, _extract_checked_library(pools, instrument_library),
			)
		except ValueError as exc:
			_log.error(
				"MIDI map reload failed extract validation - keeping current "
				"map: %s", exc,
			)
			_put_back(pools, every, player)
			return

		# reload_midi_map runs update_assignments() against the new map
		# and rolls back to the previous map + zone templates on any
		# exception — broad catch here so a runtime-validated YAML
		# error (e.g. similarity ordering without where.reference set,
		# only detectable when the query actually runs) never stops
		# live playback.
		try:
			player.reload_midi_map(result)
		except Exception as exc:
			_log.error(
				"MIDI map reload failed validation - keeping current map: %s",
				exc,
			)
			_put_back(pools, every, player)

	watched = preloaded.source_files

	midi_map_watcher = subsample.watcher.MidiMapWatcher(
		paths=watched,
		on_changed=_on_midi_map_changed,
	)

	# The map is the first of its source files; the rest are what it reads.
	if cfg.player.midi_map is not None:
		watching = cfg.player.midi_map + (
			f" and the {len(watched) - 1} file(s) it reads" if len(watched) > 1 else ""
		)
	else:
		watching = f"the {len(watched)} file(s) player.midi_maps reads"

	if not _start_watcher(midi_map_watcher, "MIDI map"):
		return None

	print(f"  MIDI map     : watching {watching} for changes")

	# Saved after it was read and before the watcher started, during the
	# library's load: the watcher never saw it (#4703, M11).
	if map_read_as is not None and _modification_times(watched) != map_read_as:
		_on_midi_map_changed(watched[0])

	return midi_map_watcher


def _start_osc_sender (cfg: subsample.config.Config, app_events: subsample.events.EventEmitter) -> None:

	"""Send OSC to osc.send_host as samples are captured and loaded, or say why not and carry on without.

	The sender resolves its host as it is made, so a name that does not
	resolve stopped Subsample, file mode too, with a traceback that named no
	setting (#4703, M10).  The receivers already carry on without OSC.
	"""

	try:
		sender = subsample.osc.OscEventSender(cfg.osc.send_host, cfg.osc.send_port)

	except ImportError:
		_log.warning("OSC enabled but python-osc not installed. pip install 'subsample[osc] @ git+https://github.com/simonholliday/subsample.git'")
		return

	except OSError as exc:
		_log.warning("OSC sender could not reach osc.send_host %r: %s - OSC sending disabled", cfg.osc.send_host, exc)
		return

	app_events.on("sample_captured", sender.on_sample_captured_event)
	app_events.on("sample_loaded", sender.on_sample_loaded_event)
	print(f"  OSC sender   : sending to {cfg.osc.send_host}:{cfg.osc.send_port}")


def _start_osc_receiver (
	cfg:                subsample.config.Config,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix:  typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager:  typing.Optional[subsample.transform.TransformManager],
	player_cell:        list[typing.Optional[subsample.player.MidiPlayer]],
	app_events:         subsample.events.EventEmitter,
	output_sample_rate: int,
	shutdown_event:     threading.Event,
) -> typing.Any:

	"""Start the OSC receiver that imports each file a /sample/import message names.

	Returns the started subsample.osc.OscReceiver, or None when python-osc is
	missing or the port is taken; either is logged and the rest of startup
	goes on.  Typed Any because subsample.osc needs python-osc only once a
	receiver is built.
	"""

	try:
		def _on_osc_import (file_path_str: str) -> None:
			"""Handle a /sample/import OSC message.

			Reads and analyses the file in place (does not copy), writing
			its analysis files beside it as a library folder's are, then
			loads it into the in-memory instrument library for immediate
			playback.  The sample is available until the next restart.
			"""

			file_path = pathlib.Path(file_path_str)

			if not file_path.is_file():
				_log.warning("OSC /sample/import: file not found: %s", file_path)
				return

			# On an analysis worker process: on this thread, in the player's
			# process, analysis makes the audio and the notes wait (#4667).
			result = subsample.parallelism.run_in_analysis_worker(
				subsample.cache.ensure_sample_assets, file_path, with_preview=cfg.recorder.previews,
			)

			if result is None:
				_log.warning("OSC /sample/import: analysis failed: %s", file_path)
				return

			audio = subsample.library.load_wav_audio(file_path, output_sample_rate)

			if audio is None:
				_log.warning("OSC /sample/import: could not read audio: %s", file_path)
				return

			record = subsample.library.SampleRecord(
				sample_id      = subsample.library.allocate_id(),
				name           = file_path.stem,
				spectral       = result.spectral,
				rhythm         = result.rhythm,
				pitch          = result.pitch,
				timbre         = result.timbre,
				level          = result.level,
				band_energy    = result.band_energy,
				params         = result.params,
				duration       = result.duration,
				audio          = audio,
				filepath       = file_path,
				channel_format = result.channel_format,
				loop           = result.loop,
				audio_sample_rate = output_sample_rate,
				audio_md5      = result.audio_md5,
			)

			# An analysis that outlasts the receiver's wait at stop ends here
			# after the transform workers it would hand variants to are gone.
			# The sample would last only until the restart anyway.
			if shutdown_event.is_set():
				_log.info("OSC /sample/import: shutting down - not adding %s", file_path)
				return

			_integrate_sample(record, instrument_library, similarity_matrix,
			                  transform_manager, player_cell, app_events)

		osc_receiver = subsample.osc.OscReceiver(
			port=cfg.osc.receive_port,
			on_import=_on_osc_import,
			host=cfg.osc.receive_host,
		)
		osc_receiver.start()
		print(f"  OSC receiver : listening on port {cfg.osc.receive_port}")

		return osc_receiver

	except ImportError:
		_log.warning("OSC receive enabled but python-osc not installed. pip install 'subsample[osc] @ git+https://github.com/simonholliday/subsample.git'")
	except OSError as exc:
		# OscReceiver binds the UDP socket in its constructor, so a busy port
		# raises OSError here (not ImportError).  Log and continue rather
		# than letting it escape and skip the rest of startup.
		_log.warning("OSC receiver could not bind port %d: %s - OSC receive disabled", cfg.osc.receive_port, exc)

	return None


def _start_osc_note_receiver (
	cfg:         subsample.config.Config,
	player_cell: list[typing.Optional[subsample.player.MidiPlayer]],
) -> typing.Any:

	"""Start the OSC receiver that plays each /note/on and /note/off through the player.

	Each note goes to the player in player_cell when it arrives, so a note
	before the player has started, or with the player off, plays nothing.
	Returns the started subsample.osc.OscNoteReceiver, or None when python-osc
	is missing or the port is taken; either is logged and the rest of startup
	goes on.  Typed Any, as _start_osc_receiver's is.
	"""

	if not cfg.player.enabled:
		_log.warning("osc.notes_enabled is on but the player is off, so OSC notes will play nothing")

	def _on_note (on: bool, channel: int, note: int, velocity: float, when: float, timed: bool) -> None:

		"""Hand one OSC note to the player, if there is one yet."""

		player = player_cell[0]

		if player is None:
			_log.debug("OSC note before the player started - not played")
			return

		player.play_osc_note(on, channel, note, velocity, when, timed)

	try:
		note_receiver = subsample.osc.OscNoteReceiver(
			port=cfg.osc.notes_port,
			on_note=_on_note,
			host=cfg.osc.notes_host,
		)
		note_receiver.start()
		print(f"  OSC notes    : listening on port {cfg.osc.notes_port}")

		return note_receiver

	except ImportError:
		_log.warning("OSC notes enabled but python-osc not installed. pip install 'subsample[osc] @ git+https://github.com/simonholliday/subsample.git'")
	except OSError as exc:
		_log.warning("OSC note receiver could not bind port %d: %s - OSC notes disabled", cfg.osc.notes_port, exc)

	return None


def _drain_captures (
	processor: typing.Optional[subsample.recorder.SampleProcessor],
	poll:      float = 0.5,
) -> None:

	"""Wait for every capture still being analysed, saying how many are left.

	A capture is written to disk only after its analysis, preview and loop
	search have finished, so anything still in the queue at shutdown is a
	recording that does not exist yet.  Shutdown used to join each subsystem
	thread for ten seconds and then hard-exit, which on a Pi, or with
	reserve_for_player narrowing the pool, is far less than a backlog needs:
	the recordings went with it, silently and with exit code 0.

	Waiting is unbounded on purpose, because the alternative is losing work the
	musician has already played.  The count is printed so the wait is explained
	rather than looking like a hang.
	"""

	if processor is None:
		return

	remaining = processor.queue_depth

	if not remaining:
		return

	_say(f"Finishing {remaining} capture(s) still being analysed… (Ctrl+C again to save them unanalysed)")

	while True:
		processor.flush()

		still_going = processor.queue_depth

		if not still_going:
			break

		# A capture that arrived while we were flushing: report and go round.
		_say(f"  {still_going} to go…")
		time.sleep(poll)


def _print_banner (cfg: subsample.config.Config, multi_bank: bool = False) -> None:

	"""Print the startup summary line.

	Each enabled subsystem contributes its OWN real settings — the recorder its
	capture format + capture-only fields (buffer, SNR, capture directory), the
	player its resolved OUTPUT format + the map and sample source it plays from.
	A player-only run therefore reports the player's output (e.g. 48000 Hz, 8ch),
	not the disabled recorder's capture rate.

	``multi_bank`` is True when the MIDI map declares ``programs:`` — the pool then
	comes from each program's own directory/map, so the banner says so instead of
	naming library.directory (which multi-bank mode ignores).
	"""

	modes = []
	if cfg.recorder.enabled:
		modes.append("recorder")
	if cfg.player.enabled:
		modes.append("player")
	mode_str = " + ".join(modes) if modes else "file-only"

	segments: list[str] = []

	if cfg.recorder.enabled:
		# channels may be None when auto-detect is configured; show "auto" until resolved.
		ch_str = (
			f"{cfg.recorder.audio.channels}ch"
			if cfg.recorder.audio.channels is not None
			else "auto"
		)
		segments.append(
			f"rec {cfg.recorder.audio.sample_rate} Hz  "
			f"{cfg.recorder.audio.bit_depth}-bit  {ch_str}  |  "
			f"buffer {cfg.recorder.buffer.max_seconds}s  |  "
			f"trigger ≥ {cfg.detection.threshold_db} dB over ambient  |  "
			f"→ {cfg.recorder.directory}"
		)

	if cfg.player.enabled:
		# Mirror the engine's own output resolution so the banner matches the
		# stream that actually opens: rate falls back to the recorder's when
		# player.audio.sample_rate is unset (_output_sample_rate); bit depth
		# likewise (MidiPlayer output_bit_depth); channels default to stereo.
		out_rate = _output_sample_rate(cfg)
		out_bits = (
			cfg.player.audio.bit_depth
			if cfg.player.audio.bit_depth is not None
			else cfg.recorder.audio.bit_depth
		)
		out_ch  = cfg.player.audio.channels if cfg.player.audio.channels is not None else 2
		map_str = cfg.player.midi_map if cfg.player.midi_map is not None else "(no map)"
		# Multi-bank mode ignores library.directory — each program supplies its own
		# pool, so naming a single directory would mislead.  A null directory is
		# the same situation for a different reason: the MIDI map's own
		# predicates supply the samples.
		if multi_bank:
			source = "per-program pools"
		elif cfg.library.directory is None:
			source = "MIDI map directories"
		else:
			source = str(cfg.library.directory)
		segments.append(
			f"out {out_rate} Hz  {out_bits}-bit  {out_ch}ch  |  "
			f"map {map_str}  |  ← {source}"
		)

	body = "  ||  ".join(segments) if segments else "file-only"
	print(f"Subsample  |  {mode_str}  |  {body}")


def _integrate_sample (
	record: subsample.library.SampleRecord,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]],
	app_events: typing.Optional[subsample.events.EventEmitter] = None,
) -> None:

	"""Add a new sample to all live subsystems.

	Called from both the live-capture on_complete callback and the directory
	watcher whenever a new sample is ready. Adds the record to the instrument
	library (evicting the oldest if over the memory limit), updates the
	similarity matrix, notifies the transform pipeline to produce variants,
	triggers a pitched-assignment update on the active player, and emits
	a ``sample_loaded`` event for integrations such as the OSC sender.

	Thread-safe: each subsystem uses an internal lock. The multi-step
	sequence (library → similarity → transforms → player) is not atomic
	across subsystems — a concurrent query between steps may see transiently
	inconsistent state (e.g. an evicted sample still present in the
	similarity matrix). This is acceptable for the current use case.
	"""

	# Identity is the resolved filepath: re-integrating the SAME file (a
	# re-analysis after an edit) replaces its prior record via add()'s path
	# de-dup; a different file that happens to share a stem (e.g. "01.wav" in
	# another take-folder) loads as a distinct sample.  No stem-collision guard
	# is needed — the old warn-and-skip here silently dropped legitimate
	# same-stem takes.
	evicted = instrument_library.add(record)
	_log.info(
		"Instrument library: %d sample(s)  [%s]",
		len(instrument_library), instrument_library.format_memory(),
	)

	if similarity_matrix is not None:
		if evicted:
			similarity_matrix.remove(evicted)
		similarity_matrix.add(record)

		scores = similarity_matrix.get_scores(record.sample_id)
		if scores:
			_log.debug(
				"Similarity: %s",
				subsample.similarity.format_similarity_scores(scores),
			)

	if transform_manager is not None:
		if evicted:
			transform_manager.on_parent_evicted(evicted)
		transform_manager.on_sample_added(record)

	if player_cell is not None and player_cell[0] is not None:
		# Defensive: a runtime query failure here (e.g. a hot-edited map
		# containing a similarity order without a reference) must not
		# kill the watcher / OSC import thread that triggered this
		# integration.  The sample is already in the library and
		# similarity matrix; only the variant pre-compute would be lost,
		# and the next note_on falls back to non-pitched playback.
		player_cell[0]._try_update_assignments(
			f"new sample {record.name!r} integration",
		)

	if app_events is not None:
		app_events.emit("sample_loaded", record=record)


def _remove_sample (
	path: pathlib.Path,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]],
) -> None:

	"""Remove a deleted sample from all live subsystems.

	Called by the directory watcher when a watched audio file is deleted or
	renamed away.  Drops the record from the instrument library — so a
	re-encoded or removed file's record does not linger as a selectable "ghost"
	that still plays its cached audio — then cascade-cleans the similarity and
	transform state and refreshes the player's assignments, mirroring the
	eviction cleanup in _integrate_sample.  A no-op if no sample at that path is
	currently loaded.
	"""

	removed_id = instrument_library.remove_by_path(path)

	if removed_id is None:
		return

	_log.info(
		"Instrument library: removed %s  [%s]",
		path.name, instrument_library.format_memory(),
	)

	if similarity_matrix is not None:
		similarity_matrix.remove([removed_id])

	if transform_manager is not None:
		transform_manager.on_parent_evicted([removed_id])

	if player_cell is not None and player_cell[0] is not None:
		# Defensive, as in _integrate_sample: a runtime query failure must not
		# kill the watcher thread — the sample is already gone from the library.
		player_cell[0]._try_update_assignments(
			f"sample removal {path.name!r}",
		)


def _make_on_complete (
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	store_audio: bool,
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]] = None,
	app_events: typing.Optional[subsample.events.EventEmitter] = None,
) -> subsample.recorder._OnCompleteCallback:

	"""Return the on_complete callback for the live-capture SampleProcessor.

	The returned callback runs on the writer thread and must not block.
	It logs the analysis result, adds the recording to the instrument
	library, updates the similarity matrix, notifies the transform
	pipeline so derivative variants can be produced in the background,
	and emits ``sample_captured`` for integrations such as the OSC sender.

	Args:
		store_audio:       When True, build a SampleRecord (with PCM audio) and
		                   integrate it into the live subsystems. Set to
		                   cfg.player.enabled — in recorder-only mode no subsystem
		                   reads from the instrument library, so integration is skipped.
		transform_manager: Optional transform pipeline coordinator. When provided,
		                   cascade-evicts derivatives for any evicted parents and
		                   triggers auto-variant production for the new sample.
		player_cell:       Single-element list holding the active MidiPlayer, or None.
		                   When provided, update_pitched_assignments() is called after
		                   each new sample is added to the similarity matrix so pitched
		                   keyboard assignments pre-compute variants for the new best match.
		app_events:        Optional event emitter; when provided, ``sample_captured``
		                   is emitted after each recording is integrated.
	"""

	def on_complete (
		filepath: pathlib.Path,
		spectral: subsample.analysis.AnalysisResult,
		rhythm: subsample.analysis.RhythmResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		level: subsample.analysis.LevelResult,
		band_energy: subsample.analysis.BandEnergyResult,
		duration: float,
		audio: numpy.ndarray,
		*,
		channel_format: str = "pcm",
		loop: typing.Optional[subsample.loopfind.LoopPoints] = None,
	) -> None:

		"""Announce a finished recording and add it to the live library and its subsystems."""

		_log.info(
			"Recorded %s: duration %.2fs, %s",
			filepath.name, duration,
			subsample.analysis.format_level_result(level),
		)

		# Emit the integration event BEFORE the store_audio gate.  The
		# event is what external listeners (such as the OSC sender) react
		# to — gating it on the player being enabled would silently break
		# the documented recorder-only-with-OSC multi-machine setup.
		if app_events is not None:
			app_events.emit(
				"sample_captured",
				filepath=filepath, spectral=spectral, rhythm=rhythm,
				pitch=pitch, timbre=timbre, level=level,
				band_energy=band_energy, duration=duration, audio=audio,
			)

		# Only build a SampleRecord and integrate into the live subsystems
		# when the player is active.  In recorder-only mode nothing reads
		# from the instrument library, similarity matrix, or transform
		# pipeline, so the work (and its log line) would be pure noise.
		if not store_audio:
			return

		record = subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = filepath.stem,
			spectral    = spectral,
			rhythm      = rhythm,
			pitch       = pitch,
			timbre      = timbre,
			level       = level,
			band_energy = band_energy,
			params      = analysis_params,
			duration    = duration,
			audio       = audio,
			filepath    = filepath,
			# Freshly-captured audio is at the recorder rate (== analysis rate),
			# not resampled to the output rate.
			audio_sample_rate = analysis_params.sample_rate,
			# Both come from the recorder, which worked them out while writing the
			# sidecar.  Without the tag a fresh ambisonic capture played through
			# the plain mix instead of the decoder until the next restart, and
			# without the loop a loopable one had no loop points.
			channel_format    = channel_format,
			loop              = loop,
		)

		_integrate_sample(record, instrument_library, similarity_matrix,
		                  transform_manager, player_cell, app_events)

	return on_complete
