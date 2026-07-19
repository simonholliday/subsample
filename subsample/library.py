"""Sample library for Subsample.

Provides two distinct in-memory collections:

  ReferenceLibrary — canonical sounds (kick, snare, hi-hat, …) loaded from
      .analysis.json sidecars. Audio is NOT stored; only analysis metadata is
      kept. Used for similarity classification. Looked up by name.

  InstrumentLibrary — the "playable" sample collection. Every sample carries
      its original-format PCM audio alongside analysis metadata. New recordings
      are added automatically during streaming; a configurable FIFO memory limit
      prevents unbounded growth. Looked up by numeric ID.

Both collections share the same SampleRecord dataclass. Each record has a
session-unique numeric ID allocated by allocate_id().

Name derivation (both libraries):
  BD0025.WAV.analysis.json  →  audio file "BD0025.WAV"  →  name "BD0025"
  kick.wav.analysis.json    →  audio file "kick.wav"     →  name "kick"

ReferenceLibrary lookup is case-insensitive.
"""

import collections
import concurrent.futures
import dataclasses
import itertools
import logging
import os
import pathlib
import threading
import typing
import wave

import librosa
import numpy

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.loopfind


_log = logging.getLogger(__name__)


# Session-unique ID counter. itertools.count is thread-safe (C extension;
# next() is atomic under the GIL), so callbacks on the writer thread and
# startup loading on the main thread can both call allocate_id() safely.
_id_counter: "itertools.count[int]" = itertools.count(1)


def allocate_id () -> int:

	"""Return the next session-unique sample ID (1, 2, 3, …).

	IDs are allocated in order across both ReferenceLibrary and InstrumentLibrary
	so that every Sample in a session has a distinct numeric identifier.
	"""

	return next(_id_counter)


def _resolve_path (path: pathlib.Path) -> pathlib.Path:

	"""Canonicalise a sample path for use as an identity key.

	The library keys samples by resolved filepath (the filename stem is a
	human label that may legitimately repeat across take-folders — e.g.
	``01.wav`` in two technique subfolders).  ``resolve()`` collapses symlinks
	and relative segments so the same file reached by two different path
	strings maps to one key.  It does not raise for a path whose file has been
	deleted (strict=False), so an evicted/ghost record still resolves to a
	stable key.  This MUST match how ``query._resolved_sample_path`` canonicalises
	the ``directory:`` / ``path:`` predicates (both call ``Path.resolve()``), or a
	same-file record could slip past a predicate.
	"""

	return path.resolve()


@dataclasses.dataclass(frozen=True)
class SampleRecord:

	"""A single sample held in memory, with required analysis data and optional audio.

	Used for both reference samples (audio=None, metadata only) and instrument
	samples (audio contains original-format PCM for playback).

	Fields:
		sample_id: Session-unique numeric ID (allocated by allocate_id()).
		name:      Stem of the audio filename (e.g. "BD0025", "kick").
		           Preserves original casing.  A human label, NOT a unique key:
		           it may repeat across take-folders (two "01.wav"), so identity
		           is sample_id / filepath (see InstrumentLibrary.find_by_path).
		spectral:    Thirteen normalised [0, 1] spectral metrics (the spectral fingerprint).
		rhythm:      Tempo, beat grid, pulse curve, onset times.
		pitch:       Fundamental frequency, chroma profile, pitch class.
		timbre:      MFCC timbral fingerprints (mfcc, mfcc_delta, mfcc_onset).
		level:       Peak and RMS amplitude (used for playback level normalisation).
		band_energy: Per-band energy fractions and decay rates (4 bands, 8 values total).
		params:    FFT parameters used when the analysis was computed.
		duration:  Recording length in seconds.
		audio:     Original capture-format PCM as a numpy array, shape
		           (n_frames, channels), dtype int16 or int32. None for reference
		           samples where only metadata is needed.
		filepath:  Path to the WAV file on disk, if known. None for in-memory-only
		           samples not yet (or never) written to disk.
		channel_format: Tag describing how the audio channels should be
		           interpreted.  "pcm" (default) means standard multichannel
		           PCM — mono, stereo, 5.1, etc. — routed directly through the
		           mix matrix.  "b_format_ambix" means first-order ambisonic
		           B-format (channel order W, Y, Z, X; SN3D) — the player
		           applies a decoder and rotation matrix before mix routing.
		loop:      Seamless loop points found for a loopable sample (None when the
		           sample is not a loop candidate or has no clean junction), used
		           by loop-mode playback.  Frame indices in this sample's timeline.
		audio_sample_rate: Sample rate of the ``audio`` array in Hz.  This can
		           differ from ``params.sample_rate`` (the rate the sidecar was
		           analysed at): disk-loaded audio is resampled to the player
		           output rate on load, while live-captured audio stays at the
		           recorder rate.  The transform engine needs the true audio rate
		           to run DSP and resample correctly.  None (no audio, or an older
		           construction site) means "assume the processor's own rate".
	"""

	sample_id:   int
	name:        str
	spectral:    subsample.analysis.AnalysisResult
	rhythm:      subsample.analysis.RhythmResult
	pitch:       subsample.analysis.PitchResult
	timbre:      subsample.analysis.TimbreResult
	level:       subsample.analysis.LevelResult
	band_energy: subsample.analysis.BandEnergyResult
	params:      subsample.analysis.AnalysisParams
	duration:    float
	audio:       typing.Optional[numpy.ndarray] = None
	filepath:    typing.Optional[pathlib.Path]  = None
	channel_format: str = "pcm"
	loop:        typing.Optional[subsample.loopfind.LoopPoints] = None
	audio_sample_rate: typing.Optional[int] = None


class ReferenceLibrary:

	"""In-memory index of named reference samples (metadata only, no audio).

	Records are keyed by uppercased name for case-insensitive lookup.
	All public methods preserve original-casing in returned values.
	"""

	def __init__ (self, records: list[SampleRecord]) -> None:

		"""Build the index from a list of SampleRecords.

		Use load_reference_library() rather than calling this directly.  Records
		are indexed by uppercased name for case-insensitive lookup; IDs are
		assigned by the loader that constructs the records, not here.
		"""

		# Store records keyed by uppercased name for O(1) case-insensitive lookup.
		# Original-cased names are preserved in each record.
		self._index: dict[str, SampleRecord] = {r.name.upper(): r for r in records}

	def get (self, name: str) -> typing.Optional[SampleRecord]:

		"""Return the record for name (case-insensitive), or None if not found."""

		return self._index.get(name.upper())

	def names (self) -> list[str]:

		"""Return all loaded sample names in sorted order (original casing)."""

		return sorted(r.name for r in self._index.values())

	def samples (self) -> list[SampleRecord]:

		"""Return all loaded sample records sorted by name (case-insensitive)."""

		return sorted(self._index.values(), key=lambda r: r.name.upper())

	def __len__ (self) -> int:

		return len(self._index)

	def __repr__ (self) -> str:

		names = ", ".join(self.names())
		return f"ReferenceLibrary({len(self)} sample(s): {names})"


class InstrumentLibrary:

	"""Mutable, memory-bounded collection of instrument samples with audio data.

	Records are stored in insertion order. When adding a sample would push total
	audio memory over the configured limit, the oldest samples are evicted (FIFO)
	until there is room. Eviction only removes samples from memory — WAV files on
	disk are never deleted.

	Samples are looked up by their numeric sample_id.

	Usage:
		lib = InstrumentLibrary(max_memory_bytes=100 * 1024 * 1024)
		evicted_ids = lib.add(record)
		sample = lib.get(record.sample_id)
	"""

	def __init__ (self, max_memory_bytes: int) -> None:

		"""Create an empty instrument library with the given memory limit.

		Args:
			max_memory_bytes: Maximum total audio memory in bytes. When this
			                  limit is exceeded, oldest samples are evicted (FIFO).
		"""

		self._index: dict[int, SampleRecord] = {}
		# Deque maintains insertion order for O(1) popleft during FIFO eviction.
		self._order: collections.deque[int] = collections.deque()
		# Primary identity index: resolved filepath → id.  A file's PATH is
		# unique; its filename stem is not (take-folders reuse "01.wav"), so
		# path is the dedup/identity key for every on-disk record.
		self._path_index: dict[pathlib.Path, int] = {}
		# Remembers each filepath-bearing record's resolved _path_index key so
		# eviction / removal delete by the EXACT key stored at insert, never by
		# re-resolving old_record.filepath (which could drift if the path's
		# symlink topology changed mid-session, orphaning a _path_index entry).
		self._id_to_key: dict[int, pathlib.Path] = {}
		# Fallback secondary index for records with no filepath (in-memory /
		# test records).  In production every InstrumentLibrary record carries a
		# filepath, so this stays empty; kept for the filepath-less case.
		self._name_index: dict[str, int] = {}
		self._total_bytes: int = 0
		self._max_bytes: int = max_memory_bytes
		# Protects multi-step add/evict operations: the recorder's writer thread
		# calls add() while the main thread may call samples() or get().
		self._lock = threading.Lock()

	def add (self, record: SampleRecord) -> list[int]:

		"""Add a sample, evicting oldest samples if the memory limit would be exceeded.

		Returns a list of sample IDs that were evicted to make room. An empty list
		means no eviction was needed.

		If the sample's audio is larger than the entire memory limit, it is still
		added (a WARNING is logged) to ensure the most recent capture is always
		available.
		"""

		sample_bytes = record.audio.nbytes if record.audio is not None else 0

		# Warn if a single sample exceeds the entire memory budget
		if sample_bytes > self._max_bytes > 0:
			_log.warning(
				"Instrument sample #%d (%s) is %.1f MB which exceeds the memory "
				"limit of %.1f MB — added anyway",
				record.sample_id, record.name,
				sample_bytes / (1024 * 1024), self._max_bytes / (1024 * 1024),
			)

		# Evict oldest samples until there is room for the new one.
		# Held under lock: this is a multi-step operation (popleft + pop + counter
		# decrement) that must be atomic with respect to samples()/get() on the
		# main thread.
		evicted: list[int] = []
		with self._lock:

			# De-dup by IDENTITY (resolved filepath).  A same-PATH re-add is a
			# normal flow — the recorder re-writes a filename, a sample is
			# re-analysed with a fresh sample_id — so drop the prior record;
			# otherwise it lingers in _order/_index as a stale duplicate (which
			# would also let a later FIFO eviction delete the index key this
			# record now owns).  Report it as evicted so callers cascade-clean
			# its similarity / transform entries, keyed by the old id.  Two
			# distinct files sharing a stem (different folders) have different
			# paths, hence different keys, so they COEXIST.  filepath-less
			# records (in-memory / tests) fall back to the name index.
			key_path = _resolve_path(record.filepath) if record.filepath is not None else None
			prior_id = (
				self._path_index.get(key_path) if key_path is not None
				else self._name_index.get(record.name)
			)

			if prior_id is not None and prior_id != record.sample_id:
				prior = self._index.pop(prior_id, None)
				self._id_to_key.pop(prior_id, None)

				if prior is not None:
					self._total_bytes -= prior.audio.nbytes if prior.audio is not None else 0
					evicted.append(prior_id)

				try:
					self._order.remove(prior_id)
				except ValueError:
					pass

			while self._order and self._total_bytes + sample_bytes > self._max_bytes:
				oldest_id = self._order.popleft()
				old_record = self._index.pop(oldest_id, None)

				if old_record is not None:
					old_bytes = old_record.audio.nbytes if old_record.audio is not None else 0
					self._total_bytes -= old_bytes

					# Drop the evicted record's secondary-index key, but only if
					# it still points at this id — a same-key re-add may have
					# repointed it.  Delete by the key STORED at insert (never a
					# re-resolve, which could drift); use whichever index the
					# record lives in.
					old_key = self._id_to_key.pop(oldest_id, None)
					if old_key is not None:
						if self._path_index.get(old_key) == oldest_id:
							del self._path_index[old_key]
					elif self._name_index.get(old_record.name) == oldest_id:
						del self._name_index[old_record.name]

					evicted.append(oldest_id)

			self._index[record.sample_id] = record
			if key_path is not None:
				self._path_index[key_path] = record.sample_id
				self._id_to_key[record.sample_id] = key_path
			else:
				self._name_index[record.name] = record.sample_id
			self._order.append(record.sample_id)
			self._total_bytes += sample_bytes

		return evicted

	def get (self, sample_id: int) -> typing.Optional[SampleRecord]:

		"""Return the sample with the given ID, or None if not present."""

		with self._lock:
			return self._index.get(sample_id)

	def find_by_path (self, path: pathlib.Path) -> typing.Optional[int]:

		"""Return the sample_id for the sample at ``path``, or None if not present.

		Path is the library's true identity key — the filename stem may repeat
		across take-folders, so a path is what uniquely identifies a sample.
		The path is resolved (realpath) before lookup, so callers need not
		pre-canonicalise.  O(1) via the path index.  Returns None if no sample
		at that path is currently loaded (never loaded or evicted).
		"""

		with self._lock:
			return self._path_index.get(_resolve_path(path))

	def remove_by_path (self, path: pathlib.Path) -> typing.Optional[int]:

		"""Remove the sample at ``path`` from the library; return its id, or None.

		The library is otherwise FIFO-only; this is the explicit removal used
		when a watched audio file is deleted or renamed away, so its record does
		not linger as a selectable "ghost" that plays cached audio for a file
		that is gone.  Keyed by resolved filepath.  Returns the removed sample_id
		so the caller can cascade-clean similarity / transform state, exactly as
		for an eviction; returns None if no sample at that path is loaded.
		"""

		key = _resolve_path(path)

		with self._lock:
			sample_id = self._path_index.pop(key, None)

			if sample_id is None:
				return None

			self._id_to_key.pop(sample_id, None)
			record = self._index.pop(sample_id, None)

			if record is not None and record.audio is not None:
				self._total_bytes -= record.audio.nbytes

			try:
				self._order.remove(sample_id)
			except ValueError:
				pass

			return sample_id

	def find_by_name (self, name: str) -> typing.Optional[int]:

		"""Return the sample_id for a filepath-less sample with the given name.

		Name is the filename stem without extension.  This is a FALLBACK lookup
		for in-memory / test records that carry no filepath; on-disk samples are
		keyed by path (see find_by_path), and stems are no longer required to be
		unique, so use find_by_path to identify an on-disk sample.  Returns None
		if no filepath-less sample with that name is loaded.
		"""

		with self._lock:
			return self._name_index.get(name)

	def samples (self) -> list[SampleRecord]:

		"""Return all samples in insertion order (oldest first)."""

		with self._lock:
			return [self._index[sid] for sid in self._order if sid in self._index]

	@property
	def memory_used (self) -> int:

		"""Current total audio memory in bytes."""

		# Locked for consistency with the rest of the class — format_memory()
		# is called from the watcher-thread callback while add() mutates
		# _total_bytes on another thread.
		with self._lock:
			return self._total_bytes

	@property
	def memory_limit (self) -> int:

		"""Configured maximum audio memory in bytes."""

		return self._max_bytes

	def format_memory (self) -> str:

		"""Return a human-readable memory usage string for logging.

		Example: '45.3 / 100.0 MB, 54% free'
		"""

		with self._lock:
			total = self._total_bytes

		used_mb  = total / (1024 * 1024)
		limit_mb = self._max_bytes / (1024 * 1024)
		pct_free = int(100 * (1.0 - total / self._max_bytes)) if self._max_bytes > 0 else 100
		return f"{used_mb:.1f} / {limit_mb:.1f} MB, {pct_free}% free"

	def __len__ (self) -> int:

		return len(self._index)

	def __repr__ (self) -> str:

		used_mb = self._total_bytes / (1024 * 1024)
		limit_mb = self._max_bytes / (1024 * 1024)
		return (
			f"InstrumentLibrary({len(self)} sample(s), "
			f"{used_mb:.1f}/{limit_mb:.1f} MB)"
		)


def load_reference_library (directory: pathlib.Path) -> ReferenceLibrary:

	"""Discover and load all .analysis.json sidecars in a directory.

	Scans the top level of directory for files ending in '.analysis.json'.
	Each sidecar is loaded via cache.load_sidecar() (version-only validation;
	audio file need not be present). Invalid or version-mismatched sidecars
	are skipped with a WARNING log.

	Each loaded record is assigned a session-unique ID.

	Logs the count of successfully loaded samples at INFO level.

	Args:
		directory: Path to search for .analysis.json sidecar files.

	Returns:
		ReferenceLibrary containing all successfully loaded records.
		If the directory does not exist or is empty, returns an empty library.
	"""

	if not directory.exists():
		_log.warning("Reference directory not found: %s — library will be empty", directory)
		return ReferenceLibrary([])

	sidecar_paths = sorted(directory.glob(f"*{subsample.cache.SIDECAR_SUFFIX}"))

	if not sidecar_paths:
		return ReferenceLibrary([])

	n_workers = max(1, ((os.cpu_count() or 1) - 2) // 2)

	# Phase 1 — parallel: load each sidecar concurrently (may trigger
	# re-analysis if the version is stale). Results are kept in sorted order
	# via the futures list so Phase 2 builds records in a deterministic sequence.
	with concurrent.futures.ThreadPoolExecutor(
		max_workers=n_workers,
		thread_name_prefix="ref-loader",
	) as executor:
		futures = [
			executor.submit(subsample.cache.load_sidecar, path)
			for path in sidecar_paths
		]
		raw_results = [f.result() for f in futures]

	# Phase 2 — sequential: construct SampleRecords in sorted filename order.
	records: list[SampleRecord] = []

	for sidecar_path, result in zip(sidecar_paths, raw_results):
		if result is None:
			continue

		audio_name = sidecar_path.name[: -len(subsample.cache.SIDECAR_SUFFIX)]
		name = pathlib.Path(audio_name).stem

		records.append(SampleRecord(
			sample_id      = allocate_id(),
			name           = name,
			spectral       = result.spectral,
			rhythm         = result.rhythm,
			pitch          = result.pitch,
			timbre         = result.timbre,
			level          = result.level,
			band_energy    = result.band_energy,
			params         = result.params,
			duration       = result.duration,
			audio          = None,
			filepath       = None,
			channel_format = result.channel_format,
			loop           = result.loop,
		))

	_log.info("Loaded %d reference sample(s) from %s", len(records), directory)

	return ReferenceLibrary(records)


@dataclasses.dataclass(frozen=True)
class _LoadedSample:

	"""Intermediate result from _load_one_sample(); holds all data needed to
	build a SampleRecord once the parallel phase is complete.

	The fields below (including ``audio`` and ``audio_path``) are produced in
	the parallel worker phase.  ``SampleRecord.sample_id`` and ``filepath`` are
	assigned later, in the sequential Phase 2, where allocate_id() is called in
	sorted order to preserve FIFO semantics — this dataclass itself carries
	neither.
	"""

	spectral:    subsample.analysis.AnalysisResult
	rhythm:      subsample.analysis.RhythmResult
	pitch:       subsample.analysis.PitchResult
	timbre:      subsample.analysis.TimbreResult
	level:       subsample.analysis.LevelResult
	band_energy: subsample.analysis.BandEnergyResult
	params:      subsample.analysis.AnalysisParams
	duration:    float
	name:        str
	audio_path:  pathlib.Path
	audio:       typing.Optional[numpy.ndarray]
	channel_format: str = "pcm"
	loop:        typing.Optional[subsample.loopfind.LoopPoints] = None


def _sweep_orphans (directory: pathlib.Path) -> None:

	"""Recursively delete .analysis.json and .preview.png sidecars whose
	audio counterpart is absent in the same directory — routine housekeeping
	after a sample is removed or renamed.

	Subsample only writes these two compound suffixes itself, so the sweep
	never touches user-created files: a `.preview.png` named like ours but
	written by a third-party tool would still be deleted, but that's the
	same name collision the user is responsible for avoiding.

	ONE exception: a directory named ``reference`` is left untouched, because
	its fingerprints are audio-less BY DESIGN — subsample ships them and
	`subsample --init` scaffolds them into ``samples/reference/``, where the
	similarity engine loads them by path with no WAV ever present.  The check
	is scoped to the swept tree (a ``reference`` folder ABOVE the project can't
	accidentally exempt everything), and covers both the case where the swept
	root itself is the reference directory and where it is a subdirectory of a
	broader ``library.directory``.

	Failures (e.g. permission denied) are logged at ERROR and skipped;
	never aborts the wider library load.
	"""

	for path in directory.rglob("*"):
		if not path.is_file():
			continue

		name = path.name

		# Crash litter: a save that was SIGKILLed (or lost power) between mkstemp
		# and the atomic rename leaves a `<sidecar>.tmpXXXX` file behind.  Clean
		# it wherever it appears — it is never a valid fingerprint.
		if ".analysis.json.tmp" in name or ".preview.png.tmp" in name:
			try:
				path.unlink()
				_log.info("Deleted crash-leftover temp file %s", path)
			except OSError as exc:
				_log.error("Failed to delete temp file %s: %s", path, exc)
			continue

		if name.endswith(subsample.cache.SIDECAR_SUFFIX):
			audio_name = name[: -len(subsample.cache.SIDECAR_SUFFIX)]
		elif name.endswith(subsample.cache.PREVIEW_PNG_SUFFIX):
			audio_name = name[: -len(subsample.cache.PREVIEW_PNG_SUFFIX)]
		else:
			continue

		audio_path = path.parent / audio_name

		if audio_path.exists():
			continue

		# Leave the curated reference directory alone (audio-less by design).
		enclosing_dirs = path.relative_to(directory).parts[:-1]
		if directory.name == "reference" or "reference" in enclosing_dirs:
			continue

		try:
			path.unlink()
			_log.info("Deleted orphaned %s (audio %s not found)", path, audio_name)
		except OSError as exc:
			_log.error("Failed to delete orphan %s: %s", path, exc)


def _load_one_sample (
	audio_path: pathlib.Path,
	load_audio: bool,
	with_preview: bool,
	target_sample_rate: typing.Optional[int] = None,
) -> typing.Optional[_LoadedSample]:

	"""Load one instrument sample (audio + sidecar) from disk.

	Designed to run on a worker thread.  Each call is fully independent
	(separate audio read, separate re-analysis if needed), so multiple
	calls can safely execute concurrently.

	Args:
		audio_path:         Path to the audio file (the sidecar/PNG are derived
		                    from this — the audio is the source of truth).
		load_audio:         When True, also load PCM data into the result for
		                    playback.  False leaves audio=None to save memory.
		with_preview:       Threaded through to ``ensure_sample_assets`` so the
		                    PNG sidecar and embedded preview block stay in
		                    sync with the audio when previews are enabled.
		target_sample_rate: When set, resample audio to this rate on load
		                    (soxr_hq quality).  None keeps the native rate.

	Returns a _LoadedSample on success, or None if any step fails (the
	reason will have already been logged by the callee).
	"""

	result = subsample.cache.ensure_sample_assets(audio_path, with_preview=with_preview)

	if result is None:
		return None

	name = audio_path.stem

	if load_audio:
		audio: typing.Optional[numpy.ndarray] = load_wav_audio(audio_path, target_sample_rate)

		if audio is None:
			return None
	else:
		audio = None

	return _LoadedSample(
		spectral=result.spectral, rhythm=result.rhythm, pitch=result.pitch, timbre=result.timbre,
		level=result.level, band_energy=result.band_energy, params=result.params, duration=result.duration,
		name=name, audio_path=audio_path, audio=audio,
		channel_format=result.channel_format, loop=result.loop,
	)


def load_instrument_library (
	directory: pathlib.Path,
	max_memory_bytes: int,
	*,
	with_preview: bool,
	load_audio: bool = True,
	target_sample_rate: typing.Optional[int] = None,
) -> InstrumentLibrary:

	"""Discover and load instrument samples from ``directory`` (recursive).

	Walks the directory tree audio-first: every supported audio file becomes
	a candidate sample, with its ``.analysis.json`` sidecar and (when
	``with_preview=True``) ``.preview.png`` regenerated on the fly if missing
	or stale.  Before the walk, an orphan sweep removes any sidecar or PNG
	whose audio counterpart is absent — keeping the directory tidy after the
	user renames or moves files outside subsample.

	Samples are added in lexicographic path order; the memory limit is
	respected using FIFO eviction.  Each loaded record is assigned a
	session-unique ID.

	Samples are identified by resolved filepath, so two files that share a
	filename stem in different subdirectories (e.g. per-technique take-folders
	each containing ``01.wav``) load as distinct samples.  A ``name:`` predicate
	then matches all of them; combine it with ``directory:`` (or ``pick``) to
	select a specific one.

	Args:
		directory:          Root of the recursive sample search.
		max_memory_bytes:   Memory limit passed to the returned InstrumentLibrary.
		with_preview:       Threaded from ``cfg.recorder.previews``; when False
		                    the orchestrator skips PNG generation and the
		                    embedded preview block (orphan PNGs are still
		                    swept unconditionally).
		load_audio:         When True (default), load PCM data into each
		                    record's audio field.  Set False to load metadata
		                    only (e.g. for an analysis-only run).
		target_sample_rate: When set, resample on load to this rate.

	Returns:
		InstrumentLibrary containing all successfully loaded samples.  If the
		directory does not exist, returns an empty library.
	"""

	lib = InstrumentLibrary(max_memory_bytes)

	if not directory.exists():
		_log.warning("Instrument directory not found: %s — library will be empty", directory)
		return lib

	# Tidy up before working.  Orphan sweep runs unconditionally so the
	# directory state at the end of load reflects exactly the audio files
	# present at the start — no stale sidecar/PNG ghosts.
	_sweep_orphans(directory)

	audio_paths = sorted(
		p for p in directory.rglob("*")
		if p.is_file() and p.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS
	)

	if not audio_paths:
		return lib

	# Worker count: same formula as SampleProcessor — reserve 2 cores for
	# audio threads, use half the remainder. At least 1 always.
	n_workers = max(1, ((os.cpu_count() or 1) - 2) // 2)

	_log.info(
		"Loading %d instrument sample(s) from %s using %d worker(s)…",
		len(audio_paths), directory, n_workers,
	)

	# Phase 1 — parallel: load each sample (sidecar regen + audio read) in
	# its own worker.  Each file is fully independent so parallelism is
	# safe.  Results are returned in submission order so Phase 2 can add
	# records deterministically.
	with concurrent.futures.ThreadPoolExecutor(
		max_workers=n_workers,
		thread_name_prefix="lib-loader",
	) as executor:
		futures = [
			executor.submit(
				_load_one_sample, path, load_audio, with_preview,
				target_sample_rate,
			)
			for path in audio_paths
		]
		# Block until all workers finish; results arrive in submitted order.
		raw_results = [f.result() for f in futures]

	# Phase 2 — sequential: construct SampleRecords and add to the library in
	# sorted path order. allocate_id() is called here (not in workers) so
	# IDs are assigned in a deterministic order and FIFO eviction works correctly.
	# Records are keyed by resolved filepath (unique per file), so two files
	# that share a stem in different folders load as distinct samples — there is
	# no collision to reject (rglob yields each path exactly once).
	loaded = 0

	for loaded_sample in raw_results:
		if loaded_sample is None:
			continue

		record = SampleRecord(
			sample_id      = allocate_id(),
			name           = loaded_sample.name,
			spectral       = loaded_sample.spectral,
			rhythm         = loaded_sample.rhythm,
			pitch          = loaded_sample.pitch,
			timbre         = loaded_sample.timbre,
			level          = loaded_sample.level,
			band_energy    = loaded_sample.band_energy,
			params         = loaded_sample.params,
			duration       = loaded_sample.duration,
			audio          = loaded_sample.audio,
			filepath       = loaded_sample.audio_path,
			channel_format = loaded_sample.channel_format,
			loop           = loaded_sample.loop,
			# Audio was resampled to target_sample_rate on load (or kept at its
			# native analysis rate when no target was requested).
			audio_sample_rate = target_sample_rate or loaded_sample.params.sample_rate,
		)

		lib.add(record)
		loaded += 1

	# A colliding take-folder tree can hold many same-stem files; their combined
	# audio may exceed the memory limit, and FIFO eviction then silently drops
	# earliest-loaded samples during the walk.  Warn only when eviction ACTUALLY
	# dropped a loaded sample (fewer resident than loaded) — a single sample that
	# alone exceeds the budget is kept resident (add()'s "added anyway" path) and
	# must not trigger this.  The per-sample over-budget notice in add() is separate.
	if lib._max_bytes > 0 and len(lib) < loaded:
		total_audio_bytes = sum(
			s.audio.nbytes for s in raw_results
			if s is not None and s.audio is not None
		)
		_log.warning(
			"Instrument library: %d sample(s) totalling %.1f MB exceed the memory "
			"limit of %.1f MB — %d were evicted (FIFO), so some are unavailable at "
			"note-on.  Raise instrument.max_memory_mb to keep them all resident.",
			loaded, total_audio_bytes / (1024 * 1024), lib._max_bytes / (1024 * 1024),
			loaded - len(lib),
		)

	_log.info("Loaded %d instrument sample(s) from %s", loaded, directory)

	return lib


def load_wav_audio (
	path: pathlib.Path,
	target_sample_rate: typing.Optional[int] = None,
) -> typing.Optional[numpy.ndarray]:

	"""Read a WAV file into a numpy array matching the capture pipeline format.

	Returns an array of shape (n_frames, channels) using the dtype that matches
	the capture pipeline (int16 for 16-bit, left-shifted int32 for 24-bit, int32
	for 32-bit). Returns None and logs a WARNING on any read error.

	When target_sample_rate is set and differs from the file's native rate,
	the audio is resampled via librosa (soxr_hq quality) so that
	in-memory audio is always at the output device rate.

	Delegates to subsample.audio.read_audio_file() for the actual reading.
	"""

	try:
		info = subsample.audio.read_audio_file(path)
	except (wave.Error, OSError, ValueError) as exc:
		_log.warning("Could not read audio from %s: %s", path.name, exc)
		return None

	audio = info.audio

	if target_sample_rate is not None and info.sample_rate != target_sample_rate:
		_log.debug(
			"Resampling %s from %d Hz to %d Hz",
			path.name, info.sample_rate, target_sample_rate,
		)

		original_dtype = audio.dtype

		# Convert to float32 for high-quality resampling.
		if original_dtype == numpy.int16:
			float_audio = audio.astype(numpy.float32) / 32768.0
		elif original_dtype == numpy.int32:
			float_audio = audio.astype(numpy.float32) / 2147483648.0
		else:
			float_audio = audio.astype(numpy.float32)

		# librosa.resample expects (channels, n_frames) — transpose.
		resampled = librosa.resample(
			float_audio.T,
			orig_sr=info.sample_rate,
			target_sr=target_sample_rate,
			res_type="soxr_hq",
		).T.astype(numpy.float32)

		# Convert back to original integer dtype.
		if original_dtype == numpy.int16:
			# 32767 is exactly representable in float32, so the clip ceiling
			# genuinely caps here — no promotion needed.
			audio = numpy.clip(resampled * 32768.0, -32768, 32767).astype(numpy.int16)
		elif original_dtype == numpy.int32:
			# Promote to float64 for the scale+clip+cast.  In float32 the ceiling
			# 2147483647 rounds UP to 2^31, so a full-scale sample that overshoots
			# on resampling (Gibbs ringing pushes the peak past 1.0) survives the
			# clip as 2^31 and then overflows int32 on the cast — warning aside,
			# the peak wraps to the full-negative rail (an audible click).  float64
			# represents 2147483647 exactly, so the clip caps and the cast is in
			# range.  Mirrors audio.float32_to_pcm_bytes's 24-bit path.
			audio = numpy.clip(
				resampled.astype(numpy.float64) * 2147483648.0, -2147483648, 2147483647,
			).astype(numpy.int32)
		else:
			audio = resampled

	return audio
