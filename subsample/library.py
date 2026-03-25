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

import numpy

import subsample.analysis
import subsample.audio
import subsample.cache


_log = logging.getLogger(__name__)

_SIDECAR_SUFFIX: str = ".analysis.json"

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


@dataclasses.dataclass(frozen=True)
class SampleRecord:

	"""A single sample held in memory, with required analysis data and optional audio.

	Used for both reference samples (audio=None, metadata only) and instrument
	samples (audio contains original-format PCM for playback).

	Fields:
		sample_id: Session-unique numeric ID (allocated by allocate_id()).
		name:      Stem of the audio filename (e.g. "BD0025", "kick").
		           Preserves original casing from the filename.
		spectral:    Eleven normalised [0, 1] spectral metrics (the spectral fingerprint).
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

	def as_vector (self) -> numpy.ndarray:

		"""Return the spectral fingerprint as a float32 1-D array.

		Delegates to spectral.as_vector() — convenience accessor so callers do
		not need to look up the nested spectral field for similarity comparison.
		"""

		return self.spectral.as_vector()


class ReferenceLibrary:

	"""In-memory index of named reference samples (metadata only, no audio).

	Records are keyed by uppercased name for case-insensitive lookup.
	All public methods preserve original-casing in returned values.
	"""

	def __init__ (self, records: list[SampleRecord]) -> None:

		"""Build the index from a list of SampleRecords.

		Use load_reference_library() rather than calling this directly.
		Each record is assigned a session-unique ID via allocate_id().
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
		# Secondary index for O(1) lookup by name (filename stem without extension).
		# Kept in sync with _index by add() and eviction.
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
			while self._order and self._total_bytes + sample_bytes > self._max_bytes:
				oldest_id = self._order.popleft()
				old_record = self._index.pop(oldest_id, None)

				if old_record is not None:
					old_bytes = old_record.audio.nbytes if old_record.audio is not None else 0
					self._total_bytes -= old_bytes
					self._name_index.pop(old_record.name, None)
					evicted.append(oldest_id)

			self._index[record.sample_id] = record
			self._name_index[record.name] = record.sample_id
			self._order.append(record.sample_id)
			self._total_bytes += sample_bytes

		return evicted

	def get (self, sample_id: int) -> SampleRecord | None:

		"""Return the sample with the given ID, or None if not present."""

		with self._lock:
			return self._index.get(sample_id)

	def find_by_name (self, name: str) -> int | None:

		"""Return the sample_id for the sample with the given name, or None if not present.

		Name is the filename stem without extension (e.g. "2026-03-23_10-04-07").
		Lookup is O(1) via a secondary index kept in sync with the main index.
		Returns None if the sample is not currently in the library (not yet loaded
		or evicted).
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

		return self._total_bytes

	@property
	def memory_limit (self) -> int:

		"""Configured maximum audio memory in bytes."""

		return self._max_bytes

	def format_memory (self) -> str:

		"""Return a human-readable memory usage string for logging.

		Example: '45.3 / 100.0 MB, 55% free'
		"""

		used_mb  = self._total_bytes / (1024 * 1024)
		limit_mb = self._max_bytes   / (1024 * 1024)
		pct_free = int(100 * (1.0 - self._total_bytes / self._max_bytes)) if self._max_bytes > 0 else 100
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

	sidecar_paths = sorted(directory.glob(f"*{_SIDECAR_SUFFIX}"))

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

		spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result

		audio_name = sidecar_path.name[: -len(_SIDECAR_SUFFIX)]
		name = pathlib.Path(audio_name).stem

		records.append(SampleRecord(
			sample_id   = allocate_id(),
			name        = name,
			spectral    = spectral,
			rhythm      = rhythm,
			pitch       = pitch,
			timbre      = timbre,
			level       = level,
			band_energy = band_energy,
			params      = params,
			duration    = duration,
			audio       = None,
			filepath    = None,
		))

	_log.info("Loaded %d reference sample(s) from %s", len(records), directory)

	return ReferenceLibrary(records)


@dataclasses.dataclass(frozen=True)
class _LoadedSample:

	"""Intermediate result from _load_one_sample(); holds all data needed to
	build a SampleRecord once the parallel phase is complete.

	sample_id and filepath are filled in during the sequential phase so that
	allocate_id() is called in sorted order, preserving FIFO semantics.
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


def _load_one_sample (
	sidecar_path: pathlib.Path,
	load_audio: bool,
	clean_orphaned_sidecars: bool,
	orphan_hint_shown: threading.Event,
) -> typing.Optional[_LoadedSample]:

	"""Load one instrument sample (sidecar + optional audio) from disk.

	Designed to run on a worker thread. Each file is fully independent, so
	multiple calls can safely execute concurrently.

	Args:
		sidecar_path:            Path to the .analysis.json sidecar file.
		load_audio:              When True, load PCM data from the WAV file.
		clean_orphaned_sidecars: When True, delete sidecars whose WAV is missing
		                         instead of just warning. When False, log a
		                         warning and a one-time hint about the option.
		orphan_hint_shown:       Shared event; set after the hint is logged so
		                         it appears at most once across all workers.

	Returns a _LoadedSample on success, or None if any step fails (the
	reason will have already been logged by the callee).
	"""

	result = subsample.cache.load_sidecar(sidecar_path)

	if result is None:
		return None

	spectral, rhythm, pitch, timbre, params, duration, level, band_energy = result

	audio_name = sidecar_path.name[: -len(_SIDECAR_SUFFIX)]
	audio_path = sidecar_path.parent / audio_name
	name = pathlib.Path(audio_name).stem

	if not audio_path.exists():

		if clean_orphaned_sidecars:
			try:
				sidecar_path.unlink()
				_log.info(
					"Deleted orphaned sidecar for %s — audio file not found: %s",
					name, audio_path,
				)
			except OSError as exc:
				_log.error(
					"Failed to delete orphaned sidecar %s: %s",
					sidecar_path, exc,
				)

		else:
			_log.warning(
				"Skipping instrument sample %s — audio file not found: %s",
				name, audio_path,
			)

			if not orphan_hint_shown.is_set():
				orphan_hint_shown.set()
				_log.info(
					"To automatically remove orphaned sidecars, set "
					"instrument.clean_orphaned_sidecars: true in config.yaml"
				)

		return None

	if load_audio:
		audio: typing.Optional[numpy.ndarray] = _load_wav_audio(audio_path)

		if audio is None:
			return None
	else:
		audio = None

	return _LoadedSample(
		spectral=spectral, rhythm=rhythm, pitch=pitch, timbre=timbre,
		level=level, band_energy=band_energy, params=params, duration=duration,
		name=name, audio_path=audio_path, audio=audio,
	)


def load_instrument_library (
	directory: pathlib.Path,
	max_memory_bytes: int,
	load_audio: bool = True,
	clean_orphaned_sidecars: bool = False,
) -> InstrumentLibrary:

	"""Discover and load instrument samples (WAV + sidecar) from a directory.

	Like load_reference_library(), but also loads the audio data from each WAV
	file (unless load_audio=False). Unlike reference samples, instrument samples
	require the WAV to be present — sidecars without a matching WAV are either
	skipped with a WARNING (default) or deleted (if clean_orphaned_sidecars is
	True).

	Scans the top level of directory only (non-recursive). Samples are added in
	sorted filename order; the memory limit is respected using FIFO eviction.

	Each loaded record is assigned a session-unique ID.

	Args:
		directory:                Path to search for .analysis.json sidecar files.
		max_memory_bytes:         Memory limit passed to the returned InstrumentLibrary.
		load_audio:               When True (default), load PCM data into each record's
		                          audio field. When False, audio is left as None to save
		                          memory — use this when playback is not required.
		clean_orphaned_sidecars:  When True, delete sidecars whose WAV is missing.
		                          When False (default), log a warning and skip.

	Returns:
		InstrumentLibrary containing all successfully loaded samples.
		If the directory does not exist or is empty, returns an empty library.
	"""

	lib = InstrumentLibrary(max_memory_bytes)

	if not directory.exists():
		_log.warning("Instrument directory not found: %s — library will be empty", directory)
		return lib

	sidecar_paths = sorted(directory.glob(f"*{_SIDECAR_SUFFIX}"))

	if not sidecar_paths:
		return lib

	# Worker count: same formula as SampleProcessor — reserve 2 cores for
	# audio threads, use half the remainder. At least 1 always.
	n_workers = max(1, ((os.cpu_count() or 1) - 2) // 2)

	_log.info(
		"Loading %d instrument sample(s) from %s using %d worker(s)…",
		len(sidecar_paths), directory, n_workers,
	)

	# Phase 1 — parallel: load sidecar + audio for each file concurrently.
	# Each file is fully independent (separate disk reads, separate re-analysis),
	# so parallelism is safe. Results are associated with their original sorted
	# position via the futures list so Phase 2 can add records in order.
	#
	# orphan_hint_shown is shared across workers so the one-time config hint
	# is emitted at most once regardless of how many orphans are found.
	orphan_hint_shown = threading.Event()

	with concurrent.futures.ThreadPoolExecutor(
		max_workers=n_workers,
		thread_name_prefix="lib-loader",
	) as executor:
		futures = [
			executor.submit(
				_load_one_sample, path, load_audio,
				clean_orphaned_sidecars, orphan_hint_shown,
			)
			for path in sidecar_paths
		]
		# Block until all workers finish; results arrive in submitted order.
		raw_results = [f.result() for f in futures]

	# Phase 2 — sequential: construct SampleRecords and add to the library in
	# sorted filename order. allocate_id() is called here (not in workers) so
	# IDs are assigned in a deterministic order and FIFO eviction works correctly.
	loaded = 0

	for loaded_sample in raw_results:
		if loaded_sample is None:
			continue

		record = SampleRecord(
			sample_id   = allocate_id(),
			name        = loaded_sample.name,
			spectral    = loaded_sample.spectral,
			rhythm      = loaded_sample.rhythm,
			pitch       = loaded_sample.pitch,
			timbre      = loaded_sample.timbre,
			level       = loaded_sample.level,
			band_energy = loaded_sample.band_energy,
			params      = loaded_sample.params,
			duration    = loaded_sample.duration,
			audio       = loaded_sample.audio,
			filepath    = loaded_sample.audio_path,
		)

		lib.add(record)
		loaded += 1

	_log.info("Loaded %d instrument sample(s) from %s", loaded, directory)

	return lib


def _load_wav_audio (path: pathlib.Path) -> typing.Optional[numpy.ndarray]:

	"""Read a WAV file into a numpy array matching the capture pipeline format.

	Returns an array of shape (n_frames, channels) using the dtype that matches
	the capture pipeline (int16 for 16-bit, left-shifted int32 for 24-bit, int32
	for 32-bit). Returns None and logs a WARNING on any read error.

	Delegates to subsample.audio.read_audio_file() for the actual reading.
	"""

	try:
		return subsample.audio.read_audio_file(path).audio

	except (wave.Error, OSError, ValueError) as exc:
		_log.warning("Could not read audio from %s: %s", path.name, exc)
		return None
