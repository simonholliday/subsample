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
session-unique numeric ID allocated by _allocate_id().

Name derivation (both libraries):
  BD0025.WAV.analysis.json  →  audio file "BD0025.WAV"  →  name "BD0025"
  kick.wav.analysis.json    →  audio file "kick.wav"     →  name "kick"

ReferenceLibrary lookup is case-insensitive.
"""

import collections
import dataclasses
import itertools
import logging
import pathlib
import threading
import typing
import wave

import subsample.audio

import numpy

import subsample.analysis
import subsample.cache


_log = logging.getLogger(__name__)

_SIDECAR_SUFFIX: str = ".analysis.json"

# Session-unique ID counter. itertools.count is thread-safe (C extension;
# next() is atomic under the GIL), so callbacks on the writer thread and
# startup loading on the main thread can both call _allocate_id() safely.
_id_counter: "itertools.count[int]" = itertools.count(1)


def _allocate_id () -> int:

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
		sample_id: Session-unique numeric ID (allocated by _allocate_id()).
		name:      Stem of the audio filename (e.g. "BD0025", "kick").
		           Preserves original casing from the filename.
		spectral:  Nine normalised [0, 1] spectral metrics (the spectral fingerprint).
		rhythm:    Tempo, beat grid, pulse curve, onset times.
		pitch:     Fundamental frequency, chroma profile, MFCC timbre vector.
		params:    FFT parameters used when the analysis was computed.
		duration:  Recording length in seconds.
		audio:     Original capture-format PCM as a numpy array, shape
		           (n_frames, channels), dtype int16 or int32. None for reference
		           samples where only metadata is needed.
		filepath:  Path to the WAV file on disk, if known. None for in-memory-only
		           samples not yet (or never) written to disk.
	"""

	sample_id: int
	name:      str
	spectral:  subsample.analysis.AnalysisResult
	rhythm:    subsample.analysis.RhythmResult
	pitch:     subsample.analysis.PitchResult
	params:    subsample.analysis.AnalysisParams
	duration:  float
	audio:     typing.Optional[numpy.ndarray] = None
	filepath:  typing.Optional[pathlib.Path]  = None

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
		Each record is assigned a session-unique ID via _allocate_id().
		"""

		# Store records keyed by uppercased name for O(1) case-insensitive lookup.
		# Original-cased names are preserved in each record.
		self._index: dict[str, SampleRecord] = {r.name.upper(): r for r in records}

	def get (self, name: str) -> SampleRecord | None:

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
					evicted.append(oldest_id)

			self._index[record.sample_id] = record
			self._order.append(record.sample_id)
			self._total_bytes += sample_bytes

		return evicted

	def get (self, sample_id: int) -> SampleRecord | None:

		"""Return the sample with the given ID, or None if not present."""

		with self._lock:
			return self._index.get(sample_id)

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

	records: list[SampleRecord] = []

	for sidecar_path in sorted(directory.glob(f"*{_SIDECAR_SUFFIX}")):
		result = subsample.cache.load_sidecar(sidecar_path)

		if result is None:
			# load_sidecar already logged the reason
			continue

		spectral, rhythm, pitch, params, duration = result

		# Derive name: strip ".analysis.json" → audio filename → strip extension
		audio_name = sidecar_path.name[: -len(_SIDECAR_SUFFIX)]
		name = pathlib.Path(audio_name).stem

		records.append(SampleRecord(
			sample_id = _allocate_id(),
			name      = name,
			spectral  = spectral,
			rhythm    = rhythm,
			pitch     = pitch,
			params    = params,
			duration  = duration,
			audio     = None,
			filepath  = None,
		))

	_log.info("Loaded %d reference sample(s) from %s", len(records), directory)

	return ReferenceLibrary(records)


def load_instrument_library (
	directory: pathlib.Path,
	max_memory_bytes: int,
) -> InstrumentLibrary:

	"""Discover and load instrument samples (WAV + sidecar) from a directory.

	Like load_reference_library(), but also loads the audio data from each WAV
	file. Unlike reference samples, instrument samples require the WAV to be
	present — sidecars without a matching WAV are skipped with a WARNING.

	Scans the top level of directory only (non-recursive). Samples are added in
	sorted filename order; the memory limit is respected using FIFO eviction.

	Each loaded record is assigned a session-unique ID.

	Args:
		directory:        Path to search for .analysis.json sidecar files.
		max_memory_bytes: Memory limit passed to the returned InstrumentLibrary.

	Returns:
		InstrumentLibrary containing all successfully loaded samples.
		If the directory does not exist or is empty, returns an empty library.
	"""

	lib = InstrumentLibrary(max_memory_bytes)

	if not directory.exists():
		_log.warning("Instrument directory not found: %s — library will be empty", directory)
		return lib

	loaded = 0

	for sidecar_path in sorted(directory.glob(f"*{_SIDECAR_SUFFIX}")):
		result = subsample.cache.load_sidecar(sidecar_path)

		if result is None:
			# load_sidecar already logged the reason
			continue

		spectral, rhythm, pitch, params, duration = result

		# Derive the audio filename from the sidecar name
		audio_name = sidecar_path.name[: -len(_SIDECAR_SUFFIX)]
		audio_path = sidecar_path.parent / audio_name
		name = pathlib.Path(audio_name).stem

		# Instrument samples require audio — skip if WAV is absent
		if not audio_path.exists():
			_log.warning(
				"Skipping instrument sample %s — audio file not found: %s",
				name, audio_path,
			)
			continue

		audio = _load_wav_audio(audio_path)

		if audio is None:
			# _load_wav_audio already logged the reason
			continue

		record = SampleRecord(
			sample_id = _allocate_id(),
			name      = name,
			spectral  = spectral,
			rhythm    = rhythm,
			pitch     = pitch,
			params    = params,
			duration  = duration,
			audio     = audio,
			filepath  = audio_path,
		)

		lib.add(record)
		loaded += 1

	_log.info("Loaded %d instrument sample(s) from %s", loaded, directory)

	return lib


def _load_wav_audio (path: pathlib.Path) -> numpy.ndarray | None:

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
