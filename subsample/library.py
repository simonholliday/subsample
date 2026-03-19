"""Reference sample library for Subsample.

Loads pre-analyzed canonical sounds (kick, snare, hi-hat, …) from a directory
of .analysis.json sidecar files into an in-memory index. The audio files do
not need to be present — only the sidecar is required.

Names are derived from the audio filename stem:
  BD0025.WAV.analysis.json  →  audio file "BD0025.WAV"  →  name "BD0025"
  kick.wav.analysis.json    →  audio file "kick.wav"     →  name "kick"

Lookup is case-insensitive: get("KICK") and get("kick") return the same record.
Names are stored with their original casing; the index key is uppercased.

This library is the foundation for similarity scoring: SampleRecord contains
all the analysis results needed to compute cosine distance between an incoming
sample and each reference sound.
"""

import dataclasses
import logging
import pathlib

import subsample.analysis
import subsample.cache


_log = logging.getLogger(__name__)

_SIDECAR_SUFFIX: str = ".analysis.json"


@dataclasses.dataclass(frozen=True)
class SampleRecord:

	"""All analysis data for a single reference sample, held in memory.

	Contains the same three result types produced by analyze_all(), plus the
	FFT params and duration. No audio data is stored — only the derived metrics.

	Fields:
		name:     Stem of the audio filename (e.g. "BD0025", "kick").
		          Preserves original casing from the filename.
		spectral: Nine normalised [0, 1] spectral metrics.
		rhythm:   Tempo, beat grid, pulse curve, onset times.
		pitch:    Fundamental frequency, chroma profile, MFCC timbre vector.
		params:   FFT parameters used when the analysis was computed.
		duration: Recording length in seconds.
	"""

	name:     str
	spectral: subsample.analysis.AnalysisResult
	rhythm:   subsample.analysis.RhythmResult
	pitch:    subsample.analysis.PitchResult
	params:   subsample.analysis.AnalysisParams
	duration: float


class ReferenceLibrary:

	"""In-memory index of named reference samples.

	Records are keyed by uppercased name for case-insensitive lookup.
	All public methods preserve original-casing in returned values.
	"""

	def __init__ (self, records: list[SampleRecord]) -> None:

		"""Build the index from a list of SampleRecords.

		Use load_reference_library() rather than calling this directly.
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

	def all (self) -> list[SampleRecord]:

		"""Return all loaded sample records sorted by name (case-insensitive)."""

		return sorted(self._index.values(), key=lambda r: r.name.upper())

	def __len__ (self) -> int:

		return len(self._index)

	def __repr__ (self) -> str:

		names = ", ".join(self.names())
		return f"ReferenceLibrary({len(self)} sample(s): {names})"


def load_reference_library (directory: pathlib.Path) -> ReferenceLibrary:

	"""Discover and load all .analysis.json sidecars in a directory.

	Scans the top level of directory for files ending in '.analysis.json'.
	Each sidecar is loaded via cache.load_sidecar() (version-only validation;
	audio file need not be present). Invalid or version-mismatched sidecars
	are skipped with a WARNING log.

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
			name     = name,
			spectral = spectral,
			rhythm   = rhythm,
			pitch    = pitch,
			params   = params,
			duration = duration,
		))

	_log.info("Loaded %d reference sample(s) from %s", len(records), directory)

	return ReferenceLibrary(records)
