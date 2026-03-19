"""Analyze one or more audio files and print their metrics to the console.

Reads any audio file supported by soundfile (WAV, FLAC, AIFF, OGG, etc.),
runs the same analysis pipeline used during live capture, and prints three
summary lines per file: rhythm, spectral, and pitch metrics.

Results are cached as a JSON sidecar file (<audio-file>.analysis.json) so
that repeated analysis of the same file is instant. The cache is
automatically invalidated if the audio file changes or the analysis
algorithm is updated.

Usage:
	python scripts/analyze_file.py <path/to/file.wav>
	python scripts/analyze_file.py ./reference/*.wav
	python scripts/analyze_file.py kick.wav snare.wav hat.wav
"""

import glob
import logging
import pathlib
import sys

import numpy
import soundfile

import subsample.analysis
import subsample.cache
import subsample.config


logging.basicConfig(
	level=logging.WARNING,
	format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
	datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)


def _analyze_file (filepath_path: pathlib.Path) -> None:

	"""Analyze a single audio file and print its metrics."""

	# Try the cache first — skips CPU-intensive analysis if nothing has changed
	cached = subsample.cache.load_cache(filepath_path)

	if cached is not None:
		result, rhythm, pitch, params, duration = cached

	else:
		try:
			# soundfile.read with dtype='float32' reads directly as float32, avoiding
			# the default float64 intermediate; always_2d ensures shape is
			# (n_frames, channels) regardless of mono/stereo.
			data, samplerate = soundfile.read(str(filepath_path), always_2d=True, dtype='float32')

		except (OSError, soundfile.SoundFileError) as exc:
			print(f"Error reading {filepath_path}: {exc}", file=sys.stderr)
			return

		# Mix down to mono — analysis functions expect shape (n_frames,)
		mono = numpy.mean(data, axis=1, dtype=numpy.float32)  # type: ignore[call-overload]

		params = subsample.analysis.compute_params(samplerate)

		# Use default analysis config (same as live capture defaults)
		rhythm_cfg = subsample.config.AnalysisConfig()

		# Run all three analyses; analyze_all() shares the pyin computation
		# between spectral and pitch, avoiding ~200-300 ms of redundant work.
		result, rhythm, pitch = subsample.analysis.analyze_all(mono, params, rhythm_cfg)

		duration = len(data) / samplerate

		# Save results for next time; log but don't fail if the filesystem is read-only
		try:
			audio_md5 = subsample.cache.compute_audio_md5(filepath_path)
			subsample.cache.save_cache(
				audio_path = filepath_path,
				audio_md5  = audio_md5,
				params     = params,
				spectral   = result,
				rhythm     = rhythm,
				pitch      = pitch,
				duration   = duration,
			)
		except OSError as exc:
			_log.warning("Could not save analysis cache for %s: %s", filepath_path.name, exc)

	print(f"rhythm:   {subsample.analysis.format_rhythm_result(rhythm)}")
	print(f"spectral: {subsample.analysis.format_result(result, duration)}")
	print(f"pitch:    {subsample.analysis.format_pitch_result(pitch)}")


def main () -> None:

	"""Analyze one or more audio files and print their metrics."""

	if not sys.argv[1:]:
		print("Usage: analyze_file <audio-file> [<audio-file> ...]", file=sys.stderr)
		sys.exit(1)

	# Expand each argument with glob so quoted wildcards work (e.g. "*.wav").
	# If an argument contains glob metacharacters but matches nothing, report
	# it immediately — the literal string is not a valid file path and soundfile
	# would produce a cryptic "System error" message.
	# If there are no metacharacters, treat it as a literal path so that the
	# normal "file not found" error is produced by the audio reader.
	_GLOB_CHARS = frozenset("*?[")

	paths: list[pathlib.Path] = []
	for arg in sys.argv[1:]:
		matches = sorted(glob.glob(arg))

		if matches:
			paths.extend(pathlib.Path(m) for m in matches)
		elif any(c in arg for c in _GLOB_CHARS):
			print(f"No files matched: {arg}", file=sys.stderr)
		else:
			paths.append(pathlib.Path(arg))

	if not paths:
		sys.exit(1)

	multi = len(paths) > 1

	for filepath_path in paths:
		if multi:
			print(f"\nAnalyzing {filepath_path.name} ...")

		_analyze_file(filepath_path)


if __name__ == "__main__":
	main()
