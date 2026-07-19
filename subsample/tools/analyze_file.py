"""Analyze one or more audio files and print their metrics to the console.

Reads any audio file supported by soundfile (WAV, FLAC, AIFF, OGG, etc.),
runs the same analysis pipeline used during live capture, and prints three
summary lines per file: rhythm, spectral, and pitch metrics.

Results are cached as a JSON sidecar file (<audio-file>.analysis.json) so
that repeated analysis of the same file is instant. The cache is
automatically invalidated if the audio file changes or the analysis
algorithm is updated.

Usage:
	subsample analyze <path/to/file.wav>
	subsample analyze ./reference/*.wav
	subsample analyze kick.wav snare.wav hat.wav
"""

import argparse
import glob
import logging
import pathlib
import sys
import typing

import numpy
import soundfile

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.config
import subsample.tools._shared


_log = logging.getLogger(__name__)


def _analyze_file (
	filepath: pathlib.Path,
	rhythm_cfg: subsample.config.AnalysisConfig,
) -> bool:

	"""Analyze a single audio file and print its metrics.

	Returns True on success, False if the file could not be read or analysed
	(so the caller can exit non-zero when every input failed)."""

	# Try the cache first — skips CPU-intensive analysis if nothing has changed
	cached = subsample.cache.load_cache(filepath)

	if cached is not None:
		result      = cached.spectral
		rhythm      = cached.rhythm
		pitch       = cached.pitch
		timbre      = cached.timbre
		params      = cached.params
		duration    = cached.duration
		level       = cached.level
		band_energy = cached.band_energy
		loop        = cached.loop

	else:
		# Hash the file BEFORE decoding and analysing it: hashing afterwards
		# would pair the analysis of the old bytes with an MD5 of whatever the
		# file became if it were overwritten mid-analysis — a permanently wrong
		# sidecar that never self-heals (cache._reanalyze_and_save documents the
		# same hash-first rule).
		try:
			audio_md5 = subsample.cache.compute_audio_md5(filepath)

		except OSError as exc:
			print(f"Error reading {filepath}: {exc}", file=sys.stderr)
			return False

		# Read through read_audio_file (not a raw soundfile.read) so this matches
		# the cache/player pipeline exactly: hot float/double sources are scaled
		# to the import ceiling, and the sidecar we write describes the audio the
		# player will actually read and play (a raw float read would poison the
		# cache with metrics 1+ dB louder than playback).
		try:
			file_info = subsample.audio.read_audio_file(filepath)

		except (OSError, ValueError) as exc:
			print(f"Error reading {filepath}: {exc}", file=sys.stderr)
			return False

		mono = subsample.analysis.to_mono_float(file_info.audio, file_info.bit_depth)
		samplerate = file_info.sample_rate

		params = subsample.analysis.compute_params(samplerate)

		# Run all three analyses; analyze_all() shares the pyin computation
		# between spectral and pitch, avoiding ~200-300 ms of redundant work.
		# A librosa failure on degenerate input skips this one file cleanly.
		try:
			result, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(mono, params, rhythm_cfg)
		except Exception as exc:
			print(f"  {filepath.name}  (skipped, could not analyze: {exc})", file=sys.stderr)
			return False

		duration = len(mono) / samplerate

		loop = subsample.cache.compute_loop(mono, samplerate, result, pitch, level, duration)

		# Save results for next time; log but don't fail if the filesystem is read-only
		try:
			subsample.cache.save_cache(
				audio_path = filepath,
				audio_md5  = audio_md5,
				params     = params,
				spectral   = result,
				rhythm     = rhythm,
				pitch      = pitch,
				timbre     = timbre,
				duration   = duration,
				level      = level,
				band_energy = band_energy,
				loop       = loop,
			)
		except OSError as exc:
			_log.warning("Could not save analysis cache for %s: %s", filepath.name, exc)

	print(f"rhythm:   {subsample.analysis.format_rhythm_result(rhythm)}")
	print(f"spectral: {subsample.analysis.format_result(result, duration)}")
	print(f"pitch:    {subsample.analysis.format_pitch_result(pitch)}")
	print(f"level:    {subsample.analysis.format_level_result(level)}")
	print(f"noisiness: {subsample.analysis.noisiness(result, level):.3f}  (0 = clean event, 1 = wall-to-wall noise)")

	if loop is not None:
		sr = params.sample_rate
		print(
			f"loop:     {loop.start / sr:.3f}s -> {loop.end / sr:.3f}s "
			f"({(loop.end - loop.start) / sr * 1000:.0f} ms, xfade {loop.crossfade / sr * 1000:.0f} ms, "
			f"junction_flux {loop.junction_flux:.2f})"
		)
	else:
		print("loop:     none (not a loop candidate, or no clean junction)")

	return True


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Analyze one or more audio files and print their metrics."""

	subsample.tools._shared.configure_logging()

	parser = argparse.ArgumentParser(
		prog="subsample analyze",
		description="Analyze audio files and print their detected metrics (rhythm, spectral, pitch, level, loop).",
	)
	parser.add_argument(
		"files",
		nargs="+",
		metavar="FILE",
		help="Audio files or quoted glob patterns to analyze",
	)
	parser.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	args = parser.parse_args(argv)

	# Wire the float ceiling and analysis tempo priors from config: analyze
	# writes a sidecar the player later trusts, so it must analyse at the same
	# scale and tuning the app itself would use.
	cfg = subsample.tools._shared.load_config_and_wire(args.config)

	# Expand each argument with glob so quoted wildcards work (e.g. "*.wav").
	# If an argument contains glob metacharacters but matches nothing, report
	# it immediately — the literal string is not a valid file path and soundfile
	# would produce a cryptic "System error" message.
	# If there are no metacharacters, treat it as a literal path so that the
	# normal "file not found" error is produced by the audio reader.
	_GLOB_CHARS = frozenset("*?[")

	paths: list[pathlib.Path] = []
	for arg in args.files:
		matches = sorted(glob.glob(arg))

		if matches:
			paths.extend(pathlib.Path(m) for m in matches)
		elif any(c in arg for c in _GLOB_CHARS):
			print(f"No files matched: {arg}", file=sys.stderr)
		else:
			paths.append(pathlib.Path(arg))

	if not paths:
		return 1

	multi = len(paths) > 1
	any_ok = False

	for filepath in paths:
		if multi:
			print(f"\nAnalyzing {filepath.name} ...")

		if _analyze_file(filepath, cfg.analysis):
			any_ok = True

	# Every input unreadable/unanalysable → non-zero exit for scripts/pipelines.
	return 0 if any_ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
