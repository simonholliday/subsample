"""Import pre-trimmed audio files into the Subsample capture library.

Reads audio files in any format supported by soundfile (WAV, BWF, FLAC, AIFF,
OGG, etc.), trims leading/trailing silence, applies safety fades to prevent
clicks, re-encodes as standard PCM WAV, runs the full analysis pipeline, and
saves a sidecar JSON alongside each imported file.

The target directory defaults to the configured output.directory from
config.yaml. Use --to to override.

Usage:
	python scripts/import_samples.py <path/to/file.wav> [...]
	python scripts/import_samples.py "/path/to/sample pack/*.wav"
	python scripts/import_samples.py --to samples/captures kick.wav snare.wav
	python scripts/import_samples.py --force --to samples/radio /mnt/sdr/audio/*.wav
"""

import glob
import logging
import math
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

# Silence threshold in float32 amplitude (~-48 dBFS).
# Samples below this are considered silence for trimming purposes.
_SILENCE_THRESHOLD: float = 0.004


def _trim_silence (audio: numpy.ndarray) -> numpy.ndarray:

	"""Trim leading and trailing silence from float32 audio.

	Finds the first and last sample whose absolute value meets or exceeds
	_SILENCE_THRESHOLD (any channel) and returns the slice between them.
	Returns the original array unchanged if no sample exceeds the threshold.

	Args:
		audio: Shape (n_frames, channels), dtype float32.

	Returns:
		Trimmed slice of audio.
	"""

	magnitude = numpy.max(numpy.abs(audio), axis=-1)
	above = numpy.where(magnitude >= _SILENCE_THRESHOLD)[0]

	if above.size == 0:
		return audio

	return audio[int(above[0]) : int(above[-1]) + 1]


def _apply_safety_fades (audio: numpy.ndarray, sample_rate: int) -> numpy.ndarray:

	"""Apply 1ms half-cosine fades at edges that don't start/end at zero.

	Checks whether the first or last sample (any channel) is non-zero. If so,
	applies an S-curve (half-cosine) fade over 1ms to prevent clicks. Edges
	already at zero are left untouched.

	Args:
		audio:       Shape (n_frames, channels), dtype float32.
		sample_rate: Audio sample rate in Hz.

	Returns:
		Audio with safety fades applied (may be a copy if fades were needed).
	"""

	n_frames = audio.shape[0]
	fade_len = max(2, int(0.001 * sample_rate))

	if n_frames < fade_len * 2:
		# Too short for separate fades — fade the whole thing
		fade_len = n_frames // 2

	needs_fade_in = numpy.max(numpy.abs(audio[0])) > 0.0
	needs_fade_out = numpy.max(numpy.abs(audio[-1])) > 0.0

	if not needs_fade_in and not needs_fade_out:
		return audio

	result = audio.copy()

	if needs_fade_in and fade_len > 1:
		ramp = ((1.0 - numpy.cos(numpy.linspace(0, numpy.pi, fade_len))) / 2.0).astype(numpy.float32)
		result[:fade_len] *= ramp[:, numpy.newaxis]

	if needs_fade_out and fade_len > 1:
		ramp = ((1.0 + numpy.cos(numpy.linspace(0, numpy.pi, fade_len))) / 2.0).astype(numpy.float32)
		result[-fade_len:] *= ramp[:, numpy.newaxis]

	return result


def _dbfs (value: float) -> str:

	"""Format a linear amplitude as dBFS."""

	if value <= 0.0:
		return "-infdBFS"

	return f"{20.0 * math.log10(value):.1f}dBFS"


def _resolve_subtype (info: soundfile._SoundFileInfo) -> str:

	"""Pick a PCM subtype that preserves the source bit depth.

	Falls back to PCM_16 for compressed or exotic formats.
	"""

	sub = info.subtype

	if sub in ("PCM_16", "PCM_24", "PCM_32"):
		return sub

	# Float sources get written as 16-bit PCM (the pipeline's native depth)
	return "PCM_16"


def _import_file (
	filepath: pathlib.Path,
	target_dir: pathlib.Path,
	force: bool,
) -> bool:

	"""Import a single audio file into the target directory.

	Returns True if the file was imported, False if skipped or failed.
	"""

	target_path = target_dir / (filepath.stem + ".wav")

	if target_path.exists() and not force:
		print(f"  {filepath.name}  (skipped, already exists)")
		return False

	# Read audio

	try:
		info = soundfile.info(str(filepath))
	except (OSError, soundfile.SoundFileError) as exc:
		print(f"  {filepath.name}  ERROR: {exc}", file=sys.stderr)
		return False

	try:
		data, samplerate = soundfile.read(str(filepath), always_2d=True, dtype="float32")
	except (OSError, soundfile.SoundFileError) as exc:
		print(f"  {filepath.name}  ERROR: {exc}", file=sys.stderr)
		return False

	if data.shape[0] == 0:
		print(f"  {filepath.name}  (skipped, empty file)")
		return False

	# Trim silence and apply safety fades

	trimmed = _trim_silence(data)

	if trimmed.shape[0] == 0:
		print(f"  {filepath.name}  (skipped, silence only)")
		return False

	faded = _apply_safety_fades(trimmed, samplerate)

	# Write as standard PCM WAV

	subtype = _resolve_subtype(info)

	try:
		soundfile.write(str(target_path), faded, samplerate, subtype=subtype)
	except (OSError, soundfile.SoundFileError) as exc:
		print(f"  {filepath.name}  ERROR writing: {exc}", file=sys.stderr)
		return False

	# Analyze

	mono = numpy.mean(faded, axis=1, dtype=numpy.float32)
	params = subsample.analysis.compute_params(samplerate)
	rhythm_cfg = subsample.config.AnalysisConfig()

	result, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(
		mono, params, rhythm_cfg,
	)

	duration = faded.shape[0] / samplerate

	# Save sidecar

	try:
		audio_md5 = subsample.cache.compute_audio_md5(target_path)

		subsample.cache.save_cache(
			audio_path  = target_path,
			audio_md5   = audio_md5,
			params      = params,
			spectral    = result,
			rhythm      = rhythm,
			pitch       = pitch,
			timbre      = timbre,
			duration    = duration,
			level       = level,
			band_energy = band_energy,
		)
	except OSError as exc:
		_log.warning("Could not save analysis cache for %s: %s", target_path.name, exc)

	# Report

	peak = float(numpy.max(numpy.abs(faded)))
	rms = float(numpy.sqrt(numpy.mean(faded ** 2)))

	print(f"  {filepath.name}  {duration:.1f}s  peak {_dbfs(peak)}  rms {_dbfs(rms)}")

	return True


def main () -> None:

	"""Import pre-trimmed audio files into the Subsample capture library."""

	import argparse

	parser = argparse.ArgumentParser(
		prog="import_samples",
		description="Import pre-trimmed audio files into the Subsample capture library.",
	)
	parser.add_argument(
		"--to",
		type=str,
		default=None,
		metavar="DIR",
		help="Target directory (default: output.directory from config.yaml). "
		     "Import to the instrument directory to make samples immediately playable.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing files in target directory",
	)
	parser.add_argument(
		"files",
		nargs="*",
		metavar="FILE",
		help="Audio files or glob patterns to import",
	)

	args = parser.parse_args()

	if not args.files:
		parser.print_usage(sys.stderr)
		sys.exit(1)

	# Resolve target directory

	if args.to is not None:
		target_dir = pathlib.Path(args.to)
	else:
		cfg = subsample.config.load_config()
		target_dir = pathlib.Path(cfg.output.directory)

	target_dir.mkdir(parents=True, exist_ok=True)

	# Expand globs

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
		print("No input files.", file=sys.stderr)
		sys.exit(1)

	# Import

	print(f"Importing {len(paths)} file(s) to {target_dir}")

	imported = 0
	skipped = 0

	for filepath in paths:

		if not filepath.exists():
			print(f"  {filepath.name}  (not found, skipping)", file=sys.stderr)
			skipped += 1
			continue

		if _import_file(filepath, target_dir, args.force):
			imported += 1
		else:
			skipped += 1

	print(f"Imported {imported} file(s), skipped {skipped}")


if __name__ == "__main__":
	main()
