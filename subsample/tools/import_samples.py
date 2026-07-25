"""Import pre-trimmed audio files into the Subsample capture library.

Reads audio files in any format supported by soundfile (WAV, BWF, FLAC, AIFF,
OGG, etc.), trims leading/trailing silence, applies safety fades to prevent
clicks, re-encodes as standard PCM WAV, runs the full analysis pipeline, and
saves a sidecar JSON alongside each imported file.

The target directory defaults to the configured recorder.directory from
config.yaml. Use --to to override.

Usage:
	subsample import <path/to/file.wav> [...]
	subsample import "/path/to/sample pack/*.wav"
	subsample import --to samples/captures kick.wav snare.wav
	subsample import --force --to samples/radio /mnt/sdr/audio/*.wav
"""

import argparse
import functools
import glob
import logging
import math
import pathlib
import sys
import typing

import numpy
import soundfile

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.config
import subsample.parallelism
import subsample.tools._shared


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

	"""Pick a PCM subtype for the imported copy.

	Integer-PCM sources keep their exact depth.  Float and double sources have
	no integer depth to preserve, so they are written at 24-bit — the highest
	depth every supported container holds, and enough that a float stem
	rendered out of a DAW is not audibly degraded on the way in.  (16-bit was
	the historical choice and cost ~8 bits of resolution silently.)  Anything
	else — compressed or exotic — falls back to 16-bit.
	"""

	sub = str(info.subtype)

	if sub in ("PCM_16", "PCM_24", "PCM_32"):
		return sub

	if sub in ("FLOAT", "DOUBLE"):
		return "PCM_24"

	return "PCM_16"


def _existing_target (target_dir: pathlib.Path, stem: str) -> typing.Optional[pathlib.Path]:

	"""Return an already-imported file with this stem, in either container.

	Import writes ``.wav`` or ``.flac`` depending on ``audio_format`` and the
	source depth, so "have I already imported this?" has to consider both —
	otherwise switching audio_format silently re-imports the whole library
	under the other extension.
	"""

	for extension in (".wav", ".flac"):
		candidate = target_dir / (stem + extension)

		if candidate.exists():
			return candidate

	return None


def _import_file (
	filepath: pathlib.Path,
	target_dir: pathlib.Path,
	force: bool,
	float_ceiling_dbfs: typing.Optional[float],
	rhythm_cfg: subsample.config.AnalysisConfig,
	audio_format: str = "wav",
) -> bool:

	"""Import a single audio file into the target directory.

	``float_ceiling_dbfs`` scales a hot float/double source down before writing
	and analysing it, so the written PCM does not clip and its sidecar describes
	the same signal (None disables, matching the config default of -1 dBFS).

	``audio_format`` is ``recorder.audio.audio_format`` — the same setting that
	governs live capture, which both its config docstring and the shipped
	config.yaml.default describe as covering captured *and imported* samples.
	A 32-bit source falls back to WAV per file, as FLAC cannot hold PCM_32.

	Returns True if the file was imported, False if skipped or failed.
	"""

	# main() claims target names before fanning out, so this guard normally never
	# fires under the pool — it stays for direct callers and as a belt-and-braces
	# check on a single-file import.
	if not force and _existing_target(target_dir, filepath.stem) is not None:
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

	# Scale a hot float/double source down to the ceiling BEFORE trim/fade/write/
	# analyse, so the 16-bit write does not hard-clip peaks above full scale and
	# the sidecar's level metrics describe the audio that was actually written.
	if info.subtype in ("FLOAT", "DOUBLE"):
		data = subsample.audio.scale_float_to_ceiling(data, float_ceiling_dbfs)

	# Trim silence and apply safety fades

	trimmed = _trim_silence(data)

	if trimmed.shape[0] == 0:
		print(f"  {filepath.name}  (skipped, silence only)")
		return False

	faded = _apply_safety_fades(trimmed, samplerate)

	# Analyze BEFORE writing anything.  A librosa failure on degenerate input
	# (very short or very low-sample-rate audio) then skips this one file
	# cleanly instead of leaving a sidecar-less WAV in the library that every
	# later startup scan re-skips.
	mono = numpy.asarray(numpy.mean(faded, axis=1), dtype=numpy.float32)
	params = subsample.analysis.compute_params(samplerate)

	try:
		result, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(
			mono, params, rhythm_cfg,
		)
	except Exception as exc:
		# librosa raises assorted exceptions (ParameterError, IndexError, ...) on
		# degenerate input; one bad file must not abort the whole import batch
		# (mirrors cache.ensure_sample_assets' skip-don't-abort contract).
		print(f"  {filepath.name}  (skipped, could not analyze: {exc})", file=sys.stderr)
		return False

	duration = faded.shape[0] / samplerate

	# Compute the seamless loop on the same analysis mono, so a bulk-imported
	# sidecar already carries its loop — a loop=None sidecar would match on
	# version + MD5 forever and never re-analyse (permanently unloopable).
	loop = subsample.cache.compute_loop(mono, samplerate, result, pitch, level, duration)

	# Pick the container now that the source depth is known, then hash the file
	# immediately — pairing the analysis above with an MD5 taken after a possible
	# concurrent overwrite of the target would write a sidecar that never
	# self-heals.  FLAC's stable subtypes stop at 24-bit, so a 32-bit source
	# falls back to WAV per file, exactly as live capture does.
	subtype = _resolve_subtype(info)

	if audio_format == "flac" and subtype != "PCM_32":
		target_path = target_dir / (filepath.stem + ".flac")
	else:
		if audio_format == "flac":
			_log.info(
				"%s: audio_format=flac cannot hold 32-bit; wrote .wav instead "
				"to preserve full precision", filepath.name,
			)

		target_path = target_dir / (filepath.stem + ".wav")

	try:
		soundfile.write(str(target_path), faded, samplerate, subtype=subtype)
	except (OSError, soundfile.SoundFileError) as exc:
		print(f"  {filepath.name}  ERROR writing: {exc}", file=sys.stderr)
		return False

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
			loop        = loop,
			# Record the actual on-disk format, not save_cache's 16/1 defaults
			# (subtype is PCM_16/24/32; faded is shape (n_frames, channels)).
			bit_depth   = int(subtype.rsplit("_", 1)[1]),
			channels    = faded.shape[1],
		)
	except OSError as exc:
		_log.warning("Could not save analysis cache for %s: %s", target_path.name, exc)

	# Report

	peak = float(numpy.max(numpy.abs(faded)))
	rms = float(numpy.sqrt(numpy.mean(faded ** 2)))

	print(f"  {filepath.name}  {duration:.1f}s  peak {_dbfs(peak)}  rms {_dbfs(rms)}")

	return True


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Import pre-trimmed audio files into the Subsample capture library."""

	subsample.tools._shared.configure_logging()


	parser = argparse.ArgumentParser(
		prog="subsample import",
		description="Import pre-trimmed audio files into the Subsample capture library.",
	)
	parser.add_argument(
		"--to",
		type=str,
		default=None,
		metavar="DIR",
		help="Target directory (default: recorder.directory from config.yaml). "
		     "Import to the instrument directory to make samples immediately playable.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing files in target directory",
	)
	parser.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	parser.add_argument(
		"files",
		nargs="*",
		metavar="FILE",
		help="Audio files or glob patterns to import",
	)

	args = parser.parse_args(argv)

	if not args.files:
		parser.print_usage(sys.stderr)
		return 1

	# Resolve target directory

	# Load config so a hot 32-bit-float source is scaled to fit the ceiling
	# rather than hard-clipped on the 16-bit write, AND so the analysis uses the
	# configured tempo priors — the same guards the live/CLI import path applies,
	# so an imported sidecar matches what the player would compute.
	cfg = subsample.tools._shared.load_config_and_wire(args.config)
	float_ceiling = cfg.recorder.audio.float_import_ceiling_dbfs

	if args.to is not None:
		target_dir = pathlib.Path(args.to)
	else:
		target_dir = pathlib.Path(cfg.recorder.directory)

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
		return 1

	# Import

	print(f"Importing {len(paths)} file(s) to {target_dir}")

	# Resolve every target path HERE, in the parent, before anything fans out.
	# _import_file's own "already exists" guard is evaluated per worker, so under
	# the pool two inputs sharing a stem (packA/kick.wav + packB/kick.wav) both
	# passed it before either had written — every worker then wrote the same
	# target, the last one won, and the run reported them all as imported.
	# Claiming targets up front restores the serial behaviour: one file wins the
	# name, the rest are skipped and counted, and the summary tells the truth.
	present: list[pathlib.Path] = []
	skipped = 0
	claimed: dict[str, pathlib.Path] = {}

	for filepath in paths:

		if not filepath.exists():
			print(f"  {filepath.name}  (not found, skipping)", file=sys.stderr)
			skipped += 1
			continue

		# Claim by stem, not by full target path: the container extension is
		# decided per file inside the worker (FLAC cannot hold 32-bit), but two
		# sources sharing a stem collide whichever container they land in.
		if filepath.stem in claimed:
			print(
				f"  {filepath.name}  (skipped, {claimed[filepath.stem].name} already "
				f"claims the name {filepath.stem!r} in this batch)",
				file=sys.stderr,
			)
			skipped += 1
			continue

		existing = _existing_target(target_dir, filepath.stem)

		if existing is not None and not args.force:
			print(f"  {filepath.name}  (skipped, already exists)")
			skipped += 1
			continue

		claimed[filepath.stem] = filepath
		present.append(filepath)

	# Fingerprinting each file is CPU-heavy and independent, so import fans out
	# across the machine (offline — no player — so it takes the larger, offline
	# share of the cores).  _import_file is self-contained: it writes its own WAV
	# and sidecar and returns True on success.  Per-file progress lines may
	# interleave under the pool, as expected for a parallel batch.
	results = subsample.parallelism.map_analysis(
		functools.partial(
			_import_file,
			target_dir=target_dir,
			force=args.force,
			float_ceiling_dbfs=float_ceiling,
			rhythm_cfg=cfg.analysis,
			audio_format=cfg.recorder.audio.audio_format,
		),
		present,
		player_active=False,
	)

	imported = sum(1 for ok in results if ok)
	skipped += sum(1 for ok in results if not ok)

	print(f"Imported {imported} file(s), skipped {skipped}")

	# Files were given but nothing was imported (all missing/unreadable/already
	# present) — exit non-zero so a script sees it rather than a false success.
	if imported == 0 and skipped > 0:
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
