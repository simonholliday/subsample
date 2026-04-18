"""Regenerate `.preview.png` sidecars for an existing sample library.

**Temporary helper** — used to iterate on the preview visualisation without
re-recording every sample.  Walks a directory, finds every audio file with
an adjacent `.analysis.json` sidecar, rebuilds the PreviewData from the
audio + existing analysis, and writes a fresh `<audio>.preview.png`
sidecar.

Does **not** touch the `.analysis.json` sidecars — the embedded `preview`
data block inside them is written only by the main capture pipeline.  PNGs
are pure raster output and can be rewritten freely.

This script will be superseded by a proper `python -m subsample
regen-previews` CLI later, which will also refresh the embedded preview
data block.

Usage:
	python scripts/regen_previews_png.py <directory>              # recursive
	python scripts/regen_previews_png.py samples/captures/
	python scripts/regen_previews_png.py ~/my-library/ -v         # verbose
"""

import argparse
import logging
import pathlib
import sys

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.preview


logging.basicConfig(
	level=logging.WARNING,
	format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
	datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)


def _regen_one (audio_path: pathlib.Path) -> bool:

	"""Regenerate the preview PNG for a single sample.  Returns True on success."""

	cached = subsample.cache.load_cache(audio_path)
	if cached is None:
		_log.warning("skip %s — no valid analysis sidecar", audio_path.name)
		return False

	spectral, rhythm, pitch, _timbre, _params, duration, level, band_energy, channel_format = cached

	try:
		file_info = subsample.audio.read_audio_file(audio_path)
	except (OSError, ValueError) as exc:
		_log.warning("skip %s — %s", audio_path.name, exc)
		return False

	# Ambisonic samples: analyse only the W (omni) channel, matching the main
	# capture pipeline.  PCM samples average channels as usual.
	channel_index = 0 if channel_format.startswith("b_format_") else None
	mono = subsample.analysis.to_mono_float(
		file_info.audio, file_info.bit_depth, channel_index=channel_index,
	)

	preview_data = subsample.preview.compute_preview_data(
		mono, file_info.sample_rate, rhythm, pitch, spectral, level, band_energy,
		duration=duration,
	)
	png_path = audio_path.with_name(audio_path.name + ".preview.png")
	subsample.preview.render_png(preview_data, png_path)
	_log.info("wrote %s", png_path.name)
	return True


def _iter_audio_files (root: pathlib.Path) -> list[pathlib.Path]:

	"""Return every audio file under root that has an adjacent `.analysis.json`
	sidecar.  File extensions matched: .wav, .flac, .aiff, .ogg, .mp3."""

	audio_files: list[pathlib.Path] = []
	for suffix in (".wav", ".flac", ".aiff", ".ogg", ".mp3"):
		for path in sorted(root.rglob(f"*{suffix}")):
			if subsample.cache.cache_path(path).exists():
				audio_files.append(path)
	return audio_files


def main () -> None:

	parser = argparse.ArgumentParser(
		description="Regenerate .preview.png sidecars for an existing library.",
	)
	parser.add_argument("directory", type=pathlib.Path,
	                    help="Library directory to walk (recursive).")
	parser.add_argument("-v", "--verbose", action="store_true",
	                    help="Log a line per sample regenerated.")
	args = parser.parse_args()

	if args.verbose:
		logging.getLogger().setLevel(logging.INFO)

	root = args.directory.expanduser().resolve()
	if not root.is_dir():
		print(f"Not a directory: {root}", file=sys.stderr)
		sys.exit(2)

	audio_files = _iter_audio_files(root)
	if not audio_files:
		print(f"No audio files with .analysis.json sidecars under {root}",
		      file=sys.stderr)
		sys.exit(1)

	print(f"Regenerating {len(audio_files)} preview PNG(s) under {root} ...")
	ok = 0
	for path in audio_files:
		if _regen_one(path):
			ok += 1

	print(f"Done.  {ok}/{len(audio_files)} samples regenerated.")


if __name__ == "__main__":
	main()
