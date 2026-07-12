"""Find and audition loop points for sustained samples.

For each loop-candidate audio file (those the is_loopable gate passes), find a
seamless loop with subsample.loopfind, print the proposed points and junction
quality, and optionally render an audition WAV so you can HEAR the loop before
committing it to a MIDI map.

This is the ear-calibration tool for loop playback: the numbers are proxies, so
the whole point is to listen.  Two renders are written per sample when --render
is given:

  <stem>_loop.wav       start -> loop x N with the 30 ms linear crossfade
  <stem>_loop_butt.wav  the same loop with NO crossfade (raw butt joint)

so you can A/B whether the crossfade earns its place on your material.

Usage:
	python scripts/suggest_loops.py /path/to/samples           # report only
	python scripts/suggest_loops.py file.wav                    # a single file
	python scripts/suggest_loops.py ~/samples --render /tmp/loop_demo
"""

import argparse
import logging
import pathlib
import sys
import typing

import numpy
import soundfile

import subsample.analysis
import subsample.cache
import subsample.loopfind


logging.basicConfig(
	level=logging.WARNING,
	format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
	datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)


def _collect_audio_files (paths: list[pathlib.Path]) -> list[pathlib.Path]:

	"""Expand each argument (file or directory) into a sorted audio-file list."""

	found: list[pathlib.Path] = []

	for p in paths:
		if p.is_dir():
			found.extend(
				q for q in sorted(p.rglob("*"))
				if q.is_file() and q.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS
			)
		elif p.is_file() and p.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS:
			found.append(p)
		else:
			print(f"Skipping {p}: not an audio file or directory", file=sys.stderr)

	return found


def _render (path: pathlib.Path, loop: subsample.loopfind.LoopPoints, audio: numpy.ndarray, sr: int, out_dir: pathlib.Path) -> None:

	"""Write the crossfaded and butt-joint audition renders for one sample."""

	out_dir.mkdir(parents=True, exist_ok=True)

	crossfaded = subsample.loopfind.render_audition(audio, loop, sr)
	soundfile.write(str(out_dir / f"{path.stem}_loop.wav"), crossfaded, sr)

	butt = subsample.loopfind.render_audition(
		audio, subsample.loopfind.LoopPoints(loop.start, loop.end, 0, loop.junction_flux), sr,
	)
	soundfile.write(str(out_dir / f"{path.stem}_loop_butt.wav"), butt, sr)


def main (argv: typing.Optional[list[str]] = None) -> None:

	"""Report (and optionally render) loop points for loop-candidate samples."""

	parser = argparse.ArgumentParser(
		prog="suggest_loops",
		description="Find and audition loop points for loop-candidate samples",
	)
	parser.add_argument("paths", type=pathlib.Path, nargs="+", help="Audio files or directories")
	parser.add_argument(
		"--render", type=pathlib.Path, default=None, metavar="DIR",
		help="Also write audition WAVs (crossfaded + butt-joint) to DIR",
	)
	parser.add_argument(
		"--all", action="store_true",
		help="Try every file, not only those the is_loopable gate passes",
	)
	args = parser.parse_args(argv)

	files = _collect_audio_files(args.paths)

	if not files:
		print("No audio files found.", file=sys.stderr)
		sys.exit(1)

	n_candidates = n_found = 0

	for path in files:
		assets = subsample.cache.ensure_sample_assets(path, with_preview=False)

		if assets is None:
			print(f"{path.name}: unreadable — skipped", file=sys.stderr)
			continue

		if not args.all and not subsample.analysis.is_loopable(assets.spectral, assets.level, assets.duration):
			continue

		n_candidates += 1

		# Load the waveform (the sidecar has no audio); keep native rate + channels.
		try:
			audio, sr = soundfile.read(str(path), always_2d=False, dtype="float32")
		except (OSError, soundfile.SoundFileError) as exc:
			print(f"{path.name}: could not read audio — {exc}", file=sys.stderr)
			continue

		pitch_hz = assets.pitch.dominant_pitch_hz if assets.pitch.dominant_pitch_hz > 0.0 else None
		loop     = subsample.loopfind.find_loop(audio, sr, pitch_hz=pitch_hz)

		if loop is None:
			print(f"{path.name:40s} loopable but no clean loop found (no sustain region, or every junction fails-musical)")
			continue

		n_found += 1
		loop_ms = (loop.end - loop.start) / sr * 1000.0
		tail_ms = (len(audio) - loop.end) / sr * 1000.0
		print(
			f"{path.name:40s} "
			f"loop {loop.start / sr:6.3f}s -> {loop.end / sr:6.3f}s "
			f"({loop_ms:6.0f} ms)  xfade {loop.crossfade / sr * 1000:4.0f} ms  "
			f"junction_flux {loop.junction_flux:.3f}  tail {tail_ms:5.0f} ms"
		)

		if args.render is not None:
			_render(path, loop, audio, sr, args.render)

	summary = f"{n_found} loop(s) found across {n_candidates} candidate(s)"
	if args.render is not None:
		summary += f"; auditions in {args.render}"
	print(summary, file=sys.stderr)


if __name__ == "__main__":
	main()
