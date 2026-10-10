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

so you can A/B whether the crossfade earns its place on your material.  Each
goes in the folder its sample was found in, under DIR, so samples of one name in
different folders keep their own, and a file already there is not replaced
unless --force is given (#4703, M30): renders wrote over whatever had their
name, a real sample included.

Usage:
	subsample loops /path/to/samples           # report only
	subsample loops file.wav                    # a single file
	subsample loops ~/samples --render /tmp/loop_demo
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
import subsample.tools._shared


_log = logging.getLogger(__name__)


def _collect_audio_files (paths: list[pathlib.Path]) -> list[tuple[pathlib.Path, pathlib.Path]]:

	"""Expand each argument (file or directory) into a sorted audio-file list, each with its folder below the argument."""

	found: list[tuple[pathlib.Path, pathlib.Path]] = []

	for p in paths:
		if p.is_dir():
			found.extend(
				(q, q.parent.relative_to(p)) for q in sorted(p.rglob("*"))
				if q.is_file() and q.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS
			)
		elif p.is_file() and p.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS:
			found.append((p, pathlib.Path()))
		else:
			print(f"Skipping {p}: not an audio file or directory", file=sys.stderr)

	return found


def _render (
	path:    pathlib.Path,
	loop:    subsample.loopfind.LoopPoints,
	audio:   numpy.ndarray,
	sr:      int,
	out_dir: pathlib.Path,
	force:   bool,
) -> int:

	"""Write the crossfaded and butt-joint audition renders for one sample; how many were not written, as they exist.

	A file already there is replaced only with ``force`` (#4703, M30): a
	render wrote over a real sample of its name.
	"""

	out_dir.mkdir(parents=True, exist_ok=True)

	butt_loop = subsample.loopfind.LoopPoints(loop.start, loop.end, 0, loop.junction_flux)
	kept      = 0

	for target, points in ((out_dir / f"{path.stem}_loop.wav", loop), (out_dir / f"{path.stem}_loop_butt.wav", butt_loop)):

		if target.exists() and not force:
			print(f"{target}: already exists - not written (--force replaces it)", file=sys.stderr)
			kept += 1
			continue

		soundfile.write(str(target), subsample.loopfind.render_audition(audio, points, sr), sr)

	return kept


# What the output means, printed at the end of --help and published with it on
# subsystem.co's command-line reference (#3843).  The help formatter keeps
# these lines as written, so they are wrapped by hand, within 79 columns.
_OUTPUT: typing.Final[str] = """\
output:
  A line for each sample tried: where its loop starts and ends in seconds, the
  loop's length, the crossfade across its join, junction_flux, about 1 or
  below for a seamless join and higher as the join grows audible, and the
  tail left after the loop for the note's release.  A sample with no clean
  loop plays without looping.
"""


def parser () -> argparse.ArgumentParser:

	"""Build the parser for `subsample loops`, without parsing anything.

	Kept apart from main() so that subsystem.co can generate the command-line
	reference from it without running the tool (#3020).
	"""

	command = argparse.ArgumentParser(
		prog="subsample loops",
		description="Find and audition loop points for loop-candidate samples",
		epilog=_OUTPUT,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	command.add_argument("paths", type=pathlib.Path, nargs="+", help="Audio files or directories")
	command.add_argument(
		"--render", type=pathlib.Path, default=None, metavar="DIR",
		help=(
			"Also write audition WAVs (crossfaded + butt-joint) to DIR, each in the folder its "
			"sample was found in; a file already there is not replaced"
		),
	)
	command.add_argument(
		"--force", action="store_true",
		help="Replace audition WAVs already in DIR",
	)
	command.add_argument(
		"--all", action="store_true",
		help="Try every file, not only those the is_loopable gate passes",
	)
	command.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	return command


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Report (and optionally render) loop points for loop-candidate samples."""

	subsample.tools._shared.configure_logging()

	args = parser().parse_args(argv)

	# ensure_sample_assets heals missing/stale sidecars, so wire the float
	# ceiling and analysis tempo priors from config first — a sidecar this tool
	# writes must match what the player would compute.
	subsample.tools._shared.load_config_and_wire(args.config)

	files = _collect_audio_files(args.paths)

	if not files:
		print("No audio files found.", file=sys.stderr)
		return 1

	n_candidates = n_found = n_kept = 0

	for path, folder in files:
		assets = subsample.cache.ensure_sample_assets(path, with_preview=False)

		if assets is None:
			print(f"{path.name}: unreadable - skipped", file=sys.stderr)
			continue

		if not args.all and not subsample.analysis.is_loopable(assets.spectral, assets.level, assets.duration):
			continue

		n_candidates += 1

		# Load the waveform (the sidecar has no audio); keep native rate + channels.
		try:
			audio, sr = soundfile.read(str(path), always_2d=False, dtype="float32")
		except (OSError, soundfile.SoundFileError) as exc:
			print(f"{path.name}: could not read audio - {exc}", file=sys.stderr)
			continue

		# Prefer the STORED loop — computed on the mono/mid at analysis time, it is
		# the exact set of points the player loops.  Re-running find_loop on this
		# raw full-channel read would diverge from playback for ambisonic and
		# multichannel samples.  --all is the explicit "explore every file" mode,
		# so it re-runs the finder (non-candidates have no stored loop anyway).
		if args.all:
			pitch_hz = assets.pitch.dominant_pitch_hz if assets.pitch.dominant_pitch_hz > 0.0 else None
			loop     = subsample.loopfind.find_loop(audio, sr, pitch_hz=pitch_hz)
		else:
			loop     = assets.loop

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
			n_kept += _render(path, loop, audio, sr, args.render / folder, args.force)

	summary = f"{n_found} loop(s) found across {n_candidates} candidate(s)"
	if args.render is not None:
		summary += f"; auditions in {args.render}"
		if n_kept:
			summary += f", {n_kept} not written as they already exist (--force replaces them)"
	print(summary, file=sys.stderr)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
