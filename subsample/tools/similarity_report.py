"""Print the top-N most similar instrument samples for each reference sample, or for one sound.

Loads reference samples from a directory (--reference-dir, defaults to the
references the app itself uses) and instrument samples from the configured
library.directory.  Builds a SimilarityMatrix and prints the top-N
matches for each reference.

Given a sound, an audio file or a library sample's name, it ranks the library
against that sound alone instead (#2676), leaving the sound itself out.

Usage:
	subsample similar
	subsample similar --top 10
	subsample similar --reference-dir samples/reference
	subsample similar ~/Downloads/snare.wav
	subsample similar kick_03 --top 10
"""

import argparse
import logging
import pathlib
import sys
import typing

import subsample.cache
import subsample.config
import subsample.library
import subsample.similarity
import subsample.tools._shared


# What the output means, printed at the end of --help and published with it on
# subsystem.co's command-line reference (#3843).  The help formatter keeps
# these lines as written, so they are wrapped by hand, within 79 columns.
_OUTPUT: typing.Final[str] = """\
output:
  For each reference, or for the sound given, the library samples most like
  it, best first: the rank, the sample's number in this run, how alike the
  two sound, where 1 is the same, the sample's name and its file. A sound
  from the library is not listed among its own matches.
"""


def parser () -> argparse.ArgumentParser:

	"""Build the parser for `subsample similar`, without parsing anything.

	Kept apart from parsing so that subsystem.co can generate the
	command-line reference from it without running the tool (#3020).
	"""

	command = argparse.ArgumentParser(
		prog="subsample similar",
		description="Show the instrument samples most similar to each reference, or to one sound",
		epilog=_OUTPUT,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	command.add_argument(
		"sound",
		nargs="?",
		default=None,
		metavar="SOUND",
		help=(
			"A sound to find more like: an audio file, analysed on the spot, or the "
			"name of a sample in the library. Left out, every reference is shown. A "
			"file's analysis is kept beside it, as the library's are."
		),
	)
	command.add_argument(
		"--top",
		type=int,
		default=5,
		metavar="N",
		help="Number of top matches to show per reference, or for the sound (default: 5)",
	)
	command.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	command.add_argument(
		"--reference-dir",
		type=pathlib.Path,
		default=None,
		metavar="DIR",
		help=(
			"Directory containing reference .analysis.json sidecar files "
			"(default: library.reference_directory, or the GM set bundled with Subsample)"
		),
	)
	return command


def _parse_args (argv: typing.Optional[list[str]] = None) -> argparse.Namespace:

	"""Parse command-line arguments."""

	return parser().parse_args(argv)


def _reference_for (
	sound:   str,
	library: subsample.library.InstrumentLibrary,
) -> typing.Optional[tuple[subsample.library.SampleRecord, typing.Optional[int]]]:

	"""The record to rank the library against, and the library's own id for it when it holds it.

	An existing file is that file: the library's record when the library holds
	it, else its analysis, made and kept beside it as the library's are.
	Anything else is the name of a sample, which must name exactly one, since
	take folders may each hold a "01".  Says why and returns None when the
	sound cannot be had.
	"""

	path = pathlib.Path(sound).expanduser()

	if path.is_file():
		own_id = library.find_by_path(path)
		own    = library.get(own_id) if own_id is not None else None

		if own is not None:
			return own, own.sample_id

		assets = subsample.cache.ensure_sample_assets(path, with_preview=False)

		if assets is None:
			print(f"Could not read or analyse {path} - nothing to compare against.", file=sys.stderr)
			return None

		record = subsample.library.SampleRecord(
			sample_id      = subsample.library.allocate_id(),
			name           = path.stem,
			spectral       = assets.spectral,
			rhythm         = assets.rhythm,
			pitch          = assets.pitch,
			timbre         = assets.timbre,
			level          = assets.level,
			band_energy    = assets.band_energy,
			params         = assets.params,
			duration       = assets.duration,
			audio          = None,
			filepath       = path,
			channel_format = assets.channel_format,
			loop           = assets.loop,
		)

		return record, None

	named = [record for record in library.samples() if record.name == sound]

	if not named:
		print(
			f"{sound!r} is neither an audio file nor the name of a sample in the "
			f"library - nothing to compare against.",
			file=sys.stderr,
		)
		return None

	if len(named) > 1:
		files = ", ".join(sorted(str(record.filepath) for record in named))
		print(
			f"{len(named)} samples in the library are named {sound!r} ({files}) - "
			f"give the path of the one you mean.",
			file=sys.stderr,
		)
		return None

	return named[0], named[0].sample_id


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Load libraries, build similarity matrix, and print per-reference rankings."""

	subsample.tools._shared.configure_logging()

	args = _parse_args(argv)

	# A sound takes the references' place, so a directory of them would be
	# read for nothing; say so rather than quietly ignoring it.
	if args.sound is not None and args.reference_dir is not None:
		print(
			"--reference-dir chooses the references, and a sound takes their "
			"place - give one or the other.",
			file=sys.stderr,
		)
		return 1

	# Loading libraries writes/heals sidecars via ensure_sample_assets, so wire
	# the float ceiling AND analysis tempo priors from config first — a sidecar
	# this tool heals must match what the player would compute.  Also gives a
	# clean one-line config error instead of a traceback.
	cfg = subsample.tools._shared.load_config_and_wire(args.config)

	# --- Load libraries ---

	reference_library: typing.Optional[subsample.library.ReferenceLibrary] = None

	if args.sound is None:
		# The same references the app itself would use, unless asked otherwise.
		# This defaulted to `samples/reference`, which --init has never created
		# and which the app does not read: the tool failed with "no reference
		# samples found" in every new project, including for the example in the
		# README.
		reference_dir = (
			args.reference_dir if args.reference_dir is not None
			else subsample.config.reference_directory(cfg)
		)
		reference_library = subsample.library.load_reference_library(reference_dir)

		if len(reference_library) == 0:
			print("No reference samples found - nothing to compare against.", file=sys.stderr)
			return 1

	# library.directory may be null — a project assembled from shared sample sets
	# loads only what its MIDI maps name, so there is no single tree to rank.
	if cfg.library.directory is None:
		print(
			"library.directory is null, so there are no instrument samples to "
			"rank - set it to the directory you want to report on.",
			file=sys.stderr,
		)
		return 1

	max_instrument_bytes = int(cfg.library.max_memory_mb * 1024 * 1024)
	instrument_library = subsample.library.load_instrument_library(
		pathlib.Path(cfg.library.directory),
		max_instrument_bytes,
		with_preview=False,   # mandatory keyword-only; this report renders no previews
		reference_directory=subsample.config.reference_directory(cfg),
	)

	if len(instrument_library) == 0:
		print("No instrument samples found - nothing to rank.", file=sys.stderr)
		return 1

	# The sound, when one is given, is the one reference.  When it is a sample
	# in the library it ranks first against itself, so it is left out below.
	own_id: typing.Optional[int] = None

	if reference_library is None:
		found = _reference_for(args.sound, instrument_library)

		if found is None:
			return 1

		reference, own_id = found
		reference_library = subsample.library.ReferenceLibrary([reference])

	# --- Build similarity matrix ---
	# Uses cfg.similarity weights — identical to the live application.

	matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
	matrix.bulk_add(instrument_library.samples())

	# --- Print report ---

	top_n = args.top
	col_width = max(len(r.name) for r in instrument_library.samples())

	for ref_name in reference_library.names():
		print(f"Reference: {ref_name}")

		# One more than asked for, when the sound itself is among them.  A
		# --top of 0 or less asks for every match, as get_matches reads it.
		wanted  = top_n + 1 if top_n > 0 and own_id is not None else top_n
		matches = [match for match in matrix.get_matches(ref_name, limit=wanted) if match.sample_id != own_id]

		if top_n > 0:
			matches = matches[:top_n]

		if not matches:
			print("  (no instrument samples)")
			print()
			continue

		for rank, match in enumerate(matches, start=1):
			record = instrument_library.get(match.sample_id)

			if record is None:
				# Should not happen — matrix and library are in sync
				print(f"  {rank}. #{match.sample_id}  {match.score:.4f}  (evicted)")
				continue

			filepath_str = str(record.filepath) if record.filepath is not None else "(no file)"
			print(
				f"  {rank}.  #{record.sample_id:<5}  {match.score:.4f}"
				f"  {record.name:<{col_width}}  {filepath_str}"
			)

		print()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
