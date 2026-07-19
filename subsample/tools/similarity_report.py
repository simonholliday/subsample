"""Print the top-N most similar instrument samples for each reference sample.

Loads reference samples from a directory (--reference-dir, defaults to
samples/reference) and instrument samples from the configured
library.directory.  Builds a SimilarityMatrix and prints the top-N
matches for each reference.

Usage:
	subsample similar
	subsample similar --top 10
	subsample similar --reference-dir samples/reference
"""

import argparse
import logging
import pathlib
import sys
import typing

import subsample.config
import subsample.library
import subsample.similarity
import subsample.tools._shared


def _parse_args (argv: typing.Optional[list[str]] = None) -> argparse.Namespace:

	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(
		prog="subsample similar",
		description="Show the top-N most similar instrument samples for each reference",
	)
	parser.add_argument(
		"--top",
		type=int,
		default=5,
		metavar="N",
		help="Number of top matches to show per reference (default: 5)",
	)
	parser.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	parser.add_argument(
		"--reference-dir",
		type=pathlib.Path,
		default=pathlib.Path("samples/reference"),
		metavar="DIR",
		help="Directory containing reference .analysis.json sidecar files (default: samples/reference)",
	)
	return parser.parse_args(argv)


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Load libraries, build similarity matrix, and print per-reference rankings."""

	subsample.tools._shared.configure_logging()

	args = _parse_args(argv)

	# Loading libraries writes/heals sidecars via ensure_sample_assets, so wire
	# the float ceiling AND analysis tempo priors from config first — a sidecar
	# this tool heals must match what the player would compute.  Also gives a
	# clean one-line config error instead of a traceback.
	cfg = subsample.tools._shared.load_config_and_wire(args.config)

	# --- Load libraries ---

	reference_library = subsample.library.load_reference_library(args.reference_dir)

	if len(reference_library) == 0:
		print("No reference samples found — nothing to compare against.", file=sys.stderr)
		return 1

	max_instrument_bytes = int(cfg.library.max_memory_mb * 1024 * 1024)
	instrument_library = subsample.library.load_instrument_library(
		pathlib.Path(cfg.library.directory),
		max_instrument_bytes,
		with_preview=False,   # mandatory keyword-only; this report renders no previews
	)

	if len(instrument_library) == 0:
		print("No instrument samples found — nothing to rank.", file=sys.stderr)
		return 1

	# --- Build similarity matrix ---
	# Uses cfg.similarity weights — identical to the live application.

	matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
	matrix.bulk_add(instrument_library.samples())

	# --- Print report ---

	top_n = args.top
	col_width = max(len(r.name) for r in instrument_library.samples())

	for ref_name in reference_library.names():
		print(f"Reference: {ref_name}")

		matches = matrix.get_matches(ref_name, limit=top_n)

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
