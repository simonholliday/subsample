"""Print the top-N most similar instrument samples for each reference sample.

Replicates the exact startup sequence from the main application:
  1. Load config (same config.yaml / config.yaml.default)
  2. Load reference library  → IDs 1..M  (sorted filename order)
  3. Load instrument library → IDs M+1..M+N (sorted filename order)
  4. Build SimilarityMatrix and score with bulk_add()

Because IDs are allocated from the same shared counter in the same order, the
numeric IDs shown here match exactly what the live application assigns.

Usage:
	python scripts/similarity_report.py
	python scripts/similarity_report.py --top 10
	python scripts/similarity_report.py --config /path/to/config.yaml
"""

import argparse
import logging
import pathlib
import sys

import subsample.config
import subsample.library
import subsample.similarity


logging.basicConfig(
	level=logging.WARNING,
	format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
	datefmt="%H:%M:%S",
)


def _parse_args () -> argparse.Namespace:

	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(
		prog="similarity_report",
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
	return parser.parse_args()


def main () -> None:

	"""Load libraries, build similarity matrix, and print per-reference rankings."""

	args = _parse_args()

	cfg = subsample.config.load_config(args.config)

	# Both libraries must be configured — without references there is nothing to
	# compare against, and without instruments there are no results to rank.
	if cfg.reference is None:
		print(
			"Error: no reference directory configured.\n"
			"Add a [reference] section to config.yaml:\n"
			"  reference:\n"
			"    directory: ./reference",
			file=sys.stderr,
		)
		sys.exit(1)

	if cfg.instrument.directory is None:
		print(
			"Error: no instrument directory configured.\n"
			"Add a [instrument] section to config.yaml:\n"
			"  instrument:\n"
			"    directory: ./samples",
			file=sys.stderr,
		)
		sys.exit(1)

	# --- Load libraries in the same order as cli.py:main() ---
	# Reference samples are loaded first so their IDs match the live application.

	reference_library = subsample.library.load_reference_library(
		pathlib.Path(cfg.reference.directory)
	)

	if len(reference_library) == 0:
		print("No reference samples found — nothing to compare against.", file=sys.stderr)
		sys.exit(1)

	max_instrument_bytes = int(cfg.instrument.max_memory_mb * 1024 * 1024)
	instrument_library = subsample.library.load_instrument_library(
		pathlib.Path(cfg.instrument.directory),
		max_instrument_bytes,
	)

	if len(instrument_library) == 0:
		print("No instrument samples found — nothing to rank.", file=sys.stderr)
		sys.exit(1)

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


if __name__ == "__main__":
	main()
