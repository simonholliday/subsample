"""Print the top-N most similar instrument samples for each reference sample.

Loads reference samples from a directory (--reference-dir, defaults to
samples/reference) and instrument samples from the configured
instrument.directory.  Builds a SimilarityMatrix and prints the top-N
matches for each reference.

Usage:
	python scripts/similarity_report.py
	python scripts/similarity_report.py --top 10
	python scripts/similarity_report.py --reference-dir samples/reference
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
	parser.add_argument(
		"--reference-dir",
		type=pathlib.Path,
		default=pathlib.Path("samples/reference"),
		metavar="DIR",
		help="Directory containing reference .analysis.json sidecar files (default: samples/reference)",
	)
	return parser.parse_args()


def main () -> None:

	"""Load libraries, build similarity matrix, and print per-reference rankings."""

	args = _parse_args()

	cfg = subsample.config.load_config(args.config)

	if cfg.instrument.directory is None:
		print(
			"Error: no instrument directory configured.\n"
			"Add an instrument section to config.yaml:\n"
			"  instrument:\n"
			"    directory: ./samples",
			file=sys.stderr,
		)
		sys.exit(1)

	# --- Load libraries ---

	reference_library = subsample.library.load_reference_library(args.reference_dir)

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
