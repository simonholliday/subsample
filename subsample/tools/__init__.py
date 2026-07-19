"""Musician-facing command-line tools, dispatched as `subsample <command>`.

import_samples (`subsample import`), catalog_samples (`subsample catalog`),
analyze_file (`subsample analyze`), similarity_report (`subsample similar`),
and suggest_loops (`subsample loops`).  Each module exposes
`main(argv) -> int`; subsample.cli routes the first CLI argument here before
falling through to the run-mode parser.
"""
