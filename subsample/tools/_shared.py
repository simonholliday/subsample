"""Shared helpers for the subsample.tools subcommands.

The tool subcommands (import/catalog/analyze/similar/loops) bypass
cli._main_impl, so they must wire the same process-wide analysis settings it
does before writing any sidecar the player later trusts.
"""

import glob
import logging
import pathlib
import sys
import typing

import subsample.audio
import subsample.cache
import subsample.config


def configure_logging () -> None:

	"""Set up console logging for a tool subcommand — call at the top of main().

	Kept out of module scope so importing a tool module (for the dispatch, or in
	a test) does not reconfigure the root logger as an import side effect.
	basicConfig is idempotent, so a repeat call is harmless.
	"""

	logging.basicConfig(
		level=logging.WARNING,
		format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
		datefmt="%H:%M:%S",
	)


def load_config_and_wire (
	config_path: typing.Optional[pathlib.Path] = None,
) -> subsample.config.Config:

	"""Load config for a tool subcommand and wire the process-wide analysis
	settings its sidecar writers depend on.

	Mirrors cli._main_impl: any sidecar a tool writes must describe the same
	audio the app itself would compute, so the float-import ceiling (hot-float
	scaling) and the analysis tempo priors are wired from the same config
	before any analysis runs.  Otherwise a non-default
	``recorder.audio.float_import_ceiling_dbfs`` or
	``analysis.start_bpm/tempo_min/tempo_max`` would make the tool write a
	sidecar the player then trusts forever yet does not match its own playback.

	A config error — a renamed key, malformed YAML, or a bad ``--config`` path
	(missing / a directory / unreadable) — prints one clean line and raises
	SystemExit(1); the tool dispatch propagates it as the process exit code
	(the same convention argparse already uses for a bad-argument SystemExit(2)).
	"""

	try:
		cfg = subsample.config.load_config(config_path)
	except (OSError, ValueError) as exc:
		print(f"Error: {exc}", file=sys.stderr)
		raise SystemExit(1)

	subsample.audio.set_float_import_ceiling(
		cfg.recorder.audio.float_import_ceiling_dbfs
	)
	subsample.cache.set_analysis_config(cfg.analysis)

	return cfg


# The characters that make an argument a pattern rather than a name.
_GLOB_CHARACTERS: typing.Final[frozenset[str]] = frozenset("*?[")


def expanded_paths (arguments: typing.Sequence[str]) -> list[pathlib.Path]:

	"""Every file the arguments name, expanding quoted patterns like `*.wav`.

	An argument that matches nothing is reported if it looks like a pattern, and
	otherwise passed through so the audio reader can produce the ordinary "file
	not found" message rather than a cryptic system error.

	A name that merely CONTAINS a pattern character is still a name.  `kick
	[01].wav` is a perfectly ordinary file in a sample pack, and glob reads the
	bracket as a character class and matches nothing — so the tools reported
	"No files matched" for a file sitting right there.  A literal path that
	exists therefore wins before the pattern question is asked at all.
	"""

	paths: list[pathlib.Path] = []

	for argument in arguments:

		# A file by that exact name settles it, brackets and all.
		if pathlib.Path(argument).exists():
			paths.append(pathlib.Path(argument))
			continue

		matches = sorted(glob.glob(argument))

		if matches:
			paths.extend(pathlib.Path(match) for match in matches)

		elif any(character in argument for character in _GLOB_CHARACTERS):
			print(f"No files matched: {argument}", file=sys.stderr)

		else:
			# Not a pattern and not there: let the reader say so properly.
			paths.append(pathlib.Path(argument))

	return paths
