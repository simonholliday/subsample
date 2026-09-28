"""Nothing Subsample tells its user holds an em dash (#3847).

subsystem.co never prints one (#2568).  Its guide shows what Subsample prints
exactly as a terminal shows it, and a line holding an em dash cannot be shown
whole, so every string the program prints, logs or raises uses the house dash,
a spaced hyphen, and so does every file it ships for a user to read.

Docstrings, comments, and the strings that stand alone to document a
constant are read only by people changing the code, and are left out.
"""

import ast
import pathlib

import pytest

import subsample


_PACKAGE = pathlib.Path(subsample.__file__).parent
_EM_DASH = "—"


def _strings_a_user_can_read (path: pathlib.Path) -> list[tuple[int, str]]:

	"""Return every string in a module, with its line, except those that document it.

	A string standing alone as a statement is a docstring or documents the line
	above it, and nothing prints it.  Every other string may reach a user: a
	message, a log line, an error, a help text.  An f-string's literal parts are
	strings here too, so a message built from pieces is checked piece by piece.
	"""

	tree = ast.parse(path.read_text(encoding = "utf-8"))

	documentation = {
		id(node.value)
		for node in ast.walk(tree)
		if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
	}

	return [
		(node.lineno, node.value)
		for node in ast.walk(tree)
		if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in documentation
	]


@pytest.mark.parametrize(
	"path",
	sorted(_PACKAGE.rglob("*.py")),
	ids = lambda path: str(path.relative_to(_PACKAGE)),
)
def test_no_module_prints_an_em_dash (path: pathlib.Path) -> None:

	"""No string a module can print, log or raise holds an em dash."""

	found = [
		f"{path.relative_to(_PACKAGE)}:{line}"
		for line, text in _strings_a_user_can_read(path)
		if _EM_DASH in text
	]

	assert found == [], "write a spaced hyphen, or a full stop between two sentences"


@pytest.mark.parametrize(
	"path",
	sorted(path for path in (_PACKAGE / "data").rglob("*") if path.suffix != ".json" and path.is_file()),
	ids = lambda path: str(path.relative_to(_PACKAGE)),
)
def test_no_shipped_file_holds_an_em_dash (path: pathlib.Path) -> None:

	"""The configuration, the maps and the notes Subsample ships are read by its users too."""

	assert _EM_DASH not in path.read_text(encoding = "utf-8")


def test_the_check_sees_a_message_built_from_pieces (tmp_path: pathlib.Path) -> None:

	"""A dash in an f-string's literal part, or in a continued piece, is still found."""

	module = tmp_path / "module.py"
	module.write_text(
		'"""A docstring — left alone."""\n'
		'LIMIT = 3\n'
		'"""Documents LIMIT — left alone."""\n'
		'def f (name):\n'
		'\traise ValueError(\n'
		'\t\tf"{name!r} is unknown "\n'
		'\t\t"— check the spelling"\n'
		'\t)\n'
		'def g (count):\n'
		'\treturn f"{count} found — {count} kept"\n',
		encoding = "utf-8",
	)

	found = sorted(line for line, text in _strings_a_user_can_read(module) if _EM_DASH in text)

	assert found == [6, 10]
