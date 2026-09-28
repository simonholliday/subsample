"""Read YAML so a number means what it looks like.

PyYAML implements YAML **1.1**, where two of its rules bite a file that is
mostly small numbers written by hand:

- a leading zero means octal, so a drum map lining its note numbers up in a
  column reads ``kick: 036`` as **30** - a different drum, and nothing says
  so;
- a colon means sexagesimal, so a cue written ``1:30`` becomes **90**.

YAML 1.2 has neither, and it reads the way people write: ``036`` is
thirty-six, ``0o42`` is thirty-four, and ``1:30`` is the text it looks like.

Every YAML file Subsample reads goes through :func:`load` - ``config.yaml``,
a MIDI map, an ensemble map, and the definitions file it shares with a
sequencer - so a number means one thing wherever it is written.

Only the number rules change.  ``yes``, ``no``, ``on``, ``off`` and ``null``
read exactly as they always have: a file full of those is not what went
wrong.

One thing is refused that PyYAML allows: a key written twice in one
mapping.  YAML says a mapping's keys are unique, and PyYAML keeps the
second and drops the first without a word, so a second ``player:``
section lost everything in the first.  :class:`DuplicateKeyError` names
the key and both its lines.

Decision 18 of subsequence#2991 settled this for both tools at once, so that
a definitions file cannot name one drum here and another there.
"""

import re
import typing

import yaml


class _Yaml12Loader (yaml.SafeLoader):

	"""SafeLoader with YAML 1.1's surprising number rules taken out.

	The resolver is replaced rather than the loaded values corrected
	afterwards, because by then the two readings are indistinguishable: 30 is
	30, whether it was written ``30`` or ``036``.
	"""


# A copy, so rewriting the resolvers below cannot reach yaml.SafeLoader
# itself - which every other user of PyYAML in this process shares.
_Yaml12Loader.yaml_implicit_resolvers = {
	first: list(resolvers)
	for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

# The sign is allowed on all three forms, where YAML 1.2's core schema allows
# it only on decimals.  A deliberate superset: under the strict reading
# `-0x10` stops being a number and becomes the string "-0x10", which 1.1 read
# as -16 - a break nobody gains from, in exchange for nothing.  Everything
# this is actually about (036, 1:30) is unaffected.
_YAML_12_INT: typing.Final[re.Pattern[str]] = re.compile(
	r"""^[-+]?(?:[0-9]+
		|0o[0-7]+
		|0x[0-9a-fA-F]+)$""",
	re.VERBOSE,
)

# A float must carry a dot or an exponent.  The 1.2 core schema's own pattern
# also matches a bare "38" and leans on the int resolver being consulted
# first, which PyYAML does not guarantee per starting character.  Spelling the
# requirement out makes the two patterns disjoint, so the order cannot matter.
# Without it every plain integer came back as a float: `snare: 38` was 38.0,
# and a MIDI note number is not a float.
_YAML_12_FLOAT: typing.Final[re.Pattern[str]] = re.compile(
	r"""^(?:[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?
		|[-+]?[0-9]+[eE][-+]?[0-9]+
		|[-+]?\.(?:inf|Inf|INF)
		|\.(?:nan|NaN|NAN))$""",
	re.VERBOSE,
)


def _use_yaml_12_numbers () -> None:

	"""Swap the int and float resolvers on the private loader for 1.2's."""

	for first, resolvers in _Yaml12Loader.yaml_implicit_resolvers.items():

		replaced: list[tuple[str, re.Pattern[str]]] = []

		for tag, pattern in resolvers:

			if tag == "tag:yaml.org,2002:int":
				replaced.append((tag, _YAML_12_INT))
			elif tag == "tag:yaml.org,2002:float":
				replaced.append((tag, _YAML_12_FLOAT))
			else:
				replaced.append((tag, pattern))

		_Yaml12Loader.yaml_implicit_resolvers[first] = replaced

	# 1.1 resolves a number on characters 1.2 does not start one with, so
	# those entries have to be added rather than only rewritten.
	for first in "0123456789-+.":

		entries = _Yaml12Loader.yaml_implicit_resolvers.setdefault(first, [])

		for tag, pattern in (
			("tag:yaml.org,2002:int", _YAML_12_INT),
			("tag:yaml.org,2002:float", _YAML_12_FLOAT),
		):
			if not any(existing == tag for existing, _ in entries):
				entries.append((tag, pattern))


_use_yaml_12_numbers()


def _construct_yaml_12_int (loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> int:

	"""Read an integer by YAML 1.2's rules: decimal, ``0o`` octal, ``0x`` hex.

	Both halves of PyYAML have to move, which is the part worth remembering:
	the *resolver* decides which tag a scalar carries, the *constructor*
	decides what value it becomes, and PyYAML's constructor is 1.1 all the way
	down.  Replacing only the resolver leaves ``kick: 036`` reading 30,
	exactly as before, which looks for all the world like the change did not
	work.
	"""

	text = str(loader.construct_scalar(node))

	sign = 1

	if text and text[0] in "+-":
		sign = -1 if text[0] == "-" else 1
		text = text[1:]

	if text.startswith(("0o", "0O")):
		return sign * int(text[2:], 8)

	if text.startswith(("0x", "0X")):
		return sign * int(text[2:], 16)

	return sign * int(text, 10)


# add_constructor copies the table onto the subclass first, so SafeLoader -
# and every other user of PyYAML in this process - keeps its own.
_Yaml12Loader.add_constructor("tag:yaml.org,2002:int", _construct_yaml_12_int)


class DuplicateKeyError (yaml.YAMLError):

	"""A key written twice in one mapping, which YAML does not allow.

	PyYAML keeps the second and drops the first without a word, so a
	``player:`` section added at the end of a config.yaml that ``--init`` wrote
	quietly lost every setting in the first, and the player then refused to
	start for want of a map.  A ``yaml.YAMLError``, so every caller that
	reports a malformed file reports this one too.
	"""

	def __init__ (self, name: str, key: str, first_line: int, second_line: int) -> None:

		"""Record the file, the key as its path of keys, and both lines it is written on."""

		self.name        = name
		self.key         = key
		self.first_line  = first_line
		self.second_line = second_line

		super().__init__(f"{name}: {self.problem}")

	@property
	def problem (self) -> str:

		"""What is wrong and how to put it right, for a caller that has named the file already."""

		return (
			f"'{self.key}' is written twice, on lines {self.first_line} and {self.second_line}. "
			"Write it once, with everything from both."
		)


# A merge (`<<: *defaults`) brings in keys that are there to be overridden.
_MERGE_TAG: typing.Final[str] = "tag:yaml.org,2002:merge"


def _refuse_repeated_keys (
	loader: _Yaml12Loader,
	node:   yaml.Node,
	path:   str,
	walked: set[int],
) -> None:

	"""Raise DuplicateKeyError at the first key under *node*, in reading order, written twice in one mapping.

	Keys are compared as the values they read as, the way the mapping built
	from them compares them, so ``36:`` and ``036:`` are one key.  *path* is
	the dotted keys that lead to *node* from the top of the document or from
	the list item that holds it.  The check runs on the parsed document before
	anything is built from it, since building it is where the first key is
	lost, and where a merge's keys are mixed in with the mapping's own.
	"""

	# An alias is the node it names, met again.  Once is enough, and a
	# structure that contains itself would otherwise never finish.
	if id(node) in walked:
		return

	walked.add(id(node))

	if isinstance(node, yaml.SequenceNode):
		for item in node.value:
			_refuse_repeated_keys(loader, item, "", walked)

		return

	if not isinstance(node, yaml.MappingNode):
		return

	first_lines: dict[typing.Any, int] = {}

	for key_node, value_node in node.value:

		if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == _MERGE_TAG:
			_refuse_repeated_keys(loader, value_node, path, walked)
			continue

		# types-PyYAML leaves construct_object unannotated; it returns the value.
		name = f"{path}.{key_node.value}" if path else str(key_node.value)
		key  = loader.construct_object(key_node, deep=True)  # type: ignore[no-untyped-call]
		line = key_node.start_mark.line + 1

		if key in first_lines:
			raise DuplicateKeyError(key_node.start_mark.name, name, first_lines[key], line)

		first_lines[key] = line

		_refuse_repeated_keys(loader, value_node, name, walked)


def load (stream: typing.Union[str, typing.IO[str]]) -> typing.Any:

	"""Read one YAML document, with its numbers meaning what they look like.

	A key written twice in one mapping raises DuplicateKeyError, naming it and
	both its lines, where PyYAML would keep the second.  That, like a malformed
	document, is a ``yaml.YAMLError``, so a caller's error handling is
	unchanged.
	"""

	# yaml.load's own steps, with the check between parsing and building.
	# types-PyYAML leaves construct_document and dispose unannotated.
	loader = _Yaml12Loader(stream)

	try:
		node = loader.get_single_node()

		if node is None:
			return None

		_refuse_repeated_keys(loader, node, "", set())

		return loader.construct_document(node)  # type: ignore[no-untyped-call]

	finally:
		loader.dispose()  # type: ignore[no-untyped-call]
