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


def load (stream: typing.Union[str, typing.IO[str]]) -> typing.Any:

	"""Read one YAML document, with its numbers meaning what they look like.

	Raises the same ``yaml.YAMLError`` as ``yaml.safe_load`` on a malformed
	document, so a caller's error handling is unchanged.
	"""

	return yaml.load(stream, Loader = _Yaml12Loader)
