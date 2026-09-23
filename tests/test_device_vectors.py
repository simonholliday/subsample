"""The device-matching cases Subsample and Subsequence must answer alike (#623).

Both apps pick a configured device out of a backend's names by one rule, and
that rule was written only as prose in each app's docstring.  The prose drifted
within hours of both sides shipping (#397): the same config ran unattended in
one app and stopped to ask a question in the other.

``tests/fixtures/device_matching.yaml`` is the executable half.  Subsequence
holds an identical copy and asserts the same cases, so the two cannot disagree
without one of them failing.  Change the file in both repositories together.
"""

import pathlib
import typing

import pytest

import subsample.devices
import subsample.yaml_numbers


_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "device_matching.yaml"


def _vectors () -> dict[str, typing.Any]:

	"""The shared cases, read once."""

	with _FIXTURE.open(encoding="utf-8") as handle:
		loaded: dict[str, typing.Any] = subsample.yaml_numbers.load(handle)

	return loaded


_VECTORS = _vectors()


@pytest.mark.parametrize(
	"case",
	[
		pytest.param(case, id=f"{case['list']}:{case['pattern']}")
		for case in _VECTORS["cases"]
	],
)
def test_a_shared_case (case: dict[str, typing.Any]) -> None:

	"""Each case selects exactly the names it says, in the order reported."""

	names = _VECTORS["lists"][case["list"]]["names"]
	matched = [names[index] for index in subsample.devices.match_device_names(case["pattern"], names)]

	assert matched == case["matches"], case["rule"]


def test_every_list_says_where_its_names_came_from () -> None:

	"""The point of the file is that its names are real, so each says whose.

	A constructed list is allowed, for a case no machine here reports, but it
	has to say so rather than pass for something a backend produced.
	"""

	for key, entry in _VECTORS["lists"].items():
		assert str(entry.get("source", "")).strip(), f"list {key!r} does not say where its names came from"
		assert entry["names"], f"list {key!r} is empty"


def test_every_case_names_a_list_the_file_holds () -> None:

	"""A misspelt list would fail as a KeyError that names nothing useful."""

	for case in _VECTORS["cases"]:
		assert case["list"] in _VECTORS["lists"], f"{case['rule']!r} names an unknown list {case['list']!r}"
