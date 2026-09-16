"""Tests for the MIDI map's declared parameters — processors, CC bindings and order entries.

subsample.processors declares which parameters each processor accepts, and
subsample.query.PROCESSOR_PARAMETERS is the parser's view of those names.  The
parser refuses anything else in strict mode, so the declaration must match the
code that reads parameters exactly: a parameter read but not declared could
never be set, and one declared but never read would be accepted and silently
ignored — the bug this declaration exists to prevent.  tests/test_processors.py
holds the rest of each declaration (defaults, words, limits) to the code.

The agreement test reads the package's source rather than a second hand-kept
list.  A parameter read is any ``<name>.get("<literal>")`` or
``_resolved(<name>, "<literal>", ...)`` inside an ``if`` that tests a
processor's name (``proc.name == "compress"``, ``step.name in
("stretch_quantize", "pad_quantize")``), which is how spec_from_process and the
player's segment mode read one.  A new reader written that way is found
automatically; one written another way reads a name this test cannot see, and
the test fails on the declaration it leaves unread.
"""

import ast
import logging
import pathlib
import typing

import pytest

import subsample.processors
import subsample.query
import subsample.transform


_PACKAGE_DIR = pathlib.Path(subsample.query.__file__).parent

_PROCESSOR_NAMES = frozenset(subsample.query.PROCESSOR_PARAMETERS) | subsample.query._LEGACY_PROCESSOR_NAMES


def _guarded_names (test: ast.expr) -> frozenset[str]:

	"""The processor names an ``if`` test selects on, or none when it tests something else."""

	if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
		return frozenset()

	left = test.left

	if not (isinstance(left, ast.Attribute) and left.attr == "name"):
		return frozenset()

	operator = test.ops[0]
	right = test.comparators[0]

	if isinstance(operator, ast.Eq) and isinstance(right, ast.Constant) and isinstance(right.value, str):
		candidates = frozenset({right.value})

	elif isinstance(operator, ast.In) and isinstance(right, (ast.Tuple, ast.List, ast.Set)):
		candidates = frozenset(
			element.value for element in right.elts
			if isinstance(element, ast.Constant) and isinstance(element.value, str)
		)

	else:
		return frozenset()

	return candidates & _PROCESSOR_NAMES


def _read_literal (call: ast.Call) -> typing.Optional[str]:

	"""The parameter name a call reads, or None when the call reads none.

	``step.get("bits")`` reads `bits`, and so does spec_from_process's
	``_resolved(proc, "bits", ...)``, which gets the value, its declared default
	and its limit in one call."""

	func = call.func

	if isinstance(func, ast.Attribute) and func.attr == "get" and isinstance(func.value, ast.Name):
		argument = call.args[0] if call.args else None

	elif isinstance(func, ast.Name) and func.id == "_resolved":
		argument = call.args[1] if len(call.args) > 1 else None

	else:
		return None

	if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
		return argument.value

	return None


def _parameter_reads () -> dict[str, set[str]]:

	"""Every parameter name the package reads under a test of each processor's name."""

	reads: dict[str, set[str]] = {}

	for module in sorted(_PACKAGE_DIR.glob("*.py")):
		tree = ast.parse(module.read_text(encoding="utf-8"))

		for node in ast.walk(tree):
			if not isinstance(node, ast.If):
				continue

			names = _guarded_names(node.test)

			if not names:
				continue

			literals = {
				literal
				for statement in node.body
				for call in ast.walk(statement)
				if isinstance(call, ast.Call)
				for literal in [_read_literal(call)]
				if literal is not None
			}

			for name in names:
				reads.setdefault(name, set()).update(literals)

	return reads


def _spec_from_process_branches () -> set[str]:

	"""The processor names spec_from_process dispatches on."""

	tree = ast.parse(pathlib.Path(subsample.transform.__file__).read_text(encoding="utf-8"))
	function = next(
		node for node in ast.walk(tree)
		if isinstance(node, ast.FunctionDef) and node.name == "spec_from_process"
	)

	return {
		name
		for node in ast.walk(function) if isinstance(node, ast.If)
		for name in _guarded_names(node.test)
	}


@pytest.fixture
def lenient () -> typing.Iterator[None]:

	"""Parse in lenient mode for one test, and always restore strict mode."""

	subsample.query.set_strict_mode(False)

	try:
		yield
	finally:
		subsample.query.set_strict_mode(True)


class TestDeclarationAgreesWithReaders:

	def test_every_declared_parameter_is_read (self) -> None:

		"""A declared parameter nothing reads would be accepted and silently ignored."""

		reads = _parameter_reads()
		unread = {
			name: sorted(set(parameters) - reads.get(name, set()))
			for name, parameters in subsample.query.PROCESSOR_PARAMETERS.items()
			if set(parameters) - reads.get(name, set())
		}

		assert not unread, f"declared in PROCESSOR_PARAMETERS but never read: {unread}"

	def test_every_parameter_read_is_declared (self) -> None:

		"""A parameter read but not declared could never be set: the parser would refuse it."""

		undeclared = {
			name: sorted(names - set(subsample.query.PROCESSOR_PARAMETERS[name]))
			for name, names in _parameter_reads().items()
			if name in subsample.query.PROCESSOR_PARAMETERS
			and names - set(subsample.query.PROCESSOR_PARAMETERS[name])
		}

		assert not undeclared, f"read but not declared in PROCESSOR_PARAMETERS: {undeclared}"

	def test_every_declared_processor_is_compiled (self) -> None:

		"""Every declared processor has a branch in spec_from_process, and every branch is declared."""

		branches = _spec_from_process_branches()

		assert set(subsample.query.PROCESSOR_PARAMETERS) <= branches
		assert branches <= _PROCESSOR_NAMES

	def test_legacy_parameter_names_translate_to_declared_ones (self) -> None:

		"""A legacy alias is only worth keeping if the name it becomes is accepted."""

		for (processor, _legacy), current in subsample.query._LEGACY_PROCESSOR_PARAMS.items():
			assert current in subsample.query.PROCESSOR_PARAMETERS[processor]

	def test_scalar_shorthands_bind_declared_parameters (self) -> None:

		"""`bit_depth: 12` sets `bits`, so `bits` must be a parameter bit_depth accepts."""

		for processor, parameter in subsample.query._SCALAR_PROCESSOR_PARAMS.items():
			assert parameter in subsample.query.PROCESSOR_PARAMETERS[processor]

	def test_scorer_parameters_belong_to_real_scorers (self) -> None:

		"""A scorer's declared parameters name a scorer an order entry can use."""

		assert set(subsample.query.SCORER_PARAMETERS) <= set(subsample.query.valid_order_names())


class TestProcessorParameters:

	def test_misspelt_parameter_is_refused (self) -> None:

		"""`treshold` used to load and leave the compressor on its automatic threshold."""

		with pytest.raises(ValueError, match="processor 'compress' has no parameter 'treshold'"):
			subsample.query.parse_process([{"compress": {"treshold": -20}}], "kick")

	def test_refusal_names_the_valid_parameters (self) -> None:

		"""The message lists what the processor does accept, in its declared order."""

		with pytest.raises(ValueError, match="Valid parameters: freq, resonance"):
			subsample.query.parse_process([{"filter_low": {"frequency": 500}}], "kick")

	def test_processor_without_parameters_refuses_any (self) -> None:

		"""reverse takes no parameters, and says so."""

		with pytest.raises(ValueError, match="Valid parameters: none"):
			subsample.query.parse_process([{"reverse": {"speed": 2}}], "kick")

	def test_every_declared_parameter_is_accepted (self) -> None:

		"""A value each declared parameter allows parses, for every processor.

		The value is the parameter's first word, the low end of its sweep, or
		its lowest allowed number; a parameter the processor requires is given
		alongside."""

		def allowed (parameter: subsample.processors.Parameter) -> typing.Any:
			if parameter.choices:
				return parameter.choice_values[0]
			if parameter.sweep is not None:
				return parameter.sweep[0]
			return parameter.limit.minimum

		for name, processor in subsample.processors.PROCESSORS.items():
			required = {p.name: allowed(p) for p in processor.parameters if p.required}

			for parameter in processor.parameters:
				value = allowed(parameter)
				spec = subsample.query.parse_process([{name: {**required, parameter.name: value}}], "test")

				assert spec.steps[0].get(parameter.name) == value, f"{name}.{parameter.name}"

	def test_legacy_parameter_name_still_accepted (self) -> None:

		"""`saturate: {amount: 6}` is a legacy alias for `drive`, not a typo."""

		spec = subsample.query.parse_process([{"saturate": {"amount": 6}}], "test")

		assert spec.steps[0].get("drive") == 6

	def test_legacy_processor_name_checks_the_current_parameters (self) -> None:

		"""`beat_quantize` accepts what `stretch_quantize` accepts, and refuses the rest."""

		spec = subsample.query.parse_process([{"beat_quantize": {"grid": 8}}], "test")

		assert spec.steps[0].get("grid") == 8

		with pytest.raises(ValueError, match="has no parameter 'gird'"):
			subsample.query.parse_process([{"beat_quantize": {"gird": 8}}], "test")

	def test_lenient_mode_warns_and_drops_the_parameter (
		self, lenient: None, caplog: pytest.LogCaptureFixture,
	) -> None:

		"""Lenient mode keeps loading an older map, names the parameter, and does not pass it on."""

		with caplog.at_level(logging.WARNING, logger="subsample.query"):
			spec = subsample.query.parse_process([{"compress": {"treshold": -20, "ratio": 8}}], "kick")

		assert spec.steps[0].params == (("ratio", 8),)
		assert "treshold" in caplog.text


class TestCcBindingKeys:

	def test_misspelt_binding_key_is_refused (self) -> None:

		"""`mni` used to be ignored, leaving the binding on its default minimum."""

		with pytest.raises(ValueError, match=r"unknown CC binding key\(s\) \['mni'\]"):
			subsample.query.parse_process(
				[{"filter_low": {"freq": {"cc": 74, "mni": 200, "max": 16000}}}], "kick",
			)

	def test_mapping_without_cc_is_refused (self) -> None:

		"""`{CC: 74}` used to reach the processor as a dict and fail when the step was compiled."""

		with pytest.raises(ValueError, match="mapping without a 'cc' key"):
			subsample.query.parse_process([{"filter_low": {"freq": {"CC": 74}}}], "kick")

	def test_every_binding_key_is_accepted (self) -> None:

		"""A binding using every key parses into a CcBinding carrying each value."""

		spec = subsample.query.parse_process(
			[{"filter_low": {"freq": {"cc": 74, "channel": 2, "min": 200, "max": 16000, "default": 800}}}],
			"kick",
		)
		binding = spec.steps[0].get("freq")

		assert isinstance(binding, subsample.query.CcBinding)
		assert (binding.cc, binding.channel, binding.min_val, binding.max_val, binding.default) == (74, 2, 200.0, 16000.0, 800.0)

	def test_lenient_mode_warns_about_binding_keys (
		self, lenient: None, caplog: pytest.LogCaptureFixture,
	) -> None:

		"""Lenient mode builds the binding from the keys it knows and names the rest."""

		with caplog.at_level(logging.WARNING, logger="subsample.query"):
			spec = subsample.query.parse_process(
				[{"filter_low": {"freq": {"cc": 74, "mni": 200}}}], "kick",
			)

		assert isinstance(spec.steps[0].get("freq"), subsample.query.CcBinding)
		assert "mni" in caplog.text


class TestOrderEntryKeys:

	def test_misspelt_dir_is_refused (self) -> None:

		"""`dri: asc` used to pass as a scorer parameter, and the order ran the default way."""

		with pytest.raises(ValueError, match=r"order entry for 'by: duration' has unknown key\(s\) \['dri'\]"):
			subsample.query.parse_select({"order": {"by": "duration", "dri": "asc"}}, "kick")

	def test_scorer_parameter_is_accepted (self) -> None:

		"""beat_match declares `pattern`, so an order entry may give it."""

		specs = subsample.query.parse_select(
			{"order": {"by": "beat_match", "pattern": [1, 0, 1, 0]}}, "kick",
		)

		assert dict(specs[0].order[0].params)["pattern"] == (1.0, 0.0, 1.0, 0.0)

	def test_parameter_of_another_scorer_is_refused (self) -> None:

		"""`pattern` belongs to beat_match; on `by: duration` it is an unknown key."""

		with pytest.raises(ValueError, match=r"Valid keys: by, dir\."):
			subsample.query.parse_select({"order": {"by": "duration", "pattern": [1, 0]}}, "kick")

	def test_lenient_mode_warns_and_drops_the_key (
		self, lenient: None, caplog: pytest.LogCaptureFixture,
	) -> None:

		"""Lenient mode keeps the clause, without the unknown key among its parameters."""

		with caplog.at_level(logging.WARNING, logger="subsample.query"):
			specs = subsample.query.parse_select({"order": {"by": "duration", "dri": "asc"}}, "kick")

		assert specs[0].order[0].params == ()
		assert "dri" in caplog.text
