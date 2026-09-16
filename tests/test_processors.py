"""Tests for subsample.processors — the processor declaration agrees with itself and with the code.

subsample.processors declares every processor parameter once: the forms it
takes, its allowed words, its limit, the sweep a CC binding travels, its
default or automatic source, and its unit.  These tests hold it to two things.

It must agree with itself.  A sweep outside its own limit, a condition naming a
word the controlling parameter does not accept, or a fixed default that is also
automatic is a declaration nobody can honour.

It must agree with the code, and the code must read it.  A declared default is
what the step compiler builds when a map leaves the parameter out, and changing
the declaration changes the compiled step.  The parser refuses at load every
value the declaration does not allow, and every CC binding a knob could not
honour.  A binding takes its missing ends, taper and resting place from the
declaration.  Every allowed word reaches a sound of its own.
"""

import dataclasses
import logging
import math
import typing

import numpy
import pytest

import subsample.processors
import subsample.query
import subsample.transform


_PROCESSORS = subsample.processors.PROCESSORS

_PARAMETERS: typing.Final[list[tuple[subsample.processors.Processor, subsample.processors.Parameter]]] = [
	(processor, parameter)
	for processor in _PROCESSORS.values()
	for parameter in processor.parameters
]
"""Every declared parameter with its processor, in declared order."""

_PARAMETER_IDS: typing.Final[list[str]] = [f"{processor.name}.{parameter.name}" for processor, parameter in _PARAMETERS]

_CONTEXT: typing.Final[dict[str, typing.Any]] = {
	"midi_note":      60,
	"target_bpm":     120.0,
	"resolution":     8,
	"reference_path": "/tmp/carrier.wav",
}
"""What the player hands spec_from_process when it compiles a step, fixed for these tests."""

_CONTEXT_VALUES: typing.Final[dict[str, typing.Any]] = {
	"note":       _CONTEXT["midi_note"],
	"tempo":      _CONTEXT["target_bpm"],
	"resolution": _CONTEXT["resolution"],
}
"""The value each automatic source that comes from the context gives a parameter."""


def _parameters_where (
	predicate: typing.Callable[[subsample.processors.Parameter], bool],
) -> tuple[list[tuple[subsample.processors.Processor, subsample.processors.Parameter]], list[str]]:

	"""The declared parameters a predicate selects, with their test ids."""

	selected = [(processor, parameter) for processor, parameter in _PARAMETERS if predicate(parameter)]

	return selected, [f"{processor.name}.{parameter.name}" for processor, parameter in selected]


def _every_limit (parameter: subsample.processors.Parameter) -> list[subsample.processors.Limit]:

	"""The parameter's own limit and each limit that holds only under a condition."""

	return [parameter.limit] + [conditional.limit for conditional in parameter.limits_when]


def _takes_integers (parameter: subsample.processors.Parameter) -> bool:

	"""True when the parameter's numbers are whole numbers."""

	return "integer" in parameter.forms and "number" not in parameter.forms


def _edges (limit: subsample.processors.Limit, integer: bool) -> tuple[list[float], list[float]]:

	"""The values at each bound a limit admits, and the values just past each bound it refuses."""

	step = 1 if integer else 0.5
	inside: list[float] = []
	outside: list[float] = []

	if limit.minimum is not None:
		inside.append(limit.minimum)
		outside.append(limit.minimum - step)

	if limit.maximum is not None:
		inside.append(limit.maximum)
		outside.append(limit.maximum + step)

	if limit.exclusive_minimum is not None:
		inside.append(limit.exclusive_minimum + step)
		outside.append(limit.exclusive_minimum)

	if limit.exclusive_maximum is not None:
		inside.append(limit.exclusive_maximum - step)
		outside.append(limit.exclusive_maximum)

	return inside, outside


def _compile (processor: str, params: typing.Mapping[str, typing.Any]) -> subsample.transform.TransformSpec:

	"""Compile one step of a processor, giving each required parameter its first declared word."""

	required = {
		parameter.name: parameter.choice_values[0]
		for parameter in _PROCESSORS[processor].parameters
		if parameter.required
	}
	step = subsample.query.ProcessorStep(name=processor, params=tuple({**required, **params}.items()))

	return subsample.transform.spec_from_process(subsample.query.ProcessSpec(steps=(step,)), **_CONTEXT)


class TestVocabularies:

	def test_every_form_is_known (self) -> None:

		"""A parameter's forms come from the closed list, and every form in it is used."""

		used = {form for _processor, parameter in _PARAMETERS for form in parameter.forms}

		assert used == set(subsample.processors.FORMS)

		for _processor, parameter in _PARAMETERS:
			assert parameter.forms, parameter.name
			assert len(set(parameter.forms)) == len(parameter.forms), parameter.name

	def test_every_unit_is_known (self) -> None:

		"""Unit words are a closed set Superconductor may act on, so a new or respelt word fails here."""

		used = {parameter.unit for _processor, parameter in _PARAMETERS if parameter.unit is not None}

		# The set covers the whole map, so the grammar uses the words the
		# processors do not; tests/test_midi_map_schema.py holds that half.
		assert used <= set(subsample.processors.UNITS)

	def test_every_automatic_source_is_known (self) -> None:

		"""Where an automatic value comes from is a closed set too, and every word in it is used."""

		used = {parameter.automatic for _processor, parameter in _PARAMETERS if parameter.automatic is not None}

		assert used == set(subsample.processors.AUTOMATIC_SOURCES)

	def test_every_taper_is_known (self) -> None:

		"""A sweep is linear or logarithmic, and nothing else."""

		for _processor, parameter in _PARAMETERS:
			assert parameter.taper in subsample.processors.TAPERS, parameter.name


class TestDeclarationAgreesWithItself:

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_at_most_one_of_default_automatic_and_required (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""The reference prints one of Default, Automatic and Required for each parameter, never two."""

		given = [parameter.default is not None, parameter.automatic is not None, parameter.required]

		assert sum(given) <= 1

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_choices_exactly_when_a_word_is_accepted (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A parameter that takes a word lists its words, once each, and no other parameter lists any."""

		assert ("choice" in parameter.forms) == bool(parameter.choices)
		assert len(set(parameter.choice_values)) == len(parameter.choice_values)

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_sweep_exactly_when_a_binding_can_drive_it (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A knob needs two ends, so every parameter a CC binding can drive declares a sweep, and no other does."""

		assert (parameter.sweep is not None) == parameter.bindable

		if parameter.sweep is not None:
			low, high = parameter.sweep

			assert low < high

			if _takes_integers(parameter):
				assert isinstance(low, int) and isinstance(high, int)

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_sweep_lies_inside_every_limit (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A binding with no ends of its own takes the sweep's, so the sweep can never break a limit.

		That includes a limit that holds only under a condition: otherwise a
		`bandwidth` knob with no range would be refused in `fm` mode because of
		the sweep the declaration chose."""

		if parameter.sweep is None:
			return

		for limit in _every_limit(parameter):
			assert limit.admits(parameter.sweep[0]) and limit.admits(parameter.sweep[1])

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_logarithmic_sweep_has_a_limit_above_zero (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A binding maps a logarithmic sweep by ratio, which only works when every allowed value is above zero."""

		if parameter.taper == "log":
			assert parameter.limit.above_zero()

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_fixed_default_is_a_value_the_parameter_allows (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""Leaving a parameter out must never produce a value writing it in would refuse."""

		default = parameter.default

		if default is None:
			return

		if isinstance(default, str):
			assert default in parameter.choice_values

		elif isinstance(default, bool):
			assert "boolean" in parameter.forms

		else:
			assert "number" in parameter.forms or "integer" in parameter.forms

			if _takes_integers(parameter):
				assert isinstance(default, int)

			for limit in _every_limit(parameter):
				assert limit.admits(default)

	@pytest.mark.parametrize(("processor", "parameter"), _PARAMETERS, ids=_PARAMETER_IDS)
	def test_conditions_name_sibling_words (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A condition names other parameters of the same processor, and words each of them accepts."""

		conditions = list(parameter.applies_when) + [conditional.when for conditional in parameter.limits_when]

		for condition in conditions:
			assert condition

			for name, words in condition.items():
				assert name != parameter.name
				assert words

				sibling = processor.parameter(name)

				assert set(words) <= set(sibling.choice_values), f"{name}: {words}"

	@pytest.mark.parametrize("processor", list(_PROCESSORS.values()), ids=list(_PROCESSORS))
	def test_names_are_unique_within_a_processor (self, processor: subsample.processors.Processor) -> None:

		"""A name, current or legacy, means one parameter of its processor."""

		current = list(processor.parameter_names)
		legacy = [name for parameter in processor.parameters for name in parameter.legacy_names]

		assert len(set(current)) == len(current)
		assert len(set(legacy)) == len(legacy)
		assert not set(legacy) & set(current)

	@pytest.mark.parametrize("processor", list(_PROCESSORS.values()), ids=list(_PROCESSORS))
	def test_shorthand_sets_a_number (self, processor: subsample.processors.Processor) -> None:

		"""`bit_depth: 12` sets `bits`, so a shorthand names a parameter that takes a number."""

		if processor.shorthand is not None:
			assert processor.parameter(processor.shorthand).bindable

	def test_legacy_processor_names (self) -> None:

		"""An old processor name is not a current one, belongs to one processor, and implies declared words."""

		seen: set[str] = set()

		for processor in _PROCESSORS.values():
			for legacy in processor.legacy_names:
				assert legacy.name not in _PROCESSORS
				assert legacy.name not in seen

				seen.add(legacy.name)

				for name, word in legacy.implies.items():
					assert word in processor.parameter(name).choice_values


def _prose () -> list[typing.Any]:

	"""Every title and description the declaration publishes, named by where it sits."""

	cases = []

	for processor in _PROCESSORS.values():
		cases.append(pytest.param(processor.title, processor.description, None, id=processor.name))

		for parameter in processor.parameters:
			cases.append(pytest.param(
				parameter.title, parameter.description, parameter.unit,
				id=f"{processor.name}.{parameter.name}",
			))

			for choice in parameter.choices:
				cases.append(pytest.param(
					choice.title, choice.description, None,
					id=f"{processor.name}.{parameter.name}.{choice.value}",
				))

	return cases


class TestProseIsWritten:

	"""The titles and descriptions subsystem.co publishes and Superconductor labels its controls with."""

	@pytest.mark.parametrize(("title", "description", "unit"), _prose())
	def test_every_term_has_a_title_and_a_description (
		self, title: str, description: str, unit: typing.Optional[str],
	) -> None:

		"""A reference entry and a control's label both need words to show."""

		assert title.strip()
		assert description.strip()

	@pytest.mark.parametrize(("title", "description", "unit"), _prose())
	def test_a_title_is_a_label_in_sentence_case (
		self, title: str, description: str, unit: typing.Optional[str],
	) -> None:

		"""A title starts with a capital and is a label, not a sentence."""

		assert title[0].isupper()
		assert not title.endswith(".")

	@pytest.mark.parametrize(("title", "description", "unit"), _prose())
	def test_a_description_is_whole_sentences (
		self, title: str, description: str, unit: typing.Optional[str],
	) -> None:

		"""A description starts a sentence and ends one, so it reads alone at its anchor."""

		assert description[0].isupper() or description.startswith("`")
		assert description.endswith(".")

	@pytest.mark.parametrize(("title", "description", "unit"), _prose())
	def test_prose_never_carries_an_em_dash (
		self, title: str, description: str, unit: typing.Optional[str],
	) -> None:

		"""subsystem.co refuses to publish an em dash."""

		assert "\u2014" not in title
		assert "\u2014" not in description

	@pytest.mark.parametrize(("title", "description", "unit"), _prose())
	def test_a_description_leaves_the_unit_to_the_declaration (
		self, title: str, description: str, unit: typing.Optional[str],
	) -> None:

		"""The unit is published beside the description, so the prose does not repeat it."""

		if unit is not None:
			assert unit not in description.split()


class TestDefaultsAgreeWithTheCompiler:

	_FIXED, _FIXED_IDS = _parameters_where(lambda parameter: parameter.default is not None)
	_FROM_CONTEXT, _FROM_CONTEXT_IDS = _parameters_where(lambda parameter: parameter.automatic in _CONTEXT_VALUES)
	_MEASURED, _MEASURED_IDS = _parameters_where(lambda parameter: parameter.automatic in ("sample", "mode"))

	@pytest.mark.parametrize(("processor", "parameter"), _FIXED, ids=_FIXED_IDS)
	def test_leaving_a_parameter_out_compiles_its_declared_default (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""The declared default is what the compiler uses: leaving it out and writing it in build the same step."""

		assert _compile(processor.name, {}) == _compile(processor.name, {parameter.name: parameter.default})

	@pytest.mark.parametrize(("processor", "parameter"), _FROM_CONTEXT, ids=_FROM_CONTEXT_IDS)
	def test_automatic_value_comes_from_where_the_note_is_played (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A `note`, `tempo` or `resolution` parameter left out takes the value the player passes in."""

		assert parameter.automatic is not None

		explicit = {parameter.name: _CONTEXT_VALUES[parameter.automatic]}

		assert _compile(processor.name, {}) == _compile(processor.name, explicit)

	@pytest.mark.parametrize(("processor", "parameter"), _MEASURED, ids=_MEASURED_IDS)
	def test_automatic_value_is_left_to_the_effect (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A `sample` or `mode` parameter left out reaches the effect unset, for it to work out.

		Writing a value in changes exactly one field of the compiled step, and
		that field is unset when the parameter is left out."""

		assert parameter.sweep is not None

		omitted = _compile(processor.name, {}).steps
		explicit = _compile(processor.name, {parameter.name: parameter.sweep[0]}).steps

		assert len(omitted) == len(explicit) == 1

		before = dataclasses.asdict(omitted[0])
		after = dataclasses.asdict(explicit[0])
		changed = [field for field in before if before[field] != after[field]]

		assert len(changed) == 1
		assert before[changed[0]] is None


def _load (processor: str, params: typing.Mapping[str, typing.Any]) -> subsample.query.ProcessSpec:

	"""Parse one step of a processor in strict mode, giving each required parameter it lacks its first word."""

	required = {
		parameter.name: parameter.choice_values[0]
		for parameter in _PROCESSORS[processor].parameters
		if parameter.required and parameter.name not in params
	}

	return subsample.query.parse_process([{processor: {**required, **params}}], "test")


def _binding (processor: str, name: str, binding: typing.Mapping[str, typing.Any]) -> subsample.query.CcBinding:

	"""The CcBinding the parser builds for one parameter's ``{cc: ...}`` mapping."""

	value = _load(processor, {name: dict(binding)}).steps[0].get(name)

	assert isinstance(value, subsample.query.CcBinding)

	return value


_LIMITED, _LIMITED_IDS = _parameters_where(lambda parameter: not parameter.limit.is_open() or bool(parameter.limits_when))
_WHOLE, _WHOLE_IDS = _parameters_where(_takes_integers)
_WORDS, _WORDS_IDS = _parameters_where(lambda parameter: bool(parameter.choices))
_WORDS_ONLY, _WORDS_ONLY_IDS = _parameters_where(lambda parameter: bool(parameter.choices) and "path" not in parameter.forms)
_BINDABLE, _BINDABLE_IDS = _parameters_where(lambda parameter: parameter.bindable)
_BOUNDED_BINDABLE, _BOUNDED_BINDABLE_IDS = _parameters_where(lambda parameter: parameter.bindable and not parameter.limit.is_open())
_NOT_BINDABLE, _NOT_BINDABLE_IDS = _parameters_where(lambda parameter: not parameter.bindable)
_REQUIRED, _REQUIRED_IDS = _parameters_where(lambda parameter: parameter.required)
_FIXED, _FIXED_IDS = _parameters_where(lambda parameter: parameter.default is not None)


class TestParserEnforcesTheDeclaration:

	@pytest.mark.parametrize(("processor", "parameter"), _WORDS, ids=_WORDS_IDS)
	def test_every_declared_word_loads (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""Each word the declaration allows loads in strict mode."""

		for word in parameter.choice_values:
			spec = _load(processor.name, {parameter.name: word})

			assert spec.steps[0].get(parameter.name) == word

	@pytest.mark.parametrize(("processor", "parameter"), _WORDS_ONLY, ids=_WORDS_ONLY_IDS)
	def test_an_undeclared_word_is_refused (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A word the declaration does not allow is refused when the map loads."""

		with pytest.raises(ValueError, match=f"{parameter.name} must be"):
			_load(processor.name, {parameter.name: "bogus"})

	@pytest.mark.parametrize(("processor", "parameter"), _WORDS_ONLY, ids=_WORDS_ONLY_IDS)
	def test_words_are_written_exactly (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""`AM` is not `am`: a word in another letter case is refused, for every parameter alike."""

		for word in parameter.choice_values:
			with pytest.raises(ValueError, match=f"{parameter.name} must be"):
				_load(processor.name, {parameter.name: word.upper()})

	@pytest.mark.parametrize(("processor", "parameter"), _LIMITED, ids=_LIMITED_IDS)
	def test_every_limit_holds_at_its_edges (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A value at each declared edge loads and a value just past it is refused.

		A limit that holds only under a condition is tested with the condition met."""

		cases: list[tuple[dict[str, str], subsample.processors.Limit]] = [({}, parameter.limit)]

		for conditional in parameter.limits_when:
			cases.append(({name: words[0] for name, words in conditional.when.items()}, conditional.limit))

		for context, limit in cases:
			inside, outside = _edges(limit, _takes_integers(parameter))

			assert inside and outside

			for value in inside:
				_load(processor.name, {**context, parameter.name: value})

			for value in outside:
				with pytest.raises(ValueError, match=f"{parameter.name} must be"):
					_load(processor.name, {**context, parameter.name: value})

	def test_a_conditional_limit_names_its_condition (self) -> None:

		"""The FM and SSB channel filters start at 300 Hz, and the refusal says that is why 250 is too low."""

		with pytest.raises(ValueError, match="bandwidth must be a number above 300 Hz when mode is fm or ssb"):
			_load("radio", {"mode": "fm", "bandwidth": 250})

		_load("radio", {"mode": "am", "bandwidth": 250})

	@pytest.mark.parametrize(("processor", "parameter"), _WHOLE, ids=_WHOLE_IDS)
	def test_whole_numbers_refuse_a_fraction (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""`grid: 2.5` used to be cut to 2 without a word; `grid: 2.0` is a whole number and loads as 2."""

		lowest = parameter.limit.minimum

		if lowest is None:
			assert parameter.sweep is not None
			lowest = parameter.sweep[0]

		with pytest.raises(ValueError, match=f"{parameter.name} must be"):
			_load(processor.name, {parameter.name: lowest + 0.5})

		value = _load(processor.name, {parameter.name: float(lowest)}).steps[0].get(parameter.name)

		assert value == lowest and isinstance(value, int)

	@pytest.mark.parametrize(("processor", "parameter"), _BINDABLE, ids=_BINDABLE_IDS)
	def test_a_number_parameter_refuses_anything_else (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A word, a boolean, a list, infinity or not-a-number is never a number a parameter can take."""

		for bad in ("loud", True, [1, 2], float("inf"), float("nan")):
			with pytest.raises(ValueError, match=f"{parameter.name} must be"):
				_load(processor.name, {parameter.name: bad})

	@pytest.mark.parametrize(("processor", "parameter"), _REQUIRED, ids=_REQUIRED_IDS)
	def test_a_required_parameter_is_refused_when_missing (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A vocoder with no carrier used to load and be dropped at render; now the map says so at load."""

		for entry in (processor.name, {processor.name: {}}, {processor.name: True}):
			with pytest.raises(ValueError, match=f"{processor.name} needs {parameter.name}"):
				subsample.query.parse_process([entry], "test")

	def test_values_are_checked_in_lenient_mode_too (self) -> None:

		"""Lenient mode forgives names it does not know, never a value outside what a parameter allows."""

		subsample.query.set_strict_mode(False)

		try:
			with pytest.raises(ValueError, match="strength must be a number from 0 to 1"):
				subsample.query.parse_process([{"pad_quantize": {"strength": 7}}], "test")
		finally:
			subsample.query.set_strict_mode(True)


class TestCcBindings:

	@pytest.mark.parametrize(("processor", "parameter"), _NOT_BINDABLE, ids=_NOT_BINDABLE_IDS)
	def test_a_binding_is_refused_where_a_knob_cannot_set_the_parameter (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""A word, a note name or a file path is chosen in the map, not swept by a knob."""

		with pytest.raises(ValueError, match="which a CC binding cannot set"):
			_load(processor.name, {parameter.name: {"cc": 20}})

	@pytest.mark.parametrize(("processor", "parameter"), _BINDABLE, ids=_BINDABLE_IDS)
	def test_a_binding_takes_each_missing_end_from_the_sweep (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""With no ends a binding travels the parameter's sweep; with one end it takes the other from the sweep."""

		assert parameter.sweep is not None

		low, high = parameter.sweep
		bare = _binding(processor.name, parameter.name, {"cc": 20})

		assert (bare.min_val, bare.max_val, bare.taper) == (low, high, parameter.taper)

		middle = bare.at_fraction(0.5)
		with_min = _binding(processor.name, parameter.name, {"cc": 20, "min": middle})
		with_max = _binding(processor.name, parameter.name, {"cc": 20, "max": middle})

		assert (with_min.min_val, with_min.max_val) == (middle, high)
		assert (with_max.min_val, with_max.max_val) == (low, middle)

	@pytest.mark.parametrize(("processor", "parameter"), _BINDABLE, ids=_BINDABLE_IDS)
	def test_a_binding_rests_as_if_the_knob_were_not_there (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""Before its first CC a binding rests at the parameter's automatic value, or at its default."""

		binding = _binding(processor.name, parameter.name, {"cc": 20})

		if parameter.automatic is not None:
			assert binding.default_value is None
		else:
			assert binding.default_value == parameter.default

	def test_a_binding_whose_travel_leaves_out_the_default_rests_at_its_middle (self) -> None:

		"""A 200 Hz to 2 kHz knob cannot reach the 16 kHz default, so it rests halfway along its logarithmic travel."""

		binding = _binding("filter_low", "freq", {"cc": 74, "min": 200, "max": 2000})

		assert binding.default_value == pytest.approx(math.sqrt(200 * 2000))

	def test_an_explicit_default_wins (self) -> None:

		"""`default:` says where the knob rests, whatever the parameter would do without it."""

		assert _binding("compress", "threshold", {"cc": 20, "default": -12}).default_value == -12.0

	@pytest.mark.parametrize(("processor", "parameter"), _BOUNDED_BINDABLE, ids=_BOUNDED_BINDABLE_IDS)
	def test_a_binding_end_or_default_outside_the_limit_is_refused (
		self, processor: subsample.processors.Processor, parameter: subsample.processors.Parameter,
	) -> None:

		"""Otherwise part of the knob's travel would be held at the limit, doing nothing."""

		_inside, outside = _edges(parameter.limit, integer=False)

		for key in ("min", "max", "default"):
			for value in outside:
				with pytest.raises(ValueError, match="outside what"):
					_load(processor.name, {parameter.name: {"cc": 20, key: value}})

	def test_a_binding_end_outside_a_conditional_limit_is_refused (self) -> None:

		"""An FM bandwidth knob may not start below the FM channel filter's 300 Hz floor."""

		with pytest.raises(ValueError, match="above 300 Hz when mode is fm or ssb"):
			_load("radio", {"mode": "fm", "bandwidth": {"cc": 70, "min": 250}})

	def test_a_binding_end_that_is_not_a_number_is_refused (self) -> None:

		"""`min: low` used to fail with a bare conversion error."""

		with pytest.raises(ValueError, match="CC binding min 'low', which is not a number"):
			_load("filter_low", {"freq": {"cc": 74, "min": "low"}})

	def test_a_logarithmic_binding_moves_in_equal_ratios (self) -> None:

		"""Each CC step on a frequency knob is the same musical interval, from 20 Hz to 20 kHz."""

		binding = _binding("filter_low", "freq", {"cc": 74})

		assert (binding.resolve(0), binding.resolve(127)) == (20.0, 20000.0)
		assert binding.resolve(64) / binding.resolve(0) == pytest.approx(binding.resolve(127) / binding.resolve(63))

	def test_a_linear_binding_moves_in_equal_steps (self) -> None:

		"""A knob on a time or a level moves the same amount for each CC step."""

		binding = _binding("compress", "knee", {"cc": 20})

		assert binding.resolve(64) - binding.resolve(0) == pytest.approx(binding.resolve(127) - binding.resolve(63))

	def test_the_filter_knob_with_no_range_now_sweeps_the_audible_range (self) -> None:

		"""`filter_low: {freq: {cc: 74}}` used to travel 0 to 1 Hz and rest at 0.5 Hz, where the filter does nothing."""

		spec = _load("filter_low", {"freq": {"cc": 74}})

		assert subsample.transform.spec_from_process(spec).steps[0].freq == 16000.0
		assert subsample.transform.spec_from_process(spec, cc_omni={74: 0}).steps[0].freq == 20.0
		assert subsample.transform.spec_from_process(spec, cc_omni={74: 127}).steps[0].freq == 20000.0

	def test_an_automatic_parameter_rests_at_its_automatic_value (self) -> None:

		"""A threshold knob leaves the compressor on its measured threshold until it is turned."""

		spec = _load("compress", {"threshold": {"cc": 20}})

		assert subsample.transform.spec_from_process(spec).steps[0].threshold_db is None
		assert subsample.transform.spec_from_process(spec, cc_omni={20: 0}).steps[0].threshold_db == -60.0

	def test_a_tempo_knob_rests_at_the_session_tempo (self) -> None:

		"""Until its first CC a tempo knob follows the session tempo, not the middle of its sweep."""

		spec = _load("stretch_quantize", {"tempo": {"cc": 2}})

		assert subsample.transform.spec_from_process(spec, target_bpm=97.0).steps[0].target_bpm == 97.0


class TestNoEffectWarning:

	@pytest.mark.parametrize(("processor", "params", "warns"), [
		("distort", {"mode": "fold", "bit_depth": 4}, True),
		("distort", {"bit_depth": 4}, True),
		("distort", {"mode": "bit_crush", "bit_depth": 4}, False),
		("distort", {"mode": "hard_clip", "downsample_factor": 8}, True),
		("distort", {"mode": "downsample", "downsample_factor": 8}, False),
		("radio", {"tune": 150}, True),
		("radio", {"mode": "ssb", "tune": 150}, False),
		("radio", {"mode": "am", "demod": "ssb", "tune": 150}, False),
		("radio", {"mode": "ssb", "demod": "am", "tune": 150}, True),
	])
	def test_a_parameter_with_no_effect_loads_with_a_warning (
		self, caplog: pytest.LogCaptureFixture, processor: str, params: dict[str, typing.Any], warns: bool,
	) -> None:

		"""The map still loads, since the value may be parked on purpose, and the log says it does nothing."""

		with caplog.at_level(logging.WARNING, logger="subsample.query"):
			spec = _load(processor, params)

		assert spec.steps
		assert ("has no effect unless" in caplog.text) == warns

	def test_the_warning_names_the_parameter_and_what_it_needs (self, caplog: pytest.LogCaptureFixture) -> None:

		"""The log line is enough to fix the map without reading the reference."""

		with caplog.at_level(logging.WARNING, logger="subsample.query"):
			_load("radio", {"tune": 150})

		assert "radio tune has no effect unless demod is ssb, or mode is ssb and demod is matched" in caplog.text


class TestCompilerReadsTheDeclaration:

	@pytest.mark.parametrize(("processor", "parameter"), _FIXED, ids=_FIXED_IDS)
	def test_a_changed_default_changes_what_compiles (
		self,
		monkeypatch: pytest.MonkeyPatch,
		processor: subsample.processors.Processor,
		parameter: subsample.processors.Parameter,
	) -> None:

		"""No default is written a second time: change the declaration and the compiled step follows it."""

		if parameter.choices:
			replacement: typing.Any = next(word for word in parameter.choice_values if word != parameter.default)
		else:
			assert parameter.sweep is not None
			replacement = parameter.sweep[1] if parameter.sweep[1] != parameter.default else parameter.sweep[0]

		changed = dataclasses.replace(parameter, default=replacement)
		patched = dataclasses.replace(
			processor,
			parameters=tuple(changed if other.name == parameter.name else other for other in processor.parameters),
		)

		monkeypatch.setitem(subsample.processors.PROCESSORS, processor.name, patched)

		assert _compile(processor.name, {}) == _compile(processor.name, {parameter.name: replacement})
		assert _compile(processor.name, {}) != _compile(processor.name, {parameter.name: parameter.default})

	def test_a_step_built_in_code_is_held_inside_its_limit (
		self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
	) -> None:

		"""The parser refuses `strength: 7`; a step built in code gets it held at 1, and the log says so once."""

		monkeypatch.setattr(subsample.transform, "_WARN_ONCE_SEEN", set())
		step = subsample.query.ProcessorStep(name="pad_quantize", params=(("strength", 7.0),))

		with caplog.at_level(logging.WARNING, logger="subsample.transform"):
			spec = subsample.transform.spec_from_process(subsample.query.ProcessSpec(steps=(step,)), target_bpm=120.0)

		assert spec.steps[0].amount == 1.0
		assert "pad_quantize strength of 7.0 is outside what it allows, so it is held at 1" in caplog.text


_RATE: typing.Final[int] = 22050

_SIGNAL: typing.Final[numpy.ndarray] = numpy.stack([
	0.8 * numpy.sin(2.0 * numpy.pi * 220.0 * numpy.arange(int(_RATE * 0.3)) / _RATE),
	0.6 * numpy.sin(2.0 * numpy.pi * 330.0 * numpy.arange(int(_RATE * 0.3)) / _RATE),
], axis=1).astype(numpy.float32)
"""A short stereo test tone, different in each channel, for rendering effects."""


def _render (processor: str, params: typing.Mapping[str, typing.Any]) -> numpy.ndarray:

	"""Compile one step and render the test tone through its effect."""

	step = _compile(processor, params).steps[0]
	handler = subsample.transform.TransformProcessor._HANDLERS[type(step)]

	return typing.cast(numpy.ndarray, handler(_SIGNAL.copy(), _RATE, None, step))


def _all_distinct (renders: dict[str, numpy.ndarray]) -> list[tuple[str, str]]:

	"""The pairs of words whose renders are the same sound."""

	words = list(renders)

	return [
		(first, second)
		for index, first in enumerate(words)
		for second in words[index + 1:]
		if renders[first].shape == renders[second].shape and numpy.allclose(renders[first], renders[second], atol=1e-6)
	]


class TestEveryWordHasItsOwnSound:

	def test_every_distort_mode_shapes_the_sound_its_own_way (self) -> None:

		"""An unknown mode used to pass the audio through unchanged; every declared mode must reach its own branch."""

		renders = {
			mode: _render("distort", {"mode": mode, "drive": 12.0, "tone": 1.0})
			for mode in _PROCESSORS["distort"].parameter("mode").choice_values
		}

		assert not _all_distinct(renders)

		for mode, audio in renders.items():
			assert not numpy.allclose(audio, _SIGNAL, atol=1e-6), mode

	def test_every_radio_mode_is_its_own_transmission (self) -> None:

		"""am, lw, fm and ssb each modulate, filter and demodulate differently."""

		renders = {mode: _render("radio", {"mode": mode}) for mode in _PROCESSORS["radio"].parameter("mode").choice_values}

		assert not _all_distinct(renders)

	def test_every_radio_demodulator_is_its_own_receiver (self) -> None:

		"""Each named demodulator sounds different, and matched is the transmission's own."""

		demods = [word for word in _PROCESSORS["radio"].parameter("demod").choice_values if word != "matched"]
		renders = {demod: _render("radio", {"mode": "ssb", "demod": demod}) for demod in demods}

		assert not _all_distinct(renders)
		assert numpy.allclose(_render("radio", {"mode": "ssb", "demod": "matched"}), renders["ssb"], atol=1e-6)

	def test_both_radio_stereo_words_are_heard (self) -> None:

		"""mono collapses the two channels to one receiver; stereo keeps a receiver for each."""

		renders = {word: _render("radio", {"stereo": word}) for word in _PROCESSORS["radio"].parameter("stereo").choice_values}

		assert not _all_distinct(renders)

	def test_every_dither_is_its_own_noise (self) -> None:

		"""none, triangular and rectangular dither each leave a different grain at 4 bits."""

		renders = {
			word: _render("bit_depth", {"bits": 4, "dither": word})
			for word in _PROCESSORS["bit_depth"].parameter("dither").choice_values
		}

		assert not _all_distinct(renders)

	def test_every_hpss_keep_builds_its_own_step (self) -> None:

		"""harmonic and percussive each compile to their own separation."""

		built = {word: type(_compile("hpss", {"keep": word}).steps[0]) for word in _PROCESSORS["hpss"].parameter("keep").choice_values}

		assert len(set(built.values())) == len(built)
