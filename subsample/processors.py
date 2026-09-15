"""Every processor a MIDI map's ``process:`` list can name, and every parameter each one takes.

This is the one declaration of the processor vocabulary, in the names and units
a map uses.  Each parameter declares what it accepts (its forms and allowed
words), the values it refuses (its limit), the travel a CC binding sweeps when
the binding gives no ends of its own, its default or where an automatic value
comes from, its unit, and when it has any effect at all.

Everything reads it.  subsample.query takes its processor and parameter names
from here, refuses at load a value a declaration does not allow, and builds a
CC binding's missing ends, curve and resting value from the sweep and default.
subsample.transform takes every default from here when it builds a step, and
holds a number inside its limit.  tests/test_processors.py fails when the
declaration disagrees with itself, or when the code stops following it.  The
prose (each title and description) is published on subsystem.co as it stands,
so it uses British spelling and never an em dash.

Nothing here imports the rest of the package, so anything may import it.
"""

import dataclasses
import math
import typing


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

FORMS: typing.Final[tuple[str, ...]] = (
	"number",       # any number, such as -20 or 0.5
	"integer",      # a whole number, such as 16
	"boolean",      # true or false
	"choice",       # one of the parameter's declared words
	"note_name",    # a note name, such as C4 or F#2
	"path",         # a file path
)
"""The forms a parameter's value can take.  A parameter lists one or more."""

UNITS: typing.Final[tuple[str, ...]] = ("Hz", "dB", "ms", "BPM", "semitones")
"""The unit words a parameter may declare: the unit a map's number is written in.

A closed set that Superconductor's adapter may act on (#2435), so a word is
never added or respelt without a test noticing.  A parameter with no unit, such
as a ratio or an amount from 0 to 1, declares none."""

AUTOMATIC_SOURCES: typing.Final[tuple[str, ...]] = ("sample", "note", "tempo", "resolution", "mode")
"""Where a parameter's value comes from when a map leaves it out and it has no fixed default.

sample:      measured from the sample being processed
note:        the MIDI note that was played
tempo:       the session tempo, which follows MIDI clock when configured to
resolution:  the configured quantise resolution, transform.quantize_resolution
mode:        the processor's own mode setting"""

SWEEP_CURVES: typing.Final[tuple[str, ...]] = ("linear", "log")
"""How a CC binding spreads its travel: in equal steps, or in equal ratios."""


# ---------------------------------------------------------------------------
# Words for messages
# ---------------------------------------------------------------------------

def number_text (value: float) -> str:

	"""A number as a person writes it: 16000, 0.5, -1."""

	return format(value, "g")


def either_text (items: typing.Sequence[str]) -> str:

	"""Alternatives in words: "a", "a or b", "a, b, or c"."""

	if len(items) <= 2:
		return " or ".join(items)

	return ", ".join(items[:-1]) + ", or " + items[-1]


def condition_text (condition: typing.Mapping[str, tuple[str, ...]]) -> str:

	"""A condition in words: "mode is fm or ssb", "mode is ssb and demod is matched"."""

	return " and ".join(f"{name} is {either_text(words)}" for name, words in condition.items())


# ---------------------------------------------------------------------------
# Declaration types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Choice:

	"""One word a parameter accepts, such as ``fold`` for distort's ``mode``."""

	value:        str
	title:        str = ""
	description:  str = ""


@dataclasses.dataclass(frozen=True)
class Limit:

	"""The values a number may take.  Every bound given must hold; an absent bound does not apply."""

	minimum:            typing.Optional[float] = None
	maximum:            typing.Optional[float] = None
	exclusive_minimum:  typing.Optional[float] = None
	exclusive_maximum:  typing.Optional[float] = None

	def admits (self, value: float) -> bool:

		"""True when value lies inside every bound this limit gives."""

		if self.minimum is not None and value < self.minimum:
			return False

		if self.maximum is not None and value > self.maximum:
			return False

		if self.exclusive_minimum is not None and value <= self.exclusive_minimum:
			return False

		if self.exclusive_maximum is not None and value >= self.exclusive_maximum:
			return False

		return True

	def above_zero (self) -> bool:

		"""True when every value this limit admits is greater than zero."""

		if self.minimum is not None and self.minimum > 0:
			return True

		return self.exclusive_minimum is not None and self.exclusive_minimum >= 0

	def is_open (self) -> bool:

		"""True when this limit gives no bound at all, so any number is admitted."""

		return all(bound is None for bound in (self.minimum, self.maximum, self.exclusive_minimum, self.exclusive_maximum))

	def clamp (self, value: float) -> float:

		"""Hold value inside this limit: at an inclusive bound, or just inside an exclusive one."""

		if self.minimum is not None and value < self.minimum:
			value = self.minimum

		if self.maximum is not None and value > self.maximum:
			value = self.maximum

		if self.exclusive_minimum is not None and value <= self.exclusive_minimum:
			value = math.nextafter(self.exclusive_minimum, math.inf)

		if self.exclusive_maximum is not None and value >= self.exclusive_maximum:
			value = math.nextafter(self.exclusive_maximum, -math.inf)

		return value

	def describe (self, unit: typing.Optional[str] = None) -> str:

		"""The values this limit admits, in words: "from 0 to 24 dB", "1 or more", "above 1 Hz"."""

		suffix = f" {unit}" if unit else ""

		if self.minimum is not None and self.maximum is not None:
			return f"from {number_text(self.minimum)} to {number_text(self.maximum)}{suffix}"

		parts: list[str] = []

		if self.minimum is not None:
			parts.append(f"{number_text(self.minimum)}{suffix} or more")

		if self.exclusive_minimum is not None:
			parts.append(f"above {number_text(self.exclusive_minimum)}{suffix}")

		if self.maximum is not None:
			parts.append(f"{number_text(self.maximum)}{suffix} or less")

		if self.exclusive_maximum is not None:
			parts.append(f"below {number_text(self.exclusive_maximum)}{suffix}")

		return " and ".join(parts) if parts else "any number"


Condition = typing.Mapping[str, tuple[str, ...]]
"""Sibling parameters of the same processor, and the words each must have.

Every key must hold.  A sibling a map leaves out counts at its default:
``distort: {bit_depth: 4}`` has distort's default mode, hard_clip, so the
condition ``{"mode": ("bit_crush",)}`` does not hold for it."""


@dataclasses.dataclass(frozen=True)
class LimitWhen:

	"""A limit that holds only while a condition on sibling parameters holds."""

	when:   Condition
	limit:  Limit


@dataclasses.dataclass(frozen=True)
class Parameter:

	"""One parameter of a processor, in the name and the unit a MIDI map uses.

	A parameter has at most one of a fixed ``default``, an ``automatic`` source
	and ``required``.  With none of them, leaving it out has its own meaning,
	which the description says (a quantise step with no ``segment`` plays every
	hit).
	"""

	name:          str
	forms:         tuple[str, ...]
	unit:          typing.Optional[str] = None
	choices:       tuple[Choice, ...] = ()
	limit:         Limit = Limit()
	limits_when:   tuple[LimitWhen, ...] = ()
	sweep:         typing.Optional[tuple[float, float]] = None
	sweep_curve:   str = "linear"
	default:       typing.Union[bool, int, float, str, None] = None
	automatic:     typing.Optional[str] = None
	required:      bool = False
	applies_when:  tuple[Condition, ...] = ()
	legacy_names:  tuple[str, ...] = ()
	title:         str = ""
	description:   str = ""

	@property
	def bindable (self) -> bool:

		"""True when a CC binding can drive this parameter: it takes a number and nothing else."""

		return self.forms in (("number",), ("integer",))

	@property
	def choice_values (self) -> tuple[str, ...]:

		"""The words this parameter accepts, in their declared order."""

		return tuple(choice.value for choice in self.choices)

	def accepts_text (self) -> str:

		"""What this parameter accepts, in words: "am, lw, fm, or ssb", "a whole number from 1 to 16"."""

		alternatives: list[str] = []

		for form in self.forms:
			if form == "boolean":
				alternatives.extend(("true", "false"))

			elif form == "choice":
				alternatives.extend(self.choice_values)

			elif form in ("number", "integer"):
				noun = "a number" if form == "number" else "a whole number"

				if not self.limit.is_open():
					alternatives.append(f"{noun} {self.limit.describe(self.unit)}")
				elif self.unit:
					alternatives.append(f"{noun} in {self.unit}")
				else:
					alternatives.append(noun)

			elif form == "note_name":
				alternatives.append("a note name such as C4")

			elif form == "path":
				alternatives.append("a file path")

		return either_text(alternatives)

	def applies_text (self) -> str:

		"""When this parameter has an effect, in words: "demod is ssb, or mode is ssb and demod is matched"."""

		return ", or ".join(condition_text(condition) for condition in self.applies_when)


@dataclasses.dataclass(frozen=True)
class LegacyName:

	"""An older processor name a map may still use, and the parameters it implies.

	``hpss_harmonic`` is ``hpss`` with ``keep: harmonic``, so it implies that
	parameter; ``beat_quantize`` is a plain rename and implies none.
	"""

	name:     str
	implies:  typing.Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Processor:

	"""One processor a MIDI map's ``process:`` list can name."""

	name:          str
	parameters:    tuple[Parameter, ...] = ()
	shorthand:     typing.Optional[str] = None
	legacy_names:  tuple[LegacyName, ...] = ()
	title:         str = ""
	description:   str = ""

	@property
	def parameter_names (self) -> tuple[str, ...]:

		"""The names of this processor's parameters, in their declared order."""

		return tuple(parameter.name for parameter in self.parameters)

	def parameter (self, name: str) -> Parameter:

		"""Return the parameter called name, raising KeyError when this processor has none."""

		for parameter in self.parameters:
			if parameter.name == name:
				return parameter

		raise KeyError(f"processor {self.name!r} has no parameter {name!r}")

	def condition_holds (self, condition: Condition, given: typing.Mapping[str, typing.Any]) -> bool:

		"""True when every sibling a condition names has one of its words in a step's given values.

		A sibling the step leaves out counts at its declared default.
		"""

		for name, words in condition.items():
			value = given.get(name, self.parameter(name).default)

			if value not in words:
				return False

		return True

	def applies (self, name: str, given: typing.Mapping[str, typing.Any]) -> bool:

		"""True when the named parameter has any effect, given a step's values for its siblings."""

		conditions = self.parameter(name).applies_when

		return not conditions or any(self.condition_holds(condition, given) for condition in conditions)

	def limits_for (
		self, name: str, given: typing.Mapping[str, typing.Any],
	) -> list[tuple[Limit, typing.Optional[Condition]]]:

		"""Every limit that holds for the named parameter, given a step's values for its siblings.

		Each comes with the condition that brought it into force, or None for the
		parameter's own limit.
		"""

		parameter = self.parameter(name)
		limits: list[tuple[Limit, typing.Optional[Condition]]] = [(parameter.limit, None)]

		for conditional in parameter.limits_when:
			if self.condition_holds(conditional.when, given):
				limits.append((conditional.limit, conditional.when))

		return limits


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

_NUMBER: typing.Final[tuple[str, ...]] = ("number",)
_INTEGER: typing.Final[tuple[str, ...]] = ("integer",)
_CHOICE: typing.Final[tuple[str, ...]] = ("choice",)

_FRACTION: typing.Final[Limit] = Limit(minimum=0.0, maximum=1.0)
_NOT_NEGATIVE: typing.Final[Limit] = Limit(minimum=0.0)


def _mix () -> Parameter:

	"""The dry and processed blend several processors share, from 0 (dry) to 1 (processed)."""

	return Parameter(
		name="mix", forms=_NUMBER,
		limit=_FRACTION, sweep=(0.0, 1.0),
		default=1.0,
	)


def _quantise_parameters () -> tuple[Parameter, ...]:

	"""The parameters stretch_quantize and pad_quantize share."""

	return (
		Parameter(
			name="tempo", forms=_NUMBER, unit="BPM",
			limit=Limit(exclusive_minimum=0.0), sweep=(60.0, 180.0),
			automatic="tempo",
			legacy_names=("bpm",),
		),
		Parameter(
			name="grid", forms=_INTEGER,
			limit=Limit(minimum=1), sweep=(1, 32),
			automatic="resolution",
		),
		Parameter(
			name="strength", forms=_NUMBER,
			limit=_FRACTION, sweep=(0.0, 1.0),
			default=1.0,
			legacy_names=("amount",),
		),
		Parameter(
			name="segment", forms=("choice", "integer"),
			choices=(Choice("round_robin"), Choice("random")),
			limit=Limit(minimum=1),
		),
	)


def _filter_frequency (default: float) -> Parameter:

	"""A filter's cutoff or centre frequency."""

	return Parameter(
		name="freq", forms=_NUMBER, unit="Hz",
		limit=Limit(exclusive_minimum=1.0), sweep=(20.0, 20000.0), sweep_curve="log",
		default=default,
	)


def _resonance () -> Parameter:

	"""The peak a filter adds at its cutoff."""

	return Parameter(
		name="resonance", forms=_NUMBER, unit="dB",
		limit=Limit(minimum=0.0, maximum=24.0), sweep=(0.0, 24.0),
		default=0.0,
	)


# ---------------------------------------------------------------------------
# The processors
# ---------------------------------------------------------------------------

_DECLARED: typing.Final[tuple[Processor, ...]] = (

	Processor(
		name="repitch",
		parameters=(
			Parameter(
				name="note", forms=("integer", "note_name"),
				limit=Limit(minimum=0, maximum=127),
				automatic="note",
			),
		),
	),

	Processor(
		name="stretch_quantize",
		parameters=_quantise_parameters(),
		legacy_names=(LegacyName("beat_quantize"),),
	),

	Processor(
		name="pad_quantize",
		parameters=_quantise_parameters(),
	),

	Processor(
		name="filter_low",
		parameters=(_filter_frequency(16000.0), _resonance()),
	),

	Processor(
		name="filter_high",
		parameters=(_filter_frequency(80.0), _resonance()),
	),

	Processor(
		name="filter_band",
		parameters=(
			_filter_frequency(1000.0),
			Parameter(
				name="q", forms=_NUMBER,
				limit=Limit(minimum=0.1, maximum=20.0), sweep=(0.5, 10.0), sweep_curve="log",
				default=0.7,
			),
			_resonance(),
		),
	),

	Processor(name="reverse"),

	Processor(
		name="saturate",
		parameters=(
			Parameter(
				name="drive", forms=_NUMBER, unit="dB",
				limit=_NOT_NEGATIVE, sweep=(0.0, 24.0),
				default=6.0,
				legacy_names=("amount",),
			),
		),
	),

	Processor(
		name="compress",
		parameters=(
			Parameter(
				name="threshold", forms=_NUMBER, unit="dB",
				sweep=(-60.0, 0.0),
				automatic="sample",
			),
			Parameter(
				name="ratio", forms=_NUMBER,
				limit=Limit(minimum=1.0), sweep=(1.0, 20.0), sweep_curve="log",
				default=4.0,
			),
			Parameter(
				name="attack", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 100.0),
				automatic="sample",
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 1000.0),
				automatic="sample",
			),
			Parameter(
				name="knee", forms=_NUMBER, unit="dB",
				limit=_NOT_NEGATIVE, sweep=(0.0, 24.0),
				default=6.0,
			),
			Parameter(
				name="makeup", forms=_NUMBER, unit="dB",
				sweep=(0.0, 24.0),
				default=0.0,
			),
			Parameter(
				name="lookahead", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 20.0),
				default=0.0,
			),
		),
	),

	Processor(
		name="limit",
		parameters=(
			Parameter(
				name="threshold", forms=_NUMBER, unit="dB",
				sweep=(-24.0, 0.0),
				default=-1.0,
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				default=50.0,
			),
			Parameter(
				name="lookahead", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 20.0),
				default=5.0,
			),
		),
	),

	Processor(
		name="hpss",
		parameters=(
			Parameter(
				name="keep", forms=_CHOICE,
				choices=(Choice("harmonic"), Choice("percussive")),
				required=True,
			),
		),
		legacy_names=(
			LegacyName("hpss_harmonic", implies={"keep": "harmonic"}),
			LegacyName("hpss_percussive", implies={"keep": "percussive"}),
		),
	),

	Processor(
		name="gate",
		parameters=(
			Parameter(
				name="threshold", forms=_NUMBER, unit="dB",
				sweep=(-80.0, 0.0),
				automatic="sample",
			),
			Parameter(
				name="attack", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 50.0),
				automatic="sample",
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				automatic="sample",
			),
			Parameter(
				name="hold", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				automatic="sample",
			),
			Parameter(
				name="lookahead", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 20.0),
				automatic="sample",
			),
		),
	),

	Processor(
		name="distort",
		parameters=(
			Parameter(
				name="mode", forms=_CHOICE,
				choices=(Choice("hard_clip"), Choice("fold"), Choice("bit_crush"), Choice("downsample")),
				default="hard_clip",
			),
			Parameter(
				name="drive", forms=_NUMBER, unit="dB",
				sweep=(0.0, 36.0),
				automatic="sample",
			),
			Parameter(
				name="tone", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				automatic="sample",
			),
			_mix(),
			Parameter(
				name="bit_depth", forms=_INTEGER,
				limit=Limit(minimum=1, maximum=16), sweep=(1, 16),
				default=8,
				applies_when=({"mode": ("bit_crush",)},),
			),
			Parameter(
				name="downsample_factor", forms=_INTEGER,
				limit=Limit(minimum=2, maximum=64), sweep=(2, 64), sweep_curve="log",
				default=4,
				applies_when=({"mode": ("downsample",)},),
			),
		),
	),

	Processor(
		name="bit_depth",
		shorthand="bits",
		parameters=(
			Parameter(
				name="bits", forms=_INTEGER,
				limit=Limit(minimum=1, maximum=16), sweep=(1, 16),
				default=12,
			),
			Parameter(
				name="dither", forms=("boolean", "choice"),
				choices=(Choice("none"), Choice("triangular"), Choice("rectangular")),
				default="none",
			),
		),
	),

	Processor(
		name="radio",
		parameters=(
			Parameter(
				name="mode", forms=_CHOICE,
				choices=(Choice("am"), Choice("lw"), Choice("fm"), Choice("ssb")),
				default="am",
			),
			Parameter(
				name="demod", forms=_CHOICE,
				choices=(Choice("matched"), Choice("am"), Choice("fm"), Choice("ssb")),
				default="matched",
			),
			Parameter(
				name="tune", forms=_NUMBER, unit="Hz",
				sweep=(-1000.0, 1000.0),
				default=0.0,
				# Tuning offsets the SSB product detector, so it only reaches
				# the sound when the receiver demodulates as SSB: named outright,
				# or matched to an SSB transmission.
				applies_when=({"demod": ("ssb",)}, {"mode": ("ssb",), "demod": ("matched",)}),
			),
			Parameter(
				name="signal", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=0.0,
			),
			Parameter(
				name="static", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=0.0,
			),
			Parameter(
				name="fade", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=0.0,
			),
			Parameter(
				name="bandwidth", forms=_NUMBER, unit="Hz",
				limit=Limit(exclusive_minimum=0.0),
				# The FM and SSB channel filters pass a band from a fixed 300 Hz,
				# so their top edge has to clear it.
				limits_when=(LimitWhen(when={"mode": ("fm", "ssb")}, limit=Limit(exclusive_minimum=300.0)),),
				sweep=(500.0, 8000.0), sweep_curve="log",
				automatic="mode",
			),
			Parameter(
				name="stereo", forms=_CHOICE,
				choices=(Choice("mono"), Choice("stereo")),
				default="mono",
			),
			_mix(),
		),
	),

	Processor(
		name="freqshift",
		shorthand="shift_hz",
		parameters=(
			Parameter(
				name="shift_hz", forms=_NUMBER, unit="Hz",
				sweep=(-2000.0, 2000.0),
				default=0.0,
			),
			_mix(),
		),
	),

	Processor(
		name="wobble",
		shorthand="depth",
		parameters=(
			Parameter(
				name="depth", forms=_NUMBER, unit="Hz",
				limit=_NOT_NEGATIVE, sweep=(0.0, 50.0),
				default=5.0,
			),
			Parameter(
				name="rate", forms=_NUMBER, unit="Hz",
				limit=Limit(exclusive_minimum=0.0), sweep=(0.05, 10.0), sweep_curve="log",
				default=0.3,
			),
			Parameter(
				name="base", forms=_NUMBER, unit="Hz",
				sweep=(-100.0, 100.0),
				default=0.0,
			),
			_mix(),
		),
	),

	Processor(
		name="reshape",
		parameters=(
			Parameter(
				name="attack", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 100.0),
				automatic="sample",
			),
			Parameter(
				name="hold", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				default=0.0,
			),
			Parameter(
				name="decay", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 1000.0),
				automatic="sample",
			),
			Parameter(
				name="sustain", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=1.0,
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 2000.0),
				automatic="sample",
			),
		),
	),

	Processor(
		name="transient",
		parameters=(
			Parameter(
				name="gain", forms=_NUMBER, unit="dB",
				sweep=(-12.0, 12.0),
				automatic="sample",
				legacy_names=("amount",),
			),
		),
	),

	Processor(
		name="vocoder",
		parameters=(
			Parameter(
				name="carrier", forms=("choice", "path"),
				choices=(Choice("reference"),),
				required=True,
			),
			Parameter(
				name="bands", forms=_INTEGER,
				limit=Limit(minimum=1), sweep=(4, 48),
				default=24,
			),
			Parameter(
				name="depth", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=1.0,
			),
			Parameter(
				name="formant_shift", forms=_INTEGER, unit="semitones",
				sweep=(-12, 12),
				default=0,
			),
		),
	),
)


PROCESSORS: typing.Final[dict[str, Processor]] = {processor.name: processor for processor in _DECLARED}
"""Every processor a map can name, by its current name, in the order a reference lists them."""
