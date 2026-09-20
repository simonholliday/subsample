"""Every processor a MIDI map's ``process:`` list can name, and every parameter each one takes.

This is the one declaration of the processor vocabulary, in the names and units
a map uses.  Each parameter declares what it accepts (its forms and allowed
words), the values it refuses (its limit), the travel a CC binding sweeps when
the binding gives no ends of its own, its default or where an automatic value
comes from, its unit, and when it has any effect at all.

Everything reads it.  subsample.query takes its processor and parameter names
from here, refuses at load a value a declaration does not allow, and builds a
CC binding's missing ends, taper and resting value from the sweep and default.
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

UNITS: typing.Final[tuple[str, ...]] = ("Hz", "dB", "ms", "s", "BPM", "semitones")
"""The unit words a map's numbers are written in, a processor parameter's and a
term of the grammar's alike.

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

TAPERS: typing.Final[tuple[str, ...]] = ("linear", "log")
"""How a sweep's travel maps onto its values, as a potentiometer's taper does: in
equal steps, or in equal ratios, so each step of a frequency knob is the same
musical interval.  Not a curve: in a MIDI map, `curve` is release's fade shape
and pick's velocity mapping."""


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

	``examples`` are values a map would really write here, one for each form
	worth showing, and they are published as the reference's examples.  A
	parameter that takes only words has none, because the words are published
	with their own prose.  Nothing shows a knob: a CC binding is published once,
	where the binding itself is declared.
	"""

	name:          str
	forms:         tuple[str, ...]
	unit:          typing.Optional[str] = None
	choices:       tuple[Choice, ...] = ()
	limit:         Limit = Limit()
	limits_when:   tuple[LimitWhen, ...] = ()
	sweep:         typing.Optional[tuple[float, float]] = None
	taper:         str = "linear"
	default:       typing.Union[bool, int, float, str, None] = None
	automatic:     typing.Optional[str] = None
	required:      bool = False
	applies_when:  tuple[Condition, ...] = ()
	legacy_names:  tuple[str, ...] = ()
	examples:      tuple[typing.Union[int, float, str], ...] = ()
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

	"""One processor a MIDI map's ``process:`` list can name.

	``examples`` are whole steps a map would write against the name: the
	shorthand number where the processor takes one, and a setting of its
	parameters that belongs together.  A processor that takes no parameters has
	none, because a map only ever names it.
	"""

	name:          str
	parameters:    tuple[Parameter, ...] = ()
	shorthand:     typing.Optional[str] = None
	legacy_names:  tuple[LegacyName, ...] = ()
	examples:      tuple[typing.Union[int, float, dict[str, typing.Any]], ...] = ()
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
		examples=(0.5,),
		title="Mix",
		description="The balance between the processed sound and the original. At 1 only the processed sound plays, and at 0 the processor has no effect.",
	)


def _quantise_parameters () -> tuple[Parameter, ...]:

	"""The parameters stretch_quantize and pad_quantize share."""

	return (
		Parameter(
			name="tempo", forms=_NUMBER, unit="BPM",
			limit=Limit(exclusive_minimum=0.0), sweep=(60.0, 180.0),
			automatic="tempo",
			legacy_names=("bpm",),
			examples=(120.0,),
			title="Tempo",
			description="The tempo the beat grid is laid out at. Left out, the session tempo, which follows MIDI clock when Subsample is set to.",
		),
		Parameter(
			name="grid", forms=_INTEGER,
			limit=Limit(minimum=1), sweep=(1, 32),
			automatic="resolution",
			examples=(16,),
			title="Grid",
			description="How many equal parts a whole note is divided into: 16 puts every hit on a sixteenth note. Left out, the quantise resolution Subsample is configured with.",
		),
		Parameter(
			name="strength", forms=_NUMBER,
			limit=_FRACTION, sweep=(0.0, 1.0),
			default=1.0,
			legacy_names=("amount",),
			examples=(0.7,),
			title="Strength",
			description="How far each hit moves toward the grid. At 1 every hit lands on the grid, and lower values move each one partway, for a looser feel.",
		),
		Parameter(
			name="segment", forms=("choice", "integer"),
			choices=(
				Choice("round_robin", "Round robin", "Each note plays the next hit in order, and starts again after the last."),
				Choice("random", "Random", "Each note plays a hit chosen at random."),
			),
			limit=Limit(minimum=1),
			examples=(3,),
			title="Segment",
			description="Plays one hit of the quantised sound per note instead of the whole sound: the next in turn, one at random, or always the same one, counted from 1. Left out, every note plays the whole sound.",
		),
	)


def _filter_frequency (default: float, example: float, title: str, description: str) -> Parameter:

	"""A filter's cutoff or centre frequency."""

	return Parameter(
		name="freq", forms=_NUMBER, unit="Hz",
		limit=Limit(exclusive_minimum=1.0), sweep=(20.0, 20000.0), taper="log",
		default=default,
		examples=(example,),
		title=title,
		description=description,
	)


def _resonance (description: str) -> Parameter:

	"""The peak a filter adds at its cutoff, or at its centre."""

	return Parameter(
		name="resonance", forms=_NUMBER, unit="dB",
		limit=Limit(minimum=0.0, maximum=24.0), sweep=(0.0, 24.0),
		default=0.0,
		examples=(6.0,),
		title="Resonance",
		description=description,
	)


def _lookahead (
	default:     typing.Optional[float],
	automatic:   typing.Optional[str],
	sweep:       tuple[float, float],
	example:     float,
	description: str,
) -> Parameter:

	"""How far ahead a dynamics processor looks, which delays the rendered sound by as much."""

	return Parameter(
		name="lookahead", forms=_NUMBER, unit="ms",
		limit=_NOT_NEGATIVE, sweep=sweep,
		default=default,
		automatic=automatic,
		examples=(example,),
		title="Look-ahead",
		description=description,
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
				examples=(48, "C3"),
				title="Note",
				description="The note to shift the sound to. Left out, the note played, so the sound follows the keyboard.",
			),
		),
		examples=({"note": "C3"},),
		title="Repitch",
		description="Shifts the sound to a note, from the pitch Subsample detected in it, without changing its length.",
	),

	Processor(
		name="stretch_quantize",
		parameters=_quantise_parameters(),
		legacy_names=(LegacyName("beat_quantize"),),
		examples=({"grid": 16, "strength": 0.7},),
		title="Stretch quantise",
		description="Moves each hit onto a beat grid by time-stretching the audio between hits, without changing its pitch.",
	),

	Processor(
		name="pad_quantize",
		parameters=_quantise_parameters(),
		examples=({"tempo": 120.0, "grid": 8},),
		title="Pad quantise",
		description="Moves each hit onto a beat grid by inserting silence between hits, so the audio keeps its own speed and timbre.",
	),

	Processor(
		name="filter_low",
		parameters=(
			_filter_frequency(
				16000.0, 800.0, "Cutoff",
				"The frequency above which the filter rolls off.",
			),
			_resonance("A peak at the cutoff. At 0 the filter is flat, and higher values make the cutoff ring more."),
		),
		examples=({"freq": 800.0, "resonance": 6.0},),
		title="Low-pass filter",
		description="Rolls off the frequencies above the cutoff, so the sound darkens as the cutoff comes down.",
	),

	Processor(
		name="filter_high",
		parameters=(
			_filter_frequency(
				80.0, 120.0, "Cutoff",
				"The frequency below which the filter rolls off.",
			),
			_resonance("A peak at the cutoff. At 0 the filter is flat, and higher values make the cutoff ring more."),
		),
		examples=({"freq": 120.0},),
		title="High-pass filter",
		description="Rolls off the frequencies below the cutoff, so the sound thins as the cutoff goes up.",
	),

	Processor(
		name="filter_band",
		parameters=(
			_filter_frequency(
				1000.0, 1200.0, "Centre",
				"The frequency at the centre of what the filter keeps.",
			),
			Parameter(
				name="q", forms=_NUMBER,
				limit=Limit(minimum=0.1, maximum=20.0), sweep=(0.5, 10.0), taper="log",
				default=0.7,
				examples=(4.0,),
				title="Q",
				description="How narrowly the filter keeps the frequencies around its centre. A lower Q keeps a wider range, and a higher Q a narrower one.",
			),
			_resonance("A peak at the centre. At 0 the filter is flat, and higher values make the centre ring more."),
		),
		examples=({"freq": 1200.0, "q": 4.0},),
		title="Band-pass filter",
		description="Keeps the frequencies around its centre and rolls off those above and below.",
	),

	Processor(
		name="reverse",
		title="Reverse",
		description="Plays the sound backwards.",
	),

	Processor(
		name="saturate",
		parameters=(
			Parameter(
				name="drive", forms=_NUMBER, unit="dB",
				limit=_NOT_NEGATIVE, sweep=(0.0, 24.0),
				default=6.0,
				legacy_names=("amount",),
				examples=(4.0,),
				title="Drive",
				description="How hard the sound is pushed into the curve. At 0 the processor has no effect, and more drive gives more distortion.",
			),
		),
		examples=({"drive": 4.0},),
		title="Saturation",
		description="Rounds off the peaks with a soft curve for warmth, then restores the peak level, which also lifts the quieter parts.",
	),

	Processor(
		name="compress",
		parameters=(
			Parameter(
				name="threshold", forms=_NUMBER, unit="dB",
				sweep=(-60.0, 0.0),
				automatic="sample",
				examples=(-20.0,),
				title="Threshold",
				description="The level above which the compressor turns the sound down, below full scale. Every sample is normalised before it is processed, so a threshold is the same depth below the peak for every sample. Left out, it sits a little below the sample's peak, so the compressor always acts.",
			),
			Parameter(
				name="ratio", forms=_NUMBER,
				limit=Limit(minimum=1.0), sweep=(1.0, 20.0), taper="log",
				default=4.0,
				examples=(8.0,),
				title="Ratio",
				description="How strongly the level is turned down above the threshold. At 1 there is no compression, and a higher ratio squashes harder.",
			),
			Parameter(
				name="attack", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 100.0),
				automatic="sample",
				examples=(10.0,),
				title="Attack",
				description="How quickly the compressor acts once the sound passes the threshold. A slower attack lets the front of a hit through, for more punch. Left out, it follows the sample: slower for a percussive sound, faster for a gradual one.",
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 1000.0),
				automatic="sample",
				examples=(120.0,),
				title="Release",
				description="How quickly the compressor lets go once the sound falls back below the threshold. Left out, it follows the sample: shorter for a quick decay, longer for a sustained sound.",
			),
			Parameter(
				name="knee", forms=_NUMBER, unit="dB",
				limit=_NOT_NEGATIVE, sweep=(0.0, 24.0),
				default=6.0,
				examples=(12.0,),
				title="Knee",
				description="How gradually compression begins around the threshold. At 0 it begins abruptly, and a wider knee eases it in.",
			),
			Parameter(
				name="makeup", forms=_NUMBER, unit="dB",
				sweep=(0.0, 24.0),
				default=0.0,
				examples=(3.0,),
				title="Makeup gain",
				description="Gain added after compression, to make up the level it took away.",
			),
			_lookahead(
				0.0, None, (0.0, 20.0), 5.0,
				"How far ahead the compressor looks, so it acts before a peak arrives. The rendered sound starts this much later and loses as much from its end. At 0 there is no look-ahead.",
			),
		),
		examples=({"threshold": -20.0, "ratio": 8.0, "attack": 10.0},),
		title="Compressor",
		description="Turns the sound down whenever it rises above the threshold, evening out its level. Left without settings, the threshold, attack and release adapt to each sample.",
	),

	Processor(
		name="limit",
		parameters=(
			Parameter(
				name="threshold", forms=_NUMBER, unit="dB",
				sweep=(-24.0, 0.0),
				default=-1.0,
				examples=(-0.5,),
				title="Ceiling",
				description="The highest level the sound may reach, below full scale.",
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				default=50.0,
				examples=(120.0,),
				title="Release",
				description="How quickly the limiter lets go after a peak.",
			),
			_lookahead(
				5.0, None, (0.0, 20.0), 10.0,
				"How far ahead the limiter looks, so it catches a peak before it passes the ceiling. The rendered sound starts this much later and loses as much from its end. At 0 there is no look-ahead.",
			),
		),
		examples=({"threshold": -0.5},),
		title="Limiter",
		description="Holds the sound below a ceiling, catching every peak that would pass it.",
	),

	Processor(
		name="hpss",
		parameters=(
			Parameter(
				name="keep", forms=_CHOICE,
				choices=(
					Choice("harmonic", "Harmonic", "The sustained, tonal part, without the hits."),
					Choice("percussive", "Percussive", "The hits and transients, without the sustained tone."),
				),
				required=True,
				title="Keep",
				description="Which part of the sound to keep.",
			),
		),
		legacy_names=(
			LegacyName("hpss_harmonic", implies={"keep": "harmonic"}),
			LegacyName("hpss_percussive", implies={"keep": "percussive"}),
		),
		examples=({"keep": "percussive"},),
		title="Harmonic and percussive split",
		description="Separates the sound into its sustained, tonal part and its hits, and keeps one of them.",
	),

	Processor(
		name="gate",
		parameters=(
			Parameter(
				name="threshold", forms=_NUMBER, unit="dB",
				sweep=(-80.0, 0.0),
				automatic="sample",
				examples=(-40.0,),
				title="Threshold",
				description="The level below which the gate closes, below full scale. Left out, it sits a little above the sample's noise floor.",
			),
			Parameter(
				name="attack", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 50.0),
				automatic="sample",
				examples=(1.0,),
				title="Attack",
				description="How quickly the gate opens once the sound rises past the threshold. Left out, it follows the sample: fast for a percussive sound, slower for a sustained one, which avoids a click.",
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				automatic="sample",
				examples=(80.0,),
				title="Release",
				description="How quickly the gate closes once the sound falls below the threshold. Left out, it follows the sample: short for a percussive sound, long for a sustained one.",
			),
			Parameter(
				name="hold", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				automatic="sample",
				examples=(20.0,),
				title="Hold",
				description="How long the gate stays open after the sound falls below the threshold, so a decaying tail does not chatter. Left out, it follows the sample's decay.",
			),
			_lookahead(
				None, "sample", (0.0, 20.0), 2.0,
				"How far ahead the gate looks, so it opens before a hit arrives. The rendered sound starts this much later and loses as much from its end. Left out, it follows the sample: a little for a percussive sound, none for a sustained one.",
			),
		),
		examples=({"threshold": -40.0, "hold": 20.0},),
		title="Noise gate",
		description="Silences the sound whenever it falls below the threshold, cutting the noise between and after hits. Left without settings, every value adapts to each sample.",
	),

	Processor(
		name="distort",
		parameters=(
			Parameter(
				name="mode", forms=_CHOICE,
				choices=(
					Choice("hard_clip", "Hard clip", "Flattens every peak at full scale, as an overdriven digital input does."),
					Choice("fold", "Fold", "Folds each peak back on itself past full scale, for brighter harmonics than clipping gives."),
					Choice("bit_crush", "Bit crush", "Snaps the driven sound to fewer amplitude levels, for lo-fi grit."),
					Choice("downsample", "Downsample", "Keeps one frame in every few and repeats it, lowering the effective sample rate and adding aliasing."),
				),
				default="hard_clip",
				title="Mode",
				description="The shape the waveshaper gives the sound.",
			),
			Parameter(
				name="drive", forms=_NUMBER, unit="dB",
				sweep=(0.0, 36.0),
				automatic="sample",
				examples=(12.0,),
				title="Drive",
				description="Gain before the waveshaper: more drive, more distortion. Left out, it follows the sample's crest factor, so a peaky sound is driven less.",
			),
			Parameter(
				name="tone", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				automatic="sample",
				examples=(0.4,),
				title="Tone",
				description="A low-pass filter after the waveshaper, which tames the harmonics it adds. At 1 the sound passes unfiltered, and lower values darken it. Left out, it follows the sample's brightness.",
			),
			_mix(),
			Parameter(
				name="bit_depth", forms=_INTEGER,
				limit=Limit(minimum=1, maximum=16), sweep=(1, 16),
				default=8,
				applies_when=({"mode": ("bit_crush",)},),
				examples=(4,),
				title="Bit depth",
				description="How many bits `bit_crush` keeps. Fewer bits sound coarser and grittier.",
			),
			Parameter(
				name="downsample_factor", forms=_INTEGER,
				limit=Limit(minimum=2, maximum=64), sweep=(2, 64), taper="log",
				default=4,
				applies_when=({"mode": ("downsample",)},),
				examples=(8,),
				title="Downsample factor",
				description="How many frames each kept frame fills in `downsample` mode. A higher factor gives a lower effective sample rate.",
			),
		),
		examples=({"mode": "fold", "drive": 12.0}, {"mode": "bit_crush", "bit_depth": 4, "mix": 0.5}),
		title="Distortion",
		description="Pushes the sound into a waveshaper, then restores its peak level. Left without settings, the drive and the tone adapt to each sample: a peaky sound is driven less, and a bright one is darkened more.",
	),

	Processor(
		name="bit_depth",
		shorthand="bits",
		parameters=(
			Parameter(
				name="bits", forms=_INTEGER,
				limit=Limit(minimum=1, maximum=16), sweep=(1, 16),
				default=12,
				examples=(8,),
				title="Bits",
				description="How many bits the sound is stored in. Fewer bits sound coarser and grittier.",
			),
			Parameter(
				name="dither", forms=("boolean", "choice"),
				choices=(
					Choice("none", "None", "No dither: quiet passages and tails break up into grit, as they do on vintage converters."),
					Choice("triangular", "Triangular", "The standard dither: the grit becomes a steady hiss that does not follow the sound."),
					Choice("rectangular", "Rectangular", "A little less hiss than triangular, with some of the hiss still rising and falling with the sound."),
				),
				default="none",
				title="Dither",
				description="Noise added before the sound is requantised, which trades the grit in quiet passages for a steady hiss, over silence too. `true` is `triangular`, and `false` is `none`.",
			),
		),
		examples=(8, {"bits": 8, "dither": "triangular"}),
		title="Bit depth",
		description="Requantises the sound to fewer bits, for the grit of a vintage sampler's converter. It adds no drive, filtering or change of level.",
	),

	Processor(
		name="radio",
		parameters=(
			Parameter(
				name="mode", forms=_CHOICE,
				choices=(
					Choice("am", "AM", "Amplitude modulation, the cleanest round trip of the four."),
					Choice("lw", "Longwave", "Amplitude modulation through a steep, narrow longwave filter, which takes away more of the top end."),
					Choice("fm", "FM", "Narrowband frequency modulation, as a two-way radio sends it."),
					Choice("ssb", "SSB", "Single sideband: a voice sent without its carrier, which sounds garbled when the receiver is mistuned."),
				),
				default="am",
				title="Transmission",
				description="How the sound is sent.",
			),
			Parameter(
				name="demod", forms=_CHOICE,
				choices=(
					Choice("matched", "Matched", "The receiver that suits the transmission."),
					Choice("am", "AM", "An AM receiver, whatever was sent. AM and FM received as each other give harsh noise."),
					Choice("fm", "FM", "An FM receiver, whatever was sent. AM and FM received as each other give harsh noise."),
					Choice("ssb", "SSB", "An SSB receiver, whatever was sent. FM received this way gives a musical warble."),
				),
				default="matched",
				title="Reception",
				description="How the sound is received, which may be deliberately wrong for how it was sent.",
			),
			Parameter(
				name="tune", forms=_NUMBER, unit="Hz",
				sweep=(-1000.0, 1000.0),
				default=0.0,
				# Tuning offsets the SSB product detector, so it only reaches
				# the sound when the receiver demodulates as SSB: named outright,
				# or matched to an SSB transmission.
				applies_when=({"demod": ("ssb",)}, {"mode": ("ssb",), "demod": ("matched",)}),
				examples=(150.0,),
				title="Tuning",
				description="How far the receiver is tuned off the station. A mistuned SSB voice sounds high or low and garbled.",
			),
			Parameter(
				name="signal", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=0.0,
				examples=(0.3,),
				title="Weak signal",
				description="How weak the station is: more hiss, clicks on FM, and the receiver's gain swelling as the signal fades. At 0 the signal is strong.",
			),
			Parameter(
				name="static", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=0.0,
				examples=(0.5,),
				title="Static",
				description="How much atmospheric crackle breaks in. At 0 there is none.",
			),
			Parameter(
				name="fade", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=0.0,
				examples=(0.6,),
				title="Fading",
				description="How deeply the signal swims in and out, as a distant shortwave station does. At 0 it holds steady.",
			),
			Parameter(
				name="bandwidth", forms=_NUMBER, unit="Hz",
				limit=Limit(exclusive_minimum=0.0),
				# The FM and SSB channel filters pass a band from a fixed 300 Hz,
				# so their top edge has to clear it.
				limits_when=(LimitWhen(when={"mode": ("fm", "ssb")}, limit=Limit(exclusive_minimum=300.0)),),
				sweep=(500.0, 8000.0), taper="log",
				automatic="mode",
				examples=(2500.0,),
				title="Bandwidth",
				description="How wide a range of audio frequencies the radio path passes. A narrower bandwidth sounds more muffled. Left out, each transmission uses its own.",
			),
			Parameter(
				name="stereo", forms=_CHOICE,
				choices=(
					Choice("mono", "Mono", "One receiver, so the sound collapses to mono, as a real radio does."),
					Choice("stereo", "Stereo", "A receiver for each audio channel, with its own hiss and the same crackle."),
				),
				default="mono",
				title="Receivers",
				description="Whether the sound is received once, or once for each audio channel.",
			),
			_mix(),
		),
		examples=({"mode": "ssb", "tune": 150.0, "signal": 0.3},),
		title="Radio",
		description="Sends the sound over a radio link and receives it again: modulated, weakened by noise and fading on the way, and demodulated, rightly or deliberately wrongly.",
	),

	Processor(
		name="freqshift",
		shorthand="shift_hz",
		parameters=(
			Parameter(
				name="shift_hz", forms=_NUMBER, unit="Hz",
				sweep=(-2000.0, 2000.0),
				default=0.0,
				examples=(50.0,),
				title="Shift",
				description="How far every partial moves. A negative shift moves them down.",
			),
			_mix(),
		),
		examples=(1000.0, {"shift_hz": 50.0, "mix": 0.5}),
		title="Frequency shift",
		description="Adds the same number of hertz to every partial, which breaks the ratios between harmonics: a small shift detunes and phases, and a large one sounds clangorous and metallic. It is not a pitch shift.",
	),

	Processor(
		name="wobble",
		shorthand="depth",
		parameters=(
			Parameter(
				name="depth", forms=_NUMBER, unit="Hz",
				limit=_NOT_NEGATIVE, sweep=(0.0, 50.0),
				default=5.0,
				examples=(6.0,),
				title="Depth",
				description="How far the tuning drifts either side of its centre.",
			),
			Parameter(
				name="rate", forms=_NUMBER, unit="Hz",
				limit=Limit(exclusive_minimum=0.0), sweep=(0.05, 10.0), taper="log",
				default=0.3,
				examples=(0.8,),
				title="Rate",
				description="How fast the tuning drifts back and forth.",
			),
			Parameter(
				name="base", forms=_NUMBER, unit="Hz",
				sweep=(-100.0, 100.0),
				default=0.0,
				examples=(10.0,),
				title="Offset",
				description="A constant shift that the drift rides on.",
			),
			_mix(),
		),
		examples=(6.0, {"depth": 6.0, "rate": 0.8}),
		title="Wobble",
		description="A slow, continuous drift of the tuning, like an unsteady oscillator.",
	),

	Processor(
		name="reshape",
		parameters=(
			Parameter(
				name="attack", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 100.0),
				automatic="sample",
				examples=(5.0,),
				title="Attack",
				description="How long the sound takes to reach its peak. Left out, the sample's own attack is kept.",
			),
			Parameter(
				name="hold", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 500.0),
				default=0.0,
				examples=(20.0,),
				title="Hold",
				description="How long the sound stays at its peak before it decays.",
			),
			Parameter(
				name="decay", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 1000.0),
				automatic="sample",
				examples=(150.0,),
				title="Decay",
				description="How long the sound takes to fall from its peak to the sustain level. Left out, the sample's own decay is kept.",
			),
			Parameter(
				name="sustain", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=1.0,
				examples=(0.5,),
				title="Sustain",
				description="The level the sound settles at after its decay, as a share of its peak. At 1 it stays at the peak.",
			),
			Parameter(
				name="release", forms=_NUMBER, unit="ms",
				limit=_NOT_NEGATIVE, sweep=(0.0, 2000.0),
				automatic="sample",
				examples=(100.0,),
				title="Release",
				description="How long the sound takes to fade out at its end. Left out, it follows the sample's decay, which tightens the tail.",
			),
		),
		examples=({"attack": 5.0, "release": 100.0},),
		title="Envelope",
		description="Reshapes how the sound's level moves over time: tighten a loose kick, cut a reverb tail short, or give a soft onset more punch. Left without settings, it tightens the tail to suit the sample.",
	),

	Processor(
		name="transient",
		parameters=(
			Parameter(
				name="gain", forms=_NUMBER, unit="dB",
				sweep=(-12.0, 12.0),
				automatic="sample",
				legacy_names=("amount",),
				examples=(6.0,),
				title="Gain",
				description="How far the hits are turned up, or down where the gain is negative. Left out, it follows the sample's crest factor, so a peaky sound is tamed and a dull one enhanced.",
			),
		),
		examples=({"gain": 6.0},),
		title="Transient shaper",
		description="Turns the hits in the sound up or down against its sustained part, then restores the peak level.",
	),

	Processor(
		name="vocoder",
		parameters=(
			Parameter(
				name="carrier", forms=("choice", "path"),
				choices=(
					Choice("reference", "Reference", "The reference sample for the note played."),
				),
				required=True,
				examples=("carriers/vowel-ah.wav",),
				title="Carrier",
				description="The sound the vocoder shapes: `reference`, or the path to an audio file.",
			),
			Parameter(
				name="bands", forms=_INTEGER,
				limit=Limit(minimum=1), sweep=(4, 48),
				default=24,
				examples=(16,),
				title="Spectral bands",
				description="How many spectral bands the vocoder divides the sound into. Fewer sound more robotic, and more sound more natural.",
			),
			Parameter(
				name="depth", forms=_NUMBER,
				limit=_FRACTION, sweep=(0.0, 1.0),
				default=1.0,
				examples=(0.8,),
				title="Depth",
				description="The balance between the vocoded sound and the original. At 1 only the vocoded sound plays, and at 0 the vocoder has no effect.",
			),
			Parameter(
				name="formant_shift", forms=_INTEGER, unit="semitones",
				sweep=(-12, 12),
				default=0,
				examples=(-5,),
				title="Formant shift",
				description="Moves the carrier's spectral bands up or down against the sound's own, which changes the apparent size of a voice.",
			),
		),
		examples=({"carrier": "reference"}, {"carrier": "carriers/vowel-ah.wav", "bands": 16, "depth": 0.8}),
		title="Vocoder",
		description="Imposes the sound's changing spectrum on a carrier, so the carrier takes on the sound's rhythm and articulation.",
	),
)


PROCESSORS: typing.Final[dict[str, Processor]] = {processor.name: processor for processor in _DECLARED}
"""Every processor a map can name, by its current name, in the order a reference lists them."""
