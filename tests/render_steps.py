"""Render steps that exist only for the tests, registered wherever this module is imported (#4667).

A render worker process imports this module when it unpickles one of these
steps, so their handlers are registered there too.  A test cannot patch what a
worker process runs, but it can hand it a step that fails, dies or warns on
purpose.
"""

import dataclasses
import os

import numpy

import subsample.library
import subsample.parallelism
import subsample.transform


@dataclasses.dataclass(frozen=True)
class Explode:

	"""A step whose render raises."""

	tag: str = "on purpose"


@dataclasses.dataclass(frozen=True)
class Die:

	"""A step that ends the worker process rendering it, as the OOM killer would."""


@dataclasses.dataclass(frozen=True)
class WarnOnce:

	"""A step that logs a once-only warning under ``key``; ``n`` tells one job from another."""

	key: str
	n:   int = 0


def _explode (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      subsample.library.SampleRecord,
	step:        Explode,
) -> numpy.ndarray:

	"""Fail the render."""

	raise RuntimeError(f"exploded {step.tag}")


def _die (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      subsample.library.SampleRecord,
	step:        Die,
) -> numpy.ndarray:

	"""End this worker process at once, or fail where that would end the tests."""

	if subsample.parallelism._in_background_worker:
		os._exit(1)

	raise RuntimeError("Die ran outside a worker process")


def _warn (
	audio:       numpy.ndarray,
	sample_rate: int,
	record:      subsample.library.SampleRecord,
	step:        WarnOnce,
) -> numpy.ndarray:

	"""Log a once-only warning, and pass the audio through."""

	subsample.transform._warn_once(step.key, f"warned about {step.key}")

	return audio


subsample.transform.TransformProcessor._HANDLERS[Explode]  = _explode
subsample.transform.TransformProcessor._HANDLERS[Die]      = _die
subsample.transform.TransformProcessor._HANDLERS[WarnOnce] = _warn
