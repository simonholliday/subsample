"""Tests for subsample.loopfind — seamless loop-point detection."""

import numpy
import pytest

import subsample.loopfind


_SR = 44100


def _steady_tone (seconds: float = 2.0, f0: float = 220.0) -> numpy.ndarray:

	"""A sustained harmonic tone with a fast attack — a clean loop candidate."""

	t = numpy.arange(int(_SR * seconds)) / _SR
	x = numpy.zeros_like(t)
	for h in range(1, 6):
		x += numpy.sin(2 * numpy.pi * f0 * h * t) / h
	# 10 ms attack, otherwise flat (steady sustain, no decay).
	attack = numpy.minimum(t / 0.010, 1.0)
	x = x * attack
	return (0.5 * x / numpy.max(numpy.abs(x))).astype(numpy.float32)


def _click () -> numpy.ndarray:

	"""A 40 ms percussive click — no sustain, not loopable."""

	t = numpy.arange(int(_SR * 0.04)) / _SR
	x = numpy.random.RandomState(0).randn(t.size) * numpy.exp(-t / 0.005)
	return (0.5 * x / numpy.max(numpy.abs(x))).astype(numpy.float32)


def _pitch_glide (seconds: float = 2.0) -> numpy.ndarray:

	"""A flat-amplitude harmonic tone that glides monotonically in pitch.

	It has a steady sustain region (level is flat) but an ever-changing timbre,
	so no two points match and every wrap jumps — a fail-musical case: a region
	exists, yet no clean loop does.
	"""

	n     = int(_SR * seconds)
	t     = numpy.arange(n) / _SR
	f     = numpy.linspace(200.0, 600.0, n)          # monotonic: no two frames match
	phase = 2 * numpy.pi * numpy.cumsum(f) / _SR
	x     = numpy.zeros(n)
	for h in range(1, 6):
		x += numpy.sin(h * phase) / h
	x = x * numpy.minimum(t / 0.010, 1.0)            # 10 ms attack, otherwise flat
	return (0.5 * x / numpy.max(numpy.abs(x))).astype(numpy.float32)


def _stereo (mono: numpy.ndarray) -> numpy.ndarray:

	"""Duplicate a mono signal to two slightly-different channels."""

	return numpy.stack([mono, mono * 0.9], axis=1).astype(numpy.float32)


class TestFindSustainRegion:

	def test_steady_tone_has_region (self) -> None:
		region = subsample.loopfind.find_sustain_region(_steady_tone(), _SR)
		assert region is not None
		lo, hi = region
		assert 0 <= lo < hi <= int(_SR * 2.0)
		# The region should span most of the 2 s tone (attack + margins trimmed).
		assert (hi - lo) / _SR > 1.0

	def test_click_has_no_region (self) -> None:
		assert subsample.loopfind.find_sustain_region(_click(), _SR) is None


class TestFindLoop:

	def test_steady_tone_loops_seamlessly (self) -> None:
		loop = subsample.loopfind.find_loop(_steady_tone(), _SR, pitch_hz=220.0)
		assert loop is not None
		assert loop.start < loop.end
		# A steady harmonic tone should splice cleanly — its wrap is barely above
		# the loop's own frame-to-frame flux (well under the fail-musical cut).
		assert loop.junction_flux < 1.5
		assert loop.crossfade > 0

	def test_loop_lies_inside_sustain_region (self) -> None:
		tone = _steady_tone()
		region = subsample.loopfind.find_sustain_region(tone, _SR)
		loop   = subsample.loopfind.find_loop(tone, _SR, pitch_hz=220.0)
		assert region is not None and loop is not None
		lo, hi = region
		# The loop start skips the attack; both ends sit within the region.
		assert lo <= loop.start < loop.end <= hi

	def test_leaves_a_tail_past_loop_end (self) -> None:
		tone = _steady_tone()
		loop = subsample.loopfind.find_loop(tone, _SR, pitch_hz=220.0)
		assert loop is not None
		# A natural tail must remain for the note-off release.
		assert loop.end < tone.size

	def test_click_returns_none (self) -> None:
		assert subsample.loopfind.find_loop(_click(), _SR) is None

	def test_fail_musical_returns_none (self) -> None:
		# A pitch glide HAS a steady sustain region, but every candidate wrap jumps
		# in timbre, so the cleanest junction still exceeds the fail-musical cut and
		# find_loop declines (returns None despite a region existing).
		glide = _pitch_glide()
		assert subsample.loopfind.find_sustain_region(glide, _SR) is not None
		assert subsample.loopfind.find_loop(glide, _SR, pitch_hz=None) is None

	def test_prefers_longer_clean_loop (self) -> None:
		# Every wrap of a steady tone is clean, so the selection keeps the longest
		# candidate, not a tiny one — it should span a large share of the region.
		tone   = _steady_tone()
		region = subsample.loopfind.find_sustain_region(tone, _SR)
		loop   = subsample.loopfind.find_loop(tone, _SR, pitch_hz=220.0)
		assert region is not None and loop is not None
		lo, hi = region
		assert (loop.end - loop.start) > 0.5 * (hi - lo)

	def test_deterministic (self) -> None:
		tone = _steady_tone()
		a = subsample.loopfind.find_loop(tone, _SR, pitch_hz=220.0)
		b = subsample.loopfind.find_loop(tone, _SR, pitch_hz=220.0)
		assert a == b

	def test_works_without_pitch_hint (self) -> None:
		# Unpitched hint falls back to a pseudo-period; still finds a loop.
		loop = subsample.loopfind.find_loop(_steady_tone(), _SR, pitch_hz=None)
		assert loop is not None

	def test_stereo_input_finds_loop (self) -> None:
		loop = subsample.loopfind.find_loop(_stereo(_steady_tone()), _SR, pitch_hz=220.0)
		assert loop is not None


class TestJunctionFlux:

	"""The selection metric: rendered spectral discontinuity at a butt-jointed wrap."""

	def test_aligned_wrap_is_cleaner_than_misaligned (self) -> None:
		# On a steady tone a whole-period loop wraps seamlessly; ending it half a
		# period short leaves a phase step the flux measure must rate as worse.
		tone   = _steady_tone(f0=220.0)
		period = int(round(_SR / 220.0))
		start  = 4000
		aligned    = subsample.loopfind._junction_flux(tone, start, start + 40 * period, _SR)
		misaligned = subsample.loopfind._junction_flux(tone, start, start + 40 * period + period // 2, _SR)
		assert aligned < misaligned

	def test_degenerate_span_is_infinite (self) -> None:
		# A non-positive span can never loop.
		assert subsample.loopfind._junction_flux(_steady_tone(), 5000, 5000, _SR) == float("inf")


class TestBakeLoopBody:

	def _loop (self) -> subsample.loopfind.LoopPoints:
		return subsample.loopfind.LoopPoints(start=1000, end=5000, crossfade=200, junction_flux=1.0)

	def test_mono_shape (self) -> None:
		tone = _steady_tone()
		body = subsample.loopfind.bake_loop_body(tone, self._loop())
		assert body.shape == (4000,)

	def test_stereo_shape_preserved (self) -> None:
		stereo = _stereo(_steady_tone())
		body   = subsample.loopfind.bake_loop_body(stereo, self._loop())
		assert body.shape == (4000, 2)

	def test_crossfade_alters_tail (self) -> None:
		tone = _steady_tone()
		loop = self._loop()
		body = subsample.loopfind.bake_loop_body(tone, loop)
		# The crossfaded tail must differ from the raw slice tail (the blend ran).
		raw_tail = tone[loop.end - loop.crossfade:loop.end]
		assert not numpy.allclose(body[-loop.crossfade:], raw_tail)

	def test_no_crossfade_is_raw_slice (self) -> None:
		tone = _steady_tone()
		loop = subsample.loopfind.LoopPoints(start=1000, end=5000, crossfade=0, junction_flux=1.0)
		body = subsample.loopfind.bake_loop_body(tone, loop)
		assert numpy.array_equal(body, tone[1000:5000])


class TestRenderAudition:

	def test_length_and_head (self) -> None:
		tone = _steady_tone()
		loop = subsample.loopfind.find_loop(tone, _SR, pitch_hz=220.0)
		assert loop is not None
		out = subsample.loopfind.render_audition(tone, loop, _SR, total_seconds=3.0)
		assert out.shape[0] == int(3.0 * _SR)
		# The real head (attack) is played before the loop begins.
		assert numpy.array_equal(out[:loop.start], tone[:loop.start])

	def test_stereo_preserved (self) -> None:
		stereo = _stereo(_steady_tone())
		loop   = subsample.loopfind.find_loop(stereo, _SR, pitch_hz=220.0)
		assert loop is not None
		out = subsample.loopfind.render_audition(stereo, loop, _SR, total_seconds=3.0)
		assert out.shape == (int(3.0 * _SR), 2)
