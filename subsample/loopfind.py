"""Seamless loop-point detection for sustained samples.

Finds a start/end pair inside a sample's steady sustaining region such that the
audio wraps from end back to start with minimal audible discontinuity, so a held
note can loop that region indefinitely (see the loop playback feature).

The method is spectral self-similarity (SSM) followed by a flux selection:
compare every short frame of the sustain region against every other by MFCC
timbre to propose several candidate loops, then RENDER each candidate's junction
and measure its actual spectral discontinuity, and keep the longest candidate
whose junction is genuinely clean.  SSM alone (taking its single best-scoring
pair) was measurably leaving quality on the table — its top pick routinely
ranked only 3rd-5th cleanest of its own candidates, because MFCC similarity does
not see the level/brightness jump the ear hears at the wrap.  Rendering the
junction does, so it is the selection metric; SSM is only the candidate
generator.  When even the cleanest candidate's junction still measures (and
sounds) discontinuous, the sample has no clean loop and find_loop returns None
(fail-musical) rather than offer a buzzing one.

SSM was chosen over a pitch-period-grid search after ear-auditioning on real
material — the period grid anchors its loop end at the region edge (often the
note's decay) and loses; the free frame-pair search finds cleaner junctions in
the steady middle, on tonal AND textural sounds.

The junction crossfade is short and LINEAR by default: on a correlated (pitched)
junction an equal-power fade adds a coherent +3 dB bump that measures — and
sounds — worse than a raw butt joint, while a linear fade is neutral-to-rescuing
everywhere.  The fade is baked into the returned loop body so a realtime player
need only move a cursor.

Points are found on a mono (mid) signal; a stereo caller applies the same points
to both channels.  Per-channel junction verification is a known refinement, not
yet done here.

This module is pure DSP (numpy/librosa in, plain values out) with no
dependency on the analysis, transform, or player layers — they import it, never
the reverse.
"""

import dataclasses
import typing

import librosa
import numpy


# STFT framing for the region features.  Matches the analysis defaults so the
# frame grid lines up with what the rest of the pipeline sees.
_N_FFT:  typing.Final[int] = 2048
_HOP:    typing.Final[int] = 512

# Sustain-region detection: the region is the longest contiguous run of the RMS
# envelope above (90th-percentile - _REGION_FLOOR_DB); the attack is skipped when
# the run's peak sits in its first fraction; a margin is trimmed off each end so a
# loop point is never right at the region boundary.
_REGION_FLOOR_DB:    typing.Final[float] = 12.0
_REGION_ATTACK_HEAD: typing.Final[float] = 0.30
_REGION_MARGIN_S:    typing.Final[float] = 0.050

# Minimum loop length.  Short loops repeat audibly (a fast "rate" the ear hears
# as artefact) on evolving/tonal material, so the length-bonus prefers longer
# loops; but a stationary texture loops cleanly even short, so the hard floor is
# lenient (0.2 s) and lets the finder pick a short loop when that is genuinely the
# best junction.  Also caps how close the end may sit to the region edge, so a
# natural tail always remains past loop_end for the note-off release.
_MIN_LOOP_S:      typing.Final[float] = 0.20
_END_TAIL_S:      typing.Final[float] = 0.030

# Crossfade: short linear by default (ear-validated).  A pitch period (or a 10 ms
# pseudo-period when unpitched) sets the correlation/refine window.
_CROSSFADE_S:     typing.Final[float] = 0.030
_PSEUDO_PERIOD_S: typing.Final[float] = 0.010

# MFCC self-similarity is penalised by the level difference between the two
# frames (a loud-vs-quiet junction jumps even when timbre matches) and MILDLY
# rewarded for length (a gentle tiebreak toward less-repetitive loops).  These
# only shape which candidates SSM proposes; the actual choice among them is made
# by rendering each junction (see the flux selection below), so the length bias
# stays gentle — a stronger one (0.03) was auditioned and reverted for looping a
# rhythmic sample across its beat.
_SSM_LEVEL_PENALTY: typing.Final[float] = 0.03   # score per dB of level jump
_SSM_LENGTH_BONUS:  typing.Final[float] = 0.01   # gentle tiebreak toward longer

# Flux selection: SSM proposes _CANDIDATE_K loops; each junction is rendered and
# scored by its spectral-flux ratio (see _junction_flux — ~1 is seamless, higher
# is a more audible wrap).  Among the candidates whose junction is within
# _FLUX_TOLERANCE of the cleanest, the LONGEST is kept ("prefer the longest clean
# loop"; a longer loop hides its repetition better).  If even the cleanest
# junction exceeds _FAIL_MUSICAL_FLUX the sample has no clean loop and find_loop
# returns None — better a gated one-shot than a buzzing loop.  Thresholds were
# ear-calibrated on a labelled corpus: every clean loop scored <= 1.75 and the
# one that audibly could not loop (detuned-oscillator filter sweep) scored >=
# 2.6, so the 2.2 cut sits in a wide empty margin, not on a knife edge.
_CANDIDATE_K:        typing.Final[int]   = 10
_FLUX_TOLERANCE:     typing.Final[float] = 0.10
_FAIL_MUSICAL_FLUX:  typing.Final[float] = 2.2
_FLUX_LAPS:          typing.Final[int]   = 5    # loops rendered to measure a junction

# The junction score is a spike RELATIVE to the loop's own median flux, so a
# perfectly stationary signal (median flux -> 0) would divide by ~0 and read as
# a huge spike from mere numerical noise.  Floor the denominator well below the
# quietest real material seen (flute ~0.009) so it never perturbs a genuine
# score, yet keeps a synthetic pure tone (or a flawless digital drone — trivially
# loopable) from a false fail-musical.
_FLUX_MEDIAN_FLOOR:  typing.Final[float] = 1e-3


@dataclasses.dataclass(frozen=True)
class LoopPoints:

	"""A found loop region and its baked crossfade.

	start / end:   Loop boundaries in frames (the wrap jumps from end back to
	               start).  Playback still begins at frame 0, keeping the attack;
	               only the wrap is confined to [start, end).
	crossfade:     Crossfade length in frames, blended into the loop body just
	               before end (linear).  A realtime player uses this by moving a
	               cursor; nothing else.
	junction_flux: Spectral-flux ratio at the raw (un-crossfaded) wrap: ~1 is
	               seamless, higher is a more audible discontinuity.  This is the
	               measure the loop was SELECTED by and it tracks the ear where
	               waveform correlation did not (a filter-swept junction can
	               correlate well yet jump in brightness).  Reported when
	               auditioning; a value near _FAIL_MUSICAL_FLUX is a borderline
	               loop.
	"""

	start:         int
	end:           int
	crossfade:     int
	junction_flux: float


def _to_mono (audio: numpy.ndarray) -> numpy.ndarray:

	"""Return a 1-D float32 mid signal; pass 1-D input through."""

	if audio.ndim == 1:
		return audio.astype(numpy.float32, copy=False)

	# numpy.mean with axis+dtype resolves to a scalar-or-ndarray union in mypy;
	# the actual return with axis=1 is always an ndarray.
	return numpy.mean(audio, axis=1, dtype=numpy.float32)  # type: ignore[return-value]


def find_sustain_region (mono: numpy.ndarray, sample_rate: int) -> typing.Optional[tuple[int, int]]:

	"""Return [lo, hi) sample bounds of the steady sustaining region, or None.

	The region is the longest contiguous run of the RMS envelope above a
	percentile-referenced floor (so a single loud spike can't sink the test),
	with the attack skipped and a margin trimmed off each end.  None when no run
	is long enough to hold a loop.
	"""

	env = librosa.feature.rms(y=mono, frame_length=_N_FFT, hop_length=_HOP)[0]

	if env.size == 0:
		return None

	floor = float(numpy.percentile(env, 90)) * 10.0 ** (-_REGION_FLOOR_DB / 20.0)
	mask  = env >= floor

	# Longest contiguous True run.
	best_lo = best_hi = 0
	i = 0
	while i < mask.size:
		if mask[i]:
			j = i
			while j + 1 < mask.size and mask[j + 1]:
				j += 1
			if j - i > best_hi - best_lo:
				best_lo, best_hi = i, j
			i = j + 1
		else:
			i += 1

	run_lo, run_hi = best_lo, best_hi

	# Skip the attack: if the run's loudest frame is in its first fraction, start
	# after it, so the loop region sits in the steady part (the note still plays
	# its real attack — only the wrap avoids it).
	if run_hi > run_lo:
		peak = run_lo + int(numpy.argmax(env[run_lo:run_hi + 1]))
		if peak - run_lo < _REGION_ATTACK_HEAD * (run_hi - run_lo):
			run_lo = peak

	margin = max(1, int(_REGION_MARGIN_S * sample_rate / _HOP))
	run_lo = min(run_lo + margin, run_hi)
	run_hi = max(run_hi - margin, run_lo)

	lo = int(run_lo * _HOP)
	hi = int(min(run_hi * _HOP + _N_FFT, mono.size))

	if hi - lo < int(_MIN_LOOP_S * sample_rate):
		return None

	return lo, hi


def _rising_zero_crossings (x: numpy.ndarray) -> numpy.ndarray:

	"""Indices where x rises through zero (<=0 then >0)."""

	return numpy.nonzero((x[:-1] <= 0.0) & (x[1:] > 0.0))[0] + 1


def _snap_to_zero_crossing (x: numpy.ndarray, index: int, window: int) -> int:

	"""Nearest rising zero-crossing to index within +/- window, else index."""

	lo = max(0, index - window)
	hi = min(x.size - 1, index + window)
	zc = _rising_zero_crossings(x[lo:hi])

	if zc.size == 0:
		return index

	zc = zc + lo
	return int(zc[numpy.argmin(numpy.abs(zc - index))])


def _ncc (a: numpy.ndarray, b: numpy.ndarray) -> float:

	"""Normalised cross-correlation of two equal-length windows, or 0 if degenerate."""

	na = float(numpy.linalg.norm(a))
	nb = float(numpy.linalg.norm(b))

	if na < 1e-9 or nb < 1e-9:
		return 0.0

	return float(numpy.dot(a, b) / (na * nb))


def _refine_start (x: numpy.ndarray, end: int, start0: int, search: int, window: int) -> int:

	"""Loop start near start0 (+/- search) maximising junction NCC vs x[end:end+window]."""

	w   = min(window, x.size - end)
	ref = x[end:end + w]

	best_start, best_c = start0, -2.0
	for s in range(max(0, start0 - search), min(end - w // 4, start0 + search + 1)):
		if s + w > x.size:
			break
		c = _ncc(x[s:s + w], ref)
		if c > best_c:
			best_c, best_start = c, s

	return best_start


def _spectral_flux (mono: numpy.ndarray) -> numpy.ndarray:

	"""Per-frame positive spectral flux, normalised by the median frame magnitude."""

	spectrum = numpy.abs(librosa.stft(mono, n_fft=_N_FFT, hop_length=_HOP))
	rising   = numpy.maximum(numpy.diff(spectrum, axis=1), 0.0)
	flux     = numpy.sqrt(numpy.sum(rising ** 2, axis=0))
	scale    = float(numpy.median(numpy.sum(spectrum, axis=0))) + 1e-9
	return numpy.asarray(flux / scale, dtype=numpy.float32)


def _junction_flux (mono: numpy.ndarray, start: int, end: int, sample_rate: int) -> float:

	"""Spectral discontinuity at a butt-jointed wrap of [start, end), or inf.

	Plays the loop straight (no crossfade) for a few laps and measures, at each
	wrap, how far the spectral flux spikes above the loop's own median flux.  ~1
	means the wrap is indistinguishable from the loop's natural frame-to-frame
	change (seamless); a higher ratio is an audible jump.  This succeeds where
	waveform correlation fails: it sees a brightness or level step even when the
	two ends line up in phase, and it works on noise textures that never
	correlate at all — which is why it, not NCC, selects the loop.
	"""

	loop_len = end - start
	if loop_len <= 0:
		return float("inf")

	# Render enough laps to expose several wrap junctions for a stable median.
	total_seconds = (start + _FLUX_LAPS * loop_len) / sample_rate
	render = render_audition(mono, LoopPoints(start, end, 0, 0.0), sample_rate, total_seconds=total_seconds)

	flux   = _spectral_flux(render)
	median = max(float(numpy.median(flux)), _FLUX_MEDIAN_FLOOR)

	# In render coordinates the wraps fall at start + k*loop_len == end + (k-1)*loop_len.
	junctions = list(range(end, len(render) - _N_FFT, loop_len))
	junctions = junctions[1:] if len(junctions) > 2 else junctions

	ratios = [
		float(numpy.max(flux[j // _HOP - 2:j // _HOP + 3]) / median)
		for j in junctions
		if 2 <= j // _HOP < flux.size - 3
	]
	return float(numpy.median(ratios)) if ratios else float("inf")


def _ssm_candidates (
	mono:        numpy.ndarray,
	region:      tuple[int, int],
	sample_rate: int,
	period:      int,
	window:      int,
) -> list[tuple[int, int]]:

	"""Propose up to _CANDIDATE_K distinct (start, end) loops by MFCC self-similarity.

	Scores every frame pair in the sustain region by timbre similarity (a 3-frame
	diagonal block, so a neighbourhood matches, not one instant), penalised by
	level jump and mildly rewarded for length, and returns the best-scoring
	DISTINCT pairs — each refined to the sample and snapped to a rising zero
	crossing.  These are only candidates: find_loop chooses among them by
	rendering their junctions, because the SSM score does not see the level or
	brightness step the ear (and _junction_flux) hears at the wrap.
	"""

	lo, hi = region
	seg    = mono[lo:hi]

	mfcc = librosa.feature.mfcc(y=seg, sr=sample_rate, n_mfcc=20, n_fft=_N_FFT, hop_length=_HOP)[1:]
	mfcc = mfcc / (numpy.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-12)
	sim  = mfcc.T @ mfcc

	if sim.shape[0] < 8:
		return []

	# 3-frame diagonal-block average so a whole neighbourhood matches, not one instant.
	blk = (sim[:-2, :-2] + sim[1:-1, 1:-1] + sim[2:, 2:]) / 3.0
	m   = blk.shape[0]

	env = librosa.feature.rms(y=seg, frame_length=_N_FFT, hop_length=_HOP)[0]
	db  = 20.0 * numpy.log10(env + 1e-9)

	level_penalty = numpy.abs(db[:m, None] - db[None, :m]) * _SSM_LEVEL_PENALTY
	length_bonus  = (numpy.arange(m)[None, :] - numpy.arange(m)[:, None]) / m * _SSM_LENGTH_BONUS
	score         = blk - level_penalty + length_bonus

	# Only consider pairs at least a minimum span apart (the loop length floor).
	min_frames = max(4, int(min(_MIN_LOOP_S * sample_rate, 0.5 * (hi - lo)) / _HOP))
	if m <= min_frames:
		return []

	upper_i, upper_j = numpy.triu_indices(m, k=min_frames)
	if upper_i.size == 0:
		return []

	order = numpy.argsort(score[upper_i, upper_j])[::-1]

	candidates: list[tuple[int, int]] = []
	taken:      list[tuple[int, int]] = []
	for idx in order:
		i, j = int(upper_i[idx]), int(upper_j[idx])
		# Skip a pair within a few frames of one already taken, so the candidates
		# span distinct loops rather than neighbours of a single score peak.
		if any(abs(i - ti) < 8 and abs(j - tj) < 8 for ti, tj in taken):
			continue
		taken.append((i, j))

		# --- Refine to the sample, snap to a rising zero-crossing ---
		end = min(lo + j * _HOP, mono.size - window)
		# Keep a natural tail past the loop end for the note-off release.
		end = min(end, mono.size - int(_END_TAIL_S * sample_rate))
		end = _snap_to_zero_crossing(mono, end, period)

		start = _refine_start(mono, end, lo + i * _HOP, _HOP, window)
		start = _snap_to_zero_crossing(mono, start, max(4, period // 8))

		if end - start >= int(_MIN_LOOP_S * sample_rate):
			candidates.append((start, end))

		if len(candidates) >= _CANDIDATE_K:
			break

	return candidates


def find_loop (
	audio:       numpy.ndarray,
	sample_rate: int,
	pitch_hz:    typing.Optional[float] = None,
) -> typing.Optional[LoopPoints]:

	"""Find a seamless loop inside audio's sustain region, or None.

	Proposes several candidate loops by MFCC self-similarity, renders each
	candidate's junction to measure its true spectral discontinuity, and keeps
	the LONGEST whose junction is within _FLUX_TOLERANCE of the cleanest (a
	longer loop hides its repetition better).  Returns None when the sample has
	no usable sustain region, when self-similarity finds no candidate, or when
	even the cleanest junction still exceeds _FAIL_MUSICAL_FLUX — fail-musical: no
	clean loop exists, so the caller should fall back to a gated one-shot rather
	than loop a buzzing junction.

	Args:
		audio:       (n,) or (n, channels) float; points are found on the mid mix.
		sample_rate: Hz.
		pitch_hz:    Detected fundamental, if known (e.g. from the sidecar) — sets
		             the correlation/refine window.  None uses a 10 ms pseudo-
		             period, which is fine for unpitched/textural material.

	Returns:
		LoopPoints (frames, with a linear crossfade length and the selected
		junction_flux), or None.
	"""

	mono = _to_mono(audio)

	region = find_sustain_region(mono, sample_rate)
	if region is None:
		return None

	period = int(round(sample_rate / pitch_hz)) if pitch_hz and pitch_hz > 0.0 else int(_PSEUDO_PERIOD_S * sample_rate)
	window = max(4 * period, int(_PSEUDO_PERIOD_S * sample_rate))

	candidates = _ssm_candidates(mono, region, sample_rate, period, window)
	if not candidates:
		return None

	scored    = [(start, end, _junction_flux(mono, start, end, sample_rate)) for start, end in candidates]
	best_flux = min(flux for _, _, flux in scored)

	if best_flux > _FAIL_MUSICAL_FLUX:
		return None

	# Prefer the LONGEST candidate whose junction is as clean as the best.
	clean            = [c for c in scored if c[2] <= best_flux + _FLUX_TOLERANCE]
	start, end, flux = max(clean, key=lambda c: c[1] - c[0])

	crossfade = min(int(_CROSSFADE_S * sample_rate), start, end - start)

	return LoopPoints(start=start, end=end, crossfade=max(0, crossfade), junction_flux=round(flux, 4))


def bake_loop_body (audio: numpy.ndarray, loop: LoopPoints) -> numpy.ndarray:

	"""Return the loop body [start, end) with the crossfade baked in (linear).

	The last ``loop.crossfade`` frames are cross-faded with the audio just before
	``start``, so wrapping end->start is seamless and a realtime player need only
	jump its cursor.  Shape and dtype follow ``audio`` (mono or multi-channel).
	"""

	start, end, xf = loop.start, loop.end, loop.crossfade
	body = audio[start:end].copy()

	if xf > 0:
		ramp = numpy.linspace(0.0, 1.0, xf, endpoint=False, dtype=numpy.float32)

		if body.ndim == 2:
			ramp_out = (1.0 - ramp)[:, numpy.newaxis]
			ramp_in  = ramp[:, numpy.newaxis]
		else:
			ramp_out, ramp_in = 1.0 - ramp, ramp

		# Blend the pre-end tail (fading out) with the pre-start lead-in (fading
		# in), so the body's end matches its own start.
		body[-xf:] = audio[end - xf:end] * ramp_out + audio[start - xf:start] * ramp_in

	return body


def render_audition (
	audio:        numpy.ndarray,
	loop:         LoopPoints,
	sample_rate:  int,
	total_seconds: float = 6.0,
) -> numpy.ndarray:

	"""Render start -> loop x N -> short fade, for A/B listening.

	Plays the real head (0..start), then repeats the baked loop body until
	total_seconds, then a 30 ms fade so the file itself ends cleanly.  Shape and
	dtype follow ``audio``.
	"""

	body = bake_loop_body(audio, loop)
	head = audio[:loop.start]
	total = int(total_seconds * sample_rate)

	out = numpy.zeros((total,) + audio.shape[1:], dtype=numpy.float32)
	pos = min(head.shape[0], total)
	out[:pos] = head[:pos]

	while pos < total and body.shape[0] > 0:
		n = min(body.shape[0], total - pos)
		out[pos:pos + n] = body[:n]
		pos += n

	fade = min(int(0.030 * sample_rate), total)
	if fade > 0:
		ramp = numpy.linspace(1.0, 0.0, fade, dtype=numpy.float32)
		out[-fade:] *= ramp[:, numpy.newaxis] if out.ndim == 2 else ramp

	return out
