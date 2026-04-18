"""Visual sample previews — raster PNG sidecars + vector data for SVG rendering.

Emits a compact preview-data block (waveform envelopes, per-band energy
strata, onset/beat markers, accent color, badge text) alongside each captured
or imported sample.  The block lives inside the existing .analysis.json
sidecar so it travels with the audio in the same file.  Two consumers:

  * recorder: renders a fixed 1024×256 PNG sidecar to `<audio>.preview.png`
    using Pillow, for browsing the library in an OS file manager.
  * Supervisor: calls ``render_svg(data, width, height)`` on demand to
    produce a scalable vector view at whatever size the dashboard layout
    needs.

Both renderers consume the same PreviewData so a sample looks recognisable
across file manager and dashboard.

PREVIEW_VERSION is independent of analysis.ANALYSIS_VERSION — cosmetic
changes to the renderers do not require a bump, and a bump here does not
invalidate the rest of the analysis cache.  Only data-shape changes
(envelope resolution, band count, quantisation, added/removed fields)
require a bump.
"""

import base64
import colorsys
import dataclasses
import math
import pathlib
import typing
import xml.sax.saxutils

import librosa
import numpy
from PIL import Image, ImageDraw, ImageFont

import subsample.analysis


PREVIEW_VERSION: int = 3
"""Schema version for the preview data block inside analysis.json.

Bump when the data shape changes (envelope resolution, band count,
quantisation, added/removed fields).  Style changes to the renderers do
not require a bump.

Version history:
  1 — initial release (5-band skyline: sub/low/low-mid/high-mid/air).
  2 — 4-band skyline aligned with analysis.BandEnergyResult
      (bass 20-250 Hz, low-mid 250-2k, high-mid 2-6k, highs 6k+);
      n_fft floor now scales with sample_rate so even short samples
      resolve every band cleanly.
  3 — band_totals field added; stratum heights now scale with each
      band's share of total energy (min-height floor so nothing
      collapses), encoding inter-band balance alongside the
      already-present temporal shape within each stratum.
"""


PNG_WIDTH:  int = 1024
PNG_HEIGHT: int =  256


_ENVELOPE_BINS: int = 400
"""Number of min/max pairs in the waveform envelope and per-band strata.
~400 bins gives a crisp silhouette at 1024 px and still renders cleanly
when the SVG is scaled down in the dashboard."""


_BAND_HZ: tuple[tuple[float, float], ...] = (
	(   20.0,   250.0),   # bass
	(  250.0,  2000.0),   # low-mid
	( 2000.0,  6000.0),   # high-mid
	( 6000.0, 20000.0),   # highs
)
"""Four frequency bands (Hz, half-open intervals) for the band-energy
skyline.  Boundaries match analysis.BandEnergyResult's four bands so a
user comparing the numeric analysis metrics against the visual preview
sees the same taxonomy in both places."""


_N_BANDS: int = len(_BAND_HZ)


_MIN_BINS_PER_BAND: int = 4
"""Minimum FFT bins that must fall inside the narrowest band.  Drives
the n_fft floor in _compute_band_envelopes() so even short samples
resolve every band instead of rendering empty strata.  Lives next to
_BAND_HZ so the two stay co-located."""


_MIN_STRATUM_PX: int = 16
"""Minimum rendered height (px) of each stratum in the band skyline.

When stratum heights scale with band_totals, a sample whose energy is
overwhelmingly concentrated in one band would otherwise collapse the
other strata to 0-2 px — the shape information inside those bands
would become unreadable.  A 16 px floor keeps every band legible
while still giving the dominant band the majority of the height."""


_BAND_RGB: tuple[tuple[int, int, int], ...] = (
	( 110,  70, 140),   # bass      — purple
	(  80, 160, 160),   # low-mid   — teal
	( 210, 170,  70),   # high-mid  — amber
	( 220, 110, 120),   # highs     — pink
)
"""Per-band solid colors, pre-blended against the dark background.

Pre-blended instead of using alpha so the PNG can be flat RGB — Pillow's
alpha handling on an RGB canvas silently ignores fill-colour alpha, so
relying on it would be a subtle source of bugs.  Both renderers draw
these colours as solid fills in layer order (bands → waveform → ticks
→ badge), so later layers can cover earlier ones where they overlap."""


_PITCH_CLASSES: tuple[str, ...] = (
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)


_BG_COLOR:       tuple[int, int, int] = ( 20,  22,  32)
_WAVEFORM_FB:    tuple[int, int, int] = (180, 180, 200)  # unpitched + silent centroid
_REFERENCE_LINE: tuple[int, int, int] = (100, 100, 120)
_ONSET_COLOR:    tuple[int, int, int] = (230, 230, 240)
_BEAT_COLOR:     tuple[int, int, int] = (130, 130, 150)
_BADGE_BG:       tuple[int, int, int] = ( 10,  10,  16)
_BADGE_FG:       tuple[int, int, int] = (220, 220, 230)


_PITCH_CONFIDENCE_THRESHOLD: float = 0.4
"""Samples with pitch_confidence below this are treated as unpitched —
accent colour is driven by spectral centroid and no pitch label is shown.
Matches the threshold used in _format_pitch_label so the badge and the
accent colour always agree on whether a sample is "pitched"."""


_SVG_MIN_BADGE_WIDTH: int = 400
"""Below this rendered width, the SVG corner badge is omitted.  Text
would be illegible at small sizes and adds ~200 bytes of markup."""


@dataclasses.dataclass(frozen=True)
class PreviewData:

	"""Self-contained visual-preview inputs for a single sample.

	Serialised under the ``preview`` key in the sample's .analysis.json
	sidecar.  Renders are deterministic: the same PreviewData always
	produces the same PNG / SVG bytes.
	"""

	version: int

	envelope_min: numpy.ndarray
	"""Shape (_ENVELOPE_BINS,), int8 in [-127, 127] representing the
	minimum float32 sample in each time bin of the source audio."""

	envelope_max: numpy.ndarray
	"""Shape (_ENVELOPE_BINS,), int8 in [-127, 127] — the maximum sample
	per bin.  Paired with envelope_min to draw the mirrored waveform."""

	bands: tuple[numpy.ndarray, ...]
	"""_N_BANDS × (_ENVELOPE_BINS,) int8 arrays in [0, 127].  Per-band
	energy envelope, normalised per-band so each band is visually
	present regardless of overall spectral balance."""

	band_totals: tuple[float, ...]
	"""_N_BANDS floats — each band's share of total spectral energy,
	summing to ~1.0.  Drives the stratum *height* in the rendered
	image, so a bass-heavy sample looks bottom-heavy at a glance.
	Sourced from analysis.BandEnergyResult.energy_fractions (same
	4-band split) to avoid recomputing the same quantity twice."""

	onset_times: tuple[float, ...]
	beat_times:  tuple[float, ...]
	tempo_bpm:   float
	duration:    float
	peak:        float
	rms:         float
	accent_rgb:  tuple[int, int, int]

	pitch_label: typing.Optional[str]
	"""Human-readable pitch label (e.g. "A3") or None when the sample is
	not confidently pitched."""

	is_rhythmic: bool
	"""True iff the sample has a detected tempo and at least two beat
	times.  Gates the beat-grid overlay and the BPM badge segment."""


# ---------------------------------------------------------------------------
# Preview-data computation
# ---------------------------------------------------------------------------


def compute_preview_data (
	mono: numpy.ndarray,
	sample_rate: int,
	rhythm: subsample.analysis.RhythmResult,
	pitch: subsample.analysis.PitchResult,
	spectral: subsample.analysis.AnalysisResult,
	level: subsample.analysis.LevelResult,
	band_energy: subsample.analysis.BandEnergyResult,
	duration: float,
) -> PreviewData:

	"""Build a PreviewData from a mono float32 audio array + analysis results.

	``mono`` is expected to be the same float32 [-1, 1] signal the analysis
	pass consumed (``to_mono_float()`` output), so envelope and onsets are
	coherent with the RhythmResult and LevelResult.

	``band_energy.energy_fractions`` drives the per-band stratum heights in
	the rendered image — passed through rather than recomputed because the
	analysis pass already split the same four bands on the same spectrum.

	Runs on the sample-worker thread after analyze_all() returns; cost is
	dominated by a short-hop STFT (~50 ms for typical samples).
	"""

	mono_f32 = mono.astype(numpy.float32, copy=False)
	env_min, env_max = _compute_waveform_envelope(mono_f32)
	bands            = _compute_band_envelopes(mono_f32, sample_rate)
	accent           = _compute_accent_rgb(pitch, spectral)
	pitch_label      = _format_pitch_label(pitch)
	is_rhythmic      = rhythm.tempo_bpm > 0.0 and len(rhythm.beat_times) >= 2

	# analysis.BandEnergyResult uses the same four edges we do (see plan
	# 2026-04-17), so energy_fractions maps straight onto our strata.  No
	# defensive handling for length mismatch — a mismatch would be a
	# schema divergence bug that should surface at type-check time.
	band_totals = tuple(float(f) for f in band_energy.energy_fractions)

	return PreviewData(
		version      = PREVIEW_VERSION,
		envelope_min = env_min,
		envelope_max = env_max,
		bands        = bands,
		band_totals  = band_totals,
		onset_times  = tuple(rhythm.onset_times),
		beat_times   = tuple(rhythm.beat_times),
		tempo_bpm    = rhythm.tempo_bpm,
		duration     = duration,
		peak         = level.peak,
		rms          = level.rms,
		accent_rgb   = accent,
		pitch_label  = pitch_label,
		is_rhythmic  = is_rhythmic,
	)


def _compute_stratum_heights (
	band_totals: tuple[float, ...],
	total_height: int,
) -> tuple[int, ...]:

	"""Return N integer stratum heights summing to ``total_height``.

	Each band gets at least ``_MIN_STRATUM_PX`` so the temporal shape inside
	every stratum stays readable.  Remaining height is distributed by the
	band's share of total energy.  Silent or degenerate inputs fall back
	to equal heights — a sensible "no information" default.

	Uses a cumulative-position construction so heights sum **exactly** to
	``total_height`` without rounding residue, which matters for a clean
	seam between the top stratum and the image edge.
	"""

	n = len(band_totals)

	if n == 0:
		return tuple()

	# Fallback to equal heights when we can't fit the minima, or when
	# there's no energy information to differentiate the bands.
	total_energy = sum(band_totals)
	if total_height < n * _MIN_STRATUM_PX or total_energy <= 0.0:
		base      = total_height // n
		remainder = total_height - base * n
		heights   = [base + (1 if i < remainder else 0) for i in range(n)]
		return tuple(heights)

	# Cumulative construction: walk the bands in order, placing each
	# stratum's bottom edge at (band_index * min_px) + (cumulative energy
	# fraction) * flex.  The final band absorbs whatever px are left so
	# the sum is exact regardless of rounding inside the loop.
	flex         = total_height - n * _MIN_STRATUM_PX
	cum_fraction = 0.0
	prev_top     = 0
	heights      = []

	for i, t in enumerate(band_totals):
		if i == n - 1:
			heights.append(total_height - prev_top)
		else:
			cum_fraction += t / total_energy
			new_top = _MIN_STRATUM_PX * (i + 1) + int(round(flex * cum_fraction))
			heights.append(new_top - prev_top)
			prev_top = new_top

	return tuple(heights)


def _compute_waveform_envelope (
	mono: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:

	"""Return int8 min/max envelopes, each of length _ENVELOPE_BINS."""

	if mono.size == 0:
		zeros = numpy.zeros(_ENVELOPE_BINS, dtype=numpy.int8)
		return zeros, zeros.copy()

	chunks  = numpy.array_split(mono, _ENVELOPE_BINS)
	env_min = numpy.zeros(_ENVELOPE_BINS, dtype=numpy.float32)
	env_max = numpy.zeros(_ENVELOPE_BINS, dtype=numpy.float32)

	for i, chunk in enumerate(chunks):
		if chunk.size > 0:
			env_min[i] = float(chunk.min())
			env_max[i] = float(chunk.max())

	return _quantise_signed(env_min), _quantise_signed(env_max)


def _compute_band_envelopes (
	mono: numpy.ndarray,
	sample_rate: int,
) -> tuple[numpy.ndarray, ...]:

	"""Return _N_BANDS int8 envelopes of length _ENVELOPE_BINS — one per
	frequency band.  Values are per-band normalised to [0, 1] before
	quantisation so each band is visually present regardless of overall
	spectral balance.

	Uses a short-hop STFT: hop_length ≈ samples / _ENVELOPE_BINS.  The
	absolute-energy figures still live in analysis.band_energy — anyone
	who needs them has them there.
	"""

	zero_band: tuple[numpy.ndarray, ...] = tuple(
		numpy.zeros(_ENVELOPE_BINS, dtype=numpy.int8) for _ in range(_N_BANDS)
	)

	if mono.size < 32:
		return zero_band

	hop_length = max(1, mono.size // _ENVELOPE_BINS)

	# n_fft floor: guarantee at least _MIN_BINS_PER_BAND bins across the
	# narrowest band (currently bass, 230 Hz wide) so even short samples
	# resolve every band.  Without this, samples under ~1.2 s at 44.1 kHz
	# collapsed n_fft to 256 and the lowest band ended up covering zero
	# FFT bins — it rendered as a flat empty stratum regardless of content.
	# Derived from sample_rate rather than hard-coded so behaviour is correct
	# across every recording rate (16/44.1/48/96 kHz all resolve cleanly).
	narrowest_hz     = min(hi - lo for lo, hi in _BAND_HZ)
	min_bin_width_hz = narrowest_hz / _MIN_BINS_PER_BAND
	min_n_fft        = 1 << int(math.ceil(math.log2(max(256.0, sample_rate / min_bin_width_hz))))
	n_fft            = max(min_n_fft, 1 << int(math.ceil(math.log2(max(hop_length * 2, min_n_fft)))))

	spec = numpy.abs(librosa.stft(
		y          = mono,
		n_fft      = n_fft,
		hop_length = hop_length,
		center     = True,
	))
	freqs    = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
	n_frames = spec.shape[1]

	if n_frames == 0:
		return zero_band

	bands_raw = numpy.zeros((_N_BANDS, n_frames), dtype=numpy.float32)
	for b, (lo, hi) in enumerate(_BAND_HZ):
		mask = (freqs >= lo) & (freqs < hi)
		if mask.any():
			bands_raw[b] = spec[mask].sum(axis=0)

	# Per-band normalisation: divide each band by its own peak so the band
	# with the least energy is still drawable.  Preserves shape (what the
	# user cares about visually) while losing the inter-band relative
	# amplitude (which analysis.band_energy still carries).
	peaks            = bands_raw.max(axis=1, keepdims=True)
	peaks[peaks == 0] = 1.0
	bands_norm       = bands_raw / peaks

	# Resample each band's envelope to exactly _ENVELOPE_BINS for alignment
	# with the waveform envelope grid.
	xp = numpy.linspace(0.0, 1.0, n_frames,       dtype=numpy.float32)
	xq = numpy.linspace(0.0, 1.0, _ENVELOPE_BINS, dtype=numpy.float32)

	result: list[numpy.ndarray] = []
	for b in range(_N_BANDS):
		resampled = numpy.interp(xq, xp, bands_norm[b])
		q         = numpy.clip(numpy.round(resampled * 127.0), 0, 127).astype(numpy.int8)
		result.append(q)

	return tuple(result)


def _quantise_signed (arr: numpy.ndarray) -> numpy.ndarray:

	"""Quantise a float32 array in [-1, 1] to int8 in [-127, 127]."""

	quantised = numpy.clip(numpy.round(arr * 127.0), -127, 127)
	return quantised.astype(numpy.int8)


def _compute_accent_rgb (
	pitch: subsample.analysis.PitchResult,
	spectral: subsample.analysis.AnalysisResult,
) -> tuple[int, int, int]:

	"""Derive the waveform stroke colour.

	Tonal samples use hue from the dominant pitch class (12 evenly-spaced
	hues around the wheel).  Non-tonal samples use hue from the spectral
	centroid (cool blue for bassy, warm orange for bright).  Gray
	fallback when both fail.
	"""

	if (
		pitch.pitch_confidence >= _PITCH_CONFIDENCE_THRESHOLD
		and 0 <= pitch.dominant_pitch_class < 12
	):
		hue = pitch.dominant_pitch_class / 12.0
		return _hsv_to_rgb(hue, 0.55, 0.85)

	centroid = spectral.spectral_centroid
	if centroid > 0.0:
		# Map [0, 1] → hue [0.6, 0.1]: bass = blue, treble = orange.
		hue = 0.6 - 0.5 * max(0.0, min(1.0, centroid))
		return _hsv_to_rgb(hue, 0.45, 0.80)

	return _WAVEFORM_FB


def _hsv_to_rgb (h: float, s: float, v: float) -> tuple[int, int, int]:

	"""HSV → sRGB 8-bit helper.  h/s/v in [0, 1]."""

	r, g, b = colorsys.hsv_to_rgb(h, s, v)
	return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _format_pitch_label (pitch: subsample.analysis.PitchResult) -> typing.Optional[str]:

	"""Return e.g. ``"A3"`` for a tonal sample, or None when unpitched."""

	if pitch.pitch_confidence < _PITCH_CONFIDENCE_THRESHOLD:
		return None
	if pitch.dominant_pitch_hz <= 0.0:
		return None
	if not (0 <= pitch.dominant_pitch_class < 12):
		return None

	midi   = 69.0 + 12.0 * math.log2(pitch.dominant_pitch_hz / 440.0)
	octave = int(midi) // 12 - 1

	return f"{_PITCH_CLASSES[pitch.dominant_pitch_class]}{octave}"


# ---------------------------------------------------------------------------
# PNG rendering (Pillow)
# ---------------------------------------------------------------------------


def render_png (data: PreviewData, path: pathlib.Path) -> None:

	"""Render ``data`` as a 1024×256 PNG at ``path``, overwriting if present.

	The composition is entirely opaque RGB (no alpha) so the output file
	is universally readable by OS file managers and image viewers.
	"""

	img  = Image.new("RGB", (PNG_WIDTH, PNG_HEIGHT), _BG_COLOR)
	draw = ImageDraw.Draw(img)

	_png_draw_band_skyline(draw, data)
	_png_draw_waveform(draw, data)
	_png_draw_reference_lines(draw, data)
	_png_draw_onset_ticks(draw, data)

	if data.is_rhythmic:
		_png_draw_beat_grid(draw, data)

	_png_draw_badge(draw, data)

	img.save(path, format="PNG", optimize=True)


def _png_draw_band_skyline (draw: ImageDraw.ImageDraw, data: PreviewData) -> None:

	"""Bottom-up strata, each filled with one band's energy envelope.

	Stratum heights scale with ``data.band_totals`` so the rendered image
	encodes inter-band balance (tall bass stratum = bass-heavy sample) on
	top of the intra-stratum temporal shape each band already carries.
	"""

	heights = _compute_stratum_heights(data.band_totals, PNG_HEIGHT)

	# Walk bands bottom to top, tracking each stratum's y-range.
	stratum_bottom = PNG_HEIGHT
	for b, band_i8 in enumerate(data.bands):
		stratum_top = stratum_bottom - heights[b]
		stratum_h   = stratum_bottom - stratum_top
		band        = band_i8.astype(numpy.float32) / 127.0
		n           = len(band)

		if n < 2 or stratum_h < 1:
			stratum_bottom = stratum_top
			continue

		points: list[tuple[int, int]] = []
		for i in range(n):
			x = int(round(i * (PNG_WIDTH - 1) / (n - 1)))
			y = int(round(stratum_bottom - float(band[i]) * stratum_h * 0.9))
			points.append((x, y))
		points.append((PNG_WIDTH - 1, stratum_bottom))
		points.append((0,             stratum_bottom))

		draw.polygon(points, fill=_BAND_RGB[b])

		stratum_bottom = stratum_top


def _png_draw_waveform (draw: ImageDraw.ImageDraw, data: PreviewData) -> None:

	"""Mirrored filled envelope, centre line at PNG_HEIGHT/2."""

	env_max = data.envelope_max.astype(numpy.float32) / 127.0
	env_min = data.envelope_min.astype(numpy.float32) / 127.0
	n       = len(env_max)

	if n < 2:
		return

	mid_y       = PNG_HEIGHT // 2
	half_height = int(PNG_HEIGHT * 0.44)

	pts_top    = [
		(int(round(i * (PNG_WIDTH - 1) / (n - 1))),
		 mid_y - int(round(float(env_max[i]) * half_height)))
		for i in range(n)
	]
	pts_bottom = [
		(int(round(i * (PNG_WIDTH - 1) / (n - 1))),
		 mid_y - int(round(float(env_min[i]) * half_height)))
		for i in range(n - 1, -1, -1)
	]

	draw.polygon(pts_top + pts_bottom, fill=data.accent_rgb)


def _png_draw_reference_lines (draw: ImageDraw.ImageDraw, data: PreviewData) -> None:

	"""Faint horizontals at ±RMS.  Peak is the waveform's own boundary."""

	if data.rms <= 0.0:
		return

	mid_y       = PNG_HEIGHT // 2
	half_height = int(PNG_HEIGHT * 0.44)
	rms_px      = int(round(data.rms * half_height))

	draw.line(
		[(0, mid_y - rms_px), (PNG_WIDTH - 1, mid_y - rms_px)],
		fill=_REFERENCE_LINE, width=1,
	)
	draw.line(
		[(0, mid_y + rms_px), (PNG_WIDTH - 1, mid_y + rms_px)],
		fill=_REFERENCE_LINE, width=1,
	)


def _png_draw_onset_ticks (draw: ImageDraw.ImageDraw, data: PreviewData) -> None:

	"""Short vertical ticks along the top edge at every onset time."""

	if data.duration <= 0.0:
		return

	tick_h = int(PNG_HEIGHT * 0.08)

	for t in data.onset_times:
		if not 0.0 <= t <= data.duration:
			continue
		x = int(round(t / data.duration * (PNG_WIDTH - 1)))
		draw.line([(x, 0), (x, tick_h)], fill=_ONSET_COLOR, width=1)


def _png_draw_beat_grid (draw: ImageDraw.ImageDraw, data: PreviewData) -> None:

	"""Dashed full-height verticals at beat times."""

	if data.duration <= 0.0:
		return

	for t in data.beat_times:
		if not 0.0 <= t <= data.duration:
			continue
		x = int(round(t / data.duration * (PNG_WIDTH - 1)))
		y = 0
		while y < PNG_HEIGHT:
			draw.line(
				[(x, y), (x, min(y + 6, PNG_HEIGHT - 1))],
				fill=_BEAT_COLOR, width=1,
			)
			y += 12


def _png_draw_badge (draw: ImageDraw.ImageDraw, data: PreviewData) -> None:

	"""Bottom-right chip: pitch, BPM, duration."""

	text = _badge_text(data)
	if not text:
		return

	font   = ImageFont.load_default()
	bbox   = draw.textbbox((0, 0), text, font=font)
	text_w = bbox[2] - bbox[0]
	text_h = bbox[3] - bbox[1]

	x = PNG_WIDTH - text_w - 16
	y = PNG_HEIGHT - text_h - 12
	draw.rectangle(
		[(x - 6, y - 4), (x + text_w + 6, y + text_h + 4)],
		fill=_BADGE_BG,
	)
	draw.text((x, y), text, font=font, fill=_BADGE_FG)


def _badge_text (data: PreviewData) -> str:

	"""Assemble the badge string from whichever fields are meaningful."""

	parts: list[str] = []
	if data.pitch_label is not None:
		parts.append(data.pitch_label)
	if data.is_rhythmic and data.tempo_bpm > 0.0:
		parts.append(f"{int(round(data.tempo_bpm))} BPM")
	parts.append(f"{data.duration:.2f}s")
	return "  ".join(parts)


# ---------------------------------------------------------------------------
# SVG rendering (hand-rolled string)
# ---------------------------------------------------------------------------


def render_svg (
	data: PreviewData,
	width: int = PNG_WIDTH,
	height: int = PNG_HEIGHT,
) -> str:

	"""Return a standalone SVG document string sized ``width``×``height`` px.

	Mirrors render_png() so the same sample is recognisable across both
	formats.  Badge text is suppressed below _SVG_MIN_BADGE_WIDTH where
	it would be illegible.
	"""

	if width <= 0 or height <= 0:
		raise ValueError(f"width and height must be positive (got {width}×{height})")

	parts: list[str] = []
	parts.append(
		f'<svg xmlns="http://www.w3.org/2000/svg" '
		f'viewBox="0 0 {width} {height}" '
		f'width="{width}" height="{height}" '
		f'preserveAspectRatio="none">'
	)
	parts.append(f'<rect width="{width}" height="{height}" fill="{_hex(_BG_COLOR)}"/>')

	_svg_draw_band_skyline(parts, data, width, height)
	_svg_draw_waveform(parts, data, width, height)
	_svg_draw_reference_lines(parts, data, width, height)
	_svg_draw_onset_ticks(parts, data, width, height)

	if data.is_rhythmic:
		_svg_draw_beat_grid(parts, data, width, height)

	if width >= _SVG_MIN_BADGE_WIDTH:
		_svg_draw_badge(parts, data, width, height)

	parts.append("</svg>")
	return "".join(parts)


def _hex (rgb: tuple[int, int, int]) -> str:

	"""Format an 8-bit RGB triple as ``#rrggbb``."""

	return "#{:02x}{:02x}{:02x}".format(*rgb)


def _svg_draw_band_skyline (
	parts: list[str], data: PreviewData, width: int, height: int,
) -> None:

	heights = _compute_stratum_heights(data.band_totals, height)

	stratum_bottom: float = float(height)
	for b, band_i8 in enumerate(data.bands):
		stratum_top = stratum_bottom - heights[b]
		stratum_h   = stratum_bottom - stratum_top
		band        = band_i8.astype(numpy.float32) / 127.0
		n           = len(band)

		if n < 2 or stratum_h < 1:
			stratum_bottom = stratum_top
			continue

		coords: list[str] = []
		for i in range(n):
			x = i * width / (n - 1)
			y = stratum_bottom - float(band[i]) * stratum_h * 0.9
			coords.append(f"{x:.2f},{y:.2f}")
		coords.append(f"{float(width):.2f},{stratum_bottom:.2f}")
		coords.append(f"0,{stratum_bottom:.2f}")

		parts.append(
			f'<polygon points="{" ".join(coords)}" fill="{_hex(_BAND_RGB[b])}"/>'
		)

		stratum_bottom = stratum_top


def _svg_draw_waveform (
	parts: list[str], data: PreviewData, width: int, height: int,
) -> None:

	env_max = data.envelope_max.astype(numpy.float32) / 127.0
	env_min = data.envelope_min.astype(numpy.float32) / 127.0
	n       = len(env_max)

	if n < 2:
		return

	mid_y       = height / 2.0
	half_height = height * 0.44

	coords: list[str] = []
	for i in range(n):
		x = i * width / (n - 1)
		y = mid_y - float(env_max[i]) * half_height
		coords.append(f"{x:.2f},{y:.2f}")
	for i in range(n - 1, -1, -1):
		x = i * width / (n - 1)
		y = mid_y - float(env_min[i]) * half_height
		coords.append(f"{x:.2f},{y:.2f}")

	parts.append(
		f'<polygon points="{" ".join(coords)}" fill="{_hex(data.accent_rgb)}"/>'
	)


def _svg_draw_reference_lines (
	parts: list[str], data: PreviewData, width: int, height: int,
) -> None:

	if data.rms <= 0.0:
		return

	mid_y       = height / 2.0
	half_height = height * 0.44
	rms_px      = data.rms * half_height
	ref_hex     = _hex(_REFERENCE_LINE)

	parts.append(
		f'<line x1="0" y1="{mid_y - rms_px:.2f}" '
		f'x2="{width}" y2="{mid_y - rms_px:.2f}" '
		f'stroke="{ref_hex}" stroke-width="1"/>'
	)
	parts.append(
		f'<line x1="0" y1="{mid_y + rms_px:.2f}" '
		f'x2="{width}" y2="{mid_y + rms_px:.2f}" '
		f'stroke="{ref_hex}" stroke-width="1"/>'
	)


def _svg_draw_onset_ticks (
	parts: list[str], data: PreviewData, width: int, height: int,
) -> None:

	if data.duration <= 0.0:
		return

	tick_h    = height * 0.08
	onset_hex = _hex(_ONSET_COLOR)

	for t in data.onset_times:
		if not 0.0 <= t <= data.duration:
			continue
		x = t / data.duration * width
		parts.append(
			f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{tick_h:.2f}" '
			f'stroke="{onset_hex}" stroke-width="1"/>'
		)


def _svg_draw_beat_grid (
	parts: list[str], data: PreviewData, width: int, height: int,
) -> None:

	if data.duration <= 0.0:
		return

	beat_hex = _hex(_BEAT_COLOR)
	for t in data.beat_times:
		if not 0.0 <= t <= data.duration:
			continue
		x = t / data.duration * width
		parts.append(
			f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" '
			f'stroke="{beat_hex}" stroke-width="1" stroke-dasharray="6,6"/>'
		)


def _svg_draw_badge (
	parts: list[str], data: PreviewData, width: int, height: int,
) -> None:

	text = _badge_text(data)
	if not text:
		return

	font_size = max(10, height // 20)
	# Approximate text width: ~0.55 px per char at font_size (close enough
	# for an sRGB badge background that will be a few px wider than the
	# actual text; browser rendering tolerates the estimate).
	text_w = int(len(text) * font_size * 0.55)
	text_h = font_size
	margin = font_size

	bx = width - text_w - margin - 6
	by = height - text_h - margin / 2

	parts.append(
		f'<rect x="{bx - 6:.2f}" y="{by - 4:.2f}" '
		f'width="{text_w + 12}" height="{text_h + 8}" '
		f'fill="{_hex(_BADGE_BG)}" rx="2"/>'
	)
	parts.append(
		f'<text x="{bx:.2f}" y="{by + text_h - 2:.2f}" '
		f'font-family="monospace" font-size="{font_size}" '
		f'fill="{_hex(_BADGE_FG)}">{xml.sax.saxutils.escape(text)}</text>'
	)


# ---------------------------------------------------------------------------
# Sidecar (de)serialisation
# ---------------------------------------------------------------------------


def serialize_for_sidecar (data: PreviewData) -> dict[str, typing.Any]:

	"""Return the JSON-serialisable dict that lives under ``preview`` in
	.analysis.json.

	Envelopes are int8 packed into base64 strings — ~1.1 KB per 400-value
	array vs ~2 KB as a plain JSON int list, which adds up across a large
	library.
	"""

	return {
		"version":      data.version,
		"envelope_min": base64.b64encode(data.envelope_min.tobytes()).decode("ascii"),
		"envelope_max": base64.b64encode(data.envelope_max.tobytes()).decode("ascii"),
		"bands":        [base64.b64encode(b.tobytes()).decode("ascii") for b in data.bands],
		"band_totals":  list(data.band_totals),
		"onset_times":  list(data.onset_times),
		"beat_times":   list(data.beat_times),
		"tempo_bpm":    data.tempo_bpm,
		"duration":     data.duration,
		"peak":         data.peak,
		"rms":          data.rms,
		"accent_rgb":   list(data.accent_rgb),
		"pitch_label":  data.pitch_label,
		"is_rhythmic":  data.is_rhythmic,
	}


def deserialize_from_sidecar (payload: dict[str, typing.Any]) -> PreviewData:

	"""Inverse of serialize_for_sidecar.  Raises ValueError on malformed input."""

	try:
		version = int(payload["version"])

		# Explicit version check with a clear message — a v=1 sidecar has a
		# 5-element bands list that would otherwise fail the length check
		# below with a confusing "bands must be a list of 4 strings" error.
		# Surfacing the real cause helps anyone debugging a mixed-library
		# where previews render for new captures but not for older ones.
		if version != PREVIEW_VERSION:
			raise ValueError(
				f"preview schema version {version} does not match current "
				f"{PREVIEW_VERSION}; regenerate via scripts/regen_previews_png.py "
				"or a new capture of the sample"
			)

		env_min   = _decode_signed_envelope(payload["envelope_min"])
		env_max   = _decode_signed_envelope(payload["envelope_max"])
		bands_raw = payload["bands"]

		if not isinstance(bands_raw, list) or len(bands_raw) != _N_BANDS:
			raise ValueError(f"bands must be a list of {_N_BANDS} strings")

		bands = tuple(_decode_unsigned_envelope(b) for b in bands_raw)

		band_totals_raw = payload.get("band_totals")
		if not isinstance(band_totals_raw, list) or len(band_totals_raw) != _N_BANDS:
			raise ValueError(f"band_totals must be a list of {_N_BANDS} floats")
		band_totals = tuple(float(t) for t in band_totals_raw)

		onset_times = tuple(float(t) for t in payload.get("onset_times", []))
		beat_times  = tuple(float(t) for t in payload.get("beat_times",  []))
		tempo_bpm   = float(payload.get("tempo_bpm", 0.0))
		duration    = float(payload.get("duration",  0.0))
		peak        = float(payload.get("peak",      0.0))
		rms         = float(payload.get("rms",       0.0))

		accent_raw = payload.get("accent_rgb", [180, 180, 200])
		if not isinstance(accent_raw, list) or len(accent_raw) != 3:
			raise ValueError("accent_rgb must be a 3-element list")
		accent_rgb = (int(accent_raw[0]), int(accent_raw[1]), int(accent_raw[2]))

		pitch_label_raw = payload.get("pitch_label")
		pitch_label     = str(pitch_label_raw) if pitch_label_raw is not None else None
		is_rhythmic     = bool(payload.get("is_rhythmic", False))

	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError(f"malformed preview payload: {exc}") from exc

	return PreviewData(
		version      = version,
		envelope_min = env_min,
		envelope_max = env_max,
		bands        = bands,
		band_totals  = band_totals,
		onset_times  = onset_times,
		beat_times   = beat_times,
		tempo_bpm    = tempo_bpm,
		duration     = duration,
		peak         = peak,
		rms          = rms,
		accent_rgb   = accent_rgb,
		pitch_label  = pitch_label,
		is_rhythmic  = is_rhythmic,
	)


def _decode_signed_envelope (b64: typing.Any) -> numpy.ndarray:

	"""Base64 → int8 ndarray of length _ENVELOPE_BINS.  Raises ValueError on
	any shape or type mismatch."""

	if not isinstance(b64, str):
		raise ValueError("envelope must be a base64 string")

	raw = base64.b64decode(b64.encode("ascii"))
	arr = numpy.frombuffer(raw, dtype=numpy.int8)

	if arr.size != _ENVELOPE_BINS:
		raise ValueError(
			f"envelope has {arr.size} samples (expected {_ENVELOPE_BINS})"
		)

	return arr


def _decode_unsigned_envelope (b64: typing.Any) -> numpy.ndarray:

	"""Like _decode_signed_envelope but clamps negative values to zero.
	Bands are always non-negative; a negative value indicates a corrupt
	payload and is clamped rather than rejected."""

	arr = _decode_signed_envelope(b64)
	if arr.min() < 0:
		clamped: numpy.ndarray = numpy.clip(arr, 0, 127).astype(numpy.int8)
		return clamped
	return arr
