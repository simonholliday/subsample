"""Tests for subsample/preview.py — visual sample previews."""

import json
import math
import pathlib

import numpy
import pytest

from PIL import Image

import subsample.preview

import tests.helpers


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_preview_data (**overrides: object) -> subsample.preview.PreviewData:

	"""Build a minimal, well-formed PreviewData with per-field overrides.

	Lets each test focus on the specific axis it cares about (e.g. only set
	is_rhythmic=False to verify the beat-grid is suppressed) without
	reconstructing the full dataclass each time.
	"""

	bins      = subsample.preview._ENVELOPE_BINS
	n_bands   = subsample.preview._N_BANDS
	# Even per-band shares by default — test fixtures can override for
	# bass-heavy / highs-heavy scenarios explicitly.
	default: dict[str, object] = {
		"version":      subsample.preview.PREVIEW_VERSION,
		"envelope_min": numpy.zeros(bins, dtype=numpy.int8),
		"envelope_max": numpy.zeros(bins, dtype=numpy.int8),
		"bands":        tuple(numpy.zeros(bins, dtype=numpy.int8) for _ in range(n_bands)),
		"band_totals":  tuple(1.0 / n_bands for _ in range(n_bands)),
		"onset_times":  (0.0, 0.25, 0.5, 0.75),
		"beat_times":   (0.0, 0.5),
		"tempo_bpm":    120.0,
		"duration":     1.0,
		"peak":         0.5,
		"rms":          0.25,
		"accent_rgb":   (200, 100, 150),
		"pitch_label":  "A4",
		"is_rhythmic":  True,
	}
	default.update(overrides)
	return subsample.preview.PreviewData(**default)  # type: ignore[arg-type]


def _synth_sine (sample_rate: int = 44100, freq_hz: float = 440.0, duration_s: float = 1.0) -> numpy.ndarray:

	"""Return a mono float32 sine wave at amplitude 0.5."""

	t = numpy.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False).astype(numpy.float32)
	return (0.5 * numpy.sin(2.0 * numpy.pi * freq_hz * t)).astype(numpy.float32)


# ---------------------------------------------------------------------------
# TestComputePreviewData
# ---------------------------------------------------------------------------


class TestComputePreviewData:

	def test_envelope_shape_and_dtype (self) -> None:

		mono   = _synth_sine(duration_s=0.5)
		rhythm = tests.helpers._make_rhythm()
		pitch  = tests.helpers._make_pitch()
		spec   = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()

		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)

		assert data.envelope_min.shape == (subsample.preview._ENVELOPE_BINS,)
		assert data.envelope_max.shape == (subsample.preview._ENVELOPE_BINS,)
		assert data.envelope_min.dtype == numpy.int8
		assert data.envelope_max.dtype == numpy.int8
		# A 0.5-amplitude sine fills roughly ±63 of the int8 range.
		assert data.envelope_max.max() > 50
		assert data.envelope_min.min() < -50

	def test_bands_shape_and_range (self) -> None:

		mono   = _synth_sine(duration_s=0.5)
		rhythm = tests.helpers._make_rhythm()
		pitch  = tests.helpers._make_pitch()
		spec   = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()

		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)

		assert len(data.bands) == subsample.preview._N_BANDS
		for band in data.bands:
			assert band.shape == (subsample.preview._ENVELOPE_BINS,)
			assert band.dtype == numpy.int8
			# Bands are non-negative per-band-normalised energies.
			assert band.min() >= 0
			assert band.max() <= 127

	def test_tonal_sample_gets_pitch_label (self) -> None:

		"""440 Hz pitch + high confidence → accent hue from pitch class, label "A4"."""

		mono   = _synth_sine(freq_hz=440.0, duration_s=0.5)
		rhythm = tests.helpers._make_rhythm()
		pitch  = tests.helpers._make_pitch(
			dominant_pitch_hz=440.0, pitch_confidence=0.9, dominant_pitch_class=9,
		)
		spec   = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()

		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)

		assert data.pitch_label == "A4"

	def test_unpitched_sample_has_no_pitch_label (self) -> None:

		mono   = _synth_sine(duration_s=0.5)
		rhythm = tests.helpers._make_rhythm()
		pitch  = tests.helpers._make_pitch(pitch_confidence=0.1)   # below threshold
		spec   = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()

		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)

		assert data.pitch_label is None

	def test_rhythmic_flag_requires_tempo_and_two_beats (self) -> None:

		mono  = _synth_sine(duration_s=0.5)
		spec  = tests.helpers._make_spectral()
		level = tests.helpers._make_level()
		pitch = tests.helpers._make_pitch()

		# Has tempo + ≥2 beat_times → is_rhythmic True.
		rhythm_ok = tests.helpers._make_rhythm()
		data = subsample.preview.compute_preview_data(mono, 44100, rhythm_ok, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)
		assert data.is_rhythmic

	def test_zero_tempo_marks_not_rhythmic (self) -> None:

		import subsample.analysis

		mono = _synth_sine(duration_s=0.5)
		spec = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()
		pitch = tests.helpers._make_pitch()

		rhythm = subsample.analysis.RhythmResult(
			tempo_bpm=0.0, beat_times=(), pulse_curve=numpy.zeros(4, dtype=numpy.float32),
			pulse_peak_times=(), onset_times=(), attack_times=(), onset_count=0,
		)
		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)
		assert not data.is_rhythmic

	def test_empty_audio_returns_zeroed_envelopes (self) -> None:

		import subsample.analysis

		mono   = numpy.zeros(0, dtype=numpy.float32)
		rhythm = subsample.analysis.RhythmResult(
			tempo_bpm=0.0, beat_times=(), pulse_curve=numpy.zeros(0, dtype=numpy.float32),
			pulse_peak_times=(), onset_times=(), attack_times=(), onset_count=0,
		)
		pitch  = tests.helpers._make_pitch(pitch_confidence=0.0)
		spec   = tests.helpers._make_spectral()
		level  = subsample.analysis.LevelResult(peak=0.0, rms=0.0)

		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.0)

		assert data.envelope_min.shape == (subsample.preview._ENVELOPE_BINS,)
		assert (data.envelope_min == 0).all()
		assert (data.envelope_max == 0).all()
		for band in data.bands:
			assert (band == 0).all()

	def test_short_sample_with_low_content_fills_bass_band (self) -> None:

		"""Regression guard: short (0.2 s) samples with content in the
		narrowest band (bass, 20-250 Hz) must render a non-zero bass
		envelope.  Previously the hop-length-driven n_fft clamp pinned
		at 256 for short samples, giving a 172 Hz bin width so no FFT
		bins fell in the lowest band — the bottom stratum rendered
		empty regardless of content.  The sample-rate-aware
		_MIN_BINS_PER_BAND floor fixes this."""

		sample_rate = 44100
		duration    = 0.2
		# Pure 60 Hz sine — all energy lives in the bass band.
		t    = numpy.linspace(0.0, duration, int(sample_rate * duration), endpoint=False).astype(numpy.float32)
		mono = (0.5 * numpy.sin(2.0 * numpy.pi * 60.0 * t)).astype(numpy.float32)

		rhythm = tests.helpers._make_rhythm()
		pitch  = tests.helpers._make_pitch()
		spec   = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()

		data = subsample.preview.compute_preview_data(
			mono, sample_rate, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=duration,
		)

		# Band 0 is bass (20-250 Hz); it must register real energy, not zero.
		assert data.bands[0].max() > 0, (
			"bass band silent for 60 Hz content — short-sample FFT floor regressed"
		)


# ---------------------------------------------------------------------------
# TestRenderPng
# ---------------------------------------------------------------------------


class TestRenderPng:

	def test_output_file_exists_and_is_valid_png (self, tmp_path: pathlib.Path) -> None:

		data = _make_preview_data()
		out  = tmp_path / "sample.preview.png"

		subsample.preview.render_png(data, out)

		assert out.exists()
		with Image.open(out) as img:
			img.verify()

	def test_output_dimensions_are_1024_by_256 (self, tmp_path: pathlib.Path) -> None:

		data = _make_preview_data()
		out  = tmp_path / "sample.preview.png"

		subsample.preview.render_png(data, out)

		with Image.open(out) as img:
			assert img.size == (subsample.preview.PNG_WIDTH, subsample.preview.PNG_HEIGHT)

	def test_output_is_opaque_rgb (self, tmp_path: pathlib.Path) -> None:

		"""RGB (no alpha channel) so it renders identically everywhere — the PNG
		will not surprise tools that don't expect transparency."""

		data = _make_preview_data()
		out  = tmp_path / "sample.preview.png"

		subsample.preview.render_png(data, out)

		with Image.open(out) as img:
			assert img.mode == "RGB"

	def test_overwrites_existing_file (self, tmp_path: pathlib.Path) -> None:

		data = _make_preview_data()
		out  = tmp_path / "sample.preview.png"

		out.write_bytes(b"stale content")
		first_size = out.stat().st_size
		subsample.preview.render_png(data, out)
		new_size = out.stat().st_size

		assert new_size != first_size or new_size > len(b"stale content")

	def test_real_audio_renders (self, tmp_path: pathlib.Path) -> None:

		"""End-to-end: compute PreviewData from real synthesised audio, render it,
		and verify the PNG looks like more than a solid fill."""

		mono   = _synth_sine(duration_s=0.5)
		rhythm = tests.helpers._make_rhythm()
		pitch  = tests.helpers._make_pitch()
		spec   = tests.helpers._make_spectral()
		level  = tests.helpers._make_level()

		data = subsample.preview.compute_preview_data(mono, 44100, rhythm, pitch, spec, level, tests.helpers._make_band_energy(), duration=0.5)
		out  = tmp_path / "sample.preview.png"
		subsample.preview.render_png(data, out)

		with Image.open(out) as img:
			pixels = numpy.array(img)

		# A non-trivial render has at least a few distinct colours — at minimum
		# the background, one band stratum, and the waveform accent.
		unique_colors = numpy.unique(pixels.reshape(-1, 3), axis=0)
		assert len(unique_colors) >= 4


# ---------------------------------------------------------------------------
# TestRenderSvg
# ---------------------------------------------------------------------------


class TestRenderSvg:

	def test_produces_well_formed_svg_root (self) -> None:

		svg = subsample.preview.render_svg(_make_preview_data())

		assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
		assert svg.endswith("</svg>")

	def test_default_dimensions_match_png (self) -> None:

		svg = subsample.preview.render_svg(_make_preview_data())
		assert f'width="{subsample.preview.PNG_WIDTH}"'  in svg
		assert f'height="{subsample.preview.PNG_HEIGHT}"' in svg

	def test_custom_dimensions_applied (self) -> None:

		svg = subsample.preview.render_svg(_make_preview_data(), width=2000, height=500)

		assert 'width="2000"'  in svg
		assert 'height="500"'  in svg
		assert 'viewBox="0 0 2000 500"' in svg

	def test_rejects_non_positive_dimensions (self) -> None:

		with pytest.raises(ValueError, match="positive"):
			subsample.preview.render_svg(_make_preview_data(), width=0, height=256)

	def test_includes_waveform_and_band_polygons (self) -> None:

		# Set non-zero envelopes so the waveform and bands actually emit polygons.
		data = _make_preview_data(
			envelope_min = (numpy.ones(subsample.preview._ENVELOPE_BINS, dtype=numpy.int8) * -60),
			envelope_max = (numpy.ones(subsample.preview._ENVELOPE_BINS, dtype=numpy.int8) *  60),
			bands        = tuple(
				numpy.ones(subsample.preview._ENVELOPE_BINS, dtype=numpy.int8) * 80
				for _ in range(subsample.preview._N_BANDS)
			),
		)
		svg = subsample.preview.render_svg(data)

		# One polygon per band + one for the waveform.
		assert svg.count("<polygon") >= subsample.preview._N_BANDS + 1

	def test_emits_onset_ticks_when_present (self) -> None:

		data = _make_preview_data()  # has 4 onset_times
		svg  = subsample.preview.render_svg(data)

		# Each onset becomes a <line>.  At least 4 are present (there are also
		# two reference lines for RMS, so we assert strictly greater).
		assert svg.count("<line") >= 4

	def test_rhythmic_sample_draws_beat_grid (self) -> None:

		data = _make_preview_data(is_rhythmic=True, beat_times=(0.0, 0.5))
		svg  = subsample.preview.render_svg(data)

		assert 'stroke-dasharray="6,6"' in svg

	def test_non_rhythmic_sample_suppresses_beat_grid (self) -> None:

		data = _make_preview_data(is_rhythmic=False)
		svg  = subsample.preview.render_svg(data)

		assert 'stroke-dasharray' not in svg

	def test_badge_included_above_threshold (self) -> None:

		data = _make_preview_data()
		svg  = subsample.preview.render_svg(data, width=1024, height=256)

		assert "A4" in svg                 # pitch label
		assert "120 BPM" in svg            # tempo
		assert "1.00s" in svg              # duration (duration=1.0 → "1.00s")

	def test_badge_suppressed_below_threshold (self) -> None:

		data = _make_preview_data()
		svg  = subsample.preview.render_svg(
			data, width=subsample.preview._SVG_MIN_BADGE_WIDTH - 1, height=100,
		)

		# When badge is suppressed, the pitch label / BPM / duration strings do
		# not appear as <text>.  Just check "A4" isn't present in a <text> tag.
		assert "<text" not in svg

	def test_escapes_xml_special_characters_in_badge (self) -> None:

		data = _make_preview_data(pitch_label="<A>")
		svg  = subsample.preview.render_svg(data)

		# Raw "<A>" must not appear inside the text element; escaped form must.
		assert "&lt;A&gt;" in svg


# ---------------------------------------------------------------------------
# TestSidecarRoundtrip
# ---------------------------------------------------------------------------


class TestSidecarRoundtrip:

	def test_roundtrip_preserves_envelopes (self) -> None:

		data1     = _make_preview_data(
			envelope_min = _ramp(-100, 100, subsample.preview._ENVELOPE_BINS),
			envelope_max = _ramp(-50, 120, subsample.preview._ENVELOPE_BINS),
		)
		payload   = subsample.preview.serialize_for_sidecar(data1)
		data2     = subsample.preview.deserialize_from_sidecar(payload)

		assert (data1.envelope_min == data2.envelope_min).all()
		assert (data1.envelope_max == data2.envelope_max).all()

	def test_roundtrip_preserves_bands (self) -> None:

		data1   = _make_preview_data()
		payload = subsample.preview.serialize_for_sidecar(data1)
		data2   = subsample.preview.deserialize_from_sidecar(payload)

		assert len(data1.bands) == len(data2.bands)
		for b1, b2 in zip(data1.bands, data2.bands):
			assert (b1 == b2).all()

	def test_roundtrip_preserves_scalar_fields (self) -> None:

		data1   = _make_preview_data(
			tempo_bpm=135.5, duration=2.34, peak=0.9, rms=0.4,
			accent_rgb=(10, 200, 50), pitch_label="C#3", is_rhythmic=False,
		)
		payload = subsample.preview.serialize_for_sidecar(data1)
		data2   = subsample.preview.deserialize_from_sidecar(payload)

		assert data2.tempo_bpm   == pytest.approx(135.5)
		assert data2.duration    == pytest.approx(2.34)
		assert data2.peak        == pytest.approx(0.9)
		assert data2.rms         == pytest.approx(0.4)
		assert data2.accent_rgb  == (10, 200, 50)
		assert data2.pitch_label == "C#3"
		assert data2.is_rhythmic is False

	def test_roundtrip_survives_json_dumps_loads (self) -> None:

		"""The serialised form must cleanly traverse an actual JSON roundtrip —
		mirrors what cache.save_cache / load_preview_data do in production."""

		data1    = _make_preview_data()
		payload1 = subsample.preview.serialize_for_sidecar(data1)

		encoded  = json.dumps(payload1)
		payload2 = json.loads(encoded)

		data2 = subsample.preview.deserialize_from_sidecar(payload2)
		assert (data1.envelope_min == data2.envelope_min).all()
		assert (data1.envelope_max == data2.envelope_max).all()

	def test_deserialize_rejects_missing_version (self) -> None:

		with pytest.raises(ValueError, match="malformed preview"):
			subsample.preview.deserialize_from_sidecar({})

	def test_deserialize_rejects_wrong_band_count (self) -> None:

		data    = _make_preview_data()
		payload = subsample.preview.serialize_for_sidecar(data)
		payload["bands"] = payload["bands"][:2]   # drop 3 bands

		with pytest.raises(ValueError, match="malformed preview"):
			subsample.preview.deserialize_from_sidecar(payload)

	def test_deserialize_rejects_wrong_envelope_length (self) -> None:

		import base64
		data    = _make_preview_data()
		payload = subsample.preview.serialize_for_sidecar(data)
		# Replace envelope_min with a base64-encoded 10-byte buffer.
		payload["envelope_min"] = base64.b64encode(b"\x00" * 10).decode("ascii")

		with pytest.raises(ValueError, match="malformed preview"):
			subsample.preview.deserialize_from_sidecar(payload)

	def test_deserialize_rejects_stale_preview_version (self) -> None:

		"""A v=1 sidecar loaded under v=2 must fail with a clear version-
		mismatch message rather than a confusing 'bands must be a list of
		N strings' error.  Matters for users on an existing library after
		upgrading — Supervisor needs to know *why* previews stopped
		rendering so they can run the regen script."""

		data    = _make_preview_data()
		payload = subsample.preview.serialize_for_sidecar(data)
		payload["version"] = subsample.preview.PREVIEW_VERSION - 1

		with pytest.raises(ValueError, match="version"):
			subsample.preview.deserialize_from_sidecar(payload)

	def test_roundtrip_preserves_band_totals (self) -> None:

		"""band_totals round-trips cleanly — the field drives stratum
		heights at render time, so corruption here would visibly
		mis-layout the image."""

		data1   = _make_preview_data(band_totals=(0.6, 0.25, 0.1, 0.05))
		payload = subsample.preview.serialize_for_sidecar(data1)
		data2   = subsample.preview.deserialize_from_sidecar(payload)

		assert data2.band_totals == pytest.approx((0.6, 0.25, 0.1, 0.05))

	def test_deserialize_rejects_wrong_band_totals_length (self) -> None:

		data    = _make_preview_data()
		payload = subsample.preview.serialize_for_sidecar(data)
		payload["band_totals"] = [0.5, 0.5]  # 2 instead of 4

		with pytest.raises(ValueError, match="malformed preview"):
			subsample.preview.deserialize_from_sidecar(payload)


# ---------------------------------------------------------------------------
# TestComputeStratumHeights
# ---------------------------------------------------------------------------


class TestComputeStratumHeights:

	"""Unit tests for the proportional-height allocation.  These cover the
	math independently of the renderers so we can exercise edge cases
	(silent input, single-dominant band) without rendering a full image."""

	def test_sum_equals_total_height (self) -> None:

		"""No rounding residue — strata must fill the image exactly."""

		for totals in [(0.25, 0.25, 0.25, 0.25), (0.7, 0.2, 0.05, 0.05), (1.0, 0.0, 0.0, 0.0)]:
			heights = subsample.preview._compute_stratum_heights(totals, 256)
			assert sum(heights) == 256, f"heights {heights} from {totals} did not sum to 256"

	def test_equal_totals_give_equal_heights (self) -> None:

		heights = subsample.preview._compute_stratum_heights((0.25, 0.25, 0.25, 0.25), 256)
		# With equal totals each band should get ~1/4 of the image; allow
		# 1-px wobble for the final band absorbing the residue.
		for h in heights:
			assert 63 <= h <= 65

	def test_min_floor_respected_for_tiny_band (self) -> None:

		"""A band with ~0% energy must still get _MIN_STRATUM_PX so its
		temporal shape is readable — the key mitigation for collapsed
		strata on extreme distributions."""

		heights = subsample.preview._compute_stratum_heights((0.99, 0.003, 0.003, 0.004), 256)
		for h in heights:
			assert h >= subsample.preview._MIN_STRATUM_PX, (
				f"stratum height {h} < min floor {subsample.preview._MIN_STRATUM_PX}"
			)

	def test_dominant_band_gets_most_height (self) -> None:

		"""A bass-heavy sample must render with bass as the tallest stratum."""

		heights = subsample.preview._compute_stratum_heights((0.8, 0.1, 0.05, 0.05), 256)
		bass    = heights[0]
		others  = heights[1:]
		assert bass > max(others), (
			f"bass={bass} not greater than max other={max(others)} for an 80% bass sample"
		)

	def test_silent_input_falls_back_to_equal_heights (self) -> None:

		"""All zeros (silent sample) → no information → equal heights.
		Avoids the otherwise-NaN divide-by-zero fallout."""

		heights = subsample.preview._compute_stratum_heights((0.0, 0.0, 0.0, 0.0), 256)
		assert sum(heights) == 256
		for h in heights:
			assert 63 <= h <= 65

	def test_rendered_png_bass_heavy_differs_from_highs_heavy (
		self, tmp_path: pathlib.Path,
	) -> None:

		"""End-to-end: two samples with opposite band balance must render
		visibly different skylines.  Uses pixel row sampling — the bass
		stratum (bottom of the image) should hold the bass colour for a
		bass-heavy sample and the highs colour for a highs-heavy one."""

		from PIL import Image

		bass_data = _make_preview_data(
			bands        = tuple(
				numpy.ones(subsample.preview._ENVELOPE_BINS, dtype=numpy.int8) * 100
				for _ in range(subsample.preview._N_BANDS)
			),
			band_totals  = (0.85, 0.10, 0.03, 0.02),
		)
		highs_data = _make_preview_data(
			bands        = tuple(
				numpy.ones(subsample.preview._ENVELOPE_BINS, dtype=numpy.int8) * 100
				for _ in range(subsample.preview._N_BANDS)
			),
			band_totals  = (0.02, 0.03, 0.10, 0.85),
		)

		bass_png  = tmp_path / "bass.png"
		highs_png = tmp_path / "highs.png"
		subsample.preview.render_png(bass_data,  bass_png)
		subsample.preview.render_png(highs_data, highs_png)

		bass_pixels  = numpy.array(Image.open(bass_png))
		highs_pixels = numpy.array(Image.open(highs_png))

		# Middle row (y=128) crosses the waveform — not useful for this
		# check.  Use y=240 (very bottom, should be inside bass stratum
		# for a bass-heavy sample and inside highs stratum for a
		# highs-heavy one, which is definitively different colour).
		assert not numpy.array_equal(bass_pixels[240], highs_pixels[240]), (
			"bass-heavy and highs-heavy samples rendered identically at y=240"
		)


# ---------------------------------------------------------------------------
# TestAccentColor
# ---------------------------------------------------------------------------


class TestAccentColor:

	def test_distinct_pitch_classes_produce_distinct_hues (self) -> None:

		"""C-class and G-class should map to different accent colours so the
		file-manager thumbnail is a per-pitch visual cue."""

		spec  = tests.helpers._make_spectral()
		c_pitch = tests.helpers._make_pitch(pitch_confidence=0.9, dominant_pitch_class=0)
		g_pitch = tests.helpers._make_pitch(pitch_confidence=0.9, dominant_pitch_class=7)

		c_rgb = subsample.preview._compute_accent_rgb(c_pitch, spec)
		g_rgb = subsample.preview._compute_accent_rgb(g_pitch, spec)

		assert c_rgb != g_rgb

	def test_unpitched_accent_from_centroid (self) -> None:

		import dataclasses as dc
		spec_cool = dc.replace(tests.helpers._make_spectral(), spectral_centroid=0.05)
		spec_warm = dc.replace(tests.helpers._make_spectral(), spectral_centroid=0.95)
		pitch     = tests.helpers._make_pitch(pitch_confidence=0.1)

		cool_rgb = subsample.preview._compute_accent_rgb(pitch, spec_cool)
		warm_rgb = subsample.preview._compute_accent_rgb(pitch, spec_warm)

		# Cool = blue-dominant (B >= R);  warm = red/orange-dominant (R >= B).
		assert cool_rgb[2] > cool_rgb[0]
		assert warm_rgb[0] > warm_rgb[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ramp (lo: int, hi: int, n: int) -> numpy.ndarray:

	"""Return an int8 ramp from lo→hi across n bins (inclusive)."""

	return numpy.linspace(lo, hi, n, dtype=numpy.float32).astype(numpy.int8)
