"""Tests for subsample.trim.trim_silence."""

import numpy
import pytest

import subsample.trim


_THRESHOLD = 1000.0  # amplitude threshold used across all tests


def _make_audio (values: list[int], channels: int = 1) -> numpy.ndarray:
	"""Build a mono or stereo int16 array from a list of per-frame values."""
	arr = numpy.array(values, dtype=numpy.int16)

	if channels == 1:
		return arr.reshape(-1, 1)

	# Stereo: duplicate the value across both channels
	return numpy.stack([arr, arr], axis=1)


def _loud (n: int = 10) -> list[int]:
	"""N samples at amplitude well above threshold."""
	return [5000] * n


def _silent (n: int = 10) -> list[int]:
	"""N samples at amplitude well below threshold."""
	return [10] * n


class TestTrimBasic:

	def test_leading_silence_trimmed (self) -> None:
		audio = _make_audio(_silent(20) + _loud(10))

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape[0] == 10
		assert numpy.all(result == 5000)

	def test_trailing_silence_trimmed (self) -> None:
		audio = _make_audio(_loud(10) + _silent(20))

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape[0] == 10
		assert numpy.all(result == 5000)

	def test_both_ends_trimmed (self) -> None:
		audio = _make_audio(_silent(15) + _loud(10) + _silent(15))

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape[0] == 10
		assert numpy.all(result == 5000)

	def test_no_silence_unchanged (self) -> None:
		audio = _make_audio(_loud(30))

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape[0] == 30

	def test_all_silence_returned_unchanged (self) -> None:
		"""A fully silent segment should be returned as-is, not discarded."""
		audio = _make_audio(_silent(50))

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape[0] == 50

	def test_empty_array_returned_unchanged (self) -> None:
		audio = numpy.empty((0, 1), dtype=numpy.int16)

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape[0] == 0


class TestTrimPadding:

	def test_pre_samples_kept (self) -> None:
		# 20 silent + 10 loud — with pre_samples=5, should keep 5 silent before loud
		audio = _make_audio(_silent(20) + _loud(10))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, pre_samples=5)

		assert result.shape[0] == 15  # 5 pre + 10 loud

	def test_post_samples_kept (self) -> None:
		# 10 loud + 20 silent — with post_samples=8, should keep 8 silent after loud
		audio = _make_audio(_loud(10) + _silent(20))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, post_samples=8)

		assert result.shape[0] == 18  # 10 loud + 8 post

	def test_pre_and_post_samples (self) -> None:
		audio = _make_audio(_silent(20) + _loud(10) + _silent(20))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, pre_samples=3, post_samples=4)

		assert result.shape[0] == 17  # 3 pre + 10 loud + 4 post

	def test_pre_samples_clamped_to_array_start (self) -> None:
		# Only 5 silent samples before the loud section; requesting 20 pre-samples
		audio = _make_audio(_silent(5) + _loud(10))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, pre_samples=20)

		# Should extend back to index 0, not past it
		assert result.shape[0] == 15  # 5 silent + 10 loud

	def test_post_samples_clamped_to_array_end (self) -> None:
		# Only 5 silent samples after the loud section; requesting 20 post-samples
		audio = _make_audio(_loud(10) + _silent(5))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, post_samples=20)

		# Should extend to end of array, not past it
		assert result.shape[0] == 15  # 10 loud + 5 silent


class TestTrimFade:

	def test_fade_in_applied (self) -> None:
		"""Padding before the signal should be faded in with an S-curve."""
		audio = _make_audio(_silent(20) + _loud(10))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, pre_samples=10)

		# First sample should be near zero (ramp starts at 0.0)
		assert abs(int(result[0, 0])) < 5

		# Fade region (indices 0–9) should be monotonically increasing —
		# S-curve ramp rises from 0.0 to 1.0 over the padding
		assert result[0, 0] <= result[4, 0] <= result[9, 0]

		# Loud section (indices 10–19) must be untouched
		assert numpy.all(result[10:] == 5000)

	def test_fade_out_applied (self) -> None:
		"""Padding after the signal should be faded out with an S-curve."""
		audio = _make_audio(_loud(10) + _silent(20))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, post_samples=10)

		# Loud section (indices 0–9) must be untouched
		assert numpy.all(result[:10] == 5000)

		# Fade region (indices 10–19) should be monotonically decreasing —
		# S-curve ramp falls from 1.0 to 0.0 over the padding
		assert result[10, 0] >= result[14, 0] >= result[19, 0]

		# Last sample should be near zero (ramp ends at 0.0)
		assert abs(int(result[-1, 0])) < 5

	def test_fade_in_and_out_combined (self) -> None:
		"""Both leading and trailing padding should be faded independently."""
		audio = _make_audio(_silent(20) + _loud(10) + _silent(20))

		result = subsample.trim.trim_silence(
			audio, _THRESHOLD, pre_samples=8, post_samples=8,
		)

		assert result.shape[0] == 26  # 8 pre + 10 loud + 8 post

		# Edges faded
		assert abs(int(result[0, 0])) < 100
		assert abs(int(result[-1, 0])) < 100

		# Loud section untouched
		assert numpy.all(result[8:18] == 5000)

	def test_no_fade_when_no_padding (self) -> None:
		"""With no padding, values should be identical to the original signal."""
		audio = _make_audio(_loud(20))

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert numpy.array_equal(result, audio)

	def test_fade_stereo (self) -> None:
		"""Fade must be applied identically to both channels."""
		audio = _make_audio(_silent(20) + _loud(10), channels=2)

		result = subsample.trim.trim_silence(audio, _THRESHOLD, pre_samples=10)

		# Both channels should be faded equally
		assert numpy.array_equal(result[:, 0], result[:, 1])

		# First sample near zero on both channels
		assert abs(int(result[0, 0])) < 100
		assert abs(int(result[0, 1])) < 100

	def test_fade_clamped_padding (self) -> None:
		"""Fade applies over however many padding samples are available."""
		# Only 3 silent samples before the loud section, but requesting 20
		audio = _make_audio(_silent(3) + _loud(10))

		result = subsample.trim.trim_silence(audio, _THRESHOLD, pre_samples=20)

		# Should extend back to index 0 (3 samples of fade, not 20)
		assert result.shape[0] == 13  # 3 pre + 10 loud

		# First sample near zero (start of fade over 3 samples)
		assert abs(int(result[0, 0])) < 100

		# Loud section untouched
		assert numpy.all(result[3:] == 5000)


	def test_fade_out_applied_when_signal_at_end (self) -> None:
		"""Fade-out must be applied even when the last sample is above threshold.

		Regression: when individual sample peaks exceed the detection threshold
		throughout the hold period (peak > RMS mismatch), above[-1] lands at the
		very last sample giving fade_out_len = 0 under the old threshold-based
		calculation. The new approach always fades the last post_samples frames.
		"""
		# Audio that is loud all the way to the final sample (no trailing silence)
		audio = _make_audio(_loud(50))

		result = subsample.trim.trim_silence(
			audio, _THRESHOLD, post_samples=10
		)

		# Last sample should be near zero (fade-out applied)
		assert abs(int(result[-1, 0])) < 100

		# Some earlier sample within the fade region should be non-zero
		assert abs(int(result[-5, 0])) > 0


class TestTrimStereo:

	def test_stereo_shape_preserved (self) -> None:
		audio = _make_audio(_silent(10) + _loud(10) + _silent(10), channels=2)

		result = subsample.trim.trim_silence(audio, _THRESHOLD)

		assert result.shape == (10, 2)

	def test_stereo_uses_max_across_channels (self) -> None:
		"""Trimming should trigger on either channel, not just the first."""
		n = 30
		arr = numpy.zeros((n, 2), dtype=numpy.int16)

		# Only channel 1 (right) has loud signal in the middle
		arr[10:20, 0] = 10     # left channel stays silent
		arr[10:20, 1] = 5000   # right channel is loud

		result = subsample.trim.trim_silence(arr, _THRESHOLD)

		assert result.shape[0] == 10
