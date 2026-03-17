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
