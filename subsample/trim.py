"""Sub-chunk silence trimming for Subsample.

After the detector emits recording boundaries, the captured segment always
starts and ends on a chunk boundary (1024 frames by default). This module
trims leading and trailing silence to sample-level precision, with optional
padding to preserve a few samples of context on each side.

When padding is kept, an S-curve (half-cosine) fade is applied over the
padding region — from silence up to the signal on the leading edge, and
from the signal back down to silence on the trailing edge. This produces
a smooth, natural-sounding onset and release without touching the signal
content itself.
"""

import numpy


def trim_silence (
	audio: numpy.ndarray,
	amplitude_threshold: float,
	pre_samples: int = 0,
	post_samples: int = 0,
) -> numpy.ndarray:

	"""Trim leading and trailing silence from an audio segment.

	Scans for the first and last sample whose absolute amplitude meets or
	exceeds amplitude_threshold, then returns the slice between those points
	(inclusive), extended by pre_samples before and post_samples after.

	When pre_samples or post_samples are non-zero, an S-curve (half-cosine)
	fade is applied over the padding region. The fade runs from 0.0 at the
	outermost edge to 1.0 at the boundary of the detected signal, so the
	signal content itself is always unaffected.

	If no sample meets the threshold (which should not normally occur, since
	the detector validated the segment), the original array is returned
	unchanged to avoid silently discarding a complete recording.

	Args:
		audio:               Shape (n_frames, channels), dtype int16/int32.
		amplitude_threshold: Minimum absolute sample value to be considered signal.
		pre_samples:         Extra frames to keep before the first loud sample.
		post_samples:        Extra frames to keep after the last loud sample.

	Returns:
		Trimmed slice of audio, same dtype, same number of channels.
		Padding regions are faded in/out with an S-curve envelope.
	"""

	n_frames = audio.shape[0]

	if n_frames == 0:
		return audio

	# Per-frame magnitude: max absolute value across channels
	# Shape (n_frames,) regardless of mono or stereo
	magnitude = numpy.max(numpy.abs(audio.astype(numpy.float64)), axis=-1)

	above = numpy.where(magnitude >= amplitude_threshold)[0]

	if above.size == 0:
		# No sample exceeded the threshold — return unchanged rather than empty
		return audio

	start_idx = max(0, int(above[0]) - pre_samples)
	end_idx = min(n_frames - 1, int(above[-1]) + post_samples)

	# Copy so the caller owns the data and we can apply fades in-place
	result = audio[start_idx : end_idx + 1].copy()

	# Fade in: S-curve over the pre-signal padding (silence → signal)
	fade_in_len = int(above[0]) - start_idx
	if fade_in_len > 1:
		ramp = (1 - numpy.cos(numpy.linspace(0, numpy.pi, fade_in_len))) / 2
		result[:fade_in_len] = (result[:fade_in_len] * ramp[:, None]).astype(audio.dtype)

	# Fade out: S-curve over the last post_samples frames of the output.
	# Using a fixed window (not end_idx - above[-1]) prevents the fade from
	# being silently skipped when individual sample peaks exceed the threshold
	# during the detector's hold period — a peak-vs-RMS mismatch that causes
	# above[-1] to land at the very last sample, giving fade_out_len = 0.
	fade_out_len = min(post_samples, len(result))
	if fade_out_len > 1:
		ramp = (1 + numpy.cos(numpy.linspace(0, numpy.pi, fade_out_len))) / 2
		result[-fade_out_len:] = (result[-fade_out_len:] * ramp[:, None]).astype(audio.dtype)

	return result
