"""Sub-chunk silence trimming for Subsample.

After the detector emits recording boundaries, the captured segment always
starts and ends on a chunk boundary (1024 frames by default). This module
trims leading and trailing silence to sample-level precision, with optional
padding to preserve a few samples of context on each side.
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

	If no sample meets the threshold (which should not normally occur, since
	the detector validated the segment), the original array is returned
	unchanged to avoid silently discarding a complete recording.

	Args:
		audio:               Shape (n_frames, channels), dtype int16.
		amplitude_threshold: Minimum absolute sample value to be considered signal.
		pre_samples:         Extra frames to keep before the first loud sample.
		post_samples:        Extra frames to keep after the last loud sample.

	Returns:
		Trimmed slice of audio, same dtype, same number of channels.
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

	return audio[start_idx : end_idx + 1]
