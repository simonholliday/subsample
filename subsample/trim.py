"""Sub-chunk silence trimming for Subsample.

After the detector emits recording boundaries, the captured segment starts and
ends on an audio-read boundary (recorder.audio.buffer_frames, default 512). This
module trims leading and trailing silence to sample-level precision, with
optional padding to preserve a few samples of context on each side.

When padding is kept, an S-curve (half-cosine) fade is applied over the
padding region — from silence up to the signal on the leading edge, and
from the signal back down to silence on the trailing edge. This produces
a smooth, natural-sounding onset and release without touching the signal
content itself.
"""

import typing

import numpy


def trim_silence (
	audio: numpy.ndarray,
	amplitude_threshold: float,
	pre_samples: int = 0,
	post_samples: int = 0,
	fade_out_samples: int = 0,
	lead_amplitude_threshold: typing.Optional[float] = None,
) -> numpy.ndarray:

	"""Trim leading and trailing silence from an audio segment.

	Scans for the first and last sample whose absolute amplitude meets or
	exceeds amplitude_threshold, then returns the slice between those points
	(inclusive), extended by pre_samples before and post_samples after.

	When pre_samples or post_samples are non-zero, an S-curve (half-cosine)
	fade is applied over the padding region.  When real silence precedes the
	signal, the fade-in covers only that silence and the signal's own attack is
	preserved.  When the signal is loud from the first sample (a fast-attack
	hi-hat/snare whose transient falls inside the pre-read), there is no silence
	to fade, so a fixed pre_samples window is ramped instead — gently
	attenuating the leading samples of real signal to avoid a hard click.

	fade_out_samples widens the trailing fade beyond the post_samples declick to
	a musical length: the last max(post_samples, fade_out_samples) frames are
	ramped down, so a cut taken part-way through a decay (a long tail ended at
	the release level, or a recording ended by the next hit) rings out smoothly
	instead of clicking.  It reaches back into real signal by design.

	The leading (attack) and trailing (tail) edges can use DIFFERENT thresholds.
	amplitude_threshold governs the tail — kept low to preserve a long decay down
	toward the noise floor.  lead_amplitude_threshold, when given, governs the
	onset: a higher value trims the attack tight to the transient instead of to the
	low-level room tone that sits above the tail threshold, so the release level of
	one sample never bleeds low-level noise into the front of the next.  When it is
	None, both edges share amplitude_threshold (historical single-threshold trim).

	If no sample meets the tail threshold (which should not normally occur, since
	the detector validated the segment), the original array is returned unchanged
	to avoid silently discarding a complete recording.

	Args:
		audio:               Shape (n_frames, channels), dtype int16/int32.
		amplitude_threshold: Minimum absolute sample value considered signal at the
		                     TRAILING edge (the tail level).
		pre_samples:         Extra frames to keep before the first loud sample.
		post_samples:        Extra frames to keep after the last loud sample.
		fade_out_samples:    Minimum trailing fade length; 0 keeps the post_samples
		                     declick.  The fade spans max(post_samples, this).
		lead_amplitude_threshold: Minimum absolute sample value for the LEADING
		                     (attack) edge; None reuses amplitude_threshold.  Should
		                     be >= amplitude_threshold so the onset never precedes
		                     the tail end.

	Returns:
		Trimmed slice of audio, same dtype, same number of channels.
		Padding regions are faded in/out with an S-curve envelope.
	"""

	n_frames = audio.shape[0]

	if n_frames == 0:
		return audio

	# Per-frame magnitude: max absolute value across channels.
	# Shape (n_frames,) regardless of mono or stereo.
	# float64 is required for correctness: int16 cannot represent abs(INT16_MIN),
	# and int32 cannot represent abs(INT32_MIN) (which appears in 24-bit left-shifted
	# and native 32-bit audio). float64 covers the full range of all supported dtypes.
	magnitude = numpy.max(numpy.abs(audio.astype(numpy.float64)), axis=-1)

	# Trailing edge: last sample above the tail threshold (preserves the decay).
	tail_above = numpy.where(magnitude >= amplitude_threshold)[0]

	if tail_above.size == 0:
		# No sample exceeded the threshold — return unchanged rather than empty
		return audio

	# Leading edge: first sample above the (higher) attack threshold, so the onset
	# trims tight to the transient.  Fall back to the tail onset when nothing
	# reaches the attack level (a quiet hit) so the sample is not mis-anchored.
	lead_threshold = amplitude_threshold if lead_amplitude_threshold is None else lead_amplitude_threshold
	lead_above = numpy.where(magnitude >= lead_threshold)[0]
	onset_idx = int(lead_above[0]) if lead_above.size else int(tail_above[0])

	start_idx = max(0, onset_idx - pre_samples)
	end_idx = min(n_frames - 1, int(tail_above[-1]) + post_samples)

	# With lead_threshold >= amplitude_threshold the onset is within the tail span,
	# so start <= end; clamp defensively for a caller that passes a lower lead.
	if end_idx < start_idx:
		end_idx = start_idx

	# Copy so the caller owns the data and we can apply fades in-place
	result = audio[start_idx : end_idx + 1].copy()

	# Fade in: S-curve over the pre-signal padding (silence → signal).
	# When there is silence before the signal (onset_idx > start_idx), fade only
	# that region, preserving the signal's own attack envelope.
	# When the signal is loud from sample 0 (onset_idx == start_idx), use a
	# fixed pre_samples window to avoid a hard click — the same peak-vs-RMS
	# mismatch that required the fixed fade-out window also affects fade-in for
	# fast-attack sounds (hi-hat, snare) whose transient falls within the
	# pre-read buffer, placing onset_idx at position 0.
	fade_in_silence = onset_idx - start_idx
	fade_in_len = fade_in_silence if fade_in_silence > 0 else min(pre_samples, len(result))
	if fade_in_len > 1:
		ramp = (1 - numpy.cos(numpy.linspace(0, numpy.pi, fade_in_len))) / 2
		result[:fade_in_len] = (result[:fade_in_len] * ramp[:, numpy.newaxis]).astype(audio.dtype)

	# Fade out: S-curve over the last max(post_samples, fade_out_samples) frames
	# of the output.  Using a fixed window (not end_idx - above[-1]) prevents the
	# fade from being silently skipped when individual sample peaks exceed the
	# threshold during the detector's hold period — a peak-vs-RMS mismatch that
	# causes above[-1] to land at the very last sample, giving fade_out_len = 0.
	# fade_out_samples widens this to a musical fade that rings out a cut taken
	# mid-decay.  Reserve the fade-in region: a very short all-loud buffer where
	# fade_in_len + window > len(result) would otherwise double-attenuate the
	# overlapping samples (two ramps multiplied) and could zero them out.
	fade_out_window = max(post_samples, fade_out_samples)
	fade_out_len = min(fade_out_window, max(0, len(result) - fade_in_len))
	if fade_out_len > 1:
		ramp = (1 + numpy.cos(numpy.linspace(0, numpy.pi, fade_out_len))) / 2
		result[-fade_out_len:] = (result[-fade_out_len:] * ramp[:, numpy.newaxis]).astype(audio.dtype)

	return result
