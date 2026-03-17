"""Circular audio buffer backed by a pre-allocated NumPy array.

Uses absolute, monotonically-increasing frame indices externally, converting
to buffer positions via modulo internally. This keeps the detector and the
buffer implementation cleanly decoupled — the detector never needs to know
about wrap-around.

Memory footprint at defaults (44100 Hz, mono, 16-bit, 60s): ~5.2 MB.
For 24-bit audio the internal dtype is int32 (4 bytes/sample), giving ~10.4 MB.
"""

import numpy


class CircularBuffer:

	"""Fixed-size circular buffer for raw PCM audio samples.

	Frames are written sequentially; when the buffer is full, old frames are
	silently overwritten. Callers can retrieve any range of frames using the
	absolute frame indices maintained by the write counter.
	"""

	def __init__ (
		self,
		max_frames: int,
		channels: int,
		dtype: numpy.dtype = numpy.dtype(numpy.int16),
	) -> None:

		"""Allocate the buffer.

		Args:
			max_frames: Total number of frames the buffer can hold.
			channels:   Number of audio channels per frame.
			dtype:      NumPy dtype for sample storage. Use int16 for 16-bit audio,
			            int32 for 24-bit (left-shifted) or 32-bit audio.
		"""

		self._data: numpy.ndarray = numpy.zeros(
			(max_frames, channels), dtype=dtype
		)
		self._dtype: numpy.dtype = dtype

		self._max_frames: int = max_frames
		self._channels: int = channels

		# Monotonically increasing count of frames written since creation.
		# The write head position is always (frames_written % max_frames).
		self.frames_written: int = 0

	@property
	def write_head (self) -> int:

		"""Current write position within the buffer array."""

		return self.frames_written % self._max_frames

	@property
	def is_full (self) -> bool:

		"""True once the buffer has been filled at least once."""

		return self.frames_written >= self._max_frames

	def write (self, chunk: numpy.ndarray) -> None:

		"""Write a chunk of frames into the buffer, wrapping if necessary.

		Args:
			chunk: Array of shape (n_frames, channels) or (n_frames,) for mono.
		"""

		# Normalise mono chunks to shape (n, 1) so the buffer is always 2-D
		if chunk.ndim == 1:
			chunk = chunk.reshape(-1, 1)

		n_frames = chunk.shape[0]
		head = self.write_head
		space_to_end = self._max_frames - head

		if n_frames <= space_to_end:
			# Fits without wrapping
			self._data[head : head + n_frames] = chunk
		else:
			# Split across the end of the buffer
			self._data[head:] = chunk[:space_to_end]
			remainder = n_frames - space_to_end
			self._data[:remainder] = chunk[space_to_end:]

		self.frames_written += n_frames

	def read_range (self, start_frame: int, end_frame: int) -> numpy.ndarray:

		"""Return a contiguous copy of audio between two absolute frame indices.

		Handles wrap-around transparently. If the requested range is older than
		the buffer capacity, the available portion is returned (older data has
		been overwritten).

		Args:
			start_frame: Absolute frame index of the first frame to retrieve.
			end_frame:   Absolute frame index one past the last frame to retrieve.

		Returns:
			Array of shape (n_frames, channels) as a new copy.
		"""

		# Clamp start so we never request frames that have been overwritten
		oldest_available = max(0, self.frames_written - self._max_frames)
		start_frame = max(start_frame, oldest_available)

		n = end_frame - start_frame
		if n <= 0:
			return numpy.empty((0, self._channels), dtype=self._dtype)

		start_pos = start_frame % self._max_frames
		end_pos = end_frame % self._max_frames

		if start_pos < end_pos:
			# No wrap in the read range
			return self._data[start_pos:end_pos].copy()

		# Wrap-around: concatenate tail + head
		return numpy.concatenate(
			[self._data[start_pos:], self._data[:end_pos]],
			axis=0,
		)
