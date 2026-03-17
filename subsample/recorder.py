"""Background WAV file writer for Subsample.

Decouples audio I/O from disk I/O by running the WAV writer on a dedicated
daemon thread. The main capture loop hands off completed recordings via a
queue, so file writes never block audio capture.
"""

import datetime
import logging
import pathlib
import queue
import threading
import typing
import wave

import numpy

import subsample.config


_log = logging.getLogger(__name__)

# Sentinel used to signal the writer thread to shut down cleanly
_SHUTDOWN: typing.Final[object] = object()

# Type alias for items placed on the queue: (audio, timestamp) or the sentinel
_QueueItem = typing.Union[tuple[numpy.ndarray, datetime.datetime], object]


class WavWriter:

	"""Writes audio recordings to WAV files on a background daemon thread.

	Usage:
		writer = WavWriter(config)
		writer.enqueue(audio_array, datetime.datetime.now())
		# … later …
		writer.shutdown()
	"""

	def __init__ (self, cfg: subsample.config.Config) -> None:

		"""Start the writer thread and ensure the output directory exists."""

		self._cfg = cfg
		self._queue: queue.Queue[_QueueItem] = queue.Queue()

		output_dir = pathlib.Path(cfg.output.directory)
		output_dir.mkdir(parents=True, exist_ok=True)
		self._output_dir = output_dir

		self._thread = threading.Thread(
			target=self._writer_loop,
			name="wav-writer",
			daemon=True,
		)
		self._thread.start()

	def enqueue (self, audio: numpy.ndarray, timestamp: datetime.datetime) -> None:

		"""Queue an audio array for writing to disk.

		Args:
			audio:     PCM samples as a NumPy integer array, shape (n_frames, channels).
			           For 24-bit audio this is int32 with samples left-shifted by 8.
			timestamp: Datetime used to generate the output filename.
		"""

		self._queue.put((audio, timestamp))

	def shutdown (self) -> None:

		"""Flush remaining recordings and stop the writer thread gracefully."""

		self._queue.put(_SHUTDOWN)
		self._thread.join()

	def _writer_loop (self) -> None:

		"""Main loop for the writer thread; runs until the shutdown sentinel arrives."""

		while True:
			item = self._queue.get()

			if item is _SHUTDOWN:
				break

			# Safe to unpack now that we've ruled out the sentinel
			audio, timestamp = typing.cast(
				tuple[numpy.ndarray, datetime.datetime], item
			)

			self._write_wav(audio, timestamp)

	def _write_wav (self, audio: numpy.ndarray, timestamp: datetime.datetime) -> None:

		"""Write a single audio segment to a WAV file.

		Args:
			audio:     PCM samples, shape (n_frames, channels).
			           16-bit: int16. 24-bit: int32 (left-shifted by 8). 32-bit: int32.
			timestamp: Used to construct the filename.
		"""

		filename = timestamp.strftime(self._cfg.output.filename_format) + ".wav"
		filepath = self._output_dir / filename

		# Ensure the array is 2-D (n_frames, channels) before writing
		if audio.ndim == 1:
			audio = audio.reshape(-1, 1)

		n_channels = audio.shape[1]
		bit_depth = self._cfg.audio.bit_depth
		sample_width = bit_depth // 8

		# 24-bit audio is stored internally as left-shifted int32; recover the
		# original 3-byte values before writing.
		if bit_depth == 24:
			frame_bytes = _pack_int24(audio)
		else:
			frame_bytes = audio.tobytes()

		with wave.open(str(filepath), "wb") as wf:
			wf.setnchannels(n_channels)
			wf.setsampwidth(sample_width)
			wf.setframerate(self._cfg.audio.sample_rate)
			wf.writeframes(frame_bytes)

		n_frames = audio.shape[0]
		duration = n_frames / self._cfg.audio.sample_rate

		_log.debug(
			"Stored recording: file=%s  frames=%d  duration=%.2fs",
			filepath.name, n_frames, duration,
		)


def _pack_int24 (audio: numpy.ndarray) -> bytes:

	"""Pack an int32 array (24-bit values left-shifted by 8) into 3-byte WAV frames.

	Reverses the left-shift applied by unpack_audio() for 24-bit streams,
	recovering the original 24-bit sample values as 3-byte little-endian integers.

	Args:
		audio: Shape (n_frames, channels), dtype int32, values occupying
		       bits 8-31 (the LSB byte is padding zeroes from capture).

	Returns:
		Raw bytes suitable for wave.writeframes().
	"""

	# Right-shift by 8 to undo the LSB padding added at capture
	samples = (audio >> 8).astype(numpy.int32)

	# View each int32 as 4 uint8 bytes (little-endian), then drop byte 3 (MSB padding)
	b = samples.view(numpy.uint8).reshape(-1, 4)

	return b[:, :3].tobytes()
