"""Background WAV file writer for Subsample.

Decouples audio I/O from disk I/O by running the WAV writer on a dedicated
daemon thread. The main capture loop hands off completed recordings via a
queue, so file writes never block audio capture.

WavWriter's sole responsibility is: receive audio → run analysis → write WAV
→ save sidecar cache → invoke on_complete callback. It has no knowledge of
similarity scoring, analysis formatting, or any other presentation concern.
Those belong in the on_complete callback supplied by the caller (see cli.py).
"""

import datetime
import logging
import pathlib
import queue
import threading
import typing
import wave

import numpy

import subsample.analysis
import subsample.cache
import subsample.config


_log = logging.getLogger(__name__)

# Sentinel used to signal the writer thread to shut down cleanly
_SHUTDOWN: typing.Final[object] = object()

# Type alias for items placed on the queue: (audio, timestamp) or the sentinel
_QueueItem = typing.Union[tuple[numpy.ndarray, datetime.datetime], object]

# Callback type invoked after each recording is written and analyzed.
# Receives the output path, all three analysis results, the recording duration,
# and the original capture-format PCM audio array for instrument sample storage.
# Runs on the writer thread — use a queue to hand data back to the main thread safely.
_OnCompleteCallback = typing.Callable[
	[
		pathlib.Path,
		subsample.analysis.AnalysisResult,
		subsample.analysis.RhythmResult,
		subsample.analysis.PitchResult,
		float,           # duration in seconds
		numpy.ndarray,   # original capture-format PCM (int16/int32, shape n_frames×channels)
	],
	None,
]


class WavWriter:

	"""Writes audio recordings to WAV files on a background daemon thread.

	Usage:
		writer = WavWriter(config, analysis_params)
		writer.enqueue(audio_array, datetime.datetime.now())
		# … later …
		writer.shutdown()

	IMPORTANT: shutdown() must be called before the process exits. The writer
	thread is a daemon, so if the main thread exits without calling shutdown(),
	any recordings still in the queue will be silently lost.
	"""

	def __init__ (
		self,
		cfg: subsample.config.Config,
		analysis_params: subsample.analysis.AnalysisParams,
		on_complete: typing.Optional[_OnCompleteCallback] = None,
	) -> None:

		"""Start the writer thread and ensure the output directory exists.

		Args:
			cfg:             Full application config.
			analysis_params: Pre-computed FFT params (from compute_params()).
			on_complete:     Optional callback invoked on the writer thread after
			                 each recording is written and analyzed. Receives
			                 (filepath, spectral, rhythm, pitch, duration).
			                 Use a queue to hand results back to the main thread.
		"""

		self._cfg = cfg
		self._analysis_params = analysis_params
		self._on_complete = on_complete
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

			try:
				# Safe to unpack now that we've ruled out the sentinel
				audio, timestamp = typing.cast(
					tuple[numpy.ndarray, datetime.datetime], item
				)

				# Convert once; all three analyses operate on the same mono float array.
				# analyze_all() shares the pyin computation between spectral and pitch
				# analysis, avoiding running it twice (~200-300 ms saving per recording).
				mono = subsample.analysis.to_mono_float(audio, self._cfg.audio.bit_depth)

				result, rhythm, pitch = subsample.analysis.analyze_all(
					mono,
					self._analysis_params,
					self._cfg.analysis,
				)

				write_result = self._write_wav(audio, timestamp, rhythm, result, pitch)

				if self._on_complete is not None and write_result is not None:
					filepath, duration = write_result
					self._on_complete(filepath, result, rhythm, pitch, duration, audio)

			except Exception as exc:
				_log.error("Failed to write recording: %s", exc, exc_info=True)

	def _write_wav (
		self,
		audio: numpy.ndarray,
		timestamp: datetime.datetime,
		rhythm: subsample.analysis.RhythmResult,
		result: subsample.analysis.AnalysisResult,
		pitch: subsample.analysis.PitchResult,
	) -> tuple[pathlib.Path, float] | None:

		"""Write a single audio segment to a WAV file and save its analysis sidecar.

		Returns (filepath, duration_seconds), or None if a free filename could
		not be found (more than 999 same-second collisions - practically impossible).

		Args:
			audio:     PCM samples, shape (n_frames, channels).
			           16-bit: int16. 24-bit: int32 (left-shifted by 8). 32-bit: int32.
			timestamp: Used to construct the filename.
			rhythm:    Rhythm analysis computed before this call.
			result:    Spectral analysis metrics computed before this call.
			pitch:     Pitch and timbre analysis computed before this call.
		"""

		# Find a free filename; cap the loop to prevent spinning forever on a
		# pathological filesystem full of same-timestamp files.
		filename_base = timestamp.strftime(self._cfg.output.filename_format)
		filepath = self._output_dir / (filename_base + ".wav")
		suffix_counter = 2
		while filepath.exists():
			if suffix_counter > 999:
				_log.error(
					"Could not find a free filename for %s (>999 collisions); "
					"recording discarded.", filename_base,
				)
				return None
			filepath = self._output_dir / (filename_base + f"_{suffix_counter}.wav")
			suffix_counter += 1

		# Ensure the array is 2-D (n_frames, channels) before writing
		if audio.ndim == 1:
			audio = audio.reshape(-1, 1)

		n_channels = audio.shape[1]
		bit_depth = self._cfg.audio.bit_depth

		# 24-bit audio is stored internally as left-shifted int32; recover the
		# original 3-byte values before writing. The sample_width must be set
		# to 3 explicitly - bit_depth // 8 = 3 only by coincidence for 24-bit,
		# but the intent is clearer stated directly.
		if bit_depth == 24:
			sample_width = 3
			frame_bytes = _pack_int24(audio)
		else:
			sample_width = bit_depth // 8
			frame_bytes = audio.tobytes()

		with wave.open(str(filepath), "wb") as wf:
			wf.setnchannels(n_channels)
			wf.setsampwidth(sample_width)
			wf.setframerate(self._cfg.audio.sample_rate)
			wf.writeframes(frame_bytes)

		n_frames = audio.shape[0]
		duration = n_frames / self._cfg.audio.sample_rate

		_log.debug("Stored: %s  frames=%d", filepath.name, n_frames)

		# Persist analysis alongside the WAV so future reads (e.g. reference
		# file loading on startup) can skip re-analysis when nothing changes.
		audio_md5 = subsample.cache.compute_audio_md5(filepath)

		subsample.cache.save_cache(
			audio_path = filepath,
			audio_md5  = audio_md5,
			params     = self._analysis_params,
			spectral   = result,
			rhythm     = rhythm,
			pitch      = pitch,
			duration   = duration,
		)

		return filepath, duration


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
	samples = audio >> 8

	# View each int32 as 4 uint8 bytes (little-endian), then drop byte 3 (MSB padding)
	b = samples.view(numpy.uint8).reshape(-1, 4)

	return b[:, :3].tobytes()
