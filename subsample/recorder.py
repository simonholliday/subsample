"""Background WAV file writer for Subsample.

Decouples audio I/O from disk I/O by running the WAV writer on a dedicated
daemon thread. The main capture loop hands off completed recordings via a
queue, so file writes never block audio capture.

WavWriter's sole responsibility is: receive audio → run analysis → write WAV
→ save sidecar cache → invoke on_complete callback. It has no knowledge of
similarity scoring, analysis formatting, or any other presentation concern.
Those belong in the on_complete callback supplied by the caller (see cli.py).
"""

import dataclasses
import datetime
import hashlib
import io
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

# Callback type invoked after each recording is written and analyzed.
# Receives the output path, all four analysis results, the recording duration,
# and the original capture-format PCM audio array for instrument sample storage.
# Runs on the writer thread — use a queue to hand data back to the main thread safely.
_OnCompleteCallback = typing.Callable[
	[
		pathlib.Path,
		subsample.analysis.AnalysisResult,
		subsample.analysis.RhythmResult,
		subsample.analysis.PitchResult,
		subsample.analysis.TimbreResult,
		float,           # duration in seconds
		numpy.ndarray,   # original capture-format PCM (int16/int32, shape n_frames×channels)
	],
	None,
]


@dataclasses.dataclass(frozen=True)
class _WriteRequest:

	"""A single audio segment queued for writing to disk.

	filename_base, sample_rate, and bit_depth are optional overrides used when
	processing audio files rather than live stream chunks. When None, the writer
	falls back to the values from the application config.
	"""

	audio: numpy.ndarray
	timestamp: datetime.datetime
	filename_base: typing.Optional[str] = None   # None → timestamp-based filename
	sample_rate: typing.Optional[int] = None      # None → use config sample rate
	bit_depth: typing.Optional[int] = None        # None → use config bit depth


# Type alias for items placed on the queue
_QueueItem = typing.Union[_WriteRequest, object]


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
			                 (filepath, spectral, rhythm, pitch, timbre, duration, audio).
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

	def enqueue (
		self,
		audio: numpy.ndarray,
		timestamp: datetime.datetime,
		filename_base: typing.Optional[str] = None,
		sample_rate: typing.Optional[int] = None,
		bit_depth: typing.Optional[int] = None,
	) -> None:

		"""Queue an audio array for writing to disk.

		Args:
			audio:         PCM samples as a NumPy integer array, shape (n_frames, channels).
			               For 24-bit audio this is int32 with samples left-shifted by 8.
			timestamp:     Datetime used to generate the output filename when
			               filename_base is None.
			filename_base: If provided, use this as the filename stem (e.g.
			               "field_recording_1") instead of the timestamp format.
			               If the file exists, it will be overwritten.
			sample_rate:   Sample rate for the WAV header. Defaults to config value.
			bit_depth:     Bit depth for WAV writing and mono conversion. Defaults
			               to config value.
		"""

		self._queue.put(_WriteRequest(audio, timestamp, filename_base, sample_rate, bit_depth))

	def flush (self) -> None:

		"""Block until all queued recordings have been written.

		Unlike shutdown(), the writer thread continues running after this returns.
		Use this to ensure file-input segments are on disk before loading the
		instrument library.
		"""

		self._queue.join()

	def shutdown (self) -> None:

		"""Flush remaining recordings and stop the writer thread gracefully.

		Safe to call more than once — subsequent calls after the thread has
		already exited are no-ops.
		"""

		if not self._thread.is_alive():
			return

		self._queue.put(_SHUTDOWN)
		self._thread.join()

	def _writer_loop (self) -> None:

		"""Main loop for the writer thread; runs until the shutdown sentinel arrives."""

		while True:
			item = self._queue.get()

			if item is _SHUTDOWN:
				self._queue.task_done()
				break

			try:
				req = typing.cast(_WriteRequest, item)

				# Use per-request overrides when provided (file-input mode); otherwise
				# fall back to the config values set for the live capture stream.
				effective_bit_depth = req.bit_depth if req.bit_depth is not None else self._cfg.audio.bit_depth

				# Convert once; all three analyses operate on the same mono float array.
				# analyze_all() shares the pyin computation between spectral and pitch
				# analysis, avoiding running it twice (~200-300 ms saving per recording).
				mono = subsample.analysis.to_mono_float(req.audio, effective_bit_depth)

				result, rhythm, pitch, timbre = subsample.analysis.analyze_all(
					mono,
					self._analysis_params,
					self._cfg.analysis,
				)

				write_result = self._write_wav(
					req.audio, req.timestamp, rhythm, result, pitch, timbre,
					filename_base=req.filename_base,
					sample_rate=req.sample_rate,
					bit_depth=req.bit_depth,
				)

				if self._on_complete is not None and write_result is not None:
					filepath, duration = write_result
					self._on_complete(filepath, result, rhythm, pitch, timbre, duration, req.audio)

			except Exception as exc:
				_log.error("Failed to analyse/cache recording: %s — WAV may be intact", exc, exc_info=True)

			finally:
				self._queue.task_done()

	def _write_wav (
		self,
		audio: numpy.ndarray,
		timestamp: datetime.datetime,
		rhythm: subsample.analysis.RhythmResult,
		result: subsample.analysis.AnalysisResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		filename_base: typing.Optional[str] = None,
		sample_rate: typing.Optional[int] = None,
		bit_depth: typing.Optional[int] = None,
	) -> tuple[pathlib.Path, float] | None:

		"""Write a single audio segment to a WAV file and save its analysis sidecar.

		If the target file already exists, it is overwritten (with INFO-level logging).
		If a filesystem error occurs during write, an ERROR is logged and None is returned.

		Returns (filepath, duration_seconds), or None on write failure.

		Args:
			audio:         PCM samples, shape (n_frames, channels).
			               16-bit: int16. 24-bit: int32 (left-shifted by 8). 32-bit: int32.
			timestamp:     Used to construct the filename when filename_base is None.
			rhythm:        Rhythm analysis computed before this call.
			result:        Spectral analysis metrics computed before this call.
			pitch:         Pitch analysis computed before this call.
			timbre:        Timbral fingerprint computed before this call.
			filename_base: If provided, used as the filename stem instead of the
			               timestamp format. Collision handling still applies.
			sample_rate:   Sample rate for the WAV header. Defaults to config value.
			bit_depth:     Bit depth for WAV writing. Defaults to config value.
		"""

		# Resolve effective format values; per-request overrides take precedence.
		effective_sample_rate = sample_rate if sample_rate is not None else self._cfg.audio.sample_rate
		effective_bit_depth   = bit_depth   if bit_depth   is not None else self._cfg.audio.bit_depth

		fname_base = filename_base if filename_base is not None else timestamp.strftime(self._cfg.output.filename_format)
		filepath = self._output_dir / (fname_base + ".wav")

		# Ensure the array is 2-D (n_frames, channels) before writing
		if audio.ndim == 1:
			audio = audio.reshape(-1, 1)

		n_channels = audio.shape[1]

		# 24-bit audio is stored internally as left-shifted int32; recover the
		# original 3-byte values before writing. The sample_width must be set
		# to 3 explicitly - bit_depth // 8 = 3 only by coincidence for 24-bit,
		# but the intent is clearer stated directly.
		if effective_bit_depth == 24:
			sample_width = 3
			frame_bytes = _pack_int24(audio)
		else:
			sample_width = effective_bit_depth // 8
			frame_bytes = audio.tobytes()

		# Build the complete WAV in memory so we can compute the MD5 from the
		# in-memory bytes rather than re-reading the file from disk after writing.
		buf = io.BytesIO()
		with wave.open(buf, "wb") as wf:
			wf.setnchannels(n_channels)
			wf.setsampwidth(sample_width)
			wf.setframerate(effective_sample_rate)
			wf.writeframes(frame_bytes)
		wav_bytes = buf.getvalue()
		audio_md5 = hashlib.md5(wav_bytes).hexdigest()

		try:
			if filepath.exists():
				_log.info("Overwriting: %s", filepath.name)
			filepath.write_bytes(wav_bytes)
		except OSError as exc:
			_log.error("Failed to write %s: %s", filepath.name, exc)
			return None

		n_frames = audio.shape[0]
		duration = n_frames / effective_sample_rate

		_log.debug("Stored: %s  frames=%d", filepath.name, n_frames)

		# Persist analysis alongside the WAV so future reads (e.g. reference
		# file loading on startup) can skip re-analysis when nothing changes.
		subsample.cache.save_cache(
			audio_path = filepath,
			audio_md5  = audio_md5,
			params     = self._analysis_params,
			spectral   = result,
			rhythm     = rhythm,
			pitch      = pitch,
			timbre     = timbre,
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
