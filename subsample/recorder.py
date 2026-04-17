"""Sample processing pipeline for Subsample.

Decouples audio capture from analysis and disk I/O by running per-sample
work on a thread-pool executor. The recorder thread hands off completed
recordings via submit(); each worker independently runs the full pipeline:
  convert → analyze → write WAV → save sidecar → invoke on_complete callback

Worker count is auto-scaled from os.cpu_count() at construction time — no
configuration needed. Two cores are reserved for audio threads (recorder +
player callback); half the remainder are used for processing. This degrades
gracefully to a single worker on a Raspberry Pi and scales up automatically
on multi-core machines.

SampleProcessor has no knowledge of similarity scoring, analysis formatting,
or any other presentation concern. Those belong in the on_complete callback
supplied by the caller (see cli.py).
"""

import concurrent.futures
import dataclasses
import datetime
import hashlib
import io
import logging
import math
import os
import pathlib
import threading
import typing
import wave

import numpy
import soundfile  # type: ignore[import-untyped]  # soundfile ships no stubs

import subsample.ambisonic
import subsample.analysis
import subsample.cache
import subsample.config


_log = logging.getLogger(__name__)


# FLAC compression level passed to soundfile.write().  python-soundfile's
# [0.0, 1.0] float maps to libsndfile's 0-8 integer levels via
# floor(value * 8).  0.75 = FLAC level 6 — the measured sweet spot:
# near-maximum compression at materially lower encode CPU than level 8.
# Leaving it at None would fall back to libsndfile's level 5 default.
_FLAC_COMPRESSION_LEVEL: float = 0.75


def _pcm_float_to_int (audio: numpy.ndarray, bit_depth: int) -> numpy.ndarray:

	"""Inverse of analysis.to_mono_float — float32 in [-1, 1] → capture-format PCM.

	The output dtype matches what the recorder's capture path produces: int16
	for 16-bit, and int32 for 24-bit (left-shifted by 8) and native 32-bit.
	Values are clipped to the full-scale range before conversion so an
	out-of-range float does not wrap around to a negative integer.
	"""

	divisor = 32768.0 if bit_depth == 16 else 2147483648.0
	max_int = int(divisor) - 1
	min_int = -int(divisor)

	clipped = numpy.clip(audio, -1.0, 1.0 - 1.0 / divisor)
	scaled  = numpy.rint(clipped * divisor).astype(numpy.int64)
	scaled  = numpy.clip(scaled, min_int, max_int)

	target_dtype: numpy.dtype[typing.Any] = numpy.dtype(numpy.int16) if bit_depth == 16 else numpy.dtype(numpy.int32)

	# For 24-bit, the caller stores values left-shifted by 8 bits inside int32.
	# to_mono_float divides by 2^31 (full int32 range) to recover the [-1, 1]
	# float value, so multiplying back by 2^31 and rounding produces the same
	# left-shifted int32 representation the rest of the pipeline expects.

	result: numpy.ndarray = scaled.astype(target_dtype)
	return result


def _compute_worker_count () -> int:

	"""Return the number of processing workers to use.

	Reserves 2 cores for audio threads (recorder input + player callback),
	then uses half the remainder — at least 1. Scales automatically from
	a Raspberry Pi (1 worker) up to a workstation (many workers).
	"""

	cpu_count = os.cpu_count() or 1
	return max(1, (cpu_count - 2) // 2)


# Callback type invoked after each recording is written and analyzed.
# Receives the output path, all analysis results, the recording duration,
# and the original capture-format PCM audio array for instrument sample storage.
# Runs on a worker thread — use a queue to hand data back to the main thread safely.
_OnCompleteCallback = typing.Callable[
	[
		pathlib.Path,
		subsample.analysis.AnalysisResult,
		subsample.analysis.RhythmResult,
		subsample.analysis.PitchResult,
		subsample.analysis.TimbreResult,
		subsample.analysis.LevelResult,
		subsample.analysis.BandEnergyResult,
		float,           # duration in seconds
		numpy.ndarray,   # original capture-format PCM (int16/int32, shape n_frames×channels)
	],
	None,
]


def _format_filename (timestamp: datetime.datetime, fmt: str) -> str:

	"""Format a filename from a timestamp using an extended strftime format.

	Supports all standard strftime codes plus ``%3f`` which expands to the
	zero-padded 3-digit millisecond value (e.g. "007", "123").  This avoids
	collisions when two recordings end in the same second.
	"""

	ms = f"{timestamp.microsecond // 1000:03d}"
	return timestamp.strftime(fmt.replace("%3f", ms))


@dataclasses.dataclass(frozen=True)
class _ProcessRequest:

	"""A single audio segment submitted for processing.

	filename_base, sample_rate, and bit_depth are optional overrides used when
	processing audio files rather than live stream chunks. When None, the worker
	falls back to the values from the application config.
	"""

	audio: numpy.ndarray
	timestamp: datetime.datetime
	filename_base: typing.Optional[str] = None   # None → timestamp-based filename
	sample_rate: typing.Optional[int] = None      # None → use config sample rate
	bit_depth: typing.Optional[int] = None        # None → use config bit depth


class SampleProcessor:

	"""Processes audio recordings on a thread-pool executor.

	Each submitted recording runs the full pipeline concurrently:
	  convert → analyze → write WAV → save sidecar → on_complete callback

	Worker count is auto-scaled from os.cpu_count() — see _compute_worker_count().
	Results may complete out of order; InstrumentLibrary and SimilarityMatrix
	are both thread-safe and handle concurrent updates correctly.

	Usage:
		processor = SampleProcessor(config, analysis_params)
		processor.enqueue(audio_array, datetime.datetime.now())
		# … later …
		processor.shutdown()

	IMPORTANT: shutdown() must be called before the process exits to ensure
	all in-flight recordings complete and their WAV files are written.
	"""

	def __init__ (
		self,
		cfg: subsample.config.Config,
		analysis_params: subsample.analysis.AnalysisParams,
		on_complete: typing.Optional[_OnCompleteCallback] = None,
		warn_backlog: bool = True,
	) -> None:

		"""Start the worker pool and ensure the output directory exists.

		Args:
			cfg:             Full application config.
			analysis_params: Pre-computed FFT params (from compute_params()).
			on_complete:     Optional callback invoked on a worker thread after
			                 each recording is written and analyzed. Receives
			                 (filepath, spectral, rhythm, pitch, timbre, level,
			                 band_energy, duration, audio). Use a queue to pass
			                 results back to the main thread if needed.
			warn_backlog:    When True (default), log a WARNING when 3+ segments
			                 are in-flight. Set False for file-input mode where
			                 faster-than-realtime enqueue is expected.
		"""

		self._cfg             = cfg
		self._analysis_params = analysis_params
		self._on_complete     = on_complete
		self._warn_backlog    = warn_backlog

		output_dir = pathlib.Path(cfg.output.directory)
		output_dir.mkdir(parents=True, exist_ok=True)
		self._output_dir = output_dir

		self._n_workers = _compute_worker_count()
		_log.info(
			"SampleProcessor: %d worker(s) (cpu_count=%d)",
			self._n_workers, os.cpu_count() or 1,
		)

		self._executor = concurrent.futures.ThreadPoolExecutor(
			max_workers=self._n_workers,
			thread_name_prefix="sample-worker",
		)

		# Track pending futures for flush() and queue_depth.
		# Protected by _futures_lock since workers complete on arbitrary threads.
		self._futures:      list[concurrent.futures.Future[None]] = []
		self._futures_lock: threading.Lock = threading.Lock()

		# Set to True when a backlog warning fires; cleared on drain.
		# Ensures the "queue drained" INFO fires once per backlog episode.
		self._was_backed_up: bool = False

	def enqueue (
		self,
		audio: numpy.ndarray,
		timestamp: datetime.datetime,
		filename_base: typing.Optional[str] = None,
		sample_rate: typing.Optional[int] = None,
		bit_depth: typing.Optional[int] = None,
	) -> None:

		"""Submit an audio array for processing.

		Returns immediately. The recording is analyzed and written to disk
		by the next available worker thread.

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

		req = _ProcessRequest(audio, timestamp, filename_base, sample_rate, bit_depth)

		future = self._executor.submit(self._process, req)
		future.add_done_callback(self._on_future_done)

		with self._futures_lock:
			# Drop completed futures to keep the list short.
			self._futures = [f for f in self._futures if not f.done()]
			self._futures.append(future)
			depth = sum(1 for f in self._futures if not f.done())

		if self._warn_backlog and depth >= 3:
			_log.warning(
				"sample-processor backlog: %d in-flight — processing may be falling behind captures",
				depth,
			)
			self._was_backed_up = True

	@property
	def queue_depth (self) -> int:

		"""Number of recordings currently being processed or waiting for a worker."""

		with self._futures_lock:
			return sum(1 for f in self._futures if not f.done())

	def flush (self) -> None:

		"""Block until all submitted recordings have finished processing.

		Unlike shutdown(), the executor continues running after this returns.
		Use this to ensure file-input segments are on disk before loading the
		instrument library.
		"""

		with self._futures_lock:
			pending = list(self._futures)

		if pending:
			concurrent.futures.wait(pending)

	def shutdown (self) -> None:

		"""Wait for all in-flight recordings to complete and stop the worker pool.

		Safe to call more than once — subsequent calls after the executor has
		already shut down are no-ops.
		"""

		self._executor.shutdown(wait=True)

	def _on_future_done (self, future: concurrent.futures.Future[None]) -> None:

		"""Done callback — logs INFO when a backlog episode fully drains.

		Called on the completing worker thread. Checks under the lock whether
		all futures are now done and a backlog warning was previously emitted.
		"""

		with self._futures_lock:
			if self._was_backed_up and all(
				f.done() for f in self._futures
			):
				_log.info("sample-processor queue drained")
				self._was_backed_up = False

	def _process (self, req: _ProcessRequest) -> None:

		"""Full processing pipeline for a single recording. Runs on a worker thread.

		Sequence: convert → analyze → write WAV → save sidecar → on_complete callback.
		Exceptions are caught and logged so one failed recording never kills the worker.
		"""

		try:
			# Use per-request overrides when provided (file-input mode); otherwise
			# fall back to the config values set for the live capture stream.
			effective_bit_depth = (
				req.bit_depth if req.bit_depth is not None
				else self._cfg.recorder.audio.bit_depth
			)
			effective_sample_rate = (
				req.sample_rate if req.sample_rate is not None
				else self._cfg.recorder.audio.sample_rate
			)

			# Ambisonic capture: convert the 4-channel PCM to canonical AmbiX
			# B-format before storage and analysis.  Downstream the WAV file on
			# disk is B-format (channel order W, Y, Z, X; SN3D) and analysis
			# feeds on the W channel (index 0) so the spectral/rhythm/pitch
			# fingerprint reflects the omnidirectional sum of the sound field
			# rather than a directionally biased mix of the velocity channels.
			ambisonic_format = self._cfg.recorder.audio.ambisonic_format
			analysis_channel_index: typing.Optional[int]
			channel_format_tag: str

			if ambisonic_format is not None:
				audio_float = req.audio.astype(numpy.float32) / (
					32768.0 if effective_bit_depth == 16 else 2147483648.0
				)
				b_format_float = subsample.ambisonic.process_capture(
					audio_float, ambisonic_format, sample_rate=effective_sample_rate,
				)
				req = dataclasses.replace(
					req, audio=_pcm_float_to_int(b_format_float, effective_bit_depth),
				)
				analysis_channel_index = 0
				channel_format_tag      = "b_format_ambix"
			else:
				analysis_channel_index = None
				channel_format_tag      = "pcm"

			# Convert once; all analyses operate on the same mono float array.
			# analyze_all() shares the pyin computation between spectral and pitch
			# analysis, avoiding running it twice (~200-300 ms saving per recording).
			mono = subsample.analysis.to_mono_float(
				req.audio, effective_bit_depth, channel_index=analysis_channel_index,
			)

			result, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(
				mono,
				self._analysis_params,
				self._cfg.analysis,
			)

			# Warn on clipped live recordings so the user knows to reduce gain.
			# Skipped for file imports (filename_base set) — clipping already occurred.
			if req.filename_base is None and level.peak >= 1.0:
				_log.warning(
					"input clipped (peak %.1f dBFS) — reduce input gain to avoid distortion",
					20.0 * math.log10(level.peak),
				)

			write_result = self._write_audio_file(
				req.audio, req.timestamp, rhythm, result, pitch, timbre, level, band_energy,
				filename_base=req.filename_base,
				sample_rate=req.sample_rate,
				bit_depth=req.bit_depth,
				channel_format=channel_format_tag,
			)

			if self._on_complete is not None and write_result is not None:
				filepath, duration = write_result
				self._on_complete(filepath, result, rhythm, pitch, timbre, level, band_energy, duration, req.audio)

		except Exception as exc:
			_log.error("Failed to process recording: %s — WAV may be intact", exc, exc_info=True)

	def _write_audio_file (
		self,
		audio: numpy.ndarray,
		timestamp: datetime.datetime,
		rhythm: subsample.analysis.RhythmResult,
		result: subsample.analysis.AnalysisResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		level: subsample.analysis.LevelResult,
		band_energy: subsample.analysis.BandEnergyResult,
		filename_base: typing.Optional[str] = None,
		sample_rate: typing.Optional[int] = None,
		bit_depth: typing.Optional[int] = None,
		channel_format: str = "pcm",
	) -> tuple[pathlib.Path, float] | None:

		"""Write a single audio segment to disk and save its analysis sidecar.

		Extension and encoder are chosen from the configured audio_format
		(``wav`` or ``flac``) and the effective bit depth:

		- ``audio_format=wav``  → always ``.wav`` (16/24/32-bit PCM).
		- ``audio_format=flac`` → ``.flac`` for 16/24-bit; ``.wav`` for
		  32-bit inputs (libsndfile's stable FLAC subtypes don't cover
		  PCM_32).  The fallback is per-file: live capture is already
		  validated at config-load time, so this only triggers for file
		  imports whose source is 32-bit.  An INFO log explains the
		  fallback so the user isn't surprised by a mixed library.

		If the target file already exists, it is overwritten (with INFO-
		level logging).  On filesystem error, logs ERROR and returns None.

		Returns (filepath, duration_seconds), or None on write failure.

		Args:
			audio:         PCM samples, shape (n_frames, channels).
			               16-bit: int16. 24-bit: int32 (left-shifted by 8). 32-bit: int32.
			timestamp:     Used to construct the filename when filename_base is None.
			rhythm:        Rhythm analysis computed before this call.
			result:        Spectral analysis metrics computed before this call.
			pitch:         Pitch analysis computed before this call.
			timbre:        Timbral fingerprint computed before this call.
			level:         Peak and RMS amplitude of the recording.
			filename_base: If provided, used as the filename stem instead of the
			               timestamp format. Collision handling still applies.
			sample_rate:   Sample rate for writing. Defaults to config value.
			bit_depth:     Bit depth for writing. Defaults to config value.
			channel_format: "pcm" or "b_format_ambix"; passed through to the
			                sidecar so the player knows whether to apply the
			                ambisonic decoder on playback.
		"""

		# Resolve effective format values; per-request overrides take precedence.
		effective_sample_rate = (
			sample_rate if sample_rate is not None
			else self._cfg.recorder.audio.sample_rate
		)
		effective_bit_depth = (
			bit_depth if bit_depth is not None
			else self._cfg.recorder.audio.bit_depth
		)
		configured_format = self._cfg.recorder.audio.audio_format

		# Per-file format decision.  FLAC's stable subtypes only cover
		# 16/24-bit, so 32-bit inputs fall back to WAV — a silent truncate
		# would lose precision and a hard reject would make mixed-source
		# libraries unusable.
		if configured_format == "flac" and effective_bit_depth == 32:
			effective_format = "wav"
			fallback_message: typing.Optional[str] = (
				"audio_format=flac cannot hold 32-bit; wrote .wav instead "
				"to preserve full precision"
			)
		else:
			effective_format = configured_format
			fallback_message = None

		extension = ".flac" if effective_format == "flac" else ".wav"

		fname_base = (
			filename_base if filename_base is not None
			else _format_filename(timestamp, self._cfg.output.filename_format)
		)
		filepath = self._output_dir / (fname_base + extension)

		# Ensure the array is 2-D (n_frames, channels) before writing
		if audio.ndim == 1:
			audio = audio.reshape(-1, 1)

		n_channels = audio.shape[1]

		# Map our internal dtype conventions to libsndfile subtypes:
		#  - 16-bit int16              → PCM_16
		#  - 24-bit int32 (<< 8)       → PCM_24 (soundfile keeps the upper
		#                                3 bytes; our LSB padding is zero)
		#  - 32-bit int32              → PCM_32
		_SUBTYPE_BY_BIT_DEPTH: dict[int, str] = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
		subtype = _SUBTYPE_BY_BIT_DEPTH[effective_bit_depth]

		# Build the complete file in memory so we can MD5 the exact bytes
		# that hit disk, then do a single atomic write.  soundfile accepts
		# any file-like object for the `file` argument.
		buf = io.BytesIO()
		write_kwargs: dict[str, typing.Any] = {
			"samplerate": effective_sample_rate,
			"subtype":    subtype,
			"format":     "FLAC" if effective_format == "flac" else "WAV",
		}

		if effective_format == "flac":
			write_kwargs["compression_level"] = _FLAC_COMPRESSION_LEVEL

		soundfile.write(buf, audio, **write_kwargs)

		file_bytes = buf.getvalue()
		audio_md5  = hashlib.md5(file_bytes).hexdigest()

		try:
			if filepath.exists():
				_log.info("Overwriting: %s", filepath.name)
			filepath.write_bytes(file_bytes)
		except OSError as exc:
			_log.error("Failed to write %s: %s", filepath.name, exc)
			return None

		if fallback_message is not None:
			_log.info("%s: %s", filepath.name, fallback_message)

		n_frames = audio.shape[0]
		duration = n_frames / effective_sample_rate

		_log.debug("Stored: %s  frames=%d", filepath.name, n_frames)

		# Persist analysis alongside the audio file so future reads (e.g.
		# reference file loading on startup) can skip re-analysis when
		# nothing changes.
		subsample.cache.save_cache(
			audio_path     = filepath,
			audio_md5      = audio_md5,
			params         = self._analysis_params,
			spectral       = result,
			rhythm         = rhythm,
			pitch          = pitch,
			timbre         = timbre,
			duration       = duration,
			level          = level,
			band_energy    = band_energy,
			bit_depth      = effective_bit_depth,
			channels       = n_channels,
			captured_at    = timestamp.isoformat(),
			channel_format = channel_format,
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
