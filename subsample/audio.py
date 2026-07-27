"""Audio device management and file I/O for Subsample.

Handles PyAudio lifecycle, input device enumeration, interactive device
selection, stream creation, and bit-depth-aware sample unpacking.
Also provides read_audio_file() for reading audio files (WAV, FLAC,
AIFF, OGG, MP3/MPEG, and other formats via libsndfile) into the same integer
array format used by the capture pipeline.
Keeps all audio I/O concerns isolated from the rest of the application.
"""

import contextlib
import dataclasses
import logging
import math
import os
import pathlib
import queue
import sys
import threading
import typing
import wave

import numpy
import pyaudio

import subsample.config
import subsample.devices
import subsample.parallelism


_log = logging.getLogger(__name__)


# Type returned by PyAudio for device info mappings
DeviceInfo = typing.Mapping[str, typing.Union[str, int, float]]

# Serialise PyAudio initialisation across threads.
# _suppress_c_stderr() redirects file descriptor 2 at the OS level; concurrent
# calls from different threads corrupt each other's fd state and crash.
_pyaudio_init_lock = threading.Lock()


@dataclasses.dataclass(frozen=True)
class AudioFileInfo:

	"""Metadata and PCM data read from an audio file.

	The audio array uses the same integer format as the capture pipeline:
	  16-bit  →  int16
	  24-bit  →  int32 (left-shifted by 8, matching unpack_audio())
	  32-bit  →  int32
	Shape is (n_frames, channels).
	"""

	audio: numpy.ndarray
	sample_rate: int
	bit_depth: int       # 16, 24, or 32
	channels: int


# Process-wide ceiling for float/double imports, applied by read_audio_file()
# whenever a caller doesn't pass one of its own.  cli.py sets it from the resolved
# config at startup; it seeds from AudioConfig's own declared default (rather than
# a literal repeated here) so a caller that never wires it — a script, a test —
# still reads audio the same way a default-config player does.  That consistency
# is not cosmetic: an unwired reader that analyses a hot float sample writes a
# sidecar the wired player will later trust, so a mismatch here poisons the cache.
_FLOAT_IMPORT_CEILING_DBFS: typing.Optional[float] = subsample.config.AudioConfig.float_import_ceiling_dbfs


class _Unset:

	"""Sentinel type: the caller named no ceiling, so the configured one applies.

	Distinct from None, which is a caller explicitly asking for the historical
	hard-clip whatever happens to be configured."""


_UNSET = _Unset()


def set_float_import_ceiling (dbfs: typing.Optional[float]) -> None:

	"""Set the process-wide float/double import ceiling.  Called from cli.py once
	the config is loaded.

	Every path that reads an audio file — the library loader, the analysis cache,
	the watcher, OSC import — funnels through read_audio_file(), so wiring the
	value here applies it to all of them consistently.  That consistency is the
	point: a sample's analysis and its playback audio are two separate reads of
	the same file, and they must agree, or the stored fingerprint (and the RMS
	that drives loudness normalisation) would describe audio that isn't what
	plays.  None restores the historical hard-clip."""

	global _FLOAT_IMPORT_CEILING_DBFS
	_FLOAT_IMPORT_CEILING_DBFS = dbfs


def scale_float_to_ceiling (
	data: numpy.ndarray,
	ceiling_dbfs: typing.Optional[float],
) -> numpy.ndarray:

	"""Scale a float array down so its peak sits at ``ceiling_dbfs``, if it exceeds it.

	Mirrors the scale-to-fit ``read_audio_file`` applies to hot float/double
	sources, exposed for the sidecar-writing scripts (import_samples,
	analyze_file) that read float PCM directly rather than through
	``read_audio_file`` — so the audio they write and the sidecar they analyse
	both describe the same, un-clipped signal.  Returns the array unchanged when
	``ceiling_dbfs`` is None, the array is empty, or the peak already fits.

	Non-finite samples are scrubbed first, matching ``read_audio_file``: a NaN
	makes the peak probe NaN (so ``peak > ceiling`` is False and no scaling
	happens at all) and an inf makes the gain zero, silencing the file.
	"""

	if ceiling_dbfs is None or data.size == 0:
		return data

	if not numpy.all(numpy.isfinite(data)):
		data = numpy.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

	peak = float(numpy.max(numpy.abs(data)))
	ceiling_lin = 10.0 ** (ceiling_dbfs / 20.0)

	if peak > ceiling_lin:
		scaled: numpy.ndarray = data * (ceiling_lin / peak)
		return scaled

	return data


def read_audio_file (
	path: pathlib.Path,
	float_ceiling_dbfs: typing.Union[float, None, _Unset] = _UNSET,
) -> AudioFileInfo:

	"""Read an audio file and return its data with format metadata.

	Tries the stdlib wave module first (WAV files).  If that fails, falls
	back to soundfile (libsndfile) which supports WAV, FLAC, AIFF, OGG,
	MP3/MPEG, and many other formats.  The reported ``bit_depth`` follows the
	source subtype — 32 for FLOAT/DOUBLE and the 32-bit PCM/ALAC variants, 24
	for the 24-bit variants, and 16 for 16-bit PCM and lossy formats — so the
	returned int array matches the source resolution rather than always being
	truncated to 16-bit.

	``float_ceiling_dbfs`` guards the one format that can legitimately exceed
	0 dBFS: 32-bit float / 64-bit double.  Those have no hard full-scale limit,
	but the internal integer array does, so peaks above full scale would hard-
	clip on conversion.  When set, a float/double source whose peak exceeds the
	ceiling is scaled down as a whole (preserving relative dynamics) so its peak
	lands at the ceiling — no clipping, no lost transients.  Integer-PCM sources
	are never scaled (a full-scale int sample is the format ceiling and may
	already be clipped at source).  Omitted, it follows the process-wide ceiling
	set by set_float_import_ceiling(); None forces the historical hard-clip.

	Args:
		path:               Path to the audio file.
		float_ceiling_dbfs: Ceiling (dBFS, <= 0) for float/double imports.  Omit to
		                    follow the configured process-wide ceiling; None disables
		                    the scale-to-fit and hard-clips as before.

	Returns:
		AudioFileInfo with audio array, sample_rate, bit_depth, and channels.

	Raises:
		ValueError: If the file format is not supported by either reader.
		OSError:    If the file cannot be opened or read.
	"""

	# An explicit argument always wins (including None, meaning "clip as before");
	# only an absent one falls back to the configured process-wide ceiling.
	ceiling_dbfs = (
		_FLOAT_IMPORT_CEILING_DBFS if isinstance(float_ceiling_dbfs, _Unset)
		else float_ceiling_dbfs
	)

	# Fast path: WAV via stdlib (avoids soundfile overhead and preserves
	# the original bit depth for 16/24/32-bit WAV files).
	try:
		with wave.open(str(path), "rb") as wf:
			channels     = wf.getnchannels()
			sample_width = wf.getsampwidth()
			sample_rate  = wf.getframerate()
			raw_bytes    = wf.readframes(wf.getnframes())

		bit_depth = sample_width * 8
		audio = unpack_audio(raw_bytes, bit_depth, channels)

		return AudioFileInfo(
			audio       = audio,
			sample_rate = sample_rate,
			bit_depth   = bit_depth,
			channels    = channels,
		)

	except wave.Error:
		pass  # Not a WAV file — fall through to soundfile.

	except EOFError:
		# A zero-byte or header-truncated WAV: wave.open/readframes raises
		# EOFError (not wave.Error).  Fall through to soundfile, which raises a
		# clean LibsndfileError→ValueError the callers already catch — so one
		# malformed file in a scanned directory becomes a skip, not an
		# EOFError that escapes read_audio_file and aborts the whole load.
		pass

	except ValueError:
		# A real WAV, but at a width unpack_audio doesn't handle (e.g. 8-bit
		# PCM_U8) — soundfile reads those fine, so fall through rather than
		# rejecting a readable file.
		pass

	# Fallback: soundfile (libsndfile) handles FLAC, AIFF, OGG, MP3, and also
	# WAV variants the stdlib wave module rejects — notably WAVE_FORMAT_IEEE_FLOAT
	# (32-bit float WAVs from DAWs and field recorders).
	#
	# Pick an input dtype that preserves the file's native bit depth:
	#  - PCM_24 / ALAC_24 files are decoded as int32 with the 24-bit value in
	#    the upper 3 bytes (matches unpack_audio's 24-bit convention); the
	#    bit_depth we return is 24 so downstream code treats it correctly.
	#  - PCM_32 / ALAC_32 files are decoded as int32 natively (full scale).
	#  - FLOAT (32-bit) and DOUBLE (64-bit) files are decoded as float first
	#    and scaled to int32 by hand: soundfile's dtype="int32" on a float
	#    source does a direct cast (1.0 → 1, not 1.0 → 2**31), silently
	#    collapsing the signal to near-silence.  Scaling here keeps the
	#    downstream int32 pipeline happy without any float-aware special
	#    case further on.
	#  - Everything else (PCM_16, ALAC_16, ULAW, ALAW, VORBIS, OPUS,
	#    MP3/MPEG, etc.) is decoded as int16 — these are either natively
	#    16-bit or lossy/companded formats whose effective precision sits
	#    comfortably inside int16.
	# Without this, a 24-bit FLAC written by the recorder and later read
	# back would be silently truncated to 16-bit, a 32-bit float WAV
	# would be silently flattened to near-zero, a 64-bit double-precision
	# WAV would do the same, and a 24-bit ALAC file would lose its upper
	# 8 bits of precision.
	try:
		import soundfile

		sf_info = soundfile.info(str(path))
		subtype = (sf_info.subtype or "").upper()

		if subtype in ("FLOAT", "DOUBLE"):
			# Read at native float precision.  DOUBLE → float64 preserves the
			# extra mantissa bits during scaling; FLOAT → float32 is enough.
			# Either way the scale-and-clamp logic below is identical.
			float_dtype = "float32" if subtype == "FLOAT" else "float64"
			float_data, sample_rate = soundfile.read(str(path), dtype=float_dtype, always_2d=True)

			# Scrub non-finite samples before anything reads them.  numpy.clip
			# passes NaN straight through and .astype(int32) then maps it to
			# INT32_MIN — a full-scale negative click on playback, from a file
			# that merely had one bad sample.  ±inf is worse: it makes the peak
			# probe below inf, so the ceiling gain is 0 and the whole file goes
			# silent, then log10(0) raises inside the broad handler and the user
			# is told their format is unsupported.  A NaN is silence and an inf
			# is a rail, which is the closest honest reading of a corrupt sample.
			if float_data.size and not numpy.all(numpy.isfinite(float_data)):
				n_bad = int(numpy.count_nonzero(~numpy.isfinite(float_data)))
				_log.warning(
					"%s: %d non-finite sample(s) (NaN/inf) replaced with silence/full scale",
					path.name, n_bad,
				)
				float_data = numpy.nan_to_num(
					float_data, nan=0.0, posinf=1.0, neginf=-1.0,
				)

			# Clip-safe conversion: float/double have no hard 0 dBFS ceiling but the
			# int32 array does.  When a ceiling is configured and this source exceeds
			# it, scale the whole file down (one gain, so inter-hit dynamics survive)
			# so its peak lands at the ceiling — losslessly, rather than hard-clipping
			# the loudest transients on the cast below.  Playback re-normalises level
			# per sample, so the attenuation is inaudible; it only prevents data loss.
			if ceiling_dbfs is not None and float_data.size:
				peak = float(numpy.max(numpy.abs(float_data)))
				ceiling_lin = 10.0 ** (ceiling_dbfs / 20.0)
				if peak > ceiling_lin:
					gain = ceiling_lin / peak
					float_data = float_data * gain
					_log.info(
						"%s: float source peaks at %+.1f dBFS; scaled down %.1f dB to a "
						"%.1f dBFS ceiling so integer conversion does not clip",
						path.name, 20.0 * math.log10(peak),
						-20.0 * math.log10(gain), ceiling_dbfs,
					)

			# Scale [-1.0, 1.0] → int32 full scale.  Multiply in float64 and
			# clip *after* multiplication so a stray sample > 1.0 (rare in
			# headroom-mixed files) doesn't wrap on cast; the float32 mantissa
			# can't represent 2**31 - 1 exactly, so clamping with int32's
			# actual limits avoids both wrap and silently-altered peak values.
			int32_min = float(numpy.iinfo(numpy.int32).min)
			int32_max = float(numpy.iinfo(numpy.int32).max)
			# copy=False avoids a redundant copy when float_data is already
			# float64 (DOUBLE); the FLOAT path still upcasts so the multiply
			# happens in float64 regardless of numpy's scalar-promotion rules.
			scaled = numpy.clip(float_data.astype(numpy.float64, copy=False) * float(2**31), int32_min, int32_max)
			audio = numpy.ascontiguousarray(scaled.astype(numpy.int32))
			bit_depth = 32

			return AudioFileInfo(
				audio       = audio,
				sample_rate = sample_rate,
				bit_depth   = bit_depth,
				channels    = audio.shape[1],
			)

		if subtype in ("PCM_24", "PCM_32", "ALAC_24", "ALAC_32"):
			read_dtype = "int32"
			bit_depth  = 32 if subtype in ("PCM_32", "ALAC_32") else 24
		else:
			read_dtype = "int16"
			bit_depth  = 16

		data, sample_rate = soundfile.read(str(path), dtype=read_dtype, always_2d=True)
		audio = numpy.ascontiguousarray(data)

		return AudioFileInfo(
			audio       = audio,
			sample_rate = sample_rate,
			bit_depth   = bit_depth,
			channels    = audio.shape[1],
		)

	except Exception as exc:
		supported = "WAV, FLAC, AIFF, OGG, MP3/MPEG"

		raise ValueError(
			f"Unsupported audio format: {path.name} — "
			f"convert to a supported format ({supported}) and try again"
		) from exc


@contextlib.contextmanager
def _suppress_c_stderr () -> typing.Generator[None, None, None]:

	"""Redirect C-level stderr to /dev/null for the duration of the block.

	Python's sys.stderr redirection does not affect C library output; this
	operates at the file-descriptor level so it catches ALSA/JACK noise too.
	Restores the original stderr even if an exception is raised.
	fd_null is opened before fd_saved so that if os.dup(2) fails, fd_null
	is closed before re-raising and no file descriptor leaks.
	"""

	# Open fd_null first. If os.dup(2) then fails, we close fd_null before
	# re-raising so no file descriptor leaks.
	fd_null = os.open(os.devnull, os.O_WRONLY)
	try:
		fd_saved = os.dup(2)
	except OSError:
		os.close(fd_null)
		raise

	try:
		os.dup2(fd_null, 2)
		yield
	finally:
		os.dup2(fd_saved, 2)
		os.close(fd_null)
		os.close(fd_saved)


def create_pyaudio () -> pyaudio.PyAudio:

	"""Create a PyAudio instance, suppressing ALSA/JACK diagnostic noise.

	Raises:
		OSError: If PortAudio cannot initialise — most often because no sound
		         server is running.  Re-raised with that hint attached, since
		         the underlying diagnostic goes to the fd 2 this suppresses.
	"""

	# PortAudio spawns its callback threads inside C, where
	# threading.active_count() cannot see them, so tell the fork guard that
	# forking is no longer safe from this process.
	subsample.parallelism.note_native_subsystem_started()

	with _pyaudio_init_lock:
		try:
			with _suppress_c_stderr():
				return pyaudio.PyAudio()

		except OSError as exc:
			# The ALSA/JACK line that says WHY went to the suppressed fd 2, so
			# without this the user gets a bare "[Errno -9999] Unanticipated
			# host error" and nothing to act on.
			raise OSError(
				f"Could not initialise the audio backend ({exc}).  Is a sound "
				f"server running?  Subsample needs PipeWire, PulseAudio or JACK "
				f"— on a headless box try `pw-jack subsample`."
			) from exc


def unpack_audio (raw_bytes: bytes, bit_depth: int, channels: int) -> numpy.ndarray:

	"""Convert raw bytes from a PyAudio stream into a NumPy integer array.

	Returns shape (n_frames, channels). For 24-bit audio, samples are stored
	as int32 with the 24-bit value occupying the upper 3 bytes (left-shifted
	by 8). This avoids NumPy's lack of a native int24 type while preserving
	full precision. ``soundfile.write(..., subtype="PCM_24")`` handles the
	24-bit packing when this format is written back to disk.

	Args:
		raw_bytes: Raw bytes from stream.read().
		bit_depth: 16, 24, or 32.
		channels:  Number of audio channels.

	Raises:
		ValueError: For unsupported bit depths.
	"""

	if bit_depth == 16:
		return numpy.frombuffer(raw_bytes, dtype=numpy.int16).reshape(-1, channels)

	if bit_depth == 24:
		# The zero-byte padding trick (zero at column 0, audio bytes at 1–3)
		# relies on little-endian byte order: the zero lands at the LSB of the
		# int32, producing a left-shift by 8. On a big-endian machine the zero
		# would be at the MSB, giving completely wrong values.
		if sys.byteorder != "little":
			raise RuntimeError(
				"24-bit audio unpacking requires a little-endian system; "
				f"this machine is {sys.byteorder}-endian."
			)
		# Reshape to (n_samples, 3) byte view, then pad each sample to 4 bytes
		# by inserting a zero at the LSB position — equivalent to << 8.
		raw = numpy.frombuffer(raw_bytes, dtype=numpy.uint8).reshape(-1, 3)
		n_samples = raw.shape[0]
		padded = numpy.zeros((n_samples, 4), dtype=numpy.uint8)
		padded[:, 1:] = raw
		return padded.view(numpy.int32).reshape(-1, channels)

	if bit_depth == 32:
		return numpy.frombuffer(raw_bytes, dtype=numpy.int32).reshape(-1, channels)

	raise ValueError(f"Unsupported bit depth {bit_depth}. Supported: 16, 24, 32")


class AudioReader:

	"""Reads audio chunks from a PyAudio stream using PortAudio's callback mode.

	Callback mode delivers audio directly from PortAudio's high-priority audio
	thread, bypassing the internal ring buffer used in blocking mode. This is
	more reliable for USB audio devices, which use isochronous USB transfers
	(no retransmit) and are sensitive to any timing jitter in the delivery path.

	The callback does minimal work — just queue.put_nowait(raw_bytes) — so the
	audio thread is never blocked by Python processing. Unpacking to numpy
	happens in the main thread via read().

	The stream is owned by AudioReader and is opened and closed internally.

	Usage:
		reader = AudioReader(pa, device_index, audio_cfg)
		chunk = reader.read()   # blocks until next chunk is ready
		reader.stop()           # stops and closes the stream
	"""

	_QUEUE_MAX: int = 64  # chunks of headroom (≈0.74s at 44100 Hz with the default 512-frame chunk; scales with buffer_frames)

	def __init__ (
		self,
		pa: pyaudio.PyAudio,
		device_index: int,
		audio_cfg: subsample.config.AudioConfig,
	) -> None:

		"""Open the audio stream in callback mode.

		Args:
			pa:          PyAudio instance.
			device_index: Index of the input device to use.
			audio_cfg:   Audio configuration (sample rate, bit depth, etc.).
		"""

		# channels must be resolved to a concrete int before AudioReader is
		# constructed.  Callers are responsible for auto-detecting via
		# get_device_channels() when AudioConfig.channels is None.
		assert audio_cfg.channels is not None, (
			"AudioConfig.channels must be resolved before opening an AudioReader. "
			"Call get_device_channels() to auto-detect from the selected device."
		)

		self._bit_depth = audio_cfg.bit_depth
		self._queue: queue.Queue[bytes] = queue.Queue(maxsize=self._QUEUE_MAX)
		self._overflow_count: int = 0

		# Input channel routing: when specific physical inputs are selected,
		# we open the stream with enough channels to cover the highest index,
		# then extract only the requested columns in read().
		self._input_map: typing.Optional[tuple[int, ...]] = audio_cfg.input

		if self._input_map is not None:
			stream_channels = max(self._input_map) + 1
		else:
			stream_channels = audio_cfg.channels

		self._stream_channels = stream_channels
		self._channels = audio_cfg.channels

		self._stream = pa.open(
			format=get_pyaudio_format(audio_cfg.bit_depth),
			channels=stream_channels,
			rate=audio_cfg.sample_rate,
			input=True,
			input_device_index=device_index,
			frames_per_buffer=audio_cfg.buffer_frames,
			stream_callback=self._callback,
		)

	def read (self, timeout: typing.Optional[float] = None) -> typing.Optional[numpy.ndarray]:

		"""Return the next audio chunk, or None if timeout elapses.

		Unpacks raw bytes from the callback queue into a numpy integer array.
		When timeout is None (default) the call blocks indefinitely until a
		chunk arrives — identical to the original behaviour.

		Args:
			timeout: Maximum seconds to wait. None = block forever.

		Returns:
			Array of shape (buffer_frames, channels), integer dtype, or None on timeout.
		"""

		try:
			raw_bytes = self._queue.get(timeout=timeout)
		except queue.Empty:
			return None

		chunk = unpack_audio(raw_bytes, self._bit_depth, self._stream_channels)

		# Select only the requested physical inputs when input routing is active.
		if self._input_map is not None:
			chunk = chunk[:, list(self._input_map)]

		return chunk

	@property
	def overflow_count (self) -> int:

		"""Number of overflow/underflow events reported by PortAudio."""

		return self._overflow_count

	def stop (self) -> None:

		"""Stop the stream and release it."""

		self._stream.stop_stream()
		self._stream.close()

	def _callback (
		self,
		in_data: typing.Optional[bytes],
		frame_count: int,
		time_info: typing.Mapping[str, float],
		status_flags: int,
	) -> tuple[typing.Optional[bytes], int]:

		"""PortAudio callback — called from PortAudio's audio thread.

		Must return quickly and must never block. Overflow/underflow events
		are counted; the chunk is dropped rather than blocking if the queue
		is full (which would stall the audio thread).
		"""

		if status_flags:
			# status_flags is a bitmask: paInputOverflow=0x2, paOutputUnderflow=0x4.
			# Any non-zero value means PortAudio discarded or lost data.
			self._overflow_count += 1

		if in_data is not None:
			try:
				self._queue.put_nowait(in_data)
			except queue.Full:
				# Main loop has fallen far behind — drop the chunk.
				# Dropping one chunk is less harmful than stalling the audio thread.
				pass

		return (None, pyaudio.paContinue)


def get_device_channels (pa: pyaudio.PyAudio, device_index: int) -> int:

	"""Return the maximum number of input channels reported by the device.

	Used to auto-detect the channel count when `recorder.audio.channels` is
	omitted from config.  The value comes from PortAudio's `maxInputChannels`
	field, which reflects the hardware capability — e.g. 2 for a stereo USB
	microphone, 1 for a mono headset.

	Raises:
		ValueError: If the device reports zero input channels (output-only device).
	"""

	info = pa.get_device_info_by_index(device_index)
	ch = int(info["maxInputChannels"])

	if ch <= 0:
		raise ValueError(
			f"Device {info['name']!r} reports no input channels — "
			"it may be an output-only device."
		)

	return ch


def get_output_device_channels (pa: pyaudio.PyAudio, device_index: int) -> int:

	"""Return the maximum number of output channels reported by the device.

	The player logs this and validates `player.audio.channels` against it, so
	an over-large request fails with a clear, actionable message instead of a
	cryptic PortAudio ``Invalid number of channels`` error at stream-open time.
	The value comes from PortAudio's `maxOutputChannels` field (the mirror of
	`maxInputChannels` used by get_device_channels for the recorder).

	Raises:
		ValueError: If the device reports zero output channels (input-only device).
	"""

	info = pa.get_device_info_by_index(device_index)
	ch = int(info["maxOutputChannels"])

	if ch <= 0:
		raise ValueError(
			f"Device {info['name']!r} reports no output channels — "
			"it may be an input-only device."
		)

	return ch


def select_input_channels (device_name: str, max_channels: int) -> tuple[int, ...]:

	"""Prompt the user to choose which input channels to record from.

	Called when both channels and input are omitted in config and the device
	reports 3+ input channels (too many to silently use all).

	Args:
		device_name:  Display name of the audio device.
		max_channels: Total number of input channels available.

	Returns:
		Tuple of 0-indexed channel indices.

	Raises:
		ValueError: If no valid channels are entered.
	"""

	print(f'\nAudio input device "{device_name}" has {max_channels} input channels.')
	print("Which channels to record? (1-indexed, comma-separated, e.g. 1,2)")

	raw = input("> ").strip()

	if not raw:
		raise ValueError("No input channels selected")

	parts = [p.strip() for p in raw.split(",")]
	selected: list[int] = []

	for part in parts:
		try:
			n = int(part)
		except ValueError:
			raise ValueError(f"Invalid channel number: {part!r}") from None

		if n < 1 or n > max_channels:
			raise ValueError(
				f"Channel {n} is out of range (device has inputs 1-{max_channels})"
			)

		if n - 1 in selected:
			raise ValueError(f"Duplicate channel: {n}")

		selected.append(n - 1)

	print(f"Recording {len(selected)} channel(s) from input(s) {', '.join(str(s + 1) for s in selected)}")

	return tuple(selected)


def _list_devices (pa: pyaudio.PyAudio, direction: str) -> list[DeviceInfo]:

	"""Return all devices with at least one channel in *direction* ("input"/"output")."""

	key = "maxInputChannels" if direction == "input" else "maxOutputChannels"
	devices: list[DeviceInfo] = []

	for i in range(pa.get_device_count()):
		info: DeviceInfo = pa.get_device_info_by_index(i)

		if int(info[key]) > 0:
			devices.append(info)

	return devices


def _find_device_by_name (pa: pyaudio.PyAudio, name: str, direction: str) -> int:

	"""Return the index of the *direction* device matching *name*.

	*name* is a glob (see subsample.devices): case-insensitive, implicit ``*`` at
	each end, so "Samson" still matches "Samson Go Mic: USB Audio (hw:1,0)" and
	"SC-U: USB Audio (hw:*,0)" survives the card index being renumbered when an
	unrelated interface is plugged in.

	Several matches prompt over just those devices rather than picking the first
	silently — resolving ambiguity by enumeration order is the same surprise the
	volatile numbering causes.

	Raises:
		ValueError: If nothing matches, listing the available device names; or
		            if several match with no terminal to choose on.
	"""

	devices = _list_devices(pa, direction)
	names = [str(device["name"]) for device in devices]
	matches = subsample.devices.match_device_names(name, names)

	if not matches:
		raise ValueError(
			f"No {direction} device matching {name!r} found.\nAvailable devices:\n"
			f"{subsample.devices.format_device_list(names)}"
		)

	if len(matches) == 1:
		return int(devices[matches[0]]["index"])

	matched_devices = [devices[index] for index in matches]

	if not subsample.devices.can_prompt():
		raise ValueError(
			subsample.devices.ambiguous_pattern_message(
				name, [names[index] for index in matches], f"audio {direction}",
			)
		)

	print(f"{len(matches)} audio {direction} devices match {name!r}:")

	return _select_device(
		matched_devices, direction,
		f"No {direction} device matching {name!r} found.",
	)


def _select_device (devices: list[DeviceInfo], direction: str, empty_hint: str) -> int:

	"""Return the device index to use, prompting the user if there are multiple.

	Auto-selects when only one device is available; raises ValueError with
	*empty_hint* when the list is empty.

	Without a terminal there is nobody to prompt: raise a message naming the
	candidates instead of blocking on input() forever, which is how this
	presented when run from a service manager or CI.
	"""

	if not devices:
		raise ValueError(empty_hint)

	if len(devices) == 1:
		name = devices[0]["name"]
		print(f"Using audio {direction}: {name}")
		return int(devices[0]["index"])

	names = [str(device["name"]) for device in devices]

	if not subsample.devices.can_prompt():
		raise ValueError(
			f"{len(devices)} audio {direction} devices are available and there is "
			f"no terminal to choose on:\n"
			f"{subsample.devices.format_device_list(names)}\n"
			f"Set the device in config.yaml — a wildcard covers a changing index "
			f"(e.g. 'SC-U: USB Audio (hw:*,0)')."
		)

	# Multiple devices — let the user choose
	print(f"Available audio {direction} devices:")
	for i, device in enumerate(devices):
		name = device["name"]
		rate = int(device["defaultSampleRate"])
		print(f"  [{i}] {name}  (default {rate} Hz)")

	while True:
		raw = input(f"Select device [0–{len(devices) - 1}]: ").strip()

		try:
			choice = int(raw)
		except ValueError:
			print("  Please enter a number.")
			continue

		if 0 <= choice < len(devices):
			return int(devices[choice]["index"])

		print(f"  Please enter a number between 0 and {len(devices) - 1}.")


def list_input_devices (pa: pyaudio.PyAudio) -> list[DeviceInfo]:

	"""Return all audio devices that have at least one input channel."""

	return _list_devices(pa, "input")


def find_device_by_name (pa: pyaudio.PyAudio, name: str) -> int:

	"""Return the index of the first input device whose name contains *name*.

	Matching is case-insensitive substring search (see _find_device_by_name).
	"""

	return _find_device_by_name(pa, name, "input")


def select_device (devices: list[DeviceInfo]) -> int:

	"""Return the input device index to use, prompting the user if there are multiple."""

	return _select_device(
		devices, "input",
		"No audio input devices found. Check that a microphone is connected.",
	)


def list_output_devices (pa: pyaudio.PyAudio) -> list[DeviceInfo]:

	"""Return all audio devices that have at least one output channel."""

	return _list_devices(pa, "output")


def find_output_device_by_name (pa: pyaudio.PyAudio, name: str) -> int:

	"""Return the index of the first output device whose name contains *name*.

	Matching is case-insensitive substring search (see _find_device_by_name).
	"""

	return _find_device_by_name(pa, name, "output")


def select_output_device (devices: list[DeviceInfo]) -> int:

	"""Return the output device index to use, prompting the user if there are multiple."""

	return _select_device(devices, "output", "No audio output devices found.")


def get_pyaudio_format (bit_depth: int) -> int:

	"""Map a bit depth integer to the corresponding PyAudio format constant.

	Raises ValueError for unsupported bit depths.
	"""

	formats: dict[int, int] = {
		16: pyaudio.paInt16,
		24: pyaudio.paInt24,
		32: pyaudio.paInt32,
	}

	if bit_depth not in formats:
		supported = ", ".join(str(b) for b in sorted(formats.keys()))
		raise ValueError(f"Unsupported bit depth {bit_depth}. Supported: {supported}")

	return formats[bit_depth]


def float32_to_pcm_bytes (audio: numpy.ndarray, bit_depth: int) -> bytes:

	"""Convert a float32 audio array to PCM bytes for the given output bit depth.

	Mirrors the bit-depth-aware byte layout expected by PortAudio's paInt16,
	paInt24, and paInt32 formats:

	  16-bit: 2 bytes per sample, signed int16, little-endian.
	  24-bit: 3 bytes per sample, signed int24, little-endian.  The 3 least-
	          significant bytes of an int32 encode the value correctly.
	  32-bit: 4 bytes per sample, signed int32, little-endian.

	Args:
		audio:     float32 array, values in [-1.0, 1.0]. Any shape — the array
		           is flattened in C (row-major) order, which produces the
		           interleaved L/R layout PortAudio expects.
		           ⓒ Values are NOT clipped here (only the 32-bit path clamps
		           for arithmetic-overflow reasons): the caller contract is
		           pre-clipped input, and the audio callback hard-clips the
		           mix to [-1.0, 1.0] immediately before calling this.
		bit_depth: 16, 24, or 32.

	Returns:
		Raw bytes suitable for returning from a PyAudio output callback.

	Raises:
		ValueError: If bit_depth is not 16, 24, or 32.
	"""

	flat = audio.flatten()

	if bit_depth == 16:
		return (flat * 32767.0).astype(numpy.int16).tobytes()

	if bit_depth == 24:
		# Scale to signed 24-bit range, store in int32, then extract the 3
		# least-significant bytes per sample.  This is the inverse of
		# unpack_audio()'s 24-bit path, which likewise guards byte order.
		scaled = (flat * 8388607.0).astype(numpy.int32)
		raw = scaled.view(numpy.uint8).reshape(-1, 4)

		# The 3 LSBs are the first three bytes on a little-endian host and the
		# last three on a big-endian one.  (unpack_audio's 24-bit READ path
		# refuses big-endian hosts outright, so the big-endian branch below
		# is defensive symmetry, not a tested round-trip.)
		if sys.byteorder == "little":
			return raw[:, :3].tobytes()

		return raw[:, 1:4].tobytes()

	if bit_depth == 32:
		# Clip in float64 before the int32 cast: in float32 the scale factor
		# 2147483647 rounds up to 2**31, which wraps to full-scale NEGATIVE on
		# cast — flipping every peak >= ~0.99999996 to a loud click.  float64
		# represents the int32 max exactly, so the clip lands on it cleanly.
		scaled64 = numpy.clip(
			flat.astype(numpy.float64) * 2147483647.0, -2147483648.0, 2147483647.0,
		)
		return scaled64.astype(numpy.int32).tobytes()

	raise ValueError(f"Unsupported bit depth {bit_depth}. Supported: 16, 24, 32")
