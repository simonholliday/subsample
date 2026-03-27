"""Audio device management and file I/O for Subsample.

Handles PyAudio lifecycle, input device enumeration, interactive device
selection, stream creation, and bit-depth-aware sample unpacking.
Also provides read_audio_file() for reading WAV files into the same
integer array format used by the capture pipeline.
Keeps all audio I/O concerns isolated from the rest of the application.
"""

import contextlib
import dataclasses
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


def read_audio_file (path: pathlib.Path) -> AudioFileInfo:

	"""Read a WAV file and return its audio data with format metadata.

	Reads using the wave module for header parsing and delegates to
	unpack_audio() for dtype conversion (including the 24-bit left-shift),
	so the returned array is identical in format to what AudioReader produces.

	Args:
		path: Path to the WAV file.

	Returns:
		AudioFileInfo with audio array, sample_rate, bit_depth, and channels.

	Raises:
		wave.Error: If the file is not a valid WAV file.
		OSError:    If the file cannot be opened or read.
		ValueError: If the bit depth is not 16, 24, or 32.
	"""

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

	"""Create a PyAudio instance, suppressing ALSA/JACK diagnostic noise."""

	with _pyaudio_init_lock:
		with _suppress_c_stderr():
			return pyaudio.PyAudio()


def unpack_audio (raw_bytes: bytes, bit_depth: int, channels: int) -> numpy.ndarray:

	"""Convert raw bytes from a PyAudio stream into a NumPy integer array.

	Returns shape (n_frames, channels). For 24-bit audio, samples are stored
	as int32 with the 24-bit value occupying the upper 3 bytes (left-shifted
	by 8). This avoids NumPy's lack of a native int24 type while preserving
	full precision. recorder._pack_int24() reverses this when writing to disk.

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

	_QUEUE_MAX: int = 64  # ~0.74s of headroom at 44100 Hz / 512 frames per chunk

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
		self._channels = audio_cfg.channels
		self._queue: queue.Queue[bytes] = queue.Queue(maxsize=self._QUEUE_MAX)
		self._overflow_count: int = 0

		self._stream = pa.open(
			format=get_pyaudio_format(audio_cfg.bit_depth),
			channels=audio_cfg.channels,
			rate=audio_cfg.sample_rate,
			input=True,
			input_device_index=device_index,
			frames_per_buffer=audio_cfg.chunk_size,
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
			Array of shape (chunk_size, channels), integer dtype, or None on timeout.
		"""

		try:
			raw_bytes = self._queue.get(timeout=timeout)
		except queue.Empty:
			return None

		return unpack_audio(raw_bytes, self._bit_depth, self._channels)

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


def list_input_devices (pa: pyaudio.PyAudio) -> list[DeviceInfo]:

	"""Return all audio devices that have at least one input channel."""

	devices: list[DeviceInfo] = []

	for i in range(pa.get_device_count()):
		info: DeviceInfo = pa.get_device_info_by_index(i)

		if int(info["maxInputChannels"]) > 0:
			devices.append(info)

	return devices


def find_device_by_name (pa: pyaudio.PyAudio, name: str) -> int:

	"""Return the index of the first input device whose name contains *name*.

	Matching is case-insensitive substring search, so "Samson" matches
	"Samson Go Mic: USB Audio (hw:1,0)".

	Args:
		pa:   Active PyAudio instance.
		name: Substring to match against device names.

	Returns:
		Device index of the first matching input device.

	Raises:
		ValueError: If no input device name contains *name*, with a list of
		            available device names to help the user correct the config.
	"""

	devices = list_input_devices(pa)
	name_lower = name.lower()

	for dev in devices:
		if name_lower in str(dev["name"]).lower():
			return int(dev["index"])

	available = "\n".join(f"  {d['name']}" for d in devices) or "  (none)"
	raise ValueError(
		f"No input device matching {name!r} found.\nAvailable devices:\n{available}"
	)


def select_device (devices: list[DeviceInfo]) -> int:

	"""Return the device index to use, prompting the user if there are multiple.

	Auto-selects when only one input device is available.
	Raises ValueError if no input devices are found.
	"""

	if not devices:
		raise ValueError("No audio input devices found. Check that a microphone is connected.")

	if len(devices) == 1:
		name = devices[0]["name"]
		print(f"Using audio input: {name}")
		return int(devices[0]["index"])

	# Multiple devices — let the user choose
	print("Available audio input devices:")
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


def list_output_devices (pa: pyaudio.PyAudio) -> list[DeviceInfo]:

	"""Return all audio devices that have at least one output channel."""

	devices: list[DeviceInfo] = []

	for i in range(pa.get_device_count()):
		info: DeviceInfo = pa.get_device_info_by_index(i)

		if int(info["maxOutputChannels"]) > 0:
			devices.append(info)

	return devices


def find_output_device_by_name (pa: pyaudio.PyAudio, name: str) -> int:

	"""Return the index of the first output device whose name contains *name*.

	Matching is case-insensitive substring search.

	Raises:
		ValueError: If no output device name contains *name*.
	"""

	devices = list_output_devices(pa)
	name_lower = name.lower()

	for device in devices:
		if name_lower in str(device["name"]).lower():
			return int(device["index"])

	available = "\n".join(f"  {d['name']}" for d in devices) or "  (none)"
	raise ValueError(
		f"No output device matching {name!r} found.\nAvailable devices:\n{available}"
	)


def select_output_device (devices: list[DeviceInfo]) -> int:

	"""Return the output device index to use, prompting the user if there are multiple.

	Auto-selects when only one output device is available.
	Raises ValueError if no output devices are found.
	"""

	if not devices:
		raise ValueError("No audio output devices found.")

	if len(devices) == 1:
		print(f"Using audio output: {devices[0]['name']}")
		return int(devices[0]["index"])

	print("Available audio output devices:")
	for i, device in enumerate(devices):
		rate = int(device["defaultSampleRate"])
		print(f"  [{i}] {device['name']}  (default {rate} Hz)")

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
		# least-significant bytes (little-endian order) per sample.
		# This is the inverse of unpack_audio()'s 24-bit path.
		scaled = (flat * 8388607.0).astype(numpy.int32)
		raw = scaled.view(numpy.uint8).reshape(-1, 4)
		return raw[:, :3].tobytes()

	if bit_depth == 32:
		return (flat * 2147483647.0).astype(numpy.int32).tobytes()

	raise ValueError(f"Unsupported bit depth {bit_depth}. Supported: 16, 24, 32")
