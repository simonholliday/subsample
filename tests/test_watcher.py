"""Tests for subsample.watcher.InstrumentWatcher.

The watcher runs a background watchdog observer thread, so tests use
threading.Event with a generous timeout to detect callback invocations
asynchronously. The 1-second debounce window means each test that
expects a callback must wait up to ~3 seconds for it to fire.

Audio-file-path tests override the module-level timing constants to
keep the test suite fast: debounce, grace, and stability windows are
all shortened to sub-second values.
"""

import pathlib
import threading
import time
import typing

import subsample.analysis
import subsample.cache
import subsample.library
import subsample.watcher

import tests.helpers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	audio_ext: str = ".wav",
) -> pathlib.Path:
	return tests.helpers._write_sidecar(directory, audio_stem, audio_ext)


def _write_wav_and_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	n_frames: int = 2048,
) -> tuple[pathlib.Path, pathlib.Path]:
	return tests.helpers._write_wav_and_sidecar(directory, audio_stem, n_frames)


# Generous timeout for debounce window plus filesystem event latency.
_TIMEOUT: float = 5.0

# Extended timeout for audio-path tests (debounce + grace + analysis).
_AUDIO_PATH_TIMEOUT: float = 15.0


def _fast_audio_timings () -> dict[str, float | int]:
	"""Speed up audio-path timing constants for tests. Returns originals."""

	originals = {
		"audio_debounce": subsample.watcher._AUDIO_DEBOUNCE_SECONDS,
		"sidecar_grace":  subsample.watcher._SIDECAR_GRACE_SECONDS,
		"stability":      subsample.watcher._STABILITY_CHECK_SECONDS,
		"stability_max":  subsample.watcher._STABILITY_MAX_CHECKS,
	}

	subsample.watcher._AUDIO_DEBOUNCE_SECONDS = 0.3
	subsample.watcher._SIDECAR_GRACE_SECONDS = 0.5
	subsample.watcher._STABILITY_CHECK_SECONDS = 0.3
	subsample.watcher._STABILITY_MAX_CHECKS = 3

	return originals


def _restore_audio_timings (originals: dict[str, float | int]) -> None:
	"""Restore audio-path timing constants after a test."""

	subsample.watcher._AUDIO_DEBOUNCE_SECONDS = originals["audio_debounce"]  # type: ignore[assignment]
	subsample.watcher._SIDECAR_GRACE_SECONDS = originals["sidecar_grace"]    # type: ignore[assignment]
	subsample.watcher._STABILITY_CHECK_SECONDS = originals["stability"]      # type: ignore[assignment]
	subsample.watcher._STABILITY_MAX_CHECKS = originals["stability_max"]     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TestInstrumentWatcher — sidecar path (existing behaviour)
# ---------------------------------------------------------------------------

class TestInstrumentWatcher:

	def _make_watcher (
		self,
		directory: pathlib.Path,
		callback: typing.Callable[[subsample.library.SampleRecord], None],
		known_sidecars: typing.Optional[set[pathlib.Path]] = None,
		known_audio: typing.Optional[set[pathlib.Path]] = None,
	) -> subsample.watcher.InstrumentWatcher:

		"""Construct a watcher with empty known sets by default."""

		return subsample.watcher.InstrumentWatcher(
			directory=directory,
			known_sidecars=known_sidecars or set(),
			on_sample_loaded=callback,
			known_audio=known_audio,
		)

	def test_new_sample_triggers_callback (self, tmp_path: pathlib.Path) -> None:

		"""A new WAV + sidecar pair written to the watched dir calls on_sample_loaded."""

		received: list[subsample.library.SampleRecord] = []
		done = threading.Event()

		def on_loaded (record: subsample.library.SampleRecord) -> None:
			received.append(record)
			done.set()

		watcher = self._make_watcher(tmp_path, on_loaded)
		watcher.start()

		try:
			_write_wav_and_sidecar(tmp_path, "kick")
			triggered = done.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()

		assert triggered, "Callback not called within timeout"
		assert len(received) == 1
		assert received[0].name == "kick"
		assert received[0].audio is not None
		assert received[0].filepath is not None

	def test_known_sidecar_ignored (self, tmp_path: pathlib.Path) -> None:

		"""Sidecars listed in known_sidecars at construction time are silently skipped."""

		# Write the sample before starting the watcher; register as known.
		wav_path, sidecar_path = _write_wav_and_sidecar(tmp_path, "snare")

		called = threading.Event()
		watcher = self._make_watcher(
			tmp_path,
			lambda _r: called.set(),
			known_sidecars={sidecar_path.resolve()},
			known_audio={wav_path.resolve()},
		)
		watcher.start()

		try:
			# Touch the sidecar to fire a filesystem event — watcher should ignore it.
			sidecar_path.write_text(
				sidecar_path.read_text(encoding="utf-8"), encoding="utf-8"
			)
			triggered = called.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()

		assert not triggered, "Callback fired for a known sidecar (should be skipped)"

	def test_sidecar_without_wav_retries_then_gives_up (self, tmp_path: pathlib.Path) -> None:

		"""A sidecar whose WAV never arrives is retried then abandoned without crashing."""

		called = threading.Event()

		# Speed up the retry cycle so the test finishes quickly.
		original_delay = subsample.watcher._RETRY_DELAY_SECONDS
		subsample.watcher._RETRY_DELAY_SECONDS = 0.2

		watcher = self._make_watcher(tmp_path, lambda _r: called.set())
		watcher.start()

		try:
			_write_sidecar(tmp_path, "ghost_wav")

			# Allow time for debounce + all retries to exhaust.
			budget = (subsample.watcher._MAX_RETRIES + 1) * subsample.watcher._RETRY_DELAY_SECONDS
			triggered = called.wait(timeout=subsample.watcher._DEBOUNCE_SECONDS + budget + 1.0)
		finally:
			subsample.watcher._RETRY_DELAY_SECONDS = original_delay
			watcher.stop()

		assert not triggered, "Callback fired for sidecar with missing WAV"

	def test_stop_is_clean (self, tmp_path: pathlib.Path) -> None:

		"""start() then stop() terminates the observer thread cleanly without hanging."""

		watcher = self._make_watcher(tmp_path, lambda _r: None)
		watcher.start()
		watcher.stop()
		# Reaching here means the observer thread joined cleanly.

	def test_multiple_samples_all_delivered (self, tmp_path: pathlib.Path) -> None:

		"""Multiple new samples arriving sequentially all trigger the callback."""

		names: list[str] = []
		lock  = threading.Lock()
		done  = threading.Event()
		expected_count = 3

		def on_loaded (record: subsample.library.SampleRecord) -> None:
			with lock:
				names.append(record.name)
				if len(names) >= expected_count:
					done.set()

		watcher = self._make_watcher(tmp_path, on_loaded)
		watcher.start()

		try:
			for stem in ("kick", "snare", "hihat"):
				_write_wav_and_sidecar(tmp_path, stem)
			triggered = done.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()

		assert triggered, f"Only {len(names)}/{expected_count} callbacks received"
		assert sorted(names) == ["hihat", "kick", "snare"]


# ---------------------------------------------------------------------------
# TestInstrumentWatcherAudioPath — audio file detection (no sidecar)
# ---------------------------------------------------------------------------

class TestInstrumentWatcherAudioPath:

	"""Tests for the audio-file detection path that handles files from
	non-subsample sources (no .analysis.json sidecar expected).
	"""

	def _make_watcher (
		self,
		directory: pathlib.Path,
		callback: typing.Callable[[subsample.library.SampleRecord], None],
		known_audio: typing.Optional[set[pathlib.Path]] = None,
	) -> subsample.watcher.InstrumentWatcher:

		return subsample.watcher.InstrumentWatcher(
			directory=directory,
			known_sidecars=set(),
			on_sample_loaded=callback,
			known_audio=known_audio,
		)

	def test_wav_without_sidecar_triggers_callback (self, tmp_path: pathlib.Path) -> None:

		"""A WAV file appearing without a sidecar triggers the audio-path callback
		after the grace period, analyzing and loading the sample.
		"""

		originals = _fast_audio_timings()

		received: list[subsample.library.SampleRecord] = []
		done = threading.Event()

		def on_loaded (record: subsample.library.SampleRecord) -> None:
			received.append(record)
			done.set()

		watcher = self._make_watcher(tmp_path, on_loaded)
		watcher.start()

		try:
			tests.helpers._make_wav(tmp_path / "hihat.wav")
			triggered = done.wait(timeout=_AUDIO_PATH_TIMEOUT)
		finally:
			watcher.stop()
			_restore_audio_timings(originals)

		assert triggered, "Callback not called for WAV without sidecar"
		assert len(received) == 1
		assert received[0].name == "hihat"
		assert received[0].audio is not None

		# Verify that a sidecar was generated.
		sidecar = subsample.cache.cache_path(tmp_path / "hihat.wav")
		assert sidecar.exists(), "Sidecar should have been generated by load_or_analyze"

	def test_sidecar_arriving_during_grace_skips_audio_path (self, tmp_path: pathlib.Path) -> None:

		"""When a sidecar appears during the grace period, the audio path
		defers to the sidecar path — no redundant analysis.
		"""

		originals = _fast_audio_timings()

		# Use a longer grace period so we have time to write the sidecar.
		subsample.watcher._SIDECAR_GRACE_SECONDS = 3.0

		received: list[subsample.library.SampleRecord] = []
		lock = threading.Lock()
		done = threading.Event()

		def on_loaded (record: subsample.library.SampleRecord) -> None:
			with lock:
				received.append(record)
				done.set()

		watcher = self._make_watcher(tmp_path, on_loaded)
		watcher.start()

		try:
			# Write the WAV first (triggers audio path debounce).
			tests.helpers._make_wav(tmp_path / "tom.wav")

			# Wait past the audio debounce but before grace expires,
			# then write the sidecar — sidecar path should handle it.
			time.sleep(subsample.watcher._AUDIO_DEBOUNCE_SECONDS + 0.5)
			_write_sidecar(tmp_path, "tom")

			triggered = done.wait(timeout=_AUDIO_PATH_TIMEOUT)
		finally:
			watcher.stop()
			_restore_audio_timings(originals)

		assert triggered, "Callback not called at all"

		# Should have received exactly one record (from the sidecar path,
		# not a second one from the audio path).
		with lock:
			assert len(received) == 1, f"Expected 1 callback, got {len(received)}"
			assert received[0].name == "tom"

	def test_known_audio_ignored (self, tmp_path: pathlib.Path) -> None:

		"""Audio files listed in known_audio at construction time are silently skipped."""

		originals = _fast_audio_timings()

		wav_path = tmp_path / "kick.wav"
		tests.helpers._make_wav(wav_path)

		called = threading.Event()
		watcher = self._make_watcher(
			tmp_path,
			lambda _r: called.set(),
			known_audio={wav_path.resolve()},
		)
		watcher.start()

		try:
			# Touch the file to fire a filesystem event.
			wav_path.write_bytes(wav_path.read_bytes())
			triggered = called.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()
			_restore_audio_timings(originals)

		assert not triggered, "Callback fired for known audio file (should be skipped)"

	def test_non_audio_extension_ignored (self, tmp_path: pathlib.Path) -> None:

		"""Files with unrecognised extensions do not trigger the audio path."""

		originals = _fast_audio_timings()

		called = threading.Event()
		watcher = self._make_watcher(tmp_path, lambda _r: called.set())
		watcher.start()

		try:
			(tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
			(tmp_path / "data.csv").write_text("1,2,3", encoding="utf-8")
			triggered = called.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()
			_restore_audio_timings(originals)

		assert not triggered, "Callback fired for non-audio file"

	def test_file_stability_retries_on_size_change (self, tmp_path: pathlib.Path) -> None:

		"""When a file's size changes between stability checks, the watcher
		retries until the size stabilises.
		"""

		originals = _fast_audio_timings()

		received: list[subsample.library.SampleRecord] = []
		done = threading.Event()

		def on_loaded (record: subsample.library.SampleRecord) -> None:
			received.append(record)
			done.set()

		watcher = self._make_watcher(tmp_path, on_loaded)
		watcher.start()

		wav_path = tmp_path / "growing.wav"

		try:
			# Write a small file first (triggers audio path).
			tests.helpers._make_wav(wav_path, n_frames=1024)

			# After debounce fires, overwrite with a larger file before
			# the grace period completes — simulates a file still being written.
			time.sleep(subsample.watcher._AUDIO_DEBOUNCE_SECONDS + 0.2)
			tests.helpers._make_wav(wav_path, n_frames=4096)

			# Eventually the file stabilises and the callback fires.
			triggered = done.wait(timeout=_AUDIO_PATH_TIMEOUT)
		finally:
			watcher.stop()
			_restore_audio_timings(originals)

		assert triggered, "Callback not called after file stabilised"
		assert len(received) == 1
		assert received[0].name == "growing"

	def test_empty_file_skipped (self, tmp_path: pathlib.Path) -> None:

		"""An empty audio file (0 bytes) is silently skipped."""

		originals = _fast_audio_timings()

		called = threading.Event()
		watcher = self._make_watcher(tmp_path, lambda _r: called.set())
		watcher.start()

		try:
			(tmp_path / "empty.wav").write_bytes(b"")

			budget = (
				subsample.watcher._AUDIO_DEBOUNCE_SECONDS
				+ subsample.watcher._SIDECAR_GRACE_SECONDS
				+ 1.0
			)
			triggered = called.wait(timeout=budget)
		finally:
			watcher.stop()
			_restore_audio_timings(originals)

		assert not triggered, "Callback fired for empty file"


# ---------------------------------------------------------------------------
# TestMidiMapWatcher
# ---------------------------------------------------------------------------

class TestMidiMapWatcher:

	def _make_midi_map_file (self, directory: pathlib.Path, name: str = "midi-map.yaml") -> pathlib.Path:

		"""Write a minimal MIDI map YAML file and return its path."""

		path = directory / name
		path.write_text("assignments: []\n", encoding="utf-8")
		return path

	def test_modified_file_triggers_callback (self, tmp_path: pathlib.Path) -> None:

		"""Modifying the watched file calls on_changed with the file path."""

		received: list[pathlib.Path] = []
		done = threading.Event()

		def on_changed (path: pathlib.Path) -> None:
			received.append(path)
			done.set()

		midi_map_path = self._make_midi_map_file(tmp_path)
		watcher = subsample.watcher.MidiMapWatcher(path=midi_map_path, on_changed=on_changed)
		watcher.start()

		try:
			# Wait briefly for the observer thread to be ready.
			import time
			time.sleep(0.2)

			midi_map_path.write_text("assignments:\n  - name: test\n", encoding="utf-8")
			triggered = done.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()

		assert triggered, "Callback not called within timeout"
		assert len(received) >= 1
		assert received[0] == midi_map_path.resolve()

	def test_debounce_coalesces_rapid_writes (self, tmp_path: pathlib.Path) -> None:

		"""Multiple rapid writes produce a single callback after the debounce window."""

		call_count = 0
		lock = threading.Lock()
		done = threading.Event()

		def on_changed (path: pathlib.Path) -> None:
			nonlocal call_count
			with lock:
				call_count += 1
			done.set()

		midi_map_path = self._make_midi_map_file(tmp_path)
		watcher = subsample.watcher.MidiMapWatcher(path=midi_map_path, on_changed=on_changed)
		watcher.start()

		try:
			import time
			time.sleep(0.2)

			# Write 5 times in rapid succession.
			for i in range(5):
				midi_map_path.write_text(f"assignments: []  # write {i}\n", encoding="utf-8")

			# Wait long enough for the debounce to fire (but only once).
			triggered = done.wait(timeout=_TIMEOUT)

			# Brief extra wait to catch any spurious second callback.
			time.sleep(subsample.watcher._MIDI_MAP_DEBOUNCE_SECONDS + 0.5)
		finally:
			watcher.stop()

		assert triggered, "Callback not called within timeout"

		with lock:
			assert call_count == 1, f"Expected 1 debounced callback, got {call_count}"

	def test_unrelated_file_ignored (self, tmp_path: pathlib.Path) -> None:

		"""Changes to other files in the same directory do not trigger the callback."""

		called = threading.Event()
		midi_map_path = self._make_midi_map_file(tmp_path)
		watcher = subsample.watcher.MidiMapWatcher(
			path=midi_map_path,
			on_changed=lambda _p: called.set(),
		)
		watcher.start()

		try:
			import time
			time.sleep(0.2)

			# Write a different file in the same directory.
			(tmp_path / "other.yaml").write_text("unrelated: true\n", encoding="utf-8")
			triggered = called.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()

		assert not triggered, "Callback fired for unrelated file"

	def test_stop_is_clean (self, tmp_path: pathlib.Path) -> None:

		"""start() then stop() terminates cleanly without hanging."""

		midi_map_path = self._make_midi_map_file(tmp_path)
		watcher = subsample.watcher.MidiMapWatcher(
			path=midi_map_path,
			on_changed=lambda _p: None,
		)
		watcher.start()
		watcher.stop()
