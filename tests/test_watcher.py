"""Tests for subsample.watcher.InstrumentWatcher.

The watcher runs a background watchdog observer thread, so tests use
threading.Event with a generous timeout to detect callback invocations
asynchronously. The 1-second debounce window means each test that
expects a callback must wait up to ~3 seconds for it to fire.
"""

import json
import pathlib
import threading
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

	"""Write a valid .analysis.json sidecar to directory.

	Does NOT create the audio file — only the sidecar. Returns the sidecar path.

	NOTE: the JSON payload below mirrors the format in subsample/cache.py. If
	the sidecar schema changes (new fields, renamed keys), this helper must be
	updated to match, otherwise watcher tests will silently use stale sidecars.
	"""

	audio_path   = directory / (audio_stem + audio_ext)
	sidecar_path = subsample.cache.cache_path(audio_path)
	spectral     = tests.helpers._make_spectral()
	rhythm       = tests.helpers._make_rhythm()
	pitch        = tests.helpers._make_pitch()
	timbre       = tests.helpers._make_timbre()
	level        = tests.helpers._make_level()
	band_energy  = tests.helpers._make_band_energy()
	params       = tests.helpers._make_params()

	payload: dict[str, typing.Any] = {
		"analysis_version": subsample.analysis.ANALYSIS_VERSION,
		"audio_md5":        "deadbeef00000000deadbeef00000000",
		"sample_rate":      params.sample_rate,
		"duration":         1.0,
		"params": {
			"n_fft":        params.n_fft,
			"hop_length":   params.hop_length,
			"sample_rate":  params.sample_rate,
		},
		"spectral": {
			"spectral_flatness":  spectral.spectral_flatness,
			"attack":             spectral.attack,
			"release":            spectral.release,
			"spectral_centroid":  spectral.spectral_centroid,
			"spectral_bandwidth": spectral.spectral_bandwidth,
			"zcr":                spectral.zcr,
			"harmonic_ratio":     spectral.harmonic_ratio,
			"spectral_contrast":  spectral.spectral_contrast,
			"voiced_fraction":    spectral.voiced_fraction,
			"log_attack_time":    spectral.log_attack_time,
			"spectral_flux":      spectral.spectral_flux,
		},
		"rhythm": {
			"tempo_bpm":        rhythm.tempo_bpm,
			"beat_times":       list(rhythm.beat_times),
			"pulse_curve":      rhythm.pulse_curve.tolist(),
			"pulse_peak_times": list(rhythm.pulse_peak_times),
			"onset_times":      list(rhythm.onset_times),
			"onset_count":      rhythm.onset_count,
		},
		"pitch": {
			"dominant_pitch_hz":    pitch.dominant_pitch_hz,
			"pitch_confidence":     pitch.pitch_confidence,
			"chroma_profile":       list(pitch.chroma_profile),
			"dominant_pitch_class": pitch.dominant_pitch_class,
			"pitch_stability":      pitch.pitch_stability,
			"voiced_frame_count":   pitch.voiced_frame_count,
		},
		"timbre": {
			"mfcc":       list(timbre.mfcc),
			"mfcc_delta": list(timbre.mfcc_delta),
			"mfcc_onset": list(timbre.mfcc_onset),
		},
		"level": {
			"peak": level.peak,
			"rms":  level.rms,
		},
		"band_energy": {
			"energy_fractions": list(band_energy.energy_fractions),
			"decay_rates":      list(band_energy.decay_rates),
		},
	}

	sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
	return sidecar_path


def _write_wav_and_sidecar (
	directory: pathlib.Path,
	audio_stem: str,
	n_frames: int = 2048,
) -> tuple[pathlib.Path, pathlib.Path]:

	"""Write a WAV file and its sidecar. Returns (wav_path, sidecar_path)."""

	wav_path     = directory / (audio_stem + ".wav")
	tests.helpers._make_wav(wav_path, n_frames=n_frames)
	sidecar_path = _write_sidecar(directory, audio_stem)
	return wav_path, sidecar_path


# Generous timeout for debounce window plus filesystem event latency.
_TIMEOUT: float = 5.0


# ---------------------------------------------------------------------------
# TestInstrumentWatcher
# ---------------------------------------------------------------------------

class TestInstrumentWatcher:

	def _make_watcher (
		self,
		directory: pathlib.Path,
		callback: typing.Callable[[subsample.library.SampleRecord], None],
		known_sidecars: typing.Optional[set[pathlib.Path]] = None,
	) -> subsample.watcher.InstrumentWatcher:

		"""Construct a watcher with an empty known-sidecars set by default."""

		return subsample.watcher.InstrumentWatcher(
			directory=directory,
			known_sidecars=known_sidecars or set(),
			on_sample_loaded=callback,
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
		_, sidecar_path = _write_wav_and_sidecar(tmp_path, "snare")

		called = threading.Event()
		watcher = self._make_watcher(
			tmp_path,
			lambda _r: called.set(),
			known_sidecars={sidecar_path.resolve()},
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

	def test_wav_only_no_callback (self, tmp_path: pathlib.Path) -> None:

		"""A WAV file appearing without a sidecar does not trigger the callback."""

		called = threading.Event()
		watcher = self._make_watcher(tmp_path, lambda _r: called.set())
		watcher.start()

		try:
			tests.helpers._make_wav(tmp_path / "hihat.wav")
			triggered = called.wait(timeout=_TIMEOUT)
		finally:
			watcher.stop()

		assert not triggered, "Callback fired for WAV-only event (should be ignored)"

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
