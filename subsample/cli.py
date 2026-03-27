"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, the WAV writer, and the MIDI player. Supports two
input modes and two run modes:

  File input   — pass WAV file paths as positional arguments; each file
                 is processed through the detection pipeline and segments
                 saved to the output directory, then picked up by the
                 instrument library loader.

  Live capture — stream from an audio input device (recorder.enabled: true).

  MIDI player  — listen for MIDI input and play instrument samples
                 (player.enabled: true).

Recorder and player run as threads so they can operate concurrently.
The main thread handles KeyboardInterrupt and coordinates shutdown via
a shared threading.Event.

Press Ctrl+C to stop cleanly.
"""

import argparse
import dataclasses
import datetime
import logging
import pathlib
import sys
import threading
import typing

import numpy
import yaml

import subsample.analysis
import subsample.audio
import subsample.buffer
import subsample.config
import subsample.detector
import subsample.library
import subsample.player
import subsample.recorder
import subsample.similarity
import subsample.transform
import subsample.trim
import subsample.watcher


_log = logging.getLogger(__name__)

# Maps config bit_depth to the NumPy dtype used for sample storage.
# 24-bit audio is stored as int32 (left-shifted by 8) — see audio.unpack_audio().
_AUDIO_DTYPE: dict[int, numpy.dtype] = {
	16: numpy.dtype(numpy.int16),
	24: numpy.dtype(numpy.int32),
	32: numpy.dtype(numpy.int32),
}


def _parse_args () -> argparse.Namespace:

	"""Parse command-line arguments.

	Returns:
		Namespace with a 'files' attribute containing a (possibly empty)
		list of pathlib.Path objects.
	"""

	parser = argparse.ArgumentParser(
		prog="subsample",
		description="Ambient audio sample recorder and analyser",
	)
	parser.add_argument(
		"files",
		nargs="*",
		type=pathlib.Path,
		metavar="FILE",
		help=(
			"WAV files to process through the detection pipeline before "
			"starting live capture. Segments are written to the configured "
			"output directory and named after the source file "
			"(e.g. recording_1.wav, recording_2.wav, …)."
		),
	)
	return parser.parse_args()


def _process_chunk (
	chunk: numpy.ndarray,
	buf: subsample.buffer.CircularBuffer,
	detector: subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
) -> typing.Optional[numpy.ndarray]:

	"""Feed one audio chunk through the detection pipeline.

	Writes the chunk to the circular buffer, runs the level detector, and
	if a recording is completed, retrieves and trims the segment.

	Args:
		chunk:         One audio chunk, shape (chunk_size, channels).
		buf:           Circular buffer receiving the audio stream.
		detector:      Level detector tracking ambient noise and recording state.
		detection_cfg: Detection settings (SNR threshold, trim params, etc.).

	Returns:
		Trimmed audio segment as a numpy array, or None if no recording completed.
	"""

	buf.write(chunk)

	result = detector.process_chunk(chunk, buf.frames_written)

	if result is None:
		return None

	start_frame, end_frame = result
	pre = detection_cfg.trim_pre_samples
	# Read back `pre` extra frames before the detector's start boundary so
	# trim_silence has raw audio available for its fade-in window.
	# trim_silence then uses the same `pre_samples` value to decide how many
	# of those frames to keep and how wide the S-curve fade should be.
	# The two uses are coordinated, not double-counted.
	segment = buf.read_range(max(0, start_frame - pre), end_frame)

	amplitude_threshold = detector.ambient_rms * (
		10.0 ** (detection_cfg.snr_threshold_db / 20.0)
	)

	trimmed = subsample.trim.trim_silence(
		segment,
		amplitude_threshold,
		pre_samples=detection_cfg.trim_pre_samples,
		post_samples=detection_cfg.trim_post_samples,
	)

	if trimmed.size == 0:
		return None

	return trimmed


def _process_input_files (
	files: list[pathlib.Path],
	cfg: subsample.config.Config,
) -> None:

	"""Process audio files through the detection pipeline, writing segments to disk.

	Each file is read with its native sample rate, bit depth, and channel count.
	Detected segments are written to cfg.output.directory with names derived from
	the source filename (e.g. field_recording_1.wav, field_recording_2.wav, …).
	Files that cannot be read are skipped with a warning.

	Args:
		files: Paths to audio files to process.
		cfg:   Application config (detection settings, output directory).
	"""

	for path in files:
		if not path.exists():
			_log.warning("Input file not found, skipping: %s", path)
			continue

		print(f"Processing {path.name}…")

		try:
			file_info = subsample.audio.read_audio_file(path)
		except (OSError, ValueError) as exc:
			_log.warning("Could not read %s: %s — skipping", path.name, exc)
			continue

		audio_dtype = _AUDIO_DTYPE.get(file_info.bit_depth)
		if audio_dtype is None:
			_log.warning(
				"Unsupported bit depth %d in %s — skipping",
				file_info.bit_depth, path.name,
			)
			continue

		max_frames = file_info.sample_rate * cfg.recorder.buffer.max_seconds
		buf = subsample.buffer.CircularBuffer(max_frames, file_info.channels, dtype=audio_dtype)

		detector = subsample.detector.LevelDetector(
			cfg.detection,
			file_info.sample_rate,
			cfg.recorder.audio.chunk_size,
			max_recording_frames=max_frames,
		)

		analysis_params = subsample.analysis.compute_params(file_info.sample_rate)

		writer = subsample.recorder.SampleProcessor(cfg, analysis_params, on_complete=None)

		segment_index = 1
		chunk_size = cfg.recorder.audio.chunk_size
		n_frames = file_info.audio.shape[0]

		for offset in range(0, n_frames, chunk_size):
			chunk = file_info.audio[offset : offset + chunk_size]

			trimmed = _process_chunk(chunk, buf, detector, cfg.detection)

			if trimmed is not None:
				writer.enqueue(
					trimmed,
					datetime.datetime.now(),
					filename_base=f"{path.stem}_{segment_index}",
					sample_rate=file_info.sample_rate,
					bit_depth=file_info.bit_depth,
				)
				segment_index += 1

		writer.flush()
		writer.shutdown()

		count = segment_index - 1
		print(f"  → {count} segment(s) written from {path.name}")


def _run_recorder (
	cfg: subsample.config.Config,
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	shutdown_event: threading.Event,
	store_audio: bool,
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]] = None,
) -> None:

	"""Set up an audio input device and run the real-time capture loop.

	Streams audio from the configured device (or interactively selected device)
	into a circular buffer. Detected recordings are trimmed, queued for WAV
	output, analyzed, and added to the instrument library. Runs until
	shutdown_event is set.

	Args:
		cfg:                Full application config.
		reference_library:  Loaded reference samples, or None if not configured.
		instrument_library: Instrument sample library to update in real time.
		analysis_params:    Pre-computed FFT params matching cfg.recorder.audio.sample_rate.
		similarity_matrix:  Similarity index to update as new samples arrive, or None.
		shutdown_event:     Set this to stop the capture loop cleanly.
		store_audio:        When True, keep PCM data in SampleRecord for playback.
		transform_manager:  Optional transform pipeline; notified of new and evicted
		                    samples so derivative variants are kept in sync.
		player_cell:        Single-element list holding the active MidiPlayer, or None.
		                    Forwarded to _make_on_complete so pitched assignments are
		                    updated when the best match changes.
	"""

	pa = subsample.audio.create_pyaudio()

	try:
		if cfg.recorder.audio.device is not None:
			device_index = subsample.audio.find_device_by_name(pa, cfg.recorder.audio.device)
		else:
			devices = subsample.audio.list_input_devices(pa)
			device_index = subsample.audio.select_device(devices)

		# Resolve channel count: if not set in config, detect from the device.
		# A stereo mic (e.g. Shure MV88+) reports maxInputChannels=2 and will
		# automatically record and play back in stereo without any config change.
		audio_cfg = cfg.recorder.audio
		if audio_cfg.channels is None:
			detected_channels = subsample.audio.get_device_channels(pa, device_index)
			audio_cfg = dataclasses.replace(audio_cfg, channels=detected_channels)
			_log.info("Auto-detected %d input channel(s) from device", detected_channels)

		reader = subsample.audio.AudioReader(pa, device_index, audio_cfg)

	except (ValueError, OSError) as exc:
		print(f"Error opening audio device: {exc}", file=sys.stderr)
		pa.terminate()
		return

	# By this point channels is always resolved (either explicit from config, or
	# auto-detected above).  The assert narrows the type for mypy.
	assert audio_cfg.channels is not None

	audio_dtype = _AUDIO_DTYPE[audio_cfg.bit_depth]
	max_frames = audio_cfg.sample_rate * cfg.recorder.buffer.max_seconds
	buf = subsample.buffer.CircularBuffer(max_frames, audio_cfg.channels, dtype=audio_dtype)

	detector = subsample.detector.LevelDetector(
		cfg.detection,
		audio_cfg.sample_rate,
		audio_cfg.chunk_size,
		max_recording_frames=max_frames,
	)

	writer = subsample.recorder.SampleProcessor(
		cfg,
		analysis_params,
		on_complete=_make_on_complete(
			reference_library, instrument_library, analysis_params,
			similarity_matrix, store_audio, transform_manager,
			player_cell=player_cell,
		),
	)

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	try:
		while not shutdown_event.is_set():
			chunk = reader.read(timeout=0.5)

			if chunk is None:
				# Timeout — loop back to check shutdown_event.
				continue

			trimmed = _process_chunk(chunk, buf, detector, cfg.detection)

			if trimmed is not None:
				writer.enqueue(trimmed, datetime.datetime.now())

	finally:
		if reader.overflow_count > 0:
			_log.warning(
				"Audio overflows detected during capture: %d — "
				"recordings may contain discontinuities",
				reader.overflow_count,
			)

		reader.stop()
		pa.terminate()
		writer.shutdown()


def _start_player (
	cfg: subsample.config.Config,
	shutdown_event: threading.Event,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: subsample.similarity.SimilarityMatrix,
	reference_library: subsample.library.ReferenceLibrary,
	player_cell: list[typing.Optional[subsample.player.MidiPlayer]],
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
) -> None:

	"""Select a MIDI input device (or create a virtual port), then run the player.

	When cfg.player.virtual_midi_port is set, Subsample creates a named virtual
	MIDI input port and skips hardware device selection entirely. Otherwise, it
	resolves a hardware device from config (substring match) or prompts the user
	interactively. Runs until shutdown_event is set.

	Args:
		cfg:                Full application config.
		shutdown_event:     Set this to stop the player cleanly.
		instrument_library: Loaded instrument samples (must have audio in memory).
		similarity_matrix:  Similarity index for note → sample lookup.
		reference_library:  Reference library; provides sorted names for note mapping.
		player_cell:        Single-element list; _start_player stores the MidiPlayer
		                    here before calling run() so the on_complete callback can
		                    call update_pitched_assignments() when the best match changes.
		transform_manager:  Optional transform pipeline; enables pitched variant
		                    playback when provided.
	"""

	# Load the MIDI routing map from config or fall back to the shipped default.
	_midi_map_path = (
		pathlib.Path(cfg.player.midi_map) if cfg.player.midi_map is not None
		else pathlib.Path(__file__).parent.parent / "midi-map.yaml.default"
	)
	try:
		midi_map = subsample.player.load_midi_map(_midi_map_path, reference_library.names())
	except (FileNotFoundError, ValueError) as exc:
		print(f"Error loading MIDI map: {exc}", file=sys.stderr)
		return

	# Virtual port mode: bypass hardware device selection entirely.
	# MidiPlayer.run() will open the named virtual port with virtual=True.
	if cfg.player.virtual_midi_port is not None:
		print(f"  MIDI input   : virtual port \"{cfg.player.virtual_midi_port}\"")
		player = subsample.player.MidiPlayer(
			"",
			shutdown_event,
			instrument_library=instrument_library,
			similarity_matrix=similarity_matrix,
			midi_map=midi_map,
			sample_rate=cfg.recorder.audio.sample_rate,
			bit_depth=cfg.recorder.audio.bit_depth,
			output_device_name=cfg.player.audio.device,
			output_bit_depth=cfg.player.audio.bit_depth,
			output_sample_rate=cfg.player.audio.sample_rate,
			transform_manager=transform_manager,
			virtual_midi_port=cfg.player.virtual_midi_port,
			max_polyphony=cfg.player.max_polyphony,
			limiter_threshold_db=cfg.player.limiter_threshold_db,
			limiter_ceiling_db=cfg.player.limiter_ceiling_db,
		)
		player_cell[0] = player
		player.update_pitched_assignments()
		try:
			player.run()
		except ValueError as exc:
			print(f"\nError starting player: {exc}", file=sys.stderr)

		return

	# Hardware port mode: resolve device name from config or interactive menu.
	try:
		devices = subsample.player.list_midi_input_devices()

		if cfg.player.midi_device is not None:
			device_name = subsample.player.find_midi_device_by_name(cfg.player.midi_device)
		else:
			device_name = subsample.player.select_midi_device(devices)

	except ValueError as exc:
		print(f"Error opening MIDI device: {exc}", file=sys.stderr)
		return

	player = subsample.player.MidiPlayer(
		device_name,
		shutdown_event,
		instrument_library=instrument_library,
		similarity_matrix=similarity_matrix,
		midi_map=midi_map,
		sample_rate=cfg.recorder.audio.sample_rate,
		bit_depth=cfg.recorder.audio.bit_depth,
		output_device_name=cfg.player.audio.device,
		output_bit_depth=cfg.player.audio.bit_depth,
		output_sample_rate=cfg.player.audio.sample_rate,
		transform_manager=transform_manager,
		max_polyphony=cfg.player.max_polyphony,
		limiter_threshold_db=cfg.player.limiter_threshold_db,
		limiter_ceiling_db=cfg.player.limiter_ceiling_db,
	)
	player_cell[0] = player
	player.update_pitched_assignments()
	try:
		player.run()
	except ValueError as exc:
		print(f"\nError starting player: {exc}", file=sys.stderr)


def main () -> None:

	"""Run the ambient audio sampler.

	Processes any input files first (if given on the command line), then
	loads libraries and starts the recorder and/or player as configured.
	Both run as threads; the main thread coordinates shutdown on Ctrl+C.
	"""

	logging.basicConfig(
		level=logging.WARNING,
		format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
		datefmt="%H:%M:%S",
	)
	logging.getLogger("subsample").setLevel(logging.INFO)

	args = _parse_args()

	cfg = subsample.config.load_config()

	_print_banner(cfg)

	# Load reference library first — before instrument samples — so that similarity
	# comparisons against reference sounds are available as instrument samples load.
	reference_library: typing.Optional[subsample.library.ReferenceLibrary] = None
	if cfg.reference is not None:
		reference_library = subsample.library.load_reference_library(
			pathlib.Path(cfg.reference.directory)
		)
		print(f"  Reference    : {len(reference_library)} sample(s) loaded from {cfg.reference.directory}")

	# Process any input files through the detection pipeline.
	# Segments are written to the output directory (which is typically the same
	# as the instrument directory), so they are picked up by the instrument loader below.
	if args.files:
		_process_input_files(args.files, cfg)

	# Create instrument library. PCM audio is only needed when the player is
	# active — skipping it saves memory when player is disabled.
	max_instrument_bytes = int(cfg.instrument.max_memory_mb * 1024 * 1024)
	if cfg.instrument.directory is not None and cfg.player.enabled:
		instrument_library = subsample.library.load_instrument_library(
			pathlib.Path(cfg.instrument.directory),
			max_instrument_bytes,
			load_audio=True,
			clean_orphaned_sidecars=cfg.instrument.clean_orphaned_sidecars,
		)
		print(
			f"  Instruments  : {len(instrument_library)} sample(s) loaded"
			f" from {cfg.instrument.directory}"
		)
		_log.info(
			"Instrument library: %d sample(s)  [%s]",
			len(instrument_library), instrument_library.format_memory(),
		)
	else:
		instrument_library = subsample.library.InstrumentLibrary(max_instrument_bytes)

	analysis_params = subsample.analysis.compute_params(cfg.recorder.audio.sample_rate)

	# Build the similarity matrix now that both libraries are populated.
	# bulk_add() is vectorised (single matrix multiply for N × M scores),
	# so loading hundreds of pre-existing instrument samples is fast.
	# Only needed when the player is enabled — similarity scores drive
	# note-to-sample routing and are not consumed by the recorder alone.
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix] = None
	if reference_library is not None and cfg.player.enabled:
		similarity_matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
		if len(instrument_library) > 0:
			similarity_matrix.bulk_add(instrument_library.samples())
		print(f"  Similarity   : {similarity_matrix}")

	# --- Transform pipeline ---
	# Only needed when the player is enabled — variants (base, pitch-shifted,
	# time-stretched) exist solely to serve MIDI-triggered playback.  Skip the
	# entire pipeline when running recorder-only to avoid unnecessary CPU/memory
	# usage and spurious "Auto-pitch" log messages.
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None

	if cfg.player.enabled:
		# Create the cache, processor, and manager now (before threads start) so
		# that the on_complete callback and the player share the same instance.
		# Phase 2: pitch shifting is active — on_sample_added() auto-enqueues
		# variants for tonal samples via _apply_pitch() (Rubber Band, offline mode).
		max_transform_bytes = int(cfg.transform.max_memory_mb * 1024 * 1024)

		_transform_cache = subsample.transform.TransformCache(
			max_memory_bytes=max_transform_bytes,
		)
		def _on_transform_complete (
			result: subsample.transform.TransformResult,
		) -> None:
			_transform_cache.put(result)

		def _on_transform_idle (completed: int) -> None:
			_log.info(
				"Transform queue idle — %d variant(s) processed  [cache: %s]",
				completed, _transform_cache.format_memory(),
			)

		# Resolve the effective output sample rate for the player.  Variants are
		# resampled to this rate in the transform worker so they arrive at the
		# player pre-converted — zero conversion overhead at trigger time.
		output_sample_rate = (
			cfg.player.audio.sample_rate
			if cfg.player.audio.sample_rate is not None
			else cfg.recorder.audio.sample_rate
		)

		# Disk cache for persistent variant storage across restarts.
		_variant_disk_cache: typing.Optional[subsample.transform.VariantDiskCache] = None

		if cfg.transform.variant_cache_dir and cfg.transform.max_disk_mb > 0:
			_variant_disk_cache = subsample.transform.VariantDiskCache(
				directory=pathlib.Path(cfg.transform.variant_cache_dir),
				max_bytes=int(cfg.transform.max_disk_mb * 1024 * 1024),
				sample_rate=output_sample_rate,
			)
			_log.info(
				"Variant disk cache: %s (max %.0f MB)",
				cfg.transform.variant_cache_dir, cfg.transform.max_disk_mb,
			)

		_transform_processor = subsample.transform.TransformProcessor(
			sample_rate=cfg.recorder.audio.sample_rate,
			output_sample_rate=output_sample_rate,
			bit_depth=cfg.recorder.audio.bit_depth,
			on_complete=_on_transform_complete,
			on_idle=_on_transform_idle,
			disk_cache=_variant_disk_cache,
		)
		transform_manager = subsample.transform.TransformManager(
			cache=_transform_cache,
			processor=_transform_processor,
			instrument_library=instrument_library,
			cfg=cfg.transform,
			disk_cache=_variant_disk_cache,
		)

		# Auto-enqueue pitch variants for all pre-loaded instrument samples.
		# Live-captured samples are handled by the on_complete callback (below).
		# Pre-loaded startup samples are never processed by that callback, so we
		# notify the manager here, before threads start.  enqueue() returns
		# immediately; variants are computed in the background by the worker pool.
		if len(instrument_library) > 0:
			for _record in instrument_library.samples():
				transform_manager.on_sample_added(_record)

	# --- Thread-based orchestration ---
	# Both the recorder and player have blocking loops, so each runs on its own
	# thread. The main thread waits on shutdown_event and forwards Ctrl+C.

	shutdown_event = threading.Event()
	threads: list[threading.Thread] = []

	# Shared cell so the on_complete callback can call update_pitched_assignments
	# when the best-matching sample changes for a pitched keyboard assignment.
	# _start_player sets this before calling player.run().
	_player_cell: list[typing.Optional[subsample.player.MidiPlayer]] = [None]

	if cfg.recorder.enabled:
		threads.append(threading.Thread(
			target=_run_recorder,
			args=(
				cfg, reference_library, instrument_library,
				analysis_params, similarity_matrix,
				shutdown_event, cfg.player.enabled,
				transform_manager, _player_cell,
			),
			name="recorder",
		))

	if cfg.player.enabled:
		if similarity_matrix is None or reference_library is None:
			print(
				"Player enabled but no reference library configured — "
				"note-to-sample routing requires a reference library. "
				"Add a reference.directory to config.yaml.",
				file=sys.stderr,
			)
		else:
			threads.append(threading.Thread(
				target=_start_player,
				args=(
					cfg, shutdown_event, instrument_library,
					similarity_matrix, reference_library, _player_cell, transform_manager,
				),
				name="player",
			))

	if not threads:
		print("Neither recorder nor player is enabled. Nothing to do.")
		return

	# --- Directory watcher ---
	# Start after the instrument library and player are configured so the
	# on_watched_sample callback can reference all live subsystems.
	# Only active when player is enabled — the watcher's purpose is to feed
	# new samples into the playback pipeline.
	instrument_watcher: typing.Optional[subsample.watcher.InstrumentWatcher] = None

	if cfg.instrument.watch and cfg.instrument.directory is not None and cfg.player.enabled:
		watch_dir = pathlib.Path(cfg.instrument.directory)

		# Collect sidecars already loaded at startup so the watcher ignores them.
		# Resolve to absolute paths — the watcher resolves incoming paths via
		# Path.resolve(), so the comparison must use the same form.
		known_sidecars: set[pathlib.Path] = {
			(fp.parent / (fp.name + ".analysis.json")).resolve()
			for r in instrument_library.samples()
			if (fp := r.filepath) is not None
		}

		def _on_watched_sample (record: subsample.library.SampleRecord) -> None:
			_log.info("Watcher: new sample arrived — %s (%.2fs)", record.name, record.duration)
			_integrate_sample(record, instrument_library, similarity_matrix,
			                  transform_manager, _player_cell)

		instrument_watcher = subsample.watcher.InstrumentWatcher(
			directory=watch_dir,
			known_sidecars=known_sidecars,
			on_sample_loaded=_on_watched_sample,
		)
		instrument_watcher.start()
		print(f"  Watcher      : monitoring {cfg.instrument.directory} for new samples")

	# --- MIDI map file watcher ---
	# Monitors the MIDI map YAML file for changes so assignments can be
	# reloaded without restarting — enables live-coding of sample routing.
	midi_map_watcher: typing.Optional[subsample.watcher.MidiMapWatcher] = None

	if (
		cfg.player.watch_midi_map
		and cfg.player.midi_map is not None
		and cfg.player.enabled
	):
		_midi_map_watch_path = pathlib.Path(cfg.player.midi_map)

		def _on_midi_map_changed (path: pathlib.Path) -> None:

			"""Reload the MIDI map and deliver it to the active player."""

			player = _player_cell[0]

			if player is None:
				return

			assert reference_library is not None

			try:
				new_map = subsample.player.load_midi_map(
					path, reference_library.names(),
				)
			except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
				_log.warning(
					"MIDI map reload failed — keeping current map: %s", exc,
				)
				return

			player.reload_midi_map(new_map)

		midi_map_watcher = subsample.watcher.MidiMapWatcher(
			path=_midi_map_watch_path,
			on_changed=_on_midi_map_changed,
		)
		midi_map_watcher.start()
		print(f"  MIDI map     : watching {cfg.player.midi_map} for changes")

	for t in threads:
		t.start()

	try:
		# Block the main thread without spinning. Event.wait() releases the GIL
		# and responds to KeyboardInterrupt between intervals.
		while not shutdown_event.is_set():
			shutdown_event.wait(timeout=1.0)

	except KeyboardInterrupt:
		print("\nStopping…")
		shutdown_event.set()

	for t in threads:
		t.join(timeout=10.0)

	if instrument_watcher is not None:
		instrument_watcher.stop()

	if midi_map_watcher is not None:
		midi_map_watcher.stop()

	# Drain any in-flight transform workers before exiting.
	if transform_manager is not None:
		transform_manager.shutdown()

	print("Done.")


def _print_banner (cfg: subsample.config.Config) -> None:

	"""Print the startup summary line."""

	modes = []
	if cfg.recorder.enabled:
		modes.append("recorder")
	if cfg.player.enabled:
		modes.append("player")
	mode_str = " + ".join(modes) if modes else "file-only"

	# channels may be None when auto-detect is configured; show "auto" until resolved.
	ch_str = (
		f"{cfg.recorder.audio.channels}ch"
		if cfg.recorder.audio.channels is not None
		else "auto"
	)

	print(
		f"Subsample  |  {mode_str}  |  "
		f"{cfg.recorder.audio.sample_rate} Hz  "
		f"{cfg.recorder.audio.bit_depth}-bit  "
		f"{ch_str}  |  "
		f"buffer {cfg.recorder.buffer.max_seconds}s  |  "
		f"SNR ≥ {cfg.detection.snr_threshold_db} dB  |  "
		f"→ {cfg.output.directory}"
	)


def _integrate_sample (
	record: subsample.library.SampleRecord,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]],
) -> None:

	"""Add a new sample to all live subsystems.

	Called from both the live-capture on_complete callback and the directory
	watcher whenever a new sample is ready. Adds the record to the instrument
	library (evicting the oldest if over the memory limit), updates the
	similarity matrix, notifies the transform pipeline to produce variants,
	and triggers a pitched-assignment update on the active player.

	Thread-safe: each subsystem uses an internal lock. The multi-step
	sequence (library → similarity → transforms → player) is not atomic
	across subsystems — a concurrent query between steps may see transiently
	inconsistent state (e.g. an evicted sample still present in the
	similarity matrix). This is acceptable for the current use case.
	"""

	evicted = instrument_library.add(record)
	_log.info(
		"Instrument library: %d sample(s)  [%s]",
		len(instrument_library), instrument_library.format_memory(),
	)

	if similarity_matrix is not None:
		if evicted:
			similarity_matrix.remove(evicted)
		similarity_matrix.add(record)

		scores = similarity_matrix.get_scores(record.sample_id)
		if scores:
			_log.debug(
				"Similarity: %s",
				subsample.similarity.format_similarity_scores(scores),
			)

	if transform_manager is not None:
		if evicted:
			transform_manager.on_parent_evicted(evicted)
		transform_manager.on_sample_added(record)

	if player_cell is not None and player_cell[0] is not None:
		player_cell[0].update_pitched_assignments()


def _make_on_complete (
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	store_audio: bool,
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]] = None,
) -> subsample.recorder._OnCompleteCallback:

	"""Return the on_complete callback for the live-capture SampleProcessor.

	The returned callback runs on the writer thread and must not block.
	It logs the analysis result, adds the recording to the instrument
	library, updates the similarity matrix, and notifies the transform
	pipeline so derivative variants can be produced in the background.

	Args:
		store_audio:       When True, keep PCM data in the SampleRecord. Set to
		                   cfg.player.enabled — audio is only needed for playback.
		transform_manager: Optional transform pipeline coordinator. When provided,
		                   cascade-evicts derivatives for any evicted parents and
		                   triggers auto-variant production for the new sample.
		player_cell:       Single-element list holding the active MidiPlayer, or None.
		                   When provided, update_pitched_assignments() is called after
		                   each new sample is added to the similarity matrix so pitched
		                   keyboard assignments pre-compute variants for the new best match.
	"""

	def on_complete (
		filepath: pathlib.Path,
		spectral: subsample.analysis.AnalysisResult,
		rhythm: subsample.analysis.RhythmResult,
		pitch: subsample.analysis.PitchResult,
		timbre: subsample.analysis.TimbreResult,
		level: subsample.analysis.LevelResult,
		band_energy: subsample.analysis.BandEnergyResult,
		duration: float,
		audio: numpy.ndarray,
	) -> None:

		_log.info(
			"Recorded %s  (%.2fs)\n  %s\n  %s\n  %s\n  %s\n  %s",
			filepath.name, duration,
			subsample.analysis.format_result(spectral, duration),
			subsample.analysis.format_rhythm_result(rhythm),
			subsample.analysis.format_pitch_result(pitch),
			subsample.analysis.format_level_result(level),
			subsample.analysis.format_band_energy_result(band_energy),
		)

		record = subsample.library.SampleRecord(
			sample_id   = subsample.library.allocate_id(),
			name        = filepath.stem,
			spectral    = spectral,
			rhythm      = rhythm,
			pitch       = pitch,
			timbre      = timbre,
			level       = level,
			band_energy = band_energy,
			params      = analysis_params,
			duration    = duration,
			audio       = audio if store_audio else None,
			filepath    = filepath,
		)

		_integrate_sample(record, instrument_library, similarity_matrix,
		                  transform_manager, player_cell)

	return on_complete
