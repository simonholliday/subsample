"""Entry point and main orchestration loop for Subsample.

Ties together config loading, device selection, the circular buffer,
the level detector, the WAV writer, and the MIDI player. Supports two
input modes and two run modes:

  File input   — pass audio file paths as positional arguments; each
                 file is processed through the detection pipeline and
                 segments are written to the output directory as new
                 ``.wav`` files with their analysis sidecars.  The
                 process then exits — the chopped segments are picked
                 up at the next subsample startup, when the recursive
                 library load discovers them.

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
import time
import typing

import numpy
import yaml

import subsample.analysis
import subsample.audio
import subsample.bank
import subsample.cache
import subsample.buffer
import subsample.config
import subsample.detector
import subsample.events
import subsample.library
import subsample.osc
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

# Cadence for in-loop progress logging during file ingestion.
_PROGRESS_LOG_INTERVAL_SEC: typing.Final[float] = 10.0


def _format_mmss (seconds: float) -> str:

	"""Format a non-negative duration in seconds as mm:ss (zero-padded).

	Minutes are not capped — a 100-minute duration renders as '100:00'.
	"""

	s = int(round(max(0.0, seconds)))
	return f"{s // 60:02d}:{s % 60:02d}"


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


def _build_trimmed_segment (
	buf: subsample.buffer.CircularBuffer,
	detector: subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
	sample_rate: int,
	start_frame: int,
	end_frame: int,
) -> typing.Optional[numpy.ndarray]:

	"""Read [start_frame, end_frame] from the buffer and trim both edges.

	Shared by the per-chunk completion path and the end-of-file flush so both
	produce identically-trimmed segments.
	"""

	pre = detection_cfg.trim_pre_samples
	# Read back `pre` extra frames before the detector's start boundary so
	# trim_silence has raw audio available for its fade-in window.
	segment = buf.read_range(max(0, start_frame - pre), end_frame)

	# Tail edge uses the CLOSE threshold (keeps a decayed tail); attack edge uses
	# the snr (open) threshold (trims tight to the transient) — both owned by the
	# detector so they cannot drift from what it ended on.
	fade_out_samples = round(detection_cfg.fade_out_ms / 1000.0 * sample_rate)

	trimmed = subsample.trim.trim_silence(
		segment,
		detector.tail_amplitude_threshold,
		pre_samples=detection_cfg.trim_pre_samples,
		post_samples=detection_cfg.trim_post_samples,
		fade_out_samples=fade_out_samples,
		lead_amplitude_threshold=detector.attack_amplitude_threshold,
	)

	return trimmed if trimmed.size > 0 else None


def _process_chunk (
	chunk: numpy.ndarray,
	buf: subsample.buffer.CircularBuffer,
	detector: subsample.detector.LevelDetector,
	detection_cfg: subsample.config.DetectionConfig,
	sample_rate: int,
) -> typing.Optional[numpy.ndarray]:

	"""Feed one audio chunk through the detection pipeline.

	Writes the chunk to the circular buffer, runs the level detector, and
	if a recording is completed, retrieves and trims the segment.

	Args:
		chunk:         One audio chunk, shape (chunk_size, channels).
		buf:           Circular buffer receiving the audio stream.
		detector:      Level detector tracking ambient noise and recording state.
		detection_cfg: Detection settings (SNR threshold, trim params, etc.).
		sample_rate:   Stream sample rate, used to convert fade_out_ms to samples.

	Returns:
		Trimmed audio segment as a numpy array, or None if no recording completed.
	"""

	buf.write(chunk)

	result = detector.process_chunk(chunk, buf.frames_written)

	if result is None:
		return None

	start_frame, end_frame = result
	return _build_trimmed_segment(buf, detector, detection_cfg, sample_rate, start_frame, end_frame)


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
			file_info = subsample.audio.read_audio_file(
				path, float_ceiling_dbfs=cfg.recorder.audio.float_import_ceiling_dbfs,
			)
		except (OSError, ValueError) as exc:
			_log.warning("Could not read %s: %s — skipping", path.name, exc)
			continue

		print(f"  {file_info.sample_rate} Hz  {file_info.bit_depth}-bit  {file_info.channels}ch")

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

		writer = subsample.recorder.SampleProcessor(cfg, analysis_params, on_complete=None, warn_backlog=False)

		segment_index = 1
		chunk_size = cfg.recorder.audio.chunk_size
		n_frames = file_info.audio.shape[0]
		file_duration_sec = n_frames / file_info.sample_rate

		_log.info("File duration: %s", _format_mmss(file_duration_sec))

		start_time = time.monotonic()
		last_progress_time = start_time

		# try/finally so an error in the read/drain loop still shuts down the
		# worker pool (this is the one SampleProcessor owner without the
		# teardown guarantee its siblings have).
		try:

			# Phase 1 — read pass: stream the file through the detector, dispatching
			# detected segments to the worker pool.  Fast for in-memory files
			# (essentially memory bandwidth + detector cost), so the periodic tick
			# below will only fire for very large files.
			for offset in range(0, n_frames, chunk_size):
				chunk = file_info.audio[offset : offset + chunk_size]

				trimmed = _process_chunk(chunk, buf, detector, cfg.detection, file_info.sample_rate)

				if trimmed is not None:
					writer.enqueue(
						trimmed,
						datetime.datetime.now(),
						filename_base=f"{path.stem}_{segment_index}",
						sample_rate=file_info.sample_rate,
						bit_depth=file_info.bit_depth,
					)
					segment_index += 1

				now = time.monotonic()
				if now - last_progress_time >= _PROGRESS_LOG_INTERVAL_SEC:
					position_sec = min(file_duration_sec, (offset + chunk_size) / file_info.sample_rate)
					pct = 100.0 * position_sec / file_duration_sec if file_duration_sec > 0 else 100.0
					wall_elapsed = now - start_time
					# Extrapolate remaining wall time from the ratio of audio read so far.
					if position_sec > 0:
						eta_wall = wall_elapsed * (file_duration_sec - position_sec) / position_sec
					else:
						eta_wall = 0.0
					_log.info(
						"Reading: %.0f%% — elapsed %s, ETA %s",
						pct, _format_mmss(wall_elapsed), _format_mmss(eta_wall),
					)
					last_progress_time = now

			# End of file: flush a recording still open (its last sound ran to
			# within hold_time of EOF) so the final segment is not silently lost.
			flush = detector.finalize(buf.frames_written)
			if flush is not None:
				trimmed = _build_trimmed_segment(
					buf, detector, cfg.detection, file_info.sample_rate, flush[0], flush[1],
				)
				if trimmed is not None:
					writer.enqueue(
						trimmed,
						datetime.datetime.now(),
						filename_base=f"{path.stem}_{segment_index}",
						sample_rate=file_info.sample_rate,
						bit_depth=file_info.bit_depth,
					)
					segment_index += 1

			# Phase 2 — drain: wait for the worker pool to finish analysing and
			# writing every enqueued segment.  For long files this is where most
			# of the wall time lives, so we poll queue_depth and log progress.
			total_segments = segment_index - 1
			if total_segments > 0:
				drain_start = time.monotonic()
				while True:
					depth = writer.queue_depth
					if depth == 0:
						break

					now = time.monotonic()
					if now - last_progress_time >= _PROGRESS_LOG_INTERVAL_SEC:
						completed = total_segments - depth
						wall_elapsed = now - start_time
						drain_elapsed = now - drain_start
						if completed > 0:
							eta_wall = drain_elapsed * depth / completed
						else:
							eta_wall = 0.0
						pct = 100.0 * completed / total_segments
						_log.info(
							"Processing: %d/%d segments (%.1f%%) — elapsed %s, ETA %s",
							completed, total_segments, pct,
							_format_mmss(wall_elapsed), _format_mmss(eta_wall),
						)
						last_progress_time = now

					time.sleep(1.0)

			writer.flush()

			wall_total = time.monotonic() - start_time
			count = segment_index - 1
			print(f"  → {count} segment(s) processed from {path.name} in {_format_mmss(wall_total)}")
		finally:
			writer.shutdown()


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
	app_events: typing.Optional[subsample.events.EventEmitter] = None,
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
		app_events:         Optional event emitter; forwarded to _make_on_complete
		                    so sample_captured and sample_loaded events are emitted.
	"""

	pa = subsample.audio.create_pyaudio()

	try:
		devices = subsample.audio.list_input_devices(pa)

		if cfg.recorder.audio.device is not None:
			try:
				device_index = subsample.audio.find_device_by_name(pa, cfg.recorder.audio.device)
			except ValueError:
				_log.warning(
					"Configured audio input device %r not found — prompting for selection",
					cfg.recorder.audio.device,
				)
				device_index = subsample.audio.select_device(devices)
		else:
			device_index = subsample.audio.select_device(devices)

		# Resolve channel count and input routing.
		audio_cfg = cfg.recorder.audio
		detected_channels = subsample.audio.get_device_channels(pa, device_index)

		if audio_cfg.input is not None:
			# Validate input indices against the device's actual channel count.
			for idx in audio_cfg.input:
				if idx >= detected_channels:
					raise ValueError(
						f"recorder.audio.input channel {idx + 1} exceeds device's "
						f"{detected_channels} input channel(s)"
					)

			if audio_cfg.channels is None:
				audio_cfg = dataclasses.replace(audio_cfg, channels=len(audio_cfg.input))

		elif audio_cfg.channels is None:
			# No explicit channels or input routing — auto-detect.
			if detected_channels >= 3:
				# Multi-channel device: prompt user to choose inputs.
				device_info = pa.get_device_info_by_index(device_index)
				selected = subsample.audio.select_input_channels(
					str(device_info["name"]), detected_channels,
				)
				audio_cfg = dataclasses.replace(
					audio_cfg,
					channels=len(selected),
					input=selected,
				)
			else:
				# 1-2 channel device: use all channels.
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

	on_complete_callback = _make_on_complete(
		reference_library, instrument_library, analysis_params,
		similarity_matrix, store_audio, transform_manager,
		player_cell=player_cell,
		app_events=app_events,
	)

	writer = subsample.recorder.SampleProcessor(
		cfg,
		analysis_params,
		on_complete=on_complete_callback,
	)

	print(f"Calibrating ambient noise for {cfg.detection.warmup_seconds:.0f}s…")

	try:
		while not shutdown_event.is_set():
			chunk = reader.read(timeout=0.5)

			if chunk is None:
				# Timeout — loop back to check shutdown_event.
				continue

			trimmed = _process_chunk(chunk, buf, detector, cfg.detection, audio_cfg.sample_rate)

			if trimmed is not None:
				writer.enqueue(trimmed, datetime.datetime.now())

	finally:
		if reader.overflow_count > 0:
			_log.warning(
				"Audio overflows detected during capture: %d — "
				"recordings may contain discontinuities",
				reader.overflow_count,
			)

		# Isolate each teardown step: if reader.stop() raises (e.g. PortAudio
		# error on an unplugged USB device), pa.terminate() and the writer's
		# drain/flush must still run rather than leak the stream and lose
		# in-flight recordings.
		try:
			reader.stop()
		except Exception as exc:
			_log.error("Recorder: reader.stop() failed during shutdown: %s", exc)

		try:
			pa.terminate()
		except Exception as exc:
			_log.error("Recorder: pa.terminate() failed during shutdown: %s", exc)

		writer.shutdown()


def _load_bank (
	defn: subsample.bank.BankDefinition,
	reference_library: subsample.library.ReferenceLibrary,
	cfg: subsample.config.Config,
	output_sample_rate: int,
	parent_map_dir: pathlib.Path,
) -> subsample.bank.Bank:

	"""Load a single program: library, similarity matrix, and transform pipeline.

	Two forms (mutually exclusive on the BankDefinition):

	  - ``directory:`` shorthand — load the directory as the sample pool;
	    the program reuses the player's top-level assignments.  The returned
	    Bank carries ``note_map=None`` (use the global rules).
	  - ``map:`` preset — load a whole mapper file (its own assignments +
	    samples).  The library is populated from the preset's own path /
	    ``directory:`` predicates (resolved relative to the preset folder),
	    and the returned Bank carries the preset's note_map / zone_templates /
	    mapped_ccs so a Program Change swaps the rules too.

	Args:
		defn:               Parsed program definition from the MIDI map.
		reference_library:  Shared reference library for similarity scoring.
		cfg:                Full application config (memory limits, transform settings).
		output_sample_rate: Effective player output sample rate for variant resampling.
		parent_map_dir:     Directory of the top-level MIDI map, used to
		                    resolve a relative ``map:`` preset path.

	Returns:
		Fully loaded Bank ready for playback.

	Raises:
		ValueError: If a ``map:`` preset path is missing or itself declares
		a nested ``programs:`` block.
	"""

	max_instrument_bytes = int(cfg.instrument.max_memory_mb * 1024 * 1024)

	preset_result: typing.Optional[subsample.player.MidiMapResult] = None

	if defn.map_path is not None:
		# `map:` preset — load the mapper file; the library starts empty and
		# is filled by the preset's own path / directory references below.
		preset_path = parent_map_dir / defn.map_path
		if not preset_path.exists():
			raise ValueError(
				f"Program {defn.name!r}: preset map {defn.map_path!r} not found "
				f"(resolved to {preset_path})"
			)
		preset_result = subsample.player.load_midi_map(
			preset_path, reference_library.names(), strict=cfg.player.strict_midi_map,
		)
		if preset_result.bank_definitions:
			raise ValueError(
				f"Program {defn.name!r}: preset {defn.map_path!r} declares its own "
				f"'programs:' — nested presets are not allowed"
			)
		directory = preset_path.parent
		instrument_library = subsample.library.InstrumentLibrary(max_instrument_bytes)
	else:
		# `directory:` shorthand — the directory IS the pool.
		directory = pathlib.Path(typing.cast(str, defn.directory))
		instrument_library = subsample.library.load_instrument_library(
			directory,
			max_instrument_bytes,
			load_audio=True,
			with_preview=cfg.recorder.previews,
			target_sample_rate=output_sample_rate,
		)

	# Similarity matrix (per-bank — rankings are relative to each bank's samples).
	similarity_matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
	if len(instrument_library) > 0:
		similarity_matrix.bulk_add(instrument_library.samples())

	# Transform pipeline (per-bank).
	max_transform_bytes = int(cfg.transform.max_memory_mb * 1024 * 1024)

	transform_cache = subsample.transform.TransformCache(
		max_memory_bytes=max_transform_bytes,
	)

	def _on_transform_complete (result: subsample.transform.TransformResult) -> None:
		transform_cache.put(result)

	def _on_transform_idle (completed: int) -> None:
		_log.info(
			"Transform queue idle [%s] — %d variant(s) processed  [cache: %s]",
			defn.name, completed, transform_cache.format_memory(),
		)

	variant_disk_cache: typing.Optional[subsample.transform.VariantDiskCache] = None
	if cfg.transform.variant_cache_dir and cfg.transform.max_disk_mb > 0:
		variant_disk_cache = subsample.transform.VariantDiskCache(
			directory=pathlib.Path(cfg.transform.variant_cache_dir),
			max_bytes=int(cfg.transform.max_disk_mb * 1024 * 1024),
			sample_rate=output_sample_rate,
		)

	transform_processor = subsample.transform.TransformProcessor(
		sample_rate=cfg.recorder.audio.sample_rate,
		output_sample_rate=output_sample_rate,
		bit_depth=cfg.recorder.audio.bit_depth,
		on_complete=_on_transform_complete,
		on_idle=_on_transform_idle,
		disk_cache=variant_disk_cache,
	)

	transform_manager = subsample.transform.TransformManager(
		cache=transform_cache,
		processor=transform_processor,
		instrument_library=instrument_library,
		cfg=cfg.transform,
		disk_cache=variant_disk_cache,
	)

	# For a `map:` preset, populate the (empty) library + similarity matrix
	# from the preset's own path / directory references — these resolve
	# relative to the preset folder (load_midi_map stamped the preset's
	# midi_map_dir into each directory predicate), so a self-contained kit
	# folder loads with no extra coupling.  Then validate its extracts.
	if preset_result is not None:
		subsample.player._resolve_path_references(
			preset_result.note_map, [similarity_matrix], instrument_library,
			target_sample_rate=output_sample_rate,
			with_preview=cfg.recorder.previews,
		)
		subsample.player._validate_assignment_extracts(preset_result.note_map, instrument_library)
		if len(preset_result.note_map) == 0:
			_log.warning("Program %r preset %s has no assignments", defn.name, defn.map_path)
		elif len(instrument_library) == 0:
			_log.warning(
				"Program %r preset %s loaded no samples — check its 'directory:' predicates",
				defn.name, defn.map_path,
			)

	# Auto-enqueue variants for loaded samples.
	if len(instrument_library) > 0:
		for record in instrument_library.samples():
			transform_manager.on_sample_added(record)

	_log.info(
		"Program %r loaded: %d sample(s) from %s  [%s]",
		defn.name, len(instrument_library), directory,
		instrument_library.format_memory(),
	)

	preset_zone_templates: typing.Optional[tuple[typing.Any, ...]] = None
	preset_mapped_ccs:     typing.Optional[set[int]]               = None
	if preset_result is not None:
		preset_zone_templates = preset_result.zone_templates
		preset_mapped_ccs     = subsample.player._collect_mapped_ccs(
			preset_result.note_map, preset_result.zone_templates,
		)

	return subsample.bank.Bank(
		name=defn.name,
		directory=directory,
		program=defn.program,
		instrument_library=instrument_library,
		similarity_matrix=similarity_matrix,
		transform_manager=transform_manager,
		note_map=preset_result.note_map if preset_result is not None else None,
		zone_templates=preset_zone_templates,
		mapped_ccs=preset_mapped_ccs,
	)


def _apply_active_preset_rules (
	player: subsample.player.MidiPlayer,
	bank_manager: typing.Optional[subsample.bank.BankManager],
) -> None:

	"""Install the active program's preset rules at startup, if it carries any.

	The player is constructed with the top-level note_map / zone_templates,
	but the BankManager's active program (e.g. a ``map:`` default_program) may
	carry its own.  Swap them in before playback so the default program's
	rules are live from the first note rather than only after the first
	Program Change.
	"""

	if bank_manager is None:
		return

	active = bank_manager.active_bank
	if active.note_map is None:
		return

	player._apply_rule_set(active.note_map, active.zone_templates or (), active.mapped_ccs or set())


def _start_player (
	cfg: subsample.config.Config,
	shutdown_event: threading.Event,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: subsample.similarity.SimilarityMatrix,
	reference_library: subsample.library.ReferenceLibrary,
	player_cell: list[typing.Optional[subsample.player.MidiPlayer]],
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	bank_manager: typing.Optional[subsample.bank.BankManager] = None,
	sv_cell: typing.Optional[list[typing.Any]] = None,
	preloaded_midi_map_result: typing.Optional[subsample.player.MidiMapResult] = None,
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
		bank_manager:       Optional bank manager for multi-bank switching via MIDI
		                    Program Change.
		sv_cell:            Single-element list holding the Supervisor instance, or None.
		                    When present, the player's CC events are subscribed after
		                    the player is created.
	"""

	# Load the MIDI routing map.  Requires an explicit path in config —
	# no hidden fallback.  A new user must set player.midi_map to get output.
	if cfg.player.midi_map is None:
		print(
			"Player enabled but no MIDI map configured — "
			"set player.midi_map in config.yaml "
			"(e.g. midi_map: \"./midi-map-gm-drums.yaml\").",
			file=sys.stderr,
		)
		_log.warning("player.midi_map is not set — player will not start")
		return

	_midi_map_path = pathlib.Path(cfg.player.midi_map)

	# Reuse the parse done at startup for bank detection — the parser is
	# pure and the reference list it sees is the same here as there
	# (path-based references go to similarity matrices, not the reference
	# library, so reference_library.names() stays empty across the two
	# call sites).  Saves the second parse and the duplicate INFO logs it
	# emits for similarity auto-injection.
	if preloaded_midi_map_result is not None:
		midi_map_result = preloaded_midi_map_result
	else:
		try:
			midi_map_result = subsample.player.load_midi_map(
				_midi_map_path,
				reference_library.names(),
				strict=cfg.player.strict_midi_map,
			)
		except (FileNotFoundError, ValueError) as exc:
			print(f"Error loading MIDI map: {exc}", file=sys.stderr)
			return

	midi_map = midi_map_result.note_map

	# Resolve path-based references and instruments from the MIDI map
	matrices: list[subsample.similarity.SimilarityMatrix] = []
	if bank_manager is not None:
		# Multi-bank mode: resolve into each bank's matrix
		for bank in bank_manager.all_banks():
			matrices.append(bank.similarity_matrix)
	else:
		# Single-bank mode: use the global similarity matrix
		matrices.append(similarity_matrix)

	# Resolve the effective output sample rate for resampling loaded samples.
	effective_output_sr = (
		cfg.player.audio.sample_rate
		if cfg.player.audio.sample_rate is not None
		else cfg.recorder.audio.sample_rate
	)

	subsample.player._resolve_path_references(
		midi_map, matrices, instrument_library,
		target_sample_rate=effective_output_sr,
		with_preview=cfg.recorder.previews,
	)

	# Validate `extract:` directives now that all candidate samples are
	# loaded: any assignment whose extract is incompatible with even one of
	# its matching samples is rejected here, before audio playback begins.
	subsample.player._validate_assignment_extracts(midi_map, instrument_library)

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
			bank_manager=bank_manager,
			target_bpm=cfg.transform.target_bpm,
			tempo_source=cfg.transform.tempo_source,
			output_channels=cfg.player.audio.channels,
			ambisonic_config=cfg.ambisonic,
			buffer_frames=cfg.player.audio.buffer_frames,
			zone_templates=midi_map_result.zone_templates,
		)
		player_cell[0] = player

		if sv_cell is not None and sv_cell[0] is not None:
			player.events.on("cc", sv_cell[0]._on_cc)

		_apply_active_preset_rules(player, bank_manager)
		player.update_pitched_assignments()
		try:
			player.run()
		except (ValueError, OSError) as exc:
			# OSError covers PortAudio rejecting the device/format and mido
			# failing to bind the port — catch it so a device-open failure prints
			# a clean message instead of escaping this thread target as a raw
			# traceback.  Returning ends the thread; the main loop notices all
			# subsystem threads have stopped and exits (see the wait loop) rather
			# than parking on shutdown_event forever.
			print(f"\nError starting player: {exc}", file=sys.stderr)

		return

	# Hardware port mode: resolve device name from config or interactive menu.
	try:
		devices = subsample.player.list_midi_input_devices()

		if cfg.player.midi_device is not None:
			try:
				device_name = subsample.player.find_midi_device_by_name(cfg.player.midi_device)
			except ValueError:
				_log.warning(
					"Configured MIDI device %r not found — prompting for selection",
					cfg.player.midi_device,
				)
				device_name = subsample.player.select_midi_device(devices)
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
		bank_manager=bank_manager,
		target_bpm=cfg.transform.target_bpm,
		tempo_source=cfg.transform.tempo_source,
		output_channels=cfg.player.audio.channels,
		ambisonic_config=cfg.ambisonic,
		buffer_frames=cfg.player.audio.buffer_frames,
		zone_templates=midi_map_result.zone_templates,
	)
	player_cell[0] = player

	if sv_cell is not None and sv_cell[0] is not None:
		player.events.on("cc", sv_cell[0]._on_cc)

	_apply_active_preset_rules(player, bank_manager)
	player.update_pitched_assignments()
	try:
		player.run()
	except (ValueError, OSError) as exc:
		# OSError: PortAudio/mido device-open failure — see the virtual-port
		# branch above for why this thread target must not let it escape.
		print(f"\nError starting player: {exc}", file=sys.stderr)


def main () -> None:

	"""Entry point — runs the ambient audio sampler."""

	_main_impl()


def _main_impl () -> None:

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

	# Wire the carrier cache budget from the resolved config.
	subsample.transform.set_carrier_cache_budget(
		int(cfg.transform.carrier_memory_mb * 1024 * 1024)
	)

	# Wire the float import ceiling so every read path — library, cache, watcher,
	# OSC import — treats a hot float source the same way, not just CLI import.
	subsample.audio.set_float_import_ceiling(
		cfg.recorder.audio.float_import_ceiling_dbfs
	)

	# Wire the analysis tempo priors so the file/library/watcher heal path honours
	# analysis.start_bpm / tempo_min / tempo_max, not just live capture.
	subsample.cache.set_analysis_config(cfg.analysis)

	# Wire the default quantize grid so transform.quantize_resolution actually
	# takes effect (it was documented but previously never read).
	subsample.player.set_default_quantize_grid(cfg.transform.quantize_resolution)

	_log.info(
		"Memory budget: instrument %.0f MB, transform %.0f MB, carrier %.0f MB, disk %.0f MB",
		cfg.instrument.max_memory_mb, cfg.transform.max_memory_mb,
		cfg.transform.carrier_memory_mb, cfg.transform.max_disk_mb,
	)

	_print_banner(cfg)

	# --- Application event emitter ---
	# Integrations (OSC sender, Supervisor dashboard, etc.) subscribe here
	# instead of being manually wired into each callback chain.
	app_events = subsample.events.EventEmitter()

	if cfg.osc.enabled:
		try:
			_osc_sender = subsample.osc.OscEventSender(cfg.osc.send_host, cfg.osc.send_port)
			app_events.on("sample_captured", _osc_sender.on_sample_captured_event)
			app_events.on("sample_loaded", _osc_sender.on_sample_loaded_event)
			print(f"  OSC sender   : sending to {cfg.osc.send_host}:{cfg.osc.send_port}")
		except ImportError:
			_log.warning("OSC enabled but python-osc not installed. pip install subsample[osc]")

	# Reference library starts empty.  Path-based references declared in
	# the MIDI map are loaded later by _resolve_path_references() during
	# player startup, which adds them to the similarity matrix dynamically.
	reference_library = subsample.library.ReferenceLibrary([])

	# Process any input files through the detection pipeline, then exit.
	# Segments are written to the output directory and analysed; the recursive
	# library loader will pick them up on the next subsample startup.  This is
	# deliberate — file-input mode is a batch chop, not a hybrid live session.
	if args.files:
		# Recorder banner shows the live-capture config; file-input mode reads
		# at each source's native format.  Clarify so a 32-bit float source
		# isn't mistaken for "24-bit" because that's what the config says.
		print("File input mode: reading each file at its native sample rate, bit depth, channel count.")
		_process_input_files(args.files, cfg)
		return

	# --- Bank detection ---
	# Pre-load the MIDI map to check for bank definitions before loading
	# instrument libraries.  Banks override cfg.instrument.directory.
	bank_manager: typing.Optional[subsample.bank.BankManager] = None
	bank_definitions: list[subsample.bank.BankDefinition] = []
	bank_channel: int = subsample.bank.DEFAULT_BANK_CHANNEL
	default_bank: typing.Optional[int] = None
	preloaded_midi_map_result: typing.Optional[subsample.player.MidiMapResult] = None

	if cfg.player.enabled and cfg.player.midi_map is not None:
		_midi_map_path = pathlib.Path(cfg.player.midi_map)
		try:
			preloaded_midi_map_result = subsample.player.load_midi_map(
				_midi_map_path, [], strict=cfg.player.strict_midi_map,
			)
			bank_definitions = preloaded_midi_map_result.bank_definitions
			bank_channel = preloaded_midi_map_result.bank_channel
			default_bank = preloaded_midi_map_result.default_bank
		except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
			_log.warning("Could not pre-load MIDI map for bank detection: %s", exc)

	# Resolve the effective output sample rate for the player.
	output_sample_rate = (
		cfg.player.audio.sample_rate
		if cfg.player.audio.sample_rate is not None
		else cfg.recorder.audio.sample_rate
	)

	# Declare shared variables before the bank/single-directory branch.
	max_instrument_bytes = int(cfg.instrument.max_memory_mb * 1024 * 1024)
	instrument_library:  subsample.library.InstrumentLibrary
	similarity_matrix:   typing.Optional[subsample.similarity.SimilarityMatrix] = None
	transform_manager:   typing.Optional[subsample.transform.TransformManager] = None

	# --- Multi-bank loading ---
	# When the MIDI map declares banks, load each one independently.
	# cfg.instrument.directory is ignored (banks take precedence).
	if bank_definitions:
		_log.info(
			"MIDI map declares %d program(s) — ignoring instrument.directory (%s)",
			len(bank_definitions), cfg.instrument.directory,
		)

		parent_map_dir = pathlib.Path(typing.cast(str, cfg.player.midi_map)).parent

		banks: list[subsample.bank.Bank] = []
		for defn in bank_definitions:
			bank = _load_bank(defn, reference_library, cfg, output_sample_rate, parent_map_dir)
			banks.append(bank)
			source = defn.map_path if defn.map_path is not None else defn.directory
			print(
				f"  Program {defn.program:<3d}  : {defn.name!r} — "
				f"{len(bank.instrument_library)} sample(s) from {source}"
			)

		bank_manager = subsample.bank.BankManager(banks, bank_channel, default_program=default_bank)

		# Eager-load memory guard (WARN only — never blocks startup).  A
		# program whose audio cannot stay fully resident within
		# instrument.max_memory_mb is evicting at load, so the first triggers
		# after a switch to it will reload from disk (lag).  And because each
		# program's library is capped independently, N programs can use up to
		# N× the configured budget in aggregate.
		for bank in banks:
			if bank.instrument_library.memory_used >= bank.instrument_library.memory_limit:
				_log.warning(
					"Program %r exceeds instrument.max_memory_mb (%.0f MB) — samples "
					"will reload on switch (lag); raise the limit to keep it resident",
					bank.name, cfg.instrument.max_memory_mb,
				)
		total_used = sum(b.instrument_library.memory_used for b in banks)
		if total_used > max_instrument_bytes:
			_log.warning(
				"Programs use %.0f MB of audio in total — more than the "
				"instrument.max_memory_mb budget (%.0f MB); eager-loading every "
				"program multiplies the budget by the program count",
				total_used / (1024 * 1024), cfg.instrument.max_memory_mb,
			)

		# The primary instrument_library/similarity/transform used by the
		# recorder on_complete callback come from the first program. Captures
		# directed at its directory are also picked up by that program's
		# watcher (see below).
		instrument_library  = banks[0].instrument_library
		similarity_matrix   = banks[0].similarity_matrix
		transform_manager   = banks[0].transform_manager

		print(
			f"  Programs     : {len(banks)} loaded — "
			f"switch via Program Change on ch {bank_channel}"
		)

	# --- Single-directory mode (no banks) ---
	else:
		# Create instrument library. PCM audio is only needed when the player is
		# active — skipping it saves memory when player is disabled.
		if cfg.player.enabled:
			instrument_library = subsample.library.load_instrument_library(
				pathlib.Path(cfg.instrument.directory),
				max_instrument_bytes,
				load_audio=True,
				with_preview=cfg.recorder.previews,
				target_sample_rate=output_sample_rate,
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

		# Build the similarity matrix.  It starts empty; path-based references
		# from the MIDI map are added dynamically during player startup via
		# _resolve_path_references().
		if cfg.player.enabled:
			similarity_matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
			if len(instrument_library) > 0:
				similarity_matrix.bulk_add(instrument_library.samples())
			print(f"  Similarity   : {similarity_matrix}")

		# --- Transform pipeline ---
		if cfg.player.enabled:
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

			if len(instrument_library) > 0:
				for _record in instrument_library.samples():
					transform_manager.on_sample_added(_record)

	analysis_params = subsample.analysis.compute_params(cfg.recorder.audio.sample_rate)

	# --- Thread-based orchestration ---
	# Both the recorder and player have blocking loops, so each runs on its own
	# thread. The main thread waits on shutdown_event and forwards Ctrl+C.

	shutdown_event = threading.Event()
	threads: list[threading.Thread] = []

	# Shared cell so the on_complete callback can call update_assignments
	# when the best-matching sample changes for a pitched keyboard assignment.
	# _start_player sets this before calling player.run().
	_player_cell: list[typing.Optional[subsample.player.MidiPlayer]] = [None]

	# --- Supervisor dashboard ---
	# Broadcasts state via WebSocket.  Created before threads start so
	# sample events can be subscribed immediately.  The player reference
	# is resolved lazily via _player_cell (the player is created on a
	# separate thread).  CC subscription happens inside _start_player
	# after the player exists.
	_sv_cell: list[typing.Any] = [None]

	if cfg.supervisor.enabled:
		try:
			import supervisor.app.subsample as _sv_module
			_sv_cell[0] = _sv_module.SubsampleSupervisor(
				player=_player_cell,
				instrument_library=instrument_library,
				recorder_processor=None,
				cfg=cfg,
				port=cfg.supervisor.port,
			)
			app_events.on("sample_captured", _sv_cell[0].on_sample_captured)
			app_events.on("sample_loaded", _sv_cell[0].on_sample_loaded)
			_sv_cell[0].start_threaded()
			print(f"  Supervisor   : ws://localhost:{cfg.supervisor.port}")
		except ImportError:
			_log.warning("Supervisor enabled but not installed. pip install subsample[supervisor]")
		except OSError as exc:
			# A busy WebSocket port (or similar bind failure) must not abort
			# the whole startup — mirror the OSC receiver's degradation.
			_log.warning(
				"Supervisor could not start on port %d: %s — dashboard disabled",
				cfg.supervisor.port, exc,
			)

	if cfg.recorder.enabled:
		threads.append(threading.Thread(
			target=_run_recorder,
			args=(
				cfg, reference_library, instrument_library,
				analysis_params, similarity_matrix,
				shutdown_event, cfg.player.enabled,
				transform_manager, _player_cell,
				app_events,
			),
			name="recorder",
		))

	if cfg.player.enabled:
		if similarity_matrix is None:
			print(
				"Player enabled but similarity matrix could not be created.",
				file=sys.stderr,
			)
		else:
			threads.append(threading.Thread(
				target=_start_player,
				args=(
					cfg, shutdown_event, instrument_library,
					similarity_matrix, reference_library, _player_cell,
					transform_manager, bank_manager, _sv_cell,
					preloaded_midi_map_result,
				),
				name="player",
			))

	if not threads:
		print("Neither recorder nor player is enabled. Nothing to do.")
		return

	# --- Directory watchers ---
	# Start after the instrument library and player are configured so the
	# on_watched_sample callback can reference all live subsystems.
	# Only active when player is enabled — the watcher's purpose is to feed
	# new samples into the playback pipeline.
	instrument_watchers: list[subsample.watcher.InstrumentWatcher] = []

	if cfg.instrument.watch and cfg.player.enabled:

		# Multi-bank mode: one watcher per bank directory.
		if bank_manager is not None:
			for bank in bank_manager.all_banks():
				_bank = bank  # capture for closure

				known_sc = {
					(fp.parent / (fp.name + subsample.cache.SIDECAR_SUFFIX)).resolve()
					for r in _bank.instrument_library.samples()
					if (fp := r.filepath) is not None
				}

				known_au = {
					fp.resolve()
					for r in _bank.instrument_library.samples()
					if (fp := r.filepath) is not None
				}

				def _make_bank_callback (b: subsample.bank.Bank) -> typing.Callable[[subsample.library.SampleRecord], None]:
					def cb (record: subsample.library.SampleRecord) -> None:
						_log.info("Watcher [%s]: new sample — %s (%.2fs)", b.name, record.name, record.duration)
						_integrate_sample(record, b.instrument_library, b.similarity_matrix,
						                  b.transform_manager, _player_cell, app_events)

					return cb

				def _make_bank_removal_callback (b: subsample.bank.Bank) -> typing.Callable[[pathlib.Path], None]:
					def rm (path: pathlib.Path) -> None:
						_log.info("Watcher [%s]: sample removed — %s", b.name, path.name)
						_remove_sample(path, b.instrument_library, b.similarity_matrix,
						               b.transform_manager, _player_cell)

					return rm

				watcher = subsample.watcher.InstrumentWatcher(
					directory=_bank.directory,
					known_sidecars=known_sc,
					on_sample_loaded=_make_bank_callback(_bank),
					target_sample_rate=output_sample_rate,
					known_audio=known_au,
					with_preview=cfg.recorder.previews,
					on_sample_removed=_make_bank_removal_callback(_bank),
				)
				watcher.start()
				instrument_watchers.append(watcher)
				print(f"  Watcher      : monitoring {_bank.directory} ({_bank.name!r})")

		# Single-directory mode.
		else:
			known_sidecars: set[pathlib.Path] = {
				(fp.parent / (fp.name + subsample.cache.SIDECAR_SUFFIX)).resolve()
				for r in instrument_library.samples()
				if (fp := r.filepath) is not None
			}

			known_audio_paths: set[pathlib.Path] = {
				fp.resolve()
				for r in instrument_library.samples()
				if (fp := r.filepath) is not None
			}

			def _on_watched_sample (record: subsample.library.SampleRecord) -> None:
				_log.info("Watcher: new sample arrived — %s (%.2fs)", record.name, record.duration)
				_integrate_sample(record, instrument_library, similarity_matrix,
				                  transform_manager, _player_cell, app_events)

			def _on_watched_sample_removed (path: pathlib.Path) -> None:
				_log.info("Watcher: sample removed — %s", path.name)
				_remove_sample(path, instrument_library, similarity_matrix,
				               transform_manager, _player_cell)

			watcher = subsample.watcher.InstrumentWatcher(
				directory=pathlib.Path(cfg.instrument.directory),
				known_sidecars=known_sidecars,
				on_sample_loaded=_on_watched_sample,
				target_sample_rate=output_sample_rate,
				known_audio=known_audio_paths,
				with_preview=cfg.recorder.previews,
				on_sample_removed=_on_watched_sample_removed,
			)
			watcher.start()
			instrument_watchers.append(watcher)
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

		# Snapshot the bank state at startup so the live-reload callback can
		# detect bank-related edits that hot-reload doesn't cover yet.
		_startup_bank_definitions = list(bank_definitions)
		_startup_bank_channel     = bank_channel
		_startup_default_bank     = default_bank

		def _on_midi_map_changed (path: pathlib.Path) -> None:

			"""Reload the MIDI map and deliver it to the active player.

			Program-set edits (the `programs:` block, `program_channel:`, and
			`default_program:`) are not hot-reloadable in this version — the
			callback warns and keeps the current program state.  Editing a
			`map:` preset's OWN file is also not watched (only the top-level
			map is) and needs a restart.  Top-level assignment edits reload
			as normal.
			"""

			player = _player_cell[0]

			if player is None:
				return

			assert reference_library is not None

			try:
				result = subsample.player.load_midi_map(
					path, reference_library.names(),
					strict=cfg.player.strict_midi_map,
				)
			except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
				_log.warning(
					"MIDI map reload failed at parse time — keeping current "
					"map: %s", exc,
				)
				return

			# Detect program-set changes the live reload can't apply, so the
			# user isn't left wondering why an edit to programs:/program_channel:/
			# default_program: has no audible effect.  The BankDefinition diff
			# also catches a changed map: path or a program retyped between
			# map: and directory: (map_path is part of the frozen equality).
			if (
				result.bank_definitions != _startup_bank_definitions
				or result.bank_channel != _startup_bank_channel
				or result.default_bank != _startup_default_bank
			):
				_log.warning(
					"MIDI map reload: programs, program_channel, or "
					"default_program changed — these only take effect on restart. "
					"Editing a map: preset's own file also needs a restart. "
					"Top-level assignment changes will still apply.",
				)

			# Load any NEW path/directory/reference predicates the edit
			# introduced — startup does this via _resolve_path_references, and
			# without it here a live-coded `path:`/`directory:` select would
			# reload "successfully" but silently match nothing until restart.
			# Targets the primary library/matrix (same as the OSC import
			# path); per-bank matrices in multi-bank mode still resolve at
			# startup/program load.  Already-loaded paths are deduped inside.
			try:
				reload_sr = (
					cfg.player.audio.sample_rate
					if cfg.player.audio.sample_rate is not None
					else cfg.recorder.audio.sample_rate
				)
				reload_matrices = [similarity_matrix] if similarity_matrix is not None else []
				subsample.player._resolve_path_references(
					result.note_map, reload_matrices, instrument_library,
					target_sample_rate=reload_sr,
					with_preview=cfg.recorder.previews,
				)
			except Exception as exc:
				_log.warning(
					"MIDI map reload: could not load new path references — "
					"affected selects may match nothing: %s", exc,
				)

			# reload_midi_map runs update_assignments() against the new map
			# and rolls back to the previous map + zone templates on any
			# exception — broad catch here so a runtime-validated YAML
			# error (e.g. similarity ordering without where.reference set,
			# only detectable when the query actually runs) never stops
			# live playback.
			try:
				player.reload_midi_map(result)
			except Exception as exc:
				_log.error(
					"MIDI map reload failed validation — keeping current map: %s",
					exc,
				)

		midi_map_watcher = subsample.watcher.MidiMapWatcher(
			path=_midi_map_watch_path,
			on_changed=_on_midi_map_changed,
		)
		midi_map_watcher.start()
		print(f"  MIDI map     : watching {cfg.player.midi_map} for changes")

	# --- OSC receiver ---
	# Listens for /sample/import messages and triggers file import.
	osc_receiver: typing.Any = None

	if cfg.osc.enabled and cfg.osc.receive_enabled:
		try:
			def _on_osc_import (file_path_str: str) -> None:
				"""Handle a /sample/import OSC message.

				Reads and analyses the file in place (does not copy), then
				loads it into the in-memory instrument library for immediate
				playback.  The sample is available until the next restart.
				"""

				file_path = pathlib.Path(file_path_str)

				if not file_path.is_file():
					_log.warning("OSC /sample/import: file not found: %s", file_path)
					return

				result = subsample.cache.ensure_sample_assets(file_path, with_preview=cfg.recorder.previews)

				if result is None:
					_log.warning("OSC /sample/import: analysis failed: %s", file_path)
					return

				audio = subsample.library.load_wav_audio(file_path, output_sample_rate)

				if audio is None:
					_log.warning("OSC /sample/import: could not read audio: %s", file_path)
					return

				record = subsample.library.SampleRecord(
					sample_id      = subsample.library.allocate_id(),
					name           = file_path.stem,
					spectral       = result.spectral,
					rhythm         = result.rhythm,
					pitch          = result.pitch,
					timbre         = result.timbre,
					level          = result.level,
					band_energy    = result.band_energy,
					params         = result.params,
					duration       = result.duration,
					audio          = audio,
					filepath       = file_path,
					channel_format = result.channel_format,
					loop           = result.loop,
					audio_sample_rate = output_sample_rate or result.params.sample_rate,
				)

				_integrate_sample(record, instrument_library, similarity_matrix,
				                  transform_manager, _player_cell, app_events)

			osc_receiver = subsample.osc.OscReceiver(
				port=cfg.osc.receive_port,
				on_import=_on_osc_import,
				host=cfg.osc.receive_host,
			)
			osc_receiver.start()
			print(f"  OSC receiver : listening on port {cfg.osc.receive_port}")

		except ImportError:
			_log.warning("OSC receive enabled but python-osc not installed. pip install subsample[osc]")
		except OSError as exc:
			# OscReceiver binds the UDP socket in its constructor, so a busy port
			# raises OSError here (not ImportError).  Log and continue rather
			# than letting it escape and skip the rest of startup.
			_log.warning("OSC receiver could not bind port %d: %s — OSC receive disabled", cfg.osc.receive_port, exc)

	for t in threads:
		t.start()

	try:
		# Block the main thread without spinning. Event.wait() releases the GIL
		# and responds to KeyboardInterrupt between intervals.
		while not shutdown_event.is_set():
			shutdown_event.wait(timeout=1.0)

			# If every subsystem thread has exited on its own — e.g. the only
			# enabled subsystem failed to start (bad map, missing device) and
			# returned without ever entering its run loop — there is nothing left
			# to keep the process alive, so stop waiting instead of parking
			# forever.  A still-alive sibling (both enabled, one failed) keeps the
			# app running.  `threads` holds only the recorder/player subsystems;
			# an empty list (watcher/OSC-only mode) never triggers this.
			if threads and not any(t.is_alive() for t in threads):
				if not shutdown_event.is_set():
					_log.error("All subsystems have stopped — exiting.")
				break

	except KeyboardInterrupt:
		print("\nStopping…")
		shutdown_event.set()

	for t in threads:
		t.join(timeout=10.0)

	for iw in instrument_watchers:
		iw.stop()

	if midi_map_watcher is not None:
		midi_map_watcher.stop()

	if osc_receiver is not None:
		osc_receiver.stop()

	if _sv_cell[0] is not None:
		_sv_cell[0].stop_threaded()

	# Drain any in-flight transform workers before exiting.
	if bank_manager is not None:
		for bank in bank_manager.all_banks():
			if bank.transform_manager is not None:
				bank.transform_manager.shutdown()
	elif transform_manager is not None:
		transform_manager.shutdown()

	print("Done.")


def _print_banner (cfg: subsample.config.Config) -> None:

	"""Print the startup summary line.

	Each enabled subsystem contributes its OWN real settings — the recorder its
	capture format + capture-only fields (buffer, SNR, capture directory), the
	player its resolved OUTPUT format + the map and sample source it plays from.
	A player-only run therefore reports the player's output (e.g. 48000 Hz, 8ch),
	not the disabled recorder's capture rate.
	"""

	modes = []
	if cfg.recorder.enabled:
		modes.append("recorder")
	if cfg.player.enabled:
		modes.append("player")
	mode_str = " + ".join(modes) if modes else "file-only"

	segments: list[str] = []

	if cfg.recorder.enabled:
		# channels may be None when auto-detect is configured; show "auto" until resolved.
		ch_str = (
			f"{cfg.recorder.audio.channels}ch"
			if cfg.recorder.audio.channels is not None
			else "auto"
		)
		segments.append(
			f"rec {cfg.recorder.audio.sample_rate} Hz  "
			f"{cfg.recorder.audio.bit_depth}-bit  {ch_str}  |  "
			f"buffer {cfg.recorder.buffer.max_seconds}s  |  "
			f"SNR ≥ {cfg.detection.snr_threshold_db} dB  |  "
			f"→ {cfg.output.directory}"
		)

	if cfg.player.enabled:
		# Mirror the engine's own output resolution so the banner matches the
		# stream that actually opens: rate falls back to the recorder's when
		# player.audio.sample_rate is unset (cli output_sample_rate); bit depth
		# likewise (MidiPlayer output_bit_depth); channels default to stereo.
		out_rate = (
			cfg.player.audio.sample_rate
			if cfg.player.audio.sample_rate is not None
			else cfg.recorder.audio.sample_rate
		)
		out_bits = (
			cfg.player.audio.bit_depth
			if cfg.player.audio.bit_depth is not None
			else cfg.recorder.audio.bit_depth
		)
		out_ch  = cfg.player.audio.channels if cfg.player.audio.channels is not None else 2
		map_str = cfg.player.midi_map if cfg.player.midi_map is not None else "(no map)"
		segments.append(
			f"out {out_rate} Hz  {out_bits}-bit  {out_ch}ch  |  "
			f"map {map_str}  |  ← {cfg.instrument.directory}"
		)

	body = "  ||  ".join(segments) if segments else "file-only"
	print(f"Subsample  |  {mode_str}  |  {body}")


def _integrate_sample (
	record: subsample.library.SampleRecord,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]],
	app_events: typing.Optional[subsample.events.EventEmitter] = None,
) -> None:

	"""Add a new sample to all live subsystems.

	Called from both the live-capture on_complete callback and the directory
	watcher whenever a new sample is ready. Adds the record to the instrument
	library (evicting the oldest if over the memory limit), updates the
	similarity matrix, notifies the transform pipeline to produce variants,
	triggers a pitched-assignment update on the active player, and emits
	a ``sample_loaded`` event for integrations (OSC sender, Supervisor).

	Thread-safe: each subsystem uses an internal lock. The multi-step
	sequence (library → similarity → transforms → player) is not atomic
	across subsystems — a concurrent query between steps may see transiently
	inconsistent state (e.g. an evicted sample still present in the
	similarity matrix). This is acceptable for the current use case.
	"""

	# Identity is the resolved filepath: re-integrating the SAME file (a
	# re-analysis after an edit) replaces its prior record via add()'s path
	# de-dup; a different file that happens to share a stem (e.g. "01.wav" in
	# another take-folder) loads as a distinct sample.  No stem-collision guard
	# is needed — the old warn-and-skip here silently dropped legitimate
	# same-stem takes.
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
		# Defensive: a runtime query failure here (e.g. a hot-edited map
		# containing a similarity order without a reference) must not
		# kill the watcher / OSC import thread that triggered this
		# integration.  The sample is already in the library and
		# similarity matrix; only the variant pre-compute would be lost,
		# and the next note_on falls back to non-pitched playback.
		player_cell[0]._try_update_assignments(
			f"new sample {record.name!r} integration",
		)

	if app_events is not None:
		app_events.emit("sample_loaded", record=record)


def _remove_sample (
	path: pathlib.Path,
	instrument_library: subsample.library.InstrumentLibrary,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	transform_manager: typing.Optional[subsample.transform.TransformManager],
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]],
) -> None:

	"""Remove a deleted sample from all live subsystems.

	Called by the directory watcher when a watched audio file is deleted or
	renamed away.  Drops the record from the instrument library — so a
	re-encoded or removed file's record does not linger as a selectable "ghost"
	that still plays its cached audio — then cascade-cleans the similarity and
	transform state and refreshes the player's assignments, mirroring the
	eviction cleanup in _integrate_sample.  A no-op if no sample at that path is
	currently loaded.
	"""

	removed_id = instrument_library.remove_by_path(path)

	if removed_id is None:
		return

	_log.info(
		"Instrument library: removed %s  [%s]",
		path.name, instrument_library.format_memory(),
	)

	if similarity_matrix is not None:
		similarity_matrix.remove([removed_id])

	if transform_manager is not None:
		transform_manager.on_parent_evicted([removed_id])

	if player_cell is not None and player_cell[0] is not None:
		# Defensive, as in _integrate_sample: a runtime query failure must not
		# kill the watcher thread — the sample is already gone from the library.
		player_cell[0]._try_update_assignments(
			f"sample removal {path.name!r}",
		)


def _make_on_complete (
	reference_library: typing.Optional[subsample.library.ReferenceLibrary],
	instrument_library: subsample.library.InstrumentLibrary,
	analysis_params: subsample.analysis.AnalysisParams,
	similarity_matrix: typing.Optional[subsample.similarity.SimilarityMatrix],
	store_audio: bool,
	transform_manager: typing.Optional[subsample.transform.TransformManager] = None,
	player_cell: typing.Optional[list[typing.Optional[subsample.player.MidiPlayer]]] = None,
	app_events: typing.Optional[subsample.events.EventEmitter] = None,
) -> subsample.recorder._OnCompleteCallback:

	"""Return the on_complete callback for the live-capture SampleProcessor.

	The returned callback runs on the writer thread and must not block.
	It logs the analysis result, adds the recording to the instrument
	library, updates the similarity matrix, notifies the transform
	pipeline so derivative variants can be produced in the background,
	and emits ``sample_captured`` for integrations (OSC sender, Supervisor).

	Args:
		store_audio:       When True, build a SampleRecord (with PCM audio) and
		                   integrate it into the live subsystems. Set to
		                   cfg.player.enabled — in recorder-only mode no subsystem
		                   reads from the instrument library, so integration is skipped.
		transform_manager: Optional transform pipeline coordinator. When provided,
		                   cascade-evicts derivatives for any evicted parents and
		                   triggers auto-variant production for the new sample.
		player_cell:       Single-element list holding the active MidiPlayer, or None.
		                   When provided, update_pitched_assignments() is called after
		                   each new sample is added to the similarity matrix so pitched
		                   keyboard assignments pre-compute variants for the new best match.
		app_events:        Optional event emitter; when provided, ``sample_captured``
		                   is emitted after each recording is integrated.
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
			"Recorded %s: duration %.2fs, %s",
			filepath.name, duration,
			subsample.analysis.format_level_result(level),
		)

		# Emit the OSC/dashboard event BEFORE the store_audio gate.  The
		# event is what external listeners (OSC sender, Supervisor) react
		# to — gating it on the player being enabled would silently break
		# the documented recorder-only-with-OSC multi-machine setup.
		if app_events is not None:
			app_events.emit(
				"sample_captured",
				filepath=filepath, spectral=spectral, rhythm=rhythm,
				pitch=pitch, timbre=timbre, level=level,
				band_energy=band_energy, duration=duration, audio=audio,
			)

		# Only build a SampleRecord and integrate into the live subsystems
		# when the player is active.  In recorder-only mode nothing reads
		# from the instrument library, similarity matrix, or transform
		# pipeline, so the work (and its log line) would be pure noise.
		if not store_audio:
			return

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
			audio       = audio,
			filepath    = filepath,
			# Freshly-captured audio is at the recorder rate (== analysis rate),
			# not resampled to the output rate.
			audio_sample_rate = analysis_params.sample_rate,
		)

		_integrate_sample(record, instrument_library, similarity_matrix,
		                  transform_manager, player_cell, app_events)

	return on_complete
