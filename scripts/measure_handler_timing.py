"""Measure per-note-on handler processing cost for the live player.

The companion script ``measure_midi_latency.py`` measures only MIDI
*delivery* latency — it uses an empty callback and never touches the
library, query engine, or transform cache.  This script measures the
other half: how long ``MidiPlayer._handle_message`` actually takes to
turn a note_on into a queued voice, against the real config, the real
instrument library, and the real MIDI map.

It builds the player as ``cli._main_impl`` does for single-directory mode
(loading the library, similarity matrix, and transform manager itself —
``_start_player`` receives those as parameters), but never opens an audio or
MIDI device — it calls
``_handle_message`` directly on this thread, so the numbers are pure
selection + variant-lookup + render compute, with no rtmidi delivery
jitter and no audio-callback lock contention mixed in.

Two phases are reported for the chosen assignment's note:

  - COLD: fired immediately after ``update_pitched_assignments`` — before
    the background transform workers have finished baking the variant.
    Exercises the cache-miss path (md5 of the audio + disk-cache probe +
    enqueue, then fallback).
  - WARM: fired after the workers drain.  Exercises the steady-state
    cache-hit path that a sustained groove actually hits.

Each hit is classified by which playback path it took (cached variant,
previous-variant fallback, base variant, or raw render) so a slow phase
can be attributed to misses rather than guessed at.

Usage:

    python scripts/measure_handler_timing.py
    python scripts/measure_handler_timing.py --assignment Kick --hits 500
"""

import argparse
import logging
import pathlib
import statistics
import sys
import threading
import time
import typing

import mido

import subsample.audio
import subsample.config
import subsample.library
import subsample.player
import subsample.similarity
import subsample.transform


class _PathCapture (logging.Handler):

	"""Capture the format-string of each player log record.

	The note-on handler logs which playback path it took with a literal
	substring ("(variant,", "(previous variant)", "(base variant)").  That
	token lives in ``record.msg`` (the format string) — not the %-args —
	so we can read it without paying the cost of formatting the message.
	"""

	def __init__ (self) -> None:
		super().__init__()
		self.last_msg: str = ""

	def emit (self, record: logging.LogRecord) -> None:
		self.last_msg = str(record.msg)


def _classify (msg: str) -> str:

	"""Map a captured format-string to a short playback-path label."""

	if "(variant," in msg:
		return "variant"
	if "(previous variant)" in msg:
		return "previous"
	if "(base variant)" in msg:
		return "base"
	if "no sample matched" in msg or "no mapping" in msg:
		return "nomatch"

	return "render"


def _percentile (values: list[float], pct: float) -> float:

	"""Return the ``pct``-th percentile (0-100) of ``values``."""

	if not values:
		return float("nan")

	ordered = sorted(values)
	idx = int(round((pct / 100.0) * (len(ordered) - 1)))
	return ordered[idx]


def _report (label: str, durations_ms: list[float], paths: dict[str, int]) -> None:

	"""Print percentile stats and the path-classification breakdown."""

	if not durations_ms:
		print(f"{label}: no samples")
		return

	path_summary = "  ".join(f"{k}={v}" for k, v in sorted(paths.items()))

	print(f"{label}  (n={len(durations_ms)})")
	print(f"  median  : {statistics.median(durations_ms):8.3f} ms")
	print(f"  95th-pct: {_percentile(durations_ms, 95.0):8.3f} ms")
	print(f"  99th-pct: {_percentile(durations_ms, 99.0):8.3f} ms")
	print(f"  max     : {max(durations_ms):8.3f} ms")
	print(f"  paths   : {path_summary}")
	print()


def _fire (
	player:   subsample.player.MidiPlayer,
	capture:  _PathCapture,
	channel:  int,
	note:     int,
	hits:     int,
	gap_s:    float,
) -> tuple[list[float], dict[str, int]]:

	"""Fire ``hits`` note_on messages, timing each ``_handle_message`` call.

	Returns the per-hit durations (ms) and a count of playback paths taken.
	The voice list is cleared between hits (outside the timed region) so the
	measurement isn't skewed by an ever-growing backlog that no audio
	callback is draining.
	"""

	durations_ms: list[float] = []
	paths: dict[str, int] = {}

	for _ in range(hits):
		msg = mido.Message("note_on", channel=channel, note=note, velocity=100)

		capture.last_msg = ""
		start_ns = time.perf_counter_ns()
		player._handle_message(msg)
		elapsed_ns = time.perf_counter_ns() - start_ns

		durations_ms.append(elapsed_ns / 1e6)
		label = _classify(capture.last_msg)
		paths[label] = paths.get(label, 0) + 1

		with player._voices_lock:
			player._voices.clear()

		if gap_s > 0.0:
			time.sleep(gap_s)

	return durations_ms, paths


def main () -> int:

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--assignment", default="Kicks", help="Assignment name to fire (default: Kicks, as in midi-map.yaml.default).")
	parser.add_argument("--hits", type=int, default=500, help="Warm-phase hit count (default: 500).")
	parser.add_argument("--gap-ms", type=float, default=5.0, help="Spacing between warm hits, ms (default: 5).")
	parser.add_argument("--drain-s", type=float, default=6.0, help="Seconds to wait for transform workers to bake variants (default: 6).")
	parser.add_argument("--profile", action="store_true", help="cProfile a warm burst and print the hottest functions.")
	args = parser.parse_args()

	logging.basicConfig(level=logging.WARNING, format="%(message)s")

	cfg = subsample.config.load_config()

	# Loading the library heals sidecars via ensure_sample_assets (read through
	# read_audio_file) — wire the float ceiling so hot floats are read as the
	# player reads them.
	subsample.audio.set_float_import_ceiling(cfg.recorder.audio.float_import_ceiling_dbfs)

	if cfg.player.midi_map is None:
		print("config.player.midi_map is not set", file=sys.stderr)
		return 1

	output_sr = (
		cfg.player.audio.sample_rate
		if cfg.player.audio.sample_rate is not None
		else cfg.recorder.audio.sample_rate
	)

	# Mirror cli._main_impl single-directory setup.  Reference library is
	# empty in production too — path references resolve into similarity
	# matrices, not here.
	reference_library = subsample.library.ReferenceLibrary([])

	midi_map_result = subsample.player.load_midi_map(
		pathlib.Path(cfg.player.midi_map), [], strict=cfg.player.strict_midi_map,
	)

	if midi_map_result.bank_definitions:
		print("This harness covers single-directory maps only; this map declares programs.", file=sys.stderr)
		return 1

	print(f"Loading instrument library from {cfg.instrument.directory} …")
	instrument_library = subsample.library.load_instrument_library(
		pathlib.Path(cfg.instrument.directory),
		int(cfg.instrument.max_memory_mb * 1024 * 1024),
		load_audio=True,
		with_preview=cfg.recorder.previews,
		target_sample_rate=output_sr,
	)
	print(f"  {len(instrument_library)} sample(s) loaded")

	similarity_matrix = subsample.similarity.SimilarityMatrix(reference_library, cfg.similarity)
	if len(instrument_library) > 0:
		similarity_matrix.bulk_add(instrument_library.samples())

	# Transform pipeline — identical wiring to production, including the
	# on-disk variant cache, so cache hit/miss behaviour matches the real run.
	transform_cache = subsample.transform.TransformCache(
		max_memory_bytes=int(cfg.transform.max_memory_mb * 1024 * 1024),
	)

	def _on_complete (result: subsample.transform.TransformResult) -> None:
		transform_cache.put(result)

	variant_disk_cache: typing.Optional[subsample.transform.VariantDiskCache] = None
	if cfg.transform.variant_cache_dir and cfg.transform.max_disk_mb > 0:
		variant_disk_cache = subsample.transform.VariantDiskCache(
			directory=pathlib.Path(cfg.transform.variant_cache_dir),
			max_bytes=int(cfg.transform.max_disk_mb * 1024 * 1024),
			sample_rate=output_sr,
		)

	transform_processor = subsample.transform.TransformProcessor(
		sample_rate=cfg.recorder.audio.sample_rate,
		output_sample_rate=output_sr,
		bit_depth=cfg.recorder.audio.bit_depth,
		on_complete=_on_complete,
		disk_cache=variant_disk_cache,
	)
	transform_manager = subsample.transform.TransformManager(
		cache=transform_cache,
		processor=transform_processor,
		instrument_library=instrument_library,
		cfg=cfg.transform,
		disk_cache=variant_disk_cache,
	)

	for record in instrument_library.samples():
		transform_manager.on_sample_added(record)

	midi_map = midi_map_result.note_map

	# Resolve path references for assignments that use them (best-effort —
	# the assignment under test may not need any).
	try:
		subsample.player._resolve_path_references(
			midi_map, [similarity_matrix], instrument_library,
			target_sample_rate=output_sr, with_preview=cfg.recorder.previews,
		)
	except Exception as exc:
		print(f"  (path-reference resolution skipped: {exc})")

	player = subsample.player.MidiPlayer(
		"",
		shutdown_event=threading.Event(),
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
		target_bpm=cfg.transform.target_bpm,
		output_channels=cfg.player.audio.channels,
		ambisonic_config=cfg.ambisonic,
		buffer_frames=cfg.player.audio.buffer_frames,
		zone_templates=midi_map_result.zone_templates,
	)

	# Find the (channel, note) the chosen assignment is mapped to.
	target: typing.Optional[tuple[int, int]] = None
	for (ch, note), entries in player._note_map.items():
		for assignment, _pick in entries:
			if assignment.name == args.assignment:
				target = (ch, note)
				break
		if target is not None:
			break

	if target is None:
		print(f"Assignment {args.assignment!r} not found in the note map.", file=sys.stderr)
		print(f"Available: {sorted({a.name for es in player._note_map.values() for a, _ in es})}", file=sys.stderr)
		return 1

	channel, note = target
	print(f"Firing assignment {args.assignment!r} → MIDI ch{channel + 1} note {note}\n")

	capture = _PathCapture()
	player_logger = logging.getLogger("subsample.player")
	player_logger.setLevel(logging.DEBUG)
	player_logger.addHandler(capture)

	# Warm the static variant cache (mirrors cli's call before run()).
	player.update_pitched_assignments()

	# COLD: workers are still baking — exercises the miss path.
	cold_durs, cold_paths = _fire(player, capture, channel, note, hits=30, gap_s=0.0)
	_report("COLD (immediately after update_assignments)", cold_durs, cold_paths)

	print(f"Draining transform workers for {args.drain_s:.1f}s …\n")
	time.sleep(args.drain_s)

	# WARM: steady state — what a sustained groove actually hits.
	warm_durs, warm_paths = _fire(player, capture, channel, note, hits=args.hits, gap_s=args.gap_ms / 1000.0)
	_report(f"WARM (after drain, {args.gap_ms:.0f} ms spacing)", warm_durs, warm_paths)

	if args.profile:
		import cProfile
		import pstats

		print("Profiling 200 warm hits …\n")
		profiler = cProfile.Profile()
		profiler.enable()
		_fire(player, capture, channel, note, hits=200, gap_s=0.0)
		profiler.disable()

		stats = pstats.Stats(profiler)
		stats.sort_stats("tottime")
		stats.print_stats(18)

	player_logger.removeHandler(capture)

	return 0


if __name__ == "__main__":
	sys.exit(main())
