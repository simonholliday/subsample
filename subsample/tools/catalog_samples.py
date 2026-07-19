"""Catalog every audio file in a sample directory as a CSV of detected properties.

Walks a directory recursively (the same walk the instrument library performs at
startup), reads each file's analysis sidecar — analysing any file that doesn't
have one yet, exactly as startup would — and writes one CSV row per file with
its detected properties.  Three capability columns show which musical behaviours
each sample is eligible for:

  pitched      — passes the stable-pitch test, so it matches `pitched: true`
                 in a MIDI map and can be re-pitched across a keyboard range
  quantizable  — has enough detected hits (2+) for stretch_quantize /
                 pad_quantize to align them to a beat grid; below that,
                 quantize degrades to a plain stretch or a pass-through
  loopable     — has a steady sustaining region (tonal or textural) worth
                 looping while a key is held; a coarse candidate flag (the
                 seamless loop point is found later from the audio itself)

All three columns call the same functions the playback engine uses
(subsample.analysis.has_stable_pitch / has_beat_map / is_loopable), so what the
catalog reports is exactly what a MIDI map would select.  All seven inputs to
the pitched test (pitch_hz, voiced_fraction, voiced_frame_count,
pitch_confidence, pitch_stability_st, harmonic_ratio, duration_s) are included
as columns, so a sample that unexpectedly fails can be diagnosed against the
thresholds documented on has_stable_pitch.

Curation aids for large, mic-captured directories:

  --group             Cluster near-duplicate samples — the same sound recorded
                      over and over — so days of captures collapse to one
                      decision per distinct sound.  Adds group / group_size /
                      group_keeper columns (biggest pile first, a suggested
                      keeper marked in each), and in paths mode emits only each
                      group's keeper (a de-duplicated set).  --similarity-
                      threshold tunes how alike samples must be to group.
  --order similarity  Order rows (or paths) as a nearest-neighbour chain, so
                      each sample is followed by its closest-sounding
                      neighbour — auditioning alike-with-alike makes keep/
                      discard a comparison rather than a cold judgement.
  snr_db / near_silent / clipping_risk columns flag the obvious junk (room tone
                      with no event; near-clipped captures) so it sorts to the
                      top of the bin pile.

With --pitched, --quantizable and/or --loopable the CSV is replaced by a plain
list of matching file paths, one per line (all requested capabilities must hold)
— pipeable into a player or file tool for auditioning and hand-curating sets:

	subsample catalog --pitched | mpv --playlist=-
	subsample catalog ~/samples --quantizable | xargs -I{} cp {} ~/curated/
	subsample catalog ~/samples --loopable | mpv --playlist=-

The first run over a directory without .analysis.json sidecars is slow (each
file is fully analysed, same cost as a startup load); results are cached as
sidecars so later runs are instant.  Progress goes to stderr, data to stdout.

Usage:
	subsample catalog                          # configured library.directory
	subsample catalog path/to/samples          # explicit directory
	subsample catalog -o samples.csv           # write CSV to a file
	subsample catalog --full                   # every property incl. MFCC vectors
	subsample catalog --group                  # cluster near-duplicates
	subsample catalog --group --pitched        # one keeper path per pitched-sound group
	subsample catalog --loopable               # paths of loop-candidate samples only
	subsample catalog --order similarity       # rows ordered by sonic similarity
	subsample catalog --pitched                # paths of pitched samples only
	subsample catalog --pitched --quantizable  # paths matching both
"""

import argparse
import csv
import logging
import math
import os
import pathlib
import sys
import typing

import pymididefs.notes

import subsample.analysis
import subsample.cache
import subsample.config
import subsample.library
import subsample.similarity
import subsample.tools._shared


# Junk-triage thresholds (module constants — corpus-dependent, easy to tune).
# The raw peak and snr_db columns are always present, so a user who distrusts
# these cutoffs can sort on the numbers instead of the boolean flags.
_CLIP_PEAK:        typing.Final[float] = 0.99    # linear peak >= this (~-0.09 dBFS) → clipping_risk
_NEAR_SILENT_PEAK: typing.Final[float] = 0.004   # linear peak <  this (~-48 dBFS)  → near_silent


# Default column set: identity, the two capability indicators, and the
# musically useful scalars.  The seven has_stable_pitch inputs are all here;
# so are the three junk-triage fields (snr_db + the two flags).
_BASE_COLUMNS: tuple[str, ...] = (
	"name", "path", "duration_s", "channel_format",
	"pitched", "quantizable", "loopable", "loop_ms", "attack_count",
	"pitch_note", "pitch_hz", "pitch_confidence", "pitch_stability_st",
	"voiced_fraction", "voiced_frame_count", "harmonic_ratio",
	"tempo_bpm", "onset_count", "beat_count",
	"peak", "rms", "crest_db", "snr_db", "near_silent", "clipping_risk", "noisiness",
	"spectral_centroid", "log_attack_time",
	"band_sub_bass", "band_low_mid", "band_high_mid", "band_presence",
)

# Near-duplicate grouping columns (--group only), prepended so the biggest
# duplicate piles and their suggested keeper read left-to-right.
_GROUP_COLUMNS: tuple[str, ...] = ("group", "group_size", "group_keeper")

# --full appends everything else in the sidecar: the remaining normalised
# spectral scalars, per-band decay rates, the chroma profile, and the three
# 13-element MFCC timbre fingerprints.
_FULL_ONLY_COLUMNS: tuple[str, ...] = (
	"sample_rate", "chroma",
	"spectral_flatness", "attack", "release", "spectral_bandwidth",
	"zcr", "spectral_contrast", "spectral_flux", "spectral_rolloff",
	"spectral_slope",
	"crest_factor", "noise_floor",
	"decay_sub_bass", "decay_low_mid", "decay_high_mid", "decay_presence",
	*(f"chroma_{note}" for note in pymididefs.notes.NOTE_CLASSES),
	*(f"mfcc_{i:02d}" for i in range(13)),
	*(f"mfcc_delta_{i:02d}" for i in range(13)),
	*(f"mfcc_onset_{i:02d}" for i in range(13)),
)


def _columns (full: bool, grouped: bool = False) -> tuple[str, ...]:

	"""Return the CSV column names for the active column set.

	Args:
		full:    Include the extended --full columns (chroma / MFCC / decay).
		grouped: Prepend the near-duplicate grouping columns (--group).
	"""

	columns = _BASE_COLUMNS

	if full:
		columns = columns + _FULL_ONLY_COLUMNS

	if grouped:
		columns = _GROUP_COLUMNS + columns

	return columns


def _fmt (value: float) -> str:

	"""Format a float for CSV: up to 4 decimal places, trailing zeros stripped."""

	text = f"{value:.4f}".rstrip("0").rstrip(".")
	return "0" if text in ("", "-", "-0") else text


def _pitch_note (hz: float) -> str:

	"""Return e.g. "A2" for a detected fundamental, or "" when unpitched.

	Same Hz→MIDI derivation as the preview renderer (_format_pitch_label in
	subsample/preview.py): round to the nearest semitone so the note name and
	octave can never disagree near a semitone boundary.
	"""

	if hz <= 0.0:
		return ""

	midi = int(round(69.0 + 12.0 * math.log2(hz / 440.0)))

	if not 0 <= midi <= 127:
		return ""

	return pymididefs.notes.note_to_name(midi)


def _snr_db (peak: float, noise_floor: float) -> str:

	"""Return the peak-to-noise-floor ratio in dB, or "" when unmeasurable.

	A high value means the sample's loudest moment stands well clear of its
	quiet-frame floor — i.e. a real event rather than room tone.  "" when either
	term is non-positive (silent sample, or a sidecar whose noise_floor was not
	computed).
	"""

	if peak <= 0.0 or noise_floor <= 0.0:
		return ""

	return _fmt(20.0 * math.log10(peak / noise_floor))


def _collect_audio_files (directory: pathlib.Path) -> list[pathlib.Path]:

	"""Return every audio file under directory, recursively, sorted.

	Same walk and extension filter as the instrument library loader, so the
	catalog covers exactly the files a startup load would pick up.
	"""

	return sorted(
		p for p in directory.rglob("*")
		if p.is_file() and p.suffix.lower() in subsample.cache.AUDIO_EXTENSIONS
	)


def _row (
	path: pathlib.Path,
	directory: pathlib.Path,
	assets: subsample.cache.SampleAssets,
) -> dict[str, str]:

	"""Build the full column→value mapping for one audio file.

	Args:
		path:      The audio file (as walked — directory-joined).
		directory: The scanned root; the "path" column is relative to it.
		assets:    The SampleAssets dataclass from subsample.cache.ensure_sample_assets.

	Returns:
		Every known column (base + full); the writer selects the active set.
	"""

	spectral       = assets.spectral
	rhythm         = assets.rhythm
	pitch          = assets.pitch
	timbre         = assets.timbre
	params         = assets.params
	duration       = assets.duration
	level          = assets.level
	band_energy    = assets.band_energy
	channel_format = assets.channel_format
	loop           = assets.loop

	attack_times = subsample.analysis.effective_attack_times(rhythm)

	values: dict[str, str] = {
		"name":               path.name,
		"path":               str(path.relative_to(directory)),
		"duration_s":         _fmt(duration),
		"channel_format":     channel_format,

		# Capability indicators — the same tests the playback engine runs.
		"pitched":            "yes" if subsample.analysis.has_stable_pitch(spectral, pitch, duration) else "no",
		"quantizable":        "yes" if subsample.analysis.has_beat_map(rhythm) else "no",
		"loopable":           "yes" if subsample.analysis.is_loopable(spectral, level, duration) else "no",
		# Length of the stored seamless loop (blank when none was found); the
		# loopable flag can be "yes" here yet loop_ms blank — a candidate whose
		# audio search found no clean junction (fail-musical).
		"loop_ms":            f"{(loop.end - loop.start) / params.sample_rate * 1000.0:.0f}" if loop is not None else "",
		"attack_count":       str(len(attack_times)),

		"pitch_note":         _pitch_note(pitch.dominant_pitch_hz),
		"pitch_hz":           _fmt(pitch.dominant_pitch_hz) if pitch.dominant_pitch_hz > 0.0 else "",
		"pitch_confidence":   _fmt(pitch.pitch_confidence),
		"pitch_stability_st": _fmt(pitch.pitch_stability),
		"voiced_fraction":    _fmt(spectral.voiced_fraction),
		"voiced_frame_count": str(pitch.voiced_frame_count),
		"harmonic_ratio":     _fmt(spectral.harmonic_ratio),

		"tempo_bpm":          _fmt(rhythm.tempo_bpm),
		"onset_count":        str(rhythm.onset_count),
		"beat_count":         str(len(rhythm.beat_times)),

		"peak":               _fmt(level.peak),
		"rms":                _fmt(level.rms),
		"crest_db":           _fmt(level.crest_factor_db),

		# Junk-triage: snr_db is the discriminating number; the flags are
		# convenience cutoffs over peak (see the module constants).  noisiness
		# rates wall-to-wall unpitched noise (radio static) high — a triggered-
		# then-noisy capture reads near 1, a clean hit near 0.
		"snr_db":             _snr_db(level.peak, level.noise_floor),
		"near_silent":        "yes" if level.peak < _NEAR_SILENT_PEAK else "no",
		"clipping_risk":      "yes" if level.peak >= _CLIP_PEAK else "no",
		"noisiness":          _fmt(subsample.analysis.noisiness(spectral, level)),

		"spectral_centroid":  _fmt(spectral.spectral_centroid),
		"log_attack_time":    _fmt(spectral.log_attack_time),

		"band_sub_bass":      _fmt(band_energy.energy_fractions[0]),
		"band_low_mid":       _fmt(band_energy.energy_fractions[1]),
		"band_high_mid":      _fmt(band_energy.energy_fractions[2]),
		"band_presence":      _fmt(band_energy.energy_fractions[3]),

		# --full columns.
		"sample_rate":        str(params.sample_rate),
		"chroma":             (
			pymididefs.notes.NOTE_CLASSES[pitch.dominant_pitch_class]
			if 0 <= pitch.dominant_pitch_class < 12 else ""
		),
		"spectral_flatness":  _fmt(spectral.spectral_flatness),
		"attack":             _fmt(spectral.attack),
		"release":            _fmt(spectral.release),
		"spectral_bandwidth": _fmt(spectral.spectral_bandwidth),
		"zcr":                _fmt(spectral.zcr),
		"spectral_contrast":  _fmt(spectral.spectral_contrast),
		"spectral_flux":      _fmt(spectral.spectral_flux),
		"spectral_rolloff":   _fmt(spectral.spectral_rolloff),
		"spectral_slope":     _fmt(spectral.spectral_slope),
		"crest_factor":       _fmt(level.crest_factor),
		"noise_floor":        _fmt(level.noise_floor),
		"decay_sub_bass":     _fmt(band_energy.decay_rates[0]),
		"decay_low_mid":      _fmt(band_energy.decay_rates[1]),
		"decay_high_mid":     _fmt(band_energy.decay_rates[2]),
		"decay_presence":     _fmt(band_energy.decay_rates[3]),
	}

	# Vector columns default to "" — degenerate samples (silence, sub-2-frame
	# captures) legitimately carry empty chroma/MFCC tuples, and a blank cell
	# is more honest than a fabricated zero.
	for note in pymididefs.notes.NOTE_CLASSES:
		values[f"chroma_{note}"] = ""

	for prefix in ("mfcc", "mfcc_delta", "mfcc_onset"):
		for i in range(13):
			values[f"{prefix}_{i:02d}"] = ""

	for note, energy in zip(pymididefs.notes.NOTE_CLASSES, pitch.chroma_profile):
		values[f"chroma_{note}"] = _fmt(energy)

	for i, coeff in enumerate(timbre.mfcc):
		values[f"mfcc_{i:02d}"] = _fmt(coeff)

	for i, coeff in enumerate(timbre.mfcc_delta):
		values[f"mfcc_delta_{i:02d}"] = _fmt(coeff)

	for i, coeff in enumerate(timbre.mfcc_onset):
		values[f"mfcc_onset_{i:02d}"] = _fmt(coeff)

	return values


def _record_from_assets (
	index:  int,
	path:   pathlib.Path,
	assets: subsample.cache.SampleAssets,
) -> subsample.library.SampleRecord:

	"""Wrap one ensure_sample_assets tuple in a SampleRecord for similarity work.

	sample_id is just the load index (the similarity helpers return indices, not
	ids) and audio is None — grouping and ordering read only the analysis fields
	that feed the shared feature vector, never the waveform.
	"""

	return subsample.library.SampleRecord(
		sample_id      = index,
		name           = path.stem,
		spectral       = assets.spectral,
		rhythm         = assets.rhythm,
		pitch          = assets.pitch,
		timbre         = assets.timbre,
		level          = assets.level,
		band_energy    = assets.band_energy,
		params         = assets.params,
		duration       = assets.duration,
		audio          = None,
		filepath       = path,
		channel_format = assets.channel_format,
		loop           = assets.loop,
	)


def _matches_capability_filter (
	assets:           subsample.cache.SampleAssets,
	want_pitched:     bool,
	want_quantizable: bool,
	want_loopable:    bool,
) -> bool:

	"""Return True if assets satisfies the requested capability filters (AND).

	An unrequested capability is not constrained; requesting several means all
	must hold.  Uses the same engine tests as the catalog columns.
	"""

	return (
		(not want_pitched or subsample.analysis.has_stable_pitch(assets.spectral, assets.pitch, assets.duration))
		and (not want_quantizable or subsample.analysis.has_beat_map(assets.rhythm))
		and (not want_loopable or subsample.analysis.is_loopable(assets.spectral, assets.level, assets.duration))
	)


def _group_keeper (records: list[subsample.library.SampleRecord], group: list[int]) -> int:

	"""Return the suggested keeper index from a near-duplicate group.

	Heuristic: the loudest take (highest peak) is usually the cleanest capture of
	a repeated sound; RMS then name break ties for determinism.  It is only a
	suggestion — the catalog marks it, the user decides.  (A loud-but-clipped
	take would win here; the clipping_risk column is the cross-check.)
	"""

	return sorted(
		group,
		key=lambda i: (-records[i].level.peak, -records[i].level.rms, records[i].name),
	)[0]


def _load_all (
	paths:         list[pathlib.Path],
	show_progress: bool,
) -> tuple[list[tuple[pathlib.Path, subsample.cache.SampleAssets]], int]:

	"""Analyze/read every path (cache-first), returning (loaded, n_skipped).

	loaded is a list of (path, assets); unreadable files are logged to stderr
	and omitted.  Progress is drawn to stderr only when it is a TTY.
	"""

	loaded:    list[tuple[pathlib.Path, subsample.cache.SampleAssets]] = []
	n_skipped: int = 0

	for i, path in enumerate(paths, start=1):
		if show_progress:
			print(f"\r[{i}/{len(paths)}] {path.name[:60]:<60}", end="", file=sys.stderr)

		assets = subsample.cache.ensure_sample_assets(path, with_preview=False)

		if assets is None:
			end = "\n" if show_progress else ""
			print(f"{end}Skipping {path}: unreadable audio or unrecoverable analysis data", file=sys.stderr)
			n_skipped += 1
			continue

		loaded.append((path, assets))

	if show_progress:
		print("\r\x1b[K", end="", file=sys.stderr)

	return loaded, n_skipped


def _emission_plan (
	loaded:  list[tuple[pathlib.Path, subsample.cache.SampleAssets]],
	records: typing.Optional[list[subsample.library.SampleRecord]],
	cfg:     typing.Optional[subsample.config.Config],
	args:    argparse.Namespace,
) -> list[tuple[int, typing.Optional[dict[str, str]]]]:

	"""Decide the emission order and per-row group metadata.

	Returns a list of (load_index, group_meta) in output order.  group_meta is
	None unless --group is active, in which case it carries the group / group_size
	/ group_keeper cells.  records and cfg are required for the --group and
	--order similarity paths and ignored otherwise.

	  --group             : cluster near-duplicates; emit largest pile first, the
	                        suggested keeper first within each pile.  (Determines
	                        row order; --order is ignored.)
	  --order similarity  : greedy nearest-neighbour chain so alike samples are
	                        adjacent.
	  neither             : load order (name-sorted, from the directory walk).
	"""

	if args.group:
		assert records is not None and cfg is not None

		groups = subsample.similarity.group_near_duplicates(
			records, cfg.similarity, args.similarity_threshold,
		)

		plan: list[tuple[int, typing.Optional[dict[str, str]]]] = []

		for group_id, group in enumerate(groups, start=1):
			keeper  = _group_keeper(records, group)
			ordered = [keeper] + sorted(
				(i for i in group if i != keeper),
				key=lambda i: records[i].name,
			)

			for index in ordered:
				plan.append((index, {
					"group":        str(group_id),
					"group_size":   str(len(group)),
					"group_keeper": "yes" if index == keeper else "no",
				}))

		return plan

	if args.order == "similarity":
		assert records is not None and cfg is not None
		order = subsample.similarity.similarity_order(records, cfg.similarity)
		return [(index, None) for index in order]

	return [(index, None) for index in range(len(loaded))]


def _parse_args (argv: typing.Optional[list[str]] = None) -> argparse.Namespace:

	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(
		prog="subsample catalog",
		description="Write a CSV of every sample's detected properties, or list samples matching a capability filter",
	)
	parser.add_argument(
		"directory",
		type=pathlib.Path,
		nargs="?",
		default=None,
		help="Sample directory to catalog (default: library.directory from config.yaml)",
	)
	parser.add_argument(
		"-o", "--output",
		type=pathlib.Path,
		default=None,
		metavar="FILE",
		help="Write output to FILE instead of stdout",
	)
	parser.add_argument(
		"--full",
		action="store_true",
		help="Include every stored property (chroma profile, MFCC timbre vectors, per-band decay rates)",
	)
	parser.add_argument(
		"--pitched",
		action="store_true",
		help="Instead of a CSV, list the paths of samples that pass the stable-pitch test (re-pitchable)",
	)
	parser.add_argument(
		"--quantizable",
		action="store_true",
		help="Instead of a CSV, list the paths of samples with enough hits to quantize to a beat grid",
	)
	parser.add_argument(
		"--loopable",
		action="store_true",
		help="Instead of a CSV, list the paths of samples that could sustain a held-note loop",
	)
	parser.add_argument(
		"--group",
		action="store_true",
		help="Cluster near-duplicate samples (same sound recorded repeatedly) and add "
		     "group/group_size/group_keeper columns, biggest pile first, suggested keeper "
		     "first; in paths mode, emit only each group's keeper (deduplicated)",
	)
	parser.add_argument(
		"--similarity-threshold",
		type=float,
		default=0.98,
		metavar="T",
		help="Cosine-similarity cutoff (0..1) for --group; higher groups only near-identical "
		     "samples, lower groups more loosely (default: 0.98)",
	)
	parser.add_argument(
		"--order",
		choices=("name", "similarity"),
		default="name",
		help="Row/path order: 'name' (default) or 'similarity' (nearest-neighbour chain so "
		     "alike samples are adjacent for auditioning). Ignored when --group is set",
	)
	parser.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	return parser.parse_args(argv)


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Catalog a sample directory to CSV, or list paths matching a capability filter."""

	subsample.tools._shared.configure_logging()

	args = _parse_args(argv)

	paths_mode = args.pitched or args.quantizable or args.loopable

	# --group and --order similarity both need every sample loaded before any
	# output (clustering / nearest-neighbour ordering are whole-set operations),
	# and both need the similarity weights from config.
	two_pass = args.group or args.order == "similarity"

	# Always load config: this tool writes sidecars the player later trusts, so
	# it must analyse at the same float-import ceiling AND tempo priors the
	# player will (a differently-scaled or differently-tuned analysis would
	# leave the player trusting a fingerprint that doesn't match its playback).
	# load_config_and_wire handles both, plus a clean one-line config error.
	cfg = subsample.tools._shared.load_config_and_wire(args.config)

	if args.directory is not None:
		directory = args.directory
	else:
		assert cfg is not None
		directory = pathlib.Path(cfg.library.directory)

	if not directory.is_dir():
		print(f"Directory not found: {directory}", file=sys.stderr)
		return 1

	paths = _collect_audio_files(directory)

	if not paths:
		print(f"No audio files found in {directory}", file=sys.stderr)

	show_progress = sys.stderr.isatty()
	grouped       = args.group

	out = open(args.output, "w", newline="", encoding="utf-8") if args.output is not None else sys.stdout

	try:
		loaded, n_skipped = _load_all(paths, show_progress)

		records: typing.Optional[list[subsample.library.SampleRecord]] = None

		if two_pass and loaded:
			if len(loaded) > 6000:
				print(
					f"Note: comparing {len(loaded)} samples pairwise is O(N²) in memory — "
					f"this may be slow.",
					file=sys.stderr,
				)
			print(f"Comparing {len(loaded)} samples by similarity…", file=sys.stderr)
			records = [_record_from_assets(i, p, a) for i, (p, a) in enumerate(loaded)]

		plan = _emission_plan(loaded, records, cfg, args)

		# lineterminator: csv defaults to \r\n; plain \n keeps the output
		# friendly to unix pipes (awk, grep) as well as spreadsheets.
		writer = csv.writer(out, lineterminator="\n")

		if paths_mode:
			for index, meta in plan:
				# Grouped paths mode is a de-duplicator: one keeper per pile.
				if grouped and meta is not None and meta["group_keeper"] == "no":
					continue

				path, assets = loaded[index]

				if _matches_capability_filter(assets, args.pitched, args.quantizable, args.loopable):
					print(path, file=out)

		else:
			columns = _columns(args.full, grouped=grouped)
			writer.writerow(columns)

			for index, meta in plan:
				path, assets = loaded[index]
				values       = _row(path, directory, assets)

				if meta is not None:
					values.update(meta)

				writer.writerow([values[c] for c in columns])

		summary = f"Cataloged {len(loaded)} sample(s) from {directory}"

		if grouped and records is not None:
			n_groups = len({meta["group"] for _index, meta in plan if meta is not None})
			summary += f" in {n_groups} group(s)"

		if n_skipped:
			summary += f", skipped {n_skipped}"

		print(summary, file=sys.stderr)

	finally:
		if out is not sys.stdout:
			out.close()

	return 0


if __name__ == "__main__":
	try:
		sys.exit(main())
	except BrokenPipeError:
		# Downstream closed the pipe early (head, or quitting mpv mid-playlist)
		# — normal pipeline behaviour, not an error.  Redirect stdout to
		# devnull so interpreter shutdown doesn't raise a second time
		# (the pattern recommended by the Python signal docs).
		os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
		sys.exit(0)
