"""Analyze one or more audio files and print their metrics to the console.

Reads any audio file supported by soundfile (WAV, FLAC, AIFF, OGG, etc.),
runs the same analysis pipeline used during live capture, and prints three
summary lines per file: rhythm, spectral, and pitch metrics.  Every detected
attack is listed under those, with where it lands and how loud it is — a count
of onsets says a take has four hits, not whether they are even enough to
quantize or whether one of them is a ghost note.

Results are cached as a JSON sidecar file (<audio-file>.analysis.json) so
that repeated analysis of the same file is instant. The cache is
automatically invalidated if the audio file changes or the analysis
algorithm is updated.

Usage:
	subsample analyze <path/to/file.wav>
	subsample analyze ./reference/*.wav
	subsample analyze kick.wav snare.wav hat.wav
"""

import argparse
import logging
import pathlib
import sys
import typing

import numpy
import soundfile

import subsample.analysis
import subsample.audio
import subsample.cache
import subsample.config
import subsample.tools._shared


_log = logging.getLogger(__name__)


def _read_mono (filepath: pathlib.Path) -> typing.Optional[numpy.ndarray]:

	"""The file's audio as the analysis reads it, or None when it cannot be read now.

	Read through read_audio_file, like the analysis path above, so a level
	measured here is measured at the same scale the player will use.
	"""

	try:
		file_info = subsample.audio.read_audio_file(filepath)

	except (OSError, ValueError) as exc:
		_log.warning("Could not read %s to measure attack levels: %s", filepath.name, exc)
		return None

	return subsample.analysis.to_mono_float(file_info.audio, file_info.bit_depth)


def _print_attacks (attack_times: typing.Sequence[float], levels: typing.Sequence[float]) -> None:

	"""Print where each detected attack lands and how loud it is.

	Levels are dB relative to the loudest moment of the sample, so the hardest
	hit reads 0.0 and a ghost note reads well below it.
	"""

	if not attack_times:
		print("attacks:  none")
		return

	if len(levels) != len(attack_times):
		print(f"attacks:  {len(attack_times)}  (levels unavailable)")
		return

	print(f"attacks:  {len(attack_times)}")

	for number, (attack, level) in enumerate(zip(attack_times, levels), start=1):
		print(f"  {number:2d}   {attack:6.3f}s  {level:5.1f}dB")


def _analyze_file (
	filepath: pathlib.Path,
	rhythm_cfg: subsample.config.AnalysisConfig,
) -> bool:

	"""Analyze a single audio file and print its metrics.

	Returns True on success, False if the file could not be read or analysed
	(so the caller can exit non-zero when every input failed)."""

	# The buffer, kept for the attack levels: the sidecar holds attack times but
	# not how loud each one is, so a cache hit still needs the audio itself.
	mono: typing.Optional[numpy.ndarray] = None

	# Try the cache first — skips CPU-intensive analysis if nothing has changed
	cached = subsample.cache.load_cache(filepath)

	if cached is not None:
		result      = cached.spectral
		rhythm      = cached.rhythm
		pitch       = cached.pitch
		timbre      = cached.timbre
		params      = cached.params
		duration    = cached.duration
		level       = cached.level
		band_energy = cached.band_energy
		loop        = cached.loop

	else:
		# Hash the file BEFORE decoding and analysing it: hashing afterwards
		# would pair the analysis of the old bytes with an MD5 of whatever the
		# file became if it were overwritten mid-analysis — a permanently wrong
		# sidecar that never self-heals (cache._reanalyze_and_save documents the
		# same hash-first rule).
		try:
			audio_md5 = subsample.cache.compute_audio_md5(filepath)

		except OSError as exc:
			print(f"Error reading {filepath}: {exc}", file=sys.stderr)
			return False

		# Read through read_audio_file (not a raw soundfile.read) so this matches
		# the cache/player pipeline exactly: hot float/double sources are scaled
		# to the import ceiling, and the sidecar we write describes the audio the
		# player will actually read and play (a raw float read would poison the
		# cache with metrics 1+ dB louder than playback).
		try:
			file_info = subsample.audio.read_audio_file(filepath)

		except (OSError, ValueError) as exc:
			print(f"Error reading {filepath}: {exc}", file=sys.stderr)
			return False

		mono = subsample.analysis.to_mono_float(file_info.audio, file_info.bit_depth)
		samplerate = file_info.sample_rate

		params = subsample.analysis.compute_params(samplerate)

		# Run all three analyses; analyze_all() shares the pyin computation
		# between spectral and pitch, avoiding ~200-300 ms of redundant work.
		# A librosa failure on degenerate input skips this one file cleanly.
		try:
			result, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(mono, params, rhythm_cfg)
		except Exception as exc:
			print(f"  {filepath.name}  (skipped, could not analyze: {exc})", file=sys.stderr)
			return False

		duration = len(mono) / samplerate

		loop = subsample.cache.compute_loop(mono, samplerate, result, pitch, level, duration)

		# Save results for next time; log but don't fail if the filesystem is read-only
		try:
			subsample.cache.save_cache(
				audio_path = filepath,
				audio_md5  = audio_md5,
				params     = params,
				spectral   = result,
				rhythm     = rhythm,
				pitch      = pitch,
				timbre     = timbre,
				duration   = duration,
				level      = level,
				band_energy = band_energy,
				loop       = loop,
			)
		except OSError as exc:
			_log.warning("Could not save analysis cache for %s: %s", filepath.name, exc)

	# Re-read only when the analysis came from the sidecar.  A cache hit already
	# reads every byte of the file to check its MD5, so this costs the decode
	# rather than another trip to the disk.
	if mono is None and rhythm.attack_times:
		mono = _read_mono(filepath)

	levels = (
		subsample.analysis.attack_levels(mono, rhythm.attack_times, params.sample_rate)
		if mono is not None else ()
	)

	print(f"rhythm:   {subsample.analysis.format_rhythm_result(rhythm)}")
	_print_attacks(rhythm.attack_times, levels)
	print(f"spectral: {subsample.analysis.format_result(result, duration)}")
	print(f"pitch:    {subsample.analysis.format_pitch_result(pitch)}")
	print(f"level:    {subsample.analysis.format_level_result(level)}")
	print(f"noisiness: {subsample.analysis.noisiness(result, level):.3f}  (0 = clean event, 1 = wall-to-wall noise)")

	if loop is not None:
		sr = params.sample_rate
		print(
			f"loop:     {loop.start / sr:.3f}s -> {loop.end / sr:.3f}s "
			f"({(loop.end - loop.start) / sr * 1000:.0f} ms, xfade {loop.crossfade / sr * 1000:.0f} ms, "
			f"junction_flux {loop.junction_flux:.2f})"
		)
	else:
		print("loop:     none (not a loop candidate, or no clean junction)")

	return True


# What the output means, printed at the end of --help and published with it on
# subsystem.co's command-line reference (#3843).  The help formatter keeps
# these lines as written, so they are wrapped by hand, within 79 columns.
_OUTPUT: typing.Final[str] = """\
output:
  rhythm     the tempo in BPM, and the beats, pulses and onsets found
  attacks    each hit's start in seconds, and its level in dB against the
             loudest hit, which reads 0.0dB; a hit far below the rest is a
             ghost note
  spectral   the length in seconds, then measures from 0 to 1:
               flatness    0 tonal, 1 noisy
               attack      0 instant, 1 a gradual build
               release     0 a sudden stop, 1 a long decay
               centroid    0 bassy, 1 bright
               bandwidth   0 a pure tone, 1 spectrally complex
               zcr         how often the wave crosses zero: 0 smooth, 1 noisy
               harmonic    0 percussive, 1 harmonic
               contrast    0 a flat spectrum, 1 strong spectral peaks
               voiced      the share of the sound with a detected pitch
               log_attack  0 an instant onset, 1 a very slow one
               flux        0 a steady spectrum, 1 a fast-changing one
               rolloff     0 energy low down, 1 energy reaching the top
               slope       0 bass-heavy, 0.5 about flat, 1 treble-heavy
  pitch      the pitch in Hz and its pitch class (chroma), or none;
             pitch_conf, from 0 to 1, is how sure the pitch is, stability is
             how far it wanders in semitones, and voiced_frames how many
             frames hold a pitch
  level      the peak, the loudness (rms) and, when it can be measured, the
             room's floor, in dBFS, and the crest factor, peak over loudness,
             in dB; rms sets the playback gain
  noisiness  from 0, a clean hit or tone, to 1, noise from end to end such as
             static; a sustained unpitched sound scores high too
  loop       the loop found, its length and crossfade, and junction_flux,
             near 1 for a seamless join; none when there is no clean loop
"""


def parser () -> argparse.ArgumentParser:

	"""Build the parser for `subsample analyze`, without parsing anything.

	Kept apart from main() so that subsystem.co can generate the command-line
	reference from it without running the tool (#3020).
	"""

	command = argparse.ArgumentParser(
		prog="subsample analyze",
		description=(
			"Analyse audio files and print their detected metrics (rhythm, spectral,\n"
			"pitch, level, loop)."
		),
		epilog=_OUTPUT,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	command.add_argument(
		"files",
		nargs="+",
		metavar="FILE",
		help="Audio files to analyse. A name may use wildcards, as in '*.wav'.",
	)
	command.add_argument(
		"--config",
		type=pathlib.Path,
		default=None,
		metavar="PATH",
		help="Path to config.yaml (default: auto-discover as per main app)",
	)
	return command


def main (argv: typing.Optional[list[str]] = None) -> int:

	"""Analyze one or more audio files and print their metrics."""

	subsample.tools._shared.configure_logging()

	args = parser().parse_args(argv)

	# Wire the float ceiling and analysis tempo priors from config: analyze
	# writes a sidecar the player later trusts, so it must analyse at the same
	# scale and tuning the app itself would use.
	cfg = subsample.tools._shared.load_config_and_wire(args.config)

	paths = subsample.tools._shared.expanded_paths(args.files)

	if not paths:
		return 1

	multi = len(paths) > 1
	any_ok = False

	for filepath in paths:
		if multi:
			print(f"\nAnalyzing {filepath.name} ...")

		if _analyze_file(filepath, cfg.analysis):
			any_ok = True

	# Every input unreadable/unanalysable → non-zero exit for scripts/pipelines.
	return 0 if any_ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
