"""Extract General MIDI percussion samples from a SoundFont and analyze them.

Renders each of the 47 GM percussion instruments (MIDI notes 35-81) from a
SoundFont file using fluidsynth, then runs the subsample analysis pipeline
to produce .analysis.json sidecar files.

WAV + JSON files are written to the output directory (default
samples/reference/ — audio stays local-only and .gitignored there), and each
.analysis.json sidecar is also copied to subsample/data/reference/, the
shipped package-data location. Commit the subsample/data/reference/ copies —
they are what a pip-installed user receives, and a test pins their
analysis_version to the current ANALYSIS_VERSION (re-run this script after
any version bump).

Prerequisites:
	- fluidsynth CLI tool (apt install fluidsynth)
	- A General MIDI SoundFont file (e.g. FluidR3_GM.sf2)

Usage:
	python scripts/extract_gm_drums.py /path/to/FluidR3_GM.sf2
	python scripts/extract_gm_drums.py /path/to/FluidR3_GM.sf2 --output reference/
"""

import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import mido
import numpy
import soundfile

import subsample.analysis
import subsample.cache
import subsample.config


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s  %(levelname)-8s  %(message)s",
	datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GM percussion map — MIDI note 35-81
# ---------------------------------------------------------------------------

GM_PERCUSSION: dict[int, str] = {
	35: "AcousticBassDrum",
	36: "BassDrum1",
	37: "SideStick",
	38: "AcousticSnare",
	39: "HandClap",
	40: "ElectricSnare",
	41: "LowFloorTom",
	42: "ClosedHiHat",
	43: "HighFloorTom",
	44: "PedalHiHat",
	45: "LowTom",
	46: "OpenHiHat",
	47: "LowMidTom",
	48: "HiMidTom",
	49: "CrashCymbal1",
	50: "HighTom",
	51: "RideCymbal1",
	52: "ChineseCymbal",
	53: "RideBell",
	54: "Tambourine",
	55: "SplashCymbal",
	56: "Cowbell",
	57: "CrashCymbal2",
	58: "Vibraslap",
	59: "RideCymbal2",
	60: "HiBongo",
	61: "LowBongo",
	62: "MuteHiConga",
	63: "OpenHiConga",
	64: "LowConga",
	65: "HighTimbale",
	66: "LowTimbale",
	67: "HighAgogo",
	68: "LowAgogo",
	69: "Cabasa",
	70: "Maracas",
	71: "ShortWhistle",
	72: "LongWhistle",
	73: "ShortGuiro",
	74: "LongGuiro",
	75: "Claves",
	76: "HiWoodBlock",
	77: "LowWoodBlock",
	78: "MuteCuica",
	79: "OpenCuica",
	80: "MuteTriangle",
	81: "OpenTriangle",
}


def _make_midi_file (note: int, output_path: pathlib.Path) -> None:

	"""Write a minimal MIDI file with a single percussion hit on channel 10."""

	mid = mido.MidiFile(type=0)
	track = mido.MidiTrack()
	mid.tracks.append(track)

	# Channel 10 in GM is channel 9 in mido (0-indexed).
	track.append(mido.Message("note_on", channel=9, note=note, velocity=100, time=0))
	track.append(mido.Message("note_off", channel=9, note=note, velocity=0, time=960))

	# A couple of seconds of silence to let the sample ring out.
	track.append(mido.MetaMessage("end_of_track", time=960))

	mid.save(str(output_path))


def _render_with_fluidsynth (
	sf2_path:  pathlib.Path,
	midi_path: pathlib.Path,
	wav_path:  pathlib.Path,
) -> bool:

	"""Render a MIDI file to WAV using fluidsynth."""

	try:
		subprocess.run(
			[
				"fluidsynth",
				"-ni",           # non-interactive, no shell
				"-g", "1.0",     # gain
				"-r", "44100",   # sample rate
				str(sf2_path),
				str(midi_path),
				"-F", str(wav_path),
			],
			check=True,
			capture_output=True,
		)
		return True

	except subprocess.CalledProcessError as exc:
		_log.error("fluidsynth failed for %s: %s", midi_path.name, exc.stderr.decode())
		return False


def _trim_silence (audio: numpy.ndarray, threshold: float = 0.001) -> numpy.ndarray:

	"""Trim trailing silence from audio.

	Finds the last sample above the threshold and trims everything after it
	(with a small fade-out tail).  Leading silence is preserved — the attack
	matters for analysis.
	"""

	mono = numpy.mean(audio, axis=1) if audio.ndim > 1 else audio
	above = numpy.where(numpy.abs(mono) > threshold)[0]

	if len(above) == 0:
		return audio

	last = above[-1]

	# Keep a small tail (100 ms at 44100 Hz) for natural decay.
	tail = min(4410, len(audio) - last - 1)
	end = last + tail + 1

	return audio[:end]


def _analyze_and_save (wav_path: pathlib.Path) -> bool:

	"""Run the analysis pipeline on a WAV file and save the sidecar JSON."""

	try:
		data, samplerate = soundfile.read(str(wav_path), always_2d=True, dtype="float32")
	except (OSError, soundfile.SoundFileError) as exc:
		_log.error("Cannot read %s: %s", wav_path.name, exc)
		return False

	# Trim trailing silence.
	data = _trim_silence(data)

	if len(data) < 1000:
		_log.warning("Skipping %s — too short after trimming (%d frames)", wav_path.name, len(data))
		return False

	# Write trimmed audio back so the sidecar MD5 matches.
	soundfile.write(str(wav_path), data, samplerate, subtype="PCM_16")

	mono = numpy.mean(data, axis=1, dtype=numpy.float32)  # type: ignore[call-overload]
	params = subsample.analysis.compute_params(samplerate)
	rhythm_cfg = subsample.config.AnalysisConfig()

	result, rhythm, pitch, timbre, level, band_energy = subsample.analysis.analyze_all(mono, params, rhythm_cfg)

	duration = len(data) / samplerate
	audio_md5 = subsample.cache.compute_audio_md5(wav_path)

	# Compute the loop so the sidecar carries it (drums are never loop
	# candidates, so this is None here — but it keeps every save_cache writer
	# consistent, since a loop=None sidecar is trusted on version+MD5 forever).
	loop = subsample.cache.compute_loop(mono, samplerate, result, pitch, level, duration)

	subsample.cache.save_cache(
		audio_path  = wav_path,
		audio_md5   = audio_md5,
		params      = params,
		spectral    = result,
		rhythm      = rhythm,
		pitch       = pitch,
		timbre      = timbre,
		duration    = duration,
		level       = level,
		band_energy = band_energy,
		loop        = loop,
		bit_depth   = 16,
		channels    = data.shape[1] if data.ndim > 1 else 1,
	)

	return True


def main () -> None:

	if shutil.which("fluidsynth") is None:
		sys.exit("fluidsynth not found.  Install it:  apt install fluidsynth")

	if len(sys.argv) < 2:
		print("Usage: python scripts/extract_gm_drums.py <soundfont.sf2> [--output <dir>]", file=sys.stderr)
		sys.exit(1)

	sf2_path = pathlib.Path(sys.argv[1])

	if not sf2_path.exists():
		sys.exit(f"SoundFont not found: {sf2_path}")

	# Optional output directory.
	output_dir = pathlib.Path("samples/reference")

	if "--output" in sys.argv:
		idx = sys.argv.index("--output")
		if idx + 1 < len(sys.argv):
			output_dir = pathlib.Path(sys.argv[idx + 1])

	output_dir.mkdir(parents=True, exist_ok=True)

	_log.info("SoundFont: %s", sf2_path)
	_log.info("Output:    %s", output_dir)
	_log.info("Extracting %d GM percussion instruments (notes 35-81)", len(GM_PERCUSSION))

	success = 0
	failed  = 0

	with tempfile.TemporaryDirectory(prefix="subsample_gm_") as tmpdir:
		tmp = pathlib.Path(tmpdir)

		for note, name in GM_PERCUSSION.items():

			stem = f"GM{note}_{name}"
			wav_path = output_dir / f"{stem}.wav"
			midi_path = tmp / f"{stem}.mid"

			_log.info("[%d/%d] %s", note - 34, len(GM_PERCUSSION), stem)

			# 1. Generate MIDI file.
			_make_midi_file(note, midi_path)

			# 2. Render via fluidsynth.
			raw_wav = tmp / f"{stem}_raw.wav"

			if not _render_with_fluidsynth(sf2_path, midi_path, raw_wav):
				failed += 1
				continue

			# 3. Read, trim, and save to output directory.
			try:
				data, sr = soundfile.read(str(raw_wav), always_2d=True, dtype="float32")
			except (OSError, soundfile.SoundFileError) as exc:
				_log.error("Cannot read rendered file %s: %s", raw_wav, exc)
				failed += 1
				continue

			data = _trim_silence(data)

			if len(data) < 1000:
				_log.warning("Skipping %s — too short (%d frames)", stem, len(data))
				failed += 1
				continue

			soundfile.write(str(wav_path), data, sr, subtype="PCM_16")

			# 4. Analyze and write sidecar.
			if _analyze_and_save(wav_path):
				success += 1
			else:
				failed += 1
				continue

			# 5. Copy the sidecar to the shipped location (subsample/data/reference/)
			#    — the tracked, wheel-packaged copy. WAVs stay local-only.
			sidecar = wav_path.parent / (wav_path.name + subsample.cache.SIDECAR_SUFFIX)
			shipped = subsample.config.data_dir() / "reference" / sidecar.name
			shutil.copyfile(sidecar, shipped)

	_log.info("Done: %d succeeded, %d failed", success, failed)
	_log.info("Shipped sidecars updated in %s — commit those", subsample.config.data_dir() / "reference")

	if failed > 0:
		sys.exit(1)


if __name__ == "__main__":
	main()
