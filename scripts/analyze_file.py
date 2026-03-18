"""Analyze a local audio file and print its metrics to the console.

Reads any audio file supported by soundfile (WAV, FLAC, AIFF, OGG, etc.),
runs the same analysis pipeline used during live capture, and prints two
summary lines to stdout: rhythm properties followed by spectral metrics.

Usage:
	python scripts/analyze_file.py <path/to/file.wav>
"""

import logging
import sys

import numpy
import soundfile

import subsample.analysis
import subsample.config


logging.basicConfig(
	level=logging.WARNING,
	format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
	datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)


def main () -> None:

	"""Analyze a single audio file and print its metrics."""

	if len(sys.argv) != 2:
		print("Usage: analyze_file <audio-file>", file=sys.stderr)
		sys.exit(1)

	filepath = sys.argv[1]

	try:
		# soundfile.read with dtype='float32' reads directly as float32, avoiding
		# the default float64 intermediate; always_2d ensures shape is
		# (n_frames, channels) regardless of mono/stereo.
		data, samplerate = soundfile.read(filepath, always_2d=True, dtype='float32')

	except (OSError, soundfile.SoundFileError) as exc:
		print(f"Error reading {filepath}: {exc}", file=sys.stderr)
		sys.exit(1)

	# Mix down to mono — both analysis functions expect shape (n_frames,)
	mono = numpy.mean(data, axis=1, dtype=numpy.float32)  # type: ignore[call-overload]

	params = subsample.analysis.compute_params(samplerate)

	# Use default analysis config (same as live capture defaults)
	rhythm_cfg = subsample.config.AnalysisConfig()

	# Rhythm, spectral, and pitch analysis
	rhythm = subsample.analysis.analyze_rhythm(mono, params, rhythm_cfg)
	result = subsample.analysis.analyze_mono(mono, params)
	pitch = subsample.analysis.analyze_pitch(mono, params)

	duration = len(data) / samplerate
	print(f"rhythm:   {subsample.analysis.format_rhythm_result(rhythm)}")
	print(f"spectral: {subsample.analysis.format_result(result, duration)}")
	print(f"pitch:    {subsample.analysis.format_pitch_result(pitch)}")


if __name__ == "__main__":
	main()
