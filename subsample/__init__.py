"""Subsample: ambient audio sample recorder and MIDI player.

Records audio signals that exceed a configurable SNR threshold above ambient
noise, saving each detected event as a timestamped WAV file with full spectral,
rhythm, and pitch analysis. Optionally listens for MIDI input to trigger
playback of collected samples.
"""

