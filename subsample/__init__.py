"""Subsample: automatic sample harvester and MIDI instrument.

Records audio signals that exceed a configurable SNR threshold above ambient
noise, saving each detected event as a timestamped WAV file with full spectral,
rhythm, pitch, timbre, level, and band energy analysis. Incoming samples are
automatically matched to a reference library using cosine similarity and mapped
to MIDI notes for polyphonic playback. Tonal samples are pitch-shifted across
the full assigned note range; samples with sufficient rhythmic content are
beat-quantised and time-stretched to a target tempo. All variants are produced
in the background using Rubber Band's offline finer engine for highest quality.
"""

