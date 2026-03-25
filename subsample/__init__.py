"""Subsample: automatic sample harvester and MIDI instrument.

Records audio signals that exceed a configurable SNR threshold above ambient
noise, saving each detected event as a timestamped WAV file with full spectral,
rhythm, pitch, timbre, level, and band energy analysis. Incoming samples are
automatically matched to a reference library using cosine similarity and mapped
to MIDI notes for polyphonic playback.
"""

