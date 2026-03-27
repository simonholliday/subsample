"""Subsample: automatic sample harvester and MIDI instrument.

Records audio signals that exceed a configurable SNR threshold above ambient
noise, saving each detected event as a timestamped WAV file with full spectral,
rhythm, pitch, timbre, level, and band energy analysis. Incoming samples are
mapped to MIDI notes via a composable select/process pipeline: select filters,
orders, and picks samples by any combination of analysis metadata; process
declares per-assignment audio transforms (repitch, beat-quantize). Tonal samples
are pitch-shifted across the full assigned note range; samples with sufficient
rhythmic content are beat-quantized and time-stretched to a target tempo. All
variants are produced in the background using Rubber Band's offline finer
engine for highest quality.
"""

