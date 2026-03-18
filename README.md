# Subsample

Subsample is an ambient audio sample recorder written in Python. It continuously monitors an audio input, learns the ambient noise floor, and automatically records any sound that rises above it - saving each event as a timestamped WAV file.

## Quick start

```bash
# Copy the default config and adjust to taste
cp config.yaml.default config.yaml

# Run
subsample
```

On first run, Subsample will list available audio input devices and let you choose one (or auto-select if only one is present). It then calibrates ambient noise for a few seconds before listening for events.

## Configuration

All settings live in `config.yaml`. The defaults are:

| Setting | Default | Description |
|---|---|---|
| `audio.sample_rate` | `44100` | Sample rate in Hz |
| `audio.bit_depth` | `16` | Bit depth (16, 24, or 32) |
| `audio.channels` | `1` | 1 = mono, 2 = stereo |
| `audio.chunk_size` | `512` | Frames per buffer read |
| `buffer.max_seconds` | `60` | Circular buffer length |
| `detection.snr_threshold_db` | `12.0` | dB above ambient to trigger recording |
| `detection.hold_time` | `0.5` | Seconds to hold recording open after signal drops |
| `detection.warmup_seconds` | `3.0` | Calibration period before detection activates |
| `detection.ema_alpha` | `0.05` | Ambient noise adaptation speed (lower = slower) |
| `detection.trim_pre_samples` | `10` | Samples to keep before signal onset (S-curve fade applied) |
| `detection.trim_post_samples` | `25` | Samples to keep after signal end (S-curve fade applied) |
| `output.directory` | `./samples` | Where WAV files are saved |
| `output.filename_format` | `%Y-%m-%d_%H-%M-%S` | strftime format for filenames |
| `analysis.start_bpm` | `120.0` | Tempo prior for beat detection (BPM) |
| `analysis.tempo_min` | `30.0` | Minimum tempo considered by pulse detector (BPM) |
| `analysis.tempo_max` | `300.0` | Maximum tempo considered by pulse detector (BPM) |

## Output

Recordings are saved as uncompressed 16, 24, or 32-bit WAV files (depending on the `audio.bit_depth` setting) in the configured output directory, named by the datetime the recording ended:

```
samples/
  2026-03-17_14-32-01.wav
  2026-03-17_14-35-44.wav
```

## Analyzing recorded files

You can analyze any WAV file (recorded by Subsample or elsewhere) using the `analyze_file` script:

```bash
python scripts/analyze_file.py samples/2026-03-17_14-32-01.wav
```

This runs the same analysis pipeline used during live capture and prints three lines:

```
rhythm:   tempo=120.2bpm  beats=4  pulses=12  onsets=4
spectral: duration=2.00s  flatness=0.001  attack=0.000  release=0.812  centroid=0.018  bandwidth=0.001  zcr=0.120  harmonic=0.821  contrast=0.310  voiced=0.940
pitch:    pitch=440.0Hz  chroma=A  pitch_conf=0.89
```

The first line shows rhythm properties (not normalised):
- **tempo** - Estimated global tempo in BPM (0.0 if no beat detected)
- **beats** - Number of beat positions detected on a regular grid
- **pulses** - Number of local pulse peaks from the PLP algorithm
- **onsets** - Number of transient onsets detected (start of each audible event)

The second line shows spectral metrics, each in the range [0.0, 1.0]:
- **duration** - Recording length in seconds
- **flatness** - 0 = tonal (e.g. sine wave), 1 = noisy
- **attack** - 0 = instant/percussive onset, 1 = gradual build-up
- **release** - 0 = sudden stop, 1 = long sustain/decay tail
- **centroid** - 0 = bassy/low-frequency, 1 = trebly/high-frequency
- **bandwidth** - 0 = narrow/pure tone, 1 = wide/spectrally complex
- **zcr** - Zero crossing rate: 0 = smooth/DC, 1 = maximally noisy
- **harmonic** - Harmonic energy fraction (HPSS): 0 = percussive, 1 = harmonic/tonal
- **contrast** - Spectral peak-vs-valley contrast: 0 = flat spectrum, 1 = strong peaks
- **voiced** - Fraction of frames with detected pitch: 0 = unpitched, 1 = clearly pitched

The third line shows pitch and timbre data (raw values, not normalised):
- **pitch** - Dominant fundamental frequency in Hz, or "none" for unpitched audio
- **pitch** - Dominant fundamental frequency in Hz, or "none" for unpitched audio
- **chroma** - Dominant pitch class (C, C#, D, … B), or "none" for unpitched audio
- **pitch_conf** - pyin pitch detection confidence [0, 1] (use with `voiced` to judge reliability)

## Requirements

- Python 3.12+
- PortAudio (required by PyAudio - install via your system package manager: `apt install portaudio19-dev` or `brew install portaudio`)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy subsample
```
