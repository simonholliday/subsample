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
| `audio.chunk_size` | `1024` | Frames per buffer read |
| `buffer.max_seconds` | `60` | Circular buffer length |
| `detection.snr_threshold_db` | `6.0` | dB above ambient to trigger recording |
| `detection.hold_time` | `0.5` | Seconds to hold recording open after signal drops |
| `detection.warmup_seconds` | `3.0` | Calibration period before detection activates |
| `detection.ema_alpha` | `0.01` | Ambient noise adaptation speed (lower = slower) |
| `output.directory` | `./samples` | Where WAV files are saved |
| `output.filename_format` | `%Y-%m-%d_%H-%M-%S` | strftime format for filenames |

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

This runs the same analysis pipeline used during live capture and prints a single line of metrics:

```
duration=0.07s  flatness=0.017  attack=0.414  release=0.000  centroid=0.728  bandwidth=0.757
```

Each metric is in the range [0.0, 1.0]:
- **duration** - Recording length in seconds
- **flatness** - 0 = tonal (e.g. sine wave), 1 = noisy
- **attack** - 0 = instant/percussive onset, 1 = gradual build-up
- **release** - 0 = sudden stop, 1 = long sustain/decay tail
- **centroid** - 0 = bassy/low-frequency, 1 = trebly/high-frequency
- **bandwidth** - 0 = narrow/pure tone, 1 = wide/spectrally complex

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
