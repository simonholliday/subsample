# Subsample

Subsample is an ambient audio sample recorder written in Python. It continuously monitors an audio input, learns the ambient noise floor, and automatically records any sound that rises above it — saving each event as a timestamped WAV file.

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

## Requirements

- Python 3.12+
- PortAudio (required by PyAudio — install via your system package manager: `apt install portaudio19-dev` or `brew install portaudio`)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy subsample
```
