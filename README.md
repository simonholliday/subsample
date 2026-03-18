# Subsample

*A combine harvester for sound.*

Subsample is an automatic field recorder, sample analyzer, and MIDI playback device.

Point a microphone at the world, and Subsample continuously listens - automatically
capturing every usable sound, trimming the silence, and analyzing each clip for its
spectral, rhythmic, and pitch character.

Most samplers ask you to manually record a chunk of audio, chop it up, and figure out
where each piece belongs. Subsample automates the entire pipeline: it harvests sounds
from a live stream or a pre-recorded file, sorts them intelligently (tonal vs.
percussive, pitched vs. unpitched), and builds a feature vector for each one.

The end goal is a live instrument - sounds are assigned to MIDI notes as they are
discovered, pitch-mapped tonal samples become playable across a keyboard, and an
external controller can trigger the whole collection in real time. The chaotic
environment becomes an instant, organized sample pack.

## Implemented

- **Live audio capture** - continuous monitoring via PyAudio callback mode; SNR-triggered
  recording with configurable threshold; auto-silence trimming with S-curve fade in/out;
  timestamped WAV output.
- **Device selection** - specify the input device by name in `config.yaml`; falls back to
  interactive menu or auto-select when one device is present.
- **Feature analysis** - 9 normalised spectral metrics (flatness, attack, release,
  centroid, bandwidth, zero-crossing rate, harmonic ratio, spectral contrast, voiced
  fraction); rhythm analysis (BPM, beats, pulses, onsets); pitch analysis (fundamental
  frequency, pitch class, chroma profile, MFCC timbre fingerprint).
- **File analysis** - `scripts/analyze_file.py` runs the same analysis pipeline on any
  local audio file (WAV, FLAC, AIFF, OGG, etc.).

## Planned

- **Sample similarity** - compare feature vectors (MFCC, spectral metrics) between
  recordings using cosine distance to find the most "kick-like", "snare-like", etc.
  sample from the library.
- **Canonical sound training** - define reference sounds (kick, snare, hi-hat, …) from
  example recordings; new samples are automatically classified by similarity.
- **MIDI note assignment** - allocate MIDI trigger notes based on sample classification;
  notes can be replaced in real time as better examples arrive; manual override supported.
- **MIDI playback device** - receive MIDI note triggers and play back the assigned sample
  for each note.
- **Pitch re-mapping** - recognise tonal samples and map them across a keyboard range
  centered on their recorded pitch, so a single sample can be played at any pitch.
- **BPM time-stretching** - warp rhythmic samples to a target tempo to fit a song or loop.
- **Sample library** - load previously collected samples, re-analyze them, and assign
  them to MIDI notes alongside newly recorded material.
- **Independent modes** - recording, analysis, MIDI assignment, and MIDI playback can
  each run separately or together in real time.

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
| `audio.device` | `none` | Audio input device name (substring match); if unset, auto-select or prompt |
| `audio.sample_rate` | `44100` | Sample rate in Hz |
| `audio.bit_depth` | `16` | Bit depth (16, 24, or 32) |
| `audio.channels` | `1` | 1 = mono, 2 = stereo |
| `audio.chunk_size` | `512` | Frames per buffer read |
| `buffer.max_seconds` | `60` | Circular buffer length |
| `detection.snr_threshold_db` | `12.0` | dB above ambient to trigger recording |
| `detection.hold_time` | `0.5` | Seconds to hold recording open after signal drops |
| `detection.warmup_seconds` | `1.0` | Calibration period before detection activates |
| `detection.ema_alpha` | `0.1` | Ambient noise adaptation speed (lower = slower) |
| `detection.trim_pre_samples` | `15` | Samples to keep before signal onset (S-curve fade applied) |
| `detection.trim_post_samples` | `45` | Samples to keep after signal end (S-curve fade applied) |
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
