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
- **Feature analysis** - 11 normalised spectral metrics (flatness, attack, release,
  centroid, bandwidth, zero-crossing rate, harmonic ratio, spectral contrast, voiced
  fraction, log-attack time, spectral flux); rhythm analysis (BPM, beats, pulses,
  onsets); pitch analysis (fundamental frequency, pitch class, chroma profile, MFCC
  timbre fingerprint, delta-MFCC timbre trajectory, onset-weighted MFCC).
- **File analysis** - `scripts/analyze_file.py` runs the same analysis pipeline on any
  local audio file (WAV, FLAC, AIFF, OGG, etc.).
- **Reference sample library** - load pre-analyzed reference sounds from disk; lookups
  are case-insensitive; audio files not required (only `.analysis.json` sidecars).
- **Spectral fingerprint similarity** - compare the 9-element [0,1] spectral vector
  (spectral flatness, attack, release, centroid, bandwidth, ZCR, harmonic ratio,
  contrast, voiced fraction) against reference samples using cosine similarity; for each
  reference, an in-memory ranked list of instrument matches is maintained and updated
  incrementally as new recordings arrive or old ones are evicted.
- **Instrument sample library** - every recording is added to an in-memory library
  with its PCM audio and full analysis; a configurable memory cap (default 100 MB)
  keeps the newest samples in RAM using FIFO eviction; an optional startup directory
  pre-populates the library from previously recorded files.
- **File input mode** - process WAV files through the detector pipeline; specify files
  as positional arguments on the command line (e.g. `subsample recording.wav` or
  `subsample ./recordings/*.wav`); files are processed using their native sample rate,
  bit depth, and channel count; detected segments are saved with the original filename
  stem plus an index (e.g. field_recording_1.wav, field_recording_2.wav); useful for
  batch processing, testing on known material, and building sample libraries without
  live capture hardware.

## In Progress

- **Extended similarity** - spectral fingerprint ranked lists are in place; next: integrate
  the new percussive features (delta-MFCC, onset-weighted MFCC, log-attack time, spectral
  flux) into the comparison vector to improve snare/cymbal/kick discrimination.
- **Automatic sample classification** - infrastructure in place; next: wire ranked match
  results to a simple classifier (e.g. "if top reference match is KICK, classify as KICK").

## Planned

- **Multi-band energy envelope** - split the spectrum into 3–5 frequency bands (sub-bass,
  low-mid, mid, presence, air) and compute per-band peak energy and decay rate; would
  directly encode the physical signature of drum types (kick = sub-bass dominant; snare =
  mid + presence; hi-hat = air) and improve classification and similarity for percussion.
- **Interactive classification** - allow live adjustment of classification thresholds;
  manually reassign samples to categories during recording session.
- **MIDI note assignment** - allocate MIDI trigger notes based on sample classification;
  notes can be replaced in real time as better examples arrive; manual override supported.
- **Audio output device selection** - choose a playback interface by name in config,
  with the same auto-select / interactive-menu fallback used for audio input.
- **MIDI playback device** - receive MIDI note triggers and play back the assigned sample
  for each note; MIDI note velocity mapped to playback volume at the point of output.
- **Polyphonic playback** - multiple samples can play simultaneously; each active voice
  contributes to the output mix.
- **Mix management** - per-voice gain staging to prevent clipping when multiple samples
  overlap; configurable mixing strategy (e.g. normalise to peak, fixed headroom).
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

# Stream from a live audio input device
subsample

# Or process audio files through the detection pipeline
subsample recording.wav                # Single file
subsample ./recordings/*.wav           # Multiple files (glob expansion)
```

**Live capture mode:** Subsample will list available audio input devices and let you choose one (or auto-select if only one is present). It then calibrates ambient noise for a few seconds before listening for events.

**File input mode:** Provide one or more WAV file paths as positional arguments. Each file is processed using its native sample rate, bit depth, and channel count; detected segments are saved to the output directory with names like `recording_1.wav`, `recording_2.wav`, etc. File processing happens before live capture starts (if configured).

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
| `instrument.max_memory_mb` | `100.0` | Max audio memory for in-memory instrument samples; oldest evicted (FIFO) when exceeded |
| `instrument.directory` | `none` | Optional directory to pre-load instrument samples from at startup (WAV + sidecar required) |
| `reference.directory` | `none` | Optional directory of reference sounds for similarity classification (sidecar only, no audio required); if unset, similarity features are disabled |

## Output

Recordings are saved as uncompressed 16, 24, or 32-bit WAV files (depending on the `audio.bit_depth` setting) in the configured output directory.

**Live capture mode** - filenames are generated from the datetime the recording ended (strftime format controlled by `output.filename_format`):

```
samples/
  2026-03-17_14-32-01.wav
  2026-03-17_14-35-44.wav
```

**File input mode** - filenames are derived from the original audio file's stem plus a segment index:

```
samples/
  field_recording_1.wav
  field_recording_2.wav
  field_recording_3.wav
```

Both modes write to the same output directory, which is typical when pointing `instrument.directory` at the same path (see **Persistent library across sessions** below).

## Instrument sample library

Every recording is automatically added to an in-memory instrument library alongside its full analysis data. The library is the foundation for MIDI playback: samples will be assigned to notes as classification develops.

A configurable memory cap (default 100 MB, `instrument.max_memory_mb`) prevents unbounded growth. When a new sample would push usage over the limit, the oldest samples are evicted from memory - newest captures are always retained. WAV files on disk are never deleted.

### Persistent library across sessions

Point `instrument.directory` at the same path as `output.directory` to get a library that grows on disk and stays fresh in memory:

```yaml
output:
  directory: "./samples"

instrument:
  directory: "./samples"
```

On startup, Subsample pre-loads all existing WAV files from `./samples` into memory. As new recordings arrive they are written to disk and added to the in-memory library in one step. The memory cap keeps only the most recent `max_memory_mb` worth of audio in RAM; the full archive on disk is unaffected. Across sessions the collection on disk grows indefinitely; in memory it always holds the freshest window of captures.

## Reference sample library

Reference samples define the canonical sound classes you want to match against - kick drum, snare, hi-hat, etc. Each reference is represented by its `.analysis.json` sidecar file only; the original audio is not required.

```yaml
reference:
  directory: "./reference"
```

Place one sidecar per sound class in the reference directory. The name is taken from the audio filename stem:

```
reference/
  BD0025.wav.analysis.json   →  "BD0025"
  SD5075.wav.analysis.json   →  "SD5075"
  CH.wav.analysis.json       →  "CH"
```

At startup, reference samples are loaded before instrument samples. For every instrument sample (pre-loaded at startup or captured live), Subsample computes cosine similarity against every reference and maintains a ranked list per reference - most similar instrument first. When a sample is evicted from the instrument library, it is also removed from the ranked lists.

You can query the ranked lists programmatically:

```python
# Most kick-like instrument in memory
sample_id = similarity_matrix.get_match("BD0025", rank=0)
```

Lookup is case-insensitive.

## Analyzing recorded files

You can analyze any WAV file (recorded by Subsample or elsewhere) using the `analyze_file` script:

```bash
python scripts/analyze_file.py samples/2026-03-17_14-32-01.wav
```

This runs the same analysis pipeline used during live capture and prints three lines:

```
rhythm:   tempo=120.2bpm  beats=4  pulses=12  onsets=4
spectral: duration=2.00s  flatness=0.001  attack=0.000  release=0.812  centroid=0.018  bandwidth=0.001  zcr=0.120  harmonic=0.821  contrast=0.310  voiced=0.940  log_attack=0.000  flux=0.312
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
- **attack** - 0 = instant/percussive onset, 1 = gradual build-up (RMS energy based)
- **release** - 0 = sudden stop, 1 = long sustain/decay tail
- **centroid** - 0 = bassy/low-frequency, 1 = trebly/high-frequency
- **bandwidth** - 0 = narrow/pure tone, 1 = wide/spectrally complex
- **zcr** - Zero crossing rate: 0 = smooth/DC, 1 = maximally noisy
- **harmonic** - Harmonic energy fraction (HPSS): 0 = percussive, 1 = harmonic/tonal
- **contrast** - Spectral peak-vs-valley contrast: 0 = flat spectrum, 1 = strong peaks
- **voiced** - Fraction of frames with detected pitch: 0 = unpitched, 1 = clearly pitched
- **log_attack** - 0 = instant spectral onset, 1 = very slow (spectral flux based)
- **flux** - Mean spectral flux: 0 = static/barely-changing spectrum, 1 = rapidly evolving

The third line shows pitch and timbre data (raw values, not normalised):
- **pitch** - Dominant fundamental frequency in Hz, or "none" for unpitched audio
- **chroma** - Dominant pitch class (C, C#, D, … B), or "none" for unpitched audio
- **pitch_conf** - pyin pitch detection confidence [0, 1] (use with `voiced` to judge reliability)

Three MFCC timbre fingerprints are computed and stored in each sample's `.analysis.json`
sidecar (not shown in the analysis script output - used for similarity matching):
- **mfcc** - 13 mean MFCC coefficients; captures average timbral character
- **mfcc_delta** - 13 mean delta-MFCC coefficients; captures timbre trajectory (attack-to-decay shift)
- **mfcc_onset** - 13 onset-weighted MFCC coefficients; emphasises the attack portion of the sound

## Requirements

- Python 3.12+
- PortAudio (required by PyAudio - install via your system package manager: `apt install portaudio19-dev` or `brew install portaudio`)

## Tests

This project uses `pytest`. Install test dependencies and run with:

```bash
pip install -e ".[dev]"
pytest
```

## Type Checking

This project uses mypy for static type checking. Run locally with:

```bash
pip install -e ".[dev]"
mypy subsample
```

## Dependencies and Credits

Subsample makes use of these excellent open-source libraries:

| Library | Purpose | License |
|---------|---------|---------|
| [PyAudio ↗](https://people.csail.mit.edu/hubert/pyaudio/) | Audio device I/O (PortAudio bindings) | MIT |
| [PyYAML ↗](https://github.com/yaml/pyyaml) | YAML config loading | MIT |
| [NumPy ↗](https://numpy.org/) | Numerical array operations | BSD-3-Clause |
| [librosa ↗](https://librosa.org/) | Audio analysis (spectral, rhythm, pitch) | ISC |
| [SciPy ↗](https://scipy.org/) | Signal processing (onset detection, filtering) | BSD-3-Clause |
| [SoundFile ↗](https://python-soundfile.readthedocs.io/) | WAV file reading for library pre-load | BSD-3-Clause |

## About the Author

Subsample was created by me, Simon Holliday ([simonholliday.com ↗](https://simonholliday.com/)), a senior technologist and a junior (but trying) musician. From running an electronic music label in the 2000s to prototyping new passive SONAR techniques for defence research, my work has often explored the intersection of code and sound.

## License

Subsample is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).

You are free to use, modify, and distribute this software under the terms of the AGPL. If you run a modified version of Subsample as part of a network service, you must make the source code available to its users.

All runtime dependencies are permissively licensed (MIT, ISC, BSD-3-Clause) and compatible with AGPLv3.

## Commercial licensing

If you wish to use Subsample in a proprietary or closed-source product without the obligations of the AGPL, please contact [simon.holliday@protonmail.com] to discuss a commercial license.
