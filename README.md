# Subsample

*A combine harvester for sound.*

Subsample is an automatic field recorder, sample analyzer, and MIDI playback device.

Point a microphone at the world and Subsample continuously listens — automatically
capturing every usable sound, trimming the silence, and analyzing each clip for its
spectral, rhythmic, and pitch character.

Most samplers ask you to manually record a chunk of audio, chop it up, and figure out
where each piece belongs. Subsample automates the entire pipeline: it harvests sounds
from a live stream or a pre-recorded file, sorts them intelligently (tonal vs.
percussive, pitched vs. unpitched), and builds a feature vector for each one.

The end goal is a live instrument — sounds are assigned to MIDI notes as they are
discovered, pitch-mapped tonal samples become playable across a keyboard, and an
external controller can trigger the whole collection in real time. The chaotic
environment becomes an instant, organised sample pack.

## Implemented

- **Live audio capture** — continuous monitoring via PyAudio callback mode; adaptive
  SNR-triggered recording using exponential moving average ambient tracking;
  auto-silence trimming with S-curve (half-cosine) fade in/out; timestamped WAV output.
- **Stereo capture** — recordings preserve the original channel count. Set
  `recorder.audio.channels: null` (or omit it) to auto-detect from the device; a
  stereo USB microphone will record and play back in stereo without any manual setting.
- **Device selection** — specify the input device by name in `config.yaml`; falls back
  to interactive menu or auto-select when one device is present.
- **Rich acoustic feature analysis** — each sample is characterised by:
  - *11 normalised spectral metrics* — flatness, attack, release, centroid, bandwidth,
    zero-crossing rate, harmonic ratio, spectral contrast, voiced fraction, log-attack
    time, spectral flux; all in [0, 1] for direct comparison.
  - *Rhythm analysis* — BPM, beat grid, pulse curve (PLP algorithm), onset times.
  - *Pitch analysis* — fundamental frequency (pyin), dominant pitch class, full 12-bin
    chroma profile, pitch confidence, pitch stability (semitone std dev across voiced
    frames), voiced frame count.
  - *Timbre fingerprinting* — three independently computed 13-coefficient MFCC vectors:
    mean (steady-state timbre), delta (how timbre changes over time), and onset-weighted
    (emphasises the attack portion). Together these encode both the tonal colour and the
    physical envelope of the sound.
  - *Amplitude metadata* — peak and RMS amplitude measured on the normalised signal,
    stored for per-sample gain normalisation at playback time.
- **Intelligent stable-pitch detection** — before producing pitch variants, a seven-criterion
  filter checks that a sample has a single, confident, stable pitch worth shifting:
  voiced fraction > 50%, at least 5 voiced frames (~60 ms), pyin confidence > 50%,
  pitch stability < 0.5 semitones, harmonic ratio > 40%, dominant Hz > 0, and duration
  ≥ 100 ms. Only samples that pass all seven produce variants; percussive sounds are
  silently skipped.
- **Composite similarity matching** — compare each recording against reference samples
  using cosine similarity on a 47-element weighted feature vector built from four
  independently L2-normalised groups: spectral shape (11 metrics), sustained MFCC timbre
  (coefficients 1–12), delta-MFCC timbre trajectory (1–12), and onset-weighted MFCC
  attack character (1–12). Each group's weight is configurable via `similarity.weight_*`
  in `config.yaml`. For each reference, an in-memory ranked list of instrument matches
  is maintained and updated incrementally as new recordings arrive or old ones are evicted.
- **Similarity-ranked note routing** — the GM drum note map resolves each MIDI note to
  `(reference_name, rank, one_shot)`. `rank=0` triggers the most-similar recorded sample;
  `rank=1` the second-most-similar. Kick_1 and Kick_2 therefore automatically select
  different recorded hits from the same reference class. If a second hit has not yet been
  recorded, it falls back to rank 0.
- **Reference sample library** — load pre-analyzed reference sounds from disk; audio
  files not required (`.analysis.json` sidecars only). Lookups are case-insensitive.
- **Instrument sample library** — every recording is added to an in-memory library with
  its PCM audio and full analysis. A configurable memory cap (default 100 MB) keeps the
  newest samples in RAM using FIFO eviction. An optional startup directory pre-populates
  the library from previously recorded files.
- **File input mode** — process WAV files through the detector pipeline by passing them
  as positional arguments (`subsample recording.wav` or `subsample ./recordings/*.wav`).
  Files are processed at their native sample rate, bit depth, and channel count; detected
  segments are saved with the original filename stem plus an index. Useful for batch
  processing, testing on known material, and building sample libraries without live
  capture hardware.
- **32-bit internal processing pipeline** — all audio is converted to 32-bit float
  immediately after capture and kept in that format through every stage of analysis,
  normalisation, pitch shifting, and mixing. The 32-bit float format has 24 bits of
  mantissa precision — lossless for 16-bit sources, and effectively lossless for 24-bit
  sources. This means peak-normalising a quiet sample or pitch-shifting it across two
  octaves introduces no measurable rounding error. The only integer conversion is the
  final output pack to the hardware's native bit depth (16, 24, or 32-bit PCM), which
  happens once per audio callback frame.
- **Variants tailored to the playback device** — when a sample enters the library, a
  background worker immediately produces a pre-converted copy (the *base variant*):
  float32, peak-normalised, resampled to the output device's sample rate if it differs
  from the recorder rate. Tonal samples also receive a full set of pitch-shifted
  variants, each at the same output format. By the time the first note fires, the work
  is already done — playback is a zero-conversion memory copy, not an on-the-fly
  calculation. The sample rate conversion uses `librosa.resample` (soxr high-quality
  algorithm). Configure the output rate via `player.audio.sample_rate`; if unset, the
  recorder's rate is used.
- **MIDI-triggered polyphonic sample playback** — `player.enabled: true` opens a MIDI
  input device and plays the best-matching instrument sample for each note trigger via a
  callback-based mix buffer. Multiple voices play simultaneously. Device selected by
  name substring match, auto-select, or interactive menu.
- **Three-tier playback lookup** — when a note fires, the player looks up: (1) a
  pre-computed pitch variant if the sample is tonal, (2) the base variant (pre-normalised
  float32 copy) if the pitch variant is not ready, (3) on-the-fly PCM conversion as a
  last resort for the very first trigger only. After the first trigger, subsequent triggers
  always use the pre-computed copy with zero conversion overhead.
- **One-shot / note-off handling** — each drum note carries a `one_shot` flag. Kicks,
  snares, and cymbals play to natural completion regardless of key-release (one-shot).
  Hi-hats respond to `note_off`, enabling the standard closed-pedal-silences-open-hat
  behaviour. A 10 ms cosine fade-out on `note_off` prevents clicks when cutting tonal samples.
- **Virtual MIDI input port** — set `player.virtual_midi_port: "Subsample Virtual MIDI"`
  to create a named virtual MIDI input port at startup instead of connecting to a hardware
  device. This is the primary way to drive Subsample from another application running on
  the same machine — for example, a Python sequencer such as
  [Subsequence](https://github.com/simonholliday/subsequence) can send a drum pattern
  directly to Subsample's virtual port without any physical MIDI hardware. From the
  sequencer's side, Subsample's port appears as a MIDI output destination while Subsample
  is running. Overrides `player.midi_device`.

  > **Performance note:** running a MIDI sequencer and Subsample simultaneously on the
  > same machine means two real-time workloads compete for CPU and I/O. This works well
  > on a modern multi-core machine but may cause xruns or timing drift on lower-powered
  > hardware. If you experience dropouts, reduce `recorder.audio.chunk_size`, lower the
  > sequencer's buffer size, or disable the recorder (`recorder.enabled: false`) to run
  > Subsample in playback-only mode.

- **Per-sample gain normalisation** — every sample carries `LevelResult` (peak and RMS
  on the normalised signal). At playback: `gain = target_rms / sample.rms × velocity²`,
  clamped by `1.0 / sample.peak` to prevent clipping. Quiet samples recorded in a noisy
  room and loud transients recorded close-up sit at the same perceived level without
  modifying the stored audio.
- **Automatic pitch variants** — tonal samples with a stable, confident pitch are
  automatically pitch-shifted to every MIDI note within ±12 semitones of the detected
  pitch (configurable via `transform.pitch_range_semitones`; default = 25 variants per
  sample). Uses Rubber Band via `pyrubberband` in offline mode with the finer engine for
  highest quality. Variants are produced in the background by a worker pool; the player
  falls back gracefully until a variant is ready.
- **Output format control** — the output stream format defaults to the recorder's sample
  rate and bit depth, preserving capture quality end-to-end. Override with
  `player.audio.sample_rate` and `player.audio.bit_depth` if the output device requires
  a different format (e.g. 16-bit only). Never configure the output higher than the
  source — upsampling adds no quality.
- **Versioned analysis cache with auto-invalidation** — each `.analysis.json` sidecar
  carries an `analysis_version` string and an MD5 of the WAV file. On startup, stale
  sidecars are detected and re-analysed automatically. The re-analysis log message
  includes the old → new version (`analysis version 7 → 8`) so the cause is always
  clear. Sidecars also record `bit_depth`, `channels`, and `captured_at` (ISO 8601
  timestamp for live captures; null for imported files), making them self-describing.
- **Analysis file script** — `scripts/analyze_file.py` runs the same analysis pipeline
  on any local WAV file and prints formatted results.
- **Similarity report script** — `scripts/similarity_report.py` prints the top-N most
  similar instrument samples for each reference, using the same config and sample IDs as
  the live application.

## In Progress

- **Automatic sample classification** — infrastructure in place; next: wire ranked match
  results to a simple classifier (e.g. "if top reference match is KICK, classify as KICK").
- **Time-stretch and envelope transforms** — `TimeStretch` and `EnvelopeAdjust` dataclasses
  are scaffolded; only the apply-function implementations and handler registrations remain.
  See `transform.target_bpm` in `config.yaml`.
- **Pitch-variant playback test channel** — MIDI channel 1 (mido ch 0) is wired as an
  exploratory channel that maps notes directly to cached pitch variants, bypassing the
  reference-similarity routing used by channel 10. This gives end-to-end hardware test
  coverage for the transform pipeline. The mapping strategy and channel constant will
  move to config in a future iteration.

## Planned

- **BPM time-stretching** — `TransformManager.on_bpm_change()` and `transform.target_bpm`
  are in place; only `_apply_time_stretch()` remains to implement and register.
- **Envelope shaping** — `EnvelopeAdjust` dataclass (attack/release in ms) is defined;
  only `_apply_envelope()` remains to implement and register.
- **Parallel startup re-analysis** — when the analysis version bumps, stale sidecars are
  currently re-analysed sequentially; for large libraries this can block for several
  minutes. Re-analysis should be parallelised using the existing `SampleProcessor` pool.
- **Multi-band energy envelope** — split the spectrum into 3–5 frequency bands and
  compute per-band peak energy and decay rate. This would directly encode the physical
  signature of drum types (kick = sub-bass dominant; snare = mid + presence; hi-hat =
  air) and improve classification and similarity for percussion.
- **Config-driven note mapping** — move MIDI channel, note range, and note→reference
  mapping from hard-coded constants to `config.yaml`; allows different pad layouts and
  instruments without code changes.
- **Mix management** — per-voice gain staging to prevent clipping when many samples
  overlap; configurable mixing strategy.
- **Interactive classification** — live adjustment of classification thresholds; manual
  sample reassignment during a session.
- **Independent modes** — recording, analysis, MIDI assignment, and playback can each
  run separately or together.

## How it works

Subsample is built around three concurrent pipelines that interact through thread-safe
shared state.

### Live capture pipeline

```
PortAudio callback → raw PCM bytes → unpack_audio() → CircularBuffer
                                                               ↓
                                              LevelDetector.process_chunk()
                                              (EMA ambient tracking + SNR gate)
                                                               ↓
                                              trim_silence() → segment PCM
                                                               ↓
                                              SampleProcessor worker pool
                                              (auto-scaled: (cpu_count - 2) / 2)
                                                               ↓
                           to_mono_float() → analyze_all() → WAV + sidecar + SampleRecord
```

The input thread is never blocked waiting for analysis. Back-to-back sounds are captured
reliably even when analysis is slow — worker threads handle each recording concurrently
and independently.

### Similarity engine

Every new instrument sample is scored against every reference using cosine similarity on
a 47-element composite vector. The vector is split into four groups, each independently
L2-normalised so that no single group dominates by scale:

```
Group 1 (×11): spectral shape   [flatness, attack, release, centroid, bandwidth, zcr,
                                  harmonic, contrast, voiced, log_attack, flux]
Group 2 (×12): sustained MFCC   [mean timbre, coefficients 1–12]
Group 3 (×12): delta-MFCC       [timbre trajectory, coefficients 1–12]
Group 4 (×12): onset-weighted   [attack character, coefficients 1–12]
```

Each group is scaled by a configurable weight (`similarity.weight_*`). This design means
the same comparison method works for both percussive (attack character dominates) and
tonal (sustained timbre dominates) sounds without needing to classify them first.

### Transform pipeline

```
SampleRecord added to library
    → TransformManager.on_sample_added()
        → enqueue base variant (always)         ← float32 peak-normalised copy
        → enqueue pitch variants (tonal only)   ← Rubber Band offline finer engine
            → TransformProcessor worker pool
                → TransformCache (parent-priority FIFO eviction, 50 MB default)
```

The base variant (identity spec: no DSP) is produced for every sample — percussive and
tonal alike — so the playback path never pays the float32 conversion cost at trigger
time. Pitch variants are additional cache entries, derived from the same PCM source.

When a variant set for a parent sample would exceed the memory budget, the entire oldest
parent's variant family is evicted together, keeping the remaining families intact and
playable.

### Playback path

```
MIDI note_on (channel 10)
    → similarity_matrix.get_match(ref_name, rank) → sample_id
    → transform_manager.get_pitched()  → pitch variant (tonal samples)
    → transform_manager.get_base()     → base variant  (all samples)
    → _render()                        → on-the-fly fallback (first trigger only)
    → _render_float(): apply gain · velocity² · anti-clip ceiling
    → append _Voice (float32 stereo, pre-rendered)
    ↓
PyAudio callback (PortAudio high-priority thread)
    → sum all active voices (float32 addition)
    → clip to [-1, 1]
    → float32_to_pcm_bytes(mixed, output_bit_depth)  → int16/24/32 bytes to hardware
```

All mixing happens in float32 — the only integer conversion is the final output packing.
Multiple simultaneous voices are summed correctly regardless of the output device's bit depth.

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

**Live capture mode:** Subsample lists available audio input devices and lets you choose
one (or auto-selects if only one is present). It calibrates ambient noise for a few
seconds before listening for events.

**File input mode:** Each file is processed at its native sample rate, bit depth, and
channel count. Detected segments are saved to the output directory.

## Configuration

All settings live in `config.yaml`. The defaults are:

| Setting | Default | Description |
|---|---|---|
| `recorder.enabled` | `true` | Enable live audio capture; set to `false` to process files only |
| `recorder.audio.device` | `none` | Audio input device name (substring match); if unset, auto-select or prompt |
| `recorder.audio.sample_rate` | `44100` | Sample rate in Hz |
| `recorder.audio.bit_depth` | `16` | Bit depth (16, 24, or 32) |
| `recorder.audio.channels` | auto | 1 = mono, 2 = stereo. Omit (or `null`) to auto-detect from device |
| `recorder.audio.chunk_size` | `512` | Frames per buffer read |
| `recorder.buffer.max_seconds` | `60` | Circular buffer length |
| `player.enabled` | `false` | Enable the MIDI player |
| `player.midi_device` | `none` | MIDI input device name (substring match); if unset, auto-select or prompt |
| `player.audio.device` | `none` | Audio output device name for playback |
| `player.audio.sample_rate` | auto | Output sample rate; defaults to recorder rate. Do not set higher than source. |
| `player.audio.bit_depth` | auto | Output bit depth (16, 24, or 32); defaults to recorder bit depth |
| `player.virtual_midi_port` | `none` | Name for a virtual MIDI input port; overrides `player.midi_device` |
| `detection.snr_threshold_db` | `12.0` | dB above ambient to trigger recording |
| `detection.hold_time` | `0.5` | Seconds to hold recording open after signal drops |
| `detection.warmup_seconds` | `1.0` | Calibration period before detection activates |
| `detection.ema_alpha` | `0.1` | Ambient noise adaptation speed (lower = slower) |
| `detection.trim_pre_samples` | `15` | Samples to keep before signal onset (S-curve fade applied) |
| `detection.trim_post_samples` | `85` | Samples to keep after signal end (S-curve fade applied) |
| `output.directory` | `./samples` | Where WAV files are saved |
| `output.filename_format` | `%Y-%m-%d_%H-%M-%S` | strftime format for filenames |
| `analysis.start_bpm` | `120.0` | Tempo prior for beat detection (BPM) |
| `analysis.tempo_min` | `30.0` | Minimum tempo considered by pulse detector (BPM) |
| `analysis.tempo_max` | `300.0` | Maximum tempo considered by pulse detector (BPM) |
| `instrument.max_memory_mb` | `100.0` | Max audio memory for in-memory samples; oldest evicted (FIFO) when exceeded |
| `instrument.directory` | `none` | Optional directory to pre-load instrument samples from at startup |
| `reference.directory` | `none` | Optional directory of reference sounds for similarity classification |
| `similarity.weight_spectral` | `1.0` | Weight for the spectral shape group (11 metrics) |
| `similarity.weight_timbre` | `1.0` | Weight for sustained MFCC timbre (coefficients 1–12) |
| `similarity.weight_timbre_delta` | `0.5` | Weight for delta-MFCC timbre trajectory |
| `similarity.weight_timbre_onset` | `1.0` | Weight for onset-weighted MFCC attack character |
| `transform.max_memory_mb` | `50.0` | Memory budget (MB) for pitch-shifted variants |
| `transform.auto_pitch` | `true` | Auto-produce pitch variants for tonal samples |
| `transform.pitch_range_semitones` | `12` | Semitones above/below detected pitch for auto-variants (12 = ±1 octave = 25 notes) |
| `transform.target_bpm` | `0.0` | Target BPM for automatic time-stretch variants; 0.0 disables |

## Output

Recordings are saved as uncompressed 16, 24, or 32-bit WAV files (depending on
`recorder.audio.bit_depth`) in the configured output directory.

**Live capture mode** — filenames from the datetime the recording ended:

```
samples/
  2026-03-17_14-32-01.wav
  2026-03-17_14-35-44.wav
```

**File input mode** — filenames from the original audio file's stem plus a segment index:

```
samples/
  field_recording_1.wav
  field_recording_2.wav
```

Both modes write to the same output directory. Point `instrument.directory` at the same
path to get a persistent library that grows on disk across sessions.

## Instrument sample library

Every recording is automatically added to an in-memory instrument library alongside its
full analysis data. A configurable memory cap (default 100 MB) prevents unbounded growth;
the oldest samples are evicted when a new one would exceed the limit. WAV files on disk
are never deleted.

### Persistent library across sessions

```yaml
output:
  directory: "./samples"

instrument:
  directory: "./samples"
```

On startup, Subsample pre-loads all existing WAV files from `./samples`. As new recordings
arrive they are written to disk and added to memory in one step. The memory cap keeps only
the most recent window of captures in RAM; the full archive on disk is unaffected.

## Reference sample library

Reference samples define the canonical sound classes you want to match against — kick
drum, snare, hi-hat, etc. Each reference is represented by its `.analysis.json` sidecar
file only; the original audio is not required.

```yaml
reference:
  directory: "./reference"
```

Place one sidecar per sound class in the reference directory. The name is taken from
the audio filename stem:

```
reference/
  BD0025.wav.analysis.json   →  "BD0025"
  SD5075.wav.analysis.json   →  "SD5075"
  CH.wav.analysis.json       →  "CH"
```

At startup, reference samples are loaded before instrument samples. For every instrument
sample, Subsample computes cosine similarity against every reference and maintains a
ranked list per reference — most similar instrument first. When a sample is evicted from
the instrument library, it is also removed from the ranked lists.

Query the ranked lists programmatically:

```python
# Most kick-like instrument in memory
sample_id = similarity_matrix.get_match("BD0025", rank=0)

# Second-most kick-like (for a separate kick_2 mapping)
sample_id = similarity_matrix.get_match("BD0025", rank=1)
```

Lookup is case-insensitive.

## Scripts

### Analyzing recorded files

```bash
python scripts/analyze_file.py samples/2026-03-17_14-32-01.wav
```

Output:
```
rhythm:   tempo=120.2bpm  beats=4  pulses=12  onsets=4
spectral: duration=2.00s  flatness=0.001  attack=0.000  release=0.812  centroid=0.018  bandwidth=0.001  zcr=0.120  harmonic=0.821  contrast=0.310  voiced=0.940  log_attack=0.000  flux=0.312
pitch:    pitch=440.0Hz  chroma=A  pitch_conf=0.89
level:    peak=0.8743 (-1.2dBFS)  rms=0.2341 (-12.6dBFS)
```

Spectral metrics (all [0, 1]):
- **flatness** — 0 = tonal, 1 = noisy
- **attack** — 0 = instant/percussive, 1 = gradual build-up
- **release** — 0 = sudden stop, 1 = long decay tail
- **centroid** — 0 = bassy, 1 = trebly
- **bandwidth** — 0 = pure tone, 1 = spectrally complex
- **zcr** — zero crossing rate: 0 = smooth, 1 = maximally noisy
- **harmonic** — 0 = percussive, 1 = harmonic/tonal (HPSS)
- **contrast** — 0 = flat spectrum, 1 = strong spectral peaks
- **voiced** — fraction of frames with detected pitch
- **log_attack** — 0 = instant spectral onset, 1 = very slow
- **flux** — 0 = static spectrum, 1 = rapidly evolving

Pitch data (raw values):
- **pitch** — dominant fundamental frequency in Hz, or "none" for unpitched audio
- **chroma** — dominant pitch class (C–B), or "none"
- **pitch_conf** — pyin confidence [0, 1]; use with `voiced` to judge reliability

Amplitude metadata:
- **peak** — peak absolute amplitude [0, 1] with dBFS equivalent
- **rms** — RMS loudness [0, 1] with dBFS equivalent; drives playback gain normalisation

Three MFCC timbre fingerprints are stored in the sidecar (used for similarity, not shown
in script output): `mfcc` (mean, average timbre), `mfcc_delta` (first-order trajectory),
and `mfcc_onset` (onset-weighted, attack emphasis).

### Similarity report

```bash
python scripts/similarity_report.py           # top 5 per reference (default)
python scripts/similarity_report.py --top 10  # top 10 per reference
```

Example output:
```
Reference: BD0025
  1.  #5     0.9412  BD0025          ./samples/BD0025.wav
  2.  #7     0.8134  kick_hard       ./samples/kick_hard.wav
  3.  #8     0.7601  kick_soft       ./samples/kick_soft.wav
```

## Requirements

- Python 3.12+
- PortAudio (required by PyAudio — `apt install portaudio19-dev` or `brew install portaudio`)
- Rubber Band (required by pyrubberband — `apt install rubberband-cli` or `brew install rubberband`)

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Type Checking

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
| [mido ↗](https://github.com/mido/mido) | MIDI message parsing and I/O | MIT |
| [python-rtmidi ↗](https://github.com/SpotlightKid/python-rtmidi) | MIDI device access (RtMidi bindings) | MIT |
| [pyrubberband ↗](https://github.com/bmcfee/pyrubberband) | Pitch shifting (Rubber Band wrapper) | ISC |

## About the Author

Subsample was created by me, Simon Holliday ([simonholliday.com ↗](https://simonholliday.com/)), a senior technologist and a junior (but trying) musician. From running an electronic music label in the 2000s to prototyping new passive SONAR techniques for defence research, my work has often explored the intersection of code and sound.

## License

Subsample is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).

You are free to use, modify, and distribute this software under the terms of the AGPL. If you run a modified version of Subsample as part of a network service, you must make the source code available to its users.

All runtime dependencies are permissively licensed (MIT, ISC, BSD-3-Clause) and compatible with AGPLv3.

## Commercial licensing

If you wish to use Subsample in a proprietary or closed-source product without the obligations of the AGPL, please contact [simon.holliday@protonmail.com] to discuss a commercial license.
