# Subsample

*A combine harvester for sound.*

Subsample is an automatic sample harvester and MIDI instrument. Point a
microphone at the world and it captures, analyses, and organises every usable
sound into a playable MIDI instrument - automatically, in real time.

Most samplers require you to manually record, chop, name, categorise, and map
every sample by hand. Subsample automates the entire pipeline: it detects
individual sounds from a live audio stream or pre-recorded files, builds a
55-element acoustic fingerprint for each one, and assigns them to MIDI notes
based on how they sound. The chaotic environment becomes an organised, playable
sample pack - without you lifting a finger.

## What sets Subsample apart

- **Automatic similarity-based MIDI assignment** - sounds are matched to your
  reference library and mapped to MIDI notes as they arrive. No manual
  drag-and-drop, no folder browsing, no per-pad assignment.
- **Classification-free matching** - the same engine works for kicks and violins
  without training data, labels, or pre-defined categories. It is pure geometry:
  similar sounds cluster together naturally in feature space.
- **Real-time capture with zero-gap detection** - the input thread is never
  blocked waiting for analysis. Back-to-back sounds are captured reliably, even
  on USB audio.
- **Pitch-aware** - tonal samples are automatically detected and mapped
  chromatically across a keyboard range, with background pitch-shifting at the
  highest available quality.
- **Config-driven** - everything is YAML. MIDI routing, similarity weights,
  detection tuning, output format. Version-controllable, reproducible, no GUI
  required.
- **Headless** - runs on anything from a Raspberry Pi to a studio rack server.
  Drive it from a DAW, hardware controller, or Python sequencer over MIDI.

## How it works

### 1. Capture

Subsample listens continuously to a live audio input and captures every distinct
sound event. An adaptive noise floor (exponential moving average) tracks the
ambient level in real time, so it works equally well in a quiet studio and a
noisy rehearsal space. Each captured sound is trimmed with smooth S-curve fades
to avoid clicks. Stereo recordings are preserved end-to-end - a stereo
microphone records and plays back in stereo without any manual setting.

You can also feed it pre-recorded WAV files - they pass through the same
detection pipeline, making it easy to build sample libraries from existing
recordings.

### 2. Analyse

Each captured sound is fingerprinted across 55 acoustic dimensions spanning five
groups:

| Group | Dimensions | What it captures |
|-------|-----------|------------------|
| Spectral shape | 11 | Brightness, noisiness, attack/release character |
| Sustained timbre | 12 | Steady-state tonal colour |
| Timbre dynamics | 12 | How the sound evolves over time |
| Attack character | 12 | Transient signature |
| Band energy | 8 | Per-band energy distribution and decay (drum-type signature) |

Tonal sounds are identified by a seven-criterion pitch stability gate - only
samples with a single, confident, stable pitch are flagged for chromatic mapping.
Percussive sounds are handled naturally by the same feature space without special
treatment.

Analysis results are cached as `.analysis.json` sidecar files alongside each
WAV. The cache is versioned and auto-invalidating - when the analysis algorithm
improves, stale sidecars are detected and re-analysed automatically on startup.

### 3. Assign

Sounds are matched to your reference library using cosine similarity on the
55-element feature vector. The best kick-like sound maps to your kick pad; the
best snare maps to your snare. When multiple notes share a reference, they
receive ranked matches: first note gets the best match, second note gets the
second-best, and so on.

As new sounds arrive, assignments update dynamically. Evicted samples are
replaced by the next-best match. The instrument stays playable and fresh without
any manual intervention.

## MIDI map

MIDI routing is defined in a YAML file - by default `midi-map.yaml` in the
project directory, referenced from `config.yaml`:

```yaml
player:
  midi_map: "./midi-map.yaml"
```

Copy `midi-map.yaml.default` as your starting point. The file lists
**assignments** - each mapping one or more MIDI notes on a given channel to
sample targets.

### Assignment fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Label shown in logs |
| `channel` | yes | MIDI channel 1-16 (standard numbering) |
| `notes` | yes | Single note, list, or range (see Note syntax below) |
| `target` | yes | Which sample(s) to play (see Target types below) |
| `one_shot` | no | `true` = play to natural end regardless of note-off (default). `false` = fade out on note-off |
| `gain` | no | Level offset in dB (default 0.0). Negative = quieter, positive = louder |
| `pan` | no | Stereo position as percentage weights e.g. `[50, 50]` = centre (default) |
| `pitch` | no | `true` = pitch-shift the matched sample to each MIDI note in the range. Works with `reference()`, `chain`, `newest()`, and `oldest()` |

### Note syntax

```yaml
notes: 36          # single MIDI note number
notes: C4          # note name (C4 = MIDI 60, same as Ableton/Logic/FL Studio)
notes: [36, 35]    # list - each gets the next similarity rank (first = best match)
notes: C2..C4      # range - expands to every MIDI note from C2 (36) to C4 (60)
notes: 36..60      # range with note numbers
```

Note names use the convention C4 = 60 (C-1 = 0, G9 = 127). Sharps: `C#4`,
`D#3`. Flats: `Db4`, `Eb3`.

### Target types

**`reference(NAME)`** - plays the recorded instrument sample most similar to the
named reference. When multiple notes share a reference, they receive ranked
matches: first note in the list gets rank 0 (best match), second gets rank 1,
and so on. Falls back to rank 0 if fewer samples than notes have been recorded.

```yaml
- name: Kicks
  channel: 10
  notes: [36, 35]          # note 36 → most kick-like; note 35 → second-most
  target: reference(BD0025)
  one_shot: true
```

The reference name must match a file in your `reference.directory`
(case-insensitive).

**`reference(NAME)` with `pitch: true`** - plays the best-matching sample,
pitch-shifted to each MIDI note in the range. Every note in the assignment maps
to rank 0 (same sample), shifted up or down from the sample's detected
fundamental pitch. Pitch variants are computed in the background when the best
match changes - no delay on the first trigger.

```yaml
- name: Bass keyboard
  channel: 1
  notes: C2..C4
  target: reference(BASS_TONE)
  pitch: true
  one_shot: false
```

The reference sample must have a confident, stable detected pitch (checked by
`has_stable_pitch()`). Samples that fail this test fall back to unpitched
playback.

**`sample(FILENAME)`** - plays a specific sample by its filename stem (without
the `.wav` extension). If the sample is not loaded or has been evicted from
memory, the note plays silence - no error. Useful for building fixed kits from
known recordings.

```yaml
- name: Fixed kick
  channel: 10
  notes: 36
  target: sample(2026-03-24_14-37-14)
  one_shot: true
```

**`pitched(SELECTOR)`** - selects a sample by position among all pitch-stable
samples in the instrument library, without requiring a reference library entry.
Every MIDI note in the range plays the selected sample pitch-shifted to match.
Variants are pre-computed in the background when the library changes - no delay
on first trigger.

Selectors:
- `pitched(oldest)` - the first (lowest ID) pitch-stable sample
- `pitched(newest)` - the most recently added pitch-stable sample
- `pitched(N)` - the Nth pitch-stable sample (0-indexed)

```yaml
- name: Latest tonal capture
  channel: 2
  notes: C2..C6
  target: pitched(newest)
  one_shot: false
```

If no pitch-stable samples exist or the index is out of range, the note plays
silence. A sample is pitch-stable when it passes the same `has_stable_pitch()`
gate used by `reference() + pitch: true`.

**`newest()`** - always resolves to the most recently added sample in the
instrument library, whether pitched or percussive. Re-evaluated on every trigger,
so the mapping always reflects the latest capture. Combine with `pitch: true` and
a note range for an instant chromatic keyboard from whatever you just recorded.

```yaml
- name: Latest capture (keyboard)
  channel: 2
  notes: C2..C6
  target: newest()
  one_shot: false
```

**`oldest()`** - complement of `newest()`. Always resolves to the first sample
added to the library in this session. Re-evaluated on every trigger - if the
oldest sample is evicted, the next trigger resolves to the new oldest.

```yaml
- name: First capture (keyboard)
  channel: 3
  notes: C2..C6
  target: oldest()
  one_shot: false
```

**`chain`** - ordered fallback. Tries a list of targets in sequence and uses the
first one that resolves to a sample. Useful for resilient mappings: try a
specific named sample, fall back to similarity matching if it has been evicted.

```yaml
- name: Kick with fallback
  channel: 10
  notes: 36
  target:
    chain:
      - sample(my-favourite-kick)
      - reference(BD0025)
  one_shot: true
```

Each sub-target uses the same syntax as a standalone target. Requires at least 2
sub-targets. Nested chains are not supported. All sub-targets are validated at
load time - one invalid sub-target skips the entire assignment.

### Pan

Pan weights are normalised to constant-power gains so perceived loudness is
equal at any pan position:

```yaml
pan: [50, 50]    # centre (default)
pan: [100, 0]    # hard left
pan: [75, 25]    # left of centre
```

Channel order follows SMPTE: `[L, R]` for stereo; `[L, R, C, LFE, Ls, Rs]` for
5.1. Multichannel output is planned - stereo is the current output format.

## Performance

### Zero-latency playback

When a sample enters the library, a background worker immediately produces a
pre-rendered copy at the output device's sample rate and format. Tonal samples
also receive a full set of pitch-shifted variants. By the time the first MIDI
note fires, the work is already done - playback is a memory copy into the mix
buffer, not an on-the-fly calculation. A three-tier fallback guarantees playback
is never blocked:

1. **Pitch variant** - pre-computed, pitch-corrected (tonal samples)
2. **Base variant** - pre-normalised, no DSP (all samples)
3. **On-the-fly render** - last resort on the very first trigger only

### End-to-end 32-bit float

Every audio sample is converted to float32 immediately after capture and stays in
that format through every stage - analysis, normalisation, pitch shifting, gain
staging, polyphonic mixing. The only integer conversion is a single pack to the
hardware's native bit depth at the output. This is the same internal format used
by professional DAWs, and means that peak-normalising a quiet recording or
pitch-shifting it across two octaves introduces no measurable quality loss.

### Non-blocking capture

The audio input thread does minimal work and returns immediately. Analysis runs
in a separate auto-scaled worker pool, so back-to-back sounds are captured
reliably even when spectral analysis is slow. This is critical for USB audio
devices, which use isochronous transfers and are sensitive to timing jitter.

### Professional gain staging

Every voice is RMS-normalised so a quiet recording and a loud one play at
comparable levels at the same MIDI velocity. A tanh soft-limiter on the mix bus
smoothly compresses peaks that approach 0 dBFS - the output never clips, no
matter how many voices overlap, and the character of the sound is preserved.

### Pitch shifting quality

Pitch variants are produced using the Rubber Band library's offline finer engine,
the highest quality pitch-shifting algorithm available. Variants are pre-computed
in the background by a worker pool; no latency is added at trigger time.

## Similarity engine

Every new sample is scored against every reference using cosine similarity on a
55-element composite feature vector built from five groups: spectral shape (11
dimensions), sustained timbre (12), timbre dynamics (12), attack character (12),
and band energy (8). Each group is independently normalised and scaled by a
configurable weight (`similarity.weight_*`), so you can emphasise whichever
acoustic qualities matter most for your material.

The key insight: **the same comparison method works for both percussive and tonal
sounds without needing to classify them first.** A kick drum naturally scores
high on attack character; a violin scores high on sustained timbre. No
classifier, no training data, no labelling - just geometry.

For each reference, an in-memory ranked list of matches is maintained and updated
incrementally as new recordings arrive or old ones are evicted. See
[Architecture](#architecture) for the full vector breakdown.

## Transforms

Tonal samples with a stable, confident pitch are automatically pitch-shifted to
every MIDI note in the assigned note range (e.g. all 128 notes for a full-keyboard
assignment). Variants are produced in the background by a worker pool and cached
in a memory-bounded store with parent-priority FIFO eviction - when a variant
family would exceed the memory budget, the entire oldest family is evicted
together, keeping remaining families intact and playable.

All variants are produced at the output device's sample rate and format using
high-quality sample rate conversion (soxr algorithm), so the playback path never
pays a conversion cost at trigger time.

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

**Live capture mode:** Subsample lists available audio input devices and lets you
choose one (or auto-selects if only one is present). It calibrates ambient noise
for a few seconds before listening for events.

**File input mode:** Each file is processed at its native sample rate, bit depth,
and channel count. Detected segments are saved to the output directory.

## Configuration

Subsample always loads `config.yaml.default` as the base, then deep-merges
your `config.yaml` on top. Your config only needs the settings you want to
change — everything else is inherited from the defaults automatically.

The most common overrides:

- **First run:** set `recorder.audio.device` (your microphone) and `output.directory`
- **For MIDI playback:** set `player.enabled: true`, `player.midi_device` or `player.virtual_midi_port`, and `player.audio.device`
- **If you hear clipping:** raise `player.max_polyphony`; the `limiter_threshold_db` and `limiter_ceiling_db` defaults protect against distortion automatically
- **If recordings miss quiet sounds or trigger on noise:** tune `detection.snr_threshold_db`

Everything else - chunk sizes, buffer lengths, transform settings, similarity
weights - is optional and rarely needs changing.

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
| `player.midi_map` | `none` | Path to MIDI routing map YAML; if unset, uses built-in GM drum defaults |
| `player.max_polyphony` | `8` | Max simultaneous voices; per-voice gain = 1/max\_polyphony. Raise if clipping; lower for louder individual voices |
| `player.limiter_threshold_db` | `-1.5` | Safety limiter threshold (dBFS); signals below this pass untouched |
| `player.limiter_ceiling_db` | `-0.1` | Maximum output level (dBFS) the limiter allows; must exceed threshold |
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
| `instrument.clean_orphaned_sidecars` | `false` | Auto-delete `.analysis.json` sidecars whose audio file has been deleted |
| `instrument.watch` | `false` | Monitor `instrument.directory` at runtime for new samples arriving from a remote recorder instance (see Multi-machine setup) |
| `reference.directory` | `none` | Optional directory of reference sounds for similarity classification |
| `similarity.weight_spectral` | `1.0` | Weight for the spectral shape group (11 metrics) |
| `similarity.weight_timbre` | `1.0` | Weight for sustained MFCC timbre (coefficients 1-12) |
| `similarity.weight_timbre_delta` | `0.5` | Weight for delta-MFCC timbre trajectory |
| `similarity.weight_timbre_onset` | `1.0` | Weight for onset-weighted MFCC attack character |
| `similarity.weight_band_energy` | `1.0` | Weight for the band energy group (4 per-band energy fractions + 4 decay rates) |
| `transform.max_memory_mb` | `50.0` | Memory budget (MB) for pitch-shifted variants |
| `transform.auto_pitch` | `true` | Pre-compute pitch variants for every MIDI note in the assigned range. Requires `rubberband-cli`. Disable if rubberband is unavailable or you prefer on-the-fly rendering (pitch still works, higher CPU at trigger time) |
| `transform.target_bpm` | `0.0` | Target BPM for automatic time-stretch variants; 0.0 disables |

## Output

Recordings are saved as uncompressed 16, 24, or 32-bit WAV files (depending on
`recorder.audio.bit_depth`) in the configured output directory.

**Live capture mode** - filenames from the datetime the recording ended:

```
samples/
  2026-03-17_14-32-01.wav
  2026-03-17_14-35-44.wav
```

**File input mode** - filenames from the original audio file's stem plus a
segment index:

```
samples/
  field_recording_1.wav
  field_recording_2.wav
```

Both modes write to the same output directory. Point `instrument.directory` at
the same path to get a persistent library that grows on disk across sessions.

## Instrument sample library

Every recording is automatically added to an in-memory instrument library
alongside its full analysis data. A configurable memory cap (default 100 MB)
prevents unbounded growth; the oldest samples are evicted when a new one would
exceed the limit. WAV files on disk are never deleted.

### Persistent library across sessions

```yaml
output:
  directory: "./samples"

instrument:
  directory: "./samples"
```

On startup, Subsample pre-loads all existing WAV files from `./samples`. As new
recordings arrive they are written to disk and added to memory in one step. The
memory cap keeps only the most recent window of captures in RAM; the full archive
on disk is unaffected.

### Multi-machine setup (remote recorder + player)

Subsample can be split across two machines: one captures and analyses audio, the
other plays it back via MIDI. The two machines share a directory (network drive,
Dropbox, or any folder sync tool). The recorder writes samples there; the player
watches the same directory and loads new samples as they arrive — no restart
required.

This separation is useful when the recording and playback environments are
different: a field recorder capturing environmental sound in one location, a
performance machine somewhere else; a backstage capture machine feeding a front-
of-house playback rig; or simply keeping CPU-intensive audio analysis on a
dedicated host.

**Recorder machine** (`config.yaml`):
```yaml
recorder:
  enabled: true

player:
  enabled: false

output:
  directory: "/mnt/shared/samples"
```

**Player machine** (`config.yaml`):
```yaml
recorder:
  enabled: false

player:
  enabled: true

instrument:
  directory: "/mnt/shared/samples"
  watch: true
```

The recorder writes each detected sample as a WAV file plus an `.analysis.json`
sidecar containing the pre-computed feature data. The player monitors the shared
directory for new sidecar files; when one arrives, it loads the sample pair
directly without re-analysing. The sidecar's arrival is used as the ready signal
because the recorder always writes the WAV first — a sidecar appearing means both
files are present and complete.

New samples become available for MIDI playback within a second or two of the
sidecar landing on disk (a short debounce window to accommodate network sync
tools). If the WAV has not yet arrived, the player retries a few times before
logging a warning; that sample will be picked up on the next restart.

## Reference sample library

Reference samples define the canonical sound classes you want to match against -
kick drum, snare, hi-hat, etc. Each reference is represented by its
`.analysis.json` sidecar file only; the original audio is not required.

```yaml
reference:
  directory: "./reference"
```

Place one sidecar per sound class in the reference directory. The name is taken
from the audio filename stem:

```
reference/
  BD0025.wav.analysis.json   →  "BD0025"
  SD5075.wav.analysis.json   →  "SD5075"
  CH.wav.analysis.json       →  "CH"
```

At startup, reference samples are loaded before instrument samples. For every
instrument sample, Subsample computes cosine similarity against every reference
and maintains a ranked list per reference - most similar instrument first. When a
sample is evicted from the instrument library, it is also removed from the ranked
lists.

Query the ranked lists programmatically:

```python
# Most kick-like instrument in memory
sample_id = similarity_matrix.get_match("BD0025", rank=0)

# Second-most kick-like (for a separate kick_2 mapping)
sample_id = similarity_matrix.get_match("BD0025", rank=1)
```

Lookup is case-insensitive.

## Virtual MIDI

Set `player.virtual_midi_port: "Subsample Virtual MIDI"` to create a named
virtual MIDI input port at startup instead of connecting to a hardware device.
This is the primary way to drive Subsample from another application running on
the same machine - for example, a Python sequencer such as
[Subsequence](https://github.com/simonholliday/subsequence) can send a drum
pattern directly to Subsample's virtual port without any physical MIDI hardware.
From the sequencer's side, Subsample's port appears as a MIDI output destination
while Subsample is running. Overrides `player.midi_device`.

> **Performance note:** running a MIDI sequencer and Subsample simultaneously on
> the same machine means two real-time workloads compete for CPU and I/O. This
> works well on a modern multi-core machine but may cause xruns or timing drift
> on lower-powered hardware. If you experience dropouts, reduce
> `recorder.audio.chunk_size`, lower the sequencer's buffer size, or disable the
> recorder (`recorder.enabled: false`) to run Subsample in playback-only mode.

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
- **flatness** - 0 = tonal, 1 = noisy
- **attack** - 0 = instant/percussive, 1 = gradual build-up
- **release** - 0 = sudden stop, 1 = long decay tail
- **centroid** - 0 = bassy, 1 = trebly
- **bandwidth** - 0 = pure tone, 1 = spectrally complex
- **zcr** - zero crossing rate: 0 = smooth, 1 = maximally noisy
- **harmonic** - 0 = percussive, 1 = harmonic/tonal (HPSS)
- **contrast** - 0 = flat spectrum, 1 = strong spectral peaks
- **voiced** - fraction of frames with detected pitch
- **log_attack** - 0 = instant spectral onset, 1 = very slow
- **flux** - 0 = static spectrum, 1 = rapidly evolving

Pitch data (raw values):
- **pitch** - dominant fundamental frequency in Hz, or "none" for unpitched audio
- **chroma** - dominant pitch class (C-B), or "none"
- **pitch_conf** - pyin confidence [0, 1]; use with `voiced` to judge reliability

Amplitude metadata:
- **peak** - peak absolute amplitude [0, 1] with dBFS equivalent
- **rms** - RMS loudness [0, 1] with dBFS equivalent; drives playback gain normalisation

Three MFCC timbre fingerprints are stored in the sidecar (used for similarity,
not shown in script output): `mfcc` (mean, average timbre), `mfcc_delta`
(first-order trajectory), and `mfcc_onset` (onset-weighted, attack emphasis).

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

## Roadmap

### MIDI expressiveness

- **Mute groups** - notes in a named group silence each other when triggered.
  Classic use: closed hi-hat silences open hi-hat.
- **Velocity layers** - different targets at different velocity ranges (pp/mp/ff
  layers) for more expressive performance.
- **Round-robin** - cycle through sample variants on repeated triggers to avoid
  the machine-gun effect on rapid notes.
- **MIDI CC mapping** - mod wheel, volume, sustain pedal, expression - map
  continuous controllers to parameters in real time.
- **Program change** - switch between MIDI map presets on the fly via MIDI
  program change messages.

### Playback and sound design

- **Time-stretching** - BPM-aware variants that match loops and phrases to a
  target tempo. Infrastructure is in place (`transform.target_bpm` in config);
  the DSP implementation is next.
- **Envelope shaping** - per-voice attack/release adjustment. The data model is
  in place; the DSP implementation is next.
- **Effects processing** - filter, reverb, delay - per-voice or per-bus effects
  for shaping the output without an external mixer.
- **Loop playback** - sustain loops for pads, drones, and textures that play
  continuously while a key is held.
- **Multichannel output** - surround sound output beyond stereo, using SMPTE
  channel ordering (5.1, 7.1). Pan weights already support arbitrary channel
  counts; the output path needs to be extended.

### Sample management

- **Auto-slicing** - chop loops and long recordings into individual hits by
  transient detection, then add each slice to the library as a separate sample.
- **Similar-to-this query** - "find more sounds like this one" by exposing the
  similarity engine as a user-facing search.
- **Parallel startup re-analysis** - when the analysis version bumps, stale
  sidecars are currently re-analysed sequentially; large libraries should use the
  existing worker pool for parallel re-analysis.

### Additional targets

- **`pitched(random)`** - random selection from all pitched samples.
- **`all_pitched()`** - distribute a note range across all pitched samples, one
  per note, each pitch-shifted to match.

## Architecture

Subsample is built around three concurrent pipelines that interact through
thread-safe shared state.

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

The input thread is never blocked waiting for analysis. Back-to-back sounds are
captured reliably even when analysis is slow - worker threads handle each
recording concurrently and independently.

### Similarity engine

Every new instrument sample is scored against every reference using cosine
similarity on a 55-element composite vector. The vector is split into five
groups, each independently L2-normalised so that no single group dominates by
scale:

```
Group 1 (x11): spectral shape   [flatness, attack, release, centroid, bandwidth, zcr,
                                  harmonic, contrast, voiced, log_attack, flux]
Group 2 (x12): sustained MFCC   [mean timbre, coefficients 1-12]
Group 3 (x12): delta-MFCC       [timbre trajectory, coefficients 1-12]
Group 4 (x12): onset-weighted   [attack character, coefficients 1-12]
Group 5 (x8):  band energy      [sub-bass/low-mid/high-mid/presence fractions + decay rates]
```

Each group is scaled by a configurable weight (`similarity.weight_*`). This
design means the same comparison method works for both percussive (attack
character dominates) and tonal (sustained timbre dominates) sounds without
needing to classify them first.

### Transform pipeline

```
SampleRecord added to library
    → TransformManager.on_sample_added()
        → enqueue base variant (always)         ← float32 peak-normalised copy
        → enqueue pitch variants (tonal only)   ← Rubber Band offline finer engine
            → TransformProcessor worker pool
                → TransformCache (parent-priority FIFO eviction, 50 MB default)
```

The base variant (identity spec: no DSP) is produced for every sample -
percussive and tonal alike - so the playback path never pays the float32
conversion cost at trigger time. Pitch variants are additional cache entries,
derived from the same PCM source.

When a variant set for a parent sample would exceed the memory budget, the entire
oldest parent's variant family is evicted together, keeping the remaining
families intact and playable.

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

All mixing happens in float32 - the only integer conversion is the final output
packing. Multiple simultaneous voices are summed correctly regardless of the
output device's bit depth.

## Requirements

- Python 3.12+
- PortAudio (required by PyAudio - `apt install portaudio19-dev` or `brew install portaudio`)
- Rubber Band (required by pyrubberband - `apt install rubberband-cli` or `brew install rubberband`)

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
