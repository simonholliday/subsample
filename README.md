# Subsample

**Cross-platform open-source Python live sampler, automatic drum-kit builder,
and MIDI sample instrument.** Point a microphone at the world (or feed in
field recordings, sample packs, or radio captures) and Subsample captures,
trims, analyses, and routes every distinct sound into a playable, mix-ready
MIDI instrument, automatically, in real time. Workflows that normally require
expensive hardware samplers, sample-pack organiser plugins, or hours of manual
chopping and tagging happen continuously in the background.

Build a custom drum kit from your favourite vinyl. Turn a walk through the
woods into a playable instrument. Slice and re-tempo a breakbeat. Feed a pile
of unsorted samples in and watch them organise themselves. All four are the
same workflow.

Traditional samplers - hardware or software - require you to manually record,
chop, name, categorise, and map every sample by hand. Subsample automates the
entire pipeline: it detects individual sounds from a live audio stream or
pre-recorded files, builds a 58-element acoustic fingerprint for each one,
assigns them to MIDI notes based on how they sound, and runs a per-sample DSP
processing chain that adapts its parameters from the audio content itself. A
chaotic environment becomes an organised, mix-ready sample instrument while
you focus on playing.


## Contents

- **1. Getting Started**
  - [Why Subsample?](#why-subsample)
  - [At a glance](#at-a-glance)
  - [Quick start](#quick-start)
- **2. Concepts**
  - [How it works](#how-it-works)
  - [MIDI map](#midi-map)
  - [Similarity engine](#similarity-engine)
  - [Transforms](#transforms)
- **3. Configuration & Operation**
  - [Configuration](#configuration)
  - [Output](#output)
  - [Instrument sample library](#instrument-sample-library)
  - [Reference sample library](#reference-sample-library)
  - [Live-coding the MIDI map](#live-coding-the-midi-map)
- **4. Integration**
  - [Virtual MIDI](#virtual-midi)
  - [OSC integration](#osc-integration)
  - [Works with Subsequence](#works-with-subsequence)
- **5. Project Info**
  - [Performance](#performance)
  - [Scripts](#scripts)
  - [Roadmap](#roadmap)
  - [Architecture](#architecture)
  - [Requirements](#requirements)
  - [Tests](#tests)
  - [Type Checking](#type-checking)
  - [Dependencies and Credits](#dependencies-and-credits)
  - [About the Author](#about-the-author)
  - [License](#license)
  - [Commercial licensing](#commercial-licensing)


## Why Subsample?

- **A studio sampler that builds itself.** Drop samples in (or record them
  live) and Subsample maps them to MIDI notes, processes them through an
  adaptive DSP chain, and presents a playable, mix-ready instrument with no
  manual chopping, naming, or mapping. Free, open-source, and runs anywhere
  CPython 3.12 does - from a Raspberry Pi in the rehearsal room to a studio
  Mac or Linux rack server.
- **Automatic similarity-based sample organisation.** A 58-dimensional
  acoustic fingerprint matches kicks to kick pads, snares to snare pads,
  hi-hats to hi-hat pads - no labels, no training data, no manual tagging.
  The same engine handles tonal samples without special treatment, and works
  equally well on a chaotic Splice library or a fresh field-recording session.
- **Real-time live sampling.** Point a microphone at the world and Subsample
  captures, trims, analyses, and adds every distinct sound event to your
  instrument library as it happens. Adaptive noise floor tracking works in
  noisy rehearsal rooms as well as quiet studios; back-to-back sounds are
  captured reliably with zero-gap detection.
- **Beat slicer and auto-quantize for loops.** Detected onsets in long samples
  are individually placed on a beat grid using onset-aligned timemaps - loops
  snap to your target BPM with musical precision. A pad-quantize mode
  preserves natural timbre by inserting silence between hits instead of
  time-stretching. Per-hit segment playback: cycle through hits with
  `round_robin`, pick randomly with `random`, or map specific segments to
  specific notes by index.
- **Pitched and percussive in one engine.** Tonal samples are auto-detected by
  a seven-criterion stability gate and pitch-shifted across the keyboard range
  at the highest available quality (Rubber Band offline finer mode). Drums,
  melodic, and effect samples share one library and one workflow.
- **17-processor DSP chain with intelligent defaults.** Compression, gating,
  transient shaping, filters, distortion, saturation, vocoder cross-synthesis,
  beat-quantize, pitch-shift, time-stretch, reverse, envelope reshape, and
  HPSS harmonic/percussive separation. Every parameter auto-adapts to each
  sample's analysis data - write `compress: true` and the right threshold,
  attack, and release are derived from the audio. Variants are pre-rendered
  in a background worker pool and ready before you press a key.
- **Sweep anything with a knob.** Bind any numeric parameter - filter cutoff,
  beat-quantize amount, distortion drive, compression threshold - to a MIDI
  CC controller. Variants are re-rendered in the background between knob
  positions and bridged smoothly, so you can play with parameters that aren't
  normally automatable on samplers at all.
- **Multichannel in, multichannel out.** Records from any subset of physical
  inputs on a multi-channel interface (e.g. inputs 3-4 of a Focusrite Scarlett
  18i20). Routes individual instruments to specific outputs (kick to outputs
  1-2, snare to outputs 3-4) for separate external processing. Standard
  ITU-R BS.775 downmix and conservative upmix for stereo, quad, 5.1, and 7.1.
- **Headless and config-driven.** Everything is YAML - version-controllable,
  reproducible, no GUI required. Runs equally well on a studio Mac, a
  Raspberry Pi in the rehearsal room, or a rack server. Drive it from any
  DAW, hardware controller, or sequencer over standard MIDI.
- **Plays nicely with the rest of your studio.** Standard MIDI input from any
  DAW or hardware controller, [virtual MIDI](#virtual-midi) ports for
  software-only routing on the same machine, [OSC integration](#osc-integration)
  for talking to sequencers and visualisers, and a ready-to-play GM drums map
  that turns any sample collection into a coherent, pre-mixed drum kit on
  first play.
- **Pairs with Subsequence.** Subsample is one half of a fully open-source
  generative sampler workstation - its sister project
  [Subsequence](https://github.com/simonholliday/subsequence) is a Python
  MIDI sequencer. Subsequence drives the patterns; Subsample provides the
  sounds. Each works independently - see [Works with Subsequence](#works-with-subsequence).


## At a glance

| | |
|---|---|
| **Live capture** | Adaptive noise floor, zero-gap back-to-back detection, S-curve fades |
| **Analysis** | 58 dimensions across 5 feature groups; cached `.analysis.json` sidecars |
| **Matching** | Cosine similarity, classification-free, ranked fallback, dynamic re-assignment |
| **DSP processors** | 17 (filter, comp, gate, distort, saturate, reshape, transient, HPSS, vocoder, repitch, beat-quantize, pad-quantize, ...) |
| **Adaptive defaults** | Compressor, gate, transient shaper, distortion, envelope reshape - all auto-derive parameters from each sample |
| **Pitch shifting** | Rubber Band offline finer (highest available quality), pre-rendered |
| **Time stretch** | Beat-quantized with onset-aligned timemaps, partial-quantize amount, pad-quantize alternative for speech |
| **Segment playback** | Per-hit round-robin, random, or indexed - for sliced loops |
| **MIDI input** | Hardware port, named virtual port, or both |
| **MIDI control** | Note on/off, Program Change for banks, CC binding for any numeric parameter |
| **OSC** | Sender + receiver (optional dependency) |
| **Audio formats in** | WAV, BWF, FLAC, AIFF, OGG, MP3/MPEG (libsndfile) |
| **Channels** | Mono through 7.1, ITU-R BS.775 downmix, conservative upmix, per-instrument output routing |
| **Audio precision** | End-to-end 32-bit float pipeline, 64-bit DSP for IIR filters and envelope followers |
| **Latency** | Pre-rendered variants - playback is a memory copy into the mix buffer |
| **Library mgmt** | Memory-bounded with FIFO eviction, persistent disk cache for variants, hot-loading from watched directories |
| **Live-coding** | Edit the MIDI map YAML and assignments reload on save |
| **Bank switching** | Multiple instrument directories swappable via MIDI Program Change |
| **GM drums** | Ready-to-play map of 47 GM percussion instruments with researched mix chain |
| **Configuration** | YAML, version-controllable, headless, no GUI |
| **Platform** | Linux, macOS, Windows (via WSL), Raspberry Pi - anywhere CPython 3.12 runs |
| **License** | AGPL-3.0 (commercial licensing on request) |


## How it works

### 1. Capture

Subsample listens continuously to a live audio input and captures every distinct
sound event. An adaptive noise floor (exponential moving average) tracks the
ambient level in real time, so it works equally well in a quiet studio and a
noisy rehearsal space. Each captured sound is trimmed with smooth S-curve fades
to avoid clicks.

All channel formats are preserved end-to-end - a stereo microphone records and
plays back in stereo, a quad recording keeps its four channels, and
multichannel samples are automatically mapped to the output layout using
standard ITU downmix coefficients. On multi-channel interfaces (e.g. Focusrite
Scarlett 18i20), `recorder.audio.input` selects which physical inputs to
record from - for example `[3, 4]` records a stereo pair from inputs 3 and 4.

You can also feed it pre-recorded WAV files - they pass through the same
detection pipeline, making it easy to build sample libraries from existing
recordings. For pre-trimmed sources (commercial sample packs, field recordings,
SDR radio captures), `import_samples.py` bypasses detection entirely and imports
files directly with silence trimming, safety fades, re-encoding, and full
analysis.

### 2. Analyse

Each captured sound is fingerprinted across 58 acoustic dimensions spanning five
groups:

| Group | Dimensions | What it captures |
|-------|-----------|------------------|
| Spectral shape | 14 | Brightness, noisiness, attack/release character |
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
58-element feature vector. The best kick-like sound maps to your kick pad; the
best snare maps to your snare. When multiple notes share a reference, they
receive ranked matches: first note gets the best match, second note gets the
second-best, and so on.

As new sounds arrive, assignments update dynamically. Evicted samples are
replaced by the next-best match. The instrument stays playable and fresh without
any manual intervention.

### 4. Process and mix

Each assigned sample passes through a per-instrument DSP processing chain before
playback. The chain is declared in the MIDI map - a sequence of processors that
can include filtering, compression, limiting, gating, distortion, saturation,
envelope reshaping, transient shaping, time-stretching, pitch-shifting, reversal,
harmonic/percussive separation, and beat quantization. Variants are computed
offline in a background worker pool and cached to disk, so by the time you press
a key the processed audio is already waiting in memory.

Every processor is designed with **intelligent defaults that adapt per sample**.
Filters default to classic console channel-strip values (80 Hz HPF, 16 kHz LPF).
The compressor analyses each sample's peak level, onset speed, and decay
character to set threshold, attack, and release automatically - a percussive kick
gets a slow attack that preserves the beater transient, while a sustained pad
gets a faster attack with longer release to avoid pumping. The gate reads the
noise floor to set its threshold. Transient shaping reads the crest factor to
decide how much punch to add or remove. Envelope reshape reads the decay
character to tighten the tail. Write `compress: true` or `transient: true` and
the right parameters are derived from the audio itself.

Beat-quantized time-stretching locks samples to a target BPM using onset-aligned
timemaps - each onset is individually placed on the beat grid with minimal
stretching between them. For speech and other material where time-stretch
artifacts are unacceptable, pad-quantize snaps onsets to the grid by inserting
silence instead, preserving natural timbre completely.

The included `midi-map-gm-drums.yaml` applies all of this across the full GM
percussion set: 47 instruments, each with researched filtering, compression
(where appropriate), panning, and gain. The result is a coherent, pre-mixed drum
kit from whatever samples you have - no manual tweaking required. Every setting
can be overridden by an experienced user who wants precise control.

## MIDI map

MIDI routing is defined in a YAML file - by default `midi-map.yaml` in the
project directory, referenced from `config.yaml`:

```yaml
player:
  midi_map: midi-map.yaml
```

Two maps are included:

- **`midi-map.yaml.default`** - template with common assignments and full format documentation
- **`midi-map-gm-drums.yaml`** - complete GM percussion kit, ready to play (see below)

Copy either file as your starting point. Each map lists **assignments** -
mapping one or more MIDI notes on a given channel to sample targets.

### The GM drums map - instant professional drum kit

`midi-map-gm-drums.yaml` covers all 47 standard GM percussion notes (35-81).
Point it at your instrument directory and every MIDI drum note automatically
finds the closest matching sample and plays it through a professional mix chain:

- **Similarity matching** - each note finds the best sample via spectral
  fingerprint comparison against GM reference sounds
- **Console-style filtering** - per-instrument HPF/LPF to carve frequency space
  (30 Hz HPF on kicks, 300 Hz on hi-hats, 1 kHz on triangles, etc.)
- **Adaptive compression** on 28 transient instruments - threshold, attack, and
  release auto-adapt to each sample's analysis data.  Foundation sounds get
  tailored settings: kicks at 6:1 with 15 ms attack (beater punch + thick body),
  snares at 5:1 with 8 ms attack (stick crack + ring), hi-hats at gentle 2:1
  (consistency without flattening dynamics).  Cymbals, shakers, and expressive
  instruments are left uncompressed.
- **Audience-perspective panning** - hi-hats left, ride right, toms spread
  across the stereo field, kick and snare near centre
- **Gain balancing** - cymbals and small percussion pulled back so the kit sits
  together without any one instrument dominating

The result: a new user with a collection of recorded samples hears a coherent,
pre-mixed drum kit on first play - no manual configuration needed.

### Assignment fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Label shown in logs |
| `channel` | yes | MIDI channel 1-16 (standard numbering) |
| `notes` | yes | Single note, list, or range (see Note syntax below) |
| `select` | yes | Which sample to play (see Select below) |
| `process` | no | How to present it (see Process below) |
| `one_shot` | no | `true` = play to natural end regardless of note-off (default). `false` = fade out on note-off |
| `gain` | no | Level offset in dB (default 0.0). Negative = quieter, positive = louder |
| `pan` | no | Stereo position as percentage weights e.g. `[50, 50]` = centre (default) |
| `output` | no | Physical output channels (1-indexed) e.g. `[3, 4]` routes to outputs 3-4 |

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

### Select - which sample to play

The `select` block defines how to choose a sample from the instrument library.
It has three parts: filter predicates (`where`), a sort order (`order_by`), and
a pick position (`pick`).

```yaml
select:
  where:
    min_duration: 1.0      # at least 1 second long
    min_onsets: 4           # at least 4 transient hits
  order_by: newest          # most recently captured first
  pick: 1                   # take the first match
```

All `where` predicates must pass (AND logic). Available predicates:

| Predicate | Type | Description |
|-----------|------|-------------|
| `min_duration` / `max_duration` | float (seconds) | Filter by sample length |
| `min_onsets` / `max_onsets` | int | Filter by detected transient count |
| `pitched` | bool | `true` = has stable pitch; `false` = not pitched |
| `min_tempo` / `max_tempo` | float (BPM) | Filter by detected tempo |
| `min_pitch` / `max_pitch` | Hz or note name | Filter by detected frequency |
| `reference` | path | Similarity match against a reference sample (path to WAV) |
| `name` | string or path | Exact filename stem match, or path to a specific WAV |
| `directory` | path | Only match samples whose file path is inside this directory (auto-loads on startup; see [Banks vs directory predicate](#banks-vs-directory-predicate)) |

Available `order_by` values: `newest`, `oldest`, `similarity` (requires
`reference`), `duration_asc`, `duration_desc`, `pitch_asc`, `pitch_desc`,
`onsets_asc`, `onsets_desc`, `tempo_asc`, `tempo_desc`, `loudest`, `quietest`.
Default: `newest`. When `reference` is in `where` and no explicit `order_by` is
given, defaults to `similarity`.

`pick` is 1-indexed. Default: 1 (first match). For multi-note assignments
without explicit `pick`, each note gets the next position (rank distribution) -
so `notes: [36, 35]` gives note 36 pick 1 (best match) and note 35 pick 2.

**Fallback chains** - `select` can be a list of specs tried in order. The first
that returns a result wins:

```yaml
select:
  - where: { name: my-favourite-kick }                               # try specific sample first
  - where: { reference: samples/reference/GM36_BassDrum1.wav }       # fall back to similarity match
```

#### Examples

```yaml
# GM kicks - ranked by similarity to a kick reference
select:
  where:
    reference: samples/reference/GM36_BassDrum1.wav
  order_by: similarity

# Pitched keyboard - oldest tonal sample, repitched per note
select:
  where:
    pitched: true
  order_by: oldest
  pick: 1

# Rhythmic loops - recent, long, with enough beats
select:
  where:
    min_duration: 1.0
    min_onsets: 4
  order_by: newest

# Highest-pitched sample in the library
select:
  where:
    pitched: true
  order_by: pitch_desc
  pick: 1
```

### Process - how to present the sample

The optional `process` block declares an ordered list of audio processors
applied after sample selection. Omit it entirely for unprocessed playback.

```yaml
process:
  - filter_low: { freq: 800, resonance: 6 }   # low-pass, then
  - repitch: true                               # pitch-shift, then
  - saturate: { amount: 4 }                     # saturation
```

Processors execute in the order you declare them - different orderings
produce different results. The full chain is pre-computed and cached.

Available processors:

| Processor | Parameters | Description |
|-----------|-----------|-------------|
| `repitch: true` | none | Pitch-shift to match the triggering MIDI note |
| `repitch: { note: C4 }` | target note | Pitch-shift to a fixed note |
| `beat_quantize: { grid: 16 }` | grid subdivision | Time-stretch to session `target_bpm` |
| `beat_quantize: { bpm: 120, grid: 8 }` | explicit BPM + grid | Time-stretch to a specific BPM |
| `beat_quantize: { amount: 0.5 }` | 0.0-1.0 (default 1.0) | Partial quantize - onsets move partway to the grid for a looser feel |
| `pad_quantize: { grid: 16 }` | bpm (config), grid (16) | Onset-aligned silence padding - snaps onsets to the beat grid by inserting silence between segments rather than time-stretching. No pitch/speed change. Ideal for speech. |
| `pad_quantize: { amount: 0.75 }` | 0.0-1.0 (default 1.0) | Partial quantize - same as beat_quantize amount but for silence-pad mode |
| `filter_low: true` | freq (Hz, default 16000), resonance (dB, default 0) | Low-pass filter (console-style default) |
| `filter_high: true` | freq (Hz, default 80), resonance (dB, default 0) | High-pass filter (console-style default) |
| `filter_band: true` | freq (Hz, default 1000), q (default 0.7), resonance (dB, default 0) | Band-pass filter (Q sets width) |
| `reverse: true` | none | Reverse the audio |
| `saturate: { amount: 6 }` | amount (dB of drive) | Soft-clip saturation with level compensation |
| `compress: true` | threshold (auto), ratio (4:1), attack (auto), release (auto), knee (6 dB), makeup (0 dB), lookahead (0 ms) | Dynamic range compressor (adapts to each sample) |
| `limit: true` | threshold (-1 dB), release (50 ms), lookahead (5 ms) | Brickwall limiter (ratio 100:1, instant attack) |
| `hpss_harmonic: true` | none | Keep only harmonic/tonal content (remove percussion) |
| `hpss_percussive: true` | none | Keep only percussive/transient content (remove harmonics) |
| `gate: true` | threshold (auto), attack (auto), release (auto), hold (auto), lookahead (auto) | Noise gate - silences audio below the noise floor. All parameters auto-adapt: threshold from noise floor, attack/release/hold from onset and decay character. |
| `distort: true` | mode (hard_clip), drive (auto), mix (1.0), tone (auto), bit_depth (8), downsample_factor (4) | Waveshaping distortion with four modes: hard_clip, fold, bit_crush, downsample. Drive adapts to crest factor; tone adapts to spectral rolloff. |
| `reshape: true` | attack (preserve), hold (0), decay (preserve), sustain (1.0), release (auto) | ADSR envelope reshaping. Default auto-tightens the tail. Set attack, decay, sustain, release to reshape specific phases. |
| `transient: true` | amount (auto) | Transient enhancement/taming via HPSS rebalancing. Auto-adapts from crest factor: peaky samples are tamed, dull samples enhanced. |
| `vocoder: { carrier: reference }` | carrier (required), bands (24), depth (1.0), formant_shift (0) | Channel vocoder cross-synthesis. Imposes the sample's spectral envelope onto a carrier signal. `carrier: reference` uses this note's reference sample; or specify a file path. |

All three filters can be used without parameters - they default to classic
console channel-strip values:

```yaml
process:
  - filter_high: true    # 80 Hz high-pass  (rumble filter)
  - filter_low: true     # 16 kHz low-pass  (analog warmth roll-off)
  - filter_band: true    # 1 kHz band-pass, Q 0.7 (wide mid sweep)
```

Override any parameter to taste. All filters are 2nd-order (12 dB/octave),
flat Butterworth by default. Add resonance for a peak at the cutoff
(Chebyshev Type I, max 24 dB). Band-pass Q controls width: lower = wider
(0.7 = gentle sweep), higher = narrower (4.0 = surgical).

The compressor and limiter share the same DSP back-end (Giannoulis et al.
feed-forward design with soft knee and look-ahead). `compress: true` adapts to
each sample automatically using the analysis data:

- **threshold** - set 6 dB below the sample's peak level (always engages)
- **attack** - slow for percussive samples (lets the transient punch through),
  fast for gradual onsets (no transient to protect)
- **release** - short for quick-decay samples (recovers before the next hit),
  long for sustained sounds (avoids pumping)

```yaml
process:
  - compress: true                                        # adapts to each sample
  - compress: { threshold: -30, ratio: 10, attack: 0.5 } # explicit - squash + raise tail
  - compress: { attack: 5 }                               # explicit attack, rest auto
  - limit: true                                           # brickwall at -1 dBFS
```

Set any parameter explicitly to override its auto value. Fixed parameters
(ratio, knee, makeup, lookahead) always use their defaults unless set.

The noise gate, distortion, and envelope reshaper follow the same pattern -
`true` gives you intelligent auto defaults, explicit parameters override:

```yaml
process:
  - gate: true                              # auto noise gate
  - gate: { threshold: -40, hold: 20 }      # explicit threshold
  - distort: true                            # hard-clip with auto drive
  - distort: { mode: fold, drive: 12 }       # foldback distortion
  - distort: { mode: bit_crush, bit_depth: 4, mix: 0.5 }
  - reshape: true                            # auto tail-tightening
  - reshape: { attack: 5, release: 100 }     # fast attack, controlled release
  - reshape: { sustain: 0.5, release: 50 }   # half sustain, tight tail
  - transient: true                          # auto: normalises punch from crest factor
  - transient: { amount: 6 }                # enhance transients by 6 dB
  - transient: { amount: -3 }               # tame transients by 3 dB
  - pad_quantize: { bpm: 120, grid: 8 }     # silence-pad onsets to eighth-note grid
  - vocoder: { carrier: reference }           # cross-synthesise with this note's reference
  - vocoder: { carrier: samples/reference/GM36_BassDrum1.wav, bands: 16, depth: 0.8 }
```

For the opposite of snappy drums (bring up room ambience and reverb tails), use
a fast attack (< 1 ms), high ratio (10:1+), and low threshold (-30 dB) to
squash transients and raise the relative level of the sustain/decay.

HPSS (Harmonic/Percussive Source Separation) decomposes audio into sustained
tonal content and transient clicks/hits. Useful as a pre-filter before repitch
(avoids pitch-shifting drum bleed) or beat_quantize (cleaner grid alignment).

When `repitch` is in the process list, all notes in a multi-note assignment
share pick 1 (same sample, pitched per note). Without `repitch`, each note gets
the next rank.

### Pan

Pan weights are normalised to constant-power gains so perceived loudness is
equal at any pan position:

```yaml
pan: [50, 50]    # centre (default)
pan: [100, 0]    # hard left
pan: [75, 25]    # left of centre
```

Channel order follows SMPTE: `[L, R]` for stereo; `[L, R, C, LFE, Ls, Rs]` for
5.1; `[L, R, C, LFE, BL, BR, SL, SR]` for 7.1. Set `player.audio.channels` in
config to match your output device (default: stereo). Samples of any channel
count are automatically mapped to the output layout using ITU-R BS.775 downmix
coefficients (surround to stereo, etc.) or conservative upmix (stereo to 5.1
uses front pair only). Pan weights define a target layout - if the output has
fewer channels, standard downmix is applied automatically.

#### Output routing

On a multi-channel interface you can route each instrument to specific physical
outputs. Numbers are 1-indexed, matching the labels on your hardware:

```yaml
kick:
  pan: [50, 50]
  output: [1, 2]       # main monitors (default when omitted)

snare:
  pan: [50, 50]
  output: [3, 4]       # separate outputs for external processing

pad:
  output: [5, 6]       # stereo sample sent to outputs 5-6
```

Set `player.audio.channels` in config to match your device (e.g. 8 for a
Focusrite Scarlett 18i20). When `output` is omitted, instruments route to the
first N outputs as before - stereo users see no change.

### Banks - switching instrument sets via MIDI

The MIDI map can optionally declare multiple instrument directories ("banks")
that are all loaded at startup. Switch between them at runtime using MIDI
Program Change messages - no restart, no disk I/O, instant switching:

```yaml
banks:
  - name: "Acoustic Kit"
    directory: samples/acoustic
    program: 0
  - name: "Electronic Kit"
    directory: samples/electronic
    program: 1

bank_channel: 10    # MIDI channel for PC messages (1-16, or 0 = any)
default_bank: 0     # program number to activate at startup (default: first in list)
```

When `banks:` is absent, the single `instrument.directory` from config.yaml is
used as before. When present, it overrides `instrument.directory`. Each bank
gets its own sample library, similarity index, and transform cache.

Assignments are bank-agnostic - they query whichever bank is active. Named
samples (`where: { name: X }`) that only exist in one bank silently produce no
match in other banks; rule-based selects (`reference:`, `pitched:`, etc.) work
naturally against whatever samples are present.

#### Banks vs directory predicate

Banks and `where: { directory: ... }` both load samples from a directory, but
they solve different problems:

- **Banks** swap the entire sample pool at once. Only one bank is active at a
  time - a MIDI Program Change switches all assignments to a new set of samples.
  Use banks when you want the same MIDI map rules to evaluate against completely
  different sample collections (e.g. "Acoustic Kit" vs "Electronic Kit").

- **`where: { directory: ... }`** filters within the active pool. It is
  per-assignment, and multiple assignments can each reference a different
  directory simultaneously. Use it when different notes in the same map need
  samples from different directories at the same time (e.g. kicks from one
  folder, hi-hats from another).

| | Banks | `where: { directory }` |
|---|---|---|
| Scope | All assignments share one active bank | Per-assignment filter |
| Switching | MIDI Program Change swaps the whole pool | Always active |
| Simultaneous directories | No (one bank at a time) | Yes (each assignment can use a different directory) |
| Use case | Swap entire kits | Mix sources within one kit |

### CC mapping - real-time parameter control

Any numeric processor parameter can be controlled by a MIDI CC message.
Replace the scalar value with a CC binding:

```yaml
process:
  - pad_quantize: { grid: 16, amount: { cc: 1 } }
  - beat_quantize: { bpm: { cc: 2, min: 60, max: 180 }, grid: 16 }
  - filter_low: { freq: { cc: 74, min: 200, max: 16000 } }
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `cc` | yes | | CC number (0-127) |
| `min` | no | `0.0` | Output value when CC = 0 |
| `max` | no | `1.0` | Output value when CC = 127 |
| `default` | no | midpoint | Value before any CC is received |
| `channel` | no | any | MIDI channel (1-16); omit for omni |

When a mapped CC changes, new variants are enqueued after a 200 ms debounce.
Until the new variant is ready, the previous processed variant continues to
play - giving smooth transitions for gradual changes.

**Important:** use stepped/discrete controllers (knobs, faders, buttons) for CC
mapping. Do not use pitch bend, aftertouch, or high-resolution continuous
controllers - these generate hundreds of messages per second and would flood the
transform queue. Each distinct CC value produces a new variant; the transform
cache evicts the oldest when its memory budget is exceeded.

## Performance

### Zero-latency playback

When a sample enters the library, a background worker immediately produces a
pre-rendered copy at the output device's sample rate and format. Tonal samples
also receive a full set of pitch-shifted variants. By the time the first MIDI
note fires, the work is already done - playback is a memory copy into the mix
buffer, not an on-the-fly calculation. A three-tier fallback guarantees playback
is never blocked:

1. **Process variant** - pre-computed with the full declared chain (pitch, filter, saturate, reverse, time-stretch, etc.)
2. **Base variant** - pre-normalised, no DSP (all samples)
3. **On-the-fly render** - last resort on the very first trigger only

### End-to-end 32-bit float

Every audio sample is converted to float32 immediately after capture and stays in
that format between pipeline stages - analysis, normalisation, pitch shifting,
gain staging, polyphonic mixing. Precision-sensitive operations (IIR filters,
compressor/gate envelope followers, gain curve generation) promote to float64
internally and return float32. The only integer conversion is a single pack to
the hardware's native bit depth at the output. This approach matches professional
DAW practice, and means that peak-normalising a quiet recording or pitch-shifting
it across two octaves introduces no measurable quality loss.

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
58-element composite feature vector built from five groups: spectral shape (14
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

Variants are also persisted to a disk cache (`samples/variant-cache/` by default) so
they survive restarts. Each variant is stored as a single binary file named by a
SHA-256 hash of the source audio, transform chain, output sample rate, and
analysis version - any change to any of these produces a different key, so stale
cache hits are impossible. Recently-used files are kept warm (LRU by modification
time); oldest files are evicted when the disk budget is exceeded.

Samples with detected rhythmic content can be time-stretched to a target tempo
using the `beat_quantize` processor in a MIDI map assignment. Detected attacks are
snapped to a quantized beat grid and the entire mapping is applied in a single
pass using Rubber Band's offline finer engine. Time-stretch variants are produced
on-demand when an assignment requests them - no global startup cost.

### Attack-accurate onset detection

Standard spectral onset detection (as used by librosa and most audio analysis
tools) identifies the frame where spectral energy changes most rapidly - the
peak of the onset strength envelope. For percussive sounds this peak typically
lags the actual attack by 10-30 ms, which is enough to make beat-quantized
hits sound noticeably off the grid.

Subsample refines each detected onset to sample-accurate precision using a
two-stage approach:

1. **Coarse detection** - librosa's onset detector finds approximate positions
   at frame resolution (~11.6 ms at 44100 Hz / hop 512).
2. **Attack refinement** - for each onset, a short-window amplitude envelope
   (32 samples, ~0.7 ms) is searched backward to find the inter-hit valley
   (quietest point between consecutive transients), then forward to find where
   energy first rises above 20% of the local peak. This threshold crossing is
   the perceptual attack start - the moment a musician would tap along.

The search is bounded by the midpoint to the previous onset (preventing bleed
into the prior hit's tail) and a maximum of 50 ms (the physical upper bound on
STFT detection lag). The result is stored as `attack_times` in the analysis
sidecar alongside the original `onset_times`, giving the time-stretch handler
precise alignment points without sacrificing the coarse onsets that other
subsystems rely on.

All DSP runs at the sample's native rate so filters and nonlinear processors
(distortion, saturation) operate at full resolution. The final downsample to
the output device rate uses very-high-quality conversion (soxr_vhq) whose
anti-alias filter catches any above-Nyquist content generated by the
processing chain. The playback path never pays a conversion cost at trigger
time.

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
change - everything else is inherited from the defaults automatically.

The most common overrides:

- **First run:** set `recorder.audio.device` (your microphone) and `output.directory`
- **For MIDI playback:** set `player.enabled: true`, `player.midi_device` or `player.virtual_midi_port`, and `player.audio.device`
- **If you hear clipping:** raise `player.max_polyphony`; the `limiter_threshold_db` and `limiter_ceiling_db` defaults protect against distortion automatically
- **If recordings miss quiet sounds or trigger on noise:** tune `detection.snr_threshold_db`

Everything else - chunk sizes, buffer lengths, transform settings, similarity
weights - is optional and rarely needs changing.

| Setting | Default | Description |
|---|---|---|
| `max_memory_mb` | auto | Total cache memory budget. Auto-detect: min(25% of system RAM, 1024 MB). Split: 60% instruments, 35% transforms, 5% carrier |
| `recorder.enabled` | `true` | Enable live audio capture; set to `false` to process files only |
| `recorder.audio.device` | `none` | Audio input device name (substring match); if unset, auto-select or prompt |
| `recorder.audio.sample_rate` | `44100` | Sample rate in Hz |
| `recorder.audio.bit_depth` | `16` | Bit depth (16, 24, or 32) |
| `recorder.audio.channels` | auto | 1 = mono, 2 = stereo. Omit (or `null`) to auto-detect from device |
| `recorder.audio.input` | `null` | Physical input channels (1-indexed list). `[3, 4]` records from inputs 3-4 |
| `recorder.audio.chunk_size` | `512` | Frames per buffer read |
| `recorder.buffer.max_seconds` | `60` | Circular buffer length |
| `player.enabled` | `false` | Enable the MIDI player |
| `player.midi_map` | `none` | Path to MIDI routing map YAML; required for player. Use `midi-map-gm-drums.yaml` for a complete GM kit |
| `player.max_polyphony` | `8` | Max simultaneous voices; per-voice gain = 1/max\_polyphony. Raise if clipping; lower for louder individual voices |
| `player.limiter_threshold_db` | `-1.5` | Safety limiter threshold (dBFS); signals below this pass untouched |
| `player.limiter_ceiling_db` | `-0.1` | Maximum output level (dBFS) the limiter allows; must exceed threshold |
| `player.midi_device` | `none` | MIDI input device name (substring match); if unset, auto-select or prompt |
| `player.audio.device` | `none` | Audio output device name for playback |
| `player.audio.sample_rate` | auto | Output sample rate; defaults to recorder rate. Do not set higher than source. |
| `player.audio.bit_depth` | auto | Output bit depth (16, 24, or 32); defaults to recorder bit depth |
| `player.audio.channels` | `null` | Output channels (2=stereo, 6=5.1, 8=7.1); null defaults to stereo. SMPTE ordering |
| `player.virtual_midi_port` | `none` | Name for a virtual MIDI input port; overrides `player.midi_device` |
| `player.watch_midi_map` | `false` | Monitor the `midi_map` file for changes and reload assignments on save (see Live-coding) |
| `detection.snr_threshold_db` | `12.0` | dB above ambient to trigger recording |
| `detection.hold_time` | `0.5` | Seconds to hold recording open after signal drops |
| `detection.warmup_seconds` | `1.0` | Calibration period before detection activates |
| `detection.ema_alpha` | `0.1` | Ambient noise adaptation speed (lower = slower) |
| `detection.trim_pre_samples` | `10` | Samples to keep before signal onset (S-curve fade applied) |
| `detection.trim_post_samples` | `90` | Samples to keep after signal end (S-curve fade applied) |
| `output.directory` | `./samples/captures` | Where WAV files are saved |
| `output.filename_format` | `%Y-%m-%d_%H-%M-%S-%3f` | strftime format for filenames (`%3f` = 3-digit milliseconds) |
| `analysis.start_bpm` | `120.0` | Tempo prior for beat detection (BPM) |
| `analysis.tempo_min` | `30.0` | Minimum tempo considered by pulse detector (BPM) |
| `analysis.tempo_max` | `300.0` | Maximum tempo considered by pulse detector (BPM) |
| `instrument.max_memory_mb` | auto | Max audio memory for in-memory samples; overrides global split. Oldest evicted (FIFO) |
| `instrument.directory` | `samples/captures` | Directory of instrument samples to load at startup (overridden by `banks:` in the MIDI map when present) |
| `instrument.clean_orphaned_sidecars` | `true` | Auto-delete `.analysis.json` sidecars whose audio file has been deleted |
| `instrument.watch` | `false` | Monitor `instrument.directory` (or each bank directory) at runtime for new audio files from any source - another Subsample instance, a DAW, or any application that writes audio (see Watching for new samples) |
| `similarity.weight_spectral` | `1.0` | Weight for the spectral shape group (14 metrics) |
| `similarity.weight_timbre` | `1.0` | Weight for sustained MFCC timbre (coefficients 1-12) |
| `similarity.weight_timbre_delta` | `0.5` | Weight for delta-MFCC timbre trajectory |
| `similarity.weight_timbre_onset` | `1.0` | Weight for onset-weighted MFCC attack character |
| `similarity.weight_band_energy` | `1.0` | Weight for the band energy group (4 per-band energy fractions + 4 decay rates) |
| `transform.max_memory_mb` | auto | Memory budget (MB) for transform variants; overrides global split |
| `transform.auto_pitch` | `true` | Pre-compute pitch variants for every MIDI note in the assigned range. Requires `rubberband-cli`. Disable if rubberband is unavailable or you prefer on-the-fly rendering (pitch still works, higher CPU at trigger time) |
| `transform.target_bpm` | `0.0` | Target BPM for automatic time-stretch variants; 0.0 disables. When > 0, qualifying samples (detected tempo + enough onsets) are beat-quantized to the target tempo |
| `transform.quantize_resolution` | `16` | Grid subdivision for time-stretch onset alignment: 1 (whole), 2 (half), 4 (quarter), 8 (eighth), 16 (sixteenth) |
| `transform.variant_cache_dir` | `samples/variant-cache` | Directory for persistent disk cache of transform variants. Empty string or null disables |
| `transform.max_disk_mb` | auto | Max disk space (MB) for cached variant files; defaults to 3x memory budget. 0 disables |
| `osc.enabled` | `false` | Enable OSC integration (send sample events, optionally receive import requests). Requires `pip install subsample[osc]` |
| `osc.send_host` | `127.0.0.1` | Destination host for outgoing `/sample/captured` and `/sample/loaded` messages |
| `osc.send_port` | `9000` | Destination UDP port for outgoing OSC messages |
| `osc.receive_enabled` | `false` | Listen for `/sample/import` messages to trigger file import from other apps |
| `osc.receive_port` | `9002` | UDP port the OSC receiver listens on |

## Output

Recordings are saved as uncompressed 16, 24, or 32-bit WAV files (depending on
`recorder.audio.bit_depth`) in the configured output directory.

**Live capture mode** - filenames from the datetime the recording ended:

```
samples/
  2026-03-17_14-32-01-472.wav
  2026-03-17_14-35-44-091.wav
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
alongside its full analysis data. A configurable memory cap prevents unbounded
growth; the oldest samples are evicted when a new one would exceed the limit.
The budget is auto-detected by default (60% of the global memory allocation -
see `max_memory_mb` in the configuration table) and can be overridden via
`instrument.max_memory_mb`. WAV files on disk are never deleted.

### Persistent library across sessions

```yaml
output:
  directory: samples/captures

instrument:
  directory: samples/captures
```

On startup, Subsample pre-loads all existing WAV files from `./samples/captures`.
As new recordings arrive they are written to disk and added to memory in one step.
The memory cap keeps only the most recent window of captures in RAM; the full
archive on disk is unaffected.

### Watching for new samples

Set `instrument.watch: true` to monitor the instrument directory for new audio
files at runtime and load them without restarting. The watcher detects audio
files from any source - another Subsample instance, a DAW, an SDR recorder, a
script, or any other application that writes audio to the watched directory.

Two detection paths run in parallel:

1. **Sidecar path** - watches for `.analysis.json` sidecar files (fastest).
   When a sidecar appears, its corresponding audio file is loaded immediately
   without re-analysing. This is the path taken when the source is another
   Subsample instance, which always writes the WAV first and the sidecar second.

2. **Audio file path** - watches for audio files (`.wav`, `.flac`, `.aiff`,
   `.ogg`, `.mp3`) from any source. After a short grace period to see if a
   sidecar follows (in case the source is Subsample), checks that the file is
   no longer being written (file-size stability), runs the full analysis
   pipeline, writes a sidecar, and loads the sample.

The audio file path handles the common case where another application writes
an audio file without any sidecar. The file-size stability check ensures that
long recordings still being written are not loaded prematurely - the watcher
waits until the file size stops changing before attempting to read it.

Supported audio formats: WAV, FLAC, AIFF, OGG, MP3/MPEG (anything supported
by libsndfile).

### Multi-machine setup (remote recorder + player)

Subsample can be split across two machines: one captures and analyses audio, the
other plays it back via MIDI. The two machines share a directory (network drive,
Dropbox, or any folder sync tool). The recorder writes samples there; the player
watches the same directory and loads new samples as they arrive - no restart
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
because the recorder always writes the WAV first - a sidecar appearing means both
files are present and complete.

Audio files from non-Subsample sources (no sidecar) are also detected
automatically. After a brief grace period, the player analyses the file, writes
a sidecar for next time, and loads the sample into memory.

New samples become available for MIDI playback within a second or two of the
sidecar landing on disk (a short debounce window to accommodate network sync
tools), or within about 10 seconds for audio files without sidecars (debounce +
grace period + analysis). If the WAV has not yet arrived or is still being
written, the player retries automatically.

## Live-coding the MIDI map

You can edit the MIDI routing map while the player is running and have changes
take effect immediately - no restart required. Set `player.watch_midi_map: true`
and point `player.midi_map` at your working copy:

```yaml
player:
  enabled: true
  midi_map: midi-map.yaml
  watch_midi_map: true
```

When you save the file, Subsample re-parses it and swaps the active note map
within about half a second. If the YAML has a syntax error, the current map is
kept and a warning is logged - playback is never interrupted. Rapid saves from
text editors are debounced into a single reload.

## Reference sample library

Reference samples define the canonical sound classes you want to match against -
kick drum, snare, hi-hat, etc. Each reference is represented by its
`.analysis.json` sidecar file alongside the original audio. References are
declared as path-based `where: { reference: ... }` predicates in the MIDI map:

```yaml
- name: Bass Drum
  notes: 36
  select:
    where:
      reference: samples/reference/GM36_BassDrum1.wav
```

During player startup, each path-based reference is loaded from its sidecar and
added to the similarity matrix. If a WAV file exists but its `.analysis.json`
sidecar is missing, Subsample generates it automatically - you can point at any
WAV file as a reference without pre-processing. For every instrument sample,
Subsample computes cosine similarity against every reference and maintains a
ranked list per reference - most similar instrument first. When a sample is
evicted from the instrument
library, it is also removed from the ranked lists.

Query the ranked lists programmatically:

```python
# Most kick-like instrument in memory
sample_id = similarity_matrix.get_match("GM36_BassDrum1", rank=0)

# Second-most kick-like (for a separate kick_2 mapping)
sample_id = similarity_matrix.get_match("GM36_BassDrum1", rank=1)
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

## OSC integration

Subsample can send and receive [OSC (Open Sound Control)](https://opensoundcontrol.stanford.edu/)
messages, so it can talk to sequencers, visualisers, custom scripts, or any
other OSC-compatible application. OSC support is an optional extra: install it
with

```bash
pip install -e ".[osc]"
```

then enable it in `config.yaml`:

```yaml
osc:
  enabled: true
  send_host: "127.0.0.1"
  send_port: 9000
  receive_enabled: true
  receive_port: 9002
```

### Outgoing messages

When `osc.enabled` is true, Subsample sends two events:

| Address | When | Arguments |
|---|---|---|
| `/sample/captured` | A new live recording has been analysed | `filepath:str, duration:float, pitch_hz:float, pitch_class:int, tempo_bpm:float, onset_count:int` |
| `/sample/loaded` | A sample has been added to the instrument library (live capture, hot-load, or OSC import) | `name:str, duration:float, pitch_hz:float, pitch_class:int` |

`pitch_class` is `0..11` for tonal samples (C=0, C#=1, ..., B=11) or `-1`
when no stable pitch is detected. `pitch_hz` is `0.0` when unpitched.

### Incoming messages

When `osc.receive_enabled` is also true, Subsample listens on
`osc.receive_port` for one address:

| Address | Effect | Arguments |
|---|---|---|
| `/sample/import` | Read the file at the given path, run the full analysis pipeline, and add it to the instrument library | `file_path:str` |

This is more targeted than the directory watcher and lets external applications
push specific files into the library on demand - for example, a radio scanner
or bird detector that wants its captures to become MIDI-playable instruments.

### Use cases

- **Drive a sequencer from incoming sounds.** A Subsequence pattern can react
  when Subsample captures a snare-like sound, triggering a fill or changing
  density.
- **Visualise the library in real time.** A TouchDesigner or Processing patch
  subscribed to `/sample/loaded` can show new samples as they arrive, mapped
  by pitch, tempo, or duration.
- **Cross-app sample handoff.** Any other tool that produces audio files can
  push them into Subsample with a single `/sample/import` message - no shared
  filesystem watching required.

## Works with Subsequence

[Subsequence](https://github.com/simonholliday/subsequence) is a sister
project: a generative MIDI sequencer and algorithmic composition engine for
Python with rock-solid timing (typical pulse jitter < 5 μs on Linux).
Together, the two form a fully open-source generative sampler workstation -
Subsequence drives the patterns, Subsample provides the sounds.

The two communicate over standard MIDI. The simplest setup is to give
Subsample a named [virtual MIDI port](#virtual-midi) and have Subsequence send
to it - no hardware MIDI cabling required, no audio routing on the host:

```yaml
# config.yaml
player:
  enabled: true
  virtual_midi_port: "Subsample Virtual MIDI"
```

From the sequencer side, Subsample's port appears as a MIDI output
destination while Subsample is running - Subsequence connects with
`composition.midi_output("Subsample Virtual MIDI")`, no special configuration
needed.

For richer integration, enable [OSC](#osc-integration) on both sides.
Subsample will forward `/sample/captured` and `/sample/loaded` events to a
Subsequence OSC listener, so a pattern can respond musically to incoming
samples - trigger a fill when a snare-like sound arrives, raise pattern
density when a busy loop is captured, or update a visualiser. Subsequence can
also send `/sample/import` messages back to Subsample to push specific files
into its library.

Each project is independently useful and has no dependency on the other.

## Scripts

### Analyzing recorded files

```bash
python scripts/analyze_file.py samples/2026-03-17_14-32-01.wav
```

Output:
```
rhythm:   tempo=120.2bpm  beats=4  pulses=12  onsets=4
spectral: duration=2.00s  flatness=0.001  attack=0.000  release=0.812  centroid=0.018  bandwidth=0.001  zcr=0.120  harmonic=0.821  contrast=0.310  voiced=0.940  log_attack=0.000  flux=0.312  rolloff=0.451  slope=0.023
pitch:    pitch=440.0Hz  chroma=A  pitch_conf=0.89  stability=0.120st  voiced_frames=86
level:    peak -1.2dBFS  rms -12.6dBFS  crest 11.4dB  floor -42.3dBFS
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
- **rolloff** - 0 = energy concentrated low, 1 = energy extends to Nyquist
- **slope** - 0 = flat spectrum, 1 = steeply tilted

Pitch data (raw values):
- **pitch** - dominant fundamental frequency in Hz, or "none" for unpitched audio
- **chroma** - dominant pitch class (C-B), or "none"
- **pitch_conf** - pyin confidence [0, 1]; use with `voiced` to judge reliability
- **stability** - pitch stability in semitones; lower = more stable
- **voiced_frames** - number of frames with detected pitch

Amplitude metadata:
- **peak** - peak level in dBFS
- **rms** - RMS loudness in dBFS; drives playback gain normalisation
- **crest** - crest factor (peak-to-RMS ratio) in dB
- **floor** - noise floor in dBFS (shown when detectable)

Three MFCC timbre fingerprints are stored in the sidecar (used for similarity,
not shown in script output): `mfcc` (mean, average timbre), `mfcc_delta`
(first-order trajectory), and `mfcc_onset` (onset-weighted, attack emphasis).

### Importing pre-trimmed samples

Import audio files from any source (SDR captures, commercial sample packs, field
recordings) directly into the capture library, bypassing the detection pipeline.
Files are silence-trimmed, safety-faded, re-encoded as standard PCM WAV, fully
analyzed, and saved with sidecar JSON.

```bash
python scripts/import_samples.py /path/to/samples/*.wav
python scripts/import_samples.py --to samples/captures /path/to/sample-pack/*.wav
python scripts/import_samples.py --force "/mnt/sdr/audio/2026-01-15/*.wav"
```

- `--to DIR` - target directory (default: `output.directory` from config.yaml)
- `--force` - overwrite existing files in target directory

Handles WAV, BWF (Broadcast Wave Format), FLAC, AIFF, OGG, and any other format
supported by libsndfile. BWF and non-WAV sources are re-encoded as standard PCM WAV
so the rest of the pipeline can load them reliably.

### Similarity report

```bash
python scripts/similarity_report.py           # top 5 per reference (default)
python scripts/similarity_report.py --top 10  # top 10 per reference
```

Example output:
```
Reference: GM36_BassDrum1
  1.  #5     0.9412  GM36_BassDrum1  ./samples/kick_deep.wav
  2.  #7     0.8134  kick_hard       ./samples/kick_hard.wav
  3.  #8     0.7601  kick_soft       ./samples/kick_soft.wav
```

### Extracting GM percussion references

Render all 47 General MIDI percussion instruments from a SoundFont file into the
reference sample directory. Requires `fluidsynth` CLI tool and a GM SoundFont.

```bash
python scripts/extract_gm_drums.py /path/to/FluidR3_GM.sf2
python scripts/extract_gm_drums.py /path/to/FluidR3_GM.sf2 --output samples/reference/
```

Produces WAV files + analysis sidecars. Only the sidecars are committed to the
repository; audio files are local-only and .gitignored.

## Roadmap

### MIDI expressiveness

- **Mute groups** - notes in a named group silence each other when triggered.
  Classic use: closed hi-hat silences open hi-hat.
- **Per-trigger sample variation** - cycle through alternative samples on
  repeated triggers to avoid the machine-gun effect on rapid notes. (Distinct
  from the existing per-hit segment round-robin, which cycles through detected
  hits inside a single sliced loop.)

### Playback and sound design

- **Loop playback** - sustain loops for pads, drones, and textures that play
  continuously while a key is held.

### Sample management

- **Auto-slicing** - chop loops and long recordings into individual hits by
  transient detection, then add each slice to the library as a separate sample.
- **Similar-to-this query** - "find more sounds like this one" by exposing the
  similarity engine as a user-facing search.

### Additional select/process features

- **Random selection** - `order_by: random` to pick a different sample on each
  trigger.

### Monitoring

- **Web dashboard** - a lightweight local web UI showing active bank, loaded
  samples, CC state, voice activity, and transform queue progress. Read-only
  visibility into what the engine is doing, without requiring a terminal.

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
similarity on a 58-element composite vector. The vector is split into five
groups, each independently L2-normalised so that no single group dominates by
scale:

```
Group 1 (x14): spectral shape   [flatness, attack, release, centroid, bandwidth, zcr,
                                  harmonic, contrast, voiced, log_attack, flux,
                                  spectral_rolloff, spectral_slope, crest_factor]
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
        → enqueue base variant (always)             ← float32 peak-normalised copy
        → enqueue pitch variants (tonal only)       ← Rubber Band offline finer engine
        → enqueue time-stretch (if BPM set + enough onsets) ← beat-quantized timemap_stretch
            → TransformProcessor worker pool
                → TransformCache (parent-priority FIFO eviction, 50 MB default)
```

The base variant (identity spec: no DSP) is produced for every sample -
percussive and tonal alike - so the playback path never pays the float32
conversion cost at trigger time. Pitch and time-stretch variants are additional
cache entries, derived from the same PCM source.

When a variant set for a parent sample would exceed the memory budget, the entire
oldest parent's variant family is evicted together, keeping the remaining
families intact and playable.

### Playback path

```
MIDI note_on
    → query engine: filter → order → pick → sample_id
        (fallback: try each select spec in order)
    → transform_manager.get_pitched()  → pitch variant (repitch assignments)
    → transform_manager.get_at_bpm()   → time-stretch variant (beat_quantize assignments)
    → transform_manager.get_base()     → base variant (all samples)
    → _render()                        → on-the-fly fallback (first trigger only)
    → _render_float(): apply gain · velocity² · anti-clip ceiling
    → append _Voice (float32 stereo, pre-rendered)
    ↓
PyAudio callback (PortAudio high-priority thread)
    → sum all active voices (float32 addition)
    → clip to [-1, 1]
    → float32_to_pcm_bytes(mixed, output_bit_depth)  → int16/24/32 bytes to hardware
```

All mixing happens in float32; precision-sensitive DSP (IIR filters, envelope
followers) promotes to float64 internally. The only integer conversion is the
final output packing. Multiple simultaneous voices are summed correctly regardless of the
output device's bit depth.

## Requirements

- Python 3.12+
- PortAudio (required by PyAudio - `apt install portaudio19-dev` or `brew install portaudio`)
- Rubber Band (required by pyrubberband - `apt install rubberband-cli` or `brew install rubberband`)

**Windows users:** install and run Subsample inside [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
(Windows Subsystem for Linux). This gives you a real Linux environment where
the `apt` instructions above just work. Audio devices need to be exposed to
WSL - see the [WSL audio guide](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)
for USB passthrough or use a network audio bridge if your interface supports
one. Subsample is not currently tested against native Windows Python.

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
| [pyrubberband ↗](https://github.com/bmcfee/pyrubberband) | Pitch shifting and time-stretching (Rubber Band wrapper) | ISC |
| [watchdog ↗](https://github.com/gorakhargosh/watchdog) | Filesystem monitoring for multi-machine sample hot-loading | Apache-2.0 |
| [PyMidiDefs ↗](https://github.com/simonholliday/PyMidiDefs) | MIDI constant definitions (notes, CC, drums, GM) | MIT |

### Academic references

The compressor/limiter DSP is based on the feed-forward design described in:

> D. Giannoulis, M. Massberg, and J. D. Reiss, "Digital Dynamic Range Compressor Design - A Tutorial and Analysis," *Journal of the Audio Engineering Society*, vol. 60, no. 6, pp. 399-408, 2012.

## About the Author

Subsample was created by me, Simon Holliday ([simonholliday.com ↗](https://simonholliday.com/)), a senior technologist and a junior (but trying) musician. From running an electronic music label in the 2000s to prototyping new passive SONAR techniques for defence research, my work has often explored the intersection of code and sound.

## License

Subsample is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).

You are free to use, modify, and distribute this software under the terms of the AGPL. If you run a modified version of Subsample as part of a network service, you must make the source code available to its users.

All runtime dependencies are permissively licensed (MIT, ISC, BSD-3-Clause) and compatible with AGPLv3.

## Commercial licensing

If you wish to use Subsample in a proprietary or closed-source product without the obligations of the AGPL, please contact [simon.holliday@protonmail.com] to discuss a commercial license.
