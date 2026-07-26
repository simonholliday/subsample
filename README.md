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
  - [Tools](#tools)
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
- **20-processor DSP chain with intelligent defaults.** Compression, gating,
  transient shaping, filters, distortion, saturation, bit-depth reduction,
  radio transmission/reception, frequency shift, tuning wobble, vocoder
  cross-synthesis, beat-quantize, pitch-shift, time-stretch, reverse,
  envelope reshape, and
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
  First-order ambisonic capture from tetrahedral mics (Rode NT-SF1,
  generic A-format, or pre-encoded B-format FuMA/AmbiX) with decoder and
  rotation at playback time - see [Ambisonic](#ambisonic-capture).
- **WAV or lossless FLAC storage.** Opt into FLAC (`audio_format: flac`) to
  shrink your sample library by ~40-60% with zero quality loss. Existing
  WAV samples continue to load unchanged alongside any new FLAC captures -
  see [Storage format](#storage-format).
- **Visual sample previews.** Every capture gets a fixed 1024x256 `.preview.png`
  thumbnail (waveform + 4-band frequency skyline + onset ticks + pitch/BPM
  badge) for browsing in an OS file manager, plus a compact preview-data
  block embedded in the analysis sidecar that the Supervisor dashboard
  renders as scalable SVG on demand - see [Sample previews](#sample-previews).
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
- **Pairs with Subsequence.** Subsample is one part of a fully open-source
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
| **DSP processors** | 20 (filter, comp, gate, distort, bit-depth, radio, freqshift, wobble, saturate, reshape, transient, HPSS, vocoder, repitch, stretch-quantize, pad-quantize, ...) |
| **Adaptive defaults** | Compressor, gate, transient shaper, distortion, envelope reshape - all auto-derive parameters from each sample |
| **Pitch shifting** | Rubber Band offline finer (highest available quality), pre-rendered |
| **Time stretch** | Beat-quantized with onset-aligned timemaps, partial-quantize amount, pad-quantize alternative for speech |
| **Segment playback** | Per-hit round-robin, random, or indexed - for sliced loops |
| **MIDI input** | Hardware port, named virtual port, or both |
| **MIDI control** | Note on/off, Program Change for programs, CC binding for any numeric parameter |
| **OSC** | Sender + receiver (optional dependency) |
| **Audio formats in** | WAV, BWF, FLAC, AIFF, OGG, MP3/MPEG (libsndfile) |
| **Channels** | Mono through 7.1, ITU-R BS.775 downmix, conservative upmix, per-instrument output routing |
| **Audio precision** | End-to-end 32-bit float pipeline, 64-bit DSP for IIR filters and envelope followers |
| **Latency** | Pre-rendered variants - playback is a memory copy into the mix buffer |
| **Library mgmt** | Memory-bounded with FIFO eviction, persistent disk cache for variants, hot-loading from watched directories |
| **Live-coding** | Edit the MIDI map YAML and assignments reload on save |
| **Program switching** | Multiple instrument directories swappable via MIDI Program Change |
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
SDR radio captures), `subsample import` bypasses detection entirely and imports
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

The MIDI map is where Subsample becomes an instrument you can *play*. It is the
most expressive MIDI routing surface of any sampler we know of: you don't just
assign samples to notes, you write *rules* that pick samples from your library
at trigger time - by similarity to a reference, by analysis metadata, by age,
by user-defined scoring functions, or by whatever combination you can write
down in a few lines of plain text. Samples can then be reshaped on the way
out through an ordered effects chain with MIDI CC control over every
parameter.

There is real complexity here - the price of a surface this expressive. The
rest of this section leads you in gently. A five-step tutorial first, each
step adding one concept on top of the last. Then the complete reference, then
the advanced features (programs, ambisonic capture, MIDI CC mapping).

MIDI routing is defined in a YAML file - by default `midi-map.yaml` in the
project directory, referenced from `config.yaml`:

```yaml
player:
  midi_map: midi-map.yaml
```

Two maps ship with Subsample, and `subsample --init` places both in your
project:

- **midi-map.yaml** (from the template
  [midi-map.yaml.default](subsample/data/midi-map.yaml.default)) - a
  heavily-commented starting point; uncomment the example you want and go.
- **midi-map-gm-drums.yaml**
  ([source](subsample/data/midi-map-gm-drums.yaml)) - a complete General MIDI
  percussion kit, ready to play against any sample library you point it at.
  Instant kit, no tweaking needed - `--init` wires it into `config.yaml` for
  you.

### Tutorial - five steps from simple to expressive

The examples below are working YAML. Each one is a self-contained
`assignments:` entry. Copy any of them into `midi-map.yaml` under
`assignments:` and reload to hear it.

#### Step 1 - play one specific sample

The simplest possible assignment: MIDI note 36 (on channel 10, the GM drum
channel) always plays one named sample.

```yaml
- name: My favourite kick
  channel: 10
  notes: 36
  select:
    where:
      name: 2026-03-24_14-37-14
```

`name` matches a sample's filename stem (no extension, no path). Strike note
36 and Subsample plays that exact sample. Everything else in the library is
ignored for this assignment.

#### Step 2 - "find me the best kick"

Now the interesting bit. Instead of naming a specific sample, describe the
*kind* of sample you want. Subsample's similarity engine will pick the closest
match from your library - every time you load new samples, the best candidate
may change, but you never have to rewrite the YAML.

```yaml
- name: Any kick
  channel: 10
  notes: 36
  select:
    where:
      reference: samples/reference/GM36_BassDrum1.wav
```

`reference` points at a reference sample by path. subsample ships (and
`subsample --init` scaffolds into `samples/reference/`) a precomputed
fingerprint sidecar for each General MIDI sound - the `.analysis.json` files
only, no audio - so the path above resolves from its fingerprint even though no
WAV sits beside it. The library's samples are ranked against that reference by
a 58-dimensional spectral/rhythmic fingerprint; the top-ranked match plays.
(When `reference` is set and no `order` is given,
`order: [{ by: similarity, dir: desc }]` is assumed - see
[Implicit defaults](#implicit-defaults) further down.)

#### Step 3 - rule-based selection

Filter the library by analysis metadata, sort the qualifying samples, and
pick one. This example plays the **oldest pitched sample** across a whole
keyboard range, pitch-shifted to each MIDI note:

```yaml
- name: Pitched keyboard
  channel: 1
  notes: C2..C6
  select:
    where:
      pitched: true              # only samples with a stable detected pitch
    order:
      - { by: age, dir: asc }    # oldest first
    pick: 1                      # take the top result
  process:
    - repitch: true              # pitch-shift each note to its MIDI value
  mode: gated                # release on note-off (sustained playback)
```

The `notes: C2..C6` range expands to every MIDI note between C2 and C6 - one
assignment, 49 notes. `repitch: true` pitch-shifts the chosen sample per note.

#### Step 4 - process the sample on the way out

Everything in `process:` is an ordered audio-effects pipeline. Order matters -
the sample flows through top to bottom.

```yaml
- name: Warm keys
  channel: 1
  notes: C2..C6
  select:
    where: { pitched: true }
    order: [{ by: age, dir: asc }]
  process:
    - filter_low: { freq: 2000, resonance: 6 }   # low-pass with resonant peak
    - saturate: { drive: 4 }                      # analog-style soft-clip
    - compress: true                              # adaptive dynamics
    - repitch: true                               # then pitch-shift
  mode: gated
```

Every processor accepts `true` for sensible defaults, or a dict for
fine-grained control. All the parameters of every processor are documented in
the [Process](#process---how-to-present-the-sample) reference below.

#### Step 5 - lock a loop to your session tempo

`stretch_quantize` time-stretches a sample to a target BPM and snaps its onsets
to a beat grid - turning any loosely-timed loop in your library into something
locked to the session. Combine it with filtering for a length+rhythm pick:

```yaml
- name: Tight loops
  channel: 2
  notes: C3..C4
  select:
    where:
      duration: { gte: 1.0, lt: 8.0 }   # at least 1 bar, less than 8
      onsets:   { gte: 4 }              # at least 4 transients
    order:
      - { by: duration, dir: desc }     # prefer longer loops
  process:
    - stretch_quantize: { strength: 0.7 }  # 70% snap - loose but locked
  mode: gated
```

`duration`, `onsets`, and other numeric predicates take per-field operator
dicts (`gte`, `lte`, `gt`, `lt`, `eq`). `strength: 0.7` is a partial-quantize
amount - fully snapped at 1.0, unchanged at 0.0.

That's the ladder. The rest of this section is the full reference - every
field, every predicate, every processor, every option - then the advanced
features (programs, ambisonic capture, MIDI CC mapping).

### The GM drums map - instant professional drum kit

Before the reference, a quick mention of the "no-config" path. If you want a
complete drum kit in under a minute, use the `midi-map-gm-drums.yaml` that
`subsample --init` placed in your project (it is even pre-wired into the
scaffolded `config.yaml`). Point Subsample at any sample collection and every
MIDI drum note automatically finds the closest matching sample and plays it
through a professional mix chain:

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

---

**Reference - every option.** From here on, this section is reference material:
every field, every predicate, every processor option. Skim it once; come back
when you want to try something the tutorial didn't show.

---

### Assignment fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Label shown in logs |
| `channel` | yes | MIDI channel 1-16 (standard numbering) |
| `notes` | yes | Single note, list, range, or `zone-tuned` for auto-derived keyboard layout (see Note syntax + Zone-tuned below) |
| `select` | yes | Which sample to play (see Select below) |
| `process` | no | How to present it (see Process below) |
| `mode` | no | Playback behaviour: `one_shot` (default) plays to the natural end and ignores note-off; `gated` fades out on note-off; `loop` holds a seamless loop while the key is held, then plays past it on release. Replaces the old `one_shot` flag - `one_shot: true` is now `mode: one_shot`, `one_shot: false` is `mode: gated`, and a leftover `one_shot:` key is a load error |
| `loop` | no | Manual loop-point override, e.g. `{ start: 1.2, end: 3.4, crossfade: 40 }` (start/end in seconds, crossfade in ms; each optional). Implies `mode: loop`. Omit to use the sample's auto-detected loop |
| `release` | no | Shape the note-off fade for a sustained voice (`mode: gated` or `loop`; ignored with a warning on `mode: one_shot`). A number of ms, `true` for an adaptive tail, `full` to play the remaining audio to its natural end with no fade (a loop rings out its real tail), or `{time, curve}` where `curve` is `cosine` (default) or `exponential`; `time` may be CC-bound. `mode: loop` defaults to the adaptive tail when unset. See Release below |
| `silenced_by` | no | Choke: the note(s) whose arrival cuts this sound with a fast ~10 ms damp (a hi-hat choke). A note, a list of notes, or `self` (a re-strike stops the previous hit), and `self` may sit in a list. Overrides `release` and rings-out; not valid on `zone-tuned`. See Choke below |
| `gain` | no | Level offset in dB (default 0.0). Negative = quieter, positive = louder |
| `pan` | no | Stereo position `-100` (hard left) to `100` (hard right), `0` = centre (default) - or a per-channel weight list for surround/asymmetric routing (`[50, 50]` = centre; ratios matter, not absolute values). Also `any` / a `{gte, lte}` range / `{position, variation}` for a fresh random position per note-on. Constant-power normalised at mix time. See Pan below |
| `output` | no | Physical output channels (1-indexed) e.g. `[3, 4]` routes to outputs 3-4 |
| `extract` | no | Collapse a multi-channel sample to one channel at playback: `omni`, `left`, `right`, `front`, `back`, `side`, `depth`, `height`, `channel.N`, or `{blend: [w1, w2, ...]}` for a weighted mix to mono (see Channel extraction below) |
| `velocity` | no | Velocity layering range — `[lo, hi]` filter only, or `{trigger: [lo, hi], rescale: …}` with optional in-band rescaling (see Velocity layering below) |
| `stack` | no | `true` lets this sound play together with other `stack: true` assignments on the same note and velocity, instead of being rejected as an overlap (see Stacking below). Default `false` |
| `template` | no | Inherit fields from one or more named templates (see Templates below). The assignment's own fields override the template's; lists (`process`) and nested blocks (`select`) are replaced wholesale, not merged |

### Release - shape how a held note fades

By default a sustained sound (`mode: gated`) fades out over a fixed 10 ms
when you lift the key - just enough to avoid a click. `release` lets you set
how long that fade is, and its shape:

```yaml
- name: Warm pad
  channel: 1
  notes: 48
  mode: gated
  release: 800                       # fade over 800 ms after key-up
  select: { where: { name: { matches: "*pad*" } } }

- name: Stab
  channel: 1
  notes: 60
  mode: gated
  release: { time: 120, curve: exponential }   # fast drop, long tail
  select: { where: { name: { matches: "*stab*" } } }

- name: Expressive pad
  channel: 1
  notes: 64
  mode: gated
  release: { time: { cc: 72, min: 20, max: 3000 } }   # a pedal sets the length
  select: { where: { name: { matches: "*pad*" } } }
```

- A bare number is the fade time in milliseconds.
- `true` gives an adaptive tail - short for percussive material, longer for
  sustained sounds.
- The mapping form sets `time` and/or `curve`. `curve` is `cosine` (smooth,
  the default) or `exponential` (fast initial drop, long tail, like a damped
  string). `time` may be a `{cc, min, max}` mapping, read the instant each note
  is struck - so the knob shapes the notes you play next, not the one already
  ringing.

`release` only applies to voices that actually receive a note-off, so it needs
`mode: gated` or `mode: loop`. Declared on a `mode: one_shot` (play-to-end)
voice it is ignored, with a warning at startup.

**`release` vs `reshape`.** These are different tools: `reshape` (a `process:`
step) shapes the sample's *own* amplitude body - baked in, fired on every hit,
the same regardless of how long you hold the key. `release` shapes the fade
when you *lift* the key - applied live, timed from the note-off. Both can be
set at once (they multiply), but for amplitude you usually want one or the
other.

**`release: full`** is the exception to a fade: on note-off the voice plays the
remaining audio to its natural end with no fade at all. On a plain sample that
just lets it ring out; on a `mode: loop` voice it stops looping and plays the
real tail - the natural decay of the held note.

One honest limit: `release` can only shape whatever sample audio is still
playing when you lift the key. A long release on a short sample fades what is
left - it cannot invent a sustain tail, and a note held past the sample's
natural end still stops there. For a sound that sustains as long as you hold the
key, use `mode: loop` (below).


### Loop - hold a sustain for as long as the key is down

`mode: loop` turns a sustaining sample into a held instrument: the attack plays
once, then a seamless slice of the steady part loops for as long as you hold the
key. Lift the key and it stops looping and plays on through the sample's real
tail, shaped by `release`.

```yaml
- name: Pad
  channel: 1
  notes: C2..C4
  select:
    where: { loopable: true }
  mode: loop            # attack, then loop the sustain while held
  release: full         # on note-off, ring out the natural tail (no fade)
```

subsample finds the loop points automatically - the steadiest slice of the
sustain, with a short crossfade so the wrap is inaudible. The `loopable` catalog
column and `subsample loops` let you preview which samples loop well. To
place the loop by hand, give `loop:` in seconds (crossfade in ms):

```yaml
  loop: { start: 1.2, end: 3.4, crossfade: 40 }
```

Any field you omit falls back to the automatic value, and a `loop:` block on its
own implies `mode: loop`. If a sample has no clean loop point - too evolving to
wrap seamlessly - it plays *gated* instead (held, then released) and a line in
the log says so.

Loop playback can't yet be combined with `repitch` or time/pad `*_quantize`
(those change the timeline the loop points are measured in); an assignment that
asks for both plays gated with a warning.

**Re-pressing a held note.** Re-striking a `mode: gated` or `mode: loop` note
that is already sounding replaces it: the sounding voice is released - it stops
looping and fades over its `release`, or rings out its natural tail under
`release: full` - while the new press starts instantly over the top. One held key
is one voice, so a trill or a re-triggering sequencer restarts the note instead
of stacking loops that never stop. (`release: full` still rings its tail out as
asked, so a released tail and the new note can overlap - your choice to let them.)
`mode: one_shot` is unchanged: repeated hits layer, the way striking a drum twice
should.

### Choke - silence one sound when another plays

A real hi-hat is one instrument: close it and the open ring stops dead. But an
open-hat sample and a closed-hat sample are two files that know nothing about each
other, so `silenced_by` lets you say "cut this sound the instant that other note
plays". It is the classic drum choke - and it is what lets a one-shot ring out on
its own yet still stop on cue.

Declare it on the sound that should be *cut*, naming the note(s) that cut it -
any note form works, including your own names from a mounted definitions file
(see [Naming your own sounds](#naming-your-own-sounds---the-definitions-file)):

```yaml
- name: Open Hi-Hat
  channel: 10
  notes: drum.hi_hat_open
  silenced_by: [self, drum.hi_hat_closed, drum.hi_hat_pedal]
  select: [ where: { reference: samples/reference/GM46_OpenHiHat.wav } ]

- name: Ride
  channel: 10
  notes: drum.ride
  silenced_by: self          # one physical cymbal - a new hit damps the last
  select: [ where: { reference: samples/reference/GM51_RideCymbal1.wav } ]
```

- The value is a single note, a list of notes, or `self` - and `self` can sit in
  a list beside notes. Notes use the same syntax as `notes:`: a `drum.*` name, a
  MIDI number, or a note name.
- `self` means the sound chokes itself - re-striking it stops the previous hit, so
  a single ride, cowbell, or triangle never piles up overlapping copies of its own
  ring.
- The cut is a fast ~10 ms damp, not a note-off. It deliberately overrides the
  sound's own `release` (even `release: full`) and any loop tail: a choke models a
  hand stopping the cymbal, so the ring genuinely stops instead of fading over its
  natural release.
- Every sounding copy is cut, not just the newest - a fast open-hat roll all damps
  together when the hat closes, the way one physical cymbal would.
- Choke acts within a channel (a kit is one channel). Build a mutual group - like
  the three hi-hat articulations above - by listing the others (and `self`) on each
  member.
- Choke works on the note, not the individual sound: if you deliberately `stack` an
  unrelated sample on a choked note, it is damped along with the group. In a normal
  kit - one instrument per note - this never arises.

`silenced_by` is what finally stops a `mode: one_shot` sound early: a one-shot
ignores note-off, so before choke the only way to end a ringing open hat was to let
it decay. It is not supported on `zone-tuned` assignments (that would make a whole
keyboard range monophonic). The shipped GM drums map uses it on the hi-hats,
cymbals, vibraslap, and triangle.

### Templates - share fields across assignments

Drum kits repeat themselves: the same `channel`, the same process chain, the
same `select` shape on every pad. Define those shared fields once in a
top-level `templates:` section and pull them into each assignment with a
`template:` reference.

```yaml
templates:
  percussion:                    # any subset of assignment fields
    channel: 10
    extract: omni
    process:
      - gate: true
      - transient: true

assignments:
  - name: Kick
    template: percussion         # inherits channel, extract, process
    notes: drum.kick
    select: { pick: any, where: { directory: Kick } }

  - name: Snare
    template: percussion
    notes: drum.snare
    select: { pick: any, where: { directory: Snare } }
```

**How a template merges.** The assignment starts from the template's fields,
then its own fields win: a field it *sets* replaces the template's, a field it
*omits* is inherited. The merge is top-level only — if the assignment sets its
own `process` (or `select`), that **replaces** the template's wholesale rather
than appending to or merging into it.

**Stacking templates.** `template: [percussion, loud]` applies several
left-to-right: a later template overrides an earlier one, and the assignment's
own fields override them all.

Templates are flat — a template cannot itself carry a `template:` (inheritance
is one level deep). A `template:` that names an undefined template is a load
error listing the templates you defined.

### Note syntax

```yaml
notes: 36          # single MIDI note number
notes: C4          # scientific pitch (C4 = MIDI 60, as in REAPER; Ableton/Logic show 60 as C3, FL as C5)
notes: drum.kick_1 # GM percussion by symbolic name (case-insensitive)
notes: [36, 35]    # list - each gets the next similarity rank (first = best match)
notes: [drum.kick_1, drum.snare_1]   # list of symbolic names
notes: C2..C4      # range - expands to every MIDI note from C2 (36) to C4 (60)
notes: 36..60      # range with note numbers
```

Note names use the convention C4 = 60 (C-1 = 0, G9 = 127). Sharps: `C#4`,
`D#3`. Flats: `Db4`, `Eb3`.

**Symbolic GM drum names** — `drum.<name>` looks up
[`pymididefs.drums.GM_DRUM_MAP`](https://github.com/simonholliday/PyMidiDefs)
(case-insensitive, so `drum.kick_1`, `drum.KICK_1`, and `Drum.kick_1` are
equivalent). Covers the full GM percussion key map (notes 27-87): kicks,
snares, hi-hats, toms, cymbals, Latin percussion, shakers, woodblocks,
triangles. Use a list for multiple drums — `drum.x..drum.y` ranges are
deliberately not supported because drum names aren't a musical sequence.
Equivalent: `notes: drum.low_floor_tom` and `notes: 41` produce the same
result; the symbolic form simply makes intent visible.

The four instruments GM defines in numbered pairs — kick, snare, crash, ride —
also accept the **bare alias** `drum.kick`, `drum.snare`, `drum.crash`,
`drum.ride`, each pointing at the GM-designated primary (`drum.kick` =
`drum.kick_1` = Bass Drum 1 = 36). Reach for the bare name when you just want
"the kick" rather than choosing between the two variants; use the numbered
form when you specifically want `_2`.

### Naming your own sounds - the definitions file

GM names cover standard drums, but your own sample sets deserve their own
vocabulary. A *definitions file* is a small YAML file, owned by your music
project, that gives names to note, CC, channel, and program numbers - so your
map reads `notes: my.dawn_chorus_pheasant` instead of `notes: 60`, and the
same file can name the same sounds in your sequencer. Neither tool depends on
the other; they just read the same trivial file.

```yaml
# project.yaml - or any filename you like
notes:
  ride_edge_soft: 53
  dawn_chorus_pheasant: 60
cc:
  sampler_release: 21
channels:
  kit: 10
  birds: 3
programs:
  brushes: 1
```

Mount it in the MIDI map under a prefix of your choosing (the path is
relative to the map file, like `reference:` paths):

```yaml
definitions: { my: project.yaml }

assignments:
  - name: Pheasant
    channel: my.birds                     # channels: section
    notes: my.dawn_chorus_pheasant        # notes: section
    silenced_by: my.ride_edge_soft        # notes: section (chokes too)
    release: { cc: my.sampler_release }   # cc: section
```

The field decides which section a name is looked up in: note positions
(`notes:`, `silenced_by:`, zone `range:`) read `notes:`; `channel:` and
`program_channel:` read `channels:`; CC bindings read `cc:`; `program:` and
`default_program:` read `programs:`.

| Section | Names | Allowed values |
|---------|-------|----------------|
| `notes` | MIDI notes | 0-127 |
| `cc` | controller numbers | 0-127 |
| `channels` | MIDI channels | 1-16 |
| `programs` | program-change numbers | 0-127 |

Rules worth knowing:

- Names are lowercase `a-z`, digits, and underscores in the file, used as
  `prefix.name`, and matched case-insensitively at the point of use - exactly
  like `drum.*`. The `drum` prefix itself is reserved.
- Sections subsample does not use are ignored without comment, so the same
  file can hold your sequencer's own vocabularies (`nrpn:`, whatever it
  needs) alongside these four.
- You can mount several files under different prefixes:
  `definitions: { kit: kit.yaml, birds: birds.yaml }`.
- The definitions file is re-read whenever the map reloads. Editing the
  definitions file alone takes effect the next time the map file is saved
  (or the player restarts) - the watcher follows the map, not the
  definitions file.

### Select - which sample to play

The `select` block defines how to choose a sample from the instrument library.
It has three parts: filter predicates (`where`), a sort order (`order`), and
a pick position (`pick`).

```yaml
select:
  where:
    duration: { gte: 1.0 }                # at least 1 second long
    onsets:   { gte: 4 }                  # at least 4 transient hits
  order:
    - { by: age, dir: desc }              # most recently captured first
  pick: 1                                 # take the first match
  # pick: [1, 3]                          # or: a different match in your top 3
                                          #     on every hit (random per trigger)
```

`select` is usually a single block like this. It can also be a *list* of
blocks for fallback chains - try the first, and if nothing matches try the
next. See [Fallback chains](#fallback-chains) below.

All `where` predicates must pass (AND logic).

Numeric predicates (`duration`, `duration_beats`, `onsets`, `tempo`, `pitch`, `quantized_beats`)
use a per-field operator dict. Operators:

| Operator | Meaning |
|-----|---------|
| `gte` | `>=` inclusive lower bound |
| `lte` | `<=` inclusive upper bound |
| `gt` | `>` strict lower bound |
| `lt` | `<` strict upper bound |
| `eq` | `==` exact equality |

Any combination on one field AND-composes. A bare scalar under a numeric field
is shorthand for `eq` — e.g. `quantized_beats: 4` is the same as
`quantized_beats: { eq: 4 }`.

| Predicate | Type | Description |
|-----------|------|-------------|
| `duration` | float (seconds) | Filter by sample length. Example: `{ gte: 1.0, lt: 5.0 }` |
| `duration_beats` | float (beats) | Filter by sample length in *beats at the session tempo* rather than seconds, so the pool stays musically tight as the tempo changes. A beat is a quarter note: a 16th note is `0.25`, an 8th `0.5`, a quarter `1`, a bar of 4/4 `4`. Example: `{ lt: 0.25 }` keeps only samples shorter than a 16th note. Needs a session tempo (`tempo.bpm`, or a followed MIDI clock) - a map that uses it will not load without one. Distinct from `quantized_beats`, which measures a quantize processor's *output*; this measures the sample's own recorded length. |
| `onsets` | int | Filter by detected transient count. Example: `{ gte: 4 }` |
| `tempo` | float (BPM) | Filter by detected tempo. Example: `{ gte: 100, lte: 140 }` |
| `pitch` | Hz or note name | Filter by detected frequency. Each operator value is either a Hz float (`{ gte: 130.8 }`) or a note name (`{ gte: C3, lt: C6 }`). The two forms are interchangeable - note names are converted to Hz at parse time. Sharps: `C#4`; flats: `Db4`. |
| `quantized_beats` | float (beats) | Filter by the beat length of the assignment's `stretch_quantize`/`pad_quantize` output. Samples whose quantized variant has not yet been computed (or whose assignment has no quantize step with a valid BPM) are excluded when this predicate is active. Non-integer values accepted. |
| `pitched` | bool | `true` = has stable pitch; `false` = not pitched |
| `loopable` | bool | `true` = has a steady sustaining region (tonal or textural) worth looping while a key is held; `false` = does not. A coarse candidate flag over existing analysis - see the loopable column of `subsample catalog` to preview which samples pass |
| `reference` | path | Similarity match against a reference sample (path to WAV) |
| `name` | string / list / dict | Filename stem match. Four forms — see below. Legacy: a path-like scalar value (containing `/` or starting with `.`) is still auto-detected as a `path:` |
| `path` | path | Match a specific WAV file at this path (relative paths resolved against the MIDI map's directory). Preferred over `name:` for file references |
| `directory` | path | Only match samples whose file path is inside this directory (auto-loads on startup; see [Programs vs directory predicate](#programs-vs-directory-predicate)) |

The `name:` predicate accepts four forms:

```yaml
where:
  name: my-kick                       # 1. exact stem match (case-sensitive)
  name: [my-kick-1, my-kick-2]        # 2. list of exact stems (case-sensitive)
  name: { matches: "*kick*" }         # 3. glob — fnmatch-style, case-insensitive
  name: { regex: "kick_\\d+" }        # 4. regex — re.fullmatch, case-insensitive
```

- The **list** form matches if the sample's stem is in the list. Pair with
  `pick: any` for uniform random selection across the whole set.
- The **glob** form (`matches:`) uses `*`, `?`, and `[abc]` character classes;
  `.` is a literal dot. Full-string match — `kick` matches only `kick`,
  `*kick*` matches any stem containing `kick`. Case-insensitive.
- The **regex** form (`regex:`) is `re.fullmatch` with `re.IGNORECASE`. Must
  match the entire stem. YAML tip: prefer double-quoted strings so `"\\d+"`
  is interpreted as `\d+`. Bad regex syntax is surfaced at map-load time.
- All forms match the **stem only** (filename without extension or path). A
  stem is a label, not a unique id - the same filename can appear in several
  folders (e.g. per-technique take-folders each holding `01.wav`), so `name:`
  may match **several** samples. Narrow to one with `directory:` (folder
  containment), `path:` (an exact file), or `pick:` (choose among the matches).

`name:` (any form) and `path:` are mutually exclusive within a single `where`
block — use one, not both. Inside a `where` block, only one of the four `name:`
forms is allowed; combine multiple patterns via a `select:` fallback chain
instead.

**Legacy `min_X` / `max_X` syntax**: the pre-2026-04 form
(`min_duration: 1.0`, `max_pitch: A4`, etc.) still works indefinitely —
the parser translates each legacy key into the equivalent operator
(`gte` for `min_`, `lte` for `max_`). Mixing both forms on the same
field in one `where` block raises an error; use one form per field. New
YAML should prefer the operator-dict form.

`order` is a list of clauses. Each clause has a `by` (scorer name), a `dir`
(`asc` or `desc`; the default is `asc`, except for the match-quality scorers
`similarity` and `beat_match`, which default to `desc` so the best match comes
first), and optional scorer-specific parameters.
Later clauses break ties on earlier ones, so primary sort + secondary
tie-breaker is natural:

```yaml
order:
  - { by: duration, dir: desc }           # primary
  - { by: onsets,   dir: asc }            # tiebreaker
```

Built-in scorers:

| `by` | What it sorts by |
|-----|------------------|
| `age` | Arrival time (sample_id) — `desc` = newest first |
| `duration` | Sample length in seconds |
| `pitch` | Dominant frequency |
| `onsets` | Detected onset count |
| `tempo` | Detected BPM |
| `level` | RMS loudness |
| `quantized_beats` | Beat length of the assignment's `stretch_quantize`/`pad_quantize` output. Samples without a computed variant park at the end regardless of direction. |
| `similarity` | Similarity rank against the reference in `where`. Only supported as the primary clause; requires `reference` in `where`. When `reference` is set and no `order` is given, `similarity` desc is assumed automatically. |
| `beat_match` | Cosine similarity between a user-supplied `pattern:` (a list of numbers in `[0, 1]` per beat) and the sample's per-beat energy profile. Requires a `stretch_quantize` or `pad_quantize` step in the same assignment; samples without a quantized variant are excluded from the result. See [Beat-pattern matching](#beat-pattern-matching) below for the full semantics. |

#### Implicit defaults

The parser fills in a few defaults that are easy to miss - they make the
common case concise, but it helps to know which ones are on:

| Omitted key | Default applied | When |
|---|---|---|
| `order` | `[{ by: age, dir: desc }]` (newest first) | No `where.reference` set |
| `order` | `[{ by: similarity, dir: desc }]` | `where.reference` **is** set |
| `pick` | `1` (best match) for the first note; incremented per note thereafter | Multi-note assignment without `repitch`, and no explicit `pick` |
| `pick` | `1` for every note | Multi-note assignment with `repitch` in `process` |
| `pick` | Same `pick` for every note (no per-note distribution) | Any explicit `pick` — scalar, range, open-ended, or `any` |
| `where` | Empty (all samples match) | `where` block omitted |
| `process` | Empty (unprocessed playback) | `process` block omitted |
| `grid` | `16` (sixteenth-note) | `stretch_quantize` / `pad_quantize` without explicit grid |
| `tempo` | Session `tempo.bpm` from `config.yaml` | `stretch_quantize` / `pad_quantize` without explicit tempo |
| `mode` | `one_shot` | Omitted from assignment |
| `gain` | `0.0` dB | Omitted from assignment |
| `pan` | Identity routing | Omitted from assignment |
| `output` | Outputs `1..N` | Omitted from assignment |

The `where.reference` → `similarity desc` coupling is worth calling out: if
you set `reference` and then later add another filter like
`duration: { gte: 1.0 }`, the ordering is still similarity - there's nothing
visible in the YAML telling you so. Add an explicit `order:` clause if you
want a different sort.

`pick` is 1-indexed. Default: 1 (first match). For multi-note assignments
without explicit `pick`, each note gets the next position (rank distribution) -
so `notes: [36, 35]` gives note 36 pick 1 (best match) and note 35 pick 2.

**Pick a different sample on every hit.** `pick` also accepts a *range*: a
fresh random rank is drawn on every note-on, so the same pad plays a different
sample each time without scripting. Two equivalent forms:

```yaml
pick: [1, 3]              # shorthand: random rank in 1..3 inclusive
pick: { gte: 1, lte: 3 }  # explicit: same vocabulary as `where:` operators
```

The dict form also accepts `gt` / `lt` / `eq` (so `pick: { gt: 1, lt: 5 }`
draws from ranks 2-4). If the upper bound exceeds the number of available
matches, the draw clamps to the last rank — matching the scalar fallback
behaviour. Any explicit `pick` (scalar or range) suppresses per-note
distribution, so a range on `notes: [60, 61, 62]` rolls independently for
each key instead of fixing different ranks to different notes.

**Pick any match — open-ended ranges.** To draw uniformly across *every*
match without counting your library, leave an end open. Write `null` in the
list, drop the upper bound from the dict, or use the `any` shortcut:

```yaml
pick: any                 # uniform draw across all matches
pick: [null, null]        # same thing, list spelling
pick: [2, null]           # rank 2 to the last match (skip the top hit)
pick: [null, 5]           # the top 5 (same as [1, 5])
pick: { gte: 2 }          # rank 2 onward, open upper bound
```

An open lower bound means "from the best match"; an open upper bound means
"to the last match", resolved against the live library on every hit — so a
new capture is automatically in the running with no map edit. When the whole
list is in play (`pick: any`), the `order:` clause makes no difference: every
match is equally likely. (`pick: {}` with no operators is rejected — write
`pick: any` if that's what you mean.)

**Map velocity to the right sample.** `pick: velocity` chooses the sample by
how hard the note is played: a gentle note-on picks from the quiet end of the
pool, a hard one from the loud end, and everything between maps across the
ranks. Point it at a folder of one sound recorded at many strengths - ghost
notes through full strikes - and each velocity plays the take that was performed
at roughly that strength.

```yaml
- notes: drum.snare
  channel: 10
  select:
    where: { directory: snare }
    order: quietest       # required - ranks the pool quiet to loud
    pick: velocity
```

The chosen sample sets the *timbre*; the note's velocity still sets the
*loudness* the usual way, so the response stays smooth as you cross from one
sample to the next rather than stepping. Ten samples covering the whole velocity
range each cover about 13 velocity steps, and the gaps are filled by gain - you
do not need one sample per velocity.

Two refinements are available in the long form:

```yaml
pick: { mode: velocity, variation: 10, curve: logarithmic }
```

- `variation` (default `0`) spreads the *choice* by up to this many velocity
  steps (here plus or minus 5), so repeated notes at one velocity vary in tone
  without varying in level - the loudness always follows the velocity you
  actually played, never the randomised value. It keeps a programmed part from
  retriggering one identical sample.
- `curve` (default `linear`) bends the velocity-to-sample mapping.
  `logarithmic` spreads gentle notes across more of the quiet samples - more
  distinct ghost-note tones at the bottom of the range; `exponential` favours
  finer resolution among the loud samples instead.
- `spacing` (default `rank`) sets how the samples are laid out across the
  velocity range. `rank` spreads them evenly by position - ten samples each own
  about a tenth of the range whatever their actual level - which is right when
  you record an even spread of strengths. `loudness` (`pick: { mode: velocity,
  spacing: loudness }`) instead places each sample at its own measured level, so
  the layout follows the real dynamics you captured. Record nine near-silent
  ghosts and one hard hit and the difference is stark: `rank` gives the hard hit
  only the top tenth of the range - firm notes still play ghosts - while
  `loudness` lets the hard hit own the whole loud half and packs the ghosts into
  the quiet end, so soft playing picks among the ghosts and anything firm snaps
  to the hit. Reach for it when your takes are clustered or lopsided; keep the
  default `rank` when they cover the range evenly. (Under `loudness` a lone
  very-soft or very-loud sample sits in a proportionally tiny velocity slice, so
  it takes an extreme note to reach - widen that end with `curve` if needed.)

Unlike `pick: any`, the `order:` here is mandatory and load-bearing - it is what
ranks the pool quiet-to-loud (`quietest` or `loudest`; either works). To lift or
narrow the *loudness* range so ghosts are not too quiet, pair it with a velocity
`rescale:` (below) - that shapes the output gain and is independent of which
sample is picked.

#### Beat-pattern matching

`beat_match` is the shape-based companion to `similarity`: where `similarity`
ranks by spectral/timbral closeness to a reference sample, `beat_match` ranks
by *rhythmic* closeness to a user-defined pattern.

**Applies only to quantized samples.** `beat_match` scores the per-beat energy
profile that `stretch_quantize` and `pad_quantize` produce as a by-product of
snapping onsets to a beat grid. Any assignment that uses `beat_match` in its
`order:` must therefore include one of those processors in its `process:`
block - without a quantize step, no sample has an energy profile to compare
against, and the result set is empty.

```yaml
select:
  where:
    duration: { gte: 1.0 }
    onsets:   { gte: 4 }
  order:
    - { by: beat_match, pattern: [1, 0, 1, 0, 1, 0, 1, 0] }
process:
  - stretch_quantize: { grid: 16 }
```

**The pattern.** A list of numbers in `[0, 1]`, one per beat. Values are
relative — only the shape matters, not the absolute magnitudes. Examples:

| Pattern | Intent |
|---|---|
| `[1, 0, 1, 0]` | energy on every other beat (back-beat feel) |
| `[0, 1, 0, 1]` | energy on the off-beats |
| `[1, 0.9, 0.8, 0.7, 0.6, 0.5]` | gentle decay from beat 1 to beat 6 |
| `[0, 0, 1, 1, 0, 0, 1, 1]` | double-hits on beats 3-4 and 7-8 |

**How a sample is scored.** Each quantized sample has a *grid energy profile* —
per-slot RMS computed after the quantize step. `beat_match` mean-pools that
profile down to per-beat energy (so an 8th-note grid and a 16th-note grid
both reduce to the same per-beat values — cross-grid invariance), then
computes cosine similarity between the pattern and the profile over
`min(len(pattern), len(beats))` elements (left-aligned). Samples with no
quantized variant score `None` and are excluded.

**Behaviour summary:**

- `dir: desc` (default) = best match first. `dir: asc` = worst match first.
- Shape-sensitive, level-insensitive: `[1, 0, 1, 0]` perfectly matches a
  sample with energy `[0.5, 0, 0.5, 0]` (score 1.0).
- Length mismatches are truncated left-aligned — no resampling, no padding.
- Ints and floats are both accepted in the pattern list; values outside
  `[0, 1]` are rejected at parse time.

#### Fallback chains

`select` can be a list of specs tried in order. The first that returns a
result wins:

```yaml
select:
  - where: { name: my-favourite-kick }                               # try specific sample first
  - where: { reference: samples/reference/GM36_BassDrum1.wav }       # fall back to similarity match
```

#### Legacy `order_by:` syntax

The pre-2026-04 `order_by:` key with a bare-string token is still accepted
indefinitely — the parser translates it into the equivalent `order:` clause.
These two forms produce identical results:

```yaml
# Legacy (still accepted)
select:
  where: { pitched: true }
  order_by: pitch_desc

# Preferred (new form)
select:
  where: { pitched: true }
  order:
    - { by: pitch, dir: desc }
```

Legacy tokens map as follows: `newest` → `{by: age, dir: desc}`, `oldest` →
`{by: age, dir: asc}`, `duration_desc` → `{by: duration, dir: desc}`,
`loudest` → `{by: level, dir: desc}`, `quietest` → `{by: level, dir: asc}`,
`quantized_beats_desc` → `{by: quantized_beats, dir: desc}`, `similarity` →
`{by: similarity, dir: desc}`, and so on — field name without the `_asc`/
`_desc` suffix goes into `by`, the suffix determines `dir`. Mixing both keys
on the same `select` entry is an error.

#### Examples

```yaml
# GM kicks - ranked by similarity to a kick reference
select:
  where:
    reference: samples/reference/GM36_BassDrum1.wav
  order:
    - { by: similarity, dir: desc }

# Pitched keyboard - oldest tonal sample, repitched per note
select:
  where:
    pitched: true
  order:
    - { by: age, dir: asc }
  pick: 1

# Rhythmic loops - recent, long, with enough beats
select:
  where:
    min_duration: 1.0
    min_onsets: 4
  order:
    - { by: age, dir: desc }

# Longest sample, with onset count breaking ties
select:
  where:
    pitched: true
  order:
    - { by: duration, dir: desc }
    - { by: onsets,   dir: desc }
  pick: 1
```

### Process - how to present the sample

The optional `process` block declares an ordered list of audio processors
applied after sample selection. Omit it entirely for unprocessed playback.

```yaml
process:
  - filter_low: { freq: 800, resonance: 6 }   # low-pass, then
  - repitch: true                               # pitch-shift, then
  - saturate: { drive: 4 }                      # saturation
```

Processors execute in the order you declare them - different orderings
produce different results. The full chain is pre-computed and cached.

Available processors:

| Processor | Parameters | Description |
|-----------|-----------|-------------|
| `repitch: true` | none | Pitch-shift to match the triggering MIDI note |
| `repitch: { note: C4 }` | target note | Pitch-shift to a fixed note |
| `stretch_quantize: true` | grid (default 16), tempo (config `tempo.bpm`), strength (default 1.0) | Time-stretch to session `tempo.bpm` with all defaults |
| `stretch_quantize: { grid: 16 }` | as above, grid overridden | Time-stretch to session `tempo.bpm` |
| `stretch_quantize: { tempo: 120, grid: 8 }` | explicit tempo + grid | Time-stretch to a specific tempo |
| `stretch_quantize: { strength: 0.5 }` | 0.0-1.0 (default 1.0) | Partial quantize - onsets move partway to the grid for a looser feel |
| `pad_quantize: true` | grid (default 16), tempo (config `tempo.bpm`), strength (default 1.0) | Silence-pad onsets with all defaults |
| `pad_quantize: { grid: 16 }` | as above, grid overridden | Onset-aligned silence padding - snaps onsets to the beat grid by inserting silence between segments rather than time-stretching. No pitch/speed change. Ideal for speech. |
| `pad_quantize: { strength: 0.75 }` | 0.0-1.0 (default 1.0) | Partial quantize - same as stretch_quantize strength but for silence-pad mode |
| `filter_low: true` | freq (Hz, default 16000), resonance (dB, default 0) | Low-pass filter (console-style default) |
| `filter_high: true` | freq (Hz, default 80), resonance (dB, default 0) | High-pass filter (console-style default) |
| `filter_band: true` | freq (Hz, default 1000), q (default 0.7), resonance (dB, default 0) | Band-pass filter (Q sets width) |
| `reverse: true` | none | Reverse the audio |
| `saturate: true` | drive (default 6 dB) | Soft-clip saturation with level compensation (moderate default warmth) |
| `saturate: { drive: 6 }` | drive (dB) | Soft-clip saturation with explicit drive |
| `compress: true` | threshold (auto), ratio (4:1), attack (auto), release (auto), knee (6 dB), makeup (0 dB), lookahead (0 ms) | Dynamic range compressor (adapts to each sample) |
| `limit: true` | threshold (-1 dB), release (50 ms), lookahead (5 ms) | Brickwall limiter (ratio 100:1, instant attack) |
| `hpss: { keep: harmonic }` | keep (required: `harmonic` or `percussive`) | Keep only harmonic/tonal content (remove percussion) |
| `hpss: { keep: percussive }` | as above | Keep only percussive/transient content (remove harmonics) |
| `gate: true` | threshold (auto), attack (auto), release (auto), hold (auto), lookahead (auto) | Noise gate - silences audio below the noise floor. All parameters auto-adapt: threshold from noise floor, attack/release/hold from onset and decay character. |
| `distort: true` | mode (hard_clip), drive (auto), mix (1.0), tone (auto), bit_depth (8), downsample_factor (4) | Waveshaping distortion with four modes: hard_clip, fold, bit_crush, downsample. Drive adapts to crest factor; tone adapts to spectral rolloff. |
| `bit_depth: 12` | bits (1-16, default 12), dither (false, true = triangular, or `triangular` / `rectangular`) | Clean bit-depth reduction - requantizes to an N-bit amplitude grid for vintage sampler grit (the MPC60 and SP-1200 store 12-bit samples). Pure quantization: no drive, tone filtering, or level changes. Dither defaults off (vintage units had none); turn it on to trade the gritty low-level distortion for a smooth hiss. |
| `radio: { mode: am }` | mode (am, lw, fm, ssb), demod (matched, am, fm, ssb), tune (Hz), signal (0-1), static (0-1), fade (0-1), bandwidth (Hz), stereo (mono, stereo), mix (1.0) | Broadcast-and-received: a full transmit -> channel -> demodulate round-trip. `lw` is AM through a steep longwave channel; `fm` is narrowband NFM; `ssb` with `tune` is the mistuned voice. `demod` different from the mode is the deliberate wrong demodulator (`fm` -> `ssb` is a musical warble; AM<->FM is harsh noise, silence-guarded if it recovers nothing). `signal` weakens the station (hiss, FM clicks, AGC swell-on-fade), `static` adds atmospheric crackle, `fade` adds shortwave swimming. Authentic mono collapse by default; `stereo` for dual-mono. |
| `freqshift: 1000` | shift_hz (signed Hz), mix (1.0) | Bode/single-sideband frequency shift - adds a constant number of Hz to every partial (harmonic ratios break; NOT a pitch shift). Small shifts detune and phase; large shifts go clangorous and metallic. |
| `wobble: { depth: 6, rate: 0.3 }` | depth (Hz), rate (Hz), base (Hz), mix (1.0) | Oscillator warble - a slow continuous drift of the tuning (microphonic / BFO wander). `depth` sets the Hz of wander, `rate` the LFO speed, `base` a constant offset to drift around. |
| `reshape: true` | attack (preserve), hold (0), decay (preserve), sustain (1.0), release (auto) | ADSR envelope reshaping. Default auto-tightens the tail. Set attack, decay, sustain, release to reshape specific phases. |
| `transient: true` | gain (auto, dB signed) | Transient enhancement/taming via HPSS rebalancing. Auto-adapts from crest factor: peaky samples are tamed, dull samples enhanced. |
| `transient: { gain: 6 }` | gain (dB, signed: +/- enhance/tame) | Explicit dB of transient enhancement or taming |
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

- **threshold** - set 6 dB below the sample's peak level (always engages),
  independent of how loud the sample was recorded
- **attack** - slow for percussive samples (lets the transient punch through),
  fast for gradual onsets (no transient to protect)
- **release** - short for quick-decay samples (recovers before the next hit),
  long for sustained sounds (avoids pumping)

Explicit thresholds (`compress: { threshold: -30 }`, `gate: { threshold: -40 }`)
are dBFS against the processing buffer, which every sample is normalised to
before the chain runs - so a given number means the same depth below the peak
for every sample, whatever its recording level.

```yaml
process:
  - compress: true                                        # adapts to each sample
  - compress: { threshold: -30, ratio: 10, attack: 0.5 } # explicit - squash + raise tail
  - compress: { attack: 5 }                               # explicit attack, rest auto
  - limit: true                                           # brickwall at -1 dBFS
```

Set any parameter explicitly to override its auto value. Fixed parameters
(ratio, knee, makeup, lookahead) always use their defaults unless set.

Look-ahead has a cost worth knowing: the rendered variant is delayed by the
look-ahead window (prepended silence) and loses the same amount off its tail,
so `limit: true` (5 ms default) shifts a one-shot's attack ~5 ms later and
trims ~5 ms of decay. Set `lookahead: 0` to disable it for short percussive
one-shots where that delay matters. The gate uses the same pad-and-truncate
look-ahead.

The noise gate, distortion, and envelope reshaper follow the same pattern -
`true` gives you intelligent auto defaults, explicit parameters override:

```yaml
process:
  - gate: true                              # auto noise gate
  - gate: { threshold: -40, hold: 20 }      # explicit threshold
  - distort: true                            # hard-clip with auto drive
  - distort: { mode: fold, drive: 12 }       # foldback distortion
  - distort: { mode: bit_crush, bit_depth: 4, mix: 0.5 }
  - bit_depth: 12                            # clean 12-bit converter grit
  - bit_depth: { bits: 12, dither: true }    # smooth (triangular dither)
  - bit_depth: { bits: { cc: 70, min: 4, max: 16 } }   # bits on a knob
  - radio: { mode: am }                      # cleanest AM round-trip
  - radio: { mode: lw, signal: 0.4, static: 0.5 }      # weak, crackly longwave
  - radio: { mode: ssb, tune: 150, signal: 0.3 }       # mistuned voice in static
  - radio: { mode: fm, demod: ssb }          # transmit FM, receive WRONG -> warble
  - radio: { mode: am, fade: 0.6, stereo: stereo }     # shortwave swimming fade
  - freqshift: 1000                          # frequency shift up 1 kHz
  - wobble: { depth: 6, rate: 0.3 }          # slow tuning warble
  - reshape: true                            # auto tail-tightening
  - reshape: { attack: 5, release: 100 }     # fast attack, controlled release
  - reshape: { sustain: 0.5, release: 50 }   # half sustain, tight tail
  - transient: true                          # auto: normalises punch from crest factor
  - transient: { gain: 6 }                  # enhance transients by 6 dB
  - transient: { gain: -3 }                 # tame transients by 3 dB
  - pad_quantize: { tempo: 120, grid: 8 }   # silence-pad onsets to eighth-note grid
  - vocoder: { carrier: reference }           # cross-synthesise with this note's reference
  - vocoder: { carrier: samples/reference/GM36_BassDrum1.wav, bands: 16, depth: 0.8 }
```

`bit_depth` and `distort: { mode: bit_crush }` share the same quantizer, but
serve different ends: `bit_depth` is the converter itself - just the grid, with
silence staying silent and levels untouched - while bit_crush wraps that grid
in a distortion chain (drive, tone filter, level compensation, mix). Dither
decides what happens to material quieter than one quantization step: without
it (the default, and what vintage hardware did), fades and tails break up into
gritty, signal-correlated distortion; with `dither: true` (triangular) that
distortion becomes a smooth, constant hiss - including over silence, which is
the trade. Use `rectangular` for marginally less hiss at the cost of slight
noise pumping. For the full vintage-sampler character, pair the converter with
its low sample rate:

```yaml
process:
  - bit_depth: 12                            # the converter grain
  - distort: { mode: downsample, downsample_factor: 2, drive: 0, tone: 1.0 }
```

The `radio` effect is a complete radio link: it modulates the sample onto a
carrier, sends it through a channel, and demodulates it back - so the sample
returns sounding broadcast and received. Every stage is real radio physics
(AM/FM/SSB modulation, a diode envelope detector, an FM discriminator, a
mistunable SSB product detector), so the character is authentic rather than an
EQ-and-noise impression. Two things make it more than a lo-fi preset. First,
the receiver can be told to demodulate with the *wrong* scheme: set `demod`
different from `mode` and you hear what a real receiver does when tuned to the
wrong mode - `fm` into `ssb` is a musical pitch-tracking warble, while crossing
AM and FM is harsh noise (a combination that recovers nothing is muted rather
than amplified into hiss). Second, the impairments are injected before
demodulation, so the detector shapes them the way a real radio does: `signal`
weakens the carrier (FM gains its bright rising hiss and threshold crackle, and
an automatic gain control makes a fading station swim in rising static rather
than merely go quiet), `static` adds genuine atmospheric crackle (Poisson
lightning impulses, not a click loop), and `fade` adds the shortwave swimming
of selective fading. Stereo files collapse to authentic mono by default (real
broadcast is mono); `stereo: stereo` runs each channel as its own receiver
sharing one sky - independent hiss, but the same lightning crackle centred
across both.

```yaml
process:
  - radio: { mode: ssb, tune: 150, signal: 0.3, static: 0.3 }   # a voice on a crowded band
  - radio: { mode: fm, demod: ssb }                             # the wrong-mode warble
```

For the opposite of snappy drums (bring up room ambience and reverb tails), use
a fast attack (< 1 ms), high ratio (10:1+), and low threshold (-30 dB) to
squash transients and raise the relative level of the sustain/decay.

HPSS (Harmonic/Percussive Source Separation) decomposes audio into sustained
tonal content and transient clicks/hits. Useful as a pre-filter before repitch
(avoids pitch-shifting drum bleed) or stretch_quantize (cleaner grid alignment).

When `repitch` is in the process list, all notes in a multi-note assignment
share pick 1 (same sample, pitched per note). Without `repitch`, each note gets
the next rank — unless an explicit `pick` (scalar or range) is given, in which
case every note uses that same `pick` (range picks roll fresh per trigger).

#### Legacy `amount:` parameter (still accepted)

Four processors previously shared an `amount:` parameter with wildly different
units (dB for `saturate` and `transient`, 0-1 fraction for the two quantizers).
The parameter has been renamed per processor so the unit is obvious at the
call site. The old `amount:` key still works indefinitely - the parser
translates each one to the appropriate canonical name:

| Processor | Legacy | New (preferred) | Unit |
|---|---|---|---|
| `saturate` | `amount` | `drive` | dB |
| `transient` | `amount` | `gain` | dB (signed: +enhance, -tame) |
| `stretch_quantize` | `amount` | `strength` | 0.0-1.0 fraction |
| `pad_quantize` | `amount` | `strength` | 0.0-1.0 fraction |

Mixing both names on the same step is rejected at parse time.

The two separate HPSS processor names `hpss_harmonic: true` and
`hpss_percussive: true` have been unified into `hpss: { keep: harmonic }` /
`hpss: { keep: percussive }`. The legacy names still work and translate
internally.

`stretch_quantize` and `pad_quantize` now accept `tempo:` instead of `bpm:` -
matching the `tempo:` where-predicate. Legacy `bpm:` is translated to
`tempo:` at parse time.

The processor formerly named `beat_quantize` is now `stretch_quantize`: both
the new name and its companion `pad_quantize` describe *how* each quantizer
aligns onsets to a grid - one stretches audio in time, the other pads with
silence between segments. The legacy name `beat_quantize` still works -
the parser translates it to `stretch_quantize` at parse time, preserving any
params.

### Pan

The simple form is a single **position**, the way a mixer pan pot reads:
`-100` hard left, `0` centre, `+100` hard right.

```yaml
pan: 0       # centre (default)
pan: -100    # hard left
pan: -50     # halfway left
pan: 30      # a touch right
```

For surround layouts or asymmetric routing, `pan` also accepts a list of
per-channel weights. The values are **relative**, not percentages: only the
ratio between channels matters. `[50, 50]`, `[1, 1]`, and `[100, 100]` all
produce centre. Either form is normalised to constant-power gains at mix
time, so perceived loudness stays equal across pan positions.

```yaml
pan: [50, 50]    # centre (same as pan: 0)
pan: [100, 0]    # hard left (same as pan: -100)
pan: [75, 25]    # left of centre (same as pan: -50)
```

Channel order follows SMPTE: `[L, R]` for stereo; `[L, R, C, LFE, Ls, Rs]` for
5.1; `[L, R, C, LFE, BL, BR, SL, SR]` for 7.1. Set `player.audio.channels` in
config to match your output device (default: stereo). Samples of any channel
count are automatically mapped to the output layout using ITU-R BS.775 downmix
coefficients (surround to stereo, etc.) or conservative upmix (stereo to 5.1
uses front pair only). Pan weights define a target layout - if the output has
fewer channels, standard downmix is applied automatically.

#### Random pan

Instead of a fixed position, `pan` can draw a **fresh random position on every
note-on** - each strike lands somewhere new in the field and holds there for
that note. Three forms:

```yaml
pan: any                              # anywhere, hard left to hard right
pan: { gte: -50, lte: 50 }            # within a range (omit an end to open it)
pan: { position: -20, variation: 40 } # around a centre: -20, spread by ±20
```

Positions use the same `-100`…`100` axis as the fixed form, and the draw is
always **constant-power** - only the position is randomised, never the channel
levels, so a note is never louder or quieter for landing off-centre. It works on
any output layout (the stereo position up/downmixes just like a fixed pan), and
stacked layers on one note each draw independently. If you also set `output:` it
must list exactly two channels.

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

`player.audio.channels` defaults to stereo (2), **not** the device maximum -
opening every physical output for a stereo set would compute a needlessly wide
mix on every audio callback, and the count would vary by whatever device is
attached. At startup the player logs the device's detected maximum (`Output
device 'X' supports up to N channel(s); using M`); if you set `channels` higher
than the device offers, startup fails with a clear error naming the device's
real maximum rather than a cryptic PortAudio one.

### Velocity layering - multiple sounds per MIDI note

A single `(channel, note)` can host multiple assignments, each declaring a
distinct velocity range. The player picks the matching layer at note-on based
on the incoming velocity. Two common uses:

- **Velocity-switched libraries** — soft taps trigger the soft-recorded sample,
  hard hits trigger the hard-recorded one (standard pattern in drum and piano
  libraries since the 1980s).
- **Trigger multiplication** — turn a single MIDI key into two or more distinct
  triggers by splitting its velocity range. Trades velocity resolution for
  more sounds per pad.

#### List shortcut - filter only

```yaml
- name: Hard snare hit
  channel: 10
  notes: 38
  velocity: [100, 127]
  select:
    where:
      name: { matches: "*snare-hard*" }
```

This assignment fires only for velocities 100-127. The original velocity reaches
the gain calculation unchanged.

#### Dict form - filter with optional rescale

```yaml
- name: Soft hat
  channel: 10
  notes: 42
  velocity:
    trigger: [0, 63]
    rescale: true                # → output range [0, 127]
  select:
    where: { name: { matches: "*hat-soft*" } }

- name: Hard hat
  channel: 10
  notes: 42
  velocity:
    trigger: [64, 127]
    rescale: [10, 100]           # custom output range
  select:
    where: { name: { matches: "*hat-hard*" } }
```

`rescale: true` is shorthand for `[0, 127]`. With rescale on, each layer
plays through its own full dynamic envelope: a vel-30 input on the
0-63 layer is treated as if it were 60 out of 127, so the sample doesn't
sound permanently quiet just because the layer only sees the low half of
the velocity range.

Omit `rescale` (or set it to `false`) to keep the input velocity literal —
useful when each layer's sample is already calibrated for the velocity range
it covers, so rescaling would inflate the loudness inappropriately.

#### Validation

| Condition | Behaviour |
|---|---|
| Velocity field omitted | Default — single layer covering all velocities (no change from pre-layering) |
| Overlapping ranges on the same note | `ValueError` at load — overlap is almost always a typo, unless every overlapping assignment sets `stack: true` (see Stacking below) |
| Coverage gap (some velocities mapped to no layer) | `WARNING` listing the gap — velocities in the gap silently play nothing |
| `trigger`/`rescale` lo > hi, or out of `[0, 127]` | `ValueError` at load |
| Unknown inner key (e.g. `trggier` typo) | `ValueError` at load |

Layers maintain independent `round_robin` segment counters and independent
variant-transition fallback caches, so two layers on the same note never
interfere with each other's state.

### Stacking - multiple samples per trigger

Velocity layering puts different samples on *different* slices of one note's
velocity range. Stacking is the opposite: it plays several samples on the
*same* note and the *same* velocity, all sounding together as one composite
hit. Reach for it to fatten a sound that no single sample delivers - a kick
layered with a sub-sine for low-end weight, a snare doubled with a noise
burst for crack, or a clap stacked with its own reversed tail.

By default two assignments whose velocity ranges overlap are rejected at load,
because that is almost always a copy-paste slip. To stack them on purpose, set
`stack: true` on **every** assignment that shares the note - the flag is your
explicit "yes, I meant these to overlap":

```yaml
- name: Kick body
  channel: 10
  notes: 36
  stack: true
  select:
    where: { name: { matches: "*kick-body*" } }

- name: Sub sine
  channel: 10
  notes: 36
  stack: true
  output: [4]                    # send the sub to its own output if you like
  select:
    where: { name: { matches: "*sub-sine*" } }
```

Each stacked sample keeps its own settings - gain, pan, output routing,
processing, and `mode` - so you can shape and route the layers
independently, then a single note-off releases them together.

Stacking composes with velocity layering: stack two samples across the soft
half of a note's velocity and a different one across the loud half, and each
velocity still triggers exactly the samples you assigned to it.

A few things to keep in mind:

- **Mutual opt-in.** If only some of the overlapping assignments set
  `stack: true`, the map still fails to load - so a genuine mistake next to a
  deliberate stack is never silently swallowed.
- **Voices add up.** A four-deep stack plays four samples per note-on; stack
  with restraint if you are near your polyphony or CPU ceiling.
- **Zone-tuned can't stack.** A zone-tuned assignment already maps one sample
  per note, so `stack: true` there is rejected at load.

### Zone-tuned - auto-distribute pitched samples across a keyboard

Instead of writing one assignment per MIDI note, declare a single assignment
that filters the library and lets Subsample lay each matching pitched sample
across a contiguous slice of the keyboard centred on its detected pitch.
Zones meet at the midpoint between adjacent samples' pitches; the
lowest-pitched sample's zone extends down to the keyboard floor and the
highest extends up to the ceiling.

The classic use case: you've captured 30 tuned synth hits, or recorded a
shelf of pitched percussion, and want them all playable across a keyboard
without manually writing 30 assignments.

#### Shortcut form - full keyboard

```yaml
- name: Pitched library
  channel: 1
  notes: zone-tuned                # covers MIDI 0-127
  process:
    - repitch: true                # required
  select:
    where:
      duration: { gte: 0.5 }       # optional additional filters
```

`repitch` is **required** — each sample is pitch-shifted at note-on to the
note being played, with its declared pitch as the source. Without it, every
note would play the same sample at the same pitch, which is never what
zone-tuned is for.

`pitched: true` is **implicit** — Subsample filters via the same
`has_stable_pitch` gate used elsewhere (`pitch_confidence`, `pitch_stability`,
`harmonic_ratio`, etc.). Unpitched samples never sneak into a zone.

#### Dict form - restricted keyboard range and split keyboards

```yaml
- name: Bass zone
  channel: 1
  notes:
    mode: zone-tuned
    range: [0, 50]                 # MIDI ints OR note names ("C-1".."G9")
  process:
    - repitch: true
  select:
    where:
      name: { matches: "*bass*" }

- name: Lead zone
  channel: 1
  notes:
    mode: zone-tuned
    range: [51, 127]
  process:
    - repitch: true
  select:
    where:
      name: { matches: "*lead*" }
```

Multiple `zone-tuned` assignments on the same channel are allowed as long
as their keyboard ranges don't overlap.

#### Live re-derivation

Whenever the active library changes — a new sample captured, a watcher
import, a library eviction, a program switch, or a MIDI map reload —
Subsample re-derives the zones. There's no manual refresh step.

#### Validation

| Condition | Behaviour |
|---|---|
| Zone-tuned assignment without `process: [- repitch: true]` | `ValueError` at load |
| Regular `notes: 36` assignment on a channel already owned by a `zone-tuned` | `ValueError` at load |
| Two `zone-tuned` on the same channel whose `range:` spans overlap | `ValueError` at load |
| Unknown inner key under the dict form (e.g. typo `mdoe:`) | `ValueError` at load |
| No matching pitched samples for a template | INFO log; the channel plays nothing until a matching sample is added |
| Sample's detected pitch falls outside the template's `range:` | Sample excluded from that template; logged at DEBUG |

### Extract - present a multi-channel sample as a sub-pattern

`extract:` collapses a multi-channel sample to a 1-channel sub-signal at playback
time, emulating a named microphone pickup pattern. The source file is unchanged
- the same sample can be used in full stereo by another assignment. Extract
runs **before** `pan` and `output`, so the routing logic distributes the mono
extract across the chosen outputs as usual.

The classic use case is a kick: a stereo recording might have slight L/R
differences from room reflection or mic placement, but you usually want the
kick to land dead-centre and mono. `extract: omni` collapses the stereo source
to its `(L+R)/√2` sum and then `pan: [50, 50]` sends the mono signal equally to
both outputs.

```yaml
- name: Kick
  channel: 10
  notes: drum.kick_1
  select:
    where: { reference: samples/reference/GM36_BassDrum1.wav }
  extract: omni     # collapse to centred mono, distributed equally to all outputs
```

When `extract:` is set and `pan:` is omitted, the mono extract is distributed
equally across every output channel (constant-power) — the natural default for
a "collapsed to mono" signal. Explicit `pan:` still works as a per-output
weighting if you want something other than uniform.

The vocabulary is microphone-pattern names: every input format has a
canonical answer for `omni` (zero-order, equal-weight) and for each cardinal
first-order pickup pattern. The dispatch is automatic - the same YAML works
for mono, stereo, quad, 5.1, 7.1, and Ambisonic B-format inputs.

| value      | pattern                                | stereo (2ch)      | B-format AmbiX (4ch) |
|------------|----------------------------------------|-------------------|----------------------|
| `omni`     | equal-energy sum / W / M of M/S        | `(L+R)/√2`        | **W** only           |
| `side`     | left-right figure-eight                | `(L-R)/√2`        | **Y**                |
| `depth`    | front-back figure-eight                | rejected          | **X**                |
| `height`   | up-down figure-eight                   | rejected          | **Z**                |
| `left`     | left-facing cardioid                   | `L` only          | `(W+Y)/√2`           |
| `right`    | right-facing cardioid                  | `R` only          | `(W-Y)/√2`           |
| `front`    | forward cardioid                       | same as `omni` ⚠ | `(W+X)/√2`           |
| `back`     | rear cardioid                          | rejected          | `(W-X)/√2`           |
| `channel.N`| literal Nth input channel (1-indexed)  | -                 | -                    |
| `{blend: [w1, w2, …]}` | your own weighted sum to mono (signed, auto-normalised) | `w1·L + w2·R` | raw channels |

Surround (quad / 5.1 / 7.1) inputs are also supported; each pattern uses the
channels that carry the requested spatial information (e.g. `front` on 5.1
sums FL+FR+FC, normalised). `omni` on 5.1 / 7.1 **excludes LFE** because LFE
is band-limited and would dominate a full-range omni sum. Mono inputs are
treated as identity for `omni`, `left`, `right`, and `front` (no spatial
information to collapse).

⚠ `front` on stereo input has no front-back information to discriminate, so
it reduces to the omni matrix. The map still loads, but a warning is logged.

**Parse-time validation.** Patterns that have no meaningful definition for
the input format are **rejected at map load** (before any audio plays). For
example, `extract: depth` on a stereo reference raises a `ValueError` naming
the assignment and the offending sample; you fix the map or change the
extract.

The `channel.N` form is the escape hatch when you really do want a literal
channel pick (e.g. `extract: channel.2` for the second input channel only).
N is 1-indexed and rejected if it exceeds the input's channel count.

**Blending two mics into mono.** `extract: { blend: [w1, w2, ...] }` sums the
input channels with your own weights - one per channel - instead of a fixed
pattern. The weights are auto-normalised so their magnitudes add up to 1, which
means changing the balance never changes the output level: you can dial the mix
without re-touching your gain. A **negative** weight flips that channel's
polarity.

The reason this exists is a snare recorded with two mics - one above, one below -
onto two channels of a single file. The two capsules face each other, so they
are roughly 180° out of phase; summing them naively (`omni`) thins the body of
the drum. Flip the bottom mic with a negative weight and set the balance to
taste - more of the bottom mic brings up the snare wires:

```yaml
- name: Snare
  channel: 10
  notes: drum.snare_1
  select:
    where: { directory: snare }
  extract: { blend: [0.7, -0.3] }   # 70% top mic + 30% bottom, bottom polarity flipped
```

Write the weights at any scale - `[7, -3]` and `[0.7, -0.3]` mean the same thing.
The number of weights must match the sample's channel count, checked at map load.

---

**Going further.** The sections that follow cover the optional advanced
features: multichannel/ambisonic capture, program switching for live kit swaps
via MIDI Program Change, and MIDI CC control for any numeric processor
parameter. None of this is needed for a basic setup - skip ahead if you're
just building a drum kit or pitched keyboard.

---

### Ambisonic capture

Four-capsule tetrahedral mics (such as the Rode NT-SF1) and pre-encoded
B-format files are supported as first-order ambisonic content. Samples are
converted to canonical AmbiX B-format (channel order W, Y, Z, X; SN3D) at
capture time and decoded at playback time through a virtual speaker array
sized to match `player.audio.channels` (mono, stereo, quad, 5.1, or 7.1).

Enable ambisonic capture in `config.yaml`:

```yaml
recorder:
  audio:
    channels: 4
    ambisonic_format: a_nt_sf1   # or a_generic, b_fuma, b_ambix

ambisonic:
  decoder: basic                  # basic | max_re | inphase
  yaw_degrees: 0.0                # rotate before decoding
  pitch_degrees: 0.0
  roll_degrees: 0.0
```

Format options:

- `a_nt_sf1` - Rode NT-SF1 A-format. Applies a capsule-matching HF shelf
  pre-matrix and a post-matrix HF shelf on X/Y/Z to compensate for
  capsule-spacing loss. Best choice for this mic.
- `a_generic` - Generic tetrahedral A-format with the standard Gerzon
  matrix, capsule order FLU/FRD/BLD/BRU. No capsule calibration applied.
- `b_fuma` - Pre-encoded B-format in FuMA order (W, X, Y, Z), MaxN.
  Reordered and renormalised to AmbiX on read.
- `b_ambix` - Pre-encoded B-format already in AmbiX order - stored
  unchanged.

Decoder choice affects the spatial character: `basic` has sharp lobes and
the best low-frequency behaviour, `max_re` trades some front-energy for
tighter localisation in the sweet spot, and `inphase` has the softest
lobes and works best when listening from off-axis positions. Rotation
(yaw/pitch/roll) is applied before the decoder and is project-wide - all
ambisonic samples rotate together.

Analysis runs on the W (omnidirectional) channel only, so spectral and
rhythmic fingerprints reflect the sound-field sum rather than a
directionally biased mix. Pad-quantize and beat-quantize work on
ambisonic samples using Rubber Band's phase-coherent multichannel engine
- inter-channel relationships survive time-stretching within tolerance.

### Programs - switching instrument sets via MIDI

The MIDI map can optionally declare multiple **programs** (presets) that are all
loaded at startup. Switch between them at runtime using MIDI Program Change
messages - no restart, no disk I/O, instant switching. A `program:` number may
also be a name from a mounted definitions file (`program: my.brushes` - see
[Naming your own sounds](#naming-your-own-sounds---the-definitions-file)):

```yaml
programs:
  - name: "Acoustic Kit"
    directory: samples/acoustic
    program: 0
  - name: "Electronic Kit"
    directory: samples/electronic
    program: 1

program_channel: 10    # MIDI channel for Program Change messages (1-16, or 0 = any)
default_program: 0     # program number to activate at startup (default: first in list)
```

Each program entry has a `name`, an optional `program` number (0-127, defaults to
its list position), and **exactly one** of two source forms:

- **`directory:`** - swap the sample pool only. The program reuses the map's
  top-level `assignments:` and just points them at a different folder. Use this
  when the same routing rules should evaluate against different samples (the
  `directory:` example above).
- **`map:`** - a full **preset**: a whole mapper file with its own `assignments:`
  *and* its own samples. A Program Change to this program swaps the routing rules
  *and* the sample pool together - the MIDI-true meaning of a program change
  (like selecting a different drum kit on a GM device).

Each entry is selected by its `program` number (0-127), addressed by a MIDI
**Program Change** message. This is deliberately *not* the MIDI **Bank Select**
mechanism (CC 0 / CC 32), which subsample does not implement - so you address up
to 128 programs by Program Change number alone. The terminology mirrors that:
what a synth calls a "program" (one Program Change slot) is one entry here; the
MIDI term "bank" (a group of 128 programs reached via Bank Select) has no role.

When `programs:` is absent, the single `library.directory` from config.yaml is
used as before - you do not need a `programs:` block for a single instrument set.
When present, it overrides `library.directory`. Each program gets its own
sample library, similarity index, and transform cache.

With the `directory:` form, assignments are program-agnostic - the same top-level
rules query whichever program is active. Named samples (`where: { name: X }`) that
only exist in one program silently produce no match in others; rule-based selects
(`reference:`, `pitched:`, etc.) work naturally against whatever samples are present.

#### Programs as presets - the `map:` form

A `map:` program is a complete, self-contained preset:

```yaml
programs:
  - name: "Acoustic Kit"
    program: 0
    map: kits/acoustic/midi-map.yaml      # full preset - own assignments + own samples
  - name: "Electronic Kit"
    program: 1
    map: kits/electronic/midi-map.yaml

program_channel: 10
default_program: 0
# no top-level `assignments:` needed when every program is a `map:` preset
```

- The `map:` path is resolved **relative to the parent map's directory**.
  A `directory:` program, by contrast, resolves **relative to the working
  directory you launch subsample from** (like `library.directory`) - if a
  kit works from the project root but not elsewhere, this asymmetry is why.
- The preset is an ordinary mapper file with its own `assignments:`. Its samples
  come from its own `where: { directory: ... }` predicates and path references,
  which resolve **relative to the preset's own folder** - so a self-contained kit
  directory (`kits/acoustic/{midi-map.yaml, Kick/*.wav, Snare/*.wav, ...}`) works
  as a drop-in unit.
- A Program Change swaps **both** the rules and the pool atomically. A broken
  preset is rolled back and logged rather than stopping playback.
- Presets are **flat**: a preset may not declare its own `programs:` block.
- The top-level `assignments:` block is **optional** when every program is a
  `map:` preset, but **required** if any program uses the `directory:` form (those
  reuse the top-level assignments).
- You can **mix** `map:` and `directory:` programs in one map.

`map:` and `directory:` programs are eager-loaded at startup so switching stays
instant. If a program's samples don't fit within `library.max_memory_mb`, a
startup warning notes that switching to it may lag (samples reload from disk).

> **Editing a preset:** the file watcher follows only the top-level map. Editing a
> `map:` preset's own file - or changing the `programs:` / `program_channel:` /
> `default_program:` settings - requires a **restart** to take effect.

#### Programs vs directory predicate

`programs:` and `where: { directory: ... }` both load samples from a directory,
but they solve different problems:

- **Programs** swap the entire sample pool at once. Only one program is active at
  a time - a MIDI Program Change switches all assignments to a new set of samples.
  Use programs when you want the same MIDI map rules to evaluate against completely
  different sample collections (e.g. "Acoustic Kit" vs "Electronic Kit").

- **`where: { directory: ... }`** filters within the active pool. It is
  per-assignment, and multiple assignments can each reference a different
  directory simultaneously. Use it when different notes in the same map need
  samples from different directories at the same time (e.g. kicks from one
  folder, hi-hats from another).

| | Programs | `where: { directory }` |
|---|---|---|
| Scope | All assignments share one active program | Per-assignment filter |
| Switching | MIDI Program Change swaps the whole pool | Always active |
| Simultaneous directories | No (one program at a time) | Yes (each assignment can use a different directory) |
| Use case | Swap entire kits | Mix sources within one kit |

### CC mapping - real-time parameter control

Any numeric processor parameter can be controlled by a MIDI CC message.
Replace the scalar value with a CC binding (`cc:` and `channel:` also accept
names from a mounted definitions file, e.g. `cc: my.brightness` - see
[Naming your own sounds](#naming-your-own-sounds---the-definitions-file)):

```yaml
process:
  - pad_quantize: { grid: 16, strength: { cc: 1 } }         # bare CC binding
  - filter_low: { freq: { cc: 74, min: 200, max: 16000 } }  # ranged CC binding
# stretch_quantize's tempo binds the same way - but a chain may hold only one
# quantizer, so use it in place of pad_quantize, not alongside it:
#   - stretch_quantize: { tempo: { cc: 2, min: 60, max: 180 }, grid: 16 }
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

### Vocabulary reference

Every enum-string value the MIDI map accepts, in one place:

| Where | Valid values |
|---|---|
| `where` operators | `gte` `lte` `gt` `lt` `eq` |
| Order `dir` | `asc` `desc` |
| Order `by` | `age` `duration` `pitch` `onsets` `tempo` `level` `quantized_beats` `similarity` `beat_match` |
| `notes` range | `<low>..<high>` (e.g. `C2..C4` or `36..60`) |
| `pitch` predicate value | Hz float (`440`) or note name (`A4`, `C#3`, `Db5`) |
| `distort` `mode` | `hard_clip` `fold` `bit_crush` `downsample` |
| `bit_depth` `dither` | `true` (= `triangular`) `false` `none` `triangular` `rectangular` |
| `radio` `mode` | `am` `lw` `fm` `ssb` |
| `radio` `demod` | `matched` `am` `fm` `ssb` |
| `radio` `stereo` | `mono` `stereo` |
| Quantize `segment` | `round_robin` `random` or integer (1-indexed) |
| `vocoder` `carrier` | `reference` (the note's reference sample) or a file path |
| `silenced_by` value | `self`, or a note / list of notes (same syntax as `notes`) |
| `definitions` file sections | `notes` (0-127) `cc` (0-127) `channels` (1-16) `programs` (0-127); other sections ignored |
| Legacy `order_by` tokens | `newest` `oldest` `duration_asc` `duration_desc` `pitch_asc` `pitch_desc` `onsets_asc` `onsets_desc` `tempo_asc` `tempo_desc` `loudest` `quietest` `similarity` `quantized_beats_asc` `quantized_beats_desc` |
| Legacy numeric-predicate keys | `min_duration` `max_duration` `min_onsets` `max_onsets` `min_tempo` `max_tempo` `min_pitch` `max_pitch` `min_quantized_beats` `max_quantized_beats` |
| Legacy processor names | `beat_quantize` (→ `stretch_quantize`) `hpss_harmonic` (→ `hpss: { keep: harmonic }`) `hpss_percussive` (→ `hpss: { keep: percussive }`) |
| Legacy processor param names | `amount` (→ `drive` / `gain` / `strength` per processor) `bpm` (→ `tempo` in quantizers) |

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

### MIDI dispatch model

Incoming MIDI is dispatched in callback mode: rtmidi delivers each message
to subsample's handler on its own dedicated thread the moment it arrives.
There is no polling loop, so there is no fixed input-latency floor. The
end-to-end path that a player feels as latency is three parts:

1. **MIDI dispatch** — under 1 ms (rtmidi callback).
2. **Per-note handling** — well under 1 ms (sample selection is pre-computed
   when the library changes, so a trigger is an indexed pick, not a query).
3. **Output latency** — the buffer→DAC floor. This is the dominant term, and
   it's what you hear as a uniform delay against a hardware instrument.

The output buffer period is set via `player.audio.buffer_frames`:

| frames | buffer period at 44.1 kHz | at 48 kHz |
|--------|--------------------------:|----------:|
| 128    | 2.9 ms                    | 2.7 ms    |
| 256    | 5.8 ms                    | 5.3 ms    |
| 512    | 11.6 ms                   | 10.7 ms   |
| 1024   | 23.2 ms                   | 21.3 ms   |

These are the *period*, not the total latency: PortAudio's ALSA backend keeps
several periods in flight, so actual output latency is a small multiple of the
figure above. The real number your device negotiated is printed at startup:

```
PortAudio output latency: 11.6 ms
```

That line is ground truth — tune against it, not the table. Lower
`buffer_frames` (e.g. 128, 64) to shrink it; the player logs an `Audio xrun`
warning if the buffer is too low for your machine to sustain (audible as
clicks), so reduce until you see those, then step back up. Leave it unset to
let the OS pick a safe default.

You can measure the two software terms on your hardware with the included
scripts — MIDI dispatch:

```bash
python scripts/measure_midi_latency.py --count 1000
```

and per-note handling cost (selection + variant lookup + render):

```bash
python scripts/measure_handler_timing.py
```

Both should report well under 1 ms; if they don't, that's a bug, not a buffer
setting.

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
in a separate worker pool, so back-to-back sounds are captured reliably even
when spectral analysis is slow — critical for USB audio devices, which use
isochronous transfers and are sensitive to timing jitter. The pool right-sizes
itself to the moment: while the player is live it keeps to a small share of the
cores so the real-time audio thread is never starved, while the one-off library
scan at startup — before anything is playing — spreads across most of the cores
in separate processes (leaving some headroom for the system), building a large
library several times faster on a multi-core machine.

### Leaves you headroom

Background analysis deliberately runs on about three-quarters of the machine's
cores rather than every last thread, so your system stays responsive while a
large library rebuild or `import` works away in the background. It pulls back
much further while the player is live, so playback is never starved. The share
is measured against the cores this process may actually use, so it does the
right thing inside a container or under a CPU allowance rather than sizing
itself to the whole host.

Note that fewer workers does not mean a cooler machine: a modern CPU draws to
its power limit whether the work is spread across four cores or forty, so the
package temperature lands in much the same place either way. What you gain is
headroom — the machine stays usable while it works.

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
time); oldest files are evicted when the disk budget is exceeded. Quantized
variants also store a grid energy profile - per-grid-slot RMS energy normalized
to [0, 1] - alongside the audio; the `beat_match` order scorer compares against it.

Samples with detected rhythmic content can be time-stretched to a target tempo
using the `stretch_quantize` processor in a MIDI map assignment. Detected attacks are
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
# Install system dependencies. On Linux, PyAudio and python-rtmidi are compiled
# from source, so they need a build toolchain and the PortAudio + ALSA/JACK
# development headers; pyrubberband needs the Rubber Band CLI. A minimal install
# (e.g. Ubuntu Server) has none of these by default.
# Debian/Ubuntu:
sudo apt install build-essential pkg-config portaudio19-dev libasound2-dev libjack-jackd2-dev rubberband-cli
# Fedora/RHEL:
sudo dnf install gcc-c++ make pkgconf-pkg-config portaudio-devel alsa-lib-devel jack-audio-connection-kit-devel rubberband
# macOS (CoreAudio/CoreMIDI are built in; Xcode command-line tools provide the compiler):
brew install portaudio rubberband

# Linux only: Subsample plays/records through PortAudio, whose JACK backend must
# reach a running sound server at startup - otherwise PyAudio aborts with
# `OSError: [Errno -9999] Unanticipated host error`, even if you never use JACK.
# Desktop Linux already runs PipeWire; on a minimal/headless install (e.g.
# Ubuntu Server) there is no audio server, so install and start one (PipeWire
# recommended - it also provides the JACK layer PortAudio needs):
sudo apt install pipewire pipewire-alsa pipewire-jack pipewire-pulse wireplumber
systemctl --user enable --now pipewire pipewire-pulse wireplumber
# Then launch Subsample under pw-jack (`pw-jack subsample`) for a clean start.
# Plain `subsample` also works once PipeWire runs, but PortAudio's JACK probe
# logs harmless `jack server is not running` warnings.

# Install (directly from GitHub, or from a local clone with `pip install .`)
pip install git+https://github.com/simonholliday/subsample.git

# Create a project: a documented config, the ready-to-play GM drum kit map,
# an editable map template, and the GM reference data the kit matches against
mkdir my-project && cd my-project
subsample --init

# Start capturing - every distinct sound lands in samples/captures/
subsample

# Or feed it existing recordings instead of a microphone
subsample recording.wav                # chop one file into samples
subsample ./recordings/*.wav           # or many (glob expansion)
```

Subsample also runs with no project at all - built-in defaults cover
everything, and a hand-written `config.yaml` containing only the settings you
want to override works the same way (everything else is inherited
automatically; see [Configuration](#configuration)).

### First sound - from install to a playable kit

1. **Get samples into the library.** Run `subsample` and make sounds at the
   microphone - each distinct sound is captured, trimmed, and analysed into
   `samples/captures/` - or bring existing sounds:
   `subsample import ~/my-pack/*.wav` (trims, fades, and analyses them in),
   or simply copy audio files (WAV, FLAC, AIFF, OGG, MP3) into
   `samples/captures/`; everything there loads at startup.
2. **Turn the player on.** In `config.yaml`, set `player.enabled: true` and
   your devices: `player.audio.device` (output), and `player.midi_device`
   (or `player.virtual_midi_port` if a sequencer on the same machine will
   drive it). The GM kit map is already wired in by `--init`.
3. **Play channel 10.** Run `subsample` again and play your controller or
   sequencer on MIDI channel 10 - kick on 36, snare on 38, hi-hats on 42/46.
   Every GM drum note plays whichever of your samples sounds closest to that
   drum, through a pre-mixed channel strip. Load more samples and the kit
   improves.

**Live capture mode:** Subsample lists available audio input devices and lets you
choose one (or auto-selects if only one is present). It calibrates ambient noise
for a few seconds before listening for events.

**File input mode:** Each file is processed at its native sample rate, bit depth,
and channel count. Detected segments are saved to the output directory. The
detector spends the first `detection.warmup_seconds` (default 1.0s) calibrating
the ambient floor, so a hit inside that opening window is not captured - leave
about a second of room tone at the start of an imported file. Lowering
`detection.warmup_seconds` shortens that window but cannot remove it: at least
one chunk always goes to calibration, and if that chunk contains the hit it also
seeds the ambient floor from it. For a file that begins exactly on a transient,
pad the front with a moment of silence instead.

### Chopping long-decay takes

By default a recording starts when the level rises above the ambient floor by
`detection.threshold_db` and ends when it falls back below that same
threshold. That works for short, well-separated hits, but it cuts a long decay
short - a ride cymbal or gong ringing down over many seconds is chopped off the
moment it drops near (but not to) the noise floor, often 15 dB above silence.

To capture the full tail, record one hit at a time - strike, let it ring down to
silence, strike again - and give the recorder a separate, lower threshold for the
*end* of each hit:

```yaml
detection:
  threshold_db: 10.5         # START: still fires only on the loud attack
  release_threshold_db: 4.0      # END: let the tail ring out to ~4 dB over the floor
  retrigger_threshold_db: 12.0   # or end the moment the next hit lands, whichever comes first
  fade_out_ms: 30                # a smooth fade so the tail never clicks
```

`release_threshold_db` decouples the end from the start: the recording still
begins on the loud attack, but only ends once the tail has decayed to the quieter
release level, so the whole ring-down is preserved. `retrigger_threshold_db` is
the safety net - if a tail is still audible over background noise when you play
the next hit, the rising attack ends the current sample and opens the next one,
so hits are never merged. Together they mean each sample ends on *silence or the
next hit, whichever comes first*. Both default to off, preserving the original
single-threshold behaviour.

`retrigger_threshold_db` assumes each hit's attack lands quickly - within
`max(hold_seconds, 0.1s)`. That fits drums, cymbals, and other fast-attack
percussion. A sound with a slow or two-stage attack (a soft onset then a louder
transient a moment later - breath before a flute note, mallet contact before a
bowl) can read its own transient as a new hit and split one gesture into two; if
that happens, raise `hold_seconds` so it spans the attack.

### Rejecting recordings of nothing

Every threshold above is **relative** - "louder than the room by N dB". That is
what lets detection work in any room without recalibration, but it also means
detection cannot, on its own, tell a quiet room's noise from a quiet sound.

A room's background level is not steady; it wobbles by several dB moment to
moment. So in a *quiet* room - a good preamp and a noise floor near -100 dBFS -
a small `threshold_db` can be cleared by nothing but air, and a small
`retrigger_threshold_db` can fire on the noise a tail has decayed into. You get
a file containing no sound at all, or a real hit with a silent head in front of
it. Counter-intuitively, the quieter the room, the easier this is to provoke.

Two settings guard against it:

```yaml
detection:
  threshold_db: 12.0             # comfortably above the room's own wobble
  retrigger_threshold_db: 15.0   # a real strike, not a fluctuation in the tail
  min_peak_db: -45.0             # and throw away anything that peaks below this
```

Keep the relative thresholds above the background's natural variation - that is
the actual fix, and `12` / `15` are good starting points. A value near `9` sits
*inside* the wobble of a quiet room.

`min_peak_db` is then the absolute backstop for whatever still slips through:
a finished recording peaking below this level is discarded rather than saved,
analysed and added to your library. Judge it against the peak of your quietest
**wanted** hit and leave headroom - set it too high and you will silently lose
real material. For close-miked percussion, -50 to -40 dBFS is typical. It is off
by default, because the right value depends entirely on how you are recording.

If real hits go missing, `min_peak_db` is the first thing to lower or unset.

## Configuration

Subsample ships its defaults built in and deep-merges your `config.yaml` on
top. Your config only needs the settings you want to change - everything else
is inherited from the defaults automatically. `subsample --init` writes a
starter `config.yaml` with every setting present and documented, ready to
edit.

### Where your config lives

Subsample looks for `./config.yaml` in the directory you run it from, or
takes an explicit path with `--config`:

```bash
cd my-project && subsample            # uses my-project/config.yaml
subsample --config ../config.yaml     # an explicit file, wherever you keep it
```

The directory you run from is the project folder: relative paths in the
config - `player.midi_map`, `library.directory`, `recorder.directory` - all
resolve against it, whichever config file is in use. That supports both ways
of laying out a multi-track project:

- **A config per track:** each track folder holds its own `config.yaml`,
  MIDI map, and samples; `cd track-01 && subsample`.
- **One shared config:** a single `config.yaml` at the project top;
  `cd track-01 && subsample --config ../config.yaml`. Shared settings come
  from the one file, while the map, samples, and recordings still belong to
  the track folder you ran from.

Paths *inside* a MIDI map - sample directories, `definitions:` files - are
the exception: they resolve relative to the map file itself, so a map and its
sounds travel together (see [MIDI map](#midi-map)).

At startup Subsample logs which configuration it is using; if you see
`built-in defaults (no config.yaml in ...)` you are running without your
config, usually because you launched from a different directory.

The most common overrides:

- **First run:** set `recorder.audio.device` (your microphone) and `recorder.directory`
- **For MIDI playback:** set `player.enabled: true`, `player.midi_device` or `player.virtual_midi_port`, and `player.audio.device`
- **If startup stops with `Cannot load the MIDI map`:** with the player enabled, a map that is missing or invalid is fatal - it is reported in the first second rather than after the sample library has loaded. `player.midi_map` resolves from the directory you launched in, and the logged absolute path shows exactly where it looked; only the maps that ship with Subsample live under `subsample/data/`
- **If you hear clipping:** raise `player.max_polyphony`; the `limiter_threshold_db` and `limiter_ceiling_db` defaults protect against distortion automatically
- **If recordings miss quiet sounds:** lower `detection.threshold_db`
- **If recordings trigger on noise, or arrive empty:** raise `detection.threshold_db` and `detection.retrigger_threshold_db`, then set `detection.min_peak_db` as a backstop - see [Rejecting recordings of nothing](#rejecting-recordings-of-nothing)

Everything else - chunk sizes, buffer lengths, transform settings, similarity
weights - is optional and rarely needs changing.

| Setting | Default | Description |
|---|---|---|
| `max_memory_mb` | auto | Total cache memory budget. Auto-detect: min(25% of system RAM, 1024 MB). Split: 60% instruments, 35% transforms, 5% carrier |
| `recorder.enabled` | `true` | Enable live audio capture; set to `false` to process files only |
| `recorder.audio.device` | `none` | Audio input device name (substring match); if unset, auto-select or prompt. `subsample --list-devices` shows the names |
| `recorder.audio.sample_rate` | `44100` | Sample rate in Hz |
| `recorder.audio.bit_depth` | `16` | Bit depth (16, 24, or 32) |
| `recorder.audio.channels` | auto | 1 = mono, 2 = stereo. Omitted by default (and `null`) = auto-detect from the selected device |
| `recorder.audio.input` | `null` | Physical input channels (1-indexed list). `[3, 4]` records from inputs 3-4 |
| `recorder.audio.buffer_frames` | `512` | Frames per buffer read |
| `recorder.audio.audio_format` | `wav` | Output container: `wav` (uncompressed, 16/24/32-bit) or `flac` (lossless compressed, ~40-60% smaller, 16/24-bit). See [Storage format](#storage-format) for behaviour around mixed bit depths |
| `recorder.audio.float_import_ceiling_dbfs` | `-1.0` | Ceiling (dBFS) for 32-bit float / 64-bit double audio, wherever it is read - CLI file-input (`subsample <file>`), `directory:`/`path:` loads, OSC import, and the watcher. Peaks above full scale (these formats have no hard 0 dBFS limit) are scaled down to fit instead of hard-clipping on the way in. Whole-file, so dynamics are preserved; inaudible, since playback re-normalises. Integer sources untouched. `null` clips as before (this was the behaviour in prior versions). A hot float sample analysed before this existed can play a decibel or two quiet until re-analysed - delete its `.analysis.json` sidecar to refresh it |
| `recorder.previews` | `true` | Emit a `.preview.png` thumbnail sidecar (1024x256, ~15-25 KB) and embed a compact `preview` data block in `.analysis.json` so the Supervisor dashboard can render a scalable SVG on demand. See [Sample previews](#sample-previews) |
| `recorder.buffer.max_seconds` | `60` | Circular buffer length |
| `player.enabled` | `false` | Enable the MIDI player |
| `player.midi_map` | `none` | Path to MIDI routing map YAML; required for player. Use `midi-map-gm-drums.yaml` for a complete GM kit |
| `player.max_polyphony` | `8` | Headroom divisor, not a voice cap: per-voice gain = 1/max\_polyphony, so this many voices at full velocity sum to full scale. Voices are never cut off. Raise if clipping; lower for louder individual voices |
| `player.limiter_threshold_db` | `-1.5` | Safety limiter threshold (dBFS); signals below this pass untouched. `0.0` disables the limiter |
| `player.limiter_ceiling_db` | `-0.1` | Maximum output level (dBFS) the limiter allows; must exceed threshold (ignored when the limiter is disabled) |
| `player.midi_device` | `none` | MIDI input device name (substring match); if unset, auto-select or prompt. `subsample --list-devices` shows the names |
| `player.audio.device` | `none` | Audio output device name for playback |
| `player.audio.sample_rate` | auto | Output sample rate; defaults to recorder rate. Do not set higher than source. |
| `player.audio.bit_depth` | auto | Output bit depth (16, 24, or 32); defaults to recorder bit depth |
| `player.audio.channels` | `null` | Output channels (2=stereo, 6=5.1, 8=7.1); null defaults to stereo. SMPTE ordering |
| `player.audio.buffer_frames` | `null` | PortAudio output buffer in frames (power of two, 32-4096); `null` (default) lets the OS pick. Smaller → lower latency; larger → safer under load. See the **MIDI dispatch model** section for the latency table |
| `player.virtual_midi_port` | `none` | Name for a virtual MIDI input port; overrides `player.midi_device` |
| `player.watch_midi_map` | `false` | Monitor the `midi_map` file for changes and reload assignments on save (see Live-coding) |
| `player.strict_midi_map` | `true` | Reject unknown `where:` keys, unknown processor names, and non-bool `pitched:` values at parse time. Set to `false` to silently ignore unknown keys when loading older or hand-edited MIDI maps |
| `detection.threshold_db` | `12.0` | dB above ambient to trigger recording |
| `detection.hold_seconds` | `0.5` | Seconds to hold recording open after signal drops |
| `detection.warmup_seconds` | `1.0` | Calibration period before detection activates |
| `detection.floor_adaptation` | `0.1` | How quickly the tracked room level follows changes, 0-1. Higher = only sudden-onset sounds trigger; lower = gradual builds trigger too |
| `detection.trim_pre_ms` | `0.25` | Milliseconds of audio kept before the detected onset, with a smooth fade-in so the start never clicks |
| `detection.trim_post_ms` | `2.0` | Milliseconds of audio kept after the detected end, with a smooth fade-out so the stop never clicks |
| `detection.release_threshold_db` | `null` | Separate CLOSE threshold (dB above ambient) for ending a recording. Lower than `threshold_db` so a long decay rings out toward silence instead of being cut short; the recording still starts on the louder attack. `null` reuses `threshold_db`. See [Chopping long-decay takes](#chopping-long-decay-takes) |
| `detection.retrigger_threshold_db` | `null` | While recording, a rise of this many dB over the decaying tail ends the current sample and starts the next - so spaced hits whose tails never fully fade are still separated. `null` disables. Pair with `release_threshold_db`. Assumes each attack lands within `max(hold_seconds, 0.1s)`; raise `hold_seconds` for slow/two-stage attacks or they over-split |
| `detection.fade_out_ms` | `0.0` | Smooth fade (ms) on each sample's trailing edge, masking a cut taken mid-decay or at the next hit. `0.0` keeps the ~2 ms declick |
| `detection.min_peak_db` | `null` | Absolute floor (dBFS) a recording's peak must reach to be kept - the backstop against a trigger on room tone, since every other threshold is relative to the room. Judge it against your quietest wanted hit. `null` disables. See [Rejecting recordings of nothing](#rejecting-recordings-of-nothing) |
| `recorder.directory` | `samples/captures` | Where recordings are saved (also the default `library.directory`, so captures join the playable library) |
| `recorder.filename_format` | `%Y-%m-%d_%H-%M-%S-%3f` | strftime format for filenames (`%3f` = 3-digit milliseconds) |
| `analysis.start_bpm` | `120.0` | Tempo prior for beat detection (BPM) |
| `analysis.tempo_min` | `30.0` | Minimum tempo considered by pulse detector (BPM) |
| `analysis.tempo_max` | `300.0` | Maximum tempo considered by pulse detector (BPM) |
| `library.max_memory_mb` | auto | Max audio memory for in-memory samples; overrides global split. Oldest evicted (FIFO) |
| `library.directory` | `samples/captures` | Root directory of instrument samples — walked recursively, so samples can be organised into subdirectories (`kicks/`, `snares/`, …) however suits the user. Overridden by `programs:` in the MIDI map when present. Missing `.analysis.json` and `.preview.png` sidecars are regenerated at startup; orphaned ones (no matching audio) are deleted |
| `library.watch` | `false` | Monitor `library.directory` (or each program directory) at runtime for new audio files from any source - another Subsample instance, a DAW, or any application that writes audio. Watches the top level of each directory only - drop new files there, not into subdirectories (see Watching for new samples) |
| `similarity.weight_spectral` | `1.0` | Weight for the spectral shape group (14 metrics) |
| `similarity.weight_timbre` | `1.0` | Weight for sustained MFCC timbre (coefficients 1-12) |
| `similarity.weight_timbre_delta` | `0.5` | Weight for delta-MFCC timbre trajectory |
| `similarity.weight_timbre_onset` | `1.0` | Weight for onset-weighted MFCC attack character |
| `similarity.weight_band_energy` | `1.0` | Weight for the band energy group (4 per-band energy fractions + 4 decay rates) |
| `transform.max_memory_mb` | auto | Memory budget (MB) for transform variants; overrides global split |
| `transform.auto_pitch` | `true` | Pre-compute pitch variants for every MIDI note in the assigned range. Requires `rubberband-cli`. Disable if rubberband is unavailable or you prefer on-the-fly rendering (pitch still works, higher CPU at trigger time) |
| `tempo.bpm` | `0.0` | Session tempo (BPM). Default for `stretch_quantize` / `pad_quantize` steps that carry no explicit `tempo:`, and the reference for the `duration_beats` selection filter. Only assignments that declare a quantize processor are quantized - there is no automatic stretch-every-rhythmic-sample behaviour. `0.0` means unset: such a quantize step is skipped, and a map that filters by `duration_beats` will not load. Also the fallback under `tempo.source: midi` until a MIDI clock arrives |
| `tempo.source` | `config` | Where the session tempo comes from. `config` uses `tempo.bpm`; `midi` follows an incoming MIDI clock so changing tempo in your sequencer doesn't leave quantized samples or beat filters on the old grid. See [Following your sequencer's tempo](#following-your-sequencers-tempo) |
| `transform.quantize_resolution` | `16` | Grid subdivision for time-stretch onset alignment: 1 (whole), 2 (half), 4 (quarter), 8 (eighth), 16 (sixteenth) |
| `transform.variant_cache_dir` | `samples/variant-cache` | Directory for persistent disk cache of transform variants. Empty string or null disables |
| `transform.max_disk_mb` | auto | Max disk space (MB) for cached variant files; defaults to 3x memory budget. 0 disables |
| `supervisor.enabled` | `false` | Enable the Supervisor web dashboard (broadcasts state via WebSocket for live monitoring). Requires `pip install "subsample[supervisor] @ git+https://github.com/simonholliday/subsample.git"` |
| `supervisor.port` | `9003` | WebSocket port the Supervisor server listens on |
| `osc.enabled` | `false` | Enable OSC integration (send sample events, optionally receive import requests). Requires `pip install "subsample[osc] @ git+https://github.com/simonholliday/subsample.git"` |
| `osc.send_host` | `127.0.0.1` | Destination host for outgoing `/sample/captured` and `/sample/loaded` messages |
| `osc.send_port` | `9000` | Destination UDP port for outgoing OSC messages |
| `osc.receive_enabled` | `false` | Listen for `/sample/import` messages to load audio files into the in-memory library from other apps (reads in place, does not copy) |
| `osc.receive_port` | `9002` | UDP port the OSC receiver listens on |
| `osc.receive_host` | `127.0.0.1` | Host/interface the OSC receiver binds to - set `0.0.0.0` to accept import requests from other machines |
| `recorder.audio.ambisonic_format` | `null` | Enable ambisonic capture. One of `a_nt_sf1`, `a_generic`, `b_fuma`, `b_ambix`; requires `channels: 4`. Converts capture to canonical AmbiX B-format on disk (see [Ambisonic capture](#ambisonic-capture)) |
| `ambisonic.decoder` | `basic` | Decoder weight mode: `basic` (flat velocity), `max_re` (tighter lobes, best HF), or `inphase` (softest lobes, no back-lobes) |
| `ambisonic.yaw_degrees` | `0.0` | Yaw rotation (degrees) applied to the B-format signal before decoding |
| `ambisonic.pitch_degrees` | `0.0` | Pitch rotation (degrees) applied to the B-format signal before decoding |
| `ambisonic.roll_degrees` | `0.0` | Roll rotation (degrees) applied to the B-format signal before decoding |
| `ambisonic.max_order` | `1` | Reserved for future higher-order support; currently must be 1 |

### Following your sequencer's tempo

`stretch_quantize` and `pad_quantize` snap a sample's onsets to a beat grid at
the session tempo, and the `duration_beats` selection filter measures sample
length in beats at that same tempo. The tempo lives in `tempo.bpm` - a fixed
number in `config.yaml`, so if you change tempo in your sequencer and forget to
change it here, quantized samples keep snapping to the old grid and beat filters
keep measuring against the old tempo - it just quietly stops matching your
sequence.

Set `tempo.source: midi` and subsample takes the tempo from the MIDI clock
arriving on the player's MIDI input instead:

```yaml
tempo:
  bpm: 125.0        # fallback, used until a clock is seen
  source: midi      # follow the sequencer
```

Worth knowing:

- **Set `tempo.bpm` anyway.** It's the fallback before any clock arrives (and
  if your sequencer never sends one). At `0.0` nothing quantizes until the
  transport starts, and a map that filters by `duration_beats` will not load.
- **The clock must reach the same MIDI input as your notes** (`player.midi_device`
  or your virtual port). Notes on one port and clock on another and it sees
  nothing.
- **The tempo is rounded to whole BPM and only adopted once it holds steady.**
  Every change re-bakes every quantized variant, so a change costs a burst of
  rendering - one-off per tempo, and cached to disk, so going back to a tempo
  you've used before is instant. Change tempo while stopped if you can.
- **It's sticky.** Stopping the transport keeps the last tempo rather than
  reverting, so stop/start doesn't re-bake anything.
- **An assignment's own `tempo:` still wins**, as does a CC-bound one. Order is
  step `tempo:` > MIDI clock > `tempo.bpm`.

Leave `tempo.source` at `config` and nothing changes - except that if a clock
*is* present and disagrees with `tempo.bpm`, you get one warning saying so
rather than silently wrong timing.

## Output

Recordings are saved as 16, 24, or 32-bit audio files (depending on
`recorder.audio.bit_depth`) in the configured output directory.  Container
format is controlled by `recorder.audio.audio_format` — `wav` (uncompressed,
the default) or `flac` (lossless compressed, see [Storage format](#storage-format)
below).

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

Both modes write to the same output directory. Point `library.directory` at
the same path to get a persistent library that grows on disk across sessions.

### Storage format

`recorder.audio.audio_format` decides whether new captures land as `.wav` or
`.flac`:

- `wav` (default) - uncompressed PCM.  Works at 16, 24, or 32-bit.
- `flac` - lossless compressed (around 40-60% smaller on typical material,
  decoded audio is bit-identical).  Works at 16 or 24-bit.

**The rule when formats don't quite line up:**

| Capture scenario | Extension written |
|---|---|
| `audio_format: wav`, any bit depth | `.wav` |
| `audio_format: flac`, live capture at 16 or 24-bit | `.flac` |
| `audio_format: flac`, 32-bit source (e.g. imported file) | `.wav` for that file, with an INFO log explaining why |
| `audio_format: flac` combined with `bit_depth: 32` (live capture) | Rejected at startup — set one or the other |

So if you flip `audio_format: flac` and then process a mix of 16/24-bit and
32-bit source files, you'll see a mix of `.flac` and `.wav` in your output
directory.  This is correct behaviour: the 32-bit fallback preserves full
precision rather than silently truncating.  Live captures share one bit
depth per session, so they stay consistent within a run.

**Existing libraries.** Upgrading to a subsample build with FLAC support does
not touch your existing `.wav` samples — they continue to load unchanged.
No migration, no bulk conversion.  FLAC only affects what gets written for
*new* captures once you flip the flag.

### Sample previews

When `recorder.previews: true` (the default), every captured or imported sample
also produces two visual-preview artefacts alongside the audio and analysis
sidecar:

- **`<sample>.preview.png`** — a fixed 1024x256 raster thumbnail (RGB, around
  15-25 KB) for browsing the library in an OS file manager.  The composition
  layers a 4-band frequency skyline behind a mirrored waveform envelope, with
  short vertical ticks at each detected onset and (when the sample is
  rhythmic) a dashed beat grid.  Stratum heights scale with each band's
  share of total energy (same four bands as `band_energy.energy_fractions`),
  so a bass-heavy kick looks bottom-heavy at a glance and a cymbal
  looks top-heavy — every band keeps at least a small minimum height
  so its temporal shape stays readable.  A bottom-right badge shows
  pitch (when tonal), BPM (when rhythmic), and duration.
- **A `preview` block embedded in `<sample>.analysis.json`** (around 4 KB) —
  the same composition's inputs (envelopes, band strata, onset/beat times,
  accent colour, badge text) in a compact form.  The Supervisor dashboard
  calls `subsample.preview.render_svg(data, width, height)` at request time
  to materialise a scalable vector preview at whatever size the layout wants.

> File managers on macOS, Windows, and Linux do **not** treat sibling PNGs as
> the audio file's own icon — the `.preview.png` appears as a separate file
> in the directory listing.  This is deliberate: embedding cover art would
> mutate the audio container, which subsample does not do.  Browse the
> previews alongside the audio files, or use the Supervisor dashboard for
> in-browser thumbnails.

Visual design (stroke weights, colours, layout) can be iterated later without
any schema bump — the `preview` block stores the underlying data, not the
rendered output.  Only a change in envelope resolution or band count
requires a `preview.version` bump.  Existing samples with no `preview`
block simply render nothing in the dashboard; they continue to play back
and analyse identically.

Set `recorder.previews: false` to skip both artefacts and save around
15-25 KB per PNG plus 4 KB of JSON per sample.

## Instrument sample library

Every recording is automatically added to an in-memory instrument library
alongside its full analysis data. A configurable memory cap prevents unbounded
growth; the oldest samples are evicted when a new one would exceed the limit.
The budget is auto-detected by default (60% of the global memory allocation -
see `max_memory_mb` in the configuration table) and can be overridden via
`library.max_memory_mb`. WAV files on disk are never deleted.

### Persistent library across sessions

```yaml
recorder:
  directory: samples/captures     # where new captures are written

library:
  directory: samples/captures     # where the playable library loads from
```

This pairing is the default - out of the box every capture lands straight in
the playable library, and the library persists and grows across sessions.
Point the two at different folders to play one collection while recording
into another.

On startup, Subsample walks `./samples/captures` recursively, so samples can be
organised into subdirectories - `kicks/`, `snares/`, `percussion/clangs/`, or
whatever scheme suits the user. Each audio file is one instrument sample,
identified by its full file path. The filename stem is a convenient label, not a
unique id - the same filename can appear in several folders (e.g. per-technique
take-folders each holding `01.wav`), and those all load as distinct samples. A
`name:` predicate then matches every sample sharing that stem; narrow to one with
`directory:`, `path:`, or `pick:` (see the Select section).

The library is self-healing across sessions: if the user renames, moves, or
removes audio files between runs (for example in an external auditioning tool
like Sononym), subsample's startup pass regenerates missing `.analysis.json`
sidecars and `.preview.png` previews from the audio at its current location, and
deletes any sidecars or preview images whose audio counterpart has gone away.
The on-disk state always reflects the audio present at startup.

As new recordings arrive they are written to disk and added to memory in one
step. The memory cap keeps only the most recent window of captures in RAM; the
full archive on disk is unaffected.

### Watching for new samples

Set `library.watch: true` to monitor the instrument directory for new audio
files at runtime and load them without restarting. The watcher detects audio
files from any source - another Subsample instance, a DAW, an SDR recorder, a
script, or any other application that writes audio to the watched directory.
The watcher monitors the top level of each watched directory only - drop new
files directly into the watched directory, not into a subdirectory (existing
subdirectories are still fully loaded at startup, which walks recursively).

Two detection paths run in parallel:

1. **Sidecar path** - watches for `.analysis.json` sidecar files (fastest).
   When a sidecar appears, its corresponding audio file is loaded immediately
   without re-analysing. This is the path taken when the source is another
   Subsample instance, which always writes the WAV first and the sidecar second.

2. **Audio file path** - watches for audio files (`.wav`, `.flac`, `.aiff`,
   `.aif`, `.ogg`, `.mp3`, `.mpeg`) from any source. After a short grace
   period to see if a sidecar follows (in case the source is Subsample),
   checks that the file is no longer being written (file-size stability),
   runs the full analysis pipeline, writes a sidecar, and loads the sample.

The audio file path handles the common case where another application writes
an audio file without any sidecar. The file-size stability check ensures that
long recordings still being written are not loaded prematurely - the watcher
waits until the file size stops changing before attempting to read it.

Deleting or renaming a watched audio file away also takes effect live: its
sample is removed from the running library, so a re-encoded or deleted file (for
example `01.wav` re-exported as `01.flac`) no longer lingers as a stale,
still-selectable sound.

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
  directory: "/mnt/shared/samples"

player:
  enabled: false
```

**Player machine** (`config.yaml`):
```yaml
recorder:
  enabled: false

player:
  enabled: true

library:
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
  channel: 10
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

Reference lookups are case-insensitive.

The GM drum references placed in your project by `subsample --init` are
sidecar files only - the acoustic fingerprints derived from the FluidR3_GM
SoundFont (MIT licence; see `samples/reference/CREDITS.md`). No audio is
distributed: the fingerprint is all the similarity engine needs, so the kit
matches against them without any reference WAVs on disk.

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
> `recorder.audio.buffer_frames`, lower the sequencer's buffer size, or disable the
> recorder (`recorder.enabled: false`) to run Subsample in playback-only mode.

## OSC integration

Subsample can send and receive [OSC (Open Sound Control)](https://opensoundcontrol.stanford.edu/)
messages, so it can talk to sequencers, visualisers, custom scripts, or any
other OSC-compatible application. OSC support is an optional extra: install it
with

```bash
pip install "subsample[osc] @ git+https://github.com/simonholliday/subsample.git"
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
| `/sample/import` | Read the file at the given path, analyse it, and load it into the in-memory instrument library for immediate playback. The file is read in place - it is not copied or moved. The sample is available until the next restart; for persistence, place the file in `library.directory` instead (or as well). | `file_path:str` |

This is more targeted than the directory watcher and lets external applications
load specific files into the library on demand - for example, a radio scanner
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
Together, they form part of a fully open-source generative sampler
workstation - Subsequence drives the patterns, Subsample provides the
sounds.

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

## Tools

Five companion tools ship inside Subsample as subcommands - run
`subsample <command> --help` for each one's full options:

| Command | What it does |
|---|---|
| `subsample import` | Import pre-trimmed audio (sample packs, field recordings) into the library |
| `subsample catalog` | CSV catalog of every sample's detected properties, with curation aids |
| `subsample analyze` | Analyze audio files and print their detected metrics |
| `subsample similar` | Rank library samples against each reference by similarity |
| `subsample loops` | Find and audition seamless loop points in sustained samples |

(A file argument that shares a command name needs a path prefix:
`subsample ./import` treats it as an input file to chop into the capture
library, while `subsample import` runs the import tool.)

### Analyzing recorded files

```bash
subsample analyze samples/2026-03-17_14-32-01.wav
```

Output:
```
rhythm:   tempo=120.2bpm  beats=4  pulses=12  onsets=4
spectral: duration=2.00s  flatness=0.001  attack=0.000  release=0.812  centroid=0.018  bandwidth=0.001  zcr=0.120  harmonic=0.821  contrast=0.310  voiced=0.940  log_attack=0.000  flux=0.312  rolloff=0.451  slope=0.023
pitch:    pitch=440.0Hz  chroma=A  pitch_conf=0.89  stability=0.120st  voiced_frames=86
level:    peak -1.2dBFS  rms -12.6dBFS  crest 11.4dB  floor -42.3dBFS
noisiness: 0.012  (0 = clean event, 1 = wall-to-wall noise)
loop:     0.412s -> 1.187s (775 ms, xfade 30 ms, junction_flux 0.08)
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
- **slope** - 0 = steeply bass-dominated, 0.5 = roughly flat, 1 = bright/treble-dominated

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

Noise-likeness:
- **noisiness** - 0-to-1 rating of how noise-like the sample is across its whole
  length; near 1 = wall-to-wall unpitched noise (radio static, a dead channel),
  near 0 = a clean hit or pitched tone. Computed as *stationarity* (the signal
  never gets quiet) times *lack of pitch*, so a held tone or a decaying hit both
  score low. Sustained unpitched textures (cymbal rolls, noise sweeps) score
  high too, so use it to sort and audition, not to auto-delete.

Three MFCC timbre fingerprints are stored in the sidecar (used for similarity,
not shown in script output): `mfcc` (mean, average timbre), `mfcc_delta`
(first-order trajectory), and `mfcc_onset` (onset-weighted, attack emphasis).

### Cataloging a sample directory

```bash
subsample catalog                          # configured library.directory
subsample catalog path/to/samples          # any directory
subsample catalog -o samples.csv           # write to a file
subsample catalog --full                   # every property incl. MFCC vectors
```

Walks a directory (recursively, the same walk the instrument library performs
at startup) and writes one CSV row per audio file with its detected
properties. Three capability columns show which musical behaviours each sample
is eligible for:

- **pitched** - passes the stable-pitch test, so `pitched: true` selects it in
  a MIDI map and it can be re-pitched across a keyboard range
- **quantizable** - has at least 2 detected hits, so `stretch_quantize` /
  `pad_quantize` can align them to a beat grid; below that, quantize degrades
  to a plain stretch or a pass-through
- **loopable** - has a steady sustaining region (a held tone, or a stationary
  textural bed) worth looping while a key is held, so `loopable: true` selects
  it. This is a coarse candidate flag - it means "worth trying to loop", not
  "here is the seamless loop point" (that is found later from the audio itself).
  The companion **loop_ms** column shows the actual detected loop length in
  milliseconds; it is blank when a sample is loopable but too evolving to wrap
  seamlessly (no clean loop point was found)

All three columns run exactly the tests the player runs, so the catalog shows
the selection pool a MIDI map would draw from.

Two further columns describe *when the sound actually happens*:

- **impact_ms** - how far into the file the loudest event begins. For a struck
  drum this is `0.0`: the hit is the first thing in the file. It is only
  non-zero for sounds with a preparatory noise in front of the musical moment -
  a hi-hat pedal close (the foot hits the pedal, then the cymbals meet ~40 ms
  later), a shaker pulled back before the downbeat, a tambourine drawn off the
  palm before the strike, a guiro whose loudest ridge arrives after the scrape
  starts
- **impact_pre_db** - the level of everything before that moment, relative to
  the sample's own peak. Blank when there is nothing before it. Around -20 dB
  means a genuinely quieter preparation; **near 0 dB is a warning** that the
  file holds several comparable events (a loop, a long take) and `impact_ms` is
  simply pointing at whichever was loudest, not measuring a pre-stroke

The pair is intended for a sequencer that wants a note's transient to land *on*
the beat: trigger the note `impact_ms` early and the hit arrives on time with
the preparation intact. Subsample's own playback does not use it - a voice
still starts at the beginning of the sample - so nothing changes in how your kit
sounds. The values also ride in each sample's `.analysis.json` sidecar as
`rhythm.impact_time` (in **seconds**) and `rhythm.impact_pre_level_db`, so any
tool that can read the sidecar can use them without going through the catalog.

Because the offset comes from how a sound was *performed* rather than from what
kind of sound it is, it varies hit to hit: across one set of pedal-close
captures the median was 42 ms but the middle half spanned 6-55 ms. Treat it as
a per-sample value; an average over a folder would mis-time most of them. All seven inputs to the pitched
test are included as columns (`pitch_hz`, `voiced_fraction`,
`voiced_frame_count`, `pitch_confidence`, `pitch_stability_st`,
`harmonic_ratio`, `duration_s`), so a sample that unexpectedly fails can be
diagnosed against the thresholds documented on the test.

With `--pitched`, `--quantizable` and/or `--loopable` (combined: all must
hold), the CSV is replaced by matching file paths, one per line - pipe them
into a player to audition a capability group, or into file tools to build
curated sets:

```bash
subsample catalog --pitched | mpv --playlist=-
subsample catalog ~/samples --quantizable | xargs -I{} cp {} ~/curated/
subsample catalog ~/samples --loopable | mpv --playlist=-
```

Files without an `.analysis.json` sidecar are analyzed on the way (slow on the
first run, same cost as a startup load - progress goes to stderr); results are
cached as sidecars so later runs are instant.

#### Curation aids

Leaving a mic running for hours or days produces thousands of samples, many of
them the same sound over and over. Three options help thin them down:

```bash
subsample catalog --group              # cluster near-duplicates
subsample catalog --group --pitched    # one keeper path per pitched-sound group
subsample catalog --order similarity   # rows ordered by how alike they sound
```

- **`--group`** clusters samples that sound nearly identical (the same event
  captured repeatedly) so days of recordings collapse to one decision per
  distinct sound. It adds `group`, `group_size`, and `group_keeper` columns -
  the biggest pile first, with a suggested keeper (the loudest, usually
  cleanest take) marked in each. In paths mode it emits only each group's
  keeper, a deduplicated set. `--similarity-threshold T` (default 0.98) tunes
  how alike samples must be to group: higher groups only near-identical takes,
  lower groups more loosely. **On a homogeneous corpus (e.g. one mic in one
  room) everything is somewhat alike, so this is sensitive - audition a group
  or two to calibrate the threshold to your ears before trusting it to bin
  samples.**
- **`--order similarity`** orders the rows (or paths) as a nearest-neighbour
  chain, so each sample is followed by its closest-sounding neighbour.
  Auditioning alike-with-alike turns keep/discard into a quick comparison
  rather than a cold judgement.
- Four **junk-triage** columns flag the obvious discards so they sort to the
  top of the bin pile: `snr_db` (how far the loudest moment rises above the
  quiet-frame floor - low means room tone with no real event), `near_silent`
  (nothing loud enough to be an event), `clipping_risk` (peak at digital full
  scale, possibly distorted), and `noisiness` (a 0-to-1 rating of how noise-like
  the sample is across its whole length - near 1 means wall-to-wall unpitched
  noise like radio static or a dead channel, near 0 means a clean hit or a
  pitched tone). `noisiness` is aimed at captures that were triggered by a
  transient but are mostly noise thereafter; sustained *unpitched* textures
  (cymbal rolls, noise sweeps) also score high, so treat it as a
  sort-and-audition signal rather than an automatic discard.

Grouping and similarity ordering compare samples on the same acoustic
fingerprint the MIDI mapper uses for its similarity matching, so "alike" means
the same thing here as in a `beat_match`/`similarity` MIDI-map query.

### Importing pre-trimmed samples

Import audio files from any source (SDR captures, commercial sample packs, field
recordings) directly into the capture library, bypassing the detection pipeline.
Files are silence-trimmed, safety-faded, re-encoded as standard PCM WAV, fully
analyzed, and saved with sidecar JSON. A large batch is fingerprinted across the
machine's cores in parallel, so importing a big sample pack is many times faster
than processing one file at a time.

```bash
subsample import /path/to/samples/*.wav
subsample import --to samples/captures /path/to/sample-pack/*.wav
subsample import --force "/mnt/sdr/audio/2026-01-15/*.wav"
```

- `--to DIR` - target directory (default: `recorder.directory` from config.yaml)
- `--force` - overwrite existing files in target directory

Handles WAV, BWF (Broadcast Wave Format), FLAC, AIFF, OGG, and any other format
supported by libsndfile. BWF and non-WAV sources are re-encoded as standard PCM WAV
so the rest of the pipeline can load them reliably.

### Similarity report

```bash
subsample similar           # top 5 per reference (default)
subsample similar --top 10  # top 10 per reference
```

Example output:
```
Reference: GM36_BassDrum1
  1.  #5     0.9412  GM36_BassDrum1  ./samples/kick_deep.wav
  2.  #7     0.8134  kick_hard       ./samples/kick_hard.wav
  3.  #8     0.7601  kick_soft       ./samples/kick_soft.wav
```

### Finding loop points

Preview and audition the seamless loops that `mode: loop` playback will use:

```bash
subsample loops samples/captures            # propose loop points for loop candidates
subsample loops pad.wav --render auditions/ # also write A/B audition WAVs
```

For each loop-candidate sample this prints the proposed loop start/end,
crossfade, and junction quality (or a note that no clean loop exists - such a
sample plays gated instead). `--render DIR` writes two audition files per
sample: the crossfaded loop and the raw butt-joint, so you can hear exactly
what the player will do before a key is ever held.

### Maintainer scripts

The remaining scripts in `scripts/` are maintainer tools and need a repo
checkout: `measure_midi_latency.py` and `measure_handler_timing.py` (the
latency guards described under [Performance](#performance)),
`regen_previews_png.py` (regenerate preview thumbnails after a format bump),
and `extract_gm_drums.py` (regenerate the shipped GM reference fingerprints
from a SoundFont).

## Roadmap

### MIDI expressiveness

- **Round-robin sample cycling** - step *deterministically* through alternative
  samples on repeated triggers. Random per-trigger variation already ships
  (`pick: [1, 3]` or `pick: any` draws a fresh sample on every hit - see
  [Select](#select---which-sample-to-play)); this item is the strict-rotation
  complement, plus per-note rotation state. (Distinct from the existing per-hit
  segment round-robin, which cycles through detected hits inside a single
  sliced loop.)

### Sample management

- **Auto-slicing** - chop loops and long recordings into individual hits by
  transient detection, then add each slice to the library as a separate sample.
- **Similar-to-this query** - "find more sounds like this one" by exposing the
  similarity engine as a user-facing search.

### Monitoring

- **Supervisor dashboard extensions** - the read-only web dashboard shipped
  (enable with `supervisor.enabled: true`); planned additions include voice
  activity metering and transform queue progress.

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
        → enqueue base variant ONLY                 ← float32 peak-normalised copy
            → TransformProcessor worker pool
                → TransformCache (parent-priority FIFO eviction, 50 MB default)

MIDI map loaded / reloaded
    → MidiPlayer.update_assignments()
        → enqueue pitch variants (tonal only)       ← Rubber Band offline finer engine
        → enqueue time-stretch (if BPM set + enough onsets) ← beat-quantized timemap_stretch
            → (same worker pool and cache)
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
    → _resolve_sample_id: indexed pick from the pre-computed candidate cache
        (rebuilt when the library changes, not per-trigger; variant-state
         selects — quantized_beats / beat_match — fall back to a live query)
    → transform_manager.get_variant(sample_id, spec)  → processed variant
        (memory cache → disk cache → enqueue + fall back to a previous/base variant)
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
- A C/C++ build toolchain and `pkg-config` (Linux) - `apt install build-essential pkg-config` or `dnf install gcc-c++ make pkgconf-pkg-config`. PyAudio and python-rtmidi are compiled from source on Linux, and a minimal install (e.g. Ubuntu Server) ships none of this by default. macOS uses the Xcode command-line tools.
- PortAudio (required by PyAudio - `apt install portaudio19-dev`, `dnf install portaudio-devel`, or `brew install portaudio`)
- ALSA development headers (Linux, required by python-rtmidi for MIDI - `apt install libasound2-dev` or `dnf install alsa-lib-devel`); macOS uses CoreMIDI and needs neither. The JACK dev headers (`libjack-jackd2-dev`, `dnf install jack-audio-connection-kit-devel`) are optional - only needed if you build against JACK; the JACK *runtime* PortAudio loads ships with PortAudio itself (see the sound-server bullet below).
- Rubber Band (required by pyrubberband - `apt install rubberband-cli`, `dnf install rubberband`, or `brew install rubberband`)
- **A running sound server (Linux).** PortAudio initialises its JACK backend at startup and aborts the whole process with `OSError: [Errno -9999] Unanticipated host error` if it cannot reach an audio server - *even if you never use JACK*, because `libportaudio` links JACK unconditionally. Desktop Linux already runs PipeWire; a minimal/headless/server install (e.g. Ubuntu Server) has none, so install and start one - PipeWire is recommended, as it also supplies the JACK layer PortAudio needs and fixes the ALSA default device: `sudo apt install pipewire pipewire-alsa pipewire-jack pipewire-pulse wireplumber` then `systemctl --user enable --now pipewire pipewire-pulse wireplumber` (add `sudo loginctl enable-linger "$USER"` for headless autostart). Once PipeWire is running, plain `subsample` works but PortAudio's JACK probe logs harmless `jack server is not running` warnings; launch under `pw-jack` (`pw-jack subsample`) for a clean startup that routes through PipeWire's JACK client instead. macOS uses CoreAudio and needs none of this.

**Windows users:** install and run Subsample inside [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
(Windows Subsystem for Linux). This gives you a real Linux environment where
the `apt` instructions above just work. Audio devices need to be exposed to
WSL - see the [WSL audio guide](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)
for USB passthrough or use a network audio bridge if your interface supports
one. Subsample is not currently tested against native Windows Python.

## Tests

For working on Subsample itself (everything above works from a plain
`pip install` - no clone needed): clone the repo, install editable with the
dev extras, and run the suite.

```bash
git clone https://github.com/simonholliday/subsample.git && cd subsample
pip install -e ".[dev]"
pytest
```

## Type Checking

Same setup as [Tests](#tests):

```bash
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
| [Pillow ↗](https://python-pillow.org/) | PNG waveform preview rendering | MIT-CMU |
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
