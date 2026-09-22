# Subsample

**A sampler that cuts separate sounds out of a live input or a recording,
sorts them by how they sound, and plays them from MIDI.**

Subsample listens to an audio input, or reads audio files, and cuts out each
separate sound. It measures each sound's spectrum, timbre, attack, and pitch,
and can compare those measurements with a set of reference sounds, so that a
kick-like sound is offered to the kick note. A MIDI map sets which sounds each
note plays and how they are processed on the way out, and a keyboard, pad
controller, or sequencer plays them.

Subsample has no graphical interface: it is set up in configuration files and
run from a terminal. It runs on Linux and macOS; on Windows it runs inside
WSL2, and is not tested on native Windows. This is an early release, and its
configuration and MIDI map formats still change between releases.

Build a custom drum kit from the sounds outside your window. Turn a walk
through the woods into a playable instrument. Slice a dawn chorus into single
calls. Feed a pile of unsorted samples in and watch them organise themselves.
All four are the same workflow.


## Contents

- **1. Getting started**
  - [Why Subsample?](#why-subsample)
  - [At a glance](#at-a-glance)
  - [Documentation](#documentation)
  - [Quick start](#quick-start)
- **2. Concepts**
  - [How it works](#how-it-works)
  - [MIDI map](#midi-map)
  - [Similarity engine](#similarity-engine)
  - [Transforms](#transforms)
- **3. Configuration and operation**
  - [Configuration](#configuration)
  - [Output](#output)
  - [Instrument sample library](#instrument-sample-library)
  - [Reference sample library](#reference-sample-library)
  - [Live-coding the MIDI map](#live-coding-the-midi-map)
- **4. Integration**
  - [Virtual MIDI](#virtual-midi)
  - [OSC integration](#osc-integration)
  - [Works with Subsequence](#works-with-subsequence)
- **5. Project information**
  - [Performance](#performance)
  - [Tools](#tools)
  - [Architecture](#architecture)
  - [Requirements](#requirements)
  - [Tests](#tests)
  - [Type checking](#type-checking)
  - [Dependencies and credits](#dependencies-and-credits)
  - [About the author](#about-the-author)
  - [Licence](#licence)
  - [Commercial licensing](#commercial-licensing)


## Why Subsample?

- **A studio sampler that builds itself.** Drop samples in (or record them
  live) and Subsample maps them to MIDI notes, processes them through an
  adaptive DSP chain, and presents a playable, mix-ready instrument with no
  manual chopping, naming, or mapping.
- **Automatic similarity-based sample organisation.** Kicks are matched to
  kick pads, snares to snare pads, hi-hats to hi-hat pads - no labels, no
  training data, no manual tagging. The same engine handles tonal samples
  without special treatment, and works equally well on a disorganised sample
  library or a fresh field-recording session.
- **Real-time live sampling.** Point a microphone at the world and Subsample
  captures, trims, analyses, and adds every distinct sound event to your
  instrument library as it happens. It works in noisy rehearsal rooms as well
  as quiet studios.
- **Beat slicer and auto-quantise for loops.** Detected onsets in long samples
  are individually placed on a beat grid, so each hit in a loop lands on the
  grid at your target BPM.
- **Pitched and percussive in one engine.** Subsample detects samples with a
  single stable pitch and shifts them across the keyboard range. Drums,
  melodic, and effect samples share one library and one workflow.
- **A DSP chain that sets its defaults from each sample.** Write
  `compress: true` and the threshold, attack, and release are derived from the
  audio. Variants are rendered in a background worker pool before they play.
- **Sweep anything with a knob.** A filter cutoff, a beat-quantise amount, a
  distortion drive, a compression threshold: variants are re-rendered in the
  background between knob positions and bridged smoothly, so you can play with
  parameters that aren't normally automatable on samplers at all.
- **Multichannel in, multichannel out.** Records from any subset of physical
  inputs on a multi-channel interface (e.g. inputs 3-4 of a Focusrite Scarlett
  18i20). Routes individual instruments to specific outputs (kick to outputs
  1-2, snare to outputs 3-4) for separate external processing. First-order
  ambisonic capture from tetrahedral mics (Rode NT-SF1, generic A-format, or
  pre-encoded B-format FuMA/AmbiX) with decoder and rotation at playback time.
- **WAV or lossless FLAC storage.** Opt into FLAC (`audio_format: flac`) to
  shrink your sample library with no loss of quality. Existing WAV samples
  continue to load unchanged alongside any new FLAC captures.
- **Visual sample previews.** Every capture gets a fixed 1024x256 `.preview.png`
  thumbnail (waveform + spectral band skyline + onset ticks + pitch/BPM
  badge) for browsing in an OS file manager, plus the compact data it is
  drawn from, kept in the analysis sidecar so a missing thumbnail can be
  redrawn without re-analysing.
- **Plays nicely with the rest of your studio.** Standard MIDI input from any
  DAW or hardware controller, virtual MIDI ports for software-only routing on
  the same machine, OSC integration for talking to sequencers and visualisers,
  and a ready-to-play GM drums map that turns any sample collection into a
  coherent, pre-mixed drum kit on first play.
- **Pairs with Subsequence.** Subsample's sister project
  [Subsequence](https://github.com/simonholliday/subsequence) is a Python
  MIDI sequencer: Subsequence drives the patterns, and Subsample provides the
  sounds. Each works independently.


## At a glance

| | |
|---|---|
| **Live capture** | Adaptive noise floor, capture that keeps recording while analysis runs, S-curve fades |
| **Analysis** | Spectral shape, sustained timbre, timbre dynamics, attack character, and spectral band energy; cached `.analysis.json` sidecars |
| **Matching** | Cosine similarity, classification-free, ranked fallback, dynamic re-assignment |
| **DSP processors** | Filters, dynamics, distortion, radio, vocoder, pitch, time-stretch, and quantise |
| **Adaptive defaults** | Compressor, gate, transient shaper, distortion, envelope reshape - all auto-derive parameters from each sample |
| **Pitch shifting** | Rubber Band offline finer engine, pre-rendered |
| **Time stretch** | Beat-quantised with onset-aligned timemaps, partial-quantise amount, pad-quantise alternative for speech |
| **Segment playback** | Per-hit round-robin, random, or indexed - for sliced loops |
| **MIDI input** | Hardware port, named virtual port, or both |
| **MIDI control** | Note on/off, Program Change for programs, CC binding for any numeric parameter |
| **OSC** | Sender + receiver (optional dependency) |
| **Audio formats in** | WAV, BWF, FLAC, AIFF, OGG, MP3/MPEG (libsndfile) |
| **Audio channels** | Mono through 7.1, ITU-R BS.775 downmix, conservative upmix, per-instrument output routing |
| **Audio precision** | End-to-end 32-bit float pipeline, 64-bit DSP for IIR filters and envelope followers |
| **Latency** | Pre-rendered variants - playback is a memory copy into the mix buffer |
| **Library mgmt** | Memory-bounded with FIFO eviction, persistent disk cache for variants, hot-loading from watched directories |
| **Live-coding** | Edit the MIDI map YAML and assignments reload on save |
| **Program switching** | Multiple instrument directories swappable via MIDI Program Change |
| **GM drums** | Ready-to-play map of the General MIDI percussion set, with a mix chain for each instrument |
| **Configuration** | YAML, version-controllable, headless, no GUI |
| **Platform** | Linux, macOS, Windows (via WSL), Raspberry Pi |


## Documentation

**Full documentation: [https://subsystem.co/subsample/](https://subsystem.co/subsample/)**

- Configuration reference: [https://subsystem.co/subsample/configuration/](https://subsystem.co/subsample/configuration/)
- MIDI map reference: [https://subsystem.co/subsample/midi-map/](https://subsystem.co/subsample/midi-map/)

Both are generated from the code that reads these files, so they describe the
release you have rather than a copy kept by hand.

## How it works

### 1. Capture

Subsample listens continuously to a live audio input and captures every distinct
sound event. An adaptive noise floor (exponential moving average) tracks the
ambient level in real time, so it works equally well in a quiet studio and a
noisy rehearsal space. Each captured sound is trimmed with smooth S-curve fades
to avoid clicks.

All audio channel formats are preserved end-to-end - a stereo microphone records and
plays back in stereo, a quad recording keeps its four channels, and
multichannel samples are automatically mapped to the output layout using
standard ITU downmix coefficients. On multi-channel interfaces (e.g. Focusrite
Scarlett 18i20), `recorder.audio.input` selects which physical inputs to
record from - for example `[3, 4]` records a stereo pair from inputs 3 and 4.

You can also feed it pre-recorded WAV files - they pass through the same
detection pipeline, so a sample library can be built from existing
recordings. For pre-trimmed sources (commercial sample packs, field recordings,
SDR radio captures), `subsample import` bypasses detection entirely and imports
files directly with silence trimming, safety fades, re-encoding, and full
analysis.

### 2. Analyse

Each captured sound is fingerprinted across five groups of acoustic measurements:

| Group | What it captures |
|-------|------------------|
| Spectral shape | Brightness, noisiness, attack/release character |
| Sustained timbre | Steady-state tonal colour |
| Timbre dynamics | How the sound evolves over time |
| Attack character | Transient signature |
| Spectral band energy | Per-band energy distribution and decay (drum-type signature) |

Tonal sounds are identified by a pitch stability test - only
samples with a single, confident, stable pitch are flagged for chromatic mapping.
Percussive sounds are handled naturally by the same feature space without special
treatment.

Analysis results are cached as `.analysis.json` sidecar files alongside each
WAV. The cache is versioned and auto-invalidating - when the analysis algorithm
improves, stale sidecars are detected and re-analysed automatically on startup.

### 3. Assign

Sounds are matched to your reference library using cosine similarity on that
fingerprint. The best kick-like sound maps to your kick pad; the
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
harmonic/percussive separation, and beat quantisation. Variants are computed
offline in a background worker pool and cached to disk, so once a variant is
ready, a note plays it without processing anything when it is triggered.

The compressor, gate, transient shaper, distortion, and envelope reshaper are
designed with **defaults that adapt to each sample**, while the filters default
to fixed console channel-strip values (80 Hz HPF, 16 kHz LPF). The compressor
analyses each sample's peak level, onset speed, and decay character to set
threshold, attack, and release automatically - a percussive kick gets a slow
attack that preserves the beater transient, while a sustained pad
gets a faster attack with longer release to avoid pumping. The gate reads the
noise floor to set its threshold. Transient shaping reads the crest factor to
decide how much punch to add or remove. Envelope reshape reads the decay
character to tighten the tail. Write `compress: true` or `transient: true` and
the right parameters are derived from the audio itself.

Beat-quantised time-stretching locks samples to a target BPM using onset-aligned
timemaps - each onset is individually placed on the beat grid with minimal
stretching between them. For speech and other material where time-stretch
artefacts are unacceptable, pad-quantise snaps onsets to the grid by inserting
silence instead, preserving natural timbre completely.

The included `midi-map-gm-drums.yaml` applies all of this across the full GM
percussion set: every instrument in it, each with its own filtering, compression
(where appropriate), panning, and gain. The result is a coherent, pre-mixed drum
kit from whatever samples you have - no manual tweaking required. Every setting
can be overridden by an experienced user who wants precise control.

## MIDI map

The MIDI map is where Subsample becomes an instrument you can *play*. You do not
assign samples to notes one by one: you write *rules* that pick samples from
your library at trigger time - by similarity to a reference, by analysis metadata, by age,
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

```yaml config
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

The simplest possible assignment: MIDI note 36 (on MIDI channel 10, the GM drum
channel) always plays one named sample.

```yaml map.assignments
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

```yaml map.assignments
- name: Any kick
  channel: 10
  notes: 36
  select:
    where:
      reference: GM36_BassDrum1
```

`reference` **names** a reference sample. Subsample ships a precomputed
fingerprint for each General MIDI sound - the `.analysis.json` files only, no
audio, since the fingerprint *is* the reference - and the name above resolves
from that built-in set on any machine where Subsample is installed. Nothing is
copied into your project, and the map stays portable: it can live on a shared
drive, be used by several projects, and still find its references.

You can also point `reference` at a file (`reference: my-refs/kick.wav`), which
resolves relative to the map file like every other path. Use that for your own
reference material; use the name form for the built-in GM set. To match against
a different reference set entirely, point `library.reference_directory` at it -
its names then replace the built-in ones.

The library's samples are ranked against that reference by their acoustic
fingerprint; the top-ranked match plays.
(When `reference` is set and no `order` is given,
`order: [{ by: similarity, dir: desc }]` is assumed - see
see [choosing a sample](https://subsystem.co/subsample/midi-map/choosing-a-sample/).)

#### Step 3 - rule-based selection

Filter the library by analysis metadata, sort the qualifying samples, and
pick one. This example plays the **oldest pitched sample** across a whole
keyboard range, pitch-shifted to each MIDI note:

```yaml map.assignments
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

```yaml map.assignments
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
the [processing reference](https://subsystem.co/subsample/midi-map/processing/).

#### Step 5 - lock a loop to your session tempo

`stretch_quantize` time-stretches a sample to a target BPM and snaps its onsets
to a beat grid - turning any loosely-timed loop in your library into something
locked to the session. Combine it with filtering for a length+rhythm pick:

```yaml map.assignments
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
dicts (`gte`, `lte`, `gt`, `lt`). `strength: 0.7` is how far each hit moves
toward the grid - every hit on it at 1.0, and at 0.0 the sample is still
stretched to the tempo with its hits left where that puts them.

That's the ladder. The rest of this section is the full reference - every
field, every predicate, every processor, every option - then the advanced
features (programs, ambisonic capture, MIDI CC mapping).

### The GM drums map - a drum kit from any sample collection

Before the reference, a quick mention of the "no-config" path. If you want a
complete drum kit without writing a map, use the `midi-map-gm-drums.yaml` that
`subsample --init` placed in your project (it is even pre-wired into the
scaffolded `config.yaml`). Point Subsample at any sample collection and every
MIDI drum note automatically finds the closest matching sample and plays it
through a mix chain set for its instrument:

- **Similarity matching** - each note finds the best sample via spectral
  fingerprint comparison against GM reference sounds
- **Console-style filtering** - per-instrument HPF/LPF to carve frequency space
  (30 Hz HPF on kicks, 300 Hz on hi-hats, 1 kHz on triangles, etc.)
- **Adaptive compression** on the transient instruments - threshold, attack, and
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

Every field a map accepts is documented term by term in the
[MIDI map reference](https://subsystem.co/subsample/midi-map/). What follows is
the part that teaches rather than lists: a tutorial, the GM kit, templates,
sample sets, ensembles and programs.

### Templates - share fields across assignments

Drum kits repeat themselves: the same `channel`, the same process chain, the
same `select` shape on every pad. Define those shared fields once in a
top-level `templates:` section and pull them into each assignment with a
`template:` reference.

```yaml map
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
*omits* is inherited. The merge is top-level only - if the assignment sets its
own `process` (or `select`), that **replaces** the template's wholesale rather
than appending to or merging into it.

**Stacking templates.** `template: [percussion, loud]` applies several
left-to-right: a later template overrides an earlier one, and the assignment's
own fields override them all.

Templates are flat - a template cannot itself carry a `template:` (inheritance
is one level deep). A `template:` that names an undefined template is a load
error listing the templates you defined.

### Sample sets - one folder, one map, reusable

A **sample set** is a directory of samples with a MIDI map at the top of it:

```
Home Kit 2026-07/
  midi-map.yaml        <- the map
  snare_muffled_1/
  hh_closed_mid/
  hh_open_mid/
```

Nothing declares that this is a set. It is one because the map's
`directory:` predicates resolve relative to the map file and its `reference:`
predicates name built-in fingerprints rather than pointing at a project. So the
folder is self-contained: copy it anywhere, share it between projects, keep it
on a network drive, and it still works.

Give the map a top-level `channel:` and the set becomes portable in the other
sense too - a project can play it on whatever MIDI channel it likes without editing
it.

A set may hold **several maps**, exposing different mappings over the same
samples - a full kit and a stripped one, say. `midi-map.yaml` is the expected
name but nothing requires it; you reference a set by naming the map file you
want:

```
Home Kit 2026-07/
  midi-map.yaml        <- the full kit
  minimal.yaml         <- kick and snare only
```

### Ensembles - several sample sets at once

An **ensemble** binds sets to MIDI channels so they all play together. It is
not a new kind of file: any map may declare a `maps:` block, and may carry its
own `assignments:` alongside.

```yaml map
# ensemble.yaml
maps:
  - "/mnt/shared/Home Kit 2026-07/midi-map.yaml"       # keeps its own channel
  - { channel: 11, map: "/mnt/shared/vocals/live.yaml" }
  - { channel: 12, map: "/mnt/shared/effects/midi-map.yaml" }
```

Point `player.midi_map` at that file and all three sets are live at once, each
on its own MIDI channel. Only the samples those sets name are loaded - set
`library.directory: null` and nothing else is read at all.

An entry is either a bare path (the set plays on the MIDI channel its own map
declares) or a mapping with `channel:` and `map:`. A bound `channel:` replaces
the set's top-level default. An assignment that names its **own** `channel:`
still wins - that is how one map deliberately spans several channels - and the
binding logs a warning when it finds one, since it will not move those entries.

The same thing can be written straight into `config.yaml` when you would rather
not keep a separate file:

```yaml config
player:
  midi_maps:
    10: "/mnt/shared/Home Kit 2026-07/midi-map.yaml"
    11: "/mnt/shared/vocals/live.yaml"
```

`midi_map` and `midi_maps` are mutually exclusive. Both forms go through the
same loader, so identical bindings give identical results. Prefer the
ensemble file when you want to name MIDI channels from a `definitions:` vocabulary
(`channel: my.kit`) or to reload while running - `definitions:` is a map-level
feature `config.yaml` cannot see, and `config.yaml` is not watched for changes.

Rules worth knowing:

- **Flat, one level.** A map included via `maps:` may not declare `maps:` of its
  own. Same rule as `map:` presets, and it means there are no cycles to worry
  about.
- **One (MIDI channel, note) per set.** If two sets claim the same MIDI channel and note,
  the load fails naming both - a silent merge would fabricate a
  velocity-switched note out of two unrelated sets.
- **No `programs:` inside an included set.** Program switching holds one active
  bank for the whole player, so per-channel switching is not expressible. Use it
  in the ensemble itself, or load the set on its own.
- **Only the ensemble file is watched.** Editing an included set takes effect
  on restart. (And a set on a network drive will not be watched at all - see
  [Live-coding](#live-coding-the-midi-map).)

### Programs - switching instrument sets via MIDI

The MIDI map can optionally declare multiple **programs** (presets) that are all
loaded at startup. Switch between them at runtime using MIDI Program Change
messages - no restart, no disk I/O, instant switching. A `program:` number may
also be a name from a mounted definitions file (`program: my.brushes` - see
[the definitions file](https://subsystem.co/subsample/midi-map/the-definitions-file/)):

```yaml map
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
  top-level `assignments:` and points them at a different folder. Use this
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

```yaml map
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

## Performance

### Pre-rendered playback

When a sample enters the library, a background worker produces a pre-rendered
copy at the output device's sample rate and format. Tonal samples also receive
a set of pitch-shifted variants. When a MIDI note arrives, playback copies the
prepared audio into the mix buffer rather than calculating it on the spot. If a
variant is not ready yet, the player falls back in this order:

1. **Process variant** - pre-computed with the full declared chain (pitch, filter, saturate, reverse, time-stretch, etc.)
2. **Base variant** - pre-normalised, no DSP (all samples)
3. **On-the-fly render** - used only on the first trigger, before any variant exists

### MIDI dispatch model

Incoming MIDI is dispatched in callback mode: rtmidi delivers each message
to Subsample's handler on its own dedicated thread as it arrives. There is no
polling loop, so there is no fixed input-latency floor. The delay a player
hears between a key and its sound has three parts:

1. **MIDI dispatch** - the rtmidi callback handing the message over.
2. **Per-note handling** - choosing the sample and starting the voice. Sample
   selection is worked out when the library changes, so a trigger picks from
   a prepared list rather than searching the library.
3. **Output latency** - the time from the output buffer to the converter. This
   is usually the largest part, and it is what you hear as a uniform delay
   against a hardware instrument.

The output buffer period is set via `player.audio.buffer_frames`:

| Frames | Buffer period at 44.1 kHz | At 48 kHz |
|--------|--------------------------:|----------:|
| 128    | 2.9 ms                    | 2.7 ms    |
| 256    | 5.8 ms                    | 5.3 ms    |
| 512    | 11.6 ms                   | 10.7 ms   |
| 1024   | 23.2 ms                   | 21.3 ms   |

These are the *period*, not the total latency: PortAudio's ALSA backend keeps
several periods in flight, so actual output latency is a small multiple of the
figure above. The latency your device negotiated is printed at start-up:

```
PortAudio output latency: 11.6 ms
```

Tune against that line, not the table. Lower `buffer_frames` (e.g. 128, 64) to
shrink it. The player logs an `Audio xrun` warning when the buffer is too small
for your machine to keep up (audible as clicks), so reduce it until those
appear, then step back up. Leave it unset to let the OS pick.

Two included scripts measure the software parts on your own hardware. MIDI
dispatch:

```bash
python scripts/measure_midi_latency.py --count 1000
```

and per-note handling (selection, variant lookup and render):

```bash
python scripts/measure_handler_timing.py
```

### End-to-end 32-bit float

Every audio sample is converted to float32 immediately after capture and stays in
that format between pipeline stages - analysis, normalisation, pitch shifting,
gain staging, polyphonic mixing. Precision-sensitive operations (IIR filters,
compressor/gate envelope followers, gain curve generation) promote to float64
internally and return float32. The only integer conversion is a single pack to
the hardware's native bit depth at the output, so peak-normalising a quiet
recording or pitch-shifting it does not round it through an integer format on
the way.

### Non-blocking capture

The audio input thread does minimal work and returns immediately. Analysis runs
in a separate worker pool, so capture keeps reading the device while a slow
spectral analysis finishes. This matters most for USB audio devices, which use
isochronous transfers and are sensitive to timing jitter. The pool sizes itself
to what is happening: while the player is live it keeps to a small share of the
cores so the audio thread always has room, while the library scan at start-up,
before anything is playing, spreads across most of the cores in separate
processes.

### Worker headroom

Background analysis runs on most of the machine's cores rather than every one of
them, so the system stays usable while a large library rebuild or `import` runs.
It pulls back further while the player is live. The share is taken from the
cores this process may actually use, so inside a container or under a CPU
allowance it sizes itself to that allowance rather than to the whole host.

Fewer workers does not mean a cooler machine: a modern CPU draws to its power
limit whether the work is spread across four cores or forty, so the package
temperature lands in much the same place either way. What you gain is a machine
that stays usable while it works.

### Gain staging

Every voice is RMS-normalised so a quiet recording and a loud one play at
comparable levels at the same MIDI velocity. A tanh soft-limiter on the mix bus
compresses peaks that approach 0 dBFS, so overlapping voices round off rather
than hard-clip.

### Pitch shifting

Pitch variants are produced with the Rubber Band library's offline finer engine.
Variants are pre-computed in the background by a worker pool, so the shifting
happens before a note is played rather than when it is triggered.

## Similarity engine

Every new sample is scored against every reference using cosine similarity on a
composite feature vector built from five groups: spectral shape, sustained
timbre, timbre dynamics, attack character, and spectral band energy. Each group is independently normalised and scaled by a
configurable weight (`similarity.weight_*`), so you can emphasise whichever
acoustic qualities matter most for your material.

The key insight: **the same comparison method works for both percussive and tonal
sounds without needing to classify them first.** A kick drum naturally scores
high on attack character; a violin scores high on sustained timbre. No
classifier, no training data, no labelling: the comparison is geometry.

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
time); oldest files are evicted when the disk budget is exceeded. Quantised
variants also store a grid energy profile - per-grid-slot RMS energy normalised
to [0, 1] - alongside the audio; the `beat_match` order scorer compares against it.

Samples with detected rhythmic content can be time-stretched to a target tempo
using the `stretch_quantize` processor in a MIDI map assignment. Detected attacks are
snapped to a quantised beat grid and the entire mapping is applied in a single
pass using Rubber Band's offline finer engine. Time-stretch variants are produced
on-demand when an assignment requests them - no global startup cost.

### Attack-accurate onset detection

Standard spectral onset detection (as used by librosa and most audio analysis
tools) identifies the frame where spectral energy changes most rapidly - the
peak of the onset strength envelope. For percussive sounds this peak typically
lags the actual attack by 10-30 ms, which is enough to make beat-quantised
hits sound noticeably off the grid.

Subsample refines each detected onset to sample-accurate precision using a
two-stage approach:

1. **Coarse detection** - librosa's onset detector finds approximate positions
   at frame resolution (~11.6 ms at 44100 Hz / hop 512).
2. **Attack refinement** - for each onset, a short-window amplitude envelope
   (32 samples, ~0.7 ms) is searched backward to find the inter-hit valley
   (quietest point between consecutive transients), then forward to find where
   energy first rises above a set fraction of the local peak. This threshold crossing is
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
# and an editable map template.  The GM reference fingerprints the kit matches
# against ship inside Subsample and are named, so nothing is copied here
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
   or copy audio files (WAV, FLAC, AIFF, OGG, MP3) into
   `samples/captures/`; everything there loads at startup.
2. **Turn the player on.** In `config.yaml`, set `player.enabled: true` and
   your devices: `player.audio.device` (output), and `player.midi_device`
   (or `player.virtual_midi_port` if a sequencer on the same machine will
   drive it). The GM kit map is already wired in by `--init`.
3. **Play MIDI channel 10.** Run `subsample` again and play your controller or
   sequencer on MIDI channel 10 - kick on 36, snare on 38, hi-hats on 42/46.
   Every GM drum note plays whichever of your samples sounds closest to that
   drum, through a pre-mixed channel strip. Load more samples and the kit
   improves.

**Live capture mode:** Subsample lists available audio input devices and lets you
choose one (or auto-selects if only one is present). It calibrates ambient noise
for a few seconds before listening for events.

**File input mode:** Each file is processed at its native sample rate, bit depth,
and audio channel count. Detected segments are saved to the output directory. The
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

```yaml config
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

```yaml config
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

### Naming a device that keeps moving

`recorder.audio.device` and `player.midi_device` take a **glob**: `*` matches any
run of characters, `?` matches exactly one, matching is case-insensitive, and the
pattern is matched anywhere in the name. A pattern with no wildcards is therefore
a plain substring, which is how these fields have always worked.

Wildcards exist because device names carry a number that moves. On Linux an audio
device is named:

```
SC-U: USB Audio (hw:2,0)
        card index ─┘ └─ subdevice
```

The **card index** is assigned by probe order, so plugging in an unrelated
interface renumbers it - the same device can be `hw:0,0` today and `hw:2,0`
tomorrow, and a pinned name silently stops matching. The **subdevice** does not
move.

MIDI has the same shape. `RtMidiIn Client:Subsample Virtual MIDI 129:0` carries an
ALSA sequencer client id (`129`) handed out in registration order, so it differs
between runs, followed by a port number (`:0`) that does not.

So the rule is: **wildcard the number that moves, keep the one that doesn't.**

```yaml config
recorder:
  audio:
    device: "SC-U: USB Audio (hw:*,0)"

player:
  midi_device: "*U6MIDI Pro *:0"
```

Keeping the trailing index matters more than it looks. A multi-port interface
reports one name per port - a 3-port MIDI interface appears as `… 16:0`, `… 16:1`
and `… 16:2` - so `*U6MIDI Pro*` matches all three and asks you which one at
every launch, while `*U6MIDI Pro *:0` names one port for good.

Prefer `*` to `?` for the moving number. `hw:?,0` works until the machine has ten
sound cards, then stops matching with no symptom beyond the device seeming to
disappear.

**When a pattern is ambiguous.** If it matches exactly one device that device is
used silently. If it matches several you are asked which, and the menu lists only
the matches rather than every device on the system. If it matches none, Subsample
reports that and lists what is available.

**A device's full name always pins that one device.** The whole name is tried
before anything else, so pasting it wins even where a longer name contains it -
`default` is also inside `Default Sink` and `Default Source`, and PipeWire
publishes a `.monitor` source for every sink. Shorten the name and you are back
to a substring, which matches everything it appears in.

> Running without a terminal - from a service manager, over SSH without a TTY, in
> CI - there is nobody to answer the menu. Rather than blocking on input forever,
> Subsample reports the candidates and exits, so the log says what happened.

**On Linux with PipeWire, there is a better name available.** PipeWire publishes
its own node names alongside the raw ALSA ones, and they are built from the
device's USB identity rather than from probe order:

```
alsa_output.usb-BEHRINGER_SC-U_0EB571140230AB13-00.multichannel-output
                              └─ the interface's serial number
```

Nothing in that moves - no card index at all - so it survives re-plugging and
reboots without a wildcard for an index:

```yaml config
recorder:
  audio:
    device: "alsa_input.usb-BEHRINGER_SC-U*"
```

The trailing `*` only covers PipeWire's profile suffix. Run
`subsample --list-devices` to see whether these names are present on your
system; if they are, they are the most stable thing to point at.

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

Every setting, with its type, default, limits and an example, is in the
[configuration reference](https://subsystem.co/subsample/configuration/). It is
generated from the code that reads your `config.yaml`, and the shipped
`config.yaml.default` is checked against it on every build.

### Following your sequencer's tempo

`stretch_quantize` and `pad_quantize` snap a sample's onsets to a beat grid at
the session tempo, and the `duration_beats` selection filter measures sample
length in beats at that same tempo. The tempo lives in `tempo.bpm` - a fixed
number in `config.yaml`, so if you change tempo in your sequencer and forget to
change it here, quantised samples keep snapping to the old grid and beat filters
keep measuring against the old tempo, and Subsample stops matching your
sequence without any warning.

Set `tempo.source: midi` and subsample takes the tempo from the MIDI clock
arriving on the player's MIDI input instead:

```yaml config
tempo:
  bpm: 125.0        # fallback, used until a clock is seen
  source: midi      # follow the sequencer
```

Worth knowing:

- **Set `tempo.bpm` anyway.** It's the fallback before any clock arrives (and
  if your sequencer never sends one). At `0.0` nothing quantises until the
  transport starts, and a map that filters by `duration_beats` will not load.
- **The clock must reach the same MIDI input as your notes** (`player.midi_device`
  or your virtual port). Notes on one port and clock on another and it sees
  nothing.
- **The tempo is rounded to whole BPM and only adopted once it holds steady.**
  Every change re-bakes every quantised variant, so a change costs a burst of
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
format is controlled by `recorder.audio.audio_format` - `wav` (uncompressed,
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
- `flac` - lossless compressed (smaller files, and the decoded audio is
  bit-identical).  Works at 16 or 24-bit.

**The rule when formats don't quite line up:**

| Capture scenario | Extension written |
|---|---|
| `audio_format: wav`, any bit depth | `.wav` |
| `audio_format: flac`, live capture at 16 or 24-bit | `.flac` |
| `audio_format: flac`, 32-bit source (e.g. imported file) | `.wav` for that file, with an INFO log explaining why |
| `audio_format: flac` combined with `bit_depth: 32` (live capture) | Rejected at startup - set one or the other |

So if you flip `audio_format: flac` and then process a mix of 16/24-bit and
32-bit source files, you'll see a mix of `.flac` and `.wav` in your output
directory.  This is correct behaviour: the 32-bit fallback preserves full
precision rather than silently truncating.  Live captures share one bit
depth per session, so they stay consistent within a run.

**Existing libraries.** Upgrading to a subsample build with FLAC support does
not touch your existing `.wav` samples - they continue to load unchanged.
No migration, no bulk conversion.  FLAC only affects what gets written for
*new* captures once you flip the flag.

### Sample previews

When `recorder.previews: true` (the default), every captured or imported sample
also produces two visual-preview artefacts alongside the audio and analysis
sidecar:

- **`<sample>.preview.png`** - a fixed 1024x256 raster thumbnail (RGB) for
  browsing the library in an OS file manager.  The layout
  layers a skyline of four spectral bands behind a mirrored waveform envelope, with
  short vertical ticks at each detected onset and (when the sample is
  rhythmic) a dashed beat grid.  Stratum heights scale with each band's
  share of total energy (same four bands as `band_energy.energy_fractions`),
  so a bass-heavy kick looks bottom-heavy at a glance and a cymbal
  looks top-heavy - every band keeps at least a small minimum height
  so its temporal shape stays readable.  A bottom-right badge shows
  pitch (when tonal), BPM (when rhythmic), and duration.
- **A `preview` block embedded in `<sample>.analysis.json`**:
  the same image's inputs (envelopes, spectral band strata, onset/beat times,
  accent colour, badge text) in a compact form.  When a thumbnail goes
  missing, Subsample redraws it from this block at the next start-up instead
  of re-analysing the audio.

> File managers on macOS, Windows, and Linux do **not** treat sibling PNGs as
> the audio file's own icon - the `.preview.png` appears as a separate file
> in the directory listing.  This is deliberate: embedding cover art would
> mutate the audio container, which subsample does not do.  Browse the
> previews alongside the audio files.

Visual design (stroke weights, colours, layout) can be iterated later without
any schema bump - the `preview` block stores the underlying data, not the
rendered output.  Only a change in envelope resolution or spectral band count
requires a `preview.version` bump.  A sample with no `preview` block still
plays back and analyses identically.

Set `recorder.previews: false` to skip both artefacts and save the disk space
they take.

## Instrument sample library

Every recording is automatically added to an in-memory instrument library
alongside its full analysis data. A configurable memory cap prevents unbounded
growth; the oldest samples are evicted when a new one would exceed the limit.
The budget is auto-detected by default (the largest share of the global memory allocation -
see `max_memory_mb` in the configuration table) and can be overridden via
`library.max_memory_mb`. WAV files on disk are never deleted.

### Persistent library across sessions

```yaml config
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
of-house playback rig; or keeping CPU-intensive audio analysis on a
dedicated host.

**Recorder machine** (`config.yaml`):
```yaml config
recorder:
  enabled: true
  directory: "/mnt/shared/samples"

player:
  enabled: false
```

**Player machine** (`config.yaml`):
```yaml config
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

Two limits worth knowing up front: only the file `player.midi_map` names is
watched, so editing a set an ensemble *includes* needs a restart; and watching
relies on filesystem notifications, which do not cross machines - a sample set on
a network drive edited from another machine generates no events at all. Both are
fine for pre-built sets, which do not change while you play.

```yaml config
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
kick drum, snare, hi-hat, etc. Each reference is its `.analysis.json`
fingerprint: the 47 GM references ship inside Subsample as fingerprints alone,
with no audio, which is what lets a map name one and have it resolve on any
machine. A reference is named in a `where: { reference: ... }` predicate:

```yaml map.assignments
- name: Bass Drum
  channel: 10
  notes: 36
  select:
    where:
      reference: GM36_BassDrum1
```

During player startup, each reference is loaded from its sidecar and added to
the similarity matrix. A path still works, for a reference of your own: if a WAV
file exists but its `.analysis.json` sidecar is missing, Subsample generates it
automatically - you can point at any
WAV file as a reference without pre-processing. For every instrument sample,
Subsample computes cosine similarity against every reference and maintains a
ranked list per reference - most similar instrument first. When a sample is
evicted from the instrument
library, it is also removed from the ranked lists.

Reference lookups are case-insensitive.

The GM drum references bundled with Subsample are sidecar files only - the
acoustic fingerprints derived from the FluidR3_GM SoundFont (MIT licence; see
`CREDITS.md` beside them in the package). No audio is distributed: the
fingerprint is all the similarity engine needs, so the kit matches against them
without any reference WAVs on disk. They are not copied into your project -
`subsample --init` leaves them in the package and the scaffolded map names them,
so the map works unchanged wherever you move it.

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

```yaml config
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
Python. Together, they form part of a generative sampler workstation -
Subsequence drives the patterns, Subsample provides the sounds.

The two communicate over standard MIDI. The simplest setup is to give
Subsample a named [virtual MIDI port](#virtual-midi) and have Subsequence send
to it - no hardware MIDI cabling required, no audio routing on the host:

```yaml config
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
| `subsample catalog` | CSV catalogue of every sample's detected properties, with curation aids |
| `subsample analyze` | Analyse audio files and print their detected metrics |
| `subsample similar` | Rank library samples against each reference by similarity |
| `subsample loops` | Find and audition click-free loop points in sustained samples |

(A file argument that shares a command name needs a path prefix:
`subsample ./import` treats it as an input file to chop into the capture
library, while `subsample import` runs the import tool.)

### Analysing recorded files

```bash
subsample analyze samples/2026-03-17_14-32-01.wav
```

Output:
```
rhythm:   tempo=120.2bpm  beats=4  pulses=12  onsets=4
attacks:  4
   1    0.020s    0.0dB
   2    0.510s   -6.0dB
   3    1.000s   -3.5dB
   4    1.490s  -21.0dB
spectral: duration=2.00s  flatness=0.001  attack=0.000  release=0.812  centroid=0.018  bandwidth=0.001  zcr=0.120  harmonic=0.821  contrast=0.310  voiced=0.940  log_attack=0.000  flux=0.312  rolloff=0.451  slope=0.023
pitch:    pitch=440.0Hz  chroma=A  pitch_conf=0.89  stability=0.120st  voiced_frames=86
level:    peak -1.2dBFS  rms -12.6dBFS  crest 11.4dB  floor -42.3dBFS
noisiness: 0.012  (0 = clean event, 1 = wall-to-wall noise)
loop:     0.412s -> 1.187s (775 ms, xfade 30 ms, junction_flux 0.08)
```

Every detected attack is listed with where it starts and how loud it is, in dB
relative to the sample's own peak - the hardest hit reads `0.0dB` and everything
else sits below it. This is what answers "is this take worth quantising": hits
spaced evenly and within a few dB of one another will land on a grid, while one
20 dB down is a ghost note. Each level is measured over the moment after its
attack, stopping at the next one, so a quiet hit just ahead of a loud one keeps
its own level rather than borrowing its neighbour's.

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
  length; near 1 = wall-to-wall unpitched noise (static, a dead radio channel),
  near 0 = a clean hit or pitched tone. Computed as *stationarity* (the signal
  never gets quiet) times *lack of pitch*, so a held tone or a decaying hit both
  score low. Sustained unpitched textures (cymbal rolls, noise sweeps) score
  high too, so use it to sort and audition, not to auto-delete.

Three MFCC timbre fingerprints are stored in the sidecar (used for similarity,
not shown in script output): `mfcc` (mean, average timbre), `mfcc_delta`
(first-order trajectory), and `mfcc_onset` (onset-weighted, attack emphasis).

### Cataloguing a sample directory

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

- **`pitched`** - passes the stable-pitch test, so `pitched: true` selects it in
  a MIDI map and it can be re-pitched across a keyboard range
- **`quantizable`** - has at least 2 detected hits, so `stretch_quantize` /
  `pad_quantize` can align them to a beat grid; below that, quantise degrades
  to a plain stretch or a pass-through
- **`loopable`** - has a steady sustaining region (a held tone, or a stationary
  textural bed) worth looping while a key is held, so `loopable: true` selects
  it. This is a coarse candidate flag - it means "worth trying to loop", not
  "here is the loop point" (that is found later from the audio itself).
  The companion **loop_ms** column shows the actual detected loop length in
  milliseconds; it is blank when a sample is loopable but too evolving to wrap
  seamlessly (no clean loop point was found)

All three columns run exactly the tests the player runs, so the catalogue shows
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
  pointing at whichever was loudest, not measuring a pre-stroke

The pair is intended for a sequencer that wants a note's transient to land *on*
the beat: trigger the note `impact_ms` early and the hit arrives on time with
the preparation intact. Subsample's own playback does not use it - a voice
still starts at the beginning of the sample - so nothing changes in how your kit
sounds. The values also ride in each sample's `.analysis.json` sidecar as
`rhythm.impact_time` (in **seconds**) and `rhythm.impact_pre_level_db`, so any
tool that can read the sidecar can use them without going through the catalogue.

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

Files without an `.analysis.json` sidecar are analysed on the way (slow on the
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
  noise like static or a dead radio channel, near 0 means a clean hit or a
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
Files are silence-trimmed, safety-faded, re-encoded in the configured capture
format (WAV, or FLAC when `recorder.audio.audio_format: flac`), fully
analysed, and saved with sidecar JSON. A large batch is fingerprinted across the
machine's cores in parallel rather than one file at a time.

```bash
subsample import /path/to/samples/*.wav
subsample import --to samples/captures /path/to/sample-pack/*.wav
subsample import --force "/mnt/sdr/audio/2026-01-15/*.wav"
```

- `--to DIR` - target directory (default: `recorder.directory` from config.yaml)
- `--force` - overwrite existing files in target directory

Handles WAV, BWF (Broadcast Wave Format), FLAC, AIFF, OGG, and any other format
supported by libsndfile. BWF and non-WAV sources are re-encoded in the configured
capture format
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

Preview and audition the click-free loops that `mode: loop` playback uses:

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
                                              (auto-scaled: a share of the usable cores)
                                                               ↓
                           to_mono_float() → analyze_all() → WAV + sidecar + SampleRecord
```

The input thread is never blocked waiting for analysis. Back-to-back sounds are
captured reliably even when analysis is slow - worker threads handle each
recording concurrently and independently.

### Similarity engine

Every new instrument sample is scored against every reference using cosine
similarity on a composite vector. The vector is split into five
groups, each independently L2-normalised so that no single group dominates by
scale:

```
Group 1: spectral shape   [flatness, attack, release, centroid, bandwidth, zcr,
                           harmonic, contrast, voiced, log_attack, flux,
                           spectral_rolloff, spectral_slope, crest_factor]
Group 2: sustained MFCC   [mean timbre, coefficients 1-12]
Group 3: delta-MFCC       [timbre trajectory, coefficients 1-12]
Group 4: onset-weighted   [attack character, coefficients 1-12]
Group 5: band energy      [sub-bass/low-mid/high-mid/presence fractions + decay rates]
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
                → TransformCache (parent-priority FIFO eviction, memory-capped)

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
         selects - quantized_beats / beat_match - fall back to a live query)
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
the `apt` instructions above work unchanged. Audio devices need to be exposed to
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

## Type checking

Same setup as [Tests](#tests):

```bash
mypy subsample
```

## Dependencies and credits

Subsample uses these libraries:

| Library | Purpose | Licence |
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

## About the author

Subsample was created by me, Simon Holliday ([simonholliday.com ↗](https://simonholliday.com/)), a senior technologist and a junior (but trying) musician. From running an electronic music label in the 2000s to prototyping new passive SONAR techniques for defence research, my work has often explored the intersection of code and sound.

## Licence

Subsample is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).

You are free to use, modify, and distribute this software under the terms of the AGPL. If you run a modified version of Subsample as part of a network service, you must make the source code available to its users.

All runtime dependencies are permissively licensed and compatible with AGPLv3.

This project is managed with [Subroutine](https://github.com/simonholliday/subroutine).

## Commercial licensing

If you wish to use Subsample in a proprietary or closed-source product without the obligations of the AGPL, please contact [simon.holliday@protonmail.com] to discuss a commercial licence.
