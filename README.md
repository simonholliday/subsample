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
| **MIDI control** | Note on/off, Program Change for programs, CC binding for any processor's numeric parameter |
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
| **Platform** | Linux, macOS, Windows (via WSL) |


## Documentation

**Full documentation: [https://subsystem.co/subsample](https://subsystem.co/subsample)**

- Guide: [https://subsystem.co/subsample/guide](https://subsystem.co/subsample/guide)
- Configuration reference: [https://subsystem.co/subsample/configuration](https://subsystem.co/subsample/configuration)
- MIDI map reference: [https://subsystem.co/subsample/midi-map](https://subsystem.co/subsample/midi-map)
- Command-line reference: [https://subsystem.co/subsample/command-line](https://subsystem.co/subsample/command-line)

The guide builds one project, from a first recording to playing live. The
references are generated from the code that reads these files and runs these
commands, so they describe the release the guide installs rather than a copy
kept by hand.

For changing Subsample's code, see [docs/architecture.md](docs/architecture.md)
and [CONTRIBUTING.md](CONTRIBUTING.md).

## Installing

Subsample is fetched from GitHub with Git, and some of its libraries compile
as they install, which needs a compiler, Python's own headers and a few
development packages. It also calls on PortAudio for sound and on a
time-stretching tool. Install them all first:

```bash
# Debian and Ubuntu
sudo apt install git build-essential python3-dev pkg-config portaudio19-dev libasound2-dev libjack-jackd2-dev rubberband-cli
# Fedora
sudo dnf install git gcc-c++ make python3-devel pkgconf-pkg-config portaudio-devel alsa-lib-devel jack-audio-connection-kit-devel rubberband
# macOS, with Homebrew
brew install portaudio rubberband
```

On macOS, the compiler and Git come with Xcode's command-line tools. On
Linux, a sound server must be running: a desktop runs one already, and the
guide's [install chapter](https://subsystem.co/subsample/guide/installing-subsample)
shows how to add one to a machine without a desktop.

Then install Subsample with [uv](https://docs.astral.sh/uv/), using the line in
the guide's [install chapter](https://subsystem.co/subsample/guide/installing-subsample),
which names the release the guide documents. The name `subsample` on the
Python Package Index belongs to another project, so install from that line
rather than by name. Then make a project in a new folder:

```bash
mkdir my-project && cd my-project
subsample --init
```

From here, the [guide](https://subsystem.co/subsample/guide) makes a first kit
from a recording.

## Dependencies and credits

Subsample uses these libraries:

| Library | Purpose | Licence |
|---------|---------|---------|
| [PyAudio ↗](https://people.csail.mit.edu/hubert/pyaudio/) | Audio device I/O (PortAudio bindings) | MIT |
| [PyYAML ↗](https://github.com/yaml/pyyaml) | YAML config loading | MIT |
| [NumPy ↗](https://numpy.org/) | Numerical array operations | BSD-3-Clause |
| [librosa ↗](https://librosa.org/) | Audio analysis (spectral, rhythm, pitch) | ISC |
| [SciPy ↗](https://scipy.org/) | Signal processing (onset detection, filtering) | BSD-3-Clause |
| [SoundFile ↗](https://python-soundfile.readthedocs.io/) | Audio file reading and writing | BSD-3-Clause |
| [Pillow ↗](https://python-pillow.org/) | PNG waveform preview rendering | MIT-CMU |
| [mido ↗](https://github.com/mido/mido) | MIDI message parsing and I/O | MIT |
| [python-rtmidi ↗](https://github.com/SpotlightKid/python-rtmidi) | MIDI device access (RtMidi bindings) | MIT |
| [pyrubberband ↗](https://github.com/bmcfee/pyrubberband) | Pitch shifting and time-stretching (Rubber Band wrapper) | ISC |
| [watchdog ↗](https://github.com/gorakhargosh/watchdog) | Watching the sample directory and the MIDI map for changes | Apache-2.0 |
| [PyMidiDefs ↗](https://github.com/simonholliday/PyMidiDefs) | MIDI constant definitions (notes, CC, drums, GM) | MIT |
| [threadpoolctl ↗](https://github.com/joblib/threadpoolctl) | Thread limits for numerical work, so analysis does not disturb the audio | BSD-3-Clause |
| [python-osc ↗](https://github.com/attwad/python-osc) | Open Sound Control messages with other software (optional) | Unlicense |

The 47 General MIDI reference fingerprints are derived from the FluidR3_GM
SoundFont (MIT licence), and no audio from it is included;
`subsample/data/reference/CREDITS.md` has the details.

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
