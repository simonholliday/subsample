# Architecture

This file is for people changing Subsample's code, and subsystem.co does not
publish it. For using Subsample, see [https://subsystem.co/subsample/](https://subsystem.co/subsample/).

Subsample is built around three concurrent pipelines that interact through
thread-safe shared state.

## Live capture pipeline

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

## Similarity engine

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

## Transform pipeline

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

## Playback path

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
polling loop, so there is no fixed input-latency floor.

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

## Sample previews

Visual design (stroke weights, colours, layout) can be iterated later without
any schema bump - the `preview` block stores the underlying data, not the
rendered output.  Only a change in envelope resolution or spectral band count
requires a `preview.version` bump.  A sample with no `preview` block still
plays back and analyses identically.
