# Architecture

This file is for people changing Subsample's code, and subsystem.co does not
publish it. For using Subsample, see [https://subsystem.co/subsample](https://subsystem.co/subsample).

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
                                              SampleProcessor: a thread per capture
                                              (auto-scaled: a share of the usable cores)
                                                               ↓
                                              analysis worker process (#4667)
                           to_mono_float() → analyze_all() → WAV + sidecar + preview
                                                               ↓
                                              back on the thread: SampleRecord (on_complete)
```

The input thread is never blocked waiting for analysis. Back-to-back sounds are
captured reliably even when analysis is slow - each recording is handled
concurrently and independently.

Converting, analysing and writing a capture run on the session's analysis worker
processes, not on the player's threads (#4667). Analysing captures on those
threads cost the audio 85 dropouts in 30 seconds at 1024-frame buffers, by making
it wait for Python's lock (#4666).

- **The thread's part.** A thread of the `SampleProcessor`'s own takes each
  capture in turn and works out its file's path. It tells the watchers the
  sidecar is the player's own, hands the capture's sound over in a file, and
  waits.
- **The worker's part.** It runs the whole pipeline and hands back the analysis,
  with the sound only when an ambisonic capture was converted. The thread then
  calls `on_complete`.
- **Where worker processes cannot start,** the whole pipeline stays on the
  thread, as before.

Details a change here has to keep:

- **Warm-up takes at least one chunk.** `LevelDetector` counts
  `round(warmup_seconds × chunks per second)` chunks, and its warm-up handler
  runs on the first chunk even when that rounds to 0, so one chunk
  (`recorder.audio.buffer_frames` frames) always seeds the moving averages, a
  hit inside it included.
- **Re-trigger has a floor.** A new hit can split a recording only after
  `max(hold_seconds, _MIN_RETRIGGER_GUARD_SECONDS)`, 0.1 s, so an attack's own
  peak cannot split it.
- **Fades are S-curves** (half-cosine), at a trim's edges and at a crop.
- **Channels are kept end to end.** A multi-channel sample reaches an output
  with fewer channels through the ITU-R BS.775 downmix matrices in
  `channel.py`, the LFE dropped.

## Analysis cache

Each sample's analysis lives in its `.analysis.json` sidecar, keyed by the audio
file's MD5 and `ANALYSIS_VERSION`. When either no longer matches and the audio is
present, the sample is analysed again at load and the sidecar rewritten; a
reference sidecar with no audio beside it is kept as it is, so a version bump
must regenerate the shipped references in the same commit (the recipe is in
`README-AGENTS.md`). A few fields worth knowing: `harmonic` comes from HPSS,
`pitch_conf` is pyin's confidence, the timbre vectors are the keys `mfcc`,
`mfcc_delta` and `mfcc_onset`, and the stable-pitch test's thresholds live on
`analysis.has_stable_pitch`.

Onset detection and attack refinement run with one analysis window (`n_fft`) of
digital silence in front of the sound, and every time is shifted back by it, so
an attack in a file's first moments rises out of silence and is found, and the
onset envelope is not scaled against the sound's own decay (#3880).

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

`SimilarityMatrix` holds only the references a loaded map names
(`player._resolve_path_references` adds them), and keeps a ranked list of the
library's samples for each, updated as samples are added and removed when the
library evicts them. An assignment covering several notes gives them successive
ranks, the first note the best match and the next the second, unless it
repitches.

Every score goes through `similarity._score_matrix`, in float64 and rounded to
nine places, whether the library is scored in one batch at start or a capture is
scored alone. Exact ties are ordinary (a duplicate import, the same kit loaded
twice) and both paths break them by insertion order, so a sample's twin ranks
the same however the two arrived.

## Transform pipeline

```
SampleRecord added to library
    → TransformManager.on_sample_added()
        → enqueue base variant ONLY                 ← float32 peak-normalised copy
            → TransformProcessor: queued in the player, handed out a few at a time
                → render worker processes (one set, shared by every program)
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

Renders run in worker processes of their own, not on the player's threads
(#4667). Python runs one thread's code at a time, so a render on one of the
player's threads made the audio callback and the note handlers wait their turn
however many cores were free. At 1024-frame buffers, compressing or gating
sounds cost the audio over 200 dropouts in 30 seconds, and the same renders in
another process cost none (#4666).

- **Who does what.** Each program's `TransformProcessor` keeps its queue, its
  failures and its callbacks in the player, on a results thread of its own. It
  hands the session's render processes, (cores − 2) / 2 of them shared by every
  program, a few renders at a time.
- **How a sound crosses.** It goes through a file in a private scratch folder
  (`parallelism.write_audio` and `read_audio`), which copies it with Python's
  lock released. Pickling would hold the lock for the whole copy: a 35 MB
  sound pickled cost 157 dropouts in 20 seconds, and the same through a file
  none.
- **Logs.** What a worker logs comes back with its render and is logged by the
  player. A once-only warning is logged once in the session.
- **The disk cache.** Workers write to it, the player counts what they write,
  and one worker at a time trims the folder to its budget.
- **Where worker processes cannot start,** renders run on threads as before,
  and the player warns once that they may make it drop out. A temporary folder
  whose path is too long for the forkserver's socket is one such case.

A render that fails leaves its note playing a previous or the base variant, and
is not tried again until a pause has passed: 30 seconds, doubling with each
repeat up to ten minutes, and forgotten once a render works. A failure can pass
(memory, a full `/tmp` under Rubber Band, a killed subprocess) or recur on every
attempt, and nothing tells the two apart, so the pause bounds the cost of
either. The first failure logs its traceback, and a repeat logs one line.

## Playback path

```
MIDI note_on, or an OSC /note/on (play_osc_note, at its bundle's time)
    → _resolve_sample_id: indexed pick from the pre-computed candidate cache
        (rebuilt when the library changes, not per-trigger; variant-state
         selects - quantized_beats / beat_match - fall back to a live query;
         a round-robin pick takes the layer's next turn from _pick_turns,
         kept per channel, note and assignment until the rules change)
    → transform_manager.get_variant(sample_id, spec, from_disk=False)  → processed variant
        (memory cache, else enqueue: the render worker loads it from the disk
         cache or renders it, and this note falls back to a previous/base variant)
    → transform_manager.get_base()     → base variant (all samples)
    → _float_audio()                   → on-the-fly fallback (first trigger only)
    → _note_gain(): gain · velocity² · anti-clip ceiling, from the render's
        measured level and true peak, never from its audio
    → append _Voice (the render itself, its gain and mix matrix, stamped
        with the note-on's arrival time, or for an OSC note in a bundle,
        that time less where the sound's hit comes: _hit_start)
    ↓
PyAudio callback (PortAudio high-priority thread)
    → _advance_span: the arrival times this buffer plays, one buffer back
    → sum all active voices (float32 addition), each from the frame its
        note-on's arrival falls on, each release from its own frame; a
        voice gains and mixes only the frames it plays (_Voice.frames)
    → clip to [-1, 1]
    → float32_to_pcm_bytes(mixed, output_bit_depth)  → int16/24/32 bytes to hardware
```

All mixing happens in float32; precision-sensitive DSP (IIR filters, envelope
followers) promotes to float64 internally. The only integer conversion is the
final output packing. Multiple simultaneous voices are summed correctly regardless of the
output device's bit depth.

A note-on does no work that grows with its sound's length (#4654). A voice holds
the cached render as it is, never a copy, with its gain and mix matrix, and the
callback gains and mixes each buffer's frames as it plays them: at most one
buffer per voice, the same arithmetic the whole render once got at note-on, a
slice at a time. Everything the gain needs is measured on the render worker
(`transform._measure`): the render's true peak, its loudest sample in any
channel, for the anti-clip ceiling, and each segment's peak and level. Gaining
and mixing a 23-second open hi-hat whole for 8 outputs took about 30 ms on the
MIDI thread, so every note that arrived behind it was late. A quantised sound
held in `mode: loop` is laid out over its bars the same way, a buffer at a time
(`_ring_layout`). Only the last-resort fallback still converts a whole sample on
the note-on, for a fresh capture whose base variant is not ready.

## Performance

### Pre-rendered playback

When a sample enters the library, a background worker produces a pre-rendered
copy at the output device's sample rate and format. Tonal samples also receive
a set of pitch-shifted variants. When a MIDI note arrives, its voice plays the
prepared audio as it is, gaining and mixing each buffer's frames as they play,
rather than calculating anything on the spot. If a variant is not ready yet, the
player falls back in this order:

1. **Process variant** - pre-computed with the full declared chain (pitch, filter, saturate, reverse, time-stretch, etc.)
2. **Base variant** - pre-normalised, no DSP (all samples)
3. **On-the-fly render** - used only on the first trigger, before any variant exists

### MIDI dispatch model

Incoming MIDI is dispatched in callback mode: rtmidi delivers each message
to Subsample's handler on its own dedicated thread as it arrives. There is no
polling loop, so there is no fixed input-latency floor. On the output side,
PortAudio's ALSA backend keeps several periods of `buffer_frames` in flight, so
the delay a note meets is a few buffers, not one.

Every message waits for the one before it, so the handler does nothing that can
block (#4488), and nothing that grows with a sound's length (#4654). A note-on
looks variants up in memory only: one that is still on disk is loaded by a
render worker, and that note plays its fallback. A Program
Change switches the program and installs its rules at once, under a lock held
only for those swaps. Ranking the new program's samples and preparing its
variants happen on the re-evaluation worker that CC changes use, and until it
has finished, each note ranks its candidates as it plays. If the new program's
rules fail there, the worker switches back to the previous program.

The OSC receiver takes `/sample/import` messages on one thread and imports them
on another, one at a time and in order, with at most 64 waiting. At stop it drops
those still waiting and gives the one being analysed ten seconds to finish; one
that finishes later is not added, since shutdown has begun.

A second OSC receiver, `OscNoteReceiver` on its own port, takes `/note/on` and
`/note/off` (#3610, #603). It reads each packet itself, walking bundles with
python-osc's `OscBundle`, because python-osc's dispatcher sleeps on the server
thread until a bundle's time and then hands the message on without it, and its
`OscPacket` gives a message on its own and one whose time has passed the same
time. Each note goes to `MidiPlayer.play_osc_note` with its bundle's time, or its
arrival for a message on its own or a bundle whose time has passed, and whether
it came in a bundle with a time of its own (`timed`, for impact timing below).
The player turns that wall-clock time into its own clock, so the note is handled
as a MIDI note arriving then would be. `_handler_lock`
serialises the receiver's thread with the MIDI thread, since `_handle_message`
is written for one at a time; a MIDI message is stamped before it waits. The
velocity, 0 to 1, picks a velocity layer scaled to 0-127 and reaches the gain
and a velocity pick unrounded. A note timed ahead is handled at once, and its
voice waits in `_voices` until its time comes round.

### Note timing

Every note plays one buffer after it arrived, at the frame within that buffer
its arrival falls on (#600), as a DAW places MIDI, so notes keep the spacing
they were played with. Before, each began at the next buffer boundary, re-timed
onto a grid one buffer coarse: 21 ms at 1024 frames. The cost is half a buffer
of latency on average, and one at most.

`_safe_handle_message` stamps each message with the player's clock
(`time.perf_counter`) before any work, so a slow handler does not make its note
late. Each voice a message starts records the time as `_Voice.starts_at`; a
note-off, a same-note steal, a choke and CC 120 and 123 record theirs as
`releases_at` (`_release`, which keeps the first). The audio callback moves a
span of arrival times on by exactly one buffer per call (`_advance_span`), so
a callback early or late by scheduling jitter moves no note, and pulls it 5% of
the way towards its own clock each call so the audio device's clock and the
player's never drift apart; a callback more than a buffer away, the first or one
after a stall, starts the span afresh. `_frame_of` places a time in the span: one
from before it plays at frame 0, late rather than lost, and one after it waits
for the next buffer. A voice with no time, as a test passing a message straight
to `_handle_message` gives, starts at frame 0 as before.

An OSC note in a bundle takes the bundle's time as its arrival (#603), so a
sender that sends ahead has its notes played on their frames however the
network delays the packets, as long as the two machines' clocks agree. A
note-off timed ahead releases its voice at its own frame in the same way, and a
note more than two seconds ahead is warned about, at most once a minute.

Such a note also starts its sound early, by where the sound's hit comes, so the
hit lands on the note's time (#604, `_hit_start`). The sequencer cannot do this
itself, since Subsample picks the sample. Where the hit comes is measured on the
audio a voice will play, after the process chain: each render records
`TransformResult.hit_time`, and `segment_hit_times` for a quantised one's
segments, measured on the render worker (`transform._measure`) as it renders or
reads a render from disk, so a note-on only reads it. The raw-sample fallback
reads the sample's own analysis. `analysis.hit_time` turns an impact into a hit
only after a lead-in at least 10 dB quieter (`HIT_LEAD_IN_DB`): the impact marks
where the loudest moment begins, which for an open hi-hat's bloom or a field
recording is not a hit, and measured on 5387 samples only 234 move. A MIDI note,
an OSC note on its own and an assignment with `align: start` never move. A note
that arrives with less lead than its hit needs keeps its early start, which
`_frame_of` places at the next buffer's first frame, so the whole sound plays
and the hit lands late by the shortfall. `_warn_hit_late` says so at most once a
minute, naming the largest lead any timed sound has needed (`_hit_lead_needed`).

A callback that runs early, before a message in its span has arrived, plays that
message at the start of the next buffer: one buffer of delay cannot absorb
jitter beyond the span. Simulated with callbacks jittering by 1 ms either way at
256 frames, hits 7 ms apart played 7 ms apart within 0.05 ms on average and
0.8 ms at worst, against 2.3 ms and 3.7 ms before. A looping voice released
mid-buffer stops looping at that buffer's start, not at its release frame, so it
may reach its tail up to a buffer early, under the fade.

### Native dependencies

On Linux, PyAudio and python-rtmidi compile from source as they install, which
is why the install needs a toolchain and the PortAudio and ALSA headers.
pyrubberband runs the Rubber Band command-line tool rather than linking a
library. `libportaudio` links JACK unconditionally, so PortAudio probes for a
sound server at start and aborts with `Unanticipated host error` (-9999) when
none runs, even though Subsample never asks for JACK.

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
in separate worker processes, so capture keeps reading the device while a slow
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

Free cores are not enough on their own: work on the player's own threads waits
for Python's lock with the audio, however many cores are idle. That is why
renders and the analysis of captures run in worker processes (see
[Transform pipeline](#transform-pipeline) and
[Live capture pipeline](#live-capture-pipeline)).

Fewer workers does not mean a cooler machine: a modern CPU draws to its power
limit whether the work is spread across four cores or forty, so the package
temperature lands in much the same place either way. What you gain is a machine
that stays usable while it works.

### Start-up checks and real-time priority

When the player starts, `subsample.performance` looks once at what on the
machine can cost it its timing, and logs what to change (#4659). Nothing is read
from the system inside the audio callback.

- **Power saving.** On Linux, under the `intel_pstate` and `amd-pstate-epp`
  drivers the governor reads `powersave` in the ordinary balanced mode, so the
  energy-performance preference decides: `power` or `balance_power` warns. Under
  any other driver the `powersave` governor holds the lowest clock, and warns. On
  macOS, Low Power Mode warns, read from `pmset -g`.
- **Memory budgets.** `library.max_memory_mb` and `transform.max_memory_mb`,
  counted once for each program, since each has its own library and render
  cache, warn past three quarters of the machine's memory (`MEMORY_SHARE`).
- **The variant cache.** It warns on a file system held in memory (tmpfs or
  ramfs: Fedora's `/tmp`, and Debian's from 13), and on a disk whose free space
  and the cache's own files together fall short of `max_disk_mb`. A full `/tmp`
  also breaks Rubber Band, which writes its temporary files there.
- **Real-time priority, on Linux.** PortAudio's ALSA backend asks for real-time
  scheduling only when the application does, and PyAudio never does, so the
  audio callback ran at ordinary priority. Its thread now asks for `SCHED_FIFO`
  at 70 on its first callback, and the MIDI input thread and the OSC note
  receiver ask for 69 on their first call, since a note is timed from when its
  handler starts. Each is capped at the user's `RLIMIT_RTPRIO`
  (`MidiPlayer._promote_thread`, `performance.promote_this_thread`), and only the
  calling thread changes. The main thread never asks, since the tests drive the
  handlers from it. The audio thread logs once what it got, and when refused,
  how to allow it. On macOS, CoreAudio and CoreMIDI already run their threads at
  real-time priority.

The xrun warning then names what costs the machine headroom: a CPU found saving
power at startup, and audio left at ordinary priority.

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

## Watchers

**The library watcher** (`watcher.InstrumentWatcher`) runs two paths over the
top level of `library.directory`, or of each program's directory when the map
declares `programs:`, whatever `library.directory` says
(`cli._start_library_watchers`). A new `.analysis.json` loads its sample after
a 1 s debounce, since the recorder writes the audio first and the sidecar's
arrival means both are complete. A new audio file waits a 2 s debounce, then a
5 s grace for a sidecar from another Subsample, then a 2 s check that its size
has stopped changing, and only then is analysed, its sidecar written and the
sample loaded, so a bare file plays about ten seconds after it lands. Both
paths retry a file still being written.

**A folder on a network drive is polled.** Both watchers otherwise hear of a
change through the operating system's notice of it, and a network drive gives
none for a file another machine writes (#3887). `mounts.network_filesystem`
tells a network drive by its file system type, from `/proc/self/mounts` on
Linux (cifs, smb3, nfs, nfs4, fuse.sshfs, and WSL's 9p and drvfs) and `statfs`
on macOS (smbfs, nfs, afpfs, webdav), and a folder on one is listed every 2 s
by `watcher._NetworkDriveObserver` instead, with no setting (#3972). A folder
whose type cannot be told counts as local. watchdog's own polling emitter
stops for good at the first listing that fails, and reads a folder that has
gone as empty, reporting every file in it deleted; `_NetworkDriveEmitter`
keeps the last listing while the folder cannot be read on the drive it was
on, so a dropped connection or an unmounted share leaves the library as it
was, and the next good listing reports only what changed meanwhile. The map
watcher keeps one observer of each kind and puts each directory on the one it
needs.

**The map watcher** (`MidiMapWatcher`) fires half a second after the last save
of any file the map is read from. The loader records those files as
`MidiMapResult.source_files`: the map, each set an ensemble includes or
`player.midi_maps` names, and every `definitions:` file any of them mounts, but
not a `map:` preset's, since programs take a restart (#389). The reload parses
the map again and swaps the active note map, keeping the old one when the new
one fails to parse or validate. Once it parses, the watcher follows the files it
was read from this time, so a set added to an ensemble is watched from then on.

## Programs and ensembles

A `programs:` block loads every program at start as a `bank.Bank`: its own
`InstrumentLibrary`, `SimilarityMatrix` and `TransformManager`. `BankManager`
holds one active bank for the whole player, so a Program Change switches the kit
on every MIDI channel (decision #3974). A switch installs the new program's
rules at once, on the MIDI thread; the re-evaluation worker then ranks its
samples against them, and switches back to the previous program if they fail
(#4488). A `map:` preset's rules are loaded with the map that names it
(`player.load_preset_map`, #3886). The switch, and a map reload's check for a
live preset and refresh of the top-level rules, take the player's
`_publish_lock`, so they never interleave. A reload that finishes validating an
edit after a switch to a `directory:` program gives that program the edit,
rather than leaving it the rules from before it.

An ensemble file's `maps:` block and `player.midi_maps` reach the same loader,
`player.load_ensemble`. Includes are flat, one level, so there are no cycles to
detect.

A set shared between projects, or kept on a network drive, carries a cost
worth knowing before designing around it. Editing it is not the cost: a set on a
network drive is polled (see Watchers), so an edit from another machine reloads
it. But an `ANALYSIS_VERSION` bump makes every project that
loads the set analyse its samples again, perhaps over the network, perhaps
several at once, and onto a share that may be read-only, so the new sidecars
cannot be written and the work repeats at every start. Publishing sidecars with
a set, rather than regenerating them on demand, would answer the second if it
comes to matter (Simon's note on #389).

## Sample previews

Visual design (stroke weights, colours, layout) can be iterated later without
any schema bump - the `preview` block stores the underlying data, not the
rendered output: the waveform envelopes, the four spectral bands' energy
envelopes and shares (the strata heights come from
`band_energy.energy_fractions`, each with a minimum), the onset and beat
times, the accent colour and the badge's text.  Only a change in envelope
resolution or spectral band count requires a `preview.version` bump.  A sample
with no `preview` block still plays back and analyses identically.
