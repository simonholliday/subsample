"""A declared schema of config.yaml, for documentation.

subsample.config builds the configuration by recording which keys it reads
(its _KeyTracker), and that stays the runtime source of truth: nothing here
validates a config or changes what loads.  This module declares the same keys
a second time, with the type, default, limits, and prose a reader needs, and
returns them as a JSON Schema that a documentation site can render.

A second list can drift from the first, so tests/test_config_schema.py is the
load-bearing half of this module.  It fails the suite when a key is declared
here but not read by the builder, or read by the builder but not declared
here, when a declared default disagrees with what the builder produces from
the shipped config.yaml.default, and when a value outside a declared limit
still loads.

The descriptions are written for musicians reading a reference page: what a
setting controls, its unit, when to change it, and how to turn it off.  They
are published as they stand, so they use British spelling and never an em
dash.
"""

import inspect
import typing


_ABSENT: typing.Final = object()
"""Marks a setting whose default is derived at load time rather than fixed."""


def json_schema () -> dict[str, typing.Any]:

	"""Return the JSON Schema of config.yaml, in the order config.yaml.default declares it."""

	# Imported here so reading the schema does not pay for numpy and scipy
	# until someone asks for it; the constants keep the enums in one place.
	import subsample.ambisonic
	import subsample.config

	return {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"title": "Subsample configuration",
		"description": _text("""
			Subsample reads `config.yaml` from the directory it runs in, or the file
			named with `--config`, and lays it over the defaults in the
			`config.yaml.default` file it ships.  A file needs only the settings it
			changes.  Relative paths resolve against the directory Subsample runs in,
			and a key Subsample does not read is reported by name at start-up and
			ignored.
		"""),
		"type": "object",
		"properties": {
			"max_memory_mb": _setting(
				"number",
				"""
				Total memory, in MB, that Subsample may use for its sample caches.
				`null` sets the budget from the machine's RAM, with a ceiling.
				Subsample divides the budget between instrument samples, transform
				variants, and the vocoder carrier cache, and sizes the variant disk
				cache from it.  `library.max_memory_mb`, `transform.max_memory_mb`,
				and `transform.max_disk_mb` override their own share.  Set a value
				on a machine that runs other memory-hungry software.
				""",
				default=None,
				nullable=True,
				exclusiveMinimum=0,
			),
			"osc": _section(
				"""
				Open Sound Control (OSC): Subsample sends a message when a sample is
				captured or loaded, and can receive requests to import audio files.
				Requires the optional `python-osc` package.
				""",
				{
					"enabled": _setting(
						"boolean",
						"""
						Turns OSC on.  When `false`, Subsample creates neither the
						sender nor the receiver, whatever the other `osc` settings say.
						""",
						default=False,
					),
					"send_host": _setting(
						"string",
						"Host that outgoing `/sample/captured` and `/sample/loaded` messages go to.",
						default="127.0.0.1",
					),
					"send_port": _setting(
						"integer",
						"UDP port that outgoing OSC messages go to.",
						default=9000,
					),
					"receive_enabled": _setting(
						"boolean",
						"""
						Starts an OSC receiver that imports the audio file named in
						each `/sample/import` message into the library.  Has no effect
						unless `enabled` is `true`.
						""",
						default=False,
					),
					"receive_port": _setting(
						"integer",
						"UDP port the OSC receiver listens on.",
						default=9002,
					),
					"receive_host": _setting(
						"string",
						"""
						Network interface the OSC receiver binds to.  The receiver
						reads whatever path a message names, without authentication,
						so the default accepts messages from this machine only.  Set
						`0.0.0.0` to accept them from other hosts on a network you
						control.
						""",
						default="127.0.0.1",
						examples=["0.0.0.0"],
					),
				},
			),
			"ambisonic": _section(
				"""
				How Subsample rotates and decodes first-order ambisonic samples at
				playback.  Applies only to samples captured with
				`recorder.audio.ambisonic_format` set, which Subsample stores as
				AmbiX B-format, and applies to every such sample in the project.
				""",
				{
					"decoder": _setting(
						"string",
						"""
						Decoder weighting that maps B-format onto the output speaker
						layout.  `basic` uses flat weights, with sharp lobes and the
						best low-frequency behaviour.  `max_re` narrows the front lobe
						for better high-frequency localisation.  `inphase` gives the
						softest lobes with no back lobes, for listening away from the
						sweet spot.
						""",
						default="basic",
						enum=sorted(subsample.ambisonic.SUPPORTED_DECODER_TYPES),
					),
					"yaw_degrees": _setting(
						"number",
						"""
						Rotation of the sound field about the vertical axis, in
						degrees, applied before decoding.  Positive values turn it
						anticlockwise seen from above, moving a sound at the front
						towards the left.
						""",
						default=0.0,
					),
					"pitch_degrees": _setting(
						"number",
						"""
						Rotation about the left-right axis, in degrees, applied before
						decoding.  Positive values tilt the front downwards, moving a
						sound at the front towards below.
						""",
						default=0.0,
					),
					"roll_degrees": _setting(
						"number",
						"""
						Rotation about the front-back axis, in degrees, applied before
						decoding.  Positive values tilt the field to the right, moving
						a sound on the left towards above.
						""",
						default=0.0,
					),
					"max_order": _setting(
						"integer",
						"Ambisonic order to decode.  Only first order is supported.",
						default=1,
						enum=[subsample.ambisonic.AMBISONIC_ORDER_SUPPORTED],
					),
				},
			),
			"recorder": _section(
				"""
				Live capture from an audio input.  Subsample tracks the room's
				background level, starts a recording when a sound rises above it,
				and writes each sound to its own file.
				""",
				{
					"enabled": _setting(
						"boolean",
						"""
						Turns live capture on.  Set `false` to process only the files
						named on the command line, or to run the player alone.
						""",
						default=True,
					),
					"directory": _setting(
						"string",
						"""
						Directory that recordings are written to, absolute or relative
						to the directory Subsample runs in.  Subsample creates it when
						it is missing.  The default is also the default
						`library.directory`, so every recording joins the playable
						library.
						""",
						default="samples/captures",
					),
					"filename_format": _setting(
						"string",
						"""
						Name of each recording, as a `strftime` format without an
						extension.  `%3f` adds zero-padded milliseconds.  The time is
						the moment the recording ended, so a format without `%3f` gives
						two recordings that end in the same second the same name, and
						the later one replaces the earlier.
						""",
						default="%Y-%m-%d_%H-%M-%S-%3f",
						examples=["%Y%m%d_%H%M%S"],
					),
					"previews": _setting(
						"boolean",
						"""
						Writes a visual preview of every captured or imported sample:
						a `.preview.png` thumbnail beside the audio, and the data it is
						drawn from inside the `.analysis.json` sidecar.  Set `false` to
						write neither.
						""",
						default=True,
					),
					"audio": _section(
						"The input device, and the format Subsample captures in.",
						{
							"device": _setting(
								"string",
								"""
								Input device to capture from, matched against device names
								without regard to case.  The value is a pattern: `*`
								matches any run of characters, `?` matches one, and the
								pattern may match anywhere in the name.  Use `*` for a
								number the system assigns at each start, such as the card
								index in `hw:2,0`.  When the pattern matches several
								devices, Subsample asks which to use.  `null` selects the
								only input device when there is one, and otherwise asks.
								`subsample --list-devices` prints the names.
								""",
								default=None,
								nullable=True,
								examples=["Samson Go Mic", "SC-U: USB Audio (hw:*,0)"],
							),
							"sample_rate": _setting(
								"integer",
								"Capture sample rate, in Hz.  The input device must support it.",
								default=44100,
								exclusiveMinimum=0,
							),
							"bit_depth": _setting(
								"integer",
								"Bits per sample, for capture and for the files Subsample writes.",
								default=16,
								enum=[16, 24, 32],
							),
							"channels": _setting(
								"integer",
								"""
								Number of input channels to capture.  `null` uses the
								channel count the selected device reports.  Must be `4`
								when `ambisonic_format` is set.
								""",
								default=None,
								nullable=True,
								exclusiveMinimum=0,
							),
							"input": _setting(
								{"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
								"""
								Physical inputs to record from, numbered from 1 as the
								interface labels them.  `null` records from the first
								inputs, as many as `channels`.  The list's length must
								match `channels`, and sets it when `channels` is `null`.
								""",
								default=None,
								nullable=True,
								examples=[[3, 4]],
							),
							"buffer_frames": _setting(
								"integer",
								"""
								Frames the input device delivers at a time.  Smaller values
								lower input latency, and larger ones use less CPU.
								""",
								default=512,
								exclusiveMinimum=0,
							),
							"audio_format": _setting(
								"string",
								"""
								File format for captured and imported samples.  `wav` is
								uncompressed and supports 16, 24, and 32-bit.  `flac` is
								lossless and smaller, and supports 16 and 24-bit: under
								`flac`, a 32-bit imported file is still written as `.wav`,
								and live capture with `bit_depth: 32` is refused at
								start-up.  Samples already in the library load whatever
								this is set to.
								""",
								default="wav",
								enum=["wav", "flac"],
							),
							"ambisonic_format": _setting(
								"string",
								"""
								Treats the four input channels as ambisonic, and stores
								each recording as first-order AmbiX B-format.  `a_generic`
								is tetrahedral A-format with the capsules in the order
								front-left-up, front-right-down, back-left-down, and
								back-right-up.  `a_nt_sf1` is the Rode NT-SF1's A-format,
								with capsule-matching correction.  `b_fuma` is B-format in
								FuMA order, which Subsample reorders and renormalises.
								`b_ambix` is B-format already in AmbiX order, stored as it
								is.  `null` turns ambisonic capture off.  Requires
								`channels: 4`.
								""",
								default=None,
								nullable=True,
								enum=sorted(subsample.ambisonic.SUPPORTED_AMBISONIC_FORMATS),
							),
							"float_import_ceiling_dbfs": _setting(
								"number",
								"""
								Highest peak level, in dBFS, for 32-bit float and 64-bit
								audio that Subsample reads from the command line, a MIDI
								map's `directory:` or `path:`, an OSC import, or the
								library watcher.  A file whose peaks go above it is scaled
								down as a whole, which keeps its dynamics, instead of
								clipping.  Integer audio is never changed.  `null` clips
								such peaks instead.
								""",
								default=-1.0,
								nullable=True,
								maximum=0,
							),
						},
					),
					"buffer": _section(
						"The rolling buffer that holds incoming audio while a recording is open.",
						{
							"max_seconds": _setting(
								"integer",
								"""
								Length of the rolling capture buffer, in seconds.  A
								recording can be no longer than this, so set it above the
								longest sound you expect to capture.
								""",
								default=60,
								exclusiveMinimum=0,
							),
						},
					),
				},
			),
			"player": _section(
				"""
				MIDI-triggered playback.  Subsample opens a MIDI input and an audio
				output, picks a sample for each incoming note by the rules in the
				MIDI map, and mixes the voices as they play.
				""",
				{
					"enabled": _setting(
						"boolean",
						"Turns the MIDI player on.  Requires `midi_map` or `midi_maps`.",
						default=False,
					),
					"max_polyphony": _setting(
						"integer",
						"""
						Number of voices at full velocity that together reach full
						scale.  Each voice's level is divided by it, and voices beyond
						it still play rather than being cut off.  Raise it when dense
						passages clip, and lower it for louder individual notes.
						""",
						default=8,
						minimum=1,
						maximum=64,
					),
					"limiter_threshold_db": _setting(
						"number",
						"""
						Level, in dBFS, above which the output limiter starts to
						soft-clip.  Signals below it pass unchanged, and lower values
						compress more.  `0` turns the limiter off.
						""",
						default=-1.5,
						minimum=-12,
						maximum=0,
					),
					"limiter_ceiling_db": _setting(
						"number",
						"""
						Highest output level, in dBFS, that the limiter allows.  Must be
						above `limiter_threshold_db`, and is ignored when the limiter
						is off.
						""",
						default=-0.1,
						minimum=-12,
						maximum=0,
					),
					"midi_map": _setting(
						"string",
						"""
						Path to the MIDI map that decides which samples each note
						plays.  The player does not start without it or `midi_maps`.
						The map may be an ensemble, declaring a `maps:` block that
						binds several sample sets to MIDI channels.  Cannot be set
						together with `midi_maps`.
						""",
						default=None,
						nullable=True,
						examples=["midi-map-gm-drums.yaml"],
					),
					"midi_maps": _setting(
						{"type": "object", "additionalProperties": {"type": "string"}},
						"""
						Sample sets to play at once, each a MIDI channel from 1 to 16
						followed by the path to a sample set's MIDI map, as in
						`10: kits/home/midi-map.yaml`.  Gives the same result as an
						ensemble map declaring the same `maps:` block.  Use an ensemble
						map instead to name channels from a `definitions:` file, or to
						reload the bindings while running, since Subsample does not
						watch `config.yaml`.  Cannot be set together with `midi_map`.
						Pair it with `library.directory: null` to load only the samples
						the sets name.
						""",
						default=None,
						nullable=True,
					),
					"watch_midi_map": _setting(
						"boolean",
						"""
						Reloads the MIDI map when its file changes, so an edit takes
						effect on the next note without a restart.  Several saves in
						quick succession count as one change.  Requires `midi_map`.
						""",
						default=False,
					),
					"strict_midi_map": _setting(
						"boolean",
						"""
						Refuses a MIDI map with an unknown `where:` key, processor,
						processor parameter, CC binding key, or order entry key, or a
						`pitched:` value that is not `true` or `false`, and names the
						valid options.  Set `false` to log a warning for each and carry
						on, when loading an older map.
						""",
						default=True,
					),
					"midi_device": _setting(
						"string",
						"""
						MIDI input to play from, matched the same way as
						`recorder.audio.device`: a pattern, without regard to case,
						where `*` matches any run of characters and `?` matches one.
						On Linux a MIDI device's name carries a client number that
						changes between runs, so use `*` for it and keep the port
						number after the colon.  `null` selects the only MIDI input
						when there is one, and otherwise asks.  Ignored when
						`virtual_midi_port` is set.
						""",
						default=None,
						nullable=True,
						examples=["Launchpad", "*Subsample Virtual MIDI *:0"],
					),
					"virtual_midi_port": _setting(
						"string",
						"""
						Name of a virtual MIDI input port that Subsample creates at
						start-up, for other software on the same machine to send notes
						to.  When set, Subsample opens no hardware MIDI input and
						ignores `midi_device`.
						""",
						default=None,
						nullable=True,
						examples=["Subsample Virtual MIDI"],
					),
					"audio": _section(
						"The output device, and the format Subsample plays in.",
						{
							"device": _setting(
								"string",
								"""
								Output device to play through, matched the same way as
								`recorder.audio.device`.  `null` uses the system's default
								output.
								""",
								default=None,
								nullable=True,
							),
							"sample_rate": _setting(
								"integer",
								"""
								Output sample rate, in Hz.  `null` uses
								`recorder.audio.sample_rate`.  Set it when the output device
								cannot run at the recorder's rate; a rate above the
								recorder's adds no quality.
								""",
								default=None,
								nullable=True,
								exclusiveMinimum=0,
							),
							"bit_depth": _setting(
								"integer",
								"""
								Output bit depth.  `null` uses `recorder.audio.bit_depth`.
								Set `16` for a device that only plays 16-bit audio.
								""",
								default=None,
								nullable=True,
								enum=[16, 24, 32],
							),
							"channels": _setting(
								"integer",
								"""
								Number of output channels, in SMPTE order.  `null` plays in
								stereo.  Set it to the interface's output count when the
								MIDI map routes instruments to separate outputs.  Subsample
								refuses to start when the device has fewer outputs.
								""",
								default=None,
								nullable=True,
								exclusiveMinimum=0,
							),
							"buffer_frames": _setting(
								"integer",
								"""
								Output buffer size in frames, as a power of two.  Smaller
								buffers lower latency, and larger ones are safer under load.
								`null` lets the operating system choose.  When the device
								refuses the size, Subsample logs an error and uses the
								device's own.
								""",
								default=None,
								nullable=True,
								minimum=32,
								maximum=4096,
							),
						},
					),
				},
			),
			"detection": _section(
				"""
				When a sound starts and ends a recording.  Every threshold except
				`min_peak_db` is measured against the room's background level, which
				Subsample tracks continuously.
				""",
				{
					"threshold_db": _setting(
						"number",
						"""
						How far, in dB, a sound must rise above the room's background
						level to start a recording.  Lower values catch quieter sounds
						and trigger more often on noise.  Raise it in a noisy room.
						""",
						default=12.0,
					),
					"floor_adaptation": _setting(
						"number",
						"""
						How quickly the tracked background level follows changes in
						the room.  Higher values follow gradual rises, so only sudden
						sounds such as drums and plucks trigger.  Lower values keep the
						background steady, so gradual swells trigger as well.
						""",
						default=0.1,
						exclusiveMinimum=0,
						maximum=1,
					),
					"hold_seconds": _setting(
						"number",
						"""
						Time, in seconds, a recording stays open after the level falls
						back, so a brief pause does not split one sound in two.  Longer
						values keep more of each tail, and more silence after it.
						""",
						default=0.5,
						exclusiveMinimum=0,
					),
					"warmup_seconds": _setting(
						"number",
						"""
						Time, in seconds, that Subsample listens to the room before
						detection starts, to measure its background level.  Raise it in
						a room whose level varies.
						""",
						default=1.0,
					),
					"trim_pre_ms": _setting(
						"number",
						"""
						Audio kept before each detected onset, in milliseconds, with a
						fade-in across it so the start does not click.
						""",
						default=0.25,
						minimum=0,
					),
					"trim_post_ms": _setting(
						"number",
						"""
						Audio kept after each detected end, in milliseconds, with a
						fade-out across it so the stop does not click.  For a longer
						fade on a cut tail, use `fade_out_ms`.
						""",
						default=2.0,
						minimum=0,
					),
					"release_threshold_db": _setting(
						"number",
						"""
						Level, in dB above the background, at which a sounding tail
						counts as finished and the recording ends.  Set it below
						`threshold_db` so a long decay such as a cymbal or a gong rings
						out instead of being cut short; the recording still starts at
						`threshold_db`.  `null` ends the recording at `threshold_db`.
						""",
						default=None,
						nullable=True,
						exclusiveMinimum=0,
					),
					"retrigger_threshold_db": _setting(
						"number",
						"""
						Rise, in dB over the decaying tail, that counts as the next hit
						while a recording is open.  The current sample ends there and a
						new recording starts, which separates hits whose tails never
						fade to silence.  Use it with `release_threshold_db` to cut a
						take of spaced hits apart.  A sound with a slow or two-stage
						attack can read its own second transient as a new hit, so raise
						`hold_seconds` to cover the attack.  `null` turns re-triggering
						off.
						""",
						default=None,
						nullable=True,
						exclusiveMinimum=0,
					),
					"fade_out_ms": _setting(
						"number",
						"""
						Length, in milliseconds, of a fade on each sample's trailing
						edge, which hides a cut made part-way down a decay or at the
						next hit.  `0` keeps only the short fade from `trim_post_ms`.
						""",
						default=0.0,
						minimum=0,
					),
					"min_peak_db": _setting(
						"number",
						"""
						Level, in dBFS, that a finished recording's peak must reach for
						Subsample to keep it.  A recording peaking below it is
						discarded, which stops a quiet room's own noise being saved as
						a sample.  Set it below the peak of the quietest sound you want
						to keep.  `null` keeps every recording.
						""",
						default=None,
						nullable=True,
						exclusiveMaximum=0,
					),
				},
			),
			"analysis": _section(
				"How Subsample detects the tempo of each sample.",
				{
					"start_bpm": _setting(
						"number",
						"""
						Starting estimate, in BPM, for beat tracking.  It biases the
						detected tempo without limiting it.  Set it near the tempo of
						the source material.
						""",
						default=120.0,
					),
					"tempo_min": _setting(
						"number",
						"""
						Slowest tempo, in BPM, the pulse detector considers when it
						looks for beats.  Must be below `tempo_max`.  It does not bound
						the tempo Subsample reports for a sample, which comes from beat
						tracking.
						""",
						default=30.0,
						exclusiveMinimum=0,
					),
					"tempo_max": _setting(
						"number",
						"""
						Fastest tempo, in BPM, the pulse detector considers when it
						looks for beats.  Must be above `tempo_min`.  It does not bound
						the tempo Subsample reports for a sample, which comes from beat
						tracking.
						""",
						default=300.0,
						exclusiveMinimum=0,
					),
				},
			),
			"similarity": _section(
				"""
				How much each group of acoustic measurements counts when Subsample
				compares a sample with a reference.  `0` leaves a group out of the
				comparison.
				""",
				{
					"weight_spectral": _similarity_weight(
						"""
						Weight of the spectral shape group: brightness, noisiness,
						attack and release time, and similar measures of a sound's
						overall character.
						""",
						default=1.0,
					),
					"weight_timbre": _similarity_weight(
						"""
						Weight of the sustained timbre group, averaged over the whole
						sound.  Raise it to tell instrument families apart in a library
						of sustained tonal sounds.
						""",
						default=1.0,
					),
					"weight_timbre_delta": _similarity_weight(
						"""
						Weight of the timbre change group, which follows how the timbre
						moves over the sound, such as the shift from attack to sustain
						in a plucked string.
						""",
						default=0.5,
					),
					"weight_timbre_onset": _similarity_weight(
						"""
						Weight of the attack timbre group, measured over the start of
						the sound.  Raise it for percussion, where kick, snare, and
						hi-hat differ most in their attacks.
						""",
						default=1.0,
					),
					"weight_band_energy": _similarity_weight(
						"""
						Weight of the band energy group: how a sound's energy and decay
						divide between the bass, low-mid, high-mid, and high frequency
						bands.  Raise it for drum libraries, and lower it for pitched
						instruments.
						""",
						default=1.0,
					),
				},
			),
			"library": _section(
				"""
				The playable sample library: where Subsample loads samples from, how
				much memory they may use, and where named references come from.
				""",
				{
					"max_memory_mb": _setting(
						"number",
						"""
						Memory, in MB, for the samples held in the library.  When a new
						sample would go over it, Subsample drops the oldest samples from
						memory; files on disk are never deleted.  When absent, it is a
						share of `max_memory_mb`.
						""",
						exclusiveMinimum=0,
					),
					"directory": _setting(
						"string",
						"""
						Directory of samples to load at start-up, including its
						subdirectories.  Subsample recreates missing `.analysis.json`
						and `.preview.png` sidecars, and deletes sidecars whose audio
						has gone.  `null` loads nothing in bulk, so every sample comes
						from the MIDI map's `directory:` and `path:` rules, and `watch`
						has nothing to watch.  A `programs:` block in the MIDI map
						overrides it.
						""",
						default="samples/captures",
						nullable=True,
					),
					"reference_directory": _setting(
						"string",
						"""
						Directory of reference fingerprints that a MIDI map can name in
						`where: { reference: ... }`.  `null` uses the General MIDI set
						that ships with Subsample, so a name such as `GM46_OpenHiHat`
						resolves on any machine.  Set it to a directory of your own
						references, whose names then replace the built-in ones.  A
						reference given as a path is not affected.
						""",
						default=None,
						nullable=True,
					),
					"watch": _setting(
						"boolean",
						"""
						Loads audio files that appear in `library.directory` while
						Subsample runs, without a restart.  Subsample waits until a
						file stops growing, analyses it when it has no sidecar, and adds
						it to the library.  Only the top level of the directory is
						watched.  Requires `library.directory` and `player.enabled`.
						""",
						default=False,
					),
				},
			),
			"transform": _section(
				"""
				Derived versions of samples that playback uses: pitch-shifted variants
				of tonal samples and time-stretched variants for quantise steps, held
				in memory and cached on disk.
				""",
				{
					"max_memory_mb": _setting(
						"number",
						"""
						Memory, in MB, for derived variants.  When a new variant would
						go over it, Subsample drops every variant of the oldest sample
						together.  When absent, it is a share of `max_memory_mb`.
						""",
						exclusiveMinimum=0,
					),
					"auto_pitch": _setting(
						"boolean",
						"""
						Renders pitch-shifted variants of each tonal sample for every
						note in its assigned range ahead of time, with Rubber Band.
						Set `false` when `rubberband-cli` is not installed, or to render
						pitch when a note plays instead, at a higher CPU cost.
						""",
						default=True,
					),
					"quantize_resolution": _setting(
						"integer",
						"""
						Grid that `stretch_quantize` and `pad_quantize` snap onsets to
						when an assignment gives no `grid:`, as a note division: `1`
						for whole notes, `2` halves, `4` quarters, `8` eighths, and
						`16` sixteenths.
						""",
						default=16,
						enum=[1, 2, 4, 8, 16],
					),
					"variant_cache_dir": _setting(
						"string",
						"""
						Directory for the disk cache of derived variants, so they
						survive a restart.  An empty value or `null` turns the disk
						cache off.
						""",
						default="samples/variant-cache",
						nullable=True,
					),
					"max_disk_mb": _setting(
						"number",
						"""
						Disk space, in MB, for cached variant files.  When the cache is
						full, Subsample deletes the least recently used files.  `0`
						turns the disk cache off.  When absent, it is sized from
						`max_memory_mb`.
						""",
					),
				},
			),
			"tempo": _section(
				"""
				The session tempo, and where it comes from.  It sets the grid for the
				`stretch_quantize` and `pad_quantize` processors, and the beat length
				for the `duration_beats` filter.
				""",
				{
					"bpm": _setting(
						"number",
						"""
						Session tempo, in BPM, for quantise steps with no `tempo:` of
						their own and for the `duration_beats` filter.  `0` leaves the
						tempo unset: quantise steps are skipped, and a map that filters
						by `duration_beats` does not load.  Under `source: midi` it is
						the tempo until a clock arrives.
						""",
						default=0.0,
					),
					"source": _setting(
						"string",
						"""
						Where the session tempo comes from.  `config` uses `bpm`.
						`midi` follows a MIDI clock arriving on the player's MIDI input,
						rounded to whole BPM, and uses `bpm` until one arrives.
						Subsample adopts a new clock tempo once it holds steady,
						re-renders quantised variants for it, and keeps it when the
						transport stops.  An assignment's own `tempo:` overrides both.
						""",
						default="config",
						enum=sorted(subsample.config.SUPPORTED_TEMPO_SOURCES),
					),
				},
			),
		},
	}


def _section (description: str, properties: dict[str, typing.Any]) -> dict[str, typing.Any]:

	"""A section of config.yaml: a mapping of named settings."""

	return {
		"description": _text(description),
		"type": "object",
		"properties": properties,
	}


def _setting (
	kind: typing.Union[str, dict[str, typing.Any]],
	description: str,
	default: typing.Any = _ABSENT,
	nullable: bool = False,
	examples: typing.Optional[list[typing.Any]] = None,
	**limits: typing.Any,
) -> dict[str, typing.Any]:

	"""One setting: its JSON type, prose, default, examples, and limits.

	``kind`` is a JSON Schema type name, or a whole schema for a list or a map.
	``limits`` are JSON Schema keywords (``minimum``, ``exclusiveMaximum``,
	``enum``, ...) and constrain the value itself, so a nullable setting carries
	them on its non-null option, where `null` does not have to satisfy them.
	"""

	value: dict[str, typing.Any] = {"type": kind} if isinstance(kind, str) else dict(kind)
	value.update(limits)

	setting: dict[str, typing.Any] = {"description": _text(description)}

	if nullable:
		setting["anyOf"] = [value, {"type": "null"}]
	else:
		setting.update(value)

	if default is not _ABSENT:
		setting["default"] = default

	if examples:
		setting["examples"] = examples

	return setting


def _similarity_weight (description: str, default: float) -> dict[str, typing.Any]:

	"""A similarity group weight, which every group bounds the same way."""

	return _setting("number", description, default=default, minimum=0, maximum=2)


def _text (description: str) -> str:

	"""Prose as written in this file, with its indentation and line breaks removed."""

	return " ".join(inspect.cleandoc(description).split())
