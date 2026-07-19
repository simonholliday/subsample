"""Tests for choke / mute groups (``silenced_by:``).

Covers the four layers of the feature:
  - ``_parse_silenced_by``   — the YAML surface (note / list / self / errors)
  - ``_build_choke_map``     — compiling declarations into the note-on table
  - ``_choke_voices``        — the runtime cut (fast declick, overrides, kill-all)
  - ``_handle_message`` + ``load_midi_map`` — integration and parse-at-load

The runtime tests drive the real player methods against a MagicMock ``self``
(the pattern the note-off / same-note-steal tests already use), so no audio
device or full player construction is needed.
"""

import logging
import pathlib
import threading
import unittest.mock

import mido
import numpy
import pytest

import subsample.config
import subsample.library
import subsample.player
import subsample.query
import subsample.similarity


_FADE_FRAMES = 441   # 10 ms at 44100 Hz


def _voice (
	note:            int = 46,
	channel:         int = 9,
	one_shot:        bool = False,
	releasing:       bool = False,
	release_to_end:  bool = False,
	release_frames:  "int | None" = None,
	release_curve:   int = 0,
	looping:         bool = False,
	fade_pos:        int = 0,
	n_frames:        int = 4410,
) -> subsample.player._Voice:
	"""A _Voice with silent audio and the given lifecycle flags."""
	audio = numpy.zeros((n_frames, 2), dtype=numpy.float32)
	return subsample.player._Voice(
		audio=audio, note=note, channel=channel, one_shot=one_shot,
		releasing=releasing, release_to_end=release_to_end,
		release_frames=release_frames, release_curve=release_curve,
		looping=looping, fade_pos=fade_pos,
	)


def _player (voices, choke_map) -> unittest.mock.MagicMock:
	"""A MagicMock player wired with just what the choke path reads."""
	player = unittest.mock.MagicMock(spec=subsample.player.MidiPlayer)
	player._voices              = list(voices)
	player._voices_lock         = threading.Lock()
	player._choke_map           = choke_map
	player._release_fade_frames = _FADE_FRAMES
	return player


def _choke (player, channel, note) -> None:
	subsample.player.MidiPlayer._choke_voices(player, channel, note)


# ---------------------------------------------------------------------------
# _parse_silenced_by — the YAML surface
# ---------------------------------------------------------------------------

class TestParseSilencedBy:

	def test_absent_forms_are_none (self) -> None:
		for raw in (None, False, []):
			assert subsample.player._parse_silenced_by(raw, "t") is None

	def test_self_scalar (self) -> None:
		spec = subsample.player._parse_silenced_by("self", "t")
		assert spec == subsample.query.ChokeSpec(is_self=True, notes=frozenset())

	def test_single_note_int (self) -> None:
		spec = subsample.player._parse_silenced_by(42, "t")
		assert spec == subsample.query.ChokeSpec(is_self=False, notes=frozenset({42}))

	def test_single_note_symbol (self) -> None:
		spec = subsample.player._parse_silenced_by("drum.hi_hat_closed", "t")
		assert spec == subsample.query.ChokeSpec(is_self=False, notes=frozenset({42}))

	def test_list_mixes_self_and_notes (self) -> None:
		spec = subsample.player._parse_silenced_by(
			["self", "drum.hi_hat_closed", "drum.hi_hat_pedal"], "t",
		)
		assert spec == subsample.query.ChokeSpec(is_self=True, notes=frozenset({42, 44}))

	def test_list_of_only_self (self) -> None:
		spec = subsample.player._parse_silenced_by(["self"], "t")
		assert spec == subsample.query.ChokeSpec(is_self=True, notes=frozenset())

	def test_custom_namespace_symbol_resolves (self) -> None:

		"""A mounted definitions name chokes like any note — threaded in via
		the namespaces param (load_midi_map passes the merged per-map view)."""

		spaces = {
			**subsample.player._SYMBOL_NAMESPACES,
			"my": {"ride_edge_soft": 53},
		}
		spec = subsample.player._parse_silenced_by(
			["self", "my.ride_edge_soft"], "t", spaces,
		)
		assert spec == subsample.query.ChokeSpec(is_self=True, notes=frozenset({53}))

	def test_note_name_resolves (self) -> None:
		spec = subsample.player._parse_silenced_by("C3", "t")
		assert spec is not None and not spec.is_self and len(spec.notes) == 1

	def test_bool_rejected (self) -> None:
		# True is an int subclass — must not silently become note 1.
		with pytest.raises(ValueError, match="silenced_by"):
			subsample.player._parse_silenced_by(True, "t")

	def test_unknown_symbol_rejected (self) -> None:
		with pytest.raises(ValueError, match="unknown"):
			subsample.player._parse_silenced_by("drum.not_a_drum", "t")

	def test_out_of_range_rejected (self) -> None:
		with pytest.raises(ValueError, match=r"\[0, 127\]"):
			subsample.player._parse_silenced_by(200, "t")


# ---------------------------------------------------------------------------
# _build_choke_map — compiling declarations into the note-on table
# ---------------------------------------------------------------------------

class TestBuildChokeMap:

	@staticmethod
	def _assign (name, silenced_by):
		return subsample.query.Assignment(name=name, select=(), silenced_by=silenced_by)

	def _note_map (self, entries):
		# entries: {(ch, note): silenced_by-spec-or-None}
		nm: dict = {}
		for (ch, note), spec in entries.items():
			nm[(ch, note)] = [(self._assign(f"{note}", spec), None)]
		return nm

	def test_three_way_hat_is_mutual (self) -> None:
		S = subsample.query.ChokeSpec
		nm = self._note_map({
			(9, 42): S(is_self=True, notes=frozenset({46, 44})),
			(9, 44): S(is_self=True, notes=frozenset({42, 46})),
			(9, 46): S(is_self=True, notes=frozenset({42, 44})),
		})
		cm = subsample.player._build_choke_map(nm)
		all_hats = frozenset({(9, 42), (9, 44), (9, 46)})
		for note in (42, 44, 46):
			assert cm[(9, note)] == all_hats   # any articulation damps all three

	def test_self_only_is_mono (self) -> None:
		S = subsample.query.ChokeSpec
		nm = self._note_map({(9, 51): S(is_self=True, notes=frozenset())})
		cm = subsample.player._build_choke_map(nm)
		assert cm == {(9, 51): frozenset({(9, 51)})}

	def test_explicit_only_has_no_self_edge (self) -> None:
		# open hat silenced_by closed (42) only, NOT self.
		S = subsample.query.ChokeSpec
		nm = self._note_map({(9, 46): S(is_self=False, notes=frozenset({42}))})
		cm = subsample.player._build_choke_map(nm)
		assert cm == {(9, 42): frozenset({(9, 46)})}   # firing 46 does NOT damp 46
		assert (9, 46) not in cm

	def test_no_declarations_is_empty (self) -> None:
		nm = self._note_map({(9, 36): None, (9, 38): None})
		assert subsample.player._build_choke_map(nm) == {}

	def test_multi_note_assignment_self (self) -> None:
		# One assignment spanning notes 60 and 62, self-choking: firing either
		# damps both (one physical instrument across two notes).
		S = subsample.query.ChokeSpec
		asgn = self._assign("wide", S(is_self=True, notes=frozenset()))
		nm = {(9, 60): [(asgn, None)], (9, 62): [(asgn, None)]}
		cm = subsample.player._build_choke_map(nm)
		both = frozenset({(9, 60), (9, 62)})
		assert cm[(9, 60)] == both
		assert cm[(9, 62)] == both

	def test_explicit_killer_is_channel_scoped (self) -> None:
		# An explicit (non-self) killer note resolves on the DECLARING
		# assignment's own channel — never a same-numbered note on another.
		S = subsample.query.ChokeSpec
		victim = self._assign("victim", S(is_self=False, notes=frozenset({42})))  # ch 9
		other  = self._assign("other", None)                                      # ch 0, note 46
		nm = {
			(9, 46): [(victim, None)],
			(0, 46): [(other, None)],
		}
		cm = subsample.player._build_choke_map(nm)
		assert cm == {(9, 42): frozenset({(9, 46)})}   # killer keyed to ch 9 only
		assert (0, 42) not in cm


# ---------------------------------------------------------------------------
# _choke_voices — the runtime cut
# ---------------------------------------------------------------------------

class TestChokeVoices:

	_MAP = {(9, 42): frozenset({(9, 46)})}   # firing 42 damps voices on 46

	def test_cuts_one_shot_voice (self) -> None:
		# The headline: a ringing one-shot is cut, which note-off and CC120/123
		# both refuse to do.
		v = _voice(note=46, one_shot=True)
		p = _player([v], self._MAP)
		_choke(p, 9, 42)
		assert v.releasing is True
		assert v.release_frames == _FADE_FRAMES
		assert v.release_curve == 0
		assert v.release_to_end is False

	def test_overrides_release_full (self) -> None:
		# release: full (release_to_end) is immune to note-off/panic; choke wins.
		v = _voice(note=46, one_shot=True, release_to_end=True)
		p = _player([v], self._MAP)
		_choke(p, 9, 42)
		assert v.release_to_end is False
		assert v.releasing is True
		assert v.release_frames == _FADE_FRAMES

	def test_overrides_configured_release (self) -> None:
		# A gated voice with a long configured release is damped fast, not slow.
		v = _voice(note=46, one_shot=False, release_frames=22050)
		p = _player([v], self._MAP)
		_choke(p, 9, 42)
		assert v.release_frames == _FADE_FRAMES

	def test_kills_all_matching_voices (self) -> None:
		# Round-robin roll: every ringing voice on the note is damped (one cymbal).
		vs = [_voice(note=46, one_shot=True) for _ in range(4)]
		p = _player(vs, self._MAP)
		_choke(p, 9, 42)
		assert all(v.releasing for v in vs)

	def test_per_channel_isolation (self) -> None:
		here  = _voice(note=46, channel=9, one_shot=True)
		other = _voice(note=46, channel=0, one_shot=True)
		p = _player([here, other], self._MAP)
		_choke(p, 9, 42)
		assert here.releasing is True
		assert other.releasing is False

	def test_stops_looping (self) -> None:
		v = _voice(note=46, looping=True)
		p = _player([v], self._MAP)
		_choke(p, 9, 42)
		assert v.looping is False
		assert v.releasing is True
		assert v.release_frames == _FADE_FRAMES

	def test_noop_when_no_victims (self) -> None:
		v = _voice(note=46, one_shot=True)
		p = _player([v], self._MAP)
		_choke(p, 9, 99)   # 99 is not a killer note
		assert v.releasing is False

	def test_forces_cosine_curve (self) -> None:
		# A sounding voice configured for the exponential release curve is damped
		# with the cosine declick (release_curve forced to 0).
		v = _voice(note=46, one_shot=True, release_curve=1)
		p = _player([v], self._MAP)
		_choke(p, 9, 42)
		assert v.release_curve == 0

	def test_mid_fade_voice_not_reclamped (self) -> None:
		# A voice already fading (fade_pos > 0) on a long release keeps its fade
		# length AND curve rather than jumping mid-ramp (would click).  Edge doc'd.
		v = _voice(note=46, releasing=True, fade_pos=100, release_frames=22050, release_curve=1)
		p = _player([v], self._MAP)
		_choke(p, 9, 42)
		assert v.release_frames == 22050   # unchanged
		assert v.release_curve == 1        # unchanged (guard skips the curve too)
		assert v.release_to_end is False
		assert v.releasing is True

	def test_choke_beats_steal_on_shared_voice (self) -> None:
		# A self-choking GATED voice re-struck: the handler runs choke BEFORE the
		# same-note steal, so the fast declick wins over the voice's own release.
		v = _voice(note=46, one_shot=False, release_frames=22050)
		p = _player([v], {(9, 46): frozenset({(9, 46)})})
		_choke(p, 9, 46)                                       # choke first
		subsample.player.MidiPlayer._release_held(p, 46, 9)    # then steal
		assert v.releasing is True
		assert v.release_frames == _FADE_FRAMES                # choke won


# ---------------------------------------------------------------------------
# _handle_message — integration on the note-on gesture
# ---------------------------------------------------------------------------

class TestChokeHandleMessage:

	def test_note_on_fires_choke_before_mapping (self) -> None:
		# Choke runs on the raw note-on gesture, ahead of the note-map lookup, so
		# a note that maps to nothing (a silent "grab") still damps.  Empty
		# note_map means the handler returns right after the choke sweep.
		v = _voice(note=46, one_shot=True)
		p = _player([v], {(9, 42): frozenset({(9, 46)})})
		p._note_map = {}
		p._choke_voices = lambda ch, note: subsample.player.MidiPlayer._choke_voices(p, ch, note)

		msg = mido.Message("note_on", channel=9, note=42, velocity=100)
		subsample.player.MidiPlayer._handle_message(p, msg)

		assert v.releasing is True

	def test_velocity_zero_note_on_does_not_choke (self) -> None:
		# A note_on with velocity 0 is mido's note-off; it must take the note-off
		# path (never the choke sweep), so a one-shot victim is left untouched.
		v = _voice(note=46, one_shot=True)
		p = _player([v], {(9, 42): frozenset({(9, 46)})})
		p._note_map = {}
		p._choke_voices = lambda ch, note: subsample.player.MidiPlayer._choke_voices(p, ch, note)
		p._release_held = lambda note, ch: subsample.player.MidiPlayer._release_held(p, note, ch)

		msg = mido.Message("note_on", channel=9, note=42, velocity=0)
		subsample.player.MidiPlayer._handle_message(p, msg)

		assert v.releasing is False


# ---------------------------------------------------------------------------
# load_midi_map — parse at load, zone rejection, unmapped-target warning
# ---------------------------------------------------------------------------

class TestChokeLoad:

	def _write_map (self, tmp_path: pathlib.Path, content: str) -> pathlib.Path:
		p = tmp_path / "test-map.yaml"
		p.write_text(content, encoding="utf-8")
		return p

	def test_silenced_by_parsed_onto_assignment (self, tmp_path: pathlib.Path) -> None:
		path = self._write_map(tmp_path, """
assignments:
  - name: Open Hi-Hat
    channel: 10
    notes: drum.hi_hat_open
    silenced_by: [self, drum.hi_hat_closed, drum.hi_hat_pedal]
    select:
      where:
        reference: BD0025
  - name: Closed Hi-Hat
    channel: 10
    notes: drum.hi_hat_closed
    select:
      where:
        reference: BD0025
  - name: Pedal Hi-Hat
    channel: 10
    notes: drum.hi_hat_pedal
    select:
      where:
        reference: BD0025
""")
		note_map = subsample.player.load_midi_map(path, ["BD0025"]).note_map
		asgn, _ = note_map[(9, 46)][0]
		assert asgn.silenced_by == subsample.query.ChokeSpec(
			is_self=True, notes=frozenset({42, 44}),
		)

	def test_zone_tuned_rejects_silenced_by (self, tmp_path: pathlib.Path) -> None:
		path = self._write_map(tmp_path, """
assignments:
  - name: Lead
    channel: 1
    notes: zone-tuned
    silenced_by: self
    process:
      - repitch: true
    select:
      where:
        pitched: true
""")
		with pytest.raises(ValueError, match="silenced_by"):
			subsample.player.load_midi_map(path, [])

	def test_unmapped_choker_note_warns (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		# Open hat is silenced_by the pedal (44), but no pedal assignment exists.
		path = self._write_map(tmp_path, """
assignments:
  - name: Open Hi-Hat
    channel: 10
    notes: drum.hi_hat_open
    silenced_by: drum.hi_hat_pedal
    select:
      where:
        reference: BD0025
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.load_midi_map(path, ["BD0025"])
		assert any("silenced_by" in r.message and "no" in r.message for r in caplog.records)

	def test_mapped_choker_note_does_not_warn (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		path = self._write_map(tmp_path, """
assignments:
  - name: Open Hi-Hat
    channel: 10
    notes: drum.hi_hat_open
    silenced_by: drum.hi_hat_closed
    select:
      where:
        reference: BD0025
  - name: Closed Hi-Hat
    channel: 10
    notes: drum.hi_hat_closed
    select:
      where:
        reference: BD0025
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.load_midi_map(path, ["BD0025"])
		assert not any("silenced_by" in r.message for r in caplog.records)

	def test_multi_note_assignment_warns_once (
		self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		# A multi-note assignment naming one unmapped choker note warns exactly
		# once (deduped across its notes), not once per note.
		path = self._write_map(tmp_path, """
assignments:
  - name: Wide
    channel: 1
    notes: [60, 62]
    silenced_by: drum.hi_hat_pedal
    select:
      where:
        reference: BD0025
""")
		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			subsample.player.load_midi_map(path, ["BD0025"])
		hits = [r for r in caplog.records if "silenced_by" in r.message]
		assert len(hits) == 1

	def test_shipped_gm_drums_map_choke_table (self) -> None:
		# Regression guard on the shipped kit: load the real GM drums map and
		# assert its choke groups compile as intended (a typo'd drum symbol or an
		# asymmetric hat group would fail here).
		gm_map = subsample.config.data_dir() / "midi-map-gm-drums.yaml"
		note_map = subsample.player.load_midi_map(gm_map, []).note_map
		cm = subsample.player._build_choke_map(note_map)

		hats = frozenset({(9, 42), (9, 44), (9, 46)})
		for hat_note in (42, 44, 46):
			assert cm[(9, hat_note)] == hats            # any articulation damps all three
		assert cm[(9, 51)] == frozenset({(9, 51)})      # ride 1 self-only
		assert cm[(9, 49)] == frozenset({(9, 49)})      # crash 1 self-only
		assert cm[(9, 58)] == frozenset({(9, 58)})      # vibraslap self-only
		assert cm[(9, 80)] == frozenset({(9, 80), (9, 81)})   # mute triangle damps mute + open
		assert cm[(9, 81)] == frozenset({(9, 81)})      # open triangle self-only


# ---------------------------------------------------------------------------
# Choke table rebuild across the live rule-swap paths (reload / program change)
# ---------------------------------------------------------------------------

class TestChokeReload:

	def _make_player (self, midi_map: dict) -> subsample.player.MidiPlayer:
		lib = unittest.mock.MagicMock(spec=subsample.library.InstrumentLibrary)
		lib.samples.return_value = []          # keep the candidate-cache rebuild cheap
		sim = unittest.mock.MagicMock(spec=subsample.similarity.SimilarityMatrix)
		return subsample.player.MidiPlayer(
			"Test Device", threading.Event(),
			instrument_library=lib, similarity_matrix=sim,
			midi_map=midi_map, sample_rate=44100, bit_depth=16,
		)

	@staticmethod
	def _asgn (silenced_by):
		return subsample.query.Assignment(
			name="x", select=(subsample.query.SelectSpec(),), silenced_by=silenced_by,
		)

	def test_init_builds_choke_map (self) -> None:
		S = subsample.query.ChokeSpec
		mm = {(9, 51): [(self._asgn(S(is_self=True)), subsample.query.PickSpec(1, 1))]}
		player = self._make_player(mm)
		assert player._choke_map == {(9, 51): frozenset({(9, 51)})}

	def test_reload_rebuilds_choke_map (self) -> None:
		S = subsample.query.ChokeSpec
		player = self._make_player({})
		assert player._choke_map == {}                 # init on an empty map

		choke_map = {(9, 51): [(self._asgn(S(is_self=True)), subsample.query.PickSpec(1, 1))]}
		with unittest.mock.patch.object(player, "update_assignments"):
			player._apply_rule_set(choke_map, (), set())
		assert player._choke_map == {(9, 51): frozenset({(9, 51)})}

		# Reload to a choke-free map — the stale table must clear, not linger.
		free_map = {(9, 36): [(self._asgn(None), subsample.query.PickSpec(1, 1))]}
		with unittest.mock.patch.object(player, "update_assignments"):
			player._apply_rule_set(free_map, (), set())
		assert player._choke_map == {}

	def test_reload_rollback_preserves_choke_map (self) -> None:
		S = subsample.query.ChokeSpec
		mm = {(9, 51): [(self._asgn(S(is_self=True)), subsample.query.PickSpec(1, 1))]}
		player = self._make_player(mm)
		before = player._choke_map

		bad = {(9, 36): [(self._asgn(S(is_self=True, notes=frozenset({42}))), subsample.query.PickSpec(1, 1))]}
		with unittest.mock.patch.object(
			player, "update_assignments", side_effect=ValueError("boom"),
		):
			with pytest.raises(ValueError, match="boom"):
				player._apply_rule_set(bad, (), set())
		assert player._choke_map == before             # untouched on rollback

	def test_multi_note_self_survives_oob_routing_strip (self) -> None:
		# A multi-note self-choke assignment with out-of-bounds output routing:
		# the routing strip must keep it ONE object, or _build_choke_map's
		# id()-grouping would collapse the cross-note self edge.
		S = subsample.query.ChokeSpec
		player = self._make_player({})
		player._output_channels = 2
		asgn = subsample.query.Assignment(
			name="wide", select=(subsample.query.SelectSpec(),),
			silenced_by=S(is_self=True), output_routing=(0, 5),   # 5 is OOB on stereo
		)
		pick = subsample.query.PickSpec(1, 1)
		base = {(9, 60): [(asgn, pick)], (9, 62): [(asgn, pick)]}

		stripped, _zones = player._strip_oob_routing_rules(base, ())
		cm = subsample.player._build_choke_map(stripped)

		both = frozenset({(9, 60), (9, 62)})
		assert cm[(9, 60)] == both     # firing 60 still damps a ringing 62 voice
		assert cm[(9, 62)] == both
