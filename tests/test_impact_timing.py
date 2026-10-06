"""Tests for impact timing: a timed note's sound starts early, so its hit lands on the note's time (#604).

The rule, the measurement on a render, the player's placing of a voice, the
note receiver telling a timed note from one sent alone, and the map's
``align:``.  The player is run buffer by buffer on the hand-moved clock
test_note_timing uses: at 44.1 kHz a buffer of 441 frames lasts 10 ms.
"""

import logging
import pathlib
import re
import time
import typing
import unittest.mock

import mido
import numpy
import pytest

import subsample.analysis
import subsample.midi_map_schema
import subsample.osc
import subsample.player
import subsample.query
import subsample.tools.catalog_samples
import subsample.transform

import tests.test_midi_map_schema as maps
import tests.test_note_timing as timing


_RATE = timing._RATE
_HIT  = 882
"""Where the hit comes in the test sound: 20 ms in, at 44.1 kHz."""


def _sound (lead_in: float = 0.05, channels: int = 1, length: int = 2000) -> numpy.ndarray:

	"""A quiet lead-in, then a hit 20 ms in that the test can find by its level."""

	audio = numpy.zeros((length, channels), dtype=numpy.float32)
	audio[:_HIT] = lead_in
	audio[_HIT]  = 0.9

	return audio


def _struck (lead_in_level: float, lead_in_seconds: float = 0.020) -> numpy.ndarray:

	"""Mono audio: noise at ``lead_in_level`` for ``lead_in_seconds``, then a decaying 200 Hz hit at full scale."""

	noise   = numpy.random.RandomState(7).uniform(-lead_in_level, lead_in_level, int(lead_in_seconds * _RATE))
	seconds = numpy.arange(int(0.3 * _RATE)) / _RATE
	hit     = numpy.sin(2.0 * numpy.pi * 200.0 * seconds) * numpy.exp(-seconds / 0.05)

	return numpy.concatenate([noise, hit]).astype(numpy.float32)


# ---------------------------------------------------------------------------
# Which sounds move, and by how much
# ---------------------------------------------------------------------------

class TestTheRule:

	"""Only a hit after a lead-in at least 10 dB quieter moves (#4513, 8a)."""

	@pytest.mark.parametrize(("impact", "pre_level", "moves"), [
		pytest.param(0.042, -17.0, True, id="pedal-close"),
		pytest.param(0.042, -10.0, True, id="exactly-at-the-line"),
		pytest.param(0.042, -120.0, True, id="silent-lead-in"),
		pytest.param(0.030, -1.8, False, id="open-hat-bloom"),
		pytest.param(7.2, -0.7, False, id="field-recording"),
		pytest.param(0.042, -9.9, False, id="just-inside-the-line"),
		pytest.param(0.0, None, False, id="hit-at-the-start"),
	])
	def test_a_hit_moves_only_after_a_quieter_lead_in (
		self, impact: float, pre_level: typing.Optional[float], moves: bool,
	) -> None:
		assert subsample.analysis.hit_time(impact, pre_level) == (impact if moves else 0.0)

	def test_a_hit_after_a_quiet_lead_in_is_found_where_it_starts (self) -> None:

		hit = subsample.analysis.measure_hit_time(_struck(0.05), _RATE)

		assert hit == pytest.approx(0.020, abs=0.001)

	def test_a_sound_that_opens_on_its_hit_does_not_move (self) -> None:

		assert subsample.analysis.measure_hit_time(_struck(0.0, lead_in_seconds=0.0), _RATE) == 0.0

	def test_a_loudest_moment_after_sound_nearly_as_loud_does_not_move (self) -> None:

		"""A busy take: a strike, a gap, then a louder one, which is not a hit after a pre-stroke."""

		first = 0.85 * _struck(0.0, lead_in_seconds=0.0)[: int(0.010 * _RATE)]
		audio = numpy.concatenate([first, numpy.zeros(int(0.010 * _RATE), dtype=numpy.float32), _struck(0.0, lead_in_seconds=0.0)])

		impact, pre_level = subsample.analysis._compute_impact(audio, _RATE)

		assert impact == pytest.approx(0.020, abs=0.001)
		assert pre_level is not None and pre_level > subsample.analysis.HIT_LEAD_IN_DB
		assert subsample.analysis.measure_hit_time(audio, _RATE) == 0.0


class TestTheWordsNameTheLine:

	"""The published words give the line the code draws."""

	def test_the_maps_reference_names_it (self) -> None:

		hit = next(
			option for option in maps._at("/$defs/assignment/properties/align")["oneOf"]
			if option["const"] == "hit"
		)

		assert f"at least {-subsample.analysis.HIT_LEAD_IN_DB:g} dB quieter" in hit["description"]

	def test_catalogs_help_names_it (self) -> None:

		help_text = subsample.tools.catalog_samples.parser().epilog or ""

		assert f"impact_pre_db is {subsample.analysis.HIT_LEAD_IN_DB:g} or" in help_text


# ---------------------------------------------------------------------------
# A render knows where its hit comes
# ---------------------------------------------------------------------------

class TestARenderKnowsWhereItsHitComes:

	def test_each_segment_has_its_own (self) -> None:

		"""A quantised sound plays one segment per note, from the segment's own start."""

		opens_on_its_hit = _struck(0.0, lead_in_seconds=0.0)
		pre_struck       = _struck(0.05)
		audio  = numpy.concatenate([opens_on_its_hit, pre_struck])[:, numpy.newaxis]
		bounds = ((0, len(opens_on_its_hit)), (len(opens_on_its_hit), len(audio)))

		_whole, segments = subsample.transform._hit_times(audio, _RATE, bounds)

		assert segments is not None
		assert segments[0] == 0.0
		assert segments[1] == pytest.approx(0.020, abs=0.001)

	def test_a_render_without_segments_has_none (self) -> None:

		hit, segments = subsample.transform._hit_times(_struck(0.05)[:, numpy.newaxis], _RATE, None)

		assert hit == pytest.approx(0.020, abs=0.001)
		assert segments is None

	def test_a_render_measures_the_sound_it_made (self) -> None:

		"""Measured after the steps: reversed, the hit is the top of the swell, most of the way in."""

		import tests.test_transform as transform_tests

		pcm    = (_struck(0.05) * 32767).astype(numpy.int16)[:, numpy.newaxis]
		record = transform_tests._make_record(audio=pcm)
		made: list[subsample.transform.TransformResult] = []

		processor = subsample.transform.TransformProcessor(sample_rate=_RATE)
		processor._on_complete = made.append
		processor._disk_cache  = None

		for spec in (subsample.transform.TransformSpec(steps=()), subsample.transform.TransformSpec(steps=(subsample.transform.Reverse(),))):
			processor._execute(record, spec, key=subsample.transform.TransformKey(record.sample_id, spec))

		as_recorded, reversed_ = made

		assert as_recorded.hit_time == pytest.approx(0.020, abs=0.001)
		assert reversed_.hit_time > 0.5 * reversed_.duration

	def test_a_render_read_back_from_disk_knows_it_too (self, tmp_path: pathlib.Path) -> None:

		cache = subsample.transform.VariantDiskCache(directory=tmp_path, max_bytes=100_000_000, sample_rate=_RATE)
		spec  = subsample.transform.TransformSpec(steps=(subsample.transform.Reverse(),))
		key   = subsample.transform.TransformKey(sample_id=1, spec=spec)
		audio = _struck(0.05)[:, numpy.newaxis]

		cache.put("md5", spec, subsample.transform.TransformResult(
			key=key, audio=audio, duration=len(audio) / _RATE,
			level=subsample.analysis.LevelResult(peak=1.0, rms=0.2),
		))
		loaded = cache.get("md5", spec, key)

		assert loaded is not None
		assert loaded.hit_time == pytest.approx(0.020, abs=0.001)

	def test_a_voice_reads_its_segments_hit_or_the_whole_renders (self) -> None:

		spec   = subsample.transform.TransformSpec(steps=())
		result = subsample.transform.TransformResult(
			key=subsample.transform.TransformKey(1, spec), audio=numpy.zeros((10, 1), dtype=numpy.float32),
			duration=0.0, level=subsample.analysis.LevelResult(peak=0.0, rms=0.0),
			hit_time=0.5, segment_hit_times=(0.0, 0.02),
		)

		assert subsample.player._hit_of(result, None) == 0.5
		assert subsample.player._hit_of(result, 1) == 0.02


# ---------------------------------------------------------------------------
# The player starts a timed note's sound early
# ---------------------------------------------------------------------------

def _render_as_is (*args: typing.Any, **_kwargs: typing.Any) -> numpy.ndarray:

	"""Stands in for _render_float and _render: the sound as given, so the test can find its hit."""

	audio: numpy.ndarray = args[0] if isinstance(args[0], numpy.ndarray) else args[0].audio

	return audio


def _player_of (
	clock:   timing._Clock,
	hit:     float = _HIT / _RATE,
	align:   str = "hit",
	base:    bool = True,
) -> subsample.player.MidiPlayer:

	"""A player whose note 36 on channel 10 plays _sound, from its base render or, with base False, from the sample."""

	player = timing._player(clock)
	audio  = _sound(channels=player._output_channels)

	record = unittest.mock.MagicMock()
	record.audio = audio
	record.name  = "Pedal"
	record.rhythm.impact_time         = hit
	record.rhythm.impact_pre_level_db = -26.0

	library = unittest.mock.MagicMock()
	library.get.return_value = record

	manager: typing.Optional[unittest.mock.MagicMock] = None

	if base:
		spec    = subsample.transform.TransformSpec(steps=())
		manager = unittest.mock.MagicMock(spec=subsample.transform.TransformManager)
		manager.get_base.return_value = subsample.transform.TransformResult(
			key=subsample.transform.TransformKey(1, spec), audio=audio, duration=len(audio) / _RATE,
			level=subsample.analysis.LevelResult(peak=0.9, rms=0.1), hit_time=hit,
		)

	player._effective_pool     = lambda: (library, manager)  # type: ignore[method-assign]
	player._resolve_sample_id  = lambda *_args, **_kwargs: 1  # type: ignore[method-assign]
	player._resolve_release    = lambda *_args: (None, 0, False)  # type: ignore[method-assign]
	player._resolve_loop       = lambda *_args: None  # type: ignore[method-assign]
	player._get_mix_matrix     = lambda *_args: numpy.eye(1, dtype=numpy.float32)  # type: ignore[method-assign]
	player._render_float       = _render_as_is  # type: ignore[method-assign]
	player._render             = _render_as_is  # type: ignore[method-assign]
	player._note_map = {(9, 36): [(
		subsample.query.Assignment(name="Pedal", select=(), align=align),
		subsample.query.PickSpec(1, 1),
	)]}

	return player


def _play (player: subsample.player.MidiPlayer, clock: timing._Clock, send: typing.Callable[[], None], at: float) -> numpy.ndarray:

	"""Run buffers every 10 ms from 10 ms to 90 ms, calling ``send`` with the clock at ``at``; what buffers 20 to 90 ms played."""

	played: list[numpy.ndarray] = []
	sent = False

	for index in range(1, 10):
		now = index * 0.010

		if not sent and at < now:
			clock.now = at
			send()
			sent = True

		buffer = timing._buffer(player, clock, now)

		if index > 1:
			played.append(buffer)

	return numpy.concatenate(played)


# The note is meant for 62.5 ms, so it plays one buffer later, from the
# buffer at 70 ms, which is the sixth one returned, at frame 110.
_MEANT_FOR = 0.0625
_HIT_LANDS = 5 * timing._FRAMES + timing._frame(0.0025)


def _starts_and_hit (played: numpy.ndarray) -> tuple[int, int]:

	"""The frame the sound starts on, and the frame its hit lands on."""

	return int(numpy.flatnonzero(played > 0.01)[0]), int(numpy.flatnonzero(played > 0.5)[0])


@pytest.fixture
def wall_clock (monkeypatch: pytest.MonkeyPatch) -> typing.Callable[[timing._Clock], None]:

	"""Ties time.time to the test's clock, so a bundle's time and the player's agree."""

	def _tie (clock: timing._Clock) -> None:
		monkeypatch.setattr(time, "time", lambda: 1000.0 + clock.now)

	return _tie


class TestATimedNoteLandsItsHit:

	def test_its_sound_starts_early_by_its_hit (self, wall_clock: typing.Callable[[timing._Clock], None]) -> None:

		"""Sent at 12 ms for 62.5 ms: the sound starts 20 ms early, and the hit lands where the note is meant."""

		clock  = timing._Clock()
		player = _player_of(clock)
		wall_clock(clock)

		played = _play(player, clock, lambda: player.play_osc_note(True, 9, 36, 1.0, 1000.0 + _MEANT_FOR, True), at=0.012)
		starts, hit = _starts_and_hit(played)

		assert hit == pytest.approx(_HIT_LANDS, abs=1)
		assert starts == pytest.approx(_HIT_LANDS - _HIT, abs=1)

	def test_from_the_sample_itself_too (self, wall_clock: typing.Callable[[timing._Clock], None]) -> None:

		"""With no render, the sample's own analysis says where its hit comes."""

		clock  = timing._Clock()
		player = _player_of(clock, base=False)
		wall_clock(clock)

		played = _play(player, clock, lambda: player.play_osc_note(True, 9, 36, 1.0, 1000.0 + _MEANT_FOR, True), at=0.012)

		assert _starts_and_hit(played)[1] == pytest.approx(_HIT_LANDS, abs=1)

	@pytest.mark.parametrize("why", ["sent-on-its-own", "align-start", "no-quieter-lead-in"])
	def test_otherwise_its_sound_starts_on_the_time (
		self, why: str, wall_clock: typing.Callable[[timing._Clock], None],
	) -> None:
		clock  = timing._Clock()
		player = _player_of(clock, align="start" if why == "align-start" else "hit", hit=0.0 if why == "no-quieter-lead-in" else _HIT / _RATE)
		wall_clock(clock)

		played = _play(player, clock, lambda: player.play_osc_note(True, 9, 36, 1.0, 1000.0 + _MEANT_FOR, why != "sent-on-its-own"), at=0.012)

		assert _starts_and_hit(played)[0] == pytest.approx(_HIT_LANDS, abs=1)

	def test_a_midi_note_is_never_moved (self) -> None:

		"""It gives no notice of its time, so its sound starts as it arrives (#4492)."""

		clock  = timing._Clock()
		player = _player_of(clock)

		played = _play(player, clock, lambda: player._safe_handle_message(mido.Message("note_on", channel=9, note=36, velocity=127)), at=_MEANT_FOR)

		assert _starts_and_hit(played)[0] == pytest.approx(_HIT_LANDS, abs=1)


class TestANoteSentTooLate:

	def test_starts_its_whole_sound_at_once_and_lands_its_hit_late (
		self, wall_clock: typing.Callable[[timing._Clock], None], caplog: pytest.LogCaptureFixture,
	) -> None:

		"""Sent 7.5 ms ahead with a hit 20 ms in: the sound starts at the next buffer, from its beginning."""

		clock  = timing._Clock()
		player = _player_of(clock)
		wall_clock(clock)

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			played = _play(player, clock, lambda: player.play_osc_note(True, 9, 36, 1.0, 1000.0 + _MEANT_FOR, True), at=0.055)

		starts, hit = _starts_and_hit(played)

		# The buffer at 60 ms, the fifth returned, plays it from its first frame.
		assert starts == pytest.approx(4 * timing._FRAMES, abs=1)
		assert hit - _HIT_LANDS == pytest.approx(timing._frame(0.0075), abs=1)

		warnings = [record.message for record in caplog.records if "lands" in record.message]

		assert len(warnings) == 1
		assert re.search(r"'Pedal' arrived \d+ ms before its time, but its hit is 20 ms into the sound", warnings[0])
		assert "need notes sent at least 20 ms ahead" in warnings[0]

	def test_is_warned_about_once_a_minute (
		self, wall_clock: typing.Callable[[timing._Clock], None], caplog: pytest.LogCaptureFixture,
	) -> None:
		clock  = timing._Clock()
		player = _player_of(clock)
		wall_clock(clock)

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			for _ in range(3):
				player.play_osc_note(True, 9, 36, 1.0, 1000.0 + clock.now, True)

		assert len([record for record in caplog.records if "lands" in record.message]) == 1

	def test_names_the_furthest_ahead_the_sounds_so_far_needed (
		self, wall_clock: typing.Callable[[timing._Clock], None], caplog: pytest.LogCaptureFixture,
	) -> None:

		"""A sound that needed 50 ms, sent in time, still counts toward the lead the set needs."""

		clock  = timing._Clock()
		player = _player_of(clock)
		wall_clock(clock)
		player._hit_lead_needed = 0.050

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			player.play_osc_note(True, 9, 36, 1.0, 1000.0 + clock.now, True)

		assert any("at least 50 ms ahead" in record.message for record in caplog.records)

	def test_a_shortfall_under_a_millisecond_is_not_warned_about (
		self, wall_clock: typing.Callable[[timing._Clock], None], caplog: pytest.LogCaptureFixture,
	) -> None:
		clock  = timing._Clock()
		player = _player_of(clock)
		wall_clock(clock)
		timing._buffer(player, clock, 0.010)
		clock.now = 0.010

		with caplog.at_level(logging.WARNING, logger="subsample.player"):
			player.play_osc_note(True, 9, 36, 1.0, 1000.0 + 0.010 + _HIT / _RATE - 0.0005, True)

		assert not [record for record in caplog.records if "lands" in record.message]


# ---------------------------------------------------------------------------
# The note receiver says which notes are timed
# ---------------------------------------------------------------------------

class TestTheReceiverSaysWhichNotesAreTimed:

	@pytest.fixture(autouse=True)
	def _needs_python_osc (self) -> None:
		pytest.importorskip("pythonosc")

	def _heard (self, datagram: bytes) -> list[tuple[typing.Any, ...]]:

		"""What a receiver hands on for one packet."""

		heard: list[tuple[typing.Any, ...]] = []
		receiver = subsample.osc.OscNoteReceiver(port=0, on_note=lambda *args: heard.append(args))

		try:
			receiver._handle_packet(datagram)
		finally:
			receiver.stop()

		return heard

	def _bundle (self, when: typing.Any, *contents: typing.Any) -> typing.Any:

		"""A built bundle of these messages or bundles."""

		import pythonosc.osc_bundle_builder

		builder = pythonosc.osc_bundle_builder.OscBundleBuilder(when)

		for content in contents:
			builder.add_content(content)

		return builder.build()

	def _note_on (self, note: int) -> typing.Any:

		import pythonosc.osc_message_builder

		message = pythonosc.osc_message_builder.OscMessageBuilder(address="/note/on")

		for arg in (10, note, 0.5):
			message.add_arg(arg)

		return message.build()

	def test_a_message_in_a_bundle_timed_ahead_is_timed (self) -> None:

		heard = self._heard(self._bundle(time.time() + 5.0, self._note_on(36)).dgram)

		assert heard[0][5] is True

	def test_one_whose_time_has_passed_is_timed_and_plays_as_it_arrives (self) -> None:

		before = time.time()
		heard  = self._heard(self._bundle(before - 5.0, self._note_on(36)).dgram)

		assert heard[0][5] is True
		assert heard[0][4] >= before

	def test_a_message_on_its_own_is_not (self) -> None:

		heard = self._heard(self._note_on(36).dgram)

		assert heard[0][5] is False

	def test_a_bundle_marked_immediately_is_not (self) -> None:

		import pythonosc.osc_bundle_builder

		heard = self._heard(self._bundle(pythonosc.osc_bundle_builder.IMMEDIATELY, self._note_on(36)).dgram)

		assert heard[0][5] is False

	def test_a_bundle_inside_a_bundle_keeps_its_own_time (self) -> None:

		import pythonosc.osc_bundle_builder

		later = time.time() + 5.0
		inner = self._bundle(later, self._note_on(38))
		heard = self._heard(self._bundle(pythonosc.osc_bundle_builder.IMMEDIATELY, self._note_on(36), inner).dgram)

		assert [(note[2], note[5]) for note in heard] == [(36, False), (38, True)]
		assert heard[1][4] == pytest.approx(later, abs=0.001)

	def test_a_packet_that_is_not_osc_is_ignored_with_a_warning (self, caplog: pytest.LogCaptureFixture) -> None:

		with caplog.at_level(logging.WARNING, logger="subsample.osc"):
			heard = self._heard(b"not osc")

		assert heard == []
		assert any("not OSC" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# The map's align:
# ---------------------------------------------------------------------------

class TestAlign:

	@pytest.mark.parametrize("value", ["impact", "", True, 1])
	def test_anything_but_its_two_words_is_refused (self, tmp_path: pathlib.Path, value: typing.Any) -> None:

		with pytest.raises(ValueError, match="invalid align"):
			maps._load(tmp_path, maps._map(assignments=[maps._assignment(align=value)]))

	def test_an_assignment_can_set_a_templates_start_back_to_hit (self, tmp_path: pathlib.Path) -> None:

		result = maps._load(tmp_path, maps._map(
			templates={"pads": {"align": "start"}},
			assignments=[maps._assignment(template="pads", align="hit")],
		))

		assert maps._first(result).align == "hit"

	def test_a_template_sets_it_for_the_assignments_that_name_it (self, tmp_path: pathlib.Path) -> None:

		result = maps._load(tmp_path, maps._map(
			templates={"pads": {"align": "start"}},
			assignments=[maps._assignment(template="pads")],
		))

		assert maps._first(result).align == "start"

	def test_a_zone_carries_it (self, tmp_path: pathlib.Path) -> None:

		result = maps._load(tmp_path, maps._map(assignments=[
			maps._assignment(notes="zone-tuned", process=[{"repitch": True}], align="start"),
		]))

		assert result.zone_templates[0].align == "start"
