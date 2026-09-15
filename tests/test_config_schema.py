"""Tests for subsample.config_schema — the documentation schema agrees with the config builder.

The builder deliberately keeps no schema of its own: it records which keys it
reads.  The documentation schema is a second, hand-declared list, and these
tests are what make it trustworthy.  If one fails, the schema has drifted from
the builder, and the fix is to change the schema to match what the builder
does, not the other way round.
"""

import json
import pathlib
import typing

import pytest
import yaml

import subsample.config
import subsample.config_schema


_SCHEMA = subsample.config_schema.json_schema()

_RETIRED_KEYS: dict[str, typing.Any] = {
	"output":                       {"directory": "/tmp/x"},
	"instrument":                   {"directory": "/tmp/x"},
	"detection.snr_threshold_db":   12.0,
	"detection.ema_alpha":          0.1,
	"detection.hold_time":          0.5,
	"detection.trim_pre_samples":   10,
	"detection.trim_post_samples":  90,
	"recorder.audio.chunk_size":    512,
	"transform.target_bpm":         120.0,
	"transform.tempo_source":       "midi",
}
"""Keys the builder reads only to refuse them and name their replacement, each
with a value that would once have been valid.  They are not settings, so the
schema does not declare them."""

_DERIVED_DEFAULTS: frozenset[str] = frozenset({"max_memory_mb"})
"""Settings whose declared default is `null`, meaning "work it out", and which
the loaded Config holds as the value worked out on this machine."""

_JSON_TYPES: dict[str, typing.Callable[[typing.Any], bool]] = {
	"boolean": lambda value: isinstance(value, bool),
	"integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
	"number":  lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
	"string":  lambda value: isinstance(value, str),
	"array":   lambda value: isinstance(value, (list, tuple)),
	"object":  lambda value: isinstance(value, dict),
}


def _declared (
	schema: dict[str, typing.Any],
	prefix: str = "",
) -> dict[str, dict[str, typing.Any]]:

	"""Every section and setting the schema declares, keyed by dotted path."""

	found: dict[str, dict[str, typing.Any]] = {}

	for name, entry in schema.get("properties", {}).items():
		path = f"{prefix}{name}"
		found[path] = entry

		if entry.get("type") == "object" and "properties" in entry:
			found.update(_declared(entry, f"{path}."))

	return found


def _consulted () -> set[str]:

	"""Every dotted key the builder consults while building the shipped default."""

	raw = subsample.config._read_yaml(subsample.config._locate_default_config())
	trackers: list[subsample.config._KeyTracker] = []

	subsample.config._build_config(raw, trackers)

	paths: set[str] = set()

	for tracker in trackers:
		prefix = "" if tracker.label == "top-level" else f"{tracker.label}."
		paths.update(f"{prefix}{key}" for key in tracker.accessed)

	return paths


def _value_schema (entry: dict[str, typing.Any]) -> dict[str, typing.Any]:

	"""The schema a setting's value must satisfy, looking past a union with null."""

	for option in entry.get("anyOf", ()):
		if option.get("type") != "null":
			return typing.cast(dict[str, typing.Any], option)

	return entry


def _leaves () -> dict[str, dict[str, typing.Any]]:

	"""The settings that hold a value rather than further settings."""

	return {
		path: entry for path, entry in _declared(_SCHEMA).items()
		if not (entry.get("type") == "object" and "properties" in entry)
	}


def _outside_limit (keyword: str, bound: typing.Any) -> typing.Any:

	"""A value that a declared limit refuses."""

	if keyword == "enum":
		numbers = [value for value in bound if isinstance(value, (int, float))]
		return max(numbers) + 1 if numbers else "not-a-valid-value"

	return {
		"minimum":          lambda: bound - 1,
		"exclusiveMinimum": lambda: bound,
		"maximum":          lambda: bound + 1,
		"exclusiveMaximum": lambda: bound,
	}[keyword]()


def _limit_cases () -> list[typing.Any]:

	"""One case per declared limit on every setting."""

	cases = []

	for path, entry in _leaves().items():
		value_schema = _value_schema(entry)

		for keyword in ("enum", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"):
			if keyword in value_schema:
				cases.append(pytest.param(path, keyword, value_schema[keyword], id=f"{path}-{keyword}"))

	return cases


def _write_override (tmp_path: pathlib.Path, path: str, value: typing.Any) -> pathlib.Path:

	"""Write a config.yaml that sets one dotted key over the shipped defaults."""

	override: dict[str, typing.Any] = {}
	node = override
	parts = path.split(".")

	for part in parts[:-1]:
		node = node.setdefault(part, {})

	node[parts[-1]] = value

	config_file = tmp_path / "config.yaml"
	config_file.write_text(yaml.safe_dump(override))

	return config_file


class TestSchemaAgreesWithBuilder:

	def test_every_declared_key_is_read_by_the_builder (self) -> None:

		"""A key the schema documents but the builder never reads would be documented and ignored."""

		unread = set(_declared(_SCHEMA)) - _consulted()

		assert not unread, f"declared in config_schema but never read by the builder: {sorted(unread)}"

	def test_every_key_the_builder_reads_is_declared (self) -> None:

		"""A key the builder reads but the schema omits would work and be undocumented."""

		undeclared = _consulted() - set(_RETIRED_KEYS) - set(_declared(_SCHEMA))

		assert not undeclared, f"read by the builder but not declared in config_schema: {sorted(undeclared)}"

	@pytest.mark.parametrize("path", sorted(_RETIRED_KEYS))
	def test_retired_key_is_refused (self, tmp_path: pathlib.Path, path: str) -> None:

		"""Each retired key is still refused, so the list of keys the schema may leave out stays honest."""

		config_file = _write_override(tmp_path, path, _RETIRED_KEYS[path])

		with pytest.raises(ValueError):
			subsample.config.load_config(config_file)

	def test_declared_defaults_match_the_shipped_configuration (self) -> None:

		"""Every declared default is the value the builder produces from config.yaml.default."""

		cfg = subsample.config.load_config(subsample.config._locate_default_config())
		mismatched: list[str] = []

		for path, entry in _leaves().items():
			if "default" not in entry or path in _DERIVED_DEFAULTS:
				continue

			loaded: typing.Any = cfg

			for part in path.split("."):
				loaded = getattr(loaded, part)

			if isinstance(loaded, tuple):
				loaded = list(loaded)

			if loaded != entry["default"]:
				mismatched.append(f"{path}: declared {entry['default']!r}, builder gives {loaded!r}")

		assert not mismatched, "\n".join(mismatched)

	def test_declared_types_match_the_shipped_configuration (self) -> None:

		"""Every setting's loaded value, where it has one, is of the type the schema declares."""

		cfg = subsample.config.load_config(subsample.config._locate_default_config())
		mistyped: list[str] = []

		for path, entry in _leaves().items():
			loaded: typing.Any = cfg

			for part in path.split("."):
				loaded = getattr(loaded, part)

			if loaded is None:
				continue

			declared = _value_schema(entry)["type"]

			if not _JSON_TYPES[declared](loaded):
				mistyped.append(f"{path}: declared {declared}, builder gives {type(loaded).__name__}")

		assert not mistyped, "\n".join(mistyped)

	@pytest.mark.parametrize(("path", "keyword", "bound"), _limit_cases())
	def test_value_outside_a_declared_limit_is_refused (
		self, tmp_path: pathlib.Path, path: str, keyword: str, bound: typing.Any,
	) -> None:

		"""A limit the schema publishes is one the builder enforces."""

		config_file = _write_override(tmp_path, path, _outside_limit(keyword, bound))

		with pytest.raises(ValueError):
			subsample.config.load_config(config_file)


class TestSchemaIsPublishable:

	def test_shipped_file_loads_from_a_string_path (self) -> None:

		"""A documentation build names the shipped file as a string and loads it with load_config."""

		cfg = subsample.config.load_config(str(subsample.config._locate_default_config()))

		assert isinstance(cfg, subsample.config.Config)

	def test_schema_is_json (self) -> None:

		"""The schema survives a JSON round trip, which is how a documentation build reads it."""

		assert json.loads(json.dumps(_SCHEMA)) == _SCHEMA

	def test_every_section_and_setting_is_described (self) -> None:

		"""A reference page has nothing to say about a setting with no description."""

		undescribed = [path for path, entry in _declared(_SCHEMA).items() if not entry.get("description", "").strip()]

		assert not undescribed

	def test_no_description_contains_an_em_dash (self) -> None:

		"""The documentation site refuses to publish an em dash."""

		dashed = [path for path, entry in _declared(_SCHEMA).items() if "—" in entry["description"]]

		assert "—" not in _SCHEMA["description"]
		assert not dashed
