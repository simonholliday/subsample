"""What the package is called where a reader installs it (#5018).

Subsample is `subsampler` on PyPI, where `subsample` is another project's
package, so a line that says how to install it must name what pyproject.toml
declares.  One that named `subsample` would install somebody else's tool.
"""

import pathlib
import tomllib
import typing

import subsample.cli


_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _project () -> dict[str, typing.Any]:

	"""The [project] table of pyproject.toml."""

	loaded: dict[str, typing.Any] = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

	project: dict[str, typing.Any] = loaded["project"]

	return project


class TestInstallName:

	"""The name a reader installs, and the names that stay `subsample`."""

	def test_only_the_install_name_differs (self) -> None:

		"""The distribution is subsampler; the command it installs is subsample."""

		project = _project()

		assert project["name"] == "subsampler"
		assert project["scripts"] == {"subsample": "subsample.cli:main"}

	def test_the_osc_warning_names_the_distribution_and_its_extra (self) -> None:

		"""What Subsample says when OSC is on without python-osc installs the right package."""

		project = _project()

		assert "osc" in project["optional-dependencies"]
		assert subsample.cli._OSC_INSTALL == f"pip install '{project['name']}[osc]'"

	def test_the_shipped_config_names_the_distribution_and_its_extra (self) -> None:

		"""The comment above the osc settings gives the same install line."""

		project = _project()
		shipped = (_ROOT / "subsample" / "data" / "config.yaml.default").read_text(encoding="utf-8")

		assert f'pip install "{project["name"]}[osc]"' in shipped
		assert "subsample[osc]" not in shipped

	def test_the_readme_installs_by_that_name (self) -> None:

		"""The README is also the PyPI page, and its install line names the distribution."""

		project = _project()
		readme = (_ROOT / "README.md").read_text(encoding="utf-8")

		assert f"uv tool install {project['name']}\n" in readme
