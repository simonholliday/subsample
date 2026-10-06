"""What every test shares."""

import typing

import pytest

import subsample.parallelism


@pytest.fixture(autouse=True)
def _stop_shared_pools () -> typing.Iterator[None]:

	"""Stop the session's shared worker pools after each test that started one.

	A pool's manager thread would otherwise outlive its test, and the tests
	that need this process to look forkable (parallelism.can_fork_safely)
	would fail or skip after it.
	"""

	yield

	subsample.parallelism.shutdown_shared_pools()
