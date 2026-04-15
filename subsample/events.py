"""Lightweight event emitter for inter-module communication.

Provides a simple `.on()` / `.emit()` system so that integrations (OSC sender,
Supervisor dashboard, etc.) can subscribe to sample and MIDI events without
tightly coupling to the callback chains in cli.py.

Handlers are called synchronously on the emitting thread.  Exceptions in
individual handlers are logged and swallowed so one broken subscriber cannot
break others.
"""

import logging
import typing


_log = logging.getLogger(__name__)


class EventEmitter:

	"""Subscribe to named events and dispatch kwargs to all registered handlers."""

	def __init__ (self) -> None:

		self._handlers: dict[str, list[typing.Callable[..., None]]] = {}

	def on (self, event: str, handler: typing.Callable[..., None]) -> None:

		"""Register a handler for the given event name."""

		self._handlers.setdefault(event, []).append(handler)

	def emit (self, event: str, **kwargs: typing.Any) -> None:

		"""Dispatch kwargs to all handlers registered for this event.

		Handlers are called in registration order.  If a handler raises,
		the exception is logged and remaining handlers still run.
		"""

		for handler in self._handlers.get(event, []):
			try:
				handler(**kwargs)
			except Exception:
				_log.warning("Event handler error for %r", event, exc_info=True)
