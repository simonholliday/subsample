"""Lightweight event emitter for inter-module communication.

Provides a simple `.on()` / `.emit()` system so that integrations (OSC sender,
Supervisor dashboard, etc.) can subscribe to sample and MIDI events without
tightly coupling to the callback chains in cli.py.

Handlers are called synchronously on the emitting thread.  Exceptions in
individual handlers are logged and swallowed so one broken subscriber cannot
break others.
"""

import logging
import threading
import typing


_log = logging.getLogger(__name__)


class EventEmitter:

	"""Subscribe to named events and dispatch kwargs to all registered handlers.

	Thread-safe: ``on()`` and ``emit()`` are guarded by a lock, and ``emit()``
	snapshots the handler list before calling (so a subscriber registering
	mid-dispatch can't corrupt the iteration).  Registration typically happens
	at startup while emits come from worker threads.
	"""

	def __init__ (self) -> None:

		self._handlers: dict[str, list[typing.Callable[..., None]]] = {}
		self._lock:     threading.Lock = threading.Lock()

	def on (self, event: str, handler: typing.Callable[..., None]) -> None:

		"""Register a handler for the given event name."""

		with self._lock:
			self._handlers.setdefault(event, []).append(handler)

	def emit (self, event: str, **kwargs: typing.Any) -> None:

		"""Dispatch kwargs to all handlers registered for this event.

		Handlers are called in registration order.  If a handler raises,
		the exception is logged and remaining handlers still run.
		"""

		# Snapshot under the lock, then call handlers outside it — a handler
		# must not be able to deadlock by (un)subscribing during dispatch.
		with self._lock:
			handlers = list(self._handlers.get(event, []))

		for handler in handlers:
			try:
				handler(**kwargs)
			except Exception:
				_log.warning("Event handler error for %r", event, exc_info=True)
