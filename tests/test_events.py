"""Tests for subsample.events — EventEmitter."""

import unittest.mock

import subsample.events


class TestEventEmitter:

	def test_emit_calls_handler (self) -> None:
		"""A registered handler is called when the event fires."""

		emitter = subsample.events.EventEmitter()
		handler = unittest.mock.MagicMock()

		emitter.on("test", handler)
		emitter.emit("test")

		handler.assert_called_once_with()

	def test_emit_with_kwargs (self) -> None:
		"""Kwargs are forwarded to the handler."""

		emitter = subsample.events.EventEmitter()
		handler = unittest.mock.MagicMock()

		emitter.on("test", handler)
		emitter.emit("test", name="kick", duration=1.5)

		handler.assert_called_once_with(name="kick", duration=1.5)

	def test_multiple_handlers (self) -> None:
		"""Multiple handlers on the same event are all called."""

		emitter = subsample.events.EventEmitter()
		handler_a = unittest.mock.MagicMock()
		handler_b = unittest.mock.MagicMock()

		emitter.on("test", handler_a)
		emitter.on("test", handler_b)
		emitter.emit("test", value=42)

		handler_a.assert_called_once_with(value=42)
		handler_b.assert_called_once_with(value=42)

	def test_handler_exception_does_not_propagate (self) -> None:
		"""An exception in one handler does not prevent others from running."""

		emitter = subsample.events.EventEmitter()
		bad_handler = unittest.mock.MagicMock(side_effect=RuntimeError("boom"))
		good_handler = unittest.mock.MagicMock()

		emitter.on("test", bad_handler)
		emitter.on("test", good_handler)
		emitter.emit("test", x=1)

		bad_handler.assert_called_once_with(x=1)
		good_handler.assert_called_once_with(x=1)

	def test_no_handlers_is_silent (self) -> None:
		"""Emitting an event with no subscribers does nothing and does not raise."""

		emitter = subsample.events.EventEmitter()

		# Should not raise.
		emitter.emit("nonexistent", foo="bar")

	def test_separate_events (self) -> None:
		"""Handlers on event A do not fire when event B is emitted."""

		emitter = subsample.events.EventEmitter()
		handler_a = unittest.mock.MagicMock()
		handler_b = unittest.mock.MagicMock()

		emitter.on("alpha", handler_a)
		emitter.on("beta", handler_b)
		emitter.emit("alpha", v=1)

		handler_a.assert_called_once_with(v=1)
		handler_b.assert_not_called()
