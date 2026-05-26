"""Measure MIDI-to-dispatch latency end-to-end.

Manual smoke test for the callback-mode MIDI dispatch.  Opens a virtual
MIDI input port, opens an output port wired to it, sends timestamped
note_on messages, and records when each message lands in the player's
callback.  Reports median + 95th-percentile dispatch latency.

Usage:

    python scripts/measure_midi_latency.py --count 1000 --interval-ms 5

With the old 10 ms polling loop, the same code would report median ≈ 5 ms
and 95th-percentile ≈ 10 ms.  After the callback-mode switch, expect
median ≪ 1 ms and 95th-percentile < 2 ms on a quiet Linux system.

Notes:
  - Sub-millisecond timing on Python depends on system load and the rtmidi
    backend's responsiveness.  Run a few times if a single result looks
    noisy.
  - This script does NOT touch the audio output stream or PortAudio — it
    only measures MIDI dispatch latency, the most important and most
    fixable part of the live-performance path.
"""

import argparse
import collections
import statistics
import sys
import threading
import time

import mido


_VIRTUAL_PORT_NAME = "subsample-latency-measure"


def _percentile (values: list[float], pct: float) -> float:

	"""Return the ``pct``-th percentile of ``values`` (0.0-100.0)."""

	if not values:
		return float("nan")

	sorted_values = sorted(values)
	idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
	return sorted_values[idx]


def main () -> int:

	parser = argparse.ArgumentParser(
		description="Measure MIDI-to-dispatch latency for the subsample player.",
	)
	parser.add_argument(
		"--count",
		type=int,
		default=1000,
		help="Number of MIDI messages to send (default: 1000).",
	)
	parser.add_argument(
		"--interval-ms",
		type=float,
		default=5.0,
		help=(
			"Wall-clock spacing between sent messages, in ms (default: 5).  "
			"Lower values stress-test sustained bursts; higher values "
			"isolate worst-case scheduling jitter."
		),
	)
	args = parser.parse_args()

	# Open the virtual port that simulates an external controller.
	# rtmidi creates this port as a system-level destination so we can
	# also open a sender attached to it.  Both ports are tracked through
	# a try/finally so a Ctrl+C mid-run doesn't leave the virtual port
	# registered with ALSA (visible in `aconnect -l` until the process
	# exits).
	in_port  = mido.open_input(_VIRTUAL_PORT_NAME, virtual=True)
	out_port = mido.open_output(_VIRTUAL_PORT_NAME)

	try:
		# We can't carry a timestamp inside the MIDI message itself — mido
		# reconstructs the receiving Message from raw MIDI bytes, which
		# don't carry timestamps.  Instead, push each send timestamp into
		# a FIFO deque and pop the oldest on receive.  Safe because mido +
		# rtmidi deliver messages in order on a single port.
		sent_times_ns: collections.deque[int] = collections.deque()
		sent_lock = threading.Lock()

		timings_ns: list[int] = []
		timings_lock = threading.Lock()

		def _callback (msg: mido.Message) -> None:
			now_ns = time.perf_counter_ns()
			with sent_lock:
				if not sent_times_ns:
					return
				sent_ns = sent_times_ns.popleft()
			with timings_lock:
				timings_ns.append(now_ns - sent_ns)

		in_port.callback = _callback

		interval_s = args.interval_ms / 1000.0

		print(f"Sending {args.count} message(s) at {args.interval_ms:.2f} ms spacing…")

		send_start = time.perf_counter()

		for i in range(args.count):
			msg = mido.Message(
				"note_on",
				channel=0,
				note=60 + (i % 12),
				velocity=64,
			)
			sent_ns = time.perf_counter_ns()
			with sent_lock:
				sent_times_ns.append(sent_ns)
			out_port.send(msg)

			# Sleep the configured interval (busy-wait the small remainder
			# so the schedule is honoured tightly).
			target = send_start + (i + 1) * interval_s
			while True:
				remaining = target - time.perf_counter()
				if remaining <= 0:
					break
				if remaining > 0.001:
					time.sleep(remaining - 0.0005)

		# Give rtmidi a generous window to deliver the last few messages.
		time.sleep(0.5)

		with timings_lock:
			samples = list(timings_ns)

	finally:
		in_port.callback = None
		in_port.close()
		out_port.close()

	if not samples:
		print("ERROR: No messages received — check that rtmidi virtual ports work on this system.", file=sys.stderr)
		return 1

	delivered = len(samples)
	timings_ms = [t / 1e6 for t in samples]

	mean_ms   = statistics.fmean(timings_ms)
	median_ms = statistics.median(timings_ms)
	p95_ms    = _percentile(timings_ms, 95.0)
	p99_ms    = _percentile(timings_ms, 99.0)
	max_ms    = max(timings_ms)

	print()
	print(f"Delivered: {delivered}/{args.count}")
	print(f"  mean    : {mean_ms:7.3f} ms")
	print(f"  median  : {median_ms:7.3f} ms")
	print(f"  95th-pct: {p95_ms:7.3f} ms")
	print(f"  99th-pct: {p99_ms:7.3f} ms")
	print(f"  max     : {max_ms:7.3f} ms")
	print()
	print(
		"After the callback-mode fix, expect median < 1 ms and 95th-pct "
		"< 2 ms.  The old 10 ms polling loop reported median ≈ 5 ms.",
	)

	return 0


if __name__ == "__main__":
	sys.exit(main())
