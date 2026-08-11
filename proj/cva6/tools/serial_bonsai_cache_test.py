#!/usr/bin/env python3
"""Measure Bonsai server startup and continued-prefix reuse over UART."""

from __future__ import annotations

import argparse
import re
import sys
import time

import serial


def write_paced(port: serial.Serial, data: bytes) -> None:
    for byte in data:
        port.write(bytes((byte,)))
        port.flush()
        time.sleep(0.005)


def receive_until_any(port: serial.Serial, markers: tuple[bytes, ...],
                      timeout: float) -> tuple[bytes, bytes]:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while True:
        for marker in markers:
            if marker in received:
                return marker, bytes(received)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {markers!r}")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            received.extend(chunk)


def receive_until(port: serial.Serial, marker: bytes, timeout: float) -> bytes:
    return receive_until_any(port, (marker,), timeout)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--model", default="/root/models/Bonsai-1.7B.gguf")
    args = parser.parse_args()

    command = (
        f"BONSAI_STARTUP_PROFILE=1 BONSAI_PARALLEL_ROWS=1 "
        f"/usr/bin/bonsai-lite --model {args.model} --server\n"
    ).encode()
    with serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=10) as port:
        port.reset_input_buffer()
        started = time.monotonic()
        write_paced(port, command)
        state, _ = receive_until_any(port, (b"READY", b"~ # ", b"cva6"), 1200)
        if state != b"READY":
            raise RuntimeError("Bonsai exited before READY")
        startup = time.monotonic() - started
        print(f"HOST_STARTUP_SECONDS {startup:.3f}", flush=True)

        write_paced(port, b"RUN 1 3 12522 5193 264\n")
        first = receive_until(port, b"DONE", 300)
        match = re.search(rb"TOKEN (\d+)", first)
        if match is None:
            raise RuntimeError("first response did not contain a token")
        emitted = int(match.group(1))
        write_paced(port, f"RUN 1 4 12522 5193 264 {emitted}\n".encode())
        second = receive_until(port, b"DONE", 300)
        prompt = re.search(rb"PROMPT (\d+) (\d+) (\d+)", second)
        if prompt is None:
            raise RuntimeError("second response did not contain prompt timing")
        evaluated, cached, nanoseconds = map(int, prompt.groups())
        print(
            f"CACHE_RESULT evaluated={evaluated} cached={cached} ns={nanoseconds}",
            flush=True,
        )
        write_paced(port, b"QUIT\n")
        if evaluated != 0 or cached != 4:
            raise SystemExit("continued prefix was not fully reused")


if __name__ == "__main__":
    main()
