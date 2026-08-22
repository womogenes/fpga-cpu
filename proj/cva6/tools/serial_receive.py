#!/usr/bin/env python3
"""Capture a UART without transmitting bytes during a boot-ROM window."""

from __future__ import annotations

import argparse
import sys
import time

import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--until", action="append", default=[])
    args = parser.parse_args()
    markers = tuple(value.encode() for value in args.until)
    deadline = time.monotonic() + args.timeout
    received = bytearray()
    with serial.Serial(args.serial, args.baud, timeout=0.1) as port:
        while time.monotonic() < deadline:
            chunk = port.read(port.in_waiting or 1)
            if not chunk:
                continue
            received.extend(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            if markers and any(marker in received for marker in markers):
                return
    raise TimeoutError(f"UART receive timeout; tail={bytes(received[-1000:])!r}")


if __name__ == "__main__":
    main()
