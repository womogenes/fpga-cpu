#!/usr/bin/env python3
"""Copy a small file into a running Linux system through its serial shell."""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
import shlex
import sys
import time

import serial


def write_paced(port: serial.Serial, data: bytes, delay: float, burst: int) -> None:
    for offset in range(0, len(data), burst):
        chunk = data[offset : offset + burst]
        port.write(chunk)
        port.flush()
        if delay > 0:
            time.sleep(delay * len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--char-delay-us", type=float, default=100.0)
    parser.add_argument("--burst", type=int, default=8, help="bytes sent before each pacing delay")
    parser.add_argument("--mode", default="0644")
    args = parser.parse_args()
    if args.burst <= 0:
        raise SystemExit("burst must be positive")

    source = args.source.read_bytes()
    expected_hash = hashlib.sha256(source).hexdigest()
    encoded = base64.b64encode(source)
    # Keep each here-document line comfortably below the Linux tty canonical
    # input limit. Long Base64 lines can overrun or be discarded before Bash
    # has a chance to feed them to the decoder on a small single-hart system.
    payload = b"\n".join(
        encoded[offset : offset + 76]
        for offset in range(0, len(encoded), 76)
    )
    destination = shlex.quote(args.destination)
    temporary = shlex.quote(args.destination + ".serial-push.tmp")
    # Feed the decoder directly and terminate its stdin with the tty EOF
    # character. An interactive Bash here-document prints a secondary prompt
    # for every encoded line and can fall behind the UART badly enough to
    # overrun the RX FIFO.
    header = f"\nstty -echo -ixon\nbase64 -d > {temporary}\n".encode()
    trailer = (
        "\n\x04stty echo\n"
        f"actual=$(sha256sum {temporary} | cut -d' ' -f1)\n"
        f"if [ \"$actual\" = {expected_hash} ]; then "
        f"chmod {shlex.quote(args.mode)} {temporary} && mv -f {temporary} {destination} && "
        f"echo PUSH_OK {expected_hash}; else rm -f {temporary}; "
        f"echo PUSH_BAD $actual {expected_hash}; fi\n"
    ).encode()
    delay = args.char_delay_us / 1_000_000.0

    with serial.Serial(args.serial, args.baud, timeout=0.1) as port:
        write_paced(port, header, delay, args.burst)
        time.sleep(1.0)
        port.reset_input_buffer()
        for offset in range(0, len(payload), 4096):
            write_paced(port, payload[offset : offset + 4096] + b"\n", delay, args.burst)
        write_paced(port, trailer, delay, args.burst)

        deadline = time.monotonic() + 30.0
        output = bytearray()
        success = f"\r\nPUSH_OK {expected_hash}\r\n".encode()
        failure = b"\r\nPUSH_BAD "
        while time.monotonic() < deadline:
            chunk = port.read(port.in_waiting or 1)
            if chunk:
                output.extend(chunk)
                if success in output or failure in output:
                    break
        sys.stdout.buffer.write(output)
        if success not in output:
            raise SystemExit("serial push did not pass its target-side SHA-256 check")


if __name__ == "__main__":
    main()
