#!/usr/bin/env python3
"""Log in on the board UART, launch Bonsai chat, and leave it at its prompt."""

from __future__ import annotations

import argparse
import sys
import time

import serial


def wait_for(port: serial.Serial, marker: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while marker not in received:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {marker!r}")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            received.extend(chunk)
            if len(received) > 65536:
                del received[:-32768]
    return bytes(received)


def send_line(port: serial.Serial, line: str, delay: float) -> None:
    for byte in line.encode() + b"\r":
        port.write(bytes((byte,)))
        port.flush()
        if delay > 0:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--character-delay-us", type=float, default=200.0)
    parser.add_argument(
        "--command",
        default=(
            "export BONSAI_MODEL=/root/models/Bonsai-1.7B.gguf "
            "BONSAI_MODEL_PRESTAGED=1 BONSAI_PARALLEL_ROWS=1; "
            "exec /usr/bin/bonsai-chat --no-progress"
        ),
    )
    args = parser.parse_args()
    delay = args.character_delay_us / 1_000_000

    with serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=10) as port:
        wait_for(port, b"buildroot login: ", args.timeout)
        send_line(port, "root", delay)
        wait_for(port, b"# ", 30.0)
        send_line(port, args.command, delay)
        wait_for(port, b"you> ", args.timeout)
        print("\nBonsai chat is ready on the FPGA console.")


if __name__ == "__main__":
    main()
