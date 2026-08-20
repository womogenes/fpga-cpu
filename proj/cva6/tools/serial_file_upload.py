#!/usr/bin/env python3
"""Copy a modest file through the running Linux serial shell.

This fallback deliberately uses only printable shell commands, so it can
install a repaired binary even when the target's binary receiver is broken.
Each base64 fragment decodes independently and is acknowledged by the shell.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import shlex
import sys
import time

import serial


ACK = b"\x1e"


def wait_for(port: serial.Serial, marker: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while marker not in received:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {marker!r}: {received[-200:]!r}")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            received.extend(chunk)
    return bytes(received)


def wait_for_any(port: serial.Serial, markers: tuple[bytes, ...],
                 timeout: float) -> tuple[bytes, bytes]:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            received.extend(chunk)
            for marker in markers:
                if marker in received:
                    return marker, bytes(received)
    raise TimeoutError(f"timed out waiting for {markers!r}: {received[-200:]!r}")


def write_paced(port: serial.Serial, data: bytes, rate: int) -> None:
    started = time.monotonic()
    sent = 0
    while sent < len(data):
        # The Linux UART driver can drop characters while the slow CVA6 is
        # handling canonical input.  One-byte writes at a conservative rate
        # are still quick for runtime binaries and preserve redirections such
        # as ">>" exactly.
        chunk = data[sent:sent + 1]
        port.write(chunk)
        port.flush()
        sent += len(chunk)
        delay = started + sent / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def shell_command(
    port: serial.Serial, command: str, rate: int, timeout: float = 30.0
) -> bytes:
    write_paced(port, command.encode() + b"\n", rate)
    return wait_for(port, ACK, timeout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    # Base64 expands by 4/3 and the surrounding shell pipeline adds roughly
    # 100 bytes.  Keep the complete command comfortably below Linux N_TTY's
    # 4096-byte canonical-input limit; crossing it silently corrupts a line.
    parser.add_argument("--raw-chunk", type=int, default=2_400)
    parser.add_argument("--rate", type=int, default=1_500)
    args = parser.parse_args()
    if args.raw_chunk <= 0 or args.raw_chunk % 3:
        raise SystemExit("--raw-chunk must be a positive multiple of three")
    if args.raw_chunk > 2_700:
        raise SystemExit("--raw-chunk must keep the encoded shell line below 4096 bytes")
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")

    data = args.source.read_bytes()
    destination = shlex.quote(args.destination)
    with serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=10) as port:
        port.reset_input_buffer()
        port.write(b"\x03\n")
        marker, _ = wait_for_any(port, (b"# ", b"$ ", b"login: "), 10.0)
        if marker == b"login: ":
            write_paced(port, b"root\n", 100)
            wait_for_any(port, (b"# ", b"$ "), 30.0)
        shell_command(port, f"rm -f {destination}; printf '\\036'", args.rate)
        count = (len(data) + args.raw_chunk - 1) // args.raw_chunk
        for index, offset in enumerate(range(0, len(data), args.raw_chunk), 1):
            encoded = base64.b64encode(data[offset:offset + args.raw_chunk]).decode()
            output = shell_command(
                port,
                f"printf %s {shlex.quote(encoded)} | base64 -d >> {destination} && printf '\\036'",
                args.rate,
            )
            if b"base64:" in output or b"not found" in output:
                raise RuntimeError(output.decode(errors="replace"))
            print(f"\rUpload: {index * 100 / count:6.2f}%", end="", flush=True)
        print()
        output = shell_command(
            port,
            f"chmod 0755 {destination}; sha256sum {destination}; printf '\\036'",
            args.rate,
        )
        sys.stdout.write(output.decode(errors="replace"))


if __name__ == "__main__":
    main()
