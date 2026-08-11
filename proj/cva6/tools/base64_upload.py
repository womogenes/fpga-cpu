#!/usr/bin/env python3
"""Reliably copy a small file through an interactive Linux serial shell."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import shlex
import time

import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--rate", type=int, default=2_500)
    args = parser.parse_args()
    if args.rate <= 0:
        raise SystemExit("rate must be positive")

    encoded = base64.encodebytes(args.source.read_bytes())
    destination = shlex.quote(args.destination)
    marker = "CODEX_BASE64_UPLOAD_EOF"
    payload = (
        f"stty -echo\nbase64 -d > {destination} <<'{marker}'\n".encode()
        + encoded
        + f"{marker}\nstty echo\nsha256sum {destination}\nsync\necho CODEX_\"\"UPLOAD_OK\n".encode()
    )

    with serial.Serial(args.serial, args.baud, timeout=0.05, write_timeout=10) as port:
        port.reset_input_buffer()
        started = time.monotonic()
        sent = 0
        next_report = 0
        while sent < len(payload):
            block = payload[sent : sent + 64]
            port.write(block)
            port.flush()
            sent += len(block)
            # Drain secondary prompts and diagnostics while the here-document
            # is active so the host-side USB input queue cannot fill.
            if port.in_waiting:
                port.read(port.in_waiting)
            delay = started + sent / args.rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if sent >= next_report or sent == len(payload):
                print(f"upload {sent * 100 / len(payload):5.1f}%", flush=True)
                next_report = sent + max(len(payload) // 20, 1)

        deadline = time.monotonic() + 30
        received = bytearray()
        while b"CODEX_UPLOAD_OK" not in received and time.monotonic() < deadline:
            chunk = port.read(port.in_waiting or 1)
            if chunk:
                received.extend(chunk)
        print(received.decode(errors="replace"), flush=True)
        if b"CODEX_UPLOAD_OK" not in received:
            raise SystemExit("target did not confirm upload")


if __name__ == "__main__":
    main()
