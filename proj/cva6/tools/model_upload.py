#!/usr/bin/env python3
"""Upload an LFM model into the running CVA6 Linux RAM filesystem."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
import time

import serial


def wait_for_any(port: serial.Serial, markers: tuple[bytes, ...], timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while True:
        for marker in markers:
            if marker in received:
                return marker
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for one of {markers!r}")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            received.extend(chunk)
            if len(received) > 8192:
                del received[:-4096]


def wait_for(port: serial.Serial, marker: bytes, timeout: float) -> None:
    wait_for_any(port, (marker,), timeout)


def read_exact(port: serial.Serial, count: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while len(received) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {count} bytes")
        chunk = port.read(count - len(received))
        if chunk:
            received.extend(chunk)
    return bytes(received)


def fnv1a(data: bytes, value: int) -> int:
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def write_paced(port: serial.Serial, data: bytes, rate: int, chunk_size: int) -> None:
    started = time.monotonic()
    sent = 0
    while sent < len(data):
        chunk = data[sent:sent + chunk_size]
        port.write(chunk)
        port.flush()
        sent += len(chunk)
        delay = started + sent / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--rate-kib", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--destination", default="/root/models/LFM2.5-350M-Q4_0.gguf")
    parser.add_argument("--already-logged-in", action="store_true")
    args = parser.parse_args()

    if args.rate_kib <= 0 or args.chunk_size <= 0:
        raise SystemExit("upload rate and chunk size must be positive")

    size = args.model.stat().st_size
    if size >= 2**32:
        raise SystemExit("model is too large for the upload protocol")

    with serial.Serial(args.serial, baudrate=args.baud, timeout=0.1, write_timeout=10) as port:
        port.reset_input_buffer()
        if not args.already_logged_in:
            print("Waiting for Linux login or shell...")
            # A fresh getty may already be sitting at a prompt whose bytes were
            # consumed by an earlier serial client. An empty line makes getty
            # print the prompt again without relying on buffered old output.
            port.write(b"\n")
            port.flush()
            state = wait_for_any(port, (b"buildroot login:", b"# "), 600.0)
            if state == b"buildroot login:":
                port.write(b"root\n")
                port.flush()
                wait_for(port, b"# ", 30.0)
        command = f"model-receive {args.destination} {size}\n".encode()
        port.write(command)
        port.flush()
        wait_for(port, b"MODEL_READY\n", 10.0)

        checksum = 2166136261
        sent = 0
        started = time.monotonic()
        next_report = 0
        with args.model.open("rb") as source:
            while block := source.read(4096):
                write_paced(port, block, args.rate_kib * 1024, args.chunk_size)
                acknowledgement = read_exact(port, 1, 5.0)
                if acknowledgement != b"A":
                    raise RuntimeError(f"model receiver returned {acknowledgement!r}")
                checksum = fnv1a(block, checksum)
                sent += len(block)
                if sent >= next_report or sent == size:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(f"\rModel upload: {sent * 100 / size:6.2f}%  {sent / elapsed / 1024:7.1f} KiB/s", end="", flush=True)
                    next_report = sent + 1024 * 1024
        print()

        remote_checksum = struct.unpack("<I", read_exact(port, 4, 5.0))[0]
        if remote_checksum != checksum:
            raise RuntimeError(f"model checksum mismatch: host {checksum:08x}, target {remote_checksum:08x}")
        wait_for(port, f"MODEL_DONE {checksum:08x}".encode(), 5.0)
        print(f"Model checksum passed ({checksum:08x}).")


if __name__ == "__main__":
    main()
