#!/usr/bin/env python3
"""Upload a file or staged LFM model into the running CVA6 Linux system."""

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


def read_ack(port: serial.Serial, timeout: float) -> None:
    """Consume delayed console residue without losing post-ACK checksum bytes."""
    deadline = time.monotonic() + timeout
    residue = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for block acknowledgement; residue={bytes(residue)!r}")
        byte = read_exact(port, 1, remaining)
        if byte == b"A":
            if residue:
                print(f"Ignored delayed UART residue {bytes(residue)!r}.")
            return
        residue.extend(byte)
        if len(residue) > 4096:
            raise RuntimeError(f"excess UART residue before acknowledgement: {bytes(residue[:80])!r}")


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


def synchronize_shell(port: serial.Serial, allow_login: bool) -> None:
    """Request a fresh prompt so stale UART state cannot eat the command."""
    port.reset_input_buffer()
    port.write(b"\n")
    port.flush()
    state = wait_for_any(port, (b"buildroot login:", b"# "), 600.0)
    if state == b"buildroot login:":
        if not allow_login:
            raise RuntimeError("Linux is at the login prompt; omit --already-logged-in")
        write_paced(port, b"root\n", 200, 1)
        wait_for(port, b"# ", 30.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--rate-kib", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--ack-timeout", type=float, default=30.0)
    parser.add_argument("--destination", default="@0xaf000000")
    parser.add_argument("--already-logged-in", action="store_true")
    args = parser.parse_args()

    if args.rate_kib <= 0 or args.chunk_size <= 0 or args.ack_timeout <= 0:
        raise SystemExit("upload rate, chunk size, and acknowledgement timeout must be positive")

    size = args.model.stat().st_size
    if size >= 2**32:
        raise SystemExit("model is too large for the upload protocol")

    with serial.Serial(args.serial, baudrate=args.baud, timeout=0.1, write_timeout=10) as port:
        if args.already_logged_in:
            print("Synchronizing with the running shell...")
            synchronize_shell(port, allow_login=False)
        else:
            print("Waiting for Linux login or shell...")
            synchronize_shell(port, allow_login=True)
        command = f"model-receive {args.destination} {size}\n".encode()
        # Command traffic does not need bulk-transfer throughput. Sending one
        # byte at a time avoids first-character loss at 3 Mbaud.
        write_paced(port, command, 200, 1)
        wait_for(port, b"MODEL_READY\n", 10.0)
        time.sleep(0.1)
        port.reset_input_buffer()

        checksum = 2166136261
        sent = 0
        started = time.monotonic()
        next_report = 0
        with args.model.open("rb") as source:
            while block := source.read(4096):
                write_paced(port, block, args.rate_kib * 1024, args.chunk_size)
                read_ack(port, args.ack_timeout)
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
