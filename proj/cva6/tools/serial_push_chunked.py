#!/usr/bin/env python3
"""Chunked, verified serial file push that survives UART input overruns.

The file is sent as base64 in small chunks. Each chunk is appended on the
target only after its SHA-256 (computed target-side on the decoded chunk)
matches; corrupted chunks are retried. Works through a logged-in shell.
"""

import argparse
import base64
import hashlib
import re
import sys
import time
from pathlib import Path

import serial


def drain(port: serial.Serial, duration: float) -> bytes:
    deadline = time.monotonic() + duration
    data = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            data.extend(chunk)
            deadline = time.monotonic() + duration
    return bytes(data)


def send_line(port: serial.Serial, text: str, char_delay: float) -> None:
    payload = (text + "\r").encode()
    # Python's sub-millisecond sleep is quantized heavily on the remote host.
    # Pace four-byte bursts instead: they fit comfortably in the UART RX FIFO
    # while avoiding one scheduler sleep and tcdrain operation per character.
    for offset in range(0, len(payload), 4):
        burst = payload[offset:offset + 4]
        port.write(burst)
        port.flush()
        time.sleep(char_delay * len(burst))


SEQUENCE = 0


def run(port: serial.Serial, command: str, char_delay: float,
        idle: float) -> str:
    """Run a command and wait for a unique end marker, so slow process
    spawns on the target cannot desynchronize command/response pairing."""
    global SEQUENCE
    SEQUENCE += 1
    marker = f"MARK{SEQUENCE}Z"
    send_line(port, f"{command}; echo {marker}", char_delay)
    deadline = time.monotonic() + max(idle, 30.0)
    data = bytearray()
    # The UART terminal echoes the command line, including the marker text.
    # Wait for its second occurrence, emitted by `echo`. Counting works across
    # CR/LF translation variants used by the several target console images.
    marker_bytes = marker.encode()
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            data.extend(chunk)
            if bytes(data).count(marker_bytes) >= 2:
                break
    return bytes(data).decode(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--char-delay-us", type=float, default=150.0)
    parser.add_argument("--chunk-bytes", type=int, default=384)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    payload = args.source.read_bytes()
    total_sha = hashlib.sha256(payload).hexdigest()
    delay = args.char_delay_us / 1_000_000

    with serial.Serial(args.serial, args.baud, timeout=0.1) as port:
        run(port, ":", delay, 0.5)
        run(port, f"rm -f {args.destination}.part", delay, 0.8)
        offset = 0
        index = 0
        while offset < len(payload):
            chunk = payload[offset:offset + args.chunk_bytes]
            chunk_b64 = base64.b64encode(chunk).decode()
            chunk_sha = hashlib.sha256(chunk).hexdigest()
            ok = False
            for attempt in range(args.retries):
                run(port,
                    f"printf %s {chunk_b64} | base64 -d > /tmp/.chunk.bin",
                    delay, 0.8)
                reply = run(port,
                            "sha256sum /tmp/.chunk.bin",
                            delay, 0.8)
                if chunk_sha not in reply:
                    print(f"chunk {index} attempt {attempt + 1} decode failed",
                          file=sys.stderr)
                    continue
                # The append itself can be corrupted by a UART overrun, so
                # confirm the resulting size and roll back on any mismatch.
                run(port,
                    f"truncate -s {offset} {args.destination}.part 2>/dev/null; "
                    f"cat /tmp/.chunk.bin >> {args.destination}.part",
                    delay, 0.8)
                reply = run(port,
                            f"wc -c < {args.destination}.part",
                            delay, 0.8)
                if re.search(rf"[\r\n]{offset + len(chunk)}[\r\n]", reply):
                    ok = True
                    break
                print(f"chunk {index} attempt {attempt + 1} append failed",
                      file=sys.stderr)
            if not ok:
                raise SystemExit(f"chunk {index} failed after retries")
            offset += len(chunk)
            index += 1
            if index % 8 == 0:
                print(f"{offset}/{len(payload)} bytes", file=sys.stderr)
        reply = run(port,
                    f"sha256sum {args.destination}.part",
                    delay, 1.5)
        if total_sha not in reply:
            raise SystemExit("final SHA mismatch")
        run(port, f"mv {args.destination}.part {args.destination}", delay, 0.8)
        print(f"PUSH_OK {total_sha}")


if __name__ == "__main__":
    main()
