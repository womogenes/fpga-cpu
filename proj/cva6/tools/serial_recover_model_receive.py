#!/usr/bin/env python3
"""Drain an interrupted model-receive invocation back to its shell."""

from __future__ import annotations

import argparse
import sys
import time

import serial


BLOCK_SIZE = 4096


def write_paced(port: serial.Serial, data: bytes, rate: int,
                chunk_size: int) -> None:
    started = time.monotonic()
    sent = 0
    for offset in range(0, len(data), chunk_size):
        chunk = data[offset:offset + chunk_size]
        port.write(chunk)
        port.flush()
        sent += len(chunk)
        delay = started + sent / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--rate-kib", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-blocks", type=int, default=2048)
    parser.add_argument("--ack-timeout", type=float, default=15.0)
    args = parser.parse_args()

    padding = bytes(BLOCK_SIZE)
    with serial.Serial(args.serial, args.baud, timeout=0.1,
                       write_timeout=10) as port:
        port.reset_input_buffer()
        for block_index in range(args.max_blocks):
            write_paced(port, padding, args.rate_kib * 1024,
                        args.chunk_size)
            deadline = time.monotonic() + args.ack_timeout
            received = bytearray()
            while time.monotonic() < deadline:
                chunk = port.read(port.in_waiting or 1)
                if not chunk:
                    continue
                received.extend(chunk)
                if b"MODEL_DONE " in received:
                    sys.stdout.buffer.write(received)
                    print(f"\nMODEL_RECEIVER_RECOVERED after {block_index + 1} blocks")
                    return
                if b"A" in received:
                    if (block_index + 1) % 128 == 0:
                        print(f"Recovery padding: {block_index + 1} blocks",
                              flush=True)
                    break
            else:
                raise TimeoutError(
                    f"no receiver acknowledgement after block {block_index + 1}; "
                    f"tail={bytes(received[-200:])!r}"
                )
    raise RuntimeError("model-receive did not finish within the padding limit")


if __name__ == "__main__":
    main()
