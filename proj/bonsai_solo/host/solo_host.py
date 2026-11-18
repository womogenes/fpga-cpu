#!/usr/bin/env python3
"""Host side of the Bonsai solo build: talks to the boot monitor and the
Bonsai app over the Genesys 2 UART (3 Mbaud).

  solo_host.py ping
  solo_host.py memtest [--mib N]
  solo_host.py load --images DIR          upload model/scratch/sum table
  solo_host.py run --prompt-ids a,b,c -n N
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import serial

ACK, DONE, NAK = 0x06, 0x04, 0x15
APP_ADDRESS = 0x1000
MONITOR_RAM_ADDRESS = 0xC000
FW_BUILD = Path(__file__).resolve().parent.parent / "fw/build"
MODEL_PHYS = 0xAF000000
SCRATCH_PHYS = 0xBE000000
SUM_TABLE_PHYS = 0xBF000000
# Only the static part of the scratch image matters: norm slots, attention
# table, sequencer descriptors, Q/K and RoPE tables (0x0 - 0x400000).
SCRATCH_STATIC_BYTES = 0x400000
CHUNK = 1 << 20


class Board:
    def __init__(self, port: str, baud: int = 3_000_000) -> None:
        # exclusive=True takes an flock; other tools that scan USB serial
        # ports (a drone bridge on this host) skip locked ports.
        self.serial = serial.Serial(port, baud, timeout=5, exclusive=True)
        self.serial.reset_input_buffer()

    def read(self, count: int, timeout: float = 5.0) -> bytes:
        self.serial.timeout = timeout
        data = b""
        deadline = time.monotonic() + timeout
        while len(data) < count and time.monotonic() < deadline:
            try:
                data += self.serial.read(count - len(data))
            except serial.SerialException:
                # pyserial reports a spurious "readiness but no data" on
                # this FT232R; the bytes arrive on the next read.
                time.sleep(0.01)
        if len(data) != count:
            raise TimeoutError(f"expected {count} bytes, got {data!r}")
        return data

    def expect_ack(self, timeout: float = 5.0) -> None:
        byte = self.read(1, timeout)[0]
        if byte != ACK:
            raise RuntimeError(f"expected ACK, got 0x{byte:02x}")

    def u32(self, timeout: float = 5.0) -> int:
        return struct.unpack("<I", self.read(4, timeout))[0]

    def ping(self, attempts: int = 5) -> bool:
        # The first bytes after opening the port are sometimes dropped.
        for attempt in range(attempts):
            self.serial.reset_input_buffer()
            self.serial.write(b"P")
            try:
                self.expect_ack(timeout=1.0)
                break
            except TimeoutError:
                if attempt + 1 == attempts:
                    raise
        ident = self.read(5)
        if ident[:4] != b"SOLO":
            raise RuntimeError(f"unexpected monitor id {ident!r}")
        self.relocated = bool(ident[4] & 0x80)
        return bool(ident[4] & 1)

    def bootstrap(self, monitor: bytes) -> None:
        """Replace the boot ROM monitor (whose UART reads can race on the
        first gateware) with the fixed copy at 0xc000."""
        self.ping()
        if self.relocated:
            return
        for attempt in range(5):
            try:
                self.write(MONITOR_RAM_ADDRESS, monitor)
                break
            except RuntimeError as error:
                print(f"monitor upload retry: {error}")
                time.sleep(0.2)
                self.ping()
        else:
            raise RuntimeError("could not upload the relocated monitor")
        self.serial.write(b"J" + struct.pack("<I", MONITOR_RAM_ADDRESS))
        self.expect_ack()
        time.sleep(0.05)
        self.ping()
        if not self.relocated:
            raise RuntimeError("relocated monitor did not start")

    def write(self, address: int, data: bytes) -> None:
        self.serial.write(b"W" + struct.pack("<II", address, len(data)))
        self.serial.write(data)
        self.expect_ack(timeout=30 + len(data) / 200_000)
        crc = self.u32()
        if crc != zlib.crc32(data):
            raise RuntimeError(f"write CRC mismatch at 0x{address:08x}")

    def crc(self, address: int, length: int) -> int:
        self.serial.write(b"C" + struct.pack("<II", address, length))
        self.expect_ack(timeout=30 + length / 20_000_000)
        return self.u32()

    def memtest(self, address: int, length: int, seed: int) -> int:
        self.serial.write(b"T" + struct.pack("<III", address, length, seed))
        self.expect_ack(timeout=60 + length / 5_000_000)
        return self.u32()

    def upload(self, address: int, data: bytes, label: str) -> None:
        if self.crc(address, len(data)) == zlib.crc32(data):
            print(f"{label}: already resident ({len(data)} bytes)")
            return
        start = time.monotonic()
        for offset in range(0, len(data), CHUNK):
            chunk = data[offset:offset + CHUNK]
            self.write(address + offset, chunk)
            done = offset + len(chunk)
            rate = done / (time.monotonic() - start) / 1024
            print(f"\r{label}: {done >> 20}/{len(data) >> 20} MiB "
                  f"({rate:.0f} KiB/s)", end="", flush=True)
        print()
        if self.crc(address, len(data)) != zlib.crc32(data):
            raise RuntimeError(f"{label}: read-back CRC mismatch")

    def start_app(self, binary: bytes) -> None:
        self.write(APP_ADDRESS, binary)
        self.serial.write(b"J" + struct.pack("<I", APP_ADDRESS))
        self.expect_ack()

    def generate(self, prompt: list[int], count: int):
        self.serial.write(b"G" + struct.pack("<II", len(prompt), count))
        self.serial.write(struct.pack(f"<{len(prompt)}I", *prompt))
        while True:
            kind = self.read(1, timeout=600)
            if kind == b"E":
                return
            if kind == b"X":
                stage, status = struct.unpack("<II", self.read(8))
                raise RuntimeError(
                    f"accelerator error at stage {stage}, status 0x{status:08x}")
            if kind != b"T":
                raise RuntimeError(f"unexpected byte {kind!r}")
            token, *profile = struct.unpack("<7I", self.read(28))
            yield token, profile

    def quit_app(self) -> None:
        self.serial.write(b"Q")
        if self.read(1)[0] != DONE:
            raise RuntimeError("app did not return to the monitor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ping")
    memtest = sub.add_parser("memtest")
    memtest.add_argument("--mib", type=int, default=1024)
    memtest.add_argument("--seed", type=lambda v: int(v, 0), default=0x1234567)
    load = sub.add_parser("load")
    load.add_argument("--images", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--prompt-ids", required=True)
    run.add_argument("-n", type=int, default=16)
    run.add_argument("--app", type=Path,
                     default=Path(__file__).resolve().parent.parent /
                     "fw/build/bonsai.bin")
    args = parser.parse_args()

    board = Board(args.port)
    calibrated = board.ping()
    board.bootstrap((FW_BUILD / "monitor_ram.bin").read_bytes())
    print(f"monitor up (relocated={board.relocated}), "
          f"DDR calibrated={calibrated}")
    if args.command == "ping":
        return
    if args.command == "memtest":
        # Test in 64 MiB pieces so progress is visible.
        errors = 0
        total = args.mib << 20
        for offset in range(0, total, 64 << 20):
            length = min(64 << 20, total - offset)
            found = board.memtest(0x80000000 + offset, length, args.seed + offset)
            errors += found
            print(f"0x{0x80000000 + offset:08x}+{length >> 20} MiB: "
                  f"{found} errors")
        print(f"memtest: {errors} word errors in {args.mib} MiB")
        sys.exit(1 if errors else 0)
    if args.command == "load":
        board.upload(MODEL_PHYS, (args.images / "model-compact.bin").read_bytes(),
                     "model")
        scratch = (args.images / "scratch.bin").read_bytes()[:SCRATCH_STATIC_BYTES]
        board.upload(SCRATCH_PHYS, scratch, "scratch")
        sums = (args.images / "sum_table.bin").read_bytes()
        sums += bytes(-len(sums) % 32)
        board.upload(SUM_TABLE_PHYS, sums, "sum table")
        return
    if args.command == "run":
        prompt = [int(x) for x in args.prompt_ids.split(",")]
        board.start_app(args.app.read_bytes())
        start = time.monotonic()
        tokens = []
        for token, profile in board.generate(prompt, args.n):
            tokens.append(token)
            ms = profile[5] / 200_000
            print(f"token {token:6d}  {ms:7.2f} ms  "
                  f"(layers {profile[2] / 200_000:.2f} ms, "
                  f"argmax {profile[4] / 200_000:.2f} ms)", flush=True)
        elapsed = time.monotonic() - start
        board.quit_app()
        print("tokens:", ",".join(map(str, tokens)))
        print(f"{len(tokens)} tokens in {elapsed:.2f} s wall")


if __name__ == "__main__":
    main()
