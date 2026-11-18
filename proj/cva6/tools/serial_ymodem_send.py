#!/usr/bin/env python3
"""Send one file to U-Boot's ``loady`` command over a serial port."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial


SOH = 0x01
STX = 0x02
EOT = 0x04
ACK = 0x06
NAK = 0x15
CAN = 0x18
CRC_REQUEST = ord("C")


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def packet(block: int, payload: bytes) -> bytes:
    if len(payload) == 128:
        marker = SOH
    elif len(payload) == 1024:
        marker = STX
    else:
        raise ValueError(f"invalid YMODEM payload length: {len(payload)}")
    crc = crc16_xmodem(payload)
    return bytes((marker, block & 0xFF, 0xFF - (block & 0xFF))) + payload + crc.to_bytes(2, "big")


def read_byte(port: serial.Serial, deadline: float) -> int | None:
    while time.monotonic() < deadline:
        value = port.read(1)
        if value:
            return value[0]
    return None


def wait_for(port: serial.Serial, wanted: set[int], timeout: float, label: str) -> int:
    deadline = time.monotonic() + timeout
    seen: list[int] = []
    while time.monotonic() < deadline:
        value = read_byte(port, deadline)
        if value is None:
            break
        if value in wanted:
            return value
        if value == CAN:
            second = read_byte(port, min(deadline, time.monotonic() + 0.5))
            if second == CAN:
                raise RuntimeError(f"receiver cancelled while waiting for {label}")
            if second is not None:
                seen.extend((value, second))
        elif len(seen) < 32:
            seen.append(value)
    rendered = " ".join(f"{value:02x}" for value in seen)
    raise TimeoutError(f"timed out waiting for {label}; received: {rendered or 'nothing'}")


def write_command(port: serial.Serial, command: str, delay: float) -> None:
    port.write(b"\r")
    port.flush()
    time.sleep(0.2)
    port.reset_input_buffer()
    for value in command.encode("ascii") + b"\r":
        port.write(bytes((value,)))
        port.flush()
        time.sleep(delay)


def paced_write(port: serial.Serial, data: bytes, chunk_size: int, chunk_delay: float) -> None:
    for offset in range(0, len(data), chunk_size):
        port.write(data[offset : offset + chunk_size])
        port.flush()
        if chunk_delay > 0 and offset + chunk_size < len(data):
            time.sleep(chunk_delay)


def send_with_retries(
    port: serial.Serial,
    encoded: bytes,
    block: int,
    timeout: float,
    retries: int,
    chunk_size: int,
    chunk_delay: float,
) -> None:
    for attempt in range(1, retries + 1):
        paced_write(port, encoded, chunk_size, chunk_delay)
        deadline = time.monotonic() + timeout
        response = None
        while time.monotonic() < deadline:
            value = read_byte(port, deadline)
            if value is None:
                break
            if value == ACK:
                return
            if value == NAK:
                response = value
                break
            if value == CAN:
                second = read_byte(port, min(deadline, time.monotonic() + 0.5))
                if second == CAN:
                    raise RuntimeError(f"receiver cancelled at block {block}")
            # A YMODEM receiver may have queued repeated CRC requests before it
            # sees the packet. Ignore those and any command echo until ACK/NAK.
        if attempt < retries:
            sys.stderr.write(f"block {block}: {'timeout' if response is None else 'NAK'}; retrying\n")
    raise RuntimeError(f"block {block} failed after {retries} attempts")


def send_file(
    port: serial.Serial,
    path: Path,
    timeout: float,
    retries: int,
    chunk_size: int,
    chunk_delay: float,
    data_only: bool,
) -> None:
    size = path.stat().st_size
    name = path.name.encode("utf-8")
    size_text = str(size).encode("ascii")
    header_data = name + b"\0" + size_text + b"\0"
    if len(header_data) > 128:
        raise ValueError("file name and size do not fit in the YMODEM header")

    if not data_only:
        wait_for(port, {CRC_REQUEST}, timeout, "initial CRC request")
        # U-Boot is still transitioning into the receive loop for block zero.
        # Give that small control packet conservative FIFO pacing independently
        # of the bulk-data delay, which can safely be much shorter.
        header_delay = max(chunk_delay, 0.001)
        send_with_retries(
            port,
            packet(0, header_data.ljust(128, b"\0")),
            0,
            timeout,
            retries,
            min(chunk_size, 16),
            header_delay,
        )
        wait_for(port, {CRC_REQUEST}, timeout, "data CRC request")

    started = time.monotonic()
    sent = 0
    block = 1
    next_report = started
    with path.open("rb", buffering=1024 * 1024) as source:
        while True:
            payload = source.read(1024)
            if not payload:
                break
            actual = len(payload)
            if actual < 1024:
                payload = payload.ljust(1024, b"\x1a")
            send_with_retries(
                port,
                packet(block, payload),
                block,
                timeout,
                retries,
                chunk_size,
                chunk_delay,
            )
            sent += actual
            block = (block + 1) & 0xFF

            now = time.monotonic()
            if now >= next_report or sent == size:
                elapsed = max(now - started, 0.001)
                mib_s = sent / elapsed / (1024 * 1024)
                percent = sent * 100.0 / size
                remaining = (size - sent) / max(sent / elapsed, 1.0)
                print(
                    f"sent {sent}/{size} bytes ({percent:.1f}%), "
                    f"{mib_s:.3f} MiB/s, ETA {remaining:.0f}s",
                    flush=True,
                )
                next_report = now + 10.0

    # U-Boot's YMODEM receiver normally NAKs the first EOT and ACKs the second.
    port.write(bytes((EOT,)))
    port.flush()
    response = wait_for(port, {ACK, NAK}, timeout, "EOT response")
    if response == NAK:
        port.write(bytes((EOT,)))
        port.flush()
        wait_for(port, {ACK}, timeout, "final EOT ACK")

    wait_for(port, {CRC_REQUEST}, timeout, "closing header CRC request")
    send_with_retries(
        port,
        packet(0, bytes(128)),
        0,
        timeout,
        retries,
        min(chunk_size, 16),
        max(chunk_delay, 0.001),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--address", default="0xaf000000")
    parser.add_argument("--command-delay", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--chunk-delay", type=float, default=0.001)
    parser.add_argument("--already-receiving", action="store_true")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="resume after a header that the receiver already accepted",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.file.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"sending {path} ({path.stat().st_size} bytes) to {args.address}", flush=True)
    with serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=args.timeout) as port:
        if not args.already_receiving:
            port.reset_input_buffer()
            write_command(port, f"loady {args.address}", args.command_delay)
        send_file(
            port,
            path,
            args.timeout,
            args.retries,
            args.chunk_size,
            args.chunk_delay,
            args.data_only,
        )

        # Capture U-Boot's transfer summary without consuming the prompt forever.
        deadline = time.monotonic() + 5.0
        output = bytearray()
        while time.monotonic() < deadline:
            chunk = port.read(port.in_waiting or 1)
            if chunk:
                output.extend(chunk)
                deadline = time.monotonic() + 0.5
        if output:
            print(output.decode("utf-8", errors="replace"), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
