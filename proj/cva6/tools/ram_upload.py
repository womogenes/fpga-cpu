#!/usr/bin/env python3
"""Upload an OpenSBI/Linux image through the CVA6 boot ROM and open its console."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
import time

import serial
from serial.tools import miniterm


class PacedMiniterm(miniterm.Miniterm):
    """Miniterm that leaves enough time for Linux to drain the UART FIFO."""

    def __init__(self, *args: object, character_delay: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.character_delay = character_delay

    def writer(self) -> None:
        menu_active = False
        try:
            while self.alive:
                try:
                    character = self.console.getkey()
                except KeyboardInterrupt:
                    character = "\x03"
                if not self.alive:
                    break
                if menu_active:
                    self.handle_menu_key(character)
                    menu_active = False
                elif character == self.menu_character:
                    menu_active = True
                elif character == self.exit_character:
                    self.stop()
                    break
                else:
                    transformed = character
                    for transformation in self.tx_transformations:
                        transformed = transformation.tx(transformed)
                    self.serial.write(self.tx_encoder.encode(transformed))
                    if self.character_delay > 0:
                        time.sleep(self.character_delay)
        except Exception:
            self.alive = False
            raise


def wait_for(port: serial.Serial, marker: bytes, timeout: float) -> None:
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
            if len(received) > 4096:
                del received[:-2048]


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


def upload_bootrom(port: serial.Serial, image: bytes, rate: int, chunk_size: int, boot_wait: float) -> None:
    # The FTDI bridge can transmit while the FPGA is still resetting, so an
    # eager key can disappear before the boot ROM enables its receiver. Wait
    # until the ROM has printed the prompt, then answer inside its countdown.
    wait_for(port, b"Hit any key to enter update mode ", boot_wait)
    port.write(b"U")
    port.flush()
    wait_for(port, b"size: ", 15.0)
    port.write(struct.pack("<I", len(image)))
    port.flush()
    wait_for(port, b"receiving ", 5.0)

    started = time.monotonic()
    sent = 0
    next_report = 0
    # CVA6 polls the APB UART one byte at a time. The boot ROM enables only the
    # 16550's 16-byte FIFO, so never leave more than one FIFO-sized write queued
    # in the host TTY. UART output and interactive input remain at the full baud
    # rate; only the large boot-ROM transfer is paced.
    while sent < len(image):
        chunk = image[sent : sent + chunk_size]
        port.write(chunk)
        port.flush()
        sent += len(chunk)
        delay = started + sent / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if sent >= next_report or sent == len(image):
            elapsed = max(time.monotonic() - started, 0.001)
            print(f"\rUploading: {sent * 100 / len(image):6.2f}%  {sent / elapsed / 1024:7.1f} KiB/s", end="", flush=True)
            next_report = sent + 1024 * 1024
    port.flush()
    print()
    wait_for(port, b" done!", 60.0)


def fnv1a(data: bytes, value: int = 2166136261) -> int:
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def upload_segment(port: serial.Serial, data: bytes, label: str, checksum: int) -> int:
    if not data:
        return checksum
    started = time.monotonic()
    sent = 0
    next_report = 0
    block_size = 4096
    while sent < len(data):
        block = data[sent : sent + block_size]
        port.write(block)
        port.flush()
        acknowledgement = read_exact(port, 1, 2.0)
        if acknowledgement != b"A":
            raise RuntimeError(f"stage-2 loader returned {acknowledgement!r}")
        checksum = fnv1a(block, checksum)
        sent += len(block)
        if sent >= next_report or sent == len(data):
            elapsed = max(time.monotonic() - started, 0.001)
            print(f"\r{label}: {sent * 100 / len(data):6.2f}%  {sent / elapsed / 1024:7.1f} KiB/s", end="", flush=True)
            next_report = sent + 1024 * 1024
    print()
    return checksum


def upload_stage2(port: serial.Serial, image: bytes, model: bytes) -> None:
    # The boot-ROM marker and S2READY can arrive in one host read. Start the
    # binary protocol with an explicit magic/ACK exchange instead of relying
    # on the informational ready line surviving that boundary.
    time.sleep(0.2)
    port.reset_input_buffer()
    port.write(b"S2UP")
    port.flush()
    wait_for(port, b"S2ACK\n", 5.0)
    port.write(struct.pack("<II", len(image), len(model)))
    port.flush()

    checksum = 2166136261
    checksum = upload_segment(port, image, "Linux upload", checksum)
    checksum = upload_segment(port, model, "Model upload", checksum)

    wait_for(port, b"\nS2DONE\n", 120.0)
    port.write(b"C")
    port.flush()
    remote_checksum = struct.unpack("<I", read_exact(port, 4, 2.0))[0]
    if remote_checksum != checksum:
        raise RuntimeError(f"stage-2 checksum mismatch: host {checksum:08x}, target {remote_checksum:08x}")
    print(f"Stage-2 checksum passed ({checksum:08x}); launching Linux.")
    port.write(b"G")
    port.flush()


def resume_stage2(port: serial.Serial, image: bytes, model: bytes) -> None:
    checksum = fnv1a(model, fnv1a(image))
    # S2DONE may have been discarded while no process had the TTY open, but
    # the loader remains blocked waiting for this explicit checksum request.
    port.reset_input_buffer()
    port.write(b"C")
    port.flush()
    remote_checksum = struct.unpack("<I", read_exact(port, 4, 120.0))[0]
    if remote_checksum != checksum:
        raise RuntimeError(f"stage-2 checksum mismatch: host {checksum:08x}, target {remote_checksum:08x}")
    print(f"Resumed stage-2 checksum passed ({checksum:08x}); launching Linux.")
    port.write(b"G")
    port.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--rate-kib", type=int, default=None, help="paced boot-ROM upload rate in KiB/s")
    parser.add_argument("--chunk-size", type=int, default=None, help="drained host write size in bytes")
    parser.add_argument("--boot-wait", type=float, default=15.0, help="seconds to wait for a freshly programmed FPGA boot-ROM prompt")
    parser.add_argument("--stage2", type=Path, default=None, help="flow-controlled stage-2 loader binary")
    parser.add_argument("--model", type=Path, default=None, help="optional model copied to reserved DDR at 0xaf000000")
    parser.add_argument("--resume-stage2", action="store_true", help="resume at the post-upload checksum gate")
    parser.add_argument("--no-console", action="store_true")
    parser.add_argument("--console-char-delay-us", type=float, default=None, help="delay pasted console input between bytes")
    args = parser.parse_args()

    import os

    device = args.serial or os.environ.get("SERIAL", "/dev/ttyUSB0")
    baud = args.baud or int(os.environ.get("UART_BAUD", "3000000"))
    rate_kib = args.rate_kib or int(os.environ.get("UPLOAD_RATE_KIB", "120"))
    chunk_size = args.chunk_size or int(os.environ.get("UPLOAD_CHUNK_SIZE", "32"))
    console_char_delay_us = args.console_char_delay_us
    if console_char_delay_us is None:
        console_char_delay_us = float(os.environ.get("CONSOLE_CHAR_DELAY_US", "100"))
    if rate_kib <= 0:
        raise SystemExit("upload rate must be positive")
    if chunk_size <= 0:
        raise SystemExit("upload chunk size must be positive")
    if args.boot_wait <= 0:
        raise SystemExit("boot wait must be positive")
    if console_char_delay_us < 0:
        raise SystemExit("console character delay must not be negative")
    image = args.image.read_bytes()
    model = args.model.read_bytes() if args.model is not None else b""
    if len(image) >= 2**32:
        raise SystemExit("image is too large for the boot ROM protocol")
    if len(model) > 240 * 1024 * 1024:
        raise SystemExit("model is too large for the 240 MiB reserved DDR model region")

    with serial.Serial(device, baudrate=baud, timeout=0.1, write_timeout=10) as port:
        if args.resume_stage2:
            resume_stage2(port, image, model)
        elif args.stage2 is None:
            print(f"Uploading {len(image) / 1024 / 1024:.1f} MiB to {device} at {baud} baud, paced to {rate_kib} KiB/s")
            upload_bootrom(port, image, rate_kib * 1024, chunk_size, args.boot_wait)
        else:
            loader = args.stage2.read_bytes()
            print(f"Uploading {len(loader)}-byte stage-2 loader through the boot ROM...")
            upload_bootrom(port, loader, rate_kib * 1024, chunk_size, args.boot_wait)
            print(f"Uploading {len(image) / 1024 / 1024:.1f} MiB Linux and {len(model) / 1024 / 1024:.1f} MiB model with block acknowledgements...")
            upload_stage2(port, image, model)
        if args.no_console:
            return
        print("Entering console; press Ctrl-] to exit.")
        console = PacedMiniterm(port, echo=False, character_delay=console_char_delay_us / 1_000_000)
        console.exit_character = chr(29)
        console.raw = True
        console.set_rx_encoding("UTF-8")
        console.set_tx_encoding("UTF-8")
        console.start()
        console.join()
        console.close()


if __name__ == "__main__":
    main()
