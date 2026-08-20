#!/usr/bin/env python3
"""Preload an SD-resident model into reserved DDR before booting Linux."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

import serial


def send(port: serial.Serial, data: bytes) -> None:
    for byte in data:
        port.write(bytes((byte,)))
        port.flush()
        time.sleep(0.01)


def receive_until(port: serial.Serial, markers: tuple[bytes, ...],
                  timeout: float) -> tuple[bytes, bytes]:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if not chunk:
            continue
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        received.extend(chunk)
        for marker in markers:
            if marker in received:
                return marker, bytes(received)
    raise TimeoutError(f"timed out; tail={bytes(received[-1500:])!r}")


def command(port: serial.Serial, text: str, timeout: float = 30.0) -> bytes:
    send(port, text.encode() + b"\r")
    _, output = receive_until(port, (b"=> ",), timeout)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--bitstream")
    parser.add_argument(
        "--reboot-linux", action="store_true",
        help="request a software reboot from an existing Linux shell",
    )
    parser.add_argument(
        "--stop-at-uboot", action="store_true",
        help="program/reset the FPGA, interrupt autoboot, and leave U-Boot at its prompt",
    )
    parser.add_argument(
        "--at-uboot-prompt", action="store_true",
        help="the board is already stopped at U-Boot; do not send a wake-up return",
    )
    parser.add_argument("--model-partition", default="0:4")
    parser.add_argument("--model-path",
                        default="/Bonsai-1.7B-packed.gguf")
    parser.add_argument("--address", type=lambda value: int(value, 0),
                        default=0xaf000000)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument(
        "--already-preloaded", action="store_true",
        help="validate reserved DDR and boot without reading the model again",
    )
    parser.add_argument(
        "--raw-extent", action="append", default=[], metavar="DEST:BLOCK:COUNT",
        help="load a known contiguous file extent with U-Boot mmc read",
    )
    parser.add_argument("--load-timeout", type=float, default=600.0)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    args = parser.parse_args()

    with serial.Serial(args.serial, args.baud, timeout=0.1,
                       write_timeout=10) as port:
        port.reset_input_buffer()
        if args.reboot_linux:
            send(port, b"reboot\r")
        if args.bitstream:
            subprocess.run([
                "openFPGALoader", "-b", "genesys2", "--freq", "1000000",
                args.bitstream,
            ], check=True)
            # Genesys 2 briefly disconnects the USB UART when configuration
            # replaces the design.  Reopen the device so pyserial does not
            # retain a stale file descriptor while the boot ROM starts.
            port.close()
            deadline = time.monotonic() + 15.0
            while True:
                try:
                    port.open()
                    break
                except serial.SerialException:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
        elif not args.at_uboot_prompt and not args.reboot_linux:
            # An existing U-Boot prompt is silent until it receives input.
            send(port, b"\r")

        if not args.at_uboot_prompt:
            marker, _ = receive_until(
                port, (b"Hit any key to stop autoboot", b"=> "), 180.0)
            if marker != b"=> ":
                port.write(b" ")
                port.flush()
                receive_until(port, (b"=> ",), 15.0)

        if args.stop_at_uboot:
            print("UBOOT_READY", flush=True)
            return

        if args.already_preloaded:
            pass
        elif args.raw_extent:
            loaded_blocks = 0
            for specification in args.raw_extent:
                try:
                    destination, block, count = (
                        int(value, 0) for value in specification.split(":"))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid raw extent {specification!r}") from error
                expected_destination = args.address + loaded_blocks * 512
                if destination != expected_destination:
                    raise RuntimeError(
                        f"raw extent destination 0x{destination:x} is not "
                        f"contiguous; expected 0x{expected_destination:x}")
                output = command(
                    port,
                    f"mmc read 0x{destination:x} 0x{block:x} 0x{count:x}",
                    args.load_timeout,
                )
                matches = re.findall(rb"(\d+) blocks read: OK", output)
                if not matches or int(matches[-1]) != count:
                    raise RuntimeError(
                        f"raw extent block-count mismatch: expected {count}")
                loaded_blocks += count
            expected_blocks = (args.expected_size + 511) // 512
            padding_bytes = loaded_blocks * 512 - args.expected_size
            if loaded_blocks < expected_blocks or padding_bytes >= 4096:
                raise RuntimeError(
                    f"raw extents cover {loaded_blocks} blocks; "
                    f"expected {expected_blocks} with less than one final "
                    "filesystem block of padding")
        else:
            output = command(
                port,
                f"ext4load mmc {args.model_partition} 0x{args.address:x} "
                f"{args.model_path}",
                args.load_timeout,
            )
            match = re.search(rb"(\d+) bytes read", output)
            if match is None or int(match.group(1)) != args.expected_size:
                raise RuntimeError(
                    f"model load size mismatch: expected {args.expected_size}")

        output = command(port, f"md.b 0x{args.address:x} 4")
        if b"42 47 55 46" not in output.lower():
            raise RuntimeError("reserved DDR does not begin with BGUF")

        send(port, b"run bootcmd\r")
        receive_until(port, (b"buildroot login:",), args.boot_timeout)
        send(port, b"root\r")
        prompt, _ = receive_until(port, (b"# ", b"$ "), 60.0)
        if prompt == b"$ ":
            send(port, b'PS1="# "\r')
            receive_until(port, (b"# ",), 15.0)
        print("MODEL_PRELOAD_BOOT_OK", flush=True)


if __name__ == "__main__":
    main()
