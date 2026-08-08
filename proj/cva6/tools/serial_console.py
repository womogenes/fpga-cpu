#!/usr/bin/env python3
"""Scripted serial console: send commands, capture output, no interactivity.

Each --cmd is written to the UART with per-character pacing, then output is
collected until the shell prompt reappears or the idle timeout expires. All
received bytes are echoed to stdout so the transcript is preserved.
"""

import argparse
import sys
import time

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


def send_paced(port: serial.Serial, text: str, char_delay: float) -> None:
    for byte in text.encode():
        port.write(bytes([byte]))
        port.flush()
        time.sleep(char_delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--char-delay-us", type=float, default=200.0)
    parser.add_argument("--idle-timeout", type=float, default=2.0,
                        help="seconds of silence that end one command's capture")
    parser.add_argument("--cmd", action="append", default=[],
                        help="command to run; may repeat")
    parser.add_argument("--interrupt", action="store_true",
                        help="send Ctrl-C first to clear a stuck foreground "
                             "process")
    parser.add_argument("--script", type=argparse.FileType("r"),
                        help="file with one command per line")
    args = parser.parse_args()

    commands = list(args.cmd)
    if args.script:
        for line in args.script:
            line = line.rstrip("\n")
            if line:
                commands.append(line)

    with serial.Serial(args.serial, args.baud, timeout=0.1) as port:
        if args.interrupt:
            port.write(b"\x03")
            port.flush()
            time.sleep(0.5)
            port.write(b"\x03")
            port.flush()
            time.sleep(0.5)
        send_paced(port, "\r", args.char_delay_us / 1_000_000)
        sys.stdout.write(drain(port, 1.0).decode(errors="replace"))
        for command in commands:
            send_paced(port, command + "\r", args.char_delay_us / 1_000_000)
            output = drain(port, args.idle_timeout)
            sys.stdout.write(output.decode(errors="replace"))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
