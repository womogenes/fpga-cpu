#!/usr/bin/env python3
"""Run one paced shell command on the FPGA UART and report its exit status."""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--char-delay-ms", type=float, default=10.0)
    args = parser.parse_args()

    token = uuid.uuid4().hex
    marker = f"__SERIAL_RC_{token}=".encode()
    result_marker = b"\r\n" + marker
    payload = (
        args.command
        + f"; rc=$?; printf '\\n{marker.decode()}%d\\n' \"$rc\"\n"
    ).encode()

    with serial.Serial(args.serial, args.baud, timeout=0.2, write_timeout=10) as port:
        port.reset_input_buffer()
        for byte in payload:
            port.write(bytes([byte]))
            port.flush()
            time.sleep(args.char_delay_ms / 1000.0)

        output = bytearray()
        deadline = time.monotonic() + args.timeout
        while result_marker not in output and time.monotonic() < deadline:
            chunk = port.read(port.in_waiting or 1)
            if chunk:
                output.extend(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

    if result_marker not in output:
        raise SystemExit("serial shell command timed out")
    suffix = output.split(result_marker, 1)[1]
    status_text = suffix.splitlines()[0].decode("ascii", "strict")
    status = int(status_text)
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
