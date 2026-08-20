#!/usr/bin/env python3
"""Interrupt the FPGA console foreground process until its shell returns."""

import sys
import time

import serial


def main() -> None:
    with serial.Serial("/dev/ttyUSB0", 3_000_000, timeout=0.2,
                       write_timeout=10) as port:
        port.reset_input_buffer()
        received = bytearray()
        deadline = time.monotonic() + 120
        attempts = 0
        while (b"# " not in received and b"$ " not in received and
               time.monotonic() < deadline):
            if b"login: " in received:
                port.write(b"root\r")
                received.clear()
            else:
                port.write(b"\x03\r")
            port.flush()
            attempts += 1
            attempt_end = time.monotonic() + 2
            while time.monotonic() < attempt_end:
                chunk = port.read(port.in_waiting or 1)
                if chunk:
                    received.extend(chunk)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                if b"# " in received or b"$ " in received:
                    break
        print(f"\nINTERRUPT_ATTEMPTS {attempts}")
        if b"# " not in received and b"$ " not in received:
            raise SystemExit("FPGA shell did not return")


if __name__ == "__main__":
    main()
