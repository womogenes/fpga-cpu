#!/usr/bin/env python3
"""Open the CVA6 serial console with safe pacing for pasted input."""

from __future__ import annotations

import argparse
import os

import serial

from ram_upload import PacedMiniterm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=os.environ.get("SERIAL", "/dev/ttyUSB0"))
    parser.add_argument("--baud", type=int, default=int(os.environ.get("UART_BAUD", "3000000")))
    parser.add_argument("--char-delay-us", type=float, default=float(os.environ.get("CONSOLE_CHAR_DELAY_US", "100")))
    args = parser.parse_args()
    if args.char_delay_us < 0:
        raise SystemExit("console character delay must not be negative")

    with serial.Serial(args.serial, baudrate=args.baud, timeout=0.1, write_timeout=10) as port:
        console = PacedMiniterm(port, echo=False, character_delay=args.char_delay_us / 1_000_000)
        console.exit_character = chr(29)
        console.raw = True
        console.set_rx_encoding("UTF-8")
        console.set_tx_encoding("UTF-8")
        console.start()
        console.join()
        console.close()


if __name__ == "__main__":
    main()
