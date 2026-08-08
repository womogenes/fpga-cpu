#!/usr/bin/env python3
"""Open UART before programming so the boot-ROM update prompt cannot be missed."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--bitstream", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--jtag-frequency", type=int, default=1_000_000)
    parser.add_argument("--boot-wait", type=float, default=180.0)
    args = parser.parse_args()

    uploader_command = [
        sys.executable,
        str(Path(__file__).with_name("ram_upload.py")),
        "--serial",
        args.serial,
        "--baud",
        str(args.baud),
        "--boot-wait",
        str(args.boot_wait),
        "--stage2",
        str(args.stage2),
    ]
    if args.model is not None:
        uploader_command.extend(("--model", str(args.model)))
    uploader_command.append(str(args.image))

    uploader = subprocess.Popen(uploader_command)
    try:
        # Give pyserial time to claim the port before configuration releases
        # the CPU. It then waits harmlessly while JTAG shifts the bitstream.
        time.sleep(0.5)
        status = uploader.poll()
        if status is not None:
            raise SystemExit(f"UART uploader exited before programming (status {status})")
        programmer = subprocess.run(
            [
                "openFPGALoader",
                "-b",
                "genesys2",
                "--freq",
                str(args.jtag_frequency),
                str(args.bitstream),
            ],
            check=False,
        )
        if programmer.returncode != 0:
            uploader.terminate()
            uploader.wait()
            raise SystemExit(f"openFPGALoader failed with status {programmer.returncode}")
        raise SystemExit(uploader.wait())
    except BaseException:
        if uploader.poll() is None:
            uploader.terminate()
            try:
                uploader.wait(timeout=5)
            except subprocess.TimeoutExpired:
                uploader.kill()
        raise


if __name__ == "__main__":
    main()
