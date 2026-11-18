#!/usr/bin/env python3
"""Validate a standard Life RLE file and pack it for the firmware."""

import argparse
import re
from pathlib import Path


HEADER = re.compile(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    content = [line for line in lines if not line.lstrip().startswith("#")]
    if not content:
        raise ValueError("RLE file has no header")

    match = HEADER.search(content[0])
    if match is None:
        raise ValueError("expected an 'x = ..., y = ...' RLE header")

    width, height = (int(value) for value in match.groups())
    if width > 0xffff or height > 0xffff:
        raise ValueError("pattern dimensions exceed the packed format")

    body = "".join("".join(content[1:]).split())
    if not body.endswith("!"):
        raise ValueError("RLE body must end with '!'")
    if re.fullmatch(r"[0-9bo$!]+", body) is None:
        raise ValueError("RLE body contains unsupported tokens")

    packed = bytes((width & 0xff, width >> 8, height & 0xff, height >> 8))
    args.output.write_bytes(packed + body.encode("ascii") + b"\0")


if __name__ == "__main__":
    main()

