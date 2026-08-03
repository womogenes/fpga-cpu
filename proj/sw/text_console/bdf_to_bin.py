#!/usr/bin/env python3

"""
Pack printable ASCII glyphs from a fixed-cell BDF into a raw bitmap.
"""

import argparse
from pathlib import Path


FIRST_CHARACTER = 32
LAST_CHARACTER = 126
def parse_bdf(path: Path, width: int, height: int) -> dict[int, bytes]:
    bytes_per_row = (width + 7) // 8
    glyphs: dict[int, bytes] = {}
    encoding: int | None = None
    bitmap_rows: list[str] | None = None

    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if line.startswith("ENCODING "):
            encoding = int(line.split()[1])
        elif line == "BITMAP":
            bitmap_rows = []
        elif line == "ENDCHAR":
            if encoding is not None and FIRST_CHARACTER <= encoding <= LAST_CHARACTER:
                if bitmap_rows is None or len(bitmap_rows) != height:
                    raise ValueError(
                        f"glyph {encoding} has {len(bitmap_rows or [])} rows; "
                        f"expected {height}"
                    )
                try:
                    bitmap = b"".join(bytes.fromhex(row) for row in bitmap_rows)
                except ValueError as error:
                    raise ValueError(
                        f"invalid bitmap data for glyph {encoding}"
                    ) from error
                expected_size = height * bytes_per_row
                if len(bitmap) != expected_size:
                    raise ValueError(
                        f"glyph {encoding} is {len(bitmap)} bytes; "
                        f"expected {expected_size}"
                    )
                glyphs[encoding] = bitmap
            encoding = None
            bitmap_rows = None
        elif bitmap_rows is not None:
            bitmap_rows.append(line)

    return glyphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    glyphs = parse_bdf(args.input, args.width, args.height)
    missing = [
        character
        for character in range(FIRST_CHARACTER, LAST_CHARACTER + 1)
        if character not in glyphs
    ]
    if missing:
        raise ValueError(f"missing printable ASCII encodings: {missing}")

    args.output.write_bytes(
        b"".join(
            glyphs[character]
            for character in range(FIRST_CHARACTER, LAST_CHARACTER + 1)
        )
    )


if __name__ == "__main__":
    main()
