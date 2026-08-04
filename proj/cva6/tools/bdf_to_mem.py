#!/usr/bin/env python3

import argparse
from pathlib import Path


FIRST = 32
LAST = 126
WIDTH = 8
HEIGHT = 16


def load_glyphs(path: Path) -> dict[int, list[str]]:
    glyphs: dict[int, list[str]] = {}
    encoding: int | None = None
    bitmap: list[str] | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("ENCODING "):
            encoding = int(line.split()[1])
        elif line == "BITMAP":
            bitmap = []
        elif line == "ENDCHAR":
            if encoding is not None and FIRST <= encoding <= LAST:
                if bitmap is None or len(bitmap) != HEIGHT:
                    raise ValueError(f"glyph {encoding} is not {WIDTH}x{HEIGHT}")
                glyphs[encoding] = bitmap
            encoding = None
            bitmap = None
        elif bitmap is not None:
            bitmap.append(line)
    return glyphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    glyphs = load_glyphs(args.input)
    missing = [code for code in range(FIRST, LAST + 1) if code not in glyphs]
    if missing:
        raise ValueError(f"missing glyphs: {missing}")
    lines = [row[-2:].upper() for code in range(FIRST, LAST + 1) for row in glyphs[code]]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
