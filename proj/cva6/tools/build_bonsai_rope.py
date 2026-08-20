#!/usr/bin/env python3
"""Generate deterministic Q16 YaRN tables for the Bonsai decoder."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

HEAD_DIM = 128
CONTEXT = 256


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def q16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    magnitude = bits & 0x7FFF_FFFF
    exponent = magnitude >> 23
    if exponent == 0:
        return 0
    significand = (magnitude & 0x007F_FFFF) | 0x0080_0000
    shift = exponent - 127 - 7
    fixed = (min(2_147_418_112, significand << shift) if shift >= 0
             else (0 if -shift >= 32 else significand >> -shift))
    return -fixed if bits >> 31 else fixed


def correlation_dimension(rotations: float) -> float:
    ratio = 8192.0 / (rotations * 2.0 * math.pi)
    return HEAD_DIM * math.log(ratio) / (2.0 * math.log(1_000_000.0))


def generate() -> tuple[list[list[int]], list[list[int]]]:
    low = f32(math.floor(correlation_dimension(32.0)))
    high = f32(math.ceil(correlation_dimension(1.0)))
    frequency_scale = f32(0.25)
    theta_scale = f32(1_000_000.0 ** (-2.0 / HEAD_DIM))
    magnitude_scale = f32(1.0 + 0.1 * math.log(4.0))
    span_reciprocal = f32(1.0 / f32(high - low))
    step_cos: list[float] = []
    step_sin: list[float] = []
    theta = f32(1.0)
    for index in range(HEAD_DIM // 2):
        y = f32(f32(f32(index) - low) * span_reciprocal)
        ramp = f32(1.0 - max(0.0, min(1.0, y)))
        mix = f32(f32(frequency_scale * f32(1.0 - ramp)) + ramp)
        angle = f32(theta * mix)
        step_cos.append(f32(math.cos(angle)))
        step_sin.append(f32(math.sin(angle)))
        theta = f32(theta * theta_scale)

    current_cos = [f32(1.0)] * (HEAD_DIM // 2)
    current_sin = [f32(0.0)] * (HEAD_DIM // 2)
    cosine: list[list[int]] = []
    sine: list[list[int]] = []
    for _ in range(CONTEXT):
        cosine.append([q16(f32(value * magnitude_scale))
                       for value in current_cos])
        sine.append([q16(f32(value * magnitude_scale))
                     for value in current_sin])
        for index in range(HEAD_DIM // 2):
            c = current_cos[index]
            s = current_sin[index]
            current_cos[index] = f32(
                f32(c * step_cos[index]) - f32(s * step_sin[index]))
            current_sin[index] = f32(
                f32(s * step_cos[index]) + f32(c * step_sin[index]))
    return cosine, sine


def emit_table(name: str, table: list[list[int]]) -> list[str]:
    lines = [f"static const int32_t {name}[256][64] = {{"]
    for row in table:
        lines.append("  {" + ", ".join(str(value) for value in row) + "},")
    lines.append("};")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cosine, sine = generate()
    lines = [
        "/* Generated deterministic Bonsai YaRN tables; do not edit. */",
        "#ifndef BONSAI_ROPE_MANIFEST_H",
        "#define BONSAI_ROPE_MANIFEST_H",
        "", "#include <stdint.h>", "",
        *emit_table("bonsai_rope_cos_fixed", cosine), "",
        *emit_table("bonsai_rope_sin_fixed", sine), "",
        "#endif", "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote deterministic {CONTEXT}x{HEAD_DIM // 2} RoPE tables "
          f"to {args.output}")


if __name__ == "__main__":
    main()
