#!/usr/bin/env python3
"""Generate the fixed Bonsai Q1_0 tensor table used by the FPGA wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from gguf import GGUFReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reader = GGUFReader(args.model, "r")
    tensors = [
        tensor
        for tensor in reader.tensors
        if int(tensor.tensor_type) == 41 and len(tensor.shape) == 2
    ]
    lines = [
        "/* Generated from Bonsai-1.7B.gguf; do not edit. */",
        "#ifndef BONSAI_MODEL_MANIFEST_H",
        "#define BONSAI_MODEL_MANIFEST_H",
        "",
        f"#define BONSAI_MODEL_FILE_SIZE UINT64_C({args.model.stat().st_size})",
        "",
        "struct bonsai_tensor_spec {",
        "  uint64_t offset;",
        "  uint32_t ne0;",
        "  uint32_t ne1;",
        "};",
        "",
        "static const struct bonsai_tensor_spec bonsai_tensors[] = {",
    ]
    for tensor in tensors:
        ne0, ne1 = (int(value) for value in tensor.shape)
        lines.append(f"  {{UINT64_C({tensor.data_offset}), {ne0}u, {ne1}u}},")
    lines.extend(
        [
            "};",
            "",
            "#define BONSAI_TENSOR_COUNT \\",
            "  (sizeof(bonsai_tensors) / sizeof(bonsai_tensors[0]))",
            "",
            "#endif",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {len(tensors)} Q1_0 matrix tensors to {args.output}")


if __name__ == "__main__":
    main()
