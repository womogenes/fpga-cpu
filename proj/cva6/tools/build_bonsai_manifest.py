#!/usr/bin/env python3
"""Generate the fixed Bonsai Q1_0 tensor table used by the FPGA wrapper."""

from __future__ import annotations

import argparse
import json
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
    all_tensors = list(reader.tensors)
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
            "struct bonsai_model_entry {",
            "  const char *name;",
            "  uint64_t offset;",
            "  uint32_t ne0;",
            "  uint32_t ne1;",
            "  uint32_t type;",
            "};",
            "",
            "static const struct bonsai_model_entry bonsai_model_entries[] = {",
        ]
    )
    for tensor in all_tensors:
        shape = [int(value) for value in tensor.shape]
        ne0 = shape[0]
        ne1 = shape[1] if len(shape) > 1 else 1
        lines.append(
            f"  {{{json.dumps(tensor.name)}, UINT64_C({tensor.data_offset}), "
            f"{ne0}u, {ne1}u, {int(tensor.tensor_type)}u}},"
        )
    lines.extend(
        [
            "};",
            "",
            "#define BONSAI_MODEL_ENTRY_COUNT \\",
            "  (sizeof(bonsai_model_entries) / sizeof(bonsai_model_entries[0]))",
            "",
            "#endif",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(
        f"wrote {len(tensors)} Q1_0 matrices and {len(all_tensors)} model "
        f"entries to {args.output}"
    )


if __name__ == "__main__":
    main()
