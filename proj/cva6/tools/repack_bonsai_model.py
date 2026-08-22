#!/usr/bin/env python3
"""Reorder Bonsai Q1_0 blocks into full-width accelerator stream groups."""

from __future__ import annotations

import argparse
import mmap
from pathlib import Path
import shutil

import numpy as np
from gguf import GGUFReader


Q1_0_TYPE = 41


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.source.resolve() == args.destination.resolve():
        raise SystemExit("source and destination must differ")
    reader = GGUFReader(args.source, "r")
    tensors = [
        tensor
        for tensor in reader.tensors
        if int(tensor.tensor_type) == Q1_0_TYPE and len(tensor.shape) == 2
    ]
    for tensor in tensors:
        ne0, _ = (int(value) for value in tensor.shape)
        if ne0 % 512 != 0:
            raise SystemExit(
                f"{tensor.name}: Q1_0 width {ne0} is not divisible by 512"
            )

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.destination)
    with args.destination.open("r+b") as output:
        mapping = mmap.mmap(output.fileno(), 0)
        try:
            for tensor in tensors:
                ne0, ne1 = (int(value) for value in tensor.shape)
                groups = ne0 // 512
                source = np.ndarray(
                    (ne1, groups, 4, 18),
                    dtype=np.uint8,
                    buffer=mapping,
                    offset=int(tensor.data_offset),
                ).copy()
                destination = np.ndarray(
                    (ne1, groups, 72),
                    dtype=np.uint8,
                    buffer=mapping,
                    offset=int(tensor.data_offset),
                )
                destination[:, :, :8] = source[:, :, :, :2].reshape(
                    ne1, groups, 8
                )
                destination[:, :, 8:] = source[:, :, :, 2:].reshape(
                    ne1, groups, 64
                )
            # BGUF marks accelerator-packed tensor data. The target maps the
            # untouched GGUF on SD for metadata and CPU-readable embeddings.
            mapping[0:4] = b"BGUF"
            mapping.flush()
        finally:
            mapping.close()
    print(
        f"repacked {len(tensors)} Q1_0 tensors into {args.destination} "
        f"({args.destination.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
