#!/usr/bin/env python3
"""Print Linux FIEMAP extents in bytes and 512-byte sectors."""

from __future__ import annotations

import argparse
import array
import fcntl
import os
import struct


FS_IOC_FIEMAP = 0xC020660B
FIEMAP_HEADER = struct.Struct("=QQIIII")
FIEMAP_EXTENT = struct.Struct("=QQQQQIIII")
FIEMAP_EXTENT_LAST = 0x00000001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--partition-lba", type=lambda value: int(value, 0), default=0)
    args = parser.parse_args()

    extent_count = 256
    buffer = array.array("B", bytes(FIEMAP_HEADER.size + extent_count * FIEMAP_EXTENT.size))
    FIEMAP_HEADER.pack_into(buffer, 0, 0, (1 << 64) - 1, 0, 0, extent_count, 0)
    with open(args.path, "rb", buffering=0) as model:
        fcntl.ioctl(model.fileno(), FS_IOC_FIEMAP, buffer, True)

    _, _, _, mapped, _, _ = FIEMAP_HEADER.unpack_from(buffer)
    size = os.stat(args.path).st_size
    print(f"FILE_SIZE {size}")
    print(f"MAPPED_EXTENTS {mapped}")
    covered = 0
    for index in range(mapped):
        offset = FIEMAP_HEADER.size + index * FIEMAP_EXTENT.size
        logical, physical, length, _, _, flags, _, _, _ = FIEMAP_EXTENT.unpack_from(buffer, offset)
        sectors = (length + 511) // 512
        lba = args.partition_lba + physical // 512
        print(
            f"EXTENT {index} logical={logical} physical={physical} length={length} "
            f"lba=0x{lba:x} sectors=0x{sectors:x} flags=0x{flags:x}"
        )
        covered += min(length, max(size - logical, 0))
        if flags & FIEMAP_EXTENT_LAST:
            break
    print(f"COVERED_BYTES {covered}")


if __name__ == "__main__":
    main()
