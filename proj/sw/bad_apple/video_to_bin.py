#!/usr/bin/env python3
"""Convert a video to the delta-compressed monochrome MVF1 format."""

import argparse
import io
from pathlib import Path
import shutil
import struct
import subprocess
import zlib


MAGIC = b"MVF1"
HEADER = struct.Struct("<4sHHHHII")
COMPRESSED_HEADER = struct.Struct("<4sI")


def encode_packbits(data: bytes) -> bytes:
    output = bytearray()
    offset = 0

    while offset < len(data):
        run_length = 1
        while (
            offset + run_length < len(data)
            and run_length < 128
            and data[offset + run_length] == data[offset]
        ):
            run_length += 1

        if run_length >= 3:
            output.extend((0x80 | (run_length - 1), data[offset]))
            offset += run_length
            continue

        literal_start = offset
        offset += 1
        while offset - literal_start < 128 and offset < len(data):
            run_length = 1
            while (
                offset + run_length < len(data)
                and run_length < 128
                and data[offset + run_length] == data[offset]
            ):
                run_length += 1
            if run_length >= 3:
                break
            offset += 1

        literal = data[literal_start:offset]
        output.append(len(literal) - 1)
        output.extend(literal)

    return bytes(output)


def pack_frame(frame: bytes, width: int, height: int, threshold: int) -> bytes:
    row_bytes = width // 8
    packed = bytearray(row_bytes * height)

    for y in range(height):
        input_row = y * width
        output_row = y * row_bytes
        for byte_x in range(row_bytes):
            value = 0
            for bit in range(8):
                if frame[input_row + byte_x * 8 + bit] >= threshold:
                    value |= 1 << bit
            packed[output_row + byte_x] = value

    return bytes(packed)


def read_exact(stream, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.read(length - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert video to delta-compressed monochrome frames."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=128)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to convert the video")
    if args.width <= 0 or args.width % 8 != 0 or args.width > 0xffff:
        raise SystemExit("width must be a positive multiple of 8 no larger than 65535")
    if args.height <= 0 or args.height > 0xffff:
        raise SystemExit("height must be between 1 and 65535")
    if args.fps <= 0 or args.fps > 0xffff:
        raise SystemExit("fps must be between 1 and 65535")
    if not 0 <= args.threshold <= 255:
        raise SystemExit("threshold must be between 0 and 255")
    if not args.input.is_file():
        raise SystemExit(f"video does not exist: {args.input}")

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(args.input),
        "-an",
        "-vf",
        f"scale={args.width}:{args.height}:flags=lanczos,fps={args.fps}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None

    raw_frame_size = args.width * args.height
    packed_frame_size = raw_frame_size // 8
    previous = bytes(packed_frame_size)
    frame_count = 0

    output = io.BytesIO()
    output.write(HEADER.pack(MAGIC, args.width, args.height, args.fps, 0, 0, packed_frame_size))

    while True:
        raw_frame = read_exact(process.stdout, raw_frame_size)
        if not raw_frame:
            break
        if len(raw_frame) != raw_frame_size:
            process.kill()
            raise SystemExit("ffmpeg produced a partial final frame")

        packed = pack_frame(raw_frame, args.width, args.height, args.threshold)
        delta = bytes(current ^ old for current, old in zip(packed, previous))
        encoded = encode_packbits(delta)
        output.write(struct.pack("<I", len(encoded)))
        output.write(encoded)
        previous = packed
        frame_count += 1

    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    if frame_count == 0:
        raise SystemExit("video contains no frames")

    output.seek(0)
    output.write(HEADER.pack(MAGIC, args.width, args.height, args.fps, 0, frame_count, packed_frame_size))
    uncompressed = output.getvalue()
    compressed = zlib.compress(uncompressed, level=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(COMPRESSED_HEADER.pack(b"MVZ1", len(uncompressed)) + compressed)

    print(
        f"wrote {frame_count} frames at {args.width}x{args.height}, "
        f"{args.fps} fps: {args.output} "
        f"({len(uncompressed)} bytes, {len(compressed)} after DEFLATE)"
    )


if __name__ == "__main__":
    main()
