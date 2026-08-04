#!/usr/bin/env python3

import argparse
import pickle
import struct
from pathlib import Path


UINT8 = 0
INT8 = 1
UINT16 = 2
INT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11
FLOAT64 = 12

SCALAR_FORMATS = {
    UINT8: "<B",
    INT8: "<b",
    UINT16: "<H",
    INT16: "<h",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    BOOL: "<?",
    UINT64: "<Q",
    INT64: "<q",
    FLOAT64: "<d",
}


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def read_string(stream):
    size = struct.unpack("<Q", read_exact(stream, 8))[0]
    return read_exact(stream, size).decode("utf-8")


def read_scalar(stream, value_type):
    if value_type == STRING:
        return read_string(stream)
    try:
        fmt = SCALAR_FORMATS[value_type]
    except KeyError as error:
        raise ValueError(f"unsupported GGUF metadata type {value_type}") from error
    return struct.unpack(fmt, read_exact(stream, struct.calcsize(fmt)))[0]


def read_value(stream, value_type):
    if value_type != ARRAY:
        return read_scalar(stream, value_type)
    element_type = struct.unpack("<I", read_exact(stream, 4))[0]
    count = struct.unpack("<Q", read_exact(stream, 8))[0]
    return [read_scalar(stream, element_type) for _ in range(count)]


def tokenizer_metadata(model):
    wanted = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}
    found = {}
    with model.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise ValueError("model is not GGUF")
        version = struct.unpack("<I", read_exact(stream, 4))[0]
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        read_exact(stream, 8)
        field_count = struct.unpack("<Q", read_exact(stream, 8))[0]
        for _ in range(field_count):
            key = read_string(stream)
            value_type = struct.unpack("<I", read_exact(stream, 4))[0]
            value = read_value(stream, value_type)
            if key in wanted:
                found[key] = value
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"missing tokenizer metadata: {', '.join(sorted(missing))}")
    return found


def main():
    parser = argparse.ArgumentParser(description="Extract the standalone LFM tokenizer from GGUF")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = tokenizer_metadata(args.model)
    payload = {
        "tokens": metadata["tokenizer.ggml.tokens"],
        "merges": metadata["tokenizer.ggml.merges"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)
    print(f"wrote {len(payload['tokens'])} tokens and {len(payload['merges'])} merges to {args.output}")


if __name__ == "__main__":
    main()
