#!/usr/bin/env python3
"""Interactive host-side Bonsai chat bridge for the FPGA serial console."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bonsai_chat import (  # noqa: E402
    EOS_TOKENS,
    MAX_CONTEXT,
    THINK_TOKENS,
    byte_decoder,
    decode_piece,
    load_tokenizer,
    render_prompt,
    tokenize,
)


def send(port: serial.Serial, data: bytes, rate: int = 3_000) -> None:
    delay = 1.0 / rate
    for offset in range(0, len(data), 32):
        chunk = data[offset : offset + 32]
        port.write(chunk)
        port.flush()
        time.sleep(delay * len(chunk))


def read_line(port: serial.Serial, deadline: float) -> str:
    data = bytearray()
    while time.monotonic() < deadline:
        byte = port.read(1)
        if not byte:
            continue
        if byte == b"\n":
            return data.rstrip(b"\r").decode("utf-8", errors="replace")
        data.extend(byte)
    raise TimeoutError("timed out waiting for the FPGA decoder")


def wait_ready(port: serial.Serial, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        line = read_line(port, deadline)
        if line == "READY":
            return
        if "failed" in line.lower() or "cannot open" in line.lower():
            raise RuntimeError(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--binary", default="/root/bonsai-lite.hybrid")
    parser.add_argument("--model", default="/root/models/Bonsai-1.7B-packed.gguf")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument(
        "--prestaged",
        action="store_true",
        help="reuse a model already validated in reserved FPGA DRAM",
    )
    args = parser.parse_args()

    tokens, token_to_id, merge_rank = load_tokenizer(args.tokenizer)
    decoder = byte_decoder()
    messages: list[tuple[str, str]] = []

    with serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=10) as port:
        port.reset_input_buffer()
        send(port, b"\x03\r", rate=1_000)
        time.sleep(0.5)
        port.reset_input_buffer()
        model_env = "BONSAI_MODEL_PRESTAGED=1 " if args.prestaged else ""
        production_env = (
            "BONSAI_MODEL_DDR=1 "
            "BONSAI_CACHED_INPUT=1 BONSAI_CACHED_OUTPUT=1 "
            "BONSAI_COMPACT_OUTPUT=1 BONSAI_PACKED_INPUT=1 "
            "BONSAI_REFERENCE_PACK=0 BONSAI_STREAM_PHASES=1 "
            "BONSAI_HW_TRANSFORMS=1 BONSAI_HW_RESIDUAL=0 "
            "BONSAI_HW_ATTENTION=1 BONSAI_HW_ATTENTION_SKIP_MASK=0x26 "
            "BONSAI_HW_ARGMAX=1 "
        )
        command = (
            f"export {model_env}{production_env}BONSAI_PARALLEL_ROWS=1; "
            "stty -echo; exec "
            f"{args.binary} --model {args.model} --server --threads {args.threads}\r"
        ).encode()
        startup = time.monotonic()
        send(port, command, rate=1_000)
        wait_ready(port, args.startup_timeout)
        startup_seconds = time.monotonic() - startup
        print(f"Bonsai-1.7B FPGA chat ready ({startup_seconds:.1f}s warm-up).")
        print("Commands: /clear, /quit")

        try:
            while True:
                user = input("\nyou> ").strip()
                if user == "/quit":
                    send(port, b"QUIT\r")
                    return
                if user == "/clear":
                    messages.clear()
                    print("history cleared")
                    continue
                if not user:
                    continue

                messages.append(("user", user))
                ids = tokenize(render_prompt(messages), token_to_id, merge_rank)
                if len(ids) + args.max_tokens >= MAX_CONTEXT:
                    messages.pop()
                    print("context limit reached; use /clear")
                    continue
                request = (
                    f"RUN {args.max_tokens} {len(ids)} "
                    + " ".join(map(str, ids))
                    + "\r"
                ).encode()
                turn_start = time.monotonic_ns()
                send(port, request)
                assistant = bytearray()
                eval_ns: list[int] = []
                output_tokens = 0
                prompt_tokens = 0
                prompt_ns = 0
                print("bonsai> ", end="", flush=True)
                deadline = time.monotonic() + 900
                while True:
                    line = read_line(port, deadline)
                    if line == "DONE":
                        break
                    fields = line.split()
                    if not fields:
                        continue
                    if fields[0] == "PROMPT":
                        prompt_tokens = int(fields[1])
                        prompt_ns = int(fields[-1])
                    elif fields[0] == "TOKEN":
                        token = int(fields[1])
                        if token not in EOS_TOKENS and token not in THINK_TOKENS:
                            output_tokens += 1
                            piece = decode_piece(tokens[token], decoder)
                            assistant.extend(piece)
                            print(piece.decode("utf-8", errors="replace"), end="", flush=True)
                    elif fields[0] == "EVAL":
                        eval_ns.append(int(fields[1]))

                turn_ns = time.monotonic_ns() - turn_start
                messages.append(("assistant", assistant.decode("utf-8", errors="replace")))
                prompt_tps = prompt_tokens * 1e9 / prompt_ns if prompt_ns else 0.0
                decode_tps = len(eval_ns) * 1e9 / sum(eval_ns) if eval_ns else 0.0
                end_to_end_tps = output_tokens * 1e9 / turn_ns if turn_ns else 0.0
                print(
                    f"\n[{prompt_tps:.3f} prompt tok/s, {decode_tps:.3f} decode tok/s, "
                    f"{end_to_end_tps:.3f} end-to-end tok/s]"
                )
        finally:
            try:
                send(port, b"QUIT\rstty echo\r")
            except Exception:
                pass


if __name__ == "__main__":
    main()
