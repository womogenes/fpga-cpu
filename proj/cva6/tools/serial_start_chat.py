#!/usr/bin/env python3
"""Log in on the board UART, launch Bonsai chat, and leave it at its prompt."""

from __future__ import annotations

import argparse
import re
import sys
import time

import serial


def wait_for(port: serial.Serial, marker: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while marker not in received:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {marker!r}")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            received.extend(chunk)
            if len(received) > 65536:
                del received[:-32768]
    return bytes(received)


def send_line(port: serial.Serial, line: str, delay: float) -> None:
    for byte in line.encode() + b"\r":
        port.write(bytes((byte,)))
        port.flush()
        if delay > 0:
            time.sleep(delay)


def report_benchmark(transcript: bytes) -> None:
    text = transcript.decode(errors="replace").replace("\r", "")
    sequence_ok = re.search(
        r"^25,220,16,24,24,15,11,279,1156$", text, re.MULTILINE
    ) is not None
    decode = re.search(
        r"decode: 8 tokens in ([0-9.]+)s \(([0-9.]+)s/token, "
        r"([0-9.]+) tok/s\)", text
    )
    profile = re.search(
        r"Bonsai FPGA profile: .* pack=([0-9.]+)ms "
        r"hardware=([0-9.]+)ms output=([0-9.]+)ms", text
    )
    prompt = re.search(r"prompt: 1 tokens in ([0-9.]+)s", text)
    hardware_argmax = "Bonsai FPGA hardware argmax enabled" in text
    if decode is None:
        print("ACCEPTANCE FAIL: decode timing was not reported")
        return
    decode_tps = float(decode.group(3))
    passed = sequence_ok and hardware_argmax and decode_tps >= 1.0
    print(
        f"ACCEPTANCE {'PASS' if passed else 'FAIL'}: exact_tokens={sequence_ok} "
        f"hardware_argmax={hardware_argmax} decode_tok_s={decode_tps:.3f}"
    )
    if profile is not None and prompt is not None:
        hardware_ms = float(profile.group(2))
        wall_ms = 1000.0 * (float(prompt.group(1)) + float(decode.group(1)))
        share = hardware_ms / wall_ms if wall_ms else 0.0
        limiter = "FPGA compute/memory" if share >= 0.5 else "CPU-side work"
        print(
            f"BOTTLENECK: {limiter}; accelerator wait is "
            f"{share * 100.0:.1f}% of measured forward wall time"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--character-delay-us", type=float, default=200.0)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--command",
        default=(
            "export BONSAI_MODEL=/root/models/Bonsai-1.7B.gguf "
            "BONSAI_MODEL_PRESTAGED=1 BONSAI_PARALLEL_ROWS=1; "
            "exec /usr/bin/bonsai-chat --no-progress"
        ),
    )
    args = parser.parse_args()
    delay = args.character_delay_us / 1_000_000

    with serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=10) as port:
        wait_for(port, b"buildroot login: ", args.timeout)
        send_line(port, "root", delay)
        wait_for(port, b"# ", 30.0)
        if not args.skip_benchmark:
            benchmark = (
                "BONSAI_PROFILE=1 /usr/bin/bonsai-lite --model "
                '"$BONSAI_MODEL" --threads 4 -n 9 --prompt-ids 12522 '
                "2>&1 | tee /root/bonsai-acceptance.log; "
                "printf '\\n__BONSAI_BENCH_DONE__\\n'"
            )
            send_line(
                port,
                "export BONSAI_MODEL=/root/models/Bonsai-1.7B.gguf "
                "BONSAI_MODEL_PRESTAGED=1 BONSAI_PARALLEL_ROWS=1; "
                + benchmark,
                delay,
            )
            transcript = wait_for(
                port, b"\r\n__BONSAI_BENCH_DONE__\r\n",
                max(args.timeout, 900.0),
            )
            report_benchmark(transcript)
        send_line(port, args.command, delay)
        wait_for(port, b"you> ", args.timeout)
        print("\nBonsai chat is ready on the FPGA console.")


if __name__ == "__main__":
    main()
