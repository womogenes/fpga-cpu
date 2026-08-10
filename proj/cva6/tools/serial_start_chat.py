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
    speed_text = text.split("__BONSAI_SPEED_BEGIN__", 1)[-1]
    speed_text = speed_text.split("__BONSAI_PROFILE_BEGIN__", 1)[0]
    profile_text = text.split("__BONSAI_PROFILE_BEGIN__", 1)[-1]
    sequence_ok = re.search(
        r"^25,220,16,24,24,15,11,279,1156$", speed_text, re.MULTILINE
    ) is not None
    decode = re.search(
        r"decode: 8 tokens in ([0-9.]+)s \(([0-9.]+)s/token, "
        r"([0-9.]+) tok/s\)", speed_text
    )
    profile = re.search(
        r"Bonsai FPGA profile: .* pack=([0-9.]+)ms "
        r"hardware=([0-9.]+)ms output=([0-9.]+)ms", profile_text
    )
    profile_prompt = re.search(
        r"prompt: 1 tokens in ([0-9.]+)s", profile_text
    )
    profile_decode = re.search(
        r"decode: 8 tokens in ([0-9.]+)s", profile_text
    )
    worker_times = {
        name: float(match.group(1))
        for name in (
            "attn_heads",
            "accel_argmax",
            "read_rows",
            "quantize",
            "attn_prep",
            "ffn_activate",
            "norm_sum",
            "norm_apply",
        )
        if (match := re.search(rf"{name}=([0-9.]+)ms/", profile_text))
    }
    hardware_argmax = "Bonsai FPGA hardware argmax enabled" in speed_text
    if decode is None:
        print("ACCEPTANCE FAIL: decode timing was not reported")
        return
    decode_tps = float(decode.group(3))
    passed = sequence_ok and hardware_argmax and decode_tps >= 1.0
    print(
        f"ACCEPTANCE {'PASS' if passed else 'FAIL'}: exact_tokens={sequence_ok} "
        f"hardware_argmax={hardware_argmax} decode_tok_s={decode_tps:.3f}"
    )
    if profile is not None and profile_prompt is not None and profile_decode is not None:
        hardware_ms = float(profile.group(2))
        wall_ms = 1000.0 * (
            float(profile_prompt.group(1)) + float(profile_decode.group(1))
        )
        if len(worker_times) == 8:
            kernel_ms = hardware_ms + sum(worker_times.values())
            share = min(kernel_ms / wall_ms, 1.0) if wall_ms else 0.0
            overhead_ms = max(wall_ms - kernel_ms, 0.0)
            limiter = "compute/memory kernels" if share >= 0.75 else "CPU overhead"
            print(
                f"BOTTLENECK: {limiter}; named FPGA and model kernels account "
                f"for {share * 100.0:.1f}% of forward wall time "
                f"({overhead_ms:.1f}ms unclassified)"
            )
        else:
            share = hardware_ms / wall_ms if wall_ms else 0.0
            print(
                "BOTTLENECK: incomplete worker profile; accelerator wait is "
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
                "printf '\\n__BONSAI_SPEED_BEGIN__\\n'; "
                "/usr/bin/bonsai-lite --model "
                '"$BONSAI_MODEL" --threads 4 -n 9 --prompt-ids 12522 '
                "2>&1 | tee /root/bonsai-speed.log; "
                "printf '\\n__BONSAI_PROFILE_BEGIN__\\n'; "
                "BONSAI_PROFILE=1 /usr/bin/bonsai-lite --model "
                '"$BONSAI_MODEL" --threads 4 -n 9 --prompt-ids 12522 '
                "2>&1 | tee /root/bonsai-profile.log; "
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
