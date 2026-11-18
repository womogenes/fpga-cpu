#!/usr/bin/env python3
"""Expose the FPGA Bonsai server as a local line-protocol subprocess."""

from __future__ import annotations

import argparse
import os
import select
import shlex
import sys
import time

import serial


def write_paced(port: serial.Serial, data: bytes, rate: int = 3_000) -> None:
    started = time.monotonic()
    for index, byte in enumerate(data, 1):
        port.write(bytes((byte,)))
        port.flush()
        delay = started + index / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def read_line(port: serial.Serial, pending: bytearray,
              deadline: float | None = None) -> bytes:
    while True:
        newline = pending.find(b"\n")
        if newline >= 0:
            line = bytes(pending[:newline + 1])
            del pending[:newline + 1]
            return line
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for FPGA serial output")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            pending.extend(chunk)


def wait_for(port: serial.Serial, pending: bytearray, marker: bytes,
             timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while marker not in received:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {marker!r}")
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            received.extend(chunk)
    pending.extend(received[received.find(marker) + len(marker):])
    return bytes(received)


def wait_for_any(port: serial.Serial, markers: tuple[bytes, ...],
                 timeout: float) -> tuple[bytes | None, bytes]:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            received.extend(chunk)
            for marker in markers:
                if marker in received:
                    return marker, bytes(received)
    return None, bytes(received)


def expand_result(line: bytes) -> bytes:
    fields = line.strip().split()
    if not fields or fields[0] != b"RESULT":
        return line.replace(b"\r", b"")
    if len(fields) < 4:
        raise RuntimeError(f"malformed FPGA result: {line!r}")
    generated_count = int(fields[1])
    evaluated_count = int(fields[2])
    decode_ns = int(fields[3])
    tokens = [int(value) for value in fields[4:]]
    if len(tokens) != generated_count or evaluated_count > generated_count:
        raise RuntimeError(f"inconsistent FPGA result: {line!r}")
    per_eval, remainder = (divmod(decode_ns, evaluated_count)
                           if evaluated_count else (0, 0))
    output = bytearray()
    for index, token in enumerate(tokens):
        output.extend(f"TOKEN {token}\n".encode())
        if index < evaluated_count:
            elapsed = per_eval + (1 if index < remainder else 0)
            output.extend(f"EVAL {elapsed}\n".encode())
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--serial", default=os.environ.get(
        "BONSAI_SERIAL", "/dev/ttyUSB0"))
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--binary", default=os.environ.get(
        "BONSAI_FPGA_RUNTIME", "/root/bonsai-lite-4tps"))
    args, _ = parser.parse_known_args()

    pending = bytearray()
    started = time.monotonic()
    def report_phase(name: str) -> None:
        print(f"[bridge] {name} {time.monotonic() - started:.3f}s",
              file=sys.stderr, flush=True)

    with serial.Serial(args.serial, args.baud, timeout=0.1,
                       write_timeout=10) as port:
        report_phase("serial-open")
        port.reset_input_buffer()
        write_paced(port, b"\r", rate=100)
        shell_markers = (b"# ", b"$ ")
        marker, _ = wait_for_any(port, shell_markers + (b"login: ",), 3)
        if marker is None:
            write_paced(port, b"\x03\r", rate=100)
            marker, _ = wait_for_any(port, shell_markers + (b"login: ",), 15)
        if marker == b"login: ":
            write_paced(port, b"root\r", rate=100)
            marker, _ = wait_for_any(port, shell_markers, 30)
        if marker not in shell_markers:
            raise TimeoutError("FPGA console did not reach a shell or login")
        report_phase("shell-ready")
        write_paced(port, b"stty -echo\r", rate=100)
        marker, _ = wait_for_any(port, shell_markers, 15)
        if marker not in shell_markers:
            raise TimeoutError("FPGA shell did not acknowledge stty")
        report_phase("stty-ready")
        source_binary = shlex.quote(args.binary)
        if args.binary.startswith("/tmp/"):
            runtime_binary = source_binary
            prepare_runtime = ""
        else:
            runtime_binary = "/tmp/bonsai-lite.run"
            prepare_runtime = (
                f"test -x {runtime_binary} || {{ "
                f"cp {source_binary} {runtime_binary} && "
                f"chmod 0755 {runtime_binary}; }}; "
            )
        command = (
            prepare_runtime +
            "BONSAI_STARTUP_PROFILE=1 BONSAI_MODEL_PRESTAGED=1 "
            "BONSAI_MODEL_DDR=1 "
            "BONSAI_CACHED_INPUT=1 BONSAI_CACHED_OUTPUT=1 "
            "BONSAI_COMPACT_OUTPUT=1 BONSAI_PACKED_INPUT=1 "
            "BONSAI_REFERENCE_PACK=0 BONSAI_STREAM_PHASES=1 "
            "BONSAI_PARALLEL_ROWS=1 BONSAI_HW_TRANSFORMS=1 "
            "BONSAI_HW_RESIDUAL=1 "
            "BONSAI_HW_ATTENTION=1 BONSAI_HW_ATTENTION_SKIP_MASK=0 "
            "BONSAI_HW_ARGMAX=1 "
            f"exec {runtime_binary} --model /root/models/"
            f"Bonsai-1.7B-packed.gguf --server --threads {args.threads}\r"
        ).encode()
        write_paced(port, command, rate=1_000)
        report_phase("runtime-command-sent")

        deadline = time.monotonic() + 900
        while True:
            line = read_line(port, pending, deadline).strip(b"\r\n")
            if line == b"READY":
                report_phase("runtime-ready")
                # Warm the immutable ChatML user-role prefix before publishing
                # readiness. The first visible prompt is therefore genuinely
                # ready to accept a request, with no hidden work still queued.
                write_paced(port, b"RUN 0 3 151644 872 198\r")
                wait_for(port, pending, b"DONE\r\n", 30)
                report_phase("prefix-ready")
                sys.stdout.write("READY\n")
                sys.stdout.flush()
                break
            if line:
                print(f"[fpga +{time.monotonic() - started:.3f}s] "
                      f"{line.decode(errors='replace')}",
                      file=sys.stderr, flush=True)
            if b"failed" in line or b"cannot open model" in line:
                raise RuntimeError(line.decode(errors="replace"))

        while True:
            # A single UART read can contain many protocol lines. Drain those
            # lines before waiting for another readability edge; otherwise a
            # decoder that batches TOKEN/EVAL output can leave DONE stranded
            # in this userspace buffer forever.
            if b"\n" in pending:
                line = read_line(port, pending)
                sys.stdout.buffer.write(expand_result(line))
                sys.stdout.buffer.flush()
                continue
            readable, _, _ = select.select([sys.stdin, port], [], [], 0.1)
            if sys.stdin in readable:
                request = sys.stdin.buffer.readline()
                if not request:
                    return
                write_paced(port, request)
                if request.strip() == b"QUIT":
                    return
            if port in readable or port.in_waiting:
                line = read_line(port, pending)
                sys.stdout.buffer.write(expand_result(line))
                sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
