#!/usr/bin/env python3
"""Run repeatable Bonsai decode benchmarks over the CVA6 UART."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
import time

import serial


PROMPT_IDS = (151644, 872, 198, 9707, 151645, 198, 151644, 77091,
              198, 151667, 151668, 271)
ACCEPT_PROMPT_IDS = (12522, 5193, 264, 882)
ACCEPT_TOKEN_PREFIX = (11, 1052, 572, 264)
ACCEPT_GENERATED_TOKENS = 48


def send(port: serial.Serial, data: bytes, rate: int = 100) -> None:
    """Write at a UART-safe pace without benchmark-only turn delay."""
    started = time.monotonic()
    for index, byte in enumerate(data, 1):
        port.write(bytes((byte,)))
        port.flush()
        delay = started + index / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def receive_until(port: serial.Serial, markers: tuple[bytes, ...],
                  timeout: float) -> tuple[bytes, bytes]:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if not chunk:
            continue
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        received.extend(chunk)
        for marker in markers:
            if marker in received:
                return marker, bytes(received)
    raise TimeoutError(f"timed out; tail={bytes(received[-1500:])!r}")


def reach_shell(port: serial.Serial) -> bytes:
    """Interrupt a foreground program and handle either Buildroot prompt."""
    send(port, b"\x03\r")
    marker, _ = receive_until(port, (b"# ", b"$ ", b"login: "), 20)
    if marker == b"login: ":
        send(port, b"root\r")
        marker, _ = receive_until(port, (b"# ", b"$ "), 30)
    return marker


def receive_run(port: serial.Serial, timeout: float) -> tuple[bytes, float]:
    """Receive one run and time decode with the host's monotonic clock.

    The FPGA's Linux timebase metadata can differ from the implemented CPU
    clock. Timing from the completed PROMPT line through DONE covers exactly
    the EVAL operations while remaining independent of the target clock.
    """
    deadline = time.monotonic() + timeout
    received = bytearray()
    decode_begin: float | None = None
    decode_end: float | None = None
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if not chunk:
            continue
        now = time.monotonic()
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        received.extend(chunk)
        if decode_begin is None:
            marker = received.find(b"DECODE_BEGIN\r\n")
            if marker >= 0:
                decode_begin = now
        if decode_begin is not None and decode_end is None:
            marker = received.find(b"DECODE_END\r\n")
            if marker >= 0:
                decode_end = now
        if b"DONE\r\n" in received:
            if decode_begin is None or decode_end is None:
                raise RuntimeError("DONE arrived without decode wall markers")
            return bytes(received), decode_end - decode_begin
    raise TimeoutError(f"run timed out; tail={bytes(received[-1500:])!r}")


def parse_run_result(output: bytes) -> tuple[list[int], list[int]]:
    compact = re.search(rb"^RESULT (\d+) (\d+) (\d+)((?: \d+)*)\r?$",
                        output, re.MULTILINE)
    if compact is None:
        return ([int(value) for value in re.findall(rb"TOKEN (\d+)", output)],
                [int(value) for value in re.findall(rb"EVAL (\d+)", output)])
    generated_count = int(compact.group(1))
    evaluated_count = int(compact.group(2))
    decode_ns = int(compact.group(3))
    tokens = [int(value) for value in compact.group(4).split()]
    if len(tokens) != generated_count or evaluated_count > generated_count:
        raise RuntimeError(f"inconsistent compact result: {compact.group(0)!r}")
    per_eval, remainder = (divmod(decode_ns, evaluated_count)
                           if evaluated_count else (0, 0))
    eval_ns = [per_eval + (1 if index < remainder else 0)
               for index in range(evaluated_count)]
    return tokens, eval_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--attention-workers", type=int, default=0)
    parser.add_argument("--residual-workers", type=int, default=0)
    parser.add_argument("--gqa-attention",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--hardware-kv-prep",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--binary", default="/usr/bin/bonsai-lite")
    parser.add_argument("--prestaged", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--staged-model",
                        default="/root/models/Bonsai-1.7B-packed.gguf")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--model-ddr", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--uncached-model", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--reference-pack", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--packed-input", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--cached-stage", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--fast-stage", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--stage-resume-mb", type=int, default=0)
    parser.add_argument("--stage-overwrite", action="store_true")
    parser.add_argument("--prompt-count", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--prompt-ids",
                        help="comma-separated token IDs overriding the preset")
    parser.add_argument(
        "--warm-prompt-ids",
        help="comma-separated prefix to evaluate privately before timing")
    parser.add_argument("--expect-tokens",
                        help="comma-separated exact generated token IDs")
    parser.add_argument("--expect-prefix",
                        help="comma-separated exact generated token prefix")
    parser.add_argument("--min-decode-tps", type=float, default=0.0)
    parser.add_argument("--min-end-to-end-tps", type=float, default=0.0)
    parser.add_argument(
        "--accept-2tps", action="store_true",
        help="run the strict full-model correctness and >=2 tok/s gate")
    parser.add_argument("--hw-transforms", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--hw-norm", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--hw-ffn", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--hw-residual",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hw-argmax", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--hw-attention", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--hw-attention-skip-mask", type=lambda value: int(value, 0),
                        default=0)
    parser.add_argument("--cached-input", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--cached-output", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--compact-output", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--layer-trace", action="store_true")
    parser.add_argument("--matvec-trace", action="store_true")
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--replay-residual", action="store_true")
    parser.add_argument("--repeat-hw-residual", action="store_true")
    parser.add_argument("--repeat-hw-attention", action="store_true")
    parser.add_argument("--hw-residual-trace", action="store_true")
    parser.add_argument("--ffn-shadow", action="store_true")
    parser.add_argument("--norm-shadow", action="store_true")
    parser.add_argument("--attn-shadow", action="store_true")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--spin-workers", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    if args.accept_2tps:
        args.tokens = ACCEPT_GENERATED_TOKENS
        args.runs = max(args.runs, 2)
        args.layers = 0
        args.prestaged = True
        args.model_ddr = True
        args.reference_pack = False
        args.packed_input = True
        args.hw_transforms = True
        args.hw_residual = True
        args.hw_argmax = True
        args.hw_attention = True
        args.compact_output = True
        # Measure the production chat path. Detailed profiling performs many
        # clock_gettime calls on CVA6 and materially slows inference itself.
        args.profile = False
        args.prompt_ids = ",".join(map(str, ACCEPT_PROMPT_IDS))
        args.expect_prefix = ",".join(map(str, ACCEPT_TOKEN_PREFIX))
        args.min_decode_tps = max(args.min_decode_tps, 2.0)
        args.min_end_to_end_tps = max(args.min_end_to_end_tps, 2.0)
    if args.prompt_ids:
        try:
            prompt_ids = tuple(int(value) for value in
                               args.prompt_ids.split(","))
        except ValueError as error:
            parser.error(f"invalid --prompt-ids: {error}")
        if not prompt_ids or any(value < 0 for value in prompt_ids):
            parser.error("--prompt-ids must contain nonnegative token IDs")
    else:
        if not 1 <= args.prompt_count <= len(PROMPT_IDS):
            parser.error(f"--prompt-count must be 1..{len(PROMPT_IDS)}")
        prompt_ids = PROMPT_IDS[:args.prompt_count]
    warm_prompt_ids: tuple[int, ...] = ()
    if args.warm_prompt_ids:
        try:
            warm_prompt_ids = tuple(
                int(value) for value in args.warm_prompt_ids.split(","))
        except ValueError as error:
            parser.error(f"invalid --warm-prompt-ids: {error}")
        if not warm_prompt_ids or any(value < 0 for value in warm_prompt_ids):
            parser.error("--warm-prompt-ids must contain nonnegative token IDs")
    expected_tokens: tuple[int, ...] | None = None
    if args.expect_tokens:
        try:
            expected_tokens = tuple(int(value) for value in
                                    args.expect_tokens.split(","))
        except ValueError as error:
            parser.error(f"invalid --expect-tokens: {error}")
        if not expected_tokens:
            parser.error("--expect-tokens cannot be empty")
        if args.tokens != len(expected_tokens):
            parser.error("--tokens must equal the --expect-tokens length")
    expected_prefix: tuple[int, ...] | None = None
    if args.expect_prefix:
        try:
            expected_prefix = tuple(int(value) for value in
                                    args.expect_prefix.split(","))
        except ValueError as error:
            parser.error(f"invalid --expect-prefix: {error}")
        if not expected_prefix:
            parser.error("--expect-prefix cannot be empty")
        if args.tokens < len(expected_prefix):
            parser.error("--tokens must cover the --expect-prefix length")

    with serial.Serial(args.serial, args.baud, timeout=0.1,
                       write_timeout=10) as port:
        port.reset_input_buffer()
        reach_shell(port)
        send(port, b"stty -echo\r")
        time.sleep(1)
        port.reset_input_buffer()
        trace_env = (("BONSAI_LAYER_TRACE=1 " if args.layer_trace else "") +
                     ("BONSAI_TRACE=1 " if args.trace else "") +
                     ("BONSAI_MATVEC_TRACE=1 " if args.matvec_trace else "") +
                     ("BONSAI_DEBUG=1 " if args.debug else "") +
                     ("BONSAI_REPEAT_CHECK=1 " if args.repeat_check else "") +
                     ("BONSAI_REPLAY_RESIDUAL=1 "
                      if args.replay_residual else "") +
                     ("BONSAI_REPEAT_HW_RESIDUAL=1 "
                      if args.repeat_hw_residual else "") +
                     ("BONSAI_REPEAT_HW_ATTENTION=1 "
                      if args.repeat_hw_attention else "") +
                     ("BONSAI_HW_RESIDUAL_TRACE=1 "
                      if args.hw_residual_trace else "") +
                     ("BONSAI_FFN_SHADOW=1 " if args.ffn_shadow else "") +
                     ("BONSAI_NORM_SHADOW=1 " if args.norm_shadow else "") +
                     ("BONSAI_ATTN_SHADOW=1 " if args.attn_shadow else ""))
        worker_env = "" if args.spin_workers else "BONSAI_SPIN_WORKERS=0 "
        attention_worker_env = (f"BONSAI_ATTN_WORKERS={args.attention_workers} "
                                if args.attention_workers else "")
        residual_worker_env = (f"BONSAI_RESIDUAL_WORKERS={args.residual_workers} "
                               if args.residual_workers else "")
        layer_env = f"BONSAI_LAYERS={args.layers} " if args.layers else ""
        model_env = ("BONSAI_MODEL_PRESTAGED=1 " if args.prestaged else
                     f"BONSAI_MODEL={args.staged_model} ")
        command = (
            f"{model_env}"
            f"BONSAI_CACHED_INPUT={int(args.cached_input)} "
            f"BONSAI_CACHED_OUTPUT={int(args.cached_output)} "
            f"BONSAI_COMPACT_OUTPUT={int(args.compact_output)} "
            f"BONSAI_MODEL_DDR={int(args.model_ddr)} "
            f"BONSAI_UNCACHED_MODEL={int(args.uncached_model)} "
            f"BONSAI_REFERENCE_PACK={int(args.reference_pack)} "
            f"BONSAI_PACKED_INPUT={int(args.packed_input)} "
            + ("BONSAI_CACHED_STAGE=1 " if args.cached_stage else "") +
            ("BONSAI_FAST_STAGE=1 " if args.fast_stage else "") +
            (f"BONSAI_STAGE_RESUME_MB={args.stage_resume_mb} "
             if args.stage_resume_mb else "") +
            ("BONSAI_STAGE_OVERWRITE=1 " if args.stage_overwrite else "") +
            ("BONSAI_PROFILE=1 " if args.profile else "") +
            ("BONSAI_GQA_ATTENTION=1 " if args.gqa_attention else "") +
            ("BONSAI_HARDWARE_KV_PREP=1 " if args.hardware_kv_prep else "") +
            "BONSAI_STREAM_PHASES=1 " +
            f"BONSAI_HW_TRANSFORMS={int(args.hw_transforms)} "
            + (f"BONSAI_HW_NORM={int(args.hw_norm)} "
               if args.hw_norm is not None else "")
            + (f"BONSAI_HW_FFN={int(args.hw_ffn)} "
               if args.hw_ffn is not None else "")
            + f"BONSAI_HW_RESIDUAL={int(args.hw_residual)} "
            + f"BONSAI_HW_ATTENTION={int(args.hw_attention)} "
            f"BONSAI_HW_ATTENTION_SKIP_MASK=0x{args.hw_attention_skip_mask:x} "
            f"BONSAI_HW_ARGMAX={int(args.hw_argmax)} "
            + ("BONSAI_NO_MANIFEST=1 " if args.no_manifest else "")
            + f"{trace_env}"
            f"{worker_env}"
            f"{attention_worker_env}"
            f"{residual_worker_env}"
            f"{layer_env}"
            "BONSAI_PARALLEL_ROWS=1 "
            f"{args.binary} "
            # The decoder understands BGUF metadata and uses its packed CPU
            # tensors directly; PRESTAGED independently selects the copy in
            # accelerator-visible DDR.
            f"--model {shlex.quote(args.staged_model)} "
            f"--server --threads {args.threads}\r"
        ).encode()
        send(port, command)
        marker, startup_output = receive_until(
            port, (b"READY\r\n", b"cannot open model", b"failed"),
            args.startup_timeout)
        if marker != b"READY\r\n":
            raise RuntimeError(f"runtime startup failed: {marker!r}")
        if args.accept_2tps:
            required_config = (b"Bonsai runtime config: layers=28/28 ",
                               b"packed_model=1 ", b"float_output=1 ",
                               b"spin_workers=1 ",
                               b"hw_transforms=1 ", b"hw_residual=1 ",
                               b"hw_attention=1 ",
                               b"hw_argmax=1")
            missing = [value for value in required_config
                       if value not in startup_output]
            if missing:
                raise RuntimeError(
                    f"strict runtime configuration missing {missing!r}")

        if warm_prompt_ids:
            warm_request = (
                f"RUN 0 {len(warm_prompt_ids)} "
                + " ".join(map(str, warm_prompt_ids)) + "\r"
            ).encode()
            send(port, warm_request, rate=3_000)
            receive_until(port, (b"DONE\r\n",), 60)

        request = (
            f"RUN {args.tokens} {len(prompt_ids)} "
            + " ".join(map(str, prompt_ids)) + "\r"
        ).encode()
        runs: list[tuple[list[int], list[int], float, float, float]] = []
        for index in range(args.runs):
            print(f"\nHOST_RUN {index + 1}", flush=True)
            turn_start = time.monotonic()
            # Match the production serial bridge. The prior 100-byte/s path
            # charged roughly a second of artificial pacing to every turn.
            send(port, request, rate=3_000)
            output, wall_decode_seconds = receive_run(port, 600)
            wall_turn_seconds = time.monotonic() - turn_start
            tokens, eval_ns = parse_run_result(output)
            target_decode_tps = (
                len(eval_ns) * 1_000_000_000 / sum(eval_ns)
                if eval_ns else 0.0)
            wall_decode_tps = (
                len(eval_ns) / wall_decode_seconds
                if eval_ns and wall_decode_seconds > 0.0 else 0.0)
            wall_end_to_end_tps = (
                len(tokens) / wall_turn_seconds if wall_turn_seconds > 0.0
                else 0.0)
            runs.append((tokens, eval_ns, target_decode_tps,
                         wall_decode_tps, wall_end_to_end_tps))
            print(f"HOST_RESULT run={index + 1} tokens={tokens} "
                  f"wall_decode_tps={wall_decode_tps:.6f} "
                  f"target_decode_tps={target_decode_tps:.6f} "
                  f"wall_end_to_end_tps={wall_end_to_end_tps:.6f} "
                  f"wall_turn_seconds={wall_turn_seconds:.6f}", flush=True)

        deterministic = all(run[0] == runs[0][0] for run in runs[1:])
        print(f"HOST_DETERMINISTIC {int(deterministic)}", flush=True)
        send(port, b"QUIT\r")
        _, shutdown_output = receive_until(port, (b"# ", b"$ "), 60)
        sys.stdout.buffer.write(shutdown_output)
        sys.stdout.buffer.flush()
        send(port, b"stty echo\r")
        if not deterministic:
            raise SystemExit("token streams differed between identical runs")
        if expected_tokens is not None:
            expected = list(expected_tokens)
            wrong = [index + 1 for index, run in enumerate(runs)
                     if run[0] != expected]
            if wrong:
                raise SystemExit(
                    f"incorrect token sequence in runs {wrong}: "
                    f"expected {expected}")
        if expected_prefix is not None:
            expected = list(expected_prefix)
            wrong = [index + 1 for index, run in enumerate(runs)
                     if run[0][:len(expected)] != expected]
            if wrong:
                raise SystemExit(
                    f"incorrect token prefix in runs {wrong}: "
                    f"expected {expected}")
        slow = [(index + 1, run[3]) for index, run in enumerate(runs)
                if run[3] < args.min_decode_tps]
        if slow:
            raise SystemExit(
                f"decode throughput below {args.min_decode_tps:.6f} "
                f"tok/s: {slow}")
        slow_end_to_end = [
            (index + 1, run[4]) for index, run in enumerate(runs)
            if run[4] < args.min_end_to_end_tps]
        if slow_end_to_end:
            raise SystemExit(
                f"end-to-end throughput below "
                f"{args.min_end_to_end_tps:.6f} tok/s: "
                f"{slow_end_to_end}")
        if args.accept_2tps:
            print("HOST_ACCEPT_2TPS 1", flush=True)


if __name__ == "__main__":
    main()
