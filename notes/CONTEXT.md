# FPGA CPU agent handoff

## Active workspace

The primary checkout is `/home/willi/coding/ai/bonsai/fpga-cpu`; all CVA6 work lives in the worktree `/home/willi/coding/ai/bonsai/fpga-cpu-cva6-smp` on branch `cva6-smp`, now at commit `dba628c` (bonsai-lite + FPU workarounds; see also 43cfdae/e420bb0/6fd63a1). The `hardware` submodule remains intentionally dirty (four-hart patches, axi_cut insertion, GPIO width config); the sibling repo `/home/willi/coding/ai/bonsai/llama.cpp` carries one uncommitted two-part change: the mul_mat dispatch in `ggml/src/ggml-cpu/ggml-cpu.c` is routed through a weak `bonsai_forward_mul_mat_hook` (default aliases the CPU path) because `ld --wrap` cannot intercept the intra-object call in current ggml. Do not revert that without also restoring a wrap-based interception.

The remote FPGA host is `thinkmax`; repository `~/fpga/fpga-cpu`, artifacts in `~/fpga/fpga-cpu/build/cva6`, UART `/dev/ttyUSB0` at 3,000,000 baud, JTAG at 1 MHz. Synthesis is local only, one Vivado job at a time. Never let a `pkill -f` pattern appear verbatim in the same remote command line that launches the process (repeated self-kill incidents), and kill any `serial_console.py`-style pollers before uploads: a leftover poller stealing UART bytes caused deterministic upload failures at 5.7%.

## Current hardware

The deployed bitstream is `build/gateware/cva6_genesys2_hdmi.bit`, SHA-256 `f3e5613d09f2b9d1317b43eb8b543ff0b6eaaa43d0d30cb2ff7b5c726ea87d0b`: four 100 MHz RV64IMAFC harts, F-only FPUs, the 160×45 HDMI terminal and software video, and the redesigned streaming Q1 accelerator (wide 64-byte-per-cycle activation banks, free-running exact-Q30 scale pipeline, non-blocking row writer, axi_cut register slice before the exclusive arbiter, undefined-state trap through the flush drain, live FSM state in status bits 23:8 for non-Q6 modes). Post-route WNS −0.580 ns on a CPU cache path; worst accelerator path −0.316 ns. Simulated Q1 rate ≈3 cycles per 128-weight block (was ≈20); all 19 unit + 4 integration sims pass, including random mid-burst stall injection.

The RAM image is `build/sdk/ram-linux.bin`, SHA-256 `a6fa7b52a56a186fcd1b2359763c13bbc770689cac63084b1499ee9126458528` (rebuilt 2026-08-08: shipped `bonsai-lite` + accelerator driver now include the fdiv/fsqrt/fcvt FPU workarounds, and `bonsai-chat` + `/usr/share/bonsai/tokenizer.pickle` are installed; supersedes `04db48ff…`, `711ad22b…`, `c97724d0…`; already staged on thinkmax): F-without-D kernel (validated: `rv64imafc` ISA line, no masking warning, four harts online, four concurrent NumPy float32 jobs bit-identical under load), `bonsai-runtime` built from `llama-completion` with the dispatch hook, packed-activation cache, parallel uncached readback, 4-thread defaults, and the tcc `stubs-lp64d.h` overlay fix.

## Physically measured results (Aug 7)

Bonsai-1.7B generates end-to-end through the accelerator: whole-matrix path engages, 73.2 G weights over 8,817 hardware matvecs in a 44-token run. Accelerator hardware time ≈0.5 s/token. End-to-end decode ≈31 s/token because ~78% of runtime is llama.cpp CPU-side graph work (attention over quantized KV, per-token graph construction, sampling over the 151,669-token vocabulary) on four D-cache-less harts; ctx 128 vs 512 makes no difference. The packed-activation cache did not show hits in its first measurement — verify with BONSAI_PROFILE (pack should drop well below 22 ms/call). Reaching interactive Bonsai rates needs an LFM-style slim custom runtime (deterministic small graph, no per-token rebuild, tiny sampler) — that is the top follow-up, worth ~10-30× on the CPU side.

Manual accelerator tests: a tcc-compiled word-store poke runs a 2048×2048 Q1 matvec in 4.2 ms wall including grant. IMPORTANT: python `/dev/mem` pokes (mmap slices AND ctypes) issue byte stores that the bus replicates across lanes — GPIO writes appear as garbage and reads mislead; only native C word stores are valid instruments. The AXI GPIO TRI register (offset 4) defaults to all-inputs; the runtime writes it, but manual readback of gpio[0] needs `g[1]=0` first.

## Boot and staging workflow

Synchronized boot from thinkmax (`nohup`-detached; console attach crashes harmlessly with a termios error after "launching Linux"):

```sh
cd ~/fpga/fpga-cpu/build/cva6 && rm -f boot.log && nohup python3 program_and_upload.py --bitstream cva6_genesys2_hdmi.bit --serial /dev/ttyUSB0 --baud 3000000 --stage2 stage2-loader.bin --model Bonsai-1.7B.gguf ram-linux.bin > boot.log 2>&1 &
```

~40 min with a model, ~21 without. After boot: `model-export 248302272 /tmp/m.gguf` copies the DDR-staged model to tmpfs (~4 min), then a stat-stamp written to `/run/fpga-model.stamp` (tcc one-liner, see memory notes) makes the runtime skip its 25-minute restage. `board_cmd.py` on thinkmax sends paced console commands (2 ms/char — unpaced pastes overrun the board RX); board output emitted while no host process holds the UART is lost forever, so keep one session attached across command+response, and Ctrl-C first if the shell might sit at a stuck foreground command or a `>` continuation prompt.

## bonsai-lite result (Aug 8)

`software/bonsai_main.c` (installed as `bonsai-lite`, also push-updatable as `/tmp/blite` via `serial_push_bin.py`) is a standalone Qwen3 decoder that keeps all 197 Q1_0 matrices on the accelerator. On hardware it reproduces the x86 llama.cpp greedy reference token-for-token ("Once upon a time" -> 11,1052,572,264,3908,3743,6941,444,10524,879,12163,304 = ", there was a young girl named Lila who lived in...") at 9.4 s/token - 3.3x faster than llama.cpp end-to-end, accelerator time 0.8 s/token. The deployed bitstream f3e5613d has three FPU silicon defects, all worked around in software: fdiv.s returns NaN (Newton rcp_f/rsqrt_f replace all hot divisions), fsqrt.s is not implemented by the FPU configuration, and fcvt.s.l returns +-1 while fcvt.s.w is correct (driver assembles floats from 32-bit halves). Only gcc-compiled word-store C probes are valid instruments (tcc emits lp64d D-instructions and SIGILLs; python /dev/mem pokes issue byte stores that the bus replicates). Next perf levers: ~5 s/token of unattributed scalar/sync overhead, an FPU-timing-fixed bitstream (recovers the workaround costs and the llama.cpp path), and the interactive chat client over bonsai-lite --server (TOKEN/EVAL protocol, clone lfm_chat.py).

## Known issues / blocked items

- SD card bulk I/O is slow **by architecture**, not degradation: the slot is SPI-attached (`CONFIG_MMC_SPI` through the Xilinx SPI core, PIO) on D-cache-less harts. Measured ~20 KiB/s for `dd conv=fsync`, ~90 KiB/s for sequential installer writes with batched writeback; `time` shows sys≈real because PIO polling is counted as system time. Before large SD writes set `vm.dirty_bytes=8388608` / `vm.dirty_background_bytes=2097152` to avoid the balance_dirty_pages stalls that previously wedged `sd-install`. The card can be power-cycled remotely: pin AE24 (`sd_reset`) drives the slot power switch, so loading the minimal `sd_power_off.bit` (on thinkmax in `build/cva6/`) for ~20 s and then reloading the CVA6 bitstream re-enumerates the card cleanly — done 2026-08-08.
- LFM deterministic-token validation on the four-hart F image was in flight at handoff: boot with `--model LFM2.5-350M-Q4_0.gguf`, then `model-export 219309792 /tmp/lfm.gguf` and `echo Hello | lfm-chat --model /tmp/lfm.gguf --max-tokens 8` (loading from tmpfs is fast; from SD it takes hours). Expected greedy token ID 521.
- llama-cli in current llama.cpp is conversation-only (ignores `-n`, rejects `-no-cnv`); `bonsai_chat.sh` passes `--conversation`, which needs re-checking against `llama-completion`'s accepted flags.
- The kernel F-only ptrace regset is still not adapted (non-critical).

## Plan status: COMPLETE (2026-08-09)

All plan steps are done and physically validated:

1. DONE: `sd-install model bonsai` completed after the remote SD power cycle — `Installed Bonsai-1.7B.gguf (3d7c6c90…)`, `RC=0`; the sha is the installer's own full read-back of the SD copy, so persistence is physically verified. Both models now live on SD p4.
2. DONE: final image `a6fa7b52…` deployed with the model staged; the *shipped* `/usr/bin/bonsai-lite` reproduces the reference tokens exactly (11,1052,572,264,3908,3743,6941,444) at 8.98 s/decode token (0.111 tok/s), stamp trick working (no re-staging).
3. DONE: `bonsai-chat` validated on hardware — `--self-test` passes (488 s tokenizer indexing), live exchange "What is 2+3?" → "2 + 3 = 5 …" at 0.104 prompt / 0.076 decode tok/s (~30-token context; attention cost grows with position).

## Remaining follow-ups (not in plan)

1. Chase the ~5 s/token unattributed scalar/sync overhead in bonsai-lite (BONSAI_PROFILE); an FPU-timing-fixed bitstream would recover the division-workaround costs and revive the llama.cpp path.
2. Consider committing the llama.cpp hook change in that repo with the user.
3. Optional: speed up bonsai-chat startup (the 8-minute CPython tokenizer indexing could be cached as a marshaled dict).
