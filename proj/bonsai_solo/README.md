# Bonsai solo: a Bonsai-1.7B-only FPGA build

This is a standalone Genesys 2 image whose only job is to decode Bonsai-1.7B (Qwen3, Q1_0 weights) as fast as the board allows. There is no CVA6, no Linux, and no HDMI terminal. A small integer-only controller sequences the existing exact accelerator, and a host PC does tokenization over the UART.

## Status (2026-11-18)

First board result, bitstream `build/bonsai_solo_ddr1600.bit` (SHA-256 `ac1f5928…`, WNS +0.020 ns at 200 MHz):

| Measurement                         | Result                                     |
|-------------------------------------|--------------------------------------------|
| Exactness gate, "Once upon a time"  | 11,1052,572,264,3908,3743,6941,444,… exact |
| Decode time per token (positions 4–19) | 95.7–100.2 ms (≈10.2 tok/s)              |
| 28 layers                           | 77.8–82.3 ms                               |
| LM head argmax                      | 7.35 ms                                    |
| 56 norm packs + embedding           | ≈10.5 ms                                   |
| CVA6 image, same gate               | ≈210 ms (4.75 tok/s)                       |
| DDR3-1600 memtest                   | 1 GiB, 0 errors                            |

The firmware drives the validated eight-command layer path; the one-command layer sequencer is unused because its retained-QKV attention writes zero K/V. Two attention cores (four did not route at 200 MHz).

Bring-up procedure: flash with `openFPGALoader -b genesys2 --freq 1000000 build/bonsai_solo_ddr1600.bit`, then `host/solo_host.py load --images DIR` (DIR from `host/trace/make_images.sh` plus `sum_table.bin` from `host/sum_table.c`; ≈15 min at 3 Mbaud, skipped when resident), then `host/solo_host.py run --prompt-ids 12522,5193,264,882 -n 16`.

## Why a separate build

The CVA6 image peaked at about 4.75 tok/s (≈210 ms/token). Simulated arithmetic for the same work is 75–110 ms/token, and the DDR weight-streaming floor is about 41 ms/token. The gap is control overhead: ~200 MMIO command/poll round trips per token through a GPIO and two clock crossings, an exclusive bus arbiter shared with four CPUs, CPU-side embedding/norm work on D-cache-less harts, and 5,376 uncached table stores per token.

Removing the CPUs also frees ~100 k LUTs, ~70 RAMB36 and ~70 DSPs for more accelerator parallelism.

## Bandwidth budget

| Quantity                          | Value                                   |
|-----------------------------------|-----------------------------------------|
| DDR3                              | 32-bit, 1600 MT/s, 6.4 GB/s peak         |
| Accelerator port                  | 256-bit at 200 MHz (MIG UI), 6.4 GB/s    |
| Measured packed QKV stream        | ≈5.9 GB/s effective                      |
| Q1 weights per token, 28 layers   | ≈198 MB (24,576 rows × 2,048 per layer)  |
| LM head (tied embedding) weights  | ≈43.7 MB                                 |
| Streaming floor                   | ≈41 ms/token at 5.9 GB/s (≈24 tok/s)     |
| Current board rate                | ≈210 ms/token (4.75 tok/s)               |

We are at roughly 20% of the memory bandwidth limit, so control and non-matmul kernels are the targets.

## Per-token budget with existing blocks (200 MHz, simulated cycles)

| Kernel                                | Per layer        | Per token      |
|---------------------------------------|------------------|----------------|
| Q1 matvecs (QKV, o, up/gate, down)    | ≈1.2 ms          | ≈34 ms         |
| LM head argmax                        | —                | ≈7.4 ms        |
| Attention, 1 core, positions 0–64     | 0.62–1.30 ms     | 17–37 ms       |
| Residual add + sum of squares (×2)    | 0.43 ms          | ≈12 ms         |
| FFN SiLU·up + Q8 pack                 | 0.34 ms          | ≈9.5 ms        |
| Norm + Q8 pack (×2)                   | 0.02 ms          | ≈0.5 ms        |
| Total                                 |                  | ≈80–100 ms     |

With zero control overhead this is already 10–12 tok/s. Reaching the streaming floor needs three RTL changes, listed in order of payoff:

1. Attention: instantiate the four-core dual-query `bonsai_attention_quad` (17–37 ms → ≈5–9 ms).
2. Residual: fold the residual add and sum of squares into the matvec row writer so it overlaps weight streaming (12 ms → ≈0).
3. FFN pack: widen SiLU/Q8 packing from one FMA to eight (9.5 ms → ≈1.5 ms).

Target after all three: ≈50 ms/token, about 20 tok/s.

## Reaching the streaming floor

Production decodes a single batch-1 stream. The first target is to make the ≈41 ms/token weight-streaming time the actual token time and then raise the bandwidth itself. Greedy output stays bit-exact throughout.

1. Maximal pipelining: DDR is never idle. Attention, norms, residual and packing currently leave the bus idle between matrices. A ≈1 MiB on-chip weight prefetch FIFO (the part has ≈2 MiB of BRAM, mostly freed by removing CVA6) keeps streaming the next matrix while those kernels run, and kernels start on partial results (row-streamed residual, packing and norm) instead of waiting for whole vectors.
2. Faster DDR. The Kintex-7 -2 HP banks support DDR3-1866. The board's DDR3 is rated 1600, so running it at 1866 MT/s is an experiment: +17% bandwidth if it calibrates and passes an on-board memory test.

| Stage                                    | Est. ms/token | Est. tok/s |
|------------------------------------------|---------------|------------|
| CVA6 image today                          | 210           | 4.75       |
| Solo build, existing blocks               | 80–100        | 10–12      |
| + quad attention, residual fold, wide FFN | ≈50           | ≈20        |
| + weight prefetch, overlapped kernels     | ≈41           | ≈24        |
| + DDR3-1866                               | ≈35           | ≈28        |
| + K=4 speculative verification            | ≈35 per pass  | ≈40–60     |

3. Speculative decoding, still a single batch-1 stream. Decode is bandwidth-bound: each weight beat is used by one dot product while the adders idle. A K-vector matvec datapath (K = 4) applies every 256-bit weight beat to the current token plus K−1 draft tokens, costing ≈K×256 int8 adds per cycle and no extra DDR traffic, so one weight pass yields the model's prediction at all K positions. Greedy acceptance keeps the longest prefix of drafts that match the model's own argmax and then emits the model's token at the first mismatch, so the output is token-for-token identical to plain greedy decoding (the integer Q30 arithmetic makes batch-K logits bit-identical to batch-1). The worst case is one token per pass. Drafts come from prompt lookup (n-gram match against the context) first, with a small on-FPGA draft head as a later option; the drafting policy lives in firmware. Attention and KV writes run per position; rejected positions are overwritten on the next pass. Batched prefill is not a goal, though it falls out of the same datapath for free.

For reference, the same model on the host laptop (Ryzen 9 7940HS) decodes at 83 tok/s on 8 CPU threads and 107 tok/s on the Radeon 780M iGPU through llama.cpp Vulkan.

## Architecture

```
            UART 3 Mbaud (host: tokenizer, chat UI, firmware/model loader)
                 │
   ┌─────────────▼──────────────┐
   │ PicoRV32 controller, 100MHz│  firmware in 64 KiB BRAM, reloadable over UART
   │  - token loop, embedding   │  (no resynthesis to change control logic)
   │  - accelerator registers   │
   │  - SD raw-sector loader    │
   └──────┬───────────┬─────────┘
          │ regs      │ 32-bit AXI (narrow DDR access, SD DMA)
   ┌──────▼─────┐  ┌──▼────────┐
   │ Bonsai     │  │ native SD │ 4-bit, ~20 MB/s: model (248 MB) loads in ~15 s
   │ accelerator│  └──┬────────┘
   │ + layer    │     │
   │ sequencer  │     │
   └──────┬─────┘     │
          │ 256-bit AXI, priority to accelerator
       ┌──▼───────────▼──┐
       │ MIG DDR3 1 GiB  │
       └─────────────────┘
```

- The controller reaches accelerator registers through a direct register file in the 200 MHz domain via one handshake CDC, replacing the GPIO/status path.
- The accelerator keeps its exact arithmetic and DDR layout (`B8UF` packed model at `0xaf000000`, KV cache at `0xa0000000`, scratch at `0xbe000000`), so the existing bit-exact simulations remain the reference.
- The whole 28-layer stack runs as one sequencer command per token. The controller only does embedding expansion (integer: fp16 scale → fp32 by field manipulation, sign bit from the Q1 mask), layer-0 RMS scale (the integer `rms_scale_f` port already exists in hardware), the final norm, and the argmax command.
- Firmware is the only thing that changes between experiments. The bitstream exposes cycle counters per kernel phase so profiling needs no rebuild.

## Host protocol

Framed binary packets over UART at 3,000,000 baud: `LOAD_FW`, `LOAD_MODEL_SD(extent)`, `LOAD_MODEL_UART`, `RESET_CTX`, `PROMPT(token ids)`, `GENERATE(n)`. The board streams back one 4-byte token record per generated token plus an optional per-token cycle profile. `tools/bonsai_solo_host.py` handles tokenization, ChatML templating and display.

## Exactness gate

Prompt token IDs `12522,5193,264,882` must produce the prefix `11,1052,572,264` (the existing known-good gate), and a 48-token greedy decode must match the CVA6 image token for token.

## Build plan

1. Copy the newest accelerator RTL (sequencer, quad attention, rms scale) from the `cva6-smp` worktree into `rtl/`, and write the new top: MIG, accelerator, PicoRV32, UART, native SD.
2. Write firmware and the host tool, and simulate the full token loop with a DDR model (on `vivado-v0`, never locally).
3. First synthesis on `vivado-v0`: existing blocks plus quad attention, per-phase cycle counters, and a DDR3-1866 MIG variant built in parallel. Measure on the board.
4. Second synthesis: residual fold, wide FFN pack, weight prefetch FIFO, kernel overlap, and the K=4 verification datapath, chosen by what the counters show. Speculative drafting and acceptance then live entirely in firmware.
