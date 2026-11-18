# Bonsai solo: a Bonsai-1.7B-only FPGA build

This is a standalone Genesys 2 image whose only job is to decode Bonsai-1.7B (Qwen3, Q1_0 weights) as fast as the board allows. There is no CVA6, no Linux, and no HDMI terminal. A small integer-only controller sequences the existing exact accelerator, and a host PC does tokenization over the UART.

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
3. First synthesis on `vivado-v0`: existing blocks plus quad attention. Measure on the board with per-phase counters.
4. Second synthesis: residual fold and wide FFN pack, chosen by what the counters show.
