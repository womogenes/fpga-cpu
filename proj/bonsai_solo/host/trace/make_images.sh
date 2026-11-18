#!/usr/bin/env bash
# Produce the two DDR images the solo build preloads, using the unmodified
# CVA6 runtime and tools so the layouts match the validated gateware:
#   model-compact.bin  compact B8UF model for 0xaf000000 (248,302,272 bytes)
#   scratch.bin        32 MiB scratch image for 0xbe000000 (sequencer
#                      descriptors, RoPE and Q/K tables, 57 norm slots)
# The runtime runs on x86 against fake_hw.c; trace.txt records its
# per-token accelerator register writes.
set -euo pipefail

out="${1:?usage: make_images.sh OUTPUT_DIR}"
cva6="${CVA6_DIR:-/home/willi/coding/ai/bonsai/fpga-cpu-cva6-smp/proj/cva6}"
llama="${LLAMA_DIR:-/home/willi/coding/ai/bonsai/llama.cpp}"
source_model="${BONSAI_SOURCE_MODEL:-/home/willi/coding/ai/bonsai/Bonsai-1.7B.gguf}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$cva6/build/bonsai-runtime"
software="$cva6/software"
limit=(systemd-run --user --scope --quiet -p MemoryMax=3G --)

mkdir -p "$out"
gcc -O2 -I"$manifest" "$cva6/tools/compact_bonsai_staged_model.c" \
    -o "$out/compact"
gcc -D_GNU_SOURCE -O1 -std=gnu11 -pthread -ffp-contract=off \
    -include "$here/fake_hw.h" -I"$manifest" -I"$software" \
    "$software/bonsai_main.c" "$software/bonsai_accelerator.c" \
    "$software/bonsai_attention_cpu.c" "$here/fake_hw.c" -lm \
    -o "$out/bonsai-lite-trace"

PYTHONPATH="$llama/gguf-py" "${limit[@]}" python3 \
    "$cva6/tools/repack_bonsai_model.py" "$source_model" \
    "$out/model-compact.bin"
"${limit[@]}" "$out/compact" "file:$out/model-compact.bin"

rm -f "$out"/fpga-model.*
(
    cd "$out"
    env FAKE_HW_MODEL="$out/model-compact.bin" FAKE_HW_RUN_DIR="$out" \
        FAKE_HW_TRACE="$out/trace.txt" FAKE_HW_SCRATCH="$out/scratch.bin" \
        BONSAI_MODEL_PRESTAGED=1 BONSAI_FULL_MODEL_MAPPING=1 \
        BONSAI_MODEL_DDR=1 BONSAI_CACHED_INPUT=0 BONSAI_CACHED_OUTPUT=0 \
        BONSAI_COMPACT_OUTPUT=1 BONSAI_FUSED_QKV=1 BONSAI_FUSED_FFN=1 \
        BONSAI_DIRECT_FFN=1 BONSAI_GQA_ATTENTION=1 BONSAI_PARALLEL_ROWS=1 \
        BONSAI_HW_TRANSFORMS=1 BONSAI_HW_RESIDUAL=1 \
        BONSAI_CHAIN_RESIDUAL=1 BONSAI_HW_ATTENTION=1 \
        BONSAI_HW_ATTENTION_SKIP_MASK=0 BONSAI_HW_ARGMAX=1 \
        BONSAI_LAYER_SEQUENCER=1 BONSAI_LAYER_SEQUENCE_ALL=1 \
        "${limit[@]}" ./bonsai-lite-trace --model unused --threads 1 \
        --prompt-ids 12522,5193 -n 2 || true
)
sha256sum "$out/model-compact.bin" "$out/scratch.bin"
