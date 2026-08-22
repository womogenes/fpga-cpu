#!/bin/sh
set -eu

model=${BONSAI_MODEL:-/root/models/Bonsai-1.7B.gguf}
export BONSAI_MODEL="$model"
context=${BONSAI_CONTEXT:-512}
threads=${BONSAI_THREADS:-4}

exec /usr/bin/bonsai-runtime \
    --model "$model" \
    --ctx-size "$context" \
    --threads "$threads" \
    --threads-batch "$threads" \
    --batch-size 64 \
    --ubatch-size 32 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --conversation \
    --simple-io \
    --no-warmup \
    "$@"
