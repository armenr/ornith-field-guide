#!/usr/bin/env bash
# Serve Ornith-1.0-9B (Dense) on a single RTX 5090. Fits fully on the GPU (~7.4 GB at Q6_K), ~130 tok/s.
# Tiny weights -> tons of VRAM headroom for KV, so it can carry a large context cheaply.
set -u
LLAMA="${LLAMA_SERVER:-llama-server}"
MODEL="${ORNITH_9B:-$HOME/models/ornith/ornith-1.0-9b-Q6_K.gguf}"
PORT="${PORT:-8096}"
CTX="${CTX:-32768}"   # bump for long-context / multi-round self-fix (9B KV is cheap; it has the headroom).
NP="${NP:-1}"         # -np 1: one request gets the FULL CTX. NP=4 splits the KV pool (each gets ~CTX/NP).
KVQ="${KVQ:-}"        # set KVQ="--cache-type-k q8_0 --cache-type-v q8_0" to halve KV if you push CTX high.
exec "$LLAMA" -m "$MODEL" \
  -ngl 99 \
  -c "$CTX" -fa on -np "$NP" $KVQ \
  --host 127.0.0.1 --port "$PORT" \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1
# Sampling is set per-request: temperature 0.6-1.0, top_p 0.95, top_k 20, min_p 0, max_tokens >= 32000.
# Chain-of-thought arrives in choices[0].message.reasoning_content (deepseek reasoning format).
