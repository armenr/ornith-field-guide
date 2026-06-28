#!/usr/bin/env bash
# Serve Ornith-1.0-9B (Dense) on a single RTX 5090. Fits fully on the GPU (~7.4 GB), ~130 tok/s.
set -u
LLAMA="${LLAMA_SERVER:-llama-server}"
MODEL="${ORNITH_9B:-$HOME/models/ornith/ornith-1.0-9b-Q6_K.gguf}"
PORT="${PORT:-8096}"
exec "$LLAMA" -m "$MODEL" \
  -ngl 99 \
  -c 32768 -fa on \
  --host 127.0.0.1 --port "$PORT" \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1
