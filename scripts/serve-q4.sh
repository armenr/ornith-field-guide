#!/usr/bin/env bash
# RECOMMENDED daily-driver config (verified 2026-06-29): Ornith-1.0-35B Q4_K_M on llama.cpp,
# FULLY on the GPU (no CPU offload), on a single RTX 5090.
#
# Why this is the optimized pick (see docs/optimized-config.md for the full study):
#   - Q4_K_M (21 GB) fits ENTIRELY on the 32 GB card -> NO --n-cpu-moe -> ~237-242 tok/s sustained
#     (faster than Q6_K's ~150 with forced offload, and faster than vLLM/NVFP4's 214).
#   - Quality is statistically indistinguishable from Q6_K (eval/trie/regex trade wins within
#     run-to-run variance; 4-bit k-quant costs nothing measurable on real code).
#   - Reasoning-loop rate is low and far below vLLM/NVFP4 (whose 67% loop rate is an
#     NVFP4+vLLM-decode artifact, NOT a bit-width problem).
set -u
LLAMA="${LLAMA_SERVER:-llama-server}"
MODEL="${ORNITH_35B_Q4:-$HOME/models/ornith/ornith-1.0-35b-Q4_K_M.gguf}"
PORT="${PORT:-8095}"
CTX="${CTX:-65536}"   # Q4 leaves KV headroom; drop to 49152/32768 if you OOM around a heavy desktop

# -ngl 99 with NO --n-cpu-moe: the whole model lives on the GPU. This is the speed unlock vs Q6_K.
exec "$LLAMA" -m "$MODEL" \
  -ngl 99 \
  -c "$CTX" -fa on \
  --host 127.0.0.1 --port "$PORT" \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1
# Sampling is set per-request: temperature 0.6-1.0, top_p 0.95, top_k 20, min_p 0, max_tokens >= 32000.
# Chain-of-thought arrives in choices[0].message.reasoning_content (deepseek reasoning format).
