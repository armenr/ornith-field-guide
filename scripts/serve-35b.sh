#!/usr/bin/env bash
# Serve Ornith-1.0-35B (MoE) on a single RTX 5090 via llama.cpp's OpenAI-compatible server.
#
# Set LLAMA_SERVER to your CUDA-enabled llama-server binary (Blackwell/sm_120 needs a
# recent build, CUDA >= 12.8). Set ORNITH_35B to the GGUF path if not the default.
set -u
LLAMA="${LLAMA_SERVER:-llama-server}"
MODEL="${ORNITH_35B:-$HOME/models/ornith/ornith-1.0-35b-Q6_K.gguf}"
PORT="${PORT:-8095}"

# --n-cpu-moe 6 : keep ATTENTION on the GPU, park the cold expert tensors of 6 layers on CPU.
#   The 35B Q6_K is ~26.6 GiB; on a 32 GB card it fits fully ONLY if <~4 GB is used by other
#   apps. n-cpu-moe lets it fit around a few GB of other GPU usage with almost no speed loss
#   (it's an MoE: only ~2-3B of 35B params fire per token). ~151 tok/s here.
#   If your card is otherwise EMPTY you can drop --n-cpu-moe and just use -ngl 99 (~144 tok/s).
#   DO NOT use whole-layer offload (-ngl 34): that drops attention to CPU too -> ~50 tok/s.
exec "$LLAMA" -m "$MODEL" \
  -ngl 99 --n-cpu-moe 6 \
  -c 32768 -fa on \
  --host 127.0.0.1 --port "$PORT" \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1
# --jinja            : use the GGUF's embedded (correct) Qwen-3.5 chat template
# --reasoning-format : split the model's <think>…</think> into a separate `reasoning_content`
#                      field so `content` is the clean answer
# --reasoning-budget -1 : let it think freely (it's a reasoning model; don't starve it)
