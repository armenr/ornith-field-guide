#!/usr/bin/env bash
# Path A (VERIFIED 2026-06-28): serve Ornith-1.0-35B-NVFP4 via vLLM in Docker on a single RTX 5090.
# Derived from the research runbook + the Li-Lee/vllm-qwen3.5-nvfp4-5090 proven recipe (same arch).
# KEY LEVER: FORCE MARLIN (W4A16). Native NVFP4-MoE CUTLASS/FlashInfer kernels crash on SM120.
set -u
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/ornith-nvfp4}"
GPU_FLAG="${GPU_FLAG:---device nvidia.com/gpu=all}"   # CDI; fallback: --gpus all
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"               # fallback: qwen3_coder (Li-Lee's proven choice)
MAXLEN="${MAXLEN:-65536}"                             # lower to 32768/16384 if OOM
NAME="${NAME:-ornith-vllm}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  $GPU_FLAG \
  --ipc=host \
  -p 8000:8000 \
  -v "$MODEL_DIR":/model:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_MOE_FORCE_MARLIN=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm \
  "$IMAGE" \
  serve /model \
    --served-model-name Ornith-1.0-35B \
    --trust-remote-code \
    --language-model-only \
    --tensor-parallel-size 1 \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "${MAX_SEQS:-1}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.78}" \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser qwen3 \
    --chat-template /model/chat_template.jinja \
    --host 0.0.0.0 --port 8000

# Fallback ladder (from research) if it won't start / misbehaves:
#  - "unrecognized arguments: --language-model-only" -> drop it; add -e to skip vision via mm limits
#  - "No NvFp4 MoE backend ..." -> add --moe-backend marlin explicitly
#  - illegal instruction / illegal memory access -> a native FP4 kernel ran; confirm Marlin warning in logs
#  - CUDA-graph crash / silent hang -> keep --enforce-eager (already set)
#  - OOM at start -> MAXLEN=32768 (then 16384), --gpu-memory-utilization 0.90
#  - tool calls as raw text / bleed into <think> -> TOOL_PARSER=qwen3_coder ; custom chat template
#  - NEVER add --cpu-offload-gb (corrupts SM120 MoE output)
