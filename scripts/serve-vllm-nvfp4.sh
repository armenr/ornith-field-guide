#!/usr/bin/env bash
# Path A (VERIFIED 2026-06-28): serve Ornith-1.0-35B-NVFP4 via vLLM in Docker on a single RTX 5090.
# Zero host installs beyond Docker + nvidia-container-toolkit. Derived from the research runbook +
# the Li-Lee/vllm-qwen3.5-nvfp4-5090 proven recipe (same arch).
# KEY LEVER: FORCE MARLIN (W4A16). Native NVFP4-MoE CUTLASS/FlashInfer kernels crash on SM120.
#
# MODE=fast   (default) -> CUDA graphs ON. ~214 tok/s. Verified stable on Marlin (3/3 sustained reqs).
# MODE=stable           -> --enforce-eager. ~26 tok/s. Use only if you hit a CUDA-graph crash/hang.
set -u
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/ornith-nvfp4}"
GPU_FLAG="${GPU_FLAG:---device nvidia.com/gpu=all}"   # CDI; fallback: --gpus all
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"               # fallback: qwen3_coder (Li-Lee's proven choice)
MAXLEN="${MAXLEN:-65536}"                             # lower to 32768/16384 if OOM
GPU_UTIL="${GPU_UTIL:-0.75}"                          # 0.75 leaves headroom for a ~3GB desktop/compositor
KV_DTYPE="${KV_DTYPE:-fp8}"                           # fp8 = max ctx; set 'auto' for f16 KV (more precise)
NAME="${NAME:-ornith-vllm}"
MODE="${MODE:-fast}"

EAGER=""
[ "$MODE" = "stable" ] && EAGER="--enforce-eager"

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
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    $EAGER \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser qwen3 \
    --chat-template /model/chat_template.jinja \
    --host 0.0.0.0 --port 8000

echo "Launched '$NAME' (MODE=$MODE, MAXLEN=$MAXLEN, GPU_UTIL=$GPU_UTIL)."
echo "Watch readiness:  docker logs -f $NAME   (wait for 'Application startup complete')"
echo "Health:           curl -s localhost:8000/health && echo OK"
echo "Smoke test:       scripts/smoke-vllm.sh"

# Fallback ladder (from research) if it won't start / misbehaves:
#  - CUDA-graph crash / silent hang during decode -> MODE=stable (adds --enforce-eager)
#  - "unrecognized arguments: --language-model-only" -> drop it; add -e to skip vision via mm limits
#  - "No NvFp4 MoE backend ..." -> add --moe-backend marlin explicitly
#  - illegal instruction / illegal memory access -> a native FP4 kernel ran; confirm Marlin warning in logs
#  - OOM at start -> MAXLEN=32768 (then 16384), or GPU_UTIL=0.70
#  - tool calls as raw text / bleed into <think> -> TOOL_PARSER=qwen3_coder ; custom chat template
#  - NEVER add --cpu-offload-gb (corrupts SM120 MoE output)
