# Path A (vLLM + NVFP4) feasibility on a single RTX 5090 — as of 2026-06

**Verdict: moderate-to-hard, and it may not work cleanly *today*. The weights are ready; the blocker
is that NVFP4-MoE on consumer Blackwell (SM120) in vLLM is bleeding-edge with documented instability.**
Worth a timeboxed attempt if you want concurrency/native tool-parsing — but llama.cpp (Path B) is
faster single-stream and rock-solid right now.

## What's ready (low risk)
- **The quant exists and fits.** `sakamakismile/Ornith-1.0-35B-NVFP4` — **21.9 GB**, single safetensors,
  `quant_method: compressed-tensors`, `format: nvfp4-pack-quantized`, arch `Qwen3_5MoeForConditionalGeneration`,
  tagged `vllm`. 21.9 GB leaves ~10 GB for KV on a 32 GB card → comfortable. (4.4K downloads, updated 2026-06-25.)

## The real blocker (high risk): vLLM NVFP4-MoE on SM120 is brand-new
This exact combination — **NVFP4 + MoE + consumer Blackwell (RTX 5090, SM120)** — is the most
bleeding-edge corner of vLLM:
- The enabling work landed *recently* as PRs: NVFP4 w4a4 CUTLASS for SM120 ([#21309](https://github.com/vllm-project/vllm/pull/21309)) and **NVFP4 MoE for SM120** ([#24968](https://github.com/vllm-project/vllm/pull/24968)).
- There are **open failures on near-identical models**: a 35B-A3B MoE NVFP4 reportedly hits *three*
  stability failure modes on a 5090 — illegal instruction, CUDA-graph-replay crash, silent engine hang
  ([HF discussion](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/discussions/9)) — and "No NvFp4
  MoE backend supports the deployment configuration" startup failures ([vLLM #35065](https://github.com/vllm-project/vllm/issues/35065),
  feature gap [#31085](https://github.com/vllm-project/vllm/issues/31085)). Ornith-35B is the same class
  of model, so expect to meet these.

## If you attempt it — the config most likely to work
- Use the **`vllm/vllm-openai:cu130-nightly` Docker image** (ships kernels compiled for SM 12.0).
- Set SM120 explicitly; don't rely on arch defaults.
- Add **`--enforce-eager`** to dodge the CUDA-graph-replay crash (costs some speed, buys stability).
- Keep the official parsers (`--tool-call-parser qwen3_xml --reasoning-parser qwen3`), single GPU
  (no `--tensor-parallel-size`), `--max-model-len 65536`, `--gpu-memory-utilization 0.92`.

## Is the reward worth it?
- **Single-stream speed:** reported NVFP4 on a 5090 is **~80 tok/s** (for a dense model; MoE TBD) —
  i.e. *possibly slower than our llama.cpp Q6_K at 151 tok/s*. The single-user win is not obvious.
- **The actual win is concurrency:** vLLM's continuous batching serves many agent requests at once and
  gives clean native tool/reasoning parsers. If you're running one agent, Path B is better today; if
  you want to serve a team or many parallel agent calls, that's vLLM's reason to exist.

## Effort estimate
- Download NVFP4: ~21.9 GB ≈ **~75 min** at ~4.5 MB/s (use `scripts/download.sh`).
- Docker image pull: several GB.
- Config trial-and-error: **1–3 hours**, with a real chance of hitting the documented failure modes
  and having to fall back.
- **Difficulty: ~6–7/10** — almost entirely tooling/stability, not concept.

## Recommendation
Timebox it (≈ an afternoon, via the cu130-nightly Docker image + `--enforce-eager`). If it serves the
tool-call sanity test cleanly, great — you've got batched agentic serving. If it hits the SM120 NVFP4-MoE
failure modes, don't fight it; Path B (llama.cpp) already does single-stream agentic coding well today,
and you can revisit vLLM in a few weeks as the SM120 kernels stabilize.

*Sources: [vLLM #21309](https://github.com/vllm-project/vllm/pull/21309) · [vLLM #24968](https://github.com/vllm-project/vllm/pull/24968) · [vLLM #35065](https://github.com/vllm-project/vllm/issues/35065) · [vLLM #31085](https://github.com/vllm-project/vllm/issues/31085) · [Qwen3.6-35B-A3B-NVFP4 stability report](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/discussions/9) · [aliez-ren/vllm-qwen3.5-nvfp4-sm120](https://github.com/aliez-ren/vllm-qwen3.5-nvfp4-sm120)*
