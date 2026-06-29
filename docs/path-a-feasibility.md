# Path A (vLLM + NVFP4) on a single RTX 5090

## UPDATE — vLLM/NVFP4 runs on a single 5090 (functional verify 2026-06-28; loop study 2026-06-29)

vLLM serves the NVFP4 35B **in Docker, on a single 5090, with zero host installs**, generating correct
code, emitting real OpenAI‑style `tool_calls`, **engaging its full `<think>` reasoning**, and decoding
at **~214 tok/s** (faster than offload-forced Q6_K's ~151, but slower than the Q4_K_M daily driver's ~237). The scary SM120 NVFP4‑MoE kernel failures
never showed up — **Marlin was auto‑selected and stable.** One command: `docker compose up -d`
(or `scripts/serve-vllm-nvfp4.sh`); verify with `scripts/smoke-vllm.sh`.

**What actually mattered (vs the predicted failure modes):**
- **Marlin auto‑selected.** Our `VLLM_*_MARLIN` env vars were "unknown" in this nightly, but vLLM fell
  back to Marlin anyway ("Your GPU does not have native support for FP4 … Marlin"). No forcing needed.
- **The real blocker was OOM, not kernels.** It loaded, then the first forward OOM'd in the
  linear‑attention op (`fla/ops/chunk_o.py`): KV cache filled the 0.85 budget, leaving no room for
  activations. Fix = `--gpu-memory-utilization 0.75 --max-num-seqs 1`.
- **CUDA graphs are FINE on Marlin.** We initially kept `--enforce-eager` (~26 tok/s) fearing the SM120
  graph crash. Dropping it (the `MODE=fast` default) enables CUDA graphs → **214 tok/s, stable across
  sustained requests.** The crash is a *native‑FP4‑kernel* problem; on Marlin (W4A16) graphs are safe.
  Keep `MODE=stable` (`--enforce-eager`) only as a fallback if you ever see a decode‑time graph crash.

### The two gotchas that look like bugs but aren't
1. **Reasoning lives in `message.reasoning`, not `reasoning_content`** (vLLM `--reasoning-parser qwen3`).
   An empty `reasoning_content` does **not** mean thinking is off — we briefly believed vLLM wasn't
   reasoning; it was, we were reading the wrong field. (llama.cpp's deepseek parser uses
   `reasoning_content`; vLLM uses `reasoning`.)
2. **`finish_reason:"length"` + empty `content` = it spent the whole budget thinking.** Same verbose‑
   reasoner trap as Path B, just bigger: on the regex task the *reasoning alone* blew past 32K **and**
   56K tokens. Raise `max_tokens` (and don't mistake "still thinking" for "wrong/empty answer").

### Quality vs Q6_K, and the regex reasoning‑loop finding (corrected by a controlled study)
NVFP4's *code* is fine — it converges `eval` in 2 rounds, identical to Q6_K. But on the hardest task (a
backtracking **regex matcher**), vLLM/NVFP4 falls into a **degenerate reasoning loop ~67% of the time**
(measured, N=15 seeds): it re-emits "*Let me use a different approach…*" until it exhausts the budget
(one capture: 127K chars, **91% repeated lines**, the same sentence ×121), never committing to code.
A controlled study (`docs/precision-and-reasoning-loops.md`) pins the cause: it is **specific to the
NVFP4 format + vLLM/Marlin decode — NOT bit-width and NOT context.** Plain 4-bit (Q4_K_M) and 6-bit
(Q6_K) on **llama.cpp loop only ~1/5**, and KV precision (fp8 vs f16) is null. *(An earlier version of
this note guessed it was "context rope, not NVFP4-specific" — the controls falsified that.)* So it's
not "4-bit degrades reasoning"; it's this format+engine combination on SM120. **For single-stream
reliability use llama.cpp Q4_K_M (`docs/optimized-config.md`); use vLLM/NVFP4 for concurrency**, with
loop-detection + retry if it matters.

### Why CUDA graphs matter so much here (26 → 214 tok/s)
At batch=1 every token otherwise pays full kernel‑launch overhead — exactly what CUDA graphs erase.
FP4 tensor cores stay unused (Marlin = 16‑bit compute), and vLLM's batching shines with *concurrency*,
not single streams — yet even single‑stream it now beats llama.cpp once graphs are on.

**Bottom line:** Path A works and is the right call for **concurrency / native tool‑parsing**. But for
**single‑stream daily use it's the wrong pick** — it loops ~67% on the hardest reasoning, and llama.cpp
**Q4_K_M** is both faster (237 vs 214 tok/s) *and* far more reliable (loops ~1/5 full-trace vs vLLM's
67%, though not loop-proof). Use vLLM when serving many parallel agents;
otherwise see `docs/optimized-config.md`.

---

*Original pre‑attempt feasibility assessment (now mostly confirmed) follows.*

**Verdict (pre‑attempt): moderate-to-hard, and it may not work cleanly *today*. The weights are ready; the blocker
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
