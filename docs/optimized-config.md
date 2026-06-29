# The optimized config (and the data behind it)

*Measured 2026-06-29 on RTX 5090 (32 GB) + 9950X3D + 128 GB DDR5. The full controlled study is in
`docs/precision-and-reasoning-loops.md`; this file is the decision.*

## TL;DR — the daily driver
```bash
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
scripts/download.sh deepreinforce-ai/Ornith-1.0-35B-GGUF ornith-1.0-35b-Q4_K_M.gguf
scripts/serve-q4.sh     # llama.cpp + Q4_K_M, -ngl 99 (NO offload), f16 KV, -c 65536, :8095
```
Query at temp **0.6–1.0 / top_p 0.95 / top_k 20 / min_p 0 / max_tokens ≥ 32K**. Reasoning lands in
`choices[0].message.reasoning_content`.

## The three contenders, measured

| | vLLM + NVFP4 | llama.cpp + Q6_K | **llama.cpp + Q4_K_M** |
|---|---|---|---|
| **Sustained decode** | 214–232 tok/s | ~150 tok/s | **237–242 tok/s** ✅ |
| **Reasoning-loop rate** | 67%→**~1/5**† | ~1/5 (occasional) | ~1/5 (occasional) |
| **Correctness battery** (eval+trie) | eval ties | 8/10 | **10/10** |
| **Size on disk / VRAM** | 21 GB | 28.5 GB | **21 GB** |
| **Fits fully on the 5090?** | yes | **no** (needs `--n-cpu-moe`) | **yes** ✅ |
| **Best for** | concurrency / many agents | max weight fidelity | **single-stream daily use** |

† **UPDATE 2026-06-30:** the vLLM "67%" was *our* low-quality NVFP4 export on forced-Marlin + a stale
container — a re-probe with a current nightly (native FlashInfer-CUTLASS) + a *properly-exported* NVFP4
(W4A16 MLP-only) loops ~25% ≈ llama.cpp's ~1/5 floor. So Q4_K_M still wins on **speed + simplicity** (it's
faster and needs no Docker/quant-quality care), **not** because "NVFP4 loops." See
`docs/precision-and-reasoning-loops.md` UPDATE.

## Why Q4_K_M wins — three findings

1. **Quality is a wash vs Q6_K.** They trade wins seed-to-seed (Q4 won the battery 10/10 vs 8/10; Q6
   emitted more regex solutions in the window; eval tied). At n=5 with large run-to-run variance there's
   **no measurable quality gap** — consistent with the textbook k-quant result that Q4_K_M sits right at
   the quality/size knee.

2. **Q4_K_M is the *fastest* option — faster than vLLM.** Not because 4-bit decode is special, but
   because **21 GB fits entirely on the 32 GB card** (25.8 GB at `-c 32768`, 26.5 GB at `-c 65536`), so
   it runs with **zero CPU offload** → 237 tok/s sustained (242 warm), holding at depth. Q6_K (28.5 GB)
   *cannot* fit fully alongside KV + your desktop, so it's forced onto `--n-cpu-moe` (experts in RAM over
   DDR5) → ~150 tok/s. vLLM's 214 is real but comes with the 67% loop.

3. **The loop is an NVFP4+vLLM artifact, not a quant-precision problem.** Clean controls: fp8-vs-f16 KV
   is null, and 4-bit (Q4_K) vs 6-bit (Q6_K) on llama.cpp is a wash — only NVFP4-on-vLLM blows up to
   67%. So choosing a k-quant sidesteps the *severe* NVFP4/vLLM loop (67% → the residual ~1/5 gradual loop
   both k-quants still show); choosing Q4 over Q6 costs nothing.

## The only operational difference between Q4 and Q6
Beyond quant/size, exactly **one** thing differs in how you run them:
- **Q4_K_M:** `-ngl 99`, **no** `--n-cpu-moe` — whole model on GPU.
- **Q6_K:** must add `--n-cpu-moe N` to fit (some expert tensors live in RAM, on DDR5 bandwidth).

Everything else is identical: KV cache **type** (f16 on both), KV **amount** per token (same
architecture), sampling, flash-attention, reasoning format. **Context window is a free knob** — not
dictated by quant; Q4's smaller footprint simply affords a larger `-c` before you OOM.

## When to pick each
- **Q4_K_M (default):** single-user agentic coding. Fastest, fits fully, quality = Q6_K.
- **Q6_K:** you want maximum weight fidelity (perplexity/KL is marginally closer to fp16) and don't mind
  ~1.6× slower + the offload. Quality difference is below what our tests could resolve.
- **vLLM + NVFP4:** serving **many concurrent** agents (vLLM's batching is its reason to exist). Accept
  the single-stream loop risk, or add loop-detection + retry. See `scripts/serve-vllm-nvfp4.sh` /
  `docker-compose.yml`.

## Caveats (so you're not surprised)
- **No config is loop-proof on the hardest open-ended problems.** Even Q4/Q6 loop ~1/5 on regex-class
  tasks — the model can get verbose and fail to commit. It's stochastic; **retry with a new seed.**
- Numbers are from one model on one machine; n=5 on the battery. Treat the *ranking* as solid and the
  *exact* per-quant quality deltas as "within noise."
