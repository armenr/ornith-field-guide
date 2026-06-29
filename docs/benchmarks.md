# Measurements & Benchmarks (RTX 5090, 32 GB)

All numbers measured on a single RTX 5090 (32 GB) across llama.cpp (Q4_K_M / Q6_K GGUF) and vLLM
(NVFP4), at the official sampling (temp 0.6 / top_p 0.95 / top_k 20). Your hardware (same GPU, top-end
Intel, lots of DDR5) should match the GPU-bound numbers closely; the CPU-offload numbers depend on DDR5
bandwidth (and the recommended Q4_K_M path uses no offload at all).

## Throughput (single-stream decode)

| Model / config | tok/s | notes |
|---|---|---|
| **35B Q4_K_M, `-ngl 99` (full GPU, NO offload)** | **237–242** | **⭐ recommended daily driver**; 21 GB fits fully (~26 GB w/ KV @ -c 65536) |
| 35B Q4_K_M, `-ngl 99 --n-cpu-moe 6` | ~145–200 | offload *hurts* a model that already fits — don't |
| 35B Q6_K, `-ngl 99 --n-cpu-moe 6` | ~150 | max-fidelity fallback; 28.5 GB can't fit fully → offload forced |
| 35B Q6_K, `-ngl 99` (full GPU, *truly empty* card) | ~144 | only fits fully with NO desktop using VRAM; with a ~3 GB desktop it doesn't fit → use `--n-cpu-moe` |
| 35B Q6_K, `-ngl 34` (whole-layer offload) | ~50 | ❌ avoid — drops attention to CPU |
| 9B Q6_K, `-ngl 99` (full GPU) | ~130–134 | fits in ~9.5 GB |
| vLLM NVFP4 (Marlin, CUDA graphs) | 214 / 232 (fp8 / f16 KV) | concurrency path; but 67% single-stream reasoning loop |

The 35B (MoE) is *faster* than the 9B (dense) despite being ~4× larger: only ~2–3B of its 35B
params are active per token. That's also why **GPU utilization looks low during generation**
(~20–27% SM, ~100 W of 550 W) — single-stream decode is memory-latency bound, not compute bound.
To actually saturate the card you need concurrency/batching (vLLM continuous batching) or a long
prompt prefill.

## Model sizes (Q6_K unless noted)

| | 9B (dense) | 35B (MoE) | 397B (MoE) |
|---|---|---|---|
| arch | `qwen3_5` | `qwen3_5_moe`, 256 experts / 8 active | `qwen3_5_moe`, 512 experts / 10 active |
| active params/token | 9B | ~2–3B | ~12–15B |
| BF16 | 18.8 GB | 70.2 GB | 793.6 GB |
| **Q6_K** | **7.4 GB** | **28.5 GB** | 342 GB |
| Q4_K_M | 5.6 GB | 21.2 GB | 242 GB |
| Q8_0 | 9.5 GB | 36.9 GB | ~435 GB |
| fits one 5090? | ✅ easily | ✅ at Q4–Q6 (**Q4_K_M recommended** — fits fully, no offload) | ❌ (needs big RAM, IQ2 ≈ 106–140 GB) |

GGUF quant ladders exist from `deepreinforce-ai/*-GGUF` (official) and `bartowski/*` (imatrix).
Hybrid linear attention (full attention only every 4th layer) makes the KV cache cheap, so large
context (`-c 32768`+) is affordable even with partial offload.

## Published benchmark scores (from the blog, for reference)

| Benchmark | 9B | 35B | 397B | Claude Opus 4.7 |
|---|---|---|---|---|
| Terminal-Bench 2.1 (Terminus-2) | 43.1 | 64.2 | 77.5 | 70.3 |
| SWE-Bench Verified | 69.4 | 75.6 | 82.4 | 80.8 |

The 397B roughly matches Claude Opus 4.7 on these; the 35B is the sweet spot for a single 5090.

## Self-correction loop (agentic debugging ability)

Loop: model writes Rust → we compile + run a behavioral test → feed exact `rustc` errors back →
repeat. Identical settings for both models.

| Problem | 9B | 35B |
|---|---|---|
| Trie (insert/search/starts_with) | 4→4→**compiles but search() broken** ❌ | **pass one-shot** ✅ |
| LRU cache (O(1), aliasing-heavy) | 20→10→9→3→**9 (regressed)** ❌ never converged | 2 errors → **pass** ✅ (2 iters) |
| Expression evaluator (precedence/parens/unary) | — | 9 errors → **pass** ✅ (2 iters) |
| Regex engine (backtracking, groups, alternation, full-match) | — | **pass** ✅ (3 iters) — *with adequate budget* |

> **Important correction.** The regex engine *first appeared* to be beyond the 35B's reach (it
> "plateaued at ~20 errors, whack-a-mole, never converged"). That was wrong — two **harness/config
> artifacts**, not a model limit: (1) our `max_tokens` was 14000, but the model reasons ~30,000
> tokens on this problem, so its code was being truncated; (2) a function-name collision made the
> compiler errors reference test code the model couldn't edit. Fix the budget (≥32K) and the
> feedback, and it **converges in 3 rounds**, passing every case including the backtracking ones
> (`(ab)+`, `a*`, `.*z`) that the broken runs failed. **We never actually found the 35B's ceiling on
> these tasks.**

> **9B column superseded (2026-06-29).** The 9B results in the table above were produced with the
> *same* broken methodology we later caught: concurrent batched decode (`-np 4`, not batch-invariant)
> and bare-`panic` feedback the model couldn't localize. Re-run single-stream with actionable feedback,
> the 9B **self-corrects the trie 3/3** (the "search() broken" cell above) and the evaluator, and is
> strong in Python/Go/TS — but still fails **Rust LRU 0/3** and writes buggier code than the 35B in a
> blind head-to-head (35B preferred on **11/14**). Full corrected numbers: **`docs/9b-assessment.md`**.

**Takeaway:** the 35B genuinely reasons about compiler feedback and converges — but *only if you
don't starve it*. Give it the right temperature (0.6–1.0), enough output budget (≥32K tokens), and
actionable feedback. Under those conditions it self-corrects hard problems to passing. The 9B, under
the *same* corrected conditions, is more capable than first reported (it self-corrects the trie and the
evaluator, and is strong in GC'd languages) but still trails the 35B on hard ownership/algorithmic
problems and on code correctness — a feedback loop *amplifies* capability, it doesn't create it. Trust
the 35B in agentic/iterative loops; treat the 9B's output as a draft to verify (`docs/9b-assessment.md`).

> **Note (2026-06-29 study):** the regex "converged in 3" was a *single lucky draw*, not a reliable
> property — regex convergence is stochastic (the model can stay verbose and fail to commit ~1/5 of
> runs on *any* config). See the loop study below and `docs/precision-and-reasoning-loops.md`.

## Quant × engine: reasoning-loop rate & the optimized config (2026-06-29)

Controlled study (`docs/precision-and-reasoning-loops.md`). Metric = reasoning-loop rate on the hardest
self-fix task (backtracking regex), **N=15 seeds**, loop = reasoning-uniqueness < 0.40.

| config | engine | quant | KV | loop-rate | 95% CI (Wilson) |
|---|---|---|---|---|---|
| NVFP4 (fp8 KV) | vLLM | NVFP4 | fp8 | **67%** | [42%, 85%] |
| NVFP4 (f16 KV) | vLLM | NVFP4 | f16 | **67%** | [42%, 85%] |
| Q6_K | llama.cpp | Q6_K | f16 | 7%* | [1%, 30%] |
| Q4_K_M | llama.cpp | Q4_K_M | f16 | 0%* | [0%, 20%] |

\* 16K-window lower bound — a full-trace (56K) analysis shows both k-quants loop **~1/5** (some loops
develop gradually past 16K). **KV precision is null** (fp8 == f16); **4-bit vs 6-bit is a wash**. The
67% is an **NVFP4-format + vLLM/Marlin-decode artifact on SM120, not bit-width** (clean controls).
vLLM is also *non-deterministic* here (same seed → uniq 0.23/0.42/0.47), so loop-*rate* is the statistic.

**Code-correctness battery** (eval+trie, 5 seeds, compiles **and** passes tests): **Q4_K_M 10/10**,
**Q6_K 8/10**, NVFP4 ties on `eval` (2 rounds) → **a wash within variance.** Quality is indistinguishable
across quants; pick by speed/fit → **Q4_K_M** (fastest, fits fully). See `docs/optimized-config.md`.
