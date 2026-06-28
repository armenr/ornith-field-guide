# Measurements & Benchmarks (RTX 5090, 32 GB)

All numbers measured on a single RTX 5090 (32 GB), Q6_K GGUF via llama.cpp, official sampling
(temp 0.6 / top_p 0.95 / top_k 20). Your hardware (same GPU, top-end Intel, lots of DDR5) should
match the GPU-bound numbers closely; the CPU-offload numbers depend on DDR5 bandwidth.

## Throughput (single-stream decode)

| Model / config | tok/s | notes |
|---|---|---|
| 9B Q6_K, `-ngl 99` (full GPU) | ~130–134 | fits in ~9.5 GB |
| 35B Q6_K, `-ngl 99 --n-cpu-moe 6` | **151** | **recommended**; fits ~25 GB, leaves room for other apps |
| 35B Q6_K, `-ngl 99` (full GPU, empty card) | ~144 | needs the card mostly free |
| 35B Q6_K, `-ngl 34` (whole-layer offload) | ~50 | ❌ avoid — drops attention to CPU |

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
| fits one 5090? | ✅ easily | ✅ at Q4–Q6 (Q6_K recommended) | ❌ (needs big RAM, IQ2 ≈ 106–140 GB) |

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

**Takeaway:** the 35B genuinely reasons about compiler feedback and converges — but *only if you
don't starve it*. Give it the right temperature (0.6–1.0), enough output budget (≥32K tokens), and
actionable feedback. Under those conditions it self-corrects hard problems to passing. The 9B, by
contrast, improves locally but oscillates/regresses and can't land — a feedback loop *amplifies*
capability, it doesn't create it. Trust the 35B in agentic/iterative loops; treat the 9B's output as
a draft to verify.
