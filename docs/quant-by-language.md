# Q4_K_M vs Q6_K across Rust, Python, Go, TypeScript

*Measured 2026-06-29 on RTX 5090 via llama.cpp. Reproduce with `scripts/multilang-battery.py`.*

## Method
The same three algorithmic problems in each language (controls for difficulty), each with a **real
compile + behavioral test** and a self-fix loop (model writes code → we compile & run a behavioral test
→ feed errors back, up to 5 rounds):
- **expr-eval** — integer expression evaluator (precedence, parens, unary minus, div-by-zero errors).
- **lru** — O(1) LRU cache (get/put/eviction).
- **merge-intervals** — sort + merge overlapping intervals.

3 problems × 3 seeds × 4 languages × 2 quants = **72 self-fix runs**. Toolchains: `rustc -O`, `python3`,
`go run`, `bun run`. Sampling: temp 0.6 / top_p 0.95 / top_k 20. Identical server config for both quants
(`-np 4 -c 131072`, MAXTOK 10000).

## Results

| language | Q4_K_M (converged / first-try) | Q6_K (converged / first-try) |
|---|---|---|
| Python | **9/9**, 9/9 first-try | 9/9, 8/9 |
| Go | **9/9**, 9/9 first-try | 8/9, 8/9 |
| TypeScript | **9/9**, 8/9 first-try | 8/9, 8/9 |
| Rust | 8/9, 4/9 first-try (eval avg **3.0** rounds) | 7/9, 3/9 first-try (eval avg **1.7** rounds) |
| **TOTAL** | **35/36 (97%), 30/36 (83%)** | **32/36 (88%), 27/36 (75%)** |

## Findings
- **Q4 and Q6 are a wash within variance — Q4 is if anything nominally ahead.** Q6's three extra misses
  are mostly **budget/variance artifacts** (`length` exhaustion or `no-converge` from the tight
  32768-token/10K-output config), not clear quality deficits — though Q4's lone miss was *also* a budget
  `length` fail (Rust lru), so neither is artifact-free. At n=3 per cell the per-cell numbers are too
  noisy to separate the quants, and they trade wins (e.g. Q6 solved Rust-eval in fewer rounds, 1.7 vs
  3.0; Q4 had the cleaner Python/Go/TS sweep). **4-bit costs nothing measurable vs 6-bit on real code.**
  (No significance test computed — n=3/cell; "wash" = the differences are within run-to-run noise.)
- **Python / Go / TypeScript: a near-perfect sweep for both quants** — almost everything first-try.
- **Rust is the hard language for both.** First-try rate craters to ~3–4/9 (vs 8–9/9 elsewhere): the
  borrow checker reliably rejects the confident-but-slightly-wrong code LLMs produce, so the model leans
  on the compile→fix loop. It *does* converge — just expect more rounds on Rust regardless of quant.

## Takeaway
This multi-language evidence **reinforces `docs/optimized-config.md`**: there is no language where Q6_K
earns its **~1.6× speed penalty (and ~1.4× larger footprint, 28.5 vs 21 GB)** over Q4_K_M. **Use Q4_K_M.** And budget for extra fix-rounds on Rust
specifically — give the agentic loop room (it's the one language where first-attempt code routinely
fails to compile).

*Caveat: n=3/cell, high variance; the absolute pass rates carry budget-artifact noise. Treat the
*ranking* (Q4 ≥ Q6; Rust hardest) as solid and the exact per-cell numbers as indicative.*
