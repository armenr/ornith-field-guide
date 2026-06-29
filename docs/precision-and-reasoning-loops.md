# Why NVFP4-on-vLLM "loops" on hard reasoning — a controlled root-cause study

*Measured 2026-06-29 on one RTX 5090 (32 GB). Variables isolated, a falsifiable mechanism, and an
honest accounting — including the intermediate claims that turned out wrong.*

## The symptom
On the hardest self-fix task (a backtracking **regex matcher**), vLLM serving the **NVFP4** 35B
frequently falls into a **degenerate reasoning loop**: it re-emits *"Let me use a different approach.
I'll use a recursive backtracking matcher…"* dozens of times, burns the whole token budget thinking,
never closes `</think>`, and emits no code. llama.cpp serving a **k-quant** (Q4_K_M / Q6_K) mostly
doesn't. The question: *why*, and is it the quantization?

It is **not** a code-quality failure — it's a failure to **commit** at a reasoning fork. On the easier
`eval` task NVFP4 converges in 2 rounds, identical to the k-quants.

## What's SOLID

### 1. It's a repetition/degeneration loop (confirmed, not inferred)
A `frequency_penalty` on vLLM breaks it: `finish_reason` flips from `length` (stuck) to `stop`, and
reasoning-uniqueness jumps to ~1.0 (seed 11: 0.72→0.94→1.0 at fp=0/0.3/0.6; seed 23: 0.14→1.0).
(It's a *diagnostic*, not a fix — a global penalty also distorts code, so penalized runs emit no usable
program.)

### 2. vLLM is NOT seed-deterministic here
Same seed, same config, same single instance, three consecutive runs of seed 7 → uniqueness **0.23 /
0.42 / 0.47**. MoE expert routing + FP4/Marlin GPU reductions are non-deterministic, so a fixed `seed`
does not pin the trajectory (σ ≈ ±0.1). **Consequence: single-sample-per-seed numbers are noise; the
loop *rate* over many seeds is the only meaningful statistic.** (This is what sank our first n=1/seed pass.)

### 3. The clean controls: it's NOT precision/bit-width — it's the NVFP4+vLLM path
Loop-rate over **N=15 seeds** (regex, `max_tokens=16000`, loop = reasoning-uniqueness < 0.40):

| config | engine | quant | KV | loop-rate | 95% CI (Wilson) |
|---|---|---|---|---|---|
| A | vLLM | NVFP4 | fp8 | **67%** (10/15) | [42%, 85%] |
| B | vLLM | NVFP4 | f16 | **67%** (10/15) | [42%, 85%] |
| C | llama.cpp | Q6_K | f16 | **7%** (1/15) | [1%, 30%] |
| D | llama.cpp | Q4_K_M | f16 | **0%** (0/15) | [0%, 20%] |

- **KV-precision axis (A vs B): NULL.** fp8 → f16 KV changes nothing (67% == 67%). (Our earlier n=5
  "clean KV effect" was sampling noise.)
- **Weight-bit axis on llama.cpp (C vs D, plus the battery below): NULL / wash.** 4-bit ≈ 6-bit.
- **{A,B} ≫ {C,D} with non-overlapping CIs.** The huge effect is the **engine+format**, not the bits.

So **"4-bit" is not the problem** — plain 4-bit k-quant (Q4_K_M) on llama.cpp is the *cleanest* config.
The 67% is specific to the **NVFP4 format (per-block fp4 scales) decoded via vLLM's Marlin W4A16 path on
consumer Blackwell (SM120)** — a bleeding-edge combination. NVFP4 ≠ "4-bit quality."

## The methodological catch: the 16K window under-counts *gradual* loops
A full-trace analysis (`max_tokens=56000`, uniqueness over a 16K-token **prefix** vs the **full** trace)
shows most loops are visible by 16K, but some llama loops develop *gradually*:

| | Q4_K_M | Q6_K |
|---|---|---|
| full-trace loops (5 seeds) | 1/5 (seed 4: prefix 0.48 → full **0.17**) | 1/5 (seed 7: prefix 0.58 → full **0.35**, "late-loop") |
| code *emitted* in trace (not compile-tested) | 2/5 | 4/5 |

Takeaways: (a) the llama loop-rates in the table above are **lower bounds** (16K misses gradual loops);
vLLM's loops are *sharp* (visible by 16K), so its 67% is unaffected. (b) **Both k-quants loop ~1/5
full-trace — comparable.** Neither is loop-proof. (c) llama.cpp also has run-to-run variance (seed 7
solved here but budget-outed elsewhere).

## Correctness is a wash between Q4 and Q6
Self-fix convergence battery (eval + trie, 5 seeds each, compiles **and** passes behavioral tests):

| problem | Q4_K_M | Q6_K |
|---|---|---|
| eval | 5/5 (avg 1.6 rounds) | 5/5 (avg 1.4) |
| trie | 5/5 (avg 1.6) | 3/5 (seed1 no-converge, seed5 budget) |
| **total** | **10/10** | **8/10** |

Q4 *edged* Q6 here — the opposite of the regex-window result. Net across all tests: **statistically
indistinguishable at this sample size.** 4-bit k-quant costs nothing measurable on real code.

## The mechanism (first principles)
A degenerate *semantic* repetition (whole sentences re-drawn, not token stutter) is the signature of a
**flattened next-token distribution failing to break symmetry** at a "commit vs reconsider" reasoning
fork. When decode noise is comparable to the logit margin, sampling keeps re-drawing "reconsider." This
predicts everything observed: hardest/most-open-ended task only (eval/trie have crisp forks → no loop);
probabilistic, not on/off; a repetition penalty breaks it. **But the clean controls prove generic
precision is *not* the source here** (KV null, Q4≈Q6 null) — for this model it's the NVFP4-format +
vLLM-Marlin decode that produces the symptom, not bit-width.

## Honesty trail — claims we made and then falsified with more data
1. *"Precision dose-response, both axes matter"* (n=1/seed) → **wrong, noise.**
2. *"Clean KV-axis effect 0.26→0.39"* (n=5) → **wrong at N=15** (A==B; KV null).
3. *"Q4 = 0% loop"* → **too strong** (full-trace ≈ 1/5; 16K under-counts gradual loops).
4. *"Q6 is more decisive (regex 4/5 vs 2/5)"* → **not robust** (battery flipped it: Q4 10/10 vs Q6 8/10). Net wash.
5. Standing conclusion: **engine+format dominates (NVFP4-vLLM bad); Q4 vs Q6 is a wash.**

## Practical guidance
- **Use llama.cpp + a k-quant for single-stream reliability.** NVFP4-on-vLLM loops ~67% on the hardest
  open-ended tasks; reserve it for **concurrency** (its real strength).
- **No llama config is loop-proof** — both Q4 and Q6 loop ~1/5 on regex-class problems. It's stochastic;
  **a retry usually fixes it.** A self-fix harness should *detect* the loop (low reasoning-uniqueness /
  `finish=length` with empty `content`) and **retry with a new seed**, not feed the truncation back.
- See `docs/optimized-config.md` for the full Q4-vs-Q6-vs-vLLM decision and the recommended config.

## Reproduce
- `scripts/loop-rate-sweep.py PORT MODEL LABEL N SEED0` — N-seed loop-rate + Wilson CI.
- `scripts/loop-window-analysis.py PORT MODEL [seeds…]` — prefix-16K vs full-trace uniqueness.
- `scripts/correctness-battery.py PORT MODEL LABEL "eval,trie" "1,2,3,4,5"` — convergence rate + rounds.
- `scripts/seed-sweep-regex.py PORT MODEL [seeds…]` — quick uniqueness sweep (reads vLLM `reasoning` or
  llama `reasoning_content`). Flip the KV axis with `KV_DTYPE=auto|fp8` on `scripts/serve-vllm-nvfp4.sh`.

## Caveats on confidence
n=5 on the battery (high variance); one model (Ornith-35B, a hybrid-MoE); literature context below is
from general knowledge, not live-cited. **Q4≈Q6 is the textbook k-quant result** (Q4_K_M is the
community sweet spot; perplexity/KL show Q6 *measurably* closer to fp16, just below task-level
resolution, and quant error may compound on long chain-of-thought — a place we had low power to detect a
gap). The **out-of-ordinary** result is NVFP4-on-vLLM's 67%, not Q4≈Q6.
