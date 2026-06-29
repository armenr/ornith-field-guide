# Ornith-1.0-9B — a fair re-assessment, and a blind 9B↔35B head-to-head

An earlier note in this repo called the 9B *"capable-looking, frequently wrong, oscillates, doesn't
converge."* That verdict was formed **before** we learned how to run and test these models. This is the
re-do, under corrected methodology, with a **blind code-quality head-to-head** against the 35B.

Every number here was produced on a single RTX 5090 (32 GB), llama.cpp, **single-stream** (`-np 1`), at
the official sampling (temp 0.6 / top_p 0.95 / top_k 20), 9B Q6_K vs 35B Q4_K_M. The 9B is the
`qwen3_5` dense member (8.95 B params; GGUF `general.architecture = qwen35`; hybrid
full-attention-every-4th-layer + Gated-Delta-Net linear attention, `n_ctx_train = 262144`).

## TL;DR

- **The old verdict was unfair.** Two methodology bugs — not the model — produced most of the "9B can't
  converge" signal. Fixed, the 9B **self-corrects the very trie test that originally damned it** (3/3),
  and nails the expression-evaluator and interval-merge across languages.
- **But the gap to the 35B is real and specific.** It concentrates on hard *ownership/algorithmic*
  problems (Rust LRU **0/3** vs 35B 3/3; regex spirals) and — the part pass/fail can't see — on **code
  quality**: blind reviewers preferred the 35B's code on **11 of 14** tasks, and adversarial
  break-testers found **more defects in the 9B's code (24 vs 16)** — including bugs in code that *passed
  our behavioral tests*.
- **The extra parameters buy correctness, not cosmetics.** Two of the 9B's "passing" solutions are
  provably spec-violating (verified by hand); the 35B's are correct.
- **The 9B does use the full 256 K context** (needle recall HIT at 128 K and 200 K; at 250 K the code
  is recalled in the reasoning trace but the 3 000-token answer budget truncated the final line).

## Why the original verdict was unfair — two confounds

We re-ran the *identical* battery the 35B was tested on and watched it produce nonsense for the 9B (a
cell that passed at one token budget *failed* at a larger one — impossible if the run were
deterministic and budget-bound). Two harness/serving bugs were unfairly sinking the 9B (and would sink
*any* model):

1. **Concurrent batched decode is not batch-invariant.** Running the battery with the server at `-np 4`
   (four languages in parallel) means a sequence's logits depend on which other sequences share its
   decode batch. At temp 0.6 those tiny differences fork into different trajectories — the same
   "compounding trajectory divergence" we documented for vLLM, but here induced *on llama.cpp by
   concurrency itself*. Per-seed results stopped being reproducible. **Fix: single-stream (`-np 1`).**

2. **Bare-panic test feedback gave the model nothing to act on.** The original harnesses failed with a
   generic `panic("eval mismatch")` / `assert` — no failing input, no expected-vs-actual. Asked to fix
   a bug it couldn't localize, the 9B would *spiral*: on one go/eval cell it reasoned **157,641
   characters** ("*I'm really struggling to find the issue… everything looks correct… Wait… No,*")
   without emitting a line of code, then gave up and re-emitted the same buggy program. **Fix: harnesses
   now report the first failing input + expected + actual**, exactly what a real test runner (pytest,
   `cargo test`, `go test`) gives.

**Proof the fixes matter:** the *same* go/eval cell — same model, same seed — that spiraled for 40 000
tokens and gave up under bare feedback **converges in 2 rounds** once the feedback names the failing
input. The corrected harness is `scripts/multilang-battery.py` / `scripts/capture-solutions.py`.

> Caveat we are honest about: these convergence numbers use **rich feedback + single-stream**. The
> 35B's old 35/36 (`docs/quant-by-language.md`) was produced under the *pre-correction* setup —
> concurrent `-np 4` decode (not batch-invariant) and the bare-feedback harness — so the head-to-head
> below **re-runs the 35B in-session under the identical corrected conditions** rather than citing that
> number.

## Self-fix convergence: 9B vs 35B (corrected methodology)

Loop: model writes code → we compile + run a behavioral test → feed the **actionable** failure back →
repeat (≤ 6 rounds). PASS = compiles **and** passes the test.

**Rust** (3 seeds each; Rust is the hardest language — the borrow checker punishes plausible-but-wrong code):

| problem | 9B | 35B |
|---|---|---|
| expression evaluator | **3/3** | 3/3 |
| interval merge | 3/3 | 3/3 |
| trie (prefix tree) | **3/3** ⭐ | 3/3 |
| LRU cache (O(1), aliasing) | **0/3** ❌ | **3/3** ✅ |
| regex engine (backtracking) | 0/1 † | 1/3 |

⭐ The trie is the *exact* task the original note used to condemn the 9B ("a trie whose `search()`
can't find an inserted word"). With fair conditions it self-corrects it every time.
† The 9B regex was sampled at one seed only: each attempt spirals to the full budget for ~6 rounds
(~25 min/seed), so we stopped at n=1. The spiral *is* the finding. Note even the **35B** only lands
regex 1/3 — backtracking-regex-in-Rust is near the ceiling for both, and the 35B's documented
"converged in 3 rounds" was a lucky single draw (we flagged this previously).

**Python / Go / TypeScript** (seed 1; the GC'd languages, no borrow checker):

| | 9B | 35B |
|---|---|---|
| Python (eval/lru/intervals) | 3/3 | 3/3 |
| Go (eval/lru/intervals) | 2/3 (eval failed) | 3/3 |
| TypeScript (eval/lru/intervals) | 3/3 | 3/3 |

The 9B is **strong in GC'd languages** — mostly first-try. Its convergence failures cluster on Rust
ownership (LRU) and on the multi-part expression evaluator.

## Blind code-quality head-to-head — what pass/fail can't measure

Convergence only asks "did it compile and pass a handful of assertions?" A solution can pass a small
test and still be wrong. So we ran a **blind** review: for each of 14 (language × problem) instances we
took both models' final solution, labeled them **A / B in randomized order**, and gave each pair to a
panel of **3 independent reviewers + 2 adversarial break-testers** (Opus, high effort). The reviewers
were told only *"grade these two LLM-generated solutions"* — **no model identity, no sizes, no hint
which should win, nothing about this project.** We held the A/B↔model map secret and revealed it only at
tally time. (70 agents total; harness: `scripts/capture-solutions.py` + a blind-judge workflow.)

| metric | 35B | 9B |
|---|---|---|
| instances won (of 14) | **11** | 2 (+ 1 tie) |
| mean blind quality (sum of 4 dims, /40) | **36.1** | 27.3 |
| distinct defects found in its code | 16 | **24** |

- **Decisive 35B wins** (3–0 votes, large gaps) were exactly where the 9B's code was *actually broken*:
  Rust eval, Rust LRU, Rust regex, Go eval, TS LRU.
- **Near-ties** (both write good code): interval-merge in every language, the trie, Python/Go LRU. The
  9B's two outright wins (Rust & TS interval-merge) were narrow — 1.0 and 1.7 points on the /40 scale.

### Convergence ≠ correctness — verified by hand

The most important result: **two of the 9B solutions that *passed our behavioral test* are provably
spec-violating.** A break-tester flagged them; we then compiled the actual captured code and confirmed:

| spec requirement | input | 9B (passed our test) | 35B |
|---|---|---|---|
| "arbitrary whitespace" (Rust eval) | `eval("2 * 3")` | **`Err(...)`** ❌ | `6` ✅ |
| "truncating integer division" (Python eval) | `evaluate("7/(-2)")` | **`-4`** (used floor `//`) ❌ | `-3` ✅ |

Our harness happened to test only positive divisions and never put a space directly before `*`/`/`, so
the bugs slipped through; the adversarial reviewer found them. (A third reviewer flag — the 9B's TS LRU
fails `tsc` type-checking via a duplicate `size` identifier — we could not independently reproduce
because the harness runs TypeScript under `bun`, which does not type-check; reported as judge-found.)
This is the whole point of the quality layer: **the 9B's *passing* code carries more latent bugs than
the 35B's.**

> Judge calibration caveat: on Python eval the reviewers scored the two nearly equal (30.0 vs 30.3)
> despite the 9B's verified truncating-division bug, i.e. they slightly under-weighted it — though the
> *vote* still went 3–0 to the 35B. Treat per-instance scores as directional, n=3 reviewers.

## Long context

The 9B's hybrid attention makes its KV cache cheap (~32 KiB/token on the 8 full-attention layers + a
fixed recurrent state), so the full trained 256 K fits in ~8 GiB of KV. Needle-in-haystack recall:

| context | depth | result |
|---|---|---|
| 128 K | 50% | **HIT** |
| 200 K | 50% | **HIT** |
| 250 K | 10% (lost-in-middle) | **HIT** (code recalled in the reasoning; the 3 000-token answer budget truncated the final line — raise it) |

Same 256 K ceiling and sectioned-rope behavior as the 35B (`docs/context-window.md`): usable to 256 K,
no clean extension beyond.

## How the two models think differently

On the identical go/eval problem, single-stream:

| | reasoning emitted | outcome |
|---|---|---|
| **9B** (a failing run) | **157,641 chars**, then 0 chars of code | spiral → gave up → re-emitted the same bug |
| **35B** | **14,144 chars** | committed to correct code, first try |

The 9B isn't *dumber* token-for-token so much as **less decisive** — it struggles to commit at a
reasoning fork and burns its budget re-deriving, especially when it can't localize a bug. The 35B
commits. (Both are verbose reasoners; budget them ≥ 32 K tokens.)

## Bottom line

The 9B is **far better than the original verdict** — a genuinely capable model that self-corrects with
good feedback, is strong in Python/Go/TS, and uses the full 256 K context. The earlier "frequently
wrong, can't converge" call was mostly **our** broken methodology (concurrent batching + useless test
feedback), the same class of mistake that produced every earlier "the model is broken" moment in this
repo.

But the 35B is still the one to trust for real work: it wins the blind quality head-to-head **11/14**,
solves the hard ownership/algorithmic problems the 9B can't (Rust LRU), and — critically — its
*passing* code is actually correct where the 9B's harbors latent spec violations. **Use the 9B as a
fast drafter (especially in GC'd languages) and verify it; reach for the 35B Q4_K_M daily driver for
anything you need to be right.**

## Reproduce

```bash
scripts/serve-9b.sh                 # 9B Q6_K, single-stream
# convergence + code capture (rich feedback, single-stream):
python3 scripts/capture-solutions.py <port> 9b <outdir> rust,python,go,ts eval,lru,intervals,trie,regex 1,2,3 6
python3 scripts/multilang-battery.py <port> 9b 9b rust,python,go,ts 1,2,3 6   # convergence-only
python3 scripts/needle-test.py <port> 200000 0.5                              # long-context recall
```
The blind judge (anonymized A/B, panel + adversarial break-test) and its aggregation are research
harnesses kept with the investigation record. Convergence ⇏ correctness — keep an adversarial check in
the loop.
