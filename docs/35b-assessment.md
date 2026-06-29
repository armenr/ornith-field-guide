# Ornith-1.0-35B — Hands-on Coding Assessment

A short, evidence-based review of the **35B MoE** variant as a local coding model. Every result
below was **compiled and behavior-tested** (Rust, `rustc 1.94`), not eyeballed. Run on a single
RTX 5090 (Q6_K GGUF here; **Q4_K_M is the faster, quality-equivalent serving choice** — see
`docs/optimized-config.md`) at the model's official sampling (temp 0.6, top_p 0.95, top_k 20).

## What it is
35B-parameter Mixture-of-Experts (256 experts, ~8 active → only **~2–3B params active per token**),
so it runs fast on one 5090 despite its size (~150 tok/s at Q6_K; **~237 tok/s at Q4_K_M**, which fits
fully on the card). It's a
**reasoning model** (emits a `<think>` block, then the answer).

## Why these tasks
We used a deliberate difficulty ladder in Rust — chosen because Rust's borrow checker punishes
"plausible but wrong" code, so it's a strong correctness filter — and split testing into two modes:
**one-shot** (can it write it?) and **agentic self-correction** (can it debug its own code when fed
compiler errors?). The second mode matters more: real coding is iterative against a compiler/tests.

| # | Task | Why we chose it |
|---|------|-----------------|
| 1 | **Bubble sort** | Trivial baseline — confirm correctness, idioms, and natural behavior. |
| 2 | **LRU cache, O(1)** | The canonical "Rust is hard" problem; the obvious linked-list design fights ownership/aliasing. Tests judgment, not just algorithms. |
| 3 | **LRU via self-fix loop** | Can it reason about its *own* compiler errors and converge? |
| 4 | **Expression evaluator** (tokenizer + precedence + parens + unary minus + errors), self-fix loop | A harder, multi-part problem — more to get right and more to debug. |
| 5 | **Trie / prefix tree** | A *fresh* task on a different axis (tree ownership/traversal, not aliasing) to rule out memorized solutions. |

## How it performed

| Task | Result |
|------|--------|
| Bubble sort | ✅ Correct & idiomatic first try (generic `T: Ord`, early-exit, doc comments). Compiled + passed. |
| LRU (one-shot) | ⚠️→✅ Complete, coherent, **logically correct** (raw-pointer DLL + HashMap + `Drop`). Had **2 trivial compile errors** (an `unsafe`-block omission; `Option`-vs-raw-pointer). A **2-line fix → compiled and passed all eviction tests** (recency, value-update, correct LRU eviction). |
| LRU (self-fix loop) | ✅ **Converged in 2 iterations** — fed the 2 errors, fixed both, passed. |
| Expression evaluator (self-fix loop) | ✅ First attempt had **9** compile errors (harder problem); fed the errors, it **fixed all nine in a single round** and passed every case (precedence, parens, unary minus, div-by-zero, malformed input). |
| Trie | ✅ **One-shot** — compiled and passed all tests on the first try. |
| Regex engine (backtracking: literals, `.`, `*`/`+`/`?`, groups, alternation, full-match) | ✅ **Converged in 3 rounds** with adequate output budget. Passed every case, including the backtracking ones (`(ab)+`, `a*`, `.*z`). *See the correction note below — this one first looked like a failure and wasn't.* |

## Assessment

**Strengths**
- **Correct logic on hard ownership problems.** It got the LRU's eviction/recency semantics right
  and chose sound architectures (raw-pointer DLL, arena/index). The gap from "passing" was syntax,
  not design.
- **Reliable self-correction — the headline.** Given exact compiler errors it converges fast,
  routinely fixing *all* errors in one round rather than oscillating. This is the property that
  makes a model useful in an agentic loop.
- **Idiomatic and well-documented** Rust; sensible API choices.

**Caveats (use it right)**
- **First-pass code often has a few trivial compile errors** on hard problems — it's an
  iterate-to-correct collaborator, not a one-shot oracle. Pair it with a compile/test loop.
- **Verbose reasoner** — it thinks for thousands–tens-of-thousands of tokens; give it budget and
  parse the chain-of-thought out (`reasoning_content`).
- **Sensitive to sampling** — must run at temp 0.6–1.0; at low temp it degenerates into loops. (Run
  it the way it was tuned; see `settings.md`.)
- Minor: occasional imprecise stdlib asides in prose (e.g. mischaracterizing `Vec::sort` internals).

**For context:** the same family's **9B** could *not* converge on these tasks — it improved with
feedback but oscillated and shipped broken code (e.g., a trie whose `search()` couldn't find an
inserted word). So this performance is specific to the 35B; the extra parameters buy *correctness
and convergence*, not just nicer-looking code.

## Correction worth reading (we got this wrong first)
The regex engine *initially* looked like the model's ceiling — it appeared to plateau at ~20 compile
errors and oscillate without converging. **That conclusion was wrong, and the cause is instructive:**
- We had `max_tokens` set to 14000, but this model **reasons for ~30,000 tokens** on a hard problem —
  so its code was being **truncated mid-output** and the "errors" were partly an artifact of the cut-off.
- A separate test-harness flaw (the model named its function `regex_matches`, colliding with Rust's
  built-in `matches!` macro) produced compiler errors that pointed at *test* code the model couldn't
  edit — feedback it literally couldn't act on.

Fix the budget (≥32K tokens) and the feedback, and it **converges in 3 rounds** and passes everything.
**Net: across all our tasks we never actually found a hard problem the 35B couldn't self-correct to
passing — every "failure" was something *we* did wrong (temperature, output budget, or feedback
quality).** Take any "the model can't do X" claim (including our earlier ones) with skepticism until
those three are verified.

## Bottom line
A trustworthy local coding model for a single 32 GB GPU **when used in a loop** with a compiler/tests
in the harness — *and run with the right settings* (temp 0.6–1.0, `max_tokens` ≥ 32K, actionable
feedback). It writes correct logic, takes feedback well, and self-corrects to passing on genuinely
hard problems — including ones it didn't nail on the first try. Its biggest operational gotcha is that
it's a *very* verbose reasoner: under-budget it and it truncates into looking broken. Give it room,
keep a build/test step in the loop, and it's a capable, fully-local pair-programmer.
