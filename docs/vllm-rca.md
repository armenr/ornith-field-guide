# vLLM-NVFP4 reasoning loop: the logit-level RCA

*Measured 2026-06-29, RTX 5090. The deeper "why" behind `docs/precision-and-reasoning-loops.md`.
Reproduce with `scripts/probe-logits.py` and `scripts/kl-sweep.py` + `scripts/compare-kl.py`.*

> **⚠️ UPDATE (2026-06-30): the *rate* was inflated by a bad checkpoint; this RCA's *mechanism* survives.**
> A re-probe (current vLLM nightly, native FlashInfer-CUTLASS, + a properly-exported NVFP4 that keeps
> attention/GatedDeltaNet in BF16) cut the loop 67% → ~25% ≈ llama.cpp's ~1/5 floor. So "67% = the
> NVFP4+vLLM path is bad" was an overstatement — much of it was a stale-container + forced-Marlin +
> low-quality-export stack (our export trips vLLM's `reduced accuracy` fused-scale warning, #36094). **But
> the core finding here is VINDICATED:** the loop is a **quant-independent, intrinsic ~1/5 commit-failure**
> at flat forks — exactly the ~20% floor a *clean* NVFP4 still shows. See the UPDATE in
> `docs/precision-and-reasoning-loops.md`. Directional (small N, vLLM nondeterministic).

## Question
vLLM-NVFP4 falls into a reasoning loop ~67% of the time on the hardest task; llama.cpp-Q4 ~1/5. The
loop-rate study proved it's the NVFP4+vLLM *path*, not bit-width. This asks the next level down: **what,
at the logit level, makes that path loop?**

## Hypothesis (and how it died)
Obvious hypothesis: *NVFP4 flattens the next-token distribution at "commit-vs-reconsider" forks, so
sampling can't break symmetry and keeps re-drawing "reconsider."* We probed it directly: feed the
**same** decision-point prefixes to both engines via raw `/v1/completions`, pull the top-20 logprobs,
compare entropy / top-1 margin. Control prefixes have a near-deterministic next token (to prove any
effect is fork-specific).

**Result — hypothesis REFUTED:**

| next-token metric | llama-Q4 | vLLM-NVFP4 |
|---|---|---|
| decision-pts: mean top-1 | 0.304 | 0.336 |
| decision-pts: mean margin | 0.134 | 0.142 |
| decision-pts: mean **entropy** | **2.244** | **2.119** |
| controls: mean entropy | 0.609 | 0.810 |

The fork distributions are **nearly identical** — if anything NVFP4 is *marginally more* decisive. The
controls validate the probe (e.g. "recursive back" → "tracking" is top-1 **1.000**, entropy **0.004** on
both). **NVFP4 does not flatten the forks.** Note the forks are inherently flat for *both* (entropy ~2.1
vs ~0.6 at controls) — that's why Ornith is loop-*prone* on hard reasoning at all, but it's quant-independent.

## The actual mechanism: compounding trajectory divergence
We then measured how much the two engines' **full distributions** differ across a real 4420-char
reasoning trajectory — 24 positions, top-1 agreement + Jensen-Shannon divergence (bound ln2 = 0.693):

- **top-1 agreement: 18/24 = 75%**
- **JS divergence: mean 0.073, median 0.081, max 0.279**

So per-token the engines are *close but not identical*: they pick the same most-likely token only ~75%
of the time (argmax disagreement ~1-in-4), and the full distributions diverge **modestly** per token
(mean JS 0.073 — ~10% of the ln2=0.693 bound — occasionally up to 0.28). **Over a ~30,000-token
reasoning chain, those small, frequent differences compound** — the two walk different paths, and the NVFP4-vLLM
path lands in the degenerate loop far more often. The divergences are not a directional bias toward
"reconsider" tokens; they're general drift that occasionally derails onto a loop.

## What this explains
- **Stochastic** (67% not 100%): trajectory drift is probabilistic, not a deterministic cliff.
- **Only the longest task** (regex, ~30K-token reasoning): short chains (eval) don't accumulate enough
  drift to derail; long chains do.
- **Code quality unaffected** (Q4≈Q6, eval ties): per-token the model is equally good (75% identical
  argmax, equal entropy) — it writes correct code, it just sometimes *wanders while thinking*.
- **Not "4-bit = bad":** it's two near-equal distributions diverging over a long horizon, not a quality collapse.

## Caveats (honest scope)
- The JS divergence **bundles** the quant (NVFP4 vs Q4_K) and the engine/sampler (vLLM/Marlin numerics +
  sampler order vs llama.cpp). We can't separate them on one GPU (can't run NVFP4 on llama.cpp or Q4_K on
  vLLM). So this localizes the cause to *the NVFP4+vLLM path's per-token divergence from the k-quant+llama
  path*, not to one component.
- n is modest (8 fork prefixes, 24 trajectory positions, one trajectory). The *direction* (no flattening;
  ~75% agreement; compounding drift) is clear; exact divergence magnitudes are indicative.
- To fully isolate quant-vs-engine you'd need a matched-format run on both engines (e.g. an AWQ/GPTQ both
  support, or NVFP4 weights on a second vLLM-vs-SGLang comparison) — future work.
