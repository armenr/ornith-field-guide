# Observations & Model Behavior Notes

## The family
Ornith-1.0 (DeepReinforce) — a self-scaffolding agentic-coding LLM family, post-trained on
Gemma 4 / Qwen 3.5 with an RL framework where the model learns its own task scaffolds.
- **9B Dense**, **31B Dense**, **35B MoE**, **397B MoE**.
- **The 31B Dense is NOT publicly released** as of this writing. The org has only 9B / 35B / 397B
  (+ GGUF/FP8 variants); searching all of HF found zero `31B` repos. It's likely the Gemma-4-based
  member (the released ones are all Qwen-3.5-based). The HF API returns 401 for the 31B path, but a
  control probe showed 401 is returned for *any* nonexistent path under that org — so the 401 is NOT
  evidence it exists privately. Treat 31B as unreleased.
- Blog: https://deep-reinforce.com/ornith_1_0.html · HF: https://huggingface.co/deepreinforce-ai

## Quality (Q6_K, official settings)

**35B (MoE) — trustworthy.** Produces correct logic with good Rust idioms (uses `arr.swap`, slices,
`Drop`, arena/index or raw-pointer DLLs). On hard problems its *first* attempt may have a few compile
errors, but given the errors it fixes them cleanly in one round. Self-corrects reliably. This is the
one to actually use for coding.

**We *mostly* never found the 35B's ceiling — but there's one genuine failure mode.** Most apparent
failures traced back to *our* configuration, not the model: the early "doom loops" were wrong
temperature; the LRU "regression" and the regex "plateau" were a too-small `max_tokens` truncating its
~30K-token reasoning; one regex "can't converge" was a misleading harness (a function-name collision).
Remove those constraints and it clears the bar, including solving a full backtracking **regex engine**.
**But** a later controlled study (`docs/precision-and-reasoning-loops.md`) found a *real* stochastic
failure on the hardest, most open-ended problems: the model sometimes **fails to commit** at a reasoning
fork and loops *"let me try a different approach"* until it exhausts the budget. On llama.cpp (any
k-quant) that's **~1/5 of runs** — usually cleared by a retry. On **vLLM/NVFP4 it's ~67%** (an
NVFP4-format + vLLM-decode artifact, *not* bit-width). So the "regex in 3 rounds" was one lucky draw,
not a guarantee. Still treat "the model can't do X" with suspicion (check temperature, budget, feedback
first) — but know that on the very hardest problems there *is* an occasional commit-failure loop:
detect it (low reasoning-uniqueness / `finish=length` with empty `content`) and retry with a new seed.

**9B (dense) — capable-looking, frequently wrong.** It writes confident, well-decorated code
(doc comments, `NonNull`, even unit tests) but on hard ownership problems it:
- invents nonexistent std APIs (`LinkedList::move_back_to_front`, `.value()`),
- misuses types (`NonNull` field access without `.as_ref()`), and
- has logic bugs (e.g. a trie whose `search()` can't find an inserted word).
It can *improve* with compiler feedback but oscillates and doesn't converge. Good for trivial code;
verify everything on anything non-trivial. Practical pattern: extract the first ```rust block and
discard its prose (its explanations hallucinate stdlib facts even when the code is fine).

## Behavior quirks
- **Verbose reasoner (this trips people up).** It thinks for **~30,000 tokens** on hard problems
  (~118K characters of `<think>` in one measured turn). Give it `max_tokens` ≥ 32000 and parse
  `reasoning_content` out. If you under-budget it, the code truncates and the model looks far worse
  than it is — see the "no ceiling found" note below.
- **Low temp = doom loops.** See settings.md. The failure signature is repeated "I apologize for the
  repeated errors…" or re-emitting the same skeleton.
- **GPU looks idle during generation** — expected for single-stream MoE decode (memory-latency bound,
  ~2–3B active). Not a misconfiguration.

## Hardware fit for *this* class of machine (RTX 5090 + lots of DDR5)
- 9B and 35B: covered above. **35B Q4_K_M (`-ngl 99`, no offload, `-c 65536`) is the daily driver** —
  fits fully, ~237 tok/s, quality a wash with Q6_K. Q6_K (`--n-cpu-moe`, ~150 tok/s) is the max-fidelity fallback.
- 397B: won't fit VRAM. With "tons of DDR5" (128 GB+) you *could* run it at IQ2_XXS (~106 GB) split
  across GPU+RAM via llama.cpp `-ot`/`--n-cpu-moe`, but expect a few tok/s and degraded quality.
  Not recommended over the 35B for day-to-day.
- Intel vs AMD is irrelevant to the GPU path; it only affects CPU-offloaded experts (DDR5 bandwidth).
  On the *offloaded* paths (Q6_K, 397B) a top-end Intel with comparable DDR5 should perform similarly —
  untested here. The optimized **Q4_K_M path is fully on GPU**, so CPU/RAM choice doesn't affect it.
