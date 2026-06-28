# Run Settings — the part that actually matters

> The single biggest lesson of this whole exercise: **Ornith is a reasoning model RL-tuned for
> high temperature. Run it at the official sampling, give it room to think, and parse the
> reasoning out. Run it cold (low temp) or starve its thinking and it degenerates.**

## Recommended sampling (from the official model card / `generation_config.json`)

| Source | temperature | top_p | top_k |
|---|---|---|---|
| `generation_config.json` (baked-in default) | **1.0** | **0.95** | **20** |
| Transformers quickstart (model card) | **0.6** | 0.95 | 20 |
| Benchmarks (SWE-bench, Terminal-Bench) | 1.0 | 0.95–1.0 | — |
| ClawEval (real-user tasks) | 0.6 | — | — |

**Use temp 0.6 for crisp/deterministic code, up to 1.0 for exploration. top_p 0.95, top_k 20.**
We standardized on **temp 0.6 / top_p 0.95 / top_k 20 / min_p 0**, no DRY, no repeat penalty.

### What NOT to do (learned the hard way)
- **Do not run at temp 0.2–0.3.** A model tuned for 0.6–1.0 degenerates at low temp: it gets
  stuck in self-correction loops ("I apologize for the repeated errors. Here is the complete
  correct implementation:" … repeated ~30× verbatim until the token cap).
- **Do not suppress or tightly cap thinking.** With `enable_thinking:false` it "thinks in the
  answer" and spirals; with a too-small `--reasoning-budget` (e.g. 3000) it gets cut off
  mid-design and emits truncated code. Let it think; give it budget.
- **Do not add aggressive `repeat_penalty`/DRY to fix loops.** Those are wrong-tool band-aids
  for a sampling problem; at the correct temperature the loops don't happen.

## It is a reasoning model

The assistant turn opens with a `<think> … </think>` block, then the final answer.
- **vLLM/SGLang:** `--reasoning-parser qwen3` → reasoning returned in `reasoning_content`.
- **llama.cpp:** `--jinja --reasoning-format deepseek` → same effect (`reasoning_content` vs `content`).
- **Manual:** split the decoded text on `</think>`.
- Give a **generous output budget — this is critical, second only to temperature.** On hard
  problems it reasons for **~30,000 tokens** (measured: ~118K characters of `<think>` in a single
  turn). Set **`max_tokens` ≥ 32000** and **`-c` ≥ 65536**, or its answer truncates mid-code and
  *looks* broken — which is not the model failing, it's you clamping it. We initially mistook a
  too-small `max_tokens` (14000) for the model "hitting a ceiling"; it wasn't. Benchmarks use
  128K–400K context, up to 131K output.

## Required runtimes (official)
- Transformers ≥ 5.8.1 · vLLM ≥ 0.19.1 · SGLang ≥ 0.5.9
- llama.cpp: any recent build with `--reasoning-format`, `--n-cpu-moe`, and Blackwell (sm_120) CUDA.
- ollama ≥ 0.20.3 understands the arch (`Qwen3_5MoeForConditionalGeneration`) but **cannot gate the
  `<think>` block on a raw-GGUF import** ("does not support thinking" error) — prefer llama.cpp.

## Example request (OpenAI-compatible)
```bash
curl -s http://127.0.0.1:8095/v1/chat/completions -d '{
  "messages":[{"role":"user","content":"…"}],
  "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"max_tokens":12000
}'
# answer is in choices[0].message.content ; chain-of-thought in choices[0].message.reasoning_content
```
