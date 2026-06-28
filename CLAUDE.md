# CLAUDE.md — Operating Ornith-1.0 locally

You are working in a project for running **Ornith-1.0** (DeepReinforce's agentic-coding LLM family)
locally on an **RTX 5090 (32 GB)**. This file is the distilled operational knowledge from a prior
session that worked all of this out. Trust it; it will save you the dead-ends documented in `docs/`.

## The one rule that matters most
Ornith is a **reasoning model RL-tuned for high temperature**. **Always run it at temp 0.6–1.0,
top_p 0.95, top_k 20**, with thinking ENABLED and a generous token budget. Running it cold
(temp ≤0.3) or starving its `<think>` block makes it degenerate into infinite self-correction
loops or truncated/empty answers. This was the cause of every "the model is broken" moment — it
wasn't the model, it was the settings.

## What fits this hardware
- **9B Q6_K (7.4 GB):** fits fully on GPU, ~130 tok/s. Fine for trivial code; verify anything hard
  (it writes confident-but-wrong code and can't reliably self-correct).
- **35B MoE Q6_K (28.5 GB): the daily driver.** ~2–3B active params/token → ~151 tok/s. Logically
  correct code, self-corrects from compiler errors in ~1 round. **Use this for real work.**
- **397B:** won't fit; skip unless experimenting with IQ2 in RAM.
- The 31B Dense is **unreleased** — don't try to download it.

## Serving (llama.cpp — preferred; it has the knobs)
Set `LLAMA_SERVER` to a CUDA llama-server binary (Blackwell needs CUDA ≥12.8). Then:
```bash
scripts/serve-35b.sh        # -ngl 99 --n-cpu-moe 6 -c 32768 -fa on --jinja --reasoning-format deepseek --reasoning-budget -1
scripts/serve-9b.sh         # -ngl 99 ...
scripts/smoke-test.sh 8095  # verify + show tok/s
```
- Start the server detached; poll `GET /health` with `curl --retry-connrefused` (NOT shell `sleep`).
- **One model per GPU at a time** (9B + 35B won't co-reside in 32 GB) — stop one before starting the other.
- `--n-cpu-moe 6` is why the 35B fits around other GPU usage at full speed; keep attention on GPU,
  experts on CPU. **Never** use whole-layer `-ngl 34` (that's the ~50 tok/s trap).

## Querying (OpenAI-compatible)
```bash
curl -s http://127.0.0.1:8095/v1/chat/completions -d '{
  "messages":[{"role":"user","content":"…"}],
  "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"max_tokens":12000
}'
```
- Final answer → `choices[0].message.content`. Chain-of-thought → `choices[0].message.reasoning_content`
  (because the server runs `--reasoning-format deepseek`). If you ever see `<think>` inside `content`,
  the reasoning parser isn't on.
- **Give big `max_tokens` (≥ 32000) and `-c` ≥ 65536.** It's an extremely verbose reasoner
  (~30,000 thinking tokens on hard problems). Too-small a budget truncates the answer and makes the
  model *look* broken — this is the #2 mistake after wrong temperature. Empty `content` +
  `finish_reason:length` = it ran out mid-think → raise `max_tokens`, keep temp 0.6.

## Downloading models
HF is throttled/Xet-flaky here. Use the resumable parallel downloader:
```bash
scripts/download.sh deepreinforce-ai/Ornith-1.0-35B-GGUF ornith-1.0-35b-Q6_K.gguf
scripts/download.sh deepreinforce-ai/Ornith-1.0-9B-GGUF  ornith-1.0-9b-Q6_K.gguf
```

## Verifying the model's code (don't trust, test)
`scripts/selffix_loop.py` is an agentic harness: it asks the model for Rust, compiles it, runs a
behavioral test, and feeds `rustc` errors back over multiple rounds.
```bash
python3 scripts/selffix_loop.py <port> <label> <max_iters> <problem>   # problems: eval | trie
```
Use it to confirm a model actually solves a task. Empirically the 35B converges (often 1 fix round);
the 9B oscillates and doesn't.

## Reflexes / gotchas (see docs/troubleshooting.md for all)
- Low/idle GPU utilization during generation is **normal** for single-stream MoE decode (latency-bound).
- ollama can't gate thinking on a raw GGUF — prefer llama.cpp.
- Intel vs AMD only affects CPU-offloaded expert speed (DDR5 bandwidth); the GPU path is identical.

## Pointers
- `README.md` — human setup walkthrough (incl. using Ornith as your editor's model via a proxy).
- `docs/settings.md` — sampling/runtime details · `docs/benchmarks.md` — all numbers ·
  `docs/observations.md` — model behavior · `docs/troubleshooting.md` — every failure + fix.
