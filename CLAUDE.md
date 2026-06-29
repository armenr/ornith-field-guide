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
- **35B MoE Q4_K_M (21 GB): THE daily driver.** Fits *fully* on the 32 GB card → run with **no CPU
  offload** → **~237 tok/s** sustained. Quality is a wash with Q6_K (measured). **Use this for real
  work.** (`scripts/serve-q4.sh`; full study `docs/optimized-config.md`.)
- **35B MoE Q6_K (28.5 GB):** max-fidelity fallback. Doesn't fit fully → needs `--n-cpu-moe` → ~150
  tok/s. Same correctness as Q4_K_M; only worth it if you specifically want maximum weight fidelity.
- **9B Q6_K (7.4 GB):** fits fully, ~130 tok/s. Capable fast drafter (strong in Python/Go/TS, self-fixes
  with *actionable* feedback) — **verify it**; trails the 35B on hard Rust (LRU 0/3) and writes buggier
  "passing" code (blind head-to-head: 35B preferred 11/14). See `docs/9b-assessment.md`.
- **397B:** won't fit; skip. The 31B Dense is **unreleased** — don't try to download it.

## Serving (llama.cpp — the daily-driver path)
Set `LLAMA_SERVER` to a CUDA llama-server binary (Blackwell needs CUDA ≥12.8). Then:
```bash
scripts/serve-q4.sh         # ⭐ Q4_K_M, -ngl 99 (NO offload), -c 65536, ~237 tok/s — the optimized default
scripts/serve-35b.sh        # Q6_K, -ngl 99 --n-cpu-moe 6 -c 32768 — max-fidelity fallback (~150 tok/s)
scripts/serve-9b.sh         # 9B, fully on GPU
scripts/smoke-test.sh 8095  # verify + show tok/s
```
- Start the server detached; poll `GET /health` with `curl --retry-connrefused` (NOT shell `sleep`).
- **One model per GPU at a time** — stop one before starting another.
- **Q4_K_M fits fully → `-ngl 99` with NO `--n-cpu-moe`** (offload only HURTS a model that already
  fits). Q6_K (28.5 GB) *doesn't* fit, so it needs `--n-cpu-moe 6` (experts→CPU, attention→GPU).
  **Never** use whole-layer `-ngl 34` (the ~50 tok/s trap).

## Context window (big codebases)
- **Native max is 256K (`n_ctx_train = 262144`) — the real ceiling.** Needle recall verified usable at
  200K and 250K; fits at full f16 KV (~30.6 GB, KV only ~5 GB). `serve-q4.sh` defaults to `-c 262144
  -np 1` (one request gets the full 256K).
- **>256K does NOT work** — sectioned rope (`rope type 40`) rejects YaRN/linear extension; llama.cpp
  caps the slot at 262144 regardless of `-c`. Don't promise 500K–1M. (`docs/context-window.md`)
- One 5090's KV budget (~256K f16) is shared: `-np 1` = full 256K to one request; `-np N` splits it.
  1M-orchestrator + big sub-agents at once is multi-GPU territory, not one card.

## Code quality by language (Q4 vs Q6)
Across Rust/Python/Go/TS, **Q4 ≈ Q6 (a wash)** — 4-bit costs nothing measurable. Python/Go/TS are
near-perfect first-try; **Rust is the hard one** (borrow checker → more compile→fix rounds, either
quant). (`docs/quant-by-language.md`)

## Serving (Path A — vLLM + NVFP4 in Docker; for concurrency / native tool-parsing)
Zero host installs beyond Docker + nvidia-container-toolkit. **Verified on the 5090 (SM120).**
```bash
export MODEL_DIR=$HOME/models/ornith-nvfp4   # model.safetensors + chat_template.jinja
docker compose up -d            # or: scripts/serve-vllm-nvfp4.sh   (MODE=fast default)
scripts/smoke-vllm.sh           # proves it answers + THINKS + reports tok/s
```
- **MODE=fast (default) keeps CUDA graphs → ~214 tok/s** — faster than offload-forced Q6_K (~150), but
  *slower* than the Q4_K_M daily driver (~237). Use vLLM for **concurrency**, not single-stream speed.
  Only fall back to `MODE=stable` (`--enforce-eager`, ~26 tok/s) if you hit a CUDA-graph crash/hang.
- **Force Marlin (W4A16)** via the env vars in the script — native NVFP4-MoE kernels crash on SM120.
- VRAM: `--gpu-memory-utilization 0.75 --max-num-seqs 1 --kv-cache-dtype fp8` (the OOM fix).
- **Caveat (UPDATED 2026-06-30): the old "vLLM/NVFP4 loops ~67%" was mostly a bad-checkpoint +
  forced-Marlin + stale-container artifact, not "NVFP4 is bad."** A current vLLM nightly auto-uses native
  FlashInfer-CUTLASS (no Marlin forcing), and a *properly-exported* NVFP4 (W4A16 MLP-only, attn+GatedDeltaNet
  kept BF16 — e.g. AEON-7's export) loops ~25% ≈ llama.cpp's intrinsic ~1/5 floor. Our own `ornith-nvfp4`
  export is low-quality (trips vLLM's `reduced accuracy` fused-scale warning, #36094). Still: for
  single-stream **simplicity/speed** use llama.cpp **Q4_K_M**; vLLM's reason to exist is **concurrency**.
  Serve clean NVFP4 with `--mamba-cache-dtype float32`. Full re-probe: `docs/precision-and-reasoning-loops.md` UPDATE.

## Querying (OpenAI-compatible)
```bash
curl -s http://127.0.0.1:8095/v1/chat/completions -d '{
  "messages":[{"role":"user","content":"…"}],
  "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"max_tokens":12000
}'
```
- Final answer → `choices[0].message.content`. Chain-of-thought field name **depends on the server**:
  - **llama.cpp** (`--reasoning-format deepseek`) → `message.reasoning_content`.
  - **vLLM** (`--reasoning-parser qwen3`) → `message.reasoning`. ← easy to miss; an empty
    `reasoning_content` on vLLM does **not** mean thinking is off — check `reasoning`.
  - If you ever see `<think>` inside `content`, the reasoning parser isn't on.
- **`finish_reason:"length"` with empty `content` = the model used the whole budget *thinking* and never
  emitted the answer.** Raise `max_tokens` (the regex task alone needed >32K thinking tokens). This is
  the single most common "it returned nothing / it's broken" cause on hard problems.
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
Use it to confirm a model actually solves a task. **Give the harness *actionable* feedback** (the
failing input + expected + actual, like `scripts/multilang-battery.py` / `scripts/capture-solutions.py`
do) — a bare `panic`/`assert` makes the model spiral. **Run single-stream** (`-np 1`): concurrent
batched decode isn't batch-invariant and corrupts per-seed results. Empirically the 35B converges
(often 1 fix round); the 9B converges on most problems too but fails hard Rust (LRU) and spirals on
regex — `docs/9b-assessment.md`. And convergence ⇏ correctness: keep an adversarial check (passing code
can still be spec-wrong).

## Reflexes / gotchas (see docs/troubleshooting.md for all)
- Low/idle GPU utilization during generation is **normal** for single-stream MoE decode (latency-bound).
- ollama can't gate thinking on a raw GGUF — prefer llama.cpp.
- Intel vs AMD only affects CPU-offloaded expert speed (DDR5 bandwidth); the GPU path is identical.

## Pointers
- `README.md` — human setup walkthrough (incl. using Ornith as your editor's model via a proxy).
- `docs/settings.md` — sampling/runtime details · `docs/benchmarks.md` — all numbers ·
  `docs/observations.md` — model behavior · `docs/troubleshooting.md` — every failure + fix.
