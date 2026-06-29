# ornith-field-guide

**A measured field guide to running [DeepReinforce's Ornith-1.0](https://deep-reinforce.com/ornith_1_0.html)
(35B / 9B) on a single RTX 5090** — real settings, measured benchmarks, the mistakes that cost us
hours, and a `CLAUDE.md` so your Claude Code knows how to drive it.

Everything here was worked out empirically on one machine; the numbers and recipes are
**measured, not guessed**. Same GPU (or close)? You should be able to reproduce all of it.

> **The 30-second version:** run it at **temp 0.6–1.0** (never low), give it **`max_tokens` ≥ 32K**
> (it's a *very* verbose reasoner), and serve the 35B as **Q4_K_M on llama.cpp, fully on the GPU**
> (21 GB fits the 32 GB card with no CPU offload → ~237 tok/s, and 4-bit costs nothing measurable vs
> Q6_K). It's a genuinely capable, fully-local coding model that **self-corrects from compiler errors**.
> Most "it's broken" moments are config, not the model. Details below.

*Captured 2026-06-28. Models: https://huggingface.co/deepreinforce-ai · MIT licensed.*

## Who this is for (hardware)
Tested on: **RTX 5090 (32 GB)** + AMD 9950X3D + 128 GB DDR5. It will transfer near-identically to:
- **Same GPU (RTX 5090, 32 GB)** — all VRAM math, quant choices, and offload flags apply verbatim.
- **Top-end Intel instead of AMD** — irrelevant to the GPU path; the CPU only does the offloaded
  expert tensors, gated by DDR5 bandwidth (fast Intel + DDR5 matches or beats the numbers here).
- **Lots of DDR5** — comfortable headroom for the `--n-cpu-moe` offload; enough to experiment with
  the 397B at low quant in RAM if you ever want (not recommended over the 35B).

## TL;DR
```bash
# 0. prereq: a CUDA-enabled llama.cpp `llama-server` (Blackwell/sm_120 -> CUDA >= 12.8). See below.
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server

# 1. download the 35B in Q4_K_M (the optimized daily driver — fits FULLY on a 32GB card)
scripts/download.sh deepreinforce-ai/Ornith-1.0-35B-GGUF ornith-1.0-35b-Q4_K_M.gguf

# 2. serve it — fully on GPU, no CPU offload
scripts/serve-q4.sh             # http://127.0.0.1:8095, ~237 tok/s

# 3. verify
scripts/smoke-test.sh 8095
```
Then talk to it at `http://127.0.0.1:8095/v1/chat/completions` (OpenAI-compatible).
**Run it at temperature 0.6–1.0, top_p 0.95, top_k 20.** That one thing is the difference between
great output and infinite "I apologize for the repeated errors" loops — see `docs/settings.md`.

## How to run it — and the optimized config
Three real options, all measured. **For single-user coding the winner is Q4_K_M on llama.cpp, fully on
the GPU** (`scripts/serve-q4.sh`) — fastest *and* quality-equivalent to the bigger quant:

| | **Q4_K_M · llama.cpp** ⭐ | **Q6_K · llama.cpp** | **NVFP4 · vLLM (Docker)** |
|---|---|---|---|
| Speed (1 stream) | **~237 tok/s** | ~150 tok/s | 214–232 tok/s |
| Fits fully on GPU? | **yes** (21 GB, no offload) | no (28.5 GB, needs `--n-cpu-moe`) | yes (21 GB) |
| Reasoning-loop rate | low (~1/5 hardest) | low (~1/5 hardest) | **67%** on hardest tasks |
| Quality | baseline (= Q6_K) | = Q4_K_M (a wash) | code OK, but loops single-stream |
| Best for | **single-stream daily use** | max weight fidelity | **concurrency / many agents** |

The short version: Q4_K_M fits fully on the card so it runs with **zero CPU offload** (the speed
unlock), and the 4-bit-vs-6-bit quality difference is below what we could measure. The vLLM "reasoning
loop" is an **NVFP4-format + vLLM-decode artifact, not a bit-width problem** — proven with clean
controls. Full study: `docs/optimized-config.md` and `docs/precision-and-reasoning-loops.md`.

**Concurrency path — vLLM/NVFP4 in one command** (Docker + nvidia-container-toolkit; ~21 GB in `$MODEL_DIR`):
```bash
export MODEL_DIR=$HOME/models/ornith-nvfp4   # model.safetensors + chat_template.jinja
docker compose up -d                          # ~214 tok/s; great for many parallel agents
scripts/smoke-vllm.sh
```
On vLLM the chain-of-thought is in `message.reasoning` (llama.cpp uses `message.reasoning_content`).
Use it for serving a team / many concurrent requests; for single-stream reliability prefer Q4_K_M above.
Details + the SM120/Marlin story: `docs/path-a-feasibility.md`.

## Prerequisites
- **GPU driver + CUDA ≥ 12.8** (RTX 5090 is Blackwell / sm_120; older CUDA won't build kernels for it).
- **llama.cpp with CUDA**, recent enough to have `--reasoning-format`, `--reasoning-budget`,
  `--n-cpu-moe`, and `--jinja`. Build from source with `-DGGML_CUDA=ON`, or grab a recent CUDA
  release binary. Point `LLAMA_SERVER` at the resulting `llama-server`.
- (Alternative, simpler but fewer knobs) **ollama ≥ 0.20.3** can run these GGUFs, but it can't gate
  the `<think>` block on a raw import — llama.cpp is preferred. The official path is vLLM ≥ 0.19.1 /
  SGLang ≥ 0.5.9 (needs NVFP4 weights to fit one 32 GB card; GGUF + llama.cpp is the practical route).

## Which model
| Model | recommended quant | size | speed (5090) | use it for |
|---|---|---|---|---|
| **35B MoE** | **Q4_K_M** | 21 GB | **~237 tok/s** | **everything** — correct code, self-corrects; fits fully on GPU |
| 9B Dense | Q6_K | 7.4 GB | ~130 tok/s | trivial code / drafts (verify the rest — it doesn't reliably converge) |
| 397B MoE | — | 342 GB | n/a | won't fit a 32 GB card |
The 35B in **Q4_K_M** is the optimized daily driver (`docs/optimized-config.md`). Q6_K (28.5 GB) is a
max-fidelity fallback that needs `--n-cpu-moe` to fit; quality vs Q4_K_M is a wash. Download the 9B for
comparison: `scripts/download.sh deepreinforce-ai/Ornith-1.0-9B-GGUF ornith-1.0-9b-Q6_K.gguf`

## Letting Claude Code use this
**Option A — Claude Code as the *operator* (recommended, zero extra infra).**
Open Claude Code in this folder. It reads `CLAUDE.md` automatically and instantly knows how to
download, serve (right flags), query, and verify Ornith — i.e. it can drive the local model for you
and avoid every pitfall documented here. This is the "their claude ends up knowing all of it" path.

**Option B — Ornith as Claude Code's *backing model* (advanced).**
Claude Code speaks the Anthropic API; `llama-server` speaks OpenAI. Put a translating proxy between
them, then point Claude Code at the proxy:
```
Claude Code ──ANTHROPIC_BASE_URL──▶ proxy (Anthropic⇄OpenAI) ──▶ llama-server (Ornith) :8095
```
Use **LiteLLM** (Anthropic-compatible passthrough) or **claude-code-router** as the proxy, configured
to forward to `http://127.0.0.1:8095/v1` with sampling temp 0.6 / top_p 0.95 / top_k 20. Set Claude
Code's `ANTHROPIC_BASE_URL` (and a dummy auth token) to the proxy. **Verify exact env-var names and
proxy flags against the current Claude Code + proxy docs** — these change; the architecture above is
the stable part. Note: a 32 GB single-stream local model is far slower/smaller than hosted Claude, so
this is best for offline/private work, not as a daily Claude replacement.

## What's in here
```
CLAUDE.md                  # operational brain — Claude Code reads this automatically
README.md                  # this file
docker-compose.yml         # vLLM + NVFP4 in one command (concurrency path): `docker compose up -d`
scripts/
  download.sh              # resumable parallel HF downloader (beats throttling/Xet)
  serve-q4.sh              # ⭐ OPTIMIZED daily driver: Q4_K_M, -ngl 99 (no offload), -c 65536, ~237 tok/s
  serve-35b.sh             # Q6_K (llama.cpp): -ngl 99 --n-cpu-moe 6 — max-fidelity fallback
  serve-9b.sh              # serve 9B fully on GPU
  serve-vllm-nvfp4.sh      # vLLM + NVFP4 in Docker (concurrency) — MODE=fast|stable, KV_DTYPE toggle
  smoke-test.sh            # llama.cpp: health + tok/s + reasoning-split check
  smoke-vllm.sh            # vLLM: answers + THINKS (reasoning field) + tok/s
  selffix_loop.py          # agentic compile→fix→retry harness (problems: eval, trie, regex)
  loop-rate-sweep.py       # N-seed reasoning-loop rate + Wilson CI (the study harness)
  loop-window-analysis.py  # prefix-16K vs full-trace uniqueness (catches gradual loops)
  correctness-battery.py   # eval/trie self-fix convergence rate + rounds, across seeds
  multilang-battery.py     # Q4-vs-Q6 convergence across Rust/Python/Go/TS (real compile+test)
  seed-sweep-regex.py      # quick per-seed uniqueness sweep vs any server
  needle-test.py           # long-context needle-in-haystack recall test
  probe-logits.py          # next-token entropy/margin at decision forks (RCA)
  kl-sweep.py              # per-position top-20 logprobs along a shared trajectory (RCA)
  compare-kl.py            # top-1 agreement + JS divergence between two engines (RCA)
docs/
  optimized-config.md      # ⭐ the daily-driver recommendation + the data behind it
  context-window.md        # how much context is real (256K native; why >256K doesn't work)
  quant-by-language.md     # Q4 vs Q6 across Rust/Python/Go/TS (a wash; Rust is hardest)
  precision-and-reasoning-loops.md  # controlled study: why NVFP4-on-vLLM loops (it's not bit-width)
  vllm-rca.md              # logit-level RCA: the loop is compounding drift, NOT logit-flattening
  settings.md              # sampling + runtime (temperature + output-budget lessons) — READ THIS
  benchmarks.md            # measured tok/s, VRAM, sizes, self-correction + the quant study
  observations.md          # model behavior, quality, "we never found its ceiling", unreleased-31B
  troubleshooting.md       # every wall we hit and the fix
  serving-guide.md         # finalized setup (llama.cpp + vLLM, tool-calling, agents)
  path-a-feasibility.md    # how hard the vLLM/NVFP4 path actually is on a 5090 today
  35b-assessment.md        # the colleague-facing review of the 35B
```

## The seven things that took longest to learn (so you don't have to)
1. **Temperature.** 0.6–1.0, never ~0.3. Cold = degenerate loops. (`docs/settings.md`)
2. **Budget its thinking — `max_tokens` ≥ 32K.** It's an extremely verbose reasoner (~30K tokens on
   hard problems). Under-budget it and the code truncates and it *looks* broken. This fooled us into
   declaring a false "ceiling." (`docs/settings.md`, `docs/troubleshooting.md`)
3. **Quant = VRAM fit, not quality.** **Q4_K_M (21 GB) fits *fully* → `-ngl 99`, no `--n-cpu-moe` →
   ~237 tok/s.** Q6_K (28.5 GB) doesn't fit, so it needs `--n-cpu-moe` (experts→CPU) and runs ~150 —
   for the *same* quality. Never use whole-layer `-ngl 34` (~50 tok/s). (`docs/optimized-config.md`)
4. **Context is cheap here.** Hybrid linear attention (full attention every 4th layer) keeps KV small —
   256K context ≈ 5 GB of KV, and the Q4 daily driver at `-c 65536` fits in ~26 GB total. Don't let a
   generic "KV is expensive" guide scare you off long context. (`docs/benchmarks.md`)
5. **Downloads:** parallel chunked curl; HF per-IP throttle + Xet stalls otherwise.
6. **Size buys correctness, not polish — and we never found its ceiling.** Both models write pretty
   code; only the 35B is *right* and self-corrects. Every "failure" we saw was *our* config
   (temperature/budget/feedback), not the model. Verify the 9B. (`docs/benchmarks.md`, `docs/observations.md`)
7. **The vLLM "reasoning loop" is an NVFP4+vLLM artifact — not 4-bit, not the model.** vLLM/NVFP4 loops
   *sharply* ~67% on the hardest reasoning; plain 4-bit (Q4_K_M) on llama.cpp loops far less (~1/5
   full-trace, and only gradually) — clean controls. Use llama.cpp for single-stream, vLLM for
   concurrency. (`docs/precision-and-reasoning-loops.md`)
