# ornith-field-guide

**A measured field guide to running [DeepReinforce's Ornith-1.0](https://deep-reinforce.com/ornith_1_0.html)
(35B / 9B) on a single RTX 5090** — real settings, measured benchmarks, the mistakes that cost us
hours, and a `CLAUDE.md` so your Claude Code knows how to drive it.

Everything here was worked out empirically on one machine; the numbers and recipes are
**measured, not guessed**. Same GPU (or close)? You should be able to reproduce all of it.

> **The 30-second version:** run it at **temp 0.6–1.0** (never low), give it **`max_tokens` ≥ 32K**
> (it's a *very* verbose reasoner), serve the 35B with **`--n-cpu-moe`** (not whole-layer offload),
> and it's a genuinely capable, fully-local coding model that **self-corrects from compiler errors**.
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

# 1. download the 35B (the daily driver) — resumable parallel downloader
scripts/download.sh deepreinforce-ai/Ornith-1.0-35B-GGUF ornith-1.0-35b-Q6_K.gguf

# 2. serve it
scripts/serve-35b.sh            # http://127.0.0.1:8095, ~151 tok/s

# 3. verify
scripts/smoke-test.sh 8095
```
Then talk to it at `http://127.0.0.1:8095/v1/chat/completions` (OpenAI-compatible).
**Run it at temperature 0.6–1.0, top_p 0.95, top_k 20.** That one thing is the difference between
great output and infinite "I apologize for the repeated errors" loops — see `docs/settings.md`.

## Prerequisites
- **GPU driver + CUDA ≥ 12.8** (RTX 5090 is Blackwell / sm_120; older CUDA won't build kernels for it).
- **llama.cpp with CUDA**, recent enough to have `--reasoning-format`, `--reasoning-budget`,
  `--n-cpu-moe`, and `--jinja`. Build from source with `-DGGML_CUDA=ON`, or grab a recent CUDA
  release binary. Point `LLAMA_SERVER` at the resulting `llama-server`.
- (Alternative, simpler but fewer knobs) **ollama ≥ 0.20.3** can run these GGUFs, but it can't gate
  the `<think>` block on a raw import — llama.cpp is preferred. The official path is vLLM ≥ 0.19.1 /
  SGLang ≥ 0.5.9 (needs NVFP4 weights to fit one 32 GB card; GGUF + llama.cpp is the practical route).

## Which model
| Model | Q6_K size | speed (5090) | use it for |
|---|---|---|---|
| **35B MoE** | 28.5 GB | **151 tok/s** | **everything** — correct code, self-corrects from errors |
| 9B Dense | 7.4 GB | ~130 tok/s | trivial code / drafts (verify the rest — it doesn't reliably converge) |
| 397B MoE | 342 GB | n/a | won't fit a 32 GB card |
Download the 9B too if you want the comparison: `scripts/download.sh deepreinforce-ai/Ornith-1.0-9B-GGUF ornith-1.0-9b-Q6_K.gguf`

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
scripts/
  download.sh              # resumable parallel HF downloader (beats throttling/Xet)
  serve-35b.sh             # serve 35B: -ngl 99 --n-cpu-moe 6, the fast/fits config
  serve-9b.sh              # serve 9B fully on GPU
  smoke-test.sh            # health + tok/s + reasoning-split check
  selffix_loop.py          # agentic compile→fix→retry harness (problems: eval, trie, regex)
docs/
  settings.md              # sampling + runtime (temperature + output-budget lessons) — READ THIS
  benchmarks.md            # measured tok/s, VRAM, sizes, self-correction results
  observations.md          # model behavior, quality, "we never found its ceiling", unreleased-31B
  troubleshooting.md       # every wall we hit and the fix
  serving-guide.md         # finalized two-path setup (llama.cpp + vLLM, tool-calling, agents)
  path-a-feasibility.md    # how hard the vLLM/NVFP4 path actually is on a 5090 today
  35b-assessment.md        # the colleague-facing review of the 35B
```

## The six things that took longest to learn (so you don't have to)
1. **Temperature.** 0.6–1.0, never ~0.3. Cold = degenerate loops. (`docs/settings.md`)
2. **Budget its thinking — `max_tokens` ≥ 32K.** It's an extremely verbose reasoner (~30K tokens on
   hard problems). Under-budget it and the code truncates and it *looks* broken. This fooled us into
   declaring a false "ceiling." (`docs/settings.md`, `docs/troubleshooting.md`)
3. **MoE offload by *role*, not by layer.** `--n-cpu-moe 6` (experts→CPU, attention→GPU) = 151 tok/s
   and fits around other GPU usage. Whole-layer `-ngl 34` = 50 tok/s. (`docs/troubleshooting.md`)
4. **Context is cheap here.** Hybrid linear attention → 256K KV is only ~5 GB (measured). Don't let a
   generic "KV is expensive" guide scare you off long context. (`docs/benchmarks.md`)
5. **Downloads:** parallel chunked curl; HF per-IP throttle + Xet stalls otherwise.
6. **Size buys correctness, not polish — and we never found its ceiling.** Both models write pretty
   code; only the 35B is *right* and self-corrects. Every "failure" we saw was *our* config
   (temperature/budget/feedback), not the model. Verify the 9B. (`docs/benchmarks.md`, `docs/observations.md`)
