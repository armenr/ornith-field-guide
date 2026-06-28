# Serving Ornith-1.0-35B on a single RTX 5090 — the finalized guide

A corrected, measured version of the two-path setup. Where a generic guide and our own
measurements disagreed, the measurement wins (noted inline). Start with **Path B (llama.cpp)** —
it works today with the official weights and zero format gambling. Graduate to **Path A (vLLM)**
only when you want high-concurrency agentic serving.

## The format fork (read this first)
The official Q4 you'd grab is a **GGUF** (`deepreinforce-ai/Ornith-1.0-35B-GGUF`), which runs best
in **llama.cpp / Ollama**. vLLM's GGUF support is experimental and historically flaky for MoE, so
don't force the GGUF into vLLM. If you want vLLM (better agentic serving, clean native tool/reasoning
parsers), you want a vLLM-native 4-bit: **AWQ/GPTQ**, or — ideal for the 5090's native FP4 — an
**NVFP4** build (e.g. `sakamakismile/Ornith-1.0-35B-NVFP4`; confirm it loads on a single GPU first).

---

## Path B — llama.cpp, rock-solid Q4/Q6 (recommended starting point)

**Build with Blackwell support** (RTX 5090 = sm_120; needs CUDA ≥ 12.8 and a recent llama.cpp —
early Blackwell support was rough):
```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build llama.cpp/build --config Release -j --target llama-server
```

**Serve it** (this is what we actually ran):
```bash
./llama.cpp/build/bin/llama-server \
  -m ./ornith-1.0-35b-Q6_K.gguf \   # Q4_K_M (~20GB) fits easily; Q5/Q6 if you want max correctness
  --host 0.0.0.0 --port 8000 \
  -ngl 99 \                 # all layers on GPU
  --n-cpu-moe 6 \           # ONLY if VRAM is tight (other apps on the GPU); see offload note
  -fa on \
  -c 65536 \                # context (KV is cheap here — see below)
  --jinja \                 # loads the chat template → tool calling parses correctly
  --reasoning-format deepseek   # splits <think> into a separate reasoning_content field
```
- **Quant choice:** Q4_K_M (~20 GB) leaves the most headroom and fits trivially; the 35B's value is
  *correctness*, so **Q5_K_M/Q6_K** (~25/28.5 GB) is worth it if it fits — and with KV being cheap,
  it does. (We ran Q6_K throughout.)
- **`--jinja`** is the important flag for agentic use (tool-call parsing). `--reasoning-format
  deepseek` surfaces the `<think>` block as `reasoning_content`, like vLLM.

---

## Path A — vLLM, max agentic throughput (needs an NVFP4/AWQ quant)

Needs a Blackwell-capable vLLM + CUDA 12.8. Single GPU → no `--tensor-parallel-size`:
```bash
vllm serve sakamakismile/Ornith-1.0-35B-NVFP4 \
  --served-model-name Ornith-1.0-35B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \   # from the official model card
  --reasoning-parser qwen3 \       # surfaces <think> as reasoning_content
  --trust-remote-code
```
This is the official tool-call + reasoning config from the model card, adapted to single-GPU + a
4-bit quant. `--kv-cache-dtype fp8` is optional here, not essential (KV is already cheap — see below).
See `path-a-feasibility.md` in this folder for the real-world difficulty of standing this up today.

---

## Context / KV cache — cheaper than any generic guide will tell you (measured)
Your binding constraint is **weights, not KV**. A normal transformer would need 40–80 GB of KV for
256K; **this model needs ~5 GB**, because of hybrid linear attention (a full-attention layer only
every 4th) + GQA with 2 KV heads → **~20 KiB/token** (measured on the card: 256K → 5120 MiB KV).
- 64K ≈ ~1.3 GB KV · 128K ≈ ~2.5 GB · **256K ≈ ~5 GB** (native trained max is 256K).
- KV quant (`-ctk q8_0 -ctv q8_0` / `--kv-cache-dtype fp8`) is a nice-to-have, **not** the critical
  lever it is elsewhere. Don't let "KV is expensive" advice scare you off long context here.
- If VRAM is tight, free it the *MoE* way: **`--n-cpu-moe N`** (park cold experts on CPU, keep
  attention on GPU) → 151 tok/s. **Do not** use whole-layer offload (`-ngl 34`) → tanks to ~50 tok/s.

## Sampling + output budget (get these wrong and it misbehaves)
- **temp 0.6, top_p 0.95, top_k 20** (benchmarks use 1.0; 0.6 is the daily-driver rec). Low temp →
  degenerate loops.
- **`max_tokens` ≥ 32000.** It's a *very* verbose reasoner (~30K thinking tokens on hard problems).
  Under-budget it and the code truncates and it looks broken. This is the #2 mistake after temperature.

## Downloads
`-hf …:Q4_K_M` auto-download (and `hf download`) can crawl/stall — HF throttles ~1.3 MB/s per
connection by IP, and its Xet transfer stalled to zero for us. If it's slow, use
`scripts/download.sh` (parallel chunked, resumable).

## Quick tool-call sanity test (do this before wiring an agent)
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Ornith-1.0-35B",
  "messages": [{"role":"user","content":"List the Python files in the current directory."}],
  "tools": [{"type":"function","function":{
    "name":"run_shell","description":"Run a shell command and return its output.",
    "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
  "tool_choice": "auto", "temperature": 0.6, "top_p": 0.95, "max_tokens": 32000
}'
```
Expect a `tool_calls` entry calling `run_shell` with e.g. `{"command":"ls *.py"}`. If it comes back
as raw text in the message instead, the tool-call parser isn't engaged — re-check `--jinja`
(llama.cpp) or `--tool-call-parser` (vLLM).

## Wiring an agent (both paths expose `…/v1`)
**OpenHands** (the official harness):
```bash
pip install openhands-ai
export LLM_MODEL="openai/Ornith-1.0-35B"      # match --served-model-name
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"
openhands
```
**Cline / Continue / any OpenAI-compatible coding agent:** base URL `http://localhost:8000/v1`, API
key anything (`EMPTY`), model `Ornith-1.0-35B`. Set temp 0.6 and a high max-tokens.

## Bottom line
Start with **Path B today** — least likely to fight you on brand-new Blackwell silicon. The three
things that decide whether it works: right **temperature** (0.6–1.0), enough **output budget**
(≥32K), and **`--n-cpu-moe`** instead of layer offload. Get those right and it's a capable,
fully-local coding model.
