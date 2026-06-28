# Troubleshooting (every wall we hit, and the fix)

### Download is glacial / stalls
- HF throttles **~1.3 MB/s per connection by IP**; the newer **Xet** transfer stalled to ~0 on our
  box (the `.incomplete` file stayed at 0 bytes). `hf download` with `HF_HUB_ENABLE_HF_TRANSFER` is
  now deprecated in favor of Xet, which didn't help.
- **Fix:** `scripts/download.sh` — many parallel range requests (12), resumable. Beats single-stream
  and Xet. If your *uplink* itself is the cap (we measured ~4.5 MB/s total), nothing helps; just wait.

### 35B won't fully fit the 32 GB card / OOM at `-ngl 99`
- Q6_K weights are ~26.6 GiB; add KV + CUDA compute buffers and it needs ~28–29 GB. If anything else
  is on the GPU (compositor, a stray process), it tips over. Check with:
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
- **Fix:** `-ngl 99 --n-cpu-moe 6` (keep attention on GPU, park cold experts on CPU). Fits around a
  few GB of other usage with ~no speed loss (151 tok/s). Bump the number (`--n-cpu-moe 8/10`) if
  still tight. **Do NOT** fall back to whole-layer `-ngl 34` — that's the ~50 tok/s trap.

### It loops / spams "I apologize for the repeated errors" / never stops
- You're running it too cold or with thinking suppressed. **Use temp 0.6–1.0**, top_p 0.95, top_k 20,
  `--reasoning-budget -1`. See settings.md. This is the #1 cause of every "broken" run we had.

### The `<think>` block leaks into the answer / `does not support thinking`
- ollama on a raw-GGUF import can't gate thinking (errors on `think:true`, no-ops on `think:false`).
- **Fix:** use **llama.cpp** with `--jinja --reasoning-format deepseek`. It uses the GGUF's embedded
  template and cleanly separates `reasoning_content` from `content`.

### Answer comes back empty / truncated / the model "can't solve" a hard task
- **This is the most deceptive failure mode** — it makes a capable model look incompetent. The cause
  is almost always **too small a `max_tokens`**. This model reasons for **~30,000 tokens** on hard
  problems; if your budget is below that, it spends the whole budget thinking and the code comes out
  empty or **truncated** (`finish_reason: length`, huge `reasoning_content`, short/cut `content`).
- We hit this ourselves: with `max_tokens: 14000` a regex-engine task "plateaued and never converged"
  and looked like a model ceiling. Raising to **≥ 32000** (with `-c` ≥ 65536) → it converged in 3
  rounds. **Before concluding the model can't do something, raise the budget and re-test.**
- Keep temp 0.6 (not lower) so it converges instead of thrashing.

### "I don't see GPU saturation"
- Normal. Single-stream decode of a low-active-param MoE is memory-latency bound (~20% SM, ~100 W).
  tok/s is the real metric. Want the card to sweat? Run concurrent requests (or vLLM batching).

### Running the server in an automated/agent context
- `llama-server` is a daemon: start it detached and poll `GET /health` (use `curl --retry-connrefused`
  rather than shell `sleep`) before sending requests. One model per GPU at a time on a 32 GB card
  (9B 9.5 GB + 35B 28 GB won't co-reside) — stop one server before starting the other.
