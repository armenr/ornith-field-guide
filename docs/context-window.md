# Context window: how much can you actually give Ornith on a 5090?

*Measured 2026-06-29 on RTX 5090 (32 GB), Q4_K_M via llama.cpp. Reproduce with `scripts/needle-test.py`.*

## The answer: 256K — and it's a *model* ceiling, not just a VRAM one
Ornith-1.0-35B's **native trained context is 262144 (256K)** (`n_ctx_train = 262144`). That is the
high-quality maximum, full stop — local or not.

### 256K appears usable (not just allocatable)
Needle-in-a-haystack recall (plant a unique code in a huge log, ask for it) — two probes, **n=1 each**:

| context | needle depth | result |
|---|---|---|
| ~200K tokens | 50% | **HIT** ✅ |
| ~250K tokens | 10% (early — "lost in the middle" worst case) | **HIT** ✅ |

Both probes hit — suggestive that the model uses the whole window (including the hard early-position
case), though that's two single runs, not exhaustive coverage. It fits at full **f16 KV in ~30.6 GB**
(`-np 1`; KV is only ~5.1 GB thanks to its cheap-KV hybrid attention). Prefill is fast and batched
(~4K tok/s — a 200K prompt processed in ~52s).

### Beyond 256K does NOT work cleanly
Ornith uses a **sectioned rope** (`rope type 40`, `dimension_sections [11,11,10,0]`) that does **not**
accept standard YaRN/linear extension in llama.cpp:
- `--rope-scaling yarn` was silently ignored (logs still showed `rope scaling = linear,
  freq_scale_train = 1`).
- With `-c 524288` llama.cpp allocated 512K of KV cells but **capped the usable slot at 262144**; a 400K
  request was rejected at that wall.
- Even if forced, past 256K is **untrained extrapolation** → degraded quality regardless.

**So 256K is the practical maximum.** Don't expect 500K–1M of *usable* context out of this model.

## VRAM budget (single 32 GB card, Q4_K_M, weights 21 GB)
KV ≈ 20 KiB/token (f16), ~10 (q8), ~5 (q4). Measured: 256K f16 KV ≈ 5 GB.

| context | f16 KV | q8 KV | q4 KV | fits 32 GB? | but… |
|---|---|---|---|---|---|
| 256K | 5 GB | 2.5 GB | 1.2 GB | ✅ (30.6 GB at f16) | native, high quality |
| 512K | 10 GB | 5 GB | 2.5 GB | only with q8/q4 | needs rope extension → **doesn't work** |
| 1M | 20 GB | 10 GB | 5 GB | only q4 | needs 4× extension → **doesn't work** |

## Orchestrator vs sub-agents on one card (the spoiled-by-1M reality)
A single 5090's KV budget is **~256K total (f16), shared across concurrent requests**:
- `-np 1` → one request gets the **full 256K** (the big-context "orchestrator" worker, sequential).
- `-np N` → splits the pool (`-c 131072 -np 4` → ~32K per concurrent request).

You **cannot** run "1M-orchestrator + 250K×N sub-agents simultaneously" on one 5090 — that's several
million KV-tokens at once = multi-GPU / cluster territory. Practical patterns:
- **All-local:** 256K orchestrator (`-np 1`), run sub-agents sequentially; or split with `-np N` for
  smaller concurrent windows.
- **Hybrid (if you truly need 1M orchestration):** a hosted/1M-trained model as the orchestrator, with
  the 5090 serving ≤256K sub-agents.

## Recommended
Daily driver at **`-c 262144 -np 1`** for max single-request context (`scripts/serve-q4.sh CTX=262144`).
If your desktop is VRAM-hungry and 30.6 GB is too tight, drop to `--cache-type-k q8_0 --cache-type-v
q8_0` (256K at ~27 GB; q8 KV *should* be near-lossless for recall — but recall was only needle-verified
at f16 KV here, so treat that as expectation, not measurement).
