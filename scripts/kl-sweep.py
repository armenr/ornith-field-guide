#!/usr/bin/env python3
# Measure how much vLLM-NVFP4 and llama-Q4 next-token DISTRIBUTIONS actually differ across a real
# reasoning trajectory. Generates a shared text once (saved), then probes the SAME ~N prefix positions
# on whichever server is given, saving top-20 logprobs per position. A separate compare step computes
# per-position top-1 agreement + KL divergence offline.
# Args: PORT MODEL OUTFILE   (env SHARED=path to shared text; generated from THIS server if missing)
import json, urllib.request, sys, os, math
PORT, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SHARED = os.environ.get("SHARED", "/home/v3ct0r/.claude/jobs/d566939e/tmp/shared_traj.txt")
Q = "Implement a regular-expression matcher in Rust with backtracking. fn matches(pattern: &str, text: &str) -> bool. Think step by step."
HEAD = f"<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n"

def gen_shared():
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": Q}],
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7, "max_tokens": 1200}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
    txt = (m.get("reasoning") or m.get("reasoning_content") or m.get("content") or "")
    open(SHARED, "w").write(txt)
    return txt

def probe(prefix):
    body = json.dumps({"model": MODEL, "prompt": HEAD + prefix, "max_tokens": 1,
                       "temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 20}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body, {"Content-Type": "application/json"})
    lp = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["logprobs"]
    if isinstance(lp, dict) and lp.get("content"):
        return {e["token"]: e["logprob"] for e in lp["content"][0]["top_logprobs"]}
    return dict(lp["top_logprobs"][0])

if not os.path.exists(SHARED):
    print("generating shared trajectory from this server...")
    txt = gen_shared()
else:
    txt = open(SHARED).read()
print(f"shared text: {len(txt)} chars")

# ~24 positions spread across the trajectory (char-based prefixes; same tokenizer -> aligned)
N = 24
L = len(txt)
positions = [int(L * (i + 1) / (N + 1)) for i in range(N)]
out = []
for p in positions:
    pref = txt[:p]
    try:
        top = probe(pref)
        out.append({"pos": p, "top": top})
    except Exception as e:
        out.append({"pos": p, "error": str(e)[:80]})
    sys.stdout.write("."); sys.stdout.flush()
print()
json.dump(out, open(OUT, "w"))
print(f"saved {len(out)} positions -> {OUT}")
