#!/usr/bin/env python3
# Experiment 1: hold EVERYTHING constant on vLLM/NVFP4, vary only the per-request seed.
# Question: is the regex reasoning-loop systematic (every seed) or variance (some seeds)?
# We cap max_tokens at 16000 -- not enough to finish a healthy solve, but plenty to DETECT a loop
# via a low reasoning-uniqueness ratio. Classify each seed: LOOP vs HEALTHY(-so-far) vs SOLVED.
import json, urllib.request, collections, sys, os
PORT = os.environ.get("PORT","8000"); MODELN = os.environ.get("MODEL","Ornith-1.0-35B")

PROMPT = ("Implement a regular-expression matcher in Rust. Exactly this API: "
          "fn matches(pattern: &str, text: &str) -> bool. It must FULL-match. Support literals, "
          "'.', '*', '+', '?', '(' ')' grouping, and '|' alternation, with backtracking. "
          "Provide the full program in one ```rust code block.")
SYS = ("You are a senior Rust engineer. Provide one complete, correct, idiomatic, compiling "
       "solution in a single ```rust code block.")
SEEDS = [int(x) for x in (sys.argv[1:] or ["7","11","23","42","99"])]

def run(seed):
    body = json.dumps({"model":MODELN,
        "messages":[{"role":"system","content":SYS},{"role":"user","content":PROMPT}],
        "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"seed":seed,"max_tokens":16000}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    ch = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    r = ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or ""
    c = ch["message"].get("content") or ""
    lines = [ln.strip() for ln in r.splitlines() if len(ln.strip()) > 25]
    uniq = len(set(lines))/max(len(lines),1)
    # most-repeated line count as a second signal
    top = collections.Counter(lines).most_common(1)
    maxrep = top[0][1] if top else 0
    has_code = "```rust" in c or "fn matches" in c
    if has_code:                 verdict = "SOLVED/emitted code"
    elif uniq < 0.35:            verdict = "LOOP"
    else:                        verdict = "healthy-but-unfinished"
    return dict(seed=seed, finish=ch["finish_reason"], rlen=len(r), clen=len(c),
                uniq=round(uniq,2), maxrep=maxrep, verdict=verdict)

print(f"{'seed':>5} {'finish':>7} {'reason_chars':>12} {'content':>8} {'uniq':>5} {'maxrep':>7}  verdict")
for s in SEEDS:
    d = run(s)
    print(f"{d['seed']:>5} {d['finish']:>7} {d['rlen']:>12} {d['clen']:>8} {d['uniq']:>5} {d['maxrep']:>7}  {d['verdict']}")
    sys.stdout.flush()
