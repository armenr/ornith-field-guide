#!/usr/bin/env python3
# Resolve two things at once: (1) does a given seed LOOP on this config, and (2) is the 16K-token
# window a reliable loop classifier, or do loops develop later? For each seed we generate up to 56K
# tokens and compute reasoning-uniqueness over the FULL trace vs a ~16K-token PREFIX (~60800 chars),
# plus where repetition concentrates. Args: PORT MODEL [seeds...]
import json, urllib.request, collections, sys, os
PORT=os.environ.get("PORT","8095"); MODEL=os.environ.get("MODEL","ornith-1.0-35b-Q4_K_M.gguf")
SEEDS=[int(x) for x in (sys.argv[1:] or ["7","1","2","3","4"])]
PROMPT=("Implement a regular-expression matcher in Rust. Exactly this API: "
        "fn matches(pattern: &str, text: &str) -> bool. It must FULL-match. Support literals, "
        "'.', '*', '+', '?', '(' ')' grouping, and '|' alternation, with backtracking. "
        "Provide the full program in one ```rust code block.")
SYS=("You are a senior Rust engineer. Provide one complete, correct, idiomatic, compiling "
     "solution in a single ```rust code block.")

def uniq(text):
    lines=[ln.strip() for ln in text.splitlines() if len(ln.strip())>25]
    if not lines: return 1.0, 0
    return len(set(lines))/len(lines), collections.Counter(lines).most_common(1)[0][1]

def run(seed):
    body=json.dumps({"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":PROMPT}],
        "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"seed":seed,"max_tokens":56000}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    ch=json.load(urllib.request.urlopen(req,timeout=1800))["choices"][0]
    r=ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or ""
    c=ch["message"].get("content") or ""
    fu,fm=uniq(r); pu,pm=uniq(r[:60800])   # full vs ~16K-token prefix
    code=("```rust" in c) or ("fn matches" in c)
    return dict(seed=seed,finish=ch["finish_reason"],rchars=len(r),
                prefix_uniq=round(pu,2),full_uniq=round(fu,2),full_maxrep=fm,code=code)

print(f"{'seed':>5} {'finish':>7} {'rchars':>8} {'pre16k_uniq':>11} {'full_uniq':>9} {'maxrep':>7} {'code':>5}  verdict")
for s in SEEDS:
    d=run(s)
    # loop if full_uniq low; "late loop" if prefix healthy but full degraded
    if d['code']: v="SOLVED"
    elif d['full_uniq']<0.4 and d['prefix_uniq']>=0.5: v="LATE-LOOP (16k missed it!)"
    elif d['full_uniq']<0.4: v="LOOP (early)"
    else: v="verbose/healthy"
    print(f"{d['seed']:>5} {d['finish']:>7} {d['rchars']:>8} {d['prefix_uniq']:>11} {d['full_uniq']:>9} {d['full_maxrep']:>7} {str(d['code']):>5}  {v}")
    sys.stdout.flush()
