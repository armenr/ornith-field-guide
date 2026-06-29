#!/usr/bin/env python3
# Larger-N regex loop-rate sweep with summary stats. Args: PORT MODEL LABEL N SEED_START
# Writes per-sample JSONL to sweep_<LABEL>.jsonl and prints a summary (mean uniq + 95% CI, loop-rate).
import json, urllib.request, collections, sys, math, os

PORT, MODEL, LABEL, N, SEED0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
OUT = f"/home/v3ct0r/.claude/jobs/d566939e/tmp/sweep_{LABEL}.jsonl"
PROMPT = ("Implement a regular-expression matcher in Rust. Exactly this API: "
          "fn matches(pattern: &str, text: &str) -> bool. It must FULL-match. Support literals, "
          "'.', '*', '+', '?', '(' ')' grouping, and '|' alternation, with backtracking. "
          "Provide the full program in one ```rust code block.")
SYS = ("You are a senior Rust engineer. Provide one complete, correct, idiomatic, compiling "
       "solution in a single ```rust code block.")
LOOP_T = 0.40  # uniq below this = degenerate loop

def run(seed):
    body = json.dumps({"model":MODEL,
        "messages":[{"role":"system","content":SYS},{"role":"user","content":PROMPT}],
        "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"seed":seed,"max_tokens":16000}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    ch = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    r = ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or ""
    c = ch["message"].get("content") or ""
    lines = [ln.strip() for ln in r.splitlines() if len(ln.strip()) > 25]
    uniq = len(set(lines))/max(len(lines),1)
    code = ("```rust" in c) or ("fn matches" in c)
    return dict(seed=seed, finish=ch["finish_reason"], uniq=round(uniq,3),
                rlen=len(r), clen=len(c), code=code, loop=(uniq < LOOP_T and not code))

open(OUT,"w").close()
print(f"{'seed':>5} {'finish':>7} {'uniq':>5} {'code':>5} {'loop':>5}")
res=[]
for s in range(SEED0, SEED0+N):
    d=run(s); res.append(d)
    open(OUT,"a").write(json.dumps(d)+"\n")
    print(f"{d['seed']:>5} {d['finish']:>7} {d['uniq']:>5} {str(d['code']):>5} {str(d['loop']):>5}")
    sys.stdout.flush()

us=[d['uniq'] for d in res]; n=len(us); m=sum(us)/n
sd=(sum((x-m)**2 for x in us)/(n-1))**0.5 if n>1 else 0
ci=1.96*sd/math.sqrt(n)
k=sum(1 for d in res if d['loop']); p=k/n
# Wilson 95% CI for loop-rate
z=1.96; den=1+z*z/n
ctr=(p+z*z/(2*n))/den; hw=(z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den
print(f"\n[{LABEL}] N={n}  mean_uniq={m:.3f} ± {ci:.3f} (95% CI)  sd={sd:.3f}")
print(f"[{LABEL}] loop-rate={k}/{n}={p:.0%}  Wilson95%=[{max(0,ctr-hw):.0%},{min(1,ctr+hw):.0%}]  code-emit={sum(d['code'] for d in res)}/{n}")
