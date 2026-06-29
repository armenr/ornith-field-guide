#!/usr/bin/env python3
# RCA probe: at "commit vs reconsider" reasoning forks, measure the next-token distribution
# (top-1 prob, top1-top2 margin, entropy over top-20). Control prefixes have a near-deterministic
# next token. Hypothesis: NVFP4+vLLM flattens the DECISION-point distribution (lower margin / higher
# entropy) vs llama.cpp k-quant, while CONTROLS agree -> the symmetry-break failure, made visible.
# Args: PORT MODEL ENGINE
import json, urllib.request, sys, math
PORT, MODEL, ENGINE = sys.argv[1], sys.argv[2], sys.argv[3]

Q = "Implement a regular-expression matcher in Rust with backtracking. Exactly: fn matches(pattern: &str, text: &str) -> bool."
HEAD = f"<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n"

# DECISION-point prefixes: model has just proposed/considered an approach; next token = commit (start
# code) vs reconsider ("Actually/Wait/Let me/But/Hmm/different approach").
DEC = {
 "d1_proposed_approach": "I need to match the full pattern against the text. I'll implement a recursive backtracking matcher that tries all possibilities. ",
 "d2_after_restate":     "Let me think. I could use a recursive backtracking matcher. Let me use a different approach and ",
 "d3_ready_to_code":     "OK, I have the design: parse the pattern into elements, then match with backtracking. Let me write the code.\n\n",
 "d4_doubt":             "I'll handle '*' by trying zero or more repetitions. Hmm, but that might not handle the empty case correctly. ",
 "d5_mid_reconsider":    "Wait, the position-based approach is cleaner. Actually, the simplest approach is to ",
}
# CONTROL prefixes: next token should be near-deterministic on ANY decent model.
CTL = {
 "c1_signature": "The required signature is `fn matches(pattern: &str, text: &str) -> ",
 "c2_word":      "I'll use a recursive back",
 "c3_arith":     "Quick check: 2 + 3 * 4 = ",
}

def probe(text):
    prompt = HEAD + text
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 1,
                       "temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 20}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body, {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    lp = d["choices"][0]["logprobs"]
    if isinstance(lp, dict) and lp.get("content"):          # llama.cpp newer style
        items = [(e["token"], e["logprob"]) for e in lp["content"][0]["top_logprobs"]]
    elif isinstance(lp, dict) and lp.get("top_logprobs"):   # vLLM / OpenAI completions style
        items = list(lp["top_logprobs"][0].items())
    else:
        raise RuntimeError(f"unknown logprobs shape: {str(lp)[:200]}")
    items = sorted(items, key=lambda kv: kv[1], reverse=True)
    logs = [v for _, v in items]
    ps = [math.exp(x) for x in logs]
    s = sum(ps); pn = [p / s for p in ps]  # renormalize over the returned top-k
    top1 = pn[0]; margin = pn[0] - (pn[1] if len(pn) > 1 else 0)
    ent = -sum(p * math.log(p) for p in pn if p > 0)
    toks = [t for t, _ in items[:5]]
    return top1, margin, ent, toks

print(f"=== {ENGINE} ({MODEL}) ===")
print(f"{'prefix':22} {'top1':>6} {'margin':>7} {'entropy':>8}  top-5 next tokens")
agg = {"DEC": [], "CTL": []}
for grp, D in [("DEC", DEC), ("CTL", CTL)]:
    for name, txt in D.items():
        t1, m, e, toks = probe(txt)
        agg[grp].append((t1, m, e))
        print(f"{name:22} {t1:6.3f} {m:7.3f} {e:8.3f}  {toks}")
def mean(xs): return sum(xs) / len(xs) if xs else 0
for grp in ("DEC", "CTL"):
    t1s = [a[0] for a in agg[grp]]; ms = [a[1] for a in agg[grp]]; es = [a[2] for a in agg[grp]]
    print(f">>> {grp}: mean top1={mean(t1s):.3f}  mean margin={mean(ms):.3f}  mean entropy={mean(es):.3f}")
