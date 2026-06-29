#!/usr/bin/env python3
# Compare two engines' per-position next-token distributions (from kl_sweep.py outputs).
# Reports top-1 agreement % and Jensen-Shannon divergence (bounded [0, ln2=0.693]) per position.
# Args: VLLM_JSON LLAMA_JSON
import json, sys, math
A = {d["pos"]: d for d in json.load(open(sys.argv[1])) if "top" in d}
B = {d["pos"]: d for d in json.load(open(sys.argv[2])) if "top" in d}
common = sorted(set(A) & set(B))

def dist(top):  # token -> prob, renormalized over the returned top-k
    ps = {t: math.exp(lp) for t, lp in top.items()}
    s = sum(ps.values()); return {t: p / s for t, p in ps.items()}, min(ps.values()) / s

def js(pa, fa, pb, fb):  # JS over union; missing tokens get the floor prob
    U = set(pa) | set(pb)
    P = {t: pa.get(t, fa) for t in U}; Q = {t: pb.get(t, fb) for t in U}
    sp = sum(P.values()); sq = sum(Q.values())
    P = {t: v / sp for t, v in P.items()}; Q = {t: v / sq for t, v in Q.items()}
    M = {t: 0.5 * (P[t] + Q[t]) for t in U}
    kl = lambda X: sum(X[t] * math.log(X[t] / M[t]) for t in U if X[t] > 0)
    return 0.5 * kl(P) + 0.5 * kl(Q)

print(f"{'pos':>6} {'top1_agree':>10} {'JS_div':>7}  vLLM_top1 / llama_top1")
agree = 0; jss = []
for p in common:
    pa, fa = dist(A[p]["top"]); pb, fb = dist(B[p]["top"])
    ta = max(A[p]["top"], key=A[p]["top"].get); tb = max(B[p]["top"], key=B[p]["top"].get)
    same = (ta == tb); agree += same
    j = js(pa, fa, pb, fb); jss.append(j)
    print(f"{p:>6} {str(same):>10} {j:7.3f}  {ta!r:14} / {tb!r}")
n = len(common)
print(f"\nN positions: {n}")
print(f"top-1 agreement: {agree}/{n} = {100*agree//n}%")
print(f"JS divergence:   mean={sum(jss)/n:.3f}  median={sorted(jss)[n//2]:.3f}  max={max(jss):.3f}  (bound ln2=0.693)")
print("Interp: low JS + high agreement => distributions ~match => loop is sampling/trajectory-chaos, NOT logit-quality.")
