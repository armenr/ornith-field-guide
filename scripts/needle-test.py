#!/usr/bin/env python3
# Needle-in-haystack long-context test. Builds ~TARGET_TOKENS of distinct filler, plants a unique
# secret code at DEPTH fraction, asks for it at the end. Reports prompt_tokens, recall hit/miss, timing.
# Args: PORT TARGET_TOKENS DEPTH_FRAC [MODELNAME]
import json, urllib.request, sys, time
PORT, TGT, DEPTH = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
MODEL = sys.argv[4] if len(sys.argv) > 4 else "x"
CODE = "XJ-4417-QODA-7731"
NEEDLE = f"\n\n*** IMPORTANT FACT TO REMEMBER: the secret access code is {CODE}. Memorize it. ***\n\n"

# ~18 tokens/line distinct filler; build to ~TGT tokens (est ~0.27 tokens/char -> aim by line count)
def line(i): return f"Entry {i:06d}: archive log records routine telemetry for batch sequence {i}, status nominal, checksum {(i*2654435761)&0xffff:04x}."
nlines = int(TGT / 33)  # measured ~33 tok/line for this filler; keeps prompt under n_ctx
ins = int(nlines * DEPTH)
buf = []
for i in range(nlines):
    if i == ins: buf.append(NEEDLE)
    buf.append(line(i))
filler = "\n".join(buf)
prompt = ("You are given a long log. Somewhere in it is a secret access code. "
          "Read carefully.\n\n" + filler +
          "\n\nQuestion: What is the secret access code stated in the log above? "
          "Reply with ONLY the code.")

body = json.dumps({"model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 3000}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
try:
    d = json.load(urllib.request.urlopen(req, timeout=3600))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:500]}")
    print(f"(prompt chars={len(prompt)}, est tokens~{len(prompt)//4})")
    raise SystemExit(1)
dt = time.time() - t0
ch = d["choices"][0]; m = ch["message"]
ans = (m.get("content") or "")
reason = (m.get("reasoning_content") or m.get("reasoning") or "")
hit = CODE in ans or CODE in reason
pt = d["usage"]["prompt_tokens"]
print(f"target~{TGT} depth={DEPTH:.0%}  prompt_tokens={pt}  elapsed={dt:.0f}s  finish={ch['finish_reason']}")
print(f"RECALL: {'HIT ✅' if hit else 'MISS ❌'}   code_expected={CODE}")
print(f"answer: {ans[:200]!r}")
if not hit and reason: print(f"reason_tail: {reason[-200:]!r}")
