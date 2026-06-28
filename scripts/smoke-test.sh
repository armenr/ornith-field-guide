#!/usr/bin/env bash
# Quick check that a served model works + prints decode speed and the reasoning split.
set -u
PORT="${1:-8095}"
curl -s --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" -d '{
  "messages":[{"role":"user","content":"Write a Rust function that reverses a string. One code block, brief."}],
  "temperature":0.6, "top_p":0.95, "top_k":20, "min_p":0, "seed":7, "max_tokens":600
}' | python3 -c '
import json,sys
d=json.load(sys.stdin)
ch=d["choices"][0]; m=ch["message"]; t=d.get("timings",{})
print("finish_reason:", ch.get("finish_reason"))
print("decode tok/s :", round(t.get("predicted_per_second",0),1))
r=m.get("reasoning_content") or ""
if r: print(f"[reasoning: {len(r)} chars hidden in reasoning_content]")
print("---- answer ----")
print(m.get("content") or "<empty>")
'
