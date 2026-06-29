#!/usr/bin/env bash
# Sanity-check a running Ornith vLLM server (Path A). Proves three things:
#   1) the server answers,  2) the model's <think> reasoning ENGAGES (reasoning field is non-empty),
#   3) measured decode speed (tok/s).
# Usage: scripts/smoke-vllm.sh [PORT]   (default 8000)
set -u
PORT="${1:-8000}"
BASE="http://127.0.0.1:${PORT}"

echo "Waiting for ${BASE}/health ..."
for _ in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health" || true)
  [ "$code" = "200" ] && break
  sleep 1
done
[ "$code" = "200" ] || { echo "server not healthy (last code: $code)"; exit 1; }

MODEL=$(curl -s "${BASE}/v1/models" | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: ${MODEL}"
echo "Sending a coding prompt (give it room to think)..."

t0=$(date +%s.%N)
resp=$(curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"${MODEL}\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Write a Rust function fn is_prime(n: u64) -> bool, then briefly justify it.\"}],
  \"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20, \"max_tokens\": 8000
}")
t1=$(date +%s.%N)

echo "$resp" | python3 -c "
import json,sys,os
d=json.load(sys.stdin)
ch=d['choices'][0]; m=ch['message']
# vLLM nightly: chain-of-thought is in 'reasoning'; some builds use 'reasoning_content'.
rc=m.get('reasoning') or m.get('reasoning_content') or ''
content=m.get('content') or ''
ctok=d['usage']['completion_tokens']
elapsed=float(os.environ['T1'])-float(os.environ['T0'])
print(f\"finish_reason : {ch['finish_reason']}\")
print(f\"thinking       : {len(rc)} chars  ({'ENGAGED ✅' if rc.strip() else 'NOT engaged ❌'})\")
print(f\"answer chars   : {len(content)}\")
print(f\"speed          : {ctok} tok in {elapsed:.1f}s  = {ctok/elapsed:.1f} tok/s\")
print('--- answer (head) ---')
print(content[:600])
" T0="$t0" T1="$t1"
