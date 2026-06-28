#!/usr/bin/env bash
# Parallel chunked, resumable Hugging Face downloader.
# WHY: HF throttles each connection to ~1.3 MB/s by IP, and its newer "Xet"
# transfer stalled to ~0 on our box. Many parallel range requests beat both.
#
# Usage:
#   ./download.sh deepreinforce-ai/Ornith-1.0-35B-GGUF ornith-1.0-35b-Q6_K.gguf
#   ./download.sh deepreinforce-ai/Ornith-1.0-9B-GGUF  ornith-1.0-9b-Q6_K.gguf
#   ./download.sh <repo> <file> [parts=12] [outdir=$HOME/models/ornith]
set -u
REPO="${1:?usage: download.sh <hf-repo> <file> [parts] [outdir]}"
FILE="${2:?need a filename}"
PARTS="${3:-12}"
OUTDIR="${4:-$HOME/models/ornith}"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
mkdir -p "$OUTDIR"; OUT="$OUTDIR/$FILE"

LEN=$(curl -sIL -A "Mozilla/5.0" "$URL" \
      | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')
[ -z "${LEN:-}" ] && { echo "ERROR: could not resolve size for $URL"; exit 1; }
echo "downloading $FILE : $LEN bytes in $PARTS parts -> $OUT"
chunk=$(( (LEN + PARTS - 1) / PARTS ))

dl_part() {
  local i=$1 s=$(( i*chunk )) e=$(( (i+1)*chunk - 1 ))
  [ $e -ge $LEN ] && e=$(( LEN - 1 ))
  local pf="$OUT.part$i" want=$(( e - s + 1 )) a h
  for a in $(seq 1 200); do
    h=0; [ -f "$pf" ] && h=$(stat -c%s "$pf" 2>/dev/null || echo 0)
    [ "$h" -ge "$want" ] && break
    curl -sL -A "Mozilla/5.0" --retry 8 --retry-delay 3 --retry-all-errors \
         -r $((s+h))-$e "$URL" >> "$pf" 2>/dev/null
  done
}
for i in $(seq 0 $((PARTS-1))); do dl_part "$i" & done
wait

tot=0; for i in $(seq 0 $((PARTS-1))); do tot=$(( tot + $(stat -c%s "$OUT.part$i" 2>/dev/null||echo 0) )); done
if [ "$tot" -eq "$LEN" ]; then
  : > "$OUT"; for i in $(seq 0 $((PARTS-1))); do cat "$OUT.part$i" >> "$OUT"; done
  rm -f "$OUT".part*
  echo "OK -> $OUT ($(stat -c%s "$OUT") bytes)"
else
  echo "INCOMPLETE got=$tot want=$LEN (re-run to resume)"; exit 1
fi
