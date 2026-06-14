#!/bin/bash
# Pull all kernel refs in priority order with 3s sleep, retry x3, log to HARVEST_LOG.md
set -u
ROOT=/Users/liucong/code/kaggle/ROGII
LOG=$ROOT/experiments/public_resources/HARVEST_LOG.md
RAW=$ROOT/experiments/public_resources/kernels_raw
PRIO=$ROOT/experiments/public_resources/kernels_by_priority.txt
mkdir -p "$RAW"

total=$(wc -l < "$PRIO")
i=0
ok=0
fail=0
skip=0
while IFS=$'\t' read -r votes ref; do
    i=$((i+1))
    slug_safe=$(echo "$ref" | tr '/' '_')
    target=$RAW/$slug_safe
    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        skip=$((skip+1))
        continue
    fi
    mkdir -p "$target"
    success=0
    for attempt in 1 2 3; do
        if kaggle kernels pull "$ref" -p "$target" -m >/dev/null 2>&1; then
            success=1
            break
        fi
        sleep 5
    done
    if [ $success -eq 1 ]; then
        ok=$((ok+1))
        if [ $((ok % 20)) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M')] pull progress: $i/$total | ok=$ok fail=$fail skip=$skip" >> "$LOG"
        fi
    else
        fail=$((fail+1))
        echo "[$(date '+%Y-%m-%d %H:%M')] FAILED kernel pull: $ref (3 retries)" >> "$LOG"
        rmdir "$target" 2>/dev/null || true
    fi
    sleep 3
done < "$PRIO"

echo "[$(date '+%Y-%m-%d %H:%M')] Phase 1A pull complete: total=$total ok=$ok fail=$fail skip=$skip" >> "$LOG"
