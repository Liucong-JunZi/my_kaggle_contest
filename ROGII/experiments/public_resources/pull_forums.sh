#!/bin/bash
# Fetch all missing forum topic messages
set -u
ROOT=/Users/liucong/code/kaggle/ROGII
LOG=$ROOT/experiments/public_resources/HARVEST_LOG.md
RAW=$ROOT/experiments/public_resources/forum_raw
OLD=$ROOT/docs/forum-snapshots/2026-06-12/messages
mkdir -p "$RAW"

ok=0; fail=0; skip=0
while IFS=',' read -r tid rest; do
    [ "$tid" = "id" ] && continue
    [ -z "$tid" ] && continue
    case "$tid" in (*[!0-9]*) continue;; esac
    target="$RAW/${tid}.txt"
    # Skip if we already have it (in old snapshot or already-fetched)
    if [ -f "$target" ] || [ -f "$OLD/${tid}.txt" ]; then
        skip=$((skip+1))
        continue
    fi
    success=0
    for attempt in 1 2 3; do
        if kaggle competitions topic-messages rogii-wellbore-geology-prediction "$tid" -n -1 -s new > "$target" 2>/dev/null; then
            if [ -s "$target" ]; then
                success=1
                break
            fi
        fi
        sleep 5
    done
    if [ $success -eq 1 ]; then
        ok=$((ok+1))
    else
        fail=$((fail+1))
        rm -f "$target"
        echo "[$(date '+%Y-%m-%d %H:%M')] FAILED forum pull tid=$tid" >> "$LOG"
    fi
    sleep 3
done < $ROOT/experiments/public_resources/all_topics.csv

echo "[$(date '+%Y-%m-%d %H:%M')] Phase 1C forum complete: ok=$ok fail=$fail skip=$skip" >> "$LOG"
