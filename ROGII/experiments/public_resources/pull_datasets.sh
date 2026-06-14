#!/bin/bash
# For each dataset: if size < 500MB download to datasets_raw, else save metadata only
set -u
ROOT=/Users/liucong/code/kaggle/ROGII
LOG=$ROOT/experiments/public_resources/HARVEST_LOG.md
META=$ROOT/experiments/public_resources/datasets
RAW=$ROOT/experiments/public_resources/datasets_raw
mkdir -p "$META" "$RAW"

ok=0; fail=0; desc=0; skip=0
while IFS=',' read -r ref title size lastUpdated dl votes usability; do
    [ "$ref" = "ref" ] && continue
    [ -z "$ref" ] && continue
    slug_safe=$(echo "$ref" | tr '/' '_')
    size_mb=$((size / 1024 / 1024))
    if [ "$size_mb" -lt 500 ]; then
        target="$RAW/$slug_safe"
        if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
            skip=$((skip+1))
            continue
        fi
        mkdir -p "$target"
        success=0
        for attempt in 1 2 3; do
            if kaggle datasets download "$ref" -p "$target" --unzip >/dev/null 2>&1; then
                success=1
                break
            fi
            sleep 5
        done
        if [ $success -eq 1 ]; then
            ok=$((ok+1))
            echo "[$(date '+%Y-%m-%d %H:%M')] dataset DL ok: $ref (${size_mb}MB)" >> "$LOG"
        else
            fail=$((fail+1))
            echo "[$(date '+%Y-%m-%d %H:%M')] dataset FAILED: $ref" >> "$LOG"
            rmdir "$target" 2>/dev/null || true
        fi
    else
        # Description only
        target_md="$META/${slug_safe}.md"
        if [ -f "$target_md" ]; then skip=$((skip+1)); continue; fi
        cat > "$target_md" <<EOF
# Dataset: $ref

- title: $title
- size_mb: $size_mb
- last_updated: $lastUpdated
- downloads: $dl
- votes: $votes
- usability: $usability
- url: https://www.kaggle.com/datasets/$ref
- decision: SKIPPED (size > 500MB)
EOF
        desc=$((desc+1))
    fi
    sleep 3
done < $ROOT/experiments/public_resources/all_datasets_meta.csv

echo "[$(date '+%Y-%m-%d %H:%M')] Phase 1B datasets complete: dl=$ok fail=$fail desc=$desc skip=$skip" >> "$LOG"
