#!/usr/bin/env bash
set -euo pipefail
# auto_hillclimb.sh — background monitor for candidate parquet files
#
# Watches results/candidates/ for new .parquet files. When count jumps by
# 3+ from the last check, triggers a hill climb run.
#
# Usage:   nohup bash orchestrator/auto_hillclimb.sh &
# Log:     /tmp/auto_hillclimb.log

ROUND_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CANDIDATE_DIR="$ROUND_DIR/results/candidates"
LOG_FILE="/tmp/auto_hillclimb.log"
SLEEP_SEC=600
THRESHOLD=3

# Signal handling — exit cleanly on SIGTERM/SIGINT
_clean_exit() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_hillclimb: received signal, exiting." >> "$LOG_FILE"
    exit 0
}
trap _clean_exit SIGTERM SIGINT

# Record starting count
prev_count=0
if [ -d "$CANDIDATE_DIR" ]; then
    while IFS= read -r -d '' f; do
        prev_count=$((prev_count + 1))
    done < <(find "$CANDIDATE_DIR" -maxdepth 1 -name '*.parquet' -print0 2>/dev/null)
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_hillclimb: started.  ROUND_DIR=$ROUND_DIR  initial_candidates=$prev_count" >> "$LOG_FILE"

while true; do
    sleep "$SLEEP_SEC"

    # Count current .parquet files
    curr_count=0
    if [ -d "$CANDIDATE_DIR" ]; then
        while IFS= read -r -d '' f; do
            curr_count=$((curr_count + 1))
        done < <(find "$CANDIDATE_DIR" -maxdepth 1 -name '*.parquet' -print0 2>/dev/null)
    fi

    delta=$((curr_count - prev_count))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_hillclimb: check  prev=$prev_count  curr=$curr_count  delta=$delta" >> "$LOG_FILE"

    if [ "$delta" -ge "$THRESHOLD" ]; then
        label="auto_$(date '+%H%M')"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_hillclimb: triggering hillclimb  label=$label  delta=$delta" >> "$LOG_FILE"

        cd "$ROUND_DIR"
        python orchestrator/hillclimb.py --label "$label" >> "$LOG_FILE" 2>&1
        exit_code=$?

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_hillclimb: hillclimb finished  label=$label  exit_code=$exit_code" >> "$LOG_FILE"
        prev_count="$curr_count"
    fi
done
