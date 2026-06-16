#!/usr/bin/env bash
# Queue runner for untrained round_010 candidates.
# Diversity-first ordering: linear → HGB → Extra Trees / RF → loss variants → boosting variants → CAT/LGB depth variants.
set -u  # NOT -e — one failure must not kill the queue.

cd /Users/liucong/code/kaggle/ROGII/experiments/round_010
LOGDIR=results/candidates_logs
mkdir -p $LOGDIR

QUEUE=(
  c40_ridge_v14
  c44_hgb_mitch
  c45_etr
  c46_rf
  c41_lgb_quantile_p50
  c42_lgb_mae
  c43_lgb_huber_a05
  c47_lgb_dart
  c48_lgb_goss
  c49_xgb_huber
  c33_cat_super_lr025
  c34_lgb_pilkwang_d127
  c35_cat_pilkwang_d8
  c36_lgb_nina_d127_lr04
  c37_cat_nina_d8_lr04
  c38_lgb_dwt_s7
  c39_cat_dwt_s7
)

OK=0; FAIL=0
echo "=== Queue: ${#QUEUE[@]} candidates ==="
for cid in "${QUEUE[@]}"; do
  if [[ -f results/candidates/$cid.parquet ]]; then
    echo "[skip] $cid — already trained"
    continue
  fi
  echo "[run] $cid …"
  t0=$(date +%s)
  if /Users/liucong/miniconda3/bin/python3 orchestrator/train_one.py "$cid" > "$LOGDIR/$cid.log" 2>&1; then
    perwell=$(grep "OVERALL" "$LOGDIR/$cid.log" | tail -1 | grep -oE 'perwell=[0-9.]+' | head -1)
    dt=$(( $(date +%s) - t0 ))
    echo "  ✓ $cid  $perwell  (${dt}s)"
    OK=$((OK+1))
  else
    dt=$(( $(date +%s) - t0 ))
    echo "  ✗ $cid  FAILED (${dt}s) — see $LOGDIR/$cid.log"
    tail -5 "$LOGDIR/$cid.log" | sed 's/^/      /'
    FAIL=$((FAIL+1))
  fi
done
echo ""
echo "=== Queue done: ok=$OK fail=$FAIL ==="
