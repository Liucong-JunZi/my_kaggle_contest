# Round 010 — Hill climb v5 with T4 MLPs: diagnostics & decision

**Date**: 2026-06-17 dawn (autonomous run while user asleep)

## TL;DR

- **T4 kernel succeeded** after 3 push iterations (file path glob, then `--accelerator NvidiaTeslaT4`).
- **5 MLP candidates trained** (c60-c64), single-fold perwell 10.13–10.70 (kernel-side).
- **Hill climb v5 honest OOF = 7.0127 — identical to v4** (no improvement). c60–c64 all ranked out of top 10 weights.
- **NNLS stacking on the same pool drops OOF to 6.9792**, but the gain comes from sum_w ≈ 1.18 amplification of predictions (pool std=9.13 vs target std=15.83 — every candidate shrinkage-biased), **not** from the new MLPs.
- Removing all 5 MLPs from NNLS pool → still 6.9792 (identical to 4 decimals). MLPs contribute exactly **0** to NNLS-based blending.
- **Decision**: do not submit. Hill climb v4 (7.0127, sum_w=1) remains the safest LB candidate; NNLS gain is amplification artifact, not robust signal.

## Single-fold OOF metrics (kernel-reported, importer-confirmed)

| cid                 | perwell  | flat   | wall  | corr_c20 | corr_resid |
|---------------------|---------|--------|-------|----------|------------|
| c60_mlp_s42         | 10.1484 | 13.01  | 3432s | 0.6456   | 0.3283     |
| c61_mlp_s7          | 10.1550 | 13.02  | 4058s | 0.6456   | 0.3294     |
| c62_mlp_s2024       | 10.2387 | 13.17  | 3525s | 0.6450   | 0.3134     |
| c63_mlp_huber_s42   | 10.1326 | 13.09  | 3415s | 0.6458   | 0.3212     |
| c64_resmlp_s42      | 10.7019 | 13.65  | 4727s | 0.5993   | 0.3173     |

Total kernel wall: 19,157s (5h20min on Kaggle T4). MLPs are ~2× weaker than ravaghi LGB (perwell 8.0) but show real diversity (corr_c20 ≈ 0.65 vs in-pool corr c20-vs-c51 = 0.69).

## Honest OOF comparison (full pool 38–43 candidates)

| Method                              | OOF     | Δ vs v4    | sum_w (avg) | LB risk |
|-------------------------------------|---------|------------|-------------|---------|
| Hill climb v4 (greedy, no MLPs)     | 7.0127  | (baseline) | 1.000       | low     |
| Hill climb v5 (greedy, with MLPs)   | 7.0127  | 0.0000     | 1.000       | low     |
| NNLS (no normalize), no MLPs        | 6.9792  | -0.0335    | 1.184       | medium  |
| NNLS (no normalize), with MLPs      | 6.9792  | -0.0335    | 1.184       | medium  |
| NNLS (normalize sum_w=1)            | 7.1278  | +0.1151    | 1.000       | low     |
| Ridge α=1, positive=True, intercept | 6.9855  | -0.0272    | 1.184       | medium  |
| Ridge α=1+, positive=False, interc. | 7.1976  | +0.1849    | 1.040       | low     |

`corr_resid` interpretation: 0.32 means c6x explains ~10% of (y - c20_pred) variance — non-trivial diversity. But hill climb's greedy step rejects them because their single-cid perwell is too far from the leaders.

NNLS gives them ~0 weight not because they are redundant, but because the pool already has enough degrees of freedom for sum_w optimization to extract amplification benefit without invoking new directions.

## Why MLPs don't help

The pool's predicted-std/target-std = 0.577. **Every existing candidate underpredicts magnitude** (shrinkage bias). NNLS exploits this by setting sum_w > 1, which is a 1-DOF "amplification" trick that any combination of weak-but-correlated candidates can do. Adding MLPs gives no new DOF for this particular optimization.

To actually leverage MLP diversity, we would need either:

1. **Stacking with non-linear meta-learner** (LGBM second-stage) — captures interactions that linear blends miss.
2. **MLPs trained on disjoint feature sets** — break the corr=0.65 correlation by giving them genuinely different inputs (e.g. only GR rolling features, only PF outputs, only geometric).
3. **Calibration in pool-construction stage** — fit per-candidate scaling on training fold so sum_w=1 doesn't lose magnitude information.

(3) is the cleanest fix and would make NNLS' 0.034 gain "real" / LB-translatable.

## Files

- `experiments/round_010/results/candidates/c60_mlp_s42.parquet … c64_resmlp_s42.parquet` — 5 candidates registered ✓
- `experiments/round_010/results/hillclimb_runs/run_v5_with_t4mlp.json` — hill climb 7.0127
- `experiments/round_010/results/hillclimb_runs/nnls_v1_full_pool.json` — NNLS 6.9792
- `experiments/round_010/results/hillclimb_runs/nnls_v1_normed.json` — normalized NNLS 7.1278
- `experiments/round_010/orchestrator/stacking_nnls.py` — new NNLS blender script
- `experiments/round_010/submission_package/t4_train_kernel/{train_t4.ipynb, train_t4_source.py, kernel-metadata.json, make_notebook.py}` — Kaggle kernel artifacts
- `experiments/round_010/t4_out/{c6X.parquet, summary.json, rogii-t4-train-mlp.log}` — kernel output (5 parquets + log)

## Suggested next steps (not auto-executed)

In priority order:

1. **Ratio-calibrate pool predictions** — for each candidate, fit per-fold scalar α so that std(α·pred_train) = std(target_train). Re-run hill climb on calibrated pool. If sum_w=1 hill climb then beats 7.0127, the gain is real.
2. **Train MLPs on disjoint feature subsets** (GR-only, PF-only, geometric-only) — would push corr below 0.5 and create true new DOF that linear blender must use.
3. **2nd-stage LGBM stacker** on top of pool — captures non-linear interactions; standard practice when 1st-stage diversity is high. ~1h to wire up locally.

Steps 1+3 are local-CPU work, step 2 needs another Kaggle kernel run.
