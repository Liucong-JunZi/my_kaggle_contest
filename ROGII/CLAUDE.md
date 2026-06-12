# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral Guidelines

These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked, no abstractions for single-use code,
  no "flexibility/configurability" that wasn't requested, no error handling
  for impossible scenarios.
- If 200 lines could be 50, rewrite it. "Would a senior engineer say this
  is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
- Touch only what you must. Don't "improve" adjacent code, comments, or
  formatting. Don't refactor things that aren't broken. Match existing style.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that *your* changes made unused;
  leave pre-existing dead code alone unless asked.
- Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step work, state a brief plan with per-step verification, then loop
independently until verified.

---

## Project Overview

Kaggle competition: predict **True Vertical Thickness (TVT, ft)** for each 1 ft
of a horizontal well's lateral, given the well's trajectory (MD/X/Y/Z), surface
formation depths, sparse horizontal GR log, and one vertical reference well
("typewell") with GR vs. TVT. Metric: RMSE in feet.

Per-well files (in `rogii-wellbore-geology-prediction/train/` and `test/`):
- `{WELL}__horizontal_well.csv` — trajectory + GR + formation columns + `TVT` (label) + `TVT_input` (NaN over the lateral, this is what we predict)
- `{WELL}__typewell.csv` — vertical GR vs. TVT for geological correlation
- `{WELL}.png` — visualization (chart axes are in **meters**, not feet)

The "known" segment of each well is the prefix where `TVT_input` is non-NaN.
Predictions are needed for the lateral rows (`TVT_input` is NaN).

## Common Commands

```bash
# Environment (conda)
conda env create -f environment.yml && conda activate rogii-cv
# Or pip-only:
pip install torch transformers h5py lightgbm scipy scikit-learn pandas opencv-python-headless

# Generate matching-grid HDF5 datasets (CONFIGS dict at top of file)
python src/gen_images.py
python src/gen_features.py            # tabular features (NPZ) for LightGBM

# Train SegFormer on a generated dataset (outputs metrics.json + best_model.pth into dataset dir)
python src/train.py --dataset data/cache/cfg-img-medium
python src/train.py --dataset data/cache/cfg-img-medium --backbone nvidia/mit-b1 --epochs 20
python src/train.py --dataset data/cache/cfg-img-medium-4ch --epochs 20

# Evaluate hengck23 reference checkpoint
python src/eval.py --ckpt docs/hengck23-reference/00004053.pth

# Run an experiment script (each round is self-contained under experiments/round_NNN/)
python3 experiments/round_008/r8_lgb_phase1.py 2>&1 | tee results/round_008/r8_lgb_phase1.log

# Kaggle CLI used for discussion mining (snapshots in docs/forum-snapshots/)
kaggle competitions topic-messages rogii-wellbore-geology-prediction <topic_id> -n -1 -s new
kaggle kernels pull <user>/<kernel-slug>
```

Device selection in `src/train.py` auto-picks `mps` → `cuda` → `cpu`. On Apple
Silicon (MPS) training is serial per card; don't try to fan out parallel
training jobs to the same device.

## Architecture

Two parallel modeling tracks share the same data layout and CV protocol; the
hybrid stack (round 7+) feeds the CNN's output into a GBDT.

### Track A — SegFormer + SDF (image track)

Files: `src/gen_images.py`, `src/train.py`, `src/decode.py`, `src/eval.py`.

1. **`gen_images.py`** builds 2D **matching grids** per well and writes HDF5:
   - `X (N, C, T, H)` — channels are `t_gr` (typewell GR replicated along H),
     `h_gr` (horizontal GR replicated along T), `history` (known-segment mask
     **must be the last channel** — `train.py` slices `X[:, C-1:C]`), and
     optionally `gr_diff`.
   - `y_sdf (N, 1, T, H)` — signed distance `(h_tvt[h] - t_tvt[t]) / sdf_scale`
     clipped to `[-3, 3]`. `sdf_scale=40` ft.
   - `y_tvt (N, H)`, `t_tvt (N, T)` — TVT axes for decoding (REQUIRED;
     re-generate any dataset missing `/t_tvt`).
   - `mask (N, H)`, `well_ids (N,)`.

   `CONFIGS` at the top of the file is the search space; named outputs go to
   `data/cache/<config_id>/{train,val}.h5 + config.json`. Width/compression
   constraint: keep `T ≥ 192` and `compression ≤ 24` or `t_tvt` degenerates to
   a constant for many wells (problem 5 in `.claude/skills/cv-orchestrator/SKILL.md`).

2. **`train.py`** is the single training entry point. `GeoSteerNet` =
   SegFormer (`nvidia/mit-b0/b1/b2`) + FPN fusion (`proj`/`fuse`) + a
   `fuse_history` Conv injecting the history channel into fused features +
   upsample head → `tanh * 3` SDF. Loss = masked MSE on `y_sdf`. Every epoch
   reports both **raw** RMSE (subpixel `argmin(|sdf|)` → `t_tvt` lookup) and
   **anchored** RMSE (R4-A partial anchor, see `decode.py`). The script keeps
   the anchored-best checkpoint.

   The legacy `src/eval.py` model class is a **separate** definition matching
   hengck23's `00004053.pth` key layout (has `fuse2`, `D`). Do not unify it
   with `train.py:GeoSteerNet` without re-saving checkpoints.

3. **`decode.py`** holds the canonical post-processing (R4-A):
   `decode_sdf_to_tvt` (subpixel parabolic argmin → linear interp into
   `t_tvt`) → `anchor_known_segment(alpha=0.75)` (per-well bias = mean of
   `tvt_pred − tvt_known` over the known prefix, shrunk by α) →
   `masked_rmse`. **Re-use these functions** rather than re-implementing
   decoding; α=0.75 is locked from R4-A2 sweep.

### Track B — Tabular / LightGBM (feature track)

Files: `src/gen_features.py`, `experiments/round_008/r8_lgb_phase1.py`,
`experiments/round_007/hybrid_sdf_lgb.py`.

**Critical target convention** (from LB 7.776 kernel, `docs/lb-references/ANALYSIS.md`):
the label is `target = TVT − last_known_TVT` (relative offset), **not** absolute
TVT. Inference adds `last_known_TVT` back. Using absolute TVT silently kills
LightGBM (~100+ RMSE vs. ~13 with relative).

Feature families that matter (per `docs/forum-snapshots/2026-06-12/INSIGHTS_UPDATE.md`):
- Geometry: `md_offset_from_last`, `z_rel`, `x_rel`, `y_rel`,
  `cumsum(-dz)` from last known (the dominant signal — `dtvt ≈ -dz`).
- Geometric tangents (test-safe, no leak): `sin/cos(dmd/dz)`, `sin/cos(dx/dy)`.
- Per-formation residuals: `Z − {ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA}`.
- GR rolling mean+std at windows `{5, 21, 51, 101}` after interpolation
  (raw GR is ~32% NaN on the lateral; interpolate or correlations collapse).
- Per-well static: `last_known_{tvt, z, gr}`, `n_known_rows`, `n_lateral_rows`.

### Cross-validation protocol

- **Group**: `GroupKFold(n_splits=5)` on `well_id`. Any other CV leaks across
  the well boundary.
- **Corpus**: the full 723 train wells. The hand-curated 50-well TRAIN/VAL
  split (`gen_images.py:TRAIN_IDS` / `VAL_IDS`) is a **narrow distribution
  that flatters models** — anything tuned against it overstates by ~2× when
  run on a random split (rounds 4-B/6 confirmed this).
- Report **per-well mean RMSE** (LB metric is flat RMSE; both should track).

## Experiments Layout

Each round is a dated, self-contained directory:
- `experiments/round_NNN*/` — design notes (`notes.md`) + one-shot scripts.
- `results/round_NNN/` — generated `metrics.json`, logs, parquet artifacts.
- `results/summary.json` — aggregated config index.

Current state (see `README.md` table for the full matrix):
- **Best image-only**: `cfg-img-medium` SegFormer mit-b0, raw 15.84 / anchored
  **13.67** ft on curated 50-well val. Fully reproducible via
  `src/train.py --dataset data/cache/cfg-img-medium`.
- **Active line**: Round 8 — LightGBM with relative target on full 723 wells
  (`experiments/round_008/r8_lgb_phase1.py`). Expected CV ~11 / LB ~13.
- **Dead ends** (do not retry without a new reason): `mit-b1` backbone on 50
  wells, `gr_diff` channel (leak), naive data scaling without sampling for
  distribution match, soft-argmin + Huber TVT loss (R5-A, gradient conflict
  with SDF MSE), CNN GR window matchers/rankers (msg 3465758: "the GR just
  repeats too much to localize").

## Reference Docs (read before changing strategy)

- `docs/competition-insights.md` + `docs/forum-snapshots/2026-06-12/INSIGHTS_UPDATE.md`
  — distilled forum findings. The latter is current and lists the
  game-changers (relative target, geometric tangents, PF baseline at LB 8.863,
  oracle plane fit at 4.82 ft).
- `docs/lb-references/ANALYSIS.md` — breakdown of the LB 7.776 public kernel
  (`smartorz/lb-7-776-rogii-ridge-sp`): a 7-signal stack (PF×2 + 14-beam +
  multi-scale NCC + per-formation + dense ANCC + selector) blended via
  LightGBM/CatBoost/Ridge.
- `docs/hengck23-reference/` — original SDF approach + pretrained `.pth`
  weights (`eval.py` runs them directly).
- `.claude/skills/cv-orchestrator/SKILL.md` — multi-agent orchestration spec
  and a "known issues" log worth reading (SDF→TVT conversion, MPS serial
  training, t_tvt degeneration, agent spawn quirks).

## Data Quirks To Remember

- `TVT_input` is NaN on the lateral; this is what's predicted. The known
  prefix anchors everything.
- PNG charts use **meters**; CSVs use feet.
- 57 unique typewell IDs appear in PNGs but 752 unique `__typewell.csv`
  files exist — don't trust PNG labels for joining.
- The organizer excluded one private outlier well from scoring (msg 707695);
  don't tune to outliers.
