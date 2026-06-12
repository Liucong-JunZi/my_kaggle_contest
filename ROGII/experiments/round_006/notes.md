# Round 6 — Full-corpus retrain (refuting "data scaling is dead")

**Date**: 2026-06-12
**Branch**: `round-5-tvt-aware-loss` (will branch on launch)
**Baseline to beat**: cfg-img-medium R4 = **raw 15.84 / anchored 13.67 ft** (50-well hand-curated)

## Motivation — re-reading R2/R4-B failures

R2/R4-B were marked "data scaling dead" in R4 notes. Re-read with fresh eyes:

> "The original TRAIN_IDS in src/gen_images.py was **hand-curated to match val geology**;
> sorted-alphabetical extras don't."

So the failure was **dilution of a hand-curated subset by alphabetical noise**, not
data scaling per se. Adding 50-150 noisy alphabetical wells to 50 curated ones gave
the model an inconsistent training signal.

**Hypothesis for R6**: train on FULL 723 non-val wells. With this much data the
model must learn invariant features rather than overfitting curation. If the
hypothesis holds, raw RMSE drops below 15.84 and anchored below 13.67. If not,
we confirm something else (val distribution itself off, data quality issues).

## Setup

- **Dataset**: `cfg-img-full` — 723 train + 50 val (same VAL_IDS as baseline).
- **Geometry**: identical to cfg-img-medium (T=192, H=576, comp=24, 3 channels).
- **Model**: SegFormer mit-b0 (b1 was capacity-saturated at 50 wells, may now help).
- **Epochs**: 20 (baseline best @ ep10; with 14.5× data, expect best later).
- **Batch size**: 2 (same as baseline — keep apples-to-apples; 4 if mps fits).
- **Eval**: R4-A pipeline (subpixel + α=0.75 anchor), anchored-best ckpt selection.

## Expected timing

- 50-well epoch: ~3.8 s | 723 wells ≈ 14.5× ≈ 55 s/epoch | 20 epochs ≈ 18 min
- Plus eval (50 val) per epoch: ~1 s | total ≈ 20 min wall

## Decision tree

| Result | Interpretation | Next |
|--------|---------------|------|
| raw < 13, anc < 11 | ⭐ Big win — data was the bottleneck all along | Try mit-b1/b2 with full data; ensemble |
| raw 13-15, anc 11-13.5 | Moderate win — both data + curation matter | Try clustered subsampling, k-fold ensemble |
| raw ≈ 15-16, anc ≈ 13-14 | Flat — data axis confirmed dead (but for new reason: model capacity) | Pivot to architecture / pretrained init |
| raw > 17, anc > 14 | Worse — distribution shift between corpora is real | Re-examine val split |

## Early diagnosis (epochs 1-3, ~3 min in)

```
ep1 | loss=0.3405 | raw=29.16 | anc=40.84  ⭐ NEW BEST
ep2 | loss=0.2441 | raw=26.96 | anc=34.72  ⭐ NEW BEST
ep3 | loss=0.2416 | raw=26.56 | anc=38.80
```

**train_loss 0.24 is already at baseline-converged 0.20**, so the model
fits 723 wells fine — but val raw 26-29 is far from baseline 15.84, and
**anc > raw is the same smoking gun as R4-B-200-fair**.

### Re-interpretation

R4-B notes assumed "extras don't match val distribution". Re-read more
carefully: **our VAL_IDS were also hand-curated**, drawn from the same
narrow geological window as TRAIN_IDS. Hence:

```
50 curated TRAIN ─same window─ 50 curated VAL  → easy 13.67 ⭐
           ↕↕ different window ↕↕
              723 full corpus
```

Implications:
1. Our 13.67 baseline is **likely overestimated**: it's RMSE on a narrow
   slice of the corpus, not the corpus as a whole.
2. Kaggle's hidden test is probably drawn from the broader corpus →
   our LB score is likely much worse than 13.67 (closer to 25?).
3. LB-top (~7 ft) competitors trained on full data with full-corpus val,
   not on a tight hand-curated subset.

## R6-B (queued) — proves the curation effect

After R6-A finishes: re-train baseline with the SAME geometry but
**random 50 val from full corpus** instead of hand-curated VAL_IDS.

If raw RMSE jumps from 15.84 → 25+, that's direct proof the baseline
score reflects val choice, not model quality.

## Files

- `experiments/round_006/gen_full.py` — full-corpus dataset generator
- `experiments/round_006/gen_full.log` — generation log
- `data/cache/cfg-img-full/` — train.h5 + val.h5 + config.json
- `results/round_006/cfg-img-full.{json,log}` — training metrics + log
- (queued) `experiments/round_006/gen_random_val.py` — R6-B 50/50 random split
- (queued) `results/round_006/cfg-img-randval.{json,log}` — R6-B metrics
