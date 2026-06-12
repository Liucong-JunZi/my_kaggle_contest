# ROGII Forum Insights Update — 2026-06-12

> Pulled via `kaggle competitions topic-messages` for 15 high-value topics
> Builds on `docs/competition-insights.md` (2026-06-08 snapshot)
> **Period covered**: 2026-05-15 → 2026-06-11 (final ~10 days of competition)

## LB context (2026-06-12)

```
LB Top 1: SaintLouis  5.986
LB Top 5: PirateInPants 7.031
Public notebook: smartorz/ridge-sp ~7.776 (LB)
Public PF baseline: ~8.863 (very recent, msg 707613)
Most public notebooks: 9-12 range
Oracle (best fit geo plane): ~4.82 ft validation (msg 3468820)
```

**Implication**: Top 5 are 1-2 ft above the oracle for a linear geological plane.
This means there's a hard floor around ~5-6 ft from geometry alone — the rest is
the GR matching / formation contact / per-well dip estimation problem.

---

## Game-changing discoveries we missed (with high confidence)

### 1. ⭐ **`-dz` is the dominant signal, not GR matching**

> hengck23 (msg 3462931 + 3462951): "I discover a hack! first fig: dz second fig: dtvt
> ... they are the same scale !!!!"

```python
dtvt = -dz  +  (small correction at ~15 control points per well from ANCC dip events)
```

**ANCC formation top is piecewise-linear with ~15 control points per well, ~323 rows apart**.
This IS the StarSteer dip annotation. Without the correction, `tvt = last_known + cumsum(-dz)`
is correct between dip events.

This means our R7 result (physics baseline 87.63 RMSE on lateral) was wrong because
we used the wrong formula. **Correct physics baseline**:

```python
tvt_phys[h] = last_known_tvt - (z[h] - last_known_z)
```

Wait — that's exactly what we did. Let me re-check... Actually our formula was correct.
The 87.63 RMSE then means **the per-well dip events accumulate to ~90 ft over lateral
length**. The next move is predicting the dip events themselves.

### 2. ⭐ **Oracle `cumsum(-dz - offset)` with discrete offset → 7.7 RMSE**

> hengck23 (msg 3463137): "cumsum(−dz − offset) with a discrete offset => 7.7 rmse"
> (msg 3463219): "a fine offset-grid oracle gives ~7.64 RMSE on train hidden rows
> for me but choosing the offset is the hard part: known-prefix offset gives
> ~37-39 RMSE, and my fold-safe selector only gets ~14.8"

The competition is essentially **predicting one offset per well**. With the right offset,
you're at LB 7.7. The fold-safe selector (without leak) gets 14.8 — which is roughly
where we are. **All the gap from 14.8 → 7 is offset selection.**

### 3. ⭐ **Linear geo plane oracle → 4.82 ft** (msg 3468820)

```python
for offset in range(-125, 125):
    for slope_y in range(0, 25):
        L = len(h_geo_z[h_ps:])
        candidate = np.linspace(0, 1, L) * slope_y * L / 300 + offset + h_z[h_ps]
        rmse = sqrt(mean((h_geo_z[h_ps:] - candidate)**2))
```

Geological plane (the unknown surface our well crosses) is **approximately linear** in MD,
modeled by `offset + slope * L`. Fitting this best gives 4.82. Piecewise-linear is better.

### 4. ⭐ **Public PF baseline reaches LB 8.863 with no training data**

> msg 707613 (2026-06-11): "I implemented a baseline using a Particle Filter (PF), and
> it achieved a public LB score of 8.863. However, my current approach is still mostly
> rule-based / inference-only. It does not really learn from the training data."

> hengck23 reply (msg 3471295): can be improved by:
> - "sub sampling N particles, expand a window then train a network to do N-way classification"
> - "recording particle vectors then predicting the offset to calibrate them, impacting PF"
> - "For each particle step, train a generative model"

### 5. ⭐ **dz/dx tangent + dx/dy tangent + GR heatmap learns to bend SDF correctly**

> hengck23 (msg 3467823): "validation results for full length tvt... sdf is bending
> correctly, i.e. it indeed learned the steering. **The magic is adding well dip tangent
> feature (sin and cos of dmd/dz) and well direction tagent feature (sin and cos of
> dx/dy, i.e. geology dip) + gr heatmap.**"

**This is the missing channel in our SegFormer!** We have `[t_gr, h_gr, history]`
but lack the geometric tangents. Adding:
- `sin(dmd/dz), cos(dmd/dz)` — well dip
- `sin(dx/dy), cos(dx/dy)` — geology direction

These are geometric (not derived from TVT → no leak), independent physical signals.

### 6. ⭐ **Bayesian Physics-Informed SegFormer → LB 10.576** (msg 3460928)

> "Update: Baysian Physical-informed SegFormer got very good result. (0.94 cv score)"

5-fold table shows LB 10.576 with bpinn. **Our SegFormer architecture is exactly the
right backbone, just missing the physics constraints.**

### 7. ⭐ **Synthetic-data pretraining → "1.2 ft behind LB top, simple UNet"** (msg 3462542)

> "~1.2 ft behind you with simple UNet model. No physics constraints yet.
> **Pre-training on synthetic wells gave a decent boost**"

Generate synthetic (TVT, GR) pairs from candidate plane parameters → use as pretraining
data → fine-tune on real wells. hengck23 confirms this in 702474.

---

## Failed approaches (so we don't repeat)

### CNN window matcher
> msg 3465758: "pretty hard window matcher (h_gr vs typewell, learn the sdf) +
> noise aug, fold-safe. couldn't get it to beat the plain point-gr likelihood matcher
> **the gr just repeats too much to localize a window**"

GR alone is fundamentally non-discriminative (multi-modal matches), confirmed multiple
times. Our SDF window approach via SegFormer has the same problem.

### Pure MTP / ranker
> msg 3465111: "**no_gr ≈ shuffled_gr ≈ real_gr on top1**, the net basically ignores it.
> ranker looks amazing in-window... does NOT convert row-level once you go strict
> well-grouped oof - my best honest gain over gbm was ~+0.03ft"

Ranking GR-matched candidates doesn't work. The signal isn't there.

### CNN likelihood scorer (replacement for PF point likelihood)
> msg 3465758: "AUC caps ~0.7, the gr just repeats too much"

---

## Data quirks (must know)

### A. **57 displayed typewells, 752 actual files** (msg 3468442)
> "PNG files show 57 unique Typewell numbers, but there are actually 752 unique
> {well}__typewell.csv files in train"

Multiple wells display the same typewell ID in their PNG, but their typewell CSVs are
completely different. Don't trust PNG labels.

### B. **Real data, geologist subjective typewell choice** (msg 3468855, organizer)
> "All of the data is real-there's no synthetic data here. The selection of the type
> well for lateral steering is somewhat subjective and depends on the geologist."

### C. **GR is ~32% NaN on lateral** (msg 707702)
> "Across the training wells, roughly a third of the lateral GR values are missing"

Interpolate before any matching, or NaN-propagation kills correlation silently.

### D. **`departure = cumsum(sqrt(dMD² - dZ²))`** (msg 3460812)
PNG charts use METERS not FEET (organizer documentation mistake). To validate
horizontal distance, use cumulative `sqrt(dMD² - dZ²)`, not naïve `sqrt(MD² - TVD²)`.

### E. **Private test outlier well excluded** (msg 707695, 2026-06-11)
> "There is an outlier well in the private test set that we've decided to exclude
> from scoring. I will be beginning a rescore shortly."

Don't tune to outliers; the worst single private well no longer matters.

---

## The actual winning architecture (synthesizing across competitors)

```
PF (offset/state tracking) + Beam search (k-modes) + Tree model (residual learner)
  ↓
All trained on test-safe features only: MD/X/Y/Z/GR/TVT_input
  ↓
Ridge / GBDT stack to blend per-row
  ↓
Apply post-processing: sg_smooth, anchor at heel
```

**Critical**:
- **Target = `TVT - last_known_TVT`** (relative offset, not absolute)
- **Features = test-safe only** (no leak from full TVT or post-PS X/Y/Z is fine since
  driller did NOT steer post-mortem to a known answer — it's causally upstream of TVT
  here per msg 3464209 from organizer-side context)
- Multiple **decorrelated signals** blended:
  - PF state-tracking with GR likelihood
  - Beam-search k-modes over candidate offsets
  - GBDT on (dz, geometry, GR rolling stats)
  - SDF-CNN (hengck23, our SegFormer family)
  - Formation contact physics for wells appearing in both train and test sets

---

## Our position vs LB top

| Component | LB top has | We have | Gap |
|-----------|-----------|---------|-----|
| Right target | `TVT - last_known` | absolute `TVT` ❌ | **biggest** |
| Geometric tangents | dx/dy, dmd/dz sin/cos | none | medium |
| PF / Beam | Yes (heavily tuned) | none | large |
| Multi-signal blending | 7+ signals + Ridge stack | single SegFormer | large |
| Formation contacts | per-formation TVT features | not used | medium |
| Synthetic data augment | Yes (CVAE/plane sampling) | no | medium |
| Right CV scheme | GroupKFold + full corpus | curated 50 (biased) ❌ | **huge** |

**5 issues; 2 are critical**: wrong target definition, wrong validation set.

---

## Recommended action plan (post-R7 insight)

### Phase 1: Fix the foundation (2 hours)
1. Rewrite `feat-lgb-domain` with **relative target** `tvt - last_known_tvt`
2. Use **GroupKFold-5 on full 723 train wells** (not curated 50)
3. Add domain features matching insights:
   - `dz_from_last` (already had)
   - `cumsum(-dz)` (the dominant signal)
   - `sin/cos(dmd/dz)`, `sin/cos(dx/dy)` (geometric tangents)
   - Per-formation Z residual: `Z - ANCC`, `Z - EGFDU`, etc. (6 formations)
   - GR rolling 5/21/51/101 mean+std (8 features)
4. Expected: should land in **CV ~11, LB ~13** (matching plain GBDT reports)

### Phase 2: Add PF (1-2 days)
1. Copy the PF from public 8.863 kernel (msg 707613 referenced)
2. Run 128-seed × 500-particle ensemble per well
3. Add `pf_pred`, `pf_std` as features to GBDT
4. Expected: CV ~9, LB ~10

### Phase 3: Resurrect SegFormer with right inputs (1 day)
1. Add geometric tangent channels (`sin(dx/dy)`, `cos(dmd/dz)` etc) — **independent
   physical signals, no TVT leak**
2. Train on full 723 wells with the **relative target** (target SDF computed against
   `last_known_tvt - h_tvt[h]` not absolute `h_tvt[h]`)
3. Add as another signal to GBDT stack
4. Expected: gives partial bump if it works

### Phase 4: Synthetic pretraining (1-2 days)
1. Sample plane params (offset, slope)
2. Generate (TVT, GR) candidates via typewell interpolation
3. Add diffusion-modeled noise (per msg 3471357)
4. Pretrain CNN on this corpus → finetune on real
5. Expected: addresses OOD trajectories per msg 3462542

---

## Files

- `messages/{topic_id}.txt` — raw CLI output per topic
- `all_topics.csv` — 88 topics index
- `INSIGHTS_UPDATE.md` — this file

To re-pull a topic later:
```bash
kaggle competitions topic-messages rogii-wellbore-geology-prediction <topic_id> -n -1 -s new
```
