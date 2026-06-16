# ROGII Public Resource Harvest — SUMMARY

**Agent**: HARVEST sub-agent
**Run window**: 2026-06-15 00:46 → ~01:40 (in progress, ~55 min wall clock)
**Status**: Phase 1 mostly drained (kernels still streaming); Phase 2/3 staged for the high-value top ~25 kernels

---

## 1. Totals

| Stream | Listed | Pulled | Failed | Skipped (>500MB) | Description-only |
|--------|--------|--------|--------|------------------|------------------|
| Kernels | 684 | ~620 (still streaming) | 0 | n/a | n/a |
| Datasets | 36 | 26 | 0 | n/a | 10 |
| Forum topics | 91 | 76 new (15 already in old snapshot) | 4 (now fixed) | n/a | n/a |

**Disk used**: kernels_raw 61MB + datasets_raw 4.3GB + forum_raw <1MB = **~4.4GB** of ~100GB budget.

---

## 2. Top 5 kernels by LB score / votes (1-line summaries)

1. **fleongg/fle3n-rogii-v5** (LB **7.528**, ~80 votes — the strongest LB-documented public kernel) — Two-engine blend `0.55·EngineA + 0.45·EngineB` (Engine A = LB 7.776 Ridge-SP, Engine B = LB 7.810 Drift-PF) producing LB 7.540, plus a **measured-optimum interpretation hedge** at weight 0.5 for gate-verified duplicate wells lifting to LB 7.528. Documents the duplicate-well leak history (org killed it; residual hedge captures the rest).
2. **lightningv08/lb-7-776-rogii-ridge-sp** (LB 7.776, 86 votes) — The reference: 3LGB+2CB Ridge stack on 100+ feature pipeline (PF-ANCC + PF-Z + 7-beam + multi-scale NCC + plane-KNN + dense ANCC + 4× anchored offset families) blended 0.3/0.7 with a 128-seed-PF heuristic selector. **Already documented in `docs/lb-references/ANALYSIS.md`.**
3. **nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based** (LB 9.251, **593 votes — most-upvoted public kernel**) — LB7.776 stack PLUS a **DTW alignment family** (4-radius Sakoe-Chiba + 12-realization stochastic DTW with Gumbel-noise traceback). Optuna-tuned PP including SG window/order. **NEW SIGNAL FAMILY.**
4. **pixiux/rogii-dual-pipeline-blend** (LB ~9, 201 votes) — Two independent pipelines blended 0.55 / 0.45 + a **guarded physical override** that's runtime-verified before applying (per-well prefix RMSE < 1 ft, ≥50 rows). Adds a **robust deg-4 IRLS U-space projection** (CV-validated −0.09 RMSE vs SG smooth).
5. **medali1992/rogii-tcn-train-with-ddp-layernorm-se-atten** (LB ~10-11, 78 votes) — Deep sequence model: 6-block dilated TCN with **LayerNorm** (not BN), **SE channel attention**, **typewell cross-attention** (TCN queries typewell GR keys/values), and **global Transformer over stride-50 pooled view**. Documents reducing CV-LB gap **6.32 → 0.27 ft** by forcing the formation imputer unconditionally at training time.

Honorable mentions:
- **kojimar/rogii-inference-stack-with-pf-beam-and-tabicl** — Inference-only artifact stack where **TabICL_A gets 71% of the stack weight**.
- **mitchgansemer/gr-features-outlier-detection** — pairwise estimator-divergence features (top-20 SHAP) + db4 wavelet GR decomposition.
- **ravaghi/wellbore-geology-prediction-hill-climbing** — replaces Ridge stack with hill-climbing weights (allows negatives).
- **horizen12/cnn-transformer-mtp-baseline** — extends hengck23's CNN-MTP with a Transformer head (multi-trajectory prediction).
- **sanidhyavijay24/9-946-rogii-geostat-softmax-ncc-hybrid** — adds **visible_gr_shift_fit** (per-well brute-force TVT shift estimator) and **GR FFT features**.

---

## 3. Top 10 unique-vs-our-stack techniques (ranked by likely impact)

| Rank | Technique | Source | Why impactful for our R10 stack |
|------|-----------|--------|--------------------------------|
| 1 | **TabICL Regressor** | kojimar / thbdh5765 v10 artifacts | Dominates GBDT in the v10 LB ~9.5 stack (71% weight). Provides strongly orthogonal predictions to LightGBM/CatBoost via prior-fitted in-context learning. Requires `tabicl` Python package + `needless090/rogii-tabicl-mirror` checkpoint dataset (101 MB, downloaded). **Likely +0.3–0.5 RMSE if it stacks well with our existing GBDT.** |
| 2 | **Force formation imputer at training time too** (CV-LB gap fix) | medali1992 TCN | Per the kernel notes: closed CV-LB gap from **6.32 → 0.27 ft**. Direct addresses our R8 distribution shift problem. **Drop self-well formation columns at train, use FormationPlaneKNN for both train+test.** Likely the single biggest LB lift available to us. |
| 3 | **DTW (Sakoe-Chiba) signal family** | nihilisticneuralnet 9-251 | Multi-scale (radii 20/50/100/200) + 12-realization stochastic DTW. Adds 20+ orthogonal features (DTW ensemble TVT, path slopes, std/cv, anchored offsets). Numba-JIT mandatory. |
| 4 | **U-space (TVT+Z) anchor-relative robust projection** | pilkwang / pixiux | Per-well robust deg-4 polynomial fit in U=TVT+Z space, β=0.75 blend. Distinct from SG smoothing — projects globally rather than locally. CV-validated −0.09 RMSE per pixiux. |
| 5 | **Guarded physical override** | pixiux / lightningv08 | Per-well runtime verification (prefix RMSE < 1 ft, ≥50 rows) before applying `tvt_from_contacts` to overlap wells. Provably never worse than the plain blend. |
| 6 | **Estimator divergence pairwise features** | mitchgansemer | 11 cheap features encoding "this is a hard well" via pairwise differences among (form, ncc, beam, pf, extrap) drifts. Top-20 SHAP per author. |
| 7 | **Two-pipeline 0.55/0.45 blend** | pixiux | Build the entire feature pipeline twice with different code/seed and blend. "Decorrelated errors → free accuracy" (per author's CV). |
| 8 | **Wavelet (db4 5-level) GR features** | mitchgansemer | Multi-resolution GR decomposition (level-3 detail energy + low-freq approximation + high-freq residual). Captures strata structure simple rolling stats miss. |
| 9 | **Visible-GR shift fit + GR FFT** | sanidhyavijay24 | Two cheap per-well features: brute-force TVT-shift correlation maximizer (-30..30 ft, step 2) + dominant GR FFT frequency. Quick wins. |
| 10 | **Azimuth-weighted plane-KNN** | aliafzal9323 | Reweight FormationPlaneKNN neighbors by `cos(Δθ)^β` (β=0.6) to favor wells drilled along similar azimuths. Domain-knowledge informed; principled add-on. |

---

## 4. Staged drop-in candidates in `experiments/public_harvest/` (10 stubs)

| File | What it provides | Source kernel |
|------|------------------|---------------|
| `feat_dtw_sakoe_chiba.py` | Multi-scale DTW + stochastic DTW signals (20+ features per well) | nihilisticneuralnet 9-251 |
| `feat_estimator_divergence.py` | 11 pairwise/aggregate divergence features (cheap, top-SHAP) | mitchgansemer |
| `feat_formation_plane_knn.py` | Spec/scaffold for `FormationPlaneKNN(train_well_ids, data_dir, k=10)` | LB7.776 |
| `feat_formation_segment_b_well.py` | Per-formation TVT × 5 segment-bias variants (30+ features per well) | LB7.776 |
| `feat_gr_detrend.py` | GR linear-detrend residual + first derivative | romantamrazov super-solution |
| `feat_gr_dwt.py` | db4 wavelet GR features (approx5, detail_energy, residual) | mitchgansemer |
| `feat_gr_fft.py` | Per-well dominant-frequency + log-power | sanidhyavijay24 |
| `feat_pf_ancc_pf_z.py` | Spec/scaffold for the twin PF estimators (numba JIT funcs from source) | LB7.776 |
| `feat_visible_gr_shift_fit.py` | 3 per-well shift/correlation/bias features | sanidhyavijay24 |
| `model_tabicl_kojimar.py` | TabICL regressor wrapping (fit_fold + predict) | kojimar |
| `postproc_u_space_projection.py` | Robust deg-4 IRLS polynomial projection in U=TVT+Z space | pilkwang / pixiux |
| `postproc_guarded_physical_override.py` | Verified-at-runtime per-well physical override | pixiux |

All stubs are **STUB ONLY** — parent agent should review and selectively integrate. The numba JIT functions for PF and DTW are referenced (not copied verbatim) to keep the stubs auditable.

---

## 5. Surprises / red flags / leaks discovered

### Surprises

- **TabICL dominates the v10 stack (71% weight)**. We had not previously identified this — most ROGII insights focus on PF/beam/NCC families. TabICL via in-context learning produces strongly orthogonal predictions to GBDT.
- **medali1992's CV-LB gap fix (6.32 → 0.27 ft)** is a documented lesson the public hasn't widely circulated: at training time, drop the real formation columns and use the imputer for both train AND test. This addresses our `MEMORY.md` `rogii-formation-col-leak.md` concern directly.
- **sunnywu27's GR interpolation lift (5.95 → 4.71 ft)** — simple `pd.Series.interpolate(limit_direction='both')` before PF gives 1.24 ft RMSE on PF alone. Many public kernels miss this.
- **TabICL/PFN-style models work well on the per-row residual target** despite the small-tabular paradigm.

### Red flags

- **Some public kernels apply the physical override blindly** (no prefix verification) and silently regress on misaligned wells. pixiux's guarded version is the safe pattern.
- **Forum thread 701691 explicitly states**: "A lot of the public notebooks have leakage which leads to a smaller CV-LB gap, but is not as trustworthy." Our 2-3 ft CV-LB gap is the **honest** gap; smaller gaps in public forks are likely leaking.
- **The `gr_diff` channel** (per CLAUDE.md) is a known leak; verified by multiple kernels avoiding it.

### Leaks discovered

- **Self-well formation column training leak**: training with real `ANCC/ASTNU/EGFDU/...` while test has imputed values. Already in our memory; medali's fix codified above.
- **Row-index physical override**: blindly using train-row[i].TVT for test-row[i] without MD-aligned interpolation can silently inject errors. pixiux's guarded version fixes this.

---

## 6. Disk used (cumulative)

- `experiments/public_resources/datasets_raw/` ≈ **4.3 GB** (26 datasets <500 MB each)
- `experiments/public_resources/kernels_raw/` ≈ **60 MB** (~620 kernel notebooks)
- `experiments/public_resources/forum_raw/` ≈ **0.5 MB** (76 forum-message text files)
- **Total**: ~4.4 GB of 100 GB budget. Plenty of headroom.

The ~10 datasets >500 MB skipped have description stubs in `experiments/public_resources/datasets/` flagging:
- `ravaghi/wellbore-geology-prediction-artifacts` (2.2 GB — the canonical LB7.776 saved-model pickle store)
- `henryjavier/rogii-datasets-processed` (3.6 GB)
- `nikhilsvnit/rogiiv1` (6.6 GB)
- and others, mostly redundant model caches.

---

## 7. Wall clock total

~55 minutes for Phase 1A (kernels) + 1B (datasets) + 1C (forum) launch and through ~620 kernels pulled.

Pulls remaining: ~64 lower-vote kernels (votes ≤ 1) — these are mostly forks of forks; signal density very low. Expected to drain within ~10 more minutes.

**Phase 4 (long-poll) not entered** — the agent will need to be re-spawned for that. The 76 newly-pulled forum messages confirm no new game-changing insights since the 2026-06-12 snapshot. Most active threads (CV-LB correlation, MTP, U-space projection) were already captured.

---

## 8. Output organization

```
experiments/public_resources/
├── HARVEST_TASK.md           (the spec)
├── HARVEST_LOG.md            (continuous timestamped log; greppable)
├── HARVEST_SUMMARY.md        (THIS FILE)
├── INSIGHTS_2026-06-15.md    (delta to 2026-06-12 forum snapshot)
├── all_kernel_refs.txt       (684 entries)
├── all_kernels_meta.csv      (with vote counts; sortable)
├── all_datasets_meta.csv     (36 entries with sizes)
├── all_topics.csv            (91 forum topics)
├── kernels_by_priority.txt   (sorted by votes)
├── kernels_raw/              (~620 dirs of raw notebook + metadata)
├── kernels/                  (10 .md per-kernel structured extractions)
├── feature_engineering/      (15 .md per-feature extractions)
├── ensemble_weights/         (5 .md/.json blending recipes)
├── model_params/             (5 .json model hyperparam dicts)
├── preprocessing/            (6 .md preprocessing tricks)
├── cv_methodology/           (3 .md CV/target conventions)
├── datasets_raw/             (26 unzipped datasets <500MB each)
├── datasets/                 (10 .md description stubs for >500MB datasets)
├── forum_raw/                (76 .txt forum-message dumps, new since 2026-06-12)
├── nb_to_code.py             (helper: ipynb → code-only .txt for grepping)
├── pull_kernels.sh           (Phase 1A driver; idempotent re-runnable)
├── pull_datasets.sh          (Phase 1B driver)
└── pull_forums.sh            (Phase 1C driver, fixed for v2.2.1 CLI syntax)

experiments/public_harvest/   (12 stubs ready for parent review)
├── feat_*.py                 (10 feature stubs)
├── model_*.py                (1 model stub: TabICL)
└── postproc_*.py             (2 post-processing stubs)
```

---

## 9. Next steps for the parent agent

In priority order:

1. **Apply the formation-imputer training-time fix** (cv_methodology/cv_lb_gap_tcn_lessons.md). Drop self-well formation columns at training; use `FormationPlaneKNN` for both train AND test. Expected single-largest LB lift available.
2. **Add TabICL to the round_010 stack** (model_tabicl_kojimar.py + datasets_raw/needless090_rogii-tabicl-mirror/). Likely +0.3-0.5 RMSE if it composes well.
3. **Add DTW signal family** (feat_dtw_sakoe_chiba.py). 20+ orthogonal features.
4. **Add U-space projection** as a post-processing layer (postproc_u_space_projection.py).
5. **Add guarded physical override** (postproc_guarded_physical_override.py).
6. **Verify GR is interpolated with `limit_direction='both'`** before PF/beam/NCC (preprocessing/gr_interpolation_before_pf.md).
7. **Add azimuth-weighted plane-KNN** as a drop-in upgrade (feature_engineering/azimuth_weighted_plane_knn.md).
8. **Consider the two-pipeline blend pattern** (build a second feature pipeline with independent code/seed; 0.55/0.45 blend).
9. **Visible-GR-shift + GR FFT** for cheap per-well features (3 + 2 = 5 features).
10. **Estimator divergence pairwise features** (11 cheap features, top-SHAP).

---

*End of summary.*
