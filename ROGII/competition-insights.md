# ROGII Competition — Discussion Forum Insights

> Collected from Kaggle discussion forums using `kaggle competitions topics`.
> Last updated: 2026-06-08

---

## 1. Competition Overview & Resources

**Source:** Topic #697416 — Welcome to ROGII - Wellbore Geology Prediction!  
**Author:** Igor Kuvaev (ROGII)

- The dataset is a **typical example of modern horizontal drilling data** containing well trajectory and Gamma Ray (GR) measurements.
- **PowerPoint presentation** attached to the competition data contains key information on horizontal well interpretation.
- Manual geosteering process video: https://youtu.be/gdK_eY5_QrE (currently private)
- This is a **real-world problem** — successful solutions will help make drilling safer, more efficient, and lower in emissions.

---

## 2. Dataset Issue (Fixed)

**Source:** Topic #697400 — [EDIT] Dataset issue - Fixed!  
**Author:** María Cruz

- Submissions were temporarily disabled due to a rerun dataset issue.
- **Status: RESOLVED** — no need to re-download any files.
- **Important:** The test data you see locally (3 wells) is just a partial copy. The true hidden test set is only provided after submitting and does not match the local test set.

---

## 3. Key Data Understanding

### 3.1 Typewell = "Pseudo-typewells"

**Source:** Topic #704839 — When will Typewell data generated?  
**Author:** doteeee, PC Jimmmy

- Vertical type wells exist **before drilling starts** — full data set is present at test time.
- However, they are **"pseudo-typewells"** (not real vertical wells) — a limited number of templates are reused across the dataset.

### 3.2 ANCC/BUDA Formation Columns — NOT in Test Set

**Source:** Topic #704001 — EGFDU, ANCC, ASTNL, ASTNU, EGFDL, BUDA and Geology?  
**Author:** Big Boss, Chris Deotte

> ⚠️ **CRITICAL:** The formation depth columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) and `Geology` are **ONLY available in training data**. They are **NOT available in the hidden test set**.
>
> As stated on the data page: "Predicted depth of various geological formations (Training only)"

**Implication:** Any model relying on these columns for inference will fail on the hidden test. They can only be used for:
- Training-time augmentation / teacher forcing
- Understanding data generation process
- Building synthetic training data

---

## 4. LB vs Local CV — Trust Your CV

**Source:** Topic #704273 — How much should we trust the LB score?  
**Author:** 寿!

- Training set: **773 wells**
- Public LB test set: only **52 wells**
- **Distribution shift exists** between train and test.
- CV-LB gap can be up to **2 RMSE** depending on approach.
- Methods with specific assumptions tend to show larger gaps.
- **Recommendation:** Trust local CV over LB, assuming validation strategy is sound.
- When making larger pipeline changes, CV-LB correlation "resets", but small tuning within the same pipeline leads to consistent improvements.

---

## 5. State-of-the-Art Methods Discussion

### 5.1 Multi-Trajectory Prediction (MTP) with CNN

**Source:** Topic #699853 — multi-trajectory prediction (MTP) with deep CNN for welllog inversion  
**Author:** hengck23 (90 comments, 40 votes)

**Architecture:**
```
[2D Heatmap Input] → [Regression Head (CNN)] → [MDN Predictor (MLP)] → [Multi-Trajectory Output]
```

**Key Ideas:**
- Use **Mixture Density Network (MDN)** for multiple path hypotheses (like k-beam search).
- If keypoints in GR signals can be identified, decide how to traverse between matched keypoints.
- **Signed Distance Function (SDF)** representation helps capture micro 2D patterns.

**Validation Results (reported):**
- CNN + SDF: good at capturing global GR waveform patterns
- Transformer on dz (without GR): captures dz prior
- PF on single-value GR: local GR match based on local state

**Important Insight — The "dz = dtvt" Leak:**
- `dz = np.gradient(Z)` and `dtvt = np.gradient(TVT)` are on the **same scale** and overlap in long stretches.
- Where formation is flat: `dtvt = -dz` exactly.
- They only diverge at dip events (~15 control points per well).
- Using `cumsum(-dz - offset)` with a discrete offset achieves **~7.7 RMSE** (oracle).
- Choosing the correct offset is the hard part: known-prefix offset gives ~37-39 RMSE, fold-safe selector gets ~14.8.

**Code & Resources:**
- Example notebook: https://www.kaggle.com/code/hengck23/cnn-mtp-example
- Arxiv paper: "Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning" (Sergey Alyaev)
- Lecture notes: https://github.com/geosteering-no/inversion_school_geosteering

### 5.2 Bayesian PINN & SegFormer

**Source:** Topic #700424 — Share an UI visualizer  
**Author:** Tom

**Methods explored by top competitors:**

| Method | CV Score | Notes |
|--------|----------|-------|
| GBDT Baseline | ~9.45 | |
| SegFormer + soft seg input | ~9.18 | |
| **Bayesian PINN** | **~9.18** | Physics-informed neural network |
| UNet (pretrained on synthetic wells) | ~9.3 | Synthetic pretraining helps |

**Key Directions:**
- **Probabilistic modeling** with pyro — build priors and minimize TVT mismatch
- **Curvature integration** with teacher forcing warm start
- **Diffeomorphic warping** — warp from a flat line (similar to Vesuvius Challenge approach)
- **Piecewise correction model** — define multiple pieces using split points along MD
- **Fourier formation perspective** — analyze in wavelet domain

**Important Comment:**
> "A fine offset-grid oracle gives ~7.64 RMSE on train hidden rows... but choosing the offset is the hard part."

### 5.3 Recommended Features for Transformer/CNN

**From hengck23's comments:**

Useful features to add:
- Self-correlation (good for identifying moving reverse)
- Neighbouring well correlation
- `x, y, z, azimuth, inclination`
- Plane Z sampled from fitted geology plane (ANCC, BUDA, etc.)
- Shared common typewell ID

**Architecture suggestion:**
```python
seq = torch.cat([
    h_dtvt_history,      # gradient of TVT
    h_tvt_mask,          # TVT_input mask
    h_dz,                # gradient of Z
    h_x, h_y, h_z,       # coordinates
    ...
])
```

---

## 6. EDA & Visualization Tools

**Source:** Topic #700424  
**Author:** Tom

- **ROGII Viewer:** https://github.com/tom99763/rogii-viewer
- Includes EDA and directions. **Read `glossary.html` first** to clarify definitions.
- The visualization helps understand GR mismatch between horizontal well and typewell.

**TVT Analogy (from Tom):**
> Imagine geology as a high-rise building where each floor is a different rock layer.
> - **Typewell** = the elevator shaft (goes vertically, records GR at each floor)
> - **Horizontal well** = a person walking in the hallway (moves along a floor, sometimes goes between floors)
> - **TVT** = which floor is this person currently on?

---

## 7. Domain Understanding Resources

**Papers & Videos:**
1. [Azimuthal LWD Data Interpretation for Geosteering Using PINN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6000576)
2. [Lattice Deduction Transformers](https://arxiv.org/html/2605.08605v1) — for mixture/DP problems
3. ROGII automatic alignment and segmentation video (timestamp 27:50)
4. Lecture series on geosteering: https://github.com/geosteering-no/inversion_school_geosteering

**Physical Constraints to Consider:**
- Forward GR = `F.interpolate(predict_tvt, typewell_tvt, typewell_GR)`
- Forward geology = `F.interpolate(predict_tvt, typewell_tvt, typewell_Geology)`
- Curvature constraints on wellbore trajectory
- Formation dip direction changes (piecewise-linear with ~15 control points)

---

## 8. Current Score Landscape

**Source:** Topic #703038 — The battle for LB9.0↓  
**Author:** NobelK

- LB has already broken below 8.0 (as of early June).
- Tom estimates **oracle score ~3.5 RMSE** (mock test achievable).
- One competitor reports oracle score **<1.0** (perfect information).
- The score barrier is being pushed by:
  - Feature engineering + model blending
  - Physics-informed approaches
  - Probabilistic modeling
  - Synthetic data pretraining

---

## 9. Common Pitfalls & Tips

1. **Submission errors:** Make sure your notebook generates `submission.csv` correctly. The error output is often unhelpful — check file format, row count, no NaN/inf values.

2. **DTW limitations:** Standard DTW assumes monotonic sequences and cannot match reverse indices. Be careful if using DTW for GR alignment.

3. **GR matching is ill-conditioned:** Even at true TVT, horizontal ↔ typewell GR correlation is only ~0.7. Offset error compounds over distance.

4. **Don't need to match all GR:** We have good dTVT estimates. Only need a few anchor points to push the whole TVT curve to correct the pace.

5. **Z prior > GR prior:** The well trajectory (X, Y, Z) provides a much stronger prior than GR matching. Use Z-based constraints.

6. **Notebook timeout:** Be careful with long-running operations. Test inference time locally.

---

---

## 10. Top Competitors' Insights (Leaderboard Ranked)

### Tucker Arrants (#3, LB 6.650)

**Source:** Topic #701691 — cv and lb correlations

| Method | CV | LB | Notes |
|--------|-----|-----|-------|
| GBDT | ~11.0 | ~9.6 | Large gap but stable; all CV improvements lead to LB improvements |
| **Single NN** | **8.5** | **7.5** | Inference in **2 minutes** |
| UNet (synthetic pretraining) | ~9.3 | — | Pretraining on synthetic wells gave decent boost |

> "With the plain jane GBDT models, CV around 11.00 split on well ID and leaderboard around 9.6. A lot of the gap comes from the structural guide vs local matcher decomposition."

### Chris Deotte (#11, LB 7.191)

**Source:** Topics #704001, #701995

- Confirmed: `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` and `Geology` are **training only**.
- On LB score variance: "The test data does not change. The reason our scores change is because many feature engineering steps are stochastic in this competition."

### Matteo Niccoli (DP/Viterbi Expert)

**Sources:** Topics #702919, #702131

**Dynamic Programming for TVT Tracking:**
- Viterbi ensemble (5 paths with different smoothness settings) + LightGBM
- OOF improved from 14.806 → 14.346, but **LB barely moved: 14.081 vs 14.082**
- This is a **textbook CV-overfitting signature** — Viterbi features fit training distribution but don't generalize
- **Key lesson:** The ~5 ft gap between ~14 and ~9 RMSE is **not** about better inference over GR-typewell match, but about adding a **second independent information channel** (spatial structure from neighboring wells)

**Domain Features Ablation (Single LightGBM):**

| Feature | Δ RMSE | Notes |
|---------|--------|-------|
| Q-3D tortuosity (Jing et al. 2022) | **−0.107** | Most useful domain feature |
| Signed drilling azimuth (sin/cos + dZ/dMD) | — | Updip/downdip distinction is real |
| Well-level AEON (Catch22 + ClaSP) | **+0.476** | Made model worse — overfits cross-well noise |
| Verde's BlockKFold | rejected | Spatial blocking too pessimistic; wells are interleaved |

**Key Geological Finding — TVT-Z Decoupling:**
- Global TVT-vs-Z correlation: **r = −0.96**
- Within a single lateral: mean slope **+0.057** (essentially zero)
- The global signal is **cross-well structural elevation** dominated by build-section geometry
- Features based on global relationship don't work until you account for this

**Recommended CV Strategy:**
- Use **StratifiedGroupKFold** stratified by: signed azimuth, median TVT, and spatial location
- Spatial blocking (BlockKFold) is too pessimistic because validation wells are interleaved with training

### Nicolas Bridelance (Literature Review)

**Sources:** Topics #701041, #701034

**ANCC/BUDA Columns — Coordinate System Trap:**
- These columns are stored as **negative TVD values** (same unit as Z, ~−9500 to −7500 ft), **NOT in TVT**
- Direct plotting against TVT (~11000–12000 ft) makes them appear off-scale
- Quick fix: map Z→TVT via linear interpolation (`scipy.interpolate.interp1d`)
- Z and TVT are almost perfectly correlated (**|r| ≈ 0.999**)

**Signal Processing Methods Benchmark:**

| Method | Handles Stretching | Amplitude-Invariant | Best Use |
|--------|:----------------:|:-------------------:|----------|
| Classical cross-corr | ❌ | ❌ | Initial rough alignment |
| NCC (sliding window) | ❌ | ✅ | Anchor point detection |
| Phase correlation | ❌ | ✅ | Coarse fault/offset detection |
| DTW (Sakoe-Chiba) | ✅ | ❌ | Local elastic alignment |
| PatchTST / TimesFM | ✅ | ✅ | Deep feature extraction |

**Foundation Models for Time Series:**
- **PatchTST**: 9/10 fit — patch-based encoding matches geological signal scale (5–15 ft windows)
- **TimesFM 2.5**: 8.5/10 — XReg feature allows passing XYZ as exogenous regressors
- **MOIRAI**: 7/10 — masked encoder, high-frequency detail
- **Chronos**: 4/10 — quantization smooths brief high-amplitude spikes (key stratigraphic markers)

**Soft-DTW** (tslearn) replaces `min` with smooth approximation, making alignment loss differentiable — can embed directly in neural network loss.

### hengck23 (CNN/MTP Pioneer)

**Sources:** Topics #699853, #697431, #699326

**Data Secrets:**
- Training set contains only **69 unique typewells** (not 773)
- Some hidden test typewells may be **offset copies** of train typewells
- If confirmed, this provides free geology information copied from train
- **dy is constant** across wells → strongly suggests synthetic data generation

**StarSteer Patent (ROGII's commercial product):**
- US20190106974A1 — "Systems and methods for horizontal well geosteering"
- Uses **Viterbi/beam search** including dip equation
- Reference: https://www.rogii.com/blog/starsteer-geoassist-enhanced-eagle-ford-reservoir

**The "dz = dtvt" Leak (Detailed):**
- `h_dtvt = np.gradient(h_tvt)` and `h_dz = np.gradient(h_z)` are on the **same scale**
- Where formation is flat: `dtvt = -dz` exactly
- They only diverge at dip events (~15 control points per well, ~323 rows apart)
- Oracle: `cumsum(-dz - offset)` achieves **~7.64 RMSE**
- But choosing the offset is hard: known-prefix → ~37-39 RMSE, fold-safe selector → ~14.8 RMSE
- **Offset values appear to be limited to a discrete set**

**MTP Architecture Evolution:**
1. CNN + SDF (signed distance function) for global GR waveform patterns
2. Transformer on dz (without GR) for dz prior
3. Mixture Density Network for multi-trajectory hypotheses
4. Brute force search over offset × future step candidates → 12.18 RMSE (one fold)

### Amged Alfaqih (Paradigm Shift)

**Source:** Topic #699289

**Why Pure Tabular Models Hit a Wall:**
- Passing [X, Y, Z, MD, GR] into LightGBM/XGBoost misses physical/spatial realities
- **Sequential reality:** PF or beam search treat TVT as moving particle updating state based on GR observations
- **Spatial reality:** Use cKDTree spatial imputation to find nearest known wells and calculate median TVT/formation depth
- **"Spatial neighborhood consensus"** as a feature massively stabilizes predictions

### PC Jimmy

**Sources:** Topics #697431, #704839

- Found only **57 unique type wells** in the entire field (shifted matches)
- Vertical type wells exist **before drilling starts** — full data at test time
- Test set has **45 test well paths** (from PPT)

---

## 11. Complete Method Stack (From Literature Review)

**Nicolas Bridelance's Recommended Layered Approach:**

| Layer | Method | Role |
|-------|--------|------|
| 1. Coarse alignment | Phase correlation / cross-corr | Detect macro offset and faults |
| 2. Anchor detection | NCC sliding window (>0.85 = high confidence) | Find stratigraphic anchor points |
| 3. Local elastic alignment | DTW (Sakoe-Chiba band) | Handle layer thickness variations |
| 4. Deep feature extraction | PatchTST / TimesFM encoder | Capture multi-scale geological patterns |
| 5. Spatial structural guide | Formation plane fits (cKDTree neighbors) | Regional dip-flattening backbone |
| 6. Sequential tracking | Particle Filter / Viterbi / Beam Search | Maintain trajectory consistency |
| 7. Final regression | LightGBM / XGBoost | Blend all features |

---

## 12. Summary of Key Insights for Modeling

| Insight | Implication |
|---------|-------------|
| ANCC/BUDA columns are **train-only** | Cannot use at test time; use for training augmentation only |
| Typewells are **pseudo-typewells** | Only 57–69 unique templates; reuse patterns across wells |
| `dtvt ≈ -dz` (same scale) | Strong geometric prior from well trajectory; offset is the key unknown |
| GR correlation ~0.7 even at true TVT | GR alone is insufficient; need trajectory constraints |
| Only need **few anchor points** | Don't try to match entire GR sequence; correct the pace |
| **Z prior >> GR prior** | Well geometry is more informative than GR |
| Public LB = 52 wells | Trust local CV; LB has high variance |
| Oracle score ~3.5 RMSE | Large room for improvement exists |
| Offset selection is the bottleneck | Finding correct formation offset is the core challenge |
| Piecewise-linear formations (~15 CPs) | Formation dips change at discrete control points |
| dy is constant | Data is likely synthetic |
| Test typewells may be train offsets | Could provide free geology information |
