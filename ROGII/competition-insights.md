# ROGII Competition — Discussion Forum Insights

> Auto-generated from Kaggle discussion forums
> Competition: rogii-wellbore-geology-prediction
> Generated: 2026-06-08
> Topics scanned: 60 | High-value topics: 25 | Top competitors tracked: 20

---

## Leaderboard Snapshot (Top 20)

| Rank | Team | Score | Entries |
|------|------|-------|---------|
| 1 | SaintLouis | 5.986 | — |
| 2 | Ruby | 6.487 | — |
| 3 | Tucker Arrants | 6.650 | — |
| 4 | Mr RRR | 6.901 | — |
| 5 | PirateInPants | 7.031 | — |
| 6 | TheLightIs | 7.058 | — |
| 7 | myominhtet_wnp | 7.066 | — |
| 8 | WhereAmI | 7.069 | — |
| 9 | Jacoby Jaeger | 7.079 | — |
| 10 | Jack | 7.185 | — |
| 11 | Chris Deotte | 7.191 | — |
| 12 | Justin Kimlim | 7.208 | — |
| 13 | akmr | 7.309 | — |
| 14 | zhuo wamg | 7.431 | — |
| 15 | tereka | 7.469 | — |
| 16 | James Day | 7.499 | — |
| 17 | Sam #3 | 7.512 | — |
| 18 | Gert dal Pozzo | 7.517 | — |
| 19 | VIbes & Wellbore Trade-Off | 7.573 | — |
| 20 | Sadam Torres | 7.629 | — |

---

## 🔥 High-Value Discussions

### Topic #697416: Welcome to ROGII - Wellbore Geology Prediction!
**Author:** Igor Kuvaev (ROGII) | **Votes:** 24 | **Comments:** 9 | **Date:** 2026-05-06

**Summary:** Official welcome post. Dataset is typical modern horizontal drilling data with well trajectory and GR measurements. PowerPoint presentation contains key information. YouTube video on manual geosteering is currently private.

**Key Insights:**
- Successful solutions will help make drilling safer, more efficient, and lower in emissions
- The dataset can be confidently interpreted by an experienced human interpreter
- YouTube video link is private (reported by Tiago Soares)

---

### Topic #697400: [EDIT] Dataset issue - Fixed!
**Author:** María Cruz | **Votes:** 12 | **Comments:** 5 | **Date:** 2026-05-05

**Summary:** Submissions were temporarily disabled due to a rerun dataset issue. Status: RESOLVED.

**Key Insights:**
- No need to re-download any files
- The test data you see locally (3 wells) is just a partial copy
- The true hidden test set is only provided after submitting and does not match the local test set

---

### Topic #699853: Multi-Trajectory Prediction (MTP) with Deep CNN
**Author:** hengck23 | **Votes:** 40 | **Comments:** 90 | **Date:** 2026-05-15

**Summary:** The most active technical discussion. hengck23 shares CNN+MTP architecture for welllog inversion, discovers `dz = dtvt` leak, and evolves the approach through multiple iterations.

**Architecture:**
```
[2D Heatmap Input] → [Regression Head (CNN)] → [MDN Predictor (MLP)] → [Multi-Trajectory Output]
```

**Key Insights:**
- **Mixture Density Network (MDN)** for multiple path hypotheses (like k-beam search)
- **Signed Distance Function (SDF)** representation helps capture micro 2D patterns
- CNN + SDF: good at capturing global GR waveform patterns
- Transformer on dz (without GR): captures dz prior
- PF on single-value GR: local GR match based on local state

**The "dz = dtvt" Leak (Critical Discovery):**
- `dz = np.gradient(Z)` and `dtvt = np.gradient(TVT)` are on the **same scale** and overlap in long stretches
- Where formation is flat: `dtvt = -dz` exactly
- They only diverge at dip events (~15 control points per well, ~323 rows apart)
- Oracle: `cumsum(-dz - offset)` achieves **~7.64 RMSE**
- But choosing the offset is hard: known-prefix → ~37-39 RMSE, fold-safe selector → ~14.8 RMSE
- **Offset values appear to be limited to a discrete set**
- Brute force search over offset × future step candidates → **12.18 RMSE** (one fold)

**Recommended Features for Transformer:**
```python
seq = torch.cat([
    h_dtvt_history,      # gradient of TVT
    h_tvt_mask,          # TVT_input mask
    h_dz,                # gradient of Z
    h_x, h_y, h_z,       # coordinates
    ...
])
```

**Data Secrets:**
- Training set contains only **69 unique typewells** (not 773)
- Some hidden test typewells may be **offset copies** of train typewells
- **dy is constant** across wells → strongly suggests synthetic data generation

**Code & Resources:**
- Example notebook: https://www.kaggle.com/code/hengck23/cnn-mtp-example
- Arxiv paper: "Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning" (Sergey Alyaev)
- Lecture notes: https://github.com/geosteering-no/inversion_school_geosteering

**Top Competitor Contributions:**
> **@Tom** (Bayesian PINN): "−dz and dtvt being the same scale and overlapping in long stretches means: wherever the formation is flat, dtvt = −dz exactly. ANCC (formation top) is ~piecewise-linear with ~15 control points per well."

> **@sleep3r**: "a fine offset-grid oracle gives ~7.64 RMSE on train hidden rows... but choosing the offset is the hard part"

> **@Tucker Arrants** (#3): "I think they need to reset. Surely providing the post-PS trajectory (X/Y/Z) is causally downstream of the answer — the driller steered based on where the formation actually was."

---

### Topic #700424: Share an UI Visualizer
**Author:** Tom | **Votes:** 41 | **Comments:** 53 | **Date:** 2026-05-17

**Summary:** Tom shares a visualization tool and discusses multiple advanced approaches including Bayesian PINN, SegFormer, Neural SDE, and diffeomorphic warping.

**TVT Analogy:**
> Imagine geology as a high-rise building where each floor is a different rock layer.
> - **Typewell** = the elevator shaft (goes vertically, records GR at each floor)
> - **Horizontal well** = a person walking in the hallway (moves along a floor, sometimes goes between floors)
> - **TVT** = which floor is this person currently on?

**Methods Explored:**

| Method | CV Score | Notes |
|--------|----------|-------|
| GBDT Baseline | ~9.45 | |
| SegFormer + soft seg input | ~9.18 | |
| **Bayesian PINN** | **~9.18** | Physics-informed neural network using pyro |
| UNet (pretrained on synthetic wells) | ~9.3 | Synthetic pretraining helps |

**Key Directions:**
- **Neural SDE** — forward-stepping curriculum can make it start to learn
- **Curvature integration** with teacher forcing warm start
- **Diffeomorphic warping** — warp from a flat line (similar to Vesuvius Challenge)
- **Piecewise correction model** — define multiple pieces using split points along MD
- **Fourier formation perspective** — analyze in wavelet domain

**Tools:**
- ROGII Viewer: https://github.com/tom99763/rogii-viewer
- Read `glossary.html` first to clarify definitions

**Top Competitor Contributions:**
> **@Tucker Arrants** (#3): "~1.2 ft behind you with simple UNet model. No physics constraints yet. Pre-training on synthetic wells gave a decent boost."

> **@hengck23**: "Normal dtw assume monotonic seq and cannot match reverse index, so be careful if you use it."

---

### Topic #702131: Domain Priors + Q-3D Tortuosity
**Author:** Matteo Niccoli | **Votes:** 15 | **Comments:** 0 | **Date:** 2026-05-21

**Summary:** Geophysical approach with single LightGBM, no particle filters. Focus on domain features and proper CV strategy.

**Notebook:** https://www.kaggle.com/code/mycarta/rogii-wellbore-geology-prediction-toolkit
**GitHub:** https://github.com/mycarta/rogii-geosteering-toolkit

**Key Findings:**

**TVT-Z Decoupling:**
- Global TVT-vs-Z correlation: **r = −0.96**
- Within a single lateral: mean slope **+0.057** (essentially zero)
- The global signal is **cross-well structural elevation** dominated by build-section geometry
- Features based on global relationship don't work until you account for this

**Domain Features Ablation (Single LightGBM):**

| Feature | Δ RMSE | Notes |
|---------|--------|-------|
| Q-3D tortuosity (Jing et al. 2022) | **−0.107** | Most useful domain feature |
| Signed drilling azimuth (sin/cos + dZ/dMD) | — | Updip/downdip distinction is real |
| Well-level AEON (Catch22 + ClaSP) | **+0.476** | Made model worse — overfits cross-well noise |
| Verde's BlockKFold | rejected | Spatial blocking too pessimistic; wells are interleaved |

**Recommended CV Strategy:**
- Use **StratifiedGroupKFold** stratified by: signed azimuth, median TVT, and spatial location
- Spatial blocking (BlockKFold) is too pessimistic because validation wells are interleaved with training

---

### Topic #702919: Dynamic Programming for TVT Tracking
**Author:** Matteo Niccoli | **Votes:** 4 | **Comments:** 0 | **Date:** 2026-05-27

**Summary:** Detailed analysis of Viterbi/DP approach for TVT tracking. OOF improved significantly but LB barely moved — textbook CV-overfitting signature.

**What Worked:**
- Viterbi ensemble mean became #1 feature by LightGBM gain importance
- OOF improved from 14.806 → 14.346

**What Didn't:**
- LB barely moved: 14.081 vs 14.082 baseline
- Fold variance increased from 0.44 to 0.87
- Viterbi features fit training distribution but don't generalize

**Key Lesson:**
> The ~5 ft gap between ~14 and ~9 RMSE is **not** about better inference over GR-typewell match, but about adding a **second independent information channel** (spatial structure from neighboring wells)

**Structural Guide + Local Matcher Pattern:**
- Top-scoring notebooks combine both: formation plane fits (structural guide) + GR-typewell matching (local matcher)
- Neither alone reaches sub-10; the combination does

---

### Topic #701041: Why Naive XGBoost Hits a Wall
**Author:** Nicolas Bridelance | **Votes:** 14 | **Comments:** 0 | **Date:** 2026-05-18

**Summary:** Comprehensive literature review from signal processing to foundation models. Progressive reading guide from geological basics to SOTA deep learning.

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

---

### Topic #701034: Surface Columns are in TVD (Z), NOT in TVT
**Author:** Nicolas Bridelance | **Votes:** 6 | **Comments:** 2 | **Date:** 2026-05-18

**Summary:** Critical coordinate system clarification. ANCC/BUDA columns are negative TVD values, not TVT.

**Key Insight:**
- ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA are stored as **negative TVD values** (~−9500 to −7500 ft)
- Direct plotting against TVT (~11000–12000 ft) makes them appear off-scale
- Z and TVT are almost perfectly correlated (**|r| ≈ 0.999**)
- Quick fix: `scipy.interpolate.interp1d` to map Z→TVT

---

### Topic #699289: Paradigm Shift — Why Pure Tabular Models Hit a Wall
**Author:** Amged Alfaqih | **Votes:** 21 | **Comments:** 4 | **Date:** 2026-05-13

**Summary:** Argues that treating this as pure tabular regression misses the physical and spatial realities.

**Key Insights:**
1. **Sequential Reality:** Treat TVT as a moving particle updating state based on GR observations — Particle Filters or Beam Search
2. **Spatial Reality:** Use cKDTree spatial imputation to find nearest known wells and calculate median TVT/formation depth
3. **"Spatial neighborhood consensus"** as a feature massively stabilizes predictions

**Top Competitor Contributions:**
> **@hengck23**: "Check ROGII patent on its product StarSteer. Its Viterbi/beam search includes using dip equation. US20190106974A1"

---

### Topic #701691: CV and LB Correlations
**Author:** Gaurav Rawat | **Votes:** 8 | **Comments:** 10 | **Date:** 2026-05-19

**Summary:** Tracking CV-LB correlation evolution for GBDT models.

**Top Competitor Contributions:**
> **@Tucker Arrants** (#3): "Single model NN update: CV 8.5, LB 7.5. Inference in 2 minutes."

> **@Tucker Arrants**: "With plain jane GBDT models, CV around 11.00 split on well ID and leaderboard around 9.6. A lot of the gap comes from the structural guide vs local matcher decomposition."

---

### Topic #704273: How Much Should We Trust the LB Score?
**Author:** 寿! | **Votes:** 11 | **Comments:** 6 | **Date:** 2026-06-04

**Summary:** Discussion on distribution shift between train and test, and whether to trust CV or LB.

**Key Insights:**
- Training set: **773 wells**
- Public LB test set: only **52 wells**
- **Distribution shift exists** between train and test
- CV-LB gap can be up to **2 RMSE** depending on approach
- Methods with specific assumptions tend to show larger gaps
- **Recommendation:** Trust local CV over LB, assuming validation strategy is sound
- When making larger pipeline changes, CV-LB correlation "resets", but small tuning within the same pipeline leads to consistent improvements

**Top Competitor Contributions:**
> **@Tucker Arrants** (#3): "When I make larger pipeline changes, my CV-LB correlation 'resets' but then any CV improvements with small tuning within that new pipeline always lead to LB improvements."

---

### Topic #703038: The Battle for LB9.0↓
**Author:** NobelK | **Votes:** 3 | **Comments:** 11 | **Date:** 2026-05-28

**Summary:** Discussion on score barriers and how top competitors are breaking through.

**Key Insights:**
- LB has already broken below 8.0 (as of early June)
- Tom estimates **oracle score ~3.5 RMSE** (mock test achievable)
- One competitor reports oracle score **<1.0** (perfect information)
- The score barrier is being pushed by:
  - Feature engineering + model blending
  - Physics-informed approaches
  - Probabilistic modeling
  - Synthetic data pretraining

---

### Topic #704001: EGFDU, ANCC, ASTNL, ASTNU, EGFDL, BUDA and Geology?
**Author:** Big Boss | **Votes:** 1 | **Comments:** 2 | **Date:** 2026-06-02

**Summary:** Confirmation that formation columns are training-only.

> **@Chris Deotte** (#11): "No, they are not [available in test]. They are only available in the train data. As said on the data page: 'ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA - Predicted depth of various geological formations (Training only).'"

---

### Topic #704839: When Will Typewell Data Generated?
**Author:** doteeee | **Votes:** 2 | **Comments:** 5 | **Date:** 2026-06-06

**Summary:** Clarification on typewell nature and timing.

**Key Insights:**
- Vertical type wells exist **before drilling starts** — full data set is present at test time
- They are **"pseudo-typewells"** (not real vertical wells) — a limited number of templates are reused
- PC Jimmy found only **57 unique type wells** in the entire field (shifted matches)

> **@PC Jimmmy**: "There are a limited number of different vertical holes in the train... 'pseudo-typewells' is the term used rather than synthetic."

---

### Topic #701995: Is the Public LB Test Set (26%) Fixed?
**Author:** Alhasan Abdellatif | **Votes:** 7 | **Comments:** 13 | **Date:** 2026-05-20

**Summary:** Investigation into why same notebook yields different LB scores.

> **@Chris Deotte** (#11): "The test data does not change. The reason our scores change is because many feature engineering steps are stochastic in this competition."

---

### Topic #697431: Besides Regression, Also DWT (Time Warping)!
**Author:** hengck23 | **Votes:** 32 | **Comments:** 36 | **Date:** 2026-05-06

**Summary:** Early discussion on DTW and data structure discoveries.

**Key Insights:**
- Only **69 unique typewells** in train data
- **dy is constant** across wells → synthetic data suspected
- Test typewells may be **offset copies** of train typewells
- ROGII StarSteer patent: US20190106974A1 — uses Viterbi/beam search including dip equation

---

## ⭐ Insights from Top Competitors

### @SaintLouis (#1, LB 5.986)

Currently leading the competition. No public discussions found.

### @Ruby (#2, LB 6.487)

No public discussions found.

### @Tucker Arrants (#3, LB 6.650)

**Sources:** Topics #701691, #699853

**Key Contributions:**
- Single NN: CV 8.5, LB 7.5, inference in **2 minutes**
- GBDT: CV ~11.0, LB ~9.6 — large gap but stable
- "A lot of the gap comes from the structural guide vs local matcher decomposition"
- UNet with synthetic pretraining: ~1.2 ft behind Tom's Bayesian PINN
- CV-LB correlation "resets" on large pipeline changes, but small tuning is consistent

### @Mr RRR (#4, LB 6.901)

No public discussions found.

### @PirateInPants (#5, LB 7.031)

No public discussions found.

### @Tom (Bayesian PINN/SegFormer, ~LB 7.0)

**Sources:** Topics #700424, #699853, #703038

**Key Contributions:**
- Bayesian PINN + SegFormer: CV ~9.18
- Neural SDE with forward-stepping curriculum
- Diffeomorphic warping approach
- Piecewise correction model
- Oracle score estimate: **~3.5 RMSE**
- TVT analogy: "geology as a high-rise building"
- ROGII Viewer: https://github.com/tom99763/rogii-viewer

### @Jack (#10, LB 7.185)

No public discussions found.

### @Chris Deotte (#11, LB 7.191)

**Sources:** Topics #704001, #701995

**Key Contributions:**
- Confirmed: ANCC/BUDA/Geology are **training only**
- Confirmed: test data does not change; score variation is from stochastic feature engineering

### @hengck23 (CNN/MTP Pioneer)

**Sources:** Topics #699853, #697431, #699326

**Key Contributions:**
- Discovered `dz = dtvt` leak (same scale, offset is key unknown)
- Only **69 unique typewells** in train
- **dy is constant** → synthetic data
- CNN+SDF+MTP architecture
- Brute force offset search: 12.18 RMSE
- "z prior is much stronger than gr prior"
- ROGII StarSteer patent reference

### @Matteo Niccoli (DP/Viterbi Expert)

**Sources:** Topics #702131, #702919

**Key Contributions:**
- Q-3D tortuosity: **−0.107 RMSE** (best domain feature)
- TVT-Z decoupling finding
- Viterbi ensemble OOF improved 0.46 but LB moved 0.001 — CV-overfitting signature
- **Key lesson:** 10→14 RMSE gap is about adding spatial structural information, not better GR matching
- StratifiedGroupKFold recommendation

### @Nicolas Bridelance (Literature Review)

**Sources:** Topics #701041, #701034

**Key Contributions:**
- ANCC/BUDA stored as **negative TVD**, not TVT
- Z and TVT: **|r| ≈ 0.999**
- Complete signal processing methods benchmark
- Foundation model evaluation (PatchTST 9/10, TimesFM 8.5/10, Chronos 4/10)
- Soft-DTW for differentiable alignment loss

### @Amged Alfaqih (Paradigm Shift)

**Source:** Topic #699289

**Key Contributions:**
- Pure tabular models miss physical/spatial realities
- Particle Filters / Beam Search for sequential tracking
- cKDTree spatial imputation for "spatial neighborhood consensus"

### @PC Jimmy

**Sources:** Topics #697431, #704839, #701995

**Key Contributions:**
- Found only **57 unique type wells** (shifted matches)
- "Pseudo-typewells" terminology
- Test set has **45 test well paths** (from PPT)

### @sleep3r

**Sources:** Topics #699853

**Key Contributions:**
- "a fine offset-grid oracle gives ~7.64 RMSE"
- "oracle anchors: k≈10 → ~4ft, k≈20 → ~1.7ft"
- Tried CNN-likelihood + CatBoost ranker over modes
- GR matching is ill-conditioned: even at true TVT, horizontal↔typewell GR correlation ~0.7

---

## 📌 Official Announcements

1. **Welcome Post** (#697416) — Igor Kuvaev (ROGII)
   - Dataset is typical modern horizontal drilling data
   - Review the PowerPoint presentation attached to competition data
   - YouTube video currently private

2. **Dataset Issue Fixed** (#697400) — María Cruz
   - Submissions temporarily disabled, now resolved
   - No need to re-download files

3. **Submission Scoring** (#697329) — Discussion on scorer being live
   - 30 comments, 8 votes

---

## 💡 Technical Deep-Dives

### Complete Method Stack (Nicolas Bridelance's Layered Approach)

| Layer | Method | Role |
|-------|--------|------|
| 1. Coarse alignment | Phase correlation / cross-corr | Detect macro offset and faults |
| 2. Anchor detection | NCC sliding window (>0.85 = high confidence) | Find stratigraphic anchor points |
| 3. Local elastic alignment | DTW (Sakoe-Chiba band) | Handle layer thickness variations |
| 4. Deep feature extraction | PatchTST / TimesFM encoder | Capture multi-scale geological patterns |
| 5. Spatial structural guide | Formation plane fits (cKDTree neighbors) | Regional dip-flattening backbone |
| 6. Sequential tracking | Particle Filter / Viterbi / Beam Search | Maintain trajectory consistency |
| 7. Final regression | LightGBM / XGBoost | Blend all features |

### Physics-Informed Approaches

1. **Bayesian PINN** (Tom) — pyro-based probabilistic modeling with physical priors, CV ~9.18
2. **Neural SDE** (Tom) — forward-stepping curriculum learning
3. **Curvature Integration** (Tom) — teacher forcing warm start
4. **Diffeomorphic Warping** (Tom) — warp from flat line

### CNN/MTP Evolution (hengck23)

1. CNN + SDF for global GR waveform patterns
2. Transformer on dz (without GR) for dz prior
3. Mixture Density Network for multi-trajectory hypotheses
4. Brute force search over offset × future step candidates

---

## Summary Table

| Insight | Source | Impact |
|---------|--------|--------|
| ANCC/BUDA columns are **train-only** | Chris Deotte (#704001) | 🔴 Critical — cannot use at test time |
| Typewells are **pseudo-typewells** (57-69 unique) | hengck23, PC Jimmy | 🟡 Reuse patterns across wells |
| `dtvt ≈ -dz` (same scale, offset is key) | hengck23 (#699853) | 🔴 Strong geometric prior |
| Offset values appear **discrete** | hengck23 (#699853) | 🟡 Classification opportunity |
| GR correlation ~0.7 even at true TVT | sleep3r (#699853) | 🟡 GR alone insufficient |
| Only need **few anchor points** | hengck23, sleep3r | 🟢 Pace correction framing |
| **Z prior >> GR prior** | hengck23 (#699853) | 🔴 Trajectory > GR matching |
| Public LB = 52 wells | 寿! (#704273) | 🟡 Trust CV over LB |
| Oracle score ~3.5 RMSE | Tom (#703038) | 🟢 Large room for improvement |
| Q-3D tortuosity: **−0.107 RMSE** | Matteo Niccoli (#702131) | 🟢 Best domain feature |
| TVT-Z decoupling within lateral | Matteo Niccoli (#702131) | 🟡 Global correlation misleading |
| Viterbi OOF↑0.46, LB↑0.001 | Matteo Niccoli (#702919) | 🔴 CV-overfitting warning |
| Spatial guide + local matcher needed | Matteo Niccoli (#702919) | 🔴 Two-channel requirement |
| PatchTST 9/10, TimesFM 8.5/10 | Nicolas Bridelance (#701041) | 🟢 Foundation model rankings |
| ANCC/BUDA in TVD (not TVT) | Nicolas Bridelance (#701034) | 🔴 Coordinate system trap |
| dy constant → synthetic data | hengck23 (#697431) | 🟡 Data generation insight |
| ROGII StarSteer patent | hengck23 (#699289) | 🟡 Viterbi/beam search reference |
| Spatial neighborhood consensus | Amged Alfaqih (#699289) | 🟢 Stabilization feature |
| Test data does not change | Chris Deotte (#701995) | 🟢 Score variation = randomness |
| Single NN: CV 8.5, LB 7.5, 2min | Tucker Arrants (#701691) | 🟢 NN inference speed |

---

## Common Pitfalls & Tips

1. **Submission errors:** Make sure your notebook generates `submission.csv` correctly. Check file format, row count, no NaN/inf values.

2. **DTW limitations:** Standard DTW assumes monotonic sequences and cannot match reverse indices. Be careful if using DTW for GR alignment.

3. **GR matching is ill-conditioned:** Even at true TVT, horizontal ↔ typewell GR correlation is only ~0.7. Offset error compounds over distance.

4. **Don't need to match all GR:** We have good dTVT estimates. Only need a few anchor points to push the whole TVT curve to correct the pace.

5. **Z prior > GR prior:** The well trajectory (X, Y, Z) provides a much stronger prior than GR matching. Use Z-based constraints.

6. **Notebook timeout:** Be careful with long-running operations. Test inference time locally.

7. **Coordinate system trap:** ANCC/BUDA are negative TVD values (~−9500 to −7500), not TVT (~11000–12000). Map Z→TVT before using.

8. **Randomness control:** Score variation on resubmission is from stochastic feature engineering, not changing test data. Fix all random seeds (including numba).

---

*Last updated: 2026-06-08. Crawled from Kaggle discussion forums using `kaggle competitions topics`.*
