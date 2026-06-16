# Round 010 — Kaggle submission package

Pure-PF entry-point for the ROGII Wellbore Geology Prediction competition.

## What's in this folder

```
submission/
├── ensemble_submit.py          # Kaggle entry-point (I/O, selector, beam, main loop)
├── training_code/
│   └── pf_128seed.py           # 128-seed log-likelihood-weighted PF (importable)
└── README.md                   # this file
```

No model weights or external Datasets are required — the PF is parameter-free
and runs purely on the test well's own GR / trajectory plus the typewell.

## Why pure-PF and not the full v3 blend?

The hill-climb v3 averaged weights are:

| candidate            | weight  | shipped here?          |
| -------------------- | ------- | ---------------------- |
| `c20_r9_pf128_full`  | 0.7785  | **yes** (this script)  |
| `p14_cat`            | 0.1089  | no — needs CatBoost model dataset  |
| `p14_lgb`            | 0.0926  | no — needs LightGBM model dataset  |
| `p5_lgb`             | 0.0131  | no — needs LightGBM model dataset  |
| `c06_xgb_default`    | 0.0069  | no — needs XGBoost model dataset   |

The PF candidate alone carries 77.85 % of the v3 blend and is the strongest
single model in the pool (per-well RMSE 7.95 vs. v3 blend 7.57). Shipping
PF-only costs ~0.4 RMSE in expectation versus the full blend, in exchange for
a self-contained notebook with zero external Dataset dependencies. When the
ML model artefacts are uploaded, swap in the missing branches and re-blend.

## Wall-clock budget

CPU-only, 100 hidden test wells, 128 seeds × 500 particles × 4 PF scales:
**≈ 5–7 hours**. Comfortably within Kaggle's 9 h notebook limit, but make
sure the Notebook Editor's session timeout is set high enough.

## How to submit

### Option A — Notebook only (simplest)

1. Create a new Kaggle Notebook attached to the
   `rogii-wellbore-geology-prediction` competition.
2. Settings: Accelerator = **None (CPU)**, Internet = **off**.
3. Paste the contents of `ensemble_submit.py` and `training_code/pf_128seed.py`
   into a single notebook cell (drop the `from training_code.pf_128seed
   import …` line and inline the PF code instead).
4. Run All. `submission.csv` lands at `/kaggle/working/submission.csv`.

### Option B — Notebook + Utility Dataset (recommended for iteration)

1. Upload this whole `submission/` directory as a Kaggle **Dataset** —
   call it e.g. `rogii-r010-pf-code`.
2. In a new Notebook, attach **two** inputs:
   - the competition data (`rogii-wellbore-geology-prediction`)
   - your code dataset (`rogii-r010-pf-code`)
3. In the first cell:

   ```python
   import sys
   sys.path.insert(0, "/kaggle/input/rogii-r010-pf-code")
   from ensemble_submit import main
   main()
   ```

4. Settings: Accelerator = **None (CPU)**, Internet = **off**.
5. Run All. `submission.csv` lands at `/kaggle/working/submission.csv`.
6. Submit the notebook to the competition.

## Local sanity check

```bash
/Users/liucong/miniconda3/bin/python \
    experiments/round_010/submission/ensemble_submit.py
```

When run outside Kaggle, the script falls back to
`rogii-wellbore-geology-prediction/{train,test}/` relative to the local repo
**only if `/kaggle/input/...` is mounted**. To smoke-test against a single
local well, edit `test_wids = test_wids[:1]` near the top of `main()`.

## Provenance

- PF core: `training_code/pf_128seed.py` is the `run_particle_filter` /
  `run_pf_lik_ensemble_scales` pair lifted verbatim from
  `experiments/round_009/r9_pf_only_submit_v2.py` (LB-tested).
- Selector + beam configs: same source.
- v3 weights:
  `experiments/round_010/results/hillclimb_runs/run_v3_after_c20.json`.
