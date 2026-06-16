# round_010 submission_package — Pure PF Kaggle Notebook submission

**Goal:** A self-contained, ML-free Kaggle Notebook submission for the
ROGII Wellbore Geology Prediction competition. Submits a 128-seed
particle-filter ensemble combined with beam search, a 6-bin per-well
selector, and a 4th-degree robust polynomial projection. Visible wells
(those that also appear in `train/`) get a deterministic physical
projection from formation contacts instead of the PF prediction.

No CatBoost / LightGBM / XGBoost weights are loaded — the Notebook
runs pure numpy / scipy and needs zero pre-trained model files.

## Files

```
submission_package/
├── README.md                    ← this file
├── requirements.txt             ← numpy, pandas, scipy, sklearn (Kaggle has all)
├── v3_weights.json              ← reference: hill climb v3 weights (not loaded
│                                  at inference; documents the offline blend)
├── ensemble_submit.py           ← Notebook entry point (run this cell)
└── training_code/
    ├── __init__.py
    └── pf_128seed.py            ← pure-numpy 128-seed PF ensemble
```

## How to upload

Two equally good ways. Pick **A** for a dedicated dataset (cleaner, repeatable),
or **B** to paste straight into the Notebook (faster).

### A. As a Kaggle Dataset (recommended)

1.  Zip the `submission_package/` directory:

    ```bash
    cd experiments/round_010
    zip -r round010_submission_package.zip submission_package
    ```

2.  Upload as a new Kaggle Dataset (e.g. slug `round010-pf-only`).

3.  In the Notebook, **Add Data → your dataset**. It will mount at
    `/kaggle/input/round010-pf-only/submission_package/`.

4.  In a Notebook code cell, run:

    ```python
    import os, sys
    PKG = "/kaggle/input/round010-pf-only/submission_package"
    # Copy to working dir so Python imports cleanly:
    !cp -r {PKG} /kaggle/working/
    sys.path.insert(0, "/kaggle/working/submission_package")
    %run /kaggle/working/submission_package/ensemble_submit.py
    ```

    `ensemble_submit.py` already auto-discovers the package from common
    Kaggle paths (`/kaggle/input/*` and `/kaggle/working/*`), so the
    `sys.path.insert` is just a safety belt.

### B. Paste directly into the Notebook

1.  Create one Notebook cell that writes the helper module:

    ```python
    import os, pathlib
    pathlib.Path("/kaggle/working/submission_package/training_code").mkdir(
        parents=True, exist_ok=True)
    # Paste the full contents of training_code/pf_128seed.py here
    open("/kaggle/working/submission_package/training_code/__init__.py","w").write("")
    open("/kaggle/working/submission_package/training_code/pf_128seed.py","w").write(r"""
    ...PF code...
    """)
    ```

2.  Then in the next cell, paste `ensemble_submit.py` and run it.

## How to run

The Notebook entry is `ensemble_submit.py`. It does **not** take any CLI args.
Run it as a script in any Notebook cell:

```python
%run /kaggle/working/submission_package/ensemble_submit.py
```

Output: `/kaggle/working/submission.csv` with columns `id`, `tvt` (absolute
TVT in feet, **not** offset).

## Wall time

The PF stage dominates: 128 seeds × 500 particles × len(lateral) rows per well.
For a typical 100-well hidden test set that is roughly **4–6 hours** on the
Kaggle CPU notebook (16 GiB / 4 vCPU). Beam + selector + projection are <2 min
per well combined. Plan for ~6 hours of Notebook commit time and start
the run before the 9-hour Kaggle CPU budget.

If a single well crashes the PF (e.g. degenerate typewell GR), the script
catches the exception, falls back to last-known-TVT, and continues. PF /
beam / projection failures are counted and printed in the final summary.

## What about the ML models in v3_weights.json?

The hill-climb v3 averaged weights are:

```
c20_r9_pf128_full   0.7785
p14_cat             0.1089
p14_lgb             0.0926
p5_lgb              0.0131
c06_xgb_default     0.0069
```

The dominant 0.78 weight is the same 128-seed PF this notebook computes.
The four ML candidates contribute <0.22 of the blend total, but they
require uploading trained CatBoost / LightGBM / XGBoost model files plus
the LB-7.776-style feature pipeline — too heavy for a single Notebook
under the 9-hour CPU cap. Per the task spec we **drop them** and submit
pure PF only. `v3_weights.json` is shipped purely as documentation;
nothing in `ensemble_submit.py` reads it.

If a future round wants to re-introduce the ML candidates, do it in a
separate dataset that ships pre-fitted model `.pkl` / `.cbm` files plus
the joined-feature parquet, and blend offline using
`experiments/round_010/orchestrator/submit_ensemble.py` instead of this
script.
