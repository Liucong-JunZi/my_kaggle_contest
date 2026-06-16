# ROGII Kernel-to-Candidate Extractor — Sub-Agent Task

## Mission

You convert pulled Kaggle kernels into **executable round_010 candidates**, NOT analysis notes. Every output you produce must either be a runnable `.py` candidate that fits round_010's training contract, or a refusal-with-reason.

The goal: **grow the round_010 hill-climb pool with diverse-source candidates as fast as possible.** Quality over quantity, but we need quantity too.

---

## Sources

`/Users/liucong/code/kaggle/ROGII/experiments/public_resources/kernels_raw/`
- ~568 directories, each is one kernel pull
- Files inside: `<kernel-slug>.ipynb` or `<kernel-slug>.py` + metadata
- HARVEST_LOG.md (parent dir) tracks what's pulled

`/Users/liucong/code/kaggle/ROGII/experiments/public_resources/kernels_by_priority.txt`
- Top kernels sorted by votes — START FROM HERE, work through this list first
- Each line: `<votes>\t<ref>` (e.g. "593\tnihilisticneuralnet/...")

`/Users/liucong/code/kaggle/ROGII/experiments/public_resources/reviewed_kernels/06_ACTION_PLAN.md`
- Already-digested action plan with prioritized techniques — read this for context

`/Users/liucong/code/kaggle/ROGII/experiments/public_harvest/`
- Earlier agent already created some stubs here. Check if they're runnable; if yes, register them. If they're TODO stubs, fill them in.

---

## Target Format — round_010 candidate contract

Every candidate you produce goes into:
`/Users/liucong/code/kaggle/ROGII/experiments/round_010/candidates/<cid>.py`

**Hard schema** (parent agent will run `python orchestrator/train_one.py <cid>`):

```python
"""<cid>: short description, source kernel <ref> (LB <score>)."""
import lightgbm as lgb  # or catboost, xgboost, sklearn, etc.

CANDIDATE_ID   = "<cid>"           # e.g. c30_lgb_lb9251
CANDIDATE_TYPE = "lightgbm"        # or catboost, xgboost, ridge, ...
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=3000, learning_rate=0.02, ...,
)


def get_features(df):
    """Return (X DataFrame, feat_cols list).
    df is the pre-joined feature dataframe from shared/data_loader.load_joined().
    Available columns: see shared/data_loader.py — feature_set_v14() returns 43 feats.
    """
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    """Train one fold. Must return a fitted model with .predict()."""
    m = lgb.LGBMRegressor(**HYPERPARAMS, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
```

**ALL candidates must use existing features in `shared/data_loader.feature_set_v14()`** unless the source kernel uses a feature we don't have — in which case skip (or log it for parent agent to add later).

The target is **TVT - last_known_tvt** (relative offset). Available in the joined df as `target` column.

---

## Numbering Convention

Use these ID ranges:
- `c20_*` — your first batch (PF/beam variants, GBDT clones from kernels)
- `c30_*` — round 2 (anything more exotic)
- `c40_*` — round 3 (NN / unusual / experimental)

Always check `ls candidates/` before naming to avoid collisions.

---

## Decision Tree (per kernel)

```
For each kernel (in priority order):
  1. Read the .py or .ipynb (use jupytext --to py if needed)
  2. CLASSIFY:
     a. "GBDT model with our features" → HIGH PRIORITY
        → Extract hyperparams, write candidate, target ID c20-29
     b. "Feature engineering on our existing inputs" → MEDIUM
        → Skip for now (parent agent will batch-add features)
        → BUT if it produces a single offset prediction (PF variant, beam variant),
          then write a candidate that wraps the prediction code
     c. "PF/beam/NCC variant producing offset predictions" → HIGH
        → Wrap the prediction logic in a candidate; the candidate's
          fit_fold can be trivial (e.g. just memorise mean) since the
          prediction comes from PF, not from training. Use seed=fold_idx
          to vary per fold. ID: c20_pf_*, c20_beam_*, c20_ncc_*
     d. "NN model (TCN, CNN, transformer)" → SKIP (too expensive)
        → Log to needed_features.md instead
     e. "TabICL / inference-only stack" → MEDIUM
        → Skip if requires GPU; try CPU version if feasible
     f. "Same as our existing candidates" (default LGB on Phase14B feats) → SKIP
        → Mark as duplicate
  3. WRITE the candidate file
  4. SMOKE TEST: `python -c "from candidates.<cid> import HYPERPARAMS, get_features, fit_fold, predict; print('OK')"`
     - If import fails, fix imports (don't leave broken candidates)
  5. LOG: append to /Users/liucong/code/kaggle/ROGII/experiments/round_010/CANDIDATES_LOG.md
  6. DO NOT TRAIN: parent agent will batch-train. Your job is candidate creation only.
```

---

## Hard Boundaries

- ✅ Read kernels_raw/, write to candidates/, write to CANDIDATES_LOG.md
- ✅ pip install missing libs (xgboost, lightgbm-extra, catboost) if needed
- ❌ DO NOT modify existing c01-c08 candidates
- ❌ DO NOT modify shared/, orchestrator/, results/
- ❌ DO NOT run train_one.py — parent does that
- ❌ DO NOT modify experiments/round_008, round_009
- ❌ DO NOT touch Kaggle CLI
- ❌ DO NOT train ML models locally (only candidate stub creation + smoke import test)

---

## Special Cases

### Kernels with their own OOF parquet
Some kernels save their OOF predictions as artifacts. If you find a kernel that produces a parquet with `(well_id, row_idx, oof_pred)` columns directly, you can register it as an "imported" candidate (like the legacy p14_/p11_/p5_ ones). See `orchestrator/import_legacy_oofs.py` for pattern. ID: `c20_imp_<source>`.

### Kernels that train PF with non-standard params
Common LB-7.776 variants change:
- N_PARTICLES (500 vs 1000)
- N_SEEDS (32 vs 128 vs 256)
- INIT_SPREAD (0.3 vs 4.5 vs 3.0)
- PN_SCALES (low/med/high)
- gr_smooth WINDOW

These are great diversity sources. Each variant = a candidate. ID: `c20_pf_seed32_init45_*`.

### Kernels with formation-column features
**REJECT** unless they explicitly drop them at train time too. See
`docs/forum-snapshots/2026-06-12/INSIGHTS_UPDATE.md` and
`MEMORY.md → rogii-formation-col-leak`. Formation columns leak.

---

## Output Contract

Every successful candidate:
1. File: `candidates/<cid>.py` matching the contract
2. Entry in `CANDIDATES_LOG.md`:
   ```
   | cid | source_kernel | source_LB | type | note |
   |-----|---------------|-----------|------|------|
   | c20_lgb_lb9251 | nihilisticneuralnet/9-251-... | 9.251 | lightgbm | DWT features dropped |
   ```

Every rejected kernel:
- Single line in CANDIDATES_LOG.md "Rejected" section with reason

When you're done OR hit 50+ candidates, write a **`CANDIDATES_LOG_SUMMARY.md`** with:
- Total candidates added
- Top 10 most promising (by source LB)
- What features are missing (parent agent should add)
- What models are missing (NN models we should consider)

---

## State Recovery

Append to `CANDIDATES_LOG.md` continuously. If you crash, check the log to find what's done. Skip already-processed kernels.

---

## START HERE

1. Read `/Users/liucong/code/kaggle/ROGII/experiments/public_resources/reviewed_kernels/06_ACTION_PLAN.md` (3 min)
2. Read `/Users/liucong/code/kaggle/ROGII/experiments/public_resources/kernels_by_priority.txt` (1 min)
3. Read `/Users/liucong/code/kaggle/ROGII/experiments/round_010/candidates/c01_lgb_default.py` (1 min, the contract template)
4. For top 30 kernels by priority, do the decision tree above. Add candidates as you go.
5. Then move to next 30. Etc.
6. When you've processed all priority kernels OR built 50+ candidates, write CANDIDATES_LOG_SUMMARY.md and exit.

You have hours. **Quality over quantity, but get quantity too.** A working c20_xgb_lb10239 is worth 10 unwritten candidates.

GO.
