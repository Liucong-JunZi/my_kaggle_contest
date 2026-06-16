# TASK: Fix Kaggle inference kernel and create competition submission

## Problem Summary

Three attempts to push an inference-only kernel to Kaggle:

1. **`smartorz/rogii-hill-climb-v3-inference-only`** (script, ver 8) — ✅ Ran SUCCESSFULLY. Models loaded. submission.csv generated. But no competition submission was created (new script kernels don't auto-submit?).

2. **`smartorz/rogii-hill-climb-v3-notebook`** (notebook, ver 2) — ❌ ERROR: `LightGBMError: Could not open /kaggle/input/rogii-pretrained-models-v1/lgb_model.txt`. Same dataset source, same code.

3. **`smartorz/rogii-hill-climb-v3`** (script, ver 1) — ❌ SAME ERROR: Can't open lgb_model.txt.

## Key Observations

- The **first** script kernel (attempt #1) loaded models SUCCESSFULLY from `/kaggle/input/rogii-pretrained-models-v1/lgb_model.txt`. The **second and third** attempts fail with "Could not open" for the exact same path.
- Attempt #1 had `INPUT_DIR = /kaggle/input/rogii-wellbore-geology-prediction`. Attempts #2/3 had `INPUT_DIR = /kaggle/input/competitions/rogii-wellbore-geology-prediction`. The difference in competition mount point might be a clue.
- All three kernels use the same `"dataset_sources": ["smartorz/rogii-pretrained-models-v1"]` in kernel-metadata.json.

## The Goal

Get the inference-only kernel to:
1. ✅ Load the pre-trained models successfully (lgb_model.txt, cat_model.cbm, feat_cols.pkl)
2. ✅ Generate submission.csv
3. ✅ Create a competition submission (appears in `kaggle competitions submissions rogii-wellbore-geology-prediction`)

## Materials

- **Kernel code**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/inference_kernel.py`
- **Metadata**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/kernel-metadata.json`
- **Dataset**: `smartorz/rogii-pretrained-models-v1` (contains lgb_model.txt 29.9MB, cat_model.cbm 6.3MB, feat_cols.pkl 528B)
- **Working directory**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/`
- **Python**: `/Users/liucong/miniconda3/bin/python`
- **Kaggle CLI**: `/Users/liucong/miniconda3/bin/kaggle`

## Approaches to Try (in order)

### Approach A: Debug and fix the model loading issue
Add debug code to the kernel (os.listdir, os.path.exists, file size checks) before loading models. Push a debug kernel to see what's actually in `/kaggle/input/rogii-pretrained-models-v1/`. This will reveal if the files are actually there, if they're symlinks/broken, etc.

### Approach B: Upload a new dataset version
Check if the dataset has multiple versions. If so, the latest version might be corrupted. Try creating a NEW dataset (e.g. `rogii-pretrained-models-v2`) and push the kernel with the new dataset.

### Approach C: Embed models in the kernel
As a fallback, embed the model files as base64 strings in the kernel script itself. This avoids dataset dependency entirely. The lgb_model.txt is 29.9MB which is large but Kaggle allows large script files.

### Approach D: Create a notebook that re-downloads models
Instead of using Kaggle Dataset API, have the notebook download models from a URL (GitHub releases, etc.) with `enable_internet: true`.

## Constraints
- The competition "rogii-wellbore-geology-prediction" requires a NOTEBOOK submission (not CSV upload).
- Script kernels CAN submit if they're correctly configured (attempt #1 proved the code works).
- Do NOT modify the training code or prediction logic — only the loading/infrastructure parts.

Pick the simplest approach that works. After fixing, verify the submission appears in `kaggle competitions submissions rogii-wellbore-geology-prediction` and report the score.
