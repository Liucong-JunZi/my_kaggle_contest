# TASK: Final fix — create new dataset + push inference kernel

## Current Status
1. Old kernel `smartorz/rogii-hill-climb-v3-inference-only` — 404 (deleted)
2. Old dataset `smartorz/rogii-pretrained-models-v1` — exists but **new kernels can't mount it** (debug kernel confirmed: `/kaggle/input/rogii-pretrained-models-v1` → exists=False)
3. Fresh dataset `smartorz/rogii-pretrained-models-v2` — upload in progress (was interrupted, may need redo)

## The Fix (2 steps)

### Step 1: Create dataset `smartorz/rogii-pretrained-models-v2`
Files to upload (located in `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/`):
- `lgb_model.txt` (29.9 MB)
- `cat_model.cbm` (6.3 MB)
- `feat_cols.pkl` (528 B)

Command: first use `kaggle datasets init -p /tmp/new_dataset`, edit dataset-metadata.json to have `"id": "smartorz/rogii-pretrained-models-v2"`, copy model files in, then `kaggle datasets create -p /tmp/new_dataset`.

**IMPORTANT**: Check if the upload was partially completed first. If dataset v2 already exists, use `kaggle datasets version -p /tmp/new_dataset -m "full upload"` instead.

### Step 2: Push inference kernel with new dataset
1. Edit `kernel-metadata.json` in the kernel directory:
   - Set `"dataset_sources": ["smartorz/rogii-pretrained-models-v2"]`
   - Set `"id": "smartorz/rogii-hill-climb-v3-final"` (new slug)
   - Keep `"competition_sources": ["rogii-wellbore-geology-prediction"]`
2. Push: `kaggle kernels push`
3. Wait for COMPLETE status
4. Check competition submission appears: `kaggle competitions submissions rogii-wellbore-geology-prediction`

### Fallback (if Step 1+2 fails)
If the new dataset also can't be mounted, use this approach in the kernel script:
```python
# Download model files directly from the old dataset
import kagglehub
path = kagglehub.dataset_download("smartorz/rogii-pretrained-models-v1")
# reads model files from path
```
Requires `"enable_internet": true` in kernel-metadata.json.

## Working Directory
`/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/`

## Paths
- Python: `/Users/liucong/miniconda3/bin/python`
- Kaggle CLI: `/Users/liucong/miniconda3/bin/kaggle`
- Inference script: `inference_kernel.py`
