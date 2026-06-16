# TASK: Convert inference kernel to notebook type and push to Kaggle

## Background
The inference-only kernel `smartorz/rogii-hill-climb-v3-inference-only` was pushed as `kernel_type: "script"` and successfully ran. However, Kaggle only creates competition submissions from **Notebook** type kernels. The submission did NOT appear in the competition submissions list.

## Steps

### 1. Convert `inference_kernel.py` to `.ipynb`
The file already has `# %%` cell markers (percent format). Use `jupytext` to convert:

```bash
pip install jupytext
jupytext --to notebook inference_kernel.py -o hillclimb_submit.ipynb
```

Or use a simple Python script with `nbformat` if jupytext isn't available.

The resulting `.ipynb` must be at:
```
/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/hillclimb_submit.ipynb
```

### 2. Update `kernel-metadata.json`
Change:
- `"kernel_type"` from `"script"` → `"notebook"`
- `"code_file"` from `"inference_kernel.py"` → `"hillclimb_submit.ipynb"`

Keep all other fields the same.

### 3. Push to Kaggle
```bash
cd /Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel
/Users/liucong/miniconda3/bin/kaggle kernels push
```

This should update the existing `smartorz/rogii-hill-climb-v3-inference-only` kernel as a notebook type.

### 4. Verify
- Check `kaggle kernels status smartorz/rogii-hill-climb-v3-inference-only` shows "running" or "complete"
- Wait for it to complete and check `kaggle competitions submissions rogii-wellbore-geology-prediction` for the new submission

## Files
- **Script**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/inference_kernel.py`
- **Metadata**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/kernel-metadata.json`
- **Target notebook**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/hillclimb_submit.ipynb`
- **Python**: `/Users/liucong/miniconda3/bin/python`
- **Kaggle CLI**: `/Users/liucong/miniconda3/bin/kaggle`

## Important
- Do NOT modify `inference_kernel.py` — leave it as-is
- The notebook should be a faithful conversion, not a rewrite
- After push, verify the kernel status before reporting back
