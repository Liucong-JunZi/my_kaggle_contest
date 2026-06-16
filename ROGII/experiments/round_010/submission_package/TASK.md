# Task: Create hill climb v3 blend Kaggle Notebook submission

## Background
- The pure PF kernel (smartorz/rogii-pf-128-seeds) is already RUNNING — leave it alone
- We need a SECOND kernel that submits the hill climb v3 ensemble blend
- The pre-computed blend CSV is at: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_v3.csv`
- Kaggle username: smartorz

## What to do

### Step 1: Upload submission_v3.csv as a Kaggle Dataset
Create a dataset under smartorz with the CSV file:
- Dataset slug: `smartorz/rogii-hillclimb-v3-blend`
- File: submission_v3.csv (just copy it)
- Dataset metadata JSON needs title + id

### Step 2: Create a minimal Kaggle script
Create a one-file script `hillclimb_submit.py` that:
```python
import pandas as pd
sub = pd.read_csv("/kaggle/input/rogii-hillclimb-v3-blend/submission_v3.csv")
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Done, wrote", len(sub), "rows")
```

### Step 3: Create kernel-metadata.json
```json
{
  "id": "smartorz/rogii-hillclimb-v3",
  "title": "ROGII hill climb v3",
  "code_file": "hillclimb_submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": ["smartorz/rogii-hillclimb-v3-blend"],
  "competition_sources": [],
  "kernel_sources": []
}
```

### Step 4: Push and verify
```bash
cd /path/to/work/dir
/Users/liucong/miniconda3/bin/kaggle kernels push
```

Then check status:
```bash
/Users/liucong/miniconda3/bin/kaggle kernels status smartorz/rogii-hillclimb-v3
```

## Notes
- This is a simple copy-notebook, should finish in <2 min on Kaggle CPU
- It will auto-submit submission.csv to the competition
- Do NOT touch the existing smartorz/rogii-pf-128-seeds kernel
- You already know the Kaggle CLI is at `/Users/liucong/miniconda3/bin/kaggle`
- Work in `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/`
