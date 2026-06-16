# Task: Analyze submission differences

## Problem
hill climb v3 blend submission got `SubmissionScoringError`: "type for a value, or invalid submission values from what is expected. See more debugging tips"

But v8 (11.383) submission was fine.

## What to do
1. Find the v8 submission Notebook. Check:
   `kaggle kernels list --mine` or check submissions for notebook links
   
2. Compare the OUTPUT format between v8 and hill climb v3:
   - Column names
   - Value types / formats
   - Row count
   - Any difference in how tvt values look

3. v8 submission notebook source should be in:
   `/Users/liucong/code/kaggle/ROGII/experiments/round_008/r8_kaggle_submit_v9.py`
   or
   `/Users/liucong/code/kaggle/ROGII/experiments/round_008/r8_submit.ipynb`

4. Compare the actual CSV output formats between a successful submission and ours. Check:
   - tvt value range
   - Decimal precision  
   - Whether values are relative (delta from last_known_TVT) vs absolute
   
5. Report what's DIFFERENT and propose fix.

## Key files
- Our submission CSV: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_v3.csv`
- v8 submission Notebook: see `kaggle kernels list --mine` → look for the submission that scored 11.383
- Sample submission: `/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/sample_submission.csv`

## Critical check
The competition description says the LABEL is `target = TVT - last_known_TVT` (relative offset). The v8 notebook might be outputting RELATIVE tvt, and our hill climb v3 is outputting ABSOLUTE tvt. That would explain the Scoring Error.