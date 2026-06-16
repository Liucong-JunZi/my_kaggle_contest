# Task: Investigate hill climb v3 blend — why score is 699.724

## Problem
Hill climb v3 blend (0.75·ML + 0.25·PF_ens_s12) scored **699.724** on Kaggle public LB.
In contrast: pure PF 128-seed scored 8.781; v8 (36feat + PF blend) scored 11.383.
All submissions use tvt values in ~11700-11900 range.

## Key files to check

1. **Blend generation script** — how `submission_v3.csv` was produced
   - Likely files: `blend_v3.py`, `ensemble.py`, or similar in round_010

2. **ML predictions** — what model generated ML component
   - Check if ML model used the *same* test set as PF
   - Check if ML model was trained on correct data split

3. **Prediction files** — find intermediate ML and PF prediction CSVs
   - Look for `predictions_*.csv`, `oof_*.csv`, `submission_*.csv` in round_010/

4. **v8 blend for comparison** — what made it work
   - v8 scored 11.383 with 0.75ML+0.25PF blend
   - Compare v8 blend script vs v3 hill climb blend script

## Hypothesis
The ML model in hill climb v3 may have been trained on a different data version, test set, or feature set than the PF model, causing the blend to produce bad predictions.

## Steps
1. `cd /Users/liucong/code/kaggle/ROGII/experiments/round_010/` and list files
2. Find the script that generated submission_v3.csv
3. Trace the ML and PF prediction sources
4. Compare with round_008's blend approach (which worked: 11.383)
5. Report findings — what went wrong and how to fix it

## Format of good submission CSV
- Columns: id, tvt
- No NaN, no Inf
- Float values
- 14151 rows
