# Scoring Error - Fix hill climb v3 submission

## Error
在 Kaggle Notebook 页面看到的：
"Submission Scoring Error · 2h ago
type for a value, or invalid submission values from what is expected. See more debugging tips"

## Root cause
hill climb v3 kernel 只是从 dataset 读取预生成的 `submission_v3.csv` 写到 `/kaggle/working/submission.csv`。这个 CSV 是本地 round_010 实验 blend 的结果，只覆盖了 sample_submission 的 14,151 行。

但 Kaggle 比赛实际 test set 可能更大（有 hidden wells），所以行数不对导致 Scoring Error。

## Fix options (from previous analysis)
A. 在 Kaggle 上完整重建 hill-climb v3 blend — 把 PF kernel + 各 ML candidate 搬上 Kaggle 做 blend
B. 等 PF kernel 作为基础，加上 ML 重新跑
C. 放弃

## Task
1. 先查清楚 Kaggle 比赛 test set 到底有多少行
2. 确认当前 submission CSV 的行数 vs 实际需求
3. 提出具体修复方案