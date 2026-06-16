# TASK: Debug hill climb v3 blend & submit via Kaggle CLI notebook

## 背景
- r9 v2 (pure PF 128-seed) LB = **8.781** ✅ — 这是 benchmark
- hill climb v3 blend (inference kernel) LB = **699.724** ❌ — 明显异常
- Hill climb v3 本地 OOF = **7.566** — 说明候选模型质量好，但 test 推理代码有 bug
- Notebook kernel `smartorz/rogii-hill-climb-v3-notebook` 已存在但 push 后不生成正确提交

## 问题诊断 (需要你来查)
预测均值 11907 ft（绝对 TVT），但 LB 得分 699.724。可能的根因：

1. **Feature 列不匹配** — `feat_cols.pkl` 中的列名与 `extract_well_features` 输出的列名不一致
   - 对比 `feat_cols.pkl` 内容 vs `inference_kernel.py` 中 rec 字典的 key
   - 阶段 14B 训练时可能用了不同特征集

2. **模型预测的 target 与推理时加的 last_known_tvt 不一致**
   - LGB/CAT 模型预测的是 `target = TVT - last_known_TVT`（相对偏移）
   - 推理时 `ml_tvt = last_known_tvt + ml_off`
   - 验证 test 的 `last_known_tvt` 值与训练时一致

3. **sample_submission 对齐有问题**
   - 本地测试时该 merge 逻辑曾报过 `SubmissionScoringError`
   - 验证 test 井的 id 格式是否与 sample_submission.csv 一致

4. **PF ensemble 在 test 上产生异常值**
   - `pf_ens_s12_abs` 可能在某些井上完全错误
   - 检查 ensemble fallback 数量

## 解决步骤

### Step 1: 本地调试
```bash
cd /Users/liucong/code/kaggle/ROGII
python3 experiments/round_010/submission_package/hillclimb_v3_kernel/inference_kernel.py 2>&1 | tee /tmp/debug_hillclimb.log
```
- 确保 `results/round_010/` 目录已创建（否则 to_csv 会报错）
- 查看 pred stats（min/max/mean/median）是否合理
- 查看 feature 列是否 36 个
- 查看 PF ensemble fallback 数量

### Step 2: 验证模型 feature 列
```python
import pickle
import pandas as pd
# 加载 feat_cols
with open(".../feat_cols.pkl", "rb") as f:
    feat_cols = pickle.load(f)
# 提取 test 特征
# 检查 test[feat_cols] 是否有 missing 列
```

### Step 3: 修复找到的 bug 并更新 notebook
编辑 `inference_kernel.py` → 同步更新 `hillclimb_submit.ipynb`

### Step 4: 用 Kaggle CLI 推送 notebook
```bash
# 工作目录
cd /Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/

# 确认 kernel-metadata.json 正确（dataset_sources 指向 pretrained-models-v2）
# 推送新版本
kaggle kernels push

# 等待完成
kaggle kernels status smartorz/rogii-hill-climb-v3-notebook

# 查看提交分数
kaggle competitions submissions rogii-wellbore-geology-prediction
```

### Step 5: 汇报结果
告诉我：
- 找到的 bug 是什么
- 修了什么
- 新提交的 LB 分数

## 工作目录
```
/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/
```

## 关键文件
- 推理代码: `inference_kernel.py`
- Notebook: `hillclimb_submit.ipynb` (需与 inference_kernel.py 保持同步)
- 元数据: `kernel-metadata.json`
- 模型文件: `lgb_model.txt`, `cat_model.cbm`, `feat_cols.pkl`
- Kaggle CLI: `/Users/liucong/miniconda3/bin/kaggle`
- Python: `/Users/liucong/miniconda3/bin/python`
