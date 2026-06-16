# 紧急任务：导出模型 + 上传 + 推 Kernel

## 背景
hill climb v3 的 blend 是在本地 3 口 sample 井上算的，Kaggle hidden test 有 ~100 口不同井，ID 对不上。现在要把 v9 的完整管线搬进 Kaggle Notebook 在线跑。

## 执行步骤

### Step 1: 本地训练 LGB + CAT 模型并导出

用 `/Users/liucong/code/kaggle/ROGII/experiments/round_008/r8_kaggle_submit_v9.py` 改造成只训练不推理。

修改要点：
- 先检查有没有缓存文件: 在 `/Users/liucong/code/kaggle/ROGII/experiments/round_008/` 下搜 `*.joblib` 或 `*.pkl`，如果有直接加载
- 如果没有，执行训练部分（读取所有 train 井 → PF 跑特征 → 训练 LGB + CAT）
- 训练完用 `joblib.dump` 保存两个模型到 `/Users/liucong/code/kaggle/ROGII/experiments/round_008/models/`
  - `lgb_model.joblib`
  - `cat_model.joblib`
- 同时保存 feature_cols 列表到 `feature_cols.joblib`

关键参数（v9 脚本第 622-639 行的超参数）：
```python
model_lgb = lgb.LGBMRegressor(
    n_estimators=2500, learning_rate=0.02, num_leaves=127,
    min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
    colsample_bytree=0.8, subsample=0.85, subsample_freq=5,
    verbose=-1, n_jobs=-1,
)
model_cat = CatBoostRegressor(
    iterations=1500, learning_rate=0.05, depth=8,
    l2_leaf_reg=3.0, subsample=0.85, rsm=0.8,
    loss_function="RMSE", eval_metric="RMSE",
    verbose=False, thread_count=-1, random_seed=42,
    bootstrap_type="Bernoulli",
)
```

注意：PF 跑 723 井很慢，用 ROGII_TRAIN_CACHE 环境变量缓存中间特征：
```
ROGII_TRAIN_CACHE=/Users/liucong/code/kaggle/ROGII/experiments/round_008/train_features_cache.parquet
```
第一次跑会把 train features 存下来，第二次就能秒加载。

### Step 2: 上传模型到 Kaggle Dataset

用 Kaggle CLI：
```bash
# 创建 dataset 目录
mkdir -p /tmp/kaggle_models_dataset

# 复制模型文件
cp /Users/liucong/code/kaggle/ROGII/experiments/round_008/models/*.joblib /tmp/kaggle_models_dataset/

# 创建 dataset-metadata.json
cat > /tmp/kaggle_models_dataset/dataset-metadata.json << 'EOF'
{
  "id": "smartorz/rogii-v9-models",
  "title": "ROGII v9 LGB+CAT models",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# push dataset
cd /tmp/kaggle_models_dataset && /Users/liucong/miniconda3/bin/kaggle datasets create -p . --dir-mode zip
如果已存在则用 update:
cd /tmp/kaggle_models_dataset && /Users/liucong/miniconda3/bin/kaggle datasets version -m "v1" --dir-mode zip
```

### Step 3: 创建新的 submission kernel

基于 hill climb v3 kernel 改造（在 `/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/` 下）。

新的 `hillclimb_submit.py` 要做的事：

```python
# 1. 读取 competition test 数据（不在本地读 CSV！）
# 2. 对每口 test 井跑 PF 特征（用 v9 脚本里的 extract_well_features）
# 3. 加载模型
# 4. 预测 + blend
# 5. 对齐 sample_submission → 写 submission.csv
```

注意 item ID 规则：
- test 数据在 `/kaggle/input/rogii-wellbore-geology-prediction/test/` 下
- 每个井的文件格式: `{well_id}__horizontal_well.csv` 和 `{well_id}__typewell.csv`
- submission ID 格式: `{well_id}_{row_idx}`（和 v9 脚本第 672 行一样）

### Step 4: 推 Kernel

```bash
/Users/liucong/miniconda3/bin/kaggle kernels push
```

推之前确认：
- kernel-metadata.json 里 `competition_sources` 包含 `rogii-wellbore-geology-prediction`
- 加上 `dataset_sources` 包含 `smartorz/rogii-v9-models`
- GPU 设为 false（CPU only）

### Step 5: 验证结果

1. 检查 kernel log 是否正常
2. 等出分
3. 报告分数

## 如果本地训练太慢

如果 Step 1 跑 723 井 PF 太慢（预计数小时），改为：
1. 只取前 200 口井训练（足够得到可用模型）
2. 或者直接在 Kaggle 上创建一个训练 output kernel，save 模型文件，再下载
3. 或者用现有的 train_features_cache 如果在 round_008 里已经存在

## 故障处理
- 如果 Kaggle push 失败：检查 kernel-metadata.json 格式
- 如果模型加载失败：检查 joblib 版本兼容性
- 如果 PF 跑 test 井出错：检查 input 目录结构
