# Task: 预训练模型方案 — 本地训练 → 导出 → 推理 Kernel

## 背景
当前 v7 kernel (`smartorz/rogii-hill-climb-v3` ver 7) 使用的是**在线全训方案**（在 Kaggle Notebook 内从头训练 LGB + CAT），用户强烈反对。需要改成**预训练模型方案**：
1. 本地运行完整训练管线 → 导出模型 .pkl/.cbm
2. 上传模型到 Kaggle Dataset
3. 创建推理-only Kernel（加载模型，只跑 feature extraction + predict）

## 环境
- **工作目录**: `/Users/liucong/code/kaggle/ROGII/experiments/round_010/`
- **Python**: `/Users/liucong/miniconda3/bin/python`
- **Kaggle CLI**: `/Users/liucong/miniconda3/bin/kaggle`
- **模型文件参考**: 现有 50 个 `.pkl` 在 experiments 目录下（需检查内容）
- **Claude Code**: `/Users/liucong/.nvm/versions/node/v25.8.2/bin/claude`

## 源文件
- `/Users/liucong/code/kaggle/ROGII/experiments/round_008/r8_kaggle_submit_v9.py` — v9 完整管线（701 行），包含：
  - PF feature extraction（numba JIT）
  - 训练 LGB + CAT
  - 预测 + blend
  - 输出 submission CSV

## 步骤

### Step 1: 创建模型导出脚本
基于 `r8_kaggle_submit_v9.py`，在训练完成后（line 628 LGB fit, line 639 CAT fit）添加模型保存：
- LGB: `model_lgb.booster_.save_model('lgb_model.txt')` — LightGBM 原生格式
- CAT: `model_cat.save_model('cat_model.cbm')` — CatBoost 原生格式  
- feat_cols: `pickle.dump(feat_cols, open('feat_cols.pkl', 'wb'))` — 特征列列表

创建新文件 `export_models.py`，保持完整管线但末尾加 export。

### Step 2: 本地运行导出模型
```bash
cd /Users/liucong/code/kaggle/ROGII/experiments/round_010/
/Users/liucong/miniconda3/bin/python export_models.py
```
预期运行时间：数小时（723 口井 PF feature extraction + 训练）。考虑设置 ROGII_TRAIN_CACHE 加速。

⚠️ **如果已经训过有 train cache**，查一下是否有 `ROGII_TRAIN_CACHE` 设置过。
```bash
ls -la /tmp/rogii_train_*.parquet 2>/dev/null
ls -la /Users/liucong/code/kaggle/ROGII/experiments/round_008/train_*.parquet 2>/dev/null
```

输出文件应放在 `submission_package/hillclimb_v3_kernel/` 中：
- `lgb_model.txt`
- `cat_model.cbm`
- `feat_cols.pkl`

### Step 3: 创建 Kaggle Dataset
用 Kaggle CLI 创建 Dataset（包含三个模型文件）：
```bash
mkdir -p /tmp/rogii-pretrained-models/
cp submission_package/hillclimb_v3_kernel/{lgb_model.txt,cat_model.cbm,feat_cols.pkl} /tmp/rogii-pretrained-models/
cd /tmp/rogii-pretrained-models/
kaggle datasets create -p . -d "ROGII Pre-trained Models v1" --public
```
或手动写 `dataset-metadata.json`：
```json
{
  "title": "rogii-pretrained-models-v1",
  "id": "smartorz/rogii-pretrained-models-v1",
  "licenses": [{"name": "CC0-1.0"}]
}
```

### Step 4: 创建推理-only Kernel
创建新文件 `submission_package/hillclimb_v3_kernel/inference_kernel.py`：

核心逻辑：
```python
# 不训练，只做推理！
# 从 Dataset 加载模型
model_lgb = lgb.Booster(model_file='/kaggle/input/rogii-pretrained-models-v1/lgb_model.txt')
model_cat = CatBoostRegressor()
model_cat.load_model('/kaggle/input/rogii-pretrained-models-v1/cat_model.cbm')
feat_cols = pickle.load(open('/kaggle/input/rogii-pretrained-models-v1/feat_cols.pkl', 'rb'))

# 做 feature extraction（和原来一样）
# 不需要 fit！直接 predict
# 然后 blend + submit
```

⚠️ **关键**：Feature extraction 代码（PF, Numba JIT 等）必须**完整保留**在 kernel 中。只有 LGB/CAT 训练被替换为 load。

### Step 5: 更新 kernel-metadata.json
修改 `submission_package/hillclimb_v3_kernel/kernel-metadata.json`：
- `"title"`: 标明是 inference-only 版本
- `"language": "python"`
- `"kernel_type": "script"`
- `"dataset_sources"`: 添加 `"smartorz/rogii-pretrained-models-v1"`

### Step 6: 推送并验证
```bash
cd /Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/
/Users/liucong/miniconda3/bin/kaggle kernels push
```

### Step 7: 检查 Kaggle Log
提交后检查日志确认模型加载成功，无 error。

## 注意事项
1. **LightGBM 模型保存**：用 `booster_.save_model()` 而非 joblib，因为 Kaggle 环境 LGB 版本可能不同。
2. **CatBoost 模型保存**：用原生 `save_model()` 格式 `.cbm`。
3. **Feature extraction 必须完整**：PF、Numba、beam 等代码不能精简，和原来一样。
4. **环境变量/路径**：Kaggle 上 INPUT_DIR = `/kaggle/input/`,
   `TEST_DIR = f"{INPUT_DIR}/rogii-wellbore-geology-prediction/test/wells/"`
   `TRAIN_DIR` 仅在训练时需要，推理 kernel 可能不需要 TRAIN_DIR（但 PF ensemble 需要全量训练井的 GR 数据来计算 ensemble 统计量）。
   
   ⚠️ **重要**：PF ensemble 需要训练井的数据（读取所有训练井的 GR 曲线来计算 ensemble 统计量），所以推理 kernel 仍然需要读取训练井的原始 CSV 数据。在 Kaggle 上，训练数据在 `/kaggle/input/rogii-wellbore-geology-prediction/train/wells/`。这没问题，Kaggle 上的数据是完整的。

5. **Numba warmup**：保留 `_warmup_numba()` 调用。
6. **提交方式**：Kaggle 比赛只接受 Notebook 提交，所以最终必须通过 `kaggle kernels push` 提交。

## 成功标准
- 模型文件成功导出并上传到 Kaggle Dataset
- 推理 kernel 在 Kaggle 上运行完成（COMPLETE）
- public score 接近 v8（~11.383），不应偏离太大
