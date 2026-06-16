# TASK: 修复 notebook kernel 并提交

## 背景
- notebook kernel `smartorz/rogii-hill-climb-v3-notebook` 状态 ERROR
- 日志错误: `ValueError: No kernel name found in notebook and no override provided`
- 原因: `hillclimb_submit.ipynb` 的 metadata 缺少 `kernelspec`（notebook 的 metadata 中没有 kernel 名称配置，Kaggle 的 papermill 执行器需要这个）

## 工作目录
```
/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/
```

## 步骤

### 1. 修复 notebook metadata
编辑 `hillclimb_submit.ipynb` 的 metadata，添加以下内容：
```json
{
  "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
  },
  "language_info": {
    "name": "python",
    "version": "3.12.3"
  }
}
```

### 2. 推送 notebook 新版本
```bash
kaggle kernels push
```

### 3. 等待 kernel 运行完成
```bash
kaggle kernels status smartorz/rogii-hill-climb-v3-notebook
```
等待直到 status 变成 COMPLETE 或 ERROR。

### 4. 检查提交
kernel 运行成功后，检查 Kaggle 提交列表：
```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
```
看有没有新生成的提交和分数。

### 5. 汇报结果
告诉我:
- kernel 运行状态
- 新提交的分数（如果有）
- 如果还是失败，把 ERROR 日志告诉我
