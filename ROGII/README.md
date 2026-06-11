# ROGII Wellbore Geology Prediction — CV Direction

预测水平井的 True Vertical Thickness (TVT)，基于井轨迹和 typewell 参考 GR 数据。

## 项目结构

```
ROGII/
├── README.md                       # 本文件
├── environment.yml                 # conda 环境定义
│
├── src/                            # 核心代码
│   ├── gen_images.py               # 2D 匹配网格图像生成 (HDF5)
│   ├── gen_features.py             # 表格特征工程 (NPZ)
│   ├── train.py                    # SegFormer 训练 + raw/anchored 双指标评估
│   ├── decode.py                   # SDF→TVT 解码 + 已知段 partial anchor (R4-A)
│   └── eval.py                     # 预训练 checkpoint 评估
│
├── data/
│   ├── cache/                      # 预处理后的数据集 (HDF5/NPZ)
│   │   ├── cfg-img-medium/         # 当前最佳: 50井, raw 15.89 / anc 14.28
│   │   ├── cfg-img-medium-100/     # R4-B fair: 100井
│   │   ├── cfg-img-medium-200/     # R2 旧版: 200井+不同val (21.77)
│   │   ├── cfg-img-medium-200-fair/# R4-B fair: 200井+同val
│   │   ├── cfg-img-medium-4ch/     # R3: +gr_diff 通道 (25.57, 反向)
│   │   ├── feat-lgb-base/          # 表格特征基线
│   │   └── feat-lgb-domain/        # +领域特征
│
├── docs/
│   ├── competition-description.md  # 比赛说明
│   ├── competition-insights.md     # 论坛精华整理
│   └── hengck23-reference/         # hengck23 参考代码 + 预训练权重
│
├── experiments/                    # 每轮设计文档 + 一次性脚本
│   ├── round_000-pf-discrete.md    # 早期 Particle Filter / Viterbi
│   ├── round_003/notes.md          # backbone/channel ablations
│   ├── round_004/notes.md          # decoding + fair data scaling
│   ├── round_004/gen_fair_scaling.py
│   └── scripts/                    # 早期 PF/offset 实验脚本
│
├── results/                        # 各轮最终指标 (JSON + 日志)
│   ├── summary.json                # 全部 config 总览
│   ├── round_002/  round_003/  round_004/
│
├── notebooks/                      # Jupyter (待用)
├── rogii-wellbore-geology-prediction/   # 原始竞赛数据 (只读)
└── .claude/skills/                 # 多 Agent 协作 skills
```

## 当前最佳结果

| Config | raw RMSE | **anchored RMSE** | 训练 | 备注 |
|--------|----------|-------------------|------|------|
| **cfg-img-medium + R4-A** | 15.89 | **14.28** ⭐ | 57s, 50井×15ep mit-b0 | 当前最佳（α=0.75 anchor） |
| cfg-img-medium (raw) | 15.89 | — | 57s | R2 基线 |
| cfg-img-medium-mitb1 | 15.93 | — | 114s | R3 - b1 持平 b0，容量饱和 |
| cfg-img-medium-4ch (+gr_diff) | 25.57 | — | 85s | R3 - gr_diff 通道反向 |
| cfg-img-medium-100 (fair) | 25.49 (raw) | — | — | R4-B 100井公平对比，**反而退化** |
| cfg-img-medium-200-fair | running | — | — | R4-B 进行中 |
| cfg-img-medium-200 (旧) | 21.77 | — | 375s | R2 旧版，val 不同 |
| feat-lgb-domain | 121.32 | — | 3s | LightGBM |
| feat-lgb-base | 211.73 | — | 3s | LightGBM |

**死路确认**：mit-b1 backbone (R3) / gr_diff 通道 (R3) / subpixel & smoothing 解码 (R4-A)。
**已锁定增益**：R4-A partial anchor (α=0.75, -1.61 ft, 零训练成本)。
**待 R4-B 完成判断**：数据扩规模轴 (50→100→200) 是否值得继续。

**参考**：hengck23 早期预训练 (00004053.pth) 在他 val 上 14.82 mean / 11.68 median。我们 anchored 14.28 已经过他。

## 核心方法

**SDF (Signed Distance Function) + SegFormer**
- 输入: (C=3, T=192, H=576) 2D 图像 — typewell GR / horizontal GR / history path
- 输出: SDF 热力图 = (h_tvt - t_tvt) / 40, clip [-3, 3]
- 解码: argmin(|sdf|) per column → t_tvt[idx] → partial anchor (R4-A)

参考 hengck23 [discussion #699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853)。

## 使用方法

```bash
# 1. 环境
pip install torch transformers h5py lightgbm scipy opencv-python-headless

# 2. 生成数据（gen_images.py 内部用 CONFIGS 字典；fair scaling 看 experiments/round_004/）
python src/gen_images.py
python src/gen_features.py

# 3. 训练（统一入口，自动报告 raw + anchored RMSE）
python src/train.py --dataset data/cache/cfg-img-medium
python src/train.py --dataset data/cache/cfg-img-medium --backbone nvidia/mit-b1 --epochs 20

# 4. 评估 hengck23 预训练
python src/eval.py --ckpt docs/hengck23-reference/00004053.pth
```

## Multi-Agent 实验流程

使用 [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python)：

```
cv-orchestrator (我)
  ├─ image-searcher  → 生成图像数据集
  ├─ feature-searcher → 生成特征矩阵
  ├─ train-validator → 训练 + 评估
  └─ experiment-tracker → 记录实验
```

详见 `.claude/skills/cv-orchestrator/SKILL.md` + `WORK_PROBLEMS.md`。
