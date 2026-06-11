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
│   ├── train.py                    # SegFormer 训练 + 评估
│   └── eval.py                     # 预训练模型评估
│
├── data/
│   ├── cache/                      # 预处理后的数据集 (HDF5/NPZ)
│   │   ├── cfg-img-medium/         # 当前最佳: 50井, RMSE 15.89
│   │   ├── cfg-img-medium-200/     # 200井版本: RMSE 21.77
│   │   ├── cfg-img-grdiff/         # 4ch (含 gr_diff) 版本
│   │   ├── feat-lgb-base/          # 基础表格特征
│   │   └── feat-lgb-domain/        # 加领域特征版本
│
├── docs/
│   ├── competition-description.md  # 比赛说明
│   ├── competition-insights.md     # 论坛精华整理
│   └── hengck23-reference/         # hengck23 的参考代码 + 预训练权重
│
├── experiments/                    # 实验记录
│   ├── round_001/                  # 冷启动宽搜索
│   ├── round_002/                  # 修 bug + 特征训练
│   ├── round_000-pf-discrete.md    # 早期 Particle Filter / 离散 Viterbi
│   ├── scripts/                    # 早期方案脚本 (PF / 离散 offset)
│   ├── index.json / leaderboard.json / search_state.json
│
├── results/                        # 各轮实验最终指标
│   ├── summary.json                # 汇总
│   ├── round_002/
│   └── round_003/
│
├── notebooks/                      # Jupyter 笔记本 (待添加)
│
├── rogii-wellbore-geology-prediction/   # 原始竞赛数据 (不可改)
│
└── .claude/skills/                 # 多 Agent 协作 skills
    ├── cv-orchestrator/            # 调度者 skill + WORK_PROBLEMS.md
    ├── image-searcher/             # 图像构造搜索 agent
    ├── feature-searcher/           # 特征工程搜索 agent
    ├── train-validator/            # 训练验证 agent
    └── experiment-tracker/         # 实验记录 agent
```

## 当前最佳结果

| Config | Val RMSE (ft) | 训练时间 | 模型 | 备注 |
|--------|---------------|----------|------|------|
| **cfg-img-medium** | **15.89** | 57s, 50井×15ep | SegFormer mit-b0 | ⭐ 基线 (R2) |
| cfg-img-medium-mitb1 | 15.93 | 114s | SegFormer mit-b1 | R3 - 持平，b0 容量已饱和 |
| cfg-img-medium-4ch | 25.57 | 85s | SegFormer mit-b0 (+gr_diff) | R3 - gr_diff 通道有害 |
| cfg-img-medium-200 | 21.77 | 375s, 200井×20ep | SegFormer mit-b0 | val 集不同 → 待 R4 公平对比 |
| feat-lgb-domain | 121.32 | 3s | LightGBM | 跨井绝对 RMSE |
| feat-lgb-base | 211.73 | 3s | LightGBM | 跨井绝对 RMSE |
| (常量基线) | 638 | — | mean TVT | |

**Round 3 死路**：mit-b1 backbone / gr_diff 第 4 通道。**Round 4 候选**：公平 200 井对比、SDF scale 细搜索、MTP head、beam search 后处理。

**参考**：hengck23 早期预训练 SegFormer (00004053.pth) 在他自己的 val 井上: 14.82 mean / 11.68 median RMSE。我们的 15.89 已接近其水平。

## 核心方法

**SDF (Signed Distance Function) + SegFormer**
- 输入: (C=3, T=192, H=576) 2D 图像 — typewell GR / horizontal GR / history path
- 输出: SDF 热力图 = (h_tvt - t_tvt) / 40
- 预测: argmin(|sdf|) per column → typewell TVT 查表

参考 hengck23 在 [discussion #699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853) 提出的 CNN+MTP 架构。

## 使用方法

```bash
# 1. 环境
pip install torch transformers h5py lightgbm scipy opencv-python-headless

# 2. 生成数据
python src/gen_images.py --config cfg-img-medium  # 生成 HDF5
python src/gen_features.py --config feat-lgb-base  # 生成 NPZ

# 3. 训练 (统一入口, 支持 --backbone / --epochs)
python src/train.py --dataset data/cache/cfg-img-medium
python src/train.py --dataset data/cache/cfg-img-medium --backbone nvidia/mit-b1 --epochs 20

# 4. 评估预训练
python src/eval.py --ckpt docs/hengck23-reference/00004053.pth
```

## Multi-Agent 实验流程

本项目使用 [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python) 实现多 Agent 协作：

```
cv-orchestrator (我)
  ├─ image-searcher  → 生成图像数据集
  ├─ feature-searcher → 生成特征矩阵
  ├─ train-validator → 训练 + 评估
  └─ experiment-tracker → 记录实验
```

详见 `.claude/skills/cv-orchestrator/SKILL.md`。
