---
name: playground-s6e4-qna
description: S6E4 irrigation baseline notebook 学习问答记录
type: project
---

# S6E4 Irrigation Baseline Notebook 学习问答

**日期：** 2026-04-01 ~ 2026-04-02
**背景：** 阅读 `irrigation-hgbc-xgb-lgbm-catb-realmlp-baseline.ipynb`，通过问答方式学习

---

## Q1: 为什么要合并外部数据集？不合并行不行？

**我的回答：** 不合并可以，我不知道这个人其他外面的数据哪来的

**补充：** 外部数据集 `miadul/irrigation-water-requirement-prediction-dataset` 是 Kaggle 上公开的。Playground 系列竞赛的数据通常是从某个真实数据集合成生成的，找到原始数据集加进去，相当于"开卷考试看到了答案的一部分"。

---

## Q2: 为什么要对类别特征算 target 的均值、标准差？

**我的回答：** 能得到这个土壤的分布，知道均值和标准差如果他这个类别对于某种分布比方说正态分布比较好，就可以合成数据

**纠正：** 不是用来合成数据的。是给模型一个"先验知识"。比如 Soil_Type=Clay 的 target 均值是 1.5（偏向 Medium/High），Sandy 均值是 0.3（偏向 Low）。模型看到均值特征就知道 Clay 更可能需要灌溉。标准差大说明该类别里 target 分散（难预测），小说明集中（容易预测）。本质是把 target 信息间接编码进特征里。

---

## Q2 追问: 他这个算的是目标值的均值是吧？就算知道了也不能直接加到特征里面啊

**回答：** 算出来之后变成新的数值特征列。所有 Soil_Type=Clay 的行，target_mean 都是同一个值。相比 OrdinalEncoder 只是编个号，target mean 直接告诉模型"这个类别 historically 对应什么 target"，信息量大得多。测试集只做 map，不会反过来影响统计量。

---

## Q3: 为什么要用 10 折交叉验证？

**我的回答：** 这个就是 bagging 了

**补充：** 方向对但不完整。10 折 CV 主要目的：1) 评估更可靠，每个样本都被验证一次；2) 充分利用数据；3) 顺带做了 Bagging，10 个模型预测取平均降方差。一举三得。

---

## Q4: Boosting 的核心思想是什么？

**我的回答：** boost 是先训练若干模型，然后把偏离值最大的那些数据加权，喂给下一个模型

**补充：** 对！这是 AdaBoost 的思路。现代 Boosting（XGB、LGBM、CatBoost、HGBC）稍有不同：不是改样本权重，而是让下一个模型去拟合残差（预测误差），一步步把误差缩小。

---

## 集成学习讨论：Bagging 不是复杂的 vote？

**我的回答：** Bagging不是复杂的vote？

**澄清：** 不完全一样但有相似之处。Voting 是不同模型各自预测取平均；Bagging 是同一模型在不同数据子集上训练取平均。区别：模型不同数据相同 vs 模型相同数据不同。RandomForest 就是 Bagging 代表。

**我的回应：** 这个就是bagging了（指 10 折 CV 取平均）

---

## Target mean 特征的进一步理解

**我的回答：** 就算我知道了也不能直接加到特征里面啊

**讨论：** 我联想到这个像 Stacking——用一个"模型"（groupby 取均值）的输出作为另一个模型的输入，是最轻量级的 Stacking。

---

## Q5: 为什么要"扔掉"特征？

**我的回答：** 特征当然不是越多越好。特征过多对于一些模型而言会降低质量

**补充：** 四个原因：1) 噪声干扰；2) 过拟合；3) 维度灾难；4) 训练变慢。用 XGBoost feature importance 筛掉贡献小的特征。

---

## Q6: 为什么用 balanced accuracy 而不是普通 accuracy？

**我的回答：** 这个我不知道什么是 ba，但是我感觉像是 auc 哪一块的，因为是分类问题，要考虑三个不同种类的准确率加权

**补充：** 方向对。Balanced Accuracy 就是每个类别的 accuracy 取平均。核心是处理类别不平衡——如果全预测 Low，普通 accuracy 可能 60%，但 balanced accuracy 只有 33%，能暴露模型其实很差。

---

## Q7: 怎么提升分数？

**我的回答：** 把十一个模型全部加上去

**纠正：** 不全对。弱模型（SGD、KNN）会拉低 ensemble 质量，SVC 在 64 万行上跑不动。正确策略是选几个强且互补的模型：XGBoost + LightGBM + CatBoost + HGBC，实现不同，犯的错误不同，取平均能互相补短板。

---

## Q8: Target mean 对整个训练集算统计量的问题怎么修复？

**我的回答：** 这个是数据泄露，但是其实不是数据泄露，因为你只是拿到一个平均值，而不是特征的具体值，所有可以，按照严格来说是数据泄露，应该只取 k 折的 train 的 mean

**补充：** 完全正确。严格来说验证集的 target 信息泄漏到了统计量里，但数据量 64 万，单个样本对均值影响极小，practically 影响不大。

---

## 总结学到的关键概念

- Target 统计特征工程（最轻量级 Stacking）
- 10 折交叉验证 = 可靠评估 + 数据利用 + 免费 Bagging
- Boosting vs Bagging 区别
- Balanced Accuracy 处理类别不平衡
- 特征选择的重要性
- Ensemble 要选强且互补的模型
