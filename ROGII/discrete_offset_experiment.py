"""
离散 Offset 实验：Viterbi 解最优离散状态路径

核心思路：
  1. offset (= dtvt + dz) 来自一个离散集合 (~40个值)
  2. offset 是分段常数，只在 ~10-40 个位置切换
  3. 用 Viterbi 在离散状态空间中找最优路径：
     - 状态：离散 offset 值
     - 转移概率：大概率保持，小概率切换
     - 观测概率：ML 预测与候选 offset 的接近程度
  4. 最优路径 → 分段常数 offset → cumsum 重建 TVT
"""

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from collections import Counter

TRAIN_DIR = '/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train'
EXPERIMENT_LOG = '/Users/liucong/code/kaggle/ROGII/discrete_offset_experiment_log.md'

# ================================================================
# 数据加载（复用 DataProcessor 逻辑）
# ================================================================

def load_well(filepath):
    df = pd.read_csv(filepath)
    ps = df['TVT_input'].notna().sum()
    n = len(df)
    x = df['X'].values.astype(np.float64)
    y = df['Y'].values.astype(np.float64)
    z = df['Z'].values.astype(np.float64)
    md = df['MD'].values.astype(np.float64)
    tvt = df['TVT'].values.astype(np.float64)
    dx = np.gradient(x); dy = np.gradient(y); dz = np.gradient(z)
    dmd = np.gradient(md); dtvt = np.gradient(tvt)
    bias = dtvt + dz
    return {
        'name': filepath.split('/')[-1].replace('__horizontal_well.csv', ''),
        'ps': ps, 'n': n,
        'x': x, 'y': y, 'z': z, 'md': md, 'tvt': tvt,
        'dx': dx, 'dy': dy, 'dz': dz, 'dmd': dmd,
        'dtvt': dtvt, 'bias': bias,
    }

def load_all_wells(max_wells=None):
    files = sorted(glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv'))
    if max_wells:
        files = files[:max_wells]
    wells = []
    for f in files:
        w = load_well(f)
        if w['ps'] >= 10 and (w['n'] - w['ps']) >= 100:
            wells.append(w)
    print(f"加载完成: {len(wells)} 口有效井")
    return wells

# ================================================================
# 特征构建
# ================================================================

def build_features(dx, dy, dz, dmd):
    eps = 1e-6
    features = []
    names = []
    def _add(arr, name, clip_val=100):
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.clip(arr, -clip_val, clip_val)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        features.append(arr); names.append(name)
    for name, val, cl in [('dx',dx,100),('dy',dy,100),('dz',dz,100),('dmd',dmd,10)]:
        _add(val, name, cl)
    _add(np.abs(dx), '|dx|'); _add(np.abs(dy), '|dy|'); _add(np.abs(dz), '|dz|')
    _add(np.sign(dx), 'sign(dx)',1); _add(np.sign(dy), 'sign(dy)',1); _add(np.sign(dz), 'sign(dz)',1)
    _add(np.arctan(dx), 'atan(dx)',2); _add(np.arctan(dy), 'atan(dy)',2); _add(np.arctan(dz), 'atan(dz)',2)
    _add(dx*dz, 'dx*dz',1000); _add(dy*dz, 'dy*dz',1000)
    _add(dx**2, 'dx^2',10000); _add(dz**2, 'dz^2',10000)
    _add(np.gradient(dz), 'd2z',100)
    return np.column_stack(features), names

# ================================================================
# 离散 Offset 词汇表
# ================================================================

def build_offset_vocabulary(wells, decimals=3, clip_range=(-0.1, 0.1)):
    """从训练井的已知段和水平段收集所有 offset 值，构建离散词汇表"""
    all_offsets = []
    for w in wells:
        ps = w['ps']
        # 已知段末尾
        known = w['bias'][max(0, ps - 300):ps]
        all_offsets.append(known)
        # 水平段
        all_offsets.append(w['bias'][ps:])

    all_off = np.concatenate(all_offsets)
    all_off = all_off[np.isfinite(all_off)]
    all_off = all_off[(all_off >= clip_range[0]) & (all_off <= clip_range[1])]

    rounded = np.round(all_off, decimals)
    vocab = np.sort(np.unique(rounded))
    print(f"Offset 词汇表: {len(vocab)} 个离散值 (精度={decimals}位)")
    print(f"  范围: [{vocab[0]:+.4f}, {vocab[-1]:+.4f}]")
    return vocab

# ================================================================
# Viterbi 解码器
# ================================================================

class DiscreteViterbi:
    """
    离散状态 Viterbi 解码

    状态: 离散 offset 值 (vocabulary)
    转移: log P(offset_j | offset_i)
    观测: log P(obs | offset_k) — 基于 ML 预测的似然
    """

    def __init__(self, vocabulary, stay_prob=0.997):
        """
        vocabulary: 离散 offset 候选值数组
        stay_prob: 保持当前 offset 的概率（越大越平滑）
        """
        self.vocab = np.asarray(vocabulary)
        self.K = len(self.vocab)
        self.stay_prob = stay_prob

        # 预计算转移矩阵 (log space)
        self.log_trans = np.full((self.K, self.K), np.log((1 - stay_prob) / (self.K - 1)))
        np.fill_diagonal(self.log_trans, np.log(stay_prob))

        # ML 观测模型（外部训练）
        self.obs_feature_mean = None
        self.obs_feature_std = None
        self.obs_weights = None

    def fit_observation_model(self, X_train, y_train, alpha=1.0):
        """训练岭回归观测模型"""
        self.obs_feature_mean = X_train.mean(axis=0)
        self.obs_feature_std = X_train.std(axis=0)
        self.obs_feature_std[self.obs_feature_std == 0] = 1.0
        X_s = (X_train - self.obs_feature_mean) / self.obs_feature_std
        X_bias = np.column_stack([np.ones(len(X_s)), X_s])
        XtX = X_bias.T @ X_bias
        XtX[np.arange(XtX.shape[0]), np.arange(XtX.shape[0])] += alpha
        Xty = X_bias.T @ y_train
        self.obs_weights = np.linalg.solve(XtX, Xty)
        print(f"观测模型训练完成: {len(y_train)} 样本")
        return self

    def predict_ml(self, X_step):
        """ML 对单步的连续预测"""
        X = np.atleast_2d(X_step)
        X_s = (X - self.obs_feature_mean) / self.obs_feature_std
        X_bias = np.column_stack([np.ones(len(X_s)), X_s])
        return float((X_bias @ self.obs_weights)[0])

    def compute_emission_logprob(self, X_lateral, obs_std=0.01):
        """
        计算每步对每个离散 offset 的观测对数概率

        X_lateral: (T, D) — 水平段轨迹特征
        obs_std: 观测噪声标准差

        返回: (T, K) log P(obs_t | state_k)
        """
        T = len(X_lateral)
        ml_preds = np.array([self.predict_ml(X_lateral[i]) for i in range(T)])

        # 高斯似然: log P(obs | offset_k) = -0.5 * ((ml_pred - offset_k) / obs_std)^2
        log_emission = np.zeros((T, self.K))
        for k in range(self.K):
            log_emission[:, k] = -0.5 * ((ml_preds - self.vocab[k]) / obs_std) ** 2

        # 数值稳定
        log_emission -= log_emission.max(axis=1, keepdims=True)
        return log_emission

    def viterbi_decode(self, log_emission, initial_logprob=None):
        """
        Viterbi 算法找最优状态序列

        log_emission: (T, K) — 每步每状态的观测对数概率
        initial_logprob: (K,) — 初始状态分布，None = 均匀分布

        返回: best_path (T,) — 每个时间步的最优状态索引
        """
        T, K = log_emission.shape

        # dp[t, k] = 到 t 步状态为 k 的最优对数概率
        dp = np.zeros((T, K))
        backtrack = np.zeros((T, K), dtype=int)

        # 初始分布
        if initial_logprob is None:
            dp[0] = log_emission[0]
        else:
            dp[0] = log_emission[0] + initial_logprob

        # 递推
        for t in range(1, T):
            for k in range(K):
                # dp[t,k] = emission[t,k] + max_j( dp[t-1,j] + trans[j,k] )
                scores = dp[t - 1] + self.log_trans[:, k]
                best_j = np.argmax(scores)
                dp[t, k] = log_emission[t, k] + scores[best_j]
                backtrack[t, k] = best_j

        # 回溯
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backtrack[t + 1, best_path[t + 1]]

        return best_path

    def decode_and_reconstruct(self, X_lateral, l_dz, start_tvt, obs_std=0.01):
        """
        完整流程：Viterbi 解码 → offset 序列 → TVT 重建

        返回: pred_tvt, offset_seq
        """
        log_emission = self.compute_emission_logprob(X_lateral, obs_std)
        best_path = self.viterbi_decode(log_emission)
        offset_seq = self.vocab[best_path]

        pred_tvt = np.zeros(len(l_dz))
        pred_tvt[0] = start_tvt + (-l_dz[0] + offset_seq[0])
        for i in range(1, len(l_dz)):
            pred_tvt[i] = pred_tvt[i - 1] + (-l_dz[i] + offset_seq[i])
        return pred_tvt, offset_seq


# ================================================================
# 评估与实验记录
# ================================================================

def reconstruct_tvt(start_tvt, dz, bias):
    n = len(dz)
    pred = np.zeros(n)
    pred[0] = start_tvt + (-dz[0] + bias[0])
    for i in range(1, n):
        pred[i] = pred[i - 1] + (-dz[i] + bias[i])
    return pred


def run_experiment():
    results_log = []
    results_log.append("# 离散 Offset Viterbi 实验记录\n")
    results_log.append(f"实验时间: {pd.Timestamp.now()}\n")

    # ---- 1. 加载数据 ----
    print("=" * 70)
    print("离散 Offset Viterbi 实验")
    print("=" * 70)
    print("\n[1/6] 加载数据...")
    wells = load_all_wells(max_wells=None)
    n_wells = len(wells)

    # 划分训练/验证集
    np.random.seed(42)
    indices = np.random.permutation(n_wells)
    n_train = int(0.7 * n_wells)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    print(f"训练井: {len(train_idx)}, 验证井: {len(val_idx)}")

    # ---- 2. 构建离散 Offset 词汇表 ----
    print(f"\n[2/6] 构建离散 Offset 词汇表...")
    train_wells = [wells[i] for i in train_idx]
    vocab = build_offset_vocabulary(train_wells, decimals=3)
    results_log.append(f"\n## 词汇表\n")
    results_log.append(f"- 精度: 3 位小数\n")
    results_log.append(f"- 唯一值数: {len(vocab)}\n")
    results_log.append(f"- 范围: [{vocab[0]:+.4f}, {vocab[-1]:+.4f}]\n")
    results_log.append(f"- 值: {list(np.round(vocab, 3))}\n")

    # ---- 3. 训练观测模型 ----
    print(f"\n[3/6] 训练观测模型（岭回归）...")
    X_list, y_list = [], []
    for idx in train_idx:
        w = wells[idx]
        ps = w['ps']
        sl = slice(max(0, ps - 300), ps)
        X, _ = build_features(w['dx'][sl], w['dy'][sl], w['dz'][sl], w['dmd'][sl])
        X_list.append(X)
        y_list.append(w['bias'][sl])
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    # ---- 4. Viterbi 超参数扫描（obs_std, stay_prob）----
    print(f"\n[4/6] Viterbi 超参数扫描...")

    obs_stds = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    stay_probs = [0.9, 0.95, 0.99, 0.995, 0.997, 0.999, 0.9995]

    sweep_wells = val_idx[:30]
    sweep_results = []

    for obs_std in obs_stds:
        for stay_prob in stay_probs:
            vtb = DiscreteViterbi(vocab, stay_prob=stay_prob)
            vtb.fit_observation_model(X_train, y_train, alpha=1.0)

            rmses = []
            for idx in sweep_wells:
                w = wells[idx]
                ps = w['ps']
                n_lat = min(w['n'] - ps, 1000)
                X_lat, _ = build_features(
                    w['dx'][ps:ps+n_lat], w['dy'][ps:ps+n_lat],
                    w['dz'][ps:ps+n_lat], w['dmd'][ps:ps+n_lat])
                pred_tvt, _ = vtb.decode_and_reconstruct(
                    X_lat, w['dz'][ps:ps+n_lat], w['tvt'][ps-1], obs_std)
                true_tvt = w['tvt'][ps:ps+n_lat]
                rmses.append(np.sqrt(np.mean((pred_tvt - true_tvt)**2)))
            sweep_results.append((np.mean(rmses), obs_std, stay_prob))

    sweep_results.sort()
    print(f"\nTop 10 参数组合 (30井×1000步):")
    print(f"{'Rank':>4} | {'RMSE':>8} | {'obs_std':>8} | {'stay_prob':>10}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for rank, (rmse, o, s) in enumerate(sweep_results[:10]):
        marker = " ← BEST" if rank == 0 else ""
        print(f"{rank+1:>4} | {rmse:>8.2f} | {o:>8.3f} | {s:>10.4f}{marker}")

    best_rmse, best_obs_std, best_stay_prob = sweep_results[0]

    results_log.append(f"\n## 超参数扫描\n")
    results_log.append(f"- 扫描范围: obs_std={obs_stds}, stay_prob={stay_probs}\n")
    results_log.append(f"- 最佳参数: obs_std={best_obs_std}, stay_prob={best_stay_prob}\n")
    results_log.append(f"- 最佳 RMSE (30井×1000步): {best_rmse:.2f}\n")
    results_log.append(f"\n### Top 10\n")
    results_log.append(f"| Rank | RMSE | obs_std | stay_prob |\n")
    results_log.append(f"|------|------|---------|-----------|\n")
    for rank, (rmse, o, s) in enumerate(sweep_results[:10]):
        results_log.append(f"| {rank+1} | {rmse:.2f} | {o:.3f} | {s:.4f} |\n")

    # ---- 5. 全量验证集评估 ----
    print(f"\n[5/6] 全量验证集评估（最佳参数）...")

    vtb = DiscreteViterbi(vocab, stay_prob=best_stay_prob)
    vtb.fit_observation_model(X_train, y_train, alpha=1.0)

    viterbi_rmses = []
    ml_rmses = []
    const_rmses = []
    oracle_rmses = []
    discrete_oracle_rmses = []

    for idx in val_idx:
        w = wells[idx]
        ps, n = w['ps'], w['n']
        n_lat = n - ps
        start_tvt = w['tvt'][ps - 1]
        l_dz = w['dz'][ps:]
        true_tvt = w['tvt'][ps:]

        X_lat, _ = build_features(
            w['dx'][ps:], w['dy'][ps:], w['dz'][ps:], w['dmd'][ps:])

        # ---- Viterbi ----
        pred_vtb, offset_seq = vtb.decode_and_reconstruct(
            X_lat, l_dz, start_tvt, best_obs_std)
        viterbi_rmses.append(np.sqrt(np.mean((pred_vtb - true_tvt)**2)))

        # ---- 纯 ML ----
        bias_ml = np.array([vtb.predict_ml(X_lat[i]) for i in range(len(X_lat))])
        pred_ml = reconstruct_tvt(start_tvt, l_dz, bias_ml)
        ml_rmses.append(np.sqrt(np.mean((pred_ml - true_tvt)**2)))

        # ---- 常数 ----
        const_bias = w['bias'][max(0, ps - 200):ps].mean()
        pred_c = reconstruct_tvt(start_tvt, l_dz, np.full(n_lat, const_bias))
        const_rmses.append(np.sqrt(np.mean((pred_c - true_tvt)**2)))

        # ---- 标量 Oracle ----
        best = float('inf')
        for b in np.arange(-0.1, 0.1, 0.001):
            pred_o = reconstruct_tvt(start_tvt, l_dz, np.full(n_lat, b))
            e = np.sqrt(np.mean((pred_o - true_tvt)**2))
            if e < best:
                best = e
        oracle_rmses.append(best)

        # ---- 离散 Oracle (真实 bias 量化到最近候选值) ----
        true_bias = w['bias'][ps:]
        quantized = np.array([vocab[np.argmin(np.abs(vocab - b))] for b in true_bias])
        pred_q = reconstruct_tvt(start_tvt, l_dz, quantized)
        discrete_oracle_rmses.append(np.sqrt(np.mean((pred_q - true_tvt)**2)))

    # ---- 6. 结果 ----
    print(f"\n[6/6] 全量验证集结果（{len(val_idx)} 口井）")
    print("=" * 70)
    print(f"{'方法':>30} | {'均值':>8} | {'中位数':>8} | {'最小':>8} | {'最大':>8}")
    print(f"{'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    methods = [
        ('Oracle (搜索标量)', oracle_rmses),
        ('离散 Oracle (量化真实bias)', discrete_oracle_rmses),
        ('常数 bias', const_rmses),
        ('纯 ML (岭回归连续)', ml_rmses),
        ('★ Viterbi (离散offset)', viterbi_rmses),
    ]

    refs = {}
    for name, arr in methods:
        a = np.array(arr)
        refs[name] = a.mean()
        print(f"{name:>30} | {a.mean():>8.2f} | {np.median(a):>8.2f} "
              f"| {a.min():>8.2f} | {a.max():>8.2f}")

    print(f"\n改进幅度:")
    print(f"  Viterbi vs 常数:      {(1 - refs['★ Viterbi (离散offset)'] / refs['常数 bias']) * 100:+.1f}%")
    print(f"  Viterbi vs 纯 ML:     {(1 - refs['★ Viterbi (离散offset)'] / refs['纯 ML (岭回归连续)']) * 100:+.1f}%")
    print(f"  Viterbi vs 离散Oracle: 差距 {refs['★ Viterbi (离散offset)'] - refs['离散 Oracle (量化真实bias)']:.2f} ft")
    print(f"  离散Oracle vs 标量Oracle: {(1 - refs['离散 Oracle (量化真实bias)'] / refs['Oracle (搜索标量)']) * 100:+.1f}%")

    # 记录到日志
    results_log.append(f"\n## 全量验证集结果（{len(val_idx)} 口井, 完整水平段）\n")
    results_log.append(f"| 方法 | 均值 RMSE | 中位数 | 最小 | 最大 |\n")
    results_log.append(f"|------|-----------|--------|------|------|\n")
    for name, arr in methods:
        a = np.array(arr)
        results_log.append(f"| {name} | {a.mean():.2f} | {np.median(a):.2f} | {a.min():.2f} | {a.max():.2f} |\n")

    results_log.append(f"\n## 改进幅度\n")
    results_log.append(f"- Viterbi vs 常数: {(1 - refs['★ Viterbi (离散offset)'] / refs['常数 bias']) * 100:+.1f}%\n")
    results_log.append(f"- Viterbi vs 纯 ML: {(1 - refs['★ Viterbi (离散offset)'] / refs['纯 ML (岭回归连续)']) * 100:+.1f}%\n")
    results_log.append(f"- Viterbi vs 离散Oracle: 差距 {refs['★ Viterbi (离散offset)'] - refs['离散 Oracle (量化真实bias)']:.2f} ft\n")

    # ---- 写入日志 ----
    with open(EXPERIMENT_LOG, 'w') as f:
        f.writelines(results_log)
    print(f"\n实验日志已保存: {EXPERIMENT_LOG}")

if __name__ == '__main__':
    run_experiment()
