"""
偏差多项式实验 v5：轻量自回归

核心发现 (v3)：
  - 滑动窗口特征 (dx,dy,dz)  → 无改善 (20.72)
  - bias历史特征 (作弊)        → 0.58  (知道history=作弊)

解决思路：
  测试段第 i 步的 bias 预测 = f(trajectory[i], bias[i-1], bias[i-5], bias[i-20])

  其中 bias[i-1] 是第 i-1 步的预测值（自回归）
  bias[i-5], bias[i-20] 来自已知段或已预测区间

  模型: 用已知段训练，已知段有完整 bias 序列
  推断: PS 之后逐步预测，用上一步 pred_bias 作为下一步的特征
"""

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler

TRAIN_DIR = '/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train'
MAX_WELLS = None


def add(features, name, feature, clip_val=100):
    f = np.clip(np.asarray(feature, dtype=np.float64), -clip_val, clip_val)
    f = np.where(np.isfinite(f), f, 0)
    features.append(f)
    return name


def create_lagged_bias(bias_seq, lags=[1, 5, 10, 20, 50]):
    """从 bias 序列创建滞后特征。前几个点用 0 填充。"""
    n = len(bias_seq)
    result = np.zeros((n, len(lags)))
    for col, lag in enumerate(lags):
        result[lag:, col] = bias_seq[:-lag]
        # 前 lag 个点：用第一个已知值填充
        if len(bias_seq) > 0:
            first_val = bias_seq[0]
            result[:lag, col] = first_val
    return result


def build_features_v5(dx, dy, dz, dmd, bias_seq=None):
    """
    bias_seq: 完整 bias 序列（训练时真实，测试时逐步填入预测值）
    如果 None，滞后特征全部为 0（退化版）
    """
    eps = 1e-6
    features = []
    names = []

    # ====== 瞬时（仅关键特征） ======
    for name, val, cl in [('dx', dx, 100), ('dy', dy, 100), ('dz', dz, 100), ('dmd', dmd, 10)]:
        names.append(add(features, name, val, cl))
    names.append(add(features, '|dx|', np.abs(dx)))
    names.append(add(features, '|dy|', np.abs(dy)))
    names.append(add(features, '|dz|', np.abs(dz)))
    names.append(add(features, 'sign(dx)', np.sign(dx), 1))
    names.append(add(features, 'sign(dy)', np.sign(dy), 1))
    names.append(add(features, 'sign(dz)', np.sign(dz), 1))
    names.append(add(features, 'atan(dx)', np.arctan(dx), 2))
    names.append(add(features, 'atan(dy)', np.arctan(dy), 2))
    names.append(add(features, 'atan(dz)', np.arctan(dz), 2))
    names.append(add(features, 'dx*dz', dx * dz, 1000))
    names.append(add(features, 'dy*dz', dy * dz, 1000))
    names.append(add(features, 'dx^2', dx**2, 10000))
    names.append(add(features, 'dz^2', dz**2, 10000))

    # ====== 加速度 ======
    d2z = np.gradient(dz)
    names.append(add(features, 'd2z', d2z, 100))

    # ====== ★ 关键：bias 滞后特征 ======
    if bias_seq is not None:
        lags = [1, 5, 10, 20, 50]
        lagged = create_lagged_bias(bias_seq, lags)
        for col, lag in enumerate(lags):
            names.append(add(features, f'bias_lag{lag}', lagged[:, col]))

        # bias 趋势 = bias 的近期变化
        # delta = bias[i] - bias[i-20]
        delta = np.zeros(len(bias_seq))
        delta[20:] = bias_seq[20:] - bias_seq[:-20]
        delta[:20] = bias_seq[:20] - bias_seq[0]
        names.append(add(features, 'bias_delta20', delta))

        # bias 加速度
        if len(bias_seq) > 2:
            d2 = np.gradient(bias_seq)
            names.append(add(features, 'bias_d2', d2))

    return np.column_stack(features), names


def load_well_data(f):
    df = pd.read_csv(f)
    ps = df['TVT_input'].notna().sum()
    return {
        'ps': ps, 'n': len(df),
        'z': df['Z'].values, 'md': df['MD'].values,
        'tvt': df['TVT'].values,
        'dx': np.gradient(df['X'].values),
        'dy': np.gradient(df['Y'].values),
        'dz': np.gradient(df['Z'].values),
        'dmd': np.gradient(df['MD'].values),
        'dtvt': np.gradient(df['TVT'].values),
        'target': np.gradient(df['TVT'].values) + np.gradient(df['Z'].values)
    }


def predict_tvt_cumsum(start_tvt, dz, bias):
    n = len(dz)
    pred = np.zeros(n)
    pred[0] = start_tvt + (-dz[0] + bias[0])
    for i in range(1, n):
        pred[i] = pred[i-1] + (-dz[i] + bias[i])
    return pred


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.weights = None

    def fit(self, X, y):
        X_s = self.scaler.fit_transform(X)
        X_bias = np.column_stack([np.ones(len(X_s)), X_s])
        XtX = X_bias.T @ X_bias
        XtX[np.arange(XtX.shape[0]), np.arange(XtX.shape[0])] += self.alpha
        Xty = X_bias.T @ y
        self.weights = np.linalg.solve(XtX, Xty)

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_bias = np.column_stack([np.ones(len(X_s)), X_s])
        return X_bias @ self.weights


def main():
    print("=" * 70)
    print("偏差多项式实验 v5：轻量自回归")
    print("=" * 70)

    files = sorted(glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv'))
    if MAX_WELLS:
        files = files[:MAX_WELLS]

    print(f"\n加载 {len(files)} 口井...")
    wells = []
    for f in files:
        w = load_well_data(f)
        if w['ps'] >= 10 and w['ps'] < w['n'] - 100:
            wells.append(w)
    print(f"有效井数: {len(wells)}")

    n_wells = len(wells)
    n_train = int(0.8 * n_wells)
    indices = np.random.RandomState(42).permutation(n_wells)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # ================================================================
    # 训练：用真实 bias 序列
    # ================================================================
    print(f"\n{'='*70}")
    print("训练模型（真实 bias 序列作特征）")
    print(f"{'='*70}")

    X_list, y_list = [], []
    for idx in train_idx:
        w = wells[idx]
        ps = w['ps']
        ts = slice(max(0, ps - 300), ps)
        X, _ = build_features_v5(
            w['dx'][ts], w['dy'][ts], w['dz'][ts], w['dmd'][ts],
            bias_seq=w['target'][ts]
        )
        X_list.append(X)
        y_list.append(w['target'][ts])

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    model = RidgeModel(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"训练完成: {len(y_train)} 样本, {X_train.shape[1]} 特征")

    _, feat_names = build_features_v5(
        wells[0]['dx'][:10], wells[0]['dy'][:10],
        wells[0]['dz'][:10], wells[0]['dmd'][:10],
        bias_seq=wells[0]['target'][:10]
    )
    print(f"\nTop 15 特征 (按 |系数|):")
    coef = model.weights[1:]
    for rank, idx in enumerate(np.argsort(np.abs(coef))[::-1][:15]):
        print(f"  {rank+1:2d}. {feat_names[idx]:20s}: {coef[idx]:12.6f}")

    # ================================================================
    # 评估
    # ================================================================
    print(f"\n{'='*70}")
    print("评估（验证集）- 自回归 vs 作弊 vs 基线")
    print(f"{'='*70}")

    results = {'oracle': [], 'const': [], 'cheat': [], 'ar': []}

    for idx in val_idx:
        w = wells[idx]
        ps = w['ps']
        n = w['n']
        n_lateral = n - ps
        if n_lateral < 100:
            continue

        start_tvt = w['tvt'][ps - 1]
        true_tvt = w['tvt'][ps:]
        l_dz = w['dz'][ps:]

        # ---- 作弊版 (真实 bias 序列) ----
        X_cheat, _ = build_features_v5(
            w['dx'], w['dy'], w['dz'], w['dmd'],
            bias_seq=w['target']
        )
        bias_cheat = model.predict(X_cheat)
        pred_cheat = predict_tvt_cumsum(start_tvt, l_dz, bias_cheat[ps:])
        results['cheat'].append(np.sqrt(np.mean((pred_cheat - true_tvt)**2)))

        # ---- 自回归版 ----
        # 构建 bias 序列：前 ps 步用真实值，后面逐步预测填入
        bias_ar = np.zeros(n)
        bias_ar[:ps] = w['target'][:ps]  # 已知段真实 bias

        for i in range(ps, n):
            # 只取最近 ctx_len 步来构建特征
            ctx_len = 100  # 足够覆盖 lag50
            ctx_start = max(0, i - ctx_len + 1)
            n_ctx = i - ctx_start + 1

            Xi, _ = build_features_v5(
                w['dx'][ctx_start:i+1],
                w['dy'][ctx_start:i+1],
                w['dz'][ctx_start:i+1],
                w['dmd'][ctx_start:i+1],
                bias_seq=bias_ar[ctx_start:i+1]  # 已预测的 bias 序列
            )
            bias_i = model.predict(Xi[-1:])[0]
            bias_ar[i] = bias_i

        pred_ar = predict_tvt_cumsum(start_tvt, l_dz, bias_ar[ps:])
        results['ar'].append(np.sqrt(np.mean((pred_ar - true_tvt)**2)))

        # ---- 常数 ----
        const_bias = w['target'][ps - 200:ps].mean()
        pred_c = predict_tvt_cumsum(start_tvt, l_dz, np.full(n_lateral, const_bias))
        results['const'].append(np.sqrt(np.mean((pred_c - true_tvt)**2)))

        # ---- Oracle ----
        best = float('inf')
        for b in np.arange(-0.1, 0.1, 0.001):
            pred_o = predict_tvt_cumsum(start_tvt, l_dz, np.full(n_lateral, b))
            rmse = np.sqrt(np.mean((pred_o - true_tvt)**2))
            if rmse < best:
                best = rmse
        results['oracle'].append(best)

    # 打印结果
    print(f"\n{'='*70}")
    print(f"结果对比（验证集 {len(results['oracle'])} 口井）")
    print(f"{'='*70}")
    print(f"{'方法':>25} | {'均值':>8} | {'中位数':>8}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}")
    labels = [
        ('Oracle (搜索标量)', 'oracle'),
        ('常数bias (已知段)', 'const'),
        ('ML 作弊 (真实bias序列)', 'cheat'),
        ('ML 自回归 (逐步预测)', 'ar'),
    ]
    ref = {}
    for name, key in labels:
        arr = np.array(results[key])
        print(f"{name:>25} | {arr.mean():>8.2f} | {np.median(arr):>8.2f}")
        ref[key] = arr.mean()

    print(f"\n改进幅度:")
    print(f"  自回归 vs 常数:          {(1 - ref['ar'] / ref['const']) * 100:.1f}%")
    print(f"  作弊版 vs Oracle:         {(1 - ref['cheat'] / ref['oracle']) * 100:.1f}%")
    print(f"  自回归 vs Oracle 剩余差距:  {ref['ar'] - ref['oracle']:.2f} ft")


if __name__ == '__main__':
    main()
