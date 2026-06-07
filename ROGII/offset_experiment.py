"""
偏差多项式实验：用 dx, dy, dz 的初等函数组合逼近 dtvt + dz

核心发现：
- dtvt ≈ -dz + bias（bias 是每步的系统性偏差）
- 地层平坦时 bias ≈ 0，倾斜时 bias 有一个小的常数偏移
- 如果能准确估计这个 bias，cumsum(-dz + bias) 能很好预测 TVT

hengck23 报告：
- Oracle (知道正确全局 offset): ~7.64 RMSE
- 已知段 offset: ~37-39 RMSE
- fold-safe selector: ~14.8 RMSE
"""

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler

# ============================================================
# 配置
# ============================================================
TRAIN_DIR = '/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train'
MAX_WELLS = None  # None = 全部井


def build_features(dx, dy, dz, dmd):
    """
    构建初等函数组合特征。每项是一个初等函数，带一个系数（通过线性回归学习）。

    特征包括：线性、平方、平方根、对数、交互、混合根式、符号、指数、atan、比值、立方、绝对值
    """
    eps = 1e-6
    features = []
    names = []

    def add(feature, name, clip_val=100):
        """安全添加特征：裁剪极端值"""
        f = np.clip(np.asarray(feature, dtype=np.float64), -clip_val, clip_val)
        # 额外检查：替换 NaN/Inf
        f = np.where(np.isfinite(f), f, 0)
        features.append(f)
        names.append(name)

    # 1. 线性项
    add(dx, 'dx', 100)
    add(dy, 'dy', 100)
    add(dz, 'dz', 100)
    add(dmd, 'dmd', 10)

    # 2. 平方项
    add(dx**2, 'dx^2', 10000)
    add(dy**2, 'dy^2', 10000)
    add(dz**2, 'dz^2', 10000)

    # 3. 平方根项
    add(np.sqrt(np.abs(dx) + eps), 'sqrt|dx|', 50)
    add(np.sqrt(np.abs(dy) + eps), 'sqrt|dy|', 50)
    add(np.sqrt(np.abs(dz) + eps), 'sqrt|dz|', 50)

    # 4. 对数项
    add(np.log1p(np.abs(dx)), 'log1p|dx|', 10)
    add(np.log1p(np.abs(dy)), 'log1p|dy|', 10)
    add(np.log1p(np.abs(dz)), 'log1p|dz|', 10)

    # 5. 交互项（乘积）
    add(dx * dy, 'dx*dy', 1000)
    add(dx * dz, 'dx*dz', 1000)
    add(dy * dz, 'dy*dz', 1000)
    add(dx * dmd, 'dx*dmd', 100)
    add(dy * dmd, 'dy*dmd', 100)
    add(dz * dmd, 'dz*dmd', 100)

    # 6. 混合根式
    add(np.sqrt(dx**2 + dy**2 + eps), 'sqrt(dx^2+dy^2)', 100)
    add(np.sqrt(dx**2 + dz**2 + eps), 'sqrt(dx^2+dz^2)', 100)
    add(np.sqrt(dy**2 + dz**2 + eps), 'sqrt(dy^2+dz^2)', 100)
    add(np.sqrt(dx**2 + dy**2 + dz**2 + eps), 'sqrt(dx^2+dy^2+dz^2)', 100)

    # 7. 符号交互
    add(np.sign(dx) * np.sign(dz), 'sign(dx)*sign(dz)', 1)
    add(np.sign(dy) * np.sign(dz), 'sign(dy)*sign(dz)', 1)

    # 8. 指数项（安全的）
    add(np.exp(-np.clip(np.abs(dx), 0, 10)), 'exp(-|dx|)', 1)
    add(np.exp(-np.clip(np.abs(dy), 0, 10)), 'exp(-|dy|)', 1)
    add(np.exp(-np.clip(np.abs(dz), 0, 10)), 'exp(-|dz|)', 1)

    # 9. atan（压缩大范围）
    add(np.arctan(dx), 'atan(dx)', 2)
    add(np.arctan(dy), 'atan(dy)', 2)
    add(np.arctan(dz), 'atan(dz)', 2)

    # 10. 安全比值
    add(dx / (np.abs(dz) + 1), 'dx/(|dz|+1)', 100)
    add(dy / (np.abs(dz) + 1), 'dy/(|dz|+1)', 100)
    add(dz / (np.abs(dx) + 1), 'dz/(|dx|+1)', 100)

    # 11. 立方
    add(dx**3, 'dx^3', 1e6)
    add(dy**3, 'dy^3', 1e6)
    add(dz**3, 'dz^3', 1e6)

    # 12. 绝对值
    add(np.abs(dx), '|dx|', 100)
    add(np.abs(dy), '|dy|', 100)
    add(np.abs(dz), '|dz|', 100)

    # 13. 方向
    add(np.sign(dx), 'sign(dx)', 1)
    add(np.sign(dy), 'sign(dy)', 1)
    add(np.sign(dz), 'sign(dz)', 1)

    return np.column_stack(features), names


def load_well_data(f):
    """加载单井数据并计算梯度"""
    df = pd.read_csv(f)
    ps = df['TVT_input'].notna().sum()

    h_x = df['X'].values
    h_y = df['Y'].values
    h_z = df['Z'].values
    h_md = df['MD'].values
    h_tvt = df['TVT'].values

    dx = np.gradient(h_x)
    dy = np.gradient(h_y)
    dz = np.gradient(h_z)
    dmd = np.gradient(h_md)
    dtvt = np.gradient(h_tvt)

    return {
        'ps': ps, 'n': len(df),
        'x': h_x, 'y': h_y, 'z': h_z, 'md': h_md, 'tvt': h_tvt,
        'dx': dx, 'dy': dy, 'dz': dz, 'dmd': dmd, 'dtvt': dtvt,
        'target': dtvt + dz  # 每步偏差 = dtvt + dz
    }


def predict_tvt_cumsum(start_tvt, dz, bias):
    """
    用 cumsum(-dz + bias) 方法预测 TVT

    pred[i] = pred[i-1] + (-dz[i] + bias[i])

    dz 和 bias 应该是已经切片的水平段数据（长度相同）
    """
    n = len(dz)
    assert len(bias) == n, f"dz length {n} != bias length {len(bias)}"

    pred = np.zeros(n)
    pred[0] = start_tvt + (-dz[0] + bias[0])
    for i in range(1, n):
        pred[i] = pred[i-1] + (-dz[i] + bias[i])
    return pred


def evaluate_methods(wells, well_indices):
    """评估三种方法：Oracle、常数 bias、ML bias"""
    oracle_rmses = []
    const_rmses = []
    ml_rmses = []

    for idx in well_indices:
        w = wells[idx]
        ps = w['ps']
        n_lateral = w['n'] - ps

        if n_lateral < 100:
            continue

        start_tvt = w['tvt'][ps - 1]
        true_tvt = w['tvt'][ps:]

        l_dz = w['dz'][ps:]

        # --- Oracle: 搜索最佳标量 bias ---
        best_rmse = float('inf')
        best_bias = 0
        for b in np.arange(-0.1, 0.1, 0.0005):
            pred = predict_tvt_cumsum(start_tvt, l_dz, np.full(n_lateral, b))
            rmse = np.sqrt(np.mean((pred - true_tvt)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_bias = b
        oracle_rmses.append(best_rmse)

        # --- 常数 bias: 用已知段均值 ---
        const_bias = w['target'][ps - 200:ps].mean()
        pred = predict_tvt_cumsum(start_tvt, l_dz, np.full(n_lateral, const_bias))
        const_rmses.append(np.sqrt(np.mean((pred - true_tvt)**2)))

        # --- ML bias: 用训练的模型预测 ---
        # (需要在外部传入模型)
        ml_rmses.append(None)

    return {
        'oracle': np.array(oracle_rmses),
        'const': np.array(const_rmses),
    }


def main():
    print("=" * 70)
    print("偏差多项式实验")
    print("=" * 70)

    # 加载所有井
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

    # ============================================================
    # 步骤 1: 基础评估（Oracle + 常数 bias）
    # ============================================================
    print(f"\n{'='*70}")
    print("步骤 1: Oracle vs 常数 bias（无需 ML）")
    print(f"{'='*70}")

    n_wells = len(wells)
    n_train = int(0.8 * n_wells)
    indices = np.random.RandomState(42).permutation(n_wells)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    results = evaluate_methods(wells, val_idx)

    print(f"\n验证集 ({len(val_idx)} 口井):")
    print(f"  Oracle (搜索最佳标量 bias): 均值={results['oracle'].mean():.2f}, 中位数={np.median(results['oracle']):.2f}")
    print(f"  常数 bias (已知段200点均值): 均值={results['const'].mean():.2f}, 中位数={np.median(results['const']):.2f}")

    # ============================================================
    # 步骤 2: 收集 ML 训练数据
    # ============================================================
    print(f"\n{'='*70}")
    print("步骤 2: 训练偏差多项式（线性回归）")
    print(f"{'='*70}")

    X_list = []
    y_list = []

    for idx in train_idx:
        w = wells[idx]
        ps = w['ps']

        # 只用已知段训练（严格无泄漏）
        known = slice(ps - 300, ps)

        X, feature_names = build_features(
            w['dx'][known], w['dy'][known], w['dz'][known], w['dmd'][known]
        )
        y = w['target'][known]

        X_list.append(X)
        y_list.append(y)

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    print(f"训练样本: {len(y_train)}, 特征数: {X_train.shape[1]}")
    print(f"目标 (dtvt+dz) 均值: {y_train.mean():.6f}, 标准差: {y_train.std():.6f}")

    # 标准化 + 正规方程（数值稳定）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_bias = np.column_stack([np.ones(len(X_train_s)), X_train_s])

    # Ridge 正则化解
    alpha = 1.0
    # (X^T X + alpha*I) w = X^T y
    XtX = X_bias.T @ X_bias
    XtX[np.arange(XtX.shape[0]), np.arange(XtX.shape[0])] += alpha
    Xty = X_bias.T @ y_train
    weights = np.linalg.solve(XtX, Xty)

    print(f"训练完成。正则化 alpha={alpha}")

    # 特征重要性
    print(f"\nTop 15 特征 (按 |系数|):")
    coef = weights[1:]  # 跳过偏置项
    for rank, idx in enumerate(np.argsort(np.abs(coef))[::-1][:15]):
        print(f"  {rank+1:2d}. {feature_names[idx]:25s}: {coef[idx]:12.6f}")

    # ============================================================
    # 步骤 3: ML 预测 TVT
    # ============================================================
    print(f"\n{'='*70}")
    print("步骤 3: ML bias 预测 TVT（验证集）")
    print(f"{'='*70}")

    ml_rmses = []
    const_rmses = []
    oracle_rmses = []

    for idx in val_idx:
        w = wells[idx]
        ps = w['ps']
        n_lateral = w['n'] - ps

        if n_lateral < 100:
            continue

        start_tvt = w['tvt'][ps - 1]
        true_tvt = w['tvt'][ps:]
        l_dz = w['dz'][ps:]

        # 构建全井特征
        X_all, _ = build_features(w['dx'], w['dy'], w['dz'], w['dmd'])
        X_all_s = scaler.transform(X_all)
        X_all_bias = np.column_stack([np.ones(len(X_all_s)), X_all_s])
        bias_pred = X_all_bias @ weights

        # ML 预测
        pred_ml = predict_tvt_cumsum(start_tvt, l_dz, bias_pred[ps:])
        ml_rmses.append(np.sqrt(np.mean((pred_ml - true_tvt)**2)))

        # 常数 bias 对比
        const_bias = w['target'][ps - 200:ps].mean()
        pred_const = predict_tvt_cumsum(start_tvt, l_dz, np.full(n_lateral, const_bias))
        const_rmses.append(np.sqrt(np.mean((pred_const - true_tvt)**2)))

        # Oracle 对比
        best = float('inf')
        for b in np.arange(-0.1, 0.1, 0.001):
            pred = predict_tvt_cumsum(start_tvt, l_dz, np.full(n_lateral, b))
            rmse = np.sqrt(np.mean((pred - true_tvt)**2))
            if rmse < best:
                best = rmse
        oracle_rmses.append(best)

    ml_rmses = np.array(ml_rmses)
    const_rmses = np.array(const_rmses)
    oracle_rmses = np.array(oracle_rmses)

    print(f"\n{'='*70}")
    print(f"结果对比（验证集 {len(ml_rmses)} 口井）")
    print(f"{'='*70}")
    print(f"{'方法':>25} | {'均值':>8} | {'中位数':>8} | {'最小':>8} | {'最大':>8}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"{'Oracle (搜索bias)':>25} | {oracle_rmses.mean():>8.2f} | {np.median(oracle_rmses):>8.2f} | {oracle_rmses.min():>8.2f} | {oracle_rmses.max():>8.2f}")
    print(f"{'常数bias (已知段)':>25} | {const_rmses.mean():>8.2f} | {np.median(const_rmses):>8.2f} | {const_rmses.min():>8.2f} | {const_rmses.max():>8.2f}")
    print(f"{'ML bias (dx,dy...)':>25} | {ml_rmses.mean():>8.2f} | {np.median(ml_rmses):>8.2f} | {ml_rmses.min():>8.2f} | {ml_rmses.max():>8.2f}")

    print(f"\n改进:")
    print(f"  ML vs 常数: {(1 - ml_rmses.mean() / const_rmses.mean()) * 100:.1f}%")
    print(f"  Oracle vs 常数: {(1 - oracle_rmses.mean() / const_rmses.mean()) * 100:.1f}%")

    # ============================================================
    # 与 hengck23 对比
    # ============================================================
    print(f"\n{'='*70}")
    print(f"与讨论区 hengck23 报告对比")
    print(f"{'='*70}")
    print(f"hengck23:")
    print(f"  - Oracle (cumsum -dz - offset): ~7.64 RMSE")
    print(f"  - 已知段 offset: ~37-39 RMSE")
    print(f"  - fold-safe selector: ~14.8 RMSE")
    print(f"")
    print(f"本实验:")
    print(f"  - Oracle (搜索标量 bias): {oracle_rmses.mean():.2f} RMSE")
    print(f"  - 常数 bias (已知段): {const_rmses.mean():.2f} RMSE")
    print(f"  - ML bias (43特征): {ml_rmses.mean():.2f} RMSE")


if __name__ == '__main__':
    main()
