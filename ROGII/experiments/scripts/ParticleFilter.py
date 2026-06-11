import numpy as np

class config :
    def __init__(self,
                 num_particles=500,
                 num_iterations=None,
                 transition_std=0.002,
                 observation_std=0.01,
                 initial_bias_std=0.05,
                 resample_method='systematic',
                 ess_threshold=0.5,
                 ridge_alpha=1.0,
                 random_seed=42):
        """
        粒子滤波配置

        Parameters
        ----------
        num_particles : int
            粒子数量。越多越精确但越慢，建议 200~2000
        num_iterations : int or None
            水平段最大预测步数。None = 自动取井的完整水平段长度
        transition_std : float
            状态转移噪声标准差。控制 bias 每步的随机漂移幅度。
            bias[i] = bias[i-1] + N(0, transition_std²)
            越小 → 越平滑，越大 → 越灵活。典型值 0.001~0.005
        observation_std : float
            观测噪声标准差。控制对 ML 预测的信任程度。
            似然: p(obs | bias) = N(ML_pred, observation_std²)
            越小 → 越信任 ML，越大 → 越依赖先验平滑
        initial_bias_std : float
            初始 bias 分布的标准差。粒子从已知段末尾 bias 均值出发，
            以该 std 散布。典型值 0.01~0.1
        resample_method : str
            重采样方法: 'systematic' | 'multinomial' | 'stratified'
        ess_threshold : float
            有效样本量阈值（占 num_particles 的比例）。
            低于此值时触发重采样。0.5 = ESS < N/2 时重采样
        ridge_alpha : float
            观测模型（岭回归）的正则化强度
        random_seed : int
            随机种子，保证可复现
        """
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.transition_std = transition_std
        self.observation_std = observation_std
        self.initial_bias_std = initial_bias_std
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold
        self.ridge_alpha = ridge_alpha
        self.random_seed = random_seed

class dataprocessor :
    def __init__(self, data):
        """
        data: 训练数据目录路径 或 单口井的 CSV 文件路径
        """
        self.data = data
        self.wells = []          # 所有井的数据列表
        self._loaded = False

    # ================================================================
    # 数据加载
    # ================================================================

    def load_all(self, max_wells=None):
        """加载训练目录下所有井的 CSV 文件"""
        import glob
        files = sorted(glob.glob(f'{self.data}/*__horizontal_well.csv'))
        if max_wells:
            files = files[:max_wells]
        for f in files:
            w = self._load_single_well(f)
            if w is not None:
                self.wells.append(w)
        self._loaded = True
        print(f"加载完成: {len(self.wells)} 口有效井")
        return self

    def _load_single_well(self, filepath):
        """加载单口井的 CSV，计算所有派生量"""
        import pandas as pd
        import numpy as np
        df = pd.read_csv(filepath)

        # 找 PS（造斜点）：TVT_input 最后一个非空位置
        ps = df['TVT_input'].notna().sum()
        n = len(df)

        # 基本列
        x = df['X'].values.astype(np.float64)
        y = df['Y'].values.astype(np.float64)
        z = df['Z'].values.astype(np.float64)
        md = df['MD'].values.astype(np.float64)
        tvt = df['TVT'].values.astype(np.float64)

        # 梯度
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        dmd = np.gradient(md)
        dtvt = np.gradient(tvt)

        # bias = dtvt + dz（核心量）
        bias = dtvt + dz

        return {
            'name': filepath.split('/')[-1].replace('__horizontal_well.csv', ''),
            'ps': ps, 'n': n,
            'x': x, 'y': y, 'z': z, 'md': md, 'tvt': tvt,
            'dx': dx, 'dy': dy, 'dz': dz, 'dmd': dmd,
            'dtvt': dtvt, 'bias': bias,
        }

    # ================================================================
    # 特征构建（观测模型用）
    # ================================================================

    def build_features(self, well, slice_obj=None):
        """
        从轨迹信息构建特征矩阵（不包含 bias 滞后特征）
        用于粒子滤波的观测模型: p(obs | state)

        返回: X (N, D) 特征矩阵, feature_names (list)
        """
        import numpy as np

        dx = well['dx'] if slice_obj is None else well['dx'][slice_obj]
        dy = well['dy'] if slice_obj is None else well['dy'][slice_obj]
        dz = well['dz'] if slice_obj is None else well['dz'][slice_obj]
        dmd = well['dmd'] if slice_obj is None else well['dmd'][slice_obj]

        eps = 1e-6
        features = []
        names = []

        def _add(arr, name, clip_val=100):
            arr = np.asarray(arr, dtype=np.float64)
            arr = np.clip(arr, -clip_val, clip_val)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            features.append(arr)
            names.append(name)

        # 瞬时值
        for name, val, cl in [('dx', dx, 100), ('dy', dy, 100), ('dz', dz, 100), ('dmd', dmd, 10)]:
            _add(val, name, cl)

        # 绝对值
        _add(np.abs(dx), '|dx|')
        _add(np.abs(dy), '|dy|')
        _add(np.abs(dz), '|dz|')

        # 符号
        _add(np.sign(dx), 'sign(dx)', 1)
        _add(np.sign(dy), 'sign(dy)', 1)
        _add(np.sign(dz), 'sign(dz)', 1)

        # 非线性变换
        _add(np.arctan(dx), 'atan(dx)', 2)
        _add(np.arctan(dy), 'atan(dy)', 2)
        _add(np.arctan(dz), 'atan(dz)', 2)

        # 交互项
        _add(dx * dz, 'dx*dz', 1000)
        _add(dy * dz, 'dy*dz', 1000)
        _add(dx**2, 'dx^2', 10000)
        _add(dz**2, 'dz^2', 10000)

        # 加速度
        _add(np.gradient(dz), 'd2z', 100)

        return np.column_stack(features), names

    # ================================================================
    # 训练数据获取
    # ================================================================

    def get_known_segment(self, well, n_context=300):
        """
        获取已知段（PS 之前）的数据

        返回: X (轨迹特征), y (bias 真实值)
        """
        ps = well['ps']
        start = max(0, ps - n_context)
        sl = slice(start, ps)
        X, _ = self.build_features(well, sl)
        y = well['bias'][sl]
        return X, y

    def get_lateral_segment(self, well):
        """
        获取水平段（PS 之后）的数据

        返回: X (轨迹特征), y (bias 真实值)
        """
        ps = well['ps']
        sl = slice(ps, well['n'])
        X, _ = self.build_features(well, sl)
        y = well['bias'][sl]
        return X, y

    def assemble_train_data(self, well_indices, n_context=300):
        """拼接多口井的已知段数据，返回 (X_train, y_train)"""
        X_list, y_list = [], []
        for idx in well_indices:
            w = self.wells[idx]
            X, y = self.get_known_segment(w, n_context)
            X_list.append(X)
            y_list.append(y)
        return np.vstack(X_list), np.concatenate(y_list)
    

    # ================================================================
    # 工具方法
    # ================================================================

    def filter_valid(self, min_ps=10, min_lateral=100):
        """过滤：PS 在合理范围 & 水平段足够长"""
        self.wells = [
            w for w in self.wells
            if w['ps'] >= min_ps and (w['n'] - w['ps']) >= min_lateral
        ]
        print(f"过滤后: {len(self.wells)} 口有效井")
        return self

    def train_val_split(self, val_ratio=0.2, seed=42):
        """按井划分训练/验证集"""
        n = len(self.wells)
        indices = np.random.RandomState(seed).permutation(n)
        n_val = int(val_ratio * n)
        return indices[n_val:], indices[:n_val]  # train_idx, val_idx

    @property
    def n_wells(self):
        return len(self.wells)


class ParticleFilter :
    def __init__(self, config):
        """
        config: config 实例，包含所有超参数
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

        # 观测模型（岭回归），由 fit_observation_model() 训练
        self.obs_weights = None     # shape (D+1,) 含截距
        self.obs_feature_mean = None
        self.obs_feature_std = None

        # 运行时状态
        self.particles = None       # shape (N_particles,)
        self.weights = None         # shape (N_particles,)  归一化后的权重

    # ================================================================
    # 观测模型（岭回归）
    # ================================================================

    def fit_observation_model(self, X_train, y_train):
        """
        用已知段数据训练岭回归：X(轨迹特征) → bias

        X_train: (N_samples, D_features)  轨迹特征矩阵
        y_train: (N_samples,)             bias 真实值
        """
        # 标准化
        self.obs_feature_mean = X_train.mean(axis=0)
        self.obs_feature_std = X_train.std(axis=0)
        self.obs_feature_std[self.obs_feature_std == 0] = 1.0
        X_s = (X_train - self.obs_feature_mean) / self.obs_feature_std

        # 加截距项
        X_bias = np.column_stack([np.ones(len(X_s)), X_s])

        # 岭回归闭式解: w = (XᵀX + αI)⁻¹ Xᵀy
        XtX = X_bias.T @ X_bias
        XtX[np.arange(XtX.shape[0]), np.arange(XtX.shape[0])] += self.config.ridge_alpha
        Xty = X_bias.T @ y_train
        self.obs_weights = np.linalg.solve(XtX, Xty)
        print(f"观测模型训练完成: {len(y_train)} 样本, {X_train.shape[1]} 特征")
        return self

    def _predict_observation(self, X_step):
        """
        对单步特征做 ML 预测

        X_step: (1, D) 或 (D,) — 单步轨迹特征
        返回: float — 预测的 bias 值
        """
        X = np.atleast_2d(X_step)
        X_s = (X - self.obs_feature_mean) / self.obs_feature_std
        X_bias = np.column_stack([np.ones(len(X_s)), X_s])
        return float((X_bias @ self.obs_weights)[0])

    # ================================================================
    # 粒子初始化
    # ================================================================

    def init_particles(self, initial_bias):
        """
        用已知段末尾的 bias 均值初始化粒子

        initial_bias: float — 已知段最后几步的平均 bias
        """
        self.particles = initial_bias + self.rng.randn(self.config.num_particles) * self.config.initial_bias_std
        self.weights = np.ones(self.config.num_particles) / self.config.num_particles

    # ================================================================
    # 单步预测-更新
    # ================================================================

    def step(self, X_step):
        """
        执行一步粒子滤波：预测 → 更新权重 → 归一化 → (可选) 重采样

        X_step: (D,) — 当前步的轨迹特征

        返回: (estimated_bias, resampled_flag)
        """
        # ---- 1. 预测（状态转移） ----
        # bias[i] = bias[i-1] + N(0, transition_std²)
        self.particles += self.rng.randn(self.config.num_particles) * self.config.transition_std

        # ---- 2. 更新（观测似然） ----
        obs_pred = self._predict_observation(X_step)

        # 似然: w *= exp( -0.5 * ((particle - obs_pred) / obs_std)² )
        log_likelihood = -0.5 * ((self.particles - obs_pred) / self.config.observation_std) ** 2
        log_likelihood -= log_likelihood.max()  # 数值稳定
        self.weights *= np.exp(log_likelihood)

        # ---- 3. 归一化 ----
        w_sum = self.weights.sum()
        if w_sum == 0:
            self.weights = np.ones(self.config.num_particles) / self.config.num_particles
        else:
            self.weights /= w_sum

        # ---- 4. 估计 ----
        estimated_bias = float(np.average(self.particles, weights=self.weights))

        # ---- 5. 重采样 ----
        resampled = False
        if self._compute_ess() < self.config.ess_threshold * self.config.num_particles:
            self._resample()
            resampled = True

        return estimated_bias, resampled

    # ================================================================
    # 重采样
    # ================================================================

    def _compute_ess(self):
        """有效样本量"""
        return 1.0 / (self.weights ** 2).sum()

    def _resample(self):
        """根据权重重采样粒子"""
        method = self.config.resample_method
        if method == 'systematic':
            indices = self._systematic_resample()
        elif method == 'multinomial':
            indices = self._multinomial_resample()
        elif method == 'stratified':
            indices = self._stratified_resample()
        else:
            raise ValueError(f"未知重采样方法: {method}")

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.config.num_particles) / self.config.num_particles

    def _systematic_resample(self):
        """系统重采样：方差最小，推荐默认使用"""
        N = self.config.num_particles
        cumsum = np.cumsum(self.weights)
        u0 = self.rng.uniform(0, 1.0 / N)
        indices = np.zeros(N, dtype=int)
        j = 0
        for i in range(N):
            u = u0 + i / N
            while j < N and cumsum[j] < u:
                j += 1
            indices[i] = min(j, N - 1)
        return indices

    def _multinomial_resample(self):
        """多项式重采样"""
        N = self.config.num_particles
        cumsum = np.cumsum(self.weights)
        u = self.rng.uniform(0, 1, N)
        return np.searchsorted(cumsum, u)

    def _stratified_resample(self):
        """分层重采样"""
        N = self.config.num_particles
        cumsum = np.cumsum(self.weights)
        indices = np.zeros(N, dtype=int)
        for i in range(N):
            u = self.rng.uniform(i / N, (i + 1) / N)
            indices[i] = np.searchsorted(cumsum, u)
        return indices

    # ================================================================
    # 完整运行
    # ================================================================

    def run(self, X_lateral, initial_bias, max_steps=None):
        """
        对整段水平段运行粒子滤波

        X_lateral: (N_steps, D) — 水平段每步的轨迹特征
        initial_bias: float — 已知段末尾的平均 bias
        max_steps: int or None — 最大步数限制（加速扫描用）

        返回: bias_pred (N_steps,) — 每步预测的 bias
        """
        n_steps = X_lateral.shape[0]
        if self.config.num_iterations is not None:
            n_steps = min(n_steps, self.config.num_iterations)
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        # 初始化
        self.init_particles(initial_bias)
        bias_pred = np.zeros(n_steps)
        resample_count = 0

        for i in range(n_steps):
            bias_pred[i], resampled = self.step(X_lateral[i])
            if resampled:
                resample_count += 1

        print(f"PF 完成: {n_steps} 步, 重采样 {resample_count} 次 "
              f"({resample_count / max(n_steps, 1) * 100:.1f}%)")
        return bias_pred

    # ================================================================
    # TVT 重建
    # ================================================================

    @staticmethod
    def reconstruct_tvt(start_tvt, dz, bias_pred):
        """
        从预测的 bias 序列重建 TVT: cumsum(-dz + bias)

        start_tvt: float — PS 位置的 TVT 值
        dz: (N,) — 水平段每步的 dz
        bias_pred: (N,) — 粒子滤波预测的 bias 序列

        返回: tvt_pred (N,) — 预测的 TVT 序列
        """
        n = len(dz)
        tvt_pred = np.zeros(n)
        tvt_pred[0] = start_tvt + (-dz[0] + bias_pred[0])
        for i in range(1, n):
            tvt_pred[i] = tvt_pred[i - 1] + (-dz[i] + bias_pred[i])
        return tvt_pred


# ================================================================
# 主程序
# ================================================================

TRAIN_DIR = '/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train'


def load_and_train(dp, train_idx, ridge_alpha=1.0):
    """加载数据并训练观测模型，返回训练好的 PF + 特征名"""
    X_train, y_train = dp.assemble_train_data(train_idx, n_context=300)
    cfg = config(ridge_alpha=ridge_alpha, random_seed=42)
    pf = ParticleFilter(cfg)
    pf.fit_observation_model(X_train, y_train)
    _, feat_names = dp.build_features(dp.wells[0], slice(0, 10))
    return pf, feat_names


def evaluate_one_well(dp, pf, w, transition_std, observation_std, initial_bias_std,
                      num_particles=200, max_steps=1000):
    """对单口井用指定参数跑 PF，返回 RMSE"""
    ps, n = w['ps'], w['n']
    n_lat = min(n - ps, max_steps)
    start_tvt = w['tvt'][ps - 1]
    true_tvt = w['tvt'][ps:ps + n_lat]
    l_dz = w['dz'][ps:ps + n_lat]
    initial_bias = w['bias'][max(0, ps - 50):ps].mean()

    X_lateral, _ = dp.build_features(w, slice(ps, ps + n_lat))

    # 临时覆盖参数
    pf.config.transition_std = transition_std
    pf.config.observation_std = observation_std
    pf.config.initial_bias_std = initial_bias_std
    pf.config.num_particles = num_particles

    bias_pf = pf.run(X_lateral, initial_bias)
    pred_pf = pf.reconstruct_tvt(start_tvt, l_dz, bias_pf)
    return np.sqrt(np.mean((pred_pf - true_tvt) ** 2))


def sweep():
    """网格搜索粒子滤波超参数"""
    # ---- 加载数据 ----
    print("=" * 70)
    print("粒子滤波超参数扫描")
    print("=" * 70)

    print("\n加载数据...")
    dp = dataprocessor(TRAIN_DIR)
    dp.load_all(max_wells=None)
    dp.filter_valid(min_ps=10, min_lateral=100)

    train_idx, val_idx = dp.train_val_split(val_ratio=0.3, seed=42)
    print(f"训练井: {len(train_idx)}, 验证井(扫描用前50口): {min(50, len(val_idx))}")

    # 训练观测模型
    pf, feat_names = load_and_train(dp, train_idx)

    # ---- 参数网格 ----
    transition_stds = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    observation_stds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    initial_bias_stds = [0.01, 0.05, 0.1]
    n_particles_list = [100]
    max_steps = 1000  # 每口井最多跑 1000 步

    # 只用前 N 口验证井加速
    sweep_wells = val_idx[:30]

    # 先算基线
    print("\n计算基线...")
    const_rmses = []
    ml_rmses = []
    oracle_rmses = []

    for idx in val_idx:  # 全量验证集算基线
        w = dp.wells[idx]
        ps, n = w['ps'], w['n']
        start_tvt = w['tvt'][ps - 1]
        true_tvt = w['tvt'][ps:]
        l_dz = w['dz'][ps:]
        X_lateral, _ = dp.build_features(w, slice(ps, n))

        # 常数
        cb = w['bias'][max(0, ps - 200):ps].mean()
        pred_c = pf.reconstruct_tvt(start_tvt, l_dz, np.full(len(l_dz), cb))
        const_rmses.append(np.sqrt(np.mean((pred_c - true_tvt) ** 2)))

        # 纯 ML
        bias_ml = np.array([pf._predict_observation(X_lateral[i]) for i in range(len(X_lateral))])
        pred_ml = pf.reconstruct_tvt(start_tvt, l_dz, bias_ml)
        ml_rmses.append(np.sqrt(np.mean((pred_ml - true_tvt) ** 2)))

        # Oracle
        best = float('inf')
        for b in np.arange(-0.1, 0.1, 0.001):
            pred_o = pf.reconstruct_tvt(start_tvt, l_dz, np.full(len(l_dz), b))
            e = np.sqrt(np.mean((pred_o - true_tvt) ** 2))
            if e < best:
                best = e
        oracle_rmses.append(best)

    print(f"  常数: {np.mean(const_rmses):.2f}  纯ML: {np.mean(ml_rmses):.2f}  Oracle: {np.mean(oracle_rmses):.2f}")

    # ---- 扫描 ----
    print(f"\n扫描 {len(transition_stds)}×{len(observation_stds)}×{len(initial_bias_stds)}×{len(n_particles_list)}"
          f" = {len(transition_stds)*len(observation_stds)*len(initial_bias_stds)*len(n_particles_list)} 组合...")

    best_result = None
    best_rmse = float('inf')
    all_results = []

    total = len(transition_stds) * len(observation_stds) * len(initial_bias_stds) * len(n_particles_list)
    count = 0

    for t_std in transition_stds:
        for o_std in observation_stds:
            for i_std in initial_bias_stds:
                for n_p in n_particles_list:
                    count += 1
                    well_rmses = []
                    for idx in sweep_wells:
                        w = dp.wells[idx]
                        rmse = evaluate_one_well(dp, pf, w, t_std, o_std, i_std, n_p, max_steps)
                        well_rmses.append(rmse)

                    mean_rmse = np.mean(well_rmses)
                    all_results.append((mean_rmse, t_std, o_std, i_std, n_p))

                    if count % 50 == 0 or count == total:
                        print(f"  [{count}/{total}] "
                              f"t={t_std:.4f} o={o_std:.3f} i={i_std:.3f} N={n_p} → {mean_rmse:.2f}")

                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best_result = (mean_rmse, t_std, o_std, i_std, n_p)

    # ---- 结果 ----
    all_results.sort()

    print(f"\n{'='*70}")
    print(f"Top 20 参数组合（验证集 {len(sweep_wells)} 口井）")
    print(f"{'='*70}")
    print(f"{'Rank':>4} | {'RMSE':>8} | {'trans_std':>10} | {'obs_std':>10} | {'init_std':>10} | {'N_particles':>11}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}")

    for rank, (rmse, t_std, o_std, i_std, n_p) in enumerate(all_results[:20]):
        marker = " ← BEST" if rank == 0 else ""
        print(f"{rank+1:>4} | {rmse:>8.2f} | {t_std:>10.4f} | {o_std:>10.3f} | {i_std:>10.3f} | {n_p:>11}{marker}")

    # 最佳 vs 基线
    best_rmse, best_t, best_o, best_i, best_n = best_result
    ml_mean = np.mean(ml_rmses)
    print(f"\n最佳 PF (30井×1000步): {best_rmse:.2f}  纯 ML: {ml_mean:.2f}  改进: {(1 - best_rmse/ml_mean)*100:+.1f}%")

    # ---- 用最佳参数跑全量验证集 ----
    print(f"\n{'='*70}")
    print(f"全量验证集评估（最佳参数）")
    print(f"{'='*70}")
    print(f"参数: t_std={best_t:.4f}  o_std={best_o:.3f}  i_std={best_i:.3f}  N={best_n}")

    pf.config.transition_std = best_t
    pf.config.observation_std = best_o
    pf.config.initial_bias_std = best_i
    pf.config.num_particles = best_n

    pf_rmses = []
    const_rmses = []
    ml_rmses = []
    oracle_rmses = []

    for idx in val_idx:
        w = dp.wells[idx]
        ps, n = w['ps'], w['n']
        start_tvt = w['tvt'][ps - 1]
        true_tvt = w['tvt'][ps:]
        l_dz = w['dz'][ps:]
        initial_bias = w['bias'][max(0, ps - 50):ps].mean()
        X_lateral, _ = dp.build_features(w, slice(ps, n))

        # PF
        bias_pf = pf.run(X_lateral, initial_bias)
        pred_pf = pf.reconstruct_tvt(start_tvt, l_dz, bias_pf)
        pf_rmses.append(np.sqrt(np.mean((pred_pf - true_tvt) ** 2)))

        # ML
        bias_ml = np.array([pf._predict_observation(X_lateral[i]) for i in range(len(X_lateral))])
        pred_ml = pf.reconstruct_tvt(start_tvt, l_dz, bias_ml)
        ml_rmses.append(np.sqrt(np.mean((pred_ml - true_tvt) ** 2)))

        # 常数
        cb = w['bias'][max(0, ps - 200):ps].mean()
        pred_c = pf.reconstruct_tvt(start_tvt, l_dz, np.full(len(l_dz), cb))
        const_rmses.append(np.sqrt(np.mean((pred_c - true_tvt) ** 2)))

        # Oracle
        best = float('inf')
        for b in np.arange(-0.1, 0.1, 0.001):
            pred_o = pf.reconstruct_tvt(start_tvt, l_dz, np.full(len(l_dz), b))
            e = np.sqrt(np.mean((pred_o - true_tvt) ** 2))
            if e < best:
                best = e
        oracle_rmses.append(best)

    print(f"\n{'方法':>30} | {'均值':>8} | {'中位数':>8} | {'最小':>8} | {'最大':>8}")
    print(f"{'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, arr in [('Oracle (搜索标量)', oracle_rmses),
                       ('常数 bias', const_rmses),
                       ('纯 ML (岭回归)', ml_rmses),
                       ('★ 粒子滤波 (最佳参数)', pf_rmses)]:
        a = np.array(arr)
        print(f"{name:>30} | {a.mean():>8.2f} | {np.median(a):>8.2f} | {a.min():>8.2f} | {a.max():>8.2f}")

    pf_mean = np.mean(pf_rmses)
    ml_mean = np.mean(ml_rmses)
    const_mean = np.mean(const_rmses)
    oracle_mean = np.mean(oracle_rmses)
    print(f"\n改进幅度:")
    print(f"  PF vs 常数:  {(1 - pf_mean/const_mean)*100:+.1f}%")
    print(f"  PF vs 纯 ML: {(1 - pf_mean/ml_mean)*100:+.1f}%")
    print(f"  PF vs Oracle: 差距 {pf_mean - oracle_mean:.2f} ft")


if __name__ == '__main__':
    sweep()

