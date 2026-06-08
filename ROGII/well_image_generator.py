"""
井剖面图像生成器 — 仅用训练/测试都有的列: MD, X, Y, Z, GR, TVT_input
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import glob, os

FIG_W, FIG_H, DPI = 512, 256, 100
GR_VMIN, GR_VMAX = 0, 200
Z_PAD = 0.3


def generate_well_image(hw_df, tw_df, out_path):
    md = hw_df['MD'].values.astype(np.float64)
    z  = hw_df['Z'].values.astype(np.float64)
    gr = hw_df['GR'].values.astype(np.float64)
    ps = hw_df['TVT_input'].notna().sum()

    tw_tvt = tw_df['TVT'].values.astype(np.float64)
    tw_gr  = tw_df['GR'].values.astype(np.float64)

    # 归一化
    z_lo, z_hi = z.min() - (z.max()-z.min())*Z_PAD, z.max() + (z.max()-z.min())*Z_PAD
    gr_norm = np.clip(np.nan_to_num(gr, 0), GR_VMIN, GR_VMAX)
    gr_norm = (gr_norm - GR_VMIN) / (GR_VMAX - GR_VMIN)
    tw_gr_norm = np.clip(np.nan_to_num(tw_gr, 0), GR_VMIN, GR_VMAX)
    tw_gr_norm = (tw_gr_norm - GR_VMIN) / (GR_VMAX - GR_VMIN)
    tw_d = (tw_tvt - tw_tvt.min()) / (tw_tvt.max() - tw_tvt.min() + 1e-8)

    fig = plt.figure(figsize=(FIG_W/DPI, FIG_H/DPI), dpi=DPI, facecolor='black')

    # 主图: MD vs Z, GR 着色
    ax = fig.add_axes([0.05, 0.08, 0.78, 0.87])
    ax.set_facecolor('black')
    if ps > 0:
        ax.scatter(md[:ps], z[:ps], c=gr_norm[:ps], cmap='viridis',
                   s=0.3, alpha=0.9, vmin=0, vmax=1, linewidths=0)
    if ps < len(md):
        ax.scatter(md[ps:], z[ps:], c=gr_norm[ps:], cmap='viridis',
                   s=0.15, alpha=0.5, vmin=0, vmax=1, linewidths=0)
    if 0 < ps < len(md):
        ax.axvline(x=md[ps-1], color='red', linewidth=0.8, alpha=0.6, linestyle='--')
    ax.set_xlim(md.min(), md.max())
    ax.set_ylim(z_lo, z_hi)
    ax.axis('off')

    # 侧边: Typewell GR
    ax2 = fig.add_axes([0.84, 0.08, 0.14, 0.87])
    ax2.set_facecolor('black')
    for i in range(len(tw_gr_norm)-1):
        ax2.plot([tw_gr_norm[i], tw_gr_norm[i+1]], [tw_d[i], tw_d[i+1]],
                 color=plt.cm.viridis(tw_gr_norm[i]), linewidth=0.5, alpha=0.8)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.axis('off')

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    plt.imsave(out_path, arr)
    return arr


if __name__ == '__main__':
    ROOT = '/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train'
    OUT  = '/Users/liucong/code/kaggle/ROGII/well_images'
    os.makedirs(OUT, exist_ok=True)

    hw_files = sorted(glob.glob(f'{ROOT}/*__horizontal_well.csv'))[:5]
    tw_files = [f.replace('__horizontal_well.csv', '__typewell.csv') for f in hw_files]

    print(f"生成 {len(hw_files)} 张示例图像...")
    for hw_f, tw_f in zip(hw_files, tw_files):
        wid = Path(hw_f).name.split('__')[0]
        hw_df = pd.read_csv(hw_f)
        tw_df = pd.read_csv(tw_f)
        out = f'{OUT}/{wid}.png'
        arr = generate_well_image(hw_df, tw_df, out)
        print(f"  {wid}.png  ({arr.shape[1]}×{arr.shape[0]})")
    print(f"\n保存至: {OUT}/")
