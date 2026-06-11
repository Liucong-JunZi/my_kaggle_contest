"""
Feature extraction for ROGII — two configs:
  feat-lgb-base:   MD, X, Y, Z, dz, GR_smooth, GR_rolling_std(20)
  feat-lgb-domain: base + tortuosity + cos_azimuth + sin_azimuth + TVT_history + dTVT_history

Only lateral-segment rows (TVT_input NaN). Saves train.npz / val.npz per config.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR  = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR   = "/Users/liucong/code/kaggle/ROGII/data/cache"

TRAIN_IDS = [
    '01869cd4','03a935ae','0dc5e64d','0dd99dc5','14a53cb3','14ab73fb','1590af81',
    '1a730ac2','1fdbba44','2ddad940','2ee0235b','353e5502','3b21ea64','3bbe1f5d',
    '4463446c','4a8ecc0b','5397ceb1','55efee7f','5a1a8fd8','5bd25f59','5cb483ac',
    '5fffa282','647f2a41','69c1bdad','70e1788b','7c607683','7d57c75d','7ff89f8f',
    '8bfa881d','96ae5806','995ff498','9efc812e','a247e7cf','a3518960','a4f989c2',
    'a6f967fb','a76db406','bffe3082','c03fba65','c1708b88','c472c0b5','d0a43760',
    'd7eb0be8','e201fd6d','e45a2a62','eac10e86','eba6605e','f03f10fe','f4d12d23',
    'fdfd57da',
]
VAL_IDS = [
    '09441b8d','0e5e560d','204cc64b','25050f63','27ebb9b9','28473855','2f19d536',
    '3407f5bf','35b3ef6a','398dce4b','3e011332','4121c517','466fc788','4a035ec2',
    '57f05c51','58850d2c','6d590e26','71ccf778','7bb17b96','7cd4bb31','8478df29',
    '85380836','877ed19c','87aa3730','896d15b9','89969c7a','91b301ce','9298ad5b',
    '9f0c7bae','a0383629','a85bb86f','ab6fe95d','aee6393a','af7a59ce','b38e3116',
    'bf39bc20','c1d046f4','ce55ba43','cf50c9d1','d1ecf309','d9d6d94d','dc5fbe29',
    'dc7f9757','df73b8f3','e46f4ef4','ee0300f7','ee4aecac','f08774c3','fb3848a1',
    'fd8f77fa',
]

SAVGOL_WINDOW = 51   # must be odd
SAVGOL_ORDER  = 2
ROLLING_WIN   = 20
TOR_WINDOW    = 50   # window for local tortuosity
TVT_SLOPE_WIN = 20   # window for dTVT trend estimation


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_local_tortuosity(x, y, z, window=TOR_WINDOW):
    """Q-3D tortuosity (Jing et al. 2022) computed in a sliding window.
    T = (sum of segment lengths) / (straight-line distance from first to last) - 1
    """
    n = len(x)
    tort = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 3:
            tort[i] = 0.0
            continue
        dx = np.diff(x[lo:hi])
        dy = np.diff(y[lo:hi])
        dz_ = np.diff(z[lo:hi])
        seg_len = np.sum(np.sqrt(dx**2 + dy**2 + dz_**2))
        straight = np.sqrt((x[hi-1]-x[lo])**2 + (y[hi-1]-y[lo])**2 + (z[hi-1]-z[lo])**2)
        if straight < 1e-6:
            tort[i] = 0.0
        else:
            tort[i] = seg_len / straight - 1.0
    return tort


def compute_tvt_features(hw_df):
    """Compute TVT_history and dTVT_history.

    TVT_history: last known TVT_input forward-filled into lateral.
    dTVT_history: for lateral rows, the mean d(TVT)/d(MD) slope from the last N
    vertical points. For vertical rows, the local gradient.
    """
    md   = hw_df['MD'].values
    tvt  = hw_df['TVT'].values          # always present (target)
    tvt_inp = hw_df['TVT_input'].values  # NaN in lateral

    n = len(md)
    mask_lateral = np.isnan(tvt_inp)

    # TVT_history: forward fill the last known TVT_input
    # Use TVT where TVT_input is known, else the forward-filled value
    known_tvt = tvt_inp.copy()
    tvt_history = pd.Series(known_tvt).ffill().values  # last known carried forward

    # dTVT_history
    dTVT_history = np.zeros(n)

    if np.any(mask_lateral):
        transition = np.argmax(mask_lateral)  # first lateral row
        if transition > TVT_SLOPE_WIN:
            # slope of TVT vs MD in vertical segment, last N points
            win_slice = slice(max(0, transition - TVT_SLOPE_WIN), transition)
            slope, _ = np.polyfit(md[win_slice], tvt[win_slice], 1)
        elif transition > 1:
            slope, _ = np.polyfit(md[:transition], tvt[:transition], 1)
        else:
            slope = 0.0

        # Assign slope to all lateral rows; local gradient to vertical
        dTVT_history[mask_lateral] = slope
        # Vertical rows: local gradient
        dTVT_history[~mask_lateral] = np.gradient(tvt[~mask_lateral], md[~mask_lateral])
    else:
        dTVT_history = np.gradient(tvt, md)

    return tvt_history, dTVT_history


def extract_well_features(well_id, data_dir):
    """Extract all raw features for a single well. Returns dict of 1D arrays,
    plus lateral_mask and target (TVT) for lateral rows."""
    hw_path = os.path.join(data_dir, f"{well_id}__horizontal_well.csv")
    df = pd.read_csv(hw_path)

    md = df['MD'].values.astype(np.float64)
    x  = df['X'].values.astype(np.float64)
    y  = df['Y'].values.astype(np.float64)
    z  = df['Z'].values.astype(np.float64)
    tvt_inp = df['TVT_input'].values
    tvt = df['TVT'].values.astype(np.float64)
    gr_raw = df['GR'].values.astype(np.float64)

    lateral_mask = np.isnan(tvt_inp)
    n = len(md)

    feats = {}

    # ── Base features (always computed) ──
    feats['MD'] = md
    feats['X']  = x
    feats['Y']  = y
    feats['Z']  = z
    feats['dz'] = np.gradient(z)  # dZ / d(index), approx dZ/dMD since MD step ≈ 1

    # ── GR features ──
    # Interpolate NaN GR
    gr_series = pd.Series(gr_raw).interpolate(method='linear', limit_direction='both')
    gr_series = gr_series.bfill().ffill()
    gr_clean = gr_series.values

    # Savgol smoothing (need at least window+1 points)
    if len(gr_clean) > SAVGOL_WINDOW:
        gr_smooth = savgol_filter(gr_clean, SAVGOL_WINDOW, SAVGOL_ORDER)
    else:
        gr_smooth = gr_clean.copy()

    # Rolling std
    gr_std = pd.Series(gr_clean).rolling(ROLLING_WIN, center=True, min_periods=1).std().fillna(0).values

    feats['GR_smooth'] = gr_smooth
    feats['GR_std'] = gr_std

    # ── Direction features ──
    dx = np.gradient(x)
    dy = np.gradient(y)
    azimuth = np.arctan2(dy, dx)
    feats['cos_azimuth'] = np.cos(azimuth)
    feats['sin_azimuth'] = np.sin(azimuth)

    # ── Tortuosity ──
    feats['tortuosity'] = compute_local_tortuosity(x, y, z, window=TOR_WINDOW)

    # ── TVT history features ──
    tvt_hist, dtvt_hist = compute_tvt_features(df)
    feats['TVT_history'] = tvt_hist
    feats['dTVT_history'] = dtvt_hist

    return feats, lateral_mask, tvt


# ── Config definitions ──────────────────────────────────────────────────────

CONFIGS = {
    'feat-lgb-base': {
        'features': ['MD', 'X', 'Y', 'Z', 'dz', 'GR_smooth', 'GR_std'],
    },
    'feat-lgb-domain': {
        'features': ['MD', 'X', 'Y', 'Z', 'dz', 'GR_smooth', 'GR_std',
                     'tortuosity', 'cos_azimuth', 'sin_azimuth',
                     'TVT_history', 'dTVT_history'],
    },
}


# ── Main ────────────────────────────────────────────────────────────────────

def build_config(config_id, config, well_ids, split_name):
    """Collect features for all wells, concatenate, save."""
    all_X = []
    all_y = []
    all_masks = []
    n_skipped = 0

    for wid in well_ids:
        try:
            feats, lateral_mask, tvt = extract_well_features(wid, DATA_DIR)
        except Exception as e:
            print(f"  ⚠ skipped {wid}: {e}")
            n_skipped += 1
            continue

        # Only lateral rows
        lat_idx = np.where(lateral_mask)[0]
        if len(lat_idx) == 0:
            n_skipped += 1
            continue

        # Stack selected features
        X_well = np.column_stack([feats[f] for f in config['features']])
        X_lat = X_well[lat_idx]
        y_lat = tvt[lat_idx]

        all_X.append(X_lat)
        all_y.append(y_lat)
        all_masks.append(np.ones(len(lat_idx), dtype=bool))  # all valid

    if not all_X:
        print(f"  ERROR: no wells processed for {split_name}")
        return None, None, None

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    mask = np.concatenate(all_masks, axis=0).astype(bool)

    print(f"  {split_name}: X={X.shape}, y={y.shape}, wells={len(all_X)}/{len(well_ids)}")
    return X, y, mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0_total = time.time()

    results = []

    for config_id, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Building config: {config_id}")
        print(f"Features ({len(config['features'])}): {config['features']}")
        t0 = time.time()

        out_path = os.path.join(OUT_DIR, config_id)
        os.makedirs(out_path, exist_ok=True)

        # Build train
        X_train, y_train, mask_train = build_config(config_id, config, TRAIN_IDS, "train")
        # Build val
        X_val, y_val, mask_val = build_config(config_id, config, VAL_IDS, "val")

        if X_train is None or X_val is None:
            print(f"  FAILED for {config_id}")
            continue

        # Save
        train_path = os.path.join(out_path, "train.npz")
        val_path   = os.path.join(out_path, "val.npz")

        np.savez_compressed(train_path,
                           X=X_train, y=y_train, mask=mask_train)
        np.savez_compressed(val_path,
                           X=X_val, y=y_val, mask=mask_val)

        # Feature names
        with open(os.path.join(out_path, "feature_names.txt"), "w") as f:
            for name in config['features']:
                f.write(name + "\n")

        # Config metadata
        elapsed = time.time() - t0
        meta = {
            "config_id": config_id,
            "feature_dim": len(config['features']),
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "feature_names": config['features'],
            "type": "tabular",
            "generation_time_sec": round(elapsed, 1),
        }
        with open(os.path.join(out_path, "config.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # File sizes
        train_mb = os.path.getsize(train_path) / 1e6
        val_mb   = os.path.getsize(val_path) / 1e6

        results.append({
            "config_id": config_id,
            "dim": len(config['features']),
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "train_mb": train_mb,
            "val_mb": val_mb,
            "time_s": round(elapsed, 1),
        })

        print(f"  Saved: {train_path} ({train_mb:.1f} MB)")
        print(f"  Saved: {val_path} ({val_mb:.1f} MB)")
        print(f"  Time: {elapsed:.1f}s")

    # ── Summary ──
    total_elapsed = time.time() - t0_total
    print(f"\n{'='*60}")
    print("feature-searcher 完成:")
    for r in results:
        print(f"  {r['config_id']:<18} | dim={r['dim']:<3}  "
              f"train{r['train_shape']} val{r['val_shape']} | "
              f"{r['train_mb']+r['val_mb']:.0f} MB | {r['time_s']:.0f}s")
    print(f"  Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
