"""
Generate SDF matching grid image datasets for ROGII TVT prediction.

Produces HDF5 datasets with (C, T, H) 2D matching grids from typewell and
horizontal well data, based on hengck23's SDF approach.
"""

import numpy as np
import pandas as pd
import h5py
import cv2
import json
import time
import os
from scipy.signal import savgol_filter

# ── Configs ────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "config_id": "cfg-img-hengck23",
        "channels": ["t_gr", "h_gr", "history"],
        "compression": 16,
        "sdf_scale": 40,
        "T_H": 128, "T_F": 128,
        "H_H": 64,  "H_F": 704,
        "gr_filter_window": 50,
    },
    {
        "config_id": "cfg-img-grdiff",
        "channels": ["t_gr", "h_gr", "gr_diff", "history"],
        "compression": 16,
        "sdf_scale": 40,
        "T_H": 128, "T_F": 128,
        "H_H": 64,  "H_F": 704,
        "gr_filter_window": 50,
    },
    {
        "config_id": "cfg-img-small",
        "channels": ["t_gr", "h_gr", "history"],
        "compression": 32,
        "sdf_scale": 40,
        "T_H": 64, "T_F": 64,
        "H_H": 32,  "H_F": 352,
        "gr_filter_window": 50,
    },
]

TRAIN_IDS = [
    "01869cd4", "03a935ae", "0dc5e64d", "0dd99dc5", "14a53cb3",
    "14ab73fb", "1590af81", "1a730ac2", "1fdbba44", "2ddad940",
    "2ee0235b", "353e5502", "3b21ea64", "3bbe1f5d", "4463446c",
    "4a8ecc0b", "5397ceb1", "55efee7f", "5a1a8fd8", "5bd25f59",
    "5cb483ac", "5fffa282", "647f2a41", "69c1bdad", "70e1788b",
    "7c607683", "7d57c75d", "7ff89f8f", "8bfa881d", "96ae5806",
    "995ff498", "9efc812e", "a247e7cf", "a3518960", "a4f989c2",
    "a6f967fb", "a76db406", "bffe3082", "c03fba65", "c1708b88",
    "c472c0b5", "d0a43760", "d7eb0be8", "e201fd6d", "e45a2a62",
    "eac10e86", "eba6605e", "f03f10fe", "f4d12d23", "fdfd57da",
]

VAL_IDS = [
    "09441b8d", "0e5e560d", "204cc64b", "25050f63", "27ebb9b9",
    "28473855", "2f19d536", "3407f5bf", "35b3ef6a", "398dce4b",
    "3e011332", "4121c517", "466fc788", "4a035ec2", "57f05c51",
    "58850d2c", "6d590e26", "71ccf778", "7bb17b96", "7cd4bb31",
    "8478df29", "85380836", "877ed19c", "87aa3730", "896d15b9",
    "89969c7a", "91b301ce", "9298ad5b", "9f0c7bae", "a0383629",
    "a85bb86f", "ab6fe95d", "aee6393a", "af7a59ce", "b38e3116",
    "bf39bc20", "c1d046f4", "ce55ba43", "cf50c9d1", "d1ecf309",
    "d9d6d94d", "dc5fbe29", "dc7f9757", "df73b8f3", "e46f4ef4",
    "ee0300f7", "ee4aecac", "f08774c3", "fb3848a1", "fd8f77fa",
]

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUTPUT_DIR = "/Users/liucong/code/kaggle/ROGII/data/cache"


# ── Helper functions ───────────────────────────────────────────────────────

def resample_typewell_by_step(t, step, target_step=0.5):
    """Resample typewell to target_step (0.5 ft)."""
    t_tvt = t["TVT"].values.astype(np.float64)
    t_gr = t["GR"].values.astype(np.float64)

    ratio = step / target_step

    if np.isclose(ratio, 1.0):
        return t_tvt, t_gr

    if ratio < 1.0:
        # step smaller than target_step → average pool
        group_size = int(round(1 / ratio))
        n = len(t)
        pad_len = (-n) % group_size
        if pad_len > 0:
            t_tvt = np.pad(t_tvt, (0, pad_len), mode="edge")
            t_gr = np.pad(t_gr, (0, pad_len), mode="edge")
        t_tvt = t_tvt.reshape(-1, group_size).mean(axis=1)
        t_gr = t_gr.reshape(-1, group_size).mean(axis=1)
    else:
        # step larger than target_step → linear interp
        up_factor = int(round(ratio))
        old_idx = np.arange(len(t))
        new_idx = np.linspace(0, len(t) - 1, (len(t) - 1) * up_factor + 1)
        t_tvt = np.interp(new_idx, old_idx, t_tvt)
        t_gr = np.interp(new_idx, old_idx, t_gr)

    return t_tvt, t_gr


def resample_horizontal_by_step(h, target_step, offset=0, gr_filter_window=50):
    """Pool horizontal well to target_step.

    Returns: h_tvt0, h_tvt1, h_gr0, h_gr1
      - 0: known segment (TVT_input not NaN)
      - 1: lateral segment (TVT_input NaN)
    """
    h_gr_filled = h["GR"].interpolate().bfill().ffill().values.astype(np.float64)
    h_gr_smooth = savgol_filter(h_gr_filled, gr_filter_window, 2)
    h["GR_smooth"] = h_gr_smooth

    h_ps = int(np.flatnonzero(h["TVT_input"].notna().values)[-1]) + offset

    cols = ["X", "Y", "Z", "TVT", "GR_smooth"]
    before = h[cols].iloc[: h_ps + 1].values
    after = h[cols].iloc[h_ps + 1 :].values

    # Pad before to be divisible by target_step
    pad_before = (-len(before)) % target_step
    if pad_before < target_step // 2:
        before = np.pad(before, ((pad_before, 0), (0, 0)), mode="edge")
    else:
        before = before[(target_step - pad_before) :]

    # Pad after to be divisible by target_step
    pad_after = (-len(after)) % target_step
    if pad_after < target_step // 2:
        after = np.pad(after, ((0, pad_after), (0, 0)), mode="edge")
    else:
        after = after[: -(target_step - pad_after)]

    before = before.reshape(len(before) // target_step, target_step, -1).mean(axis=1)
    after = after.reshape(len(after) // target_step, target_step, -1).mean(axis=1)

    h_tvt0 = before[:, 3]
    h_tvt1 = after[:, 3]
    h_gr0 = before[:, 4]
    h_gr1 = after[:, 4]
    return h_tvt0, h_tvt1, h_gr0, h_gr1


def get_crop_index_and_pad_1d(n, center, history, future):
    raw_i0 = center - history
    raw_i1 = center + future
    i0 = max(raw_i0, 0)
    i1 = min(raw_i1, n)
    pad_left = max(0, -raw_i0)
    pad_right = max(0, raw_i1 - n)
    return i0, i1, pad_left, pad_right


def load_one_sample(sample_id, cfg, offset=0):
    """Load and process one well into a (C, T, H) image + targets."""
    compression = cfg["compression"]
    sdf_scale = cfg["sdf_scale"]
    T_H, T_F = cfg["T_H"], cfg["T_F"]
    H_H, H_F = cfg["H_H"], cfg["H_F"]
    gr_filter_window = cfg.get("gr_filter_window", 50)
    channel_names = cfg["channels"]

    T = T_H + T_F
    H = H_H + H_F

    # ── Typewell ──
    typewell_csv = f"{DATA_DIR}/{sample_id}__typewell.csv"
    t = pd.read_csv(typewell_csv)
    t_step = (t["TVT"].diff().dropna().round(3)).mode().iloc[0]
    t_tvt, t_gr = resample_typewell_by_step(t, step=t_step, target_step=0.5)

    # ── Horizontal ──
    horizontal_csv = f"{DATA_DIR}/{sample_id}__horizontal_well.csv"
    h = pd.read_csv(horizontal_csv)
    h_tvt0, h_tvt1, h_gr0, h_gr1 = resample_horizontal_by_step(
        h, target_step=compression, offset=offset, gr_filter_window=gr_filter_window
    )

    # ── Crop typewell ──
    last_tvt = h_tvt0[-1]
    last_idx = np.abs(t_tvt - last_tvt).argmin()

    j0, j1, pad_left, pad_right = get_crop_index_and_pad_1d(
        len(t_tvt), last_idx + 1, history=T_H, future=T_F
    )
    t_seg_mask = np.ones(j1 - j0)
    t_seg_tvt = t_tvt[j0:j1]
    t_seg_gr = t_gr[j0:j1]

    # Pad
    t_seg_mask = np.pad(t_seg_mask, (pad_left, pad_right), mode="edge")
    t_seg_tvt = np.pad(t_seg_tvt, (pad_left, pad_right), mode="edge")
    t_seg_gr = np.pad(t_seg_gr, (pad_left, pad_right), mode="edge")

    assert len(t_seg_tvt) == T, f"Typewell crop: {len(t_seg_tvt)} != {T}"

    # ── Crop horizontal known segment ──
    j0, j1, pad_left, pad_right = get_crop_index_and_pad_1d(
        len(h_tvt0), len(h_tvt0), history=H_H, future=0
    )
    h_seg_mask0 = np.ones(j1 - j0)
    h_seg_tvt0 = h_tvt0[j0:j1]
    h_seg_gr0 = h_gr0[j0:j1]

    h_seg_mask0 = np.pad(h_seg_mask0, (pad_left, pad_right), mode="edge")
    h_seg_tvt0 = np.pad(h_seg_tvt0, (pad_left, pad_right), mode="edge")
    h_seg_gr0 = np.pad(h_seg_gr0, (pad_left, pad_right), mode="edge")

    # ── Crop horizontal lateral segment ──
    j0, j1, pad_left, pad_right = get_crop_index_and_pad_1d(
        len(h_tvt1), 0, history=0, future=H_F
    )
    h_seg_mask1 = np.ones(j1 - j0)
    h_seg_tvt1 = h_tvt1[j0:j1]
    h_seg_gr1 = h_gr1[j0:j1]

    h_seg_mask1 = np.pad(h_seg_mask1, (pad_left, pad_right), mode="edge")
    h_seg_tvt1 = np.pad(h_seg_tvt1, (pad_left, pad_right), mode="edge")
    h_seg_gr1 = np.pad(h_seg_gr1, (pad_left, pad_right), mode="edge")

    # ── Concatenate known + lateral ──
    h_seg_mask = np.concatenate([h_seg_mask0, h_seg_mask1])
    h_seg_tvt = np.concatenate([h_seg_tvt0, h_seg_tvt1])
    h_seg_gr = np.concatenate([h_seg_gr0, h_seg_gr1])

    assert len(h_seg_gr) == H, f"Horizontal crop: {len(h_seg_gr)} != {H}"

    # ── Compute SDF ──
    sdf = (h_seg_tvt[None, :] - t_seg_tvt[:, None]) / sdf_scale
    sdf = np.clip(sdf, -3.0, 3.0).astype(np.float32)

    # ── Compute matching path (label) for history ──
    diff = np.abs(t_seg_tvt[:, None] - h_seg_tvt[None, :])
    matched = diff.argmin(axis=0)
    matched_mask_raw = (diff.min(axis=0) < 1.0).astype(np.float32)
    matched_mask = matched_mask_raw * h_seg_mask

    label = np.zeros((T, H), dtype=np.float32)
    for i in range(H - 1):
        if matched_mask[i] == 0:
            continue
        if matched_mask[i + 1] == 0:
            continue
        cv2.line(label, (i, matched[i]), (i + 1, matched[i + 1]), 1.0, 6, cv2.LINE_AA)

    history = label.copy()
    history[:, H_H + 1 :] = 0.0  # zero out future segment

    # ── Compute t_mask, h_mask ──
    t_mask = t_seg_mask.astype(np.float32)
    h_mask = h_seg_mask.astype(np.float32)

    # ── Build channel grid ──
    t_gr_grid = np.tile(t_seg_gr[:, None], (1, H)).astype(np.float32)
    h_gr_grid = np.tile(h_seg_gr[None, :], (T, 1)).astype(np.float32)
    gr_diff_grid = (t_gr_grid - h_gr_grid).astype(np.float32)

    channel_map = {
        "t_gr": t_gr_grid,
        "h_gr": h_gr_grid,
        "gr_diff": gr_diff_grid,
        "history": history.astype(np.float32),
    }

    channels = np.stack([channel_map[ch] for ch in channel_names], axis=0).astype(
        np.float32
    )

    return {
        "image": channels,                 # (C, T, H)
        "sdf": sdf[np.newaxis, :, :],      # (1, T, H)
        "tvt": h_seg_tvt.astype(np.float32),  # (H,) horizontal TVT (target)
        "t_tvt": t_seg_tvt.astype(np.float32),  # (T,) typewell TVT grid for SDF→TVT lookup
        "mask": h_seg_mask.astype(np.float32),  # (H,)
        "well_id": sample_id,
    }


def process_wells(well_ids, cfg, label=""):
    """Process list of wells for a given config. Returns list of sample dicts."""
    results = []
    for i, wid in enumerate(well_ids):
        try:
            r = load_one_sample(wid, cfg)
            results.append(r)
        except Exception as e:
            print(f"  [{label}] FAILED well={wid}: {e}")
        if (i + 1) % 10 == 0 or i == len(well_ids) - 1:
            print(f"  [{label}] {i+1}/{len(well_ids)} wells processed")
    return results


def save_h5(samples, filepath, cfg):
    """Save list of sample dicts to HDF5 file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    N = len(samples)
    C = len(cfg["channels"])
    T = cfg["T_H"] + cfg["T_F"]
    H = cfg["H_H"] + cfg["H_F"]

    X = np.empty((N, C, T, H), dtype=np.float32)
    y_sdf = np.empty((N, 1, T, H), dtype=np.float32)
    y_tvt = np.empty((N, H), dtype=np.float32)
    t_tvt = np.empty((N, T), dtype=np.float32)
    mask = np.empty((N, H), dtype=np.float32)

    # Fixed-length byte strings for well_ids
    max_id_len = max(len(s["well_id"]) for s in samples)
    well_ids = np.zeros(N, dtype=f"S{max_id_len}")

    for i, s in enumerate(samples):
        X[i] = s["image"]
        y_sdf[i] = s["sdf"]
        y_tvt[i] = s["tvt"]
        t_tvt[i] = s["t_tvt"]
        mask[i] = s["mask"]
        well_ids[i] = s["well_id"].encode("ascii")

    with h5py.File(filepath, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y_sdf", data=y_sdf, compression="gzip", compression_opts=4)
        f.create_dataset("y_tvt", data=y_tvt, compression="gzip", compression_opts=4)
        f.create_dataset("t_tvt", data=t_tvt, compression="gzip", compression_opts=4)
        f.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
        f.create_dataset("well_ids", data=well_ids)

    disk_mb = os.path.getsize(filepath) / (1024 * 1024)
    return disk_mb


def generate_config(cfg):
    """Generate train.h5 and val.h5 for a single config."""
    config_id = cfg["config_id"]
    out_dir = os.path.join(OUTPUT_DIR, config_id)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # ── Config JSON ──
    C = len(cfg["channels"])
    T = cfg["T_H"] + cfg["T_F"]
    H = cfg["H_H"] + cfg["H_F"]

    # Process train wells
    t_train_start = time.time()
    train_samples = process_wells(TRAIN_IDS, cfg, label="train")
    t_train = time.time() - t_train_start

    # Process val wells
    t_val_start = time.time()
    val_samples = process_wells(VAL_IDS, cfg, label="val")
    t_val = time.time() - t_val_start

    # Save HDF5
    train_path = os.path.join(out_dir, "train.h5")
    val_path = os.path.join(out_dir, "val.h5")
    train_mb = save_h5(train_samples, train_path, cfg)
    val_mb = save_h5(val_samples, val_path, cfg)

    total_time = time.time() - t0

    # ── Config JSON ──
    config_json = {
        "config_id": config_id,
        "channels": cfg["channels"],
        "C": C,
        "T": T,
        "H": H,
        "T_H": cfg["T_H"],
        "T_F": cfg["T_F"],
        "H_H": cfg["H_H"],
        "H_F": cfg["H_F"],
        "compression": cfg["compression"],
        "sdf_scale": cfg["sdf_scale"],
        "gr_filter_window": cfg.get("gr_filter_window", 50),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "train_shape": [len(train_samples), C, T, H],
        "val_shape": [len(val_samples), C, T, H],
        "disk_mb": round(train_mb + val_mb, 1),
        "generation_time_sec": round(total_time, 1),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    return {
        "config_id": config_id,
        "C": C,
        "T": T,
        "H": H,
        "compression": cfg["compression"],
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "train_mb": round(train_mb, 1),
        "val_mb": round(val_mb, 1),
        "total_mb": round(train_mb + val_mb, 1),
        "time_sec": round(total_time, 1),
        "status": "OK",
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Image Searcher — Generating SDF matching grid datasets")
    print("=" * 70)

    results = []
    for cfg in CONFIGS:
        print(f"\n{'─' * 70}")
        print(f"Config: {cfg['config_id']}")
        print(f"  channels={cfg['channels']}, C={len(cfg['channels'])}")
        print(
            f"  T_H={cfg['T_H']} T_F={cfg['T_F']} H_H={cfg['H_H']} H_F={cfg['H_F']}"
        )
        print(f"  compression={cfg['compression']} sdf_scale={cfg['sdf_scale']}")
        print(f"{'─' * 70}")

        result = generate_config(cfg)
        results.append(result)

        print(
            f"  → Done: train({result['n_train']},{result['C']},{result['T']},{result['H']}) "
            f"val({result['n_val']},{result['C']},{result['T']},{result['H']}) "
            f"| {result['total_mb']} MB | {result['time_sec']}s"
        )

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['config_id']:24s} | C={r['C']} T={r['T']} H={r['H']} comp={r['compression']} "
            f"| train({r['n_train']},{r['C']},{r['T']},{r['H']}) "
            f"val({r['n_val']},{r['C']},{r['T']},{r['H']}) "
            f"| {r['total_mb']} MB | {r['time_sec']}s"
        )
    print("=" * 70)
    print("Done.")
