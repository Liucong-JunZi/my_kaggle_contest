#!/usr/bin/env python3
"""Generate cfg-img-randval: 50 train + 50 val randomly drawn from full corpus.

R6-B: refutation of "baseline 13.67 reflects model quality".

If raw RMSE on a RANDOM 50-val split jumps significantly above 15.84,
that proves the hand-curated VAL_IDS sit in a narrow geological window
that flatters the model. Same model + same train COUNT + same geometry —
ONLY the well selection changes.

Fixed seed for reproducibility.
"""
import os
import sys
import time
import json
import random

sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/src")
from gen_images import (
    DATA_DIR, OUTPUT_DIR,
    process_wells, save_h5,
)

GEOMETRY = {
    "channels": ["t_gr", "h_gr", "history"],
    "compression": 24,
    "sdf_scale": 40,
    "T_H": 96, "T_F": 96,        # T=192
    "H_H": 48, "H_F": 528,       # H=576
    "gr_filter_window": 50,
}

CONFIG_ID = "cfg-img-randval"
SEED = 20260612
N_TRAIN = 50
N_VAL = 50


def random_split():
    all_wells = sorted(set(
        f.replace("__horizontal_well.csv", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith("__horizontal_well.csv")
    ))
    rng = random.Random(SEED)
    rng.shuffle(all_wells)
    train = all_wells[:N_TRAIN]
    val = all_wells[N_TRAIN:N_TRAIN + N_VAL]
    assert len(set(train) & set(val)) == 0
    print(f"Pool: {len(all_wells)} total | seed={SEED} | train={N_TRAIN} val={N_VAL}")
    print(f"train sample: {sorted(train)[:3]} ...")
    print(f"val   sample: {sorted(val)[:3]} ...")
    return train, val


def main():
    train_ids, val_ids = random_split()
    cfg = dict(GEOMETRY)
    cfg["config_id"] = CONFIG_ID

    out_dir = os.path.join(OUTPUT_DIR, CONFIG_ID)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\nConfig: {CONFIG_ID}\n{'='*70}")

    t0 = time.time()
    train_samples = process_wells(train_ids, cfg, label="train")
    val_samples = process_wells(val_ids, cfg, label="val")
    t_proc = time.time() - t0

    train_path = os.path.join(out_dir, "train.h5")
    val_path = os.path.join(out_dir, "val.h5")
    train_mb = save_h5(train_samples, train_path, cfg)
    val_mb = save_h5(val_samples, val_path, cfg)
    total = time.time() - t0

    C = len(cfg["channels"])
    T = cfg["T_H"] + cfg["T_F"]
    H = cfg["H_H"] + cfg["H_F"]
    config_json = {
        "config_id": CONFIG_ID,
        "channels": cfg["channels"],
        "C": C, "T": T, "H": H,
        "T_H": cfg["T_H"], "T_F": cfg["T_F"],
        "H_H": cfg["H_H"], "H_F": cfg["H_F"],
        "compression": cfg["compression"],
        "sdf_scale": cfg["sdf_scale"],
        "gr_filter_window": cfg["gr_filter_window"],
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "train_shape": [len(train_samples), C, T, H],
        "val_shape": [len(val_samples), C, T, H],
        "disk_mb": round(train_mb + val_mb, 1),
        "generation_time_sec": round(total, 1),
        "type": "image",
        "scope": "random_split",
        "seed": SEED,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    print(
        f"  -> {CONFIG_ID} done | train={len(train_samples)} val={len(val_samples)} "
        f"| {train_mb+val_mb:.1f} MB | proc={t_proc:.1f}s total={total:.1f}s"
    )

    # Degeneracy check
    import h5py
    for s in ["train", "val"]:
        with h5py.File(os.path.join(out_dir, f"{s}.h5"), "r") as f:
            t = f["t_tvt"][:]
            degen = sum(1 for i in range(t.shape[0]) if t[i].std() < 0.5)
            print(f"  [verify] {s}: X={f['X'].shape} t_tvt={t.shape} degen={degen}/{t.shape[0]}")
            if degen > 0:
                pct = 100 * degen / t.shape[0]
                print(f"  WARN: {degen}/{t.shape[0]} ({pct:.1f}%) degenerate wells in {s}!")


if __name__ == "__main__":
    main()
