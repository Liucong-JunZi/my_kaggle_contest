#!/usr/bin/env python3
"""Generate cfg-img-medium-100 and cfg-img-medium-200-fair datasets.

Fair scaling experiment: keep VAL_IDS identical to baseline, only vary
train count (50 -> 100 -> 200) using extra wells from the train/ pool.
"""
import os
import sys
import time
import json

sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/src")
import gen_images  # noqa: E402
from gen_images import (
    TRAIN_IDS, VAL_IDS, DATA_DIR, OUTPUT_DIR,
    process_wells, save_h5,
)

GEOMETRY = {
    "channels": ["t_gr", "h_gr", "history"],
    "compression": 24,
    "sdf_scale": 40,
    "T_H": 96, "T_F": 96,
    "H_H": 48, "H_F": 528,
    "gr_filter_window": 50,
}


def pick_extra_wells():
    all_wells = sorted(set(
        f.replace("__horizontal_well.csv", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith("__horizontal_well.csv")
    ))
    existing = set(TRAIN_IDS) | set(VAL_IDS)
    extra_pool = [w for w in all_wells if w not in existing]
    print(f"Pool: {len(all_wells)} total, {len(existing)} reserved, {len(extra_pool)} extra")
    extra_50 = extra_pool[:50]
    extra_150 = extra_pool[:150]
    return extra_50, extra_150


def run_one(config_id, train_ids):
    cfg = dict(GEOMETRY)
    cfg["config_id"] = config_id

    out_dir = os.path.join(OUTPUT_DIR, config_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\nConfig: {config_id}  (train={len(train_ids)} val={len(VAL_IDS)})")
    print(f"{'='*70}")

    t0 = time.time()
    train_samples = process_wells(train_ids, cfg, label="train")
    val_samples = process_wells(VAL_IDS, cfg, label="val")

    train_path = os.path.join(out_dir, "train.h5")
    val_path = os.path.join(out_dir, "val.h5")
    train_mb = save_h5(train_samples, train_path, cfg)
    val_mb = save_h5(val_samples, val_path, cfg)
    total = time.time() - t0

    C = len(cfg["channels"])
    T = cfg["T_H"] + cfg["T_F"]
    H = cfg["H_H"] + cfg["H_F"]
    config_json = {
        "config_id": config_id,
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
        "fair_scaling": True,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    print(f"  -> {config_id} done | train={len(train_samples)} val={len(val_samples)} "
          f"| {train_mb+val_mb:.1f} MB | {total:.1f}s")
    return config_json


def verify_degeneracy(config_id):
    import h5py
    out_dir = os.path.join(OUTPUT_DIR, config_id)
    print(f"\n[verify] {config_id}")
    for s in ["train", "val"]:
        with h5py.File(os.path.join(out_dir, f"{s}.h5"), "r") as f:
            t = f["t_tvt"][:]
            degen = sum(1 for i in range(t.shape[0]) if t[i].std() < 0.5)
            print(f"  {s}: X={f['X'].shape} t_tvt={t.shape} degen={degen}/{t.shape[0]}")
            if degen > 0:
                print(f"  WARN: {degen} degenerate wells in {config_id}/{s}!")


if __name__ == "__main__":
    extra_50, extra_150 = pick_extra_wells()
    train_100 = TRAIN_IDS + extra_50
    train_200 = TRAIN_IDS + extra_150

    assert len(set(train_100) & set(VAL_IDS)) == 0, "train_100 overlaps VAL!"
    assert len(set(train_200) & set(VAL_IDS)) == 0, "train_200 overlaps VAL!"
    assert len(set(train_100)) == 100, f"train_100 has dupes: {len(set(train_100))}"
    assert len(set(train_200)) == 200, f"train_200 has dupes: {len(set(train_200))}"

    print(f"train_100 IDs (first 5): {train_100[:5]} ... (last 5): {train_100[-5:]}")
    print(f"train_200 IDs (first 5): {train_200[:5]} ... (last 5): {train_200[-5:]}")

    run_one("cfg-img-medium-100", train_100)
    run_one("cfg-img-medium-200-fair", train_200)

    verify_degeneracy("cfg-img-medium-100")
    verify_degeneracy("cfg-img-medium-200-fair")
    print("\nAll done.")
