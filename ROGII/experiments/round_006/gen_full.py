#!/usr/bin/env python3
"""Generate cfg-img-full: ALL 723 non-val wells as training set.

R6 hypothesis: R2/R4-B "data scaling is dead" was wrong — it was
"diluting hand-curated train set is dead". With FULL data the model must
learn invariant features. Same geometry as cfg-img-medium (T=192, H=576,
comp=24, 3 channels) for direct comparison.

Val set stays identical to baseline (50 wells from gen_images.VAL_IDS)
so all R3-R5 numbers remain comparable.
"""
import os
import sys
import time
import json

sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/src")
from gen_images import (
    VAL_IDS, DATA_DIR, OUTPUT_DIR,
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

CONFIG_ID = "cfg-img-full"


def all_train_wells():
    all_wells = sorted(set(
        f.replace("__horizontal_well.csv", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith("__horizontal_well.csv")
    ))
    val_set = set(VAL_IDS)
    train = [w for w in all_wells if w not in val_set]
    assert len(set(train) & val_set) == 0, "train overlaps val!"
    print(f"Pool: {len(all_wells)} total, {len(val_set)} val, {len(train)} train")
    return train


def main():
    train_ids = all_train_wells()
    cfg = dict(GEOMETRY)
    cfg["config_id"] = CONFIG_ID

    out_dir = os.path.join(OUTPUT_DIR, CONFIG_ID)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\nConfig: {CONFIG_ID}  (train={len(train_ids)} val={len(VAL_IDS)})")
    print(f"{'='*70}")

    t0 = time.time()
    train_samples = process_wells(train_ids, cfg, label="train")
    t_train = time.time() - t0
    val_samples = process_wells(VAL_IDS, cfg, label="val")
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
        "scope": "full_corpus",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    print(
        f"  -> {CONFIG_ID} done | train={len(train_samples)} val={len(val_samples)} "
        f"| {train_mb+val_mb:.1f} MB | proc={t_proc:.1f}s save={total-t_proc:.1f}s total={total:.1f}s"
    )

    # Degeneracy check (R3 lesson)
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
