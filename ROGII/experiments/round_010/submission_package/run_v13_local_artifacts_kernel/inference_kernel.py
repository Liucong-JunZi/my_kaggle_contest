"""ROGII run_v13 inference: local artifact dataset + hidden-safe live inference."""
# %%
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_kaggle_input(slug="rogii-wellbore-geology-prediction"):
    for path in (f"/kaggle/input/competitions/{slug}", f"/kaggle/input/{slug}"):
        if os.path.isdir(path):
            return path
    return None


def _find_dataset_input(*slugs, local=None):
    for slug in slugs:
        for path in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{slug}"):
            if os.path.isdir(path):
                return path
    if local and os.path.isdir(local):
        return local
    return None


INPUT_DIR = _find_kaggle_input() or "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction"
ARTIFACT_DIR = _find_dataset_input(
    "rogii-run-v13-local-artifacts",
    "smartorz/rogii-run-v13-local-artifacts",
    local="/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/run_v13_local_artifact_dataset",
)
OUT_PATH = "/kaggle/working/submission.csv" if _find_kaggle_input() else "/Users/liucong/code/kaggle/ROGII/results/round_010/submission_inference_v13_local_artifacts.csv"

print(f"INPUT_DIR = {INPUT_DIR}")
print(f"ARTIFACT_DIR = {ARTIFACT_DIR}")
assert os.path.isdir(f"{INPUT_DIR}/train"), f"train missing under {INPUT_DIR}"
assert os.path.isdir(f"{INPUT_DIR}/test"), f"test missing under {INPUT_DIR}"
assert ARTIFACT_DIR and os.path.isdir(ARTIFACT_DIR), "artifact dataset missing"

sys.path.insert(0, str(Path(ARTIFACT_DIR) / "src"))
from native_predict import predict_component
from pf_features import PF_N_PARTICLES, PF_N_SEEDS, run_pf_lik_ensemble_scales
from pipeline import predict_pipeline
import ravaghi_features


# %%
def load_json(rel):
    return json.loads((Path(ARTIFACT_DIR) / rel).read_text())


def test_wells():
    test_dir = Path(INPUT_DIR) / "test"
    return sorted(p.stem.replace("__horizontal_well", "") for p in test_dir.glob("*__horizontal_well.csv"))


def build_pf_component(scale_key="pf_scale_8"):
    rows = []
    t0 = time.time()
    for i, wid in enumerate(test_wells(), 1):
        hw_path = Path(INPUT_DIR) / "test" / f"{wid}__horizontal_well.csv"
        tw_path = Path(INPUT_DIR) / "test" / f"{wid}__typewell.csv"
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)
        ev_mask = hw["TVT_input"].isna().to_numpy()
        if not ev_mask.any():
            continue
        last_known_tvt = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])
        pf_by_scale = run_pf_lik_ensemble_scales(hw, tw, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS)
        if scale_key not in pf_by_scale:
            raise KeyError(f"PF scale {scale_key} missing; keys={sorted(pf_by_scale)}")
        pred_abs = pf_by_scale[scale_key]
        idxs = np.flatnonzero(ev_mask)
        last_known_md = float(hw.loc[hw["TVT_input"].notna(), "MD"].iloc[-1])
        md_since = (hw["MD"].to_numpy(np.float64)[idxs] - last_known_md).astype(np.float64)
        rows.append(pd.DataFrame({
            "well": wid,
            "row_idx": idxs.astype(np.int32),
            "last_known_tvt": np.float32(last_known_tvt),
            "md_since": md_since,
            "c20_r9_pf128_full": (pred_abs[idxs] - last_known_tvt).astype(np.float32),
        }))
        if i % 25 == 0:
            print(f"  PF {i}/{len(test_wells())} wells | rows={sum(len(x) for x in rows):,} | {time.time()-t0:.0f}s", flush=True)
    if not rows:
        raise ValueError("PF component produced no rows")
    out = pd.concat(rows, ignore_index=True)
    print(f"  PF component: rows={len(out):,} wells={out['well'].nunique()} built in {time.time()-t0:.0f}s")
    return out


def build_ravaghi_components(base):
    t0 = time.time()
    rav = ravaghi_features.build_test_features(INPUT_DIR)
    if len(rav) == 0:
        raise ValueError("ravaghi feature builder returned no rows")
    rav["well"] = rav["well"].astype(str)
    rav["row_idx"] = rav["id"].str.rsplit("_", n=1).str[1].astype("int32")
    aligned = base[["well", "row_idx"]].merge(rav, on=["well", "row_idx"], how="left", validate="one_to_one")
    missing = int(aligned["id"].isna().sum())
    print(f"  ravaghi features: rows={len(aligned):,} cols={len(rav.columns)} missing={missing} built in {time.time()-t0:.0f}s")
    if missing:
        raise ValueError(f"missing ravaghi feature rows: {missing}")

    out = {}
    model_root = Path(ARTIFACT_DIR) / "models" / "ravaghi"
    for cid_dir in sorted(model_root.iterdir()):
        if not cid_dir.is_dir():
            continue
        pred = predict_component(cid_dir, aligned)
        out[cid_dir.name] = pred
        print(f"    {cid_dir.name:18s} mean={pred.mean():.3f} std={pred.std():.3f} range=[{pred.min():.1f},{pred.max():.1f}]")
    return out


# %%
def main():
    t0 = time.time()
    print("=== ROGII run_v13: local artifact dataset + hidden-safe inference ===")
    manifest = load_json("artifact_manifest.json")
    weights_cfg = load_json("ensemble_weights.json")
    weights = weights_cfg["weights"]
    pp_params = weights_cfg.get("pp_params")
    sg_params = weights_cfg.get("sg_params")
    pf_cid = weights_cfg.get("pf_offset_component", "c20_r9_pf128_full")
    print(f"artifact_version = {manifest.get('artifact_version')}")
    print(f"weights_label = {weights_cfg.get('label')}")
    print(f"pp_params = {pp_params}  sg_params = {sg_params}")
    print(f"PF source: n_particles={PF_N_PARTICLES} n_seeds={PF_N_SEEDS} scale=pf_scale_8")

    pf = build_pf_component("pf_scale_8")
    # sg_smooth requires rows grouped by well in order; PF is built well-by-well so already grouped.
    components = {"c20_r9_pf128_full": pf["c20_r9_pf128_full"].to_numpy(np.float32)}
    components.update(build_ravaghi_components(pf))

    well_codes = pd.factorize(pf["well"].values, sort=False)[0].astype(np.int32)
    pred_tvt = predict_pipeline(
        components, weights,
        last_known_tvt=pf["last_known_tvt"].to_numpy(np.float64),
        md_since=pf["md_since"].to_numpy(np.float64),
        well_codes=well_codes,
        pf_offset=components[pf_cid],
        pp_params=pp_params,
        sg_params=sg_params,
        return_absolute=True,
    )
    print(f"  pipeline: weight_sum={sum(weights.values()):.6f} pp={'on' if pp_params else 'off'} sg={'on' if sg_params else 'off'}")
    sub = pd.DataFrame({
        "id": pf["well"] + "_" + pf["row_idx"].astype(str),
        "tvt": pred_tvt.astype(np.float32),
    })

    sample = pd.read_csv(Path(INPUT_DIR) / "sample_submission.csv")
    sub = sample[["id"]].merge(sub, on="id", how="left")
    missing = int(sub["tvt"].isna().sum())
    if missing:
        first = sub.loc[sub["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"missing predictions for {missing} sample rows, first={first}")
    sub.to_csv(OUT_PATH, index=False)
    print(f"  → {OUT_PATH} ({len(sub)} rows)")
    print(sub.head().to_string(index=False))
    print(f"  pred stats: min={sub['tvt'].min():.0f}, max={sub['tvt'].max():.0f}, mean={sub['tvt'].mean():.0f}, median={sub['tvt'].median():.0f}")
    print(f"Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
