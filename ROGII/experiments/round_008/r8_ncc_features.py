"""R8 Phase 6: multi-scale NCC features.

Implements `multi_scale_ncc` from the LB-7.776 kernel
(docs/lb-references/lb-7-776-rogii-ridge-sp.py:647). For each well, scans
a windowed cosine-similarity match between the known-segment GR template
and the lateral GR signal at three window scales (8/15/25 ft, stride 3 ft).

Each scale produces (tvt_pred, ncc_score) per lateral row. We also keep
the score-weighted ensemble (softmax over scores).

Output features per row (8):
  ncc8_offset, ncc15_offset, ncc25_offset, ncc_ens_offset
  ncc8_score, ncc15_score, ncc25_score, ncc_ens_disagreement

Saves: results/round_008/ncc_features.parquet
"""
import os, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    """Multi-scale NCC. Returns list of (tvt_pred, score) + ensemble."""
    out = []
    nh = len(hgr); nk = len(kgr)
    for hw in hws:
        win = 2 * hw + 1
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1] if len(ktvt) else 0.0, np.float32),
                        np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))

    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


def main():
    print("=== R8 Phase 6: multi-scale NCC features ===\n")
    all_wells = sorted({f.replace("__horizontal_well.csv", "")
                        for f in os.listdir(DATA_DIR)
                        if f.endswith("__horizontal_well.csv")})
    print(f"Wells: {len(all_wells)}\n")

    records = []
    n_ok = n_fail = 0
    t0 = time.time()
    for i, wid in enumerate(all_wells):
        try:
            hw = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
        except Exception:
            n_fail += 1; continue
        if hw["TVT_input"].notna().sum() < 30 or hw["TVT_input"].isna().sum() < 10:
            n_fail += 1; continue

        # Known segment: TVT_input non-NaN with valid GR
        kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(kn) < 50 or len(ev) == 0:
            n_fail += 1; continue

        kgr  = kn["GR"].values.astype(np.float32)
        ktvt = kn["TVT_input"].values.astype(np.float32)
        # Sort kgr/ktvt by ktvt for proper template
        order = np.argsort(ktvt)
        kgr, ktvt = kgr[order], ktvt[order]

        # Lateral GR (interpolated)
        hgr_full = (hw["GR"].interpolate(limit_direction="both")
                    .fillna(np.nanmean(kgr) if len(kgr) else 0).values.astype(np.float32))
        ev_idx = ev.index.values
        hgr_lat = hgr_full[ev_idx]

        try:
            scales_out, ens = multi_scale_ncc(kgr, ktvt, hgr_lat)
        except Exception as e:
            print(f"  ! {wid}: {e}"); n_fail += 1; continue

        (t8, s8), (t15, s15), (t25, s25) = scales_out
        df_n = pd.DataFrame({
            "well": wid, "row_idx": ev_idx.astype(np.int32),
            "ncc8":   t8,  "ncc8_score":  s8,
            "ncc15":  t15, "ncc15_score": s15,
            "ncc25":  t25, "ncc25_score": s25,
            "ncc_ens": ens,
        })
        records.append(df_n)
        n_ok += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s | ok={n_ok} fail={n_fail}")
    df = pd.concat(records, ignore_index=True)
    print(f"Rows: {len(df):,}, wells: {df['well'].nunique()}")
    out = OUT_DIR / "ncc_features.parquet"
    df.to_parquet(out)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
