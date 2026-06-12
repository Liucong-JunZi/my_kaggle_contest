"""R8 Phase 3A: compute beam-search features for all 723 train wells.

Extracts the 14-config beam search from the LB-7.776 kernel
(docs/lb-references/lb-7-776-rogii-ridge-sp.py) using the numba JIT path.
Each config produces one TVT path per lateral row.  We keep the per-config
predictions (so the GBDT can pick), plus the mean / std / median across
the 14, plus a tag-based "consensus vs smooth-5" reference signal that the
kernel uses internally.

Output: results/round_008/beam_features.parquet
   columns: well, row_idx,
            beam_mean, beam_std, beam_med,
            beam_cons (cfg0), beam_sm5 (cfg3, r=5)
            beam_min_minus_last, beam_max_minus_last  (range as uncertainty)
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 14 beam configs (verbatim from kernel: bs, mc, es, smooth_radius)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),   # the "sm5" smoothed reference
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]
CONS_IDX = 0   # "consensus" reference (kernel's bpaths['cons'])
SM5_IDX  = 3   # "smoothed" reference (kernel's bpaths['sm5'])


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search ±2 delta — verbatim from kernel."""
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx  = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30);      bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64);   cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v): return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean()
            if r > 0 else s).to_numpy(np.float32)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def run_beam_all(hw, tw):
    """Return (n_lat, 14) array of TVT predictions, one column per config."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) == 0:
        return None, None

    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(np.float64)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(np.float64)

    gr_all = (hw["GR"].interpolate(limit_direction="both")
                       .fillna(tw_gr.mean()).values.astype(np.float64))
    hgr = gr_all[ev.index]

    paths = np.zeros((len(ev), len(BEAM_CONFIGS)), dtype=np.float32)
    for k, (bs, mc, es, r) in enumerate(BEAM_CONFIGS):
        paths[:, k] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    return paths, ev.index.values


def _warmup():
    _beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15.0, 100.0)


def main():
    print("=== R8 Phase 3A: beam-search features for full corpus ===\n")
    print("[warmup] compiling numba beam ...")
    t0 = time.time(); np.random.seed(0); _warmup()
    print(f"  done in {time.time() - t0:.1f}s\n")

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
            tw = pd.read_csv(f"{DATA_DIR}/{wid}__typewell.csv")
        except Exception:
            n_fail += 1; continue

        if hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 10:
            n_fail += 1; continue
        if len(tw) < 10 or tw["GR"].isna().all():
            n_fail += 1; continue

        try:
            paths, ev_idx = run_beam_all(hw, tw)
        except Exception as e:
            print(f"  ! {wid}: {e}"); n_fail += 1; continue
        if paths is None:
            n_fail += 1; continue

        beam_mean = paths.mean(axis=1)
        beam_std  = paths.std(axis=1)
        beam_med  = np.median(paths, axis=1)
        beam_min  = paths.min(axis=1)
        beam_max  = paths.max(axis=1)
        beam_cons = paths[:, CONS_IDX]
        beam_sm5  = paths[:, SM5_IDX]

        df_b = pd.DataFrame({
            "well":      wid,
            "row_idx":   ev_idx.astype(np.int32),
            "beam_mean": beam_mean.astype(np.float32),
            "beam_std":  beam_std.astype(np.float32),
            "beam_med":  beam_med.astype(np.float32),
            "beam_range": (beam_max - beam_min).astype(np.float32),
            "beam_cons": beam_cons,
            "beam_sm5":  beam_sm5,
        })
        records.append(df_b)
        n_ok += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(all_wells) - i - 1)
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s | ok={n_ok} fail={n_fail}")
    df = pd.concat(records, ignore_index=True)
    print(f"Rows: {len(df):,}  Wells: {df['well'].nunique()}")
    out_path = OUT_DIR / "beam_features.parquet"
    df.to_parquet(out_path)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
