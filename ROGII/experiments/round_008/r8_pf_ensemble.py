"""R8 Phase 14A: multi-seed PF likelihood-weighted ensemble.

Replicates the LB-7.776 kernel's `run_pf_lik_ensemble_scales` (line 227 of
lb-7-776-rogii-ridge-sp.py): run the position-tracking PF (our `_pf_ancc`,
which is structurally identical to the kernel's `run_particle_filter`)
multiple times with different seeds, then for each scale s in {3, 5, 8, 12}:

    weights[k] = softmax(total_log_lik[k] / s)
    pf_scale_s[h] = sum_k weights[k] * preds[k, h]

The scale temperature controls how peaked the seed-mixture is. Lower scale
trusts only the highest-likelihood seeds; higher scale averages broadly.

Kernel uses 128 seeds; we run 16 (~42 min on 773 wells, ~8× cheaper).
Output: results/round_008/pf_ensemble.parquet
   columns: well, row_idx, pf_ens_s3, pf_ens_s5, pf_ens_s8, pf_ens_s12, pf_ens_mean
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")

# Reuse JIT kernels + helpers from PF v9 (sp45 + GR preinterp + log-lik return)
sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/experiments/round_008")
from r8_pf_features import (
    _pf_ancc, _grid, _gr_sig, _warmup,
    ANCC_N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR,
    PF_RESAMP, PF_GR_SIG_DEF,
)

N_SEEDS = 16
SCALES  = [3.0, 5.0, 8.0, 12.0]


def run_one_seed(hw, tw_tvt, tw_gr, seed, gs, ls, ir, gg, gmin, gst):
    """Run a single PF and return (preds_per_row, total_log_lik_scalar)."""
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return np.array([]), 0.0
    gr_full = hw['GR'].interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    gr_v = gr_full.loc[ev.index].values.astype(np.float64)

    np.random.seed(seed)
    pts, std, loglk = _pf_ancc(
        ev['MD'].values.astype(np.float64),
        ev['Z'].values.astype(np.float64),
        gr_v, gg, gmin, gst,
        gs, ls, ir, ANCC_N,
        ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP,
    )
    # Total log-evidence at the final row = cumulative scalar
    total_ll = float(loglk[-1]) if len(loglk) > 0 else 0.0
    return pts.astype(np.float64), total_ll


def main():
    print(f"=== R8 Phase 14A: PF {N_SEEDS}-seed log-lik ensemble ===\n")
    print("[warmup] compiling numba kernels...")
    t0 = time.time()
    np.random.seed(0); _warmup()
    print(f"  done in {time.time()-t0:.1f}s\n")

    all_wells = sorted({f.replace("__horizontal_well.csv","")
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

        tw_s = tw.sort_values('TVT')
        tw_tvt = tw_s['TVT'].values.astype(np.float64)
        tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(np.float64)
        if len(tw_tvt) < 10 or np.isnan(tw_gr).all():
            n_fail += 1; continue
        if hw['TVT_input'].notna().sum() < 10 or hw['TVT_input'].isna().sum() < 10:
            n_fail += 1; continue

        # Pre-compute well-level constants (same for every seed)
        kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
        gs = _gr_sig(hw, tw_tvt, tw_gr)
        ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
        tail = kn.tail(30)
        dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values)
        dm = np.diff(tail['MD'].values); m = dm > 0
        ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
        gg, gmin, gst = _grid(tw_tvt, tw_gr)

        try:
            preds_all = []   # [N_SEEDS, n_ev]
            lls       = []   # [N_SEEDS]
            for s in range(N_SEEDS):
                p, ll = run_one_seed(hw, tw_tvt, tw_gr, s, gs, ls, ir, gg, gmin, gst)
                preds_all.append(p); lls.append(ll)
            preds_arr = np.stack(preds_all, axis=0)         # [S, n_ev]
            lls_arr = np.array(lls, dtype=np.float64)        # [S]
        except Exception as e:
            print(f"  ! {wid}: {e}")
            n_fail += 1; continue

        # Numerical stability: subtract max before exp
        lls_n = lls_arr - lls_arr.max()
        ev_idx = ev.index.values

        out = {"well": wid, "row_idx": ev_idx.astype(np.int32)}
        for s in SCALES:
            w = np.exp(lls_n / s); w = w / w.sum()
            ens = (w[:, None] * preds_arr).sum(0)            # [n_ev]
            out[f"pf_ens_s{int(s)}"] = ens.astype(np.float32)
        out["pf_ens_mean"] = preds_arr.mean(0).astype(np.float32)

        records.append(pd.DataFrame(out))
        n_ok += 1

        if (i+1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(all_wells) - i - 1)
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s | ok={n_ok} fail={n_fail}")
    df = pd.concat(records, ignore_index=True)
    print(f"Rows: {len(df):,}  Wells: {df['well'].nunique()}")
    for s in SCALES:
        col = f"pf_ens_s{int(s)}"
        print(f"  {col:18s} med={df[col].median():.2f}  std={df[col].std():.2f}")
    print(f"  pf_ens_mean        med={df['pf_ens_mean'].median():.2f}  std={df['pf_ens_mean'].std():.2f}")

    out_path = OUT_DIR / "pf_ensemble.parquet"
    df.to_parquet(out_path)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
