#!/usr/bin/env python3
"""Train 5 PyTorch MLP candidates on local M5 MPS — write OOFs directly via shared.oof_writer.

Same model defs / hyperparams as t4_train_kernel/train_t4_source.py, but:
  - reads joined_features.parquet locally
  - uses MPS device on M5 (CUDA unavailable on macOS)
  - writes results/candidates/c6X.parquet directly through write_oof
  - no Kaggle upload, no import step

Estimated wall time on M5: ~45-60 min for all 5 candidates × 5 folds × 30 epochs.
"""
import sys, gc, time, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.oof_writer import write_oof
from shared.metrics import perwell_rmse, flat_rmse


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Match feature_set_v14() — hard-coded so we don't drag in shared.data_loader
# (which auto-builds joined_features from round_008 sources, slow startup).
FEAT_COLS = [
    "md_offset", "z_rel", "x_rel", "y_rel", "cumsum_neg_dz",
    "sin_dmd_dz", "cos_dmd_dz", "sin_dx_dy", "cos_dx_dy",
    "gr_smooth", "gr_diff_from_last",
    "last_known_tvt", "last_known_z", "last_known_gr",
    "n_known_rows", "n_lateral_rows", "row_position_norm",
    "gr_mean_5", "gr_std_5", "gr_mean_21", "gr_std_21",
    "gr_mean_51", "gr_std_51", "gr_mean_101", "gr_std_101",
    "pf_ancc_std", "pf_z_std",
    "pf_ancc_offset", "pf_z_offset", "pf_disagreement", "pf_mean_offset",
    "beam_mean_offset", "beam_med_offset", "beam_cons_offset", "beam_sm5_offset",
    "beam_vs_pf",
    "pf_ens_s3_offset", "pf_ens_s5_offset", "pf_ens_s8_offset",
    "pf_ens_s12_offset", "pf_ens_mean_offset",
    "pf_ens_vs_ancc", "pf_ens_scale_disag",
]
assert len(FEAT_COLS) == 43


# ----- Model architectures -----

class MLP(nn.Module):
    """4-layer MLP with LayerNorm + dropout — used by c60-c63.

    LayerNorm (not BatchNorm): more stable on MPS, doesn't depend on batch stats.
    """
    def __init__(self, n_in, hidden=(256, 128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, h, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(h, h); self.ln1 = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, h); self.ln2 = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z = self.ln1(torch.relu(self.fc1(x)))
        z = self.drop(z)
        z = self.ln2(self.fc2(z))
        return torch.relu(x + z)


class ResMLP(nn.Module):
    """3 residual blocks + linear out — used by c64."""
    def __init__(self, n_in, hidden=256, n_blocks=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(n_in, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        z = torch.relu(self.proj(x))
        for b in self.blocks:
            z = b(z)
        return self.out(z).squeeze(-1)


def make_model(spec, n_in):
    if spec["arch"] == "mlp":    return MLP(n_in)
    if spec["arch"] == "resmlp": return ResMLP(n_in)
    raise ValueError(spec["arch"])


def make_loss(spec):
    if spec["loss"] == "mse":   return nn.MSELoss()
    if spec["loss"] == "huber": return nn.HuberLoss(delta=10.0)
    raise ValueError(spec["loss"])


# Candidate specs — same as t4 kernel
CANDIDATE_SPECS = [
    dict(cid="c60_mlp_s42",       arch="mlp",    seed=42,   epochs=30, lr=1e-3, loss="mse",   batch=8192),
    dict(cid="c61_mlp_s7",        arch="mlp",    seed=7,    epochs=30, lr=1e-3, loss="mse",   batch=8192),
    dict(cid="c62_mlp_s2024",     arch="mlp",    seed=2024, epochs=30, lr=1e-3, loss="mse",   batch=8192),
    dict(cid="c63_mlp_huber_s42", arch="mlp",    seed=42,   epochs=30, lr=1e-3, loss="huber", batch=8192),
    dict(cid="c64_resmlp_s42",    arch="resmlp", seed=42,   epochs=40, lr=1e-3, loss="mse",   batch=8192),
]


def train_one_fold(spec, X_tr, y_tr, X_va, y_va, device):
    torch.manual_seed(spec["seed"]); np.random.seed(spec["seed"])

    # Per-fold standardization fit on train only — then clip to ±10 std to
    # kill extreme outliers (some rows on this dataset hit ~80 σ on md_offset).
    mu = X_tr.mean(0).astype(np.float32)
    sd = X_tr.std(0).astype(np.float32) + 1e-6
    X_tr_s = np.clip((X_tr - mu) / sd, -10.0, 10.0)
    X_va_s = np.clip((X_va - mu) / sd, -10.0, 10.0)

    ds_tr = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va))
    # MPS doesn't like num_workers>0 on macOS; CPU subprocess fork can be flaky too — use 0
    ld_tr = DataLoader(ds_tr, batch_size=spec["batch"],   shuffle=True,  num_workers=0)
    ld_va = DataLoader(ds_va, batch_size=spec["batch"]*2, shuffle=False, num_workers=0)

    model = make_model(spec, n_in=X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec["epochs"])
    loss_fn = make_loss(spec)

    best_va_pw = math.inf
    best_pred = None
    best_epoch = -1
    for ep in range(spec["epochs"]):
        model.train()
        tr_loss_sum = 0.0; tr_n = 0
        for xb, yb in ld_tr:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                # Defensive: skip the bad batch (rare numeric blowup on MPS).
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss_sum += loss_val * xb.size(0); tr_n += xb.size(0)
        sched.step()

        model.eval()
        va_preds = []
        with torch.no_grad():
            for xb, _yb in ld_va:
                xb = xb.to(device, non_blocking=True)
                va_preds.append(model(xb).cpu().numpy())
        va_pred = np.concatenate(va_preds)
        # Sanitize stray NaN/Inf cells from MPS arithmetic — replace with mean of finite preds
        finite = np.isfinite(va_pred)
        if not finite.all():
            mean_finite = va_pred[finite].mean() if finite.any() else 0.0
            va_pred = np.where(finite, va_pred, mean_finite).astype(np.float32)

        # Use perwell-RMSE-on-val proxy (just flat-RMSE here for cheap selection)
        va_rmse = float(np.sqrt(((va_pred - y_va) ** 2).mean()))
        if math.isfinite(va_rmse) and va_rmse < best_va_pw:
            best_va_pw = va_rmse
            best_pred = va_pred
            best_epoch = ep
        if ep == 0 or ep == spec["epochs"] - 1 or ep % 5 == 0:
            tr_loss_print = (tr_loss_sum / tr_n) if tr_n else float("nan")
            print(f"      ep{ep:>2d}  tr_loss={tr_loss_print:.4f}  va_rmse={va_rmse:.4f}  "
                  f"best={best_va_pw:.4f}@ep{best_epoch}", flush=True)

    if best_pred is None:
        # Fall back to zeros — flagged as fold failure by perwell metrics
        print(f"      WARN: no finite val rmse across {spec['epochs']} epochs; returning zeros")
        best_pred = np.zeros(len(y_va), dtype=np.float32)
    return best_pred.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Comma-separated cids to run (default: all). Eg --only c60_mlp_s42,c64_resmlp_s42")
    ap.add_argument("--device", default=None, help="Force device: mps|cuda|cpu (default: auto)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"=== train_mlps_local — device={device} ===\n")

    t0 = time.time()
    print(f"[load] joined_features.parquet ...")
    df = pd.read_parquet(ROUND_DIR / "results" / "joined_features.parquet",
                         columns=["well", "row_idx", "target", "fold"] + FEAT_COLS)
    print(f"  rows={len(df):,}  cols={df.shape[1]}  in {time.time()-t0:.1f}s")

    X = df[FEAT_COLS].astype(np.float32).copy()
    nan_counts = X.isna().sum()
    if nan_counts.sum():
        print(f"  filling NaN with median (cols affected: "
              f"{nan_counts[nan_counts>0].to_dict()})")
        X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6).values
    assert np.isfinite(X).all(), "non-finite in X after fillna"

    y      = df["target"].astype(np.float32).values
    folds  = df["fold"].values.astype(np.int8)
    wells  = df["well"].values
    rowidx = df["row_idx"].values.astype(np.int32)

    selected = set(args.only.split(",")) if args.only else None
    specs = [s for s in CANDIDATE_SPECS if (selected is None or s["cid"] in selected)]
    print(f"  running {len(specs)} candidates: {[s['cid'] for s in specs]}\n")

    for spec in specs:
        cid = spec["cid"]
        print(f"\n{'='*60}\n=== {cid}  spec={spec}\n{'='*60}")
        t_cid = time.time()

        oof = np.zeros(len(df), dtype=np.float32)
        fold_metrics = {}
        for fold in range(5):
            print(f"  fold {fold}:")
            tr_mask = folds != fold
            va_mask = folds == fold
            t_f = time.time()
            va_pred = train_one_fold(spec,
                                     X[tr_mask], y[tr_mask],
                                     X[va_mask], y[va_mask], device)
            oof[va_mask] = va_pred
            pw = perwell_rmse(y[va_mask], va_pred, wells[va_mask])
            fold_metrics[f"fold_{fold}"] = {"perwell": float(pw), "n": int(va_mask.sum())}
            print(f"    fold {fold} perwell={pw:.4f}  ({time.time()-t_f:.0f}s)")
            del va_pred
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        overall_pw = perwell_rmse(y, oof, wells)
        overall_fl = flat_rmse(y, oof)
        wall = time.time() - t_cid
        print(f"\n  → {cid}  honest perwell={overall_pw:.4f}  flat={overall_fl:.4f}  ({wall:.0f}s)")

        df_oof = pd.DataFrame({
            "well":     wells,
            "row_idx":  rowidx,
            "fold":     folds,
            "target":   y,
            "oof_pred": oof,
        })
        out = write_oof(
            candidate_id   = cid,
            df_oof         = df_oof,
            candidate_type = "torch_mlp" if spec["arch"] == "mlp" else "torch_resmlp",
            features_used  = FEAT_COLS,
            hyperparams    = {k: v for k, v in spec.items() if k != "cid"},
            seed           = spec["seed"],
            train_time_sec = wall,
            fold_metrics   = fold_metrics,
            extra_meta     = {"trained_on": f"local {device.type}",
                              "feat_count": len(FEAT_COLS)},
        )
        print(f"  → wrote {out}")

    print(f"\n=== ALL DONE  total wall: {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
