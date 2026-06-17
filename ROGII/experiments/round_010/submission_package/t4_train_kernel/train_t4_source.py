"""ROGII T4 train kernel — produce 5-fold OOF for tabular MLP candidates.

Runs on Kaggle Notebook with T4 x2 (we use 1 GPU, simpler). Loads pre-built
joined_features.parquet (3.78M rows × 66 cols, fold column already encoded),
trains 5 MLP candidates, writes /kaggle/working/<cid>.parquet — schema
matches our shared.oof_writer expectations: well, row_idx, fold, target, oof_pred.

Total runtime estimate on T4: ~10-20 min per candidate × 5 = 1-2h.
"""

# %%
import os, gc, time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print(f"torch {torch.__version__}  cuda? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("/kaggle/working")

# %%
# Discover joined_features.parquet — Kaggle mount path can vary so we glob.
import glob
candidates = sorted(glob.glob("/kaggle/input/**/joined_features.parquet", recursive=True))
print(f"input mount /kaggle/input contains: {os.listdir('/kaggle/input')}")
print(f"joined_features candidates: {candidates}")
assert candidates, "joined_features.parquet not found under /kaggle/input"
PARQUET_PATH = candidates[0]

# %%
# Load joined features (fold column is pre-built: sha256(well) % 5)
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"loaded {len(df):,} rows × {df.shape[1]} cols  in {time.time()-t0:.1f}s")
print(f"fold sizes: {df.groupby('fold').size().to_dict()}")

# %%
# feature_set_v14 — 43 cols, hard-coded so the kernel is self-contained
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
missing = set(FEAT_COLS) - set(df.columns)
assert not missing, f"missing feature cols: {missing}"
print(f"all 43 feature cols present")

# %%
# Build X, y, metadata. NaN→median, clip extremes (MLP doesn't tolerate raw NaN/Inf)
X = df[FEAT_COLS].astype(np.float32).copy()
nan_counts = X.isna().sum()
if nan_counts.sum() > 0:
    print(f"NaN columns:\n{nan_counts[nan_counts > 0]}")
medians = X.median()
X = X.fillna(medians).clip(lower=-1e6, upper=1e6).values  # → ndarray
print(f"X shape {X.shape}  has_nan={np.isnan(X).any()}  has_inf={np.isinf(X).any()}")

# Standardize per-feature using train-fold statistics inside the loop. Here we
# pre-compute global stats too (used as initial scaling for all-folds run).
y      = df["target"].astype(np.float32).values
folds  = df["fold"].values.astype(np.int8)
wells  = df["well"].values
rowidx = df["row_idx"].values.astype(np.int32)
print(f"y stats: mean={y.mean():.3f}  std={y.std():.3f}  min={y.min():.2f}  max={y.max():.2f}")

# %%
# Simple MLP architectures — used by candidate specs

class MLP(nn.Module):
    """Standard 4-layer MLP with BN + dropout. Used by c60-c63."""
    def __init__(self, n_in, hidden=(256, 128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ResBlock(nn.Module):
    def __init__(self, h, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(h, h); self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h); self.bn2 = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z = self.bn1(torch.relu(self.fc1(x)))
        z = self.drop(z)
        z = self.bn2(self.fc2(z))
        return torch.relu(x + z)

class ResMLP(nn.Module):
    """3 residual blocks + linear out. Used by c64."""
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


# %%
# Candidate specs — first-batch 5 MLPs with different seeds + losses + arch
CANDIDATE_SPECS = [
    dict(cid="c60_mlp_s42",        arch="mlp",    seed=42,   epochs=30, lr=1e-3,  loss="mse",   batch=4096),
    dict(cid="c61_mlp_s7",         arch="mlp",    seed=7,    epochs=30, lr=1e-3,  loss="mse",   batch=4096),
    dict(cid="c62_mlp_s2024",      arch="mlp",    seed=2024, epochs=30, lr=1e-3,  loss="mse",   batch=4096),
    dict(cid="c63_mlp_huber_s42",  arch="mlp",    seed=42,   epochs=30, lr=1e-3,  loss="huber", batch=4096),
    dict(cid="c64_resmlp_s42",     arch="resmlp", seed=42,   epochs=40, lr=1e-3,  loss="mse",   batch=4096),
]
print(f"{len(CANDIDATE_SPECS)} candidates queued")


# %%
# Per-well RMSE helper for in-kernel monitoring (vectorized via np.bincount)
def perwell_rmse(target, pred, wells_arr):
    codes, _ = pd.factorize(wells_arr, sort=False)
    counts = np.bincount(codes).astype(np.float64)
    diff = target - pred
    ss = np.bincount(codes, weights=(diff * diff).astype(np.float64), minlength=len(counts))
    return float(np.sqrt(ss / counts).mean())


def make_model(spec, n_in):
    if spec["arch"] == "mlp":
        return MLP(n_in)
    if spec["arch"] == "resmlp":
        return ResMLP(n_in)
    raise ValueError(spec["arch"])


def make_loss(spec):
    if spec["loss"] == "mse":
        return nn.MSELoss()
    if spec["loss"] == "huber":
        return nn.HuberLoss(delta=10.0)
    raise ValueError(spec["loss"])


def train_one_fold(spec, X_tr, y_tr, X_va, y_va):
    """Train one fold, return (model, va_pred). Uses train-fold standardization.

    Returns predictions on validation in original target space (no scaling on y).
    """
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])

    # Per-fold standardization: fit on TRAIN ONLY
    mu = X_tr.mean(0).astype(np.float32)
    sd = X_tr.std(0).astype(np.float32) + 1e-6
    X_tr_s = (X_tr - mu) / sd
    X_va_s = (X_va - mu) / sd

    ds_tr = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va))
    ld_tr = DataLoader(ds_tr, batch_size=spec["batch"], shuffle=True,  num_workers=2, pin_memory=True)
    ld_va = DataLoader(ds_va, batch_size=spec["batch"]*2, shuffle=False, num_workers=2, pin_memory=True)

    model = make_model(spec, n_in=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec["epochs"])
    loss_fn = make_loss(spec)

    best_va = math.inf
    best_pred = None
    best_epoch = -1
    for ep in range(spec["epochs"]):
        model.train()
        tr_loss = 0.0; tr_n = 0
        for xb, yb in ld_tr:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0); tr_n += xb.size(0)
        sched.step()

        model.eval()
        va_preds = []
        va_loss = 0.0; va_n = 0
        with torch.no_grad():
            for xb, yb in ld_va:
                xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
                pred = model(xb)
                va_loss += loss_fn(pred, yb).item() * xb.size(0); va_n += xb.size(0)
                va_preds.append(pred.cpu().numpy())
        va_loss /= va_n
        va_pred = np.concatenate(va_preds)
        # We use flat MSE on val as the in-loop metric (perwell needs wells_va, computed outside)
        if va_loss < best_va:
            best_va = va_loss
            best_pred = va_pred
            best_epoch = ep
        if ep % 5 == 0 or ep == spec["epochs"] - 1:
            print(f"      ep{ep:>2d}  tr_loss={tr_loss/tr_n:.4f}  va_loss={va_loss:.4f}  "
                  f"best={best_va:.4f}@ep{best_epoch}", flush=True)

    return best_pred.astype(np.float32)


# %%
# Main training loop — one candidate at a time
all_results = []
t_total = time.time()

for spec in CANDIDATE_SPECS:
    cid = spec["cid"]
    print(f"\n{'='*60}\n=== {cid}  spec={spec}\n{'='*60}")
    t_cid = time.time()

    oof = np.zeros(len(df), dtype=np.float32)
    fold_metrics = {}
    for fold in range(5):
        print(f"  fold {fold}:")
        tr_mask = folds != fold
        va_mask = folds == fold
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_va, y_va = X[va_mask], y[va_mask]
        wells_va = wells[va_mask]
        t_f = time.time()
        va_pred = train_one_fold(spec, X_tr, y_tr, X_va, y_va)
        oof[va_mask] = va_pred
        pw = perwell_rmse(y_va, va_pred, wells_va)
        fold_metrics[f"fold_{fold}"] = {"perwell": pw, "n": int(va_mask.sum())}
        print(f"    fold {fold} perwell={pw:.4f}  ({time.time()-t_f:.0f}s)")
        del X_tr, y_tr, X_va, y_va, va_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_pw = perwell_rmse(y, oof, wells)
    overall_flat = float(np.sqrt(((oof - y) ** 2).mean()))
    print(f"\n  → {cid}  honest perwell={overall_pw:.4f}  flat={overall_flat:.4f}  "
          f"({time.time()-t_cid:.0f}s)")

    # Write per-cid OOF parquet, schema = shared/oof_writer expectations
    df_oof = pd.DataFrame({
        "well":     wells,
        "row_idx":  rowidx,
        "fold":     folds,
        "target":   y,
        "oof_pred": oof,
    })
    out = OUTPUT_DIR / f"{cid}.parquet"
    df_oof.to_parquet(out)
    print(f"  → wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    all_results.append({
        "cid": cid, "spec": spec,
        "perwell": overall_pw, "flat": overall_flat,
        "fold_metrics": fold_metrics,
        "wall_sec": time.time() - t_cid,
    })

# %%
# Final summary
print(f"\n\n{'='*60}\n=== ALL DONE  total wall: {(time.time()-t_total)/60:.1f} min\n{'='*60}\n")
for r in all_results:
    print(f"  {r['cid']:25s}  perwell={r['perwell']:.4f}  flat={r['flat']:.4f}  "
          f"({r['wall_sec']:.0f}s)")

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nsummary.json written")
