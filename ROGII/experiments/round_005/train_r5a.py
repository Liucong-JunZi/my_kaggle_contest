#!/usr/bin/env python3
"""R5-A: TVT-aware training — Soft-argmin + Huber TVT loss.

Hypothesis: train_loss (SDF MSE) plateaus at 0.20 while val RMSE stops
improving — the train objective doesn't directly optimize what we evaluate.
Solution: add a differentiable TVT estimate via soft-argmin on |sdf| and
penalize TVT prediction with Huber.

Loss = α · masked_MSE(sdf_pred, sdf_target) + β · masked_Huber(tvt_pred, tvt_true)

where tvt_pred[b,h] = Σ_t softmax(-|sdf_pred|/τ)[b,t,h] * t_tvt[b,t].

Sweep:
    --tau (softmax temperature; lower = sharper, closer to hard argmin)
    --beta (TVT loss weight; α fixed at 1.0)
"""

import argparse, json, time, warnings, sys
from pathlib import Path

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/src")
from decode import decode_sdf_to_tvt, anchor_known_segment, masked_rmse

warnings.filterwarnings("ignore")


class H5WellDataset(Dataset):
    def __init__(self, p):
        self.p = p
        with h5py.File(p, "r") as f:
            self.N, self.C, self.T, self.H = f["X"].shape
            if "t_tvt" not in f:
                raise RuntimeError(f"{p} missing /t_tvt")

    def __len__(self): return self.N

    def __getitem__(self, i):
        with h5py.File(self.p, "r") as f:
            return (
                torch.from_numpy(f["X"][i]).float(),
                torch.from_numpy(f["mask"][i]).float(),
                torch.from_numpy(f["y_sdf"][i]).float(),
                torch.from_numpy(f["y_tvt"][i]).float(),
                torch.from_numpy(f["t_tvt"][i]).float(),
            )


class GeoSteerNet(nn.Module):
    """Identical to src/train.py — reproduce here so R5-A is self-contained."""

    def __init__(self, in_channels=3, backbone_name="nvidia/mit-b0"):
        super().__init__()
        from transformers import SegformerModel
        self.backbone = SegformerModel.from_pretrained(
            backbone_name, num_channels=in_channels, ignore_mismatched_sizes=True)
        hsizes = self.backbone.config.hidden_sizes
        self.proj = nn.ModuleList([nn.Conv2d(hsizes[i], 128, 1) for i in range(4)])
        self.fuse = nn.Conv2d(128 * 4, 128, 1)
        self.fuse_history = nn.Conv2d(1, 128, 1)
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x, history):
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        features = out.hidden_states
        fH, fW = features[0].shape[-2:]
        pooled = []
        for i in range(4):
            f = self.proj[i](features[i])
            if f.shape[2] != fH or f.shape[3] != fW:
                f = F.interpolate(f, size=(fH, fW), mode="bilinear", align_corners=False)
            pooled.append(f)
        fused = self.fuse(torch.cat(pooled, dim=1)) + self.fuse_history(
            F.interpolate(history, size=(fH, fW), mode="bilinear", align_corners=False))
        return torch.tanh(self.head(fused)) * 3.0


def masked_mse_sdf(pred, target, mask_2d):
    mse = F.mse_loss(pred, target, reduction="none")
    return (mse * mask_2d).sum() / mask_2d.sum().clamp(min=1)


def soft_argmin_tvt(sdf, t_tvt, tau):
    """Differentiable TVT estimate.
    sdf:    (B, 1, T, H) signed (will be |.| to give soft argmin of magnitude)
    t_tvt:  (B, T)
    Returns (B, H).
    """
    abs_sdf = sdf.abs().squeeze(1)               # (B, T, H)
    w = F.softmax(-abs_sdf / tau, dim=1)         # (B, T, H), peaks where |sdf|≈0
    return (w * t_tvt.unsqueeze(-1)).sum(dim=1)  # (B, H)


def masked_huber(pred, target, mask, delta=10.0):
    """Huber loss in TVT space (ft). delta in ft — quadratic below, linear above."""
    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=err.device))
    lin = abs_err - quad
    huber = 0.5 * quad ** 2 + delta * lin
    return (huber * mask).sum() / mask.sum().clamp(min=1)


def train(args):
    data_dir = Path(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_id = out_dir.name

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"R5-A: {config_id} | dev={device} | bb={args.backbone}")
    print(f"      tau={args.tau} beta={args.beta} alpha={args.alpha} huber_delta={args.huber_delta}")

    train_ds = H5WellDataset(data_dir / "train.h5")
    val_ds = H5WellDataset(data_dir / "val.h5")
    C, T, H = train_ds.C, train_ds.T, train_ds.H
    print(f"Shape: C={C} T={T} H={H} | train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GeoSteerNet(in_channels=C, backbone_name=args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    H_H = args.h_known
    train_losses, train_sdf_losses, train_tvt_losses = [], [], []
    val_rmses, val_rmses_anc = [], []
    best_anc_rmse = float("inf"); best_rmse = float("inf"); best_epoch = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # R5-A warmup: ramp β linearly over args.warmup_epochs to avoid SDF disruption
        beta_eff = args.beta * min(1.0, (epoch - 1) / max(1, args.warmup_epochs))

        model.train()
        ep_l, ep_l_sdf, ep_l_tvt = [], [], []
        for X, mask_h, y_sdf, y_tvt, t_tvt in train_loader:
            X = X.to(device); mask_h = mask_h.to(device); y_sdf = y_sdf.to(device)
            y_tvt = y_tvt.to(device); t_tvt = t_tvt.to(device)
            B, _, T_dim, _ = X.shape

            mask_2d = torch.ones(B, T_dim, device=device)[:, None, :, None] * mask_h[:, None, None, :]
            history = X[:, C-1:C]
            sdf_pred = model(X, history)

            loss_sdf = masked_mse_sdf(sdf_pred, y_sdf, mask_2d)
            tvt_pred = soft_argmin_tvt(sdf_pred, t_tvt, tau=args.tau)
            loss_tvt = masked_huber(tvt_pred, y_tvt, mask_h, delta=args.huber_delta)
            loss = args.alpha * loss_sdf + beta_eff * loss_tvt

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_l.append(loss.item()); ep_l_sdf.append(loss_sdf.item()); ep_l_tvt.append(loss_tvt.item())

        train_losses.append(float(np.mean(ep_l)))
        train_sdf_losses.append(float(np.mean(ep_l_sdf)))
        train_tvt_losses.append(float(np.mean(ep_l_tvt)))

        # Eval (same pipeline as src/train.py — raw argmin + anchored)
        model.eval()
        sdfs, masks, ytvts, ttvts = [], [], [], []
        with torch.no_grad():
            for X, mask_h, _, y_tvt, t_tvt in val_loader:
                X = X.to(device)
                sdfs.append(model(X, X[:, C-1:C]).abs().squeeze(1).cpu().numpy())
                masks.append(mask_h.numpy())
                ytvts.append(y_tvt.numpy()); ttvts.append(t_tvt.numpy())
        sdf_abs = np.concatenate(sdfs, 0); mask_np = np.concatenate(masks, 0)
        y_tvt_np = np.concatenate(ytvts, 0); t_tvt_np = np.concatenate(ttvts, 0)

        tvt_raw = decode_sdf_to_tvt(sdf_abs, t_tvt_np, subpixel=False)
        tvt_sub = decode_sdf_to_tvt(sdf_abs, t_tvt_np, subpixel=True)
        tvt_anc = anchor_known_segment(tvt_sub, y_tvt_np[:, :H_H], mask_np[:, :H_H], alpha=0.75)
        val_rmse = float(masked_rmse(tvt_raw, y_tvt_np, mask_np).mean())
        val_rmse_anc = float(masked_rmse(tvt_anc, y_tvt_np, mask_np).mean())
        val_rmses.append(val_rmse); val_rmses_anc.append(val_rmse_anc)

        flag = "✓ NEW BEST" if val_rmse_anc < best_anc_rmse else f"({best_anc_rmse:.2f} best)"
        if val_rmse_anc < best_anc_rmse:
            best_anc_rmse = val_rmse_anc; best_rmse = val_rmse; best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pth")

        print(
            f"ep {epoch:2d} | β={beta_eff:.4f} L={train_losses[-1]:.4f} (sdf={train_sdf_losses[-1]:.3f} "
            f"tvt={train_tvt_losses[-1]:.2f}) | raw={val_rmse:.2f} anc={val_rmse_anc:.2f} | {flag}",
            flush=True,
        )

    t_total = time.time() - t_start
    metrics = {
        "config_id": config_id,
        "dataset_path": str(data_dir),
        "model": f"SegFormer-{args.backbone.split('/')[-1]}",
        "channels": C, "T": T, "H": H,
        "loss_type": "R5-A: SDF_MSE + Huber_TVT (soft-argmin)",
        "loss_params": {
            "alpha": args.alpha, "beta": args.beta,
            "tau": args.tau, "huber_delta": args.huber_delta,
        },
        "epochs": args.epochs, "best_epoch": best_epoch,
        "best_val_rmse_raw": round(best_rmse, 4),
        "best_val_rmse_anchored": round(best_anc_rmse, 4),
        "final_val_rmse_raw": round(val_rmses[-1], 4),
        "final_val_rmse_anchored": round(val_rmses_anc[-1], 4),
        "train_loss_total": [round(l, 4) for l in train_losses],
        "train_loss_sdf": [round(l, 4) for l in train_sdf_losses],
        "train_loss_tvt": [round(l, 4) for l in train_tvt_losses],
        "val_rmse_raw_per_epoch": [round(r, 4) for r in val_rmses],
        "val_rmse_anc_per_epoch": [round(r, 4) for r in val_rmses_anc],
        "training_time_sec": round(t_total, 1),
        "batch_size": args.batch_size, "device": str(device),
        "tvt_method": "t_tvt_grid_lookup",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBest: raw={best_rmse:.2f} anc={best_anc_rmse:.2f} ft @ ep{best_epoch} | {t_total:.0f}s")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--backbone", default="nvidia/mit-b0",
                    choices=["nvidia/mit-b0", "nvidia/mit-b1", "nvidia/mit-b2"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--h-known", type=int, default=48)
    # R5-A loss controls
    ap.add_argument("--alpha", type=float, default=1.0, help="SDF MSE weight")
    ap.add_argument("--beta",  type=float, default=0.01, help="TVT Huber weight (final)")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="Linearly ramp beta from 0 to --beta over N epochs")
    ap.add_argument("--tau",   type=float, default=0.1, help="Softmax temperature for soft argmin")
    ap.add_argument("--huber-delta", type=float, default=10.0, help="Huber transition (ft)")
    train(ap.parse_args())
