#!/usr/bin/env python3
"""Train SegFormer-based GeoSteerNet on any cfg-img-* HDF5 dataset.

Usage:
    python src/train.py --dataset data/cache/cfg-img-medium
    python src/train.py --dataset data/cache/cfg-img-medium --backbone nvidia/mit-b1
    python src/train.py --dataset data/cache/cfg-img-medium-4ch --epochs 20

HDF5 schema (produced by src/gen_images.py):
    X        (N, C, T, H)  — input image
    y_sdf    (N, 1, T, H)  — SDF target
    y_tvt    (N, H)        — horizontal TVT (target for RMSE)
    t_tvt    (N, T)        — typewell TVT grid for SDF→TVT lookup
    mask     (N, H)        — horizontal validity mask
    well_ids (N,)          — bytes

Eval: argmin(|sdf|) per column → t_tvt[idx] → masked RMSE vs y_tvt.
Convention: `history` channel is the LAST channel of X (channel C-1).
"""

import argparse, json, time, warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ── Dataset ─────────────────────────────────────────────────────────────────
class H5WellDataset(Dataset):
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.N = f["X"].shape[0]
            self.C = f["X"].shape[1]
            self.T = f["X"].shape[2]
            self.H = f["X"].shape[3]
            if "t_tvt" not in f:
                raise RuntimeError(
                    f"{h5_path} missing /t_tvt — regenerate with current src/gen_images.py"
                )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            X = f["X"][idx]
            mask = f["mask"][idx]
            y_sdf = f["y_sdf"][idx]
            y_tvt = f["y_tvt"][idx]
            t_tvt = f["t_tvt"][idx]
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(y_sdf).float(),
            torch.from_numpy(y_tvt).float(),
            torch.from_numpy(t_tvt).float(),
        )


# ── Model ───────────────────────────────────────────────────────────────────
class GeoSteerNet(nn.Module):
    """SegFormer backbone + FPN fusion + history-aware head → SDF prediction."""

    def __init__(self, in_channels: int = 3, backbone_name: str = "nvidia/mit-b0"):
        super().__init__()
        from transformers import SegformerModel

        self.backbone = SegformerModel.from_pretrained(
            backbone_name, num_channels=in_channels, ignore_mismatched_sizes=True
        )
        hsizes = self.backbone.config.hidden_sizes  # e.g. [32,64,160,256] for b0
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

    def forward(self, x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
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
            F.interpolate(history, size=(fH, fW), mode="bilinear", align_corners=False)
        )
        sdf = self.head(fused)
        return torch.tanh(sdf) * 3.0


def masked_mse(pred, target, mask_2d):
    mse = F.mse_loss(pred, target, reduction="none")
    return (mse * mask_2d).sum() / mask_2d.sum().clamp(min=1)


# ── Train loop ──────────────────────────────────────────────────────────────
def train(args):
    data_dir = Path(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_id = out_dir.name

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Config: {config_id} | device: {device} | backbone: {args.backbone}")

    train_ds = H5WellDataset(data_dir / "train.h5")
    val_ds = H5WellDataset(data_dir / "val.h5")
    C, T, H = train_ds.C, train_ds.T, train_ds.H
    print(f"Shape: C={C} T={T} H={H} | train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GeoSteerNet(in_channels=C, backbone_name=args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_rmses = [], []
    best_rmse, best_epoch = float("inf"), 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        for X, mask_h, y_sdf, _, _ in train_loader:
            X, mask_h, y_sdf = X.to(device), mask_h.to(device), y_sdf.to(device)
            B, _, T_dim, _ = X.shape
            mask_2d = torch.ones(B, T_dim, device=device)[:, None, :, None] * mask_h[:, None, None, :]
            history = X[:, C - 1 : C]  # last channel is history (by gen_images.py convention)
            loss = masked_mse(model(X, history), y_sdf, mask_2d)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_losses.append(loss.item())
        train_losses.append(float(np.mean(ep_losses)))

        model.eval()
        all_rmse = []
        with torch.no_grad():
            for X, mask_h, _, y_tvt, t_tvt in val_loader:
                X, mask_h = X.to(device), mask_h.to(device)
                y_tvt, t_tvt = y_tvt.to(device), t_tvt.to(device)
                history = X[:, C - 1 : C]
                sdf_abs = model(X, history).abs().squeeze(1)
                best_t = sdf_abs.argmin(dim=1)  # (B, H)
                tvt_pred = torch.gather(t_tvt, 1, best_t)
                rmse = torch.sqrt(
                    ((tvt_pred - y_tvt) ** 2 * mask_h).sum(dim=1)
                    / mask_h.sum(dim=1).clamp(min=1)
                )
                all_rmse.extend(rmse.cpu().numpy().tolist())
        val_rmse = float(np.mean(all_rmse))
        val_rmses.append(val_rmse)

        status = "✓ NEW BEST" if val_rmse < best_rmse else f"({best_rmse:.2f} best)"
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            torch.save(model.state_dict(), out_dir / "best_model.pth")
        print(
            f"epoch {epoch:2d} | loss={train_losses[-1]:.4f} | val_rmse={val_rmse:.2f} | {status}",
            flush=True,
        )

    t_total = time.time() - t_start
    metrics = {
        "config_id": config_id,
        "dataset_path": str(data_dir),
        "model": f"SegFormer-{args.backbone.split('/')[-1]}",
        "channels": C, "T": T, "H": H,
        "epochs": args.epochs, "best_epoch": best_epoch,
        "best_val_rmse": round(best_rmse, 4),
        "final_val_rmse": round(val_rmses[-1], 4),
        "train_loss": [round(l, 4) for l in train_losses],
        "val_rmse_per_epoch": [round(r, 4) for r in val_rmses],
        "training_time_sec": round(t_total, 1),
        "batch_size": args.batch_size, "device": str(device),
        "tvt_method": "t_tvt_grid_lookup",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBest: {best_rmse:.2f} ft @ ep{best_epoch} | {t_total:.0f}s | → {out_dir}/metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset dir with train.h5 + val.h5")
    ap.add_argument("--output-dir", default=None, help="Output dir (default: same as dataset)")
    ap.add_argument("--backbone", default="nvidia/mit-b0",
                    choices=["nvidia/mit-b0", "nvidia/mit-b1", "nvidia/mit-b2"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    train(ap.parse_args())
