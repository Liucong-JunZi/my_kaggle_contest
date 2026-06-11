#!/usr/bin/env python3
"""Evaluate hengck23's pretrained GeoSteerNet model on validation data."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerModel


# ------------------------------------------------------------
# Model (EXACT match for checkpoint keys)
# ------------------------------------------------------------
class GeoSteerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Parameter(torch.zeros(1))
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/mit-b0", num_channels=3, ignore_mismatched_sizes=True)
        dims = [32, 64, 160, 256]
        self.proj = nn.ModuleList([nn.Conv2d(d, 128, 1) for d in dims])
        self.fuse  = nn.Conv2d(512, 128, 1)
        self.fuse1 = nn.Conv2d(1, 128, 1)
        self.fuse2 = nn.Conv2d(1, 128, 1)  # present in checkpoint
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, batch_img):
        # batch_img: (B, 3, 256, 768) = [t_gr, h_gr, history]
        B, C, T, H = batch_img.shape
        t_gr1  = batch_img[:,0:1].expand(B,1,T,H)
        h_gr1  = batch_img[:,1:2].expand(B,1,T,H)
        history = batch_img[:,2:3]
        image = torch.cat([t_gr1, h_gr1, history], dim=1)
        out = self.backbone(pixel_values=image, output_hidden_states=True)
        feats = out.hidden_states
        fH, fW = feats[0].shape[-2:]
        pooled = []
        for i, f in enumerate(feats):
            f = self.proj[i](f)
            if f.shape[3] != fW:
                f = F.interpolate(f, size=(fH,fW), mode="bilinear", align_corners=False)
            pooled.append(f)
        pooled = torch.cat(pooled, dim=1)
        pooled = self.fuse(pooled) + self.fuse1(
            F.interpolate(history, size=(fH,fW), mode="bilinear", align_corners=False))
        sdf = self.head(pooled)
        sdf = F.interpolate(sdf, size=(T, H), mode="bilinear", align_corners=False)
        return torch.tanh(sdf) * 3


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class H5DS(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")
    def __len__(self):
        return len(self.f["X"])
    def __getitem__(self, i):
        return (torch.from_numpy(self.f["X"][i]).float(),
                torch.from_numpy(self.f["y_tvt"][i]).float(),
                torch.from_numpy(self.f["mask"][i]).float(),
                torch.from_numpy(self.f["t_tvt"][i]).float())


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def evaluate(checkpoint_path, device="mps"):
    print(f"Loading checkpoint: {checkpoint_path}")
    t0 = time.time()

    print("Building model (downloads SegFormer backbone if needed)...")
    model = GeoSteerNet().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    ds = H5DS("data/cache/cfg-img-hengck23/val.h5")
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    all_rmse = []
    per_well = {}
    t0 = time.time()

    with torch.no_grad():
        for bi, (X, y_tvt, mask, t_tvt_grid) in enumerate(dl):
            X = X.to(device)
            sdf = model(X)[:,0].cpu().numpy()  # (B, T, H)
            for b in range(X.shape[0]):
                well_idx = bi * dl.batch_size + b
                if well_idx >= len(ds):
                    break

                # For each column (horizontal position), find the depth index
                # where SDF is closest to zero (the geological surface)
                t_idx = np.abs(sdf[b]).argmin(axis=0)  # (H,)

                # Correct TVT mapping: t_tvt[t*] = h_tvt[h] - sdf[t*,h] * sdf_scale
                # When sdf ≈ 0: t_tvt[t*] ≈ h_tvt[h]
                tvt_grid_np = t_tvt_grid[b].numpy()
                tvt_pred = tvt_grid_np[t_idx]  # (H,)

                y_true = y_tvt[b].numpy()
                valid = mask[b].numpy() > 0.5
                if valid.sum() > 10:
                    rmse = float(np.sqrt(np.mean(
                        (tvt_pred[valid] - y_true[valid]) ** 2)))
                    all_rmse.append(rmse)
                    per_well[int(well_idx)] = rmse
                else:
                    per_well[int(well_idx)] = None

    elapsed = time.time() - t0
    print(f"\nEvaluated {len(ds)} wells in {elapsed:.1f}s")

    results = {
        "checkpoint": checkpoint_path,
        "n_wells_evaluated": len(all_rmse),
        "n_wells_skipped": len(per_well) - len(all_rmse),
        "mean_rmse": float(np.mean(all_rmse)),
        "median_rmse": float(np.median(all_rmse)),
        "min_rmse": float(np.min(all_rmse)),
        "max_rmse": float(np.max(all_rmse)),
        "std_rmse": float(np.std(all_rmse)),
        "per_well_rmse": per_well,
    }
    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    checkpoint_paths = [
        "archive/00007913.pth",
        "archive/00062000.pth",
    ]

    all_results = {}
    for ckpt in checkpoint_paths:
        if not os.path.exists(ckpt):
            print(f"Skipping missing: {ckpt}")
            continue
        key = os.path.splitext(os.path.basename(ckpt))[0]
        try:
            results = evaluate(ckpt, device=device)
            all_results[key] = results

            print(f"\n--- {key} ---")
            print(f"  Wells evaluated: {results['n_wells_evaluated']}")
            print(f"  Mean RMSE:   {results['mean_rmse']:.4f} ft")
            print(f"  Median RMSE: {results['median_rmse']:.4f} ft")
            print(f"  Min RMSE:    {results['min_rmse']:.4f} ft")
            print(f"  Max RMSE:    {results['max_rmse']:.4f} ft")
            print(f"  Std RMSE:    {results['std_rmse']:.4f} ft")
        except Exception as e:
            print(f"\n--- {key} ---")
            print(f"  FAILED: {e}")
            all_results[key] = {"status": "FAILED", "reason": str(e)}

    # Save
    out_path = "data/cache/cfg-img-hengck23/pretrained_eval.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
