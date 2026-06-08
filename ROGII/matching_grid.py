"""
2D Matching Grid Image Generator for ROGII Wellbore Geology Prediction

Idea (hengck23): Each image is a (C, H, T) grid where
  H = compressed horizontal MD steps
  T = typewell TVT steps

Pixel (h, t) encodes features at the potential match of horizontal step h
against typewell depth t — the CNN sees this as a 2D "where am I?" problem.

Configurable channels:
  t_gr      — typewell GR at TVT depth t (broadcast across H)
  h_gr      — horizontal GR at MD step h (broadcast across T)
  gr_diff   — t_gr - h_gr (the matching signal)
  sdf       — Signed Distance Function: t_tvt - h_tvt_history
  z         — Z coordinate (broadcast)
  dz        — gradient of Z (broadcast)
  x, y      — coordinates (broadcast)
  tvt_mask  — known-TVT mask (broadcast)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import glob, os

# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class MatchingGridConfig:
    # Image dimensions
    compression: int = 8           # MD downsampling factor
    target_height: Optional[int] = None  # fixed H; None = derived from MD/compression
    target_width:  Optional[int] = None  # fixed T; None = use typewell TVT length

    # Channel selection
    channels: List[str] = field(default_factory=lambda: [
        "t_gr", "h_gr", "gr_diff", "sdf"
    ])

    # Preprocessing
    normalize: str = "per_image"   # "per_image" | "global" | "none"
    sdf_clip: Optional[float] = 200.0  # clip SDF magnitude (ft)
    interpolation: str = "nearest" # resize interpolation
    fill_h_gr_nan: str = "interpolate"  # "zero" | "interpolate" | "mean"

    # Global stats (set externally for "global" normalize)
    gr_mean: float = 80.0
    gr_std:  float = 30.0
    z_mean:  float = -9000.0
    z_std:   float = 500.0
    coord_mean: float = 0.0
    coord_std:  float = 5000.0


# ── Generator ───────────────────────────────────────────────────────────

class MatchingGridGenerator:
    """Generate (C, H, T) matching grid images from well data."""

    def __init__(self, config: MatchingGridConfig):
        self.cfg = config

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pool1d(arr, factor: int) -> np.ndarray:
        """Average-pool 1d array by factor."""
        if factor <= 1:
            return arr.copy()
        n = len(arr)
        # truncate to multiple of factor
        valid = n - (n % factor)
        if valid == 0:
            return arr.copy()
        return arr[:valid].reshape(-1, factor).mean(axis=1)

    @staticmethod
    def _interp_fill(x: np.ndarray) -> np.ndarray:
        """Linear interpolate NaN values."""
        mask = np.isnan(x)
        if not mask.any():
            return x.copy()
        ok = ~mask
        if not ok.any():
            return np.zeros_like(x)
        xp = np.flatnonzero(ok)
        fp = x[ok]
        y = np.interp(np.arange(len(x)), xp, fp)
        # extrapolate edges
        y[:xp[0]] = fp[0]
        y[xp[-1]+1:] = fp[-1]
        return y

    @staticmethod
    def _normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (arr - mean) / max(std, 1e-8)

    # ── main generate ────────────────────────────────────────────────

    def generate(self, hw_df: pd.DataFrame, tw_df: pd.DataFrame) -> np.ndarray:
        """
        Generate (C, H, T) float32 matching grid for one well.

        Parameters
        ----------
        hw_df : DataFrame with columns [MD, X, Y, Z, GR, TVT, TVT_input]
        tw_df : DataFrame with columns [TVT, GR]
        """
        cfg = self.cfg

        # ── extract raw data ──────────────────────────────────────────
        md   = hw_df["MD"].values.astype(np.float64)
        z    = hw_df["Z"].values.astype(np.float64)
        gr_h = hw_df["GR"].values.astype(np.float64)
        tvt_h = hw_df["TVT"].values.astype(np.float64)
        tvt_mask = hw_df["TVT_input"].notna().values.astype(np.float32)
        x    = hw_df.get("X", pd.Series(np.zeros(len(md)))).values.astype(np.float64)
        y    = hw_df.get("Y", pd.Series(np.zeros(len(md)))).values.astype(np.float64)

        tvt_t = tw_df["TVT"].values.astype(np.float64)
        gr_t  = tw_df["GR"].values.astype(np.float64)

        N_h = len(md)
        N_t = len(tvt_t)

        # ── fill GR NaN ───────────────────────────────────────────────
        if cfg.fill_h_gr_nan == "interpolate":
            gr_h = self._interp_fill(gr_h)
        elif cfg.fill_h_gr_nan == "mean":
            m = np.nanmean(gr_h)
            gr_h = np.where(np.isnan(gr_h), m if not np.isnan(m) else 0, gr_h)
        else:  # "zero"
            gr_h = np.nan_to_num(gr_h, 0)

        gr_t = np.nan_to_num(gr_t, 0)

        # ── compute dz ────────────────────────────────────────────────
        dz = np.gradient(z)

        # ── compress MD dimension ─────────────────────────────────────
        factor = cfg.compression
        md_c    = self._pool1d(md, factor)
        gr_h_c  = self._pool1d(gr_h, factor)
        z_c     = self._pool1d(z, factor)
        dz_c    = self._pool1d(dz, factor)
        x_c     = self._pool1d(x, factor)
        y_c     = self._pool1d(y, factor)
        tvt_h_c = self._pool1d(tvt_h, factor)
        tvt_mask_c = self._pool1d(tvt_mask, factor)
        # threshold mask: >0.5 means mostly known
        tvt_mask_c = (tvt_mask_c > 0.5).astype(np.float32)

        H = len(md_c)

        # ── target dimensions ─────────────────────────────────────────
        T = cfg.target_width or N_t
        H_out = cfg.target_height or H

        # ── interpolate typewell to target T grid ─────────────────────
        if T != N_t:
            idx_t = np.linspace(0, N_t - 1, T)
            idx_t_clipped = np.clip(idx_t, 0, N_t - 1).astype(int)
            tvt_t_grid = tvt_t[idx_t_clipped]
            gr_t_grid  = gr_t[idx_t_clipped]
        else:
            tvt_t_grid = tvt_t
            gr_t_grid  = gr_t

        # ── build known TVT history (last known value per step) ───────
        h_tvt_history = np.full(H, np.nan, dtype=np.float64)
        last_known = np.nan
        for i in range(H):
            if tvt_mask_c[i] > 0:
                last_known = tvt_h_c[i]
            h_tvt_history[i] = last_known

        # Determine H_out; resize if needed
        if H_out != H:
            # nearest-neighbor down/up-sample for H
            idx_h = np.linspace(0, H - 1, H_out).astype(int)
            gr_h_c   = gr_h_c[idx_h]
            z_c      = z_c[idx_h]
            dz_c     = dz_c[idx_h]
            x_c      = x_c[idx_h]
            y_c      = y_c[idx_h]
            h_tvt_history = h_tvt_history[idx_h]
            tvt_mask_c    = tvt_mask_c[idx_h]
            H = H_out

        # ── build channels ────────────────────────────────────────────
        channel_list = []

        for ch in cfg.channels:
            if ch == "t_gr":
                # typewell GR broadcast: (H, T)
                c = np.tile(gr_t_grid.reshape(1, T), (H, 1))

            elif ch == "h_gr":
                # horizontal GR broadcast: (H, T)
                c = np.tile(gr_h_c.reshape(H, 1), (1, T))

            elif ch == "gr_diff":
                # GR difference
                t_gr_brd = np.tile(gr_t_grid.reshape(1, T), (H, 1))
                h_gr_brd = np.tile(gr_h_c.reshape(H, 1), (1, T))
                c = t_gr_brd - h_gr_brd

            elif ch == "sdf":
                # Signed Distance Function: t_tvt - h_tvt_history
                sdf = tvt_t_grid.reshape(1, T) - h_tvt_history.reshape(H, 1)
                mask = tvt_mask_c.reshape(H, 1)
                sdf = sdf * mask  # zero out where history is unknown
                if cfg.sdf_clip is not None:
                    sdf = np.clip(sdf, -cfg.sdf_clip, cfg.sdf_clip)
                c = sdf

            elif ch == "sdf_sign":
                sdf = tvt_t_grid.reshape(1, T) - h_tvt_history.reshape(H, 1)
                mask = tvt_mask_c.reshape(H, 1)
                c = np.sign(sdf) * mask

            elif ch == "z":
                c = np.tile(z_c.reshape(H, 1), (1, T))

            elif ch == "dz":
                c = np.tile(dz_c.reshape(H, 1), (1, T))

            elif ch == "x":
                c = np.tile(x_c.reshape(H, 1), (1, T))

            elif ch == "y":
                c = np.tile(y_c.reshape(H, 1), (1, T))

            elif ch == "tvt_mask":
                c = np.tile(tvt_mask_c.reshape(H, 1), (1, T))

            elif ch == "gr_corr":
                # Local GR correlation in sliding window
                # For each (h, t), compute correlation of h_gr window
                # around h with t_gr window around t
                win = 10  # half-window size
                corr_map = np.zeros((H, T), dtype=np.float32)
                for h in range(H):
                    h0, h1 = max(0, h - win), min(H, h + win + 1)
                    h_seg = gr_h_c[h0:h1]
                    if len(h_seg) < 3:
                        continue
                    h_mean, h_std = h_seg.mean(), h_seg.std()
                    if h_std < 1e-6:
                        continue
                    for t in range(T):
                        t0, t1 = max(0, t - win), min(T, t + win + 1)
                        t_seg = gr_t_grid[t0:t1]
                        if len(t_seg) < 3:
                            continue
                        t_mean, t_std = t_seg.mean(), t_seg.std()
                        if t_std < 1e-6:
                            continue
                        # Interpolate one to the other's length
                        common = min(len(h_seg), len(t_seg))
                        corr = np.corrcoef(h_seg[:common], t_seg[:common])[0, 1]
                        corr_map[h, t] = 0 if np.isnan(corr) else corr
                c = corr_map

            else:
                raise ValueError(f"Unknown channel: {ch}")

            # ── normalize channel ─────────────────────────────────────
            if cfg.normalize == "per_image":
                c_mean, c_std = c.mean(), c.std()
                if c_std < 1e-8:
                    c_std = 1.0
                c = (c - c_mean) / c_std
            elif cfg.normalize == "global":
                if ch in ("t_gr", "h_gr", "gr_diff", "gr_corr"):
                    c = (c - cfg.gr_mean) / max(cfg.gr_std, 1e-8)
                elif ch == "z":
                    c = (c - cfg.z_mean) / max(cfg.z_std, 1e-8)
                elif ch in ("x", "y"):
                    c = (c - cfg.coord_mean) / max(cfg.coord_std, 1e-8)
                # SDF and others: per-image normalize fallback

            channel_list.append(c.astype(np.float32))

        # Stack: (C, H, T)
        return np.stack(channel_list, axis=0)

    def generate_batch(
        self, well_ids: List[str], data_dir: str
    ) -> np.ndarray:
        """Generate (B, C, H, T) batch for a list of well IDs."""
        images = []
        for wid in well_ids:
            hw_f = os.path.join(data_dir, f"{wid}__horizontal_well.csv")
            tw_f = os.path.join(data_dir, f"{wid}__typewell.csv")
            hw_df = pd.read_csv(hw_f)
            tw_df = pd.read_csv(tw_f)
            img = self.generate(hw_df, tw_df)
            images.append(img)
        return np.stack(images, axis=0)

    def generate_target(self, hw_df: pd.DataFrame) -> np.ndarray:
        """
        Generate TVT target for the compressed MD grid (lateral segment only).

        Returns (H,) float32, NaN where TVT is unknown (prediction target).
        """
        cfg = self.cfg
        tvt = hw_df["TVT"].values.astype(np.float64)
        tvt_c = self._pool1d(tvt, cfg.compression)
        return tvt_c.astype(np.float32)


# ── Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"

    # Define config variants to search
    configs = {
        "basic": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf"],
            compression=8,
        ),
        "basic_sdf_sign": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf_sign"],
            compression=8,
        ),
        "geometry": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf", "z", "dz"],
            compression=8,
        ),
        "minimal": MatchingGridConfig(
            channels=["t_gr", "h_gr"],
            compression=8,
        ),
        "rich": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf", "z", "dz", "x", "y"],
            compression=8,
        ),
        "high_res": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf"],
            compression=4,
        ),
        "low_res": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf"],
            compression=16,
        ),
        "global_norm": MatchingGridConfig(
            channels=["t_gr", "h_gr", "gr_diff", "sdf"],
            compression=8,
            normalize="global",
        ),
    }

    print("=" * 70)
    print("Matching Grid Image Generator — Configuration Test")
    print("=" * 70)

    # Test on first 3 wells
    hw_files = sorted(glob.glob(f"{DATA}/*__horizontal_well.csv"))[:3]

    for cfg_name, cfg in configs.items():
        print(f"\n{'─' * 70}")
        print(f"Config: {cfg_name}")
        print(f"  Channels: {cfg.channels}")
        print(f"  Compression: {cfg.compression}")
        print(f"  Normalize: {cfg.normalize}")
        gen = MatchingGridGenerator(cfg)

        for hw_f in hw_files:
            well_id = Path(hw_f).name.split("__")[0]
            tw_f = hw_f.replace("__horizontal_well.csv", "__typewell.csv")
            hw_df = pd.read_csv(hw_f)
            tw_df = pd.read_csv(tw_f)

            arr = gen.generate(hw_df, tw_df)
            target = gen.generate_target(hw_df)
            n_known = hw_df["TVT_input"].notna().sum()
            n_total = len(hw_df)

            print(f"  {well_id}: image={arr.shape} (C×H×T), "
                  f"target={target.shape}, "
                  f"val_range=[{arr.min():.3f}, {arr.max():.3f}], "
                  f"known={n_known}/{n_total} ({100*n_known/n_total:.0f}%)")

    print(f"\n{'=' * 70}")
    print("All configs tested successfully.")
    print(f"Next: run baseline CNN training to compare these configs.")
