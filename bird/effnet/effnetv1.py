#!/usr/bin/env python3
"""
EfficientNet B3 - BirdCLEF 2026 Training
从缓存频谱图训练 EfficientNet，支持 GroupKFold OOF、AUC/Loss 早停、MixUp 增强
"""

import gc
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import timm
from torchvision.transforms import functional as TF

warnings.filterwarnings("ignore")


class config:
    def __init__(self):
        self.BASE = Path("/kaggle/input/competitions/birdclef-2026")
        self.WORK_DIR = Path("/kaggle/working")

        self.SPEC_ORIG_PATH = Path(
            "/kaggle/input/full-perch-spectrograms-fp16/full_perch_spectrograms_fp16.npz"
        )
        self.SPEC_MIXUP_PATH = Path(
            "/kaggle/input/mixupcache/mixup_perch_spectrograms.npz"
        )
        self.MIXUP_LABELS_PATH = Path(
            "/kaggle/input/mixupcache/train_mixup_labels.csv"
        )

        self.EFFNET_PRETRAINED = Path(
            "/kaggle/input/models/timm/tf-efficientnet/pytorch/tf-efficientnet-b3/1"
        )

        self.N_WINDOWS = 12
        self.N_CLASSES = 234

        self.SMOKE_TEST = False
        self.SMOKE_MAX_FILES = 50
        self.MAX_MIXUP = 5000
        self.USE_MIXUP = True
        self.RANDOM_SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.effnet_config = {
            "model_name": "tf_efficientnet_b3",
            "image_size": 192,
            "n_epochs": 3 if self.SMOKE_TEST else 100,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "patience": 3 if self.SMOKE_TEST else 10,
            "dropout": 0.2,
            "oof_n_splits": 2 if self.SMOKE_TEST else 5,
            "scheduler_T0": 10,
            "scheduler_T_mult": 1,
            "scheduler_eta_min": 1e-6,
            "mixup_alpha": 0.0 if self.SMOKE_TEST else 0.2,
        }


def safe_auc_score(y_true, y_score):
    if y_true.ndim == 1:
        return roc_auc_score(y_true, y_score)
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.sum() == 0 or (1 - yt).sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(yt, y_score[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.5


def file_level_auc(y_true, y_score, n_windows=12):
    n_files = y_true.shape[0] // n_windows
    file_true = y_true.reshape(n_files, n_windows, -1).max(axis=1)
    file_pred = y_score.reshape(n_files, n_windows, -1).max(axis=1)
    return safe_auc_score(file_true, file_pred)


class BirdDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.PRIMARY_LABELS = None
        self.label_to_idx = None
        self.spec_original = None
        self.spec_mixup = None
        self.Y_original = None
        self.Y_mixup = None
        self.spec_all = None
        self.Y_all = None
        self.file_groups = None

    def load_labels(self):
        print("加载标签数据...")
        sample_sub = pd.read_csv(self.cfg.BASE / "sample_submission.csv")
        self.PRIMARY_LABELS = sample_sub.columns[1:].tolist()
        self.cfg.N_CLASSES = len(self.PRIMARY_LABELS)
        self.label_to_idx = {c: i for i, c in enumerate(self.PRIMARY_LABELS)}

        soundscape_labels = pd.read_csv(
            self.cfg.BASE / "train_soundscapes_labels.csv"
        )
        soundscape_labels["primary_label"] = soundscape_labels[
            "primary_label"
        ].astype(str)

        def parse_soundscape_labels(x):
            if pd.isna(x):
                return []
            return [t.strip() for t in str(x).split(";") if t.strip()]

        def union_labels(series):
            return sorted(
                set(lbl for x in series for lbl in parse_soundscape_labels(x))
            )

        sc_clean = (
            soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
            .apply(union_labels)
            .reset_index(name="label_list")
        )
        sc_clean["end_sec"] = (
            pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
        )
        sc_clean["row_id"] = sc_clean["filename"].str.replace(
            ".ogg", "", regex=False
        ) + "_" + sc_clean["end_sec"].astype(str)

        windows_per_file = sc_clean.groupby("filename").size()
        full_files = sorted(
            windows_per_file[
                windows_per_file == self.cfg.N_WINDOWS
            ].index.tolist()
        )
        sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

        Y_SC = np.zeros((len(sc_clean), self.cfg.N_CLASSES), dtype=np.uint8)
        for i, labels in enumerate(sc_clean["label_list"].values):
            idxs = [
                self.label_to_idx[lbl]
                for lbl in labels
                if lbl in self.label_to_idx
            ]
            if idxs:
                Y_SC[i, idxs] = 1

        self.sc_clean = sc_clean
        self.Y_SC = Y_SC

        self._full_truth = (
            sc_clean[sc_clean["file_fully_labeled"]]
            .sort_values(["filename", "end_sec"])
            .reset_index(drop=False)
        )
        print(f"  sc_clean: {sc_clean.shape}, Y_SC: {Y_SC.shape}")

    def load_spectrograms(self):
        print("\n加载频谱图缓存...")
        if not self.cfg.SPEC_ORIG_PATH.exists():
            print(f"  频谱图不存在: {self.cfg.SPEC_ORIG_PATH}")
            return

        spec_data = np.load(self.cfg.SPEC_ORIG_PATH)
        self.spec_original = spec_data["spec_full"].astype(np.float32)
        print(f"  原始频谱图: {self.spec_original.shape}")

        n_orig = len(self.spec_original)
        if getattr(self.cfg, "SMOKE_TEST", False):
            max_samples = self.cfg.SMOKE_MAX_FILES * self.cfg.N_WINDOWS
            self.spec_original = self.spec_original[:max_samples]
            n_orig = len(self.spec_original)
            print(f"  [SMOKE] 截断到 {self.cfg.SMOKE_MAX_FILES} 文件 ({n_orig} windows)")

        full_truth = self._full_truth
        row_ids = full_truth["row_id"].values[:n_orig]
        meta_sorted = self.sc_clean.set_index("row_id")
        aligned = meta_sorted.loc[row_ids].reset_index(drop=True)
        self.Y_original = self.Y_SC[
            aligned["index"].values if "index" in aligned.columns
            else full_truth["index"].values[:n_orig]
        ]
        print(f"  原始标签: {self.Y_original.shape}")

        if self.cfg.USE_MIXUP and self.cfg.SPEC_MIXUP_PATH.exists():
            print("\n加载 MixUp 频谱图...")
            mixup_spec_data = np.load(self.cfg.SPEC_MIXUP_PATH)
            self.spec_mixup = mixup_spec_data["spec_mixup"].astype(np.float16)
            print(f"  MixUp 频谱图: {self.spec_mixup.shape}")

            if self.cfg.MIXUP_LABELS_PATH.exists():
                mixup_labels = pd.read_csv(self.cfg.MIXUP_LABELS_PATH)
                mixup_labels["primary_label"] = mixup_labels[
                    "primary_label"
                ].astype(str)

                def parse_labels(x):
                    if pd.isna(x):
                        return []
                    return [t.strip() for t in str(x).split(";") if t.strip()]

                self.Y_mixup = np.zeros(
                    (len(mixup_labels), self.cfg.N_CLASSES), dtype=np.uint8
                )
                for i, label_str in enumerate(mixup_labels["primary_label"].values):
                    labels = parse_labels(label_str)
                    idxs = [
                        self.label_to_idx[lbl]
                        for lbl in labels
                        if lbl in self.label_to_idx
                    ]
                    if idxs:
                        self.Y_mixup[i, idxs] = 1
                print(f"  MixUp 标签: {self.Y_mixup.shape}")
            else:
                self.spec_mixup = None
                self.Y_mixup = None

        if self.spec_mixup is not None and self.Y_mixup is not None:
            max_mixup = self.cfg.MAX_MIXUP
            n_use = min(len(self.spec_mixup), len(self.Y_mixup), max_mixup)
            print(f"  使用 MixUp 样本: {n_use}/{len(self.spec_mixup)}")
            self.spec_all = np.concatenate(
                [self.spec_original, self.spec_mixup[:n_use].astype(np.float32)], axis=0
            )
            self.Y_all = np.concatenate(
                [self.Y_original, self.Y_mixup[:n_use]], axis=0
            )
            n_orig_files = len(self.spec_original) // self.cfg.N_WINDOWS
            orig_groups = np.repeat(np.arange(n_orig_files), self.cfg.N_WINDOWS)
            mixup_groups = np.arange(n_orig_files, n_orig_files + n_use)
            self.file_groups = np.concatenate([orig_groups, mixup_groups])
            print(f"\n合并后: {self.spec_all.shape}, 标签: {self.Y_all.shape}")
        else:
            self.spec_all = self.spec_original
            self.Y_all = self.Y_original
            n_files = len(self.spec_original) // self.cfg.N_WINDOWS
            self.file_groups = np.repeat(np.arange(n_files), self.cfg.N_WINDOWS)
            print(f"\n仅原始数据: {self.spec_all.shape}")

    def run_all(self):
        self.load_labels()
        self.load_spectrograms()


class EfficientNetBird(nn.Module):
    def __init__(
        self,
        model_name="tf_efficientnet_b3",
        n_classes=234,
        image_size=192,
        dropout=0.2,
        checkpoint_path=None,
    ):
        super().__init__()
        self.image_size = image_size

        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, in_chans=3,
        )
        self.feature_dim = self.backbone.num_features
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"  预训练权重已加载 (missing={len(missing)}, unexpected={len(unexpected)})")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, n_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = self._spec_to_img(x)
        features = self.backbone(x)
        return self.classifier(features)

    def _spec_to_img(self, spec):
        s_min = spec.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
        s_max = spec.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
        normed = (spec - s_min) / (s_max - s_min + 1e-8)
        img = normed.unsqueeze(1).expand(-1, 3, -1, -1)
        img = TF.resize(img, [self.image_size, self.image_size])
        return img


class Trainer:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data

    def _create_model(self):
        pretrained_dir = self.cfg.EFFNET_PRETRAINED
        ckpt_path = None
        if pretrained_dir.exists():
            for f in sorted(pretrained_dir.iterdir()):
                if f.suffix in (".pth", ".bin", ".pt") and f.is_file():
                    ckpt_path = str(f)
                    break
            if ckpt_path:
                print(f"  加载预训练权重: {ckpt_path}")

        model = EfficientNetBird(
            model_name=self.cfg.effnet_config["model_name"],
            n_classes=self.cfg.N_CLASSES,
            image_size=self.cfg.effnet_config["image_size"],
            dropout=self.cfg.effnet_config["dropout"],
            checkpoint_path=ckpt_path,
        )
        model.to(self.cfg.DEVICE)
        return model

    def _mixup_data(self, specs, labels, alpha=0.4):
        if alpha <= 0:
            return specs, labels
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(len(specs)).to(specs.device)
        mixed_specs = lam * specs + (1 - lam) * specs[idx]
        mixed_labels = lam * labels + (1 - lam) * labels[idx]
        return mixed_specs, mixed_labels

    def _train_one_epoch(self, model, loader, optimizer, device, mixup_alpha=0.4):
        model.train()
        total_loss = 0.0
        for batch_spec, batch_labels in loader:
            batch_spec = batch_spec.to(device)
            batch_labels = batch_labels.to(device)
            if mixup_alpha > 0:
                batch_spec, batch_labels = self._mixup_data(
                    batch_spec, batch_labels, mixup_alpha
                )
            optimizer.zero_grad()
            logits = model(batch_spec)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _validate(self, model, loader, device):
        model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_spec, batch_labels in loader:
                batch_spec = batch_spec.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_spec)
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(batch_labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        val_auc = safe_auc_score(targets, preds)
        return total_loss / max(len(loader), 1), val_auc, preds, targets

    def train_oof(self):
        print("\n" + "=" * 60)
        print("开始 EfficientNet OOF 训练")
        print("=" * 60)

        spec_orig = self.data.spec_original
        labels_orig = self.data.Y_original
        n_orig = len(spec_orig)

        spec_mixup = self.data.spec_mixup
        labels_mixup = self.data.Y_mixup

        if spec_orig is None or labels_orig is None:
            print("  数据不存在，跳过")
            return None

        n_files = n_orig // self.cfg.N_WINDOWS
        orig_groups = np.repeat(np.arange(n_files), self.cfg.N_WINDOWS)

        ec = self.cfg.effnet_config
        n_splits = ec["oof_n_splits"]
        n_epochs = ec["n_epochs"]
        batch_size = ec["batch_size"]
        patience = ec["patience"]
        device = self.cfg.DEVICE
        mixup_alpha = ec.get("mixup_alpha", 0.4)

        oof_preds = np.zeros((n_orig, self.cfg.N_CLASSES), dtype=np.float32)
        gkf = GroupKFold(n_splits=n_splits)

        for fold_i, (train_idx, val_idx) in enumerate(
            gkf.split(spec_orig, groups=orig_groups)
        ):
            print(
                f"\n--- Fold {fold_i + 1}/{n_splits} "
                f"(train={len(train_idx)}, val={len(val_idx)}) ---"
            )

            train_spec = spec_orig[train_idx]
            train_labels = labels_orig[train_idx]

            if self.cfg.USE_MIXUP and spec_mixup is not None and labels_mixup is not None:
                max_mixup = self.cfg.MAX_MIXUP
                n_use = min(len(spec_mixup), len(labels_mixup), max_mixup)
                train_spec = np.concatenate(
                    [train_spec, spec_mixup[:n_use].astype(np.float32)], axis=0
                )
                train_labels = np.concatenate(
                    [train_labels, labels_mixup[:n_use]], axis=0
                )
                print(f"  训练集含 MixUp: {len(train_spec)} samples")

            train_ds = TensorDataset(
                torch.tensor(train_spec, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32),
            )
            val_ds = TensorDataset(
                torch.tensor(spec_orig[val_idx], dtype=torch.float32),
                torch.tensor(labels_orig[val_idx], dtype=torch.float32),
            )
            train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = self._create_model()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=ec["lr"], weight_decay=ec["weight_decay"]
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=ec["scheduler_T0"],
                T_mult=ec["scheduler_T_mult"],
                eta_min=ec["scheduler_eta_min"],
            )

            best_auc, best_loss, best_state, wait = 0.0, float("inf"), None, 0

            for epoch in range(n_epochs):
                train_loss = self._train_one_epoch(
                    model, train_loader, optimizer, device, mixup_alpha
                )
                scheduler.step()
                val_loss, val_auc, _, _ = self._validate(model, val_loader, device)

                if epoch % 5 == 0 or epoch == n_epochs - 1:
                    print(
                        f"  Epoch {epoch:3d}: "
                        f"train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_auc={val_auc:.4f}"
                    )

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"  Early stopping at epoch {epoch}, best_auc={best_auc:.4f}")
                        break

            if best_state:
                model.load_state_dict(best_state)

            _, _, fold_preds, _ = self._validate(model, val_loader, device)
            oof_preds[val_idx] = fold_preds
            print(f"  Fold {fold_i + 1} best_val_auc={best_auc:.4f}")

            del model, optimizer, scheduler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_orig = len(self.data.spec_original)
        orig_auc = safe_auc_score(labels_orig[:n_orig], oof_preds[:n_orig])
        orig_file_auc = file_level_auc(labels_orig[:n_orig], oof_preds[:n_orig], n_windows=self.cfg.N_WINDOWS)
        print(f"\nOOF 完成: window_auc={orig_auc:.4f}, file_auc={orig_file_auc:.4f} (仅原始数据)")

        print("\n用全量数据训练最终模型...")
        final_model = self._create_model()
        final_opt = torch.optim.AdamW(
            final_model.parameters(), lr=ec["lr"], weight_decay=ec["weight_decay"]
        )
        final_sch = CosineAnnealingWarmRestarts(
            final_opt,
            T_0=ec["scheduler_T0"],
            T_mult=ec["scheduler_T_mult"],
            eta_min=ec["scheduler_eta_min"],
        )

        all_ds = TensorDataset(
            torch.tensor(spec_orig, dtype=torch.float32),
            torch.tensor(labels_orig, dtype=torch.float32),
        )
        all_loader = TorchDataLoader(all_ds, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            train_loss = self._train_one_epoch(
                final_model, all_loader, final_opt, device, mixup_alpha
            )
            final_sch.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss={train_loss:.4f}")

        save_dir = self.cfg.WORK_DIR / "effnet_v1"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), save_dir / "effnet_v1_weights.pth")
        np.savez_compressed(save_dir / "oof_preds.npz", oof_preds=oof_preds)
        print(f"模型保存到: {save_dir}")

        self.model = final_model
        self.oof_preds = oof_preds
        return oof_preds


def main():
    print("=" * 60)
    print("EfficientNet B3 - BirdCLEF 2026 Training")
    print("=" * 60)

    cfg = config()
    print(f"Device: {cfg.DEVICE}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"timm: {timm.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = BirdDataLoader(cfg)
    data.run_all()

    trainer = Trainer(cfg, data)
    trainer.train_oof()

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
