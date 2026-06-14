# Kernel: medali1992/rogii-tcn-train-with-ddp-layernorm-se-atten

**Author**: medali1992 (deep-learning specialist)
**Last run**: ~2026-05-17
**Total votes**: ~78 (paired with rogii-tcn-infer)
**Files**: rogii-tcn-train-with-ddp-layernorm-se-atten.ipynb (training); paired infer kernel

## Architecture (one-paragraph)
A serious sequence model: per-well 1D dilated TCN (6 residual blocks, dilations 1/2/4/8/16/32 → receptive field ~248 ft) with **LayerNorm** (replaces BatchNorm for variable-length-padded sequences), **Squeeze-and-Excitation channel attention** in each block, **typewell cross-attention** (TCN hidden states query a small typewell-GR-encoded keys/values, 4 heads, 2× FFN), and a **global Transformer self-attention** stage over a stride-50 average-pooled view (so a 10,000-row well becomes 200 super-positions giving a tractable 200×200 attention matrix). Sinusoidal positional encodings, pre-norm, key padding masks throughout. Trained with `CosineAnnealingWarmRestarts(T_0=10)`, 5-fold GroupKFold, DDP for multi-GPU.

## Key Techniques

### Feature Engineering
- 7 logarithmically-spaced GR rolling windows
- 5 GR diff lags
- GR-to-typewell inverse lookup features at 150 ft and 300 ft search windows
- Same plane-KNN formation imputer (forced unconditionally for both train+test to avoid distribution shift)

### Model & Hyperparams
- HIDDEN_SIZE = 128 (cells); NUM_BLOCKS = 6 (TCN)
- TW_NUM_HEADS = 4 (cross-attention)
- KERNEL_SIZE = 5; dilations = [1, 2, 4, 8, 16, 32]
- LEARNING_RATE = 2e-3; EPOCHS = 60; BATCH_SIZE = 16
- DROPOUT = 0.10; GRAD_CLIP = 1.0
- Cosine annealing with warm restarts T_0=10
- DDP + grad accumulation; LayerNorm1d wrapper for Conv1d shape

### Particle Filter / Beam Search
- None (pure NN).

### Ensemble / Blending
- Final inference averages 5 fold checkpoints.

### CV Methodology
- GroupKFold(n_splits=5) on well_id
- Target: residual over `last_known_TVT` (matches LB7.776 convention)
- Loss: masked MSE only on hidden eval rows

## Anything novel vs LB-7.776 kernel?
**YES — different track entirely.** This is a deep sequence model, not the GBDT stack. Notable novelties:
1. **Typewell cross-attention** — TCN hidden states query an embedded typewell GR sequence (replaces hand-crafted anchored offsets)
2. **Global Transformer over downsampled view** — captures long-range geological context the TCN's receptive field cannot
3. **LayerNorm in Conv1d** correctly handles padded variable-length sequences (BatchNorm leaks padded statistics)
4. **Squeeze-and-Excitation in residual blocks** — channel-wise re-weighting per well
5. The notebook documents reducing CV-LB gap from 6.32 → 0.27 by fixing 3 train/test inconsistencies — invaluable lessons for our R8 distribution-shift concern.

## Score-relevant constants extracted
| name | value |
|------|-------|
| HIDDEN_SIZE | 128 |
| NUM_BLOCKS | 6 |
| KERNEL_SIZE | 5 |
| Dilations | 1,2,4,8,16,32 |
| LR | 2e-3 |
| EPOCHS | 60 |
| BATCH_SIZE | 16 |
| TW_NUM_HEADS | 4 |
| Global pool stride | 50 |
| LR schedule | CosineAnnealingWarmRestarts T_0=10 |

## Cross-refs
- preprocessing/layernorm_for_padded_seq.md
- preprocessing/se_attention_block.md
- model_params/tcn_medali1992.json