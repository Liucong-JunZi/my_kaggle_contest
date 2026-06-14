# Kernel: medali1992/rogii-tcn-train-with-ddp-layernorm-se-atten (Deep Learning)

**Author**: medali1992 | **Votes**: ~78 (paired with infer kernel)
**File**: rogii-tcn-train-with-ddp-layernorm-se-atten.ipynb

## Architecture

A **deep learning alternative** to the GBDT/PF approach. Not directly comparable to our R10 track but worth noting for future hybrid integration.

```
Input features (per well):
  - 7 log-spaced GR rolling windows
  - 5 GR diff lags
  - GR-to-typewell inverse lookup at 150ft and 300ft
  - Plane-KNN formation imputation (forced unconditional)

                    ↓
┌─────────────────────────────┐
│  1D Dilated TCN (6 blocks)  │  dilations: 1,2,4,8,16,32
│  - Kernel size: 5           │  receptive field ~248 ft
│  - LayerNorm (not BatchNorm)│  ← critical for padded sequences
│  - SE channel attention      │  squeeze-and-excitation per block
│  - Hidden size: 128         │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│  Typewell Cross-Attention    │  TCN states query typewell GR
│  - 4 heads                   │  keys/values from encoded typewell
│  - 2× FFN                    │  learns GR correlation patterns
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│  Global Transformer          │
│  - Stride-50 avg pool        │  10000 rows → 200 super-positions
│  - Self-attention over 200   │  captures long-range (>248 ft) context
│  - Sinusoidal pos encoding   │
└──────────┬──────────────────┘
           ↓
      Output: TVT offset
```

## Key Innovations (Novel vs LB 7.776)

### 1. Typewell Cross-Attention
Instead of hand-crafted anchored GR offsets (44 engineered features in LB 7.776), this model learns the typewell-to-horizontal GR mapping via cross-attention. **Potentially more powerful** because:
- Captures nonlinear correlations (not just pointwise differences)
- Can attend to multiple relevant typewell regions simultaneously
- Learns which GR patterns are discriminative for depth matching

### 2. LayerNorm for Padded Sequences
BatchNorm leaks padded-position statistics → train/test mismatch. LayerNorm operates per-timestep, making it correct for variable-length sequences.

### 3. CV-LB Gap Analysis (INVALUABLE LESSON)

> "The notebook documents reducing CV-LB gap from 6.32 → 0.27 ft by fixing 3 train/test inconsistencies."

This is a critical lesson for us. The three fixes (from the medali1992 post):
1. **Force plane-KNN imputation on train AND test unconditionally** — using real formation cols in train but imputed in test creates a distribution shift
2. **Match GR interpolation method** between train and test pipelines
3. **Consistent NaN handling** for missing formation columns

**Our R10 should audit: Do we have any train/test inconsistencies?**

### 4. Training Details
```python
HIDDEN_SIZE = 128
NUM_BLOCKS = 6
KERNEL_SIZE = 5
DILATIONS = [1, 2, 4, 8, 16, 32]  # receptive field ~248
LR = 2e-3
EPOCHS = 60
BATCH_SIZE = 16
DROPOUT = 0.10
GRAD_CLIP = 1.0
LR_SCHEDULE = CosineAnnealingWarmRestarts(T_0=10)
```

## Actionable Takeaway for R10

### Immediate: Audit train/test consistency
Check that our `features_full.parquet` generation pipeline uses the same code path for train and test. Specifically:
- Is PF imputation identical? (any self-well masking difference?)
- Is GR interpolation identical?
- Are formation column handling identical?

### Medium-term: Typewell cross-attention as additional feature
Could extract cross-attention alignment weights and feed them as features into LGB/CB. This is a "best of both worlds" hybrid — deep model provides alignment signal, GBDT handles the rest.

### Note: Not a priority
The TCN alone (without PF/beam signals) likely scores ~10-12 LB. Our current GBDT+PF approach with 9.182 OOF is already stronger. The cross-attention concept is the only novel insight worth cherry-picking.
