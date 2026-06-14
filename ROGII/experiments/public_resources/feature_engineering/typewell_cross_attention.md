# Architecture: Typewell cross-attention + Global Transformer

**Source kernel**: medali1992/rogii-tcn-train-with-ddp-layernorm-se-atten

## What it does
Instead of hand-crafted anchored GR offset features (LB7.776), use **learned cross-attention** between TCN hidden states and the typewell GR sequence:

1. **Typewell encoding**: normalised (TVT, GR) typewell arrays projected to d_model=96 via linear layer + LayerNorm.
2. **Cross-attention**: TCN hidden states (seq_len, 128) → query; typewell encodings (tw_len, 96) → keys + values. Multi-head attention (h=4). This is the learnable analog of the multi-scale NCC + anchored offsets.
3. **Post-attention FFN**: 128 → 256 → 128 with LayerNorm residuals.
4. **Global Transformer** over a downsampled view of the well: stride-50 average pooling → 200 super-positions per well → sinusoidal positional encoding → 2-layer Transformer encoder (4 heads, pre-norm, 2× FFN expansion) → linear upsample back → residual merge with TCN features.

## Why it's novel
- Replaces 44+ hard-coded anchored offset features with a **differentiable, learned matching** mechanism.
- The typewell cross-attention can attend to typewell positions consistent with **all** hidden state dimensions simultaneously (trajectory, GR pattern, physics), rather than just the raw GR scalar at discrete anchor offsets.
- The global Transformer provides long-range context the TCN's 248-ft receptive field cannot reach.

## Architecture diagram
```
Input features (seq_len, n_features)
    │
TCN (6× ResidualBlock, dilations 1/2/4/8/16/32)
    │ → hidden (seq_len, 128)
    │
TypewellCrossAttention ── queries from TCN, keys/values from typewell
    │ → (seq_len, 128) with residual
    │
GlobalTransformerEncoder ── stride-50 pool → 2× Transformer → upsample
    │ → (seq_len, 128) with residual
    │
Head: Conv1d(128→64) → SiLU → Conv1d(64→1) → residual prediction
```

## Score-relevant constants
| name | value |
|------|-------|
| TCN hidden size | 128 |
| Cross-attn heads | 4 |
| Cross-attn d_model | 96 (TW_NUM_HEADS * 24) |
| Global pool stride | 50 |
| Global attn layers | 2 |
| Global attn heads | 4 |
| Global FFN expansion | 2× |

## Cross-refs
- kernels/medali1992_rogii-tcn-train-with-ddp-layernorm-se-atten.md
- model_params/tcn_medali1992.json
- feature_engineering/anchored_gr_offsets.md (the hand-crafted counterpart)