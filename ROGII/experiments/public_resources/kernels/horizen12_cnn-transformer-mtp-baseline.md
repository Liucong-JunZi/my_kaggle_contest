# Kernel: horizen12/cnn-transformer-mtp-baseline

**Author**: horizen12
**Files**: cnn-transformer-mtp-baseline.ipynb

## Architecture (one-paragraph)
A **CNN-Transformer hybrid for multi-trajectory prediction**: extends hengck23/cnn-mtp-example by replacing the flat MLP head with a Transformer encoder. The pipeline operates on per-well GR-difference heatmaps split into 4D tensors `(batch, time, bins, channels)`. A 2-layer 2D CNN extracts (batch, time, hidden) sequence features (mean-pooled across bins), then a 1-layer (or n-layer) Transformer encoder with positional embeddings + key padding mask refines the sequence. A multi-modal head outputs K candidate trajectories + K logits.

## Key Techniques

### Architecture details
- HeatmapCNNEncoder: 2× Conv2d(48 hidden, kernel=3) + BatchNorm + SiLU + Dropout2d(0.10), mean-pool over the `bins` axis to produce per-time hidden features, then linear projection + LayerNorm to 64-dim.
- TrajectoryTransformer: 1 layer (configurable), 4 heads, dim_feedforward=128 (2× hidden), norm_first=True (pre-norm), positional embedding parameter, key-padding mask.
- HeatmapMTPModel: CNN encoder → Transformer → multi-mode head (K trajectories + K logits).

### Multi-trajectory loss
Same MTP formulation as hengck23: winner-takes-all regression + cross-entropy on the selected trajectory.

### Hyperparams
- in_channels: variable (depends on heatmap construction)
- heatmap_bins: HEATMAP_BINS (configurable)
- chunk_len: CHUNK_LEN (sequence length for transformer attention)
- k_modes: K_MODES (number of candidate trajectories)
- cnn_hidden: 48
- transformer_hidden: 64
- n_heads: 4
- n_layers: 1
- dropout: 0.10

## Anything novel vs LB-7.776 kernel?
**YES — extends MTP (hengck23) with a Transformer**. This is a small-architecture sequence-to-trajectories model that's complementary to:
- The PF/beam/NCC physical signals (those are sequential point estimates).
- The TCN+CrossAttention+GlobalTransformer model (medali1992) (heavier, sequence-only).

The advantage of CNN-Transformer-MTP over TCN: explicit per-row K-mode multimodality (vs. single regression).

## Cross-refs
- kernels/hengck23_cnn-mtp-example.md (the original MTP idea)
- kernels/medali1992_rogii-tcn-train-with-ddp-layernorm-se-atten.md (TCN counterpart)
- feature_engineering/typewell_cross_attention.md