## CV vs LB consistency tricks (medali1992 TCN)

**Source kernel**: medali1992/rogii-tcn-train-with-ddp-layernorm-se-atten

The kernel notes a **6.32 → 0.27** ft CV/LB gap reduction — invaluable for our distribution-shift problem.

### Three fixes
1. **Force the spatial imputer unconditionally for `b_well` computation** — at training time, even when self-well formation columns are present, IMPUTE them via `FormationPlaneKNN` so the b_well distribution is identical to test. Otherwise training b_well uses real values, test uses imputed → distribution shift kills LB.
2. **Sequence padding mask in global Transformer** — without it, padded super-positions pollute attention; real positions get spurious weight from zeros. Add `seq_key_padding_mask` and pass to `nn.MultiheadAttention`.
3. **Per-fold seeding** — without it, fold-to-fold variance contaminates the OOF stack. `torch.manual_seed(SEED + fold)` resolves this.
4. **ResidualBlock conv2 bug** — original had `nn.Conv1d(channels, kernel_size, kernel_size)` (out_channels=5) instead of `nn.Conv1d(channels, channels, kernel_size)`. Bottlenecks all 96 channels through 5 dimensions silently.

### Implication for our pipeline
Our R8 hybrid SDF→LGB stack uses real per-formation columns at training time; test only has X/Y. We should explicitly impute formation columns at training time (drop the real ones, plug `FormationPlaneKNN` in) to match the test distribution. This may close some of our CV/LB ~2-4 ft gap.

### Cross-refs
- feature_engineering/plane_knn_formation.md
- preprocessing/layernorm_for_padded_seq.md