# Feature: Anchored GR offsets

**Source kernels**: lightningv08/lb-7-776-rogii-ridge-sp; nihilisticneuralnet/9-251 DWT-based

## What it does
For each candidate "anchor TVT" (last_known_TVT, beam_ref, sc_ens, pf_use, etc.), compute differences between the horizontal-well GR and the typewell GR sampled at `anchor + offset` for a grid of TVT offsets:

```python
ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS   = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)
PF_OFFS   = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)

# Per-row features
{f'tda{o}' : hgr − tw_gr(last_known_tvt + o) for o in ANCH_OFFS}      # 11 features
{f'tdbc{o}': hgr − tw_gr(beam_ref + o)        for o in BEAM_OFFS}     # 11
{f'tdsc{o}': hgr − tw_gr(sc_ens   + o)        for o in SC_OFFS}       # 11
{f'tdpf{o}': hgr − tw_gr(pf_use   + o)        for o in PF_OFFS}       # 11
```

## Why it matters
At each row the model sees how the horizontal GR aligns with multiple candidate TVT positions. This essentially feeds the GBDT a **scoring profile** instead of a single matched score, letting it learn nonlinear "soft argmin" behavior. The 4 anchor families give 44 features per row total — a substantial fraction of the LB7.776 stack.

## Variants seen
- LB-7.776 uses 4 anchor families, 11 offsets each (44 features).
- DWT-based kernel adds a 5th `DTW_OFFS = [-20,-10,-5,-2,0,2,5,10,20]` (9 offsets) anchored on `dtw_ens`.

## Cross-refs
- `multi_scale_ncc.md` — produces `sc_ens` anchor