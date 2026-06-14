# Feature: Multi-scale NCC + score-weighted ensemble

**Source kernel**: lightningv08/lb-7-776-rogii-ridge-sp

## What it does
For each window half-size hw ∈ {8, 15, 25} (so windows 17/31/51), with stride 3:

1. Smooth the typewell GR (`kgr`) and horizontal GR (`hgr`) with rolling-5 mean.
2. Build all `sts = arange(0, nk-win+1, stride)` template windows from the typewell.
3. Z-normalize each template `Cn = (C − C.mean) / C.std`.
4. For each horizontal row i: compute NCC `Hn[i] @ Cn.T / win` against all templates; pick the best by argmax.
5. Output: `tvt_match = ktvt[best + hw]` (clipped) and `score = ncc_max`.

Then the **score-weighted ensemble**:
```
sw = exp(3.0 * scores)            # softmax temperature 3
sw /= sw.sum(axis=1)
sc_ens = (tvts * sw).sum(axis=1)
```

## Why it matters
Three scales of GR template matching capture:
- 17-row (hw=8) — fine-grained matches
- 31-row (hw=15) — typical horizontal feature size
- 51-row (hw=25) — broader stratigraphic context

Score-weighted (vs simple average) lets high-confidence matches dominate locally while remaining robust where confidence is low. The ensemble plus the 3 individual signals + scores gives 8 features per row.

## Score-relevant constants
| name | value |
|------|-------|
| half-windows | 8, 15, 25 |
| stride | 3 |
| typewell smoothing | rolling(5, center=True) |
| score softmax temperature | 3.0 |

## Cross-refs
- `anchored_gr_offsets.md` — uses `sc_ens` as one of 4 anchor families