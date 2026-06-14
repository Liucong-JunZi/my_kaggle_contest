# Feature: Sakoe-Chiba constrained DTW alignment

**Source kernel**: nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based (LB 9.251)

## What it does (NEW SIGNAL FAMILY beyond LB7.776 stack)
Adds Dynamic Time Warping signals to the per-well feature matrix as an additional alignment family. Two variants:

### Multi-scale DTW (deterministic)
Align the **full horizontal-well GR** against the **typewell GR** under a Sakoe-Chiba band of radius r. For radii `r ∈ {20, 50, 100, 200}`:
1. Z-normalize both sequences.
2. Run constrained DTW (`_dtw_sakoe_chiba`) — diagonal + radius envelope, squared-error local cost.
3. Backtrack the warping path; for each query row i, find its corresponding typewell index → `tvt_pred[i] = tw_tvt[j_for_i[i]]`.
4. Compute local **path slope** `dj/di` smoothed over a 5-row window — captures TVT rate.

Cost-weighted ensemble: `weights ∝ 1/(cost + 1e-6)`, then `dtw_ens = sum(weights * tvt_preds)`.

### Stochastic DTW (uncertainty quantification)
Sample K=12 noisy realizations by adding Gumbel noise (temperature=3.0) to the cost matrix:
```python
noise = -temperature * log(-log(uniform(1e-10, 1.0)))
cost = D_base[i, j] + noise
```
For each realization, traceback gives a different path. Outputs:
- `mean_tvt = realizations.mean(axis=0)`
- `std_tvt  = realizations.std(axis=0)`
- `cv_tvt = std / |mean|`

The std/cv quantify how confident DTW is at each row.

## Why it matters
DTW finds **global** GR alignment whereas multi-scale NCC only finds best **local** windows. The two approaches are complementary, especially in regions with structural drift or repeated GR patterns. Per the kernel note, this is the primary innovation beyond the LB7.776 stack.

## Score-relevant constants
| name | value |
|------|-------|
| Sakoe-Chiba radii | 20, 50, 100, 200 |
| Stochastic K (realizations) | 12 |
| Stochastic temperature | 3.0 |
| Path slope smooth window | 5 |
| Cost-weight epsilon | 1e-6 |

## Outputs added to feature matrix
- `dtw_ens_d` (cost-weighted ensemble TVT − last_known_tvt)
- `dtw_mean_d`, `dtw_std`, `dtw_cv` (stochastic realization aggregates)
- per-radius slopes `dtw_slope_{r}_ev` for r ∈ {20,50,100,200}
- aggregated `dtw_slope_mean_ev`
- DTW cost stats `dtw_cost_min`, `dtw_cost_range`
- Anchored offsets `tdtw{o}` for o ∈ DTW_OFFS = `[-20,-10,-5,-2,0,2,5,10,20]` (9 features)

## Implementation note
JIT-compiled with numba for tractable speed; ~minutes per well at radius 200, full horizontal length × full typewell length.

## Cross-refs
- `multi_scale_ncc.md` — local matching counterpart
- `kernels/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.md`