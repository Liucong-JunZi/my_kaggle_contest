# Kernel: fleongg/fle3n-rogii-v5 (LB 7.540 → 7.528 with hedge)

**Author**: fleongg
**Last run**: ~2026-06
**Total votes**: ~80 (paired with v4 which had 71 votes)
**Files**: fle3n-rogii-v5.ipynb

## Architecture (one-paragraph)
The strongest LB-documented public kernel: **LB 7.540 from the A⊕B 0.55/0.45 blend, LB 7.528 with a gated interpretation hedge**. Two parallel pipelines:
- **Engine A "Ridge-SP"** (LB 7.776): tracker selector + GBM/Ridge stack + robust polynomial U-space projection (≡ pixiux's pipeline A).
- **Engine B "Drift-PF"** (LB 7.810): 128-seed likelihood-weighted PF + GBM stack + warm-up/SG smoothing (≡ Branch B in pixiux).

Then `0.55*A + 0.45*B` produces LB 7.540. Finally, the **interpretation hedge** (LB-measured-optimum 0.5) for gate-verified duplicate wells lifts to LB 7.528.

## Key Techniques

### Interpretation hedge (the LB 7.528 trick)
- The competition originally had a duplicate-well leak (some test wells were identical to train wells). Organizers killed it (msg 707695, see `docs/forum-snapshots/2026-06-12/`).
- Even after the kill, fleongg measured that mixing the train copy's TVT into the blend at weight 0.5 gives a tiny LB lift (-0.012 RMSE) — likely from residual coordinate-level matching.
- Per the kernel: "the corpse was turned into a measured, parabola-optimal hedge that beats every other variant."

### Other techniques
- All Engine A techniques per LB7.776
- Engine B uses 128-seed PF (likelihood-weighted)
- Same GBM stack on both
- Robust polynomial U-space projection (Engine A)
- SG smoothing (Engine B)

## Score-relevant constants
| name | value |
|------|-------|
| Engine blend (A/B) | 0.55 / 0.45 |
| Interpretation hedge weight | 0.5 (LB-measured optimum) |
| Engine A LB | 7.776 |
| Engine B LB | 7.810 |
| Pure blend LB | 7.540 |
| With hedge LB | 7.528 |

## Lessons
- The two-engine blend `0.55·A + 0.45·B` is the single biggest "free" lift on top of LB7.776 (-0.236 RMSE).
- Even after the duplicate-well leak was killed, a parabola-optimal hedge weight (0.5 from LB grid search) provides a small additional gain.
- The fleongg kernel is the **closest public proxy to top-3 LB performance** as of 2026-06.

## Cross-refs
- kernels/pixiux_rogii-dual-pipeline-blend.md (parallel implementation of the two-engine pattern)
- preprocessing/u_space_projection.md (Engine A's projection step)
- preprocessing/guarded_physical_override.md (the safe variant of the hedge)
- ensemble_weights/ridge_pp_smooth.md (Engine A's stack)