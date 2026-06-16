# Feature: Azimuth-weighted Plane-KNN

**Source kernel**: aliafzal9323/rogii-typewell-gr-alignment-pf-gbdt-stack

## What it adds vs LB7.776 plane-KNN
The base `FormationPlaneKNN` only weights by Euclidean (X,Y) distance. This variant **also weights by drilling-azimuth similarity**:

```python
def _well_azimuth(x, y):
    """Median drilling-direction azimuth (radians)."""
    dx = np.diff(x); dy = np.diff(y)
    return float(np.arctan2(np.nansum(dy), np.nansum(dx)))

# For each train well, store its azimuth in addition to median (X,Y, F1..F6).
# At query, compute query well's azimuth, then:
az_sim = 0.5 * (1.0 + np.cos(neighbor_az - query_az))     # in [0, 1]
w = inv_distance * (az_sim ** az_beta)                     # az_beta = 0.6 default
```

A neighbor drilled in nearly the same direction (cos(Δθ) ≈ 1) keeps full weight; one drilled perpendicular (cos = 0) gets weight 0.5; an antiparallel one (cos = −1) gets weight 0.

## Why it matters
- Per the competition description: "dip behaviour depends on drilling direction" — wells drilled along similar azimuths see similar structural picture.
- Wells drilled across-strike vs along-strike would see different formation geometry; mixing them in a plane fit dilutes the local signal.
- This is a **principled domain-knowledge-informed reweighting** — should compose cleanly on top of the base kernel without changing other features.

## Score-relevant constants
| name | value |
|------|-------|
| FORMATION_K | 10 (same as base) |
| FORMATION_AZ_BETA | 0.6 (0=ignore azimuth, 1=heavy) |
| Azimuth definition | `arctan2(sum(dy), sum(dx))` over the well trajectory |

## Cross-refs
- feature_engineering/plane_knn_formation.md (the base un-weighted version)