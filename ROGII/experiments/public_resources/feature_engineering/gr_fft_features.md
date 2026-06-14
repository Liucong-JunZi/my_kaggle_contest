# Feature: GR FFT features

**Source kernel**: sanidhyavijay24/9-946-rogii-geostat-softmax-ncc-hybrid

## What it does
Captures the dominant frequency of the post-anchor GR signal:

```python
def gr_fft_features(gr_post):
    valid = gr_post[~np.isnan(gr_post)]
    if len(valid) < 32: return 0.0, 0.0
    centered = valid - valid.mean()
    spec = np.abs(np.fft.rfft(centered)) ** 2
    dom_idx = np.argmax(spec[1:]) + 1   # skip DC
    return (
        dom_idx / len(valid),    # normalised dominant frequency
        log1p(spec[dom_idx]),    # log power of dominant component
    )
```

## Why it matters
Encodes the **stratigraphic periodicity** of the post-anchor GR — useful for distinguishing wells with strong layered structure from those with chaotic GR. The dominant frequency loosely correlates with the formation thickness regime; the log power flags how confident we are about there being any periodic structure.

Cheap, single-feature-pair addition (2 features per well), useful as a global covariate.

## Score-relevant constants
| name | value |
|------|-------|
| Min valid points | 32 |
| FFT type | real, single-sided |

## Cross-refs
- feature_engineering/gr_detrend_resid.md (alternative spectral preprocessing)