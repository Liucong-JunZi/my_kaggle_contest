# ROGII Public Resource Harvester — Sub-Agent Task

## Mission

You are a long-running harvest agent. Your job: **systematically collect and structure all public Kaggle resources** for the `rogii-wellbore-geology-prediction` competition, extract key insights, and stage them for the parent agent to fold into the round_010 hill-climb pool.

You run **autonomously** for hours. No need to ask questions — execute the protocol below, log everything to `experiments/public_resources/HARVEST_LOG.md`, and return a final summary when you've drained the public-resource pool.

---

## Boundaries (HARD — do not cross)

- ✅ Bash, Read, Write, Edit, WebFetch, WebSearch, Grep, Glob — all allowed
- ✅ `kaggle` CLI — kernels list/pull, datasets list/download, with **3-second sleep between requests** (avoid rate limiting)
- ✅ Write to `experiments/public_resources/` and `experiments/public_harvest/` only
- ❌ DO NOT run training (no `train_one.py`, no `python orchestrator/...`)
- ❌ DO NOT submit to LB
- ❌ DO NOT modify `experiments/round_010/results/` or any candidates/* files
- ❌ DO NOT modify `experiments/round_008/` or `experiments/round_009/`
- ❌ DO NOT delete anything outside `experiments/public_resources/` and `experiments/public_harvest/`
- 📦 Dataset filter: download only datasets < 500MB (check `kaggle datasets list -p` size first); for larger ones, save **description only** to `experiments/public_resources/datasets/<slug>.md`
- 💾 Disk budget: 100 GB total. Track cumulative download size, stop downloads if exceeded.

---

## Phase 1 — Full Harvest

### 1A. Kernels

```bash
mkdir -p experiments/public_resources/kernels_raw
cd experiments/public_resources/kernels_raw

# Page through all kernels — keep going until empty page
page=1
while :; do
    out=$(kaggle kernels list -c rogii-wellbore-geology-prediction -p $page 2>&1)
    if echo "$out" | grep -q "No kernels found"; then break; fi
    echo "$out" | tail -n +3 | awk '{print $1}' > /tmp/kernels_p${page}.txt
    sleep 3
    page=$((page + 1))
    if [ $page -gt 20 ]; then break; fi  # safety
done

# Dedupe + sort all refs
cat /tmp/kernels_p*.txt | sort -u > experiments/public_resources/all_kernel_refs.txt
```

For each kernel ref:
```bash
slug_safe=$(echo "$ref" | tr '/' '_')
kaggle kernels pull "$ref" -p experiments/public_resources/kernels_raw/$slug_safe -m  # -m grabs metadata
sleep 3
```

If pull fails 3 times for a ref, log it to `HARVEST_LOG.md` and skip.

### 1B. Datasets

```bash
kaggle datasets list -s rogii-wellbore-geology-prediction --sort-by hottest 2>&1 | tee /tmp/datasets.txt
# Also try without filter — competition-related datasets may not have 'rogii' in name:
kaggle datasets list -s wellbore --sort-by hottest 2>&1 | tee -a /tmp/datasets.txt
```

For each dataset ref:
```bash
size_mb=$(kaggle datasets metadata -p /tmp $ref 2>/dev/null | grep -oE '"totalBytes":\s*[0-9]+' | awk '{print int($2/1024/1024)}')
if [ "${size_mb:-9999}" -lt 500 ]; then
    kaggle datasets download $ref -p experiments/public_resources/datasets_raw --unzip
else
    # Save description only
    kaggle datasets metadata $ref -p experiments/public_resources/datasets/ 
fi
sleep 3
```

### 1C. Forum Topics (bonus — high signal)

```bash
kaggle competitions topics-list -c rogii-wellbore-geology-prediction --csv > /tmp/topics.csv
# Pull all topic messages
for tid in $(cat /tmp/topics.csv | tail -n +2 | cut -d, -f1); do
    kaggle competitions topic-messages $tid -n -1 > experiments/public_resources/forum_${tid}.txt
    sleep 3
done
```

---

## Phase 2 — Structured Extraction

For each pulled kernel, write `experiments/public_resources/kernels/<slug_safe>.md`:

```markdown
# Kernel: <ref> (LB <score>)

**Author**: <user>
**Last run**: <date>
**Total votes**: <n>
**Files**: <list>

## Architecture (one-paragraph)
<what does it do — 5-10 lines max>

## Key Techniques

### Feature Engineering
- <bullet 1: e.g. "GR rolling mean at windows 11/31/51/101">
- <bullet 2>

### Model & Hyperparams
- <model>: <key hyperparams>

### Particle Filter / Beam Search
- <PF params if present>

### Ensemble / Blending
- <weights, recipe>

### CV Methodology
- <fold scheme, target convention>

## Anything novel vs LB-7.776 kernel?
<short — what's different>

## Score-relevant constants extracted
| name | value | source line |
|------|-------|-------------|
| ... | ... | ... |
```

Then **route the extracted info** to the right subdir:

- Feature code snippets → `experiments/public_resources/feature_engineering/<feature_name>.md` (with citation back to kernel)
- Model params dicts → `experiments/public_resources/model_params/<algo>_<kernel_slug>.json`
- Ensemble blending recipes → `experiments/public_resources/ensemble_weights/<recipe_name>.md`
- Preprocessing tricks → `experiments/public_resources/preprocessing/<trick>.md`
- CV scheme variants → `experiments/public_resources/cv_methodology/<scheme>.md`

**Cross-reference everything** — every md should link back to source kernel and any related extraction in other subdirs.

---

## Phase 3 — Fusion Staging (NO training, just staging)

For each extracted technique, write a **drop-in candidate stub** to `experiments/public_harvest/`:

- New feature → `experiments/public_harvest/feat_<name>.py` (a pure feature-extraction function ready to be added to round_010's data_loader)
- New model config → `experiments/public_harvest/model_<algo>_<source>.py` (a candidate stub matching round_010's contract: CANDIDATE_ID, get_features, fit_fold, predict)
- New blending recipe → `experiments/public_harvest/blend_<name>.json`

These are stubs only — DO NOT execute them. The parent agent will review and selectively integrate.

---

## Phase 4 — Long Poll

After Phase 1-3 drain, sleep 2 hours, then re-run kernel list + dataset list. If new refs appear:
- Add to `experiments/public_resources/all_kernel_refs.txt`
- Pull new kernels only
- Extract + stage as in Phase 2-3

Repeat until 2 consecutive 2-hour polls return zero new refs. Then write a final summary report and exit.

---

## Output Contract

When you finish (or hit a wall), produce **`experiments/public_resources/HARVEST_SUMMARY.md`** with:

1. Total kernels pulled / failed
2. Total datasets pulled / skipped (with reasons)
3. Top 5 kernels by LB score (with brief 1-line summary)
4. Top 10 unique-vs-our-stack techniques discovered (ranked by likely impact)
5. List of staged drop-in candidates in `experiments/public_harvest/` with one-line description each
6. Anything that surprised you / red flags / leaks discovered
7. Disk used (cumulative)
8. Wall clock total

This summary goes back to the parent agent. Be **terse and dense** — no filler.

---

## Logging

Append to `experiments/public_resources/HARVEST_LOG.md` continuously:
```
[YYYY-MM-DD HH:MM] Phase 1A start: 60 kernels listed
[YYYY-MM-DD HH:MM] Pulled smartorz/lb-7-776-rogii-ridge-sp (235 KB)
[YYYY-MM-DD HH:MM] FAILED user/some-kernel: 3 retries exhausted
[YYYY-MM-DD HH:MM] Phase 2 extract: kernel-X → feature_engineering/sg_filter_multi.md
...
```

Make it greppable for the parent agent to follow your progress.

---

## State Recovery

If you crash / time out and re-spawn, check `experiments/public_resources/HARVEST_LOG.md` for the last completed step and resume from there. Phase 1 is idempotent (kaggle pull skips existing dirs gracefully if you check with `[ -d ]` before).

---

## START HERE

1. Read `experiments/public_resources/HARVEST_LOG.md` if it exists, otherwise create it.
2. Confirm `kaggle` CLI works: `kaggle competitions list -c rogii-wellbore-geology-prediction`
3. Begin Phase 1A (kernel listing).
4. Work through phases sequentially.
5. **Do not stop until you exhaust Phase 4 termination condition or hit a hard error.**

Go.
