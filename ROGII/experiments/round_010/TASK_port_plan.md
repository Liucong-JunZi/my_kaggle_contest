# Task: Plan full port of PF + ML + blend into Kaggle

User wants: run BOTH PF inference AND ML inference (LGB+CAT) inside the Kaggle kernel,
then blend them (0.75*ML + 0.25*PF_ens_s12) inside the kernel, producing a submission
CSV that matches the hidden test IDs.

## Part 1: Understand what PF kernel does
1. Read current PF kernel source: look in round_008 and round_010 for PF kernel scripts
2. Search for PF-related kernel scripts with `search_files(pattern="PF.*kernel|kaggle.*submit|pf.*submit", path="/Users/liucong/code/kaggle/ROGII/experiments/")`
3. Identify: What data does PF read from /kaggle/input/? What features does it compute? How long does it run?

## Part 2: Understand what ML models need
1. Find the trained LGB + CAT models used in v8 (scored 11.383)
2. What features do they need? (36 base + PF features?)
3. What's the feature extraction pipeline for the 36 base features?
4. Where are the model files saved?

## Part 3: Understand Kaggle constraints
- Check kernel-metadata.json in round_010/submission_package/hillclimb_v3_kernel/
- What datasets are already uploaded to Kaggle?
- What competition input does the PF kernel use?

## Part 4: Output plan
Write a concrete, actionable plan to:
- Port PF inference to the hill climb kernel (or create a combined kernel)
- Port ML inference (LGB + CAT models + feature pipeline)
- Run blend inside kernel
- Submit

Write the plan to: /Users/liucong/code/kaggle/ROGII/experiments/round_010/PORT_PLAN.md
