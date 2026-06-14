# Dataset: ravaghi/wellbore-geology-prediction-artifacts

- title: Wellbore Geology Prediction | Artifacts
- size_mb: 2254
- last_updated: 2026-06-04 14:39:56.717000
- downloads: 1538
- votes: 6
- usability: 0.29411766
- url: https://www.kaggle.com/datasets/ravaghi/wellbore-geology-prediction-artifacts
- decision: SKIPPED (size > 500MB)
- IMPORTANT: this is the artifact dataset that LB-7.776 and most ridge-stack forks load pre-trained
  3×LGB + 2×CB pickled models from. Path: `models/lightgbm-{1,2,3}/*.pkl` and `models/catboost-{1,2}/*.pkl`,
  plus an aggregated `data/train.csv` feature matrix. If exact LB-7.776 reproduction is needed, this is
  the source — but at 2.2 GB it exceeds the 500 MB budget. The alternative is re-extracting features +
  re-training the stack from scratch using the params in `model_params/lightgbm_lb7776.json` and
  `model_params/catboost_lb7776.json`.
