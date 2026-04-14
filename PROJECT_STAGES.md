# Project Stages and Execution Steps

This file is the master guide for the full Home Credit pipeline (Stages 1-9),
including what each stage does, why it exists, how to run it, and what it produces.

## Current Status

All stages are implemented and completed:

1. Stage 1: Raw data audit and cleaning
2. Stage 2: Target construction
3. Stage 3: Feature engineering
4. Stage 4: Chronological splitting
5. Stage 5: Centralized model training
6. Stage 6: Error analysis and threshold tuning
7. Stage 7: Simulated federated learning
8. Stage 8: Ethical evaluation
9. Stage 9: Final reporting

---

## Dataset and Artifact Locations

- Raw dataset: `data/home_credit_full/`
- Processed data: `artifacts/processed/`
- Stage checks/manifests: `artifacts/raw_checks/`
- Stage metric CSVs: `artifacts/metrics/`
- Plot images: `artifacts/plots/`
- Final report assets: `reports/`
- Trained models: `artifacts/models/`

---

## Environment Setup

```powershell
cd y:\College\ML
.\.venv\Scripts\activate
```

Core dependencies used across stages:

```powershell
pip install pandas numpy scikit-learn lightgbm imbalanced-learn shap joblib matplotlib psutil
# Optional Intel acceleration:
pip install scikit-learn-intelex
```

---

## Recommended Execution Order

1. Stage 1 notebook
2. Stage 2 script
3. Stage 3 script
4. Stage 4 script
5. Stage 5 script
6. Stage 6 script
7. Stage 7 script
8. Stage 8 script
9. Stage 9 script

---

## Stage 1: Raw Data Audit and Cleaning

Purpose:

- validate raw tables and schema
- detect missingness, duplicates, anomalies
- produce cleaned Parquet files for downstream stages

Main notebook:

- `notebooks/01_data_audit_and_cleaning.ipynb`

Main outputs:

- `artifacts/processed/cleaned_application_train.parquet`
- `artifacts/processed/cleaned_installments_payments.parquet`
- `artifacts/processed/cleaned_installment_events.parquet`
- `artifacts/processed/cleaned_pos_cash_balance.parquet`
- `artifacts/processed/cleaned_previous_application.parquet`
- `artifacts/processed/cleaned_bureau.parquet`
- `artifacts/processed/cleaned_bureau_balance.parquet`

Audit outputs:

- `artifacts/raw_checks/raw_audit_summary.csv`
- `artifacts/raw_checks/raw_missingness_summary.csv`
- `artifacts/raw_checks/raw_duplicate_summary.csv`
- `artifacts/raw_checks/raw_anomaly_notes.md`
- `artifacts/raw_checks/stage1_manifest.json`

---

## Stage 2: Target Construction

Purpose:

- build borrower-time snapshots
- create labels: `missed_upcoming_emi`, `future_dpd30`

Script:

- `scripts/stage2_target_construction_fast.py`

Run:

```powershell
python scripts\stage2_target_construction_fast.py
```

Optional quick test:

```powershell
python scripts\stage2_target_construction_fast.py --limit-borrowers 20000
```

Outputs:

- `artifacts/processed/target_missed_upcoming_emi.parquet`
- `artifacts/processed/target_future_dpd30.parquet`
- `artifacts/raw_checks/stage2_target_summary.csv`
- `artifacts/raw_checks/stage2_target_notes.md`
- `artifacts/raw_checks/stage2_manifest.json`

---

## Stage 3: Feature Engineering

Purpose:

- create model-ready numeric/categorical feature matrix
- cache expensive feature families for faster reruns

Script:

- `scripts/stage3_build_model_features.py`

Run:

```powershell
python scripts\stage3_build_model_features.py --verbose
```

Force cache rebuild:

```powershell
python scripts\stage3_build_model_features.py --verbose --rebuild-cache
```

Outputs:

- `artifacts/processed/snapshot_feature_base.parquet`
- `artifacts/processed/model_features_missed_upcoming_emi.parquet`
- `artifacts/processed/model_features_future_dpd30.parquet`
- `artifacts/raw_checks/stage3_feature_dictionary.csv`
- `artifacts/raw_checks/stage3_feature_quality_report.csv`
- `artifacts/raw_checks/stage3_manifest.json`

---

## Stage 4: Chronological Splits

Purpose:

- create leakage-safe train/validation/test splits by time
- persist consistent column schema for downstream stages

Script:

- `scripts/stage4_create_model_splits.py`

Run:

```powershell
python scripts\stage4_create_model_splits.py --verbose
```

Outputs:

- `artifacts/processed/model_features_{target}_train.parquet`
- `artifacts/processed/model_features_{target}_validation.parquet`
- `artifacts/processed/model_features_{target}_test.parquet`
- `artifacts/raw_checks/stage4_snapshot_day_split_map.csv`
- `artifacts/raw_checks/stage4_split_summary.csv`
- `artifacts/raw_checks/stage4_column_manifest.json`
- `artifacts/raw_checks/stage4_manifest.json`

---

## Stage 5: Centralized Baseline Modeling

Purpose:

- train RF + LightGBM baselines on pooled data
- compute core classification metrics and confusion matrices

Script:

- `scripts/stage5_train_models.py`

Run:

```powershell
python scripts\stage5_train_models.py --target both --verbose
```

Outputs:

- `artifacts/models/rf_{target}.pkl`
- `artifacts/models/gb_{target}.pkl`
- `artifacts/models/preprocessor_{target}.pkl`
- `artifacts/metrics/stage5_metrics.csv`
- `artifacts/metrics/stage5_confusion_matrices.csv`
- `artifacts/metrics/stage5_threshold_summary.csv`
- `artifacts/raw_checks/stage5_notes.md`
- `artifacts/raw_checks/stage5_manifest.json`

---

## Stage 6: Error Analysis and Diagnostics

Purpose:

- optimize thresholds using validation PR curve
- analyze false positives/false negatives
- measure calibration and subgroup performance

Script:

- `scripts/stage6_error_analysis.py`

Run:

```powershell
python scripts\stage6_error_analysis.py --verbose
```

Outputs:

- `artifacts/metrics/stage6_optimal_thresholds.csv`
- `artifacts/raw_checks/stage6_predictions_{target}_{model}.parquet`
- `artifacts/metrics/stage6_fp_fn_analysis.csv`
- `artifacts/metrics/stage6_slice_performance.csv`
- `artifacts/metrics/stage6_calibration.csv`
- `artifacts/raw_checks/stage6_notes.md`
- `artifacts/raw_checks/stage6_manifest.json`

---

## Stage 7: Simulated Federated Learning

Purpose:

- compare centralized vs local vs federated training quality
- estimate privacy-preserving performance tradeoff

Script:

- `scripts/stage7_federated_learning.py`

Run:

```powershell
python scripts\stage7_federated_learning.py --target both --n-clients 5 --verbose
```

Outputs:

- `artifacts/metrics/stage7_client_metrics.csv`
- `artifacts/metrics/stage7_comparison.csv`
- `artifacts/raw_checks/stage7_notes.md`
- `artifacts/raw_checks/stage7_manifest.json`

---

## Stage 8: Ethical Evaluation

Purpose:

- fairness analysis across demographic slices
- harm analysis (FP/FN burden)
- SHAP explainability and calibration-fairness checks

Script:

- `scripts/stage8_ethical_evaluation.py`

Run:

```powershell
python scripts\stage8_ethical_evaluation.py --target both --verbose
```

Faster variant (skip SHAP):

```powershell
python scripts\stage8_ethical_evaluation.py --target both --skip-shap --verbose
```

Outputs:

- `artifacts/metrics/stage8_fairness_metrics.csv`
- `artifacts/metrics/stage8_harm_analysis.csv`
- `artifacts/metrics/stage8_calibration_fairness.csv`
- `artifacts/metrics/stage8_shap_importance.csv`
- `artifacts/raw_checks/stage8_notes.md`
- `artifacts/raw_checks/stage8_manifest.json`

---

## Stage 9: Final Report and Visuals

Purpose:

- aggregate Stage 5-8 outputs into final summary tables and figures
- generate complete markdown report

Script:

- `scripts/stage9_final_report.py`

Run:

```powershell
python scripts\stage9_final_report.py --verbose
```

Outputs (`reports/` + `artifacts/plots/`):

- `README.md`
- `FINAL_REPORT.md`
- `stage9_performance_summary.csv`
- `stage9_federated_summary.csv`
- `stage9_fairness_summary.csv`
- `artifacts/plots/fig_01_roc_auc_comparison.png`
- `artifacts/plots/fig_02_shap_importance.png`
- `artifacts/plots/fig_03_fairness_gender.png`
- `artifacts/plots/fig_04_harm_analysis.png`
- `artifacts/plots/fig_05_federated_comparison.png`

Manifest:

- `reports/stage9_manifest.json`

---

## Important Markdown Files and What Each Explains

- `PROJECT_STAGES.md`
  - master runbook for all stages and commands

- `docs/stage1_notes.md`
  - raw data audit logic, cleaning decisions, leakage-safe handling

- `docs/stage2_notes.md`
  - snapshot and label design (`missed_upcoming_emi`, `future_dpd30`)

- `docs/stage3_notes.md`
  - feature families, why each is included, caching strategy

- `docs/stage4_notes.md`
  - chronological split rationale and schema manifest role

- `docs/stage5_notes.md`
  - preprocessing choices, imbalance strategy, RF/GB training rationale

- `docs/stage6_notes.md`
  - threshold optimization, calibration, subgroup error diagnostics

- `docs/stage7_notes.md`
  - federated simulation design and centralized-vs-federated interpretation

- `docs/stage8_notes.md`
  - privacy/fairness/harm/explainability framework and findings

- `docs/stage9_notes.md`
  - report assembly logic, figure intent, and delivery criteria

- `reports/README.md`
  - share package overview and key headline results

- `reports/FINAL_REPORT.md`
  - complete narrative report with all tables and figure references

---

## Team Share Packaging

Build the final team package zip:

```powershell
.\share\pack_team_share.ps1
```

This produces `team_share.zip` at workspace root.
