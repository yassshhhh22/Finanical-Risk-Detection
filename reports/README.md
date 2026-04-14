# Project: Ethical and Performance Evaluation of Ensemble and Federated Learning Models in Financial Applications

## Overview

This project implements a full end-to-end ML pipeline on the Home Credit dataset
to evaluate credit-risk prediction models across **performance**, **fairness**, and
**privacy** dimensions.

**Two prediction targets:**
- `missed_upcoming_emi` — will a borrower miss an upcoming EMI? (4.9% positive rate)
- `future_dpd30` — will a borrower go 30+ days past due? (0.78% positive rate)

**Two model families:**
- Random Forest (RF) — via Intel sklearnex-accelerated scikit-learn
- Gradient Boosting (GB) — via LightGBM

---

## What's in This Package

```
team_share/
├── README.md                  ← you are here
├── PROJECT_STAGES.md          ← pipeline overview, run commands, stage status
│
├── scripts/                   ← all Python scripts (stages 2–9)
│
├── reports/                   ← final report + figures
│   ├── FINAL_REPORT.md        ← complete written report with all findings
│   ├── fig_01_roc_auc_comparison.png
│   ├── fig_02_shap_importance.png
│   ├── fig_03_fairness_gender.png
│   ├── fig_04_harm_analysis.png
│   ├── fig_05_federated_comparison.png
│   ├── stage9_performance_summary.csv
│   ├── stage9_federated_summary.csv
│   └── stage9_fairness_summary.csv
│
├── stage_notes/               ← per-stage explanation and decisions
│   ├── stage5_notes.md        ← centralized model training decisions
│   ├── stage6_notes.md        ← error analysis and optimal thresholds
│   ├── stage7_notes.md        ← federated learning design
│   └── stage8_notes.md        ← privacy, fairness, harm framework
│
└── metrics/                   ← all raw metrics CSVs for further analysis
    ├── stage5_metrics.csv
    ├── stage5_confusion_matrices.csv
    ├── stage5_threshold_summary.csv
    ├── stage6_optimal_thresholds.csv
    ├── stage6_fp_fn_analysis.csv
    ├── stage6_slice_performance.csv
    ├── stage6_calibration.csv
    ├── stage7_client_metrics.csv
    ├── stage7_comparison.csv
    ├── stage8_fairness_metrics.csv
    ├── stage8_harm_analysis.csv
    ├── stage8_calibration_fairness.csv
    └── stage8_shap_importance.csv
```

---

## Pipeline Stages

| Stage | Purpose | Script |
|---|---|---|
| 1 | Raw data audit and cleaning | notebook |
| 2 | Target construction (labels) | `stage2_target_construction_fast.py` |
| 3 | Feature engineering (50+ features) | `stage3_build_model_features.py` |
| 4 | Chronological train/val/test splits | `stage4_create_model_splits.py` |
| 5 | Centralized RF + LightGBM training | `stage5_train_models.py` |
| 6 | Error analysis + optimal thresholds | `stage6_error_analysis.py` |
| 7 | Simulated federated learning (K=5) | `stage7_federated_learning.py` |
| 8 | Ethical evaluation (fairness + SHAP) | `stage8_ethical_evaluation.py` |
| 9 | Final report + figures | `stage9_final_report.py` |

---

## Key Results at a Glance

### Centralized Model Performance (test set)

| Target | Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| missed_upcoming_emi | RF | 0.7801 | 0.1390 | 0.1994 |
| missed_upcoming_emi | GB | 0.7789 | 0.1406 | 0.2006 |
| future_dpd30 | RF | 0.8217 | 0.2059 | 0.3070 |
| future_dpd30 | GB | 0.7701 | 0.1188 | 0.2037 |

### Federated vs Centralized (ROC-AUC gap)

| Target | Model | Centralized | Federated | Gap |
|---|---|---|---|---|
| missed_upcoming_emi | RF | 0.7801 | 0.7776 | −0.0025 |
| missed_upcoming_emi | GB | 0.7789 | 0.7808 | +0.0019 |
| future_dpd30 | RF | 0.8217 | 0.8058 | −0.0159 |
| future_dpd30 | GB | 0.7701 | 0.7698 | −0.0003 |

**Finding:** Federated learning loses < 0.02 ROC-AUC vs centralized training.

### Top SHAP Features (both targets)

Both models rank **recent payment history** as the dominant signal:
1. `hist_on_time_streak` — consecutive on-time payments
2. `hist_late_count_last_6` — late payments in last 6 months
3. `hist_recent_delay` — days overdue on most recent payment
4. `prev_application_count` — number of prior loan applications
5. `hist_delay_max` — maximum historical delay

### Fairness Summary (gender, RF)

All Disparate Impact Ratios fall within acceptable range (0.8–1.25):
- `missed_upcoming_emi`: Female DI = 1.031, Male DI = 0.987
- `future_dpd30`: Female DI = 1.127, Male DI = 1.081

Larger disparities observed for **education level** — Academic degree holders
show lower positive prediction rates (DI ≈ 0.70), consistent with genuinely
lower prevalence in that group.

---

## How to Run

Requirements: Python 3.10+, packages in requirements below.

```bash
pip install scikit-learn lightgbm imbalanced-learn shap joblib pandas numpy matplotlib psutil
# For Intel acceleration (optional):
pip install scikit-learn-intelex
```

Run stages in order:
```powershell
python scripts\stage5_train_models.py --target both --verbose
python scripts\stage6_error_analysis.py --verbose
python scripts\stage7_federated_learning.py --target both --verbose
python scripts\stage8_ethical_evaluation.py --target both --skip-shap --verbose
python scripts\stage9_final_report.py --verbose
```

> **Note:** Stages 1–4 require the full Home Credit dataset (~3 GB).
> Stages 5–9 require the processed parquet files from Stage 4 (~650 MB).

---

## Dataset

Home Credit Default Risk dataset:
- Source: Home Credit Group (Kaggle competition)
- ~300,000 loan applications, expanded to ~6M borrower-time snapshots
- Features: 50 numeric + 6 categorical after engineering

---

## Authors

ML Project — College Submission
