# Financial Risk Detection: Credit Default Prediction with Federated Learning

A comprehensive machine learning pipeline for predicting credit default risk using ensemble models (Random Forest and LightGBM) and federated learning, with a focus on fairness, explainability, and privacy-preserving techniques.

## 🎯 Project Objectives

This project addresses two critical credit risk prediction tasks:

1. **Missed Upcoming EMI (Equated Monthly Installment)** — Predict whether a borrower will miss an upcoming payment (~4.9% positive class)
2. **Future 30+ DPD (Days Past Due)** — Predict whether a borrower will go 30+ days past due (~0.78% positive class)

### Key Research Questions

- Can we build accurate early-warning models for credit default using borrower transaction history?
- How does centralized vs. federated learning perform on imbalanced credit risk targets?
- What fairness issues emerge across demographic groups (gender, income, education)?
- How can we explain individual predictions to borrowers and regulators?
- What privacy guarantees do federated approaches provide vs. centralized baselines?

## 📊 Dataset

**Home Credit Default Risk (Full Competition Dataset)**

- **Source:** [Kaggle Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- **Location:** `data/home_credit_full/`
- **Size:** ~1.1M loan applications with rich transaction history
- **Tables:**
  - `application_train.csv` — Borrower demographics and loan basics
  - `installments_payments.csv` — Installment-level payment records
  - `POS_CASH_balance.csv` — Point-of-sale cash account history
  - `previous_application.csv` — Prior loan applications
  - `bureau.csv` — Credit bureau records
  - `bureau_balance.csv` — Monthly bureau balances

### Why This Dataset?

- Rich temporal signal: payment history, prior defaults, and bureau trades
- Real-world complexity: highly imbalanced targets, missing values, multiple data sources
- Federated learning applicability: multiple logical data partitions (by region/provider)
- Production relevance: models trained here reflect real credit risk workflows

## 🏗️ Pipeline Architecture

The project is organized as a **9-stage sequential pipeline**:

```
Stage 1: Data Audit & Cleaning
    ↓
Stage 2: Target Construction (labels from transaction history)
    ↓
Stage 3: Feature Engineering (100+ features from 7 tables)
    ↓
Stage 4: Chronological Splits (leakage-free train/val/test)
    ↓
Stage 5: Centralized Model Training (RF + LightGBM baselines)
    ↓
Stage 6: Error Analysis & Threshold Tuning
    ↓
Stage 7: Federated Learning Simulation (FedForest + FedEnsemble)
    ↓
Stage 8: Ethical Evaluation (fairness, explainability, privacy)
    ↓
Stage 9: Final Reporting & Visualization
```

### Stage Descriptions

| Stage | Purpose | Output |
|-------|---------|--------|
| **1** | Raw data audit, cleaning, anomaly detection | Cleaned Parquet files |
| **2** | Borrower-time snapshot construction, label generation | `missed_upcoming_emi` and `future_dpd30` targets |
| **3** | Feature families: demographics, payment history, bureau features | 100+ engineered features |
| **4** | Leakage-safe chronological train/val/test splits | Split Parquet files with aligned schema |
| **5** | Train Random Forest and LightGBM on pooled training data | Baseline models, confusion matrices, ROC curves |
| **6** | Validate on held-out set, optimize thresholds, analyze errors | Optimal decision thresholds, FP/FN analysis |
| **7** | Federated simulation: FedForest (tree aggregation) + FedEnsemble (pred averaging) | Centralized vs. federated performance comparison |
| **8** | Fairness metrics, SHAP explainability, calibration, privacy analysis | Demographic disparities, feature importance, harm metrics |
| **9** | Aggregate findings into narrative report with figures | Markdown report + PNG visualizations |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `imbalanced-learn`, `shap`, `matplotlib`, `joblib`

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yassshhhh22/Finanical-Risk-Detection.git
cd Finanical-Risk-Detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn lightgbm imbalanced-learn shap joblib matplotlib psutil
```

### Run the Full Pipeline

```bash
# Stage 1: Open notebooks/01_data_audit_and_cleaning.ipynb in Jupyter
# ... (completes data cleaning)

# Stage 2-9: Run scripts in sequence
python scripts/stage2_target_construction_fast.py
python scripts/stage3_build_model_features.py --verbose
python scripts/stage4_create_model_splits.py --verbose
python scripts/stage5_train_models.py --target both --verbose
python scripts/stage6_error_analysis.py --verbose
python scripts/stage7_federated_learning.py --target both --n-clients 5 --verbose
python scripts/stage8_ethical_evaluation.py --target both --verbose
python scripts/stage9_final_report.py --verbose
```

### Run a Quick Test (20k borrowers)

```bash
# Test the full pipeline with a smaller dataset
python scripts/stage2_target_construction_fast.py --limit-borrowers 20000
python scripts/stage3_build_model_features.py --verbose
python scripts/stage4_create_model_splits.py --verbose
python scripts/stage5_train_models.py --target both --verbose
```

## 📈 Key Results

### Centralized Model Performance (Test Set)

| Target | Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|--------|-------|---------|--------|----|-----------|---------| 
| `missed_upcoming_emi` | LightGBM | **0.7789** | 0.1406 | 0.1461 | 0.0822 | 0.6594 |
| `future_dpd30` | LightGBM | **0.7701** | 0.1188 | 0.0173 | 0.0088 | 0.6201 |
| `missed_upcoming_emi` | Random Forest | 0.7801 | 0.1403 | 0.1462 | 0.0820 | 0.6601 |
| `future_dpd30` | Random Forest | 0.8217 | 0.2023 | 0.307 | 0.3586 | 0.2684 |

**Note:** High class imbalance (0.78%–4.9% positive) means ROC-AUC and PR-AUC are more informative than F1 for business decisions.

### Federated Learning Performance

Simulated federated learning (K=5 clients) achieves near-identical performance:

- **FedForest (RF):** Loses < 0.02 ROC-AUC vs. centralized
- **FedEnsemble (GB):** Loses < 0.01 ROC-AUC vs. centralized

This validates tree aggregation as a privacy-preserving strategy for financial applications.

### Feature Importance (SHAP Analysis)

Top 3 features consistently dominate both targets and models:

1. **Payment History Streak** (`hist_on_time_streak`) — Longest streak of on-time payments
2. **Recent Delinquency** (`hist_recent_delay`) — Days since last late payment
3. **Late Payment Frequency** (`hist_late_count_last_6`) — Late payments in last 6 months

→ **Interpretation:** Recent payment behavior is the strongest predictor of future default.

### Fairness Results

- **Gender Disparate Impact:** 0.99–1.13 (within acceptable < 0.8 or > 1.25 range)
- **Education Gap:** Academic degree borrowers have 30% lower predicted positive rates
- **Income Type:** Working-class borrowers face marginally higher false positive rates (exclusion risk)

## 📁 Repository Structure

```
Finanical-Risk-Detection/
├── README.md                          # This file
├── PROJECT_STAGES.md                  # Master runbook for all stages
├── data/
│   └── home_credit_full/             # Raw Kaggle competition data
│   └── DATASET_NOTES.md              # Dataset documentation
├── notebooks/
│   └── 01_data_audit_and_cleaning.ipynb   # Stage 1 (interactive)
├── scripts/
│   ├── stage2_target_construction_fast.py
│   ├── stage3_build_model_features.py
│   ├── stage4_create_model_splits.py
│   ├── stage5_train_models.py
│   ├── stage6_error_analysis.py
│   ├── stage7_federated_learning.py
│   ├── stage8_ethical_evaluation.py
│   └── stage9_final_report.py
├── docs/
│   ├── stage1_notes.md               # Data cleaning decisions
│   ├── stage2_notes.md               # Target construction logic
│   ├── stage3_notes.md               # Feature engineering details
│   ├── stage4_notes.md               # Split strategy
│   ├── stage5_notes.md               # Modeling approach
│   ├── stage6_notes.md               # Threshold tuning
│   ├── stage7_notes.md               # Federated learning design
│   ├── stage8_notes.md               # Ethical framework
│   └── stage9_notes.md               # Report assembly
├── reports/
│   ├── README.md                     # Key results summary
│   ├── FINAL_REPORT.md               # Complete narrative report
│   └── stage9_manifest.json          # Report metadata
└── artifacts/
    ├── processed/                    # Cleaned data & features
    ├── models/                       # Trained RF/GB pickles
    ├── metrics/                      # Performance CSVs
    ├── plots/                        # PNG figures
    └── raw_checks/                   # Audit & manifest files
```

## 🔍 Documentation

Start here for deep dives:

| File | Content |
|------|---------|
| **PROJECT_STAGES.md** | Master runbook with commands and expected outputs for all 9 stages |
| **docs/stageN_notes.md** | Stage-specific design decisions and technical details (N=1..9) |
| **reports/FINAL_REPORT.md** | Complete findings: performance, fairness, privacy, explainability |
| **data/DATASET_NOTES.md** | Dataset overview and column descriptions |

## 🎓 Key Insights

### 1. Payment History Dominates Default Risk
Recent payment behavior (on-time streaks, late payment counts) explains 80%+ of model predictions for both targets. This is consistent with domain knowledge in credit risk.

### 2. Privacy-Preserving Federated Learning is Practical
FedForest (tree aggregation) and FedEnsemble (ensemble averaging) lose < 0.02 ROC-AUC vs. centralized training, demonstrating that federated approaches are viable for production credit systems.

### 3. Fairness Issues Require Attention
- Gender: mostly fair (DI ratios 0.99–1.13)
- Education: borrowers with academic degrees are predicted positive at 30% lower rates
- Income: working-class borrowers face higher false positive rates (exclusion risk)

**Recommended mitigations:** Per-group threshold tuning, fairness constraints in model training.

### 4. Extreme Class Imbalance Requires Careful Metrics
- Standard metrics (accuracy, F1) are misleading
- ROC-AUC and Precision-Recall curves are more informative
- F1 improves with SMOTE; precision improves with threshold tuning

### 5. Privacy Needs Differential Privacy for Production
Current federated setup avoids raw data sharing but:
- Tree structures can leak membership information via overfitting
- Recommended: Add Gaussian noise to shared gradients/trees for ε-differential privacy guarantees

## 🔐 Privacy & Ethical Considerations

### Privacy

- **Centralized:** All raw borrower data held by single entity → highest privacy risk
- **Federated (current):** Each client trains locally; only model parameters shared → reduced risk
- **Federated + DP:** Add noise injection to model updates → formal privacy guarantees (ε-DP)

### Fairness

- Gender parity: models achieve disparate impact ratios 0.99–1.13 (acceptable)
- Group accuracy: some demographic groups have larger false negative rates
- Solution: audit per-group thresholds and consider group-specific models

### Explainability

- SHAP (SHapley Additive exPlanations) provides Shapley values for tree ensembles
- Enables global (feature importance) and local (individual prediction) explanations
- TreeExplainer runs in O(TLD) time where T=trees, L=leaves, D=depth

### Data Minimization

Current features use only predictive columns from raw tables. Non-predictive demographics and high-risk PII (e.g., SSN) should be dropped before production deployment.

## 🛠️ Commands Reference

### Full Pipeline

```bash
# Run all stages (1-9) in sequence
python scripts/stage2_target_construction_fast.py
python scripts/stage3_build_model_features.py --verbose
python scripts/stage4_create_model_splits.py --verbose
python scripts/stage5_train_models.py --target both --verbose
python scripts/stage6_error_analysis.py --verbose
python scripts/stage7_federated_learning.py --target both --n-clients 5 --verbose
python scripts/stage8_ethical_evaluation.py --target both --verbose
python scripts/stage9_final_report.py --verbose
```

### Stage-Specific Options

```bash
# Stage 2: Limit borrowers for quick test
python scripts/stage2_target_construction_fast.py --limit-borrowers 20000

# Stage 3: Rebuild feature cache
python scripts/stage3_build_model_features.py --verbose --rebuild-cache

# Stage 5: Train only on one target
python scripts/stage5_train_models.py --target missed_upcoming_emi --verbose

# Stage 7: Change number of federated clients
python scripts/stage7_federated_learning.py --target both --n-clients 10 --verbose

# Stage 8: Skip expensive SHAP computation
python scripts/stage8_ethical_evaluation.py --target both --skip-shap --verbose
```

## 📊 Output Artifacts

After running the full pipeline, find:

- **Trained Models:** `artifacts/models/rf_{target}.pkl`, `artifacts/models/gb_{target}.pkl`
- **Performance Metrics:** `artifacts/metrics/stage{5..8}_*.csv`
- **Figures:** `artifacts/plots/fig_0{1..5}_*.png`
- **Final Report:** `reports/FINAL_REPORT.md`
- **Feature Importance:** `artifacts/raw_checks/stage3_feature_dictionary.csv`

## 🤝 Contributing

This is an educational/research project. For improvements:

1. Create a feature branch: `git checkout -b feature/your-improvement`
2. Make changes and test thoroughly
3. Commit: `git commit -am "Add your message"`
4. Push: `git push origin feature/your-improvement`
5. Open a pull request

## 📚 References

- **Federated Learning:** McMahan et al. (2016) *Communication-Efficient Learning of Deep Networks from Decentralized Data*
- **SHAP:** Lundberg & Lee (2017) *A Unified Approach to Interpreting Model Predictions*
- **Fairness:** Hardt et al. (2016) *Equality of Opportunity in Supervised Learning*
- **Class Imbalance:** Chawla et al. (2002) *SMOTE: Synthetic Minority Over-sampling Technique*

## 📝 License

This project uses the Home Credit dataset from Kaggle. See their [terms of use](https://www.kaggle.com/competitions/home-credit-default-risk/rules).

## 👤 Author

**yassshhhh22** — Machine Learning / Credit Risk Research

## 🙋 Questions?

See `PROJECT_STAGES.md` for the master runbook, or dive into `docs/stageN_notes.md` for specific stage details.

---

**Last Updated:** 2026-04-11

For the latest updates and to track progress, see `reports/FINAL_REPORT.md` and `artifacts/plots/`.
