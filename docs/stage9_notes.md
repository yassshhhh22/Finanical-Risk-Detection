# Stage 9: Final Reporting and Delivery

## Purpose

Stage 9 converts all technical outputs from Stages 5-8 into a publication-ready
reporting package for reviewers, teammates, and stakeholders.

This stage is not model training. It is synthesis and communication:

- consolidate metrics into clean summary tables
- generate figures with consistent styling
- write a complete narrative report linking performance, federation, fairness,
  explainability, and harm

---

## Inputs

From Stage 5:

- `stage5_metrics.csv`

From Stage 7:

- `stage7_comparison.csv`

From Stage 8:

- `stage8_fairness_metrics.csv`
- `stage8_harm_analysis.csv`
- `stage8_shap_importance.csv`

These are joined into report-level summaries under `reports/`.

---

## Reporting Workflow

1. Performance summary

- Centralized test metrics are extracted and normalized
- Output: `stage9_performance_summary.csv`

2. Federated comparison summary

- Centralized vs federated metrics aligned by target/model
- Output: `stage9_federated_summary.csv`

3. Fairness summary extraction

- key demographic parity/equalized-odds rows curated
- Output: `stage9_fairness_summary.csv`

4. Figure generation

- fig_01: centralized ROC-AUC comparison
- fig_02: SHAP top-feature importance
- fig_03: fairness by gender (DI, TPR, FPR)
- fig_04: harm scatter (FP vs FN rates)
- fig_05: centralized vs federated comparison

5. Markdown report assembly

- writes `FINAL_REPORT.md` with integrated tables, interpretation, and caveats

---

## Why This Stage Is Critical

Without Stage 9, the project remains a collection of raw outputs spread across
multiple CSV files and logs.

Stage 9 creates decision-grade artifacts:

- reproducible and auditable summaries
- visual evidence for key claims
- concise narrative explaining implications and limitations

This is what makes the work evaluable by non-ML stakeholders.

---

## Figure Logic and Interpretation Intent

### Figure 1: ROC-AUC comparison

Purpose:

- establish baseline quality across targets and model families

### Figure 2: SHAP importance

Purpose:

- show what drives predictions globally
- support explainability and governance discussion

### Figure 3: Fairness by gender

Purpose:

- present parity and error-rate behavior in one compact view

### Figure 4: Harm analysis

Purpose:

- visualize tradeoff between exclusion harm (FP) and institutional harm (FN)
  across groups

### Figure 5: Federated comparison

Purpose:

- quantify privacy-preserving performance cost relative to centralized models

---

## Quality Checks in Stage 9

Before finalizing outputs, the stage verifies:

- required Stage 5-8 input files exist
- summary tables are non-empty
- figures are successfully written to disk
- manifest includes generated assets and timestamp

If an expected file is missing, report generation should fail loudly rather than
silently producing incomplete conclusions.

---

## Outputs

```
reports/
    stage9_performance_summary.csv
    stage9_federated_summary.csv
    stage9_fairness_summary.csv
  FINAL_REPORT.md
  README.md
  stage9_manifest.json

artifacts/plots/
    fig_01_roc_auc_comparison.png
    fig_02_shap_importance.png
    fig_03_fairness_gender.png
    fig_04_harm_analysis.png
    fig_05_federated_comparison.png
```

---

## Re-run Commands

```powershell
python scripts\stage9_final_report.py --verbose
```

If upstream stages were re-run, regenerate Stage 9 so figures/tables stay in sync.

---

## Project Completion Criteria

A full successful pipeline run means:

1. Stage 1-4 data preparation complete
2. Stage 5-8 metrics and diagnostics available
3. Stage 9 report assets generated without missing dependencies
4. Team-share package includes report, figures, summaries, and stage notes

At this point, the project is submission-ready and review-ready.
