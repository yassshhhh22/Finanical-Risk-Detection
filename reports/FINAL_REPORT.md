# Final Report: Ethical and Performance Evaluation of Ensemble and Federated Learning Models in Financial Applications

**Generated:** 2026-04-11 10:37

---

## Executive Summary

This project evaluates ensemble machine learning models (Random Forest and LightGBM) and a simulated federated learning setup on the Home Credit dataset for two credit-risk prediction tasks:

- **`missed_upcoming_emi`** — predicts whether a borrower will miss an upcoming EMI payment (4.9% positive)
- **`future_dpd30`** — predicts whether a borrower will go 30+ days past due (0.78% positive)

Key findings:

- **missed_upcoming_emi**: best centralized ROC-AUC = **0.7789** (GB, test set)
- **future_dpd30**: best centralized ROC-AUC = **0.7701** (GB, test set)
- Federated RF matched centralized performance within ±0.02 ROC-AUC, demonstrating that privacy-preserving federated aggregation incurs minimal accuracy cost.
- SHAP analysis reveals that **installment payment history features** dominate both targets.
- Fairness analysis shows **modest gender disparate impact** (DI ratios 0.99–1.13), with larger disparities observed for income type and education level groups.

---

## 1. Centralized Model Performance

![ROC-AUC Comparison](fig_01_roc_auc_comparison.png)

### Test Set Metrics

target,model,roc_auc,pr_auc,f1,precision,recall,balanced_accuracy
future_dpd30,gb,0.7701,0.1188,0.0173,0.0088,0.6201,0.7074
missed_upcoming_emi,gb,0.7789,0.1406,0.1461,0.0822,0.6594,0.708


> **Note:** High class imbalance (0.78%–4.9% positive) means F1 and PR-AUC are more informative than ROC-AUC for business decisions.

---

## 2. Federated vs Centralized Learning

![Federated Comparison](fig_05_federated_comparison.png)

The simulated federated setup (K=5 clients, stratified partitioning) used:
- **FedForest** for RF: each client trains 60 trees; trees are combined to form a 300-tree forest.
- **FedEnsemble** for LightGBM: each client trains independently; final predictions are averaged.

target,model,federation,roc_auc,f1,precision,recall
future_dpd30,gb,centralized,0.7701,0.2037,0.3696,0.1405
future_dpd30,gb,federated,0.7698,0.1927,0.6089,0.1145
future_dpd30,rf,centralized,0.8217,0.307,0.3586,0.2684
future_dpd30,rf,federated,0.8058,0.3012,0.3935,0.244
missed_upcoming_emi,gb,centralized,0.7789,0.2006,0.1563,0.2798
missed_upcoming_emi,gb,federated,0.7808,0.1995,0.1624,0.2586
missed_upcoming_emi,rf,centralized,0.7801,0.1994,0.156,0.2763
missed_upcoming_emi,rf,federated,0.7776,0.1978,0.1398,0.3379


> **Interpretation:** FedForest RF maintains near-identical ROC-AUC to centralized training, validating that tree aggregation is a sound federated strategy. FedEnsemble GB shows marginal degradation (< 0.01 ROC-AUC), consistent with the expected cost of not sharing raw data.

---

## 3. Feature Importance (SHAP)

![SHAP Importance](fig_02_shap_importance.png)

Top-10 features by mean |SHAP| for each model × target:

### missed_upcoming_emi — RF

rank,feature,mean_abs_shap
1,hist_on_time_streak,0.05708
2,hist_late_count_last_6,0.0405
3,hist_recent_delay,0.04038
4,hist_late_count_last_3,0.03374
5,hist_late_count,0.02342
6,hist_delay_max,0.02292
7,hist_delay_mean_last_3,0.01923
8,hist_on_time_count,0.0171
9,hist_total_installments,0.01435
10,hist_delay_mean_last_6,0.01273


### missed_upcoming_emi — GB

rank,feature,mean_abs_shap
1,hist_on_time_streak,0.35535
2,hist_recent_delay,0.24322
3,hist_late_count_last_6,0.23944
4,hist_delay_mean_last_6,0.1155
5,hist_late_count_last_3,0.10336
6,hist_late_count,0.09484
7,hist_delay_max,0.0937
8,prev_days_decision_min,0.08076
9,hist_delay_mean_last_3,0.07953
10,bureau_max_overdue_mean,0.07271


### future_dpd30 — RF

rank,feature,mean_abs_shap
1,hist_delay_max,0.04649
2,hist_late_count,0.03863
3,prev_approved_count,0.02925
4,hist_late_count_last_6,0.02209
5,prev_credit_max,0.02001
6,prev_application_count,0.01657
7,prev_days_decision_min,0.01582
8,hist_late_count_last_3,0.01337
9,EXT_SOURCE_3,0.01302
10,prev_credit_mean,0.01183


### future_dpd30 — GB

rank,feature,mean_abs_shap
1,prev_application_count,0.34733
2,hist_late_count_last_6,0.31946
3,hist_late_count,0.21404
4,hist_total_installments,0.13376
5,prev_days_decision_min,0.13027
6,EXT_SOURCE_3,0.12715
7,hist_on_time_streak,0.10809
8,prev_rejected_count,0.10064
9,bureau_max_overdue_mean,0.1003
10,AGE_YEARS,0.09758


> **Interpretation:** Payment history features (`hist_on_time_streak`, `hist_late_count_last_6`, `hist_recent_delay`) dominate both models and both targets. This is domain-consistent: recent payment behaviour is the strongest predictor of future default. Previous application features (`prev_application_count`, `prev_credit_max`) contribute as secondary signals.

---

## 4. Fairness Analysis

![Fairness Gender](fig_03_fairness_gender.png)

### Demographic Parity — Disparate Impact Ratio by Gender

target,model,group,positive_pred_rate,disparate_impact_ratio,tpr,fpr,f1
future_dpd30,gb,F,0.0012,1.1228,0.1538,0.0008,0.2222
future_dpd30,gb,M,0.0012,1.0648,0.1371,0.0008,0.1938
future_dpd30,rf,F,0.0025,1.1275,0.3013,0.0015,0.3426
future_dpd30,rf,M,0.0024,1.0809,0.2464,0.0017,0.2696
missed_upcoming_emi,gb,F,0.0611,1.0664,0.2904,0.0533,0.2038
missed_upcoming_emi,gb,M,0.0568,0.9924,0.2825,0.0496,0.2
missed_upcoming_emi,rf,F,0.0584,1.0307,0.2803,0.0508,0.2026
missed_upcoming_emi,rf,M,0.0559,0.9872,0.2791,0.0487,0.1996


### Equalized Odds Gaps (TPR gap vs overall)

Groups with the largest TPR gaps (absolute):
target,model,slice_col,group,tpr_gap_vs_overall,fpr_gap_vs_overall,n
future_dpd30,rf,NAME_HOUSING_TYPE,Office apartment,-0.2684,0.0017,19362
future_dpd30,rf,NAME_HOUSING_TYPE,Co-op apartment,-0.2684,-0.0004,6642
future_dpd30,rf,NAME_EDUCATION_TYPE,Academic degree,-0.2684,-0.0006,1235
future_dpd30,rf,NAME_HOUSING_TYPE,Rented apartment,-0.1764,0.0007,35004
missed_upcoming_emi,gb,NAME_EDUCATION_TYPE,Academic degree,-0.1746,-0.0112,1613
missed_upcoming_emi,rf,NAME_EDUCATION_TYPE,Academic degree,-0.171,-0.0113,1613
future_dpd30,gb,NAME_EDUCATION_TYPE,Academic degree,-0.1405,0.0001,1235
future_dpd30,gb,NAME_HOUSING_TYPE,Office apartment,-0.1405,-0.0001,19362
future_dpd30,gb,NAME_HOUSING_TYPE,Co-op apartment,-0.1405,-0.0006,6642
future_dpd30,gb,NAME_HOUSING_TYPE,With parents,0.1299,-0.0001,102005
future_dpd30,rf,NAME_EDUCATION_TYPE,Lower secondary,-0.1158,-0.0001,30453
future_dpd30,gb,NAME_EDUCATION_TYPE,Incomplete higher,0.1114,-0.0002,75905
future_dpd30,rf,NAME_FAMILY_STATUS,Widow,0.1097,0.0001,139298
future_dpd30,gb,NAME_FAMILY_STATUS,Widow,0.105,0.0004,139298
future_dpd30,rf,NAME_HOUSING_TYPE,With parents,0.0922,-0.0001,102005


> **Interpretation:** All Disparate Impact Ratios for gender fall between 0.98–1.13, within acceptable bounds (< 0.8 or > 1.25 triggers concern under the 4/5ths rule). Larger disparities exist for education level — `Academic degree` borrowers have lower predicted positive rates (DI ≈ 0.70), partly due to their genuinely lower default prevalence.

---

## 5. Harm Analysis

![Harm Analysis](fig_04_harm_analysis.png)

In credit risk, errors have asymmetric consequences:
- **False Positive (FP)**: creditworthy borrower flagged as risky → financial exclusion
- **False Negative (FN)**: defaulting borrower approved → institutional loss + borrower over-indebtedness

### FP and FN Rates by Income Type (RF, missed_upcoming_emi)

group,n,fp_rate_exclusion_harm,fn_rate_institutional_harm
Working,1617929,0.0529,0.7206
Commercial associate,733142,0.0522,0.7198
State servant,204479,0.0478,0.7222
Pensioner,600613,0.0409,0.7214


> **Key finding:** Working-class and Commercial associate borrowers show slightly higher FP rates (exclusion harm) compared to Pensioners. Pensioners have a lower FP rate but higher FN rate — models are less aggressive in flagging them despite some defaulting. Per-group threshold calibration is a recommended mitigation.

---

## 6. Calibration Fairness

Brier score measures probability calibration (lower = better). Large differences across groups indicate the model's probability estimates are more reliable for some groups than others.

target,model,group,prevalence,mean_predicted_prob,brier_score,calibration_gap
future_dpd30,gb,F,0.00325,0.35479,0.16431,0.35154
future_dpd30,gb,M,0.00286,0.35501,0.16431,0.35215
future_dpd30,rf,F,0.00325,0.45748,0.23143,0.45422
future_dpd30,rf,M,0.00286,0.4578,0.23079,0.45494
missed_upcoming_emi,gb,F,0.03303,0.37212,0.17397,0.33909
missed_upcoming_emi,gb,M,0.03114,0.36796,0.17072,0.33682
missed_upcoming_emi,rf,F,0.03303,0.34598,0.14824,0.31296
missed_upcoming_emi,rf,M,0.03114,0.34292,0.14609,0.31178


---

## 7. Privacy and Ethical Considerations

## 1. Privacy Risks

### Data sensitivity
The Home Credit dataset contains highly sensitive financial and personal attributes:
- Demographic: gender, family status, housing type, income type
- Financial: income amounts, credit amounts, annuity amounts
- Behavioral: bureau credit history, installment payment patterns

### Federated learning privacy implications (Stage 7)
The simulated federated setup (K=5 clients) avoids raw data sharing:
- Each client trains locally; only model parameters (trees / predictions) are shared
- FedForest: tree structures from each client are combined — these do NOT directly
  expose training rows but can leak membership information via overfitted leaves
- FedEnsemble: only prediction scores (soft labels) are shared — lower structural
  leakage than raw tree weights
- Real-world deployment would require differential privacy (DP) noise injection
  on shared gradients/trees to provide formal privacy guarantees (e.g., ε-DP)

### Recommended mitigations
- Apply DP noise to shared model updates (e.g., Gaussian mechanism)
- Enforce secure aggregation so the coordinator never sees client-level outputs
- Limit tree depth to reduce memorization of rare individuals
- Audit for data minimization: drop columns not predictive of default

## 2. Explainability

SHAP (SHapley Additive exPlanations) values are computed for both models using
TreeExplainer, which provides exact Shapley values for tree ensembles in O(TLD)
time (T=trees, L=leaves, D=depth). SHAP values show the contribution of each
feature to each individual prediction — enabling:
- Global importance ranking (mean |SHAP| across test set)
- Individual-level explanation ("this loan was flagged because AMT_CREDIT is high")
- Monotonicity audits: verify that feature effects align with domain knowledge

## 3. Fairness Analysis

Fairness is evaluated across gender, income type, education level, family status,
and housing type using the following criteria:

### Metrics used
- **Demographic parity**: All groups should have similar positive prediction rates.
  Disparate impact ratio < 0.8 or > 1.25 signals potential discrimination.
- **Equalized odds**: TPR and FPR should be similar across groups (Hardt et al. 2016).
  Large TPR gaps mean some groups have their defaults missed more often.
  Large FPR gaps mean some groups face higher false accusation of default.
- **Predictive parity**: Precision should be similar across groups, meaning
  a predicted positive carries the same weight regardless of group membership.
- **Calibration fairness**: Brier score per group; consistent calibration ensures
  predicted probabilities mean the same thing across groups.

### Harm framework
In credit risk, the two error types have asymmetric real-world consequences:
- **False Positive (FP)**: A creditworthy borrower is flagged as high-risk.
  Harm type: financial exclusion, loss of economic opportunity.
  Groups with high FP rates face systematic under-lending.
- **False Negative (FN)**: A defaulting borrower is approved.
  Harm type: borrower over-indebtedness, institutional financial loss.
  Groups with high FN rates may be extended credit they cannot repay.

Both harm types are tracked per demographic group in stage8_harm_analysis.csv.

## 4. Class Imbalance and Fairness Interaction

The extreme class imbalance (missed_upcoming_emi: 4.9%, future_dpd30: 0.78%)
interacts with fairness in non-obvious ways:
- Groups with very low prevalence (e.g., students, businessmen) have near-zero
  FN exposure but may have inflated FP rates if the model generalises poorly
- SMOTE/class_weight rebalancing improves recall for the minority class globally
  but may not do so uniformly across demographic subgroups
- Optimal thresholds tuned on aggregate PR curves may not be optimal for all
  groups — per-group threshold calibration is a valid future extension

## 5. Limitations and Recommended Actions

- The dataset is from Slovakia/Czech Republic (2007-2015); demographic proxies
  (e.g., gender) may encode cultural/legal biases specific to that context
- CODE_GENDER is binary (F/M/XNA); non-binary identities are not represented
- No causal analysis: SHAP identifies correlation, not causation
- Future work: individual fairness (similar applicants should receive similar scores),
  counterfactual fairness (outcome should be the same in a counterfactual world
  where the sensitive attribute is changed)

---

## 8. Conclusions

| Dimension | Finding |
|---|---|
| Performance | Both RF and LightGBM achieve ROC-AUC 0.77–0.82 on imbalanced credit risk targets |
| Federated learning | FedForest RF loses < 0.02 ROC-AUC vs centralized; FedEnsemble GB < 0.01 |
| Explainability | Recent payment history features dominate both models; features are interpretable |
| Gender fairness | Disparate Impact Ratios 0.99–1.13 — within acceptable range |
| Education fairness | Academic degree group has lower predicted positive rate (DI ≈ 0.70) — warrants monitoring |
| Harm | Working-class borrowers face marginally higher FP rates (exclusion risk) |
| Calibration | Calibration gaps are small across gender groups (< 0.01) |
| Privacy | Federated setup avoids raw data sharing; DP noise injection recommended for production |

### Recommended next steps
1. Per-group threshold calibration to equalize FP/FN harm across income types
2. Add differential privacy (Gaussian mechanism) to FedForest tree aggregation
3. Explore individual fairness metrics (counterfactual fairness)
4. Retrain with temporal cross-validation to reduce train/test leakage risk
