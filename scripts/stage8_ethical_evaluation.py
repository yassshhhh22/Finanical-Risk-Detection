"""
Stage 8: Ethical Evaluation
============================
Uses Stage 5 models + Stage 6 predictions to produce a full ethical audit:

  1. Fairness metrics across demographic groups
       - Demographic parity (positive prediction rate per group)
       - Equalized odds (TPR and FPR per group)
       - Predictive parity (precision per group)
       - Disparate impact ratio
       - Per-group ROC-AUC, PR-AUC, F1
  2. Harm analysis
       - FP harm  (credit denied to creditworthy borrower → financial exclusion)
       - FN harm  (credit given to defaulter → institutional loss)
       - Relative harm rates across demographic groups
  3. SHAP-based explainability
       - TreeExplainer for RF and LightGBM (fast on tree models)
       - Mean |SHAP| per feature (global importance)
       - Top-20 feature importance table per model × target
  4. Privacy discussion (written to notes file)
  5. Calibration fairness
       - Brier score per group

Outputs:
    artifacts/metrics/stage8_fairness_metrics.csv
    artifacts/metrics/stage8_harm_analysis.csv
    artifacts/metrics/stage8_shap_importance.csv
    artifacts/metrics/stage8_calibration_fairness.csv
  artifacts/raw_checks/stage8_notes.md
  artifacts/raw_checks/stage8_manifest.json

Usage:
  python scripts/stage8_ethical_evaluation.py --verbose
  python scripts/stage8_ethical_evaluation.py --target missed_upcoming_emi --verbose
  python scripts/stage8_ethical_evaluation.py --skip-shap --verbose   # skip SHAP (slow)
"""

import argparse
import gc
import json
import logging
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "artifacts" / "processed"
MODELS    = ROOT / "artifacts" / "models"
CHECKS    = ROOT / "artifacts" / "raw_checks"
METRICS   = ROOT / "artifacts" / "metrics"
MANIFEST  = CHECKS / "stage4_column_manifest.json"

TARGETS     = ["missed_upcoming_emi", "future_dpd30"]
MODEL_NAMES = ["rf", "gb"]

# Demographic slice columns available in Stage 6 prediction parquets
SLICE_COLS = [
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]

# Groups with too few samples get skipped to avoid noisy metrics
MIN_GROUP_SIZE = 500

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    for noisy in ("sklearnex", "onedal", "matplotlib", "shap", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def elapsed(t0: float) -> str:
    return str(timedelta(seconds=int(time.time() - t0)))


def mem() -> str:
    try:
        import psutil, os
        mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        return f"  [RAM {mb:,.0f} MB]"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob):
    if y_true.sum() == 0:
        return float("nan")
    return average_precision_score(y_true, y_prob)


def compute_group_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           y_pred: np.ndarray) -> dict:
    n      = len(y_true)
    n_pos  = int(y_true.sum())
    n_pred = int(y_pred.sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    tpr = tp / n_pos   if n_pos > 0             else float("nan")
    fpr = fp / (n - n_pos) if (n - n_pos) > 0  else float("nan")
    fnr = fn / n_pos   if n_pos > 0             else float("nan")
    ppv = tp / n_pred  if n_pred > 0            else float("nan")   # precision

    return {
        "n": n,
        "n_actual_pos": n_pos,
        "n_predicted_pos": n_pred,
        "positive_pred_rate": n_pred / n if n > 0 else float("nan"),
        "roc_auc":   safe_roc_auc(y_true, y_prob),
        "pr_auc":    safe_pr_auc(y_true, y_prob),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": ppv,
        "recall":    tpr,   # TPR
        "tpr":       tpr,
        "fpr":       fpr,
        "fnr":       fnr,
        "brier":     brier_score_loss(y_true, y_prob) if n_pos > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------
def compute_fairness(df: pd.DataFrame, model: str, target: str,
                     log: logging.Logger) -> list[dict]:
    rows = []
    y_true = df["label"].values
    y_prob = df["y_prob"].values
    y_pred = df["y_pred_optimal"].values

    # Overall baseline
    base = compute_group_metrics(y_true, y_prob, y_pred)
    base_ppr = base["positive_pred_rate"]  # demographic parity reference
    base_tpr = base["tpr"]
    base_fpr = base["fpr"]

    for col in SLICE_COLS:
        if col not in df.columns:
            continue
        groups = df[col].dropna().unique()
        for grp in sorted(groups):
            mask = df[col] == grp
            if mask.sum() < MIN_GROUP_SIZE:
                log.debug(f"    skip {col}={grp}: only {mask.sum()} rows")
                continue
            sub_y  = y_true[mask]
            sub_p  = y_prob[mask]
            sub_pr = y_pred[mask]
            m = compute_group_metrics(sub_y, sub_p, sub_pr)

            # Disparate impact = group PPR / overall PPR
            di = (m["positive_pred_rate"] / base_ppr
                  if base_ppr and base_ppr > 0 else float("nan"))
            # Equalized odds gap
            tpr_gap = m["tpr"] - base_tpr if not np.isnan(m["tpr"]) else float("nan")
            fpr_gap = m["fpr"] - base_fpr if not np.isnan(m["fpr"]) else float("nan")

            row = {
                "target": target,
                "model":  model,
                "slice_col": col,
                "group":  grp,
                **m,
                "disparate_impact_ratio": di,
                "tpr_gap_vs_overall": tpr_gap,
                "fpr_gap_vs_overall": fpr_gap,
            }
            rows.append(row)
            log.debug(f"    {col}={grp}: n={m['n']:,}  ROC-AUC={m['roc_auc']:.4f}"
                      f"  DI={di:.3f}  TPR-gap={tpr_gap:+.4f}")
    return rows


# ---------------------------------------------------------------------------
# Harm analysis
# ---------------------------------------------------------------------------
def compute_harm(df: pd.DataFrame, model: str, target: str,
                 log: logging.Logger) -> list[dict]:
    """
    FP harm: wrongly flagging a creditworthy borrower as risky
             → denied credit, financial exclusion harm
    FN harm: wrongly clearing a defaulting borrower as safe
             → credit extended, institutional financial loss
    Rates reported per demographic group.
    """
    rows = []
    y_true = df["label"].values
    y_pred = df["y_pred_optimal"].values

    for col in SLICE_COLS:
        if col not in df.columns:
            continue
        groups = df[col].dropna().unique()
        for grp in sorted(groups):
            mask  = df[col] == grp
            n     = mask.sum()
            if n < MIN_GROUP_SIZE:
                continue
            sub_y  = y_true[mask]
            sub_pr = y_pred[mask]

            n_neg    = int((sub_y == 0).sum())
            n_pos    = int((sub_y == 1).sum())
            fp       = int(((sub_pr == 1) & (sub_y == 0)).sum())
            fn       = int(((sub_pr == 0) & (sub_y == 1)).sum())
            fp_rate  = fp / n_neg if n_neg > 0 else float("nan")   # FP among actual negatives
            fn_rate  = fn / n_pos if n_pos > 0 else float("nan")   # FN among actual positives

            rows.append({
                "target":     target,
                "model":      model,
                "slice_col":  col,
                "group":      grp,
                "n":          int(n),
                "n_actual_neg": n_neg,
                "n_actual_pos": n_pos,
                "fp":         fp,
                "fn":         fn,
                "fp_rate_exclusion_harm": fp_rate,
                "fn_rate_institutional_harm": fn_rate,
            })
    return rows


# ---------------------------------------------------------------------------
# Calibration fairness
# ---------------------------------------------------------------------------
def compute_calibration_fairness(df: pd.DataFrame, model: str, target: str,
                                  log: logging.Logger) -> list[dict]:
    rows = []
    for col in SLICE_COLS:
        if col not in df.columns:
            continue
        for grp in sorted(df[col].dropna().unique()):
            mask = df[col] == grp
            if mask.sum() < MIN_GROUP_SIZE:
                continue
            sub_y = df.loc[mask, "label"].values
            sub_p = df.loc[mask, "y_prob"].values
            if sub_y.sum() == 0:
                continue
            brier = brier_score_loss(sub_y, sub_p)
            rows.append({
                "target": target, "model": model,
                "slice_col": col, "group": grp,
                "n": int(mask.sum()),
                "prevalence": float(sub_y.mean()),
                "mean_predicted_prob": float(sub_p.mean()),
                "brier_score": brier,
                "calibration_gap": float(sub_p.mean() - sub_y.mean()),
            })
    return rows


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------
def compute_shap(model_name: str, target: str, log: logging.Logger,
                 n_background: int = 500, n_explain: int = 100) -> list[dict]:
    """
    TreeExplainer (fast for tree-based models).
    Uses a random subsample to keep memory manageable.
    """
    try:
        import shap
    except ImportError:
        log.warning("  shap not installed — skipping SHAP. Run: pip install shap")
        return []

    log.info(f"  Loading {model_name.upper()} model for SHAP ...")
    model_path = MODELS / f"{model_name}_{target}.pkl"
    if not model_path.exists():
        log.warning(f"  Model not found: {model_path}")
        return []

    model = joblib.load(model_path)
    prep  = joblib.load(MODELS / f"preprocessor_{target}.pkl")

    # Load a sample of the test set for SHAP
    test_path = PROCESSED / f"model_features_{target}_test.parquet"
    with open(MANIFEST) as f:
        manifest = json.load(f)
    num_cols = manifest["numeric_feature_columns"]
    cat_cols = manifest["categorical_feature_columns"]
    feat_cols = num_cols + cat_cols

    log.info(f"  Reading test sample for SHAP ({n_explain} rows, reduced for speed) ...")
    df_test = pd.read_parquet(test_path, columns=feat_cols + ["label"])
    df_test = df_test.dropna(subset=["label"])

    rng = np.random.default_rng(42)
    idx = rng.choice(len(df_test), size=min(n_explain, len(df_test)), replace=False)
    df_sample = df_test.iloc[idx].reset_index(drop=True)

    X_sample = prep.transform(df_sample[feat_cols])

    # Feature names after transformation
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_sample.shape[1])]

    # Strip transformer prefixes (e.g. "num__" / "cat__")
    clean_names = []
    for name in feature_names:
        if "__" in name:
            clean_names.append(name.split("__", 1)[1])
        else:
            clean_names.append(name)

    log.info(f"  Running TreeExplainer on {n_explain} samples (may take several minutes) ...")
    t0 = time.time()

    # Heartbeat thread — prints every 30s so terminal doesn't look frozen
    _stop_hb = threading.Event()
    def _heartbeat():
        while not _stop_hb.wait(30):
            log.info(f"  ... SHAP still running  +{elapsed(t0)}{mem()}")
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        _stop_hb.set()
        log.warning(f"  TreeExplainer failed: {e}")
        return []
    finally:
        _stop_hb.set()

    # SHAP can return:
    #   list of 2 arrays (n_samples, n_features) — old format, take index 1
    #   3D array (n_samples, n_features, n_classes) — new format, take [:,:,1]
    #   2D array (n_samples, n_features) — single output, use as-is
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]   # positive class
    elif shap_arr.ndim == 4:
        # list-of-arrays stacked: shape (2, n_samples, n_features) → take [1]
        shap_arr = shap_arr[1]

    mean_abs = np.abs(shap_arr).mean(axis=0)
    # ensure 1D
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)
    log.info(f"  SHAP done in {elapsed(t0)}")

    rows = []
    for i, (fname, imp) in enumerate(zip(clean_names, mean_abs)):
        rows.append({
            "target":    target,
            "model":     model_name,
            "rank":      i + 1,
            "feature":   fname,
            "mean_abs_shap": float(imp),
        })

    # Sort by importance descending and re-rank
    rows.sort(key=lambda r: -r["mean_abs_shap"])
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank

    return rows[:50]   # top-50 features


# ---------------------------------------------------------------------------
# Privacy discussion text
# ---------------------------------------------------------------------------
PRIVACY_TEXT = """
# Stage 8: Ethical Evaluation Notes

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
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="both",
                   choices=["missed_upcoming_emi", "future_dpd30", "both"])
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip SHAP computation (saves ~10 min)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    log    = setup_logging(args.verbose)
    t_glob = time.time()

    targets = TARGETS if args.target == "both" else [args.target]

    log.info("")
    log.info("─" * 60)
    log.info("  STAGE 8: ETHICAL EVALUATION")
    log.info("─" * 60)
    log.info(f"  Target(s)  : {args.target}")
    log.info(f"  SHAP       : {'disabled' if args.skip_shap else 'enabled'}")
    log.info(f"  Slice cols : {SLICE_COLS}")
    log.info("")

    all_fairness   = []
    all_harm       = []
    all_calibration = []
    all_shap       = []

    for t_idx, target in enumerate(targets, 1):
        log.info(f"\n  *** TARGET {t_idx}/{len(targets)}: {target} ***\n")
        log.info("─" * 60)
        log.info(f"  TARGET: {target.upper()}")
        log.info("─" * 60)

        for model_name in MODEL_NAMES:
            pred_path = CHECKS / f"stage6_predictions_{target}_{model_name}.parquet"
            if not pred_path.exists():
                log.warning(f"  Predictions not found: {pred_path.name} — skipping")
                continue

            log.info(f"\n  [{model_name.upper()}] Loading predictions ...")
            df = pd.read_parquet(pred_path)
            log.info(f"  Loaded {len(df):,} rows  columns={df.columns.tolist()}{mem()}")

            # ── Fairness ────────────────────────────────────────────────────
            log.info(f"  [{model_name.upper()}] Computing fairness metrics ...")
            fair_rows = compute_fairness(df, model_name, target, log)
            all_fairness.extend(fair_rows)
            log.info(f"    → {len(fair_rows)} group×slice rows")

            # ── Harm analysis ───────────────────────────────────────────────
            log.info(f"  [{model_name.upper()}] Computing harm analysis ...")
            harm_rows = compute_harm(df, model_name, target, log)
            all_harm.extend(harm_rows)
            log.info(f"    → {len(harm_rows)} group×slice rows")

            # ── Calibration fairness ────────────────────────────────────────
            log.info(f"  [{model_name.upper()}] Computing calibration fairness ...")
            cal_rows = compute_calibration_fairness(df, model_name, target, log)
            all_calibration.extend(cal_rows)
            log.info(f"    → {len(cal_rows)} group×slice rows")

            del df
            gc.collect()

            # ── SHAP ────────────────────────────────────────────────────────
            if not args.skip_shap:
                log.info(f"  [{model_name.upper()}] Computing SHAP values ...")
                shap_rows = compute_shap(model_name, target, log)
                all_shap.extend(shap_rows)
                log.info(f"    → {len(shap_rows)} features ranked")
                gc.collect()

        gc.collect()

    # ── Save outputs ──────────────────────────────────────────────────────────
    log.info("")
    log.info("─" * 60)
    log.info("  Saving reports")
    log.info("─" * 60)

    CHECKS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    # Fairness metrics
    df_fair = pd.DataFrame(all_fairness)
    df_fair.to_csv(METRICS / "stage8_fairness_metrics.csv", index=False)
    log.info("  stage8_fairness_metrics.csv")

    # Harm analysis
    df_harm = pd.DataFrame(all_harm)
    df_harm.to_csv(METRICS / "stage8_harm_analysis.csv", index=False)
    log.info("  stage8_harm_analysis.csv")

    # Calibration fairness
    df_cal = pd.DataFrame(all_calibration)
    df_cal.to_csv(METRICS / "stage8_calibration_fairness.csv", index=False)
    log.info("  stage8_calibration_fairness.csv")

    # SHAP
    if all_shap:
        df_shap = pd.DataFrame(all_shap)
        df_shap.to_csv(METRICS / "stage8_shap_importance.csv", index=False)
        log.info("  stage8_shap_importance.csv")
    else:
        log.info("  stage8_shap_importance.csv  (skipped)")

    # Notes
    notes_path = CHECKS / "stage8_notes.md"
    notes_path.write_text(PRIVACY_TEXT.strip(), encoding="utf-8")
    log.info("  stage8_notes.md")

    # Manifest
    manifest = {
        "stage": 8,
        "run_at": datetime.now().isoformat(),
        "targets": targets,
        "shap_computed": not args.skip_shap and len(all_shap) > 0,
        "fairness_rows": len(all_fairness),
        "harm_rows":     len(all_harm),
        "calibration_rows": len(all_calibration),
        "shap_rows":     len(all_shap),
        "outputs": [
            "stage8_fairness_metrics.csv",
            "stage8_harm_analysis.csv",
            "stage8_calibration_fairness.csv",
            "stage8_shap_importance.csv",
            "stage8_notes.md",
        ],
    }
    (CHECKS / "stage8_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info("  stage8_manifest.json")

    # ── Summary printout ──────────────────────────────────────────────────────
    log.info("")
    log.info("─" * 60)
    log.info("  ALL DONE")
    log.info("─" * 60)
    log.info(f"  Total time : {elapsed(t_glob)}")
    log.info("")

    if len(df_fair) > 0:
        log.info("  Fairness highlights (Disparate Impact Ratio by gender):")
        gender_fair = df_fair[
            (df_fair["slice_col"] == "CODE_GENDER") &
            (df_fair["disparate_impact_ratio"].notna())
        ][["target", "model", "group", "positive_pred_rate",
           "disparate_impact_ratio", "tpr", "fpr", "f1"]].copy()
        gender_fair = gender_fair.sort_values(["target", "model", "group"])
        for _, row in gender_fair.iterrows():
            log.info(
                f"    {row['target']:<30} {row['model']}  gender={row['group']}"
                f"  PPR={row['positive_pred_rate']:.4f}"
                f"  DI={row['disparate_impact_ratio']:.3f}"
                f"  TPR={row['tpr']:.4f}  FPR={row['fpr']:.4f}"
            )

    if len(all_shap) > 0:
        log.info("")
        log.info("  Top-5 SHAP features per model×target:")
        df_shap_top = pd.DataFrame(all_shap)
        for (tgt, mdl), grp in df_shap_top.groupby(["target", "model"]):
            top5 = grp.nsmallest(5, "rank")
            feats = "  |  ".join(
                f"{r['feature']} ({r['mean_abs_shap']:.4f})"
                for _, r in top5.iterrows()
            )
            log.info(f"    {tgt} [{mdl}]: {feats}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
