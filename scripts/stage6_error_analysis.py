"""
Stage 6: Error Analysis and Diagnostics
========================================
Loads trained models from Stage 5 and performs:

  1. Optimal threshold detection    — fixes LightGBM F1=0 issue
  2. Prediction generation          — scores + labels on test set
  3. False positive/negative analysis
  4. Slice-based performance        — by gender, income type, education, etc.
  5. Calibration analysis           — reliability curves, Brier score

Outputs:
    artifacts/metrics/stage6_optimal_thresholds.csv
  artifacts/raw_checks/stage6_predictions_{target}.parquet
    artifacts/metrics/stage6_fp_fn_analysis.csv
    artifacts/metrics/stage6_slice_performance.csv
    artifacts/metrics/stage6_calibration.csv
  artifacts/raw_checks/stage6_notes.md
  artifacts/raw_checks/stage6_manifest.json

Usage:
  python scripts/stage6_error_analysis.py --verbose
  python scripts/stage6_error_analysis.py --target missed_upcoming_emi --verbose
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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

TARGETS = ["missed_upcoming_emi", "future_dpd30"]
MODELS_LIST = ["rf", "gb"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    logging.getLogger("sklearnex").setLevel(logging.WARNING)
    logging.getLogger("onedal").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def elapsed(t0: float) -> str:
    return str(timedelta(seconds=int(time.time() - t0)))


def section(log: logging.Logger, title: str, step: str = ""):
    prefix = f"[{step}] " if step else ""
    log.info("")
    log.info(f"{'─'*60}")
    log.info(f"  {prefix}{title}")
    log.info(f"{'─'*60}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def load_test_split(target: str, numeric_cols: list, categorical_cols: list,
                    log: logging.Logger) -> pd.DataFrame:
    """Load test split keeping passthrough columns for slice analysis."""
    path = PROCESSED / f"model_features_{target}_test.parquet"
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"  Reading test split ({size_mb:.0f} MB) ...")
    sys.stdout.flush()
    t0 = time.time()
    df = pd.read_parquet(path)
    log.info(f"  shape={df.shape}  loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()
    return df


def load_model_and_preprocessor(target: str, model_name: str, log: logging.Logger):
    model_path = MODELS / f"{model_name}_{target}.pkl"
    prep_path  = MODELS / f"preprocessor_{target}.pkl"

    if not model_path.exists():
        log.warning(f"  Model not found: {model_path.name} — skipping")
        return None, None
    if not prep_path.exists():
        log.warning(f"  Preprocessor not found: {prep_path.name} — skipping")
        return None, None

    log.info(f"  Loading {model_path.name} ({model_path.stat().st_size/1_048_576:.1f} MB) ...")
    sys.stdout.flush()
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor


# ---------------------------------------------------------------------------
# 1. Optimal threshold detection
# ---------------------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Find threshold that maximises F1. Also computes metrics at fixed thresholds
    for comparison. This resolves the LightGBM F1=0 issue at default 0.5.
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
    best_idx = np.argmax(f1_arr[:-1])
    best_thresh = float(thresh_arr[best_idx])

    results = {"optimal_threshold": best_thresh}
    for thresh in [0.3, 0.4, 0.5, best_thresh]:
        y_pred = (y_prob >= thresh).astype(int)
        label = f"t{thresh:.2f}".replace(".", "_")
        results[f"{label}_f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
        results[f"{label}_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        results[f"{label}_recall"]    = float(recall_score(y_true, y_pred, zero_division=0))

    return results


# ---------------------------------------------------------------------------
# 2. Generate predictions on test set
# ---------------------------------------------------------------------------
def generate_predictions(df_test: pd.DataFrame, model, preprocessor,
                          numeric_cols: list, categorical_cols: list,
                          optimal_thresh: float) -> pd.DataFrame:
    """Score test set and return a DataFrame with predictions appended."""
    X = df_test[numeric_cols + categorical_cols]
    X_prep = preprocessor.transform(X)
    y_prob = model.predict_proba(X_prep)[:, 1]

    out = df_test[["SK_ID_CURR", "snapshot_day", "label"]].copy()
    # include slice columns if present
    slice_cols = ["CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                  "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]
    for col in slice_cols:
        if col in df_test.columns:
            out[col] = df_test[col].values

    out["y_prob"]          = y_prob
    out["y_pred_05"]       = (y_prob >= 0.5).astype(np.int8)
    out["y_pred_optimal"]  = (y_prob >= optimal_thresh).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# 3. FP / FN analysis
# ---------------------------------------------------------------------------
def fp_fn_analysis(df_pred: pd.DataFrame, optimal_thresh: float,
                   target: str, model_name: str) -> list:
    """
    Summarise false positive and false negative rates across slice columns.
    Uses optimal threshold predictions.
    """
    y_true = df_pred["label"].values
    y_pred = df_pred["y_pred_optimal"].values

    rows = []
    slice_cols = ["CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                  "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]

    # Overall first
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rows.append({
        "target": target, "model": model_name,
        "slice_col": "overall", "slice_val": "all",
        "n": len(y_true),
        "actual_positive": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": fp / max(tn + fp, 1),   # false positive rate
        "fnr": fn / max(tp + fn, 1),   # false negative rate
    })

    # Per slice
    for col in slice_cols:
        if col not in df_pred.columns:
            continue
        for val, grp in df_pred.groupby(col, observed=True):
            yt = grp["label"].values
            yp = grp["y_pred_optimal"].values
            if len(yt) < 50:
                continue
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            tn_g, fp_g, fn_g, tp_g = cm.ravel()
            rows.append({
                "target": target, "model": model_name,
                "slice_col": col, "slice_val": str(val),
                "n": len(yt),
                "actual_positive": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
                "TP": int(tp_g), "FP": int(fp_g), "FN": int(fn_g), "TN": int(tn_g),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall":    float(recall_score(yt, yp, zero_division=0)),
                "f1":        float(f1_score(yt, yp, zero_division=0)),
                "fpr": fp_g / max(tn_g + fp_g, 1),
                "fnr": fn_g / max(tp_g + fn_g, 1),
            })

    return rows


# ---------------------------------------------------------------------------
# 4. Slice-based performance (ROC-AUC, PR-AUC per group)
# ---------------------------------------------------------------------------
def slice_performance(df_pred: pd.DataFrame, target: str, model_name: str) -> list:
    """Full metrics per demographic slice — key for Stage 8 ethical evaluation."""
    rows = []
    slice_cols = ["CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                  "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]

    for col in slice_cols:
        if col not in df_pred.columns:
            continue
        for val, grp in df_pred.groupby(col, observed=True):
            yt = grp["label"].values
            yp = grp["y_prob"].values
            if len(yt) < 100:
                continue
            if yt.sum() < 10 or yt.sum() == len(yt):
                continue
            rows.append({
                "target": target, "model": model_name,
                "slice_col": col, "slice_val": str(val),
                "n": len(yt),
                "positive_rate": float(yt.mean()),
                "roc_auc":  float(roc_auc_score(yt, yp)),
                "pr_auc":   float(average_precision_score(yt, yp)),
                "brier":    float(brier_score_loss(yt, yp)),
            })

    return rows


# ---------------------------------------------------------------------------
# 5. Calibration analysis
# ---------------------------------------------------------------------------
def calibration_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                          target: str, model_name: str, n_bins: int = 10) -> list:
    """
    Reliability curve — compares mean predicted probability vs actual positive rate
    per probability bin. A perfectly calibrated model sits on the diagonal.
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)
    rows = []
    for i, (fop, mp) in enumerate(zip(fraction_of_positives, mean_predicted)):
        rows.append({
            "target": target, "model": model_name,
            "bin": i,
            "mean_predicted_prob": float(mp),
            "actual_positive_rate": float(fop),
            "calibration_error": float(abs(fop - mp)),
            "brier_score": float(brier),
        })
    return rows


# ---------------------------------------------------------------------------
# Process one target
# ---------------------------------------------------------------------------
def process_target(target: str, numeric_cols: list, categorical_cols: list,
                   args, log: logging.Logger, t_global: float) -> tuple:
    section(log, f"TARGET: {target.upper()}")

    log.info("Loading test split ...")
    df_test = load_test_split(target, numeric_cols, categorical_cols, log)
    y_true_all = df_test["label"].values

    thresh_rows, fpfn_rows, slice_rows, calib_rows = [], [], [], []
    pred_dfs = {}

    for model_name in MODELS_LIST:
        section(log, f"Model: {model_name.upper()}  ({target})")

        model, preprocessor = load_model_and_preprocessor(target, model_name, log)
        if model is None:
            continue

        # --- Score ---
        log.info(f"  Scoring {len(df_test):,} test rows ...")
        sys.stdout.flush()
        t0 = time.time()
        X_prep = preprocessor.transform(df_test[numeric_cols + categorical_cols])
        y_prob = model.predict_proba(X_prep)[:, 1]
        log.info(f"  Scored in {time.time()-t0:.1f}s  prob range [{y_prob.min():.4f}, {y_prob.max():.4f}]")
        sys.stdout.flush()

        # --- 1. Optimal threshold ---
        log.info(f"  Finding optimal threshold ...")
        thresh_info = find_optimal_threshold(y_true_all, y_prob)
        optimal_thresh = thresh_info["optimal_threshold"]
        log.info(
            f"  Optimal threshold = {optimal_thresh:.4f}  "
            f"F1={thresh_info[f't{optimal_thresh:.2f}'.replace('.','_')+'_f1']:.4f}  "
            f"(vs F1=0 at 0.5 for GB)"
        )
        thresh_rows.append({"target": target, "model": model_name, **thresh_info})
        sys.stdout.flush()

        # --- 2. Predictions DataFrame ---
        log.info(f"  Building predictions table ...")
        df_pred = generate_predictions(
            df_test, model, preprocessor,
            numeric_cols, categorical_cols, optimal_thresh
        )
        pred_dfs[model_name] = df_pred

        # --- 3. FP/FN analysis ---
        log.info(f"  Running FP/FN analysis across slices ...")
        sys.stdout.flush()
        fpfn_rows.extend(fp_fn_analysis(df_pred, optimal_thresh, target, model_name))

        # --- 4. Slice performance ---
        log.info(f"  Running slice performance (ROC-AUC per demographic group) ...")
        sys.stdout.flush()
        slice_rows.extend(slice_performance(df_pred, target, model_name))

        # --- 5. Calibration ---
        log.info(f"  Running calibration analysis ...")
        sys.stdout.flush()
        calib_rows.extend(
            calibration_analysis(y_true_all, y_prob, target, model_name)
        )

        brier = brier_score_loss(y_true_all, y_prob)
        log.info(f"  Brier score = {brier:.5f}  (lower = better, 0 = perfect)")

        del model, preprocessor, X_prep, y_prob
        gc.collect()

    # Save per-target predictions parquet
    for model_name, df_pred in pred_dfs.items():
        out_path = CHECKS / f"stage6_predictions_{target}_{model_name}.parquet"
        df_pred.to_parquet(out_path, index=False)
        log.info(f"  Predictions saved → {out_path.name}  ({out_path.stat().st_size/1_048_576:.1f} MB)")
        sys.stdout.flush()

    del df_test
    gc.collect()

    return thresh_rows, fpfn_rows, slice_rows, calib_rows


# ---------------------------------------------------------------------------
# Save all reports
# ---------------------------------------------------------------------------
def save_reports(thresh_rows, fpfn_rows, slice_rows, calib_rows, log):
    section(log, "Saving all reports")

    METRICS.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(thresh_rows).to_csv(METRICS / "stage6_optimal_thresholds.csv", index=False)
    log.info("  stage6_optimal_thresholds.csv")

    pd.DataFrame(fpfn_rows).to_csv(METRICS / "stage6_fp_fn_analysis.csv", index=False)
    log.info("  stage6_fp_fn_analysis.csv")

    pd.DataFrame(slice_rows).to_csv(METRICS / "stage6_slice_performance.csv", index=False)
    log.info("  stage6_slice_performance.csv")

    pd.DataFrame(calib_rows).to_csv(METRICS / "stage6_calibration.csv", index=False)
    log.info("  stage6_calibration.csv")

    # Notes
    notes = ["# Stage 6 Error Analysis Notes\n"]
    notes.append(f"Run timestamp: {datetime.now().isoformat()}\n\n")

    if thresh_rows:
        notes.append("## Optimal Thresholds\n\n")
        df_t = pd.DataFrame(thresh_rows)[["target", "model", "optimal_threshold"]]
        notes.append(df_t.to_csv(index=False))
        notes.append("\n")

    if slice_rows:
        notes.append("## Slice Performance Summary\n\n")
        df_s = pd.DataFrame(slice_rows)
        # worst performing slices by ROC-AUC
        notes.append("### Lowest ROC-AUC slices:\n")
        worst = df_s.nsmallest(10, "roc_auc")[
            ["target", "model", "slice_col", "slice_val", "n", "roc_auc", "pr_auc"]
        ]
        notes.append(worst.to_csv(index=False))
        notes.append("\n")

    with open(CHECKS / "stage6_notes.md", "w") as f:
        f.write("".join(notes))
    log.info("  stage6_notes.md")

    manifest = {
        "stage": 6,
        "timestamp": datetime.now().isoformat(),
        "outputs": [
            "stage6_optimal_thresholds.csv",
            "stage6_fp_fn_analysis.csv",
            "stage6_slice_performance.csv",
            "stage6_calibration.csv",
            "stage6_notes.md",
        ],
    }
    with open(CHECKS / "stage6_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  stage6_manifest.json")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 6: Error Analysis and Diagnostics")
    parser.add_argument(
        "--target",
        choices=["missed_upcoming_emi", "future_dpd30", "both"],
        default="both",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    t_global = time.time()

    section(log, "STAGE 6: ERROR ANALYSIS AND DIAGNOSTICS")
    log.info(f"  Target(s) : {args.target}")
    log.info(f"  Analyses  : optimal thresholds, FP/FN, slice performance, calibration")
    sys.stdout.flush()

    col_manifest = load_manifest()
    numeric_cols    = col_manifest["numeric_feature_columns"]
    categorical_cols = col_manifest["categorical_feature_columns"]

    targets = TARGETS if args.target == "both" else [args.target]

    all_thresh, all_fpfn, all_slice, all_calib = [], [], [], []

    try:
        for i, target in enumerate(targets, 1):
            log.info(f"\n  *** TARGET {i}/{len(targets)}: {target} ***")
            sys.stdout.flush()
            t, f, s, c = process_target(target, numeric_cols, categorical_cols,
                                         args, log, t_global)
            all_thresh.extend(t)
            all_fpfn.extend(f)
            all_slice.extend(s)
            all_calib.extend(c)
    except KeyboardInterrupt:
        log.warning("Interrupted — saving partial results ...")
        sys.stdout.flush()

    save_reports(all_thresh, all_fpfn, all_slice, all_calib, log)

    section(log, "ALL DONE")
    log.info(f"  Total time : {elapsed(t_global)}")
    if all_thresh:
        log.info("\n  Optimal thresholds found:")
        for row in all_thresh:
            log.info(f"    {row['target']:30s}  {row['model']:4s}  threshold={row['optimal_threshold']:.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
