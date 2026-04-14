"""
Stage 5: Centralized Baseline Modeling
=======================================
Trains Random Forest and LightGBM (Gradient Boosting) on both credit-risk targets:
  - missed_upcoming_emi
  - future_dpd30

Class imbalance handling: native class weighting (class_weight / scale_pos_weight).
No oversampling — preserves real data distribution, required for ethical evaluation
and compatible with federated learning (Stage 7).

Optimized for Intel CPU (Iris Xe / no CUDA):
  - Intel Extension for Scikit-Learn accelerates Random Forest if installed
  - LightGBM is faster than XGBoost on CPU and uses less memory

Usage:
  python scripts/stage5_train_models.py --verbose
  python scripts/stage5_train_models.py --target missed_upcoming_emi --skip-gb --verbose
  python scripts/stage5_train_models.py --target both --verbose
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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Intel Extension for Scikit-Learn — patch before any sklearn import
# ---------------------------------------------------------------------------
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    SKLEARNEX = True
except ImportError:
    SKLEARNEX = False

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "artifacts" / "processed"
MODELS = ROOT / "artifacts" / "models"
CHECKS = ROOT / "artifacts" / "raw_checks"
METRICS = ROOT / "artifacts" / "metrics"
MANIFEST_PATH = CHECKS / "stage4_column_manifest.json"

MODELS.mkdir(parents=True, exist_ok=True)
CHECKS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

TARGETS = ["missed_upcoming_emi", "future_dpd30"]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout,
                        force=True)
    # Force flush on every log line so PowerShell shows it immediately
    for h in logging.getLogger().handlers:
        h.flush = lambda: sys.stdout.flush()
    # Suppress noisy sklearnex/onedal debug output
    logging.getLogger("sklearnex").setLevel(logging.WARNING)
    logging.getLogger("onedal").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def mem() -> str:
    """Current process RAM usage as a readable string."""
    if not PSUTIL_AVAILABLE:
        return ""
    mb = psutil.Process().memory_info().rss / 1_048_576
    return f"  [RAM {mb:,.0f} MB]"


def elapsed(t0: float) -> str:
    return str(timedelta(seconds=int(time.time() - t0)))


def section(log: logging.Logger, title: str, step: str = ""):
    """Print a clearly visible section header."""
    prefix = f"[{step}] " if step else ""
    log.info("")
    log.info(f"{'─'*60}")
    log.info(f"  {prefix}{title}")
    log.info(f"{'─'*60}")


def tick(log: logging.Logger, msg: str, t0: float):
    """Log a message with elapsed time and memory."""
    log.info(f"  {msg}  (+{elapsed(t0)}){mem()}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Load column manifest
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load split parquet
# ---------------------------------------------------------------------------
def load_split(target: str, split: str, numeric_cols: list, categorical_cols: list,
               log: logging.Logger) -> tuple:
    path = PROCESSED / f"model_features_{target}_{split}.parquet"
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"  Reading {split} ({size_mb:.0f} MB) ...")
    t0 = time.time()
    needed = numeric_cols + categorical_cols + ["label"]
    df = pd.read_parquet(path, columns=needed)
    y = df["label"].values.astype(np.int8)
    X = df.drop(columns=["label"])
    log.info(
        f"  {split:12s} shape={X.shape}  "
        f"positive={y.mean():.4f} ({y.sum():,}/{len(y):,})  "
        f"loaded in {time.time()-t0:.1f}s{mem()}"
    )
    sys.stdout.flush()
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, split, model, target) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "target": target, "model": model, "split": split,
        "n_samples": len(y_true),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_confusion(y_true, y_prob, threshold, split, model, target) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "target": target, "model": model, "split": split,
        "threshold": threshold,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def compute_threshold_summary(y_true, y_prob, split, model, target) -> list:
    rows = []
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        y_pred = (y_prob >= thresh).astype(int)
        rows.append({
            "target": target, "model": model, "split": split,
            "threshold": thresh, "threshold_type": "fixed",
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        })
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
    best_idx = np.argmax(f1_arr[:-1])
    best_thresh = float(thresh_arr[best_idx])
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    rows.append({
        "target": target, "model": model, "split": split,
        "threshold": best_thresh, "threshold_type": "optimal_f1",
        "f1": float(f1_score(y_true, y_pred_opt, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_opt, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_opt, zero_division=0)),
    })
    return rows


def evaluate(model, model_name, X_val, y_val, X_test, y_test, target, log, t_global) -> tuple:
    metrics_rows, confusion_rows, threshold_rows = [], [], []
    for split_name, X_ev, y_ev in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        log.info(f"  [{model_name}] scoring {split_name} ({len(y_ev):,} rows) ...")
        sys.stdout.flush()
        t0 = time.time()
        y_prob = model.predict_proba(X_ev)[:, 1]
        metrics_rows.append(compute_metrics(y_ev, y_prob, split_name, model_name, target))
        confusion_rows.append(compute_confusion(y_ev, y_prob, 0.5, split_name, model_name, target))
        threshold_rows.extend(compute_threshold_summary(y_ev, y_prob, split_name, model_name, target))
        m = metrics_rows[-1]
        log.info(
            f"  [{model_name}] {split_name:10s} → "
            f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
            f"F1={m['f1']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  "
            f"scored in {time.time()-t0:.1f}s  [total {elapsed(t_global)}]"
        )
        sys.stdout.flush()
    return metrics_rows, confusion_rows, threshold_rows


# ---------------------------------------------------------------------------
# LightGBM callback: logs every N iterations with metric + elapsed time
# ---------------------------------------------------------------------------
def lgb_logging_callback(log: logging.Logger, t0: float, period: int = 50):
    def _callback(env):
        if env.iteration % period == 0 or env.iteration == env.end_iteration - 1:
            metrics_str = "  ".join(
                f"{entry[1]}={entry[2]:.5f}" for entry in env.evaluation_result_list
            )
            log.info(
                f"  [gb] iter {env.iteration:>4}/{env.end_iteration}  "
                f"{metrics_str}  (+{elapsed(t0)}){mem()}"
            )
            sys.stdout.flush()
    _callback.order = 10
    return _callback


# ---------------------------------------------------------------------------
# Random Forest with incremental batch logging (warm_start trick)
# ---------------------------------------------------------------------------
def fit_rf_with_progress(rf_params: dict, X, y: np.ndarray,
                          log: logging.Logger) -> RandomForestClassifier:
    total = rf_params["n_estimators"]
    t0 = time.time()
    log.info(f"  [rf] fitting {total} trees ...")
    sys.stdout.flush()
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X, y)
    rate = total / max(time.time() - t0, 1)
    log.info(f"  [rf] DONE  {total} trees  rate={rate:.1f} trees/s  +{elapsed(t0)}{mem()}")
    sys.stdout.flush()
    return rf


# ---------------------------------------------------------------------------
# Process one target
# ---------------------------------------------------------------------------
def process_target(target, numeric_cols, categorical_cols, args, log, t_global) -> tuple:
    section(log, f"TARGET: {target.upper()}")

    # Step 1: Load data
    section(log, "Loading data splits", step="1/5")
    X_train, y_train = load_split(target, "train",      numeric_cols, categorical_cols, log)
    X_val,   y_val   = load_split(target, "validation", numeric_cols, categorical_cols, log)
    X_test,  y_test  = load_split(target, "test",       numeric_cols, categorical_cols, log)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / n_pos
    log.info(f"  Class ratio neg/pos = {pos_weight:.1f}  (→ scale_pos_weight for GB)")
    sys.stdout.flush()

    # Step 2: Preprocess
    section(log, "Preprocessing (impute + encode)", step="2/5")
    t0 = time.time()
    log.info("  Fitting on train, transforming all splits ...")
    sys.stdout.flush()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_prep = preprocessor.fit_transform(X_train)
    log.info(f"  train transformed  shape={X_train_prep.shape}  (+{elapsed(t0)}){mem()}")
    sys.stdout.flush()
    X_val_prep   = preprocessor.transform(X_val)
    log.info(f"  val   transformed  shape={X_val_prep.shape}  (+{elapsed(t0)}){mem()}")
    sys.stdout.flush()
    X_test_prep  = preprocessor.transform(X_test)
    log.info(f"  test  transformed  shape={X_test_prep.shape}  (+{elapsed(t0)}){mem()}")
    sys.stdout.flush()

    prep_path = MODELS / f"preprocessor_{target}.pkl"
    joblib.dump(preprocessor, prep_path)
    log.info(f"  preprocessor saved → {prep_path.name}  (+{elapsed(t0)})")
    sys.stdout.flush()

    del X_train, X_val, X_test
    gc.collect()

    all_metrics, all_confusions, all_thresholds = [], [], []

    # Step 3: Random Forest
    if not args.skip_rf:
        rf_path = MODELS / f"rf_{target}.pkl"
        section(log, "Random Forest  (class_weight='balanced')", step="3/5")

        if rf_path.exists() and not args.retrain:
            log.info(f"  CHECKPOINT: {rf_path.name} already exists — loading and skipping training")
            log.info(f"  (delete the file or use --retrain to force retraining)")
            sys.stdout.flush()
            rf = joblib.load(rf_path)
        else:
            log.info(f"  n_estimators=300  max_depth=15  n_jobs=-1")
            log.info(f"  Training on {len(y_train):,} rows — logging every 30 trees ...")
            sys.stdout.flush()

            rf_params = dict(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=100,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            t0 = time.time()
            rf = fit_rf_with_progress(rf_params, X_train_prep, y_train, log)
            log.info(f"  RF fit complete in {elapsed(t0)}{mem()}")
            sys.stdout.flush()
            joblib.dump(rf, rf_path)
            log.info(f"  RF model saved → {rf_path.name}")
            sys.stdout.flush()

        section(log, "Evaluating RF", step="3b/5")
        m, c, t = evaluate(rf, "rf", X_val_prep, y_val, X_test_prep, y_test, target, log, t_global)
        all_metrics.extend(m)
        all_confusions.extend(c)
        all_thresholds.extend(t)
        append_metrics(m, c, t, log)
        del rf
        gc.collect()

    # Step 4: LightGBM
    if not args.skip_gb:
        if not LIGHTGBM_AVAILABLE:
            log.warning("LightGBM not installed — skipping GB. Run: pip install lightgbm")
        else:
            gb_path = MODELS / f"gb_{target}.pkl"
            section(log, f"LightGBM  (scale_pos_weight={pos_weight:.1f})", step="4/5")

            if gb_path.exists() and not args.retrain:
                log.info(f"  CHECKPOINT: {gb_path.name} already exists — loading and skipping training")
                log.info(f"  (delete the file or use --retrain to force retraining)")
                sys.stdout.flush()
                gb = joblib.load(gb_path)
            else:
                log.info(f"  n_estimators=1000  lr=0.05  max_depth=6  early_stopping=50")
                log.info(f"  Training on {len(y_train):,} rows — logging every 50 iterations ...")
                sys.stdout.flush()

                gb = lgb.LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=pos_weight,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                )
                t0 = time.time()
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False, first_metric_only=True),
                    lgb_logging_callback(log, t0, period=50),
                ]
                gb.fit(
                    X_train_prep, y_train,
                    eval_set=[(X_val_prep, y_val)],
                    eval_metric="average_precision",
                    callbacks=callbacks,
                )
                log.info(
                    f"  LightGBM fit complete — best iteration={gb.best_iteration_}  "
                    f"total time {elapsed(t0)}{mem()}"
                )
                sys.stdout.flush()
                joblib.dump(gb, gb_path)
                log.info(f"  GB model saved → {gb_path.name}")
                sys.stdout.flush()

            section(log, "Evaluating LightGBM", step="4b/5")
            m, c, t = evaluate(gb, "gb", X_val_prep, y_val, X_test_prep, y_test, target, log, t_global)
            all_metrics.extend(m)
            all_confusions.extend(c)
            all_thresholds.extend(t)
            append_metrics(m, c, t, log)
            del gb
            gc.collect()

    del X_train_prep, y_train, X_val_prep, y_val, X_test_prep, y_test
    gc.collect()

    return all_metrics, all_confusions, all_thresholds


# ---------------------------------------------------------------------------
# Incremental metrics append — called after every single model finishes
# so Ctrl+C never loses completed work
# ---------------------------------------------------------------------------
def append_metrics(metrics_rows: list, confusion_rows: list, threshold_rows: list, log: logging.Logger):
    """Append rows to CSVs immediately after each model finishes."""
    for path, rows in [
        (METRICS / "stage5_metrics.csv",            metrics_rows),
        (METRICS / "stage5_confusion_matrices.csv", confusion_rows),
        (METRICS / "stage5_threshold_summary.csv",  threshold_rows),
    ]:
        if not rows:
            continue
        df_new = pd.DataFrame(rows)
        if path.exists():
            df_new.to_csv(path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(path, index=False)
    log.info("  metrics checkpointed to CSV")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Save reports (final pass — rewrites with full sorted content)
# ---------------------------------------------------------------------------
def save_reports(all_metrics, all_confusions, all_thresholds, log):
    section(log, "Saving reports", step="5/5")

    pd.DataFrame(all_metrics).to_csv(METRICS / "stage5_metrics.csv", index=False)
    log.info("  stage5_metrics.csv")

    pd.DataFrame(all_confusions).to_csv(METRICS / "stage5_confusion_matrices.csv", index=False)
    log.info("  stage5_confusion_matrices.csv")

    pd.DataFrame(all_thresholds).to_csv(METRICS / "stage5_threshold_summary.csv", index=False)
    log.info("  stage5_threshold_summary.csv")

    df = pd.DataFrame(all_metrics)
    with open(CHECKS / "stage5_notes.md", "w") as f:
        f.write("# Stage 5 Notes\n\n")
        f.write(f"Run timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"Intel sklearnex active: {SKLEARNEX}\n")
        f.write(f"LightGBM available: {LIGHTGBM_AVAILABLE}\n")
        f.write("Imbalance handling: class_weight='balanced' (RF), scale_pos_weight (LightGBM)\n\n")
        f.write("## Metric Summary\n\n")
        if not df.empty:
            f.write(df.to_csv(index=False))
        f.write("\n")
    log.info("  stage5_notes.md")

    manifest = {
        "stage": 5,
        "timestamp": datetime.now().isoformat(),
        "intel_sklearnex": SKLEARNEX,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "imbalance_method": "class_weight / scale_pos_weight",
        "outputs": {
            "models": [p.name for p in MODELS.glob("*.pkl")],
            "metrics": "stage5_metrics.csv",
        },
    }
    with open(CHECKS / "stage5_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  stage5_manifest.json")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 5: Centralized Baseline Modeling")
    parser.add_argument(
        "--target",
        choices=["missed_upcoming_emi", "future_dpd30", "both"],
        default="both",
    )
    parser.add_argument("--skip-rf", action="store_true", help="Skip Random Forest")
    parser.add_argument("--skip-gb", action="store_true", help="Skip Gradient Boosting")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore checkpoints and retrain all models from scratch")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    t_global = time.time()

    section(log, "STAGE 5: CENTRALIZED BASELINE MODELING")
    log.info(f"  Target(s)        : {args.target}")
    log.info(f"  Imbalance method : class_weight (RF) + scale_pos_weight (LightGBM)")
    log.info(f"  Intel sklearnex  : {'YES — RF accelerated' if SKLEARNEX else 'NO  (pip install scikit-learn-intelex)'}")
    log.info(f"  LightGBM         : {'YES' if LIGHTGBM_AVAILABLE else 'NO  (pip install lightgbm)'}")
    log.info(f"  psutil memory    : {'YES' if PSUTIL_AVAILABLE else 'NO  (pip install psutil)'}")
    log.info(f"  Skip RF          : {args.skip_rf}")
    log.info(f"  Skip GB          : {args.skip_gb}")
    sys.stdout.flush()

    manifest = load_manifest()
    numeric_cols = manifest["numeric_feature_columns"]
    categorical_cols = manifest["categorical_feature_columns"]
    log.info(f"  Features         : {len(numeric_cols)} numeric + {len(categorical_cols)} categorical")
    sys.stdout.flush()

    targets = TARGETS if args.target == "both" else [args.target]

    all_metrics, all_confusions, all_thresholds = [], [], []
    try:
        for i, target in enumerate(targets, 1):
            log.info(f"\n  *** TARGET {i}/{len(targets)}: {target} ***")
            sys.stdout.flush()
            m, c, t = process_target(target, numeric_cols, categorical_cols, args, log, t_global)
            all_metrics.extend(m)
            all_confusions.extend(c)
            all_thresholds.extend(t)
    except KeyboardInterrupt:
        log.warning("")
        log.warning("Interrupted by user (Ctrl+C)")
        log.warning("All completed models are already saved in artifacts/models/")
        log.warning("Metrics up to this point are saved in artifacts/metrics/stage5_metrics.csv")
        log.warning("Re-run without --retrain to resume from checkpoints automatically")
        sys.stdout.flush()

    if all_metrics:
        save_reports(all_metrics, all_confusions, all_thresholds, log)
    else:
        log.warning("No metrics to save — nothing completed before interruption")

    section(log, "ALL DONE")
    log.info(f"  Total time : {elapsed(t_global)}")
    log.info(f"  Models     : {MODELS}")
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        log.info("\n  Final metrics summary:")
        for _, row in df.iterrows():
            log.info(
                f"    {row['target']:30s} {row['model']:4s} {row['split']:12s} "
                f"ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  F1={row['f1']:.4f}"
            )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
