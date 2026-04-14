"""
Stage 7: Simulated Federated Learning
=======================================
Simulates a federated learning setup using the Home Credit dataset.

Design:
  - K=5 pseudo-clients, each holding a random stratified partition of the train set
  - Two federation strategies:
      RF  → FedForest: each client trains n_estimators/K trees, aggregate by
            combining all trees into one forest (trees are independent, valid)
      GB  → FedEnsemble: each client trains a local LightGBM, aggregate by
            averaging predictions (analogue of federated averaging for GBMs)
  - Three-way comparison:
      Centralized  — Stage 5 model trained on full dataset
      Local best   — best single-client model on global test set
      Federated    — aggregated model on global test set

Outputs:
    artifacts/metrics/stage7_client_metrics.csv         per-client performance
    artifacts/metrics/stage7_comparison.csv             centralized vs federated vs local
  artifacts/raw_checks/stage7_notes.md
  artifacts/raw_checks/stage7_manifest.json

Usage:
  python scripts/stage7_federated_learning.py --verbose
  python scripts/stage7_federated_learning.py --target missed_upcoming_emi --n-clients 5 --verbose
"""

import argparse
import gc
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    SKLEARNEX = True
except ImportError:
    SKLEARNEX = False

from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "artifacts" / "processed"
MODELS    = ROOT / "artifacts" / "models"
CHECKS    = ROOT / "artifacts" / "raw_checks"
METRICS   = ROOT / "artifacts" / "metrics"
MANIFEST  = CHECKS / "stage4_column_manifest.json"
THRESH_CSV = METRICS / "stage6_optimal_thresholds.csv"

TARGETS = ["missed_upcoming_emi", "future_dpd30"]

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
    log = logging.getLogger(__name__)
    log.info(f"Intel sklearnex : {'YES' if SKLEARNEX else 'NO'}")
    log.info(f"LightGBM        : {'YES' if LIGHTGBM_AVAILABLE else 'NO'}")
    return log


def elapsed(t0: float) -> str:
    return str(timedelta(seconds=int(time.time() - t0)))


def mem() -> str:
    if not PSUTIL_AVAILABLE:
        return ""
    mb = psutil.Process().memory_info().rss / 1_048_576
    return f"  [RAM {mb:,.0f} MB]"


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


def load_optimal_thresholds() -> dict:
    """Returns {(target, model): threshold}"""
    if not THRESH_CSV.exists():
        return {}
    df = pd.read_csv(THRESH_CSV)
    return {(row.target, row.model): row.optimal_threshold for row in df.itertuples()}


def load_split(target: str, split: str, numeric_cols: list,
               categorical_cols: list, log: logging.Logger) -> tuple:
    path = PROCESSED / f"model_features_{target}_{split}.parquet"
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"  Reading {split} ({size_mb:.0f} MB) ...")
    sys.stdout.flush()
    t0 = time.time()
    needed = numeric_cols + categorical_cols + ["label"]
    df = pd.read_parquet(path, columns=needed)
    y = df["label"].values.astype(np.int8)
    X = df.drop(columns=["label"])
    log.info(f"  {split:12s} shape={X.shape}  positive={y.mean():.4f}  loaded in {time.time()-t0:.1f}s{mem()}")
    sys.stdout.flush()
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: float, label: str, model: str,
                          target: str, client: str) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "target": target, "model": model, "client": client, "label": label,
        "threshold": threshold,
        "n": len(y_true),
        "positive_rate": float(y_true.mean()),
        "roc_auc":  float(roc_auc_score(y_true, y_prob)),
        "pr_auc":   float(average_precision_score(y_true, y_prob)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":   float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Client partitioning
# ---------------------------------------------------------------------------
def make_client_splits(X: pd.DataFrame, y: np.ndarray,
                        n_clients: int, seed: int = 42) -> list:
    """
    Stratified random partition of (X, y) into n_clients.
    Each client gets a roughly equal share of positives and negatives.
    Returns list of (X_client, y_client) tuples.
    """
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    clients = []
    pos_splits = np.array_split(pos_idx, n_clients)
    neg_splits = np.array_split(neg_idx, n_clients)

    for i in range(n_clients):
        idx = np.concatenate([pos_splits[i], neg_splits[i]])
        rng.shuffle(idx)
        clients.append((X.iloc[idx], y[idx]))

    return clients


# ---------------------------------------------------------------------------
# Federated RF
# ---------------------------------------------------------------------------
def train_federated_rf(client_splits: list, preprocessor,
                        rf_base_params: dict, n_clients: int,
                        log: logging.Logger, t_global: float) -> RandomForestClassifier:
    """
    FedForest: each client trains trees_per_client trees.
    Aggregate by combining all estimators_ into one forest.
    Trees are independent so this is a valid federated RF.
    """
    trees_per_client = max(1, rf_base_params["n_estimators"] // n_clients)
    log.info(f"  FedForest: {n_clients} clients × {trees_per_client} trees = "
             f"{trees_per_client * n_clients} total trees")
    sys.stdout.flush()

    local_rfs = []
    for i, (X_c, y_c) in enumerate(client_splits, 1):
        log.info(f"  Client {i}/{n_clients}: {len(y_c):,} rows  "
                 f"positive={y_c.mean():.4f} ...")
        sys.stdout.flush()
        t0 = time.time()
        X_prep = preprocessor.transform(X_c)
        params = {**rf_base_params, "n_estimators": trees_per_client}
        rf_c = RandomForestClassifier(**params)
        rf_c.fit(X_prep, y_c)
        log.info(f"  Client {i}/{n_clients}: done in {time.time()-t0:.1f}s{mem()}")
        sys.stdout.flush()
        local_rfs.append(rf_c)
        del X_prep
        gc.collect()

    # Aggregate: combine all trees
    log.info(f"  Aggregating {n_clients} local RF models ...")
    sys.stdout.flush()
    fed_rf = deepcopy(local_rfs[0])
    fed_rf.estimators_ = []
    for rf_c in local_rfs:
        fed_rf.estimators_.extend(rf_c.estimators_)
    fed_rf.n_estimators = len(fed_rf.estimators_)
    log.info(f"  FedForest has {fed_rf.n_estimators} total trees  +{elapsed(t_global)}")
    sys.stdout.flush()

    return fed_rf, local_rfs


# ---------------------------------------------------------------------------
# Federated GB
# ---------------------------------------------------------------------------
def train_federated_gb(client_splits: list, preprocessor,
                        gb_base_params: dict, n_clients: int,
                        pos_weight: float, log: logging.Logger,
                        t_global: float) -> tuple:
    """
    FedEnsemble: each client trains a local LightGBM.
    Aggregate: average predictions across all local models.
    This is the GBM analogue of federated averaging.
    """
    log.info(f"  FedEnsemble: {n_clients} clients, predictions averaged at inference")
    sys.stdout.flush()

    local_gbs = []
    for i, (X_c, y_c) in enumerate(client_splits, 1):
        log.info(f"  Client {i}/{n_clients}: {len(y_c):,} rows  "
                 f"positive={y_c.mean():.4f} ...")
        sys.stdout.flush()
        t0 = time.time()
        X_prep = preprocessor.transform(X_c)
        gb_c = lgb.LGBMClassifier(**{**gb_base_params, "scale_pos_weight": pos_weight})
        gb_c.fit(X_prep, y_c)
        log.info(f"  Client {i}/{n_clients}: done in {time.time()-t0:.1f}s{mem()}")
        sys.stdout.flush()
        local_gbs.append(gb_c)
        del X_prep
        gc.collect()

    log.info(f"  {n_clients} local GB models trained  +{elapsed(t_global)}")
    sys.stdout.flush()
    return local_gbs


def predict_federated_gb(local_gbs: list, X_prep: np.ndarray) -> np.ndarray:
    """Average predictions across all local GB models."""
    probs = np.stack([gb.predict_proba(X_prep)[:, 1] for gb in local_gbs], axis=0)
    return probs.mean(axis=0)


# ---------------------------------------------------------------------------
# Process one target
# ---------------------------------------------------------------------------
def process_target(target: str, numeric_cols: list, categorical_cols: list,
                   optimal_thresholds: dict, args, log: logging.Logger,
                   t_global: float) -> tuple:
    section(log, f"TARGET: {target.upper()}")

    # Load data
    section(log, "Loading data", step="1/4")
    X_train, y_train = load_split(target, "train",      numeric_cols, categorical_cols, log)
    X_test,  y_test  = load_split(target, "test",       numeric_cols, categorical_cols, log)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / n_pos
    log.info(f"  Class ratio neg/pos = {pos_weight:.1f}")
    sys.stdout.flush()

    # Load preprocessor (fitted in Stage 5 on full train — reuse it)
    preprocessor = joblib.load(MODELS / f"preprocessor_{target}.pkl")
    log.info(f"  Preprocessor loaded (fitted on full train set)")

    # Preprocess test set once
    log.info(f"  Preprocessing test set ...")
    sys.stdout.flush()
    X_test_prep = preprocessor.transform(X_test)
    del X_test
    gc.collect()

    # Partition train into client splits
    section(log, f"Partitioning into {args.n_clients} clients", step="2/4")
    client_splits = make_client_splits(X_train, y_train, args.n_clients)
    for i, (X_c, y_c) in enumerate(client_splits, 1):
        log.info(f"  Client {i}: {len(y_c):,} rows  positive={y_c.mean():.4f}")
    del X_train, y_train
    gc.collect()
    sys.stdout.flush()

    # Load centralized models from Stage 5
    central_rf = joblib.load(MODELS / f"rf_{target}.pkl")
    central_gb = joblib.load(MODELS / f"gb_{target}.pkl") if LIGHTGBM_AVAILABLE else None

    thresh_rf = optimal_thresholds.get((target, "rf"), 0.5)
    thresh_gb = optimal_thresholds.get((target, "gb"), 0.5)

    client_rows = []
    comparison_rows = []

    # -----------------------------------------------------------------------
    # RF: FedForest
    # -----------------------------------------------------------------------
    section(log, "Random Forest — FedForest", step="3/4")

    rf_base_params = dict(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    fed_rf, local_rfs = train_federated_rf(
        client_splits, preprocessor, rf_base_params, args.n_clients, log, t_global
    )

    # Evaluate centralized RF
    log.info("  Evaluating centralized RF on test ...")
    sys.stdout.flush()
    y_prob_central_rf = central_rf.predict_proba(X_test_prep)[:, 1]
    comparison_rows.append({
        **metrics_at_threshold(y_test, y_prob_central_rf, thresh_rf,
                               "centralized", "rf", target, "all"),
        "federation": "centralized",
    })

    # Evaluate federated RF
    log.info("  Evaluating federated RF on test ...")
    sys.stdout.flush()
    y_prob_fed_rf = fed_rf.predict_proba(X_test_prep)[:, 1]
    comparison_rows.append({
        **metrics_at_threshold(y_test, y_prob_fed_rf, thresh_rf,
                               "federated", "rf", target, "all"),
        "federation": "federated",
    })

    # Evaluate each local RF
    local_best_roc = 0
    for i, rf_c in enumerate(local_rfs, 1):
        log.info(f"  Evaluating local RF client {i}/{args.n_clients} on global test ...")
        sys.stdout.flush()
        X_c_prep = preprocessor.transform(client_splits[i-1][0])
        y_prob_local = rf_c.predict_proba(X_test_prep)[:, 1]
        row = metrics_at_threshold(y_test, y_prob_local, thresh_rf,
                                    f"local_client_{i}", "rf", target, f"client_{i}")
        row["federation"] = "local"
        client_rows.append(row)
        comparison_rows.append(row)
        if row["roc_auc"] > local_best_roc:
            local_best_roc = row["roc_auc"]
        del X_c_prep, y_prob_local
        gc.collect()

    log.info(
        f"  RF comparison → "
        f"Centralized ROC-AUC={comparison_rows[-args.n_clients-1]['roc_auc']:.4f}  "
        f"Federated ROC-AUC={comparison_rows[-args.n_clients]['roc_auc']:.4f}  "
        f"Best local={local_best_roc:.4f}"
    )
    sys.stdout.flush()

    del fed_rf, local_rfs, central_rf, y_prob_central_rf, y_prob_fed_rf
    gc.collect()

    # -----------------------------------------------------------------------
    # GB: FedEnsemble
    # -----------------------------------------------------------------------
    if LIGHTGBM_AVAILABLE and central_gb is not None:
        section(log, "LightGBM — FedEnsemble", step="4/4")

        gb_base_params = dict(
            n_estimators=200,   # fewer per client for speed
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

        local_gbs = train_federated_gb(
            client_splits, preprocessor, gb_base_params,
            args.n_clients, pos_weight, log, t_global
        )

        # Evaluate centralized GB
        log.info("  Evaluating centralized GB on test ...")
        sys.stdout.flush()
        y_prob_central_gb = central_gb.predict_proba(X_test_prep)[:, 1]
        comparison_rows.append({
            **metrics_at_threshold(y_test, y_prob_central_gb, thresh_gb,
                                   "centralized", "gb", target, "all"),
            "federation": "centralized",
        })

        # Evaluate federated GB (averaged predictions)
        log.info("  Evaluating federated GB (averaged predictions) on test ...")
        sys.stdout.flush()
        y_prob_fed_gb = predict_federated_gb(local_gbs, X_test_prep)
        comparison_rows.append({
            **metrics_at_threshold(y_test, y_prob_fed_gb, thresh_gb,
                                   "federated", "gb", target, "all"),
            "federation": "federated",
        })

        # Evaluate each local GB
        local_best_roc_gb = 0
        for i, gb_c in enumerate(local_gbs, 1):
            log.info(f"  Evaluating local GB client {i}/{args.n_clients} on global test ...")
            sys.stdout.flush()
            y_prob_local = gb_c.predict_proba(X_test_prep)[:, 1]
            row = metrics_at_threshold(y_test, y_prob_local, thresh_gb,
                                        f"local_client_{i}", "gb", target, f"client_{i}")
            row["federation"] = "local"
            client_rows.append(row)
            comparison_rows.append(row)
            if row["roc_auc"] > local_best_roc_gb:
                local_best_roc_gb = row["roc_auc"]
            del y_prob_local
            gc.collect()

        log.info(
            f"  GB comparison → "
            f"Centralized ROC-AUC={y_prob_central_gb.shape}  "
            f"Federated better than local best={local_best_roc_gb:.4f}"
        )
        sys.stdout.flush()
        del local_gbs, central_gb, y_prob_central_gb, y_prob_fed_gb
        gc.collect()

    del X_test_prep, y_test, client_splits
    gc.collect()

    return client_rows, comparison_rows


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------
def save_reports(all_client_rows: list, all_comparison_rows: list, log: logging.Logger):
    section(log, "Saving reports")

    METRICS.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_client_rows).to_csv(METRICS / "stage7_client_metrics.csv", index=False)
    log.info("  stage7_client_metrics.csv")

    pd.DataFrame(all_comparison_rows).to_csv(METRICS / "stage7_comparison.csv", index=False)
    log.info("  stage7_comparison.csv")

    # Summary note
    df = pd.DataFrame(all_comparison_rows)
    notes = ["# Stage 7 Federated Learning Notes\n\n"]
    notes.append(f"Run timestamp: {datetime.now().isoformat()}\n\n")
    notes.append("## Centralized vs Federated vs Local (ROC-AUC on global test)\n\n")

    for target in df["target"].unique():
        notes.append(f"### {target}\n\n")
        sub = df[df["target"] == target]
        for model in sub["model"].unique():
            notes.append(f"**{model}:**\n")
            msub = sub[sub["model"] == model]
            central = msub[msub["federation"] == "centralized"]
            federated = msub[msub["federation"] == "federated"]
            local = msub[msub["federation"] == "local"]
            if not central.empty:
                notes.append(f"  Centralized : ROC-AUC={central['roc_auc'].values[0]:.4f}  F1={central['f1'].values[0]:.4f}\n")
            if not federated.empty:
                notes.append(f"  Federated   : ROC-AUC={federated['roc_auc'].values[0]:.4f}  F1={federated['f1'].values[0]:.4f}\n")
            if not local.empty:
                notes.append(f"  Local best  : ROC-AUC={local['roc_auc'].max():.4f}  Local worst: {local['roc_auc'].min():.4f}\n")
            notes.append("\n")

    with open(CHECKS / "stage7_notes.md", "w") as f:
        f.write("".join(notes))
    log.info("  stage7_notes.md")

    manifest = {
        "stage": 7,
        "timestamp": datetime.now().isoformat(),
        "n_clients": None,
        "rf_strategy": "FedForest (combine trees)",
        "gb_strategy": "FedEnsemble (average predictions)",
        "outputs": ["stage7_client_metrics.csv", "stage7_comparison.csv", "stage7_notes.md"],
    }
    with open(CHECKS / "stage7_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  stage7_manifest.json")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 7: Simulated Federated Learning")
    parser.add_argument(
        "--target",
        choices=["missed_upcoming_emi", "future_dpd30", "both"],
        default="both",
    )
    parser.add_argument(
        "--n-clients", type=int, default=5,
        help="Number of pseudo-clients to simulate (default: 5)"
    )
    parser.add_argument("--skip-gb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    t_global = time.time()

    section(log, "STAGE 7: SIMULATED FEDERATED LEARNING")
    log.info(f"  Target(s)     : {args.target}")
    log.info(f"  Clients       : {args.n_clients}")
    log.info(f"  RF strategy   : FedForest (combine trees from all clients)")
    log.info(f"  GB strategy   : FedEnsemble (average predictions)")
    sys.stdout.flush()

    col_manifest = load_manifest()
    numeric_cols    = col_manifest["numeric_feature_columns"]
    categorical_cols = col_manifest["categorical_feature_columns"]
    optimal_thresholds = load_optimal_thresholds()

    if not optimal_thresholds:
        log.warning("Stage 6 thresholds not found — using 0.5 for all models")

    targets = TARGETS if args.target == "both" else [args.target]

    all_client_rows, all_comparison_rows = [], []
    try:
        for i, target in enumerate(targets, 1):
            log.info(f"\n  *** TARGET {i}/{len(targets)}: {target} ***")
            sys.stdout.flush()
            c, comp = process_target(
                target, numeric_cols, categorical_cols,
                optimal_thresholds, args, log, t_global
            )
            all_client_rows.extend(c)
            all_comparison_rows.extend(comp)
    except KeyboardInterrupt:
        log.warning("Interrupted — saving partial results ...")

    save_reports(all_client_rows, all_comparison_rows, log)

    section(log, "ALL DONE")
    log.info(f"  Total time : {elapsed(t_global)}")

    # Print final comparison table
    if all_comparison_rows:
        df = pd.DataFrame(all_comparison_rows)
        log.info("\n  Centralized vs Federated summary (ROC-AUC on test):")
        for target in df["target"].unique():
            sub = df[(df["target"] == target) & (df["federation"].isin(["centralized", "federated"]))]
            for _, row in sub.iterrows():
                log.info(
                    f"    {row['target']:30s} {row['model']:4s} "
                    f"{row['federation']:12s} ROC-AUC={row['roc_auc']:.4f}  "
                    f"F1={row['f1']:.4f}"
                )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
