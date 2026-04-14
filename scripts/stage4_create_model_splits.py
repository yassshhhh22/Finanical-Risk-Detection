from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_parquet(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    logging.info("Loading %s ...", name)
    df = pd.read_parquet(path)
    logging.info("%s shape=%s", name, df.shape)
    return df


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    exclude = {"snapshot_id", "SK_ID_CURR", "SK_ID_PREV", "borrower_event_order", "snapshot_day", "snapshot_month_approx", "label", "split"}
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
    passthrough_cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            passthrough_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, passthrough_cols


def create_chronological_split_map(snapshot_days: pd.Series) -> pd.DataFrame:
    ordered_days = np.sort(snapshot_days.dropna().unique())
    if len(ordered_days) == 0:
        raise ValueError("No snapshot days available for split construction.")

    train_idx = int(np.floor(len(ordered_days) * TRAIN_FRAC))
    val_idx = int(np.floor(len(ordered_days) * (TRAIN_FRAC + VAL_FRAC)))

    train_idx = min(max(train_idx, 1), len(ordered_days) - 2)
    val_idx = min(max(val_idx, train_idx + 1), len(ordered_days) - 1)

    train_cutoff = ordered_days[train_idx - 1]
    val_cutoff = ordered_days[val_idx - 1]

    split_map = pd.DataFrame({"snapshot_day": ordered_days})
    split_map["split"] = np.where(
        split_map["snapshot_day"] <= train_cutoff,
        "train",
        np.where(split_map["snapshot_day"] <= val_cutoff, "validation", "test"),
    )
    return split_map


def apply_split(df: pd.DataFrame, split_map: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(split_map, on="snapshot_day", how="left", validate="many_to_one")
    if out["split"].isna().any():
        raise ValueError("Split assignment failed for some rows.")
    return out


def summarize_split(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    summary = (
        df.groupby("split", as_index=False)
        .agg(
            row_count=("label", "size"),
            positive_count=("label", "sum"),
            borrower_count=("SK_ID_CURR", "nunique"),
            snapshot_day_min=("snapshot_day", "min"),
            snapshot_day_max=("snapshot_day", "max"),
        )
        .sort_values("split")
        .reset_index(drop=True)
    )
    summary["positive_rate"] = summary["positive_count"] / summary["row_count"].replace({0: np.nan})

    borrower_overlap = {
        split: set(group["SK_ID_CURR"].astype("int64").tolist())
        for split, group in df.groupby("split", sort=False)
    }
    overlap_report = {
        "train_validation_overlap": len(borrower_overlap.get("train", set()) & borrower_overlap.get("validation", set())),
        "train_test_overlap": len(borrower_overlap.get("train", set()) & borrower_overlap.get("test", set())),
        "validation_test_overlap": len(borrower_overlap.get("validation", set()) & borrower_overlap.get("test", set())),
    }
    manifest = {
        "target_name": target_name,
        "row_count": int(len(df)),
        "split_summary": summary.to_dict(orient="records"),
        "borrower_overlap_counts": overlap_report,
    }
    return summary, manifest


def save_split_datasets(processed_dir: Path, df: pd.DataFrame, target_name: str) -> list[str]:
    filenames: list[str] = []
    for split_name, split_df in df.groupby("split", sort=False):
        output_name = f"{target_name}_{split_name}.parquet"
        split_df.to_parquet(processed_dir / output_name, index=False)
        filenames.append(output_name)
        logging.info("Saved: %s", processed_dir / output_name)
    full_name = f"{target_name}_all_splits.parquet"
    df.to_parquet(processed_dir / full_name, index=False)
    filenames.append(full_name)
    logging.info("Saved: %s", processed_dir / full_name)
    return filenames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 chronological split creation for Home Credit modeling tables.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    root = project_root()
    processed_dir = root / "artifacts" / "processed"
    raw_checks_dir = root / "artifacts" / "raw_checks"

    total_start = time.time()

    missed_df = load_parquet(processed_dir, "model_features_missed_upcoming_emi.parquet")
    dpd30_df = load_parquet(processed_dir, "model_features_future_dpd30.parquet")

    split_map = create_chronological_split_map(missed_df["snapshot_day"])
    split_map_path = raw_checks_dir / "stage4_snapshot_day_split_map.csv"
    split_map.to_csv(split_map_path, index=False)
    logging.info("Saved: %s", split_map_path)

    missed_split = apply_split(missed_df, split_map)
    dpd30_split = apply_split(dpd30_df, split_map)

    missed_summary, missed_manifest = summarize_split(missed_split, "model_features_missed_upcoming_emi")
    dpd30_summary, dpd30_manifest = summarize_split(dpd30_split, "model_features_future_dpd30")

    missed_files = save_split_datasets(processed_dir, missed_split, "model_features_missed_upcoming_emi")
    dpd30_files = save_split_datasets(processed_dir, dpd30_split, "model_features_future_dpd30")

    numeric_cols, categorical_cols, passthrough_cols = infer_column_types(missed_split)
    column_manifest = {
        "numeric_feature_columns": numeric_cols,
        "categorical_feature_columns": categorical_cols,
        "passthrough_columns": passthrough_cols,
    }
    column_manifest_path = raw_checks_dir / "stage4_column_manifest.json"
    column_manifest_path.write_text(json.dumps(column_manifest, indent=2), encoding="utf-8")
    logging.info("Saved: %s", column_manifest_path)

    split_summary = pd.concat(
        [
            missed_summary.assign(target_name="missed_upcoming_emi"),
            dpd30_summary.assign(target_name="future_dpd30"),
        ],
        ignore_index=True,
    )
    split_summary_path = raw_checks_dir / "stage4_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)
    logging.info("Saved: %s", split_summary_path)

    manifest = {
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
        "snapshot_day_split_map": split_map_path.name,
        "column_manifest": column_manifest_path.name,
        "split_summary": split_summary_path.name,
        "outputs": {
            "missed_upcoming_emi": missed_files,
            "future_dpd30": dpd30_files,
        },
        "target_manifests": [missed_manifest, dpd30_manifest],
    }
    manifest_path = raw_checks_dir / "stage4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Saved: %s", manifest_path)

    logging.info("Split summary:\n%s", split_summary.to_string(index=False))
    logging.info("Stage 4 completed in %.1fs", time.time() - total_start)


if __name__ == "__main__":
    main()
