from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_parquet(processed_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = processed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    logging.info("Loading %s ...", name)
    df = pd.read_parquet(path, columns=columns)
    logging.info("%s shape=%s", name, df.shape)
    return df


def maybe_load_cached_feature_family(processed_dir: Path, cache_name: str) -> pd.DataFrame | None:
    path = processed_dir / cache_name
    if path.exists():
        logging.info("Loading cached feature family %s ...", cache_name)
        df = pd.read_parquet(path)
        logging.info("%s shape=%s", cache_name, df.shape)
        return df
    return None


def save_cached_feature_family(processed_dir: Path, cache_name: str, df: pd.DataFrame) -> Path:
    path = processed_dir / cache_name
    df.to_parquet(path, index=False)
    logging.info("Saved cached feature family: %s", path)
    return path


def build_application_features(app: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    logging.info("Building application features...")
    t0 = time.time()
    app = app.copy()

    categorical_cols = [
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
    ]
    for col in categorical_cols:
        app[col] = app[col].fillna("missing").astype("category")

    feature_cols = [
        "SK_ID_CURR",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "DAYS_EMPLOYED_ANOMALY_FLAG",
        "CREDIT_TO_INCOME_RATIO",
        "ANNUITY_TO_INCOME_RATIO",
        "GOODS_TO_CREDIT_RATIO",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        *categorical_cols,
    ]
    features = app[feature_cols].drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)
    logging.info("Application features built in %.1fs", time.time() - t0)

    metadata = [
        {"feature_name": col, "feature_family": "application", "source_table": "cleaned_application_train.parquet"}
        for col in feature_cols
        if col != "SK_ID_CURR"
    ]
    return features, metadata


def build_previous_application_features(prev: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    logging.info("Building previous application aggregates...")
    t0 = time.time()
    prev = prev.copy()
    prev["is_approved"] = (prev["status_group"] == "approved").astype("int8")
    prev["is_rejected"] = (prev["status_group"] == "rejected").astype("int8")
    prev["is_active_prev"] = (prev["status_group"] == "active").astype("int8")

    grouped = prev.groupby("SK_ID_CURR", as_index=False).agg(
        prev_application_count=("SK_ID_PREV", "count"),
        prev_approved_count=("is_approved", "sum"),
        prev_rejected_count=("is_rejected", "sum"),
        prev_active_count=("is_active_prev", "sum"),
        prev_credit_mean=("AMT_CREDIT", "mean"),
        prev_credit_max=("AMT_CREDIT", "max"),
        prev_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_down_payment_mean=("AMT_DOWN_PAYMENT", "mean"),
        prev_days_decision_max=("DAYS_DECISION", "max"),
        prev_days_decision_min=("DAYS_DECISION", "min"),
    )
    grouped["prev_approval_rate"] = grouped["prev_approved_count"] / grouped["prev_application_count"].replace({0: np.nan})
    logging.info("Previous application aggregates built in %.1fs", time.time() - t0)

    metadata = [
        {"feature_name": col, "feature_family": "previous_application", "source_table": "cleaned_previous_application.parquet"}
        for col in grouped.columns
        if col != "SK_ID_CURR"
    ]
    return grouped, metadata


def build_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    logging.info("Building bureau aggregates...")
    t0 = time.time()
    bureau = bureau.copy()
    bureau["bureau_active_flag"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")
    bureau["bureau_overdue_flag"] = (bureau["CREDIT_DAY_OVERDUE"].fillna(0) > 0).astype("int8")

    bb = bureau_balance.copy()
    bb["bureau_balance_dpd_flag"] = bb["status_group"].isin(
        ["dpd_1_30", "dpd_31_60", "dpd_61_90", "dpd_91_120", "dpd_120_plus"]
    ).astype("int8")
    bb["bureau_balance_dpd30_flag"] = bb["status_group"].isin(
        ["dpd_31_60", "dpd_61_90", "dpd_91_120", "dpd_120_plus"]
    ).astype("int8")
    bb_agg = bb.groupby("SK_ID_BUREAU", as_index=False).agg(
        bb_record_count=("MONTHS_BALANCE", "count"),
        bb_any_dpd=("bureau_balance_dpd_flag", "max"),
        bb_any_dpd30=("bureau_balance_dpd30_flag", "max"),
        bb_dpd_month_count=("bureau_balance_dpd_flag", "sum"),
    )

    merged = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    grouped = merged.groupby("SK_ID_CURR", as_index=False).agg(
        bureau_record_count=("SK_ID_BUREAU", "count"),
        bureau_active_count=("bureau_active_flag", "sum"),
        bureau_overdue_count=("bureau_overdue_flag", "sum"),
        bureau_credit_sum_mean=("AMT_CREDIT_SUM", "mean"),
        bureau_credit_sum_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_max_overdue_mean=("AMT_CREDIT_MAX_OVERDUE", "mean"),
        bureau_days_credit_max=("DAYS_CREDIT", "max"),
        bureau_days_credit_min=("DAYS_CREDIT", "min"),
        bureau_bb_any_dpd=("bb_any_dpd", "max"),
        bureau_bb_any_dpd30=("bb_any_dpd30", "max"),
        bureau_bb_dpd_month_count=("bb_dpd_month_count", "sum"),
    )
    logging.info("Bureau aggregates built in %.1fs", time.time() - t0)

    metadata = [
        {"feature_name": col, "feature_family": "bureau", "source_table": "cleaned_bureau.parquet|cleaned_bureau_balance.parquet"}
        for col in grouped.columns
        if col != "SK_ID_CURR"
    ]
    return grouped, metadata


def grouped_shifted_rolling_sum(values: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    csum = values.groupby(groups, sort=False).cumsum()
    prior = csum.groupby(groups, sort=False).shift(1)
    prior_window = csum.groupby(groups, sort=False).shift(window + 1)
    return prior.fillna(0) - prior_window.fillna(0)


def compute_previous_streak(flag_series: pd.Series, group_series: pd.Series) -> np.ndarray:
    flags = flag_series.to_numpy(dtype=np.int8, copy=False)
    groups = group_series.to_numpy(copy=False)
    result = np.zeros(len(flags), dtype=np.int32)
    running = 0
    for i in range(len(flags)):
        if i == 0 or groups[i] != groups[i - 1]:
            running = 0
        else:
            running = running + 1 if flags[i - 1] == 1 else 0
        result[i] = running
    return result


def build_installment_history_features(inst_events: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    logging.info("Building leakage-safe installment history features...")
    t0 = time.time()

    sort_cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"]
    sorted_path = project_root() / "artifacts" / "processed" / "cleaned_installment_events_sorted.parquet"
    if sorted_path.exists():
        logging.info("Loading cached sorted installment events...")
        df = pd.read_parquet(
            sorted_path,
            columns=[
                "SK_ID_PREV",
                "SK_ID_CURR",
                "NUM_INSTALMENT_VERSION",
                "NUM_INSTALMENT_NUMBER",
                "DAYS_INSTALMENT",
                "aggregated_payment_delay_days",
                "missing_payment_flag",
                "underpaid_after_aggregation_flag",
                "AMT_INSTALMENT",
                "total_amount_paid",
            ],
        )
    else:
        df = inst_events[
            [
                "SK_ID_PREV",
                "SK_ID_CURR",
                "NUM_INSTALMENT_VERSION",
                "NUM_INSTALMENT_NUMBER",
                "DAYS_INSTALMENT",
                "aggregated_payment_delay_days",
                "missing_payment_flag",
                "underpaid_after_aggregation_flag",
                "AMT_INSTALMENT",
                "total_amount_paid",
            ]
        ].copy()
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df.to_parquet(sorted_path, index=False)
        logging.info("Saved sorted installment cache: %s", sorted_path)

    g = df.groupby("SK_ID_CURR", sort=False)
    df["borrower_event_order"] = g.cumcount().astype("int32")
    df["snapshot_day"] = df["DAYS_INSTALMENT"]
    df["snapshot_month_approx"] = np.floor(df["snapshot_day"] / 30.0).astype("int32")
    df["snapshot_id"] = (
        df["SK_ID_CURR"].astype("int64").astype(str) + "_" + df["borrower_event_order"].astype(str)
    )

    df["is_late"] = (df["aggregated_payment_delay_days"].fillna(-999999) > 0).astype("int8")
    df["is_missed"] = (df["missing_payment_flag"].fillna(0) > 0).astype("int8")
    df["is_partial"] = (df["underpaid_after_aggregation_flag"].fillna(0) > 0).astype("int8")
    df["is_on_time"] = ((df["is_late"] == 0) & (df["is_missed"] == 0)).astype("int8")
    df["payment_ratio"] = (df["total_amount_paid"] / df["AMT_INSTALMENT"].replace({0: np.nan})).astype("float32")
    df["delay_nonnull"] = df["aggregated_payment_delay_days"].fillna(0).astype("float32")

    df["hist_total_installments"] = g.cumcount().astype("int32")
    df["hist_late_count"] = g["is_late"].cumsum().shift(1, fill_value=0).astype("int32")
    df["hist_missed_count"] = g["is_missed"].cumsum().shift(1, fill_value=0).astype("int32")
    df["hist_partial_count"] = g["is_partial"].cumsum().shift(1, fill_value=0).astype("int32")
    df["hist_on_time_count"] = g["is_on_time"].cumsum().shift(1, fill_value=0).astype("int32")
    df["hist_delay_sum"] = g["delay_nonnull"].cumsum().shift(1, fill_value=0.0).astype("float32")
    denom = df["hist_total_installments"].replace({0: np.nan})
    df["hist_delay_mean"] = (df["hist_delay_sum"] / denom).astype("float32")
    df["hist_delay_max"] = (
        df.groupby("SK_ID_CURR", sort=False)["delay_nonnull"].cummax().groupby(df["SK_ID_CURR"], sort=False).shift(1)
    ).astype("float32")
    df["hist_recent_delay"] = g["aggregated_payment_delay_days"].shift(1).astype("float32")
    df["hist_recent_payment_ratio"] = g["payment_ratio"].shift(1).astype("float32")

    for window in (3, 6):
        delay_sum = grouped_shifted_rolling_sum(df["delay_nonnull"], df["SK_ID_CURR"], window)
        df[f"hist_delay_mean_last_{window}"] = (delay_sum / window).astype("float32")
        late_sum = grouped_shifted_rolling_sum(df["is_late"].astype("int32"), df["SK_ID_CURR"], window)
        df[f"hist_late_count_last_{window}"] = late_sum.astype("float32")

    df["hist_on_time_streak"] = compute_previous_streak(df["is_on_time"], df["SK_ID_CURR"])
    df["hist_late_streak"] = compute_previous_streak(df["is_late"], df["SK_ID_CURR"])

    feature_cols = [
        "snapshot_id",
        "SK_ID_CURR",
        "SK_ID_PREV",
        "borrower_event_order",
        "snapshot_day",
        "snapshot_month_approx",
        "hist_total_installments",
        "hist_late_count",
        "hist_missed_count",
        "hist_partial_count",
        "hist_on_time_count",
        "hist_delay_mean",
        "hist_delay_max",
        "hist_recent_delay",
        "hist_recent_payment_ratio",
        "hist_delay_mean_last_3",
        "hist_delay_mean_last_6",
        "hist_late_count_last_3",
        "hist_late_count_last_6",
        "hist_on_time_streak",
        "hist_late_streak",
    ]
    features = df[feature_cols].copy()
    logging.info("Installment history features built in %.1fs", time.time() - t0)

    metadata = [
        {"feature_name": col, "feature_family": "installment_history", "source_table": "cleaned_installment_events.parquet"}
        for col in feature_cols
        if col not in {"snapshot_id", "SK_ID_CURR", "SK_ID_PREV", "borrower_event_order", "snapshot_day", "snapshot_month_approx"}
    ]
    return features, metadata


def minimal_target_columns(target_name: str) -> list[str]:
    base = ["snapshot_id", "SK_ID_CURR", "SK_ID_PREV", "borrower_event_order", "snapshot_day", "snapshot_month_approx"]
    if target_name == "missed_upcoming_emi":
        return base + ["missed_upcoming_emi"]
    return base + ["future_dpd30"]


def build_or_load_feature_family(
    processed_dir: Path,
    cache_name: str,
    builder,
    *builder_args,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    cached = maybe_load_cached_feature_family(processed_dir, cache_name)
    if cached is not None:
        return cached, []
    features, metadata = builder(*builder_args)
    save_cached_feature_family(processed_dir, cache_name, features)
    return features, metadata


def build_shared_snapshot_feature_base(
    target_df: pd.DataFrame,
    installment_features: pd.DataFrame,
    application_features: pd.DataFrame,
    prev_features: pd.DataFrame,
    bureau_features: pd.DataFrame,
) -> pd.DataFrame:
    logging.info("Building shared snapshot feature base...")
    df = target_df.merge(
        installment_features,
        on=["snapshot_id", "SK_ID_CURR", "SK_ID_PREV", "borrower_event_order", "snapshot_day", "snapshot_month_approx"],
        how="left",
        validate="one_to_one",
    )
    df = df.merge(application_features, on="SK_ID_CURR", how="left", validate="many_to_one")
    df = df.merge(prev_features, on="SK_ID_CURR", how="left", validate="many_to_one")
    df = df.merge(bureau_features, on="SK_ID_CURR", how="left", validate="many_to_one")
    return df


def attach_label(feature_base: pd.DataFrame, target_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    logging.info("Attaching label %s ...", label_col)
    return feature_base.merge(
        target_df[["snapshot_id", label_col]],
        on="snapshot_id",
        how="inner",
        validate="one_to_one",
    ).rename(columns={label_col: "label"})


def build_quality_report(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table_name, df in feature_tables.items():
        for col in df.columns:
            rows.append(
                {
                    "table_name": table_name,
                    "column_name": col,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_pct": float(df[col].isna().mean()),
                    "dtype": str(df[col].dtype),
                }
            )
    return pd.DataFrame(rows).sort_values(["table_name", "missing_pct", "column_name"], ascending=[True, False, True])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized Stage 3 feature engineering for Home Credit targets.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cached feature-family parquet files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    root = project_root()
    processed_dir = root / "artifacts" / "processed"
    raw_checks_dir = root / "artifacts" / "raw_checks"

    total_start = time.time()

    if args.rebuild_cache:
        for cache_name in [
            "features_application.parquet",
            "features_previous_application.parquet",
            "features_bureau.parquet",
            "features_installment_history.parquet",
            "cleaned_installment_events_sorted.parquet",
        ]:
            path = processed_dir / cache_name
            if path.exists():
                path.unlink()
                logging.info("Deleted cache: %s", path)

    application = load_parquet(processed_dir, "cleaned_application_train.parquet")
    previous = load_parquet(processed_dir, "cleaned_previous_application.parquet")
    bureau = load_parquet(processed_dir, "cleaned_bureau.parquet")
    bureau_balance = load_parquet(processed_dir, "cleaned_bureau_balance.parquet")
    installment_events = load_parquet(processed_dir, "cleaned_installment_events.parquet")
    target_missed = load_parquet(
        processed_dir, "target_missed_upcoming_emi.parquet", columns=minimal_target_columns("missed_upcoming_emi")
    )
    target_dpd30 = load_parquet(
        processed_dir, "target_future_dpd30.parquet", columns=minimal_target_columns("future_dpd30")
    )

    application_features, app_meta = build_or_load_feature_family(
        processed_dir, "features_application.parquet", build_application_features, application
    )
    previous_features, prev_meta = build_or_load_feature_family(
        processed_dir, "features_previous_application.parquet", build_previous_application_features, previous
    )
    bureau_features, bureau_meta = build_or_load_feature_family(
        processed_dir, "features_bureau.parquet", build_bureau_features, bureau, bureau_balance
    )
    installment_features, inst_meta = build_or_load_feature_family(
        processed_dir, "features_installment_history.parquet", build_installment_history_features, installment_events
    )

    shared_feature_base = build_shared_snapshot_feature_base(
        target_df=target_missed.drop(columns=["missed_upcoming_emi"]),
        installment_features=installment_features,
        application_features=application_features,
        prev_features=previous_features,
        bureau_features=bureau_features,
    )
    shared_base_path = processed_dir / "snapshot_feature_base.parquet"
    shared_feature_base.to_parquet(shared_base_path, index=False)

    missed_model = attach_label(shared_feature_base, target_missed, "missed_upcoming_emi")
    dpd30_feature_base = build_shared_snapshot_feature_base(
        target_df=target_dpd30.drop(columns=["future_dpd30"]),
        installment_features=installment_features,
        application_features=application_features,
        prev_features=previous_features,
        bureau_features=bureau_features,
    )
    dpd30_model = attach_label(dpd30_feature_base, target_dpd30, "future_dpd30")

    missed_path = processed_dir / "model_features_missed_upcoming_emi.parquet"
    dpd30_path = processed_dir / "model_features_future_dpd30.parquet"
    missed_model.to_parquet(missed_path, index=False)
    dpd30_model.to_parquet(dpd30_path, index=False)

    feature_metadata = pd.DataFrame(app_meta + prev_meta + bureau_meta + inst_meta).sort_values(
        ["feature_family", "feature_name"]
    )
    feature_metadata_path = raw_checks_dir / "stage3_feature_dictionary.csv"
    feature_metadata.to_csv(feature_metadata_path, index=False)

    quality_report = build_quality_report(
        {
            "snapshot_feature_base": shared_feature_base,
            "model_features_missed_upcoming_emi": missed_model,
            "model_features_future_dpd30": dpd30_model,
        }
    )
    quality_path = raw_checks_dir / "stage3_feature_quality_report.csv"
    quality_report.to_csv(quality_path, index=False)

    manifest = {
        "outputs": [shared_base_path.name, missed_path.name, dpd30_path.name],
        "cached_feature_families": [
            "features_application.parquet",
            "features_previous_application.parquet",
            "features_bureau.parquet",
            "features_installment_history.parquet",
            "cleaned_installment_events_sorted.parquet",
        ],
        "feature_dictionary": feature_metadata_path.name,
        "feature_quality_report": quality_path.name,
    }
    (raw_checks_dir / "stage3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Saved: %s", shared_base_path)
    logging.info("Saved: %s", missed_path)
    logging.info("Saved: %s", dpd30_path)
    logging.info("Saved: %s", feature_metadata_path)
    logging.info("Saved: %s", quality_path)
    logging.info(
        "Stage 3 completed in %.1fs | base_rows=%s | missed_rows=%s | dpd30_rows=%s",
        time.time() - total_start,
        f"{len(shared_feature_base):,}",
        f"{len(missed_model):,}",
        f"{len(dpd30_model):,}",
    )


if __name__ == "__main__":
    main()
