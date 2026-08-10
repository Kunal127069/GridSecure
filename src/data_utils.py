import pandas as pd
import numpy as np
from src.config import DATASET_PATH, METADATA_COLS, TARGET_COL

ENGINEERED_FEATURES = [
    "Urban_Rural", "Consumer_Type", "Avg_Consumption", "Median_Consumption",
    "Max_Consumption", "Min_Consumption", "Std_Consumption", "Variance_Consumption",
    "Consumption_Range", "Consumption_CV", "Peak_Average_Ratio",
    "Zero_Consumption_Days", "Zero_Consumption_Percentage", "Low_Consumption_Days",
    "Low_Consumption_Percentage", "First_30_Day_Avg", "Last_30_Day_Avg",
    "Long_Term_Change", "Long_Term_Change_Percentage", "Sudden_Drop_Days",
    "Sudden_Spike_Days", "Largest_Drop_Percentage", "Largest_Spike_Percentage",
    "Consumption_Trend", "Summer_Avg", "Monsoon_Avg", "Winter_Avg",
    "Seasonal_Variation", "Consumption_Stability_Score", "Behavioural_Anomaly_Score",
    "Consumption_Consistency", "Behaviour_Cluster",
]

_dataset_stats_cache = None


def load_dataset(path=None):
    if path is None:
        path = DATASET_PATH
    return pd.read_csv(path)


def clean_data(df):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].replace([np.inf, -np.inf], np.nan)

    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna("Unknown")

    return df_clean


def separate_features_and_target(df):
    df_clean = clean_data(df)
    meta_cols = [c for c in METADATA_COLS if c in df_clean.columns]

    y = df_clean[TARGET_COL] if TARGET_COL in df_clean.columns else None

    drop_cols = meta_cols.copy()
    if TARGET_COL in df_clean.columns:
        drop_cols.append(TARGET_COL)

    feature_df = df_clean.drop(columns=drop_cols)
    date_cols = [
        c for c in feature_df.columns
        if c.startswith("2014") or c.startswith("2015") or c.startswith("2016")
    ]
    feature_df = feature_df.drop(columns=date_cols)

    return feature_df, y


def compute_dataset_stats(df=None):
    global _dataset_stats_cache
    if _dataset_stats_cache is not None:
        return _dataset_stats_cache

    if df is None:
        df = load_dataset()

    feature_df, _ = separate_features_and_target(df)
    stats = {}
    for col in feature_df.columns:
        if feature_df[col].dtype in [np.float64, np.int64, float, int]:
            stats[col] = {
                "median": float(feature_df[col].median()),
                "mean": float(feature_df[col].mean()),
                "std": float(feature_df[col].std()) if feature_df[col].std() > 0 else 1.0,
            }
        else:
            mode_val = feature_df[col].mode()
            stats[col] = {"mode": str(mode_val.iloc[0]) if not mode_val.empty else "Unknown"}

    _dataset_stats_cache = stats
    return stats


def _is_missing(val):
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, np.floating) and np.isnan(val):
        return True
    if isinstance(val, str):
        return val.strip() == "" or val.strip().lower() in {"nan", "none", "null", "n/a", "na", "unknown"}
    return False


def _coerce_float(val, default=0.0):
    if _is_missing(val):
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(val, default=0):
    if _is_missing(val):
        return int(default)
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return int(default)


def complete_consumer_record(record, stats=None):
    """Fill missing engineered features using correlations and dataset statistics."""
    if stats is None:
        stats = compute_dataset_stats()

    out = dict(record)

    for meta in METADATA_COLS:
        if _is_missing(out.get(meta)):
            out[meta] = stats.get(meta, {}).get("mode", "Unknown")

    for cat in ["Urban_Rural", "Consumer_Type"]:
        if _is_missing(out.get(cat)):
            out[cat] = stats.get(cat, {}).get("mode", "Unknown")

    def _default(name):
        return stats.get(name, {}).get("median", 0.0)

    avg = out.get("Avg_Consumption")
    if _is_missing(avg):
        avg = out.get("Median_Consumption")
    if _is_missing(avg):
        avg = _default("Avg_Consumption")
    avg = _coerce_float(avg, _default("Avg_Consumption"))

    zero_days = _coerce_int(out.get("Zero_Consumption_Days", 0), 0)
    sudden_drops = _coerce_int(out.get("Sudden_Drop_Days", 0), 0)
    anomaly = _coerce_float(out.get("Behavioural_Anomaly_Score", _default("Behavioural_Anomaly_Score")), _default("Behavioural_Anomaly_Score"))
    cluster = _coerce_int(out.get("Behaviour_Cluster", _default("Behaviour_Cluster")), _default("Behaviour_Cluster"))

    derived = {
        "Avg_Consumption": avg,
        "Median_Consumption": _coerce_float(out.get("Median_Consumption"), avg * 0.92),
        "Max_Consumption": _coerce_float(out.get("Max_Consumption"), max(avg * 2.8, _default("Max_Consumption"))),
        "Min_Consumption": _coerce_float(out.get("Min_Consumption"), 0.0),
        "Std_Consumption": _coerce_float(out.get("Std_Consumption"), max(avg * 0.55, _default("Std_Consumption"))),
        "Zero_Consumption_Days": zero_days,
        "Zero_Consumption_Percentage": _coerce_float(
            out.get("Zero_Consumption_Percentage"), min(zero_days / 10.34 * 100, 100.0)
        ),
        "Sudden_Drop_Days": sudden_drops,
        "Largest_Drop_Percentage": _coerce_float(
            out.get("Largest_Drop_Percentage"), min(40 + sudden_drops * 8, 95.0)
        ),
        "First_30_Day_Avg": _coerce_float(out.get("First_30_Day_Avg"), avg * 1.15),
        "Last_30_Day_Avg": _coerce_float(out.get("Last_30_Day_Avg"), avg * (0.5 if zero_days > 15 else 0.85)),
        "Behavioural_Anomaly_Score": anomaly,
        "Behaviour_Cluster": cluster,
    }

    derived["Variance_Consumption"] = _coerce_float(
        out.get("Variance_Consumption"), derived["Std_Consumption"] ** 2
    )
    derived["Consumption_Range"] = _coerce_float(
        out.get("Consumption_Range"), derived["Max_Consumption"] - derived["Min_Consumption"]
    )
    derived["Consumption_CV"] = _coerce_float(
        out.get("Consumption_CV"), (derived["Std_Consumption"] / avg if avg > 0 else _default("Consumption_CV"))
    )
    derived["Peak_Average_Ratio"] = _coerce_float(
        out.get("Peak_Average_Ratio"), (derived["Max_Consumption"] / avg if avg > 0 else _default("Peak_Average_Ratio"))
    )
    derived["Low_Consumption_Days"] = _coerce_int(out.get("Low_Consumption_Days"), zero_days + sudden_drops)
    derived["Low_Consumption_Percentage"] = _coerce_float(
        out.get("Low_Consumption_Percentage"), min(derived["Low_Consumption_Days"] / 10.34 * 100, 100.0)
    )
    derived["Long_Term_Change"] = _coerce_float(
        out.get("Long_Term_Change"), derived["Last_30_Day_Avg"] - derived["First_30_Day_Avg"]
    )
    first_avg = derived["First_30_Day_Avg"]
    derived["Long_Term_Change_Percentage"] = _coerce_float(
        out.get("Long_Term_Change_Percentage"), ((derived["Long_Term_Change"] / first_avg * 100) if first_avg > 0 else _default("Long_Term_Change_Percentage"))
    )
    derived["Sudden_Spike_Days"] = _coerce_int(out.get("Sudden_Spike_Days"), max(0, 3 - sudden_drops))
    derived["Largest_Spike_Percentage"] = _coerce_float(out.get("Largest_Spike_Percentage"), _default("Largest_Spike_Percentage"))
    derived["Consumption_Trend"] = _coerce_float(
        out.get("Consumption_Trend"), np.sign(derived["Long_Term_Change"]) * min(abs(derived["Long_Term_Change"]), 50)
    )
    derived["Summer_Avg"] = _coerce_float(out.get("Summer_Avg"), avg * 1.08)
    derived["Monsoon_Avg"] = _coerce_float(out.get("Monsoon_Avg"), avg * 0.95)
    derived["Winter_Avg"] = _coerce_float(out.get("Winter_Avg"), avg * 1.02)
    seasonal_vals = [derived["Summer_Avg"], derived["Monsoon_Avg"], derived["Winter_Avg"]]
    derived["Seasonal_Variation"] = _coerce_float(
        out.get("Seasonal_Variation"), (max(seasonal_vals) - min(seasonal_vals))
    )
    derived["Consumption_Stability_Score"] = _coerce_float(
        out.get("Consumption_Stability_Score"), max(0.0, 1.0 - derived["Consumption_CV"])
    )
    derived["Consumption_Consistency"] = _coerce_float(
        out.get("Consumption_Consistency"), max(0.0, 1.0 - anomaly)
    )

    for name in ENGINEERED_FEATURES:
        if _is_missing(out.get(name)):
            out[name] = derived.get(name, _default(name))

    return out


def get_consumer_by_id(cons_no, df=None):
    if df is None:
        df = load_dataset()

    cons_str = str(cons_no).strip()
    match = df[df["CONS_NO"].astype(str).str.upper() == cons_str.upper()]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None


def get_sample_consumers(df=None, n_each=2):
    if df is None:
        df = load_dataset()

    theft = df[df[TARGET_COL] == 1].head(n_each)
    normal = df[df[TARGET_COL] == 0].head(n_each)

    samples = []
    for _, row in theft.iterrows():
        samples.append({
            "consumer_id": row["CONS_NO"],
            "label": "Confirmed Theft",
            "theft_flag": 1,
            "zero_days": int(row.get("Zero_Consumption_Days", 0)),
            "anomaly_score": round(float(row.get("Behavioural_Anomaly_Score", 0)), 3),
        })
    for _, row in normal.iterrows():
        samples.append({
            "consumer_id": row["CONS_NO"],
            "label": "Normal Usage",
            "theft_flag": 0,
            "zero_days": int(row.get("Zero_Consumption_Days", 0)),
            "anomaly_score": round(float(row.get("Behavioural_Anomaly_Score", 0)), 3),
        })
    return samples


def get_cluster_analytics(df=None):
    if df is None:
        df = load_dataset()

    clusters = []
    for cluster_id, group in df.groupby("Behaviour_Cluster"):
        theft_rate = round(group[TARGET_COL].mean() * 100, 2) if TARGET_COL in group.columns else 0.0
        status = "CRITICAL" if theft_rate >= 25 else "HIGH" if theft_rate >= 12 else "MEDIUM" if theft_rate >= 8 else "LOW"
        clusters.append({
            "id": f"CLUSTER_{int(cluster_id)}",
            "name": f"Behavior Cluster {int(cluster_id)}",
            "cluster_id": int(cluster_id),
            "meters": int(len(group)),
            "theft_rate": theft_rate,
            "theft_cases": int(group[TARGET_COL].sum()) if TARGET_COL in group.columns else 0,
            "status": status,
        })

    return sorted(clusters, key=lambda c: c["theft_rate"], reverse=True)


def get_consumption_trend(df=None, max_months=12):
    if df is None:
        df = load_dataset()

    date_cols = [
        c for c in df.columns
        if c.startswith("2014") or c.startswith("2015") or c.startswith("2016")
    ]
    if not date_cols:
        return []

    month_map = {}
    for col in date_cols:
        parts = col.split("-")
        if len(parts) >= 2:
            month_key = f"{parts[0]}-{parts[1]}"
            month_map.setdefault(month_key, []).append(col)

    trend = []
    for month_key in sorted(month_map.keys())[:max_months]:
        cols = month_map[month_key]
        avg_total = float(df[cols].sum(axis=1).mean())
        theft_mask = df[TARGET_COL] == 1 if TARGET_COL in df.columns else pd.Series([False] * len(df))
        theft_avg = float(df.loc[theft_mask, cols].sum(axis=1).mean()) if theft_mask.any() else 0.0
        trend.append({
            "month": month_key,
            "label": month_key.split("-")[1] + "/" + month_key.split("-")[0][2:],
            "avg_consumption_kwh": round(avg_total, 1),
            "theft_avg_consumption_kwh": round(theft_avg, 1),
        })

    return trend


def estimate_theft_loss_gwh(df=None):
    if df is None:
        df = load_dataset()

    if TARGET_COL not in df.columns:
        return 0.0

    theft_df = df[df[TARGET_COL] == 1]
    if theft_df.empty or "Avg_Consumption" not in theft_df.columns:
        return 0.0

    total_daily_kwh = float(theft_df["Avg_Consumption"].sum())
    return round(total_daily_kwh * 365 / 1_000_000, 2)
