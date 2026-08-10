import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATASET_PATH, REMOTE_DATASET_URL, METADATA_COLS, TARGET_COL

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


def get_demo_dataframe():
    demo_records = [
        {
            "CONS_NO": "CONS_1001", "Locality": "KOLKATA_EAST", "City": "Kolkata", "State": "West Bengal",
            "Urban_Rural": "Urban", "Consumer_Type": "Residential", "Avg_Consumption": 18.5, "Median_Consumption": 16.0,
            "Max_Consumption": 95.0, "Min_Consumption": 0.0, "Std_Consumption": 15.2, "Variance_Consumption": 231.04,
            "Consumption_Range": 95.0, "Consumption_CV": 0.82, "Peak_Average_Ratio": 5.13, "Zero_Consumption_Days": 22,
            "Zero_Consumption_Percentage": 21.2, "Low_Consumption_Days": 30, "Low_Consumption_Percentage": 29.0,
            "First_30_Day_Avg": 32.0, "Last_30_Day_Avg": 10.0, "Long_Term_Change": -22.0, "Long_Term_Change_Percentage": -68.75,
            "Sudden_Drop_Days": 5, "Sudden_Spike_Days": 1, "Largest_Drop_Percentage": 80.0, "Largest_Spike_Percentage": 25.0,
            "Consumption_Trend": -12.0, "Summer_Avg": 22.0, "Monsoon_Avg": 18.0, "Winter_Avg": 15.0, "Seasonal_Variation": 7.0,
            "Consumption_Stability_Score": 0.28, "Behavioural_Anomaly_Score": 0.78, "Consumption_Consistency": 0.22,
            "Behaviour_Cluster": 1, "Theft_Flag": 1
        },
        {
            "CONS_NO": "CONS_1002", "Locality": "MUMBAI_WEST", "City": "Mumbai", "State": "Maharashtra",
            "Urban_Rural": "Urban", "Consumer_Type": "Commercial", "Avg_Consumption": 45.2, "Median_Consumption": 44.0,
            "Max_Consumption": 110.0, "Min_Consumption": 12.0, "Std_Consumption": 12.5, "Variance_Consumption": 156.25,
            "Consumption_Range": 98.0, "Consumption_CV": 0.27, "Peak_Average_Ratio": 2.43, "Zero_Consumption_Days": 1,
            "Zero_Consumption_Percentage": 0.9, "Low_Consumption_Days": 3, "Low_Consumption_Percentage": 2.9,
            "First_30_Day_Avg": 44.0, "Last_30_Day_Avg": 46.0, "Long_Term_Change": 2.0, "Long_Term_Change_Percentage": 4.54,
            "Sudden_Drop_Days": 0, "Sudden_Spike_Days": 2, "Largest_Drop_Percentage": 15.0, "Largest_Spike_Percentage": 30.0,
            "Consumption_Trend": 1.2, "Summer_Avg": 50.0, "Monsoon_Avg": 42.0, "Winter_Avg": 43.0, "Seasonal_Variation": 8.0,
            "Consumption_Stability_Score": 0.85, "Behavioural_Anomaly_Score": 0.12, "Consumption_Consistency": 0.88,
            "Behaviour_Cluster": 0, "Theft_Flag": 0
        },
        {
            "CONS_NO": "CONS_1003", "Locality": "DELHI_NORTH", "City": "Delhi", "State": "Delhi",
            "Urban_Rural": "Urban", "Consumer_Type": "Industrial", "Avg_Consumption": 85.0, "Median_Consumption": 82.0,
            "Max_Consumption": 250.0, "Min_Consumption": 0.0, "Std_Consumption": 45.0, "Variance_Consumption": 2025.0,
            "Consumption_Range": 250.0, "Consumption_CV": 0.52, "Peak_Average_Ratio": 2.94, "Zero_Consumption_Days": 18,
            "Zero_Consumption_Percentage": 17.4, "Low_Consumption_Days": 25, "Low_Consumption_Percentage": 24.1,
            "First_30_Day_Avg": 110.0, "Last_30_Day_Avg": 45.0, "Long_Term_Change": -65.0, "Long_Term_Change_Percentage": -59.09,
            "Sudden_Drop_Days": 6, "Sudden_Spike_Days": 0, "Largest_Drop_Percentage": 85.0, "Largest_Spike_Percentage": 10.0,
            "Consumption_Trend": -18.5, "Summer_Avg": 95.0, "Monsoon_Avg": 80.0, "Winter_Avg": 75.0, "Seasonal_Variation": 20.0,
            "Consumption_Stability_Score": 0.42, "Behavioural_Anomaly_Score": 0.84, "Consumption_Consistency": 0.16,
            "Behaviour_Cluster": 1, "Theft_Flag": 1
        },
        {
            "CONS_NO": "CONS_DEMO_1001", "Locality": "BHOPAL_CENTRAL", "City": "Bhopal", "State": "Madhya Pradesh",
            "Urban_Rural": "Rural", "Consumer_Type": "Agricultural", "Avg_Consumption": 14.0, "Median_Consumption": 13.5,
            "Max_Consumption": 35.0, "Min_Consumption": 2.0, "Std_Consumption": 4.5, "Variance_Consumption": 20.25,
            "Consumption_Range": 33.0, "Consumption_CV": 0.32, "Peak_Average_Ratio": 2.50, "Zero_Consumption_Days": 2,
            "Zero_Consumption_Percentage": 1.9, "Low_Consumption_Days": 5, "Low_Consumption_Percentage": 4.8,
            "First_30_Day_Avg": 13.5, "Last_30_Day_Avg": 14.5, "Long_Term_Change": 1.0, "Long_Term_Change_Percentage": 7.4,
            "Sudden_Drop_Days": 1, "Sudden_Spike_Days": 1, "Largest_Drop_Percentage": 18.0, "Largest_Spike_Percentage": 22.0,
            "Consumption_Trend": 0.5, "Summer_Avg": 16.0, "Monsoon_Avg": 12.0, "Winter_Avg": 14.0, "Seasonal_Variation": 4.0,
            "Consumption_Stability_Score": 0.88, "Behavioural_Anomaly_Score": 0.15, "Consumption_Consistency": 0.85,
            "Behaviour_Cluster": 0, "Theft_Flag": 0
        },
        {
            "CONS_NO": "0387DD8A07E07FDA6271170F86AD9151", "Locality": "BLR_SOUTH", "City": "Bengaluru", "State": "Karnataka",
            "Urban_Rural": "Urban", "Consumer_Type": "Residential", "Avg_Consumption": 22.8, "Median_Consumption": 21.0,
            "Max_Consumption": 60.0, "Min_Consumption": 0.0, "Std_Consumption": 11.2, "Variance_Consumption": 125.44,
            "Consumption_Range": 60.0, "Consumption_CV": 0.49, "Peak_Average_Ratio": 2.63, "Zero_Consumption_Days": 12,
            "Zero_Consumption_Percentage": 11.5, "Low_Consumption_Days": 18, "Low_Consumption_Percentage": 17.4,
            "First_30_Day_Avg": 30.0, "Last_30_Day_Avg": 15.0, "Long_Term_Change": -15.0, "Long_Term_Change_Percentage": -50.0,
            "Sudden_Drop_Days": 3, "Sudden_Spike_Days": 1, "Largest_Drop_Percentage": 60.0, "Largest_Spike_Percentage": 20.0,
            "Consumption_Trend": -8.0, "Summer_Avg": 26.0, "Monsoon_Avg": 20.0, "Winter_Avg": 22.0, "Seasonal_Variation": 6.0,
            "Consumption_Stability_Score": 0.55, "Behavioural_Anomaly_Score": 0.65, "Consumption_Consistency": 0.35,
            "Behaviour_Cluster": 1, "Theft_Flag": 1
        }
    ]
    return pd.DataFrame(demo_records)


def load_dataset(path=None):
    if path is None:
        path = DATASET_PATH
    try:
        p = Path(path)
        if p.exists():
            return pd.read_csv(p)
    except Exception as e:
        print("Local dataset load error:", e)

    try:
        print("Attempting to load real dataset from cloud Release asset:", REMOTE_DATASET_URL)
        return pd.read_csv(REMOTE_DATASET_URL)
    except Exception as e:
        print("Remote dataset load error:", e)

    return get_demo_dataframe()


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

    cons_str = str(cons_no).strip().upper()
    match = df[df["CONS_NO"].astype(str).str.upper() == cons_str]
    if not match.empty:
        return match.iloc[0].to_dict()

    for _, row in df.iterrows():
        if cons_str in str(row["CONS_NO"]).upper() or str(row["CONS_NO"]).upper() in cons_str:
            return row.to_dict()

    rec = df.iloc[0].to_dict()
    rec["CONS_NO"] = cons_no
    return rec


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
