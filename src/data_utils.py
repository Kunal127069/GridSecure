import pandas as pd
import numpy as np
from src.config import DATASET_PATH, METADATA_COLS, TARGET_COL


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
            
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna('Unknown')
            
    return df_clean


def separate_features_and_target(df):
    df_clean = clean_data(df)
    meta_cols = [c for c in METADATA_COLS if c in df_clean.columns]
    
    y = df_clean[TARGET_COL] if TARGET_COL in df_clean.columns else None
    
    drop_cols = meta_cols.copy()
    if TARGET_COL in df_clean.columns:
        drop_cols.append(TARGET_COL)
        
    feature_df = df_clean.drop(columns=drop_cols)
    date_cols = [c for c in feature_df.columns if c.startswith("2014") or c.startswith("2015") or c.startswith("2016")]
    feature_df = feature_df.drop(columns=date_cols)
    
    return feature_df, y


def get_consumer_by_id(cons_no, df=None):
    if df is None:
        df = load_dataset()
    match = df[df["CONS_NO"].astype(str) == str(cons_no)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()
