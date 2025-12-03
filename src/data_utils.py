# src/data_utils.py

import pandas as pd
from .config import FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_RATIO


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip().str.lower()

    print(" Columns found in CSV:", list(df.columns))

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Required column '{col}' not found in CSV.")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[TARGET_COLUMN] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.iloc[:-1].reset_index(drop=True)
    return df


def train_test_time_split(df: pd.DataFrame):
    n = len(df)
    split_idx = int(n * TRAIN_RATIO)

    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    return df_train, df_test


def get_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    return X, y
