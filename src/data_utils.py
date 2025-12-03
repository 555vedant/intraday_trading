# src/data_utils.py

import pandas as pd
from .config import FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_RATIO


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    print("Columns found:", list(df.columns))

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f" Missing column: {col}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["hl_range"] = df["high"] - df["low"]
    df["co_diff"] = df["close"] - df["open"]
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    future_return = (df["close"].shift(-1) - df["close"]) / df["close"]
    df[TARGET_COLUMN] = (future_return > 0.001).astype(int)

    df = df.iloc[:-1].reset_index(drop=True)
    return df


def train_test_time_split(df: pd.DataFrame):
    n = len(df)
    split = int(n * TRAIN_RATIO)

    return (
        df.iloc[:split].reset_index(drop=True),
        df.iloc[split:].reset_index(drop=True),
    )


def get_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    return X, y
