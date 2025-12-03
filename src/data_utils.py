
import pandas as pd
from .config import FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_RATIO


def load_data(data_path):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()

    print("Columns:", list(df.columns))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_technical_features(df):
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["hl_range"] = df["high"] - df["low"]
    df["co_diff"] = df["close"] - df["open"]
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def add_target_column(df):
    df = df.copy()

    df[TARGET_COLUMN] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def balance_dataset(df):
    df_major = df[df[TARGET_COLUMN] == 0]
    df_minor = df[df[TARGET_COLUMN] == 1]

    df_major = df_major.sample(
        n=min(len(df_major), len(df_minor) * 2),
        random_state=42
    )

    df_balanced = pd.concat([df_major, df_minor])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(" Balanced target distribution:")
    print(df_balanced[TARGET_COLUMN].value_counts())

    return df_balanced


def train_test_time_split(df):
    split = int(len(df) * TRAIN_RATIO)
    return (
        df.iloc[:split].reset_index(drop=True),
        df.iloc[split:].reset_index(drop=True),
    )


def get_features_and_target(df):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    return X, y
