# src/trading.py

import pandas as pd
import numpy as np


def add_model_call(df_test: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df = df_test.copy()
    df["Predicted"] = y_pred
    df["model_call"] = np.where(df["Predicted"] == 1, "buy", "sell")
    return df


def add_model_pnl(df_test_with_calls: pd.DataFrame) -> pd.DataFrame:
    
    df = df_test_with_calls.copy()
    pnl = 0.0
    pnl_list = []

    for _, row in df.iterrows():
        close_price = row["Close"]
        if row["model_call"] == "buy":
            pnl -= close_price
        else:  # "sell"
            pnl += close_price
        pnl_list.append(pnl)

    df["model_pnl"] = pnl_list
    return df
