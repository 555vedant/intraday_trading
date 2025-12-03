import numpy as np


def add_model_call(df_test, y_pred):
    df = df_test.copy()
    df["Predicted"] = y_pred
    df["model_call"] = np.where(df["Predicted"] == 1, "buy", "sell")
    return df


def add_model_pnl(df):
    pnl = 0
    pnl_list = []

    for _, row in df.iterrows():
        if row["model_call"] == "buy":
            pnl -= row["close"]
        else:
            pnl += row["close"]

        pnl_list.append(pnl)

    df["model_pnl"] = pnl_list
    return df
