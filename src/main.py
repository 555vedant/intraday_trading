# src/main.py

import os
from pprint import pprint
import numpy as np

from .config import DATA_PATH, PROBA_THRESHOLD
from .data_utils import (
    load_data,
    add_technical_features,
    add_target_column,
    balance_dataset,
    train_test_time_split,
    get_features_and_target,
)
from .models_def import get_models
from .metrics_utils import evaluate_predictions
from .trading import add_model_call, add_model_pnl


def main():

    print("🔹 Loading data...")
    df = load_data(DATA_PATH)

    print("🔹 Adding technical features...")
    df = add_technical_features(df)

    print("🔹 Creating target...")
    df = add_target_column(df)

    print("🔹 Balancing dataset...")
    df = balance_dataset(df)

    df_train, df_test = train_test_time_split(df)

    X_train, y_train = get_features_and_target(df_train)
    X_test, y_test = get_features_and_target(df_test)

    models = get_models()

    best_model = None
    best_name = None
    best_score = -1

    print("\n🔹 Training models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba > PROBA_THRESHOLD).astype(int)

        metrics = evaluate_predictions(y_test, y_pred)
        balanced_score = (
            metrics["accuracy"] +
            metrics["precision"] +
            metrics["recall"] +
            metrics["f1score"]
        ) / 4

        print(f"{name.upper()} → {metrics}")

        if balanced_score > best_score:
            best_score = balanced_score
            best_model = model
            best_name = name

    print(f"\n Best Balanced Model: {best_name}")

    proba_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = (proba_best > PROBA_THRESHOLD).astype(int)

    df_test = add_model_call(df_test, y_pred_best)
    df_test = add_model_pnl(df_test)

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/test_with_predictions_and_pnl.csv"
    df_test.to_csv(output_path, index=False)

    print(f"\n Final Output Saved: {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
