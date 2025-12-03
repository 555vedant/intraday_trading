# src/main.py

import os
from pprint import pprint

from .config import DATA_PATH
from .data_utils import (
    load_data,
    add_technical_features,
    add_target_column,
    train_test_time_split,
    get_features_and_target,
)
from .models_def import get_models
from .metrics_utils import evaluate_predictions
from .trading import add_model_call, add_model_pnl


def main():

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Adding technical features...")
    df = add_technical_features(df)

    print(" Creating target...")
    df = add_target_column(df)

    df_train, df_test = train_test_time_split(df)

    X_train, y_train = get_features_and_target(df_train)
    X_test, y_test = get_features_and_target(df_test)

    models = get_models()

    all_metrics = {}
    best_model = None
    best_name = None
    best_acc = -1

    print("\nTraining models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        all_metrics[name] = metrics

        pprint(f"{name.upper()} → {metrics}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_model = model
            best_name = name

    print("\nAll Model Metrics:")
    pprint(all_metrics)

    print(f"\nBest Model: {best_name} | Accuracy: {best_acc:.4f}")

    y_pred_best = best_model.predict(X_test)
    df_test = add_model_call(df_test, y_pred_best)
    df_test = add_model_pnl(df_test)

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/test_with_predictions_and_pnl.csv"
    df_test.to_csv(output_path, index=False)

    print(f"\n Final Output Saved: {output_path}")
    print("\n Done.")


if __name__ == "__main__":
    main()
