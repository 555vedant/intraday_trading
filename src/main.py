# src/main.py

import os
from pprint import pprint

from .config import DATA_PATH
from .data_utils import (
    load_data,
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

    # 1 if next Close > current Close else 0
    print("Creating target column...")
    df = add_target_column(df)

    # 2. Split 
    print("Splitting into train and test...")
    df_train, df_test = train_test_time_split(df)

    X_train, y_train = get_features_and_target(df_train)
    X_test, y_test = get_features_and_target(df_test)

    # 4. models
    models = get_models()

    all_metrics = {}
    best_model_name = None
    best_accuracy = -1.0
    best_model = None

    # 5.evaluate each model 
    print("Training and evaluating models...")
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        all_metrics[name] = metrics

        print(f"Metrics for {name}:")
        pprint(metrics)

        # best by accuracy
        if metrics["accuracy"] > best_accuracy: 
            best_accuracy = metrics["accuracy"]
            best_model_name = name
            best_model = model

    print("\nAll model metrics:")
    pprint(all_metrics)

    print(f"\nBest model: {best_model_name} with accuracy = {best_accuracy:.4f}")

    # 6. best model 
    print("\nGenerating final predictions with best model...")
    y_pred_best = best_model.predict(X_test)

   
    df_test_with_calls = add_model_call(df_test, y_pred_best)
    df_test_with_pnl = add_model_pnl(df_test_with_calls)

    # 7. final data frame save 
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "test_with_predictions_and_pnl.csv")
    df_test_with_pnl.to_csv(output_path, index=False)
    print(f"\nSaved final test output to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
