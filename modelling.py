import sys
import os
import mlflow
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import load_dataset, save_dataset
from src.data_preprocessing import preprocess_data
from src.model_training import (
    setup_mlflow, train_logistic_regression,
    train_random_forest, train_gradient_boosting, compare_models
)
from src.model_evaluation import evaluate_model, log_evaluation_artifacts


def main():
    print("=" * 60)
    print("Wine Classification - Model Training Pipeline")
    print("=" * 60)

    setup_mlflow()

    print("\n[1] Loading dataset...")
    df, target_names = load_dataset()
    save_dataset(df)
    print(f"Dataset shape: {df.shape}")

    print("\n[2] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, scaler_path="artifacts/scaler.pkl"
    )

    print("\n[3] Training models...")
    results = {}

    model_lr, metrics_lr, run_id_lr = train_logistic_regression(X_train, y_train, X_test, y_test)
    results["logistic_regression"] = {"model": model_lr, "metrics": metrics_lr, "run_id": run_id_lr}

    model_rf, metrics_rf, run_id_rf = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = {"model": model_rf, "metrics": metrics_rf, "run_id": run_id_rf}

    model_gb, metrics_gb, run_id_gb = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results["gradient_boosting"] = {"model": model_gb, "metrics": metrics_gb, "run_id": run_id_gb}

    print("\n[4] Comparing models...")
    best_name, best_model = compare_models(results)

    print(f"\n[5] Evaluating best model ({best_name})...")
    eval_results = evaluate_model(best_model, X_test, y_test, class_names=target_names)

    best_run_id = results[best_name]["run_id"]
    with mlflow.start_run(run_id=best_run_id):
        log_evaluation_artifacts(eval_results)

    summary = {
        "best_model": best_name,
        "best_metrics": results[best_name]["metrics"],
        "all_results": {name: r["metrics"] for name, r in results.items()}
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Best model: {best_name}")
    print(f"Best accuracy: {results[best_name]['metrics']['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()