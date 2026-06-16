import sys
import os
import mlflow
import mlflow.sklearn
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.data_ingestion import load_dataset
from src.data_preprocessing import preprocess_data
from src.model_training import setup_mlflow, compute_metrics


def tune_logistic_regression(X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500, 1000],
    }
    base_model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"Best params: {best_params}")
    print(f"Best CV score: {best_cv_score:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")

    with mlflow.start_run(run_name="logistic_regression_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("best_cv_score", round(best_cv_score, 4))
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="wine-classifier-lr-tuned")

    return best_model, metrics, best_params


def main():
    print("=" * 60)
    print("Wine Classification - Hyperparameter Tuning")
    print("=" * 60)

    setup_mlflow()

    print("\n[1] Loading dataset...")
    df, target_names = load_dataset()

    print("\n[2] Preprocessing data...")
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    print("\n[3] Running GridSearchCV for Logistic Regression...")
    best_model, metrics, best_params = tune_logistic_regression(X_train, y_train, X_test, y_test)

    os.makedirs("artifacts", exist_ok=True)
    result = {"best_params": best_params, "metrics": metrics}
    with open("artifacts/tuning_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Tuning complete! Best accuracy: {metrics['accuracy']:.4f}")
    print(f"Best params: {best_params}")
    print("=" * 60)


if __name__ == "__main__":
    main()