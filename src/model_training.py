import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


EXPERIMENT_NAME = "breast-cancer-classification-experiment"


def setup_mlflow(tracking_uri="sqlite:///mlflow.db"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {EXPERIMENT_NAME}")


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            pass
    return metrics


def train_logistic_regression(X_train, y_train, X_test, y_test):
    params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42}
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model", registered_model_name="breast-cancer-classifier-lr")

        run_id = mlflow.active_run().info.run_id
        print(f"[LR] accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_weighted']:.4f}, run_id={run_id}")
        return model, metrics, run_id


def train_random_forest(X_train, y_train, X_test, y_test):
    params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": 42, "n_jobs": -1}
    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log feature importances as artifact
        import json
        feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else list(range(X_train.shape[1]))
        importances = dict(zip(feature_names, model.feature_importances_.tolist()))
        with open("feature_importances.json", "w") as f:
            json.dump(importances, f, indent=2)
        mlflow.log_artifact("feature_importances.json")

        mlflow.sklearn.log_model(model, "model", registered_model_name="breast-cancer-classifier-rf")

        run_id = mlflow.active_run().info.run_id
        print(f"[RF] accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_weighted']:.4f}, run_id={run_id}")
        return model, metrics, run_id


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42}
    with mlflow.start_run(run_name="gradient_boosting"):
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model", registered_model_name="breast-cancer-classifier-gb")

        run_id = mlflow.active_run().info.run_id
        print(f"[GB] accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_weighted']:.4f}, run_id={run_id}")
        return model, metrics, run_id


def compare_models(results):
    """Compare models and return the best one by accuracy."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    best = results[best_name]
    print(f"\nBest model: {best_name} with accuracy={best['metrics']['accuracy']:.4f}")
    return best_name, best["model"]
