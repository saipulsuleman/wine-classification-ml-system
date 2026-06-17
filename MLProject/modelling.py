import os
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

EXPERIMENT_NAME = "breast-cancer-classification-experiment"
TRACKING_URI = "sqlite:///mlflow.db"


def load_and_preprocess():
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df['target'] = bc.target

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    feat_cols = [c for c in df.columns if c != 'target']
    z_scores = np.abs(stats.zscore(df[feat_cols]))
    df_clean = df[(z_scores < 3.0).all(axis=1)].reset_index(drop=True)

    X = df_clean.drop('target', axis=1).values
    y = df_clean['target'].values
    return X, y, df_clean


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_model(name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        mlflow.autolog()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(model, "model",
                                 registered_model_name=f"breast-cancer-classifier-{name}")
        run_id = mlflow.active_run().info.run_id
    return acc, f1, run_id


def main():
    print("=" * 60)
    print("Breast Cancer Classification - MLProject Training")
    print("=" * 60)

    setup_mlflow()

    print("\n[1] Loading and preprocessing data...")
    X, y, df_clean = load_and_preprocess()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")

    df_clean.to_csv("breast_cancer_preprocessing.csv", index=False)
    print("Saved: breast_cancer_preprocessing.csv")

    print("\n[2] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n[3] Training models...")
    models = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        acc, f1, run_id = train_model(name, model, X_train, y_train, X_test, y_test)
        results[name] = {"accuracy": acc, "f1": f1, "run_id": run_id}
        print(f"  {name}: accuracy={acc:.4f}, f1={f1:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n[4] Best model: {best_name} (f1={results[best_name]['f1']:.4f})")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
