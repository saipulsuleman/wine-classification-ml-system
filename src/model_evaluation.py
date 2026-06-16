import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)
import mlflow
import json
import os


def evaluate_model(model, X_test, y_test, class_names=None, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix - Wine Classification")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix_path": cm_path,
        "report_path": os.path.join(output_dir, "classification_report.json"),
    }


def log_evaluation_artifacts(eval_results, mlflow_run_id=None):
    if mlflow.active_run():
        mlflow.log_artifact(eval_results["confusion_matrix_path"])
        mlflow.log_artifact(eval_results["report_path"])
        print("Evaluation artifacts logged to MLflow")
