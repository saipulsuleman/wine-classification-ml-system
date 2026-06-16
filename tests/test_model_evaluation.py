import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock


def test_evaluate_model_returns_all_keys(trained_lr_model, tmp_path):
    from src.model_evaluation import evaluate_model
    model, scaler, X_test, y_test = trained_lr_model
    result = evaluate_model(model, X_test, y_test, output_dir=str(tmp_path))
    for key in ['classification_report', 'confusion_matrix', 'accuracy', 'f1_weighted',
                'confusion_matrix_path', 'report_path']:
        assert key in result, f"Missing key: {key}"


def test_evaluate_model_creates_files(trained_lr_model, tmp_path):
    from src.model_evaluation import evaluate_model
    model, scaler, X_test, y_test = trained_lr_model
    result = evaluate_model(model, X_test, y_test, output_dir=str(tmp_path))
    assert os.path.exists(result['confusion_matrix_path'])
    assert os.path.exists(result['report_path'])


def test_evaluate_model_accuracy_range(trained_lr_model, tmp_path):
    from src.model_evaluation import evaluate_model
    model, scaler, X_test, y_test = trained_lr_model
    result = evaluate_model(model, X_test, y_test, output_dir=str(tmp_path))
    assert 0.0 <= result['accuracy'] <= 1.0


def test_evaluate_model_with_class_names(trained_lr_model, tmp_path):
    from src.model_evaluation import evaluate_model
    model, scaler, X_test, y_test = trained_lr_model
    class_names = ['class_0', 'class_1', 'class_2']
    result = evaluate_model(model, X_test, y_test,
                            class_names=class_names, output_dir=str(tmp_path))
    assert 'class_0' in result['classification_report']


def test_log_evaluation_artifacts_with_active_run(tmp_path):
    from src.model_evaluation import log_evaluation_artifacts
    cm_path = str(tmp_path / "confusion_matrix.png")
    report_path = str(tmp_path / "classification_report.json")
    open(cm_path, 'w').close()
    open(report_path, 'w').close()
    eval_results = {'confusion_matrix_path': cm_path, 'report_path': report_path}
    with patch('src.model_evaluation.mlflow') as mock_mf:
        mock_mf.active_run.return_value = MagicMock()
        log_evaluation_artifacts(eval_results)
        assert mock_mf.log_artifact.call_count == 2


def test_log_evaluation_artifacts_no_active_run(tmp_path):
    from src.model_evaluation import log_evaluation_artifacts
    eval_results = {
        'confusion_matrix_path': str(tmp_path / "cm.png"),
        'report_path': str(tmp_path / "report.json"),
    }
    with patch('src.model_evaluation.mlflow') as mock_mf:
        mock_mf.active_run.return_value = None
        log_evaluation_artifacts(eval_results)
        mock_mf.log_artifact.assert_not_called()