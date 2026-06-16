import pytest
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def train_test_data(sample_wine_df):
    from src.data_preprocessing import preprocess_data
    return preprocess_data(sample_wine_df.copy(), test_size=0.3, random_state=42)


def test_compute_metrics_returns_all_keys():
    from src.model_training import compute_metrics
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 2])
    result = compute_metrics(y_true, y_pred)
    assert 'accuracy' in result
    assert 'f1_weighted' in result
    assert 'precision_weighted' in result
    assert 'recall_weighted' in result


def test_compute_metrics_perfect_predictions():
    from src.model_training import compute_metrics
    y = np.array([0, 1, 2, 0, 1, 2])
    result = compute_metrics(y, y)
    assert result['accuracy'] == pytest.approx(1.0)
    assert result['f1_weighted'] == pytest.approx(1.0)


def test_compute_metrics_with_proba_adds_roc_auc():
    from src.model_training import compute_metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    result = compute_metrics(y_true, y_pred, y_proba)
    assert 'roc_auc_ovr' in result


def test_compute_metrics_roc_auc_fails_silently():
    from src.model_training import compute_metrics
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_proba = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    result = compute_metrics(y_true, y_pred, y_proba)
    assert isinstance(result, dict)


def test_compare_models_selects_best():
    from src.model_training import compare_models
    mock_a = MagicMock()
    mock_b = MagicMock()
    results = {
        'model_a': {'metrics': {'accuracy': 0.90}, 'model': mock_a},
        'model_b': {'metrics': {'accuracy': 0.95}, 'model': mock_b},
    }
    best_name, best_model = compare_models(results)
    assert best_name == 'model_b'
    assert best_model is mock_b


def test_compare_models_single_entry():
    from src.model_training import compare_models
    mock_model = MagicMock()
    results = {'only_model': {'metrics': {'accuracy': 0.85}, 'model': mock_model}}
    best_name, best_model = compare_models(results)
    assert best_name == 'only_model'


def test_train_logistic_regression_returns_tuple(train_test_data, mock_mlflow):
    from src.model_training import train_logistic_regression
    X_train, X_test, y_train, y_test, _ = train_test_data
    model, metrics, run_id = train_logistic_regression(X_train, y_train, X_test, y_test)
    assert model is not None
    assert 'accuracy' in metrics
    assert run_id == 'test-run-id-123'


def test_train_logistic_regression_model_predicts(train_test_data, mock_mlflow):
    from src.model_training import train_logistic_regression
    X_train, X_test, y_train, y_test, _ = train_test_data
    model, _, _ = train_logistic_regression(X_train, y_train, X_test, y_test)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_random_forest_returns_tuple(train_test_data, mock_mlflow, tmp_path, monkeypatch):
    from src.model_training import train_random_forest
    monkeypatch.chdir(tmp_path)
    X_train, X_test, y_train, y_test, _ = train_test_data
    model, metrics, run_id = train_random_forest(X_train, y_train, X_test, y_test)
    assert 'accuracy' in metrics
    assert run_id == 'test-run-id-123'


def test_train_gradient_boosting_returns_tuple(train_test_data, mock_mlflow):
    from src.model_training import train_gradient_boosting
    X_train, X_test, y_train, y_test, _ = train_test_data
    model, metrics, run_id = train_gradient_boosting(X_train, y_train, X_test, y_test)
    assert 'accuracy' in metrics
    assert run_id == 'test-run-id-123'