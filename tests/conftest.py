import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_bc_df():
    np.random.seed(42)
    n = 30
    data = {
        'mean radius': np.random.uniform(6.9, 28.1, n),
        'mean texture': np.random.uniform(9.7, 39.3, n),
        'mean perimeter': np.random.uniform(43.8, 188.5, n),
        'mean area': np.random.uniform(143.5, 2501.0, n),
        'mean smoothness': np.random.uniform(0.05, 0.16, n),
        'mean compactness': np.random.uniform(0.02, 0.35, n),
        'mean concavity': np.random.uniform(0.0, 0.43, n),
        'mean concave points': np.random.uniform(0.0, 0.20, n),
        'mean symmetry': np.random.uniform(0.11, 0.30, n),
        'mean fractal dimension': np.random.uniform(0.05, 0.10, n),
        'radius error': np.random.uniform(0.11, 2.87, n),
        'texture error': np.random.uniform(0.36, 4.88, n),
        'perimeter error': np.random.uniform(0.76, 21.98, n),
        'area error': np.random.uniform(6.80, 542.2, n),
        'smoothness error': np.random.uniform(0.001, 0.031, n),
        'compactness error': np.random.uniform(0.002, 0.135, n),
        'concavity error': np.random.uniform(0.0, 0.396, n),
        'concave points error': np.random.uniform(0.0, 0.053, n),
        'symmetry error': np.random.uniform(0.008, 0.079, n),
        'fractal dimension error': np.random.uniform(0.001, 0.03, n),
        'worst radius': np.random.uniform(7.9, 36.0, n),
        'worst texture': np.random.uniform(12.0, 49.5, n),
        'worst perimeter': np.random.uniform(50.4, 251.2, n),
        'worst area': np.random.uniform(185.2, 4254.0, n),
        'worst smoothness': np.random.uniform(0.07, 0.22, n),
        'worst compactness': np.random.uniform(0.03, 1.06, n),
        'worst concavity': np.random.uniform(0.0, 1.25, n),
        'worst concave points': np.random.uniform(0.0, 0.29, n),
        'worst symmetry': np.random.uniform(0.16, 0.66, n),
        'worst fractal dimension': np.random.uniform(0.055, 0.208, n),
        'target': [0] * 15 + [1] * 15,
        'target_name': ['malignant'] * 15 + ['benign'] * 15,
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_with_missing(sample_bc_df):
    df = sample_bc_df.copy()
    df.loc[0, 'mean radius'] = np.nan
    df.loc[5, 'mean texture'] = np.nan
    return df


@pytest.fixture
def mock_mlflow():
    with patch('src.model_training.mlflow') as mock_mf:
        mock_run = MagicMock()
        mock_run.info.run_id = 'test-run-id-123'
        mock_mf.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mf.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mf.active_run.return_value = mock_run
        yield mock_mf


@pytest.fixture
def trained_lr_model(sample_bc_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = sample_bc_df.drop(columns=['target', 'target_name'])
    y = sample_bc_df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X_scaled, y
