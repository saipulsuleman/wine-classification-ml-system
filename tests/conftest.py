import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_wine_df():
    np.random.seed(42)
    n = 30
    data = {
        'alcohol': np.random.uniform(11, 15, n),
        'malic_acid': np.random.uniform(0.7, 5.8, n),
        'ash': np.random.uniform(1.4, 3.2, n),
        'alcalinity_of_ash': np.random.uniform(10, 30, n),
        'magnesium': np.random.uniform(70, 162, n),
        'total_phenols': np.random.uniform(0.9, 3.9, n),
        'flavanoids': np.random.uniform(0.3, 5.1, n),
        'nonflavanoid_phenols': np.random.uniform(0.1, 0.7, n),
        'proanthocyanins': np.random.uniform(0.4, 3.6, n),
        'color_intensity': np.random.uniform(1.3, 13, n),
        'hue': np.random.uniform(0.5, 1.7, n),
        'od280_od315_of_diluted_wines': np.random.uniform(1.3, 4.0, n),
        'proline': np.random.uniform(278, 1680, n),
        'target': [0] * 10 + [1] * 10 + [2] * 10,
        'target_name': ['class_0'] * 10 + ['class_1'] * 10 + ['class_2'] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_with_missing(sample_wine_df):
    df = sample_wine_df.copy()
    df.loc[0, 'alcohol'] = np.nan
    df.loc[5, 'malic_acid'] = np.nan
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
def trained_lr_model(sample_wine_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = sample_wine_df.drop(columns=['target', 'target_name'])
    y = sample_wine_df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X_scaled, y