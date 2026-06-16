import pytest
import numpy as np
import pandas as pd


def test_handle_missing_values_fills_nan(df_with_missing):
    from src.data_preprocessing import handle_missing_values
    result = handle_missing_values(df_with_missing)
    assert result['alcohol'].isna().sum() == 0
    assert result['malic_acid'].isna().sum() == 0


def test_handle_missing_values_uses_median(df_with_missing):
    from src.data_preprocessing import handle_missing_values
    expected_median = df_with_missing['alcohol'].median()
    result = handle_missing_values(df_with_missing)
    assert result.loc[0, 'alcohol'] == pytest.approx(expected_median)


def test_handle_missing_values_no_nan_unchanged(sample_wine_df):
    from src.data_preprocessing import handle_missing_values
    result = handle_missing_values(sample_wine_df.copy())
    assert result['alcohol'].isna().sum() == 0


def test_remove_outliers_keeps_normal_data(sample_wine_df):
    from src.data_preprocessing import remove_outliers
    result = remove_outliers(sample_wine_df.copy(), threshold=3.0)
    assert len(result) > 0


def test_remove_outliers_removes_extreme_values(sample_wine_df):
    from src.data_preprocessing import remove_outliers
    df = sample_wine_df.copy()
    df.loc[0, 'alcohol'] = 9999.0
    result = remove_outliers(df, threshold=3.0)
    assert len(result) < len(df)
    assert 9999.0 not in result['alcohol'].values


def test_remove_outliers_excludes_target_column(sample_wine_df):
    from src.data_preprocessing import remove_outliers
    df = sample_wine_df.copy()
    df.loc[0, 'target'] = 9999
    result = remove_outliers(df, target_col='target', threshold=3.0)
    assert 9999 in result['target'].values


def test_preprocess_data_returns_5_tuple(sample_wine_df):
    from src.data_preprocessing import preprocess_data
    result = preprocess_data(sample_wine_df.copy())
    assert len(result) == 5


def test_preprocess_data_split_sizes(sample_wine_df):
    from src.data_preprocessing import preprocess_data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        sample_wine_df.copy(), test_size=0.2, random_state=42
    )
    total = len(X_train) + len(X_test)
    assert len(X_test) / total == pytest.approx(0.2, abs=0.1)


def test_preprocess_data_scaler_saved(sample_wine_df, tmp_path):
    from src.data_preprocessing import preprocess_data
    import joblib
    scaler_path = str(tmp_path / "scaler.pkl")
    preprocess_data(sample_wine_df.copy(), scaler_path=scaler_path)
    loaded = joblib.load(scaler_path)
    assert hasattr(loaded, 'transform')


def test_preprocess_data_aligned_lengths(sample_wine_df):
    from src.data_preprocessing import preprocess_data
    X_train, X_test, y_train, y_test, _ = preprocess_data(sample_wine_df.copy())
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)