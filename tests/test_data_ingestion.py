import pytest
import pandas as pd
import os


def test_load_dataset_shape():
    from src.data_ingestion import load_dataset
    df, target_names = load_dataset()
    assert df.shape == (569, 32)


def test_load_dataset_columns():
    from src.data_ingestion import load_dataset
    df, _ = load_dataset()
    assert 'target' in df.columns
    assert 'target_name' in df.columns


def test_load_dataset_target_names():
    from src.data_ingestion import load_dataset
    _, target_names = load_dataset()
    assert len(target_names) == 2
    assert 'malignant' in target_names


def test_load_dataset_target_mapping():
    from src.data_ingestion import load_dataset
    df, _ = load_dataset()
    assert set(df['target'].unique()) == {0, 1}
    assert set(df['target_name'].unique()) == {'malignant', 'benign'}


def test_save_dataset_creates_file(tmp_path, sample_bc_df):
    from src.data_ingestion import save_dataset
    out_path = str(tmp_path / "output.csv")
    save_dataset(sample_bc_df, path=out_path)
    assert os.path.exists(out_path)


def test_save_dataset_readable(tmp_path, sample_bc_df):
    from src.data_ingestion import save_dataset
    out_path = str(tmp_path / "output.csv")
    save_dataset(sample_bc_df, path=out_path)
    loaded = pd.read_csv(out_path)
    assert loaded.shape == sample_bc_df.shape


def test_save_dataset_creates_directory(tmp_path):
    from src.data_ingestion import save_dataset, load_dataset
    df, _ = load_dataset()
    nested_path = str(tmp_path / "subdir" / "data.csv")
    save_dataset(df, path=nested_path)
    assert os.path.exists(nested_path)
