import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer


def load_dataset():
    """Load breast cancer dataset from sklearn and return as DataFrame."""
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df['target'] = bc.target
    df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})
    return df, bc.target_names.tolist()


def save_dataset(df, path='data/breast_cancer_data.csv'):
    """Save dataset to CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    return path


def get_dataset_info(df):
    """Return basic information about the dataset."""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['target'].value_counts().to_dict()
    }
    return info


if __name__ == '__main__':
    df, target_names = load_dataset()
    save_dataset(df)
    info = get_dataset_info(df)
    print(f"Dataset loaded: {info['shape'][0]} rows, {info['shape'][1]} columns")
    print(f"Classes: {target_names}")
    print(f"Target distribution: {info['target_distribution']}")
