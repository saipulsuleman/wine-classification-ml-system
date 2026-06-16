import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def remove_outliers(df, target_col='target', threshold=3.0):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores < threshold).all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    print(f"Rows before outlier removal: {len(df)}, after: {len(df_clean)}")
    return df_clean


def split_features_target(df, target_col='target', drop_cols=None):
    if drop_cols is None:
        drop_cols = [target_col, 'target_name']
    drop_cols_existing = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols_existing)
    y = df[target_col]
    return X, y


def preprocess_data(df, target_col='target', test_size=0.2, random_state=42, scaler_path=None):
    df = handle_missing_values(df)
    df = remove_outliers(df, target_col=target_col)
    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    print(f"Train size: {len(X_train_scaled)}, Test size: {len(X_test_scaled)}")
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler
