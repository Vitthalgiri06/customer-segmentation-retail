"""Leak-free 60/20/20 Train/Validation/Test splitting and feature scaling."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Clustering inputs exclude ID-like columns (CustomerID), redundant aggregates,
# and geographic/sensitive attributes (PrimaryCountry, IsUK kept for auditing only).
MODEL_FEATURES: List[str] = ["RecencyDays", "Frequency", "Monetary", "UniqueProducts"]
AUDIT_COLUMNS: List[str] = ["CustomerID", "PrimaryCountry", "IsUK"]


def split_data(
    features_df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split customer features into 60% Train, 20% Validation, and 20% Test subsets."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        features_df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def log_transform(df: pd.DataFrame, feature_cols: List[str] | None = None) -> np.ndarray:
    """Apply log1p compression to right-skewed behavioral counts and spending amounts."""
    cols = feature_cols or MODEL_FEATURES
    arr = df[cols].to_numpy(dtype=float)
    return np.log1p(np.clip(arr, a_min=0.0, a_max=None))


def fit_transform_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fit StandardScaler strictly on the Training set and transform Train, Val, and Test."""
    cols = feature_cols or MODEL_FEATURES
    scaler = StandardScaler().fit(log_transform(train_df, cols))
    X_train = scaler.transform(log_transform(train_df, cols))
    X_val = scaler.transform(log_transform(val_df, cols))
    X_test = scaler.transform(log_transform(test_df, cols))
    return scaler, X_train, X_val, X_test
