from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEEDS: list[int] = [42, 123, 456, 789, 1024]


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    seed: int


def create_splits(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    group_col: str = "patient_nbr",
    target_col: str = "readmitted",
) -> SplitResult:
    """Two-stage GroupShuffleSplit: 70/30 train/temp, then 50/50 temp → val/test.

    Splits are grouped by patient_nbr so all of a patient's encounters stay in
    a single partition (no patient leakage across folds).
    """
    groups = df[group_col].values
    y = df[target_col]
    X = df.drop(columns=[target_col, group_col, "encounter_id"], errors="ignore")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    X_temp = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    groups_temp = groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    inner_val_idx, inner_test_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    val_idx = temp_idx[inner_val_idx]
    test_idx = temp_idx[inner_test_idx]

    _assert_disjoint(df, train_idx, val_idx, test_idx, group_col)

    return SplitResult(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_val=X.iloc[val_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_val=y.iloc[val_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        seed=seed,
    )


def _assert_disjoint(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    group_col: str,
) -> None:
    train_p = set(df.iloc[train_idx][group_col])
    val_p = set(df.iloc[val_idx][group_col])
    test_p = set(df.iloc[test_idx][group_col])
    assert train_p.isdisjoint(val_p), "Train/val patient overlap!"
    assert train_p.isdisjoint(test_p), "Train/test patient overlap!"
    assert val_p.isdisjoint(test_p), "Val/test patient overlap!"
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert all_idx == set(range(len(df))), (
        "Index mismatch — engineered.parquet may have changed between train and evaluate. "
        "Re-run preprocess.py + train.py + evaluate.py as a single pipeline."
    )
