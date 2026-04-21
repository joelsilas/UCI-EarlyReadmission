

import logging
from typing import Self

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Optuna-tuned (50 trials, TPE sampler, seed=42).
TUNED_HPARAMS: dict = {
    "depth": 6,
    "learning_rate": 0.006070980631259233,
    "iterations": 2000,
    "l2_leaf_reg": 0.19150811880863894,
    "random_strength": 0.5460194353464073,
    "bagging_temperature": 0.40426743243555785,
    "eval_metric": "Logloss",
    "random_seed": 42,
    "verbose": 0,
}


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Cast non-numeric columns to string for CatBoost's native categorical handling."""
    X_cb = X.copy()
    for col in X_cb.select_dtypes(exclude=["number", "bool"]).columns:
        X_cb[col] = X_cb[col].astype(str)
    return X_cb


def _cat_indices(X: pd.DataFrame) -> list[int]:
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns
    return [X.columns.get_loc(c) for c in non_numeric]


class FocalLossObjective:
    """Focal loss custom objective for CatBoost.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Derivatives w.r.t. raw logit for CatBoost's calc_ders_range API.

    TODO: EXPAND THIS AND SEE IF I CAN GET THIS TO RUN WITH NUMBA

    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def calc_ders_range(self, approxes, targets, weights):
        alpha, gamma = self.alpha, self.gamma
        approxes = np.array(approxes, dtype=np.float64)
        targets = np.array(targets, dtype=np.float64)
        # Numerically stable sigmoid.
        p = np.where(
            approxes >= 0,
            1.0 / (1.0 + np.exp(-approxes)),
            np.exp(approxes) / (1.0 + np.exp(approxes)),
        )
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        is_pos = targets >= 0.5
        pt = np.where(is_pos, p, 1.0 - p)
        at = np.where(is_pos, alpha, 1.0 - alpha)
        log_pt = np.log(pt)
        common = at * (1.0 - pt) ** gamma * (gamma * pt * log_pt + pt - 1.0)
        sign = np.where(is_pos, 1.0, -1.0)
        der1 = sign * common
        der2 = at * (1.0 - pt) ** gamma * p * (1.0 - p)
        der2 = np.maximum(der2, 1e-7)
        return list(zip(der1.tolist(), der2.tolist(), strict=True))


class CatBoostBase:
    """Wrapper around CatBoostClassifier for native categorical handling."""

    def __init__(self, **extra_params) -> None:
        params = {**TUNED_HPARAMS, **extra_params}
        self._model = CatBoostClassifier(**params)
        self._cat_features: list[int] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:
        self._cat_features = _cat_indices(X_train)
        self._model.fit(_prepare(X_train), y_train, cat_features=self._cat_features)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(_prepare(X))[:, 1]


class CatBoostWeighted(CatBoostBase):

    def __init__(self) -> None:
        super().__init__(auto_class_weights="Balanced")


class CatBoostUnweighted(CatBoostBase):

    def __init__(self) -> None:
        super().__init__()


class CatBoostFocal(CatBoostBase):

    def __init__(self) -> None:
        super().__init__(loss_function=FocalLossObjective(alpha=0.75, gamma=2.0))


class DiverseStackingEnsemble:
    """Stacking with three CatBoost bases + isotonic calibratied Logistic Regression to improve F1 scores
    """

    N_FOLDS = 3
    CALIB_FOLDS = 5

    def __init__(self) -> None:
        self._base_learners: list = []
        self._meta_learner = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=42),
            method="isotonic",
            cv=self.CALIB_FOLDS,
        )

    @staticmethod
    def _make_bases() -> list:
        return [CatBoostWeighted(), CatBoostUnweighted(), CatBoostFocal()]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:
        kfold = StratifiedKFold(n_splits=self.N_FOLDS, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X_train), 3))

        for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            logger.info(f"  OOF fold {fold_idx + 1}/{self.N_FOLDS}")
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val_fold = X_train.iloc[val_idx]
            for model_idx, model in enumerate(self._make_bases()):
                logger.info(f"    base {model_idx + 1}/3 ({type(model).__name__})")
                model.fit(X_tr, y_tr)
                oof_predictions[val_idx, model_idx] = model.predict_proba(X_val_fold)

        logger.info("  fitting isotonic-calibrated LR meta-learner on OOF predictions")
        self._meta_learner.fit(oof_predictions, y_train)

        logger.info("  refitting bases on full training data")
        self._base_learners = self._make_bases()
        for model in self._base_learners:
            logger.info(f"    {type(model).__name__}")
            model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_probas = np.column_stack([m.predict_proba(X) for m in self._base_learners])
        return self._meta_learner.predict_proba(base_probas)[:, 1]
