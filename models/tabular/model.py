"""Tabular baseline models: XGBoost (Tweedie) and LightGBM for clinical prediction.

This module provides gradient boosting models for:
- Cost trajectory prediction (Tweedie loss for XGBoost)
- Readmission risk classification
- Mortality risk prediction

Features:
- 5-fold time-aware cross-validation
- Feature importance extraction
- Early stopping
- Configuration-driven hyperparameters
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score

from healthrisk_ai.config import ModelConfig, build_config
from healthrisk_ai.exceptions import ModelError
from healthrisk_ai.numerics import time_aware_train_test_split

logger = logging.getLogger("healthrisk_ai.models.tabular")


class TabularModel:
    """Gradient boosting model for tabular clinical data.

    Supports:
    - XGBoost with Tweedie loss for cost prediction
    - LightGBM for classification or regression

    Provides training, prediction, feature importance, and serialization.
    """

    def __init__(
        self,
        model_type: str = "xgboost_tweedie",
        selection_threshold: float = 0.01,
        time_series_split_folds: int = 5,
        device: str = "auto",
    ) -> None:
        self.model_type = model_type
        self.selection_threshold = selection_threshold
        self.time_series_split_folds = time_series_split_folds
        self.device = device

        self._model: Any = None
        self._feature_names: list[str] | None = None
        self._is_trained = False
        self._feature_importance: dict[str, float] | None = None

    def train(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        *,
        feature_names: list[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int = 50,
        epochs: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        alpha: float = 0.0,
        lambda_reg: float = 1.0,
        random_state: int = 20260914,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train the tabular model.

        Args:
            X: Feature matrix.
            y: Target values.
            feature_names: Column names for features.
            eval_set: Optional validation set (X_val, y_val).
            early_stopping_rounds: Early stopping patience.
            epochs: Number of boosting rounds.
            learning_rate: Learning rate (eta).
            max_depth: Maximum tree depth.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns.
            min_child_weight: Minimum child weight for XGBoost.
            alpha: L1 regularization.
            lambda_reg: L2 regularization.
            random_state: Random seed.
            verbose: Log progress.

        Returns:
            Training history with evaluation metrics.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_array = X.values
        elif isinstance(X, np.ndarray):
            self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        else:
            msg = f"Unsupported X type: {type(X)}"
            raise ModelError(msg)

        if len(X_array) != len(y):
            msg = f"X ({len(X_array)}) and y ({len(y)}) must have same length"
            raise ModelError(msg)

        if self.model_type == "xgboost_tweedie":
            self._model = xgb.XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=1.5,
                n_estimators=epochs,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=alpha,
                reg_lambda=lambda_reg,
                random_state=random_state,
                early_stopping_rounds=early_stopping_rounds if eval_set else None,
                verbose=verbose if hasattr(xgb.XGBRegressor(), 'verbose') else 0,
                nthread=1,
            )
        elif self.model_type == "lightgbm_regressor":
            self._model = LGBMRegressor(
                objective="regression",
                n_estimators=epochs,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_samples=min_child_weight,
                reg_alpha=alpha,
                reg_lambda=lambda_reg,
                random_state=random_state,
                verbose=-1 if not verbose else 1,
            )
        elif self.model_type == "lightgbm_classifier":
            self._model = LGBMClassifier(
                objective="binary",
                n_estimators=epochs,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_samples=min_child_weight,
                reg_alpha=alpha,
                reg_lambda=lambda_reg,
                random_state=random_state,
                verbose=-1 if not verbose else 1,
            )
        else:
            msg = f"Unknown model type: {self.model_type}"
            raise ModelError(msg)

        # Prepare eval set
        eval_set_list = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            eval_set_list = [(X_val, y_val)]

        # Train
        if self.model_type == "xgboost_tweedie":
            self._model.fit(
                X_array,
                y,
                eval_set=eval_set_list,
                verbose=verbose,
            )
        else:
            self._model.fit(
                X_array,
                y,
                eval_set=eval_set_list,
                verbose=verbose,
            )

        # Compute feature importance
        self._compute_feature_importance()

        self._is_trained = True
        logger.info("Tabular model training complete")

        return self._get_training_history()

    def _compute_feature_importance(self) -> None:
        """Compute and store feature importance."""
        if self._model is None or self._feature_names is None:
            self._feature_importance = {}
            return

        if self.model_type == "xgboost_tweedie":
            importance = self._model.feature_importances_
        else:
            importance = self._model.feature_importances_

        self._feature_importance = {
            name: float(imp)
            for name, imp in zip(self._feature_names, importance)
        }

    def _get_training_history(self) -> dict[str, Any]:
        """Get training history if available."""
        if self._model is None:
            return {}

        history: dict[str, Any] = {}

        if self.model_type == "xgboost_tweedie" and hasattr(self._model, "evals_result"):
            # evals_result() raises when training ran without an eval_set,
            # so degrade gracefully to an empty history in that case.
            try:
                evals_result = self._model.evals_result()
            except Exception:  # noqa: BLE001 - xgboost raises XGBoostError
                evals_result = {}
            if evals_result:
                history["validation_0"] = {
                    k: list(v) for k, v in evals_result.get("validation_0", {}).items()
                }

        return history

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        if not self._is_trained:
            msg = "Model must be trained before prediction"
            raise ModelError(msg)

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if self.model_type in ("lightgbm_classifier", "xgboost_classifier"):
            return self._model.predict_proba(X_array)[:, 1]
        return self._model.predict(X_array)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classifier models).

        Args:
            X: Feature matrix.

        Returns:
            Probability array [n_samples, n_classes].
        """
        if not self._is_trained:
            msg = "Model must be trained before prediction"
            raise ModelError(msg)

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        return self._model.predict_proba(X_array)

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        metric: str = "mse",
    ) -> float:
        """Evaluate model on test data.

        Args:
            X: Feature matrix.
            y: True values.
            metric: Metric to compute ("mse", "rmse", "mae", "r2", "auc").

        Returns:
            Metric value.
        """
        predictions = self.predict(X)

        if metric == "mse":
            return float(mean_squared_error(y, predictions))
        elif metric == "rmse":
            return float(np.sqrt(mean_squared_error(y, predictions)))
        elif metric == "mae":
            return float(np.mean(np.abs(y - predictions)))
        elif metric == "r2":
            return float(r2_score(y, predictions))
        elif metric == "auc":
            return float(roc_auc_score(y, predictions))
        else:
            msg = f"Unknown metric: {metric}"
            raise ModelError(msg)

    def get_feature_importance(self, normalize: bool = True) -> dict[str, float]:
        """Get feature importance scores.

        Args:
            normalize: If True, normalize importance to sum to 1.

        Returns:
            Dict mapping feature names to importance scores.
        """
        if self._feature_importance is None:
            return {}

        importance = dict(self._feature_importance)

        if normalize and importance:
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

        return importance

    def get_selected_features(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float | None = None,
    ) -> list[str]:
        """Get features above importance threshold.

        Args:
            X: Feature matrix (for feature names).
            threshold: Importance threshold (uses config if None).

        Returns:
            List of selected feature names.
        """
        if threshold is None:
            threshold = self.selection_threshold

        importance = self.get_feature_importance(normalize=False)
        if not importance:
            if isinstance(X, pd.DataFrame):
                return list(X.columns)
            return []

        return [name for name, imp in importance.items() if imp >= threshold]

    def save(self, path: str) -> None:
        """Save model.

        Args:
            path: Directory to save to.
        """
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "tabular_model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "model_type": self.model_type,
                    "feature_names": self._feature_names,
                    "selection_threshold": self.selection_threshold,
                },
                f,
            )

        logger.info("Saved tabular model to %s", path)

    @classmethod
    def load(cls, path: str) -> TabularModel:
        """Load a saved model.

        Args:
            path: Directory containing saved model.

        Returns:
            Loaded TabularModel instance.
        """
        import os
        import pickle

        with open(os.path.join(path, "tabular_model.pkl"), "rb") as f:
            data = pickle.load(f)

        model = cls(
            model_type=data["model_type"],
            selection_threshold=data.get("selection_threshold", 0.01),
        )
        model._model = data["model"]
        model._feature_names = data.get("feature_names")
        model._is_trained = True

        if model._model is not None and model._feature_names is not None:
            model._compute_feature_importance()

        logger.info("Loaded tabular model from %s", path)
        return model

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained

    def cross_validate_time_aware(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        *,
        n_folds: int | None = None,
        **train_kwargs: Any,
    ) -> dict[str, list[float]]:
        """Perform time-aware cross-validation.

        Uses chronological splitting to prevent temporal leakage.

        Args:
            X: Feature matrix.
            y: Target values.
            timestamps: Timestamps for time-aware splitting.
            n_folds: Number of folds (uses config if None).
            **train_kwargs: Additional training arguments.

        Returns:
            Dict with fold-wise metrics.
        """
        if n_folds is None:
            n_folds = self.time_series_split_folds

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        X_sorted = X_array[sorted_indices]
        y_sorted = y[sorted_indices]

        # Expanding-window time-aware CV (sklearn TimeSeriesSplit style):
        # each fold trains on all data before its validation window so every
        # fold yields metrics while strictly preventing temporal leakage.
        test_size = len(X_sorted) // (n_folds + 1)
        if test_size == 0:
            msg = (
                f"Not enough samples ({len(X_sorted)}) for {n_folds} "
                "time-aware CV folds"
            )
            raise ModelError(msg)

        metrics: dict[str, list[float]] = {"train_metric": [], "val_metric": []}

        for fold in range(n_folds):
            val_start = (fold + 1) * test_size
            val_end = (fold + 2) * test_size if fold < n_folds - 1 else len(X_sorted)

            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)

            X_train = X_sorted[train_idx]
            y_train = y_sorted[train_idx]
            X_val = X_sorted[val_idx]
            y_val = y_sorted[val_idx]

            # Create new model instance for each fold
            fold_model = TabularModel(
                model_type=self.model_type,
                selection_threshold=self.selection_threshold,
                time_series_split_folds=self.time_series_split_folds,
            )

            # Guard against callers passing verbose via train_kwargs.
            fold_kwargs = dict(train_kwargs)
            fold_kwargs.setdefault("verbose", False)

            fold_model.train(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                **fold_kwargs,
            )

            train_metric = fold_model.evaluate(X_train, y_train, metric="rmse")
            val_metric = fold_model.evaluate(X_val, y_val, metric="rmse")

            metrics["train_metric"].append(train_metric)
            metrics["val_metric"].append(val_metric)

            logger.info(
                "Fold %d/%d: train_rmse=%.4f, val_rmse=%.4f",
                fold + 1,
                n_folds,
                train_metric,
                val_metric,
            )

        return metrics


def build_tabular_model(config: ModelConfig | None = None) -> TabularModel:
    """Build a TabularModel from config.

    Args:
        config: Model config; uses default if None.

    Returns:
        Initialized TabularModel.
    """
    if config is None:
        _, _, _, _, _, config, _ = build_config()

    tabular_cfg = config.tabular
    return TabularModel(
        model_type="xgboost_tweedie",
        selection_threshold=tabular_cfg.selection_threshold,
        time_series_split_folds=tabular_cfg.time_series_split_folds,
    )
