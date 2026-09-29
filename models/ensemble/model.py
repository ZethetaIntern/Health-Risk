"""Stacking ensemble: Ridge regression meta-learner combining all model outputs.

This module implements a stacking ensemble that combines predictions from:
- Bio_ClinicalBERT embeddings (64-dim)
- GATv2 GNN patient embeddings (32-dim)
- DeepSurv hazard rates
- XGBoost/LightGBM predictions

The meta-learner (Ridge regression) learns optimal weights to combine these
diverse signals into a final prediction with R² > 0.25 target.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from healthrisk_ai.config import ModelConfig, build_config
from healthrisk_ai.exceptions import ModelError

logger = logging.getLogger("healthrisk_ai.models.ensemble")


class StackingEnsemble:
    """Stacking ensemble that combines predictions from multiple models.

    Architecture:
        Base models produce predictions -> Meta-learner (Ridge) combines them
        -> Final prediction

    The ensemble expects base model predictions as input, learning optimal
    weights to combine them.

    Args:
        meta_learner: Type of meta-learner ("ridge" or "ridge_cv").
        stack_target_r2: Target R² for the ensemble.
        normalize_inputs: Whether to standardize base predictions.
    """

    def __init__(
        self,
        meta_learner: str = "ridge",
        stack_target_r2: float = 0.25,
        normalize_inputs: bool = True,
    ) -> None:
        self.meta_learner = meta_learner
        self.stack_target_r2 = stack_target_r2
        self.normalize_inputs = normalize_inputs

        self.meta_model: Ridge | RidgeCV | None = None
        self.scaler: StandardScaler | None = None
        self._is_trained = False
        self._feature_names: list[str] | None = None
        self._coefficients: np.ndarray | None = None
        self._intercept: float | None = None

    def train(
        self,
        base_predictions: dict[str, np.ndarray],
        target: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        cv: int = 5,
        alphas: list[float] | None = None,
        random_state: int = 20260914,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train the stacking meta-learner.

        Args:
            base_predictions: Dict mapping model names to prediction arrays.
                              Each array should be shape (n_samples,).
            target: True target values (n_samples,).
            sample_weights: Optional sample weights.
            cv: Number of CV folds for RidgeCV.
            alphas: Alpha values to try for RidgeCV.
            random_state: Random seed.
            verbose: Log progress.

        Returns:
            Training info including coefficients and R².
        """
        if not base_predictions:
            msg = "base_predictions cannot be empty"
            raise ModelError(msg)

        if len(target) == 0:
            msg = "target cannot be empty"
            raise ModelError(msg)

        # Validate all prediction arrays have same length
        n_samples = len(target)
        for name, preds in base_predictions.items():
            if len(preds) != n_samples:
                msg = f"Prediction array for {name} has length {len(preds)}, expected {n_samples}"
                raise ModelError(msg)

        # Build meta-features matrix
        self._feature_names = list(base_predictions.keys())
        meta_features = np.column_stack([base_predictions[name] for name in self._feature_names])

        # Normalize if requested
        if self.normalize_inputs:
            self.scaler = StandardScaler()
            meta_features = self.scaler.fit_transform(meta_features)

        # Train meta-learner
        if self.meta_learner == "ridge_cv":
            if alphas is None:
                alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
            self.meta_model = RidgeCV(
                alphas=alphas,
                cv=cv,
                scoring="r2",
                store_cv_values=True,
            )
        else:
            self.meta_model = Ridge(alpha=1.0, random_state=random_state)

        self.meta_model.fit(meta_features, target, sample_weight=sample_weights)

        # Store coefficients
        self._coefficients = self.meta_model.coef_
        self._intercept = self.meta_model.intercept_

        # Mark trained before scoring so predict() accepts the inputs.
        self._is_trained = True

        # Compute training R²
        predictions = self.predict(base_predictions)
        train_r2 = float(r2_score(target, predictions))

        logger.info(
            "Stacking ensemble trained. Train R²=%.4f, target R²=%.4f",
            train_r2,
            self.stack_target_r2,
        )

        return {
            "train_r2": train_r2,
            "coefficients": dict(zip(self._feature_names, self._coefficients.tolist())),
            "intercept": float(self._intercept),
            "best_alpha": getattr(self.meta_model, "alpha_", None),
        }

    def predict(
        self,
        base_predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Make predictions using base model outputs.

        Args:
            base_predictions: Dict mapping model names to prediction arrays.

        Returns:
            Ensemble predictions.
        """
        if not self._is_trained:
            msg = "Ensemble must be trained before prediction"
            raise ModelError(msg)

        if set(base_predictions.keys()) != set(self._feature_names or []):
            msg = f"Expected base predictions for: {self._feature_names}"
            raise ModelError(msg)

        # Build meta-features matrix
        meta_features = np.column_stack(
            [base_predictions[name] for name in self._feature_names]
        )

        # Normalize if requested
        if self.normalize_inputs and self.scaler is not None:
            meta_features = self.scaler.transform(meta_features)

        return self.meta_model.predict(meta_features)

    def evaluate(
        self,
        base_predictions: dict[str, np.ndarray],
        target: np.ndarray,
        metric: str = "r2",
    ) -> float:
        """Evaluate ensemble on test data.

        Args:
            base_predictions: Base model predictions.
            target: True values.
            metric: Metric to compute.

        Returns:
            Metric value.
        """
        predictions = self.predict(base_predictions)

        if metric == "r2":
            return float(r2_score(target, predictions))
        else:
            msg = f"Unknown metric: {metric}"
            raise ModelError(msg)

    def get_model_weights(self) -> dict[str, float]:
        """Get the learned weights for each base model.

        Returns:
            Dict mapping model names to coefficients.
        """
        if self._coefficients is None or self._feature_names is None:
            return {}

        return dict(zip(self._feature_names, self._coefficients.tolist()))

    def save(self, path: str) -> None:
        """Save ensemble.

        Args:
            path: Directory to save to.
        """
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "ensemble.pkl"), "wb") as f:
            pickle.dump(
                {
                    "meta_model": self.meta_model,
                    "scaler": self.scaler,
                    "meta_learner": self.meta_learner,
                    "stack_target_r2": self.stack_target_r2,
                    "normalize_inputs": self.normalize_inputs,
                    "feature_names": self._feature_names,
                    "is_trained": self._is_trained,
                    "coefficients": self._coefficients,
                    "intercept": self._intercept,
                },
                f,
            )

        logger.info("Saved stacking ensemble to %s", path)

    @classmethod
    def load(cls, path: str) -> StackingEnsemble:
        """Load a saved ensemble.

        Args:
            path: Directory containing saved ensemble.

        Returns:
            Loaded StackingEnsemble instance.
        """
        import os
        import pickle

        with open(os.path.join(path, "ensemble.pkl"), "rb") as f:
            data = pickle.load(f)

        ensemble = cls(
            meta_learner=data["meta_learner"],
            stack_target_r2=data["stack_target_r2"],
            normalize_inputs=data["normalize_inputs"],
        )
        ensemble.meta_model = data["meta_model"]
        ensemble.scaler = data["scaler"]
        ensemble._feature_names = data["feature_names"]
        ensemble._is_trained = data["is_trained"]
        ensemble._coefficients = data["coefficients"]
        ensemble._intercept = data["intercept"]

        logger.info("Loaded stacking ensemble from %s", path)
        return ensemble

    @property
    def is_trained(self) -> bool:
        """Whether the ensemble has been trained."""
        return self._is_trained


def build_stacking_ensemble(config: ModelConfig | None = None) -> StackingEnsemble:
    """Build a StackingEnsemble from config.

    Args:
        config: Model config; uses default if None.

    Returns:
        Initialized StackingEnsemble.
    """
    if config is None:
        _, _, _, _, _, config, _ = build_config()

    ensemble_cfg = config.ensemble
    return StackingEnsemble(
        meta_learner=ensemble_cfg.meta_learner,
        stack_target_r2=ensemble_cfg.stack_target_r2,
    )
