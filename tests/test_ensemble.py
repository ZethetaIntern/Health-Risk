"""Unit tests for the Stacking Ensemble module."""

from __future__ import annotations

import numpy as np
import pytest

from healthrisk_ai.models.ensemble.model import (
    StackingEnsemble,
    build_stacking_ensemble,
)


class TestStackingEnsemble:
    def test_init_default(self) -> None:
        ensemble = StackingEnsemble(
            meta_learner="ridge",
            stack_target_r2=0.25,
        )
        assert ensemble.meta_learner == "ridge"
        assert not ensemble.is_trained

    def test_init_ridge_cv(self) -> None:
        ensemble = StackingEnsemble(meta_learner="ridge_cv")
        assert ensemble.meta_learner == "ridge_cv"

    def test_train_and_predict(self) -> None:
        ensemble = StackingEnsemble(meta_learner="ridge")

        n_samples = 100
        # Simulate base model predictions
        base_predictions = {
            "nlp": np.random.randn(n_samples) * 0.5 + 5,
            "gnn": np.random.randn(n_samples) * 0.3 + 5,
            "survival": np.random.randn(n_samples) * 0.4 + 5,
            "tabular": np.random.randn(n_samples) * 0.6 + 5,
        }
        # Target is a weighted combination + noise
        target = (
            0.3 * base_predictions["nlp"]
            + 0.2 * base_predictions["gnn"]
            + 0.3 * base_predictions["survival"]
            + 0.2 * base_predictions["tabular"]
            + np.random.randn(n_samples) * 0.1
        )

        info = ensemble.train(base_predictions, target, verbose=False)

        assert ensemble.is_trained
        assert "train_r2" in info
        assert "coefficients" in info
        assert info["train_r2"] > 0  # Should learn something

    def test_predict_requires_train(self) -> None:
        ensemble = StackingEnsemble()

        base_predictions = {
            "nlp": np.random.randn(10),
            "gnn": np.random.randn(10),
        }

        with pytest.raises(Exception):  # ModelError
            ensemble.predict(base_predictions)

    def test_predict_mismatched_models(self) -> None:
        ensemble = StackingEnsemble()
        ensemble._is_trained = True
        ensemble._feature_names = ["nlp", "gnn"]

        # Wrong keys
        base_predictions = {
            "survival": np.random.randn(10),
            "tabular": np.random.randn(10),
        }

        with pytest.raises(Exception):  # ModelError
            ensemble.predict(base_predictions)

    def test_evaluate(self) -> None:
        ensemble = StackingEnsemble(meta_learner="ridge")

        n_samples = 100
        base_predictions = {
            "model1": np.random.randn(n_samples),
            "model2": np.random.randn(n_samples),
        }
        target = base_predictions["model1"] * 0.5 + base_predictions["model2"] * 0.5

        ensemble.train(base_predictions, target, verbose=False)

        test_preds = {
            "model1": np.random.randn(20),
            "model2": np.random.randn(20),
        }
        test_target = test_preds["model1"] * 0.5 + test_preds["model2"] * 0.5

        r2 = ensemble.evaluate(test_preds, test_target, metric="r2")
        assert r2 <= 1.0

    def test_get_model_weights(self) -> None:
        ensemble = StackingEnsemble(meta_learner="ridge")

        n_samples = 100
        base_predictions = {
            "nlp": np.random.randn(n_samples),
            "gnn": np.random.randn(n_samples),
            "tabular": np.random.randn(n_samples),
        }
        target = np.random.randn(n_samples)

        ensemble.train(base_predictions, target, verbose=False)

        weights = ensemble.get_model_weights()
        assert "nlp" in weights
        assert "gnn" in weights
        assert "tabular" in weights
        assert len(weights) == 3

    def test_save_and_load(self, tmp_path: str) -> None:
        ensemble = StackingEnsemble(meta_learner="ridge")

        n_samples = 50
        base_predictions = {
            "model1": np.random.randn(n_samples),
            "model2": np.random.randn(n_samples),
        }
        target = base_predictions["model1"] + base_predictions["model2"]

        ensemble.train(base_predictions, target, verbose=False)
        ensemble.save(tmp_path)

        loaded = StackingEnsemble.load(tmp_path)
        assert loaded.is_trained
        assert loaded.meta_learner == "ridge"

        # Verify predictions match
        test_preds = {
            "model1": np.random.randn(10),
            "model2": np.random.randn(10),
        }
        pred1 = ensemble.predict(test_preds)
        pred2 = loaded.predict(test_preds)
        np.testing.assert_allclose(pred1, pred2)

    def test_train_empty_predictions_raises(self) -> None:
        ensemble = StackingEnsemble()
        with pytest.raises(Exception):  # ModelError
            ensemble.train({}, np.array([1, 2, 3]))

    def test_train_mismatched_lengths_raises(self) -> None:
        ensemble = StackingEnsemble()
        base_predictions = {
            "model1": np.random.randn(100),
            "model2": np.random.randn(50),  # Mismatched
        }
        target = np.random.randn(100)

        with pytest.raises(Exception):  # ModelError
            ensemble.train(base_predictions, target)


class TestBuildStackingEnsemble:
    def test_build_from_config(self) -> None:
        from healthrisk_ai.models.ensemble.model import build_stacking_ensemble

        ensemble = build_stacking_ensemble()
        assert ensemble.meta_learner == "ridge"
        assert ensemble.stack_target_r2 == 0.25
