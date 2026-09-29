"""Unit tests for the Tabular Models module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthrisk_ai.models.tabular.model import (
    TabularModel,
    build_tabular_model,
)


class TestTabularModel:
    def test_init_default(self) -> None:
        model = TabularModel(
            model_type="xgboost_tweedie",
            selection_threshold=0.01,
            time_series_split_folds=5,
        )
        assert model.model_type == "xgboost_tweedie"
        assert not model.is_trained

    def test_init_lightgbm_regressor(self) -> None:
        model = TabularModel(model_type="lightgbm_regressor")
        assert model.model_type == "lightgbm_regressor"

    def test_train_xgboost_tweedie(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 100
        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.randn(n_samples),
            }
        )
        # Tweedie-distributed target (positive, skewed)
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(
            X,
            y,
            epochs=10,
            learning_rate=0.1,
            verbose=False,
        )

        assert model.is_trained
        assert model._feature_names is not None

    def test_predict(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 50
        X = pd.DataFrame(
            {
                "f1": np.random.randn(n_samples),
                "f2": np.random.randn(n_samples),
            }
        )
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(X, y, epochs=10, verbose=False)

        X_test = pd.DataFrame(
            {
                "f1": np.random.randn(10),
                "f2": np.random.randn(10),
            }
        )
        predictions = model.predict(X_test)
        assert len(predictions) == 10
        assert np.all(predictions >= 0)  # Tweedie produces positive values

    def test_train_numpy_input(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 50
        X = np.random.randn(n_samples, 3)
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(
            X,
            y,
            epochs=10,
            verbose=False,
            feature_names=["f1", "f2", "f3"],
        )

        assert model.is_trained

    def test_evaluate(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 50
        X = pd.DataFrame({"f1": np.random.randn(n_samples)})
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(X, y, epochs=10, verbose=False)

        mse = model.evaluate(X, y, metric="mse")
        rmse = model.evaluate(X, y, metric="rmse")
        r2 = model.evaluate(X, y, metric="r2")

        assert mse >= 0
        assert rmse >= 0
        assert r2 <= 1.0

    def test_feature_importance(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 50
        X = pd.DataFrame(
            {
                "important": np.random.randn(n_samples),
                "unimportant": np.random.randn(n_samples),
            }
        )
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(X, y, epochs=10, verbose=False)

        importance = model.get_feature_importance()
        assert "important" in importance
        assert "unimportant" in importance
        assert sum(importance.values()) > 0

    def test_selected_features(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 50
        X = pd.DataFrame(
            {
                "f1": np.random.randn(n_samples),
                "f2": np.random.randn(n_samples),
                "f3": np.random.randn(n_samples),
            }
        )
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(X, y, epochs=10, verbose=False)

        selected = model.get_selected_features(X, threshold=0.01)
        assert len(selected) > 0

    def test_save_and_load(self, tmp_path: str) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        n_samples = 50
        X = pd.DataFrame({"f1": np.random.randn(n_samples)})
        y = np.abs(np.random.randn(n_samples)) ** 2

        model.train(X, y, epochs=10, verbose=False)
        model.save(tmp_path)

        loaded = TabularModel.load(tmp_path)
        assert loaded.is_trained
        assert loaded.model_type == "xgboost_tweedie"

        # Verify predictions match
        X_test = pd.DataFrame({"f1": np.random.randn(5)})
        pred1 = model.predict(X_test)
        pred2 = loaded.predict(X_test)
        np.testing.assert_allclose(pred1, pred2)

    def test_cross_validate_time_aware(self) -> None:
        model = TabularModel(
            model_type="xgboost_tweedie",
            time_series_split_folds=3,
        )

        n_samples = 60
        timestamps = np.arange(n_samples)  # Chronological order
        X = pd.DataFrame({"f1": np.random.randn(n_samples)})
        y = np.abs(np.random.randn(n_samples)) ** 2

        cv_metrics = model.cross_validate_time_aware(
            X,
            y,
            timestamps,
            epochs=5,
            verbose=False,
        )

        assert "train_metric" in cv_metrics
        assert "val_metric" in cv_metrics
        assert len(cv_metrics["train_metric"]) == 3
        assert len(cv_metrics["val_metric"]) == 3

    def test_train_mismatched_length_raises(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        X = pd.DataFrame({"f1": np.random.randn(50)})
        y = np.random.randn(30)  # Mismatched

        with pytest.raises(Exception):  # ModelError
            model.train(X, y, epochs=10, verbose=False)

    def test_predict_before_train_raises(self) -> None:
        model = TabularModel(model_type="xgboost_tweedie")

        X = pd.DataFrame({"f1": np.random.randn(10)})

        with pytest.raises(Exception):  # ModelError
            model.predict(X)


class TestBuildTabularModel:
    def test_build_from_config(self) -> None:
        from healthrisk_ai.models.tabular.model import build_tabular_model

        model = build_tabular_model()
        assert model.model_type == "xgboost_tweedie"
        assert model.selection_threshold == 0.01
        assert model.time_series_split_folds == 5
