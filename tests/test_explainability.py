"""Unit tests for the Explainability module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthrisk_ai.explainability.model import (
    ExplainabilityModule,
    ModelCardGenerator,
    SHAPExplainer,
)


class TestSHAPExplainer:
    def test_init_auto_model_type(self) -> None:
        # Mock model
        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        model = MockModel()
        explainer = SHAPExplainer(model, feature_names=["f1", "f2"])
        assert explainer.feature_names == ["f1", "f2"]

    def test_init_with_model_type(self) -> None:
        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        model = MockModel()
        explainer = SHAPExplainer(model, model_type="linear")
        assert explainer.model_type == "linear"

    def test_detect_model_type(self) -> None:
        class MockXGBoost:
            pass

        class MockLinear:
            pass

        explainer = SHAPExplainer(MockXGBoost())
        assert explainer._detect_model_type() == "xgboost" or explainer._detect_model_type() == "unknown"

    def test_explain_without_fit(self) -> None:
        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        explainer = SHAPExplainer(MockModel())
        X = np.random.randn(10, 5)

        # Without fitting, should raise ValueError
        with pytest.raises(ValueError, match="Explainer must be fitted"):
            explainer.explain(X)

    def test_get_feature_importance(self) -> None:
        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        explainer = SHAPExplainer(MockModel(), feature_names=["f1", "f2", "f3"])
        X = np.random.randn(50, 3)

        # Mock fit
        explainer._is_fitted = True

        importance = explainer.get_feature_importance(X, n_samples=10)
        assert len(importance) == 3
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_get_local_explanation(self) -> None:
        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        explainer = SHAPExplainer(MockModel(), feature_names=["f1", "f2", "f3"])
        x = np.random.randn(3)

        explainer._is_fitted = True
        explainer._explainer = None  # Simulate no SHAP

        explanation = explainer.get_local_explanation(x, top_k=2)
        # Without SHAP, returns warning
        assert "all_features" in explanation or "warning" in explanation


class TestModelCardGenerator:
    def test_init_default(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        assert card.model_name == "TestModel"
        assert card.model_type == "Tabular"
        assert card.version == "1.0.0"

    def test_set_metadata(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_metadata(
            training_data="MIMIC-IV synthetic data",
            hyperparameters={"learning_rate": 0.01, "epochs": 100},
        )

        assert card.metadata["training_data"] == "MIMIC-IV synthetic data"
        assert card.metadata["hyperparameters"]["learning_rate"] == 0.01

    def test_set_performance_metrics(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_performance_metrics({
            "accuracy": 0.85,
            "f1_score": 0.82,
            "auc": 0.88,
        })

        assert card.performance_metrics["accuracy"] == 0.85

    def test_add_ethical_consideration(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.add_ethical_consideration("May exhibit bias on underrepresented populations")
        card.add_ethical_consideration("Not intended for direct clinical diagnosis")

        assert len(card.ethical_considerations) == 2
        assert "bias" in card.ethical_considerations[0]

    def test_add_limitation(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.add_limitation("Requires structured EHR data for input")
        card.add_limitation("Performance degrades on rare conditions")

        assert len(card.limitations) == 2

    def test_set_intended_use(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_intended_use(
            use_cases=["Risk stratification", "Population health management"],
            out_of_scope=["Individual diagnosis", "Treatment recommendation"],
        )

        assert len(card.intended_use_cases) == 2
        assert len(card.out_of_scope_use_cases) == 2

    def test_generate_html(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_metadata(training_data="Test data")
        card.set_performance_metrics({"accuracy": 0.85})
        card.set_intended_use(["Risk stratification"])
        card.add_limitation("Test limitation")
        card.add_ethical_consideration("Test consideration")

        html = card.generate_html()

        assert "TestModel" in html
        assert "Tabular" in html
        assert "0.85" in html
        assert "Risk stratification" in html

    def test_generate_html_with_feature_importance(self) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_performance_metrics({"accuracy": 0.85})

        feature_importance = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "importance": [0.5, 0.3, 0.2],
        })

        html = card.generate_html(feature_importance=feature_importance)

        assert "f1" in html
        assert "0.5" in html or "0.5000" in html

    def test_save_html(self, tmp_path: str) -> None:
        from pathlib import Path
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_performance_metrics({"accuracy": 0.85})

        path = f"{tmp_path}/model_card.html"
        card.save_html(path)

        assert Path(path).exists()
        content = Path(path).read_text()
        assert "TestModel" in content

    def test_save_yaml(self, tmp_path: str) -> None:
        card = ModelCardGenerator("TestModel", "Tabular")
        card.set_metadata(training_data="Test data")
        card.set_performance_metrics({"accuracy": 0.85})

        path = f"{tmp_path}/model_card.yaml"
        card.save_yaml(path)

        from pathlib import Path
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "TestModel" in content
        assert "accuracy" in content


class TestExplainabilityModule:
    def test_init(self) -> None:
        module = ExplainabilityModule()
        assert len(module.explainers) == 0
        assert len(module.model_cards) == 0

    def test_register_model(self) -> None:
        module = ExplainabilityModule()

        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        module.register_model("test_model", MockModel(), feature_names=["f1", "f2"])

        assert "test_model" in module.explainers
        assert "test_model" in module.model_cards

    def test_explain_model(self) -> None:
        module = ExplainabilityModule()

        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        module.register_model("test_model", MockModel(), feature_names=["f1", "f2"])
        X = np.random.randn(20, 2)

        explanation = module.explain_model("test_model", X)

        # May or may not have SHAP values depending on install
        assert "feature_names" in explanation

    def test_get_feature_importance(self) -> None:
        module = ExplainabilityModule()

        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        module.register_model("test_model", MockModel(), feature_names=["f1", "f2"])
        X = np.random.randn(20, 2)

        importance = module.get_feature_importance("test_model", X)

        assert len(importance) == 2
        assert "feature" in importance.columns

    def test_generate_model_card(self, tmp_path: str) -> None:
        module = ExplainabilityModule()

        class MockModel:
            def predict(self, X):
                return np.mean(X, axis=1)

        module.register_model("test_model", MockModel(), feature_names=["f1", "f2"])
        X = np.random.randn(20, 2)

        importance = module.get_feature_importance("test_model", X)
        html = module.generate_model_card(
            "test_model",
            feature_importance=importance,
            output_path=f"{tmp_path}/test_model_card.html",
        )

        assert "test_model" in html
        assert "f1" in html

    def test_explain_unregistered_model_raises(self) -> None:
        module = ExplainabilityModule()

        with pytest.raises(ValueError, match="Model not registered"):
            module.explain_model("non_existent_model", np.array([[1, 2]]))
