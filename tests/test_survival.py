"""Unit tests for the Survival analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from healthrisk_ai.models.survival.model import (
    DeepSurv,
    DynamicDeepHit,
    SurvivalDataset,
    SurvivalModel,
)


class TestSurvivalDataset:
    def test_dataset_creation(self) -> None:
        features = np.random.randn(100, 10)
        times = np.random.exponential(10, 100)
        events = np.random.binomial(1, 0.3, 100)

        dataset = SurvivalDataset(features, times, events)
        assert len(dataset) == 100

    def test_dataset_getitem(self) -> None:
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        times = np.array([10.0, 20.0])
        events = np.array([1.0, 0.0])

        dataset = SurvivalDataset(features, times, events)
        item = dataset[0]

        assert "features" in item
        assert "time" in item
        assert "event" in item
        assert item["features"].shape == (2,)
        assert item["time"] == 10.0
        assert item["event"] == 1.0


class TestDeepSurv:
    def test_model_creation(self) -> None:
        model = DeepSurv(input_dim=20, hidden_layers=[64, 32])
        assert model.network is not None

    def test_forward_pass(self) -> None:
        model = DeepSurv(input_dim=20, hidden_layers=[64, 32])
        x = np.random.randn(5, 20).astype(np.float32)
        x_tensor = np.array(x)

        import torch
        output = model(torch.tensor(x_tensor))
        assert output.shape == (5,)

    def test_cox_loss(self) -> None:
        import torch
        model = DeepSurv(input_dim=10, hidden_layers=[32])
        x = torch.randn(20, 10)
        times = torch.rand(20) * 100
        events = torch.binomial(torch.ones(20), torch.tensor(0.3))

        loss = model.cox_partial_likelihood_loss(x, times, events)
        assert loss.item() >= 0


class TestDynamicDeepHit:
    def test_model_creation(self) -> None:
        model = DynamicDeepHit(input_dim=20, max_time_steps=30)
        assert model.shared_network is not None
        assert len(model.time_heads) == 30

    def test_forward_pass_all_times(self) -> None:
        import torch
        model = DynamicDeepHit(input_dim=20, max_time_steps=10)
        x = torch.randn(5, 20)

        output = model(x)
        assert output.shape == (5, 10, 2)

    def test_forward_pass_selected_times(self) -> None:
        import torch
        model = DynamicDeepHit(input_dim=20, max_time_steps=30)
        x = torch.randn(5, 20)
        times = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0])

        output = model(x, times)
        assert output.shape == (5, 2)


class TestSurvivalModel:
    def test_init_deepsurv(self) -> None:
        model = SurvivalModel(
            input_dim=20,
            model_type="deepsurv",
            hidden_layers=[64, 32],
            device="cpu",
        )
        assert model.model_type == "deepsurv"
        assert not model.is_trained

    def test_init_deepdephit(self) -> None:
        model = SurvivalModel(
            input_dim=20,
            model_type="deepdephit",
            hidden_layers=[64, 32],
            max_time_steps=20,
            device="cpu",
        )
        assert model.model_type == "deepdephit"

    def test_train_small(self) -> None:
        model = SurvivalModel(
            input_dim=10,
            model_type="deepsurv",
            hidden_layers=[16],
            device="cpu",
        )

        n_samples = 50
        features = np.random.randn(n_samples, 10)
        times = np.random.exponential(10, n_samples)
        events = np.random.binomial(1, 0.4, n_samples)

        history = model.train(
            features,
            times,
            events,
            epochs=5,
            batch_size=16,
            verbose=False,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 5
        assert model.is_trained

    def test_predict_hazard(self) -> None:
        model = SurvivalModel(
            input_dim=10,
            model_type="deepsurv",
            hidden_layers=[16],
            device="cpu",
        )

        # Train briefly
        features = np.random.randn(30, 10)
        times = np.random.exponential(10, 30)
        events = np.random.binomial(1, 0.4, 30)
        model.train(features, times, events, epochs=3, verbose=False)

        # Predict
        test_features = np.random.randn(5, 10)
        hazards = model.predict_hazard(test_features)
        assert hazards.shape == (5,)

    def test_predict_survival(self) -> None:
        model = SurvivalModel(
            input_dim=10,
            model_type="deepsurv",
            hidden_layers=[16],
            device="cpu",
        )

        features = np.random.randn(30, 10)
        times = np.random.exponential(10, 30)
        events = np.random.binomial(1, 0.4, 30)
        model.train(features, times, events, epochs=3, verbose=False)

        test_features = np.random.randn(5, 10)
        survival = model.predict_survival_probability(test_features)
        assert survival.shape == (5,)
        assert np.all(survival >= 0) and np.all(survival <= 1)

    def test_save_and_load(self, tmp_path: str) -> None:
        model = SurvivalModel(
            input_dim=10,
            model_type="deepsurv",
            hidden_layers=[16],
            device="cpu",
        )
        model._is_trained = True
        model.save(tmp_path)

        loaded = SurvivalModel.load(tmp_path, device="cpu")
        assert loaded.is_trained
        assert loaded.model_type == "deepsurv"

    def test_invalid_model_type(self) -> None:
        with pytest.raises(Exception):  # ModelError
            SurvivalModel(
                input_dim=10,
                model_type="invalid",
                device="cpu",
            )


class TestBuildSurvivalModel:
    def test_build_from_config(self) -> None:
        from healthrisk_ai.models.survival.model import build_survival_model

        model = build_survival_model()
        assert model.input_dim == 96  # 64 (NLP) + 32 (GNN)
        assert model.model_type == "deepsurv"
        assert model.cindex_target == 0.70
