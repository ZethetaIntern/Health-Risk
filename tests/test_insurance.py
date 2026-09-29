"""Unit tests for the Actuarial Desk module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthrisk_ai.insurance.model import (
    ActuarialDesk,
    BornhuetterFergusonEstimator,
    ChainLadderEstimator,
    LossRatioCalculator,
)


class TestChainLadderEstimator:
    def test_fit_on_triangle(self) -> None:
        # Simple triangle: rows are origin years, columns are development periods
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A", "A", "A"],
                "origin_year": [2020, 2021, 2022],
                "1": [100.0, 110.0, 120.0],
                "2": [180.0, 200.0, np.nan],
                "3": [240.0, np.nan, np.nan],
            }
        )

        estimator = ChainLadderEstimator()
        estimator.fit(triangle)

        assert estimator.average_development_factor is not None
        assert estimator.average_development_factor > 0

    def test_predict_ultimate(self) -> None:
        estimator = ChainLadderEstimator()
        estimator.average_development_factor = 1.5  # Mock fitted

        ultimate = estimator.predict_ultimate(
            latest_cumulative=100.0,
            development_period=2,
            total_periods=3,
        )

        # Should be 100 * 1.5^1 = 150
        assert ultimate == pytest.approx(150.0)

    def test_predict_ibnr(self) -> None:
        estimator = ChainLadderEstimator()
        estimator.average_development_factor = 1.5

        ibnr = estimator.predict_ibnr(
            latest_cumulative=100.0,
            development_period=2,
            total_periods=3,
        )

        # Ultimate = 150, IBNR = 150 - 100 = 50
        assert ibnr == pytest.approx(50.0)

    def test_predict_ibnr_by_triangle(self) -> None:
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A", "A"],
                "origin_year": [2020, 2021],
                "1": [100.0, 110.0],
                "2": [180.0, np.nan],
            }
        )

        estimator = ChainLadderEstimator()
        estimator.average_development_factor = 1.3
        estimator.fit(triangle)

        results = estimator.predict_ibnr_by_triangle(triangle, total_periods=2)
        assert len(results) > 0
        assert "predicted_ibnr" in results.columns

    def test_not_fitted_raises(self) -> None:
        estimator = ChainLadderEstimator()

        with pytest.raises(Exception):  # ModelError
            estimator.predict_ultimate(100.0, 1, 3)


class TestBornhuetterFergusonEstimator:
    def test_fit(self) -> None:
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A", "A"],
                "origin_year": [2020, 2021],
                "1": [100.0, 110.0],
                "2": [180.0, np.nan],
            }
        )

        estimator = BornhuetterFergusonEstimator(expected_loss_ratio=0.70)
        estimator.fit(triangle)

        assert estimator.chain_ladder is not None
        assert len(estimator.cumulative_factors) > 0

    def test_predict_ibnr(self) -> None:
        estimator = BornhuetterFergusonEstimator(expected_loss_ratio=0.70)
        estimator.cumulative_factors = {1: 0.5, 2: 0.8, 3: 1.0}  # Mock fitted

        ibnr = estimator.predict_ibnr(
            earned_premium=1000.0,
            development_period=1,
        )

        # Expected ultimate = 0.70 * 1000 = 700
        # IBNR = 700 * (1 - 0.5) = 350
        assert ibnr == pytest.approx(350.0)

    def test_predict_ibnr_with_current_cumulative(self) -> None:
        estimator = BornhuetterFergusonEstimator(expected_loss_ratio=0.70)
        estimator.cumulative_factors = {1: 0.6}

        ibnr = estimator.predict_ibnr(
            earned_premium=1000.0,
            development_period=1,
            current_cumulative=400.0,
        )

        # The current_cumulative is not used in basic BF, only in adjustments
        assert ibnr > 0

    def test_predict_ibnr_by_triangle(self) -> None:
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A", "A"],
                "origin_year": [2020, 2021],
                "1": [100.0, 110.0],
                "2": [180.0, np.nan],
            }
        )

        premium = pd.DataFrame(
            {
                "triangle_id": ["A", "A"],
                "origin_year": [2020, 2021],
                "premium": [1000.0, 1200.0],
            }
        )

        estimator = BornhuetterFergusonEstimator(expected_loss_ratio=0.70)
        estimator.fit(triangle, premium)

        results = estimator.predict_ibnr_by_triangle(triangle, premium)
        assert len(results) > 0
        assert "bf_ibnr" in results.columns
        assert "chain_ladder_ibnr" in results.columns


class TestLossRatioCalculator:
    def test_basic_loss_ratio(self) -> None:
        lr = LossRatioCalculator.calculate_loss_ratio(
            claims=700.0,
            premium=1000.0,
        )
        assert lr == pytest.approx(0.70)

    def test_loss_ratio_with_ibnr(self) -> None:
        lr = LossRatioCalculator.calculate_loss_ratio(
            claims=500.0,
            premium=1000.0,
            include_ibnr=True,
            ibnr=200.0,
        )
        assert lr == pytest.approx(0.70)

    def test_loss_ratio_zero_premium(self) -> None:
        lr = LossRatioCalculator.calculate_loss_ratio(
            claims=100.0,
            premium=0.0,
        )
        assert np.isnan(lr)

    def test_combined_ratio(self) -> None:
        combined = LossRatioCalculator.calculate_combined_ratio(
            loss_ratio=0.70,
            expense_ratio=0.30,
        )
        assert combined == pytest.approx(1.0)

    def test_premium_rate(self) -> None:
        premium = LossRatioCalculator.calculate_premium_rate(
            expected_loss=700.0,
            expected_loss_ratio=0.70,
            load_factor=1.10,
        )
        # Base premium = 700 / 0.70 = 1000
        # With load = 1000 * 1.10 = 1100
        assert premium == pytest.approx(1100.0)


class TestActuarialDesk:
    def test_init_default(self) -> None:
        desk = ActuarialDesk(method="blended")
        assert desk.method == "blended"
        assert not desk.is_fitted

    def test_init_with_method(self) -> None:
        desk = ActuarialDesk(method="chain_ladder")
        assert desk.method == "chain_ladder"

    def test_fit(self) -> None:
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A", "A"],
                "origin_year": [2020, 2021],
                "1": [100.0, 110.0],
                "2": [180.0, np.nan],
            }
        )

        desk = ActuarialDesk()
        desk.fit(triangle)

        assert desk.is_fitted

    def test_estimate_ibnr(self) -> None:
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A", "A"],
                "origin_year": [2020, 2021],
                "1": [100.0, 110.0],
                "2": [180.0, np.nan],
            }
        )

        desk = ActuarialDesk()
        desk.fit(triangle)

        estimates = desk.estimate_ibnr(
            latest_cumulative=100.0,
            earned_premium=150.0,
            development_period=1,
            total_periods=2,
        )

        assert "chain_ladder" in estimates
        assert "bornhuetter_ferguson" in estimates
        assert "blended" in estimates
        assert estimates["chain_ladder"] >= 0
        assert estimates["bornhuetter_ferguson"] >= 0

    def test_estimate_ibnr_not_fitted_raises(self) -> None:
        desk = ActuarialDesk()

        with pytest.raises(Exception):  # ModelError
            desk.estimate_ibnr(100.0, 150.0, 1, 2)

    def test_calculate_premium_rate(self) -> None:
        desk = ActuarialDesk(expected_loss_ratio=0.70)
        desk._is_fitted = True

        premium = desk.calculate_premium_rate(
            expected_loss=700.0,
            load_factor=1.10,
        )

        assert premium == pytest.approx(1100.0)

    def test_save_and_load(self, tmp_path: str) -> None:
        triangle = pd.DataFrame(
            {
                "triangle_id": ["A"],
                "origin_year": [2020],
                "1": [100.0],
                "2": [180.0],
            }
        )

        desk = ActuarialDesk()
        desk.fit(triangle)
        desk.save(tmp_path)

        loaded = ActuarialDesk.load(tmp_path)
        assert loaded.is_fitted
        assert loaded.method == "blended"
