"""Unit tests for the Hospital Credit Scorecard module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthrisk_ai.credit_risk.model import (
    CreditPortfolio,
    HospitalCreditScorecard,
)


class TestHospitalCreditScorecard:
    def test_init_default(self) -> None:
        scorecard = HospitalCreditScorecard()
        assert len(scorecard.weights) > 0
        assert "dscr" in scorecard.weights
        assert not scorecard.is_fitted

    def test_calculate_score_basic(self) -> None:
        scorecard = HospitalCreditScorecard()

        financial_ratios = {
            "dscr": 1.5,
            "days_cash_on_hand": 90,
            "current_ratio": 1.5,
            "operating_margin": 0.05,
            "debt_to_revenue": 0.15,
        }

        result = scorecard.calculate_score(financial_ratios)

        assert "credit_score" in result
        assert "probability_of_default" in result
        assert "credit_rating" in result
        assert "factor_scores" in result

        assert 300 <= result["credit_score"] <= 850
        assert 0 < result["probability_of_default"] < 1
        assert isinstance(result["credit_rating"], str)

    def test_calculate_score_strong_hospital(self) -> None:
        scorecard = HospitalCreditScorecard()

        financial_ratios = {
            "dscr": 2.5,
            "days_cash_on_hand": 200,
            "current_ratio": 2.5,
            "operating_margin": 0.12,
            "debt_to_revenue": 0.05,
            "readmission_rate": 0.08,
            "infection_rate": 0.01,
            "cms_penalty": 0,
            "occupancy_rate": 0.85,
            "cmi": 1.2,
        }

        result = scorecard.calculate_score(financial_ratios)

        assert result["credit_score"] >= 600  # Should be investment grade
        assert result["probability_of_default"] < 0.05

    def test_calculate_score_distressed_hospital(self) -> None:
        scorecard = HospitalCreditScorecard()

        financial_ratios = {
            "dscr": 0.5,
            "days_cash_on_hand": 15,
            "current_ratio": 0.5,
            "operating_margin": -0.10,
            "debt_to_revenue": 0.40,
            "readmission_rate": 0.20,
            "infection_rate": 0.05,
            "cms_penalty": 1,
            "occupancy_rate": 0.50,
            "cmi": 0.8,
        }

        result = scorecard.calculate_score(financial_ratios)

        assert result["credit_score"] < 500  # Should be below investment grade
        assert result["probability_of_default"] > 0.10

    def test_pd_surface(self) -> None:
        scorecard = HospitalCreditScorecard()

        base_ratios = {
            "dscr": 1.5,
            "days_cash_on_hand": 90,
        }

        surface = scorecard.calculate_pd_surface(
            base_ratios,
            vary_factor="dscr",
            vary_range=(0.5, 3.0, 10),
        )

        assert len(surface) == 10
        assert "dscr" in surface.columns
        assert "probability_of_default" in surface.columns

        # Higher DSCR should generally mean lower PD
        assert surface["probability_of_default"].iloc[-1] < surface["probability_of_default"].iloc[0]

    def test_credit_rating_mapping(self) -> None:
        scorecard = HospitalCreditScorecard()

        # Test rating boundaries
        assert scorecard._score_to_rating(800) == "AAA"
        assert scorecard._score_to_rating(760) == "AA"
        assert scorecard._score_to_rating(720) == "A"
        assert scorecard._score_to_rating(660) == "BBB"
        assert scorecard._score_to_rating(610) == "BB"
        assert scorecard._score_to_rating(560) == "B"
        assert scorecard._score_to_rating(510) == "CCC"
        assert scorecard._score_to_rating(410) == "CC"
        assert scorecard._score_to_rating(310) == "C"
        assert scorecard._score_to_rating(200) == "D"

    def test_pd_calibration(self) -> None:
        scorecard = HospitalCreditScorecard()

        # High score should mean low PD
        pd_high = scorecard._score_to_pd(800)
        pd_low = scorecard._score_to_pd(300)

        assert pd_high < pd_low
        # With default calibration params (a=-0.04, b=20), PD is high for all scores
        # This test just checks relative ordering

    def test_stress_test(self) -> None:
        scorecard = HospitalCreditScorecard()

        base_ratios = {
            "dscr": 1.5,
            "days_cash_on_hand": 90,
            "operating_margin": 0.05,
        }

        scenarios = {
            "Mild Stress": {"dscr": 0.8, "operating_margin": 0.7},
            "Severe Stress": {"dscr": 0.5, "days_cash_on_hand": 0.3, "operating_margin": 0.3},
        }

        results = scorecard.stress_test(base_ratios, scenarios)

        assert len(results) == 3  # Base + 2 scenarios
        assert "scenario" in results.columns

        # Severe stress should have worse PD
        severe = results[results["scenario"] == "Severe Stress"].iloc[0]
        base = results[results["scenario"] == "Base Case"].iloc[0]

        assert severe["probability_of_default"] > base["probability_of_default"]

    def test_predict_pd_heuristic(self) -> None:
        scorecard = HospitalCreditScorecard()

        X = pd.DataFrame({
            "dscr": [1.5, 0.8, 2.0],
            "days_cash_on_hand": [90, 30, 180],
            "operating_margin": [0.05, -0.02, 0.10],
        })

        pd_values = scorecard.predict_pd(X)
        assert len(pd_values) == 3
        assert np.all(pd_values > 0)
        assert np.all(pd_values < 1)

    def test_predict_pd_with_scores(self) -> None:
        scorecard = HospitalCreditScorecard()

        X = pd.DataFrame({
            "dscr": [1.5, 0.8],
            "days_cash_on_hand": [90, 30],
        })

        result = scorecard.predict_pd(X, return_scores=True)
        assert "pd" in result
        assert "scores" in result
        assert len(result["pd"]) == 2
        assert len(result["scores"]) == 2

    def test_save_and_load(self, tmp_path: str) -> None:
        scorecard = HospitalCreditScorecard()
        scorecard._is_fitted = True

        scorecard.save(tmp_path)
        loaded = HospitalCreditScorecard.load(tmp_path)

        assert loaded.is_fitted
        assert loaded.weights == scorecard.weights


class TestCreditPortfolio:
    def test_init(self) -> None:
        exposures = pd.DataFrame({
            "hospital_id": ["H1", "H2", "H3"],
            "exposure": [100.0, 200.0, 300.0],
        })
        pds = np.array([0.01, 0.02, 0.03])

        portfolio = CreditPortfolio(exposures, pds, loss_given_default=0.40)
        assert portfolio.lgd == 0.40

    def test_expected_loss(self) -> None:
        exposures = pd.DataFrame({
            "hospital_id": ["H1", "H2"],
            "exposure": [100.0, 200.0],
        })
        pds = np.array([0.01, 0.02])

        portfolio = CreditPortfolio(exposures, pds, loss_given_default=0.40)
        els = portfolio.calculate_expected_loss()

        # EL1 = 0.01 * 0.40 * 100 = 0.4
        # EL2 = 0.02 * 0.40 * 200 = 1.6
        np.testing.assert_allclose(els, [0.4, 1.6])

    def test_portfolio_expected_loss(self) -> None:
        exposures = pd.DataFrame({
            "hospital_id": ["H1", "H2", "H3"],
            "exposure": [100.0, 200.0, 300.0],
        })
        pds = np.array([0.01, 0.02, 0.03])

        portfolio = CreditPortfolio(exposures, pds, loss_given_default=0.40)
        total_el = portfolio.portfolio_expected_loss()

        # Total = 0.4 + 1.6 + 3.6 = 5.6
        assert total_el == pytest.approx(5.6)

    def test_concentration_ratio(self) -> None:
        exposures = pd.DataFrame({
            "hospital_id": ["H1", "H2", "H3", "H4", "H5"],
            "exposure": [500.0, 100.0, 100.0, 100.0, 100.0],
        })
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        portfolio = CreditPortfolio(exposures, pds)
        concentration = portfolio.concentration_ratio(top_n=1)

        # H1 has 500 out of 900 total = 55.6%
        assert concentration == pytest.approx(500 / 900)
