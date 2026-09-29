"""Hospital Credit Scorecard: Probability of Default estimation.

This module translates clinical quality metrics and financial ratios into
Probability of Default (PD) estimates for hospital credit scoring.

Features:
- Multi-factor scorecard based on financial health and clinical quality
- PD calibration using logistic regression style mapping
- Credit rating assignment (AAA to D)
- Stress testing under adverse scenarios
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from healthrisk_ai.exceptions import ModelError

logger = logging.getLogger("healthrisk_ai.credit_risk")


class HospitalCreditScorecard:
    """Credit scorecard for hospital probability of default.

    Inputs:
    - Financial ratios: DSCR, Days Cash on Hand, Current Ratio, Operating Margin
    - Clinical quality: Readmission rate, infection rate, CMS penalty flag
    - Operational: Occupancy rate, CMI, case mix

    Output:
    - Credit score (300-850 scale)
    - Probability of Default (PD)
    - Credit rating (AAA, AA, A, BBB, BB, B, CCC, CC, C, D)

    Scoring methodology:
    - Each factor contributes points based on its value
    - Total points mapped to credit score
    - Credit score mapped to PD via logistic function
    """

    # Credit rating thresholds (score -> rating)
    RATING_THRESHOLDS = [
        (800, "AAA"),
        (750, "AA"),
        (700, "A"),
        (650, "BBB"),
        (600, "BB"),
        (550, "B"),
        (500, "CCC"),
        (400, "CC"),
        (300, "C"),
        (0, "D"),
    ]

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        calibration_params: dict[str, float] | None = None,
    ) -> None:
        """Initialize the scorecard.

        Args:
            weights: Factor weights for scoring. Defaults to balanced weights.
            calibration_params: PD calibration parameters.
        """
        # Default weights for each factor (higher = more important)
        self.weights = weights or {
            "dscr": 0.15,
            "days_cash_on_hand": 0.15,
            "current_ratio": 0.10,
            "operating_margin": 0.15,
            "readmission_rate": 0.10,
            "infection_rate": 0.10,
            "cms_penalty": 0.10,
            "occupancy_rate": 0.05,
            "cmi": 0.05,
            "debt_to_revenue": 0.05,
        }

        # Calibration: maps score to PD via logistic function
        # PD = 1 / (1 + exp(a * score + b)) with a > 0 and b < 0 so that PD
        # decreases monotonically as the credit score increases (PD = 50% at
        # score = -b / a, i.e. 600 on the default calibration).
        self.calibration_params = calibration_params or {
            "a": 0.02,
            "b": -12.0,
        }

        self._scaler: StandardScaler | None = None
        self._logistic_model: LogisticRegression | None = None
        self._is_fitted = False

    def _score_factor(
        self,
        value: float,
        factor_name: str,
        *,
        higher_is_better: bool = True,
        target_value: float | None = None,
    ) -> float:
        """Score a single factor on a 0-100 scale.

        Args:
            value: Factor value.
            factor_name: Name of the factor.
            higher_is_better: Whether higher values are better.
            target_value: Ideal/target value (for normalization).

        Returns:
            Score from 0 (worst) to 100 (best).
        """
        weight = self.weights.get(factor_name, 0.05)

        # Factor-specific scoring logic
        if factor_name == "dscr":
            # DSCR: Debt Service Coverage Ratio
            # < 1.0 = distressed, > 2.0 = excellent
            if value <= 0:
                return 0.0
            score = min(100, max(0, (value - 1.0) / 1.0 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "days_cash_on_hand":
            # Days Cash on Hand
            # < 30 = distressed, > 180 = excellent
            if value <= 0:
                return 0.0
            score = min(100, max(0, (value - 30) / 150 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "current_ratio":
            # Current Ratio
            # < 0.5 = distressed, > 3.0 = excellent
            if value <= 0:
                return 0.0
            score = min(100, max(0, (value - 0.5) / 2.5 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "operating_margin":
            # Operating Margin (as decimal, e.g. 0.05 = 5%)
            # -0.10 = worst (score 0), +0.15 = best (score 100)
            score = min(100, max(0, (value + 0.10) / 0.25 * 100))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "readmission_rate":
            # Readmission rate (as decimal)
            # > 0.20 = poor, < 0.10 = excellent
            # The formula already rewards lower values, so returning the score
            # as-is under the default (higher_is_better=True) call gives the
            # correct orientation.
            score = min(100, max(0, (0.20 - value) / 0.10 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "infection_rate":
            # Hospital-acquired infection rate (as decimal)
            # > 0.05 = poor, < 0.01 = excellent
            # Same orientation as readmission_rate: the formula already
            # rewards lower values.
            score = min(100, max(0, (0.05 - value) / 0.04 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "cms_penalty":
            # CMS penalty flag (1 = penalized, 0 = no penalty)
            score = 100 if value == 0 else 20
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "occupancy_rate":
            # Occupancy rate (as decimal)
            # < 50% = poor, > 90% = excellent
            score = min(100, max(0, (value - 0.50) / 0.40 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "cmi":
            # Case Mix Index
            # < 0.8 = low complexity, > 2.0 = high complexity
            # Moderate CMI is good (indicates appropriate acuity)
            if value <= 0:
                return 50.0
            if value < 1.0:
                score = 50 + (value - 0.8) / 0.2 * 25
            else:
                score = 75 - (value - 1.0) / 1.0 * 25
            score = min(100, max(0, score))
            if higher_is_better:
                return score
            return 100 - score

        elif factor_name == "debt_to_revenue":
            # Debt to Revenue ratio (as decimal)
            # > 0.30 = high leverage, < 0.10 = low leverage
            score = min(100, max(0, (0.30 - value) / 0.20 * 50 + 50))
            if higher_is_better:
                return score
            return 100 - score

        # Default scoring for unknown factors
        return 50.0

    def calculate_score(
        self,
        financial_ratios: dict[str, float],
        clinical_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Calculate credit score from financial and clinical metrics.

        Args:
            financial_ratios: Dict of financial ratios (dscr, days_cash_on_hand, etc.).
            clinical_metrics: Optional dict of clinical quality metrics.

        Returns:
            Dict with total_score, factor_scores, credit_rating, pd.
        """
        clinical_metrics = clinical_metrics or {}

        # Combine all factors
        all_factors = {**financial_ratios, **clinical_metrics}

        # Calculate individual factor scores
        factor_scores: dict[str, float] = {}
        total_weighted_score = 0.0
        total_weight = 0.0

        for factor_name, value in all_factors.items():
            if factor_name not in self.weights:
                continue

            score = self._score_factor(value, factor_name)
            weight = self.weights[factor_name]

            factor_scores[factor_name] = score
            total_weighted_score += score * weight
            total_weight += weight

        # Normalize to get final score (0-100 scale, then map to 300-850)
        if total_weight > 0:
            normalized_score = total_weighted_score / total_weight
        else:
            normalized_score = 50.0

        # Map to credit score scale (300-850)
        credit_score = 300 + (normalized_score / 100) * 550

        # Calculate PD
        pd_estimate = self._score_to_pd(credit_score)

        # Assign rating
        rating = self._score_to_rating(credit_score)

        return {
            "credit_score": float(credit_score),
            "normalized_score": float(normalized_score),
            "probability_of_default": float(pd_estimate),
            "credit_rating": rating,
            "factor_scores": factor_scores,
        }

    def _score_to_pd(self, score: float) -> float:
        """Convert credit score to Probability of Default.

        Uses logistic function: PD = 1 / (1 + exp(a * score + b))

        Args:
            score: Credit score (300-850).

        Returns:
            Annual probability of default (0-1).
        """
        a = self.calibration_params["a"]
        b = self.calibration_params["b"]

        logit = a * score + b
        pd = 1.0 / (1.0 + np.exp(logit))

        return float(np.clip(pd, 0.0001, 0.9999))

    def _score_to_rating(self, score: float) -> str:
        """Convert credit score to rating category.

        Args:
            score: Credit score.

        Returns:
            Rating string (AAA, AA, A, BBB, etc.).
        """
        for threshold, rating in self.RATING_THRESHOLDS:
            if score >= threshold:
                return rating
        return "D"

    def calculate_pd_surface(
        self,
        base_ratios: dict[str, float],
        vary_factor: str,
        vary_range: tuple[float, float, int],
        *,
        clinical_metrics: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Calculate PD across a range of values for one factor.

        Useful for stress testing and sensitivity analysis.

        Args:
            base_ratios: Base financial ratios.
            vary_factor: Factor to vary.
            vary_range: (start, end, num_points) for the factor.
            clinical_metrics: Optional clinical metrics.

        Returns:
            DataFrame with factor values and corresponding PDs.
        """
        clinical_metrics = clinical_metrics or {}
        start, end, num = vary_range
        values = np.linspace(start, end, num)

        results: list[dict[str, Any]] = []

        for value in values:
            # Copy base ratios and vary the selected factor
            ratios = base_ratios.copy()
            ratios[vary_factor] = value

            score_result = self.calculate_score(ratios, clinical_metrics)

            results.append({
                vary_factor: value,
                "credit_score": score_result["credit_score"],
                "probability_of_default": score_result["probability_of_default"],
                "credit_rating": score_result["credit_rating"],
            })

        return pd.DataFrame(results)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        factor_columns: list[str] | None = None,
        cv: int = 5,
        random_state: int = 20260914,
    ) -> HospitalCreditScorecard:
        """Fit the scorecard using historical default data.

        Args:
            X: Feature matrix (financial ratios, clinical metrics).
            y: Default indicator (1 = defaulted, 0 = survived).
            factor_columns: Columns to use (default: all).
            cv: Cross-validation folds.
            random_state: Random seed.

        Returns:
            Self for chaining.
        """
        if factor_columns is None:
            factor_columns = list(X.columns)

        # Filter to known factors
        known_factors = [c for c in factor_columns if c in self.weights]
        if not known_factors:
            logger.warning("No known factors found in data; using defaults")
            known_factors = list(self.weights.keys())[: min(5, len(X.columns))]

        X_filtered = X[known_factors].values
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_filtered)

        self._logistic_model = LogisticRegression(
            cv=cv,
            random_state=random_state,
            max_iter=1000,
        )
        self._logistic_model.fit(X_scaled, y)

        # Update weights based on model coefficients
        if self._logistic_model is not None:
            coefs = self._logistic_model.coef_[0]
            for i, col in enumerate(known_factors):
                if col in self.weights:
                    # Scale weight by absolute coefficient
                    self.weights[col] = float(np.abs(coefs[i]))

        self._is_fitted = True
        logger.info(
            "Scorecard fitted on %d samples with %d factors",
            len(X),
            len(known_factors),
        )

        return self

    def predict_pd(
        self,
        X: pd.DataFrame,
        *,
        return_scores: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict PD for new observations.

        Args:
            X: Feature matrix.
            return_scores: If True, also return credit scores.

        Returns:
            PD array or dict with 'pd' and optionally 'scores'.
        """
        if not self._is_fitted:
            # Use heuristic scoring if not fitted
            results = []
            for _, row in X.iterrows():
                result = self.calculate_score(
                    row.to_dict(),
                    clinical_metrics={},
                )
                results.append(result["probability_of_default"])

            if return_scores:
                scores = []
                for _, row in X.iterrows():
                    result = self.calculate_score(row.to_dict(), {})
                    scores.append(result["credit_score"])
                return {"pd": np.array(results), "scores": np.array(scores)}

            return np.array(results)

        if self._scaler is None or self._logistic_model is None:
            msg = "Model not properly fitted"
            raise ModelError(msg)

        factor_columns = [c for c in X.columns if c in self.weights]
        if not factor_columns:
            factor_columns = list(self.weights.keys())[: min(5, len(X.columns))]

        X_filtered = X[factor_columns].values
        X_scaled = self._scaler.transform(X_filtered)

        pd_values = self._logistic_model.predict_proba(X_scaled)[:, 1]

        if return_scores:
            # Approximate score from PD
            scores = np.array([
                (np.log(pd / (1 - pd)) - self.calibration_params["b"])
                / self.calibration_params["a"]
                for pd in pd_values
            ])
            return {"pd": pd_values, "scores": scores}

        return pd_values

    def stress_test(
        self,
        base_ratios: dict[str, float],
        scenarios: dict[str, dict[str, float]],
        *,
        clinical_metrics: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Run stress testing under multiple scenarios.

        Args:
            base_ratios: Base case financial ratios.
            scenarios: Dict of scenario name -> factor adjustments.
                      Adjustments are multiplicative (e.g. {"dscr": 0.5} halves DSCR).
            clinical_metrics: Optional clinical metrics.

        Returns:
            DataFrame with scenario results.
        """
        clinical_metrics = clinical_metrics or {}
        results: list[dict[str, Any]] = []

        # Base case
        base_result = self.calculate_score(base_ratios, clinical_metrics)
        results.append({
            "scenario": "Base Case",
            **base_result,
        })

        # Stress scenarios
        for scenario_name, adjustments in scenarios.items():
            stressed_ratios = base_ratios.copy()

            for factor, multiplier in adjustments.items():
                if factor in stressed_ratios:
                    stressed_ratios[factor] *= multiplier

            result = self.calculate_score(stressed_ratios, clinical_metrics)
            results.append({
                "scenario": scenario_name,
                **result,
            })

        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        """Save scorecard."""
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "scorecard.pkl"), "wb") as f:
            pickle.dump(
                {
                    "weights": self.weights,
                    "calibration_params": self.calibration_params,
                    "is_fitted": self._is_fitted,
                    "scaler": self._scaler,
                    "logistic_model": self._logistic_model,
                },
                f,
            )

        logger.info("Saved credit scorecard to %s", path)

    @classmethod
    def load(cls, path: str) -> HospitalCreditScorecard:
        """Load a saved scorecard."""
        import os
        import pickle

        with open(os.path.join(path, "scorecard.pkl"), "rb") as f:
            data = pickle.load(f)

        scorecard = cls(
            weights=data["weights"],
            calibration_params=data["calibration_params"],
        )
        scorecard._is_fitted = data["is_fitted"]
        scorecard._scaler = data.get("scaler")
        scorecard._logistic_model = data.get("logistic_model")

        logger.info("Loaded credit scorecard from %s", path)
        return scorecard

    @property
    def is_fitted(self) -> bool:
        """Whether the scorecard has been fitted."""
        return self._is_fitted


class CreditPortfolio:
    """Portfolio-level credit risk aggregation.

    Aggregates individual hospital PDs into portfolio metrics:
    - Expected loss (EL)
    - Value at Risk (VaR)
    - Expected shortfall (ES)
    - Concentration metrics
    """

    def __init__(
        self,
        exposures: pd.DataFrame,
        pd_estimates: np.ndarray,
        loss_given_default: float = 0.40,
    ) -> None:
        """Initialize portfolio.

        Args:
            exposures: DataFrame with 'hospital_id' and 'exposure' columns.
            pd_estimates: Array of PDs for each hospital.
            loss_given_default: Loss given default (0-1).
        """
        self.exposures = exposures
        self.pd_estimates = np.array(pd_estimates)
        self.lgd = loss_given_default
        self._expected_losses: np.ndarray | None = None

    def calculate_expected_loss(self) -> np.ndarray:
        """Calculate expected loss for each obligor.

        EL = PD * LGD * Exposure

        Returns:
            Array of expected losses.
        """
        self._expected_losses = (
            self.pd_estimates
            * self.lgd
            * self.exposures["exposure"].values
        )
        return self._expected_losses

    def portfolio_expected_loss(self) -> float:
        """Calculate total portfolio expected loss.

        Returns:
            Total EL.
        """
        if self._expected_losses is None:
            self.calculate_expected_loss()
        return float(self._expected_losses.sum())

    def portfolio_var(self, confidence_level: float = 0.99) -> float:
        """Calculate Value at Risk at given confidence level.

        Uses normal approximation for portfolio loss distribution.

        Args:
            confidence_level: Confidence level (e.g. 0.99 for 99% VaR).

        Returns:
            Portfolio VaR.
        """
        if self._expected_losses is None:
            self.calculate_expected_losses()

        exposures = self.exposures["exposure"].values

        # Mean loss
        mean_loss = self._expected_losses.sum()

        # Variance approximation (assuming independence)
        variances = (
            self.pd_estimates
            * (1 - self.pd_estimates)
            * (self.lgd ** 2)
            * (exposures ** 2)
        )
        std_loss = np.sqrt(variances.sum())

        # Normal VaR
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)

        var = mean_loss + z_score * std_loss
        return float(var)

    def portfolio_expected_shortfall(self, confidence_level: float = 0.99) -> float:
        """Calculate Expected Shortfall (CVaR) at given confidence level.

        Args:
            confidence_level: Confidence level.

        Returns:
            Portfolio Expected Shortfall.
        """
        var = self.portfolio_var(confidence_level)

        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        std_loss = np.sqrt(
            (self.pd_estimates * (1 - self.pd_estimates) * (self.lgd ** 2) *
             self.exposures["exposure"].values ** 2).sum()
        )
        mean_loss = self.portfolio_expected_loss()

        # ES = mean + std * (phi(z) / (1 - alpha))
        phi_z = norm.pdf(z_score)
        es = mean_loss + std_loss * phi_z / (1 - confidence_level)

        return float(max(0, es))

    def concentration_ratio(self, top_n: int = 5) -> float:
        """Calculate concentration ratio for top N exposures.

        Args:
            top_n: Number of largest exposures to consider.

        Returns:
            Percentage of total exposure in top N.
        """
        total_exposure = self.exposures["exposure"].sum()
        top_exposures = self.exposures.nlargest(top_n, "exposure")["exposure"].sum()

        return float(top_exposures / total_exposure) if total_exposure > 0 else 0.0
