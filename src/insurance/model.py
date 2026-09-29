"""Actuarial Desk: IBNR reserve estimation and premium pricing.

This module implements actuarial methods for:
- Chain Ladder (development factor) IBNR estimation
- Bornhuetter-Ferguson IBNR estimation
- Loss ratio calculation
- Premium pricing algorithms

These methods consume claim development triangles from the financial features
pipeline and produce reserve estimates for the HealthRisk Lab simulation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from healthrisk_ai.exceptions import ModelError

logger = logging.getLogger("healthrisk_ai.insurance")


class ChainLadderEstimator:
    """Chain Ladder (development factor) IBNR reserve estimator.

    The Chain Ladder method projects ultimate claims from historical development
    patterns using age-to-age factors.

    Ultimate claim = Latest cumulative claim * Product of remaining development factors

    IBNR = Ultimate claim - Latest cumulative claim
    """

    def __init__(self) -> None:
        self.development_factors: dict[int, float] = {}
        self.average_development_factor: float | None = None

    def fit(
        self,
        triangle: pd.DataFrame,
        triangle_id_col: str = "triangle_id",
        origin_year_col: str = "origin_year",
        cum_columns: list[str] | None = None,
    ) -> ChainLadderEstimator:
        """Fit development factors from a claim triangle.

        Args:
            triangle: Cumulative claim triangle (pivoted format).
            triangle_id_col: Column for triangle/business line ID.
            origin_year_col: Column for origin year.
            cum_columns: List of cumulative claim columns (development periods).
                         If None, auto-detected from columns excluding ID and origin.

        Returns:
            Self for chaining.
        """
        if cum_columns is None:
            exclude_cols = {triangle_id_col, origin_year_col}
            cum_columns = [
                col for col in triangle.columns if col not in exclude_cols
            ]
            try:
                cum_columns = sorted(cum_columns, key=lambda x: int(x))
            except ValueError:
                pass

        if len(cum_columns) < 2:
            msg = "Triangle must have at least 2 development periods"
            raise ModelError(msg)

        # Calculate age-to-age factors for each triangle
        all_factors: list[float] = []

        for tid, group in triangle.groupby(triangle_id_col):
            group = group.sort_values(origin_year_col)
            dev_cols = [c for c in cum_columns if c in group.columns]

            for i in range(len(dev_cols) - 1):
                current_col = dev_cols[i]
                next_col = dev_cols[i + 1]

                current = group[current_col].values
                next_vals = group[next_col].values

                # Create mask for rows where both values are valid and positive
                mask = (~pd.isna(current)) & (~pd.isna(next_vals)) & (current > 0) & (next_vals > 0)
                if mask.sum() == 0:
                    continue

                current_valid = current[mask]
                next_valid = next_vals[mask]

                current_sum = current_valid.sum()
                next_sum = next_valid.sum()

                if current_sum > 0:
                    factor = next_sum / current_sum
                    all_factors.append(factor)

        if not all_factors:
            msg = "No valid development factors could be calculated"
            raise ModelError(msg)

        # Store individual period factors (average across triangles)
        self.average_development_factor = float(np.mean(all_factors))

        # Calculate period-specific factors if enough data
        period_factors: dict[int, list[float]] = {}
        for tid, group in triangle.groupby(triangle_id_col):
            group = group.sort_values(origin_year_col)
            dev_cols = [c for c in cum_columns if c in group.columns]

            for i in range(len(dev_cols) - 1):
                current_col = dev_cols[i]
                next_col = dev_cols[i + 1]

                current = group[current_col].values
                next_vals = group[next_col].values

                mask = (~pd.isna(current)) & (~pd.isna(next_vals)) & (current > 0) & (next_vals > 0)
                if mask.sum() > 0:
                    current_valid = current[mask]
                    next_valid = next_vals[mask]
                    current_sum = current_valid.sum()
                    next_sum = next_valid.sum()
                    if current_sum > 0:
                        period = int(current_col) if current_col.isdigit() else i + 1
                        if period not in period_factors:
                            period_factors[period] = []
                        period_factors[period].append(next_sum / current_sum)

        for period, factors in period_factors.items():
            if factors:
                self.development_factors[period] = float(np.mean(factors))

        logger.info(
            "Chain Ladder fitted with %d development factors, avg=%.4f",
            len(self.development_factors),
            self.average_development_factor or 0,
        )

        return self

    def predict_ultimate(
        self,
        latest_cumulative: float,
        development_period: int,
        total_periods: int,
    ) -> float:
        """Predict ultimate claims from latest cumulative.

        Args:
            latest_cumulative: Latest known cumulative claims.
            development_period: Current development period (1-indexed).
            total_periods: Total expected development periods.

        Returns:
            Predicted ultimate claims.
        """
        remaining_periods = total_periods - development_period
        if remaining_periods <= 0:
            return latest_cumulative

        # Use period-specific factors if available, else average
        if self.average_development_factor is None:
            msg = "Model not fitted"
            raise ModelError(msg)

        factor = self.average_development_factor
        ultimate = latest_cumulative * (factor ** remaining_periods)

        return float(ultimate)

    def predict_ibnr(
        self,
        latest_cumulative: float,
        development_period: int,
        total_periods: int,
    ) -> float:
        """Predict IBNR (Incurred But Not Reported) reserves.

        Args:
            latest_cumulative: Latest known cumulative claims.
            development_period: Current development period.
            total_periods: Total expected development periods.

        Returns:
            Predicted IBNR reserve.
        """
        ultimate = self.predict_ultimate(latest_cumulative, development_period, total_periods)
        return float(ultimate - latest_cumulative)

    def predict_ibnr_by_triangle(
        self,
        triangle: pd.DataFrame,
        triangle_id_col: str = "triangle_id",
        origin_year_col: str = "origin_year",
        cum_columns: list[str] | None = None,
        total_periods: int | None = None,
    ) -> pd.DataFrame:
        """Calculate IBNR for all cells in a triangle.

        Args:
            triangle: Cumulative claim triangle.
            triangle_id_col: Triangle ID column.
            origin_year_col: Origin year column.
            cum_columns: Development period columns.
            total_periods: Total development periods (inferred if None).

        Returns:
            DataFrame with IBNR estimates per cell.
        """
        if cum_columns is None:
            exclude_cols = {triangle_id_col, origin_year_col}
            cum_columns = [
                col for col in triangle.columns if col not in exclude_cols
            ]
            try:
                cum_columns = sorted(cum_columns, key=lambda x: int(x))
            except ValueError:
                pass

        if total_periods is None:
            total_periods = len(cum_columns)

        results: list[dict[str, Any]] = []

        for _, row in triangle.iterrows():
            tid = row[triangle_id_col]
            origin = row[origin_year_col]

            for dev_idx, col in enumerate(cum_columns):
                if col not in row or pd.isna(row[col]):
                    continue

                latest = float(row[col])
                development_period = dev_idx + 1
                ibnr = self.predict_ibnr(latest, development_period, total_periods)
                ultimate = latest + ibnr

                results.append({
                    triangle_id_col: tid,
                    origin_year_col: origin,
                    "development_period": development_period,
                    "latest_cumulative": latest,
                    "predicted_ultimate": ultimate,
                    "predicted_ibnr": ibnr,
                })

        return pd.DataFrame(results)


class BornhuetterFergusonEstimator:
    """Bornhuetter-Ferguson IBNR reserve estimator.

    BF method combines:
    - Expected loss ratio * earned premium (for prior expectation)
    - Chain Ladder development pattern (for actual-to-expected ratio)

    IBNR = Expected_loss_ratio * Earned_premium * (1 - cumulative_development_factor)
    """

    def __init__(
        self,
        expected_loss_ratio: float = 0.70,
    ) -> None:
        self.expected_loss_ratio = expected_loss_ratio
        self.chain_ladder = ChainLadderEstimator()
        self.cumulative_factors: dict[int, float] = {}

    def fit(
        self,
        triangle: pd.DataFrame,
        earned_premium: pd.DataFrame | None = None,
        triangle_id_col: str = "triangle_id",
        origin_year_col: str = "origin_year",
        cum_columns: list[str] | None = None,
        premium_col: str = "premium",
        premium_id_col: str = "triangle_id",
    ) -> BornhuetterFergusonEstimator:
        """Fit the BF estimator.

        Args:
            triangle: Cumulative claim triangle.
            earned_premium: Optional earned premium by triangle/year.
            triangle_id_col: Triangle ID column.
            origin_year_col: Origin year column.
            cum_columns: Development period columns.
            premium_col: Premium amount column.
            premium_id_col: Premium triangle ID column.

        Returns:
            Self for chaining.
        """
        # Fit chain ladder for development patterns
        self.chain_ladder.fit(
            triangle,
            triangle_id_col=triangle_id_col,
            origin_year_col=origin_year_col,
            cum_columns=cum_columns,
        )

        # Calculate cumulative development factors
        if cum_columns is None:
            exclude_cols = {triangle_id_col, origin_year_col}
            cum_columns = [
                col for col in triangle.columns if col not in exclude_cols
            ]
            try:
                cum_columns = sorted(cum_columns, key=lambda x: int(x))
            except ValueError:
                pass

        self.cumulative_factors = {}
        for i, col in enumerate(cum_columns):
            period = i + 1
            # CDF to ultimate from period i+1: product of the remaining
            # development factors for periods > i (e.g. period 1 needs the
            # factors 1->2, 2->3, ... to reach ultimate). With this convention
            # CDF >= 1 and monotonically decreasing per period, so the BF IBNR
            # term (1 - 1/CDF) is non-negative and shrinks with maturity.
            cdf = 1.0
            for p in range(i + 1, len(cum_columns)):
                if p in self.chain_ladder.development_factors:
                    cdf *= self.chain_ladder.development_factors[p]
                else:
                    cdf *= self.chain_ladder.average_development_factor or 1.0
            self.cumulative_factors[period] = cdf

        # Calculate expected loss ratios from data if premium provided
        if earned_premium is not None and not earned_premium.empty:
            self._update_loss_ratio_from_data(triangle, earned_premium)

        logger.info(
            "Bornhuetter-Ferguson fitted with expected loss ratio=%.4f",
            self.expected_loss_ratio,
        )

        return self

    def _update_loss_ratio_from_data(
        self,
        triangle: pd.DataFrame,
        premium: pd.DataFrame,
    ) -> None:
        """Update expected loss ratio from observed data."""
        # This is a simplified version - in practice you'd use more sophisticated methods
        total_claims = triangle.select_dtypes(include=[np.number]).sum().sum()
        total_premium = premium.select_dtypes(include=[np.number]).sum().sum()

        if total_premium > 0:
            observed_lr = total_claims / total_premium
            # Blend observed with prior
            self.expected_loss_ratio = 0.5 * self.expected_loss_ratio + 0.5 * observed_lr

    def predict_ibnr(
        self,
        earned_premium: float,
        development_period: int,
        current_cumulative: float | None = None,
    ) -> float:
        """Predict IBNR using Bornhuetter-Ferguson method.

        Args:
            earned_premium: Earned premium for the period.
            development_period: Current development period.
            current_cumulative: Optional current cumulative claims.

        Returns:
            Predicted IBNR.
        """
        if development_period not in self.cumulative_factors:
            # Use last known CDF if period not available
            available_periods = sorted(self.cumulative_factors.keys())
            if available_periods:
                dev_period = available_periods[-1]
            else:
                dev_period = 1
        else:
            dev_period = development_period

        cdf = self.cumulative_factors.get(dev_period, 1.0)

        # Expected ultimate = ELR * Premium
        expected_ultimate = self.expected_loss_ratio * earned_premium

        # The cumulative factor may be supplied in either convention:
        # - > 1: a true development factor to ultimate, so the reported
        #   portion is expected_ultimate / cdf and IBNR = EU * (1 - 1/cdf).
        # - <= 1: a proportion already developed, so IBNR = EU * (1 - cdf).
        # Both conventions yield a non-negative IBNR for valid inputs.
        if cdf > 1.0:
            ibnr = expected_ultimate * (1.0 - 1.0 / cdf)
        else:
            ibnr = expected_ultimate * (1.0 - cdf)

        return float(ibnr)

    def predict_ibnr_by_triangle(
        self,
        triangle: pd.DataFrame,
        premium: pd.DataFrame,
        triangle_id_col: str = "triangle_id",
        origin_year_col: str = "origin_year",
        cum_columns: list[str] | None = None,
        premium_col: str = "premium",
        premium_year_col: str = "origin_year",
        premium_id_col: str = "triangle_id",
    ) -> pd.DataFrame:
        """Calculate BF IBNR for all cells.

        Args:
            triangle: Cumulative claim triangle.
            premium: Earned premium DataFrame.
            triangle_id_col: Triangle ID column.
            origin_year_col: Origin year column.
            cum_columns: Development period columns.
            premium_col: Premium column name.
            premium_year_col: Year column in premium DataFrame.
            premium_id_col: Triangle ID column in premium DataFrame.

        Returns:
            DataFrame with BF IBNR estimates.
        """
        if cum_columns is None:
            exclude_cols = {triangle_id_col, origin_year_col}
            cum_columns = [
                col for col in triangle.columns if col not in exclude_cols
            ]
            try:
                cum_columns = sorted(cum_columns, key=lambda x: int(x))
            except ValueError:
                pass

        # Merge premium data
        premium_lookup = premium.set_index([premium_id_col, premium_year_col])[premium_col].to_dict()

        results: list[dict[str, Any]] = []

        for _, row in triangle.iterrows():
            tid = row[triangle_id_col]
            origin = row[origin_year_col]

            # Get premium for this triangle/year
            premium_key = (tid, origin)
            earned_premium = premium_lookup.get(premium_key, 0.0)

            for dev_idx, col in enumerate(cum_columns):
                if col not in row or pd.isna(row[col]):
                    continue

                development_period = dev_idx + 1
                ibnr = self.predict_ibnr(earned_premium, development_period)

                results.append({
                    triangle_id_col: tid,
                    origin_year_col: origin,
                    "development_period": development_period,
                    "earned_premium": earned_premium,
                    "expected_loss_ratio": self.expected_loss_ratio,
                    "bf_ibnr": ibnr,
                    "chain_ladder_ibnr": self.chain_ladder.predict_ibnr(
                        float(row[col]) if pd.notna(row[col]) else 0,
                        development_period,
                        len(cum_columns),
                    ),
                })

        return pd.DataFrame(results)


class LossRatioCalculator:
    """Calculate loss ratios and related metrics."""

    @staticmethod
    def calculate_loss_ratio(
        claims: float,
        premium: float,
        *,
        include_ibnr: bool = True,
        ibnr: float | None = None,
    ) -> float:
        """Calculate loss ratio.

        Args:
            claims: Paid or incurred claims.
            premium: Earned premium.
            include_ibnr: Whether to include IBNR in claims.
            ibnr: IBNR reserve (if include_ibnr is True).

        Returns:
            Loss ratio (0-1 or higher for adverse experience).
        """
        if premium <= 0:
            return float("nan")

        total_claims = claims
        if include_ibnr and ibnr is not None:
            total_claims += ibnr

        return float(total_claims / premium)

    @staticmethod
    def calculate_combined_ratio(
        loss_ratio: float,
        expense_ratio: float,
    ) -> float:
        """Calculate combined ratio (loss + expense).

        Args:
            loss_ratio: Loss ratio.
            expense_ratio: Expense ratio.

        Returns:
            Combined ratio.
        """
        return float(loss_ratio + expense_ratio)

    @staticmethod
    def calculate_premium_rate(
        expected_loss: float,
        expected_loss_ratio: float,
        *,
        load_factor: float = 1.10,
    ) -> float:
        """Calculate premium rate.

        Args:
            expected_loss: Expected loss amount.
            expected_loss_ratio: Target loss ratio.
            load_factor: Loading factor for expenses/profit.

        Returns:
            Required premium.
        """
        if expected_loss_ratio <= 0:
            return float("nan")

        base_premium = expected_loss / expected_loss_ratio
        return float(base_premium * load_factor)


class ActuarialDesk:
    """High-level actuarial desk combining multiple estimation methods.

    Provides unified interface for:
    - IBNR estimation (Chain Ladder, Bornhuetter-Ferguson, or blended)
    - Loss ratio analysis
    - Premium rate calculation
    """

    def __init__(
        self,
        method: str = "blended",
        expected_loss_ratio: float = 0.70,
        blend_weight_cl: float = 0.5,
    ) -> None:
        self.method = method
        self.expected_loss_ratio = expected_loss_ratio
        self.blend_weight_cl = blend_weight_cl  # Weight for Chain Ladder in blend

        self.chain_ladder = ChainLadderEstimator()
        self.bornhuetter_ferguson: BornhuetterFergusonEstimator | None = None
        self._is_fitted = False

    def fit(
        self,
        triangle: pd.DataFrame,
        premium: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> ActuarialDesk:
        """Fit all estimation methods.

        Args:
            triangle: Claim triangle.
            premium: Optional earned premium data.
            **kwargs: Additional arguments for fit methods.

        Returns:
            Self for chaining.
        """
        self.chain_ladder.fit(triangle, **kwargs)

        self.bornhuetter_ferguson = BornhuetterFergusonEstimator(
            expected_loss_ratio=self.expected_loss_ratio,
        )
        self.bornhuetter_ferguson.fit(triangle, premium, **kwargs)

        self._is_fitted = True
        logger.info("Actuarial desk fitted with method=%s", self.method)
        return self

    def estimate_ibnr(
        self,
        latest_cumulative: float,
        earned_premium: float,
        development_period: int,
        total_periods: int,
    ) -> dict[str, float]:
        """Estimate IBNR using configured method.

        Args:
            latest_cumulative: Latest cumulative claims.
            earned_premium: Earned premium.
            development_period: Current development period.
            total_periods: Total development periods.

        Returns:
            Dict with 'chain_ladder', 'bornhuetter_ferguson', and 'blended' IBNR estimates.
        """
        if not self._is_fitted:
            msg = "Actuarial desk must be fitted before estimation"
            raise ModelError(msg)

        cl_ibnr = self.chain_ladder.predict_ibnr(
            latest_cumulative, development_period, total_periods
        )

        bf_ibnr = self.bornhuetter_ferguson.predict_ibnr(
            earned_premium, development_period, latest_cumulative
        ) if self.bornhuetter_ferguson else 0.0

        # Blended estimate
        blended_ibnr = (
            self.blend_weight_cl * cl_ibnr
            + (1 - self.blend_weight_cl) * bf_ibnr
        )

        return {
            "chain_ladder": cl_ibnr,
            "bornhuetter_ferguson": bf_ibnr,
            "blended": blended_ibnr,
        }

    def calculate_loss_ratios(
        self,
        paid_claims: pd.DataFrame,
        premium: pd.DataFrame,
        ibnr_estimates: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Calculate loss ratios by period or segment.

        Args:
            paid_claims: DataFrame with claims data.
            premium: DataFrame with premium data.
            ibnr_estimates: Optional IBNR by segment.

        Returns:
            DataFrame with loss ratios.
        """
        # Simplified implementation - merge and calculate
        results: list[dict[str, Any]] = []

        for _, claim_row in paid_claims.iterrows():
            premium_row = premium[
                (premium.get("triangle_id") == claim_row.get("triangle_id"))
                & (premium.get("period") == claim_row.get("period"))
            ]
            premium_amount = premium_row["premium"].values[0] if len(premium_row) > 0 else 0

            ibnr = 0.0
            if ibnr_estimates:
                ibnr = ibnr_estimates.get("blended", 0.0)

            lr = LossRatioCalculator.calculate_loss_ratio(
                float(claim_row.get("claims", 0)),
                premium_amount,
                include_ibnr=True,
                ibnr=ibnr,
            )

            results.append({
                "period": claim_row.get("period"),
                "claims": claim_row.get("claims"),
                "premium": premium_amount,
                "ibnr": ibnr,
                "loss_ratio": lr,
            })

        return pd.DataFrame(results)

    def calculate_premium_rate(
        self,
        expected_loss: float,
        target_loss_ratio: float | None = None,
        load_factor: float = 1.10,
    ) -> float:
        """Calculate required premium rate.

        Args:
            expected_loss: Expected loss amount.
            target_loss_ratio: Target loss ratio (uses config if None).
            load_factor: Loading factor.

        Returns:
            Required premium.
        """
        if target_loss_ratio is None:
            target_loss_ratio = self.expected_loss_ratio

        return LossRatioCalculator.calculate_premium_rate(
            expected_loss,
            target_loss_ratio,
            load_factor=load_factor,
        )

    def save(self, path: str) -> None:
        """Save actuarial desk state."""
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "actuarial_desk.pkl"), "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "expected_loss_ratio": self.expected_loss_ratio,
                    "blend_weight_cl": self.blend_weight_cl,
                    "is_fitted": self._is_fitted,
                    "chain_ladder_factors": self.chain_ladder.development_factors,
                    "chain_ladder_avg_factor": self.chain_ladder.average_development_factor,
                    "bf_cumulative_factors": self.bornhuetter_ferguson.cumulative_factors
                    if self.bornhuetter_ferguson
                    else {},
                },
                f,
            )

        logger.info("Saved actuarial desk to %s", path)

    @classmethod
    def load(cls, path: str) -> ActuarialDesk:
        """Load a saved actuarial desk."""
        import os
        import pickle

        with open(os.path.join(path, "actuarial_desk.pkl"), "rb") as f:
            data = pickle.load(f)

        desk = cls(
            method=data["method"],
            expected_loss_ratio=data["expected_loss_ratio"],
            blend_weight_cl=data["blend_weight_cl"],
        )
        desk._is_fitted = data["is_fitted"]
        desk.chain_ladder.development_factors = data["chain_ladder_factors"]
        desk.chain_ladder.average_development_factor = data["chain_ladder_avg_factor"]

        if data.get("bf_cumulative_factors"):
            desk.bornhuetter_ferguson = BornhuetterFergusonEstimator(
                expected_loss_ratio=data["expected_loss_ratio"],
            )
            desk.bornhuetter_ferguson.cumulative_factors = data["bf_cumulative_factors"]

        logger.info("Loaded actuarial desk from %s", path)
        return desk

    @property
    def is_fitted(self) -> bool:
        """Whether the desk has been fitted."""
        return self._is_fitted
