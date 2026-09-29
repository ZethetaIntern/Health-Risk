"""Pharma Analytics: rNPV Monte Carlo and clinical trial velocity tracking.

This module implements:
- Risk-adjusted Net Present Value (rNPV) calculation with Monte Carlo simulation
- Clinical trial enrollment velocity tracking
- Pipeline valuation
- Portfolio optimization across drug candidates
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import lognorm, beta

from healthrisk_ai.exceptions import ModelError

logger = logging.getLogger("healthrisk_ai.pharma")


class TrialVelocityTracker:
    """Track and project clinical trial enrollment velocity.

    Uses actual enrollment data to project completion timelines and
    identify trials at risk of delayed enrollment.
    """

    def __init__(
        self,
        velocity_window_days: int = 365,
        min_enrollment_rate: float = 0.1,
    ) -> None:
        self.velocity_window_days = velocity_window_days
        self.min_enrollment_rate = min_enrollment_rate

    def compute_velocity(
        self,
        trials: pd.DataFrame,
        *,
        nct_id_col: str = "nct_id",
        start_date_col: str = "study_start_date",
        completion_date_col: str = "primary_completion_date",
        intended_col: str = "enrollment_intended",
        actual_col: str = "enrollment_actual",
        status_col: str = "overall_status",
    ) -> pd.DataFrame:
        """Compute enrollment velocity metrics for trials.

        Args:
            trials: Clinical trials DataFrame.
            nct_id_col: NCT ID column.
            start_date_col: Study start date column.
            completion_date_col: Primary completion date column.
            intended_col: Intended enrollment column.
            actual_col: Actual enrollment column.
            status_col: Overall status column.

        Returns:
            DataFrame with velocity metrics.
        """
        results: list[dict[str, Any]] = []

        for _, row in trials.iterrows():
            start = pd.Timestamp(row[start_date_col])
            completion = pd.Timestamp(row[completion_date_col])
            intended = float(row[intended_col]) if pd.notna(row[intended_col]) else 0.0
            actual = float(row[actual_col]) if pd.notna(row[actual_col]) else 0.0
            status = str(row[status_col]) if pd.notna(row[status_col]) else "Unknown"

            # Calculate duration
            duration_days = (completion - start).days
            duration_years = duration_days / 365.0 if duration_days > 0 else float("nan")

            # Enrollment rate (patients per year)
            enrollment_rate = actual / duration_years if duration_years and duration_years > 0 else 0.0

            # Attainment (actual / intended)
            attainment = actual / intended if intended > 0 else 0.0

            # Velocity status
            if status in {"Completed"}:
                velocity_status = "completed"
            elif status in {"Recruiting", "Active, not recruiting"}:
                if enrollment_rate >= self.min_enrollment_rate:
                    velocity_status = "on_track"
                else:
                    velocity_status = "at_risk"
            elif status == "Terminated":
                velocity_status = "terminated"
            else:
                velocity_status = "unknown"

            # Projected completion (if still recruiting)
            projected_completion = None
            if status in {"Recruiting", "Active, not recruiting"} and enrollment_rate > 0:
                remaining = intended - actual
                if remaining > 0:
                    days_remaining = remaining / enrollment_rate * 365
                    projected_completion = start + pd.Timedelta(days=int(duration_days + days_remaining))

            results.append({
                nct_id_col: row[nct_id_col],
                "start_date": start,
                "completion_date": completion,
                "intended_enrollment": intended,
                "actual_enrollment": actual,
                "duration_years": duration_years,
                "enrollment_rate_per_year": enrollment_rate,
                "attainment_pct": attainment * 100,
                "velocity_status": velocity_status,
                "projected_completion": projected_completion,
                "is_active": status in {"Recruiting", "Active, not recruiting"},
                "is_completed": status == "Completed",
            })

        return pd.DataFrame(results)

    def identify_at_risk_trials(
        self,
        trials: pd.DataFrame,
        *,
        nct_id_col: str = "nct_id",
        status_col: str = "overall_status",
        intended_col: str = "enrollment_intended",
        actual_col: str = "enrollment_actual",
        start_date_col: str = "study_start_date",
    ) -> pd.DataFrame:
        """Identify trials at risk of enrollment delays.

        Args:
            trials: Clinical trials DataFrame.
            nct_id_col: NCT ID column.
            status_col: Status column.
            intended_col: Intended enrollment column.
            actual_col: Actual enrollment column.
            start_date_col: Start date column.

        Returns:
            DataFrame of at-risk trials with risk scores.
        """
        velocities = self.compute_velocity(
            trials,
            nct_id_col=nct_id_col,
            start_date_col=start_date_col,
            completion_date_col="primary_completion_date",
            intended_col=intended_col,
            actual_col=actual_col,
            status_col=status_col,
        )

        at_risk = velocities[velocities["velocity_status"] == "at_risk"].copy()

        if at_risk.empty:
            return at_risk

        # Calculate risk score (0-100, higher = more at risk)
        at_risk["risk_score"] = at_risk.apply(
            lambda row: self._calculate_risk_score(row),
            axis=1,
        )

        return at_risk.sort_values("risk_score", ascending=False)

    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate enrollment risk score for a trial.

        Args:
            row: Velocity metrics row.

        Returns:
            Risk score 0-100.
        """
        score = 0.0

        # Low attainment increases risk
        attainment = row.get("attainment_pct", 100)
        if attainment < 25:
            score += 40
        elif attainment < 50:
            score += 30
        elif attainment < 75:
            score += 20
        elif attainment < 90:
            score += 10

        # Slow enrollment rate increases risk
        rate = row.get("enrollment_rate_per_year", 0)
        if rate < 10:
            score += 30
        elif rate < 25:
            score += 20
        elif rate < 50:
            score += 10

        # Longer duration increases risk
        duration = row.get("duration_years", 0)
        if duration > 5:
            score += 20
        elif duration > 3:
            score += 10
        elif duration > 2:
            score += 5

        return min(100, score)


class RNPVModel:
    """Risk-adjusted Net Present Value calculator with Monte Carlo simulation.

    rNPV = sum over time periods of:
        (Probability of success * Expected revenue * Discount factor) - Costs

    Monte Carlo simulation accounts for uncertainty in:
    - Clinical trial success probabilities
    - Market size and penetration
    - Pricing and revenue
    - Development costs and timeline
    """

    def __init__(
        self,
        discount_rate: float = 0.10,
        simulation_runs: int = 10000,
        random_seed: int = 20260914,
    ) -> None:
        """Initialize rNPV model.

        Args:
            discount_rate: Annual discount rate (e.g. 0.10 = 10%).
            simulation_runs: Number of Monte Carlo iterations.
            random_seed: Random seed for reproducibility.
        """
        self.discount_rate = discount_rate
        self.simulation_runs = simulation_runs
        self.random_seed = random_seed

        self.rng = np.random.default_rng(random_seed)

    def calculate_deterministic_rnpv(
        self,
        pipeline: pd.DataFrame,
        *,
        start_year_col: str = "start_year",
        duration_col: str = "duration_years",
        prob_success_col: str = "prob_success",
        peak_revenue_col: str = "peak_revenue",
        cost_col: str = "development_cost",
        time_to_market_col: str = "time_to_market",
    ) -> pd.DataFrame:
        """Calculate deterministic rNPV for each drug in pipeline.

        Args:
            pipeline: DataFrame with drug candidate data.
            start_year_col: Year development starts.
            duration_col: Development duration in years.
            prob_success_col: Probability of technical success (0-1).
            peak_revenue_col: Expected peak annual revenue.
            cost_col: Total development cost.
            time_to_market_col: Years until market launch.

        Returns:
            DataFrame with rNPV and supporting metrics.
        """
        results: list[dict[str, Any]] = []

        for _, drug in pipeline.iterrows():
            start_year = int(drug[start_year_col]) if pd.notna(drug[start_year_col]) else 0
            duration = float(drug[duration_col]) if pd.notna(drug[duration_col]) else 10.0
            prob_success = float(drug[prob_success_col]) if pd.notna(drug[prob_success_col]) else 0.5
            peak_revenue = float(drug[peak_revenue_col]) if pd.notna(drug[peak_revenue_col]) else 100e6
            cost = float(drug[cost_col]) if pd.notna(drug[cost_col]) else 50e6
            time_to_market = float(drug[time_to_market_col]) if pd.notna(drug[time_to_market_col]) else duration

            # Revenue ramp-up (simplified: linear to peak over 5 years)
            ramp_years = 5
            revenue_stream = self._project_revenue_stream(
                peak_revenue, time_to_market, duration - time_to_market, ramp_years
            )

            # Calculate rNPV
            rnpv = self._calculate_rnpv(
                revenue_stream,
                cost,
                prob_success,
                start_year,
                discount_rate=self.discount_rate,
            )

            # Success-adjusted NPV (without probability)
            base_npv = self._calculate_rnpv(
                revenue_stream,
                cost,
                1.0,  # No probability adjustment
                start_year,
                discount_rate=self.discount_rate,
            )

            results.append({
                "drug_id": drug.get("drug_id", "Unknown"),
                "peak_revenue": peak_revenue,
                "prob_success": prob_success,
                "success_adjusted_revenue": peak_revenue * prob_success,
                "development_cost": cost,
                "deterministic_rnpv": rnpv,
                "base_npv": base_npv,
                "net_value": rnpv,
                "roi": (rnpv - cost) / cost if cost > 0 else float("nan"),
            })

        return pd.DataFrame(results)

    def _project_revenue_stream(
        self,
        peak_revenue: float,
        launch_year: float,
        patent_life: float,
        ramp_years: float = 5,
    ) -> list[tuple[int, float]]:
        """Project annual revenue stream.

        Args:
            peak_revenue: Peak annual revenue.
            launch_year: Year of market launch.
            patent_life: Years of revenue after launch.
            ramp_years: Years to reach peak revenue.

        Returns:
            List of (year, revenue) tuples.
        """
        stream: list[tuple[int, float]] = []

        for year in range(int(launch_year), int(launch_year + patent_life) + 1):
            years_since_launch = year - launch_year

            if years_since_launch <= 0:
                revenue = 0.0
            elif years_since_launch <= ramp_years:
                # Linear ramp-up
                revenue = peak_revenue * (years_since_launch / ramp_years)
            else:
                # Patent cliff decay (simplified: 20% annual decline after peak)
                years_post_peak = years_since_launch - ramp_years
                revenue = peak_revenue * (0.8 ** years_post_peak)

            stream.append((year, max(0, revenue)))

        return stream

    def _calculate_rnpv(
        self,
        revenue_stream: list[tuple[int, float]],
        development_cost: float,
        prob_success: float,
        start_year: int,
        discount_rate: float,
    ) -> float:
        """Calculate risk-adjusted NPV.

        Args:
            revenue_stream: List of (year, revenue) tuples.
            development_cost: Total development cost.
            prob_success: Probability of technical success.
            start_year: Development start year.
            discount_rate: Annual discount rate.

        Returns:
            rNPV value.
        """
        # PV of costs (assume costs spread over development period)
        cost_pv = development_cost / ((1 + discount_rate) ** (start_year + 0.5))

        # PV of revenues (success-adjusted)
        revenue_pv = 0.0
        for year, revenue in revenue_stream:
            pv_factor = 1.0 / ((1 + discount_rate) ** year)
            revenue_pv += revenue * pv_factor

        # Apply probability of success
        rnpv = prob_success * revenue_pv - cost_pv

        return float(rnpv)

    def monte_carlo_rnpv(
        self,
        drug: dict[str, Any],
        *,
        n_trials: int | None = None,
        success_prob_params: dict[str, float] | None = None,
        revenue_params: dict[str, float] | None = None,
        cost_params: dict[str, float] | None = None,
        duration_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation for rNPV distribution.

        Args:
            drug: Drug parameters (base case values).
            n_trials: Number of simulations (uses config if None).
            success_prob_params: Beta distribution params for success prob (alpha, beta).
            revenue_params: Lognormal distribution params for revenue (mu, sigma).
            cost_params: Lognormal distribution params for cost (mu, sigma).
            duration_params: Normal distribution params for duration (mean, std).

        Returns:
            Dict with rNPV distribution statistics.
        """
        n_trials = n_trials or self.simulation_runs

        # Distribution parameters
        success_alpha = success_prob_params.get("alpha", 2) if success_prob_params else 2
        success_beta = success_prob_params.get("beta", 2) if success_prob_params else 2

        revenue_mu = revenue_params.get("mu", np.log(drug.get("peak_revenue", 100e6))) if revenue_params else np.log(drug.get("peak_revenue", 100e6))
        revenue_sigma = revenue_params.get("sigma", 0.3) if revenue_params else 0.3

        cost_mu = cost_params.get("mu", np.log(drug.get("development_cost", 50e6))) if cost_params else np.log(drug.get("development_cost", 50e6))
        cost_sigma = cost_params.get("sigma", 0.2) if cost_params else 0.2

        duration_mean = duration_params.get("mean", drug.get("duration_years", 10)) if duration_params else drug.get("duration_years", 10)
        duration_std = duration_params.get("std", 2.0) if duration_params else 2.0

        rnpv_samples = np.zeros(n_trials)

        for i in range(n_trials):
            # Sample from distributions
            prob_success = self.rng.beta(success_alpha, success_beta)
            peak_revenue = self.rng.lognormal(revenue_mu, revenue_sigma)
            development_cost = self.rng.lognormal(cost_mu, cost_sigma)
            duration = max(1, self.rng.normal(duration_mean, duration_std))

            # Calculate rNPV for this sample
            revenue_stream = self._project_revenue_stream(
                peak_revenue,
                drug.get("time_to_market", duration * 0.7),
                duration,
            )

            rnpv = self._calculate_rnpv(
                revenue_stream,
                development_cost,
                prob_success,
                drug.get("start_year", 0),
                discount_rate=self.discount_rate,
            )

            rnpv_samples[i] = rnpv

        # Calculate statistics
        return {
            "mean_rnpv": float(np.mean(rnpv_samples)),
            "median_rnpv": float(np.median(rnpv_samples)),
            "std_rnpv": float(np.std(rnpv_samples)),
            "p10_rnpv": float(np.percentile(rnpv_samples, 10)),
            "p50_rnpv": float(np.percentile(rnpv_samples, 50)),
            "p90_rnpv": float(np.percentile(rnpv_samples, 90)),
            "probability_positive": float((rnpv_samples > 0).mean()),
            "skewness": float(
                np.mean((rnpv_samples - np.mean(rnpv_samples)) ** 3)
                / (np.std(rnpv_samples) ** 3 + 1e-10)
            ),
            "samples": rnpv_samples,
        }

    def portfolio_optimization(
        self,
        pipeline: pd.DataFrame,
        *,
        budget: float = float("inf"),
        max_candidates: int | None = None,
        risk_aversion: float = 0.5,
    ) -> pd.DataFrame:
        """Optimize portfolio selection under budget constraint.

        Uses a simple mean-variance framework to select drug candidates
        that maximize risk-adjusted return.

        Args:
            pipeline: DataFrame with drug candidates and their deterministic_rnpv.
            budget: Total budget constraint.
            max_candidates: Maximum number of candidates to select.
            risk_aversion: Risk aversion parameter (0 = risk-neutral, 1 = very risk-averse).

        Returns:
            DataFrame with selected candidates and allocation.
        """
        if pipeline.empty:
            return pipeline

        # Calculate risk metric (use coefficient of variation as proxy)
        pipeline = pipeline.copy()
        pipeline["risk"] = pipeline["peak_revenue"] / (pipeline.get("peak_revenue", 1) + 1)
        pipeline[" Sharpe-like"] = pipeline["deterministic_rnpv"] / (pipeline["risk"] + 1)

        # Score candidates
        pipeline["score"] = (
            (1 - risk_aversion) * pipeline["deterministic_rnpv"].rank(pct=True)
            - risk_aversion * pipeline["risk"].rank(pct=True)
        )

        # Sort by score
        pipeline = pipeline.sort_values("score", ascending=False)

        # Greedy selection under budget
        selected: list[dict[str, Any]] = []
        total_cost = 0.0

        for _, row in pipeline.iterrows():
            cost = row.get("development_cost", 0)

            if total_cost + cost <= budget:
                if max_candidates is None or len(selected) < max_candidates:
                    selected.append({
                        **row.to_dict(),
                        "selected": True,
                        "allocation_order": len(selected) + 1,
                    })
                    total_cost += cost

        # Mark non-selected
        selected_ids = {s["drug_id"] for s in selected}
        for _, row in pipeline.iterrows():
            if row.get("drug_id") not in selected_ids:
                selected.append({
                    **row.to_dict(),
                    "selected": False,
                    "allocation_order": None,
                })

        return pd.DataFrame(selected)


class PharmaPipelineAnalyzer:
    """High-level analyzer for pharmaceutical pipeline valuation.

    Combines trial velocity tracking and rNPV analysis for
    comprehensive pipeline assessment.
    """

    def __init__(
        self,
        discount_rate: float = 0.10,
        simulation_runs: int = 10000,
    ) -> None:
        self.velocity_tracker = TrialVelocityTracker()
        self.rnpv_model = RNPVModel(
            discount_rate=discount_rate,
            simulation_runs=simulation_runs,
        )

    def analyze_pipeline(
        self,
        pipeline: pd.DataFrame,
        trials: pd.DataFrame | None = None,
        *,
        drug_id_col: str = "drug_id",
        trial_id_col: str = "nct_id",
        drug_trial_mapping: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Comprehensive pipeline analysis.

        Args:
            pipeline: Drug pipeline DataFrame.
            trials: Optional clinical trials DataFrame.
            drug_id_col: Drug ID column.
            trial_id_col: Trial NCT ID column.
            drug_trial_mapping: Optional mapping of drug_id -> list of trial NCT IDs.

        Returns:
            Dict with pipeline summary metrics.
        """
        # Calculate deterministic rNPV for each drug
        rnpv_results = self.rnpv_model.calculate_deterministic_rnpv(pipeline)

        # Analyze trial velocity if trials provided
        trial_analysis = None
        at_risk_count = 0

        if trials is not None:
            velocities = self.velocity_tracker.compute_velocity(trials)
            at_risk = self.velocity_tracker.identify_at_risk_trials(trials)
            at_risk_count = len(at_risk)
            trial_analysis = {
                "total_trials": len(trials),
                "active_trials": int(velocities["is_active"].sum()),
                "completed_trials": int(velocities["is_completed"].sum()),
                "at_risk_trials": at_risk_count,
                "average_attainment": float(velocities["attainment_pct"].mean()),
                "average_enrollment_rate": float(velocities["enrollment_rate_per_year"].mean()),
            }

        # Portfolio-level metrics
        total_rnpv = rnpv_results["deterministic_rnpv"].sum()
        total_cost = rnpv_results["development_cost"].sum()
        portfolio_roi = (total_rnpv - total_cost) / total_cost if total_cost > 0 else float("nan")

        # Risk-adjusted analysis
        avg_prob_success = rnpv_results["prob_success"].mean()
        weighted_rnpv = (rnpv_results["deterministic_rnpv"] * rnpv_results["prob_success"]).sum()

        return {
            "pipeline_summary": {
                "num_candidates": len(pipeline),
                "total_deterministic_rnpv": float(total_rnpv),
                "total_development_cost": float(total_cost),
                "portfolio_roi": float(portfolio_roi),
                "average_prob_success": float(avg_prob_success),
                "weighted_rnpv": float(weighted_rnpv),
                "expected_value_at_risk": float(total_rnpv * (1 - avg_prob_success)),
            },
            "trial_analysis": trial_analysis,
            "rnpv_detail": rnpv_results.to_dict(orient="records"),
        }

    def run_drug_monte_carlo(
        self,
        drug: dict[str, Any],
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Run Monte Carlo analysis for a single drug.

        Args:
            drug: Drug parameters.
            n_trials: Number of simulations.

        Returns:
            Monte Carlo results.
        """
        return self.rnpv_model.monte_carlo_rnpv(drug, n_trials=n_trials)

    def optimize_portfolio(
        self,
        pipeline: pd.DataFrame,
        budget: float,
        *,
        max_candidates: int | None = None,
        risk_aversion: float = 0.5,
    ) -> pd.DataFrame:
        """Optimize portfolio selection.

        Args:
            pipeline: Drug pipeline.
            budget: Budget constraint.
            max_candidates: Maximum candidates.
            risk_aversion: Risk aversion.

        Returns:
            Optimized portfolio.
        """
        return self.rnpv_model.portfolio_optimization(
            pipeline,
            budget=budget,
            max_candidates=max_candidates,
            risk_aversion=risk_aversion,
        )
