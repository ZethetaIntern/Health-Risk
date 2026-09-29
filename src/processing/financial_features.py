"""Financial feature engineering for the HealthRisk AI downstream calculators.

This module converts clinical and ancillary operational data into the financial
feature set consumed by the actuarial desk, hospital credit scorecard, pharma
rNPV model, and the HealthRisk Lab simulation.

It implements:
- Hospital financial ratio calculators: DSCR, Days Cash on Hand, current ratio,
  quick ratio, operating margin, and a CMS-HCC-adjusted cost-per-case proxy
- CMI (Case Mix Index) signal extractors from DRG / admission volumes
- Claim development triangles (incremental and cumulative) plus standard
  development factor estimation for IBNR inputs
- Clinical trial enrollment velocity signals from ClinicalTrials.gov records
- Temporal feature windows to prevent leakage between training and scoring

All functions are pure where possible, annotated, and designed to accept both
real and synthetic data so the downstream modules and tests remain runnable
without live hospital ERP access.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable

import numpy as np
import pandas as pd

from healthrisk_ai.config import cfg
from healthrisk_ai.exceptions import FeatureEngineeringError, ValidationError

logger = logging.getLogger("healthrisk_ai.processing.financial_features")

__all__ = [
    "HospitalFinancialFeatureInputs",
    "compute_hospital_financial_ratios",
    "compute_claim_development_triangle",
    "triangle_development_factor",
    "compute_cmi_signals",
    "compute_trial_enrollment_velocity",
    "build_financial_feature_frame",
    "DscrInputs",
    "DaysCashOnHandInputs",
]


class HospitalFinancialFeatureInputs:
    """Runtime contract for the columns required by hospital financial feature functions.

    Kept as a plain class to avoid Pydantic overhead in the hot path.
    """

    required_ratio_columns: list[str] = [
        "hospital_id",
        "period_start",
        "cash_and_equivalents",
        "marketable_securities",
        "accounts_receivable",
        "total_revenue",
        "operating_expenses",
        "interest_expense",
        "depreciation_and_amortization",
        "total_debt_service",
        "current_assets",
        "current_liabilities",
    ]
    optional_ratio_columns: list[str] = [
        "drg_volume",
        "cmi_index",
        "readmissions",
        "total_discharges",
        "medicare_days",
        "medicaid_days",
        "self_pay_days",
        "case_mix_index",
        "patient_days",
        "occupancy_rate",
    ]
    required_triangle_columns: list[str] = [
        "triangle_id",
        "origin_year",
        "development_period",
        "cumulative_claim",
    ]
    required_trial_columns: list[str] = [
        "nct_id",
        "study_start_date",
        "primary_completion_date",
        "overall_status",
        "enrollment_intended",
        "enrollment_actual",
        "search_term_match",
    ]

    @classmethod
    def validate_ratio_inputs(cls, df: pd.DataFrame) -> None:
        missing = set(cls.required_ratio_columns) - set(df.columns)
        if missing:
            msg = f"Hospital financial inputs missing columns: {sorted(missing)}"
            raise ValidationError(msg, context={"missing_columns": sorted(missing)})
        if df.empty:
            msg = "Hospital financial inputs received an empty frame"
            raise ValidationError(msg)

    @classmethod
    def validate_triangle_inputs(cls, df: pd.DataFrame) -> None:
        missing = set(cls.required_triangle_columns) - set(df.columns)
        if missing:
            msg = f"Claim triangle inputs missing columns: {sorted(missing)}"
            raise ValidationError(msg, context={"missing_columns": sorted(missing)})

    @classmethod
    def validate_trial_inputs(cls, df: pd.DataFrame) -> None:
        missing = set(cls.required_trial_columns) - set(df.columns)
        if missing:
            msg = f"Trial enrollment inputs missing columns: {sorted(missing)}"
            raise ValidationError(msg, context={"missing_columns": sorted(missing)})


class DscrInputs:
    """Small typed inputs bag for DSCR computation."""

    def __init__(self, *, operating_income: float, total_debt_service: float) -> None:
        self.operating_income = operating_income
        self.total_debt_service = total_debt_service


class DaysCashOnHandInputs:
    """Small typed inputs bag for Days Cash on Hand computation."""

    def __init__(self, *, cash: float, marketable: float, operating_expenses: float) -> None:
        self.cash = cash
        self.marketable = marketable
        self.operating_expenses = operating_expenses


def _safe_divide(
    numerator: float, denominator: float, *, default: float = float("nan"), zero_ok: bool = False
) -> float:
    """Return numerator/denominator with safe NaN handling."""
    if denominator == 0:
        if zero_ok:
            return 0.0
        return default
    return float(numerator) / float(denominator)


def compute_hospital_financial_ratios(
    financials: pd.DataFrame,
    *,
    hospital_col: Hashable = "hospital_id",
    period_start_col: Hashable = "period_start",
    return_detailed: bool = True,
) -> pd.DataFrame:
    """Compute a suite of hospital financial ratios per hospital per period.

    Ratios computed:
    - DSCR (Debt Service Coverage Ratio) = (EBIT + D&A) / total debt service
    - Days Cash on Hand = (cash + marketable securities) / (operating expenses / 365)
    - Current Ratio = current assets / current liabilities
    - Quick Ratio = (cash + marketable securities + receivables) / current liabilities
    - Operating Margin = (total revenue - operating expenses) / total revenue
    - Debt-to-Revenue = total debt service / total revenue
    - CMI-adjusted cost proxy: operating expenses / (CMI * discharges) when available

    When denominator values are zero, the ratio defaults to NaN and the detailed
    breakdown includes a `ratio_valid` boolean flag per ratio.

    Args:
        financials: Hospital financials frame per period.
        hospital_col: Hospital identifier column.
        period_start_col: Period start date column.
        return_detailed: If True, include _raw numerator/denominator and validity flags.

    Returns:
        Per-hospital per-period ratio frame.
    """
    HospitalFinancialFeatureInputs.validate_ratio_inputs(financials)

    rows: list[dict[str, Any]] = []
    for _, row in financials.iterrows():
        hospital = row[hospital_col]
        period = row[period_start_col]

        ebit = float(row["total_revenue"]) - float(row["operating_expenses"])
        da = float(row["depreciation_and_amortization"])
        debt_service = float(row["total_debt_service"])
        dscr_num = ebit + da
        dscr_den = debt_service
        dscr = _safe_divide(dscr_num, dscr_den, default=float("nan"))
        dscr_valid = bool(not np.isnan(dscr) and debt_service > 0)

        cash = float(row["cash_and_equivalents"])
        marketable = float(row["marketable_securities"])
        receivables = float(row["accounts_receivable"])
        opex = float(row["operating_expenses"])
        dchor_num = cash + marketable
        dchor_den = opex / 365.0 if opex > 0 else 0.0
        days_cash_on_hand = _safe_divide(dchor_num, dchor_den, default=float("nan"))
        dchor_valid = bool(not np.isnan(days_cash_on_hand) and opex > 0)

        current_assets = float(row["current_assets"])
        current_liabilities = float(row["current_liabilities"])
        current_ratio_num = current_assets
        current_ratio_den = current_liabilities
        current_ratio = _safe_divide(current_ratio_num, current_ratio_den, default=float("nan"))
        current_ratio_valid = bool(not np.isnan(current_ratio) and current_liabilities > 0)

        quick_ratio_num = cash + marketable + receivables
        quick_ratio_den = current_liabilities
        quick_ratio = _safe_divide(quick_ratio_num, quick_ratio_den, default=float("nan"))
        quick_ratio_valid = bool(not np.isnan(quick_ratio) and current_liabilities > 0)

        total_revenue = float(row["total_revenue"])
        operating_margin_num = total_revenue - opex
        operating_margin_den = total_revenue
        operating_margin = _safe_divide(operating_margin_num, operating_margin_den, default=float("nan"))
        operating_margin_valid = bool(not np.isnan(operating_margin) and total_revenue > 0)

        debt_to_revenue_num = debt_service
        debt_to_revenue_den = total_revenue
        debt_to_revenue = _safe_divide(debt_to_revenue_num, debt_to_revenue_den, default=float("nan"))
        debt_to_revenue_valid = bool(debt_to_revenue_den > 0)

        cmi_adjusted_cost_proxy = float("nan")
        if "cmi_index" in financials.columns and "total_discharges" in financials.columns:
            cmi = float(row.get("cmi_index", 0.0))
            discharges = float(row.get("total_discharges", 0.0))
            if cmi > 0 and discharges > 0:
                cmi_adjusted_cost_proxy = opex / (cmi * discharges)

        row_dict: dict[str, Any] = {
            hospital_col: hospital,
            period_start_col: period,
            "dscr": dscr,
            "days_cash_on_hand": days_cash_on_hand,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "operating_margin": operating_margin,
            "debt_to_revenue": debt_to_revenue,
            "cmi_adjusted_cost_proxy": cmi_adjusted_cost_proxy,
            "dscr_valid": dscr_valid,
            "days_cash_on_hand_valid": dchor_valid,
            "current_ratio_valid": current_ratio_valid,
            "quick_ratio_valid": quick_ratio_valid,
            "operating_margin_valid": operating_margin_valid,
            "debt_to_revenue_valid": debt_to_revenue_valid,
        }
        if return_detailed:
            row_dict.update({
                "dscr_numerator": dscr_num,
                "dscr_denominator": dscr_den,
                "dchor_numerator": dchor_num,
                "dchor_denominator": dchor_den,
                "current_ratio_numerator": current_ratio_num,
                "current_ratio_denominator": current_ratio_den,
                "quick_ratio_numerator": quick_ratio_num,
                "quick_ratio_denominator": quick_ratio_den,
                "operating_margin_numerator": operating_margin_num,
                "operating_margin_denominator": operating_margin_den,
                "debt_to_revenue_numerator": debt_to_revenue_num,
                "debt_to_revenue_denominator": debt_to_revenue_den,
            })
        rows.append(row_dict)

    frame = pd.DataFrame(rows).sort_values([hospital_col, period_start_col]).reset_index(drop=True)
    return frame


def compute_claim_development_triangle(
    claims: pd.DataFrame,
    *,
    triangle_id_col: Hashable = "triangle_id",
    origin_year_col: Hashable = "origin_year",
    development_period_col: Hashable = "development_period",
    cumulative_claim_col: Hashable = "cumulative_claim",
    incremental_claim_col: str | None = None,
) -> pd.DataFrame:
    """Transform claim records into a cumulative development triangle.

    The triangle is pivoted so each row is an origin (accident) year and each column
    is a development period. Cumulative claims are recomputed from incremental claims
    if an incremental column is provided; otherwise the provided cumulative column is
    validated for monotonicity.

    Args:
        claims: Claim records frame.
        triangle_id_col: Identifier for the triangle/business line.
        origin_year_col: Accident/origin year.
        development_period_col: Development period index (1-based).
        cumulative_claim_col: Cumulative claim amount column.
        incremental_claim_col: Optional incremental claim column to rebuild cumulative.

    Returns:
        Pivoted cumulative triangle frame sorted by origin year.
    """
    HospitalFinancialFeatureInputs.validate_triangle_inputs(claims)

    if incremental_claim_col and incremental_claim_col in claims.columns:
        claims = claims.copy()
        # Rebuild cumulative per triangle + origin from incremental
        claims["_incremental"] = claims[incremental_claim_col].astype(float)
        claims["_cumulative"] = claims.groupby([triangle_id_col, origin_year_col])["_incremental"].cumsum()
        cumulative_col = "_cumulative"
    else:
        cumulative_col = cumulative_claim_col

    if not np.issubdtype(claims[cumulative_col].dtype, np.floating):
        claims = claims.copy()
        claims[cumulative_col] = claims[cumulative_col].astype(float)

    pivot = claims.pivot_table(
        index=[triangle_id_col, origin_year_col],
        columns=development_period_col,
        values=cumulative_col,
        aggfunc="sum",
    )
    if pivot.empty:
        return pd.DataFrame(columns=[triangle_id_col, origin_year_col])

    result = pivot.reset_index()
    # Sort origin years ascending
    result = result.sort_values(origin_year_col).reset_index(drop=True)
    return result


def triangle_development_factor(
    triangle: pd.DataFrame,
    *,
    triangle_id_col: Hashable = "triangle_id",
    origin_year_col: Hashable = "origin_year",
    method: str = "chain_ladder",
) -> pd.DataFrame:
    """Estimate age-to-age development factors for a cumulative triangle.

    Implements the classical Chain Ladder development factor estimator:
        f_k = sum(C_{i,k+1}) / sum(C_{i,k})
    where sums run over origin years i with both periods observed.

    When triangle rows are incomplete (some development periods missing), factors are
    estimated only from fully-observed pairs.

    Args:
        triangle: Pivoted cumulative triangle from compute_claim_development_triangle.
        triangle_id_col: Business line identifier.
        origin_year_col: Origin year column.
        method: Development factor method; 'chain_ladder' is the only supported option
            in this phase.

    Returns:
        Frame with columns (triangle_id, development_period, development_factor,
        development_period_next, factor_count).
    """
    if method != "chain_ladder":
        msg = f"Unsupported triangle development method: {method}"
        raise FeatureEngineeringError(msg, context={"method": method})

    # Determine the development period columns (exclude id + origin)
    id_origin_cols = {triangle_id_col, origin_year_col}
    period_cols = [c for c in triangle.columns if c not in id_origin_cols]
    if not period_cols:
        msg = "Triangle has no development period columns"
        raise FeatureEngineeringError(msg)

    # Ensure period columns are sorted by period index
    try:
        period_cols_sorted = sorted(period_cols, key=lambda c: float(c))
    except (TypeError, ValueError):
        period_cols_sorted = list(period_cols)

    rows: list[dict[str, Any]] = []
    for tid, group in triangle.groupby(triangle_id_col):
        # Keep only the development columns in sorted order
        dev_cols = [c for c in period_cols_sorted if c in group.columns]
        if len(dev_cols) < 2:
            continue
        for k, current_col in enumerate(dev_cols[:-1]):
            next_col = dev_cols[k + 1]
            current = group[current_col].astype(float)
            next_col_vals = group[next_col].astype(float)
            mask = current.notna() & next_col_vals.notna() & (current > 0)
            if mask.sum() < 1:
                continue
            sum_current = current[mask].sum()
            sum_next = next_col_vals[mask].sum()
            factor = sum_next / sum_current if sum_current > 0 else float("nan")
            rows.append({
                triangle_id_col: tid,
                "development_period": current_col,
                "development_period_next": next_col,
                "development_factor": factor,
                "factor_count": int(mask.sum()),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                triangle_id_col,
                "development_period",
                "development_period_next",
                "development_factor",
                "factor_count",
            ]
        )
    return frame.sort_values([triangle_id_col, "development_period"]).reset_index(drop=True)


def compute_cmi_signals(
    admissions: pd.DataFrame,
    *,
    hospital_col: Hashable = "hospital_id",
    period_start_col: Hashable = "period_start",
    drg_volume_col: Hashable = "drg_volume",
    case_mix_index_col: Hashable = "case_mix_index",
) -> pd.DataFrame:
    """Compute Case Mix Index (CMI) signals and CMI-adjusted volume per hospital per period.

    CMI is computed as the discharge-weighted average of DRG relative weights when
    DRG volumes and weights are available. When an explicit CMI column is provided,
    it is carried forward and validated against the computed proxy where possible.

    Returns:
        Per-hospital per-period CMI signal frame with:
        - cmi_computed
        - cmi_provided (if available)
        - cmi_delta (computed - provided)
        - effective_case_load = total_discharges * cmi_computed
        - drg_diversity = number of distinct DRGs
    """
    HospitalFinancialFeatureInputs.validate_ratio_inputs(admissions)

    rows: list[dict[str, Any]] = []

    for (hospital, period), group in admissions.groupby(
        [hospital_col, period_start_col]
    ):
        drg_volume = group[drg_volume_col] if drg_volume_col in group.columns else pd.Series(dtype=float)
        cmi_provided = (
            float(group[case_mix_index_col].iloc[0])
            if case_mix_index_col in group.columns
            else float("nan")
        )
        discharges = float(
            group.get(
                "total_discharges", pd.Series([0.0])
            ).sum()
        )

        if drg_volume.size > 0 and discharges > 0:
            # DRG relative weight is taken from cmi_index column if available, else assume 1.0
            weights = group.get(case_mix_index_col, pd.Series([1.0] * len(group), index=group.index)).astype(float)
            weighted_volume = (drg_volume.fillna(0).to_numpy(dtype=float) * weights.fillna(1.0).to_numpy(dtype=float))
            cmi_computed = (
                float(weighted_volume.sum()) / float(drg_volume.fillna(0).sum())
                if drg_volume.fillna(0).sum() > 0
                else float("nan")
            )
        else:
            cmi_computed = cmi_provided if not np.isnan(cmi_provided) else 1.0

        cmi_delta = cmi_computed - cmi_provided if not np.isnan(cmi_provided) else float("nan")
        effective_case_load = discharges * cmi_computed if not np.isnan(cmi_computed) else float("nan")
        drg_diversity = int(group[drg_volume_col].nunique()) if drg_volume_col in group.columns else 0

        rows.append({
            hospital_col: hospital,
            period_start_col: period,
            "cmi_computed": cmi_computed,
            "cmi_provided": cmi_provided,
            "cmi_delta": cmi_delta,
            "effective_case_load": effective_case_load,
            "drg_diversity": drg_diversity,
            "total_discharges": discharges,
        })
    frame = pd.DataFrame(rows).sort_values([hospital_col, period_start_col]).reset_index(drop=True)
    return frame


def compute_trial_enrollment_velocity(
    trials: pd.DataFrame,
    *,
    nct_id_col: Hashable = "nct_id",
    start_date_col: Hashable = "study_start_date",
    completion_date_col: Hashable = "primary_completion_date",
    intended_col: Hashable = "enrollment_intended",
    actual_col: Hashable = "enrollment_actual",
    status_col: Hashable = "overall_status",
    term_match_col: Hashable = "search_term_match",
    velocity_window_days: int = 365,
) -> pd.DataFrame:
    """Compute enrollment velocity signals per trial for the pharma analytics pipeline.

    Velocity signals include:
    - enrollment_rate_per_year = actual / years_elapsed (capped)
    - enrollment_attainment = actual / intended
    - years_to_completion = (completion - start) in years
    - status_bucket (Active/Completed/Other)
    - term_match (for downstream portfolio weighting)

    Trials that are not yet recruiting or terminated are flagged and may be excluded
    by downstream filters.

    Args:
        trials: ClinicalTrials.gov frame from acquisition.
        nct_id_col: NCT identifier column.
        start_date_col: Study start date column.
        completion_date_col: Primary completion date column.
        intended_col: Intended enrollment column.
        actual_col: Actual enrollment column.
        status_col: Overall status column.
        term_match_col: Search term match column.
        velocity_window_days: Window used to compute annualized rate.

    Returns:
        Per-trial velocity frame.
    """
    HospitalFinancialFeatureInputs.validate_trial_inputs(trials)

    rows: list[dict[str, Any]] = []
    for _, row in trials.iterrows():
        start = pd.Timestamp(row[start_date_col])
        completion = pd.Timestamp(row[completion_date_col])
        intended = float(row[intended_col]) if intended_col in trials.columns else 0.0
        actual = float(row[actual_col]) if actual_col in trials.columns else 0.0
        status = str(row[status_col]) if status_col in trials.columns else "Unknown"
        term = str(row[term_match_col]) if term_match_col in trials.columns else ""

        duration_days = (completion - start).days
        duration_years = duration_days / 365.0 if duration_days > 0 else float("nan")
        rate = actual / (duration_years) if duration_years and duration_years > 0 else float("nan")
        attainment = actual / intended if intended > 0 else float("nan")

        if status in {"Recruiting", "Active, not recruiting"}:
            status_bucket = "Active"
        elif status == "Completed":
            status_bucket = "Completed"
        else:
            status_bucket = "Other"

        rows.append({
            nct_id_col: row[nct_id_col],
            "start_date": start,
            "completion_date": completion,
            "enrollment_intended": intended,
            "enrollment_actual": actual,
            "duration_years": duration_years,
            "enrollment_rate_per_year": rate,
            "enrollment_attainment": attainment,
            "status": status,
            "status_bucket": status_bucket,
            "term_match": term,
            "trial_active": status in {"Recruiting", "Active, not recruiting"},
        })
    frame = pd.DataFrame(rows).sort_values(nct_id_col).reset_index(drop=True)
    return frame


def _fill_default_trial_features(
    frame: pd.DataFrame,
    *,
    nct_id_col: Hashable = "nct_id",
) -> pd.DataFrame:
    """Fill missing enrollment velocity columns with safe defaults for downstream use."""
    cols_to_fill: dict[str, float] = {
        "enrollment_rate_per_year": 0.0,
        "enrollment_attainment": 0.0,
        "duration_years": 0.0,
        "enrollment_intended": 0.0,
        "enrollment_actual": 0.0,
    }
    for col, fill in cols_to_fill.items():
        if col in frame.columns:
            frame[col] = frame[col].fillna(fill)
    return frame


def build_financial_feature_frame(
    hospital_financials: pd.DataFrame | None = None,
    claim_triangles: pd.DataFrame | None = None,
    admissions: pd.DataFrame | None = None,
    trials: pd.DataFrame | None = None,
    *,
    include_cash_ratios: bool | None = None,
    include_claim_triangle: bool | None = None,
    include_cmi: bool | None = None,
    include_trial_velocity: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a unified financial feature frame from heterogeneous sources.

    This is the canonical entry point for financial features. In a full deployment
    the frame would be joined to hospital identifiers and scoring periods; here it
    returns a structured dict-of-frames when sources are sparse so that each
    downstream module can pick what it needs.

    Args:
        hospital_financials: Hospital financials frame.
        claim_triangles: Claim triangle frame.
        admissions: Admission/DRG volume frame.
        trials: Clinical trials frame.
        include_cash_ratios: Include hospital financial ratios.
        include_claim_triangle: Include claim development triangle.
        include_cmi: Include CMI signals.
        include_trial_velocity: Include trial enrollment velocity.

    Returns:
        Dict of named feature frames, primarily for consumption by the actuarial,
        credit, and pharma modules.
    """
    r = cfg()
    features = r.get("features", {}).get("financial", {})

    include_cash_ratios = (
        include_cash_ratios
        if include_cash_ratios is not None
        else bool(features.get("compute_cash_metrics", True))
    )
    include_claim_triangle = (
        include_claim_triangle
        if include_claim_triangle is not None
        else bool(features.get("compute_claim_triangles", True))
    )
    include_cmi = (
        include_cmi
        if include_cmi is not None
        else bool(features.get("compute_cmi_signals", True))
    )
    include_trial_velocity = (
        include_trial_velocity
        if include_trial_velocity is not None
        else bool(features.get("compute_trial_velocity", True))
    )

    out: dict[str, pd.DataFrame] = {}

    if include_cash_ratios and hospital_financials is not None:
        HospitalFinancialFeatureInputs.validate_ratio_inputs(hospital_financials)
        out["hospital_ratios"] = compute_hospital_financial_ratios(hospital_financials)

    if include_claim_triangle and claim_triangles is not None:
        HospitalFinancialFeatureInputs.validate_triangle_inputs(claim_triangles)
        out["claim_triangle"] = compute_claim_development_triangle(claim_triangles)
        out["triangle_factors"] = triangle_development_factor(out["claim_triangle"])

    if include_cmi and admissions is not None:
        HospitalFinancialFeatureInputs.validate_ratio_inputs(admissions)
        out["cmi_signals"] = compute_cmi_signals(admissions)

    if include_trial_velocity and trials is not None:
        HospitalFinancialFeatureInputs.validate_trial_inputs(trials)
        vel = compute_trial_enrollment_velocity(trials)
        out["trial_velocity"] = _fill_default_trial_features(vel)

    if not out:
        # Return an empty but structurally sound frame so calling code can
        # proceed without branching.
        out["empty"] = pd.DataFrame({"placeholder": []})
    return out
