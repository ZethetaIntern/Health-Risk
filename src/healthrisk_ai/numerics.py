"""Numeric and array helpers used across the clinical, financial, and simulation modules.

All functions are pure where possible, annotated, and documented so they can be
tested in isolation without touching model or I/O code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "safe_log1p",
    "rate_limited_sleep",
    "clip",
    "rolling_zscore",
    "jensen_shannon_divergence",
    "random_state",
    "quantile_safe",
    "age_to_cms_hcc_bucket",
    "time_aware_train_test_split",
    "is_monotonically_increasing",
    "as_numeric",
]


def safe_log1p(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable log(1 + x) that clips negatives to zero before the transform.

    This is useful for financial and clinical counts that can temporarily dip below
    zero due to corrections and must never pass negative values into a log.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_arr = np.clip(x_arr, a_min=0.0, a_max=None)
    out = np.log1p(x_arr)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(out)
    return out


def clip(value: float, *, lower: float | None = None, upper: float | None = None) -> float:
    """Clip a scalar to optional bounds.

    Kept separate from numpy.clip to avoid numpy dependency in the pure scalar path
    and to make the intent explicit in financial formulas.
    """
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return float(value)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score on a series with a fixed window.

    Missing values in the output are preserved as NaN where the window is not yet
    filled; callers are expected to handle that downstream.
    """
    if window <= 0:
        msg = f"rolling_zscore requires a positive window, got {window}"
        raise ValueError(msg)

    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence between two discrete distributions.

    Inputs are normalized internally. Returns 0 when both distributions are identical.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, a_min=0.0, a_max=None)
    q = np.clip(q, a_min=0.0, a_max=None)
    p_norm = p / p.sum() if p.sum() > 0 else p
    q_norm = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p_norm + q_norm)
    kl_pm = _kl_divergence(p_norm, m) + _kl_divergence(q_norm, m)
    jsd = 0.5 * kl_pm
    if np.isnan(jsd) or np.isinf(jsd):
        return 0.0
    return float(jsd)


def _kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    mask = a > 0
    return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], a_min=1e-12, a_max=None))))


def quantile_safe(values: np.ndarray, q: float) -> float:
    """Compute the q-th quantile with robust handling of empty inputs.

    When the input is empty, returns NaN; callers should decide how to handle that.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def age_to_cms_hcc_bucket(age: int) -> str:
    """Map a patient age to the CMS-HCC age bucket label used in risk adjustment.

    The buckets follow the standard HCC age grouping used in the Medicare risk
    adjustment model. Ages outside the supported range are mapped to a boundary
    bucket rather than raising.
    """
    if age < 0:
        return "age_unknown"
    if age < 1:
        return "age_0"
    if age < 2:
        return "age_1"
    if age < 5:
        return "age_2_4"
    if age < 10:
        return "age_5_9"
    if age < 15:
        return "age_10_14"
    if age < 20:
        return "age_15_19"
    if age < 25:
        return "age_20_24"
    if age < 30:
        return "age_25_29"
    if age < 35:
        return "age_30_34"
    if age < 40:
        return "age_35_39"
    if age < 45:
        return "age_40_44"
    if age < 50:
        return "age_45_49"
    if age < 55:
        return "age_50_54"
    if age < 60:
        return "age_55_59"
    if age < 65:
        return "age_60_64"
    if age < 70:
        return "age_65_69"
    if age < 75:
        return "age_70_74"
    if age < 80:
        return "age_75_79"
    if age < 85:
        return "age_80_84"
    return "age_85_plus"


def random_state(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator seeded deterministically.

    Pass ``None`` to use an internal entropy source; pass an integer for
    reproducibility across acquisition and simulation.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return np.random.default_rng(seed)


def is_monotonically_increasing(values: np.ndarray) -> bool:
    """Return True if the 1D array is monotonically non-decreasing."""
    values = np.asarray(values)
    if values.ndim != 1:
        msg = f"is_monotonically_increasing expects a 1D array, got shape {values.shape}"
        raise ValueError(msg)
    if values.size <= 1:
        return True
    return bool(np.all(np.diff(values) >= 0))


def as_numeric(value: Any, *, nan_if_bad: bool = True) -> float | None:
    """Coerce a value to a float, returning None on failure if requested."""
    try:
        result = float(value)
        return result
    except (TypeError, ValueError):
        return None if nan_if_bad else float("nan")


def time_aware_train_test_split(
    *,
    dates: pd.Series,
    labels: pd.Series,
    features: pd.DataFrame,
    split_date: str | pd.Timestamp,
    shuffle_training: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into training and testing sets using a temporal cutoff.

    This is the canonical splitter for every model in the ensemble to prevent
    temporal leakage. It returns (X_train, X_test, y_train, y_test).
    """
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    cutoff = pd.Timestamp(split_date)
    dates_tz = getattr(dates.dt, "tz", None)
    if cutoff.tzinfo is None and dates_tz is not None:
        cutoff = cutoff.tz_localize(dates_tz)
    elif cutoff.tzinfo is not None and dates_tz is None:
        raise ValueError("split_date timezone must match dates timezone or be naive")

    train_mask = (dates < cutoff).to_numpy()
    test_mask = (dates >= cutoff).to_numpy()

    if train_mask.sum() == 0:
        msg = f"No training rows before split_date={split_date}"
        raise ValueError(msg)
    if test_mask.sum() == 0:
        msg = f"No testing rows on or after split_date={split_date}"
        raise ValueError(msg)

    X_train = features.loc[train_mask]
    X_test = features.loc[test_mask]
    y_train = labels.loc[train_mask]
    y_test = labels.loc[test_mask]

    if shuffle_training is None:
        shuffle_training = False
    if shuffle_training:
        train_idx = X_train.sample(frac=1.0, random_state=0).index
        X_train = X_train.loc[train_idx]
        y_train = y_train.loc[train_idx]

    return X_train, X_test, y_train, y_test


def rate_limited_sleep(elapsed_seconds: float, *, limit_per_second: float) -> float:
    """Sleep to respect a per-second rate limit, returning the next available time.

    This helper is used by acquisition modules. It returns the recommended delay
    before the next request so callers can implement backpressure precisely.
    """
    if limit_per_second <= 0:
        msg = f"rate_limited_sleep requires a positive rate, got {limit_per_second}"
        raise ValueError(msg)
    min_interval = 1.0 / limit_per_second
    if elapsed_seconds < min_interval:
        return min_interval - elapsed_seconds
    return 0.0
