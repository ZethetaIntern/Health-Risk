"""Custom domain exceptions for the HealthRisk AI platform.

Exceptions are centralized here so callers can catch broadly by category while
still carrying structured metadata (dataset name, quarter, asset, etc.) for
observability and retries.

The module deliberately avoids importing heavy model or I/O libraries; it is a
pure-logic exception hierarchy.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HealthRiskError",
    "IngestionError",
    "FeatureEngineeringError",
    "ModelError",
    "SimulationError",
    "ValidationError",
    "ConfigError",
]


class HealthRiskError(Exception):
    """Base class for all platform errors.

    All custom exceptions in this package inherit from this class so that
    top-level handlers can intercept platform-level failures without catching
    unrelated stdlib errors.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class ConfigError(HealthRiskError):
    """Raised when the runtime configuration is invalid, missing, or inconsistent."""


class IngestionError(HealthRiskError):
    """Raised when an acquisition/ingestion step fails after retries are exhausted."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        record_count: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.source = source
        self.record_count = record_count


class FeatureEngineeringError(HealthRiskError):
    """Raised when clinical or financial feature derivation fails validation checks."""


class ModelError(HealthRiskError):
    """Raised when model construction, training, inference, or serialization fails."""


class SimulationError(HealthRiskError):
    """Raised when the HealthRisk Lab simulation loop produces an invalid state."""

    def __init__(
        self,
        message: str,
        *,
        quarter: int | None = None,
        capital: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.quarter = quarter
        self.capital = capital


class ValidationError(HealthRiskError):
    """Raised when input data fails schema, range, or temporal sanity checks."""
