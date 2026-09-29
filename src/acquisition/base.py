"""Shared acquisition primitives: retry strategy, rate limiting, and schema contracts.

Every adapter in this package is expected to use the retry strategy and rate limiter
defined here so ingestion behavior is uniform across MIMIC-IV, openFDA, and
ClinicalTrials.gov.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

import httpx

from healthrisk_ai.config import cfg
from healthrisk_ai.exceptions import IngestionError

logger = logging.getLogger("healthrisk_ai.acquisition")

T = TypeVar("T")


__all__ = [
    "RetryConfig",
    "rate_limited_request",
    "DataFrameSchema",
]


class RetryConfig:
    """Tenacity-style retry parameters derived from the global config.

    This wrapper exists so the acquisition adapters don't need to know the config
    schema; they just ask for a RetryConfig and consume its attributes.
    """

    def __init__(self) -> None:
        raw = cfg().get("retry", {})
        self.max_attempts = int(raw.get("max_attempts", 5))
        self.base_backoff_seconds = float(raw.get("base_backoff_seconds", 1.0))
        self.max_backoff_seconds = float(raw.get("max_backoff_seconds", 32.0))
        self.rate_limit_per_second = float(raw.get("rate_limit_per_second", 2.0))

    @classmethod
    def from_config(cls) -> RetryConfig:
        return cls()


def rate_limited_request(
    client: httpx.Client,
    *,
    method: str = "GET",
    url: str,
    params: dict[str, Any] | None = None,
    request_label: str = "request",
    retry_config: RetryConfig | None = None,
) -> httpx.Response:
    """Issue an HTTP request with retry and per-second rate limiting.

    Args:
        client: Reused httpx client.
        method: HTTP method.
        url: Request URL path (absolute URL or path appended to client base_url).
        params: Query parameters.
        request_label: Human-readable label for logging.
        retry_config: Retry parameters; when None, derives from config.

    Returns:
        The final httpx Response that was returned from the server.
    """
    if retry_config is None:
        retry_config = RetryConfig.from_config()

    last_error: Exception | None = None
    for attempt in range(1, retry_config.max_attempts + 1):
        before = time.monotonic()
        try:
            response = client.request(method, url, params=params)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, OSError, ConnectionError) as exc:
            last_error = exc
            logger.debug(
                "%s attempt %d/%d failed: %s",
                request_label,
                attempt,
                retry_config.max_attempts,
                exc,
            )
            if attempt >= retry_config.max_attempts:
                break
            elapsed = time.monotonic() - before
            sleep_for = _compute_backoff(attempt, retry_config)
            rate_sleep = _rate_limit_sleep(elapsed, retry_config.rate_limit_per_second)
            time.sleep(max(sleep_for, rate_sleep))

    assert last_error is not None
    raise last_error


def _compute_backoff(attempt: int, retry_config: RetryConfig) -> float:
    raw = retry_config.base_backoff_seconds * (2 ** (attempt - 1))
    return min(raw, retry_config.max_backoff_seconds)


def _rate_limit_sleep(elapsed_seconds: float, limit_per_second: float) -> float:
    if limit_per_second <= 0:
        return 0.0
    min_interval = 1.0 / limit_per_second
    if elapsed_seconds < min_interval:
        return min_interval - elapsed_seconds
    return 0.0





class DataFrameSchema:
    """Lightweight schema contract for ingested DataFrames.

    Adapters subclass this and implement ``validate`` so downstream feature
    engineering code can depend on a known set of columns.
    """

    required_columns: list[str] = []
    optional_columns: list[str] = []

    def validate(self, df: Any, *, dataset_name: str = "frame") -> Any:
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            msg = f"{dataset_name} missing required columns: {sorted(missing)}"
            raise IngestionError(msg, source=dataset_name)
        return df
