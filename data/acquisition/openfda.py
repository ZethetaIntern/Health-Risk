"""openFDA adverse-event ingestion adapter.

This adapter pulls adverse-event records from the openFDA device/exporter endpoints
and normalizes them into a tabular FAERS-flavored frame. It includes retry logic,
per-second rate limiting, pagination handling, and deterministic mock fallback so
the pharma signal and explainability pipelines can be exercised without live API
access in every environment.

No API keys are required; when ``OPENFDA_API_KEY`` is present it is passed as a
header for higher rate limits, otherwise public access is used.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import httpx
import numpy as np
import pandas as pd
from healthrisk_ai.acquisition.base import DataFrameSchema, RetryConfig, rate_limited_request

from healthrisk_ai.config import cfg
from healthrisk_ai.exceptions import IngestionError, ValidationError
from healthrisk_ai.numerics import random_state

logger = logging.getLogger("healthrisk_ai.acquisition.openfda")

__all__ = ["load_openfda_adverse_events", "OpenFdaAdverseEventSchema"]


class OpenFdaAdverseEventSchema(DataFrameSchema):
    required_columns = [
        "event_id",
        "report_date",
        "patient_id",
        "device_name",
        "adverse_event_term",
        "serious_flag",
        "outcome_term",
        "country",
    ]
    optional_columns = [
        "manufacturer_name",
        "report_type",
        "mdr_report_purpose",
        "product_category",
    ]

    def validate(self, df: pd.DataFrame, *, dataset_name: str = "openfda") -> pd.DataFrame:
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            msg = f"{dataset_name} missing required columns: {sorted(missing)}"
            raise IngestionError(msg, source=dataset_name)
        if df["event_id"].duplicated().any():
            msg = f"{dataset_name} contains duplicate event_id rows"
            raise ValidationError(msg, context={"dataset": dataset_name})
        if not np.issubdtype(df["report_date"].dtype, np.datetime64):
            msg = "openfda report_date must be datetime64"
            raise ValidationError(msg, context={"column": "report_date"})
        return df


def _seed_from_config() -> int:
    return int(cfg().get("default_seed", 20260914))


def _mock_adverse_events(
    *,
    start: str = "2017-01-01",
    end: str = "2026-01-01",
    seed: int = 20260914,
) -> pd.DataFrame:
    """Generate deterministic mock FAERS-flavored adverse events."""
    rng = random_state(seed)
    rows: list[dict[str, Any]] = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    days_range = max(1, int((end_dt - start_dt).total_seconds() / 86400))
    device_names = ["Cardiac Monitor", "Infusion Pump", "Ventilator", "MRI Coil", "Defibrillator"]
    terms = ["device_malfunction", "infection", "hypotension", "arrhythmia", "thrombosis", "hemorrhage"]
    outcomes = ["recovered", "recovered_with_residual_effects", "life_threatening", "death"]
    countries = ["US", "DE", "JP", "GB", "FR", "CA", "AU"]
    num_events = int(rng.integers(500, 2000))

    for _ in range(num_events):
        event_dt = start_dt + pd.Timedelta(days=int(rng.integers(0, days_range)))
        rows.append({
            "event_id": int(rng.integers(10_000_000, 99_999_999)),
            "report_date": event_dt,
            "patient_id": int(rng.integers(1, 100_000)),
            "device_name": rng.choice(device_names),
            "adverse_event_term": rng.choice(terms),
            "serious_flag": int(rng.binomial(1, 0.35)),
            "outcome_term": rng.choice(outcomes),
            "country": rng.choice(countries),
            "manufacturer_name": rng.choice(["MedDevice A", "MedDevice B", "MedDevice C"]),
            "report_type": rng.choice(["initial", "follow-up"]),
            "mdr_report_purpose": rng.choice(["malfunction", "adverse_event"]),
            "product_category": rng.choice(["monitor", "pump", "ventilation", "imaging", "cardiac"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=OpenFdaAdverseEventSchema.required_columns)
    return df


def _build_client() -> httpx.Client:
    headers: dict[str, Any] = {}
    api_key = cfg().get("paths", {}).get("openfda_api_key") or ""
    import os
    env_key = os.environ.get("OPENFDA_API_KEY", "")
    key = api_key or env_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return httpx.Client(base_url="https://api.fda.gov", headers=headers)


def _fetch_openfda(
    start: str,
    end: str,
    max_results: int,
    *,
    force_mock: bool = False,
) -> pd.DataFrame:
    """Fetch openFDA adverse events for a window, with pagination and retry."""
    if force_mock:
        return _mock_adverse_events(start=start, end=end, seed=_seed_from_config())

    client: httpx.Client | None = None
    try:
        client = _build_client()
        retry_config = RetryConfig.from_config()
        rows: list[dict[str, Any]] = []
        skip = 0
        while True:
            params = {
                "search": f"report_date:{start}+TO+{end}",
                "limit": retry_config.rate_limit_per_second,
                "skip": skip,
            }
            response = rate_limited_request(
                client,
                method="GET",
                url="/device/enforcement.json",
                params=params,
                request_label="openfda_adverse_events",
                retry_config=retry_config,
            )
            body = response.json()
            results = body.get("results", [])
            if not results:
                break
            for item in results:
                row = {
                    "event_id": int(item.get(" penn", 0) or 0),
                    "report_date": pd.Timestamp(item.get("report_date", start)),
                    "patient_id": int(item.get("patient_id", 0) or 0),
                    "device_name": str(item.get("device_name", "Unknown")),
                    "adverse_event_term": str(item.get("adverse_event_term", "unknown")),
                    "serious_flag": int(item.get("serious", 0) or 0),
                    "outcome_term": str(item.get("outcome_term", "unknown")),
                    "country": str(item.get("country", "unknown")),
                    "manufacturer_name": str(item.get("manufacturer_name", "")),
                    "report_type": str(item.get("report_type", "")),
                    "mdr_report_purpose": str(item.get("mdr_report_purpose", "")),
                    "product_category": str(item.get("product_category", "")),
                }
                rows.append(row)
            skip += len(results)
            if skip >= max_results or len(results) < retry_config.rate_limit_per_second:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=OpenFdaAdverseEventSchema.required_columns)
        return df
    except IngestionError:
        raise
    except Exception as exc:
        logger.warning("openFDA live fetch failed; falling back to mock: %s", exc)
        return _mock_adverse_events(start=start, end=end, seed=_seed_from_config())
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()


def load_openfda_adverse_events(*, force_mock: bool = False) -> pd.DataFrame:
    """Load FAERS-flavored adverse events from openFDA.

    Returns a DataFrame conforming to ``OpenFdaAdverseEventSchema``. When the live
    fetch fails (network, rate limits, missing key), deterministic mock data is used
    and a WARNING is logged.
    """
    r = cfg()
    start = str(
        r.get("acquisition", {}).get("openfda", {}).get("adverse_event_start", "2017-01-01")
    )
    end = str(
        r.get("acquisition", {}).get("openfda", {}).get("adverse_event_end", "2026-01-01")
    )
    max_results = int(
        r.get("acquisition", {}).get("openfda", {}).get("max_results_per_request", 100)
    )
    df = _fetch_openfda(start=start, end=end, max_results=max_results, force_mock=force_mock)
    return OpenFdaAdverseEventSchema().validate(df, dataset_name="openfda")
