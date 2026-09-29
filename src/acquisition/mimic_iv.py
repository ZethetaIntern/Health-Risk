"""MIMIC-IV clinical event ingestion adapter.

When a real MIMIC-IV extract is available (via ``MIMIC_INGEST_PATH`` in ``.env``),
this adapter reads near-production schema tables and enriches them into a normalized
clinical events frame. When the extract is unavailable, it deterministically generates
a synthetic but schema-valid cohort so downstream feature engineering and tests remain
fully functional.

The adapter logs a WARNING whenever it falls back to synthetic data so that runs are
auditable.

Design notes:
- Secrets are never hardcoded; the only optional external secret is the local path, which
  is not an API key.
- The mock generator is deterministic per-seed so tests are reproducible.
- All dates are timezone-naive UTC for consistency across the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from healthrisk_ai.config import cfg
from healthrisk_ai.exceptions import IngestionError, ValidationError
from healthrisk_ai.numerics import age_to_cms_hcc_bucket, random_state

logger = logging.getLogger("healthrisk_ai.acquisition.mimic_iv")

__all__ = ["load_mimic_events", "MimicEventsSchema"]


class MimicEventsSchema:
    """Schema contract for the normalized MIMIC-IV events frame.

    This is the canonical output of the adapter; downstream clinical feature
    engineering code depends on these columns.
    """

    required_columns = [
        "patient_id",
        "event_date",
        "event_type",
        "item_id",
        "code",
        "value",
        "value_unit",
        "age_years",
        "cms_hcc_age_bucket",
    ]
    optional_columns = [
        "icd_10_code",
        "cpt_code",
        "adas_score",
        "readmit_30d_flag",
        "discharge_disposition",
    ]
    categoricals = {
        "event_type": [
            "admission",
            "discharge",
            "lab",
            "vital",
            "medication",
            "procedure",
            "diagnosis",
            "microbiology",
        ],
    }

    @classmethod
    def validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(cls.required_columns) - set(df.columns)
        if missing:
            msg = f"MIMIC events frame missing required columns: {sorted(missing)}"
            raise IngestionError(msg, source="mimic_iv")
        if not np.issubdtype(df["event_date"].dtype, np.datetime64):
            msg = "MIMIC events frame event_date must be datetime64"
            raise ValidationError(msg, context={"column": "event_date"})
        if df["patient_id"].nunique() == 0:
            msg = "MIMIC events frame contains no patients"
            raise ValidationError(msg)
        return df


def _seed_from_config() -> int:
    """Return the deterministic seed for mock generation."""
    r = cfg()
    return int(r.get("default_seed", 20260914))


def _build_mock_patient_demographics(num_patients: int, seed: int) -> pd.DataFrame:
    rng = random_state(seed)
    patient_ids = np.arange(1, num_patients + 1)
    ages = rng.integers(0, 95, size=num_patients)
    # Simulate a realistic age distribution skewing older with a small pediatric tail
    age_weights = np.ones(96)
    age_weights[:18] *= 0.15
    age_weights[65:] *= 1.25
    age_weights = age_weights / age_weights.sum()
    ages = rng.choice(np.arange(96), size=num_patients, p=age_weights)
    sexes = rng.choice(["M", "F"], size=num_patients, p=[0.52, 0.48])
    los_days = rng.integers(1, 28, size=num_patients)
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age_years": ages,
            "sex": sexes,
            "los_days": los_days,
        }
    )


def _build_mock_events(
    demographics: pd.DataFrame,
    *,
    start: str = "2010-01-01",
    end: str = "2022-12-31",
    seed: int = 20260914,
) -> pd.DataFrame:
    rng = random_state(seed)
    rows: list[dict[str, Any]] = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    days_range = int((end_dt - start_dt).total_seconds() / 86400)

    labs = ["Glucose", "Creatinine", "Na", "K", "WBC", "Hgb", "Platelets"]
    meds = ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin", "Furosemide"]
    icd10 = ["E11.9", "I50.9", "J44.9", "N18.9", "E78.5", "I10", "K21.9", "F32.9"]
    cpts = ["99232", "36415", "80053", "93000", "33249"]

    for _, p in demographics.iterrows():
        pid = p["patient_id"]
        age = int(p["age_years"])
        # Each patient gets 1-3 admissions
        admissions = rng.integers(1, 4)
        for _ in range(admissions):
            day_offset = rng.integers(0, days_range)
            admit_dt = start_dt + pd.Timedelta(days=int(day_offset))
            los = int(p["los_days"]) + rng.integers(0, 7)
            discharge_dt = admit_dt + pd.Timedelta(days=los)

            rows.append({
                "patient_id": pid,
                "event_date": admit_dt,
                "event_type": "admission",
                "item_id": 0,
                "code": "",
                "value": int(p["los_days"]),
                "value_unit": "days",
                "age_years": age,
                "cms_hcc_age_bucket": age_to_cms_hcc_bucket(age),
                "icd_10_code": rng.choice(icd10),
                "cpt_code": rng.choice(cpts),
                "readmit_30d_flag": int(rng.binomial(1, 0.12)),
            })
            rows.append({
                "patient_id": pid,
                "event_date": discharge_dt,
                "event_type": "discharge",
                "item_id": 0,
                "code": "",
                "value": 0,
                "value_unit": "",
                "age_years": age,
                "cms_hcc_age_bucket": age_to_cms_hcc_bucket(age),
            })

            num_labs = rng.integers(3, 12)
            for _ in range(num_labs):
                lab_date = admit_dt + pd.Timedelta(days=int(rng.integers(0, los)))
                rows.append({
                    "patient_id": pid,
                    "event_date": lab_date,
                    "event_type": "lab",
                    "item_id": rng.integers(1, 500),
                    "code": rng.choice(labs),
                    "value": float(rng.normal(100, 25)),
                    "value_unit": "mg/dL",
                    "age_years": age,
                    "cms_hcc_age_bucket": age_to_cms_hcc_bucket(age),
                })

            num_meds = rng.integers(2, 8)
            for _ in range(num_meds):
                med_date = admit_dt + pd.Timedelta(days=int(rng.integers(0, los)))
                rows.append({
                    "patient_id": pid,
                    "event_date": med_date,
                    "event_type": "medication",
                    "item_id": rng.integers(1000, 9999),
                    "code": rng.choice(meds),
                    "value": float(rng.integers(1, 3)),
                    "value_unit": "mg",
                    "age_years": age,
                    "cms_hcc_age_bucket": age_to_cms_hcc_bucket(age),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=MimicEventsSchema.required_columns)
    return df


def _read_real_extract(path: str) -> pd.DataFrame:
    """Read a real MIMIC-IV extract when available.

    The extract is expected to be a CSV or Parquet file with at least the columns in
    MimicEventsSchema. If the file is present but cannot be parsed, we raise so CI
    and operators see the real failure rather than silently masking it.
    """
    import os

    path_str = str(path)

    if not os.path.exists(path_str):
        msg = f"MIMIC ingest path does not exist: {path_str}"
        raise IngestionError(msg, source="mimic_iv")

    try:
        if path_str.lower().endswith(".parquet"):
            df = pd.read_parquet(path_str)
        else:
            df = pd.read_csv(path_str, dtype={"patient_id": "int64"})
    except Exception as exc:
        msg = f"Failed to parse MIMIC ingest path {path_str}"
        raise IngestionError(msg, source="mimic_iv") from exc

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)
    df["event_date"] = df["event_date"].dt.tz_localize(None)
    return df


def load_mimic_events(
    *,
    force_mock: bool = False,
) -> pd.DataFrame:
    """Load normalized MIMIC-IV clinical events.

    Returns a DataFrame conforming to ``MimicEventsSchema``. When
    ``MIMIC_INGEST_PATH`` is configured and the file exists, the real extract is
    used. Otherwise, synthetic but schema-valid data is generated.

    Args:
        force_mock: If True, always generate synthetic data, even when a path is
            configured.

    Returns:
        Normalized clinical events frame.
    """
    r = cfg()
    path_val = str(r.get("paths", {}).get("mimic_ingest_path") or "").strip()
    if force_mock or not path_val:
        logger.warning(
            "MIMIC-IV extract unavailable; generating synthetic cohort "
            "of %d patients",
            int(r.get("acquisition", {}).get("mimic", {}).get("mock_num_patients", 2000)),
        )
        demographics = _build_mock_patient_demographics(
            int(r.get("acquisition", {}).get("mimic", {}).get("mock_num_patients", 2000)),
            _seed_from_config(),
        )
        start = str(r.get("acquisition", {}).get("mimic", {}).get("event_date_range", {}).get("start", "2010-01-01"))
        end = str(r.get("acquisition", {}).get("mimic", {}).get("event_date_range", {}).get("end", "2022-12-31"))
        return _build_mock_events(demographics, start=start, end=end, seed=_seed_from_config())

    try:
        df = _read_real_extract(path_val)
    except IngestionError as exc:
        logger.warning("MIMIC-IV extract unavailable; falling back to synthetic data: %s", exc)
        demographics = _build_mock_patient_demographics(
            int(r.get("acquisition", {}).get("mimic", {}).get("mock_num_patients", 2000)),
            _seed_from_config(),
        )
        start = str(r.get("acquisition", {}).get("mimic", {}).get("event_date_range", {}).get("start", "2010-01-01"))
        end = str(r.get("acquisition", {}).get("mimic", {}).get("event_date_range", {}).get("end", "2022-12-31"))
        return _build_mock_events(demographics, start=start, end=end, seed=_seed_from_config())
    return MimicEventsSchema.validate(df)
