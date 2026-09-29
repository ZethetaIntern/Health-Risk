"""Shared pytest fixtures for Phase 1 tests.

This module provides reusable synthetic frames for the acquisition and feature
engineering modules so tests can be written against a common, deterministic
baseline without pulling in heavy I/O.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Generator
import numpy as np
import pandas as pd
import pytest
from healthrisk_ai.acquisition.mimic_iv import MimicEventsSchema
from healthrisk_ai.processing.clinical_features import build_clinical_matrix



@pytest.fixture
def seed() -> int:
    return 20260914


@pytest.fixture
def datetime_index(seed: int) -> pd.DatetimeIndex:
    _rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    return pd.date_range(start=start, periods=100, freq="D").to_series().sample(
        n=10, random_state=seed
    ).sort_values().dt.floor("D").values


@pytest.fixture
def mimic_events_frame(datetime_index: Any) -> pd.DataFrame:
    """Return a small but schema-valid MIMIC-IV events frame."""
    codes = [
        "I21.0",
        "I25.10",
        "E11.9",
        "I50.9",
        "K21.9",
        "C78.00",
        "N18.9",
        "Z87.891",
    ]
    events: list[dict[str, Any]] = []
    patient_ids = [1001, 1002, 1003]
    for pid in patient_ids:
        for i, dt in enumerate(datetime_index[:5]):
            events.append(
                {
                    "patient_id": pid,
                    "event_date": pd.Timestamp(dt),
                    "event_type": ["admission", "lab", "medication", "diagnosis"][i % 4],
                    "item_id": i,
                    "code": codes[i % len(codes)],
                    "value": float(42.0 + i),
                    "value_unit": "mg/dL",
                    "age_years": 72 + (pid % 3),
                    "cms_hcc_age_bucket": "age_65_69",
                    "readmit_30d_flag": int(i % 2),
                    "discharge_disposition": "home",
                    "icd_10_code": codes[i % len(codes)],
                }
            )
        # Ensure at least one medication event for polypharmacy testing
        events.append(
            {
                "patient_id": pid,
                "event_date": pd.Timestamp(datetime_index[0]),
                "event_type": "medication",
                "item_id": 10,
                "code": f"MED_{pid}_1",
                "value": 1.0,
                "value_unit": "mg",
                "age_years": 72,
                "cms_hcc_age_bucket": "age_65_69",
            }
        )
        events.append(
            {
                "patient_id": pid,
                "event_date": pd.Timestamp(datetime_index[0]),
                "event_type": "medication",
                "item_id": 11,
                "code": f"MED_{pid}_2",
                "value": 1.0,
                "value_unit": "mg",
                "age_years": 72,
                "cms_hcc_age_bucket": "age_65_69",
            }
        )

    df = pd.DataFrame(events)
    # Normalize columns to satisfy schema
    df = MimicEventsSchema.validate(df)
    return df


@pytest.fixture
def patient_demographics(mimic_events_frame: pd.DataFrame) -> pd.DataFrame:
    return build_clinical_matrix(mimic_events_frame).loc[:, ["patient_id", "age_years"]].copy()


@pytest.fixture
def hospital_financials_frame() -> pd.DataFrame:
    """Return a small hospital financials frame for ratio testing."""
    df = pd.DataFrame(
        {
            "hospital_id": ["H1", "H1", "H2"],
            "period_start": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-01-01"]),
            "cash_and_equivalents": [50_000_000, 55_000_000, 10_000_000],
            "marketable_securities": [10_000_000, 12_000_000, 2_000_000],
            "accounts_receivable": [20_000_000, 22_000_000, 5_000_000],
            "total_revenue": [200_000_000, 220_000_000, 50_000_000],
            "operating_expenses": [180_000_000, 195_000_000, 55_000_000],
            "interest_expense": [5_000_000, 5_500_000, 2_000_000],
            "depreciation_and_amortization": [15_000_000, 16_000_000, 5_000_000],
            "total_debt_service": [20_000_000, 22_000_000, 8_000_000],
            "current_assets": [100_000_000, 120_000_000, 25_000_000],
            "current_liabilities": [40_000_000, 45_000_000, 10_000_000],
            "cmi_index": [1.2, 1.3, 0.9],
            "total_discharges": [5000, 5500, 1000],
        }
    )
    return df


@pytest.fixture
def claim_triangle_frame() -> pd.DataFrame:
    """Return a small cumulative claim development triangle."""
    rows: list[dict[str, Any]] = []
    for origin in [2018, 2019, 2020]:
        for dev in [1, 2, 3]:
            base = origin * 1000 + dev * 100
            rows.append(
                {
                    "triangle_id": "A",
                    "origin_year": origin,
                    "development_period": dev,
                    "cumulative_claim": float(base) if dev <= (2020 - origin + 1) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def admissions_frame(hospital_financials_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a small admissions/DRG volume frame for CMI testing."""
    df = hospital_financials_frame[["hospital_id", "period_start"]].copy()
    df["drg_volume"] = [500, 600, 100]
    df["case_mix_index"] = [1.2, 1.3, 0.9]
    df["total_discharges"] = [5000, 5500, 1000]
    return df


@pytest.fixture
def trials_frame() -> pd.DataFrame:
    """Return a small ClinicalTrials.gov frame for enrollment velocity testing."""
    df = pd.DataFrame(
        {
            "nct_id": ["NCT000001", "NCT000002", "NCT000003"],
            "study_start_date": pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01"]),
            "primary_completion_date": pd.to_datetime(
                ["2020-01-01", "2021-01-01", "2022-01-01"]
            ),
            "overall_status": ["Completed", "Active, not recruiting", "Terminated"],
            "phase": ["Phase 3", "Phase 2", "Phase 1"],
            "enrollment_intended": [500, 300, 100],
            "enrollment_actual": [480, 150, 40],
            "intervention_type": ["Drug", "Device", "Biological"],
            "conditions": ["Myocardial Infarction", "Heart Failure", "Atrial Fibrillation"],
            "search_term_match": ["cardiovascular", "cardiovascular", "diabetes"],
        }
    )
    return df


@pytest.fixture
def sample_comorbidity_resource_path(tmp_path: Path) -> Path:
    """Create an on-disk comorbidity YAML resource for loading tests."""
    resource = tmp_path / "comorbidity.yaml"
    payload = {
        "version": "1.0.0",
        "conditions": {
            "myocardial_infarction": ["I21", "I22", "I23"],
            "congestive_heart_failure": ["I50", "I11.0"],
        },
    }
    resource.write_text(json.dumps(payload), encoding="utf-8")
    return resource


@pytest.fixture
def elixhauser_resource_path(tmp_path: Path) -> Path:
    """Create an on-disk Elixhauser YAML resource for loading tests."""
    resource = tmp_path / "elixhauser.yaml"
    payload = {
        "version": "1.0.0",
        "conditions": {
            "psychoses": ["F20", "F21", "F22"],
            "fluid_electrolyte_disorder": ["E26", "E27"],
        },
    }
    resource.write_text(json.dumps(payload), encoding="utf-8")
    return resource


@pytest.fixture
def clinical_feature_config_fixture() -> Generator[dict[str, Any], None, None]:
    """Patch config so clinical features load custom resource paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        charlson = tmp / "charlson.yaml"
        elixhauser = tmp / "elixhauser.yaml"
        charlson.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "conditions": {
                        "myocardial_infarction": ["I21", "I22", "I23"],
                        "congestive_heart_failure": ["I50"],
                    },
                }
            ),
            encoding="utf-8",
        )
        elixhauser.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "conditions": {
                        "psychoses": ["F20"],
                        "fluid_electrolyte_disorder": ["E26"],
                    },
                }
            ),
            encoding="utf-8",
        )
    with (patch("healthrisk_ai.processing.clinical_features._comorbidity_resource_path", charlson),
          patch("healthrisk_ai.processing.clinical_features._elixhauser_resource_path", elixhauser)):
        yield {"charlson_path": charlson, "elixhauser_path": elixhauser}
