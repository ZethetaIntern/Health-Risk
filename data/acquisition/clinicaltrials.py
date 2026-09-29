"""ClinicalTrials.gov study ingestion adapter.

This adapter queries the ClinicalTrials.gov v2 API for interventional studies matching
configured search terms, extracts study-level features relevant to the pharma enrollment
velocity pipeline, and returns a normalized studies frame.

Because the public API can be slow and rate-limited in some environments, the adapter
includes retry logic, polite rate limiting (with a courtesy email header when configured),
pagination, and deterministic mock fallback so the pharma analytics modules remain
runnable offline.

No API keys are required.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

from healthrisk_ai.acquisition.base import DataFrameSchema, RetryConfig, rate_limited_request

from healthrisk_ai.config import cfg
from healthrisk_ai.exceptions import IngestionError, ValidationError
from healthrisk_ai.numerics import random_state

logger = logging.getLogger("healthrisk_ai.acquisition.clinicaltrials")

__all__ = ["load_clinical_trials", "ClinicalTrialsSchema"]


class ClinicalTrialsSchema(DataFrameSchema):
    required_columns = [
        "nct_id",
        "study_start_date",
        "primary_completion_date",
        "overall_status",
        "phase",
        "enrollment_intended",
        "enrollment_actual",
        "intervention_type",
        "conditions",
        "search_term_match",
    ]
    optional_columns = [
        "sponsor",
        "official_title",
        "study_type",
        "allocation",
        "intervention_model",
        "primary_purpose",
        "masks",
    ]

    def validate(self, df: pd.DataFrame, *, dataset_name: str = "clinicaltrials") -> pd.DataFrame:
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            msg = f"{dataset_name} missing required columns: {sorted(missing)}"
            raise IngestionError(msg, source=dataset_name)
        duplicated = df["nct_id"].duplicated()
        if duplicated.any():
            msg = f"{dataset_name} contains duplicate nct_id rows"
            raise ValidationError(msg, context={"dataset": dataset_name})
        if not np.issubdtype(df["study_start_date"].dtype, np.datetime64):
            msg = "clinicaltrials study_start_date must be datetime64"
            raise ValidationError(msg, context={"column": "study_start_date"})
        return df


def _seed_from_config() -> int:
    return int(cfg().get("default_seed", 20260914))


def _mock_trials(
    *,
    search_terms: Iterable[str],
    max_studies: int,
    seed: int = 20260914,
) -> pd.DataFrame:
    """Generate deterministic mock ClinicalTrials.gov records relevant to pharma velocity."""
    rng = random_state(seed)
    rows: list[dict[str, Any]] = []
    statuses = ["Recruiting", "Active, not recruiting", "Completed", "Terminated"]
    phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Early Phase 1"]
    intervention_types = ["Drug", "Device", "Procedure", "Behavioral", "Biological"]

    conditions_by_term: dict[str, list[str]] = {
        "cardiovascular": ["Myocardial Infarction", "Heart Failure", "Atrial Fibrillation"],
        "oncology": ["Non-Small Cell Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
        "diabetes": ["Type 2 Diabetes Mellitus", "Diabetic Nephropathy", "Diabetic Retinopathy"],
    }

    base_date = pd.Timestamp("2005-01-01")
    months_range = 216  # ~18 years
    terms = list(search_terms)
    total_terms = max(1, len(terms))

    for _ in range(max_studies):
        nct_id = "NCT" + str(int(rng.integers(1_000_000_000, 9_999_999_999)))
        study_start = base_date + pd.Timedelta(days=int(rng.integers(0, months_range * 30)))
        primary_completion = study_start + pd.Timedelta(days=int(rng.integers(365, 1500)))
        status = rng.choice(statuses, p=[0.25, 0.20, 0.45, 0.10])
        phase = rng.choice(phases, p=[0.15, 0.35, 0.30, 0.15, 0.05])
        intervention_type = rng.choice(intervention_types)

        term_weights = np.ones(total_terms)
        term = terms[int(rng.choice(np.arange(total_terms), p=term_weights / term_weights.sum()))]
        condition_list = conditions_by_term.get(term, ["Other Condition"])
        condition = rng.choice(condition_list)

        enrollment_intended = int(rng.integers(20, 5000))
        if status == "Completed":
            enrollment_actual = int(enrollment_intended * rng.uniform(0.4, 1.2))
        else:
            enrollment_actual = int(enrollment_intended * rng.uniform(0.0, 0.8))

        rows.append({
            "nct_id": nct_id,
            "study_start_date": study_start,
            "primary_completion_date": primary_completion,
            "overall_status": status,
            "phase": phase,
            "enrollment_intended": enrollment_intended,
            "enrollment_actual": enrollment_actual,
            "intervention_type": intervention_type,
            "conditions": condition,
            "search_term_match": term,
            "sponsor": rng.choice(["PharmaCo A", "BiotechB", "MedDeviceC"]),
            "official_title": f"Study of {condition} with {intervention_type}",
            "study_type": "INTERVENTIONAL",
            "allocation": rng.choice(["Randomized", "Non-Randomized"]),
            "intervention_model": rng.choice(["Parallel", "Crossover", "Sequential"]),
            "primary_purpose": rng.choice(["Treatment", "Prevention", "Supportive Care"]),
            "masks": rng.choice(["None", "Single", "Double"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=ClinicalTrialsSchema.required_columns)
    return df


def _build_client() -> httpx.Client:
    headers: dict[str, Any] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    email = cfg().get("paths", {}).get("clinicaltrials_email")
    import os
    env_email = os.environ.get("CLINICALTRIALS_EMAIL", "")
    final_email = email or env_email
    if final_email:
        headers["From"] = final_email
    return httpx.Client(base_url="https://clinicaltrials.gov/api/v2", headers=headers)


def _fetch_clinical_trials(
    *,
    search_terms: Iterable[str],
    max_studies: int,
    force_mock: bool = False,
) -> pd.DataFrame:
    """Query ClinicalTrials.gov v2 and return a normalized studies frame."""
    terms = list(search_terms)
    if not terms:
        return _mock_trials(search_terms=[], max_studies=max_studies, seed=_seed_from_config())

    if force_mock:
        return _mock_trials(search_terms=terms, max_studies=max_studies, seed=_seed_from_config())

    client: httpx.Client | None = None
    rows: list[dict[str, Any]] = []
    try:
        client = _build_client()
        retry_config = RetryConfig.from_config()
        per_page = 100
        fetched = 0
        for term in terms:
            if fetched >= max_studies:
                break
            page_token: str | None = None
            while fetched < max_studies:
                params: dict[str, Any] = {
                    "query.cond": term,
                    "query.term": term,
                    "fields": (
                        "nctId,overallStatus,phase,"
                        "startDatePrimaryCompletionDate,"
                        "enrollmentCount,interventionType,conditions"
                    ),
                    "pageSize": per_page,
                    "count": False,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = rate_limited_request(
                    client,
                    method="GET",
                    url="/studies",
                    params=params,
                    request_label=f"clinicaltrials_{term}",
                    retry_config=retry_config,
                )
                body = response.json()
                studies = body.get("studies", [])
                for study in studies:
                    protocol = study.get("protocolSection", {})
                    identification = protocol.get("identificationModule", {})
                    status_module = protocol.get("statusModule", {})
                    design_module = protocol.get("designModule", {})

                    nct_id = str(identification.get("nctId", ""))
                    if not nct_id:
                        continue

                    try:
                        start_date = pd.Timestamp(
                            identification.get("orgStudyIdInfo", {}).get("orgStudyId", "")
                        )
                    except Exception:
                        start_date = pd.NaT

                    completion_text = status_module.get("primaryCompletionDateStruct", {})
                    completion = pd.to_datetime(
                        completion_text.get("date", ""), errors="coerce"
                    )
                    status_text = status_module.get("overallStatus", "")

                    phase_info = design_module.get("phases", [])
                    phase = ",".join(phase_info) if phase_info else ""

                    enroll_count = (
                        study.get("protocolSection", {})
                        .get("designModule", {})
                        .get("enrollmentInfo", {})
                    )
                    enrollment_intended = int(enroll_count.get("count", 0) or 0)
                    enrollment_actual = enrollment_intended

                    intervention_info = design_module.get("interventionModelInfo", {})
                    intervention_type = intervention_info.get("type", "")

                    conditions_list = (
                        protocol.get("conditionsModule", {}).get("conditions", [])
                    )
                    conditions = ",".join(conditions_list or [])

                    rows.append({
                        "nct_id": nct_id,
                        "study_start_date": start_date,
                        "primary_completion_date": completion,
                        "overall_status": status_text,
                        "phase": phase,
                        "enrollment_intended": enrollment_intended,
                        "enrollment_actual": enrollment_actual,
                        "intervention_type": intervention_type,
                        "conditions": conditions,
                        "search_term_match": term,
                        "sponsor": "",
                        "official_title": "",
                        "study_type": "INTERVENTIONAL",
                        "allocation": "",
                        "intervention_model": "",
                        "primary_purpose": "",
                        "masks": "",
                    })

                    fetched += 1
                    if fetched >= max_studies:
                        break

                page_token = body.get("nextPageToken")
                if not page_token:
                    break
                # polite delay between pages
                time.sleep(1.0)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=ClinicalTrialsSchema.required_columns)
        return df
    except IngestionError:
        raise
    except Exception as exc:
        logger.warning(
            "ClinicalTrials.gov live fetch failed; falling back to mock: %s", exc
        )
        return _mock_trials(search_terms=terms, max_studies=max_studies, seed=_seed_from_config())
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()


def load_clinical_trials(*, force_mock: bool = False) -> pd.DataFrame:
    """Load ClinicalTrials.gov studies and normalize them for pharma enrollment velocity.

    Returns a DataFrame conforming to ``ClinicalTrialsSchema``. When the live fetch
    fails, deterministic mock data is used and a WARNING is logged.
    """
    r = cfg()
    terms = list(
        r.get("acquisition", {})
        .get("clinicaltrials", {})
        .get("search_terms", ["cardiovascular", "oncology", "diabetes"])
    )
    max_studies = int(
        r.get("acquisition", {}).get("clinicaltrials", {}).get("max_studies", 500)
    )
    df = _fetch_clinical_trials(
        search_terms=terms, max_studies=max_studies, force_mock=force_mock
    )
    return ClinicalTrialsSchema().validate(df, dataset_name="clinicaltrials")
