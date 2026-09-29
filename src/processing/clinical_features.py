"""Clinical feature engineering for the HealthRisk AI engine.

This module transforms the normalized MIMIC-IV events frame (see
`src/acquisition/mimic_iv.py`) into model-ready clinical feature matrices used
by the NLP, GNN, survival, and tabular submodels.

It implements:
- Charlson and Elixhauser comorbidity index computation from ICD-10 codes,
  backed by separate configurable code sets loaded from a resource file
  (YAML/JSON) at deployment time, with a built-in sample fallback so the
  pipeline and tests remain runnable without a provided resource
- CMS-HCC age-bucket and condition flag mapping
- Polypharmacy scoring from medication events
- Longitudinal lab trajectory calculators (slope, volatility, last value,
  time-to-abnormal)
- Patient-level aggregation with time-aware sorting to prevent leakage

All functions are deterministic given a seed, accept explicit column references,
and raise typed configuration or validation errors when the input frame is missing
required columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable

from healthrisk_ai.processing.resources import (
    COMorbidityCodeSet,
    _lookup_conditions,
    load_comorbidity_code_set,
)

from healthrisk_ai.config import cfg
from healthrisk_ai.exceptions import FeatureEngineeringError, ValidationError

logger = logging.getLogger("healthrisk_ai.processing.clinical_features")

__all__ = [
    "ClinicalFeatureInputs",
    "charlson_comorbidity_index",
    "elixhauser_comorbidity_index",
    "polypharmacy_score",
    "cms_hcc_condition_flags",
    "lab_trajectory_stats",
    "build_clinical_matrix",
    "COMorbidityCodeSet",
]


def _comorbidity_resource_path() -> Path | None:
    r = cfg()
    raw = r.get("features", {}).get("clinical", {})
    path_val = raw.get("comorbidity_resource_path")
    if path_val is None:
        return None
    return Path(path_val)


def _load_comorbidity_code_set(name: str) -> COMorbidityCodeSet:
    path = _comorbidity_resource_path()
    try:
        return load_comorbidity_code_set(path, name=name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load comorbidity resource for %s at %s; using sample fallback: %s",
            name,
            path,
            exc,
        )
        return load_comorbidity_code_set(None, name=name)


def charlson_comorbidity_index(
    events: pd.DataFrame,
    *,
    patient_col: Hashable = "patient_id",
    code_col: Hashable = "icd_10_code",
    weight_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute a Charlson Comorbidity Index per patient from ICD-10 codes.

    The index aggregates weighted comorbidity conditions observed in a patient's
    events. Weights follow the classic Charlson weighting scheme adapted for ICD-10
    where available; the code set is loaded from the configured resource (or its
    built-in sample fallback), and the Elixhauser code set is intentionally separate.

    Returns a frame with one row per patient containing:
    - cci_score
    - cci_<condition> flags for each condition present in the loaded code set.
    """
    if code_col not in events.columns:
        msg = f"events missing code column: {code_col}"
        raise FeatureEngineeringError(msg, context={"missing_column": code_col})
    ClinicalFeatureInputs.validate_input(events)

    code_set = _load_comorbidity_code_set("charlson")
    default_weights = {
        "myocardial_infarction": 1,
        "congestive_heart_failure": 1,
        "peripheral_vascular_disease": 1,
        "cerebrovascular_disease": 1,
        "dementia": 1,
        "chronic_pulmonary_disease": 1,
        "connective_tissue_disease": 1,
        "peptic_ulcer_disease": 1,
        "liver_mild": 1,
        "diabetes_uncomplicated": 1,
        "diabetes_complicated": 2,
        "paraplegia_and_hemiplegia": 1,
        "renal_disease": 1,
        "cancer": 1,
        "liver_severe": 2,
        "metastatic_solid_tumor": 2,
        "hiv": 1,
    }
    weights = weight_map if weight_map is not None else default_weights

    conditions = [c for c in code_set.condition_keys() if c in weights]
    patient_ids = events[patient_col].unique()
    rows: list[dict[str, Any]] = []

    for pid in patient_ids:
        patient_codes = events.loc[events[patient_col] == pid, code_col].dropna().tolist()
        condition_flags: dict[str, bool] = dict.fromkeys(conditions, False)
        for code in patient_codes:
            if not isinstance(code, str):
                continue
            flags = _lookup_conditions(code, code_set.as_dict())
            for c, present in flags.items():
                if present and c in condition_flags:
                    condition_flags[c] = True

        score = sum(weights[c] for c, present in condition_flags.items() if present)
        row = {"patient_id": pid, "cci_score": score}
        row.update({f"cci_{c}": int(condition_flags[c]) for c in conditions})
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return frame


def elixhauser_comorbidity_index(
    events: pd.DataFrame,
    *,
    patient_col: Hashable = "patient_id",
    code_col: Hashable = "icd_10_code",
) -> pd.DataFrame:
    """Compute an Elixhauser comorbidity count per patient from ICD-10 codes.

    The Elixhauser measure uses a separate comorbidity code set (loaded from the
    configured resource or its built-in sample fallback) and reports a simple count
    of distinct comorbidity categories present, plus a per-condition flag matrix.
    This is intended as a feature input rather than a clinical diagnostic; the full
    Quail/van Walraven weighting would be loaded later.

    Returns a frame with:
    - elixhauser_count
    - elixhauser_<condition> flags for each condition in the Elixhauser code set.
    """
    if code_col not in events.columns:
        msg = f"events missing code column: {code_col}"
        raise FeatureEngineeringError(msg, context={"missing_column": code_col})
    ClinicalFeatureInputs.validate_input(events)

    code_set = _load_comorbidity_code_set("elixhauser")
    conditions = list(code_set.condition_keys())

    patient_ids = events[patient_col].unique()
    rows: list[dict[str, Any]] = []

    for pid in patient_ids:
        patient_codes = events.loc[events[patient_col] == pid, code_col].dropna().tolist()
        condition_flags: dict[str, bool] = dict.fromkeys(conditions, False)
        for code in patient_codes:
            if not isinstance(code, str):
                continue
            flags = _lookup_conditions(code, code_set.as_dict())
            for c, present in flags.items():
                if present and c in condition_flags:
                    condition_flags[c] = True

        count = int(sum(condition_flags.values()))
        row = {"patient_id": pid, "elixhauser_count": count}
        row.update({f"elixhauser_{c}": int(condition_flags[c]) for c in conditions})
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return frame


def polypharmacy_score(
    events: pd.DataFrame,
    *,
    patient_col: Hashable = "patient_id",
    code_col: Hashable = "code",
    event_type_col: Hashable = "event_type",
    threshold: int | None = None,
) -> pd.DataFrame:
    """Compute a polypharmacy score per patient from medication events.

    Polypharmacy is defined as the count of distinct medication codes observed for
    the patient within the observation window. A flag is set when the count exceeds
    the configured threshold.

    By default, the threshold comes from `configs/config.yaml` features.clinical
    .polypharmacy_threshold; pass an explicit value to override.

    Returns:
        Per-patient frame with polypharmacy_count and polypharmacy_flag.
    """
    if patient_col not in events.columns:
        msg = f"polypharmacy_score missing patient column: {patient_col}"
        raise FeatureEngineeringError(msg, context={"missing_column": patient_col})
    ClinicalFeatureInputs.validate_input(events)
    if threshold is None:
        r = cfg()
        threshold = int(
            r.get("features", {}).get("clinical", {}).get("polypharmacy_threshold", 5)
        )

    meds = events[events[event_type_col] == "medication"]
    if meds.empty:
        return pd.DataFrame(
            {
                patient_col: events[patient_col].unique(),
                "polypharmacy_count": 0,
                # object dtype keeps the values plain Python bools.
                "polypharmacy_flag": pd.Series([False] * len(events[patient_col].unique()), dtype=object),
            }
        ).sort_values(patient_col).reset_index(drop=True)

    grouped = (
        meds.groupby(patient_col)[code_col]
        .nunique()
        .rename("polypharmacy_count")
        .reset_index()
    )
    # object dtype keeps the values plain Python bools so callers can rely on
    # `is True` / `is False` identity checks.
    grouped["polypharmacy_flag"] = pd.Series(
        [bool(count > threshold) for count in grouped["polypharmacy_count"]],
        dtype=object,
    )
    return grouped.sort_values(patient_col).reset_index(drop=True)


class ClinicalFeatureInputs:
    """Typed contract describing the columns expected by the clinical feature pipeline.

    This class is a runtime contract, not a Pydantic model, so it stays cheap to
    import across the package.
    """

    required_event_columns: list[str] = [
        # Core contract shared by every clinical feature function. Feature-specific
        # columns (icd_10_code, cms_hcc_age_bucket, readmit_30d_flag,
        # discharge_disposition) are validated by the individual functions that
        # actually consume them.
        "patient_id",
        "event_date",
        "event_type",
        "code",
        "value",
    ]
    required_patient_columns: list[str] = [
        "patient_id",
        "age_years",
    ]

    @classmethod
    def validate_input(
        cls, events: pd.DataFrame, patients: pd.DataFrame | None = None
    ) -> None:
        missing: set[str] = set(cls.required_event_columns) - set(events.columns)
        if missing:
            msg = f"Clinical feature inputs missing columns: {sorted(missing)}"
            raise ValidationError(msg, context={"missing_columns": sorted(missing)})
        if events.empty:
            msg = "Clinical feature inputs received an empty events frame"
            raise ValidationError(msg)
        if patients is not None:
            missing_patients = set(cls.required_patient_columns) - set(patients.columns)
            if missing_patients:
                msg = f"Patient demographics missing columns: {sorted(missing_patients)}"
                raise ValidationError(msg, context={"missing_columns": sorted(missing_patients)})


def cms_hcc_condition_flags(
    events: pd.DataFrame,
    *,
    patient_col: Hashable = "patient_id",
    code_col: Hashable = "icd_10_code",
) -> pd.DataFrame:
    """Map patient conditions to CMS-HCC flag columns for risk adjustment features.

    This function exposes a compact set of HCC-style flags derived from the pooled
    ICD-10 codes in the events frame. The flag set is intentionally aligned with the
    Charlson code set so the same code table drives both clinical indices and HCC
    style features.

    Returns:
        Per-patient frame with one flag column per HCC-aligned condition.
    """
    ClinicalFeatureInputs.validate_input(events)
    charlson_frame = charlson_comorbidity_index(
        events, patient_col=patient_col, code_col=code_col
    )
    hcc_map = {
        "cci_myocardial_infarction": "HCC_MI",
        "cci_congestive_heart_failure": "HCC_CHD",
        "cci_chronic_pulmonary_disease": "HCC_COPD",
        "cci_diabetes_uncomplicated": "HCC_DIABETES_UNCOMP",
        "cci_diabetes_complicated": "HCC_DIABETES_COMP",
        "cci_renal_disease": "HCC_RENAL",
        "cci_liver_severe": "HCC_LIVER",
        "cci_cancer": "HCC_CANCER",
        "cci_metastatic_solid_tumor": "HCC_METASTASIS",
        "cci_hiv": "HCC_HIV",
        "cci_dementia": "HCC_DEMENTIA",
        "cci_cerebrovascular_disease": "HCC_CEREBROVASC",
    }
    renamed = charlson_frame.rename(columns=hcc_map)
    hcc_cols = [c for c in hcc_map.values() if c in renamed.columns]
    return renamed[[patient_col] + hcc_cols].copy()


def lab_trajectory_stats(
    events: pd.DataFrame,
    *,
    patient_col: Hashable = "patient_id",
    event_date_col: Hashable = "event_date",
    code_col: Hashable = "code",
    value_col: Hashable = "value",
    window_days: int | None = None,
    reference_start_col: Hashable | None = None,
) -> pd.DataFrame:
    """Compute per-patient, per-lab trajectory statistics.

    Trajectories are computed within a sliding window anchored to each patient's
    first lab event. For each lab test code, we compute:
    - first_value
    - last_value
    - slope_per_day (least-squares regression of value on day index)
    - volatility (std of residuals)
    - trend_direction (rising/falling/stable)
    - abnormal_count (count of values outside a generic clinically-safe band)

    The generic safe band is a placeholder; real bands would be loaded per-test.

    Args:
        events: Normalized events frame containing lab events.
        patient_col: Patient identifier column.
        event_date_col: Date column (must be datetime-like).
        code_col: Lab test code column.
        value_col: Numeric lab measurement column.
        window_days: Optional window length overriding config.
        reference_start_col: Optional column with a per-patient start date anchor;
            if None, the first lab event date is used.

    Returns:
        Multi-index frame indexed by (patient_id, lab_code) with trajectory columns.
    """
    if event_date_col not in events.columns or not np.issubdtype(
        events[event_date_col].dtype, np.datetime64
    ):
        msg = f"lab_trajectory_stats requires datetime64 {event_date_col}"
        raise FeatureEngineeringError(msg, context={"column": event_date_col})
    if value_col not in events.columns:
        msg = f"lab_trajectory_stats missing value column: {value_col}"
        raise FeatureEngineeringError(msg, context={"missing_column": value_col})
    ClinicalFeatureInputs.validate_input(events)
    if window_days is None:
        r = cfg()
        window_days = int(
            r.get("features", {}).get("clinical", {}).get("lab_window_days", 365)
        )

    labs = events[events["event_type"] == "lab"].copy()
    if labs.empty:
        index: pd.MultiIndex = pd.MultiIndex.from_arrays(
            [[], []], names=[patient_col, code_col]
        )
        return pd.DataFrame(
            {
                "first_value": pd.Series(dtype=float),
                "last_value": pd.Series(dtype=float),
                "slope_per_day": pd.Series(dtype=float),
                "volatility": pd.Series(dtype=float),
                "trend_direction": pd.Series(dtype=str),
                "abnormal_count": pd.Series(dtype=int),
            },
            index=index,
        )

    lab_start = labs[event_date_col].min()
    labs["day_index"] = (labs[event_date_col] - lab_start).dt.days.astype(int)

    safe_bands: dict[str, tuple[float, float]] = {
        "Glucose": (70.0, 200.0),
        "Creatinine": (0.6, 1.5),
        "Na": (135.0, 145.0),
        "K": (3.5, 5.0),
        "WBC": (4.0, 11.0),
        "Hgb": (12.0, 17.5),
        "Platelets": (150.0, 400.0),
    }
    default_band: tuple[float | None, float | None] = (None, None)

    rows: list[dict[str, Any]] = []
    for pid, patient in labs.groupby(patient_col):
        for lab_code, group in patient.groupby(code_col):
            group = group.sort_values(event_date_col)
            if group["day_index"].max() - group["day_index"].min() > window_days:
                group = group[
                    group["day_index"] <= group["day_index"].min() + window_days
                ]
            if len(group) < 2:
                first_val = float(group[value_col].iloc[0]) if len(group) else float("nan")
                last_val = float(group[value_col].iloc[-1]) if len(group) else float("nan")
                band = safe_bands.get(lab_code, default_band)
                low, high = band
                abnormal = 0
                if low is not None and low is not None:
                    abnormal += int((group[value_col] < low).sum())
                if high is not None:
                    abnormal += int((group[value_col] > high).sum())
                rows.append(
                    {
                        patient_col: pid,
                        code_col: lab_code,
                        "first_value": first_val,
                        "last_value": last_val,
                        "slope_per_day": float("nan"),
                        "volatility": float("nan"),
                        "trend_direction": "unknown",
                        "abnormal_count": abnormal,
                    }
                )
                continue

            x = group["day_index"].to_numpy(dtype=float)
            y = group[value_col].to_numpy(dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            intercept = float(np.polyfit(x, y, 1)[1])
            predicted = intercept + slope * x
            residuals = y - predicted
            volatility = (
                float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
            )
            trend = (
                "rising"
                if slope > 1e-6
                else ("falling" if slope < -1e-6 else "stable")
            )
            band = safe_bands.get(lab_code, default_band)
            low, high = band
            abnormal = 0
            if low is not None:
                abnormal += int((y < low).sum())
            if high is not None:
                abnormal += int((y > high).sum())
            rows.append(
                {
                    patient_col: pid,
                    code_col: lab_code,
                    "first_value": float(y[0]),
                    "last_value": float(y[-1]),
                    "slope_per_day": slope,
                    "volatility": volatility,
                    "trend_direction": trend,
                    "abnormal_count": abnormal,
                }
            )

    frame = (
        pd.DataFrame(rows)
        .set_index([patient_col, code_col])
        .sort_index()
    )
    return frame


def build_clinical_matrix(
    events: pd.DataFrame,
    patients: pd.DataFrame | None = None,
    *,
    include_comorbidity: bool | None = None,
    include_polypharmacy: bool | None = None,
    include_lab_trajectories: bool | None = None,
    include_cms_hcc: bool | None = None,
) -> pd.DataFrame:
    """Build a single model-ready clinical feature matrix per patient.

    This is the canonical entry point for clinical features. It assembles:
    - Demographics (age bucket dummies, sex)
    - CMS-HCC age bucket numeric proxy
    - Charlson and Elixhauser indices
    - Polypharmacy score and flag
    - Selected lab trajectory statistics (mean slope, mean volatility, mean
      abnormal count across all labs per patient)
    - CMS-HCC condition flags

    Time-aware sorting is enforced inside sub-functions; the returned matrix is
    suitable for downstream model ingestion.

    Args:
        events: Normalized MIMIC-IV events frame.
        patients: Optional patient demographics frame; generated from events if None.
        include_comorbidity: Whether to include comorbidity indices.
        include_polypharmacy: Whether to include polypharmacy features.
        include_lab_trajectories: Whether to include aggregated lab trajectory stats.
        include_cms_hcc: Whether to include CMS-HCC flags.

    Returns:
        Patient-level feature matrix with one row per patient.
    """
    ClinicalFeatureInputs.validate_input(events)
    r = cfg()
    features = r.get("features", {}).get("clinical", {})

    include_comorbidity = (
        include_comorbidity
        if include_comorbidity is not None
        else bool(features.get("compute_comorbidity_indices", True))
    )
    include_polypharmacy = (
        include_polypharmacy
        if include_polypharmacy is not None
        else bool(features.get("compute_polypharmacy_score", True))
    )
    include_lab_trajectories = (
        include_lab_trajectories
        if include_lab_trajectories is not None
        else bool(features.get("compute_lab_trajectories", True))
    )
    include_cms_hcc = (
        include_cms_hcc
        if include_cms_hcc is not None
        else bool(features.get("compute_comorbidity_indices", True))
    )

    patient_ids = events["patient_id"].unique()
    if patients is None:
        patients = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "age_years": events.groupby("patient_id")["age_years"].first().values,
            }
        )
    else:
        patients = patients[patients["patient_id"].isin(patient_ids)].copy()

    from healthrisk_ai.numerics import age_to_cms_hcc_bucket

    matrix = patients.copy()
    matrix["cms_hcc_age_num"] = matrix["age_years"]
    matrix["cms_hcc_age_bucket"] = matrix["age_years"].map(age_to_cms_hcc_bucket)
    matrix["cms_hcc_age_bucket_idx"] = matrix["cms_hcc_age_bucket"].map(
        {
            "age_0": 0,
            "age_1": 1,
            "age_2_4": 2,
            "age_5_9": 3,
            "age_10_14": 4,
            "age_15_19": 5,
            "age_20_24": 6,
            "age_25_29": 7,
            "age_30_34": 8,
            "age_35_39": 9,
            "age_40_44": 10,
            "age_45_49": 11,
            "age_50_54": 12,
            "age_55_59": 13,
            "age_60_64": 14,
            "age_65_69": 15,
            "age_70_74": 16,
            "age_75_79": 17,
            "age_80_84": 18,
            "age_85_plus": 19,
            "age_unknown": -1,
        }
    ).fillna(-1).astype(int)

    if "sex" in events.columns:
        sex_by_patient = events.groupby("patient_id")["sex"].first()
        matrix = matrix.set_index("patient_id").join(sex_by_patient).reset_index()
        matrix["sex_M"] = (matrix["sex"] == "M").astype(int)
    else:
        matrix["sex_M"] = 0

    if include_comorbidity:
        cci = charlson_comorbidity_index(events)
        matrix = matrix.merge(cci, on="patient_id", how="left")
        cci_cols = [c for c in cci.columns if c != "patient_id" and c.startswith("cci_")]
        for c in cci_cols:
            matrix[c] = matrix[c].fillna(0)

        elix = elixhauser_comorbidity_index(events)
        elix_cols = [c for c in elix.columns if c != "patient_id"]
        matrix = matrix.merge(elix, on="patient_id", how="left")
        for c in elix_cols:
            matrix[c] = matrix[c].fillna(0)

    if include_polypharmacy:
        pp = polypharmacy_score(events)
        matrix = matrix.merge(pp, on="patient_id", how="left")
        matrix["polypharmacy_count"] = matrix["polypharmacy_count"].fillna(0).astype(int)
        matrix["polypharmacy_flag"] = matrix["polypharmacy_flag"].infer_objects(copy=False).fillna(False).astype(bool)

    if include_lab_trajectories:
        lab_stats = lab_trajectory_stats(events)
        if not lab_stats.empty:
            agg = lab_stats.groupby(level="patient_id").agg(
                mean_slope=("slope_per_day", "mean"),
                mean_volatility=("volatility", "mean"),
                mean_abnormal_count=("abnormal_count", "mean"),
                lab_count=("slope_per_day", "size"),
            ).reset_index()
            matrix = matrix.merge(agg, on="patient_id", how="left")
            for c in [
                "mean_slope",
                "mean_volatility",
                "mean_abnormal_count",
                "lab_count",
            ]:
                matrix[c] = matrix[c].fillna(0)

    if include_cms_hcc:
        hcc = cms_hcc_condition_flags(events)
        matrix = matrix.merge(hcc, on="patient_id", how="left")
        hcc_cols = [c for c in hcc.columns if c != "patient_id"]
        for c in hcc_cols:
            matrix[c] = matrix[c].fillna(0)

    return matrix.sort_values("patient_id").reset_index(drop=True)
