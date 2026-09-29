"""Unit tests for clinical feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest
from healthrisk_ai.processing.clinical_features import (
    ClinicalFeatureInputs,
    build_clinical_matrix,
    charlson_comorbidity_index,
    cms_hcc_condition_flags,
    elixhauser_comorbidity_index,
    lab_trajectory_stats,
    polypharmacy_score,
)
from healthrisk_ai.processing.resources import _lookup_conditions, load_comorbidity_code_set

from healthrisk_ai.exceptions import FeatureEngineeringError, ValidationError


class TestClinicalFeatureInputs:
    def test_validate_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"patient_id": [1]})
        with pytest.raises(ValidationError, match="missing columns"):
            ClinicalFeatureInputs.validate_input(df)

    def test_validate_empty_events_raises(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": pd.Series([], dtype=int),
                "event_date": pd.Series([], dtype="datetime64[ns]"),
                "event_type": pd.Series([], dtype=str),
                "code": pd.Series([], dtype=str),
                "value": pd.Series([], dtype=float),
                "cms_hcc_age_bucket": pd.Series([], dtype=str),
                "readmit_30d_flag": pd.Series([], dtype=int),
                "discharge_disposition": pd.Series([], dtype=str),
                "icd_10_code": pd.Series([], dtype=str),
            }
        )
        with pytest.raises(ValidationError, match="empty events frame"):
            ClinicalFeatureInputs.validate_input(df)


class TestCharlsonComorbidityIndex:
    def test_no_code_column_raises(self) -> None:
        df = pd.DataFrame({"patient_id": [1], "event_date": [pd.Timestamp("2020-01-01")]})
        with pytest.raises(FeatureEngineeringError, match="missing code column"):
            charlson_comorbidity_index(df)

    def test_sample_code_set_produces_score(self, mimic_events_frame: pd.DataFrame) -> None:
        frame = charlson_comorbidity_index(mimic_events_frame)
        assert "cci_score" in frame.columns
        assert frame["patient_id"].nunique() == mimic_events_frame["patient_id"].nunique()

    def test_code_set_has_expected_conditions(self) -> None:
        cs = load_comorbidity_code_set(None, name="charlson")
        assert "myocardial_infarction" in cs.condition_keys()

    def test_resource_lookup_mitral_infaction(self) -> None:
        cs = load_comorbidity_code_set(None, name="charlson")
        flags = _lookup_conditions("I21.0", cs.as_dict())
        assert flags["myocardial_infarction"] is True


class TestElixhauserComorbidityIndex:
    def test_separate_code_set(self) -> None:
        cs = load_comorbidity_code_set(None, name="elixhauser")
        assert cs.name == "elixhauser"
        keys = cs.condition_keys()
        # Elixhauser sample includes psychoses etc.
        assert "psychoses" in keys or "fluid_electrolyte_disorder" in keys

    def test_elixhauser_count_non_negative(self, mimic_events_frame: pd.DataFrame) -> None:
        frame = elixhauser_comorbidity_index(mimic_events_frame)
        assert "elixhauser_count" in frame.columns
        assert (frame["elixhauser_count"] >= 0).all()


class TestPolypharmacyScore:
    def test_no_patient_column_raises(self) -> None:
        df = pd.DataFrame({"event_date": [pd.Timestamp("2020-01-01")]})
        with pytest.raises(FeatureEngineeringError, match="missing patient column"):
            polypharmacy_score(df)

    def test_polypharmacy_flag_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": [1, 1, 1, 2],
                "event_date": [pd.Timestamp("2020-01-01")] * 4,
                "event_type": ["medication"] * 4,
                "code": ["A", "B", "C", "D"],
                "value": [1.0, 2.0, 3.0, 4.0],
                "cms_hcc_age_bucket": ["age_65_69"] * 4,
            }
        )
        out = polypharmacy_score(df, threshold=2)
        assert out.loc[out["patient_id"] == 1, "polypharmacy_flag"].iloc[0] is True
        assert out.loc[out["patient_id"] == 2, "polypharmacy_flag"].iloc[0] is False


class TestCmsHccConditionFlags:
    def test_returns_hcc_columns(self, mimic_events_frame: pd.DataFrame) -> None:
        frame = cms_hcc_condition_flags(mimic_events_frame)
        assert "patient_id" in frame.columns
        assert any(col.startswith("HCC_") for col in frame.columns)


class TestLabTrajectoryStats:
    def test_requires_datetime_column(self) -> None:
        df = pd.DataFrame({
            "patient_id": [1],
            "event_date": ["2020-01-01"],
            "event_type": ["lab"],
            "code": ["Glucose"],
            "value": [100.0],
        })
        with pytest.raises(FeatureEngineeringError, match="requires datetime64"):
            lab_trajectory_stats(df)

    def test_empty_lab_events_returns_empty(self, mimic_events_frame: pd.DataFrame) -> None:
        df = pd.DataFrame({
            "patient_id": [1],
            "event_date": pd.to_datetime([pd.Timestamp("2020-01-01")]),
            "event_type": ["admission"],
            "code": ["I21.0"],
            "value": [0.0],
            "cms_hcc_age_bucket": ["age_65_69"],
        })
        out = lab_trajectory_stats(df)
        assert out.empty


class TestBuildClinicalMatrix:
    def test_returns_one_row_per_patient(self, mimic_events_frame: pd.DataFrame) -> None:
        matrix = build_clinical_matrix(mimic_events_frame)
        assert matrix["patient_id"].nunique() == mimic_events_frame["patient_id"].nunique()

    def test_includes_age_bucket_columns(self, mimic_events_frame: pd.DataFrame) -> None:
        matrix = build_clinical_matrix(mimic_events_frame)
        assert "cms_hcc_age_bucket" in matrix.columns
        assert "cms_hcc_age_bucket_idx" in matrix.columns


class TestResourceLoader:
    def test_loads_external_yaml(self, sample_comorbidity_resource_path: str) -> None:
        cs = load_comorbidity_code_set(sample_comorbidity_resource_path, name="charlson")
        assert cs.condition_keys() == ["myocardial_infarction", "congestive_heart_failure"]

    def test_missing_resource_raises_file_not_found(self, tmp_path: str) -> None:
        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="Comorbidity resource not found"):
            load_comorbidity_code_set(missing, name="charlson")

    def test_loads_external_json(self, tmp_path: str) -> None:
        resource = tmp_path / "sample.json"
        resource.write_text(
            '{"version":"1.0.0","conditions":{"test_condition":["I21"]}}',
            encoding="utf-8",
        )
        cs = load_comorbidity_code_set(str(resource), name="charlson")
        assert cs.condition_keys() == ["test_condition"]
