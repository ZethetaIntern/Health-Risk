"""Resource loaders for clinical code sets used in clinical feature engineering.

This package provides a small, deterministic loader for ICD-10 code sets used by
the Charlson and Elixhauser comorbidity indices and by the CMS-HCC flag mapper.

Design:
- A YAML resource file is the canonical source at deployment time.
- A built-in sample resource is shipped so the pipeline and tests run without a
  provided file; the sample covers a representative code subset, not the full CMS
  code universe.
- Charlson and Elixhauser are intentionally separate code sets so each index can
  evolve independently and be swapped via configuration.
"""

from __future__ import annotations

import collections.abc
from pathlib import Path
from typing import Any

if False:
    _t = collections.abc

__all__ = [
    "COMorbidityCodeSet",
    "load_comorbidity_code_set",
    "DEFAULT_COMORBIDITY_RESOURCE",
]


class COMorbidityCodeSet:
    """A typed comorbidity code set for one index variant.

    Attributes:
        name: Human-readable name (e.g. "charlson", "elixhauser").
        conditions: Mapping from condition key to a sequence of ICD-10 codes (or
            code prefixes). Codes may be full codes like "I21" or prefix-like entries
            like "I21.*"; lookups normalize to the 3-character parent.
        version: Optional version label for the resource.
    """

    def __init__(
        self,
        *,
        name: str,
        conditions: collections.abc.Mapping[str, collections.abc.Sequence[str]],
        version: str | None = None,
    ) -> None:
        self.name = name
        self.conditions = conditions
        self.version = version or "sample"

    def as_dict(self) -> collections.abc.Mapping[str, collections.abc.Sequence[str]]:
        return dict(self.conditions)

    def condition_keys(self) -> collections.abc.Sequence[str]:
        return list(self.conditions.keys())


def _normalize_icd10_code(code: str) -> str:
    if not isinstance(code, str) or not code:
        return ""
    code = code.strip()
    code = code.split(".")[0]
    return code[:3].upper()


def _lookup_conditions(
    code: str,
    conditions: collections.abc.Mapping[str, collections.abc.Sequence[str]],
) -> dict[str, bool]:
    parent = _normalize_icd10_code(code)
    flags: dict[str, bool] = dict.fromkeys(conditions, False)
    if not parent:
        return flags
    for key, codes in conditions.items():
        for c in codes:
            c_norm = _normalize_icd10_code(c)
            if parent == c_norm or parent == c[:3] or c_norm.startswith(parent):
                flags[key] = True
                break
    return flags


def load_comorbidity_code_set(
    resource_path: Path | str | None = None,
    *,
    name: str = "charlson",
) -> COMorbidityCodeSet:
    """Load a comorbidity code set from a YAML/JSON resource.

    When a path is provided, the loader reads either a YAML or JSON file with a
    ``conditions`` mapping. When no path is provided, or the file is missing, a
    built-in sample resource is returned so the pipeline remains runnable.

    Args:
        resource_path: Optional path to a YAML or JSON resource file.
        name: Name used for the returned code set object.

    Returns:
        A COMorbidityCodeSet for the requested index variant.
    """
    if resource_path is not None:
        path = Path(resource_path)
        if not path.exists():
            raise FileNotFoundError(f"Comorbidity resource not found: {path}")

        try:
            import yaml
            with path.open("r", encoding="utf-8") as handle:
                payload: dict[str, Any] = yaml.safe_load(handle) or {}
            conditions = _load_conditions_from_payload(payload)
            version = payload.get("version") or payload.get("meta", {}).get("version")
            return COMorbidityCodeSet(name=name, conditions=conditions, version=version)
        except Exception as exc:
            # Be permissive: treat malformed resource as fallback without hiding the
            # error entirely in non-test environments.
            raise RuntimeError(f"Failed to load comorbidity resource {path}: {exc}") from exc

    return _default_sample_code_set(name=name)


def _load_conditions_from_payload(
    payload: collections.abc.Mapping[str, object],
) -> collections.abc.Mapping[str, collections.abc.Sequence[str]]:
    # Accept both {conditions: {key: [...]} } and top-level {key: [...]} shapes
    conditions = payload.get("conditions")
    if isinstance(conditions, dict):
        return {str(k): list(v) for k, v in conditions.items()}
    # Fall back to top-level mapping if no "conditions" key is present
    return {
        str(k): list(v)
        for k, v in payload.items()
        if isinstance(v, collections.abc.Sequence) and not isinstance(v, str)
    }


def _default_sample_code_set(name: str = "charlson") -> COMorbidityCodeSet:
    """Return the built-in sample comorbidity code set.

    The sample covers a representative set of ICD-10 condition groups. It is not
    exhaustive and is intended only to make the feature pipeline and tests runnable
    when no external resource is provided.

    Charlson and Elixhauser are intentionally separate code sets; this function
    returns different sample subsets depending on `name` so the two indices can
    evolve independently.
    """
    if name == "elixhauser":
        return _default_sample_elixhauser_code_set()
    return _default_sample_charlson_code_set()


def _default_sample_charlson_code_set() -> COMorbidityCodeSet:
    """Built-in sample Charlson comorbidity code set."""
    sample = {
        "myocardial_infarction": [
            "I21", "I22", "I23", "I24", "I25.0", "I25.1", "I25.2", "I25.5",
            "I25.6", "I25.7", "I25.8",
        ],
        "congestive_heart_failure": [
            "I50", "I11.0", "I13.0", "I13.2", "I97.13", "I97.130",
        ],
        "peripheral_vascular_disease": [
            "I70", "I71", "I72", "I73", "I74", "I75", "I77", "I78", "I79",
        ],
        "cerebrovascular_disease": [
            "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69",
        ],
        "dementia": [
            "F00", "F01", "F02", "F03", "F05.1", "G30", "G31.0", "G31.1",
        ],
        "chronic_pulmonary_disease": [
            "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60",
            "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J68", "J70",
        ],
        "connective_tissue_disease": [
            "M05", "M06", "M07", "M08", "M09", "M30", "M31", "M32", "M33",
            "M34", "M35", "M36",
        ],
        "peptic_ulcer_disease": [
            "K25", "K26", "K27", "K28",
        ],
        "liver_mild": [
            "B18", "B19", "K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77",
        ],
        "diabetes_uncomplicated": [
            "E10", "E11", "E12", "E13", "E14", "E23.1", "E35.0",
        ],
        "diabetes_complicated": [
            "E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8",
            "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8",
            "E12.2", "E12.3", "E12.4", "E12.5", "E12.6", "E12.7", "E12.8",
            "E13.2", "E13.3", "E13.4", "E13.5", "E13.6", "E13.7", "E13.8",
            "E14.2", "E14.3", "E14.4", "E14.5", "E14.6", "E14.7", "E14.8",
        ],
        "paraplegia_and_hemiplegia": [
            "G81", "G82", "G83", "G04.1", "G04.2", "G04.3", "G80.0", "G80.1",
            "G80.2", "G80.3", "G80.4", "G80.8", "G80.9", "G82.5", "G83.0",
            "G83.1", "G83.2", "G83.3", "G83.4", "G83.5", "G83.6", "G83.7",
            "G83.8", "G83.9",
        ],
        "renal_disease": [
            "N00", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08",
            "N09", "N10", "N11", "N12", "N13", "N14", "N15", "N16", "N17",
            "N18", "N19", "N20", "N25",
        ],
        "cancer": [
            "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
            "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
            "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
            "C27", "C28", "C29", "C30", "C31", "C32", "C33", "C34", "C35",
            "C36", "C37", "C38", "C39", "C40", "C41", "C42", "C43", "C44",
        ],
        "liver_severe": [
            "K70.0", "K70.1", "K70.2", "K70.3", "K70.4", "K70.5", "K70.6",
            "K70.7", "K70.9", "K71.0", "K71.1", "K71.2", "K71.3", "K71.4",
            "K71.5", "K71.6", "K71.7", "K71.8", "K71.9", "K72.0", "K72.1",
            "K72.2", "K72.3", "K72.4", "K72.5", "K72.6", "K72.9", "K73.0",
            "K73.1", "K73.2", "K73.3", "K73.4", "K73.5", "K73.6", "K73.8",
            "K73.9", "K74.0", "K74.1", "K74.2", "K74.3", "K74.4", "K74.5",
            "K74.6", "K74.7", "K74.8", "K74.9",
        ],
        "metastatic_solid_tumor": [
            "C77", "C78", "C79", "C80",
        ],
        "hiv": [
            "B20", "B21", "B22", "B23", "B24",
        ],
    }
    return COMorbidityCodeSet(name="charlson", conditions=sample, version="sample")


def _default_sample_elixhauser_code_set() -> COMorbidityCodeSet:
    """Built-in sample Elixhauser comorbidity code set.

    Elixhauser uses a broader condition set than Charlson. The sample here is a
    representative subset aligned with the same code universe so the two indices
    can be compared and benchmarked independently.
    """
    sample = {
        "myocardial_infarction": [
            "I21", "I22", "I23", "I24", "I25.0", "I25.1", "I25.2", "I25.5",
            "I25.6", "I25.7", "I25.8",
        ],
        "congestive_heart_failure": [
            "I50", "I11.0", "I13.0", "I13.2", "I97.13", "I97.130",
        ],
        "peripheral_vascular_disease": [
            "I70", "I71", "I72", "I73", "I74", "I75", "I77", "I78", "I79",
        ],
        "cerebrovascular_disease": [
            "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69",
        ],
        "dementia": [
            "F00", "F01", "F02", "F03", "F05.1", "G30", "G31.0", "G31.1",
        ],
        "chronic_pulmonary_disease": [
            "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60",
            "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J68", "J70",
        ],
        "connective_tissue_disease": [
            "M05", "M06", "M07", "M08", "M09", "M30", "M31", "M32", "M33",
            "M34", "M35", "M36",
        ],
        "peptic_ulcer_disease": [
            "K25", "K26", "K27", "K28",
        ],
        "liver_mild": [
            "B18", "B19", "K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77",
        ],
        "diabetes_uncomplicated": [
            "E10", "E11", "E12", "E13", "E14", "E23.1", "E35.0",
        ],
        "diabetes_complicated": [
            "E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8",
            "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8",
            "E12.2", "E12.3", "E12.4", "E12.5", "E12.6", "E12.7", "E12.8",
            "E13.2", "E13.3", "E13.4", "E13.5", "E13.6", "E13.7", "E13.8",
            "E14.2", "E14.3", "E14.4", "E14.5", "E14.6", "E14.7", "E14.8",
        ],
        "paraplegia_and_hemiplegia": [
            "G81", "G82", "G83", "G04.1", "G04.2", "G04.3", "G80.0", "G80.1",
            "G80.2", "G80.3", "G80.4", "G80.8", "G80.9", "G82.5", "G83.0",
            "G83.1", "G83.2", "G83.3", "G83.4", "G83.5", "G83.6", "G83.7",
            "G83.8", "G83.9",
        ],
        "renal_disease": [
            "N00", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08",
            "N09", "N10", "N11", "N12", "N13", "N14", "N15", "N16", "N17",
            "N18", "N19", "N20", "N25",
        ],
        "cancer": [
            "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
            "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
            "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
            "C27", "C28", "C29", "C30", "C31", "C32", "C33", "C34", "C35",
            "C36", "C37", "C38", "C39", "C40", "C41", "C42", "C43", "C44",
        ],
        "liver_severe": [
            "K70.0", "K70.1", "K70.2", "K70.3", "K70.4", "K70.5", "K70.6",
            "K70.7", "K70.9", "K71.0", "K71.1", "K71.2", "K71.3", "K71.4",
            "K71.5", "K71.6", "K71.7", "K71.8", "K71.9", "K72.0", "K72.1",
            "K72.2", "K72.3", "K72.4", "K72.5", "K72.6", "K72.9", "K73.0",
            "K73.1", "K73.2", "K73.3", "K73.4", "K73.5", "K73.6", "K73.8",
            "K73.9", "K74.0", "K74.1", "K74.2", "K74.3", "K74.4", "K74.5",
            "K74.6", "K74.7", "K74.8", "K74.9",
        ],
        "metastatic_solid_tumor": [
            "C77", "C78", "C79", "C80",
        ],
        "hiv": [
            "B20", "B21", "B22", "B23", "B24",
        ],
        "psychoses": [
            "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28", "F29",
            "F30.2", "F31.2", "F31.3", "F31.4", "F31.5", "F31.6", "F31.7", "F31.8",
            "F31.9", "F32.3", "F33.3",
        ],
        "fluid_electrolyte_disorder": [
            "E26", "E27", "E28", "E29",
        ],
        "blood_loss_anemia": [
            "D50", "D51", "D52", "D53",
        ],
        "deficiency_anemia": [
            "D54", "D55", "D56", "D57", "D58", "D59",
        ],
        "alcohol_abuse": [
            "F10",
        ],
        "drug_abuse": [
            "F11",
        ],
    }
    return COMorbidityCodeSet(name="elixhauser", conditions=sample, version="sample")


DEFAULT_COMORBIDITY_RESOURCE: Path | None = None
