# HealthRisk AI / HealthRisk Lab — typed config contract
#
# This module converts the raw YAML config into typed dataclasses that the rest
# of the codebase can import without touching YAML parsing logic.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from healthrisk_ai import load_config

_CFG: dict[str, Any] | None = None


def cfg() -> dict[str, Any]:
    """Lazily load the resolved config dictionary.

    Intended for use by modules that need dynamic config at runtime without
    importing YAML parsing logic themselves.
    """
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


@dataclass(frozen=True)
class SeedConfig:
    default_seed: int


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    cache_root: Path
    models_root: Path
    artifacts_root: Path
    mimic_ingest_path: str | None


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    base_backoff_seconds: float
    max_backoff_seconds: float
    rate_limit_per_second: float


@dataclass(frozen=True)
class AcquisitionConfig:
    mimic: MimicAcquisitionConfig
    openfda: OpenFdaAcquisitionConfig
    clinicaltrials: ClinicaltrialsAcquisitionConfig


@dataclass(frozen=True)
class MimicAcquisitionConfig:
    mock_num_patients: int
    event_date_range_start: str
    event_date_range_end: str


@dataclass(frozen=True)
class OpenFdaAcquisitionConfig:
    adverse_event_start: str
    adverse_event_end: str
    max_results_per_request: int


@dataclass(frozen=True)
class ClinicaltrialsAcquisitionConfig:
    search_terms: list[str]
    max_studies: int


@dataclass(frozen=True)
class FeatureConfig:
    clinical: ClinicalFeatureConfig
    financial: FinancialFeatureConfig


@dataclass(frozen=True)
class ClinicalFeatureConfig:
    comorbidity_resource_path: str | None
    elixhauser_resource_path: str | None
    compute_comorbidity_indices: bool
    compute_polypharmacy_score: bool
    compute_lab_trajectories: bool
    lab_window_days: int
    polypharmacy_threshold: int


@dataclass(frozen=True)
class FinancialFeatureConfig:
    compute_cash_metrics: bool
    compute_claim_triangles: bool
    compute_cmi_signals: bool
    triangle_approach_years: int


@dataclass(frozen=True)
class ModelConfig:
    nlp: NlpModelConfig
    graph: GraphModelConfig
    survival: SurvivalModelConfig
    tabular: TabularModelConfig
    ensemble: EnsembleModelConfig


@dataclass(frozen=True)
class NlpModelConfig:
    model_name: str
    embedding_dim: int
    max_seq_length: int
    device: str


@dataclass(frozen=True)
class GraphModelConfig:
    model_name: str
    patient_embedding_dim: int
    num_heads: int
    dropout: float


@dataclass(frozen=True)
class SurvivalModelConfig:
    deep_surv_hidden: list[int]
    cindex_target: float
    use_deepdeephit: bool


@dataclass(frozen=True)
class TabularModelConfig:
    selection_threshold: float
    time_series_split_folds: int


@dataclass(frozen=True)
class EnsembleModelConfig:
    meta_learner: str
    stack_target_r2: float


@dataclass(frozen=True)
class SimulationConfig:
    initial_capital: int
    quarters: int
    asset_universe: list[str]
    yearly_burn_rate_pct: float
    epidemiologic_shock_prob_per_quarter: float
    fda_warning_prob_per_quarter: float
    cms_rate_cut_prob_per_quarter: float


def build_config() -> tuple[
    SeedConfig,
    PathConfig,
    RetryConfig,
    AcquisitionConfig,
    FeatureConfig,
    ModelConfig,
    SimulationConfig,
]:
    """Build all typed config objects from the raw YAML once.

    This function is the sanctioned entry point for config access outside of the
    `healthrisk_ai.config` namespace.
    """
    raw = cfg()

    seed_config = SeedConfig(default_seed=raw.get("default_seed", 20260914))

    paths = raw.get("paths", {})
    path_config = PathConfig(
        data_root=Path(paths.get("data_root", "./data")),
        cache_root=Path(paths.get("cache_root", "./data/cache")),
        models_root=Path(paths.get("models_root", "./models")),
        artifacts_root=Path(paths.get("artifacts_root", "./models/artifacts")),
        mimic_ingest_path=str(paths.get("mimic_ingest_path") or ""),
    )

    retry = raw.get("retry", {})
    retry_config = RetryConfig(
        max_attempts=int(retry.get("max_attempts", 5)),
        base_backoff_seconds=float(retry.get("base_backoff_seconds", 1.0)),
        max_backoff_seconds=float(retry.get("max_backoff_seconds", 32.0)),
        rate_limit_per_second=float(retry.get("rate_limit_per_second", 2.0)),
    )

    acq = raw.get("acquisition", {})
    mimic = acq.get("mimic", {})
    openfda = acq.get("openfda", {})
    clinicaltrials = acq.get("clinicaltrials", {})
    acquisition_config = AcquisitionConfig(
        mimic=MimicAcquisitionConfig(
            mock_num_patients=int(mimic.get("mock_num_patients", 2000)),
            event_date_range_start=str(mimic.get("event_date_range", {}).get("start", "2010-01-01")),
            event_date_range_end=str(mimic.get("event_date_range", {}).get("end", "2022-12-31")),
        ),
        openfda=OpenFdaAcquisitionConfig(
            adverse_event_start=str(openfda.get("adverse_event_start", "2017-01-01")),
            adverse_event_end=str(openfda.get("adverse_event_end", "2026-01-01")),
            max_results_per_request=int(openfda.get("max_results_per_request", 100)),
        ),
        clinicaltrials=ClinicaltrialsAcquisitionConfig(
            search_terms=list(clinicaltrials.get("search_terms", ["cardiovascular", "oncology", "diabetes"])),
            max_studies=int(clinicaltrials.get("max_studies", 500)),
        ),
    )

    features = raw.get("features", {})
    clinical = features.get("clinical", {})
    financial = features.get("financial", {})
    feature_config = FeatureConfig(
        clinical=ClinicalFeatureConfig(
            comorbidity_resource_path=str(clinical.get("comorbidity_resource_path") or "") or None,
            elixhauser_resource_path=str(clinical.get("elixhauser_resource_path") or "") or None,
            compute_comorbidity_indices=bool(clinical.get("compute_comorbidity_indices", True)),
            compute_polypharmacy_score=bool(clinical.get("compute_polypharmacy_score", True)),
            compute_lab_trajectories=bool(clinical.get("compute_lab_trajectories", True)),
            lab_window_days=int(clinical.get("lab_window_days", 365)),
            polypharmacy_threshold=int(clinical.get("polypharmacy_threshold", 5)),
        ),
        financial=FinancialFeatureConfig(
            compute_cash_metrics=bool(financial.get("compute_cash_metrics", True)),
            compute_claim_triangles=bool(financial.get("compute_claim_triangles", True)),
            compute_cmi_signals=bool(financial.get("compute_cmi_signals", True)),
            triangle_approach_years=int(financial.get("triangle_approach_years", 10)),
        ),
    )

    models = raw.get("models", {})
    nlp = models.get("nlp", {})
    graph = models.get("graph", {})
    survival = models.get("survival", {})
    tabular = models.get("tabular", {})
    ensemble = models.get("ensemble", {})
    model_config = ModelConfig(
        nlp=NlpModelConfig(
            model_name=str(nlp.get("model_name", "emilyalsentzer/Bio_ClinicalBERT")),
            embedding_dim=int(nlp.get("embedding_dim", 64)),
            max_seq_length=int(nlp.get("max_seq_length", 512)),
            device=str(nlp.get("device", "auto")),
        ),
        graph=GraphModelConfig(
            model_name=str(graph.get("model_name", "gatv2_hetero")),
            patient_embedding_dim=int(graph.get("patient_embedding_dim", 32)),
            num_heads=int(graph.get("num_heads", 4)),
            dropout=float(graph.get("dropout", 0.2)),
        ),
        survival=SurvivalModelConfig(
            deep_surv_hidden=list(survival.get("deep_surv_hidden", [128, 64, 32])),
            cindex_target=float(survival.get("cindex_target", 0.70)),
            use_deepdeephit=bool(survival.get("use_deepdeephit", True)),
        ),
        tabular=TabularModelConfig(
            selection_threshold=float(tabular.get("selection_threshold", 0.01)),
            time_series_split_folds=int(tabular.get("time_series_split_folds", 5)),
        ),
        ensemble=EnsembleModelConfig(
            meta_learner=str(ensemble.get("meta_learner", "ridge")),
            stack_target_r2=float(ensemble.get("stack_target_r2", 0.25)),
        ),
    )

    sim = raw.get("simulation", {})
    simulation_config = SimulationConfig(
        initial_capital=int(sim.get("initial_capital", 500_000_000)),
        quarters=int(sim.get("quarters", 40)),
        asset_universe=list(
            sim.get(
                "asset_universe",
                ["insurance", "hospital_bonds", "pharma_equities", "credit_facilities"],
            )
        ),
        yearly_burn_rate_pct=float(sim.get("yearly_burn_rate_pct", 0.05)),
        epidemiologic_shock_prob_per_quarter=float(sim.get("epidemiologic_shock_prob_per_quarter", 0.15)),
        fda_warning_prob_per_quarter=float(sim.get("fda_warning_prob_per_quarter", 0.05)),
        cms_rate_cut_prob_per_quarter=float(sim.get("cms_rate_cut_prob_per_quarter", 0.10)),
    )

    return (
        seed_config,
        path_config,
        retry_config,
        acquisition_config,
        feature_config,
        model_config,
        simulation_config,
    )
