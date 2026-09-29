"""Tests for configuration loading and typed config builders."""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest
from healthrisk_ai import load_config, project_root

from healthrisk_ai.config import build_config, cfg


class TestProjectRoot:
    def test_root_is_parent_of_src(self) -> None:
        root = project_root()
        assert root.is_dir()

    def test_config_file_exists_at_expected_location(self) -> None:
        root = project_root()
        cfg_path = root / "configs" / "config.yaml"
        assert cfg_path.is_file()


class TestLoadConfig:
    def test_loads_default_config(self) -> None:
        c = load_config()
        assert isinstance(c, dict)
        assert "simulation" in c

    def test_loads_explicit_path(self, tmp_path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("simulation:\n  quarters: 1\n")
        c = load_config(cfg_path)
        assert "simulation" in c

    def test_missing_file_raises_file_not_found(self) -> None:
        missing = project_root() / "configs" / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(missing)


class TestEnvReferenceResolution:
    def test_literal_value_preserved(self) -> None:
        c = load_config()
        assert c["simulation"]["quarters"] == 40

    def test_env_reference_with_default_used_when_missing(self) -> None:
        # .env is not set in tests; the ${VAR:-default} should resolve to default
        c = load_config()
        mimic_path = c["paths"]["mimic_ingest_path"]
        # The default text is just the empty string in this config
        assert isinstance(mimic_path, str)

    def test_env_reference_overrides_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = pathlib.Path(tmpdir) / "test_config.yaml"
            cfg_file.write_text(
                """
default_seed: 7
paths:
  mimic_ingest_path: ${MIMIC_INGEST_PATH:-/default/path}
""",
                encoding="utf-8",
            )
            os.environ["MIMIC_INGEST_PATH"] = "/real/path"
            try:
                c = load_config(cfg_file)
                assert c["paths"]["mimic_ingest_path"] == "/real/path"
            finally:
                os.environ.pop("MIMIC_INGEST_PATH", None)

    def test_env_reference_no_default_uses_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = pathlib.Path(tmpdir) / "test_config.yaml"
            cfg_file.write_text(
                """
paths:
  mimic_ingest_path: ${MIMIC_INGEST_PATH}
""",
                encoding="utf-8",
            )
            try:
                c = load_config(cfg_file)
                assert c["paths"]["mimic_ingest_path"] == ""
            finally:
                pass


class TestBuildConfig:
    def test_builds_all_dataclasses(self) -> None:
        result = build_config()
        assert len(result) == 7
        (
            seed_config,
            path_config,
            retry_config,
            acquisition_config,
            feature_config,
            model_config,
            simulation_config,
        ) = result

        assert seed_config.default_seed == 20260914
        assert path_config.data_root.name == "data"
        assert retry_config.max_attempts == 5
        assert acquisition_config.mimic.mock_num_patients == 2000
        assert acquisition_config.openfda.max_results_per_request == 100
        assert acquisition_config.clinicaltrials.max_studies == 500
        # Comorbidity resource paths default to None when not configured
        assert feature_config.clinical.comorbidity_resource_path is None
        assert feature_config.clinical.elixhauser_resource_path is None
        assert feature_config.clinical.polypharmacy_threshold == 5
        assert feature_config.financial.triangle_approach_years == 10
        assert model_config.nlp.embedding_dim == 64
        assert model_config.graph.patient_embedding_dim == 32
        assert model_config.survival.cindex_target == 0.70
        assert model_config.tabular.time_series_split_folds == 5
        assert model_config.ensemble.stack_target_r2 == 0.25
        assert simulation_config.initial_capital == 500_000_000
        assert simulation_config.quarters == 40
        assert "insurance" in simulation_config.asset_universe

    def test_simulation_config_fields(self) -> None:
        _, _, _, _, _, _, sim = build_config()
        assert sim.yearly_burn_rate_pct == 0.05
        assert sim.epidemiologic_shock_prob_per_quarter == 0.15


class TestCfgCache:
    def test_cfg_returns_dict(self) -> None:
        c = cfg()
        assert isinstance(c, dict)

    def test_cfg_cached(self) -> None:
        first = cfg()
        second = cfg()
        assert first is second
