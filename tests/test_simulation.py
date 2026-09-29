"""Unit tests for the HealthRisk Lab simulation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthrisk_ai.simulation.model import (
    AssetUniverse,
    HealthRiskLab,
    SimulationState,
    SimulationStrategy,
    ShockGenerator,
)


class TestAssetUniverse:
    def test_init_default(self) -> None:
        universe = AssetUniverse()
        assets = universe.get_asset_names()
        assert len(assets) == 4
        assert "insurance" in assets
        assert "hospital_bonds" in assets

    def test_get_asset_config(self) -> None:
        universe = AssetUniverse()
        config = universe.get_asset_config("insurance")
        assert "base_return" in config
        assert "shock_sensitivity" in config

    def test_sample_returns(self) -> None:
        universe = AssetUniverse()
        allocations = {
            "insurance": 0.25,
            "hospital_bonds": 0.25,
            "pharma_equities": 0.25,
            "credit_facilities": 0.25,
        }

        returns = universe.sample_returns(allocations)
        assert len(returns) == 4
        for asset in allocations:
            assert asset in returns


class TestShockGenerator:
    def test_init_default(self) -> None:
        generator = ShockGenerator()
        assert generator.epidemiologic_prob == 0.15
        assert generator.fda_warning_prob == 0.05
        assert generator.cms_rate_cut_prob == 0.10

    def test_generate_no_shocks(self) -> None:
        generator = ShockGenerator()
        # Force no shocks by setting probabilities to 0
        generator.epidemiologic_prob = 0.0
        generator.fda_warning_prob = 0.0
        generator.cms_rate_cut_prob = 0.0

        shocks = generator.generate_shocks(1)
        assert all(v == 0 for v in shocks.values())

    def test_describe_shock(self) -> None:
        generator = ShockGenerator()
        desc = generator.describe_shock("epidemiologic", 0.0)
        assert desc == "No shock"

        desc = generator.describe_shock("epidemiologic", 0.5)
        assert "Moderate" in desc or "epidemiologic" in desc.lower()


class TestSimulationState:
    def test_init(self) -> None:
        universe = AssetUniverse()
        state = SimulationState(500_000_000, universe)

        assert state.initial_capital == 500_000_000
        assert state.capital == 500_000_000
        assert state.portfolio_value == 500_000_000

    def test_allocate(self) -> None:
        universe = AssetUniverse()
        state = SimulationState(500_000_000, universe)

        allocations = {
            "insurance": 0.3,
            "hospital_bonds": 0.3,
            "pharma_equities": 0.2,
            "credit_facilities": 0.2,
        }

        state.allocate(allocations)

        assert len(state.holdings) == 4
        assert len(state.allocations) == 4

    def test_allocate_invalid_sum(self) -> None:
        universe = AssetUniverse()
        state = SimulationState(500_000_000, universe)

        allocations = {
            "insurance": 0.6,
            "hospital_bonds": 0.5,
            "pharma_equities": 0.0,
            "credit_facilities": 0.0,
        }

        # This should raise because sum != 1.0
        with pytest.raises(Exception):  # SimulationError
            state.allocate(allocations)

    def test_record_history(self) -> None:
        universe = AssetUniverse()
        state = SimulationState(500_000_000, universe)

        returns = {"insurance": 0.01, "hospital_bonds": 0.02, "pharma_equities": 0.03, "credit_facilities": 0.01}
        shocks = {"epidemiologic": 0.5, "fda_warning": 0.0, "cms_rate_cut": 0.0}

        state.record_history(1, returns, shocks)

        assert len(state.history) == 1
        assert len(state.shock_history) == 1


class TestHealthRiskLab:
    def test_init_default(self) -> None:
        lab = HealthRiskLab()
        assert lab.initial_capital == 500_000_000
        assert lab.quarters == 40
        assert not lab.is_running
        assert not lab.is_complete

    def test_init_with_seed(self) -> None:
        lab1 = HealthRiskLab(seed=42)
        lab2 = HealthRiskLab(seed=42)

        # Same seed should give reproducible results
        result1 = lab1.step()
        lab2.reset(seed=42)
        result2 = lab2.step()

        assert result1["player_capital"] == result2["player_capital"]

    def test_step(self) -> None:
        lab = HealthRiskLab()
        result = lab.step()

        assert result["quarter"] == 1
        assert result["player_capital"] > 0
        assert result["ai_capital"] > 0
        assert "shocks" in result
        assert "player_returns" in result

    def test_step_complete(self) -> None:
        lab = HealthRiskLab()
        lab.quarters = 3  # Short simulation for test
        lab.reset()

        for _ in range(3):
            lab.step()

        assert lab.is_complete

        with pytest.raises(Exception):  # SimulationError
            lab.step()

    def test_run_full_simulation(self) -> None:
        lab = HealthRiskLab()
        lab.quarters = 5  # Short for test

        results = lab.run_full_simulation()

        assert len(results) == 5
        assert "quarter" in results.columns
        assert "player_capital" in results.columns

    def test_run_with_strategy(self) -> None:
        lab = HealthRiskLab()
        lab.quarters = 3

        results = lab.run_full_simulation(player_strategy=SimulationStrategy.balanced)

        assert len(results) == 3

    def test_get_leaderboard(self) -> None:
        lab = HealthRiskLab()
        lab.quarters = 3

        lab.run_full_simulation()

        leaderboard = lab.get_leaderboard()

        assert len(leaderboard) == 2
        assert "Player" in leaderboard["name"].values
        assert "AI Opponent" in leaderboard["name"].values

    def test_get_summary(self) -> None:
        lab = HealthRiskLab()
        lab.quarters = 3

        lab.run_full_simulation()

        summary = lab.get_summary()

        assert summary["initial_capital"] == 500_000_000
        assert summary["quarters_completed"] == 3
        assert "player" in summary
        assert "ai" in summary

    def test_reset(self) -> None:
        lab = HealthRiskLab()
        lab.quarters = 3
        lab.run_full_simulation()

        assert lab.is_complete

        lab.reset()

        assert not lab.is_complete
        assert lab.current_quarter == 0


class TestSimulationStrategy:
    def test_conservative(self) -> None:
        state = SimulationState(500_000_000, AssetUniverse())
        alloc = SimulationStrategy.conservative(1, state)

        assert alloc["hospital_bonds"] >= 0.35
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_aggressive(self) -> None:
        state = SimulationState(500_000_000, AssetUniverse())
        alloc = SimulationStrategy.aggressive(1, state)

        assert alloc["pharma_equities"] >= 0.40
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_balanced(self) -> None:
        state = SimulationState(500_000_000, AssetUniverse())
        alloc = SimulationStrategy.balanced(1, state)

        for asset, frac in alloc.items():
            assert frac == pytest.approx(0.25)

    def test_adaptive(self) -> None:
        state = SimulationState(500_000_000, AssetUniverse())
        alloc = SimulationStrategy.adaptive(1, state)

        assert sum(alloc.values()) == pytest.approx(1.0)
        assert len(alloc) == 4
