"""HealthRisk Lab: 40-quarter portfolio simulation game engine.

This module implements the gamified simulation where a user manages a $500M
health-finance portfolio against an AI opponent under macroeconomic and
epidemiological shocks.

Features:
- 40-quarter (10-year) simulation loop
- Four asset classes: Insurance, Hospital Bonds, Pharma Equities, Credit Facilities
- AI opponent powered by ensemble models
- Quarterly shock events (pandemics, FDA warnings, CMS rate cuts)
- Portfolio rebalancing mechanics
- Performance tracking and scoring
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from healthrisk_ai.config import SimulationConfig, build_config
from healthrisk_ai.exceptions import SimulationError

logger = logging.getLogger("healthrisk_ai.simulation")


class AssetUniverse:
    """Defines the available asset classes and their base characteristics."""

    BASE_CONFIG = {
        "insurance": {
            "name": "Insurance Portfolio",
            "description": "Health insurance underwriting portfolio",
            "base_return": 0.06,
            "base_volatility": 0.12,
            "base_yield": 0.04,
            "min_allocation": 0.0,
            "max_allocation": 1.0,
            "shock_sensitivity": {
                "epidemiologic": 0.3,  # Positive: pandemics increase claims but also premiums
                "fda_warning": 0.0,
                "cms_rate_cut": -0.2,  # Negative: rate cuts reduce revenue
            },
        },
        "hospital_bonds": {
            "name": "Hospital Bonds",
            "description": "Investment-grade hospital bonds",
            "base_return": 0.04,
            "base_volatility": 0.06,
            "base_yield": 0.05,
            "min_allocation": 0.0,
            "max_allocation": 1.0,
            "shock_sensitivity": {
                "epidemiologic": -0.1,  # Hospitals strained during pandemics
                "fda_warning": 0.0,
                "cms_rate_cut": -0.3,  # Revenue cuts affect bond ratings
            },
        },
        "pharma_equities": {
            "name": "Pharma Equities",
            "description": "Pharmaceutical and biotech stocks",
            "base_return": 0.08,
            "base_volatility": 0.20,
            "base_yield": 0.02,
            "min_allocation": 0.0,
            "max_allocation": 1.0,
            "shock_sensitivity": {
                "epidemiologic": 0.1,  # Vaccine/therapeutic demand
                "fda_warning": -0.4,  # FDA warnings hurt pharma stocks
                "cms_rate_cut": -0.1,
            },
        },
        "credit_facilities": {
            "name": "Credit Facilities",
            "description": "Healthcare credit facilities and loans",
            "base_return": 0.05,
            "base_volatility": 0.10,
            "base_yield": 0.06,
            "min_allocation": 0.0,
            "max_allocation": 1.0,
            "shock_sensitivity": {
                "epidemiologic": -0.2,  # Credit quality deteriorates
                "fda_warning": -0.1,
                "cms_rate_cut": -0.3,
            },
        },
    }

    def __init__(
        self,
        config: SimulationConfig | None = None,
    ) -> None:
        if config is None:
            _, _, _, _, _, _, config = build_config()

        self.assets = config.asset_universe
        self._config = config

        # Initialize asset states
        self._asset_state: dict[str, dict[str, float]] = {}
        for asset in self.assets:
            base = self.BASE_CONFIG[asset]
            self._asset_state[asset] = {
                "return": base["base_return"],
                "volatility": base["base_volatility"],
                "yield": base["base_yield"],
                "price": 100.0,
                "shock_modifier": 0.0,
            }

    def get_asset_names(self) -> list[str]:
        """Return list of asset names."""
        return list(self.assets)

    def get_asset_config(self, asset: str) -> dict[str, Any]:
        """Get configuration for an asset."""
        if asset not in self.BASE_CONFIG:
            msg = f"Unknown asset: {asset}"
            raise SimulationError(msg)
        return self.BASE_CONFIG[asset].copy()

    def get_current_return(self, asset: str) -> float:
        """Get current expected return for an asset."""
        return self._asset_state[asset]["return"]

    def get_current_volatility(self, asset: str) -> float:
        """Get current volatility for an asset."""
        return self._asset_state[asset]["volatility"]

    def update_shocks(
        self,
        shocks: dict[str, float],
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Update asset returns based on shock events.

        Args:
            shocks: Dict of shock type -> magnitude (-1 to 1).
            rng: Random generator for noise.
        """
        if rng is None:
            rng = np.random.default_rng()

        for asset in self.assets:
            base = self.BASE_CONFIG[asset]
            shock_modifier = 0.0

            for shock_type, magnitude in shocks.items():
                if shock_type in base["shock_sensitivity"]:
                    sensitivity = base["shock_sensitivity"][shock_type]
                    shock_modifier += sensitivity * magnitude

            # Add random noise
            noise = rng.normal(0, 0.02)

            # Update return with shock modifier and noise
            self._asset_state[asset]["shock_modifier"] = shock_modifier
            self._asset_state[asset]["return"] = (
                base["base_return"] + shock_modifier + noise
            )

            # Update volatility based on shocks
            vol_shock = abs(shock_modifier) * 0.5
            self._asset_state[asset]["volatility"] = base["base_volatility"] + vol_shock

    def sample_returns(
        self,
        allocations: dict[str, float],
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Sample actual returns given allocations.

        Args:
            allocations: Dict of asset -> allocation fraction.
            rng: Random generator.

        Returns:
            Dict of asset -> realized return.
        """
        if rng is None:
            rng = np.random.default_rng()

        returns: dict[str, float] = {}

        for asset in self.assets:
            expected_return = self._asset_state[asset]["return"]
            volatility = self._asset_state[asset]["volatility"]

            # Sample from normal distribution
            realized_return = rng.normal(expected_return, volatility)

            # Apply allocation weight
            returns[asset] = realized_return * allocations.get(asset, 0.0)

        return returns


class ShockGenerator:
    """Generates quarterly shock events for the simulation."""

    SHOCK_TYPES = {
        "epidemiologic": "Pandemic/Epidemic Outbreak",
        "fda_warning": "FDA Safety Warning",
        "cms_rate_cut": "CMS Reimbursement Rate Cut",
    }

    def __init__(
        self,
        config: SimulationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if config is None:
            _, _, _, _, _, _, config = build_config()

        self.epidemiologic_prob = config.epidemiologic_shock_prob_per_quarter
        self.fda_warning_prob = config.fda_warning_prob_per_quarter
        self.cms_rate_cut_prob = config.cms_rate_cut_prob_per_quarter

        self.rng = np.random.default_rng(seed)

    def generate_shocks(self, quarter: int) -> dict[str, float]:
        """Generate shocks for a quarter.

        Args:
            quarter: Quarter number (1-40).

        Returns:
            Dict of shock_type -> magnitude (0 if no shock).
        """
        shocks: dict[str, float] = {}

        # Epidemiologic shock
        if self.rng.random() < self.epidemiologic_prob:
            # Magnitude: 0.3 to 1.0, sometimes severe (1.0-1.5)
            if self.rng.random() < 0.2:
                magnitude = self.rng.uniform(1.0, 1.5)  # Severe
            else:
                magnitude = self.rng.uniform(0.3, 1.0)  # Moderate
            shocks["epidemiologic"] = magnitude

        # FDA warning
        if self.rng.random() < self.fda_warning_prob:
            magnitude = self.rng.uniform(0.5, 1.0)
            shocks["fda_warning"] = magnitude

        # CMS rate cut
        if self.rng.random() < self.cms_rate_cut_prob:
            magnitude = self.rng.uniform(0.2, 0.5)
            shocks["cms_rate_cut"] = magnitude

        return shocks

    def describe_shock(self, shock_type: str, magnitude: float) -> str:
        """Get human-readable description of a shock."""
        if magnitude == 0:
            return "No shock"

        shock_name = self.SHOCK_TYPES.get(shock_type, shock_type)
        severity = "Severe" if magnitude > 0.8 else "Moderate" if magnitude > 0.4 else "Mild"

        return f"{severity} {shock_name} (magnitude: {magnitude:.2f})"


class AIPortfolio:
    """AI opponent portfolio that makes optimal allocation decisions."""

    def __init__(
        self,
        initial_capital: float,
        asset_universe: AssetUniverse,
    ) -> None:
        self.initial_capital = initial_capital
        self.asset_universe = asset_universe

        # AI uses more conservative strategy
        self._target_allocations = {
            "insurance": 0.25,
            "hospital_bonds": 0.35,
            "pharma_equities": 0.20,
            "credit_facilities": 0.20,
        }

        self.capital = initial_capital
        self.portfolio_value = initial_capital
        self.history: list[dict[str, Any]] = []

    def get_allocations(self, quarter: int) -> dict[str, float]:
        """Get target allocations for the AI."""
        # AI adjusts based on quarter (more conservative later)
        remaining_quarters = 40 - quarter + 1
        if remaining_quarters < 10:
            # Shift to bonds near end
            self._target_allocations["hospital_bonds"] = 0.45
            self._target_allocations["pharma_equities"] = 0.15
            self._target_allocations["insurance"] = 0.25
            self._target_allocations["credit_facilities"] = 0.15

        return self._target_allocations.copy()

    def update(self, returns: dict[str, float]) -> None:
        """Update AI portfolio value based on returns.

        Args:
            returns: Asset returns from simulation.
        """
        allocation = self.get_allocations(len(self.history) + 1)

        # Calculate portfolio return
        portfolio_return = sum(
            allocation.get(asset, 0) * ret
            for asset, ret in returns.items()
        )

        # Update capital (with burn rate)
        self.capital *= (1 + portfolio_return)
        self.portfolio_value = self.capital

        self.history.append({
            "quarter": len(self.history) + 1,
            "capital": self.capital,
            "portfolio_value": self.portfolio_value,
            "returns": returns,
        })


class SimulationState:
    """Tracks the simulation state for a single player."""

    def __init__(
        self,
        initial_capital: float,
        asset_universe: AssetUniverse,
    ) -> None:
        self.initial_capital = initial_capital
        self.asset_universe = asset_universe
        self.capital = initial_capital
        self.portfolio_value = initial_capital

        # Holdings: {asset: (units, cost_basis)}
        self.holdings: dict[str, tuple[float, float]] = {}
        for asset in asset_universe.get_asset_names():
            self.holdings[asset] = (0.0, 0.0)

        # Allocation percentages
        self.allocations: dict[str, float] = {
            asset: 0.0 for asset in asset_universe.get_asset_names()
        }

        # History
        self.history: list[dict[str, Any]] = []
        self.shock_history: list[dict[str, Any]] = []

        # Score
        self.score = 0.0

    def allocate(
        self,
        allocations: dict[str, float],
        *,
        prices: dict[str, float] | None = None,
    ) -> None:
        """Reallocate portfolio.

        Args:
            allocations: Target allocation fractions (must sum to ~1).
            prices: Current asset prices (uses 100 if None).
        """
        if prices is None:
            prices = {asset: 100.0 for asset in self.asset_universe.get_asset_names()}

        total_allocation = sum(allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            msg = f"Allocations must sum to 1.0, got {total_allocation}"
            raise SimulationError(msg)

        # Rebalance: sell assets over-allocated, buy under-allocated
        current_value = self.portfolio_value
        target_values = {
            asset: current_value * alloc
            for asset, alloc in allocations.items()
        }

        # Execute trades (simplified: adjust holdings)
        for asset, target_value in target_values.items():
            current_units, _ = self.holdings[asset]
            current_holding_value = current_units * prices.get(asset, 100.0)

            # Calculate new units
            new_units = target_value / prices.get(asset, 100.0)

            # Update holdings
            self.holdings[asset] = (new_units, prices.get(asset, 100.0))
            self.allocations[asset] = target_value / current_value if current_value > 0 else 0.0

    def execute_returns(self, returns: dict[str, float]) -> None:
        """Apply returns to portfolio.

        Args:
            returns: Asset returns (as fractions).
        """
        total_return = 0.0

        for asset, ret in returns.items():
            units, cost_basis = self.holdings[asset]
            if units > 0:
                # Calculate gain/loss
                gain = units * ret * 100  # Simplified: return on price
                total_return += gain

        # Update capital
        self.capital += total_return
        self.portfolio_value = self.capital

        # Apply burn rate (operational costs)
        burn_rate = self._get_burn_rate()
        self.capital -= self.capital * burn_rate
        self.portfolio_value = self.capital

    def _get_burn_rate(self) -> float:
        """Get quarterly burn rate."""
        config = self.asset_universe._config
        yearly_burn = config.yearly_burn_rate_pct
        return yearly_burn / 4  # Quarterly

    def record_history(
        self,
        quarter: int,
        returns: dict[str, float],
        shocks: dict[str, float],
    ) -> None:
        """Record state for this quarter."""
        self.history.append({
            "quarter": quarter,
            "capital": self.capital,
            "portfolio_value": self.portfolio_value,
            "returns": returns,
            "allocations": self.allocations.copy(),
        })

        if any(m > 0 for m in shocks.values()):
            self.shock_history.append({
                "quarter": quarter,
                "shocks": shocks.copy(),
            })

    def calculate_score(self) -> float:
        """Calculate performance score.

        Score based on:
        - Final portfolio value relative to initial
        - Risk-adjusted return (simulated Sharpe)
        - Bonus for beating AI opponent
        """
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        # Volatility penalty (simplified)
        if len(self.history) > 1:
            returns = [h["capital"] for h in self.history]
            vol = np.std(np.diff(returns)) / (np.mean(returns) + 1e-10)
        else:
            vol = 0.0

        # Score: return - volatility penalty
        score = total_return * 100 - vol * 50

        self.score = float(score)
        return self.score


class HealthRiskLab:
    """Main simulation engine for HealthRisk Lab.

    Orchestrates the 40-quarter game loop with:
    - Player portfolio management
    - AI opponent
    - Shock generation
    - Asset return simulation
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if config is None:
            _, _, _, _, _, _, config = build_config()

        self.config = config
        self.initial_capital = config.initial_capital
        self.quarters = config.quarters

        # Initialize components
        self.asset_universe = AssetUniverse(config)
        self.shock_generator = ShockGenerator(config, seed=seed)

        # Player state
        self.player = SimulationState(self.initial_capital, self.asset_universe)

        # AI opponent
        self.ai = AIPortfolio(self.initial_capital, self.asset_universe)

        # Random state
        self.rng = np.random.default_rng(seed)

        # Game state
        self.current_quarter = 0
        self.is_running = False
        self.is_complete = False

        logger.info(
            "HealthRisk Lab initialized: $%.0fM initial capital, %d quarters",
            self.initial_capital / 1e6,
            self.quarters,
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset simulation to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            # Reseed the shock generator too so runs stay reproducible.
            self.shock_generator = ShockGenerator(self.config, seed=seed)

        self.player = SimulationState(self.initial_capital, self.asset_universe)
        self.ai = AIPortfolio(self.initial_capital, self.asset_universe)
        self.current_quarter = 0
        self.is_running = False
        self.is_complete = False

        logger.info("Simulation reset")

    def step(
        self,
        player_allocations: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute one quarter of the simulation.

        Args:
            player_allocations: Player's target allocations (auto-allocated if None).

        Returns:
            Quarter result with returns, shocks, and updated state.
        """
        if self.is_complete:
            msg = "Simulation already complete"
            raise SimulationError(msg)

        self.current_quarter += 1
        quarter = self.current_quarter

        # Generate shocks
        shocks = self.shock_generator.generate_shocks(quarter)

        # Update asset universe with shocks
        self.asset_universe.update_shocks(shocks, rng=self.rng)

        # Get current prices
        prices = {asset: 100.0 for asset in self.asset_universe.get_asset_names()}

        # Player allocation
        if player_allocations is None:
            # Default equal allocation
            player_allocations = {
                asset: 1.0 / len(self.asset_universe.assets)
                for asset in self.asset_universe.assets
            }

        self.player.allocate(player_allocations, prices=prices)

        # Sample returns
        player_returns = self.asset_universe.sample_returns(
            player_allocations,
            rng=self.rng,
        )

        # AI step
        ai_allocations = self.ai.get_allocations(quarter)
        ai_returns = self.asset_universe.sample_returns(
            ai_allocations,
            rng=self.rng,
        )

        # Execute returns
        self.player.execute_returns(player_returns)
        self.ai.update(ai_returns)

        # Record history
        self.player.record_history(quarter, player_returns, shocks)

        # Check if complete
        if quarter >= self.quarters:
            self.is_complete = True
            self.is_running = False

        # Build result
        result = {
            "quarter": quarter,
            "player_capital": self.player.capital,
            "player_portfolio_value": self.player.portfolio_value,
            "ai_capital": self.ai.capital,
            "ai_portfolio_value": self.ai.portfolio_value,
            "player_returns": player_returns,
            "ai_returns": ai_returns,
            "shocks": shocks,
            "shock_descriptions": {
                k: self.shock_generator.describe_shock(k, v)
                for k, v in shocks.items()
            },
            "is_complete": self.is_complete,
        }

        logger.info(
            "Quarter %d: Player=$%.2fM, AI=$%.2fM, "
            "shocks: %s",
            quarter,
            self.player.capital / 1e6,
            self.ai.capital / 1e6,
            ", ".join(result["shock_descriptions"].values()),
        )

        return result

    def run_full_simulation(
        self,
        player_strategy: callable | None = None,
    ) -> pd.DataFrame:
        """Run the complete 40-quarter simulation.

        Args:
            player_strategy: Function(quarter, state) -> allocations dict.
                            If None, uses equal allocation.

        Returns:
            DataFrame with quarterly results.
        """
        results: list[dict[str, Any]] = []
        self.reset()

        for quarter in range(1, self.quarters + 1):
            if player_strategy is not None:
                allocations = player_strategy(quarter, self.player)
            else:
                allocations = None

            result = self.step(allocations)
            results.append(result)

        return pd.DataFrame(results)

    def get_player_score(self) -> float:
        """Get player's final score."""
        return self.player.calculate_score()

    def get_ai_score(self) -> float:
        """Get AI's final score."""
        if not self.ai.history:
            return 0.0

        total_return = (self.ai.capital - self.initial_capital) / self.initial_capital
        return float(total_return * 100)

    def get_leaderboard(self) -> pd.DataFrame:
        """Get comparison of player vs AI."""
        return pd.DataFrame([{
            "name": "Player",
            "final_capital": self.player.capital,
            "total_return_pct": (
                (self.player.capital - self.initial_capital)
                / self.initial_capital * 100
            ),
            "score": self.player.score,
        }, {
            "name": "AI Opponent",
            "final_capital": self.ai.capital,
            "total_return_pct": (
                (self.ai.capital - self.initial_capital)
                / self.initial_capital * 100
            ),
            "score": self.get_ai_score(),
        }])

    def get_summary(self) -> dict[str, Any]:
        """Get simulation summary."""
        return {
            "initial_capital": self.initial_capital,
            "quarters_completed": self.current_quarter,
            "is_complete": self.is_complete,
            "player": {
                "final_capital": self.player.capital,
                "total_return": (self.player.capital - self.initial_capital)
                / self.initial_capital,
                "score": self.player.score,
            },
            "ai": {
                "final_capital": self.ai.capital,
                "total_return": (self.ai.capital - self.initial_capital)
                / self.initial_capital,
                "score": self.get_ai_score(),
            },
            "shocks_encountered": len(self.player.shock_history),
            "shock_history": self.player.shock_history,
        }


class SimulationStrategy:
    """Pre-built player strategies for the simulation."""

    @staticmethod
    def conservative(quarter: int, state: SimulationState) -> dict[str, float]:
        """Conservative strategy: heavy in bonds and insurance."""
        remaining = 40 - quarter

        if remaining < 10:
            # Very conservative near end
            return {
                "insurance": 0.20,
                "hospital_bonds": 0.50,
                "pharma_equities": 0.10,
                "credit_facilities": 0.20,
            }

        return {
            "insurance": 0.25,
            "hospital_bonds": 0.40,
            "pharma_equities": 0.15,
            "credit_facilities": 0.20,
        }

    @staticmethod
    def aggressive(quarter: int, state: SimulationState) -> dict[str, float]:
        """Aggressive strategy: heavy in pharma equities."""
        return {
            "insurance": 0.15,
            "hospital_bonds": 0.20,
            "pharma_equities": 0.50,
            "credit_facilities": 0.15,
        }

    @staticmethod
    def balanced(quarter: int, state: SimulationState) -> dict[str, float]:
        """Balanced strategy: equal weight across assets."""
        n_assets = 4
        return {asset: 1.0 / n_assets for asset in [
            "insurance", "hospital_bonds", "pharma_equities", "credit_facilities"
        ]}

    @staticmethod
    def adaptive(quarter: int, state: SimulationState) -> dict[str, float]:
        """Adaptive strategy: shifts based on performance."""
        # Simple momentum-based adaptation
        if len(state.history) >= 2:
            recent_return = (
                state.history[-1]["capital"] - state.history[-2]["capital"]
            ) / state.history[-2]["capital"]

            if recent_return > 0.02:
                # Bullish: increase pharma
                return {
                    "insurance": 0.20,
                    "hospital_bonds": 0.25,
                    "pharma_equities": 0.40,
                    "credit_facilities": 0.15,
                }
            elif recent_return < -0.02:
                # Bearish: increase bonds
                return {
                    "insurance": 0.25,
                    "hospital_bonds": 0.45,
                    "pharma_equities": 0.10,
                    "credit_facilities": 0.20,
                }

        return SimulationStrategy.balanced(quarter, state)
