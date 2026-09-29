"""HealthRisk Lab: Interactive Streamlit dashboard for portfolio simulation.

This is the primary user interface for the HealthRisk Lab simulation.
Users can:
- Configure simulation parameters
- Run the 40-quarter simulation
- View portfolio performance vs AI opponent
- Analyze shock events and their impact
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from healthrisk_ai.config import build_config
from healthrisk_ai.simulation.model import (
    HealthRiskLab,
    SimulationStrategy,
)

logger = logging.getLogger("healthrisk_ai.ui")

# Page configuration
st.set_page_config(
    page_title="HealthRisk Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .winner {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    .loser {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize Streamlit session state."""
    if "simulation" not in st.session_state:
        st.session_state.simulation = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "strategy" not in st.session_state:
        st.session_state.strategy = "balanced"


def format_currency(value: float) -> str:
    """Format currency for display."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.0f}K"
    else:
        return f"${value:.2f}"


def format_percent(value: float) -> str:
    """Format percentage for display."""
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def render_sidebar() -> dict[str, Any]:
    """Render sidebar with simulation configuration."""
    st.sidebar.title("🏥 HealthRisk Lab")
    st.sidebar.markdown("""
    Manage a **$500M health-finance portfolio** against an AI opponent
    over 40 quarters (10 years) under simulated shocks.
    """)

    st.sidebar.header("Configuration")

    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10_000_000,
        max_value=1_000_000_000,
        value=500_000_000,
        step=50_000_000,
    )

    quarters = st.sidebar.select_slider(
        "Simulation Length (Quarters)",
        options=[10, 20, 30, 40],
        value=40,
    )

    strategy_name = st.sidebar.selectbox(
        "Investment Strategy",
        options=["balanced", "conservative", "aggressive", "adaptive"],
        index=0,
    )

    st.sidebar.header("Strategy Description")
    strategies = {
        "balanced": "Equal weight across all asset classes. Moderate risk.",
        "conservative": "Heavy in hospital bonds. Lower risk, lower potential return.",
        "aggressive": "Heavy in pharma equities. Higher risk, higher potential return.",
        "adaptive": "Adjusts based on recent performance. Dynamic risk management.",
    }
    st.sidebar.info(strategies.get(strategy_name, ""))

    st.sidebar.divider()

    st.sidebar.header("About")
    st.sidebar.info("""
    **HealthRisk Lab** is a simulation game where you manage a health-finance
    portfolio against an AI opponent. Navigate through 40 quarters of market
    movements, epidemiological shocks, FDA warnings, and CMS policy changes.

    Can you beat the AI?
    """)

    return {
        "initial_capital": initial_capital,
        "quarters": quarters,
        "strategy": strategy_name,
    }


def render_dashboard(simulation: HealthRiskLab, results: Any) -> None:
    """Render the main dashboard."""
    st.title("📊 Portfolio Dashboard")

    # Current status
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Your Portfolio",
            format_currency(simulation.player.capital),
            delta=f"{(simulation.player.capital - simulation.initial_capital) / simulation.initial_capital * 100:.2f}%",
        )

    with col2:
        st.metric(
            "AI Opponent",
            format_currency(simulation.ai.capital),
            delta=f"{(simulation.ai.capital - simulation.initial_capital) / simulation.initial_capital * 100:.2f}%",
        )

    with col3:
        player_return = (simulation.player.capital - simulation.initial_capital) / simulation.initial_capital
        ai_return = (simulation.ai.capital - simulation.initial_capital) / simulation.initial_capital
        diff = player_return - ai_return
        st.metric(
            "You vs AI",
            f"{diff * 100:+.2f}%",
            delta="Leading" if diff > 0 else "Trailing" if diff < 0 else "Tied",
        )

    with col4:
        st.metric(
            "Quarter",
            f"{simulation.current_quarter} / {simulation.quarters}",
            delta="Complete" if simulation.is_complete else "In Progress",
        )

    st.divider()

    # Performance chart
    st.subheader("📈 Performance Over Time")

    if results is not None and len(results) > 0:
        chart_data = results[["quarter", "player_capital", "ai_capital"]].copy()
        chart_data = chart_data.rename(columns={
            "player_capital": "Your Portfolio",
            "ai_capital": "AI Opponent",
        })

        st.line_chart(
            chart_data.set_index("quarter"),
            height=400,
        )

    # Current standings
    st.subheader("🏆 Current Standings")

    leaderboard = simulation.get_leaderboard()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="{'winner' if leaderboard.iloc[0]['total_return_pct'] > leaderboard.iloc[1]['total_return_pct'] else 'loser'}">
            <h3>{leaderboard.iloc[0]['name']}</h3>
            <p>Final: {format_currency(leaderboard.iloc[0]['final_capital'])}</p>
            <p>Return: {format_percent(leaderboard.iloc[0]['total_return_pct'] / 100)}</p>
            <p>Score: {leaderboard.iloc[0]['score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="{'winner' if leaderboard.iloc[1]['total_return_pct'] > leaderboard.iloc[0]['total_return_pct'] else 'loser'}">
            <h3>{leaderboard.iloc[1]['name']}</h3>
            <p>Final: {format_currency(leaderboard.iloc[1]['final_capital'])}</p>
            <p>Return: {format_percent(leaderboard.iloc[1]['total_return_pct'] / 100)}</p>
            <p>Score: {leaderboard.iloc[1]['score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Asset allocation
    st.subheader("💼 Current Allocation")

    if simulation.player.allocations:
        alloc_data = pd.DataFrame([
            {"Asset": asset.replace("_", " ").title(), "Allocation": alloc * 100}
            for asset, alloc in simulation.player.allocations.items()
        ])

        st.bar_chart(alloc_data.set_index("Asset"), height=300)


def render_shocks(simulation: HealthRiskLab) -> None:
    """Render shock event history."""
    st.title("⚡ Shock Events")

    if not simulation.player.shock_history:
        st.info("No major shock events occurred during this simulation.")
        return

    st.subheader("Shock Timeline")

    for shock_event in simulation.player.shock_history:
        quarter = shock_event["quarter"]
        shocks = shock_event["shocks"]

        active_shocks = {k: v for k, v in shocks.items() if v > 0}

        if active_shocks:
            with st.expander(f"Quarter {quarter}", expanded=False):
                for shock_type, magnitude in active_shocks.items():
                    descriptions = {
                        "epidemiologic": "Pandemic/Epidemic Outbreak",
                        "fda_warning": "FDA Safety Warning",
                        "cms_rate_cut": "CMS Reimbursement Rate Cut",
                    }
                    severity = "Severe" if magnitude > 0.8 else "Moderate" if magnitude > 0.4 else "Mild"

                    st.warning(f"**{severity} {descriptions.get(shock_type, shock_type)}**")
                    st.write(f"Magnitude: {magnitude:.2f}")


def render_strategy_analysis(simulation: HealthRiskLab, results: Any) -> None:
    """Render strategy performance analysis."""
    st.title("📋 Strategy Analysis")

    if results is None or len(results) == 0:
        st.info("Run a simulation to see analysis.")
        return

    # Quarterly breakdown
    st.subheader("Quarterly Performance")

    results_display = results.copy()
    results_display["Your Return"] = results_display["player_capital"].pct_change()
    results_display["AI Return"] = results_display["ai_capital"].pct_change()
    results_display["Outperformed AI"] = results_display["Your Return"] > results_display["AI Return"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Quarters Outperformed AI",
            f"{(results_display['Outperformed AI'].sum())} / {len(results_display)}",
        )

    with col2:
        win_rate = results_display["Outperformed AI"].mean()
        st.metric(
            "Win Rate vs AI",
            f"{win_rate * 100:.1f}%",
        )

    # Quarterly returns chart
    st.subheader("Quarterly Returns Comparison")
    returns_chart = results_display[["quarter", "Your Return", "AI Return"]].copy()
    returns_chart = returns_chart.iloc[1:]  # Remove first quarter (NaN)

    st.bar_chart(
        returns_chart.set_index("quarter"),
        height=300,
    )


def render_about() -> None:
    """Render about page."""
    st.title("ℹ️ About HealthRisk Lab")

    st.markdown("""
    ## What is HealthRisk Lab?

    HealthRisk Lab is an educational simulation that demonstrates how
    health-finance portfolios perform under various macroeconomic and
    epidemiological conditions.

    ## How to Play

    1. **Configure** your simulation in the sidebar
    2. **Choose** an investment strategy
    3. **Run** the simulation to see 40 quarters of market activity
    4. **Analyze** your performance vs the AI opponent

    ## Asset Classes

    | Asset | Description | Risk | Typical Return |
    |-------|-------------|------|----------------|
    | **Insurance Portfolio** | Health insurance underwriting | Medium | 6% |
    | **Hospital Bonds** | Investment-grade hospital debt | Low | 4% |
    | **Pharma Equities** | Pharmaceutical/biotech stocks | High | 8% |
    | **Credit Facilities** | Healthcare loans | Medium | 5% |

    ## Shock Events

    - **Pandemic/Epidemic**: Affects all assets, especially insurance and hospital bonds
    - **FDA Warning**: Severely impacts pharma equities
    - **CMS Rate Cut**: Reduces revenue for insurance and hospital bonds

    ## AI Opponent

    The AI opponent uses a conservative strategy with some adaptive elements.
    It's designed to be a challenging but beatable benchmark.

    ## Technical Details

    This simulation is powered by the HealthRisk AI platform, which includes:
    - Bio_ClinicalBERT for clinical text analysis
    - GATv2 GNN for patient graph modeling
    - DeepSurv for survival analysis
    - XGBoost/LightGBM for tabular prediction
    - Ridge stacking ensemble for final predictions
    """)


def main() -> None:
    """Main Streamlit app entry point."""
    logging.basicConfig(level=logging.INFO)

    # Initialize state
    initialize_session_state()

    # Sidebar
    config = render_sidebar()

    # Main content based on tab selection
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Shock Events", "Strategy Analysis", "About"],
    )

    if page == "About":
        render_about()
        return

    # Check if we need to run simulation
    if st.session_state.simulation is None or st.session_state.results is None:
        st.header("🚀 Run Simulation")

        st.markdown("""
        Configure your simulation parameters and click **Run Simulation**
        to start managing your health-finance portfolio!
        """)

        col1, col2 = st.columns([1, 4])

        with col1:
            run_button = st.button(
                "🚀 Run Simulation",
                type="primary",
                use_container_width=True,
            )

        if run_button:
            with st.spinner("Running simulation..."):
                # Build a SimulationConfig from the sidebar values, keeping the
                # remaining parameters (asset universe, shocks) from config.yaml.
                _, _, _, _, _, _, sim_cfg = build_config()
                sim_cfg = replace(
                    sim_cfg,
                    initial_capital=int(config["initial_capital"]),
                    quarters=int(config["quarters"]),
                )

                lab = HealthRiskLab(config=sim_cfg)

                # Run with selected strategy
                strategy_map = {
                    "balanced": SimulationStrategy.balanced,
                    "conservative": SimulationStrategy.conservative,
                    "aggressive": SimulationStrategy.aggressive,
                    "adaptive": SimulationStrategy.adaptive,
                }

                strategy_fn = strategy_map.get(config["strategy"], SimulationStrategy.balanced)

                results = lab.run_full_simulation(player_strategy=strategy_fn)

                # Store in session state
                st.session_state.simulation = lab
                st.session_state.results = results
                st.session_state.strategy = config["strategy"]

                st.success("Simulation complete!")

                # Rerun to show dashboard
                st.rerun()

    else:
        # Resume existing simulation
        simulation = st.session_state.simulation
        results = st.session_state.results

        if page == "Dashboard":
            render_dashboard(simulation, results)
        elif page == "Shock Events":
            render_shocks(simulation)
        elif page == "Strategy Analysis":
            render_strategy_analysis(simulation, results)


if __name__ == "__main__":
    main()
