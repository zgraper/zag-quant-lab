"""
ZAG Financial Lab - Stress Testing & Scenario Analysis Module

A research tool for analyzing portfolio behavior under stress scenarios.
This module focuses on risk assessment and scenario exploration,
not on trading, backtesting, or performance optimization.

Author: ZAG Financial Lab
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from zag_financial_lab.data import load_price_data, validate_price_data
from zag_financial_lab.analysis import (
    calculate_portfolio_returns,
    calculate_portfolio_volatility,
    calculate_max_drawdown,
    calculate_correlation_matrix,
    apply_return_shock,
    apply_volatility_shock,
    apply_correlation_shock,
    calculate_stress_metrics
)
from zag_financial_lab.plotting import (
    plot_stress_comparison_equity,
    plot_stress_comparison_drawdown,
    plot_stress_comparison_volatility,
    plot_correlation_comparison
)

# Page configuration
st.set_page_config(
    page_title="Stress Testing - ZAG Quant Lab",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("⚠️ Stress Testing & Scenario Analysis")
st.subheader("Research-Focused Portfolio Stress Testing")

st.markdown("""
This module provides stress testing and scenario analysis for portfolios from a research perspective.
It analyzes how portfolios behave under hypothetical and historical stress conditions to understand
risk characteristics.

**Important Notes:**
- This is a **research and educational tool** only
- Focus is on risk analysis and scenario exploration
- No claims are made about future performance, alpha, or profitability
- Results describe hypothetical scenarios based on historical patterns
- This is NOT a trading system or predictive tool
""")

st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Portfolio Configuration")

# Asset selection
st.sidebar.markdown("### Asset Selection")
st.sidebar.markdown("Enter asset tickers (one per line)")

default_tickers = "SPY\nTLT\nGLD"
ticker_input = st.sidebar.text_area(
    "Tickers",
    value=default_tickers,
    height=100,
    help="Enter stock/ETF tickers, one per line (e.g., SPY, TLT, GLD)"
)

# Parse tickers
tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

if len(tickers) < 1:
    st.sidebar.warning("⚠️ Please enter at least 1 ticker")

# Weights
st.sidebar.markdown("### Portfolio Weights")
st.sidebar.markdown("Weights should sum to 1.0 (100%)")

weights = {}
total_weight = 0.0

for ticker in tickers:
    default_weight = 1.0 / len(tickers) if len(tickers) > 0 else 0.0
    weight = st.sidebar.number_input(
        f"{ticker}",
        min_value=0.0,
        max_value=1.0,
        value=default_weight,
        step=0.05,
        format="%.2f",
        key=f"weight_{ticker}"
    )
    weights[ticker] = weight
    total_weight += weight

# Display total weight
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f}, not 1.0")
else:
    st.sidebar.success(f"✓ Weights sum to {total_weight:.2f}")

# Data period
st.sidebar.markdown("### Analysis Parameters")

period = st.sidebar.selectbox(
    "Data Period",
    options=["1y", "2y", "3y", "5y", "max"],
    index=2,
    help="Historical data period to analyze"
)

vol_window = st.sidebar.slider(
    "Volatility Window (days)",
    min_value=10,
    max_value=90,
    value=20,
    help="Rolling window for volatility calculation"
)

# Stress scenario selection
st.sidebar.markdown("### Stress Scenario Selection")

scenario_type = st.sidebar.selectbox(
    "Scenario Type",
    options=[
        "Return Shock",
        "Volatility Shock",
        "Correlation Shock",
        "Historical: 2008 GFC",
        "Historical: 2020 COVID",
        "Historical: 2022 Drawdown"
    ],
    help="Select the type of stress scenario to apply"
)

# Scenario-specific parameters
shock_magnitude = 0.0  # Initialize for all scenarios
vol_multiplier = 1.0   # Initialize for all scenarios
corr_increase = 0.0    # Initialize for all scenarios

if scenario_type == "Return Shock":
    shock_pct = st.sidebar.slider(
        "Return Shock (%)",
        min_value=-50,
        max_value=50,
        value=-10,
        step=5,
        help="Percentage shock to apply to returns"
    )
    shock_magnitude = shock_pct / 100.0
elif scenario_type == "Volatility Shock":
    vol_multiplier = st.sidebar.slider(
        "Volatility Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Factor to multiply volatility by (e.g., 2.0 = double volatility)"
    )
elif scenario_type == "Correlation Shock":
    corr_increase = st.sidebar.slider(
        "Correlation Increase",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="How much to increase correlations (0-1)"
    )
elif "Historical" in scenario_type:
    # Define historical stress periods
    historical_periods = {
        "Historical: 2008 GFC": ("2008-09-01", "2009-03-31"),
        "Historical: 2020 COVID": ("2020-02-01", "2020-04-30"),
        "Historical: 2022 Drawdown": ("2022-01-01", "2022-10-31")
    }
    stress_start, stress_end = historical_periods.get(scenario_type, (None, None))
    st.sidebar.markdown(f"**Period**: {stress_start} to {stress_end}")

run_analysis = st.sidebar.button("Run Stress Test", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About Stress Testing

This module analyzes:
- Portfolio behavior under stress scenarios
- Risk metrics comparison (baseline vs stressed)
- Drawdown analysis under stress
- Correlation changes under stress

**Focus**: Understanding portfolio risk, not predicting future events.

All scenarios are hypothetical or historical and should not be
interpreted as predictions.
""")

# Main content area
if run_analysis:
    if len(tickers) < 1:
        st.error("Please enter at least 1 ticker to analyze.")
        st.stop()
    
    if abs(total_weight - 1.0) > 0.01:
        st.error(f"Portfolio weights must sum to 1.0. Current sum: {total_weight:.2f}")
        st.stop()
    
    # Load data
    with st.spinner("Loading price data..."):
        try:
            price_data = {}
            for ticker in tickers:
                data = load_price_data(ticker, period=period)
                validate_price_data(data)
                price_data[ticker] = data['Close']
            
            # Combine into DataFrame
            prices_df = pd.DataFrame(price_data)
            # Drop rows with any missing values
            prices_df = prices_df.dropna()
            
            if len(prices_df) < 30:
                st.error("Insufficient overlapping data for selected assets.")
                st.stop()
            
            st.success(f"✓ Loaded {len(prices_df)} days of overlapping price data")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    
    # Calculate baseline portfolio metrics
    with st.spinner("Calculating baseline metrics..."):
        try:
            # Baseline returns
            baseline_returns = calculate_portfolio_returns(prices_df, weights)
            
            # Baseline volatility
            baseline_vol = calculate_portfolio_volatility(baseline_returns, window=vol_window)
            
            # Baseline max drawdown
            baseline_max_dd, baseline_dd_series = calculate_max_drawdown(baseline_returns)
            
            # Baseline correlation
            baseline_corr = calculate_correlation_matrix(prices_df)
            
            st.success("✓ Calculated baseline metrics")
            
        except Exception as e:
            st.error(f"Error calculating baseline metrics: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    
    # Apply stress scenario
    with st.spinner(f"Applying stress scenario: {scenario_type}..."):
        try:
            if scenario_type == "Return Shock":
                stressed_returns = apply_return_shock(baseline_returns, shock_magnitude)
                scenario_desc = f"{shock_pct:+.0f}% Return Shock"
            
            elif scenario_type == "Volatility Shock":
                stressed_returns = apply_volatility_shock(baseline_returns, vol_multiplier)
                scenario_desc = f"{vol_multiplier:.1f}x Volatility Shock"
            
            elif scenario_type == "Correlation Shock":
                stressed_returns = apply_correlation_shock(prices_df, weights, corr_increase)
                scenario_desc = f"+{corr_increase:.1f} Correlation Shock"
            
            elif "Historical" in scenario_type:
                # For historical scenarios, we'll use the actual returns from that period
                # and compare them to baseline
                if stress_start and stress_end:
                    # Check if we have data in this period
                    try:
                        historical_slice = baseline_returns.loc[stress_start:stress_end]
                        if len(historical_slice) > 0:
                            # Use historical returns as the stressed scenario
                            stressed_returns = historical_slice
                            scenario_desc = scenario_type.replace("Historical: ", "")
                        else:
                            st.warning(f"No data available for {scenario_type}. Using baseline returns.")
                            stressed_returns = baseline_returns
                            scenario_desc = "Baseline (no historical data)"
                    except Exception:
                        st.warning(f"Could not extract historical period. Using volatility shock instead.")
                        stressed_returns = apply_volatility_shock(baseline_returns, 2.0)
                        scenario_desc = "2x Volatility Shock (fallback)"
                else:
                    stressed_returns = baseline_returns
                    scenario_desc = "Baseline"
            else:
                stressed_returns = baseline_returns
                scenario_desc = "Baseline"
            
            # Calculate stressed metrics
            stressed_vol = calculate_portfolio_volatility(stressed_returns, window=vol_window)
            stressed_max_dd, stressed_dd_series = calculate_max_drawdown(stressed_returns)
            
            # For correlation comparison (only for non-historical scenarios with same length)
            if "Historical" not in scenario_type and len(stressed_returns) == len(baseline_returns):
                # Recalculate prices from stressed returns for correlation
                stressed_prices = {}
                returns_df = prices_df.pct_change().dropna()
                for ticker in tickers:
                    if scenario_type == "Return Shock":
                        stressed_prices[ticker] = (1 + returns_df[ticker] + shock_magnitude).cumprod() * prices_df[ticker].iloc[0]
                    elif scenario_type == "Volatility Shock":
                        mean_ret = returns_df[ticker].mean()
                        stressed_rets = mean_ret + (returns_df[ticker] - mean_ret) * vol_multiplier
                        stressed_prices[ticker] = (1 + stressed_rets).cumprod() * prices_df[ticker].iloc[0]
                    else:
                        stressed_prices[ticker] = prices_df[ticker]
                
                stressed_prices_df = pd.DataFrame(stressed_prices)
                stressed_corr = calculate_correlation_matrix(stressed_prices_df)
            else:
                stressed_corr = baseline_corr
            
            st.success(f"✓ Applied stress scenario: {scenario_desc}")
            
        except Exception as e:
            st.error(f"Error applying stress scenario: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    
    # Calculate comparative metrics
    with st.spinner("Calculating stress metrics..."):
        try:
            stress_metrics = calculate_stress_metrics(baseline_returns, stressed_returns)
            
        except Exception as e:
            st.error(f"Error calculating stress metrics: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    
    # Display results
    st.markdown("---")
    st.header("Stress Testing Results")
    
    # Scenario description
    st.subheader(f"Scenario: {scenario_desc}")
    
    if scenario_type == "Return Shock":
        st.markdown(f"""
        Applied a **{shock_pct:+.0f}%** shock to all daily returns.
        
        **Interpretation**: This simulates a scenario where the portfolio experiences
        a consistent {abs(shock_pct):.0f}% {'decline' if shock_pct < 0 else 'increase'} 
        in returns. This is a simplified stress test to understand sensitivity.
        """)
    elif scenario_type == "Volatility Shock":
        st.markdown(f"""
        Multiplied volatility by **{vol_multiplier:.1f}x** while preserving mean returns.
        
        **Interpretation**: This simulates increased market turbulence where price
        swings become {vol_multiplier:.1f}x larger. This helps understand how the portfolio
        behaves in high-volatility environments.
        """)
    elif scenario_type == "Correlation Shock":
        st.markdown(f"""
        Increased asset correlations by **{corr_increase:.1f}**.
        
        **Interpretation**: This simulates crisis conditions where diversification breaks down
        and assets move more closely together. Higher correlations reduce diversification benefits.
        """)
    elif "Historical" in scenario_type:
        st.markdown(f"""
        Analyzing returns from **{stress_start}** to **{stress_end}**.
        
        **Interpretation**: This shows how the portfolio would have performed during this
        historical stress period. Past performance does not predict future results.
        """)
    
    # Summary metrics comparison
    st.subheader("Risk Metrics Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{stress_metrics['stressed_total_return'] * 100:.2f}%",
            delta=f"{(stress_metrics['stressed_total_return'] - stress_metrics['baseline_total_return']) * 100:.2f}%",
            delta_color="normal"
        )
    with col2:
        st.metric(
            "Annualized Volatility",
            f"{stress_metrics['stressed_volatility'] * 100:.2f}%",
            delta=f"{(stress_metrics['stressed_volatility'] - stress_metrics['baseline_volatility']) * 100:.2f}%",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "Maximum Drawdown",
            f"{stress_metrics['stressed_max_dd'] * 100:.2f}%",
            delta=f"{(stress_metrics['stressed_max_dd'] - stress_metrics['baseline_max_dd']) * 100:.2f}%",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            "Sharpe-like Ratio",
            f"{stress_metrics['stressed_sharpe']:.2f}",
            delta=f"{(stress_metrics['stressed_sharpe'] - stress_metrics['baseline_sharpe']):.2f}",
            delta_color="normal"
        )
    
    st.markdown("""
    **Note**: Metrics show the comparison between baseline and stressed scenarios.
    Green/red deltas indicate changes under stress. These are descriptive metrics only.
    """)
    
    # Detailed comparison table
    st.subheader("Detailed Metrics Table")
    
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Return',
            'Annualized Volatility',
            'Maximum Drawdown',
            'Sharpe-like Ratio'
        ],
        'Baseline': [
            f"{stress_metrics['baseline_total_return'] * 100:.2f}%",
            f"{stress_metrics['baseline_volatility'] * 100:.2f}%",
            f"{stress_metrics['baseline_max_dd'] * 100:.2f}%",
            f"{stress_metrics['baseline_sharpe']:.2f}"
        ],
        'Stressed': [
            f"{stress_metrics['stressed_total_return'] * 100:.2f}%",
            f"{stress_metrics['stressed_volatility'] * 100:.2f}%",
            f"{stress_metrics['stressed_max_dd'] * 100:.2f}%",
            f"{stress_metrics['stressed_sharpe']:.2f}"
        ],
        'Change': [
            f"{(stress_metrics['stressed_total_return'] - stress_metrics['baseline_total_return']) * 100:.2f}%",
            f"{(stress_metrics['stressed_volatility'] - stress_metrics['baseline_volatility']) * 100:.2f}%",
            f"{(stress_metrics['stressed_max_dd'] - stress_metrics['baseline_max_dd']) * 100:.2f}%",
            f"{(stress_metrics['stressed_sharpe'] - stress_metrics['baseline_sharpe']):.2f}"
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown("---")
    st.header("Stress Scenario Visualizations")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Equity Curves",
        "📉 Drawdown Analysis",
        "🌡️ Volatility Analysis",
        "🔗 Correlation Analysis"
    ])
    
    with tab1:
        st.subheader("Cumulative Returns Comparison")
        st.markdown("""
        Shows how portfolio value evolves under baseline vs stressed scenarios.
        The stressed scenario shows hypothetical performance under the applied stress condition.
        """)
        
        try:
            fig_equity = plot_stress_comparison_equity(
                baseline_returns,
                stressed_returns,
                scenario_name=scenario_desc
            )
            st.plotly_chart(fig_equity, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating equity curve plot: {str(e)}")
    
    with tab2:
        st.subheader("Drawdown Comparison")
        st.markdown("""
        Drawdown shows the decline from previous peak. Comparing baseline vs stressed
        helps understand how much worse drawdowns could be under stress conditions.
        """)
        
        try:
            fig_dd = plot_stress_comparison_drawdown(
                baseline_returns,
                stressed_returns,
                scenario_name=scenario_desc
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating drawdown plot: {str(e)}")
        
        st.markdown(f"""
        **Baseline Max Drawdown**: {baseline_max_dd*100:.2f}%
        
        **Stressed Max Drawdown**: {stressed_max_dd*100:.2f}%
        
        **Difference**: {(stressed_max_dd - baseline_max_dd)*100:.2f} percentage points
        """)
    
    with tab3:
        st.subheader("Rolling Volatility Comparison")
        st.markdown(f"""
        Rolling {vol_window}-day annualized volatility shows how portfolio risk
        evolves over time under baseline vs stressed conditions.
        """)
        
        try:
            fig_vol = plot_stress_comparison_volatility(
                baseline_returns,
                stressed_returns,
                window=vol_window,
                scenario_name=scenario_desc
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating volatility plot: {str(e)}")
    
    with tab4:
        st.subheader("Correlation Matrix Comparison")
        st.markdown("""
        Asset correlations can change under stress. Higher correlations reduce
        diversification benefits. This comparison shows baseline vs stressed correlations.
        """)
        
        try:
            if "Historical" not in scenario_type:
                fig_corr = plot_correlation_comparison(baseline_corr, stressed_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Calculate average correlation change
                baseline_avg = baseline_corr.values[np.triu_indices_from(baseline_corr.values, k=1)].mean()
                stressed_avg = stressed_corr.values[np.triu_indices_from(stressed_corr.values, k=1)].mean()
                
                st.markdown(f"""
                **Average Baseline Correlation**: {baseline_avg:.3f}
                
                **Average Stressed Correlation**: {stressed_avg:.3f}
                
                **Change**: {(stressed_avg - baseline_avg):.3f}
                """)
            else:
                st.info("Correlation comparison not available for historical scenarios.")
        except Exception as e:
            st.error(f"Error creating correlation plot: {str(e)}")
    
    # Limitations and disclaimers
    st.markdown("---")
    st.markdown("""
    ### Important Limitations
    
    ⚠️ **These results are for research and educational purposes only:**
    
    - **Not Predictive**: Stress scenarios do not predict future market events
    - **Simplified Assumptions**: Real stress scenarios are more complex than these models
    - **Historical Bias**: Historical scenarios may not repeat
    - **No Trading Advice**: This analysis does not constitute trading recommendations
    - **Research Tool**: Designed for understanding risk characteristics, not for decision-making
    
    **Methodology Notes:**
    - Return shocks are applied uniformly (actual shocks are heterogeneous)
    - Volatility shocks preserve mean returns (actual crises may have negative returns)
    - Correlation shocks use simplified modeling (actual correlation dynamics are complex)
    - Historical scenarios assume portfolio composition during those periods
    
    Always validate methodologies extensively before any practical application.
    Consult qualified professionals for investment decisions.
    """)

else:
    # Initial state - show instructions
    st.info("""
    👈 Configure your portfolio and stress scenario in the sidebar, then click **Run Stress Test**.
    
    **Quick Start:**
    1. Enter asset tickers (one per line, e.g., SPY, TLT, GLD)
    2. Set portfolio weights (must sum to 1.0)
    3. Select time period for baseline analysis
    4. Choose a stress scenario type
    5. Configure scenario-specific parameters
    6. Click "Run Stress Test"
    7. Explore results in different tabs
    """)
    
    st.markdown("---")
    st.markdown("""
    ### What is Stress Testing?
    
    **Stress testing** analyzes how portfolios behave under extreme but plausible scenarios.
    This helps understand portfolio vulnerabilities and risk characteristics.
    
    #### Scenario Types
    
    **Parameterized Shocks**
    - **Return Shock**: Apply a uniform return change (e.g., -10%)
    - **Volatility Shock**: Increase volatility by a multiplier (e.g., 2x)
    - **Correlation Shock**: Simulate correlation increases during crises
    
    **Historical Scenarios**
    - **2008 GFC**: Global Financial Crisis period
    - **2020 COVID**: COVID-19 market drawdown
    - **2022 Drawdown**: Recent bear market
    
    #### Key Metrics Analyzed
    
    **Volatility**
    - How much does risk increase under stress?
    - Rolling volatility shows time-varying risk
    
    **Maximum Drawdown**
    - Largest peak-to-trough decline
    - Key metric for understanding worst-case scenarios
    
    **Correlations**
    - How do asset relationships change?
    - Higher stress correlations reduce diversification
    
    **Returns**
    - Descriptive metric showing cumulative performance
    - Not a prediction or performance claim
    
    #### What This Module Does NOT Do
    
    This module explicitly avoids:
    - ❌ Predicting future market events
    - ❌ Trading signals or recommendations
    - ❌ Optimization or alpha claims
    - ❌ Performance guarantees
    
    #### Research Focus
    
    The analysis is purely descriptive and educational:
    - ✅ Understanding portfolio risk characteristics
    - ✅ Exploring hypothetical scenarios
    - ✅ Comparing baseline vs stressed outcomes
    - ✅ Identifying potential vulnerabilities
    
    All scenarios are hypothetical or historical. Past stress events do not
    predict future ones. This tool is for research and learning only.
    
    #### Methodology Notes
    
    **Leakage-Safe Design:**
    - All metrics use backward-looking calculations
    - No look-ahead bias in any analysis
    - Consistent with research standards
    
    **Conservative Presentation:**
    - Clear limitations and assumptions stated
    - No claims about future performance
    - Appropriate caveats throughout
    - Research and educational focus
    
    **Standard Assumptions:**
    - 252 trading days per year for annualization
    - Constant weights (no rebalancing)
    - Daily data and daily returns
    - Simplified stress modeling (for educational purposes)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
<p><strong>ZAG Quant Lab</strong> - Stress Testing & Scenario Analysis</p>
<p>This tool is for research and educational purposes only.</p>
<p>It does not provide investment advice or predict future events.</p>
<p>All scenarios are hypothetical or historical and should not be interpreted as predictions.</p>
</div>
""", unsafe_allow_html=True)
