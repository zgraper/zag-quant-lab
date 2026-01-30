"""
ZAG Financial Lab - Signal Research Module

A research tool for evaluating trading signals from a statistical perspective.
This module focuses on signal properties, stability, and relationships with
forward returns - not on trading or backtesting.

Author: ZAG Financial Lab
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from zag_financial_lab.data import load_price_data, validate_price_data
from zag_financial_lab.signals import (
    calculate_momentum,
    calculate_mean_reversion,
    calculate_volatility_signal,
    calculate_trend_strength,
    get_signal_description
)
from zag_financial_lab.analysis import (
    calculate_forward_returns,
    calculate_information_coefficient,
    calculate_rolling_ic,
    calculate_quantile_analysis,
    calculate_signal_statistics
)

# Page configuration
st.set_page_config(
    page_title="Signal Research - ZAG Quant Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📈 Signal Research")
st.subheader("Statistical Analysis of Trading Signals")

st.markdown("""
This module evaluates trading signals from a purely statistical perspective.
We analyze signal properties, their relationship with forward returns, and
stability over time.

**Important Notes:**
- This is a **research tool**, not a trading system
- Analysis is statistical and backward-looking, not predictive
- No claims are made about profitability or alpha generation
- Signals are evaluated for educational purposes only
""")

st.markdown("---")

# Sidebar configuration
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY",
    help="Enter a valid stock ticker (e.g., SPY, AAPL, QQQ)"
)

period = st.sidebar.selectbox(
    "Data Period",
    options=["1y", "2y", "3y", "5y", "max"],
    index=2,
    help="Historical data period to analyze"
)

st.sidebar.markdown("### Signal Selection")

available_signals = {
    'Momentum': 'momentum',
    'Mean Reversion': 'mean_reversion',
    'Volatility': 'volatility',
    'Trend Strength': 'trend_strength'
}

selected_signal_names = st.sidebar.multiselect(
    "Select Signals to Analyze",
    options=list(available_signals.keys()),
    default=['Momentum', 'Mean Reversion'],
    help="Choose one or more signals to evaluate"
)

st.sidebar.markdown("### Analysis Parameters")

forward_horizon = st.sidebar.selectbox(
    "Forward Return Horizon",
    options=[1, 5, 10, 20],
    index=1,
    help="Number of days for forward return calculation (1d, 5d, 10d, 20d)"
)

signal_window = st.sidebar.slider(
    "Signal Window (days)",
    min_value=10,
    max_value=60,
    value=20,
    help="Lookback window for signal calculation"
)

rolling_ic_window = st.sidebar.slider(
    "Rolling IC Window (days)",
    min_value=30,
    max_value=120,
    value=60,
    help="Window for rolling IC calculation"
)

run_analysis = st.sidebar.button("Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About Signal Research

This module calculates various technical signals and analyzes their
statistical relationship with future returns.

**Signals:**
- **Momentum**: Rolling return over N days
- **Mean Reversion**: Z-score vs moving average
- **Volatility**: Rolling standard deviation
- **Trend Strength**: Slope of recent prices

**Analysis:**
- Information Coefficient (IC)
- Quantile analysis
- Rolling IC stability
- All operations are leakage-safe
""")

# Main analysis
if run_analysis:
    if len(selected_signal_names) == 0:
        st.warning("Please select at least one signal to analyze.")
        st.stop()
    
    # Load data
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            data = load_price_data(ticker, period=period)
            validate_price_data(data)
            st.success(f"✓ Loaded {len(data)} days of data for {ticker}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    
    # Calculate signals
    with st.spinner("Calculating signals..."):
        try:
            prices = data['Close']
            signals_df = pd.DataFrame(index=data.index)
            
            for signal_name in selected_signal_names:
                signal_key = available_signals[signal_name]
                
                if signal_key == 'momentum':
                    signals_df[signal_key] = calculate_momentum(prices, window=signal_window)
                elif signal_key == 'mean_reversion':
                    signals_df[signal_key] = calculate_mean_reversion(prices, window=signal_window)
                elif signal_key == 'volatility':
                    signals_df[signal_key] = calculate_volatility_signal(prices, window=signal_window)
                elif signal_key == 'trend_strength':
                    signals_df[signal_key] = calculate_trend_strength(prices, window=signal_window)
            
            st.success(f"✓ Calculated {len(selected_signal_names)} signal(s)")
        except Exception as e:
            st.error(f"Error calculating signals: {str(e)}")
            st.stop()
    
    # Calculate forward returns
    with st.spinner("Calculating forward returns..."):
        try:
            forward_returns = calculate_forward_returns(prices, horizon=forward_horizon)
            st.success(f"✓ Calculated {forward_horizon}-day forward returns")
        except Exception as e:
            st.error(f"Error calculating forward returns: {str(e)}")
            st.stop()
    
    # Display results
    st.markdown("---")
    st.header("Analysis Results")
    
    # Summary metrics
    st.subheader("Signal Statistics Summary")
    
    stats_list = []
    for signal_name in selected_signal_names:
        signal_key = available_signals[signal_name]
        stats = calculate_signal_statistics(
            signals_df[signal_key],
            forward_returns,
            signal_name=signal_name
        )
        stats_list.append(stats)
    
    stats_table = pd.DataFrame(stats_list)
    
    # Format display
    display_stats = stats_table.copy()
    display_stats['ic_spearman'] = display_stats['ic_spearman'].map('{:.4f}'.format)
    display_stats['ic_pearson'] = display_stats['ic_pearson'].map('{:.4f}'.format)
    display_stats['signal_mean'] = display_stats['signal_mean'].map('{:.4f}'.format)
    display_stats['signal_std'] = display_stats['signal_std'].map('{:.4f}'.format)
    
    st.dataframe(
        display_stats[[
            'name', 'n_observations', 'ic_spearman', 'ic_pearson',
            'signal_mean', 'signal_std'
        ]].rename(columns={
            'name': 'Signal',
            'n_observations': 'Observations',
            'ic_spearman': 'Spearman IC',
            'ic_pearson': 'Pearson IC',
            'signal_mean': 'Signal Mean',
            'signal_std': 'Signal Std Dev'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
    **Interpretation:**
    - **Spearman IC**: Rank correlation between signal and forward returns (robust to outliers)
    - **Pearson IC**: Linear correlation between signal and forward returns
    - **|IC| > 0.05**: Generally considered meaningful (context-dependent)
    - **|IC| > 0.10**: Strong relationship (rare in practice)
    """)
    
    # Per-signal analysis tabs
    st.markdown("---")
    st.header("Detailed Signal Analysis")
    
    tabs = st.tabs([name for name in selected_signal_names])
    
    for idx, signal_name in enumerate(selected_signal_names):
        signal_key = available_signals[signal_name]
        signal_series = signals_df[signal_key]
        
        with tabs[idx]:
            st.subheader(f"{signal_name} Analysis")
            st.markdown(f"*{get_signal_description(signal_key)}*")
            
            # Time series plot
            st.markdown("#### Signal Time Series")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=data.index,
                y=signal_series,
                mode='lines',
                name=signal_name,
                line=dict(color='blue', width=1)
            ))
            fig_ts.update_layout(
                title=f"{signal_name} Over Time",
                xaxis_title="Date",
                yaxis_title="Signal Value",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Scatter plot: Signal vs Forward Returns
            st.markdown("#### Signal vs Forward Returns")
            
            # Create scatter data
            scatter_df = pd.DataFrame({
                'signal': signal_series,
                'forward_return': forward_returns
            }).dropna()
            
            fig_scatter = px.scatter(
                scatter_df,
                x='signal',
                y='forward_return',
                opacity=0.5,
                trendline='ols',
                title=f"{signal_name} vs {forward_horizon}-Day Forward Return"
            )
            fig_scatter.update_layout(
                xaxis_title=signal_name,
                yaxis_title=f"{forward_horizon}d Forward Return",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Rolling IC
            st.markdown("#### Rolling Information Coefficient")
            
            rolling_ic = calculate_rolling_ic(
                signal_series,
                forward_returns,
                window=rolling_ic_window
            )
            
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Scatter(
                x=rolling_ic.index,
                y=rolling_ic,
                mode='lines',
                name='Rolling IC',
                line=dict(color='green', width=2)
            ))
            fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_ic.add_hline(y=0.05, line_dash="dot", line_color="lightgreen", annotation_text="IC=0.05")
            fig_ic.add_hline(y=-0.05, line_dash="dot", line_color="lightcoral", annotation_text="IC=-0.05")
            
            fig_ic.update_layout(
                title=f"Rolling {rolling_ic_window}-Day Spearman IC",
                xaxis_title="Date",
                yaxis_title="IC",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_ic, use_container_width=True)
            
            st.markdown("""
            **Interpretation:** Rolling IC shows how signal-return correlation varies over time.
            Instability suggests the signal may be regime-dependent or non-robust.
            """)
            
            # Quantile analysis
            st.markdown("#### Quantile Analysis")
            
            quantile_stats = calculate_quantile_analysis(
                signal_series,
                forward_returns,
                n_quantiles=5
            )
            
            if len(quantile_stats) > 0:
                # Display table
                display_quant = quantile_stats.copy()
                display_quant['mean_return'] = (display_quant['mean_return'] * 100).map('{:.3f}%'.format)
                display_quant['std_return'] = (display_quant['std_return'] * 100).map('{:.3f}%'.format)
                display_quant['mean_signal'] = display_quant['mean_signal'].map('{:.4f}'.format)
                display_quant['sharpe'] = display_quant['sharpe'].map('{:.3f}'.format)
                
                st.dataframe(
                    display_quant.rename(columns={
                        'quantile': 'Quintile',
                        'count': 'Count',
                        'mean_signal': 'Avg Signal',
                        'mean_return': f'Avg {forward_horizon}d Return',
                        'std_return': 'Std Dev',
                        'sharpe': 'Sharpe-like'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Bar chart of returns by quantile
                fig_quant = go.Figure()
                fig_quant.add_trace(go.Bar(
                    x=quantile_stats['quantile'],
                    y=quantile_stats['mean_return'] * 100,
                    name='Mean Return',
                    marker_color='steelblue'
                ))
                fig_quant.update_layout(
                    title=f"Average {forward_horizon}d Return by Signal Quintile",
                    xaxis_title="Signal Quintile (1=Lowest, 5=Highest)",
                    yaxis_title="Mean Return (%)",
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_quant, use_container_width=True)
                
                st.markdown("""
                **Interpretation:** A good signal should show a monotonic relationship - 
                higher signal quintiles should correspond to higher (or lower) returns.
                """)
            else:
                st.warning("Insufficient data for quantile analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### Research Notes
    
    This analysis is purely statistical and describes historical relationships.
    Key points:
    
    - **Not Predictive**: Past signal-return correlations do not guarantee future performance
    - **No Trading Claims**: This is not a backtest of a trading strategy
    - **Educational Purpose**: Designed to understand signal properties and stability
    - **Regime Dependence**: Signal effectiveness may vary by market conditions
    - **Sample Sensitivity**: Results depend on time period and parameter choices
    
    **For interview discussion**: This demonstrates understanding of signal evaluation
    methodology, leakage-free analysis, and conservative presentation of research findings.
    """)

else:
    # Initial instructions
    st.info("""
    👈 Configure your analysis in the sidebar and click **Run Analysis** to begin.
    
    **Quick Start:**
    1. Enter a ticker symbol (default: SPY)
    2. Select signals to analyze (momentum, mean reversion, etc.)
    3. Choose forward return horizon (1d, 5d, 20d)
    4. Click "Run Analysis"
    5. Explore signal statistics and visualizations
    """)
    
    st.markdown("---")
    st.markdown("""
    ### What is Signal Research?
    
    Signal research focuses on understanding the **statistical properties** of trading signals
    without actually trading or backtesting strategies. This is distinct from:
    
    - **Trading Systems**: We don't generate trade decisions
    - **Backtesting**: We don't calculate PnL or risk-adjusted returns
    - **Alpha Research**: We make no claims about profitability
    
    Instead, we analyze:
    
    #### Information Coefficient (IC)
    The correlation between a signal at time t and the forward return from t to t+h.
    This measures whether the signal contains information about future returns.
    
    #### Quantile Analysis
    Divide the signal into buckets (quintiles) and examine average returns in each bucket.
    A good signal should show monotonic relationship across quintiles.
    
    #### Rolling IC
    Track how IC varies over time to understand signal stability and regime dependence.
    Unstable IC suggests the signal may not be robust.
    
    #### Signal Types
    
    **Momentum**: Measures recent price trend (positive = rising, negative = falling)
    - Hypothesis: Trends may continue in the short term
    
    **Mean Reversion**: Measures deviation from average (z-score)
    - Hypothesis: Prices may revert to their mean
    
    **Volatility**: Measures recent price variability
    - Hypothesis: Volatility may predict returns or regime changes
    
    **Trend Strength**: Measures slope of recent price movement
    - Hypothesis: Strong trends may be more persistent
    
    ### Methodology Notes
    
    **Leakage-Safe Design:**
    - All signals use only historical data available at time t
    - Forward returns are properly aligned (known only after signal time)
    - No look-ahead bias in any calculation
    
    **Statistical Focus:**
    - Emphasize correlation and rank statistics (IC)
    - Distribution analysis (quantiles)
    - Stability over time (rolling IC)
    - No optimization or curve-fitting
    
    **Conservative Presentation:**
    - Results describe historical patterns only
    - No claims about future performance
    - Appropriate caveats and limitations
    - Research and educational focus
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
<p><strong>ZAG Quant Lab</strong> - Signal Research Module</p>
<p>This tool is for research and educational purposes only. It does not provide investment advice or trading signals.</p>
</div>
""", unsafe_allow_html=True)
