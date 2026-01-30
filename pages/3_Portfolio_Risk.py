"""
ZAG Financial Lab - Portfolio & Risk Analysis Module

A research tool for analyzing portfolio risk characteristics.
This module focuses on risk metrics and asset relationships,
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
    calculate_rolling_correlation,
    calculate_risk_contribution
)
from zag_financial_lab.plotting import (
    plot_correlation_heatmap,
    plot_drawdown_curve,
    plot_rolling_volatility,
    plot_rolling_correlation,
    plot_risk_contribution,
    plot_cumulative_returns
)

# Page configuration
st.set_page_config(
    page_title="Portfolio & Risk - ZAG Quant Lab",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("💼 Portfolio & Risk Analysis")
st.subheader("Research-Focused Risk Assessment Tool")

st.markdown("""
This module provides portfolio risk analysis from a research perspective.
It computes risk metrics, correlations, and visualizations to understand
portfolio characteristics.

**Important Notes:**
- This is a **research and educational tool** only
- Focus is on risk analysis, not trading or backtesting
- No claims are made about performance, alpha, or profitability
- Results describe historical characteristics only
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

if len(tickers) < 2:
    st.sidebar.warning("⚠️ Please enter at least 2 tickers")

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

corr_window = st.sidebar.slider(
    "Correlation Window (days)",
    min_value=30,
    max_value=180,
    value=60,
    help="Rolling window for correlation calculation"
)

run_analysis = st.sidebar.button("Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About Portfolio & Risk

This module analyzes:
- Portfolio returns and volatility
- Maximum drawdown
- Asset correlations
- Rolling correlations
- Risk contribution by asset

**Focus**: Risk characteristics, not trading performance.

All calculations use leakage-safe, backward-looking data.
""")

# Main content area
if run_analysis:
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers to analyze.")
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
    
    # Calculate portfolio metrics
    with st.spinner("Calculating portfolio metrics..."):
        try:
            # Portfolio returns
            portfolio_returns = calculate_portfolio_returns(prices_df, weights)
            
            # Rolling volatility
            portfolio_vol = calculate_portfolio_volatility(portfolio_returns, window=vol_window)
            
            # Maximum drawdown
            max_dd, drawdown_series = calculate_max_drawdown(portfolio_returns)
            
            # Correlation matrix
            corr_matrix = calculate_correlation_matrix(prices_df)
            
            # Risk contribution
            risk_contrib = calculate_risk_contribution(prices_df, weights)
            
            st.success("✓ Calculated portfolio metrics")
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    
    # Display results
    st.markdown("---")
    st.header("Portfolio Analysis Results")
    
    # Summary metrics
    st.subheader("Risk Metrics Summary")
    
    # Calculate summary statistics
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return * 100:.2f}%")
    with col2:
        st.metric("Annualized Volatility", f"{annualized_vol * 100:.2f}%")
    with col3:
        st.metric("Maximum Drawdown", f"{max_dd * 100:.2f}%")
    with col4:
        st.metric("Sharpe-like Ratio", f"{sharpe_ratio:.2f}")
    
    st.markdown("""
    **Note**: These metrics describe historical characteristics and should not
    be interpreted as predictions or performance claims.
    """)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Returns & Drawdown",
        "🌡️ Volatility",
        "🔗 Correlation",
        "⚖️ Risk Contribution",
        "📈 Asset Prices"
    ])
    
    with tab1:
        st.subheader("Cumulative Returns")
        st.markdown("""
        Shows the growth of the portfolio over time.
        """)
        fig_cum_returns = plot_cumulative_returns(portfolio_returns, 
                                                   title="Portfolio Cumulative Returns")
        st.plotly_chart(fig_cum_returns, use_container_width=True)
        
        st.subheader("Drawdown Analysis")
        st.markdown("""
        Drawdown shows the decline from the previous peak.
        Maximum drawdown is a key risk metric.
        """)
        fig_drawdown = plot_drawdown_curve(drawdown_series, 
                                           title=f"Portfolio Drawdown (Max: {max_dd*100:.2f}%)")
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        st.markdown(f"""
        **Maximum Drawdown**: {max_dd*100:.2f}%
        
        This represents the largest peak-to-trough decline during the period.
        Lower drawdowns indicate more stable portfolios.
        """)
    
    with tab2:
        st.subheader("Rolling Volatility")
        st.markdown(f"""
        Rolling {vol_window}-day annualized volatility shows how portfolio risk
        varies over time.
        """)
        fig_vol = plot_rolling_volatility(portfolio_vol,
                                          title=f"{vol_window}-Day Rolling Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.markdown(f"""
        **Current Volatility**: {portfolio_vol.iloc[-1]*100:.2f}%
        
        **Average Volatility**: {portfolio_vol.mean()*100:.2f}%
        
        Higher volatility indicates higher risk. Volatility often increases during
        market stress periods.
        """)
    
    with tab3:
        st.subheader("Correlation Matrix")
        st.markdown("""
        Correlations show how assets move together:
        - **1.0**: Perfect positive correlation (move together)
        - **0.0**: No correlation (independent)
        - **-1.0**: Perfect negative correlation (move opposite)
        
        Lower correlations provide better diversification.
        """)
        fig_corr = plot_correlation_heatmap(corr_matrix,
                                           title="Asset Return Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Rolling correlation analysis
        if len(tickers) >= 2:
            st.subheader("Rolling Correlation Analysis")
            st.markdown(f"""
            Rolling {corr_window}-day correlation between asset pairs.
            Shows how relationships change over time.
            """)
            
            # Select asset pair for rolling correlation
            col1, col2 = st.columns(2)
            with col1:
                asset1 = st.selectbox("Asset 1", tickers, index=0, key="corr_asset1")
            with col2:
                asset2_options = [t for t in tickers if t != asset1]
                asset2 = st.selectbox("Asset 2", asset2_options, index=0, key="corr_asset2")
            
            if asset1 and asset2 and asset1 != asset2:
                rolling_corr = calculate_rolling_correlation(
                    prices_df, asset1, asset2, window=corr_window
                )
                fig_rolling_corr = plot_rolling_correlation(
                    rolling_corr, asset1, asset2
                )
                st.plotly_chart(fig_rolling_corr, use_container_width=True)
    
    with tab4:
        st.subheader("Risk Contribution by Asset")
        st.markdown("""
        Shows how much each asset contributes to total portfolio risk.
        
        Risk contribution depends on:
        - Asset weight in portfolio
        - Asset volatility
        - Correlations with other assets
        
        Sum of all contributions equals total portfolio volatility.
        """)
        fig_risk = plot_risk_contribution(risk_contrib,
                                          title="Risk Contribution by Asset")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Display risk contribution table
        st.markdown("#### Risk Contribution Details")
        risk_df = pd.DataFrame({
            'Asset': risk_contrib.index,
            'Weight': [weights[t] for t in risk_contrib.index],
            'Risk Contribution (%)': risk_contrib.values * 100,
            'Risk Share (%)': (risk_contrib.values / risk_contrib.sum() * 100)
        })
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        **Total Portfolio Volatility**: {risk_contrib.sum()*100:.2f}%
        
        Note: An asset with a small weight can still contribute significantly
        to risk if it's highly volatile or correlated with other assets.
        """)
    
    with tab5:
        st.subheader("Normalized Asset Prices")
        st.markdown("""
        Asset prices normalized to 100 at the start of the period.
        This allows comparison of relative performance.
        """)
        
        # Normalize prices
        normalized_prices = prices_df / prices_df.iloc[0] * 100
        
        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Normalized Asset Prices (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # Initial state - show instructions
    st.info("""
    👈 Configure your portfolio in the sidebar and click **Run Analysis** to begin.
    
    **Quick Start:**
    1. Enter asset tickers (one per line, e.g., SPY, TLT, GLD)
    2. Set portfolio weights (must sum to 1.0)
    3. Select time period and analysis parameters
    4. Click "Run Analysis"
    5. Explore results in different tabs
    """)
    
    st.markdown("---")
    st.markdown("""
    ### What is Portfolio Risk Analysis?
    
    This module focuses on **risk characteristics** of portfolios without
    making any trading decisions or performance claims.
    
    #### Key Metrics
    
    **Volatility**
    - Measures price variability (risk)
    - Annualized standard deviation of returns
    - Higher values = more risk
    
    **Maximum Drawdown**
    - Largest peak-to-trough decline
    - Key risk metric for understanding worst-case scenarios
    - Lower drawdowns indicate more stability
    
    **Correlation**
    - Measures how assets move together
    - Range: -1 (opposite) to +1 (together)
    - Lower correlations improve diversification
    
    **Risk Contribution**
    - Shows which assets drive portfolio risk
    - Depends on weight, volatility, and correlation
    - Useful for understanding portfolio construction
    
    #### What This Module Does NOT Do
    
    This module explicitly avoids:
    - ❌ Trading signals or recommendations
    - ❌ Performance optimization or alpha claims
    - ❌ PnL backtesting
    - ❌ Order execution or portfolio rebalancing
    
    #### Research Focus
    
    The analysis is purely descriptive and educational:
    - ✅ Historical risk characteristics
    - ✅ Asset relationships and correlations
    - ✅ Drawdown analysis
    - ✅ Time-varying risk metrics
    
    All calculations are leakage-safe and use only historical data
    available at each point in time.
    
    #### Methodology Notes
    
    **Leakage-Safe Design:**
    - Returns calculated using only past prices
    - Rolling metrics use backward-looking windows
    - No look-ahead bias in any calculation
    
    **Conservative Presentation:**
    - Results describe historical patterns only
    - No claims about future performance
    - Appropriate caveats and limitations
    - Research and educational focus
    
    **Standard Assumptions:**
    - 252 trading days per year for annualization
    - Constant weights (buy-and-hold, no rebalancing)
    - Daily data and daily returns
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
<p><strong>ZAG Quant Lab</strong> - Portfolio & Risk Analysis</p>
<p>This tool is for research and educational purposes only.</p>
<p>It does not provide investment advice or trading recommendations.</p>
</div>
""", unsafe_allow_html=True)
