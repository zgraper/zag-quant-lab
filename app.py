"""
ZAG Financial Lab - Market Regime Detection Application

A research tool for analyzing market regimes using Hidden Markov Models.
This application is designed for educational and research purposes only.

Author: ZAG Financial Lab
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from zag_financial_lab.data import load_price_data, validate_price_data
from zag_financial_lab.features import prepare_regime_features, get_feature_array
from zag_financial_lab.models import GaussianRegimeDetector
from zag_financial_lab.stats import (
    calculate_regime_statistics,
    assign_regime_labels,
    get_regime_summary
)
from zag_financial_lab.plotting import (
    plot_price_and_regimes,
    plot_regime_features,
    plot_regime_distribution,
    plot_volatility_timeseries,
    plot_transition_matrix
)


# Page configuration
st.set_page_config(
    page_title="ZAG Financial Lab - Regime Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📊 ZAG Financial Lab")
st.subheader("Market Regime Detection Using Hidden Markov Models")

st.markdown("""
This application performs statistical analysis of market regimes using a Gaussian Hidden Markov Model (HMM).
It identifies distinct market states based on historical price patterns, returns, and volatility.

**Important Notes:**
- This is a **research and educational tool** only
- Results describe historical patterns and are not predictive
- Do not use for trading decisions without extensive validation
- No claims are made about future performance or alpha generation
""")

st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY",
    help="Enter a valid stock ticker (e.g., SPY, AAPL, QQQ). Note: If live data is unavailable, sample data will be used automatically for demonstration."
)

period = st.sidebar.selectbox(
    "Data Period",
    options=["1y", "2y", "3y", "5y", "max"],
    index=1,
    help="Historical data period to analyze"
)

n_regimes = st.sidebar.slider(
    "Number of Regimes",
    min_value=2,
    max_value=5,
    value=3,
    help="Number of market states to detect"
)

vol_window = st.sidebar.slider(
    "Volatility Window (days)",
    min_value=10,
    max_value=60,
    value=20,
    help="Rolling window for volatility calculation"
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=9999,
    value=42,
    help="For reproducible results"
)

run_analysis = st.sidebar.button("Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
**ZAG Financial Lab** provides quantitative research tools for market analysis.

This regime detection model uses:
- Log returns (leakage-safe)
- Rolling volatility (backward-looking)
- Gaussian HMM (hmmlearn library)

All features are computed using only historical data to avoid look-ahead bias.
""")

# Main content area
if run_analysis:
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            # Load data
            is_sample = ticker.upper() == "SAMPLE"
            data = load_price_data(ticker, period=period)
            validate_price_data(data)
            
            # Display success message
            st.success(f"✓ Loaded {len(data)} days of data for {ticker}")
            # Indicate if using sample data
            if is_sample or ticker.upper() == "SAMPLE":
                st.info("📊 Using sample/synthetic data for demonstration purposes.")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
    
    with st.spinner("Preparing features..."):
        try:
            # Prepare features
            features = prepare_regime_features(data, vol_window=vol_window)
            feature_array, feature_dates = get_feature_array(features)
            
            st.success(f"✓ Prepared {len(features)} feature observations")
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            st.stop()
    
    with st.spinner("Fitting regime detection model..."):
        try:
            # Fit HMM model
            model = GaussianRegimeDetector(
                n_regimes=n_regimes,
                n_iter=100,
                random_state=random_seed
            )
            model.fit(feature_array)
            
            # Predict regimes
            regimes = model.predict(feature_array)
            
            st.success(f"✓ Model fitted with {n_regimes} regimes")
            
        except Exception as e:
            st.error(f"Error fitting model: {str(e)}")
            st.stop()
    
    # Calculate statistics and labels
    regime_stats = calculate_regime_statistics(features, regimes)
    regime_labels = assign_regime_labels(regime_stats)
    
    # Display results
    st.markdown("---")
    st.header("Analysis Results")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", len(regimes))
    with col2:
        st.metric("Number of Regimes", n_regimes)
    with col3:
        current_regime = regimes[-1]
        current_label = regime_labels.get(current_regime, f"Regime {current_regime}")
        st.metric("Most Recent Regime", current_label)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Regimes",
        "🎯 Feature Space",
        "📊 Statistics",
        "⏱️ Timeline",
        "🔄 Transitions"
    ])
    
    with tab1:
        st.subheader("Price History with Detected Regimes")
        st.markdown("""
        This chart shows the price history colored by the detected market regime.
        Each regime represents a distinct market state identified by the HMM.
        """)
        
        # Align data and regimes
        aligned_data = data.loc[feature_dates]
        fig_price = plot_price_and_regimes(aligned_data, regimes, regime_labels, ticker)
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.subheader("Volatility Over Time")
        fig_vol = plot_volatility_timeseries(features, regimes, regime_labels)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab2:
        st.subheader("Regime Characteristics in Feature Space")
        st.markdown("""
        This scatter plot shows how regimes are distributed in the feature space
        (returns vs. volatility). Clusters indicate distinct market behaviors.
        """)
        
        fig_features = plot_regime_features(features, regimes, regime_labels)
        st.plotly_chart(fig_features, use_container_width=True)
    
    with tab3:
        st.subheader("Regime Statistics")
        st.markdown("""
        Descriptive statistics for each detected regime. These describe historical
        patterns and should not be interpreted as predictions.
        """)
        
        # Format and display statistics table
        display_stats = regime_stats.copy()
        display_stats['regime_label'] = display_stats['regime'].map(regime_labels)
        display_stats['frequency'] = display_stats['frequency'].map('{:.1%}'.format)
        display_stats['log_return_mean'] = (display_stats['log_return_mean'] * 100).map('{:.4f}%'.format)
        display_stats['rolling_vol_mean'] = display_stats['rolling_vol_mean'].map('{:.3f}'.format)
        
        st.dataframe(
            display_stats[[
                'regime_label', 'count', 'frequency',
                'log_return_mean', 'rolling_vol_mean'
            ]].rename(columns={
                'regime_label': 'Regime',
                'count': 'Observations',
                'frequency': 'Frequency',
                'log_return_mean': 'Avg Daily Return',
                'rolling_vol_mean': 'Avg Volatility'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("Summary")
        summary_text = get_regime_summary(features, regimes, regime_labels)
        st.text(summary_text)
    
    with tab4:
        st.subheader("Regime Timeline")
        st.markdown("""
        Timeline showing how market regimes evolved over the analysis period.
        """)
        
        fig_timeline = plot_regime_distribution(regimes, feature_dates, regime_labels)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab5:
        st.subheader("Regime Transition Probabilities")
        st.markdown("""
        Estimated probability of transitioning from one regime to another.
        Based on the fitted HMM transition matrix.
        """)
        
        transition_matrix = model.get_transition_matrix()
        fig_transitions = plot_transition_matrix(transition_matrix, regime_labels)
        st.plotly_chart(fig_transitions, use_container_width=True)
        
        st.markdown("""
        **Interpretation:** Each cell shows the probability of moving from one regime (row)
        to another regime (column). Diagonal elements represent regime persistence.
        """)

else:
    # Initial state - show instructions
    st.info("""
    👈 Configure your analysis in the sidebar and click **Run Analysis** to begin.
    
    **Quick Start:**
    1. Enter a ticker symbol (default: SPY)
    2. Select time period and number of regimes
    3. Click "Run Analysis"
    4. Explore results in different tabs
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Methodology
    
    This application uses a **Gaussian Hidden Markov Model (HMM)** to detect market regimes:
    
    1. **Feature Extraction**: Calculates log returns and rolling volatility from price data
    2. **Model Fitting**: Uses EM algorithm to estimate HMM parameters
    3. **Regime Detection**: Applies Viterbi algorithm to identify most likely regime sequence
    4. **Analysis**: Computes statistics and visualizations for each regime
    
    **Key Assumptions:**
    - Markets alternate between distinct statistical regimes
    - Each regime has characteristic return and volatility distributions
    - Regime transitions follow a Markov process
    
    **Limitations:**
    - Past patterns do not predict future regimes
    - Model assumes Gaussian distributions (may not fit all market conditions)
    - Results are sensitive to parameter choices
    - This is a research tool, not a trading system
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
<p><strong>ZAG Financial Lab</strong> - Quantitative Market Research</p>
<p>This tool is for research and educational purposes only. It does not provide investment advice or trading signals.</p>
</div>
""", unsafe_allow_html=True)
