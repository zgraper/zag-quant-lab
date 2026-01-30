"""
ZAG Quant Lab - Home Page

A quantitative research sandbox for exploring market dynamics.
This application is designed for educational and research purposes only.

Author: ZAG Financial Lab
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Page configuration
st.set_page_config(
    page_title="ZAG Quant Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("🔬 ZAG Quant Lab")
st.subheader("Quantitative Research Sandbox")

st.markdown("""
Welcome to **ZAG Quant Lab**, a research-oriented platform for exploring quantitative 
market analysis techniques. This lab provides tools for statistical analysis of 
financial markets, with a focus on rigorous methodology and educational value.

## Available Modules

### 📊 Regime Detection
Identify market regimes using Hidden Markov Models (HMM). This module analyzes 
historical price patterns to detect distinct market states characterized by 
different return and volatility distributions.

**Key Features:**
- Gaussian HMM with configurable number of regimes
- Leakage-safe feature engineering (log returns, rolling volatility)
- Interactive visualizations of regime characteristics
- Transition probability analysis

### 📈 Signal Research
Evaluate trading signals from a statistical perspective. This module focuses on 
understanding signal properties, stability, and relationships with forward returns.

**Key Features:**
- Multiple signal types (momentum, mean reversion, volatility, trend)
- Information coefficient (IC) analysis
- Quantile/bucket analysis
- Rolling IC stability tracking
- Optional regime-conditioned analysis

### 💼 Portfolio & Risk
Analyze portfolio risk characteristics without making trading decisions or 
performance claims. This module provides tools for understanding portfolio 
construction and risk metrics.

**Key Features:**
- Multi-asset portfolio construction
- Returns, volatility, and maximum drawdown calculation
- Correlation and rolling correlation analysis
- Risk contribution by asset
- Interactive visualizations (correlation matrix, drawdown curve, rolling metrics)

## Research Philosophy

**ZAG Quant Lab** is designed with the following principles:

### Conservative & Academic
All analysis is presented with appropriate caveats and disclaimers. We emphasize 
statistical relationships rather than making claims about profitability or alpha.

### Research-Oriented
This is not a trading system. The focus is on understanding market dynamics, 
testing hypotheses, and educational exploration of quantitative methods.

### Leakage-Safe
All features and signals are computed using only historical data available at 
each point in time, avoiding look-ahead bias.

### Regime-Aware
Where applicable, analysis can be conditioned on market regimes to understand 
how relationships vary across different market states.

## Important Disclaimers

⚠️ **This lab is for research and educational purposes only.**

- **Not Investment Advice**: Nothing in this application constitutes investment 
  advice or trading recommendations
- **No Performance Claims**: We make no claims about future returns, alpha 
  generation, or profitability
- **Historical Analysis**: All results describe historical patterns and should 
  not be interpreted as predictions
- **Validation Required**: Any methodology would require extensive validation, 
  risk management, and professional oversight before practical application
- **Model Limitations**: All models have assumptions and limitations that must 
  be understood

## Getting Started

👈 Use the sidebar to navigate between modules:
1. **Regime Detection** - Analyze market regimes
2. **Signal Research** - Evaluate trading signals
3. **Portfolio & Risk** - Analyze portfolio risk characteristics

Each module provides interactive controls for configuring analysis parameters 
and exploring results through visualizations and statistics.

## Methodology Notes

### Statistical Focus
Analysis emphasizes:
- Correlation and information coefficients
- Distribution characteristics
- Stability over time
- Regime dependence

### No Trading Logic
This lab explicitly avoids:
- Live trading connections
- Portfolio PnL backtesting
- Execution algorithms
- Risk-adjusted performance metrics

The focus is purely on statistical properties and relationships in historical data.

## Academic Use

This application is designed to be portfolio-worthy and suitable for discussion 
in academic or professional settings. It demonstrates:

- Clean code architecture with modular design
- Rigorous avoidance of look-ahead bias
- Conservative presentation of research findings
- Appropriate use of statistical methods
- Professional data visualization

---

## Technical Details

**Built with:**
- Python 3.8+
- Streamlit for interactive UI
- Pandas & NumPy for data manipulation
- Plotly for visualizations
- hmmlearn for regime detection
- yfinance for market data

**Code Structure:**
```
src/zag_financial_lab/
├── data/          # Data loading and validation
├── features/      # Feature engineering
├── signals/       # Signal calculations
├── analysis/      # Statistical analysis
├── models/        # HMM and other models
├── stats/         # Regime statistics
└── plotting/      # Visualizations
```

---
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
<p><strong>ZAG Quant Lab</strong> - Quantitative Market Research</p>
<p>This tool is for research and educational purposes only.</p>
<p>It does not provide investment advice or trading signals.</p>
<p>Past performance does not predict future results.</p>
</div>
""", unsafe_allow_html=True)
