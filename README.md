# ZAG Quant Lab

A modular Python framework for quantitative market research, providing multiple analysis tools with a focus on statistical rigor and educational value.

## Overview

**ZAG Quant Lab** is a research-oriented platform for exploring quantitative market analysis techniques. The lab provides multiple modules for analyzing market dynamics, evaluating signals, and understanding statistical relationships in financial data.

**Important:** This is a research and educational tool only. It is not designed for trading or alpha generation. All analysis is backward-looking and descriptive, not predictive.

## Available Modules

### 📊 Regime Detection
Identify market regimes using Hidden Markov Models (HMM). This module analyzes historical price patterns to detect distinct market states characterized by different return and volatility distributions.

### 📈 Signal Research
Evaluate trading signals from a statistical perspective. This module focuses on understanding signal properties, stability, and relationships with forward returns—without actual trading or backtesting.

### 💼 Portfolio & Risk Analysis
Analyze portfolio risk characteristics without making trading decisions or performance claims. This module provides tools for understanding portfolio construction, correlations, drawdowns, and risk metrics.

### ⚠️ Stress Testing & Scenario Analysis
Research-focused stress testing module for portfolios. Analyze hypothetical and historical stress scenarios and their impact on portfolio risk. Understand how portfolios behave under various stress conditions including return shocks, volatility increases, and correlation changes.

## Features

- 🔬 **Multi-Module Architecture**: Clean separation between regime detection and signal research
- 📊 **Interactive Streamlit Application**: User-friendly multi-page interface
- 🔬 **Leakage-Safe Operations**: All features and signals use only historical data
- 🎯 **Statistical Focus**: Emphasis on correlations, distributions, and stability
- 📈 **Plotly Visualizations**: Interactive charts for exploring results
- 🧩 **Modular Codebase**: Clean separation of concerns across modules
- 📝 **Comprehensive Documentation**: Clear docstrings and conservative research tone

## Project Structure

```
zag-quant-lab/
├── Home.py                  # Main entry point - Home page
├── pages/
│   ├── 1_Regime_Detection.py    # Regime detection module
│   ├── 2_Signal_Research.py     # Signal research module
│   ├── 3_Portfolio_Risk.py      # Portfolio & risk analysis module
│   └── 4_Stress_Testing.py      # Stress testing & scenario analysis module
├── src/zag_financial_lab/
│   ├── __init__.py
│   ├── data/           # Data loading and validation
│   ├── features/       # Feature engineering (log returns, volatility)
│   ├── signals/        # Signal calculations (momentum, mean reversion, etc.)
│   ├── analysis/       # Signal analysis (IC, quantiles, forward returns)
│   ├── models/         # Gaussian HMM implementation
│   ├── stats/          # Statistical analysis and regime characterization
│   └── plotting/       # Plotly-based visualizations
├── requirements.txt    # Project dependencies
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zgraper/zag-quant-lab.git
cd zag-quant-lab
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

Launch the interactive multi-page web application:

```bash
streamlit run Home.py
```

The app will open in your default browser. You'll see a home page with links to:
1. **Regime Detection** - Analyze market regimes using HMM
2. **Signal Research** - Evaluate trading signals statistically
3. **Portfolio & Risk** - Analyze portfolio risk characteristics
4. **Stress Testing** - Test portfolios under stress scenarios

### Module 1: Regime Detection

From the regime detection page you can:
1. Enter a ticker symbol (e.g., SPY, AAPL, QQQ)
2. Configure analysis parameters (time period, number of regimes, volatility window)
3. Run the analysis
4. Explore results through interactive visualizations

### Module 2: Signal Research

From the signal research page you can:
1. Enter a ticker symbol
2. Select signals to analyze (momentum, mean reversion, volatility, trend strength)
3. Choose forward return horizon (1d, 5d, 10d, 20d)
4. View Information Coefficient (IC), quantile analysis, and rolling IC stability
5. Explore signal-return relationships through interactive charts

### Module 3: Portfolio & Risk Analysis

From the portfolio & risk page you can:
1. Enter multiple asset tickers
2. Configure portfolio weights
3. Select time period and analysis parameters
4. View portfolio risk metrics (volatility, max drawdown, correlations)
5. Explore rolling correlations and risk contributions

### Module 4: Stress Testing & Scenario Analysis

From the stress testing page you can:
1. Configure a portfolio with multiple assets and weights
2. Select a stress scenario type:
   - Parameterized shocks (return, volatility, correlation)
   - Historical scenarios (2008 GFC, 2020 COVID, 2022 drawdown)
3. Compare baseline vs stressed outcomes
4. Analyze equity curves, drawdowns, and volatility under stress
5. Understand portfolio vulnerabilities and risk characteristics

### Using as a Python Library

You can also import and use the modules directly:

```python
from zag_financial_lab.data import load_price_data
from zag_financial_lab.features import prepare_regime_features
from zag_financial_lab.models import GaussianRegimeDetector
from zag_financial_lab.signals import calculate_momentum, calculate_mean_reversion
from zag_financial_lab.analysis import calculate_forward_returns, calculate_information_coefficient

# Load data
data = load_price_data('SPY', period='2y')

# Regime Detection
features = prepare_regime_features(data, vol_window=20)
model = GaussianRegimeDetector(n_regimes=3, random_state=42)
model.fit(features.values)
regimes = model.predict(features.values)

# Signal Research
prices = data['Close']
momentum_signal = calculate_momentum(prices, window=20)
forward_returns = calculate_forward_returns(prices, horizon=5)
ic = calculate_information_coefficient(momentum_signal, forward_returns)
```

## Methodology

### Module 1: Regime Detection

#### Feature Engineering

The model uses two leakage-safe features:

1. **Log Returns**: `log(P_t / P_{t-1})`
   - Preferred over simple returns for statistical properties
   - Additive over time
   - Better approximation to normal distribution

2. **Rolling Volatility**: Standard deviation of returns over a rolling window
   - Backward-looking only (uses historical data)
   - Annualized using √252 convention
   - Default window: 20 days (≈ 1 trading month)

#### Hidden Markov Model

The Gaussian HMM assumes:
- Markets transition between distinct regimes (hidden states)
- Each regime has characteristic Gaussian distributions for returns and volatility
- Transitions follow a Markov process (memoryless)

The model uses:
- **EM Algorithm**: For parameter estimation
- **Viterbi Algorithm**: For finding most likely regime sequence
- **Full Covariance**: Captures correlations between features

### Module 2: Signal Research

#### Signal Types

1. **Momentum**: Rolling percentage return over N days
   - Measures recent price trend
   - Hypothesis: Trends may persist in short term

2. **Mean Reversion**: Z-score relative to moving average
   - Measures deviation from historical average
   - Hypothesis: Prices may revert to mean

3. **Volatility**: Rolling standard deviation of returns
   - Measures recent price variability
   - Can indicate regime changes

4. **Trend Strength**: Slope of linear regression on recent prices
   - Measures direction and magnitude of trend
   - Normalized and annualized

#### Analysis Methods

1. **Information Coefficient (IC)**: Spearman rank correlation between signal and forward returns
   - |IC| > 0.05: Generally meaningful
   - |IC| > 0.10: Strong (rare)

2. **Quantile Analysis**: Divide signal into buckets and compare returns across buckets
   - Tests for monotonic relationship
   - Shows signal discriminative power

3. **Rolling IC**: Time-varying correlation to assess stability
   - Identifies regime dependence
   - Highlights structural breaks

All operations are leakage-safe: signals at time t use only data available at or before t.

### Module 3: Portfolio & Risk Analysis

#### Risk Metrics

The module calculates key portfolio risk metrics:

1. **Portfolio Returns**: Weighted sum of asset returns
   - Constant weights (buy-and-hold)
   - Daily rebalancing not assumed

2. **Volatility**: Rolling standard deviation of returns
   - Annualized using √252 convention
   - Shows time-varying risk

3. **Maximum Drawdown**: Largest peak-to-trough decline
   - Key risk metric for understanding worst-case scenarios
   - Always ≤ 0 (at peak, drawdown = 0)

4. **Correlation**: Pairwise asset return correlations
   - Range: -1 (opposite) to +1 (together)
   - Lower correlations provide better diversification

5. **Risk Contribution**: How much each asset contributes to total risk
   - Depends on weight, volatility, and correlations
   - Sum of contributions = total portfolio volatility

### Module 4: Stress Testing & Scenario Analysis

#### Stress Scenarios

The module implements several types of stress tests:

1. **Return Shock**: Apply uniform return change
   - Example: -10% shock to all daily returns
   - Simple sensitivity test

2. **Volatility Shock**: Multiply volatility by a factor
   - Example: 2x volatility (double the swings)
   - Preserves mean returns while increasing dispersion

3. **Correlation Shock**: Increase asset correlations
   - Simulates crisis conditions where diversification breaks down
   - Returns blend toward market average

4. **Historical Scenarios**: Replay historical stress periods
   - 2008 GFC (Sep 2008 - Mar 2009)
   - 2020 COVID (Feb 2020 - Apr 2020)
   - 2022 Drawdown (Jan 2022 - Oct 2022)

#### Stress Analysis Methods

1. **Comparative Metrics**: Side-by-side baseline vs stressed
   - Total return (descriptive only)
   - Volatility increase
   - Maximum drawdown worsening
   - Sharpe-like ratio changes

2. **Visual Comparisons**:
   - Equity curves (baseline vs stressed)
   - Drawdown profiles
   - Rolling volatility evolution
   - Correlation matrices

3. **Risk Assessment Focus**:
   - Understand portfolio vulnerabilities
   - Identify concentration risks
   - Evaluate diversification effectiveness under stress
   - Research tool only - not predictive

All stress scenarios are hypothetical or historical. They do not predict future events
and are designed purely for educational and research purposes.

## Visualizations

### Regime Detection Module

- **Price & Regimes**: Historical prices colored by detected regime
- **Feature Space**: Scatter plot showing regime clustering in return-volatility space
- **Statistics**: Descriptive statistics for each regime
- **Timeline**: Evolution of regimes over time
- **Transition Matrix**: Regime transition probabilities

### Signal Research Module

- **Signal Time Series**: How signals evolve over time
- **Signal vs Returns**: Scatter plot showing relationship with forward returns
- **Rolling IC**: Time-varying information coefficient
- **Quantile Analysis**: Returns by signal bucket
- **Statistics Table**: IC, signal properties, and observation counts

### Portfolio & Risk Module

- **Cumulative Returns**: Portfolio performance over time
- **Drawdown Curve**: Peak-to-trough declines
- **Rolling Volatility**: Time-varying risk
- **Correlation Heatmap**: Asset return correlations
- **Rolling Correlation**: Time-varying pairwise correlations
- **Risk Contribution**: How each asset contributes to total risk

### Stress Testing Module

- **Equity Curve Comparison**: Baseline vs stressed cumulative returns
- **Drawdown Comparison**: Baseline vs stressed peak-to-trough declines
- **Volatility Comparison**: Baseline vs stressed rolling volatility
- **Correlation Comparison**: Side-by-side correlation matrices
- **Metrics Table**: Comprehensive baseline vs stressed statistics

## Limitations and Disclaimers

⚠️ **Important Limitations:**

- **Not Predictive**: Past regime patterns do not predict future regimes
- **Research Only**: This is not a trading system or investment tool
- **No Alpha Claims**: We make no claims about profitability or market outperformance
- **Model Assumptions**: Assumes Gaussian distributions and Markov transitions (may not fit all market conditions)
- **Parameter Sensitivity**: Results depend on configuration choices
- **Historical Analysis**: All features and results are backward-looking

**Use Responsibly**: This tool is designed for research, education, and understanding market dynamics. It should not be used for trading decisions without extensive validation, risk management, and professional advice.

## Dependencies

- Python ≥ 3.8
- streamlit ≥ 1.28.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- plotly ≥ 5.18.0
- hmmlearn ≥ 0.3.0
- scipy ≥ 1.11.0
- yfinance ≥ 0.2.0

## Contributing

This is a research project. Contributions focused on improving methodology, documentation, or educational value are welcome.

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Disclaimer**: This software is provided for educational and research purposes only. It does not constitute investment advice, trading signals, or recommendations. Users assume all responsibility for any decisions made based on this software. Past performance does not indicate future results.
