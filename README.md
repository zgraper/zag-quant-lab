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
│   └── 2_Signal_Research.py     # Signal research module
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
