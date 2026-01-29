# ZAG Financial Lab

A modular Python framework for quantitative market research, focusing on market regime detection using statistically sound methods.

## Overview

**ZAG Financial Lab** provides tools for analyzing market dynamics through regime detection. The project uses a Gaussian Hidden Markov Model (HMM) to identify distinct market states based on historical price data, returns, and volatility patterns.

**Important:** This is a research and educational tool only. It is not designed for trading or alpha generation. All analysis is backward-looking and descriptive, not predictive.

## Features

- 📊 **Interactive Streamlit Application**: User-friendly interface for regime analysis
- 🔬 **Leakage-Safe Features**: Log returns and rolling volatility computed using only historical data
- 🎯 **Gaussian Hidden Markov Model**: Statistical regime detection using hmmlearn
- 📈 **Plotly Visualizations**: Interactive charts for exploring market regimes
- 🧩 **Modular Architecture**: Clean separation of concerns (data, features, models, stats, plotting)
- 📝 **Comprehensive Documentation**: Clear docstrings and conservative research tone

## Project Structure

```
zag-quant-lab/
├── src/zag_financial_lab/
│   ├── __init__.py
│   ├── data/           # Data loading and validation
│   ├── features/       # Feature engineering (log returns, volatility)
│   ├── models/         # Gaussian HMM implementation
│   ├── stats/          # Statistical analysis and regime characterization
│   └── plotting/       # Plotly-based visualizations
├── app.py              # Main Streamlit application
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

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your default browser. From there you can:

1. Enter a ticker symbol (e.g., SPY, AAPL, QQQ)
2. Configure analysis parameters (time period, number of regimes, volatility window)
3. Run the analysis
4. Explore results through interactive visualizations

### Using as a Python Library

You can also import and use the modules directly:

```python
from zag_financial_lab.data import load_price_data
from zag_financial_lab.features import prepare_regime_features
from zag_financial_lab.models import GaussianRegimeDetector

# Load data
data = load_price_data('SPY', period='2y')

# Prepare features
features = prepare_regime_features(data, vol_window=20)

# Fit HMM model
model = GaussianRegimeDetector(n_regimes=3, random_state=42)
model.fit(features.values)

# Predict regimes
regimes = model.predict(features.values)
```

## Methodology

### Feature Engineering

The model uses two leakage-safe features:

1. **Log Returns**: `log(P_t / P_{t-1})`
   - Preferred over simple returns for statistical properties
   - Additive over time
   - Better approximation to normal distribution

2. **Rolling Volatility**: Standard deviation of returns over a rolling window
   - Backward-looking only (uses historical data)
   - Annualized using √252 convention
   - Default window: 20 days (≈ 1 trading month)

### Hidden Markov Model

The Gaussian HMM assumes:
- Markets transition between distinct regimes (hidden states)
- Each regime has characteristic Gaussian distributions for returns and volatility
- Transitions follow a Markov process (memoryless)

The model uses:
- **EM Algorithm**: For parameter estimation
- **Viterbi Algorithm**: For finding most likely regime sequence
- **Full Covariance**: Captures correlations between features

## Visualizations

The application provides several interactive charts:

- **Price & Regimes**: Historical prices colored by detected regime
- **Feature Space**: Scatter plot showing regime clustering
- **Statistics**: Descriptive statistics for each regime
- **Timeline**: Evolution of regimes over time
- **Transition Matrix**: Regime transition probabilities

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
