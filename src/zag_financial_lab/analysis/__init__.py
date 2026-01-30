"""
Analysis module for ZAG Financial Lab.

This module provides statistical analysis tools for evaluating signals,
including forward return calculations, information coefficients, and
quantile analysis. All operations are leakage-safe.

Additionally provides portfolio risk analysis tools including returns,
volatility, correlation, and drawdown calculations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.stats import spearmanr, pearsonr


def calculate_forward_returns(
    prices: pd.Series,
    horizon: int = 1
) -> pd.Series:
    """
    Calculate forward returns in a leakage-safe manner.
    
    Forward return at time t is the return from t to t+horizon.
    This is properly aligned so that the forward return at index t
    uses prices at t and t+horizon.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
    horizon : int, default=1
        Forward horizon in days (e.g., 1, 5, 20)
        
    Returns
    -------
    pd.Series
        Forward returns. Last horizon values will be NaN.
        
    Notes
    -----
    LEAKAGE-SAFE: The forward return at time t is:
        (price[t+horizon] - price[t]) / price[t]
    
    This is the return you would experience if you acted on a signal at time t
    and held for 'horizon' days. The last 'horizon' observations will be NaN
    because we don't have future prices.
    
    Important: When evaluating signals, only use signal-return pairs where
    the signal at time t is computed using data up to time t (not beyond).
    """
    # Shift prices backward to align with current time
    future_prices = prices.shift(-horizon)
    forward_returns = (future_prices - prices) / prices
    
    return forward_returns


def calculate_information_coefficient(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = 'spearman'
) -> float:
    """
    Calculate Information Coefficient (IC) between signal and forward returns.
    
    IC measures the rank correlation between a signal and subsequent returns.
    It's a key metric for evaluating predictive signals in quantitative finance.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    forward_returns : pd.Series
        Forward returns aligned with signal
    method : str, default='spearman'
        Correlation method: 'spearman' (rank) or 'pearson' (linear)
        
    Returns
    -------
    float
        Information coefficient (-1 to 1). NaN if insufficient data.
        
    Notes
    -----
    Spearman IC (rank correlation) is preferred in finance because:
    - Robust to outliers
    - Captures monotonic relationships
    - More relevant for ranking-based strategies
    
    Interpretation:
    - IC > 0: Signal positively correlates with returns
    - IC < 0: Signal negatively correlates with returns
    - |IC| > 0.05: Often considered meaningful (though context-dependent)
    - |IC| > 0.10: Strong signal (rare)
    
    This measures historical correlation and does not guarantee future performance.
    """
    # Remove NaN values
    mask = signal.notna() & forward_returns.notna()
    valid_signal = signal[mask]
    valid_returns = forward_returns[mask]
    
    if len(valid_signal) < 10:
        return np.nan
    
    if method == 'spearman':
        ic, _ = spearmanr(valid_signal, valid_returns)
    elif method == 'pearson':
        ic, _ = pearsonr(valid_signal, valid_returns)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
    
    return ic


def calculate_rolling_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 60,
    method: str = 'spearman'
) -> pd.Series:
    """
    Calculate rolling Information Coefficient over time.
    
    Shows how signal-return correlation varies over time, useful for
    understanding signal stability and regime dependence.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    forward_returns : pd.Series
        Forward returns aligned with signal
    window : int, default=60
        Rolling window in days (e.g., 60 ≈ 3 months)
    method : str, default='spearman'
        Correlation method: 'spearman' or 'pearson'
        
    Returns
    -------
    pd.Series
        Rolling IC values. First window values will be NaN.
        
    Notes
    -----
    Rolling IC helps identify:
    - Periods when signal works well vs poorly
    - Regime dependence of signal effectiveness
    - Structural breaks in signal-return relationship
    
    Instability in IC suggests the signal may not be robust or may work
    differently in different market conditions.
    """
    rolling_ic = pd.Series(index=signal.index, dtype=float)
    
    # Combine into DataFrame for easier rolling operations
    df = pd.DataFrame({'signal': signal, 'returns': forward_returns})
    df = df.dropna()
    
    if len(df) < window:
        return rolling_ic
    
    for i in range(window, len(df) + 1):
        window_data = df.iloc[i-window:i]
        
        if method == 'spearman':
            ic, _ = spearmanr(window_data['signal'], window_data['returns'])
        else:
            ic, _ = pearsonr(window_data['signal'], window_data['returns'])
        
        rolling_ic.iloc[i-1] = ic
    
    return rolling_ic


def calculate_quantile_analysis(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Perform quantile (bucket) analysis of signal vs forward returns.
    
    Divides signal into quantiles and calculates average forward return
    for each bucket. This shows if high signal values correspond to
    high returns (monotonic relationship).
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    forward_returns : pd.Series
        Forward returns aligned with signal
    n_quantiles : int, default=5
        Number of quantiles (buckets) to create
        
    Returns
    -------
    pd.DataFrame
        Statistics by quantile with columns:
        - quantile: quantile number (1 = lowest signal, n = highest)
        - count: number of observations
        - mean_signal: average signal value in bucket
        - mean_return: average forward return in bucket
        - std_return: standard deviation of returns in bucket
        - sharpe: mean_return / std_return (as a simple metric)
        
    Notes
    -----
    A good signal should show monotonic relationship:
    - Higher signal quantiles → higher returns (for positive signal)
    - Lower signal quantiles → lower returns
    
    This is a form of univariate backtest at the signal level (not a PnL backtest).
    It describes historical patterns and should not be interpreted as future predictions.
    """
    # Combine and remove NaN
    df = pd.DataFrame({'signal': signal, 'returns': forward_returns})
    df = df.dropna()
    
    if len(df) < n_quantiles:
        return pd.DataFrame()
    
    # Assign quantiles (1 = lowest, n = highest)
    df['quantile'] = pd.qcut(df['signal'], q=n_quantiles, labels=False, duplicates='drop') + 1
    
    # Calculate statistics by quantile
    results = []
    for q in sorted(df['quantile'].unique()):
        q_data = df[df['quantile'] == q]
        
        mean_ret = q_data['returns'].mean()
        std_ret = q_data['returns'].std()
        sharpe = mean_ret / std_ret if std_ret > 0 else 0
        
        results.append({
            'quantile': int(q),
            'count': len(q_data),
            'mean_signal': q_data['signal'].mean(),
            'mean_return': mean_ret,
            'std_return': std_ret,
            'sharpe': sharpe
        })
    
    return pd.DataFrame(results)


def calculate_signal_statistics(
    signal: pd.Series,
    forward_returns: pd.Series,
    signal_name: str = "Signal"
) -> Dict:
    """
    Calculate comprehensive statistics for a signal.
    
    Convenience function that computes all key metrics in one call.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    forward_returns : pd.Series
        Forward returns aligned with signal
    signal_name : str
        Name of the signal for display purposes
        
    Returns
    -------
    dict
        Dictionary with keys:
        - name: signal name
        - ic_spearman: Spearman IC
        - ic_pearson: Pearson IC
        - n_observations: number of valid signal-return pairs
        - signal_mean: mean signal value
        - signal_std: signal standard deviation
        - return_mean: mean forward return
        - return_std: forward return standard deviation
        
    Notes
    -----
    This provides a quick overview of signal characteristics and
    relationship with forward returns.
    """
    # Remove NaN
    mask = signal.notna() & forward_returns.notna()
    valid_signal = signal[mask]
    valid_returns = forward_returns[mask]
    
    stats = {
        'name': signal_name,
        'n_observations': len(valid_signal),
        'signal_mean': valid_signal.mean() if len(valid_signal) > 0 else np.nan,
        'signal_std': valid_signal.std() if len(valid_signal) > 0 else np.nan,
        'return_mean': valid_returns.mean() if len(valid_returns) > 0 else np.nan,
        'return_std': valid_returns.std() if len(valid_returns) > 0 else np.nan,
        'ic_spearman': calculate_information_coefficient(signal, forward_returns, method='spearman'),
        'ic_pearson': calculate_information_coefficient(signal, forward_returns, method='pearson')
    }
    
    return stats


def analyze_signal_by_regime(
    signal: pd.Series,
    forward_returns: pd.Series,
    regimes: np.ndarray,
    regime_labels: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Analyze signal performance conditioned on market regime.
    
    Calculates IC separately for each regime to understand if signal
    effectiveness varies across market states.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    forward_returns : pd.Series
        Forward returns aligned with signal
    regimes : np.ndarray
        Regime labels aligned with signal
    regime_labels : dict, optional
        Mapping from regime numbers to descriptive labels
        
    Returns
    -------
    pd.DataFrame
        IC by regime with columns:
        - regime: regime number
        - regime_label: descriptive label
        - n_observations: sample size in regime
        - ic_spearman: Spearman IC within regime
        - ic_pearson: Pearson IC within regime
        
    Notes
    -----
    Regime-conditioned analysis helps understand:
    - Does signal work better in some market conditions?
    - Is signal robust across regimes or regime-dependent?
    
    Small sample sizes within regimes can lead to unreliable IC estimates.
    This is exploratory research, not a basis for trading decisions.
    """
    if regime_labels is None:
        regime_labels = {r: f"Regime {r}" for r in np.unique(regimes)}
    
    # Align data
    df = pd.DataFrame({
        'signal': signal,
        'returns': forward_returns,
        'regime': regimes
    })
    df = df.dropna()
    
    results = []
    for regime in sorted(df['regime'].unique()):
        regime_data = df[df['regime'] == regime]
        
        if len(regime_data) < 10:
            continue
        
        ic_spearman = calculate_information_coefficient(
            regime_data['signal'],
            regime_data['returns'],
            method='spearman'
        )
        
        ic_pearson = calculate_information_coefficient(
            regime_data['signal'],
            regime_data['returns'],
            method='pearson'
        )
        
        results.append({
            'regime': int(regime),
            'regime_label': regime_labels.get(regime, f"Regime {regime}"),
            'n_observations': len(regime_data),
            'ic_spearman': ic_spearman,
            'ic_pearson': ic_pearson
        })
    
    return pd.DataFrame(results)


# ============================================================================
# Portfolio Risk Analysis Functions
# ============================================================================

def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.Series:
    """
    Calculate portfolio returns from component prices and weights.
    
    Computes daily returns for each asset and combines them using
    portfolio weights to produce a portfolio return series.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices. Columns are asset tickers.
    weights : dict
        Portfolio weights as {ticker: weight}. Should sum to 1.0.
        
    Returns
    -------
    pd.Series
        Daily portfolio returns
        
    Notes
    -----
    LEAKAGE-SAFE: Returns at time t are computed using prices at t and t-1.
    This is standard practice for portfolio analysis.
    
    Weights are assumed constant (buy-and-hold, no rebalancing).
    For rebalancing analysis, this would need extension.
    """
    # Calculate daily returns for each asset
    returns = prices.pct_change()
    
    # Apply weights and sum
    weighted_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            weighted_returns += returns[ticker] * weight
    
    return weighted_returns


def calculate_portfolio_volatility(
    returns: pd.Series,
    window: int = 20,
    annualization_factor: float = np.sqrt(252)
) -> pd.Series:
    """
    Calculate rolling portfolio volatility.
    
    Computes rolling standard deviation of returns and annualizes it.
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    window : int, default=20
        Rolling window in days
    annualization_factor : float, default=sqrt(252)
        Factor to annualize daily volatility
        
    Returns
    -------
    pd.Series
        Annualized rolling volatility
        
    Notes
    -----
    LEAKAGE-SAFE: Uses only backward-looking data at each point.
    Standard assumption: 252 trading days per year.
    """
    rolling_std = returns.rolling(window=window).std()
    annualized_vol = rolling_std * annualization_factor
    
    return annualized_vol


def calculate_max_drawdown(
    returns: pd.Series
) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series.
    
    Maximum drawdown is the largest peak-to-trough decline in cumulative returns.
    This is a key risk metric for portfolios.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    max_dd : float
        Maximum drawdown as a fraction (e.g., -0.15 for -15%)
    drawdown_series : pd.Series
        Drawdown at each point in time
        
    Notes
    -----
    Drawdown at time t is:
        (cumulative_value[t] - running_max[t]) / running_max[t]
    
    This is always <= 0 (at peak, drawdown = 0).
    A drawdown of -0.20 means current value is 20% below previous peak.
    """
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown series
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown is the minimum value
    max_dd = drawdown.min()
    
    return max_dd, drawdown


def calculate_correlation_matrix(
    prices: pd.DataFrame,
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix of asset returns.
    
    Computes pairwise correlations between asset returns.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices. Columns are asset tickers.
    window : int, optional
        If specified, uses only the last 'window' days.
        If None, uses all available data.
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
        
    Notes
    -----
    Returns are calculated as pct_change(), which is leakage-safe.
    Correlation is computed on the full period (or last window days).
    
    High correlations indicate assets move together.
    For diversification, lower correlations are preferred.
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Use window if specified
    if window is not None:
        returns = returns.tail(window)
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    return corr_matrix


def calculate_rolling_correlation(
    prices: pd.DataFrame,
    asset1: str,
    asset2: str,
    window: int = 60
) -> pd.Series:
    """
    Calculate rolling correlation between two assets.
    
    Shows how correlation varies over time, useful for understanding
    regime changes in asset relationships.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices
    asset1 : str
        First asset ticker
    asset2 : str
        Second asset ticker
    window : int, default=60
        Rolling window in days
        
    Returns
    -------
    pd.Series
        Rolling correlation over time
        
    Notes
    -----
    Correlation changes over time and may depend on market regime.
    High correlation during crises (correlations go to 1) reduces diversification benefits.
    """
    # Calculate returns
    returns = prices[[asset1, asset2]].pct_change().dropna()
    
    # Calculate rolling correlation
    rolling_corr = returns[asset1].rolling(window=window).corr(returns[asset2])
    
    return rolling_corr


def calculate_risk_contribution(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    window: Optional[int] = None
) -> pd.Series:
    """
    Calculate risk contribution of each asset to portfolio volatility.
    
    Risk contribution shows how much each asset contributes to overall
    portfolio risk. This is useful for risk budgeting and understanding
    portfolio construction.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices
    weights : dict
        Portfolio weights as {ticker: weight}
    window : int, optional
        If specified, uses only the last 'window' days for covariance.
        If None, uses all available data.
        
    Returns
    -------
    pd.Series
        Risk contribution by asset (sum = portfolio volatility)
        
    Notes
    -----
    Risk contribution is calculated as:
        RC_i = w_i * (Cov @ w)_i / portfolio_volatility
    
    Where Cov is the covariance matrix and w is the weight vector.
    
    This decomposes total portfolio risk into contributions from each asset.
    Assets with high weights and/or high volatility contribute more to risk.
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Use window if specified
    if window is not None:
        returns = returns.tail(window)
    
    # Get ordered tickers and weights
    tickers = list(weights.keys())
    weight_array = np.array([weights[t] for t in tickers])
    
    # Calculate covariance matrix (annualized)
    cov_matrix = returns[tickers].cov() * 252
    
    # Calculate portfolio variance
    portfolio_variance = weight_array @ cov_matrix @ weight_array
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate marginal contribution to risk
    marginal_contrib = cov_matrix @ weight_array
    
    # Calculate risk contribution
    risk_contrib = weight_array * marginal_contrib / portfolio_volatility
    
    # Return as Series
    return pd.Series(risk_contrib, index=tickers)
