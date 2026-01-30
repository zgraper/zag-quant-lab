"""
Analysis module for ZAG Financial Lab.

This module provides statistical analysis tools for evaluating signals,
including forward return calculations, information coefficients, and
quantile analysis. All operations are leakage-safe.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
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
