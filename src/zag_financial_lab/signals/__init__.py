"""
Signals module for ZAG Financial Lab.

This module provides leakage-safe signal calculations for evaluating
trading signals from a statistical perspective. All signals are computed
using only historical data available at each point in time.

Signals are research tools and should not be interpreted as trading recommendations.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_momentum(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate momentum signal as rolling return.
    
    Momentum is defined as the percentage price change over a lookback window.
    This is a simple but widely-studied signal in quantitative finance research.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
    window : int, default=20
        Lookback window in days (e.g., 20 ≈ 1 month)
        
    Returns
    -------
    pd.Series
        Momentum signal (percentage return). First window values will be NaN.
        
    Notes
    -----
    This signal uses only historical data. The momentum at time t is calculated
    using prices from t-window to t, which are all available at time t.
    
    Interpretation: Positive momentum suggests recent upward price movement,
    negative suggests downward. This is a descriptive measure, not a prediction.
    """
    return prices.pct_change(periods=window)


def calculate_mean_reversion(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 1.0
) -> pd.Series:
    """
    Calculate mean reversion signal as z-score versus moving average.
    
    Measures how far current price deviates from its moving average,
    normalized by standard deviation. This captures mean-reverting tendencies.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
    window : int, default=20
        Window for moving average and standard deviation
    num_std : float, default=1.0
        Number of standard deviations for normalization (usually keep at 1.0)
        
    Returns
    -------
    pd.Series
        Z-score signal. First window values will be NaN.
        
    Notes
    -----
    Leakage-safe: Uses only historical data for moving average and std dev.
    
    Z-score = (Price - MA) / (StdDev)
    
    Interpretation: 
    - Positive values: price above average (may revert down)
    - Negative values: price below average (may revert up)
    - |z| > 2: typically considered a significant deviation
    
    This is a statistical measure describing current price relative to history.
    It does not predict future mean reversion.
    """
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    # Avoid division by zero
    zscore = (prices - ma) / (std * num_std)
    
    return zscore


def calculate_volatility_signal(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate volatility signal as rolling standard deviation of returns.
    
    Measures recent price variability. Can be used to study volatility regimes
    or as a risk proxy in signal evaluation.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
    window : int, default=20
        Rolling window for volatility calculation
    annualize : bool, default=True
        If True, annualize volatility (multiply by sqrt(252))
        
    Returns
    -------
    pd.Series
        Volatility signal. First window values will be NaN.
        
    Notes
    -----
    Leakage-safe: Uses backward-looking rolling window.
    
    Volatility at time t is calculated from returns during [t-window, t],
    which are all available at time t.
    
    This measures historical volatility and does not predict future volatility.
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_trend_strength(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate trend strength as slope of moving average.
    
    Measures the direction and magnitude of the trend by fitting a linear
    regression to recent prices. Positive slope indicates uptrend, negative
    indicates downtrend.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
    window : int, default=20
        Window for trend calculation
        
    Returns
    -------
    pd.Series
        Trend slope (annualized percentage). First window values will be NaN.
        
    Notes
    -----
    Leakage-safe: Uses only historical prices in rolling window.
    
    For each time t, fits a linear regression to prices from t-window to t.
    The slope is normalized by the mean price to get a percentage trend.
    
    Interpretation:
    - Positive: upward trend
    - Negative: downward trend
    - Magnitude: strength of trend
    
    This describes recent price movement and should not be interpreted as
    a prediction of future trends.
    """
    # Calculate rolling linear regression slope
    slopes = pd.Series(index=prices.index, dtype=float)
    
    for i in range(window, len(prices)):
        y = prices.iloc[i-window:i].values
        x = np.arange(len(y))
        
        # Linear regression: y = a + b*x
        # We want b (slope)
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator > 0:
            slope = numerator / denominator
            # Normalize by mean price to get percentage trend
            # Annualize: multiply by 252/window
            slopes.iloc[i] = (slope / y_mean) * (252 / window)
    
    return slopes


def calculate_all_signals(
    prices: pd.Series,
    momentum_window: int = 20,
    mean_reversion_window: int = 20,
    volatility_window: int = 20,
    trend_window: int = 20
) -> pd.DataFrame:
    """
    Calculate all available signals for a price series.
    
    Convenience function to compute all signals at once with consistent indexing.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
    momentum_window : int, default=20
        Window for momentum signal
    mean_reversion_window : int, default=20
        Window for mean reversion signal
    volatility_window : int, default=20
        Window for volatility signal
    trend_window : int, default=20
        Window for trend strength signal
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each signal type
        
    Notes
    -----
    All signals are leakage-safe and use only historical data.
    Rows with any NaN values are NOT dropped - caller should handle as needed.
    """
    signals = pd.DataFrame(index=prices.index)
    
    signals['momentum'] = calculate_momentum(prices, window=momentum_window)
    signals['mean_reversion'] = calculate_mean_reversion(prices, window=mean_reversion_window)
    signals['volatility'] = calculate_volatility_signal(prices, window=volatility_window)
    signals['trend_strength'] = calculate_trend_strength(prices, window=trend_window)
    
    return signals


def get_signal_description(signal_name: str) -> str:
    """
    Get a description of what a signal measures.
    
    Parameters
    ----------
    signal_name : str
        Name of the signal
        
    Returns
    -------
    str
        Description of the signal
    """
    descriptions = {
        'momentum': 'Rolling percentage return measuring recent price movement',
        'mean_reversion': 'Z-score measuring deviation from moving average',
        'volatility': 'Rolling standard deviation of returns (annualized)',
        'trend_strength': 'Slope of recent price trend (annualized %)'
    }
    
    return descriptions.get(signal_name, 'Unknown signal')
