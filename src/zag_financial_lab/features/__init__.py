"""
Feature engineering module for ZAG Financial Lab.

This module provides leakage-safe feature calculation for market regime detection.
All features are computed using only historical data to avoid look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Log returns are preferred over simple returns for several statistical reasons:
    - They are additive over time
    - They approximate normal distribution better
    - They are symmetric (log(P2/P1) = -log(P1/P2))
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices
        
    Returns
    -------
    pd.Series
        Log returns. First value will be NaN.
        
    Notes
    -----
    Calculation: log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
    """
    return np.log(prices / prices.shift(1))


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    This is a backward-looking measure that uses only historical data,
    making it leakage-safe for research applications.
    
    Parameters
    ----------
    returns : pd.Series
        Time series of returns (typically log returns)
    window : int, default=20
        Rolling window size in trading days (20 ≈ 1 month)
    annualize : bool, default=True
        If True, annualize volatility (multiply by sqrt(252))
        
    Returns
    -------
    pd.Series
        Rolling volatility. First (window-1) values will be NaN.
        
    Notes
    -----
    Annualization assumes 252 trading days per year, which is standard
    in equity markets. For other asset classes, this may need adjustment.
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
        
    return vol


def prepare_regime_features(
    data: pd.DataFrame,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Prepare feature matrix for regime detection model.
    
    Creates a leakage-safe feature set containing:
    - Log returns
    - Rolling volatility
    
    Parameters
    ----------
    data : pd.DataFrame
        Price data with 'Close' column
    vol_window : int, default=20
        Window size for volatility calculation
        
    Returns
    -------
    pd.DataFrame
        Feature matrix with columns: 'log_return', 'rolling_vol'
        Rows with NaN values are dropped
        
    Notes
    -----
    This function ensures temporal consistency - all features at time t
    use only information available at or before time t.
    """
    features = pd.DataFrame(index=data.index)
    
    # Calculate log returns
    features['log_return'] = calculate_log_returns(data['Close'])
    
    # Calculate rolling volatility
    features['rolling_vol'] = calculate_rolling_volatility(
        features['log_return'],
        window=vol_window
    )
    
    # Drop rows with NaN values
    features = features.dropna()
    
    return features


def get_feature_array(features: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Convert feature DataFrame to numpy array for model input.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame from prepare_regime_features
        
    Returns
    -------
    tuple
        (feature_array, datetime_index)
        - feature_array: shape (n_samples, n_features)
        - datetime_index: corresponding dates
    """
    return features.values, features.index
