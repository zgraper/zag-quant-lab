"""
Sample data generation utilities for testing and demonstration.

This module provides functions to generate synthetic market data when
live data sources are not available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_sample_price_data(
    n_days: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.15,
    trend: float = 0.0005,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic price data with realistic characteristics.
    
    This creates sample data with log-normal returns for testing and
    demonstration purposes.
    
    Parameters
    ----------
    n_days : int, default=500
        Number of trading days to generate
    start_price : float, default=100.0
        Initial price level
    volatility : float, default=0.15
        Annualized volatility (e.g., 0.15 = 15%)
    trend : float, default=0.0005
        Daily drift/trend
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Synthetic price data with OHLCV columns and DatetimeIndex
        
    Notes
    -----
    This generates synthetic data for demonstration only. Real market data
    has different statistical properties.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(n_days * 1.4))  # Account for weekends
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
    
    # Generate returns with different regime periods
    returns = np.zeros(n_days)
    
    # Low vol regime (first third)
    n1 = n_days // 3
    returns[:n1] = np.random.normal(trend, volatility/np.sqrt(252) * 0.5, n1)
    
    # High vol regime (middle third)
    n2 = n_days // 3
    returns[n1:n1+n2] = np.random.normal(trend - 0.001, volatility/np.sqrt(252) * 1.5, n2)
    
    # Medium vol regime (last third)
    returns[n1+n2:] = np.random.normal(trend, volatility/np.sqrt(252), n_days - n1 - n2)
    
    # Generate price series
    price = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    daily_range = np.abs(np.random.normal(0, volatility/np.sqrt(252), n_days))
    
    data = pd.DataFrame({
        'Open': price * (1 - daily_range * 0.3),
        'High': price * (1 + daily_range * 0.5),
        'Low': price * (1 - daily_range * 0.5),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Adj Close': price
    }, index=dates)
    
    return data


def save_sample_data(filename: str = "data/sample_spy.csv", **kwargs):
    """
    Generate and save sample data to CSV file.
    
    Parameters
    ----------
    filename : str
        Path to save the CSV file
    **kwargs
        Additional arguments passed to generate_sample_price_data
    """
    data = generate_sample_price_data(**kwargs)
    data.to_csv(filename)
    return data
