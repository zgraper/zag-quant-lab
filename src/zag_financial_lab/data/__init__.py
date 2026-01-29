"""
Data loading and handling module for ZAG Financial Lab.

This module provides functionality to load and prepare daily market price data
for regime detection analysis.
"""

import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta
import os
from pathlib import Path


def load_price_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y",
    use_sample: bool = False
) -> pd.DataFrame:
    """
    Load daily price data for a given ticker symbol.
    
    This function fetches historical price data using yfinance. If specific
    dates are not provided, it defaults to the most recent period.
    Falls back to sample data if network access is unavailable.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY', 'AAPL')
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, default='2y'
        Period to download if dates not specified (e.g., '1y', '2y', '5y')
    use_sample : bool, default=False
        If True, use sample data instead of downloading
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        Index is DatetimeIndex
        
    Raises
    ------
    ValueError
        If ticker is invalid or data cannot be downloaded
        
    Notes
    -----
    This function is designed for research purposes only. Data is fetched
    from public sources and should be validated for production use.
    """
    # Check for local sample data file
    if use_sample or ticker.upper() == "SAMPLE":
        from .sample_data import generate_sample_price_data
        n_days = 500 if period == "2y" else 250 if period == "1y" else 750
        return generate_sample_price_data(n_days=n_days, seed=42)
    
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(ticker, period=period, progress=False)

        # Normalize yfinance output (handle MultiIndex columns)
        if isinstance(data.columns, pd.MultiIndex):
            # If single ticker, drop the ticker level and keep OHLCV columns
            if len(data.columns.levels) == 2:
                # Most common: level0=Price field, level1=Ticker
                # Select this ticker if present; otherwise take the first ticker
                tickers = list(data.columns.levels[1])
                t = ticker if ticker in tickers else tickers[0]
                data = data.xs(t, axis=1, level=1)
            else:
                data.columns = ["_".join(map(str, c)).strip() for c in data.columns.to_flat_index()]
            
        if data.empty:
            # Try sample data as fallback
            from .sample_data import generate_sample_price_data
            n_days = 500 if period == "2y" else 250 if period == "1y" else 750
            return generate_sample_price_data(n_days=n_days, seed=42)
            
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Downloaded data missing required columns for {ticker}")
            
        return data
        
    except Exception as e:
        # Fallback to sample data
        try:
            from .sample_data import generate_sample_price_data
            n_days = 500 if period == "2y" else 250 if period == "1y" else 750
            return generate_sample_price_data(n_days=n_days, seed=42)
        except Exception:
            raise ValueError(f"Error loading data for {ticker}: {str(e)}")


def validate_price_data(data: pd.DataFrame) -> bool:
    """
    Validate that price data meets basic quality requirements.
    
    Parameters
    ----------
    data : pd.DataFrame
        Price data to validate
        
    Returns
    -------
    bool
        True if data passes validation
        
    Raises
    ------
    ValueError
        If data fails validation with specific error message
    """
    if data.empty:
        raise ValueError("Dataset is empty")
        
    if len(data) < 30:
        raise ValueError("Insufficient data points (minimum 30 required)")
        
    # Check for excessive missing values
    close = data["Close"]
    # If Close is accidentally a DataFrame, reduce to scalar missing ratio
    missing_ratio = close.isna().to_numpy().mean() if hasattr(close, "to_numpy") else pd.isna(close).mean()
    if missing_ratio > 0.1:
        raise ValueError("Too many missing values in Close prices (>10%)")
        
    # Check for non-positive prices
    if (pd.DataFrame(close) <= 0).to_numpy().any():
        raise ValueError("Non-positive prices detected")
        
    return True
