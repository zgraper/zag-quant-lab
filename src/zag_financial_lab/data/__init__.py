"""
Data loading and handling module for ZAG Financial Lab.

This module provides functionality to load and prepare daily market price data
for regime detection analysis.
"""

import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta


def load_price_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y"
) -> pd.DataFrame:
    """
    Load daily price data for a given ticker symbol.
    
    This function fetches historical price data using yfinance. If specific
    dates are not provided, it defaults to the most recent period.
    
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
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(ticker, period=period, progress=False)
            
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Downloaded data missing required columns for {ticker}")
            
        return data
        
    except Exception as e:
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
    if data['Close'].isna().sum() / len(data) > 0.1:
        raise ValueError("Too many missing values in Close prices (>10%)")
        
    # Check for non-positive prices
    if (data['Close'] <= 0).any():
        raise ValueError("Non-positive prices detected")
        
    return True
