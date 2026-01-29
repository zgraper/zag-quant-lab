"""
Statistical analysis module for ZAG Financial Lab.

This module provides functions for analyzing detected market regimes,
including descriptive statistics and regime characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_regime_statistics(
    features: pd.DataFrame,
    regimes: np.ndarray
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each detected regime.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with columns like 'log_return', 'rolling_vol'
    regimes : np.ndarray
        Regime labels for each observation
        
    Returns
    -------
    pd.DataFrame
        Statistics by regime with columns for each feature's mean, std, etc.
        
    Notes
    -----
    Statistics are computed on historical data and should not be interpreted
    as predictive of future regime behavior.
    """
    stats_list = []
    
    for regime in np.unique(regimes):
        regime_mask = regimes == regime
        regime_features = features[regime_mask]
        
        stats = {
            'regime': regime,
            'count': regime_mask.sum(),
            'frequency': regime_mask.sum() / len(regimes)
        }
        
        # Calculate statistics for each feature
        for col in features.columns:
            stats[f'{col}_mean'] = regime_features[col].mean()
            stats[f'{col}_std'] = regime_features[col].std()
            stats[f'{col}_min'] = regime_features[col].min()
            stats[f'{col}_max'] = regime_features[col].max()
            
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def assign_regime_labels(
    regime_stats: pd.DataFrame,
    sort_by: str = 'rolling_vol_mean'
) -> Dict[int, str]:
    """
    Assign interpretable labels to regimes based on their characteristics.
    
    Parameters
    ----------
    regime_stats : pd.DataFrame
        Statistics from calculate_regime_statistics
    sort_by : str, default='rolling_vol_mean'
        Feature to use for sorting regimes
        
    Returns
    -------
    dict
        Mapping from regime number to descriptive label
        
    Notes
    -----
    Labels are descriptive only and should not be interpreted as predictive
    categories. Market regimes are statistical constructs identified from
    historical data.
    """
    # Sort regimes by the specified feature
    sorted_stats = regime_stats.sort_values(sort_by)
    
    if len(sorted_stats) == 2:
        labels = ['Low Volatility', 'High Volatility']
    elif len(sorted_stats) == 3:
        labels = ['Low Volatility', 'Medium Volatility', 'High Volatility']
    elif len(sorted_stats) == 4:
        labels = ['Very Low Vol', 'Low Vol', 'High Vol', 'Very High Vol']
    else:
        labels = [f'Regime {i+1}' for i in range(len(sorted_stats))]
    
    regime_map = {}
    for idx, (_, row) in enumerate(sorted_stats.iterrows()):
        regime_map[int(row['regime'])] = labels[idx]
    
    return regime_map


def calculate_regime_transitions(
    regimes: np.ndarray,
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Analyze regime transition events.
    
    Parameters
    ----------
    regimes : np.ndarray
        Regime labels for each observation
    dates : pd.DatetimeIndex
        Corresponding dates
        
    Returns
    -------
    pd.DataFrame
        Transition events with columns: date, from_regime, to_regime
        
    Notes
    -----
    This identifies historical regime changes. Past transitions do not
    predict future regime changes.
    """
    transitions = []
    
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append({
                'date': dates[i],
                'from_regime': regimes[i-1],
                'to_regime': regimes[i]
            })
    
    return pd.DataFrame(transitions)


def get_regime_summary(
    features: pd.DataFrame,
    regimes: np.ndarray,
    regime_labels: Dict[int, str]
) -> str:
    """
    Generate a text summary of regime characteristics.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix
    regimes : np.ndarray
        Regime labels
    regime_labels : dict
        Mapping from regime numbers to descriptive labels
        
    Returns
    -------
    str
        Formatted text summary
        
    Notes
    -----
    Summary is for research and educational purposes. It describes historical
    patterns and should not be used for trading decisions.
    """
    stats = calculate_regime_statistics(features, regimes)
    
    summary_lines = [
        "Market Regime Analysis Summary",
        "=" * 50,
        "",
        f"Analysis based on {len(regimes)} observations",
        f"Number of regimes detected: {len(np.unique(regimes))}",
        ""
    ]
    
    for _, row in stats.iterrows():
        regime_num = int(row['regime'])
        label = regime_labels.get(regime_num, f"Regime {regime_num}")
        
        summary_lines.extend([
            f"{label}:",
            f"  Frequency: {row['frequency']:.1%} of observations",
            f"  Avg Return: {row['log_return_mean']*100:.3f}% (daily)",
            f"  Avg Volatility: {row['rolling_vol_mean']:.3f} (annualized)",
            ""
        ])
    
    summary_lines.append(
        "Note: This analysis describes historical patterns in market data. "
        "It is a research tool and should not be used for trading decisions."
    )
    
    return "\n".join(summary_lines)
