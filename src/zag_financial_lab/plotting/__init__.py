"""
Plotting module for ZAG Financial Lab.

This module provides Plotly-based visualizations for market regime analysis.
All charts are designed for research and educational purposes.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional


def plot_price_and_regimes(
    data: pd.DataFrame,
    regimes: np.ndarray,
    regime_labels: Dict[int, str],
    ticker: str = ""
) -> go.Figure:
    """
    Create an interactive plot showing price history colored by regime.
    
    Parameters
    ----------
    data : pd.DataFrame
        Price data with 'Close' column and DatetimeIndex
    regimes : np.ndarray
        Regime labels aligned with data
    regime_labels : dict
        Mapping from regime numbers to descriptive labels
    ticker : str, optional
        Ticker symbol for chart title
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    Visualizations show historical regime classifications and should not
    be interpreted as predictive indicators.
    """
    fig = go.Figure()
    
    # Create a dataframe for easier plotting
    plot_df = pd.DataFrame({
        'date': data.index,
        'price': data['Close'].values,
        'regime': regimes
    })
    
    # Plot each regime separately for color coding
    for regime_num in np.unique(regimes):
        regime_mask = plot_df['regime'] == regime_num
        regime_data = plot_df[regime_mask]
        
        label = regime_labels.get(regime_num, f"Regime {regime_num}")
        
        fig.add_trace(go.Scatter(
            x=regime_data['date'],
            y=regime_data['price'],
            mode='lines+markers',
            name=label,
            marker=dict(size=4),
            line=dict(width=2)
        ))
    
    title = f"Price History with Market Regimes"
    if ticker:
        title = f"{ticker} - {title}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_regime_features(
    features: pd.DataFrame,
    regimes: np.ndarray,
    regime_labels: Dict[int, str]
) -> go.Figure:
    """
    Create a scatter plot of features colored by regime.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with 'log_return' and 'rolling_vol' columns
    regimes : np.ndarray
        Regime labels
    regime_labels : dict
        Mapping from regime numbers to descriptive labels
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    This visualization helps understand the feature space characteristics
    of each detected regime.
    """
    fig = go.Figure()
    
    plot_df = pd.DataFrame({
        'return': features['log_return'].values * 100,  # Convert to percentage
        'volatility': features['rolling_vol'].values,
        'regime': regimes
    })
    
    for regime_num in np.unique(regimes):
        regime_mask = plot_df['regime'] == regime_num
        regime_data = plot_df[regime_mask]
        
        label = regime_labels.get(regime_num, f"Regime {regime_num}")
        
        fig.add_trace(go.Scatter(
            x=regime_data['volatility'],
            y=regime_data['return'],
            mode='markers',
            name=label,
            marker=dict(size=5, opacity=0.6)
        ))
    
    fig.update_layout(
        title="Market Regimes in Feature Space",
        xaxis_title="Rolling Volatility (Annualized)",
        yaxis_title="Daily Log Return (%)",
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_regime_distribution(
    regimes: np.ndarray,
    dates: pd.DatetimeIndex,
    regime_labels: Dict[int, str]
) -> go.Figure:
    """
    Create a timeline showing regime distribution over time.
    
    Parameters
    ----------
    regimes : np.ndarray
        Regime labels
    dates : pd.DatetimeIndex
        Corresponding dates
    regime_labels : dict
        Mapping from regime numbers to descriptive labels
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Map regime numbers to labels for display
    regime_names = [regime_labels.get(r, f"Regime {r}") for r in regimes]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=regimes,
        mode='lines+markers',
        marker=dict(size=3),
        line=dict(width=1),
        text=regime_names,
        hovertemplate='<b>%{text}</b><br>Date: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Market Regime Timeline",
        xaxis_title="Date",
        yaxis_title="Regime",
        yaxis=dict(
            tickmode='array',
            tickvals=list(regime_labels.keys()),
            ticktext=[regime_labels[k] for k in sorted(regime_labels.keys())]
        ),
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_volatility_timeseries(
    features: pd.DataFrame,
    regimes: np.ndarray,
    regime_labels: Dict[int, str]
) -> go.Figure:
    """
    Plot rolling volatility over time, colored by regime.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with 'rolling_vol' column and DatetimeIndex
    regimes : np.ndarray
        Regime labels
    regime_labels : dict
        Mapping from regime numbers to descriptive labels
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    plot_df = pd.DataFrame({
        'date': features.index,
        'volatility': features['rolling_vol'].values,
        'regime': regimes
    })
    
    for regime_num in np.unique(regimes):
        regime_mask = plot_df['regime'] == regime_num
        regime_data = plot_df[regime_mask]
        
        label = regime_labels.get(regime_num, f"Regime {regime_num}")
        
        fig.add_trace(go.Scatter(
            x=regime_data['date'],
            y=regime_data['volatility'],
            mode='lines',
            name=label,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Rolling Volatility Over Time by Regime",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    regime_labels: Dict[int, str]
) -> go.Figure:
    """
    Visualize regime transition probability matrix as a heatmap.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probability matrix from HMM
    regime_labels : dict
        Mapping from regime numbers to descriptive labels
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    labels = [regime_labels.get(i, f"Regime {i}") 
              for i in range(len(transition_matrix))]
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=np.round(transition_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title="Regime Transition Probability Matrix",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        template='plotly_white',
        height=500,
        width=600
    )
    
    return fig
