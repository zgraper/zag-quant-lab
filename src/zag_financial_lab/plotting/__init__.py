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


# ============================================================================
# Portfolio Risk Visualizations
# ============================================================================

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix"
) -> go.Figure:
    """
    Visualize correlation matrix as a heatmap.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (typically from calculate_correlation_matrix)
    title : str
        Chart title
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    Color scale:
    - Red: Negative correlation (assets move opposite)
    - White: No correlation
    - Blue: Positive correlation (assets move together)
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        template='plotly_white',
        height=500,
        width=600
    )
    
    return fig


def plot_drawdown_curve(
    drawdown: pd.Series,
    title: str = "Portfolio Drawdown"
) -> go.Figure:
    """
    Visualize drawdown over time.
    
    Parameters
    ----------
    drawdown : pd.Series
        Drawdown series (typically from calculate_max_drawdown)
    title : str
        Chart title
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    Drawdown shows the decline from previous peak.
    Lower values indicate larger losses from the peak.
    Periods at 0 indicate new all-time highs.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,  # Convert to percentage
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_rolling_volatility(
    volatility: pd.Series,
    title: str = "Rolling Portfolio Volatility"
) -> go.Figure:
    """
    Visualize rolling volatility over time.
    
    Parameters
    ----------
    volatility : pd.Series
        Rolling volatility series
    title : str
        Chart title
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    Shows how portfolio risk varies over time.
    Higher volatility indicates higher risk periods.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volatility.index,
        y=volatility * 100,  # Convert to percentage
        mode='lines',
        name='Volatility',
        line=dict(color='steelblue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_rolling_correlation(
    rolling_corr: pd.Series,
    asset1: str,
    asset2: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Visualize rolling correlation between two assets.
    
    Parameters
    ----------
    rolling_corr : pd.Series
        Rolling correlation series
    asset1 : str
        First asset name
    asset2 : str
        Second asset name
    title : str, optional
        Chart title (auto-generated if None)
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if title is None:
        title = f"Rolling Correlation: {asset1} vs {asset2}"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr,
        mode='lines',
        name='Correlation',
        line=dict(color='green', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dot", line_color="lightgreen", annotation_text="Corr=0.5")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="lightcoral", annotation_text="Corr=-0.5")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_risk_contribution(
    risk_contrib: pd.Series,
    title: str = "Risk Contribution by Asset"
) -> go.Figure:
    """
    Visualize risk contribution of each asset.
    
    Parameters
    ----------
    risk_contrib : pd.Series
        Risk contribution by asset (from calculate_risk_contribution)
    title : str
        Chart title
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    Shows how much each asset contributes to total portfolio risk.
    Sum of all contributions equals total portfolio volatility.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=risk_contrib.index,
        y=risk_contrib * 100,  # Convert to percentage
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Risk Contribution (% volatility)",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_cumulative_returns(
    returns: pd.Series,
    title: str = "Cumulative Returns"
) -> go.Figure:
    """
    Visualize cumulative returns over time.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    title : str
        Chart title
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    cumulative = (1 + returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=(cumulative - 1) * 100,  # Convert to percentage gain
        mode='lines',
        name='Cumulative Return',
        line=dict(color='navy', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


# ============================================================================
# Stress Testing Visualizations
# ============================================================================

def plot_stress_comparison_equity(
    baseline_returns: pd.Series,
    stressed_returns: pd.Series,
    scenario_name: str = "Stressed Scenario"
) -> go.Figure:
    """
    Compare cumulative returns between baseline and stressed scenarios.
    
    Parameters
    ----------
    baseline_returns : pd.Series
        Baseline portfolio returns
    stressed_returns : pd.Series
        Stressed portfolio returns
    scenario_name : str
        Name of the stress scenario
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Notes
    -----
    Shows how portfolio value evolves under baseline vs stress scenarios.
    """
    baseline_cum = (1 + baseline_returns).cumprod()
    stressed_cum = (1 + stressed_returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=baseline_cum.index,
        y=(baseline_cum - 1) * 100,
        mode='lines',
        name='Baseline',
        line=dict(color='navy', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=stressed_cum.index,
        y=(stressed_cum - 1) * 100,
        mode='lines',
        name=scenario_name,
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title="Cumulative Returns: Baseline vs Stressed",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def plot_stress_comparison_drawdown(
    baseline_returns: pd.Series,
    stressed_returns: pd.Series,
    scenario_name: str = "Stressed Scenario"
) -> go.Figure:
    """
    Compare drawdown between baseline and stressed scenarios.
    
    Parameters
    ----------
    baseline_returns : pd.Series
        Baseline portfolio returns
    stressed_returns : pd.Series
        Stressed portfolio returns
    scenario_name : str
        Name of the stress scenario
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Calculate drawdowns
    baseline_cum = (1 + baseline_returns).cumprod()
    baseline_running_max = baseline_cum.expanding().max()
    baseline_dd = (baseline_cum - baseline_running_max) / baseline_running_max
    
    stressed_cum = (1 + stressed_returns).cumprod()
    stressed_running_max = stressed_cum.expanding().max()
    stressed_dd = (stressed_cum - stressed_running_max) / stressed_running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=baseline_dd.index,
        y=baseline_dd * 100,
        mode='lines',
        name='Baseline',
        line=dict(color='steelblue', width=2),
        fill='tozeroy',
        fillcolor='rgba(70, 130, 180, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=stressed_dd.index,
        y=stressed_dd * 100,
        mode='lines',
        name=scenario_name,
        line=dict(color='red', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title="Drawdown: Baseline vs Stressed",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.01, y=0.01)
    )
    
    return fig


def plot_stress_comparison_volatility(
    baseline_returns: pd.Series,
    stressed_returns: pd.Series,
    window: int = 20,
    scenario_name: str = "Stressed Scenario"
) -> go.Figure:
    """
    Compare rolling volatility between baseline and stressed scenarios.
    
    Parameters
    ----------
    baseline_returns : pd.Series
        Baseline portfolio returns
    stressed_returns : pd.Series
        Stressed portfolio returns
    window : int, default=20
        Rolling window for volatility calculation
    scenario_name : str
        Name of the stress scenario
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    baseline_vol = baseline_returns.rolling(window=window).std() * np.sqrt(252)
    stressed_vol = stressed_returns.rolling(window=window).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=baseline_vol.index,
        y=baseline_vol * 100,
        mode='lines',
        name='Baseline',
        line=dict(color='steelblue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=stressed_vol.index,
        y=stressed_vol * 100,
        mode='lines',
        name=scenario_name,
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"Rolling {window}-Day Volatility: Baseline vs Stressed",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def plot_correlation_comparison(
    baseline_corr: pd.DataFrame,
    stressed_corr: pd.DataFrame
) -> go.Figure:
    """
    Create side-by-side correlation heatmaps for baseline vs stressed.
    
    Parameters
    ----------
    baseline_corr : pd.DataFrame
        Baseline correlation matrix
    stressed_corr : pd.DataFrame
        Stressed correlation matrix
        
    Returns
    -------
    go.Figure
        Plotly figure object with subplots
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Baseline Correlations", "Stressed Correlations"),
        horizontal_spacing=0.15
    )
    
    # Baseline heatmap
    fig.add_trace(
        go.Heatmap(
            z=baseline_corr.values,
            x=baseline_corr.columns,
            y=baseline_corr.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(baseline_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=False,
            name='Baseline'
        ),
        row=1, col=1
    )
    
    # Stressed heatmap
    fig.add_trace(
        go.Heatmap(
            z=stressed_corr.values,
            x=stressed_corr.columns,
            y=stressed_corr.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(stressed_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            name='Stressed'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Correlation Matrix Comparison",
        template='plotly_white',
        height=500,
        width=1000
    )
    
    return fig
