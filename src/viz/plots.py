"""Plotting utilities for visualizing experiment results."""

from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_metrics_over_time(
    results: List[Dict[str, Any]],
    metrics: List[str] = ['stego_detection_rate', 'reasoning_accuracy'],
    save_path: Optional[str] = None,
    use_plotly: bool = True
) -> go.Figure:
    """
    Plot metrics over time (sample progression).
    
    Args:
        results: List of result dictionaries from experiments
        metrics: List of metric names to plot
        save_path: Optional path to save figure
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Plotly figure or matplotlib figure
    """
    # Extract data
    sample_ids = [r.get('sample_id', i) for i, r in enumerate(results)]
    
    if use_plotly:
        fig = go.Figure()
        
        for metric in metrics:
            values = []
            for r in results:
                m = r.get('metrics', {})
                values.append(m.get(metric, 0.0))
            
            fig.add_trace(go.Scatter(
                x=sample_ids,
                y=values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Metrics Over Time',
            xaxis_title='Sample ID',
            yaxis_title='Metric Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics:
            values = []
            for r in results:
                m = r.get('metrics', {})
                values.append(m.get(metric, 0.0))
            
            ax.plot(sample_ids, values, marker='o', label=metric, linewidth=2)
        
        ax.set_xlabel('Sample ID')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_stego_success_rate(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    use_plotly: bool = True
) -> go.Figure:
    """
    Plot steganography success rate distribution.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save figure
        use_plotly: Whether to use Plotly
        
    Returns:
        Figure object
    """
    stego_success = []
    for r in results:
        m = r.get('metrics', {})
        stego_success.append(m.get('stego_success', 0.0))
    
    if use_plotly:
        fig = px.histogram(
            x=stego_success,
            nbins=2,
            title='Steganography Success Rate Distribution',
            labels={'x': 'Stego Success (0=Failed, 1=Success)', 'count': 'Frequency'}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(stego_success, bins=2, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Stego Success (0=Failed, 1=Success)')
        ax.set_ylabel('Frequency')
        ax.set_title('Steganography Success Rate Distribution')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_entropy_distribution(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    use_plotly: bool = True
) -> go.Figure:
    """
    Plot entropy distribution of outputs.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save figure
        use_plotly: Whether to use Plotly
        
    Returns:
        Figure object
    """
    entropies = []
    for r in results:
        m = r.get('metrics', {})
        entropies.append(m.get('entropy', 0.0))
    
    if use_plotly:
        fig = px.histogram(
            x=entropies,
            nbins=30,
            title='Output Entropy Distribution',
            labels={'x': 'Entropy (bits)', 'count': 'Frequency'}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(entropies, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Entropy (bits)')
        ax.set_ylabel('Frequency')
        ax.set_title('Output Entropy Distribution')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_experiment_dashboard(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save HTML dashboard
        
    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Metrics Over Time',
            'Stego Success Rate',
            'Entropy Distribution',
            'Combined Score'
        ),
        specs=[[{"secondary_y": False}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Extract data
    sample_ids = [r.get('sample_id', i) for i, r in enumerate(results)]
    stego_rates = [r.get('metrics', {}).get('stego_detection_rate', 0.0) for r in results]
    accuracies = [r.get('metrics', {}).get('reasoning_accuracy', 0.0) for r in results]
    entropies = [r.get('metrics', {}).get('entropy', 0.0) for r in results]
    combined_scores = [r.get('metrics', {}).get('combined_score', 0.0) for r in results]
    
    # Metrics over time
    fig.add_trace(
        go.Scatter(x=sample_ids, y=stego_rates, name='Stego Rate', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=sample_ids, y=accuracies, name='Accuracy', line=dict(color='red')),
        row=1, col=1
    )
    
    # Stego success histogram
    stego_success = [r.get('metrics', {}).get('stego_success', 0.0) for r in results]
    fig.add_trace(
        go.Histogram(x=stego_success, nbinsx=2, name='Stego Success'),
        row=1, col=2
    )
    
    # Entropy distribution
    fig.add_trace(
        go.Histogram(x=entropies, nbinsx=30, name='Entropy'),
        row=2, col=1
    )
    
    # Combined score
    fig.add_trace(
        go.Scatter(x=sample_ids, y=combined_scores, mode='lines+markers', name='Combined Score'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="StegoBenchy Experiment Dashboard",
        showlegend=True,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

