"""
Visualization utilities for volatility surfaces.

Provides plotting functions for volatility surfaces, smiles, and model comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple


def plot_volatility_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    surface: np.ndarray,
    title: str = "Volatility Surface",
    save_path: Optional[str] = None
):
    """
    Plot a 3D volatility surface.
    
    Args:
        strikes: Array of strike prices
        maturities: Array of maturities
        surface: 2D array of volatilities (strikes x maturities)
        title: Plot title
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    X, Y = np.meshgrid(maturities, strikes)
    
    # Plot surface
    surf = ax.plot_surface(
        X, Y, surface,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.8
    )
    
    # Labels and title
    ax.set_xlabel('Time to Maturity (years)', fontsize=10)
    ax.set_ylabel('Strike', fontsize=10)
    ax.set_zlabel('Implied Volatility', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_volatility_smile(
    strikes: np.ndarray,
    volatilities: np.ndarray,
    forward: Optional[float] = None,
    market_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Volatility Smile",
    save_path: Optional[str] = None
):
    """
    Plot a volatility smile for a single maturity.
    
    Args:
        strikes: Array of strike prices
        volatilities: Array of implied volatilities
        forward: Optional forward price (for reference line)
        market_data: Optional tuple of (market_strikes, market_vols) for comparison
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot model curve
    ax.plot(strikes, volatilities, 'b-', linewidth=2, label='Model')
    
    # Plot market data if provided
    if market_data is not None:
        market_strikes, market_vols = market_data
        ax.plot(market_strikes, market_vols, 'ro', markersize=8, 
                label='Market', alpha=0.7)
    
    # Add ATM line if forward is provided
    if forward is not None:
        ax.axvline(x=forward, color='gray', linestyle='--', 
                   linewidth=1, alpha=0.5, label='ATM')
    
    # Labels and formatting
    ax.set_xlabel('Strike', fontsize=11)
    ax.set_ylabel('Implied Volatility', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_models(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    model_results: Dict[str, np.ndarray],
    forward: Optional[float] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None
):
    """
    Compare multiple volatility models against market data.
    
    Args:
        strikes: Array of strike prices
        market_vols: Array of market implied volatilities
        model_results: Dictionary mapping model names to volatility arrays
        forward: Optional forward price
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Volatilities
    ax1.plot(strikes, market_vols, 'ko', markersize=8, 
             label='Market', alpha=0.7, zorder=10)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, model_vols) in enumerate(model_results.items()):
        color = colors[i % len(colors)]
        ax1.plot(strikes, model_vols, '-', color=color, 
                linewidth=2, label=model_name, alpha=0.8)
    
    if forward is not None:
        ax1.axvline(x=forward, color='gray', linestyle='--', 
                   linewidth=1, alpha=0.5, label='ATM')
    
    ax1.set_xlabel('Strike', fontsize=11)
    ax1.set_ylabel('Implied Volatility', fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Errors
    for i, (model_name, model_vols) in enumerate(model_results.items()):
        color = colors[i % len(colors)]
        errors = (model_vols - market_vols) * 10000  # in basis points
        ax2.plot(strikes, errors, 'o-', color=color, 
                linewidth=2, label=model_name, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Strike', fontsize=11)
    ax2.set_ylabel('Error (basis points)', fontsize=11)
    ax2.set_title('Calibration Errors', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_evolution(
    maturities: np.ndarray,
    parameters: Dict[str, np.ndarray],
    title: str = "Parameter Term Structure",
    save_path: Optional[str] = None
):
    """
    Plot how model parameters evolve across maturities.
    
    Args:
        maturities: Array of maturities
        parameters: Dictionary mapping parameter names to arrays of values
        title: Plot title
        save_path: Optional path to save figure
    """
    num_params = len(parameters)
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 3*num_params))
    
    if num_params == 1:
        axes = [axes]
    
    for ax, (param_name, param_values) in zip(axes, parameters.items()):
        ax.plot(maturities, param_values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Time to Maturity (years)', fontsize=10)
        ax.set_ylabel(param_name, fontsize=10)
        ax.set_title(f'{param_name} Term Structure', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(
    strikes: np.ndarray,
    maturities: np.ndarray,
    residuals: np.ndarray,
    title: str = "Calibration Residuals",
    save_path: Optional[str] = None
):
    """
    Plot calibration residuals as a heatmap.
    
    Args:
        strikes: Array of strike prices
        maturities: Array of maturities
        residuals: 2D array of residuals (strikes x maturities)
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(
        residuals * 10000,  # Convert to basis points
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest',
        extent=[maturities[0], maturities[-1], strikes[0], strikes[-1]],
        origin='lower'
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Error (basis points)', fontsize=10)
    
    # Labels
    ax.set_xlabel('Time to Maturity (years)', fontsize=11)
    ax.set_ylabel('Strike', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
