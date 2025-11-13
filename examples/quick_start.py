"""
Quick Start Example for Volatility Surface Calibration

A simple example demonstrating basic usage of each model.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volatility_surface.models.sabr import SABRModel
from volatility_surface.models.svi import SVIModel
from volatility_surface.models.spline import SplineModel
from volatility_surface.calibration.calibrator import Calibrator


def main():
    print("\n" + "=" * 70)
    print("Volatility Surface Calibration - Quick Start")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # Example 1: SABR Model
    # ========================================================================
    print("Example 1: SABR Model")
    print("-" * 70)
    
    # Market data
    forward = 100.0
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    market_vols = np.array([0.25, 0.21, 0.20, 0.21, 0.25])
    
    # Calibrate SABR
    calibrator = Calibrator(model_type="sabr")
    params = calibrator.calibrate_sabr(
        forward=forward,
        strikes=strikes,
        maturities=maturities,
        market_vols=market_vols,
        beta=0.5  # Fix beta parameter
    )
    
    print(f"Calibrated SABR parameters:")
    print(f"  alpha = {params['alpha']:.4f}")
    print(f"  beta  = {params['beta']:.4f} (fixed)")
    print(f"  rho   = {params['rho']:.4f}")
    print(f"  nu    = {params['nu']:.4f}")
    print(f"  RMSE  = {params['rmse']:.6f}")
    
    # Use the calibrated model
    model = SABRModel()
    model.set_parameters(
        params['alpha'], params['beta'],
        params['rho'], params['nu']
    )
    
    # Price a new option
    new_strike = 105
    vol = model.implied_volatility(
        forward, new_strike, 1.0,
        params['alpha'], params['beta'],
        params['rho'], params['nu']
    )
    print(f"\nImplied vol at strike {new_strike}: {vol:.4f}")
    print()
    
    # ========================================================================
    # Example 2: SVI Model
    # ========================================================================
    print("Example 2: SVI Model")
    print("-" * 70)
    
    # Calibrate SVI
    svi_calibrator = Calibrator(model_type="svi")
    svi_params = svi_calibrator.calibrate_svi(
        forward=forward,
        strikes=strikes,
        time_to_maturity=1.0,
        market_vols=market_vols
    )
    
    print(f"Calibrated SVI parameters:")
    print(f"  a     = {svi_params['a']:.4f}")
    print(f"  b     = {svi_params['b']:.4f}")
    print(f"  rho   = {svi_params['rho']:.4f}")
    print(f"  m     = {svi_params['m']:.4f}")
    print(f"  sigma = {svi_params['sigma']:.4f}")
    print(f"  RMSE  = {svi_params['rmse']:.6f}")
    
    # Check arbitrage conditions
    svi_model = SVIModel()
    svi_model.set_parameters(
        svi_params['a'], svi_params['b'],
        svi_params['rho'], svi_params['m'], svi_params['sigma']
    )
    
    is_arbitrage_free = svi_model.check_arbitrage_conditions()
    print(f"\nArbitrage-free: {is_arbitrage_free}")
    print()
    
    # ========================================================================
    # Example 3: Spline Model
    # ========================================================================
    print("Example 3: Spline Interpolation")
    print("-" * 70)
    
    # Create a 2D surface for spline fitting
    strikes_2d = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    maturities_2d = np.array([0.5, 1.0, 2.0])
    
    # Create sample volatility surface
    vols_2d = np.array([
        [0.28, 0.26, 0.24],
        [0.25, 0.23, 0.21],
        [0.23, 0.21, 0.19],
        [0.21, 0.19, 0.18],
        [0.20, 0.18, 0.17],
        [0.21, 0.19, 0.18],
        [0.23, 0.21, 0.19],
        [0.25, 0.23, 0.21],
        [0.28, 0.26, 0.24]
    ])
    
    # Fit spline
    spline_model = SplineModel()
    spline_model.fit(
        strikes=strikes_2d,
        maturities=maturities_2d,
        volatilities=vols_2d,
        smoothing=0.0
    )
    
    print("Spline model fitted successfully")
    
    # Interpolate at a new point
    new_strike = 97.5
    new_maturity = 1.5
    vol = spline_model.implied_volatility(new_strike, new_maturity)
    print(f"\nInterpolated vol at strike {new_strike}, maturity {new_maturity}: {vol:.4f}")
    print()
    
    # ========================================================================
    # Example 4: Generate Full Surface
    # ========================================================================
    print("Example 4: Computing Full Volatility Surface")
    print("-" * 70)
    
    # Define a grid
    strike_grid = np.linspace(80, 120, 21)
    maturity_grid = np.array([0.5, 1.0, 2.0])
    
    # Compute SABR surface
    surface = model.compute_surface(forward, strike_grid, maturity_grid)
    
    print(f"Computed surface shape: {surface.shape}")
    print(f"  Strikes: {len(strike_grid)}")
    print(f"  Maturities: {len(maturity_grid)}")
    print(f"  Volatility range: [{surface.min():.4f}, {surface.max():.4f}]")
    print()
    
    print("=" * 70)
    print("Quick start examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
