"""
Volatility Surface Calibration Comparison Example

This script demonstrates the calibration and comparison of three different
volatility surface models: SABR, SVI, and Spline-based interpolation.

It generates synthetic market data, calibrates each model using nonlinear
optimization, and compares their performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volatility_surface.models.sabr import SABRModel
from volatility_surface.models.svi import SVIModel
from volatility_surface.models.spline import SplineModel
from volatility_surface.calibration.calibrator import Calibrator
from volatility_surface.utils.market_data import MarketDataGenerator
from volatility_surface.utils.visualization import (
    plot_volatility_smile,
    compare_models,
    plot_volatility_surface
)


def main():
    """Run volatility surface calibration comparison."""
    
    print("=" * 70)
    print("Volatility Surface Calibration Comparison")
    print("=" * 70)
    print()
    
    # ========================================================================
    # 1. Generate Synthetic Market Data
    # ========================================================================
    print("1. Generating synthetic market data...")
    
    forward = 100.0
    time_to_maturity = 1.0
    
    # Generate strikes around ATM
    num_strikes = 15
    strikes = np.linspace(forward * 0.8, forward * 1.2, num_strikes)
    
    # Generate market data using SABR model with known parameters
    data_gen = MarketDataGenerator(seed=42)
    true_params = {
        "alpha": 0.25 * forward**0.5,
        "beta": 0.5,
        "rho": -0.3,
        "nu": 0.4
    }
    
    _, maturities, market_vols = data_gen.generate_sabr_data(
        forward=forward,
        strikes=strikes,
        maturities=np.full(num_strikes, time_to_maturity),
        alpha=true_params["alpha"],
        beta=true_params["beta"],
        rho=true_params["rho"],
        nu=true_params["nu"],
        noise_level=0.01  # Add 1% noise
    )
    
    print(f"   Forward: {forward}")
    print(f"   Time to maturity: {time_to_maturity} years")
    print(f"   Number of strikes: {num_strikes}")
    print(f"   Strike range: [{strikes[0]:.2f}, {strikes[-1]:.2f}]")
    print(f"   Volatility range: [{market_vols.min():.4f}, {market_vols.max():.4f}]")
    print()
    
    # ========================================================================
    # 2. Calibrate SABR Model
    # ========================================================================
    print("2. Calibrating SABR model...")
    
    sabr_calibrator = Calibrator(model_type="sabr")
    sabr_params = sabr_calibrator.calibrate_sabr(
        forward=forward,
        strikes=strikes,
        maturities=maturities,
        market_vols=market_vols,
        beta=0.5  # Fix beta
    )
    
    print(f"   SABR Parameters:")
    print(f"      alpha: {sabr_params['alpha']:.6f}")
    print(f"      beta:  {sabr_params['beta']:.6f} (fixed)")
    print(f"      rho:   {sabr_params['rho']:.6f}")
    print(f"      nu:    {sabr_params['nu']:.6f}")
    print(f"   RMSE: {sabr_params['rmse']:.6f}")
    print(f"   Success: {sabr_params['success']}")
    print()
    
    # Compute SABR model volatilities
    sabr_model = SABRModel()
    sabr_model.set_parameters(
        sabr_params["alpha"],
        sabr_params["beta"],
        sabr_params["rho"],
        sabr_params["nu"]
    )
    sabr_vols = np.array([
        sabr_model.implied_volatility(
            forward, strike, time_to_maturity,
            sabr_params["alpha"], sabr_params["beta"],
            sabr_params["rho"], sabr_params["nu"]
        )
        for strike in strikes
    ])
    
    # ========================================================================
    # 3. Calibrate SVI Model
    # ========================================================================
    print("3. Calibrating SVI model...")
    
    svi_calibrator = Calibrator(model_type="svi")
    svi_params = svi_calibrator.calibrate_svi(
        forward=forward,
        strikes=strikes,
        time_to_maturity=time_to_maturity,
        market_vols=market_vols
    )
    
    print(f"   SVI Parameters:")
    print(f"      a:     {svi_params['a']:.6f}")
    print(f"      b:     {svi_params['b']:.6f}")
    print(f"      rho:   {svi_params['rho']:.6f}")
    print(f"      m:     {svi_params['m']:.6f}")
    print(f"      sigma: {svi_params['sigma']:.6f}")
    print(f"   RMSE: {svi_params['rmse']:.6f}")
    print(f"   Success: {svi_params['success']}")
    print()
    
    # Compute SVI model volatilities
    svi_model = SVIModel()
    svi_model.set_parameters(
        svi_params["a"], svi_params["b"],
        svi_params["rho"], svi_params["m"], svi_params["sigma"]
    )
    svi_vols = np.array([
        svi_model.implied_volatility(
            forward, strike, time_to_maturity,
            svi_params["a"], svi_params["b"],
            svi_params["rho"], svi_params["m"], svi_params["sigma"]
        )
        for strike in strikes
    ])
    
    # ========================================================================
    # 4. Fit Spline Model
    # ========================================================================
    print("4. Fitting spline model...")
    
    spline_model = SplineModel()
    
    # For spline, we need a 2D surface. Create artificial maturities around our target
    # This allows the 2D spline to work properly
    maturity_grid = np.array([time_to_maturity * 0.9, time_to_maturity, time_to_maturity * 1.1])
    vol_grid = np.column_stack([market_vols, market_vols, market_vols])
    
    spline_model.fit(
        strikes=strikes,
        maturities=maturity_grid,
        volatilities=vol_grid,
        smoothing=0.0001  # Small smoothing to handle noise
    )
    
    spline_vols = np.array([
        spline_model.implied_volatility(strike, time_to_maturity)
        for strike in strikes
    ])
    
    # Compute RMSE for spline
    spline_rmse = np.sqrt(np.mean((spline_vols - market_vols)**2))
    print(f"   RMSE: {spline_rmse:.6f}")
    print()
    
    # ========================================================================
    # 5. Compare Models
    # ========================================================================
    print("5. Comparing model performance...")
    print()
    
    # Compute errors
    sabr_errors = sabr_vols - market_vols
    svi_errors = svi_vols - market_vols
    spline_errors = spline_vols - market_vols
    
    print("   Error Statistics (in basis points):")
    print(f"   {'Model':<15} {'RMSE':>10} {'MAE':>10} {'Max Error':>12}")
    print("   " + "-" * 50)
    
    for name, errors in [("SABR", sabr_errors), ("SVI", svi_errors), ("Spline", spline_errors)]:
        rmse = np.sqrt(np.mean(errors**2)) * 10000
        mae = np.mean(np.abs(errors)) * 10000
        max_err = np.max(np.abs(errors)) * 10000
        print(f"   {name:<15} {rmse:>10.2f} {mae:>10.2f} {max_err:>12.2f}")
    
    print()
    
    # ========================================================================
    # 6. Visualize Results
    # ========================================================================
    print("6. Creating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot comparison
    model_results = {
        "SABR": sabr_vols,
        "SVI": svi_vols,
        "Spline": spline_vols
    }
    
    fig = compare_models(
        strikes=strikes,
        market_vols=market_vols,
        model_results=model_results,
        forward=forward,
        title="Volatility Surface Model Comparison",
        save_path=os.path.join(output_dir, "model_comparison.png")
    )
    print(f"   Saved comparison plot to: {output_dir}/model_comparison.png")
    
    # ========================================================================
    # 7. Generate Full Surface Comparison
    # ========================================================================
    print()
    print("7. Generating full volatility surfaces...")
    
    # Generate a fuller surface with multiple maturities
    surface_data = data_gen.generate_surface_data(
        forward=forward,
        num_strikes=21,
        num_maturities=8,
        strike_range=(0.7, 1.3),
        maturity_range=(0.25, 3.0)
    )
    
    # Plot SABR surface
    sabr_surface_model = SABRModel()
    sabr_surface_model.set_parameters(
        sabr_params["alpha"],
        sabr_params["beta"],
        sabr_params["rho"],
        sabr_params["nu"]
    )
    sabr_surface = sabr_surface_model.compute_surface(
        forward=forward,
        strikes=surface_data["strikes"],
        maturities=surface_data["maturities"]
    )
    
    fig = plot_volatility_surface(
        strikes=surface_data["strikes"],
        maturities=surface_data["maturities"],
        surface=sabr_surface,
        title="SABR Volatility Surface",
        save_path=os.path.join(output_dir, "sabr_surface.png")
    )
    print(f"   Saved SABR surface to: {output_dir}/sabr_surface.png")
    
    # Fit and plot spline surface
    spline_surface_model = SplineModel()
    spline_surface_model.fit(
        strikes=surface_data["strikes"],
        maturities=surface_data["maturities"],
        volatilities=surface_data["surface"],
        smoothing=0.0001
    )
    spline_surface = spline_surface_model.compute_surface(
        strikes=surface_data["strikes"],
        maturities=surface_data["maturities"]
    )
    
    fig = plot_volatility_surface(
        strikes=surface_data["strikes"],
        maturities=surface_data["maturities"],
        surface=spline_surface,
        title="Spline-Interpolated Volatility Surface",
        save_path=os.path.join(output_dir, "spline_surface.png")
    )
    print(f"   Saved Spline surface to: {output_dir}/spline_surface.png")
    
    print()
    print("=" * 70)
    print("Calibration comparison completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
