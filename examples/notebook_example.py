"""
Interactive Example for Jupyter Notebook

This script can be run cell-by-cell in a Jupyter notebook or IPython.
To use in Jupyter:
1. jupyter notebook
2. Copy cells below into notebook
3. Run interactively

Or convert this file to a notebook:
jupyter nbconvert --to notebook notebook_example.py
"""

# %% [markdown]
# # Volatility Surface Calibration Tutorial
# 
# This notebook demonstrates calibration of volatility surfaces using:
# - SABR model
# - SVI model
# - Spline interpolation

# %% [markdown]
# ## Setup and Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
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

# Set matplotlib to inline mode for notebooks
# %matplotlib inline

# %% [markdown]
# ## 1. Generate Market Data

# %%
# Initialize data generator
gen = MarketDataGenerator(seed=42)

# Generate a realistic volatility smile
data = gen.generate_smile_data(
    forward=100.0,
    num_strikes=15,
    strike_range=(0.8, 1.2),
    time_to_maturity=1.0
)

forward = data['forward']
strikes = data['strikes']
maturities = data['maturities']
market_vols = data['volatilities']

print(f"Generated {len(strikes)} market quotes")
print(f"Strike range: [{strikes.min():.2f}, {strikes.max():.2f}]")
print(f"Vol range: [{market_vols.min():.4f}, {market_vols.max():.4f}]")

# %% [markdown]
# ## 2. Calibrate SABR Model

# %%
# Calibrate SABR
sabr_calibrator = Calibrator(model_type="sabr")
sabr_params = sabr_calibrator.calibrate_sabr(
    forward=forward,
    strikes=strikes,
    maturities=maturities,
    market_vols=market_vols,
    beta=0.5
)

print("SABR Parameters:")
for key, value in sabr_params.items():
    if key not in ['success']:
        print(f"  {key}: {value:.6f}")

# Create model and compute volatilities
sabr_model = SABRModel()
sabr_model.set_parameters(
    sabr_params['alpha'], sabr_params['beta'],
    sabr_params['rho'], sabr_params['nu']
)

sabr_vols = np.array([
    sabr_model.implied_volatility(
        forward, strike, 1.0,
        sabr_params['alpha'], sabr_params['beta'],
        sabr_params['rho'], sabr_params['nu']
    )
    for strike in strikes
])

# %% [markdown]
# ## 3. Calibrate SVI Model

# %%
# Calibrate SVI
svi_calibrator = Calibrator(model_type="svi")
svi_params = svi_calibrator.calibrate_svi(
    forward=forward,
    strikes=strikes,
    time_to_maturity=1.0,
    market_vols=market_vols
)

print("SVI Parameters:")
for key, value in svi_params.items():
    if key not in ['success']:
        print(f"  {key}: {value:.6f}")

# Create model and compute volatilities
svi_model = SVIModel()
svi_model.set_parameters(
    svi_params['a'], svi_params['b'],
    svi_params['rho'], svi_params['m'], svi_params['sigma']
)

svi_vols = np.array([
    svi_model.implied_volatility(
        forward, strike, 1.0,
        svi_params['a'], svi_params['b'],
        svi_params['rho'], svi_params['m'], svi_params['sigma']
    )
    for strike in strikes
])

# %% [markdown]
# ## 4. Fit Spline Model

# %%
# Fit spline (create artificial maturities for 2D surface)
spline_model = SplineModel()
maturity_grid = np.array([0.9, 1.0, 1.1])
vol_grid = np.column_stack([market_vols, market_vols, market_vols])

spline_model.fit(
    strikes=strikes,
    maturities=maturity_grid,
    volatilities=vol_grid,
    smoothing=0.0001
)

spline_vols = np.array([
    spline_model.implied_volatility(strike, 1.0)
    for strike in strikes
])

print("Spline model fitted successfully")

# %% [markdown]
# ## 5. Compare Models

# %%
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
    title="Volatility Model Comparison"
)
plt.show()

# %% [markdown]
# ## 6. Error Analysis

# %%
# Compute errors
for name, vols in model_results.items():
    errors = vols - market_vols
    rmse = np.sqrt(np.mean(errors**2)) * 10000  # in bps
    mae = np.mean(np.abs(errors)) * 10000
    max_err = np.max(np.abs(errors)) * 10000
    
    print(f"{name:10s}: RMSE={rmse:6.2f}bp, MAE={mae:6.2f}bp, Max={max_err:6.2f}bp")

# %% [markdown]
# ## 7. Generate Full Surface

# %%
# Generate full surface data
surface_data = gen.generate_surface_data(
    forward=100.0,
    num_strikes=21,
    num_maturities=8,
    strike_range=(0.7, 1.3),
    maturity_range=(0.25, 3.0)
)

# Compute SABR surface
sabr_surface = sabr_model.compute_surface(
    forward=forward,
    strikes=surface_data['strikes'],
    maturities=surface_data['maturities']
)

# Plot 3D surface
fig = plot_volatility_surface(
    strikes=surface_data['strikes'],
    maturities=surface_data['maturities'],
    surface=sabr_surface,
    title="SABR Volatility Surface"
)
plt.show()

# %% [markdown]
# ## 8. Interactive Exploration
# 
# Try modifying parameters and seeing the effects:

# %%
# Example: Vary rho parameter
rho_values = [-0.9, -0.5, 0.0, 0.5, 0.9]
test_strike = 110  # OTM call

print("Effect of rho on OTM call volatility:")
print(f"Strike: {test_strike}, ATM: {forward}")
print()

for rho in rho_values:
    vol = sabr_model.implied_volatility(
        forward, test_strike, 1.0,
        sabr_params['alpha'], sabr_params['beta'],
        rho, sabr_params['nu']
    )
    print(f"  rho = {rho:+5.2f}  =>  vol = {vol:.4f}")

# %% [markdown]
# ## Conclusion
# 
# This notebook demonstrated:
# - Generating synthetic market data
# - Calibrating three different models (SABR, SVI, Spline)
# - Comparing model performance
# - Visualizing volatility surfaces
# - Interactive parameter exploration
# 
# Try experimenting with different parameters, market data, and models!
