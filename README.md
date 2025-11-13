# Volatility Surface Calibration

A comprehensive Python library for building and calibrating volatility surfaces to market data using nonlinear optimization. This project compares three popular methods: SABR, SVI, and spline-based interpolation.

## Features

- **Three Calibration Methods**:
  - **SABR** (Stochastic Alpha Beta Rho): Industry-standard model capturing volatility smile dynamics
  - **SVI** (Stochastic Volatility Inspired): Parametric model ensuring arbitrage-free surfaces
  - **Spline Interpolation**: Non-parametric cubic spline interpolation for flexible surface fitting

- **Nonlinear Optimization**: Advanced calibration using scipy's optimization routines
- **Comprehensive Visualization**: 3D surface plots, volatility smiles, and model comparison charts
- **Synthetic Data Generation**: Built-in market data generators for testing
- **Full Test Coverage**: Unit tests for all models and calibration methods

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Adrivero/VolatilitySurfaceCalibration.git
cd VolatilitySurfaceCalibration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package (optional):
```bash
pip install -e .
```

## Quick Start

### Running the Example

The included example demonstrates calibration and comparison of all three methods:

```bash
python examples/calibration_comparison.py
```

This will:
1. Generate synthetic market data with realistic volatility smile
2. Calibrate SABR, SVI, and Spline models
3. Compare performance metrics (RMSE, MAE, Max Error)
4. Create visualization plots saved to `examples/output/`

### Using the Library

```python
import numpy as np
from volatility_surface.models.sabr import SABRModel
from volatility_surface.calibration.calibrator import Calibrator

# Market data
forward = 100.0
strikes = np.array([80, 90, 100, 110, 120])
maturities = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
market_vols = np.array([0.25, 0.21, 0.20, 0.21, 0.25])

# Calibrate SABR model
calibrator = Calibrator(model_type="sabr")
params = calibrator.calibrate_sabr(
    forward=forward,
    strikes=strikes,
    maturities=maturities,
    market_vols=market_vols,
    beta=0.5
)

print(f"Calibrated parameters: {params}")
print(f"RMSE: {params['rmse']:.6f}")

# Use calibrated model
model = SABRModel()
model.set_parameters(
    params['alpha'], params['beta'], 
    params['rho'], params['nu']
)

# Compute volatility for a new strike
new_strike = 105
vol = model.implied_volatility(
    forward, new_strike, 1.0,
    params['alpha'], params['beta'],
    params['rho'], params['nu']
)
print(f"Implied vol at strike {new_strike}: {vol:.4f}")
```

## Models Overview

### SABR Model

The SABR (Stochastic Alpha Beta Rho) model is widely used in interest rate and FX markets. It models the forward rate and its volatility as correlated stochastic processes:

```
dF = σ F^β dW₁
dσ = ν σ dW₂
dW₁ dW₂ = ρ dt
```

**Parameters**:
- `alpha`: Initial volatility
- `beta`: Elasticity (typically fixed, e.g., 0.5)
- `rho`: Correlation between forward and volatility
- `nu`: Volatility of volatility

**Use Cases**: Best for capturing realistic smile dynamics in liquid markets

### SVI Model

The SVI (Stochastic Volatility Inspired) model parametrizes total implied variance as a function of log-moneyness:

```
w(k) = a + b(ρ(k - m) + √((k - m)² + σ²))
```

**Parameters**:
- `a`: Overall variance level
- `b`: Slope of smile
- `rho`: Skewness
- `m`: Horizontal shift
- `sigma`: Smoothness

**Use Cases**: Fast calibration, ensures no calendar arbitrage, good for single maturity slices

### Spline Interpolation

Uses 2D cubic splines to interpolate volatilities across strikes and maturities. Non-parametric approach providing smooth surfaces.

**Use Cases**: Most flexible, best when you need exact interpolation of market quotes

## Project Structure

```
VolatilitySurfaceCalibration/
├── volatility_surface/          # Main package
│   ├── models/                  # Volatility models
│   │   ├── sabr.py             # SABR implementation
│   │   ├── svi.py              # SVI implementation
│   │   └── spline.py           # Spline interpolation
│   ├── calibration/            # Calibration engine
│   │   └── calibrator.py       # Nonlinear optimization
│   └── utils/                  # Utilities
│       ├── market_data.py      # Data generation
│       └── visualization.py    # Plotting functions
├── examples/                    # Example scripts
│   └── calibration_comparison.py
├── tests/                       # Unit tests
│   ├── test_sabr.py
│   ├── test_svi.py
│   └── test_spline.py
├── requirements.txt
├── setup.py
└── README.md
```

## Running Tests

Run all unit tests:

```bash
python -m unittest discover tests
```

Run specific model tests:

```bash
python -m unittest tests.test_sabr
python -m unittest tests.test_svi
python -m unittest tests.test_spline
```

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Optimization and interpolation
- **matplotlib**: Visualization
- **pandas**: Data handling (optional)

## Performance Comparison

Typical calibration results for a 15-point volatility smile:

| Model  | RMSE (bps) | Calibration Time | Flexibility |
|--------|-----------|------------------|-------------|
| SABR   | 5-15      | Fast (~0.1s)     | Moderate    |
| SVI    | 3-10      | Fast (~0.1s)     | Moderate    |
| Spline | 1-5       | Very Fast        | High        |

*Note: Results depend on market conditions and number of calibration points*

## Advanced Usage

### Custom Optimization

```python
from volatility_surface.calibration.calibrator import Calibrator

# Use global optimization for difficult problems
calibrator = Calibrator(model_type="sabr")
params = calibrator.calibrate_sabr_global(
    forward=forward,
    strikes=strikes,
    maturities=maturities,
    market_vols=market_vols,
    beta=0.5
)
```

### Surface Generation

```python
from volatility_surface.utils.market_data import MarketDataGenerator

# Generate full surface
gen = MarketDataGenerator(seed=42)
data = gen.generate_surface_data(
    forward=100.0,
    num_strikes=21,
    num_maturities=8,
    strike_range=(0.7, 1.3),
    maturity_range=(0.25, 3.0)
)

# Visualize
from volatility_surface.utils.visualization import plot_volatility_surface

plot_volatility_surface(
    strikes=data['strikes'],
    maturities=data['maturities'],
    surface=data['surface'],
    title="Synthetic Volatility Surface"
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## References

1. Hagan, P. S., et al. (2002). "Managing Smile Risk". Wilmott Magazine.
2. Gatheral, J., & Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces". Quantitative Finance.
3. De Boor, C. (2001). "A Practical Guide to Splines". Springer.

## Author

Adrivero

## Acknowledgments

This implementation is based on industry-standard volatility modeling techniques used in quantitative finance.