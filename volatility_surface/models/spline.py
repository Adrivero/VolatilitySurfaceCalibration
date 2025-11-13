"""
Spline-based volatility surface interpolation.

This module provides cubic spline interpolation for volatility surfaces,
offering a flexible non-parametric approach to surface construction.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from typing import Optional, Tuple


class SplineModel:
    """
    Spline-based volatility surface model using cubic interpolation.
    
    This model uses 2D cubic splines to interpolate volatility values
    across strikes and maturities, providing smooth surfaces.
    """
    
    def __init__(self):
        """Initialize spline model."""
        self.calibrated_strikes: Optional[np.ndarray] = None
        self.calibrated_maturities: Optional[np.ndarray] = None
        self.calibrated_vols: Optional[np.ndarray] = None
        self.spline: Optional[RectBivariateSpline] = None
        
    def fit(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        volatilities: np.ndarray,
        smoothing: float = 0.0
    ):
        """
        Fit a 2D cubic spline to volatility data.
        
        Args:
            strikes: Array of strike prices (must be sorted)
            maturities: Array of maturities (must be sorted)
            volatilities: 2D array of volatilities (strikes x maturities)
            smoothing: Smoothing factor (0 = interpolating spline)
        """
        # Validate inputs
        if len(strikes.shape) != 1 or len(maturities.shape) != 1:
            raise ValueError("Strikes and maturities must be 1D arrays")
        
        if volatilities.shape != (len(strikes), len(maturities)):
            raise ValueError(
                f"Volatilities shape {volatilities.shape} does not match "
                f"strikes x maturities ({len(strikes)}, {len(maturities)})"
            )
        
        # Ensure sorted
        if not np.all(strikes[:-1] <= strikes[1:]):
            raise ValueError("Strikes must be sorted in ascending order")
        if not np.all(maturities[:-1] <= maturities[1:]):
            raise ValueError("Maturities must be sorted in ascending order")
        
        # Store calibration data
        self.calibrated_strikes = strikes.copy()
        self.calibrated_maturities = maturities.copy()
        self.calibrated_vols = volatilities.copy()
        
        # Fit 2D spline
        # Note: RectBivariateSpline expects (x, y, z) where z[i,j] = f(x[i], y[j])
        self.spline = RectBivariateSpline(
            strikes,
            maturities,
            volatilities,
            s=smoothing,
            kx=min(3, len(strikes) - 1),  # Cubic or lower if insufficient points
            ky=min(3, len(maturities) - 1)
        )
    
    def implied_volatility(
        self,
        strike: float,
        time_to_maturity: float
    ) -> float:
        """
        Interpolate implied volatility at a given strike and maturity.
        
        Args:
            strike: Strike price
            time_to_maturity: Time to maturity in years
            
        Returns:
            Interpolated implied volatility
        """
        if self.spline is None:
            raise ValueError("Model must be fitted before computing volatility")
        
        # Evaluate spline (returns scalar when grid=False)
        vol = float(self.spline(strike, time_to_maturity, grid=False))
        
        return max(vol, 1e-8)  # Ensure positive volatility
    
    def compute_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Compute volatility surface for given strikes and maturities.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of time to maturity values
            
        Returns:
            2D array of implied volatilities (strikes x maturities)
        """
        if self.spline is None:
            raise ValueError("Model must be fitted before computing surface")
        
        # Evaluate spline on grid
        surface = self.spline(strikes, maturities, grid=True)
        
        # Ensure positive volatilities
        surface = np.maximum(surface, 1e-8)
        
        return surface
    
    def get_calibration_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the calibration data used to fit the spline.
        
        Returns:
            Tuple of (strikes, maturities, volatilities) or None if not fitted
        """
        if self.calibrated_strikes is None:
            return None
        
        return (
            self.calibrated_strikes.copy(),
            self.calibrated_maturities.copy(),
            self.calibrated_vols.copy()
        )
    
    def extrapolate_flat(
        self,
        strike: float,
        time_to_maturity: float
    ) -> float:
        """
        Compute volatility with flat extrapolation beyond calibration range.
        
        Args:
            strike: Strike price
            time_to_maturity: Time to maturity in years
            
        Returns:
            Implied volatility (extrapolated if outside calibration range)
        """
        if self.spline is None:
            raise ValueError("Model must be fitted before computing volatility")
        
        # Clip to calibration range
        strike_clipped = np.clip(
            strike,
            self.calibrated_strikes[0],
            self.calibrated_strikes[-1]
        )
        maturity_clipped = np.clip(
            time_to_maturity,
            self.calibrated_maturities[0],
            self.calibrated_maturities[-1]
        )
        
        # Evaluate at clipped point (returns scalar when grid=False)
        vol = float(self.spline(strike_clipped, maturity_clipped, grid=False))
        
        return max(vol, 1e-8)
