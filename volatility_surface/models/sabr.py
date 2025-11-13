"""
SABR (Stochastic Alpha Beta Rho) volatility model implementation.

The SABR model is widely used in interest rate and FX markets for modeling
the volatility smile. It has four parameters: alpha, beta, rho, and nu.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class SABRModel:
    """
    SABR volatility model for pricing options and calibrating to market data.
    
    The model captures the volatility smile through the relationship:
    dF = sigma * F^beta * dW1
    dsigma = nu * sigma * dW2
    where dW1 and dW2 have correlation rho.
    """
    
    def __init__(self):
        """Initialize SABR model with default parameters."""
        self.alpha: Optional[float] = None  # Initial volatility
        self.beta: float = 0.5  # Elasticity parameter (often fixed)
        self.rho: Optional[float] = None  # Correlation
        self.nu: Optional[float] = None  # Volatility of volatility
        
    def implied_volatility(
        self,
        forward: float,
        strike: float,
        time_to_maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> float:
        """
        Calculate implied volatility using SABR formula (Hagan et al. approximation).
        
        Args:
            forward: Forward price
            strike: Strike price
            time_to_maturity: Time to maturity in years
            alpha: Initial volatility parameter
            beta: Elasticity parameter (0 <= beta <= 1)
            rho: Correlation parameter (-1 <= rho <= 1)
            nu: Volatility of volatility parameter
            
        Returns:
            Implied volatility
        """
        # Handle ATM case separately for numerical stability
        if abs(strike - forward) < 1e-10:
            return self._atm_volatility(forward, time_to_maturity, alpha, beta, rho, nu)
        
        # Calculate log-moneyness
        log_moneyness = np.log(forward / strike)
        
        # Calculate FK average
        fk = (forward * strike) ** 0.5
        
        # Calculate z parameter
        z = (nu / alpha) * (forward * strike) ** ((1 - beta) / 2) * log_moneyness
        
        # Calculate x(z) using approximation that handles small z
        if abs(z) < 1e-6:
            x_z = 1.0
        else:
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            x_z = z / x_z if abs(x_z) > 1e-10 else 1.0
        
        # Main SABR formula components
        numerator = alpha
        denominator = (forward * strike) ** ((1 - beta) / 2) * (
            1 + ((1 - beta)**2 / 24) * log_moneyness**2 +
            ((1 - beta)**4 / 1920) * log_moneyness**4
        )
        
        # Time adjustment term
        time_adj = 1 + time_to_maturity * (
            ((1 - beta)**2 / 24) * (alpha**2 / fk**(2 - 2 * beta)) +
            (0.25 * rho * beta * nu * alpha / fk**(1 - beta)) +
            ((2 - 3 * rho**2) / 24) * nu**2
        )
        
        implied_vol = (numerator / denominator) * x_z * time_adj
        
        return max(implied_vol, 1e-8)  # Ensure positive volatility
    
    def _atm_volatility(
        self,
        forward: float,
        time_to_maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> float:
        """
        Calculate ATM (at-the-money) implied volatility.
        
        Args:
            forward: Forward price
            time_to_maturity: Time to maturity in years
            alpha: Initial volatility parameter
            beta: Elasticity parameter
            rho: Correlation parameter
            nu: Volatility of volatility parameter
            
        Returns:
            ATM implied volatility
        """
        time_adj = 1 + time_to_maturity * (
            ((1 - beta)**2 / 24) * (alpha**2 / forward**(2 - 2 * beta)) +
            (0.25 * rho * beta * nu * alpha / forward**(1 - beta)) +
            ((2 - 3 * rho**2) / 24) * nu**2
        )
        
        atm_vol = (alpha / forward**(1 - beta)) * time_adj
        
        return max(atm_vol, 1e-8)
    
    def set_parameters(self, alpha: float, beta: float, rho: float, nu: float):
        """
        Set SABR model parameters.
        
        Args:
            alpha: Initial volatility parameter
            beta: Elasticity parameter
            rho: Correlation parameter
            nu: Volatility of volatility parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current SABR model parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu
        }
    
    def compute_surface(
        self,
        forward: float,
        strikes: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Compute volatility surface for given strikes and maturities.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            maturities: Array of time to maturity values
            
        Returns:
            2D array of implied volatilities (strikes x maturities)
        """
        if self.alpha is None or self.rho is None or self.nu is None:
            raise ValueError("Model parameters must be set before computing surface")
        
        surface = np.zeros((len(strikes), len(maturities)))
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                surface[i, j] = self.implied_volatility(
                    forward, strike, maturity,
                    self.alpha, self.beta, self.rho, self.nu
                )
        
        return surface
