"""
SVI (Stochastic Volatility Inspired) model implementation.

The SVI model provides a parametric form for the implied variance smile
as a function of log-moneyness. It ensures no arbitrage and has good
fitting properties.
"""

import numpy as np
from typing import Dict, Optional


class SVIModel:
    """
    SVI (Stochastic Volatility Inspired) volatility model.
    
    The model parametrizes total implied variance as:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    where k is log-moneyness and w is total variance (sigma^2 * T).
    """
    
    def __init__(self):
        """Initialize SVI model with default parameters."""
        self.a: Optional[float] = None  # Overall variance level
        self.b: Optional[float] = None  # Slope of the smile
        self.rho: Optional[float] = None  # Skewness (-1 <= rho <= 1)
        self.m: Optional[float] = None  # Horizontal shift
        self.sigma: Optional[float] = None  # Smoothness parameter
        
    def total_variance(
        self,
        log_moneyness: float,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
    ) -> float:
        """
        Calculate total implied variance using SVI formula.
        
        Args:
            log_moneyness: Log of strike/forward ratio
            a: Overall variance level
            b: Slope parameter (b >= 0)
            rho: Skewness parameter (-1 <= rho <= 1)
            m: Horizontal shift parameter
            sigma: Smoothness parameter (sigma > 0)
            
        Returns:
            Total implied variance
        """
        k = log_moneyness
        delta_k = k - m
        
        # SVI formula
        total_var = a + b * (rho * delta_k + np.sqrt(delta_k**2 + sigma**2))
        
        return max(total_var, 1e-8)  # Ensure positive variance
    
    def implied_volatility(
        self,
        forward: float,
        strike: float,
        time_to_maturity: float,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
    ) -> float:
        """
        Calculate implied volatility from SVI parameters.
        
        Args:
            forward: Forward price
            strike: Strike price
            time_to_maturity: Time to maturity in years
            a: Overall variance level
            b: Slope parameter
            rho: Skewness parameter
            m: Horizontal shift parameter
            sigma: Smoothness parameter
            
        Returns:
            Implied volatility
        """
        # Calculate log-moneyness
        log_moneyness = np.log(strike / forward)
        
        # Get total variance
        total_var = self.total_variance(log_moneyness, a, b, rho, m, sigma)
        
        # Convert to implied volatility
        # total_var = sigma^2 * T, so sigma = sqrt(total_var / T)
        implied_vol = np.sqrt(total_var / time_to_maturity)
        
        return max(implied_vol, 1e-8)
    
    def set_parameters(
        self,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
    ):
        """
        Set SVI model parameters.
        
        Args:
            a: Overall variance level
            b: Slope parameter
            rho: Skewness parameter
            m: Horizontal shift parameter
            sigma: Smoothness parameter
        """
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current SVI model parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma
        }
    
    def check_arbitrage_conditions(self) -> bool:
        """
        Check if current parameters satisfy no-arbitrage conditions.
        
        Returns:
            True if parameters are arbitrage-free, False otherwise
        """
        if any(p is None for p in [self.a, self.b, self.rho, self.m, self.sigma]):
            return False
        
        # Basic conditions
        if self.b < 0:
            return False
        if abs(self.rho) > 1:
            return False
        if self.sigma <= 0:
            return False
        
        # Calendar spread arbitrage condition
        if self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) < 0:
            return False
        
        # Butterfly arbitrage conditions (simplified check)
        if self.b * (1 + abs(self.rho)) > 4:
            return False
        
        return True
    
    def compute_surface(
        self,
        forward: float,
        strikes: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Compute volatility surface for given strikes and maturities.
        
        Note: For simplicity, this uses the same parameters for all maturities.
        In practice, parameters should be calibrated separately for each maturity slice.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            maturities: Array of time to maturity values
            
        Returns:
            2D array of implied volatilities (strikes x maturities)
        """
        if any(p is None for p in [self.a, self.b, self.rho, self.m, self.sigma]):
            raise ValueError("Model parameters must be set before computing surface")
        
        surface = np.zeros((len(strikes), len(maturities)))
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                surface[i, j] = self.implied_volatility(
                    forward, strike, maturity,
                    self.a, self.b, self.rho, self.m, self.sigma
                )
        
        return surface
