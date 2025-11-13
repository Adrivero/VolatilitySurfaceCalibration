"""
Market data generation and handling utilities.

Provides synthetic market data generation for testing and demonstration.
"""

import numpy as np
from typing import Tuple, Dict


class MarketDataGenerator:
    """
    Generate synthetic market volatility data for testing.
    
    Can generate data with realistic volatility smile and term structure
    characteristics.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize market data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_sabr_data(
        self,
        forward: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        alpha: float = 0.3,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4,
        noise_level: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic market data using SABR model.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            maturities: Array of maturities
            alpha: SABR alpha parameter
            beta: SABR beta parameter
            rho: SABR rho parameter
            nu: SABR nu parameter
            noise_level: Standard deviation of noise to add (as fraction of vol)
            
        Returns:
            Tuple of (strikes, maturities, volatilities)
        """
        from ..models.sabr import SABRModel
        
        model = SABRModel()
        volatilities = np.zeros(len(strikes))
        
        for i, (strike, maturity) in enumerate(zip(strikes, maturities)):
            vol = model.implied_volatility(
                forward, strike, maturity, alpha, beta, rho, nu
            )
            
            # Add noise if requested
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * vol)
                vol = max(vol + noise, 1e-6)
            
            volatilities[i] = vol
        
        return strikes, maturities, volatilities
    
    def generate_svi_data(
        self,
        forward: float,
        strikes: np.ndarray,
        time_to_maturity: float,
        a: float = 0.04,
        b: float = 0.1,
        rho: float = -0.4,
        m: float = 0.0,
        sigma: float = 0.2,
        noise_level: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic market data using SVI model.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            time_to_maturity: Time to maturity
            a: SVI a parameter
            b: SVI b parameter
            rho: SVI rho parameter
            m: SVI m parameter
            sigma: SVI sigma parameter
            noise_level: Standard deviation of noise to add
            
        Returns:
            Tuple of (strikes, volatilities)
        """
        from ..models.svi import SVIModel
        
        model = SVIModel()
        volatilities = np.zeros(len(strikes))
        
        for i, strike in enumerate(strikes):
            vol = model.implied_volatility(
                forward, strike, time_to_maturity, a, b, rho, m, sigma
            )
            
            # Add noise if requested
            if noise_level > 0:
                noise = np.random.normal(0, noise_level)
                vol = max(vol + noise, 1e-6)
            
            volatilities[i] = vol
        
        return strikes, volatilities
    
    def generate_smile_data(
        self,
        forward: float = 100.0,
        num_strikes: int = 15,
        strike_range: Tuple[float, float] = (0.8, 1.2),
        time_to_maturity: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate a realistic volatility smile for a single maturity.
        
        Args:
            forward: Forward price
            num_strikes: Number of strike points
            strike_range: Relative strike range (as fraction of forward)
            time_to_maturity: Time to maturity in years
            
        Returns:
            Dictionary with 'strikes', 'maturities', and 'volatilities'
        """
        # Generate strikes
        strikes = np.linspace(
            forward * strike_range[0],
            forward * strike_range[1],
            num_strikes
        )
        
        maturities = np.full(num_strikes, time_to_maturity)
        
        # Generate volatilities using SABR with realistic parameters
        _, _, volatilities = self.generate_sabr_data(
            forward=forward,
            strikes=strikes,
            maturities=maturities,
            alpha=0.25 * forward**0.5,
            beta=0.5,
            rho=-0.3,
            nu=0.4,
            noise_level=0.01
        )
        
        return {
            "forward": forward,
            "strikes": strikes,
            "maturities": maturities,
            "volatilities": volatilities
        }
    
    def generate_surface_data(
        self,
        forward: float = 100.0,
        num_strikes: int = 11,
        num_maturities: int = 6,
        strike_range: Tuple[float, float] = (0.8, 1.2),
        maturity_range: Tuple[float, float] = (0.25, 2.0)
    ) -> Dict[str, np.ndarray]:
        """
        Generate a realistic volatility surface.
        
        Args:
            forward: Forward price
            num_strikes: Number of strike points
            num_maturities: Number of maturity points
            strike_range: Relative strike range (as fraction of forward)
            maturity_range: Maturity range in years
            
        Returns:
            Dictionary with 'strikes', 'maturities', 'surface' (2D array)
        """
        from ..models.sabr import SABRModel
        
        # Generate grid
        strikes = np.linspace(
            forward * strike_range[0],
            forward * strike_range[1],
            num_strikes
        )
        
        maturities = np.linspace(
            maturity_range[0],
            maturity_range[1],
            num_maturities
        )
        
        # Generate surface using SABR
        model = SABRModel()
        alpha = 0.25 * forward**0.5
        beta = 0.5
        rho = -0.3
        nu = 0.4
        
        surface = np.zeros((num_strikes, num_maturities))
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                vol = model.implied_volatility(
                    forward, strike, maturity, alpha, beta, rho, nu
                )
                
                # Add small noise
                noise = np.random.normal(0, 0.005)
                vol = max(vol + noise, 1e-6)
                
                surface[i, j] = vol
        
        return {
            "forward": forward,
            "strikes": strikes,
            "maturities": maturities,
            "surface": surface
        }
