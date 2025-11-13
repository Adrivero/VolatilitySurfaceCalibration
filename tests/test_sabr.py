"""Tests for SABR volatility model"""

import unittest
import numpy as np
from volatility_surface.models.sabr import SABRModel


class TestSABRModel(unittest.TestCase):
    """Test cases for SABR model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SABRModel()
        self.forward = 100.0
        self.alpha = 0.25
        self.beta = 0.5
        self.rho = -0.3
        self.nu = 0.4
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNone(self.model.alpha)
        self.assertEqual(self.model.beta, 0.5)
        self.assertIsNone(self.model.rho)
        self.assertIsNone(self.model.nu)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        self.model.set_parameters(self.alpha, self.beta, self.rho, self.nu)
        
        self.assertEqual(self.model.alpha, self.alpha)
        self.assertEqual(self.model.beta, self.beta)
        self.assertEqual(self.model.rho, self.rho)
        self.assertEqual(self.model.nu, self.nu)
    
    def test_atm_volatility(self):
        """Test ATM volatility calculation."""
        time_to_maturity = 1.0
        
        atm_vol = self.model.implied_volatility(
            self.forward, self.forward, time_to_maturity,
            self.alpha, self.beta, self.rho, self.nu
        )
        
        # ATM volatility should be positive and reasonable
        self.assertGreater(atm_vol, 0)
        self.assertLess(atm_vol, 2.0)
    
    def test_otm_volatility(self):
        """Test OTM volatility calculation."""
        strike = self.forward * 1.1
        time_to_maturity = 1.0
        
        otm_vol = self.model.implied_volatility(
            self.forward, strike, time_to_maturity,
            self.alpha, self.beta, self.rho, self.nu
        )
        
        # OTM volatility should be positive
        self.assertGreater(otm_vol, 0)
    
    def test_volatility_smile(self):
        """Test that model produces a volatility smile."""
        strikes = np.array([80, 90, 100, 110, 120])
        time_to_maturity = 1.0
        
        vols = np.array([
            self.model.implied_volatility(
                self.forward, strike, time_to_maturity,
                self.alpha, self.beta, self.rho, self.nu
            )
            for strike in strikes
        ])
        
        # All volatilities should be positive
        self.assertTrue(np.all(vols > 0))
        
        # With negative rho, OTM puts should have higher vol than ATM
        # (smile effect)
        self.assertGreater(vols[0], vols[2])
    
    def test_compute_surface(self):
        """Test surface computation."""
        self.model.set_parameters(self.alpha, self.beta, self.rho, self.nu)
        
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.5, 1.0, 2.0])
        
        surface = self.model.compute_surface(self.forward, strikes, maturities)
        
        # Check shape
        self.assertEqual(surface.shape, (len(strikes), len(maturities)))
        
        # All values should be positive
        self.assertTrue(np.all(surface > 0))


if __name__ == '__main__':
    unittest.main()
