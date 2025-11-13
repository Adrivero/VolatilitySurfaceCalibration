"""Tests for SVI volatility model"""

import unittest
import numpy as np
from volatility_surface.models.svi import SVIModel


class TestSVIModel(unittest.TestCase):
    """Test cases for SVI model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SVIModel()
        self.forward = 100.0
        self.a = 0.04
        self.b = 0.1
        self.rho = -0.4
        self.m = 0.0
        self.sigma = 0.2
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNone(self.model.a)
        self.assertIsNone(self.model.b)
        self.assertIsNone(self.model.rho)
        self.assertIsNone(self.model.m)
        self.assertIsNone(self.model.sigma)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        self.model.set_parameters(self.a, self.b, self.rho, self.m, self.sigma)
        
        self.assertEqual(self.model.a, self.a)
        self.assertEqual(self.model.b, self.b)
        self.assertEqual(self.model.rho, self.rho)
        self.assertEqual(self.model.m, self.m)
        self.assertEqual(self.model.sigma, self.sigma)
    
    def test_total_variance(self):
        """Test total variance calculation."""
        log_moneyness = 0.0  # ATM
        
        total_var = self.model.total_variance(
            log_moneyness, self.a, self.b, self.rho, self.m, self.sigma
        )
        
        # Total variance should be positive
        self.assertGreater(total_var, 0)
    
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        strike = self.forward
        time_to_maturity = 1.0
        
        vol = self.model.implied_volatility(
            self.forward, strike, time_to_maturity,
            self.a, self.b, self.rho, self.m, self.sigma
        )
        
        # Volatility should be positive and reasonable
        self.assertGreater(vol, 0)
        self.assertLess(vol, 2.0)
    
    def test_arbitrage_conditions(self):
        """Test arbitrage-free conditions."""
        # Valid parameters
        self.model.set_parameters(self.a, self.b, self.rho, self.m, self.sigma)
        self.assertTrue(self.model.check_arbitrage_conditions())
        
        # Invalid: negative b
        self.model.set_parameters(self.a, -0.1, self.rho, self.m, self.sigma)
        self.assertFalse(self.model.check_arbitrage_conditions())
        
        # Invalid: |rho| > 1
        self.model.set_parameters(self.a, self.b, 1.5, self.m, self.sigma)
        self.assertFalse(self.model.check_arbitrage_conditions())
    
    def test_compute_surface(self):
        """Test surface computation."""
        self.model.set_parameters(self.a, self.b, self.rho, self.m, self.sigma)
        
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.5, 1.0, 2.0])
        
        surface = self.model.compute_surface(self.forward, strikes, maturities)
        
        # Check shape
        self.assertEqual(surface.shape, (len(strikes), len(maturities)))
        
        # All values should be positive
        self.assertTrue(np.all(surface > 0))


if __name__ == '__main__':
    unittest.main()
