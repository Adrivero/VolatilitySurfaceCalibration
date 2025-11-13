"""Tests for spline-based volatility model"""

import unittest
import numpy as np
from volatility_surface.models.spline import SplineModel


class TestSplineModel(unittest.TestCase):
    """Test cases for spline model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SplineModel()
        
        # Create sample data
        self.strikes = np.array([90, 95, 100, 105, 110])
        self.maturities = np.array([0.5, 1.0, 2.0])
        self.volatilities = np.array([
            [0.22, 0.20, 0.19],
            [0.21, 0.19, 0.18],
            [0.20, 0.18, 0.17],
            [0.21, 0.19, 0.18],
            [0.22, 0.20, 0.19]
        ])
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNone(self.model.spline)
        self.assertIsNone(self.model.calibrated_strikes)
    
    def test_fit(self):
        """Test spline fitting."""
        self.model.fit(self.strikes, self.maturities, self.volatilities)
        
        self.assertIsNotNone(self.model.spline)
        self.assertIsNotNone(self.model.calibrated_strikes)
        np.testing.assert_array_equal(self.model.calibrated_strikes, self.strikes)
    
    def test_interpolation(self):
        """Test volatility interpolation."""
        self.model.fit(self.strikes, self.maturities, self.volatilities)
        
        # Interpolate at a calibration point
        vol = self.model.implied_volatility(100, 1.0)
        
        # Should be close to the calibrated value
        expected = self.volatilities[2, 1]
        self.assertAlmostEqual(vol, expected, places=2)
    
    def test_compute_surface(self):
        """Test surface computation."""
        self.model.fit(self.strikes, self.maturities, self.volatilities)
        
        new_strikes = np.linspace(90, 110, 10)
        new_maturities = np.linspace(0.5, 2.0, 5)
        
        surface = self.model.compute_surface(new_strikes, new_maturities)
        
        # Check shape
        self.assertEqual(surface.shape, (len(new_strikes), len(new_maturities)))
        
        # All values should be positive
        self.assertTrue(np.all(surface > 0))
    
    def test_flat_extrapolation(self):
        """Test flat extrapolation beyond calibration range."""
        self.model.fit(self.strikes, self.maturities, self.volatilities)
        
        # Extrapolate beyond strike range
        vol_extrap = self.model.extrapolate_flat(120, 1.0)
        
        # Should return edge value
        vol_edge = self.model.implied_volatility(110, 1.0)
        self.assertAlmostEqual(vol_extrap, vol_edge, places=4)
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Unsorted strikes
        unsorted_strikes = np.array([100, 90, 110])
        with self.assertRaises(ValueError):
            self.model.fit(unsorted_strikes, self.maturities, self.volatilities[:3])
        
        # Wrong shape
        wrong_vols = np.array([[0.2, 0.2]])
        with self.assertRaises(ValueError):
            self.model.fit(self.strikes, self.maturities, wrong_vols)


if __name__ == '__main__':
    unittest.main()
