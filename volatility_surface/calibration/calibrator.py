"""
Calibration engine for volatility surface models using nonlinear optimization.

This module provides tools to calibrate SABR, SVI, and other parametric models
to market volatility data using scipy's optimization routines.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable
import warnings


class Calibrator:
    """
    Calibration engine for volatility surface models.
    
    Uses nonlinear optimization to find model parameters that best fit
    observed market volatilities.
    """
    
    def __init__(self, model_type: str = "sabr"):
        """
        Initialize calibrator for a specific model type.
        
        Args:
            model_type: Type of model to calibrate ("sabr", "svi", or "spline")
        """
        self.model_type = model_type.lower()
        self.calibration_result: Optional[Dict] = None
        
    def calibrate_sabr(
        self,
        forward: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_vols: np.ndarray,
        beta: float = 0.5,
        method: str = "L-BFGS-B",
        initial_guess: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calibrate SABR model parameters to market volatilities.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            maturities: Array of maturities (same length as strikes)
            market_vols: Array of market implied volatilities
            beta: Beta parameter (typically fixed)
            method: Optimization method
            initial_guess: Initial parameter guesses
            
        Returns:
            Dictionary of calibrated parameters
        """
        from ..models.sabr import SABRModel
        
        model = SABRModel()
        model.beta = beta
        
        # Set initial guess
        if initial_guess is None:
            alpha_init = np.mean(market_vols) * forward**(1 - beta)
            initial_guess = {
                "alpha": alpha_init,
                "rho": 0.0,
                "nu": 0.3
            }
        
        x0 = np.array([initial_guess["alpha"], initial_guess["rho"], initial_guess["nu"]])
        
        # Parameter bounds
        bounds = [
            (1e-6, 10.0),  # alpha > 0
            (-0.999, 0.999),  # -1 < rho < 1
            (1e-6, 10.0)  # nu > 0
        ]
        
        # Objective function: sum of squared errors
        def objective(params):
            alpha, rho, nu = params
            
            predicted_vols = np.array([
                model.implied_volatility(
                    forward, strike, maturity, alpha, beta, rho, nu
                )
                for strike, maturity in zip(strikes, maturities)
            ])
            
            # Weighted least squares (can adjust weights)
            errors = predicted_vols - market_vols
            return np.sum(errors**2)
        
        # Run optimization
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract calibrated parameters
        alpha_cal, rho_cal, nu_cal = result.x
        
        calibrated_params = {
            "alpha": alpha_cal,
            "beta": beta,
            "rho": rho_cal,
            "nu": nu_cal,
            "rmse": np.sqrt(result.fun / len(market_vols)),
            "success": result.success
        }
        
        self.calibration_result = calibrated_params
        return calibrated_params
    
    def calibrate_svi(
        self,
        forward: float,
        strikes: np.ndarray,
        time_to_maturity: float,
        market_vols: np.ndarray,
        method: str = "L-BFGS-B",
        initial_guess: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calibrate SVI model parameters to market volatilities for a single maturity.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            time_to_maturity: Time to maturity in years
            market_vols: Array of market implied volatilities
            method: Optimization method
            initial_guess: Initial parameter guesses
            
        Returns:
            Dictionary of calibrated parameters
        """
        from ..models.svi import SVIModel
        
        model = SVIModel()
        
        # Set initial guess
        if initial_guess is None:
            avg_vol = np.mean(market_vols)
            initial_guess = {
                "a": avg_vol**2 * time_to_maturity * 0.5,
                "b": 0.1,
                "rho": -0.3,
                "m": 0.0,
                "sigma": 0.2
            }
        
        x0 = np.array([
            initial_guess["a"],
            initial_guess["b"],
            initial_guess["rho"],
            initial_guess["m"],
            initial_guess["sigma"]
        ])
        
        # Parameter bounds
        bounds = [
            (1e-6, 1.0),  # a
            (1e-6, 2.0),  # b > 0
            (-0.999, 0.999),  # -1 < rho < 1
            (-0.5, 0.5),  # m
            (1e-6, 1.0)  # sigma > 0
        ]
        
        # Objective function
        def objective(params):
            a, b, rho, m, sigma = params
            
            predicted_vols = np.array([
                model.implied_volatility(
                    forward, strike, time_to_maturity, a, b, rho, m, sigma
                )
                for strike in strikes
            ])
            
            errors = predicted_vols - market_vols
            return np.sum(errors**2)
        
        # Run optimization
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract calibrated parameters
        a_cal, b_cal, rho_cal, m_cal, sigma_cal = result.x
        
        calibrated_params = {
            "a": a_cal,
            "b": b_cal,
            "rho": rho_cal,
            "m": m_cal,
            "sigma": sigma_cal,
            "rmse": np.sqrt(result.fun / len(market_vols)),
            "success": result.success
        }
        
        self.calibration_result = calibrated_params
        return calibrated_params
    
    def calibrate_sabr_global(
        self,
        forward: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_vols: np.ndarray,
        beta: float = 0.5
    ) -> Dict[str, float]:
        """
        Calibrate SABR model using global optimization (differential evolution).
        
        This is slower but more robust for difficult calibration problems.
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            maturities: Array of maturities
            market_vols: Array of market implied volatilities
            beta: Beta parameter (typically fixed)
            
        Returns:
            Dictionary of calibrated parameters
        """
        from ..models.sabr import SABRModel
        
        model = SABRModel()
        model.beta = beta
        
        # Parameter bounds for global optimization
        bounds = [
            (1e-3, 5.0),  # alpha
            (-0.95, 0.95),  # rho
            (1e-3, 5.0)  # nu
        ]
        
        # Objective function
        def objective(params):
            alpha, rho, nu = params
            
            predicted_vols = np.array([
                model.implied_volatility(
                    forward, strike, maturity, alpha, beta, rho, nu
                )
                for strike, maturity in zip(strikes, maturities)
            ])
            
            errors = predicted_vols - market_vols
            return np.sum(errors**2)
        
        # Run global optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=500,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        # Extract calibrated parameters
        alpha_cal, rho_cal, nu_cal = result.x
        
        calibrated_params = {
            "alpha": alpha_cal,
            "beta": beta,
            "rho": rho_cal,
            "nu": nu_cal,
            "rmse": np.sqrt(result.fun / len(market_vols)),
            "success": result.success
        }
        
        self.calibration_result = calibrated_params
        return calibrated_params
    
    def compute_calibration_errors(
        self,
        model,
        forward: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_vols: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute various error metrics for a calibrated model.
        
        Args:
            model: Calibrated model instance
            forward: Forward price
            strikes: Array of strike prices
            maturities: Array of maturities
            market_vols: Array of market implied volatilities
            
        Returns:
            Dictionary of error metrics
        """
        # Compute model volatilities
        if hasattr(model, 'implied_volatility'):
            if self.model_type == "sabr":
                params = model.get_parameters()
                model_vols = np.array([
                    model.implied_volatility(
                        forward, strike, maturity,
                        params["alpha"], params["beta"], params["rho"], params["nu"]
                    )
                    for strike, maturity in zip(strikes, maturities)
                ])
            elif self.model_type == "svi":
                params = model.get_parameters()
                model_vols = np.array([
                    model.implied_volatility(
                        forward, strike, maturity,
                        params["a"], params["b"], params["rho"], params["m"], params["sigma"]
                    )
                    for strike, maturity in zip(strikes, maturities)
                ])
            elif self.model_type == "spline":
                model_vols = np.array([
                    model.implied_volatility(strike, maturity)
                    for strike, maturity in zip(strikes, maturities)
                ])
        
        # Compute errors
        errors = model_vols - market_vols
        
        return {
            "rmse": np.sqrt(np.mean(errors**2)),
            "mae": np.mean(np.abs(errors)),
            "max_error": np.max(np.abs(errors)),
            "mean_error": np.mean(errors)
        }
