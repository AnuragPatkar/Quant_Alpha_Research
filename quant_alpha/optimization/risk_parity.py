"""
Risk Parity Portfolio Optimization
Equal risk contribution from each asset
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """
    Risk Parity Optimization
    
    Goal: Each asset contributes equally to portfolio risk
    
    Risk Contribution_i = w_i * (Σw)_i / sqrt(w^T Σ w)
    
    Objective: Minimize sum of squared differences from target
    
    Example:
        optimizer = RiskParityOptimizer()
        
        weights = optimizer.optimize(
            covariance_matrix=cov_matrix,
            tickers=['AAPL', 'MSFT', 'GOOGL']
        )
    """
    
    def __init__(self, target_risk: Optional[Dict[str, float]] = None):
        """
        Args:
            target_risk: Optional target risk contributions
                        If None, assumes equal (1/N for each asset)
        """
        self.target_risk = target_risk
    
    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        tickers: List[str]
    ) -> Dict[str, float]:
        """
        Optimize for risk parity
        
        Args:
            covariance_matrix: Covariance matrix
            tickers: List of tickers
        
        Returns:
            Dictionary of {ticker: weight}
        """
        # 1. Safety: Intersect tickers with available data
        available_tickers = [t for t in tickers if t in covariance_matrix.index]
        if len(available_tickers) != len(tickers):
            missing = set(tickers) - set(available_tickers)
            logger.warning(f"⚠️ Missing covariance data for: {missing}. Optimizing available assets only.")
            
        if not available_tickers:
            return {}

        n = len(available_tickers)
        # Extract numpy array for faster matrix math
        Sigma = covariance_matrix.loc[available_tickers, available_tickers].values
        
        # Target risk contributions (equal by default)
        if self.target_risk is None:
            target = np.ones(n) / n
        else:
            target = np.array([self.target_risk.get(t, 1.0/n) for t in available_tickers])
            target = target / target.sum()  # Normalize
        
        # Objective function: minimize sum of squared deviations
        def objective(w):
            # Portfolio volatility
            port_var = w @ Sigma @ w
            # FIX: Use epsilon to prevent division by zero
            port_vol = np.sqrt(max(port_var, 1e-12))
            
            # Risk contributions
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / port_vol
            
            # Difference from target
            diff = risk_contrib - target
            
            # FIX: Scale up gradient for better convergence
            return np.sum(diff ** 2) * 1000
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]
        
        # Bounds (long only)
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess (equal weight)
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            optimal_weights = result.x
            
            weight_dict = {
                available_tickers[i]: float(weight)
                for i, weight in enumerate(optimal_weights)
                if weight > 1e-6
            }
            
            # Strict Normalization to prevent 'Insufficient Cash' errors
            total_sum = sum(weight_dict.values())
            if total_sum > 0:
                weight_dict = {k: v / total_sum for k, v in weight_dict.items()}
            
            logger.info(f"Risk parity optimized: {len(weight_dict)} positions")
            
            return weight_dict
        else:
            # ==========================================
            # SMART FALLBACK: Inverse Volatility
            # ==========================================
            logger.warning(f"⚠️ Risk parity failed ({result.message}). Using Inverse Volatility fallback.")
            
            variances = np.diag(Sigma)
            # Avoid division by zero for assets with 0 variance
            volatilities = np.sqrt(np.maximum(variances, 1e-8)) 
            inv_vol = 1.0 / volatilities
            fallback_weights = inv_vol / np.sum(inv_vol)
            
            weight_dict = {
                available_tickers[i]: float(w)
                for i, w in enumerate(fallback_weights)
                if w > 1e-6
            }
            
            # Strict Normalization to prevent 'Insufficient Cash' errors
            total_sum = sum(weight_dict.values())
            if total_sum > 0:
                weight_dict = {k: v / total_sum for k, v in weight_dict.items()}
            
            return weight_dict