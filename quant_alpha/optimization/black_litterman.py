"""
Black-Litterman Optimization Model - Production Grade
Blends market equilibrium returns with ML-based Alpha views.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from .mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)

class BlackLittermanModel:
    """
    Black-Litterman Model for Portfolio Optimization.
    
    Mathematical Formulation:
    1. Implied Returns (Prior): 
       $$ \Pi = \delta \Sigma w_{mkt} $$
    2. Posterior Distribution (N(E[R], \Sigma_{post})):
       Updates both Expected Returns and Covariance based on View Confidence.
    """

    def __init__(
        self, 
        tau: float = 0.05, 
        risk_aversion: float = 2.5
    ):
        """
        Args:
            tau: Scalar indicating the uncertainty of the prior (market equilibrium).
                 Typically between 0.01 and 0.05.
            risk_aversion: Market risk aversion parameter (delta).
        """
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.mvo = MeanVarianceOptimizer(risk_aversion=risk_aversion)
        logger.info(f"BlackLittermanModel initialized: tau={tau}, delta={risk_aversion}")

    def get_implied_returns(
        self, 
        covariance_matrix: pd.DataFrame, 
        market_caps: Dict[str, float]
    ) -> pd.Series:
        """
        Calculates the Market Implied Returns (Pi) based on current market cap weights.
        """
        tickers = list(covariance_matrix.columns)
        
        # Calculate market weights with validation
        weights = []
        missing_caps = []
        for t in tickers:
            cap = market_caps.get(t, 0.0)
            weights.append(cap)
            if cap == 0:
                missing_caps.append(t)
        
        if len(missing_caps) > len(tickers) * 0.1:
            logger.warning(f"⚠️ {len(missing_caps)}/{len(tickers)} stocks missing market cap data. Equilibrium may be distorted.")

        total_mcap = sum(weights)
        if total_mcap == 0:
            raise ValueError("Total market cap is zero. Cannot calculate implied returns.")
            
        w_mkt = np.array(weights) / total_mcap
        Sigma = covariance_matrix.values
        
        # Pi = delta * Sigma * w_mkt
        pi = self.risk_aversion * (Sigma @ w_mkt)
        return pd.Series(pi, index=tickers)

    def generate_absolute_views(
        self, 
        ml_predictions: Dict[str, float], 
        tickers: List[str],
        confidence_level: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper to convert ML Alpha predictions into Black-Litterman matrices.
        """
        n = len(tickers)
        
        # Validate confidence level
        if not (0 <= confidence_level <= 1):
             logger.warning(f"Confidence level {confidence_level} out of bounds [0,1]. Clipping to 0.99.")
             confidence_level = np.clip(confidence_level, 0.0, 0.99)

        # FIX: Only create views for tickers that exist in the universe
        valid_views = {t: v for t, v in ml_predictions.items() if t in tickers}
        k = len(valid_views)
        
        if k == 0:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))
        
        # P matrix (Pick Matrix): K x N
        P = np.zeros((k, n))
        # Q vector (View Vector): K x 1
        Q = np.zeros(k)
        
        # Map ticker string to index
        ticker_map = {t: i for i, t in enumerate(tickers)}
        
        for i, (ticker, pred) in enumerate(valid_views.items()):
            col_idx = ticker_map[ticker]
            P[i, col_idx] = 1.0
            Q[i] = pred
                
        # Omega (Confidence Matrix): K x K diagonal matrix
        # FIX: Add epsilon to prevent singularity if confidence is 1.0
        uncertainty = (1.0 - confidence_level) * 0.1 + 1e-6
        Omega = np.eye(k) * uncertainty
        
        return P, Q, Omega

    def calculate_posterior(
        self, 
        pi: pd.Series, 
        covariance_matrix: pd.DataFrame, 
        P: np.ndarray, 
        Q: np.ndarray, 
        Omega: np.ndarray
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculates the Posterior Expected Returns AND Covariance.
        """
        tickers = pi.index
        Sigma = covariance_matrix.loc[tickers, tickers].values
        Pi_vec = pi.values
        
        # If no views, return Prior
        if len(Q) == 0:
            return pi, covariance_matrix
        
        tau_Sigma = self.tau * Sigma
        
        try:
            # Using pseudo-inverse for numerical stability
            inv_tau_Sigma = np.linalg.pinv(tau_Sigma)
            inv_Omega = np.linalg.pinv(Omega)
            
            # M = inverse of uncertainty in the posterior mean
            # M = [(tau * Sigma)^-1 + P^T * Omega^-1 * P]
            M = inv_tau_Sigma + P.T @ inv_Omega @ P
            
            # Posterior Covariance of the Mean (Uncertainty reduction)
            # Sigma_post = Sigma + M^-1
            M_inv = np.linalg.pinv(M)
            Sigma_post = Sigma + M_inv
            
            # Posterior Expected Returns
            # E[R] = M^-1 * [ (tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
            term2 = (inv_tau_Sigma @ Pi_vec) + (P.T @ inv_Omega @ Q)
            posterior_er = M_inv @ term2
            
            return pd.Series(posterior_er, index=tickers), pd.DataFrame(Sigma_post, index=tickers, columns=tickers)
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed in BL calculation: {e}")
            return pi, covariance_matrix # Fallback to prior

    def optimize(
        self,
        ml_predictions: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        market_caps: Dict[str, float],
        confidence_level: float = 0.5,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        End-to-end Black-Litterman optimization.
        """
        tickers = list(covariance_matrix.columns)
        
        # 1. Get Market Implied Returns (The Anchor)
        pi = self.get_implied_returns(covariance_matrix, market_caps)
        
        # 2. Formulate Views from ML Predictions
        P, Q, Omega = self.generate_absolute_views(ml_predictions, tickers, confidence_level)
        
        # 3. Calculate Posterior Distribution (Returns & Covariance)
        post_returns, post_covariance = self.calculate_posterior(pi, covariance_matrix, P, Q, Omega)
        
        # 4. Feed into Hardened Mean-Variance Optimizer
        # FIX: Use Posterior Covariance (Sigma_post) instead of Prior
        logger.info("Feeding Black-Litterman posterior estimates into Mean-Variance Optimizer.")
        return self.mvo.optimize(post_returns.to_dict(), post_covariance, constraints)