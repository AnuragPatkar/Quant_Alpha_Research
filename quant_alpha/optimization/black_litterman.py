"""
Black-Litterman Portfolio Optimization Framework.
===============================================

Provides Bayesian alpha-view blending over market equilibrium priors.

Purpose
-------
This module implements the Black-Litterman asset allocation model, combining 
market equilibrium returns (CAPM prior) with proprietary machine learning alpha views 
to produce a mathematically stabilized posterior expected return vector.

Role in Quantitative Workflow
-----------------------------
Serves as a robust expectation engine, mitigating the extreme estimation error 
sensitivity inherent to classical Mean-Variance optimization by anchoring 
predictions to market-capitalization-derived structural priors.

Mathematical Dependencies
-------------------------
- **NumPy/Pandas**: Matrix inversion and cross-sectional data alignment.
- **CVXPY**: Indirectly utilized via the downstream Mean-Variance optimizer.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from .mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)


class BlackLittermanModel:
    """
    Black-Litterman asset allocation model.

    Combines market equilibrium returns (CAPM prior) with proprietary
    ML alpha views to produce a posterior expected return vector that
    is fed into Mean-Variance optimization.

    Mathematical foundation:
        Prior:    $E[R] \sim \mathcal{N}(\Pi, \tau\Sigma)$
        Views:    $P \cdot E[R] = Q + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \Omega)$
        Posterior:
            $M         = (\tau\Sigma)^{-1} + P^T \Omega^{-1} P$
            $E[R]_{post} = M^{-1} [ (\tau\Sigma)^{-1} \Pi + P^T \Omega^{-1} Q ]$
            $\Sigma_{post}    = \Sigma + (\tau\Sigma) M^{-1} (\tau\Sigma)$
    """

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        """
        Initializes the Bayesian framework configurations.
        
        Args:
            tau (float): Uncertainty scalar bound for the prior. Small values (e.g., 0.01) 
                anchor the portfolio tightly to the market baseline, whereas large values 
                (e.g., 1.0) permit algorithmic views to structurally dominate. Defaults to 0.05.
            risk_aversion (float): Structural market risk aversion $\delta$ bounding 
                reverse-optimization prior extraction. Defaults to 2.5.
        """
        self.tau          = tau
        self.risk_aversion = risk_aversion
        self.mvo          = MeanVarianceOptimizer(risk_aversion=risk_aversion)
        logger.info(
            f"BlackLittermanModel initialized: tau={tau}, delta={risk_aversion}"
        )

    # ------------------------------------------------------------------
    # Step 1: market-implied prior returns
    # ------------------------------------------------------------------

    def get_implied_returns(
        self,
        covariance_matrix: pd.DataFrame,
        market_caps: Dict[str, float],
    ) -> pd.Series:
        """
        Computes Market Implied Returns ($\Pi$) utilizing reverse optimization.

        Formula: $\Pi = \delta \Sigma w_{mkt}$
        
        Args:
            covariance_matrix (pd.DataFrame): Systemic covariance structure ($\Sigma$).
            market_caps (Dict[str, float]): Mapped capitalizations enforcing $w_{mkt}$ ratios.
            
        Returns:
            pd.Series: Continuous vector array bounding extracted equilibrium expected returns.
        """
        tickers = list(covariance_matrix.columns)

        weights      = [market_caps.get(t, 0.0) for t in tickers]
        missing_caps = [t for t, w in zip(tickers, weights) if w == 0]

        if len(missing_caps) > len(tickers) * 0.1:
            logger.debug(
                f"⚠️ {len(missing_caps)}/{len(tickers)} stocks missing market cap. "
                "Using default cap = 0."
            )

        total_mcap = sum(weights)
        w_mkt = (
            np.array(weights) / total_mcap
            if total_mcap > 0
            else np.ones(len(tickers)) / len(tickers)
        )

        Sigma = covariance_matrix.values
        pi    = self.risk_aversion * (Sigma @ w_mkt)
        return pd.Series(pi, index=tickers)

    # ------------------------------------------------------------------
    # Step 2: convert ML predictions → view matrices
    # ------------------------------------------------------------------

    def generate_absolute_views(
        self,
        ml_predictions: Dict[str, float],
        tickers: List[str],
        confidence_level: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Translates machine learning alpha signals directly into BL view structures ($P, Q, \Omega$).
        
        Args:
            ml_predictions (Dict[str, float]): Absolute return forecasts isolated from estimators.
            tickers (List[str]): Universe of target constituents for sequence alignment.
            confidence_level (float): The targeted structural confidence mapping to view uncertainty. Defaults to 0.5.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Standardized arrays detailing:
                - $P$: Pick matrix projecting execution targets ($K \times N$).
                - $Q$: Directional view vector ($K$).
                - $\Omega$: Diagonal bounding uncertainty matrix ($K \times K$).
        """
        n = len(tickers)

        if not (0 <= confidence_level <= 1):
            logger.warning(
                f"Confidence level {confidence_level} out of bounds [0,1]. "
                "Clipping to 0.99."
            )
            confidence_level = float(np.clip(confidence_level, 0.0, 0.99))

        valid_views = {t: v for t, v in ml_predictions.items() if t in tickers}
        k           = len(valid_views)

        if k == 0:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

        P          = np.zeros((k, n))
        Q          = np.zeros(k)
        ticker_map = {t: i for i, t in enumerate(tickers)}

        for i, (ticker, pred) in enumerate(valid_views.items()):
            P[i, ticker_map[ticker]] = 1.0
            Q[i]                     = pred

        uncertainty = (1.0 - confidence_level) * 0.1 + 1e-6
        Omega       = np.eye(k) * uncertainty
        return P, Q, Omega

    # ------------------------------------------------------------------
    # Step 3: Bayesian posterior update
    # ------------------------------------------------------------------

    def calculate_posterior(
        self,
        pi: pd.Series,
        covariance_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculates posterior probability distribution boundaries $E[R]_{post}$ and $\Sigma_{post}$.

        Applies the He & Litterman (1999) scaling theorem integrating prior covariance directly:
            $\Sigma_{post} = \Sigma + (\tau\Sigma) M^{-1} (\tau\Sigma)$

        Args:
            pi (pd.Series): Reverse-optimized structural market implied returns.
            covariance_matrix (pd.DataFrame): Systemic baseline covariance map.
            P (np.ndarray): Pick matrix designating active views.
            Q (np.ndarray): The mathematical magnitude defining ML forecasts.
            Omega (np.ndarray): Evaluated uncertainty scalar bounded to ML views.
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: The optimized posterior expectations and posterior covariance.
        """
        tickers = pi.index
        Sigma   = covariance_matrix.loc[tickers, tickers].values
        Pi_vec  = pi.values

        if len(Q) == 0:
            # No views → posterior = prior
            return pi, covariance_matrix

        tau_Sigma = self.tau * Sigma

        try:
            inv_tau_Sigma = np.linalg.pinv(tau_Sigma)
            inv_Omega     = np.linalg.pinv(Omega)

            M     = inv_tau_Sigma + P.T @ inv_Omega @ P
            M_inv = np.linalg.pinv(M)

            # Computes the Bayesian posterior covariance strictly bounding the scaled uncertainty prior
            Sigma_post = Sigma + tau_Sigma @ M_inv @ tau_Sigma

            term2       = inv_tau_Sigma @ Pi_vec + P.T @ inv_Omega @ Q
            post_er     = M_inv @ term2

            return (
                pd.Series(post_er, index=tickers),
                pd.DataFrame(Sigma_post, index=tickers, columns=tickers),
            )

        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed in BL calculation: {e}")
            return pi, covariance_matrix

    # ------------------------------------------------------------------
    # End-to-end optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        ml_predictions: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        market_caps: Dict[str, float],
        confidence_level: float = 0.5,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Executes end-to-end Black-Litterman blending and optimal structural constraint resolution.

        Args:
            ml_predictions (Dict[str, float]): Absolute directional signals mapped to assets.
            covariance_matrix (pd.DataFrame): Empirical prior risk representation map.
            market_caps (Dict[str, float]): Aggregate capitalization sizes establishing standard baseline.
            confidence_level (float): The overarching algorithmic view conviction. Defaults to 0.5.
            constraints (Optional[Dict]): Positional overrides strictly passed to the inner optimization engine.
            
        Returns:
            Dict[str, float]: Normalized terminal allocation weights structurally resolving ML targets 
                and systematic baseline assumptions.
        """
        tickers = list(covariance_matrix.columns)

        pi         = self.get_implied_returns(covariance_matrix, market_caps)
        P, Q, Omega = self.generate_absolute_views(
            ml_predictions, tickers, confidence_level
        )
        post_returns, _ = self.calculate_posterior(
            pi, covariance_matrix, P, Q, Omega
        )

        effective_constraints = constraints if constraints is not None else {
            "max_weight": 1.0
        }

        logger.info(
            "Feeding BL posterior returns + prior covariance into MVO."
        )
        return self.mvo.optimize(
            post_returns.to_dict(),
            covariance_matrix,         # Injects pure prior covariance bounding practitioner standard
            effective_constraints,
        )