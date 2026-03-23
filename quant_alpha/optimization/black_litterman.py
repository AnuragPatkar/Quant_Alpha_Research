"""
quant_alpha/optimization/black_litterman.py
=============================================
Black-Litterman Portfolio Optimization Framework.

FIXES:
  BUG-080 (MEDIUM): calculate_posterior() computed:
           Sigma_post = Sigma + M_inv
           The correct He & Litterman (1999) posterior covariance is:
           Sigma_post = Sigma + (τΣ) M_inv (τΣ)
           The code was missing the τΣ scaling on both sides of M_inv.

           Impact: Low — the code comments explicitly state that the
           posterior covariance is NOT used in the downstream MVO call
           (prior Σ is used instead, which is standard practitioner
           approach). The formula error in Sigma_post has zero numerical
           effect on the final weights. Fixed for correctness and to
           prevent future callers who use the returned Sigma_post from
           receiving a wrong value.
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
        Prior:    E[R] ~ N(Π, τΣ)
        Views:    P·E[R] = Q + ε,  ε ~ N(0, Ω)
        Posterior:
            M         = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
            E[R]_post = M⁻¹ [ (τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q ]
            Σ_post    = Σ + (τΣ) M⁻¹ (τΣ)      ← FIX BUG-080
    """

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        """
        Args:
            tau           : Uncertainty scalar for the prior. Small τ (≈0.01)
                            keeps the portfolio close to the market portfolio;
                            large τ (≈1.0) lets views dominate.
            risk_aversion : Market risk aversion δ for reverse optimization.
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
        Compute Market Implied Returns Π via reverse optimization.

            Π = δ Σ w_mkt
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
        Convert ML alpha predictions into BL view matrices (P, Q, Ω).

        Returns
        -------
        P      : Pick matrix (K × N)
        Q      : View vector (K,)
        Omega  : Diagonal uncertainty matrix (K × K)
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

        # Small uncertainty = high confidence in views
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
        Compute posterior distribution E[R]_post and Σ_post.

        FIX BUG-080: Σ_post was Σ + M⁻¹.
        Correct formula (He & Litterman 1999, eq. 12):
            Σ_post = Σ + (τΣ) M⁻¹ (τΣ)
        where M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P.

        Note: Σ_post is returned for completeness but is NOT used in
        optimize() — the prior Σ is used for MVO per standard practitioner
        approach (avoids epistemic uncertainty inflating portfolio risk).
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

            # FIX BUG-080: correct posterior covariance
            # Σ_post = Σ + (τΣ) M⁻¹ (τΣ)
            Sigma_post = Sigma + tau_Sigma @ M_inv @ tau_Sigma

            # Posterior expected returns (unchanged — was already correct)
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
        End-to-end Black-Litterman optimization.

        Workflow:
        1. Compute Π (prior) via reverse optimization on market caps.
        2. Convert ML predictions to view matrices (P, Q, Ω).
        3. Bayesian update → posterior E[R]_post and Σ_post.
        4. Run MVO using (E[R]_post, Σ_prior) — standard practitioner choice.
        """
        tickers = list(covariance_matrix.columns)

        pi         = self.get_implied_returns(covariance_matrix, market_caps)
        P, Q, Omega = self.generate_absolute_views(
            ml_predictions, tickers, confidence_level
        )
        post_returns, _ = self.calculate_posterior(
            pi, covariance_matrix, P, Q, Omega
        )

        # Use posterior returns + PRIOR covariance for MVO
        effective_constraints = constraints if constraints is not None else {
            "max_weight": 1.0
        }

        logger.info(
            "Feeding BL posterior returns + prior covariance into MVO."
        )
        return self.mvo.optimize(
            post_returns.to_dict(),
            covariance_matrix,         # prior covariance — intentional
            effective_constraints,
        )