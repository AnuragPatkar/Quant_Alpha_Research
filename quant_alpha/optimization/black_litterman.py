"""
Black-Litterman Portfolio Optimization Framework
================================================
Bayesian portfolio construction blending market equilibrium assumptions with
proprietary alpha views.

Purpose
-------
The `BlackLittermanModel` refines the Mean-Variance Optimization (MVO) framework by
addressing its sensitivity to input estimates. Instead of optimizing directly on raw
predictions, it calculates a posterior distribution of expected returns by combining:
1.  **The Prior**: Market Equilibrium Returns ($\Pi$) derived from the Capital Asset
    Pricing Model (CAPM).
2.  **The Likelihood**: Investor Views ($Q$) with associated confidence matrices ($\Omega$).

Usage
-----
Intended for generating robust return vectors for downstream Mean-Variance optimization.

.. code-block:: python

    bl = BlackLittermanModel(tau=0.05, risk_aversion=2.5)
    
    # Generate posterior estimates
    weights = bl.optimize(
        ml_predictions=alpha_signals,
        covariance_matrix=risk_model,
        market_caps=caps
    )

Importance
----------
-   **Estimation Error Mitigation**: Shrinks ML predictions towards the market consensus
    where signal confidence is low, reducing "extremal" weights common in standard MVO.
-   **Intuitive Priors**: Ensures that in the absence of views, the portfolio defaults
    to the market portfolio (CAPM), guaranteeing a sensible baseline.
-   **Bayesian Updating**: Formally incorporates the precision of alpha signals ($\Omega$)
    into the portfolio construction process.

Tools & Frameworks
------------------
-   **NumPy**: Linear algebra operations for matrix inversion and posterior derivation.
-   **Pandas**: Data alignment for tickers and time-series.
-   **MeanVarianceOptimizer**: Solves the final quadratic program using the posterior inputs.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from .mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)


class BlackLittermanModel:
    """
    Implementation of the Black-Litterman asset allocation model.

    Theoretical Foundation:
    -----------------------
    The model treats expected returns as a random variable distributed as:
    .. math::
        E[R] \sim N(\Pi, \tau \Sigma)

    Views are expressed as:
    .. math::
        P \cdot E[R] = Q + \epsilon, \quad \epsilon \sim N(0, \Omega)

    The posterior distribution is derived via Bayesian updating:
    .. math::
        E[R]_{post} = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5
    ):
        """
        Args:
            tau (float): Scalar indicating uncertainty of the prior (market equilibrium).
                 - Small $\tau$ ($\approx 0.01$): Prior dominates, views are discounted.
                 - Large $\tau$ ($\approx 1.0+$): Views dominate (Diffuse Prior).
            risk_aversion (float): Market risk aversion coefficient ($\delta$).
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
        Calculates the Market Implied Returns ($\Pi$) via Reverse Optimization.
        
        Math:
        .. math::
            \Pi = \delta \Sigma w_{mkt}
        """
        tickers = list(covariance_matrix.columns)

        weights = []
        missing_caps = []
        for t in tickers:
            cap = market_caps.get(t, 0.0)
            weights.append(cap)
            if cap == 0:
                missing_caps.append(t)

        if len(missing_caps) > len(tickers) * 0.1:
            logger.debug(
                f"⚠️ {len(missing_caps)}/{len(tickers)} stocks missing market cap data. "
                "Using defaults."
            )

        total_mcap = sum(weights)
        if total_mcap == 0:
            # Fallback: equal-weight equilibrium when no market cap data available
            w_mkt = np.ones(len(tickers)) / len(tickers)
        else:
            w_mkt = np.array(weights) / total_mcap

        Sigma = covariance_matrix.values
        pi = self.risk_aversion * (Sigma @ w_mkt)
        return pd.Series(pi, index=tickers)

    def generate_absolute_views(
        self,
        ml_predictions: Dict[str, float],
        tickers: List[str],
        confidence_level: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts ML Alpha predictions into Black-Litterman view matrices.

        Constructs:
        - **P**: Pick matrix ($K \times N$) mapping views to assets.
        - **Q**: View vector ($K \times 1$) containing expected returns.
        - **Omega**: Uncertainty matrix ($K \times K$), diagonalized by confidence.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (P, Q, Omega)
        """
        n = len(tickers)

        if not (0 <= confidence_level <= 1):
            logger.warning(
                f"Confidence level {confidence_level} out of bounds [0,1]. Clipping to 0.99."
            )
            confidence_level = float(np.clip(confidence_level, 0.0, 0.99))

        valid_views = {t: v for t, v in ml_predictions.items() if t in tickers}
        k = len(valid_views)

        if k == 0:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

        P = np.zeros((k, n))
        Q = np.zeros(k)
        ticker_map = {t: i for i, t in enumerate(tickers)}

        for i, (ticker, pred) in enumerate(valid_views.items()):
            P[i, ticker_map[ticker]] = 1.0
            Q[i] = pred

        # Omega: small uncertainty = high confidence in views
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
        Computes the Bayesian Posterior distribution.

        Math:
        .. math::
            M = (\tau \Sigma)^{-1} + P^T \Omega^{-1} P
            E[R]_{post} = M^{-1} [ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q ]
            \Sigma_{post} = \Sigma + M^{-1}
        """
        tickers = pi.index
        Sigma = covariance_matrix.loc[tickers, tickers].values
        Pi_vec = pi.values

        if len(Q) == 0:
            return pi, covariance_matrix

        tau_Sigma = self.tau * Sigma

        try:
            inv_tau_Sigma = np.linalg.pinv(tau_Sigma)
            inv_Omega = np.linalg.pinv(Omega)

            M = inv_tau_Sigma + P.T @ inv_Omega @ P
            M_inv = np.linalg.pinv(M)

            Sigma_post = Sigma + M_inv
            term2 = (inv_tau_Sigma @ Pi_vec) + (P.T @ inv_Omega @ Q)
            posterior_er = M_inv @ term2

            return (
                pd.Series(posterior_er, index=tickers),
                pd.DataFrame(Sigma_post, index=tickers, columns=tickers),
            )

        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed in BL calculation: {e}")
            return pi, covariance_matrix

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

        Workflow:
          1. Compute Market-Implied Returns ($\Pi$) as the Prior.
          2. Transform ML predictions into Views ($Q$) and Uncertainty ($\Omega$).
          3. Perform Bayesian Update to derive Posterior $E[R]$.
          4. Execute Mean-Variance Optimization using Posterior Return vector.
        """
        tickers = list(covariance_matrix.columns)

        # 1. Market Implied Returns (Prior)
        pi = self.get_implied_returns(covariance_matrix, market_caps)

        # 2. Views from ML predictions
        P, Q, Omega = self.generate_absolute_views(ml_predictions, tickers, confidence_level)

        # 3. Posterior Distribution
        post_returns, post_covariance = self.calculate_posterior(
            pi, covariance_matrix, P, Q, Omega
        )

        # Optimization Strategy:
        # We utilize the Posterior Expected Returns ($E[R]_{post}$) as the signal vector,
        # but retain the Prior Covariance ($\Sigma$) as the risk model.
        #
        # Rationale:
        # The Posterior Covariance ($\Sigma_{post} = \Sigma + M^{-1}$) incorporates
        # 'Epistemic Uncertainty' (uncertainty in the estimate of the mean).
        # Including this in the MVO risk model often results in overly conservative
        # portfolios that dampen the signal strength of the views ($\tau$ sensitivity).
        # Standard practitioner approaches (e.g., Idzorek) recommend optimizing
        # on ($E[R]_{post}$, $\Sigma_{prior}$).
        effective_constraints = constraints if constraints is not None else {"max_weight": 1.0}

        logger.info(
            "Feeding Black-Litterman posterior returns + prior covariance "
            "into Mean-Variance Optimizer."
        )
        # Execute MVO with Posterior Returns and Prior Covariance
        return self.mvo.optimize(
            post_returns.to_dict(), covariance_matrix, effective_constraints
        )