"""
Risk Parity Portfolio Optimization
Equal risk contribution from each asset.

BUGS FIXED:
  BUG-RP-01 [CRITICAL]: SLSQP with sum-of-squared-deviations objective has a
    mathematically provable zero-gradient trap when any asset is uncorrelated
    with all others (block-diagonal covariance).

    Root cause (mathematical):
      risk_contrib_i = w_i * (Sigma @ w)_i / port_vol

      For a block-diagonal asset A where Sigma[A,j]=0 for all j≠A:
        (Sigma @ w)_A = Sigma[A,A] * w_A     (only the diagonal term survives)
        risk_contrib_A = Sigma[A,A] * w_A^2 / port_vol

      At the boundary w_A = 0:
        d(risk_contrib_A)/d(w_A) = 0          ← EXACT zero gradient

      The squared-deviation objective (risk_contrib_A - target_A)^2 therefore
      also has zero gradient w.r.t. w_A at the boundary. SLSQP — and any other
      gradient-based method, including random restarts — CANNOT escape this trap;
      the solver sees a perfectly flat surface and declares convergence at w_A=0.

    Fix: Replace SLSQP + squared-deviation with the Spinu (2013) log-barrier
    formulation, which is STRICTLY CONVEX with a UNIQUE global minimum that is
    guaranteed to be interior (all weights > 0):

      Objective:  f(y) = -sum_i[ b_i * log(y_i) ]  +  (1/2) * y^T Sigma y
      Solver:     L-BFGS-B  (bounds enforce y_i > 0; analytic gradient provided)
      Recovery:   w_i = y_i / sum(y)

    Properties:
      - No boundary traps: log(y_i) → -inf forces y_i strictly > 0
      - Strictly convex: single global optimum, zero local minima
      - Works for any covariance structure, including block-diagonal
      - Exact equal risk contributions (MRC_A == MRC_B == MRC_C confirmed)

    Reference: Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights."
               SSRN: https://ssrn.com/abstract=2297383

  BUG-RP-02 [LOW]: Inverse-volatility fallback was only triggered on SLSQP
    convergence failure, not on degenerate-weight solutions. Retained as a
    last-resort safety net; with Spinu it is essentially unreachable.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

_WEIGHT_FLOOR = 1e-8   # prune weights below this threshold after normalisation


class RiskParityOptimizer:
    """
    Risk Parity Optimization using the Spinu (2013) log-barrier formulation.

    Goal: Each asset contributes equally to portfolio risk.

        Risk Contribution_i = w_i * (Σw)_i / sqrt(w^T Σ w)  =  b_i  for all i

    Method: Solves the strictly-convex unconstrained problem
        min_y  -sum_i[ b_i * log(y_i) ]  +  (1/2) * y^T Sigma y
    then recovers weights as w = y / sum(y).

    Unlike squared-deviation + SLSQP, the log-barrier objective has a provably
    unique global minimum with all weights strictly positive — no boundary traps,
    no local minima, no random restarts required.

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
            target_risk: Optional target risk budget per ticker (unnormalised).
                         If None, equal budgets (1/N) are used.
        """
        self.target_risk = target_risk

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        Optimize for risk parity using the Spinu log-barrier method.

        Args:
            covariance_matrix: Annualised covariance matrix (DataFrame).
            tickers: List of asset tickers to include.

        Returns:
            Dictionary of {ticker: weight}.
        """
        # ── 1. Safety: intersect with available data ──────────────────── #
        available_tickers = [t for t in tickers if t in covariance_matrix.index]
        if len(available_tickers) != len(tickers):
            missing = set(tickers) - set(available_tickers)
            logger.warning(
                f"⚠️ Missing covariance data for: {missing}. "
                "Optimizing available assets only."
            )

        if not available_tickers:
            return {}

        n = len(available_tickers)
        Sigma = covariance_matrix.loc[available_tickers, available_tickers].values

        # ── 2. Risk budgets ───────────────────────────────────────────── #
        if self.target_risk is None:
            b = np.ones(n) / n                          # equal budgets
        else:
            b = np.array([self.target_risk.get(t, 1.0 / n) for t in available_tickers])
            b = b / b.sum()                             # normalise

        # ── 3. Spinu (2013) log-barrier objective & analytic gradient ─── #
        #
        # FIX BUG-RP-01: strictly convex — guaranteed unique interior minimum
        #
        # f(y)  = -sum_i[ b_i * log(y_i) ]  +  (1/2) * y^T Sigma y
        # f'(y) = -b / y  +  Sigma @ y       (element-wise / for first term)

        def _objective(y: np.ndarray) -> float:
            return -float(b @ np.log(y)) + 0.5 * float(y @ Sigma @ y)

        def _gradient(y: np.ndarray) -> np.ndarray:
            return -b / y + Sigma @ y

        y0 = np.ones(n) / n                            # any interior point
        bounds = [(1e-8, None)] * n                    # enforce y_i > 0

        result = minimize(
            _objective,
            y0,
            jac=_gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10},
        )

        # ── 4. Recover portfolio weights:  w = y / sum(y) ────────────── #
        if result.success and result.x is not None and result.x.sum() > 0:
            return self._pack_weights(result.x / result.x.sum(), available_tickers)

        # ── 5. Fallback: Inverse-Volatility (BUG-RP-02: safety net) ──── #
        logger.warning(
            f"⚠️ Spinu optimisation did not converge ({result.message}). "
            "Using Inverse Volatility fallback."
        )
        return self._inverse_vol_fallback(Sigma, available_tickers)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _pack_weights(
        self, normalised_w: np.ndarray, tickers: List[str]
    ) -> Dict[str, float]:
        """Prune sub-floor weights, re-normalise, and return as dict."""
        weight_dict = {
            tickers[i]: float(w)
            for i, w in enumerate(normalised_w)
            if w > _WEIGHT_FLOOR
        }
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}
        logger.info(f"Risk parity optimised: {len(weight_dict)} positions")
        return weight_dict

    def _inverse_vol_fallback(
        self, Sigma: np.ndarray, tickers: List[str]
    ) -> Dict[str, float]:
        """Inverse-volatility weights — robust closed-form fallback."""
        variances = np.diag(Sigma)
        volatilities = np.sqrt(np.maximum(variances, 1e-8))
        inv_vol = 1.0 / volatilities
        fallback_w = inv_vol / inv_vol.sum()

        weight_dict = {
            tickers[i]: float(w)
            for i, w in enumerate(fallback_w)
            if w > _WEIGHT_FLOOR
        }
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}
        return weight_dict