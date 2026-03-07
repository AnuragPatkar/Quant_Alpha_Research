"""
Mean-Variance Optimization (Markowitz) - Production Grade
Mathematically sound formulation using CVXPY with dynamic constraints.

BUGS FIXED:
  BUG-MV-01 [CRITICAL]: ECOS solver not installed — solve_max_sharpe always fell back to
    Equal Weight, causing Sharpe to be 0.48 instead of ≥ 0.75.
    Fix: Try ECOS first, then fall back to SCS, then CLARABEL. CLARABEL ships with
    modern CVXPY (≥1.3) and handles QCQP/SOCP natively without extra install.

  BUG-MV-02 [HIGH]: dynamic_max_w override in optimize() destroyed the max-weight
    constraint intent. For n=3 assets, max(0.05, 1/3+0.01) = 0.343, capping ALL
    assets at the same ceiling and forcing equal allocation even when one asset clearly
    dominates — making test_mean_variance_dominant_asset impossible to pass.
    Fix: Only apply the dynamic floor if the user explicitly passes no constraints.
    When constraints are provided, respect the requested max_weight and only
    raise a floor of 1/n (not 1/n+0.01) as a strict infeasibility guard.

  BUG-MV-03 [HIGH]: Same dynamic_max_w bug existed in solve_max_sharpe, applied to
    the y <= dynamic_max_w * kappa constraint, distorting Charnes-Cooper weights.
    Fix: Same treatment as BUG-MV-02 — only apply floor when no constraints given.
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


def _get_available_solvers() -> List[str]:
    """Return CVXPY solvers available in the current environment."""
    return cp.installed_solvers()


class MeanVarianceOptimizer:
    def __init__(self, risk_aversion: float = 1.0):
        """
        Initializes the optimizer. Market variables like risk_free_rate
        are passed directly to the optimization methods to ensure dynamic fetching.
        """
        self.risk_aversion = risk_aversion

    def _prepare_data(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame
    ) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        """
        Centralized data preparation to prevent DRY violations.
        Assumes covariance_matrix is pre-calculated (and shrunk if necessary) upstream.
        """
        available_tickers = set(covariance_matrix.index) & set(covariance_matrix.columns)
        tickers = [t for t in expected_returns.keys() if t in available_tickers]
        n = len(tickers)

        if n == 0:
            raise ValueError(
                "No valid tickers found in intersection of expected_returns and covariance_matrix."
            )

        mu = np.array([expected_returns[t] for t in tickers])
        Sigma = covariance_matrix.loc[tickers, tickers].values

        return tickers, mu, Sigma, n

    def _safe_normalize(self, optimized_w: np.ndarray, tickers: List[str]) -> Dict[str, float]:
        """
        Safely extracts and normalizes weights.
        Raises an error if the solver returned severely infeasible weights.
        """
        n = len(tickers)
        weight_dict = {
            tickers[i]: round(float(optimized_w[i]), 8)
            for i in range(n) if optimized_w[i] > 1e-6
        }

        total_sum = sum(weight_dict.values())

        if abs(total_sum - 1.0) < 1e-4:
            if total_sum > 0:
                return {k: v / total_sum for k, v in weight_dict.items()}
            return weight_dict
        else:
            raise ValueError(f"Solver returned infeasible weights summing to {total_sum:.4f}")

    def _resolve_max_weight(self, n: int, constraints: Optional[Dict]) -> float:
        """
        FIX BUG-MV-02 / BUG-MV-03:
        Compute the effective per-asset max weight.

        Rules:
        - If no constraints provided, use a generous default of 1.0 (unconstrained).
          This allows the optimizer to freely pick dominant assets.
        - If constraints provided, use the requested max_weight.
        - In both cases, apply a strict infeasibility floor of 1/n so the problem
          is always feasible (every asset can at least hold its equal-weight share).
        - We intentionally DO NOT add the +0.01 buffer that caused the original bug:
          that buffer turned an infeasibility guard into a hard cap that smothered
          dominant assets when n is small (e.g. n=3 → cap = 0.343).
        """
        if constraints is None:
            requested = 1.0  # unconstrained: let optimizer decide freely
        else:
            requested = constraints.get('max_weight', 1.0)

        infeasibility_floor = 1.0 / n
        return max(requested, infeasibility_floor)

    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Standard Mean-Variance Optimization.
        Maximizes:  mu^T w  -  lambda * w^T Sigma w
        Subject to: sum(w) == 1, w >= 0 (long-only by default), w <= max_weight
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        w = cp.Variable(n)

        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_variance)

        constraints_list = [cp.sum(w) == 1.0]

        if not constraints or not constraints.get('allow_short', False):
            constraints_list.append(w >= 0)

        # FIX BUG-MV-02: use _resolve_max_weight instead of inline dynamic_max_w
        effective_max_w = self._resolve_max_weight(n, constraints)
        constraints_list.append(w <= effective_max_w)

        prob = cp.Problem(objective, constraints_list)

        try:
            prob.solve(solver=cp.OSQP)

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.SolverError(f"Solver status: {prob.status}")

            weight_dict = self._safe_normalize(w.value, tickers)
            logger.info(f"Optimization successful for {len(weight_dict)} assets.")
            return weight_dict

        except (cp.SolverError, ValueError) as e:
            logger.error(f"Optimization failed: {e}. Falling back to Equal Weight.")
            return {t: 1.0 / n for t in tickers}

    def solve_max_sharpe(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Direct Sharpe Maximization using Charnes-Cooper transformation.

        Solves the equivalent QCQP/SOCP:
            max  (mu - rf)^T y
            s.t. y^T Sigma y <= 1
                 sum(y) == kappa
                 y >= 0
                 y <= max_weight * kappa
                 kappa >= 0
        Then recovers weights as w = y / kappa.

        FIX BUG-MV-01: ECOS is not always installed. We try a chain of solvers
        that can handle quadratic constraints / SOCP problems:
            1. ECOS   — preferred, handles QCQP natively
            2. SCS    — open-source, ships with CVXPY, handles SOCP
            3. CLARABEL — default solver in CVXPY ≥1.3, handles SOCP/QCQP

        FIX BUG-MV-03: per-asset cap uses _resolve_max_weight (same fix as optimize).
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        y = cp.Variable(n)
        kappa = cp.Variable()

        objective = cp.Maximize((mu - risk_free_rate) @ y)

        # FIX BUG-MV-03: use _resolve_max_weight instead of inline dynamic_max_w
        effective_max_w = self._resolve_max_weight(n, constraints)

        constraints_list = [
            cp.quad_form(y, Sigma) <= 1,   # Quadratic Constraint (QCQP / SOCP cone)
            cp.sum(y) == kappa,
            y >= 0,
            y <= effective_max_w * kappa,
            kappa >= 0,
        ]

        prob = cp.Problem(objective, constraints_list)

        # FIX BUG-MV-01: solver priority chain for QCQP-capable solvers
        qcqp_solvers = ["ECOS", "SCS", "CLARABEL"]
        installed = _get_available_solvers()
        solver_chain = [s for s in qcqp_solvers if s in installed]

        if not solver_chain:
            logger.error(
                "No QCQP-capable solver available (tried ECOS, SCS, CLARABEL). "
                "Install one via: pip install ecos  OR  pip install clarabel"
            )
            return {t: 1.0 / n for t in tickers}

        last_error = None
        for solver_name in solver_chain:
            try:
                solver_const = getattr(cp, solver_name)
                prob.solve(solver=solver_const)

                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    raise cp.SolverError(f"Solver {solver_name} status: {prob.status}")

                if y.value is None or kappa.value is None or float(kappa.value) <= 1e-6:
                    raise cp.SolverError(
                        f"Solver {solver_name} returned None or near-zero kappa "
                        f"(kappa={kappa.value})"
                    )

                weights = y.value / float(kappa.value)
                weight_dict = self._safe_normalize(weights, tickers)
                logger.info(
                    f"Sharpe optimization successful via {solver_name} "
                    f"for {len(weight_dict)} assets."
                )
                return weight_dict

            except (cp.SolverError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Solver {solver_name} failed: {e}. Trying next solver.")
                last_error = e
                continue

        logger.error(
            f"All QCQP solvers failed. Last error: {last_error}. "
            "Falling back to Equal Weight."
        )
        return {t: 1.0 / n for t in tickers}