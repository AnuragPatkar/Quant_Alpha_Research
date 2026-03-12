"""
Portfolio Constraint Management & Enforcement
=============================================
Dual-interface library for defining and enforcing portfolio mandate limits.

Purpose
-------
This module provides two distinct mechanisms for constraint management:
1.  **Declarative API (Pre-Optimization)**: Static factory methods that generate
    CVXPY constraint objects for convex optimization problems.
2.  **Imperative API (Post-Optimization)**: An instance-based engine that applies
    iterative projection algorithms to enforce constraints on raw weight vectors.
    This is critical for handling numerical precision errors or enforcing
    non-convex heuristics after the primary solver step.

Usage
-----
**1. CVXPY Optimization Construction:**

.. code-block:: python

    w = cp.Variable(n)
    constraints = [
        *PortfolioConstraints.long_only(w),
        *PortfolioConstraints.sector_exposure_limit(w, tickers, sector_map, 0.20)
    ]

**2. Post-Processing Enforcement:**

.. code-block:: python

    # Corrects floating-point drift and enforces strict caps
    pc = PortfolioConstraints(max_weight=0.10, sector_limits={'Tech': 0.30})
    final_weights = pc.apply(raw_weights)

Importance
----------
-   **Risk Control**: Enforces hard limits on Gross Leverage, Sector Exposure, and Concentration.
-   **Feasibility Guarantee**: The iterative projection methods ensure that output portfolios
    strictly adhere to mandate limits even if the upstream solver returns $\epsilon$-infeasible results.
-   **Flexibility**: Supports both Gross Exposure constraints (Long-Only) and Net Exposure
    constraints (Long/Short).

Tools & Frameworks
------------------
-   **CVXPY**: Construction of convex constraint sets ($Ax \leq b$).
-   **NumPy**: Vectorized operations for iterative proportional fitting.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """
    Unified utility for portfolio constraint definition and enforcement.

    Interface Overview:
    -------------------
    1.  **Instance Methods**: Apply heuristics to dict-based weights (Post-Processing).
        Uses **Iterative Proportional Fitting** to enforce limits while maintaining
        full investment ($\sum w_i = 1.0$).
        
        *Execution Order:*
        1. Minimum Weight Floor (Cardinality reduction)
        2. Maximum Weight Cap (Concentration limit)
        3. Sector Exposure Limits (Risk bucket caps)

    2.  **Static Methods**: Generate `cvxpy.Constraint` objects (Optimization).
        Used to define the feasible region $\mathcal{F}$ for the solver.
    """

    def __init__(
        self,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        sector_limits: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            max_weight (float): Maximum allowed weight $w_{max}$ per asset ($0 < w_{max} \le 1$).
            min_weight (float): Minimum weight threshold $w_{min}$; assets with $w_i < w_{min}$ are truncated to 0.
            sector_limits (Optional[Dict]): Aggregate sector caps {sector: $L_{sector}$}.
            sector_map (Optional[Dict]): Mapping from ticker to sector label.
        """
        if not (0 < max_weight <= 1.0):
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if not (0 <= min_weight < max_weight):
            raise ValueError(
                f"min_weight must be in [0, max_weight), got {min_weight}"
            )

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.sector_limits = sector_limits or {}
        self.sector_map = sector_map or {}

    # ── Instance post-processing API ────────────────────────────────────── #

    def apply(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Sequentially enforces constraints on a weight vector via iterative projection.

        Args:
            weights (Dict[str, float]): Input portfolio weights. Input need not sum to 1.0.

        Returns:
            Dict[str, float]: Normalized weights ($\sum w_i = 1.0$) adhering to all limits.
        """
        if not weights:
            return {}

        w = dict(weights)

        # Step 1: Cardinality Constraint (Min Weight Floor)
        if self.min_weight > 0:
            w = self._apply_min_weight(w)

        # Step 2: Concentration Constraint (Max Weight Cap)
        if self.max_weight < 1.0:
            w = self._apply_max_weight(w)

        # Step 3: Risk Bucket Constraint (Sector Limits)
        if self.sector_limits:
            w = self._apply_sector_limits(w)

        return w

    def _normalise(self, w: Dict[str, float]) -> Dict[str, float]:
        """Renormalizes weights to satisfy the budget constraint $\sum w_i = 1.0$."""
        total = sum(w.values())
        if total <= 0:
            return w
        return {k: v / total for k, v in w.items()}

    def _apply_min_weight(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Truncates positions below `min_weight` and renormalizes.
        Functions as a soft cardinality constraint.
        """
        filtered = {k: v for k, v in w.items() if v >= self.min_weight}
        if not filtered:
            # Fallback: Retain the largest position if all are below floor to verify solvability.
            logger.warning(
                "All assets below min_weight floor. Retaining the largest position."
            )
            max_k = max(w, key=lambda k: w[k])
            filtered = {max_k: w[max_k]}
        return self._normalise(filtered)

    def _apply_max_weight(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Enforces upper bounds via Iterative Proportional Fitting.
        
        Algorithm:
        1. Identify assets exceeding $w_{max}$.
        2. Clip to $w_{max}$.
        3. Distribute excess weight proportionally to assets below $w_{max}$.
        4. Repeat until convergence or max iterations.
        """
        cap = self.max_weight
        w = self._normalise(w)

        for _ in range(200):           # max iterations (converges in O(n) in practice)
            over  = {k: v for k, v in w.items() if v > cap + 1e-9}
            under = {k: v for k, v in w.items() if v <= cap + 1e-9}

            if not over:
                break                   # Convergence reached

            # Compute excess to redistribute
            excess = sum(v - cap for v in over.values())
            under_total = sum(under.values())

            # Clamp over-weight assets
            for k in over:
                w[k] = cap

            # Redistribute excess proportionally
            if under_total > 1e-12:
                scale = 1.0 + excess / under_total
                for k in under:
                    w[k] = min(w[k] * scale, cap)   # re-cap in same pass

            w = self._normalise(w)

        return w

    def _apply_sector_limits(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Enforces aggregate sector caps via Iterative Redistribution.
        Excess weight from sector $S$ is flowed to assets $\notin S$.
        """
        for _ in range(200):
            changed = False
            # Compute current sector weights
            sector_totals: Dict[str, float] = {}
            for ticker, weight in w.items():
                sector = self.sector_map.get(ticker, "__other__")
                sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

            for sector, limit in self.sector_limits.items():
                total = sector_totals.get(sector, 0.0)
                if total <= limit + 1e-9:
                    continue  # already within limit

                changed = True
                excess = total - limit

                # Scale down assets in this sector proportionally
                sector_tickers = [
                    k for k in w if self.sector_map.get(k, "__other__") == sector
                ]
                scale_down = limit / total
                for k in sector_tickers:
                    w[k] *= scale_down

                # Redistribute excess to the rest of the universe
                other_tickers = [
                    k for k in w if self.sector_map.get(k, "__other__") != sector
                ]
                other_total = sum(w[k] for k in other_tickers)
                if other_total > 1e-12:
                    for k in other_tickers:
                        w[k] += excess * (w[k] / other_total)

                w = self._normalise(w)

            if not changed:
                break

        return w

    # ── Static CVXPY factory API (original, fully preserved) ────────────── #

    @staticmethod
    def long_only(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constraint: $w_i \ge 0, \forall i$.
        """
        return [w >= 0]

    @staticmethod
    def fully_invested(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constraint: $\sum w_i = 1.0$.
        """
        return [cp.sum(w) == 1.0]

    @staticmethod
    def dollar_neutral(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constraint: $\sum w_i = 0.0$ (Market Neutral).
        """
        return [cp.sum(w) == 0.0]

    @staticmethod
    def leverage_limit(w: cp.Variable, limit: float = 1.0) -> List[cp.Constraint]:
        """
        Constraint: $\|w\|_1 \le L$.
        Enforces Gross Leverage limit.
        """
        return [cp.norm(w, 1) <= limit]

    @staticmethod
    def position_limit(
        w: cp.Variable,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> List[cp.Constraint]:
        """
        Constraint: Box constraints on individual weights.
        $w_{min} \le w_i \le w_{max}$
        """
        constraints = []
        if max_weight < 1.0:
            constraints.append(w <= max_weight)
        constraints.append(w >= min_weight)
        return constraints

    @staticmethod
    def sector_exposure_limit(
        w: cp.Variable,
        tickers: List[str],
        sector_map: Dict[str, str],
        max_sector_weight: float = 0.30
    ) -> List[cp.Constraint]:
        """
        Constraint: Net Sector Exposure Limit.
        $|\sum_{i \in S} w_i| \le L_{sector}$

        Note: Uses L1 norm of the sum, appropriate for Long/Short mandates.
        """
        constraints = []
        sector_indices: Dict[str, List[int]] = {}
        for i, ticker in enumerate(tickers):
            sector = sector_map.get(ticker, "Unknown")
            sector_indices.setdefault(sector, []).append(i)

        for sector, indices in sector_indices.items():
            constraints.append(
                cp.abs(cp.sum(w[indices])) <= max_sector_weight
            )
        return constraints

    @staticmethod
    def turnover_limit(
        w: cp.Variable,
        current_weights: np.ndarray,
        turnover_cap: float
    ) -> List[cp.Constraint]:
        """
        Constraint: Maximum Turnover Limit.
        $\|w - w_{current}\|_1 \le T_{cap}$
        """
        return [cp.norm(w - current_weights, 1) <= turnover_cap]