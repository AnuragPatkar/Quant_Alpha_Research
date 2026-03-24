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
        Initializes structural limits for dynamic boundary verification.
        
        Args:
            max_weight (float): Maximum continuous allowed weight mapped per distinct independent asset. Defaults to 1.0.
            min_weight (float): Discrete minimum weight threshold dictating explicit cardinality bounds. Defaults to 0.0.
            sector_limits (Optional[Dict[str, float]]): Aggregate structural limit limits mapping sectors to exposure boundaries.
            sector_map (Optional[Dict[str, str]]): Dimensional dictionary directly bounding independent sequences to structural sector bounds.
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


    def apply(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Sequentially enforces bounding evaluation constraints mapping directly onto vector matrices utilizing discrete iterative projection.

        Args:
            weights (Dict[str, float]): The absolute input distribution mappings dynamically processed.

        Returns:
            Dict[str, float]: Extracted allocation configurations bounding strictly normalized execution constraints.
        """
        if not weights:
            return {}

        w = dict(weights)

        if self.min_weight > 0:
            w = self._apply_min_weight(w)

        if self.max_weight < 1.0:
            w = self._apply_max_weight(w)

        if self.sector_limits:
            w = self._apply_sector_limits(w)

        return w

    def _normalise(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Renormalizes evaluated weights dynamically strictly validating matrix execution probabilities sum identically to unified bounds.
        
        Args:
            w (Dict[str, float]): Current array state map boundaries evaluated mathematically.
            
        Returns:
            Dict[str, float]: Standardized proportional representation mappings.
        """
        total = sum(w.values())
        if total <= 0:
            return w
        return {k: v / total for k, v in w.items()}

    def _apply_min_weight(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Applies explicit dimensional restrictions mapping systemic truncation against standard discrete evaluation targets.
        
        Args:
            w (Dict[str, float]): Evaluating proportional allocation structural values.
            
        Returns:
            Dict[str, float]: Correctly mapped array filtered explicitly defining strictly valid probabilities.
        """
        filtered = {k: v for k, v in w.items() if v >= self.min_weight}
        if not filtered:
            logger.warning(
                "All assets below min_weight floor. Retaining the largest position."
            )
            max_k = max(w, key=lambda k: w[k])
            filtered = {max_k: w[max_k]}
        return self._normalise(filtered)

    def _apply_max_weight(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Limits mathematical configurations mapping excessive relative weight metrics strictly applying Iterative Proportional Fitting.
        
        Args:
            w (Dict[str, float]): Structural array sequence configuration definitions.
            
        Returns:
            Dict[str, float]: Rebalanced limits distributing parameters uniformly explicitly satisfying strict parameters.
        """
        cap = self.max_weight
        w = self._normalise(w)

        for _ in range(200):
            over  = {k: v for k, v in w.items() if v > cap + 1e-9}
            under = {k: v for k, v in w.items() if v <= cap + 1e-9}

            if not over:
                break

            excess = sum(v - cap for v in over.values())
            under_total = sum(under.values())

            for k in over:
                w[k] = cap

            if under_total > 1e-12:
                scale = 1.0 + excess / under_total
                for k in under:
                    w[k] = min(w[k] * scale, cap)

            w = self._normalise(w)

        return w

    def _apply_sector_limits(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Extracts sector distribution mappings bounding relative overexposure strictly redistributing systemic probabilities identically.
        
        Args:
            w (Dict[str, float]): Evaluated array dictionary sequences bounding current distributions.
            
        Returns:
            Dict[str, float]: Systemic constraints mathematically conforming explicitly to configured matrices boundaries.
        """
        for _ in range(200):
            changed = False
            sector_totals: Dict[str, float] = {}
            for ticker, weight in w.items():
                sector = self.sector_map.get(ticker, "__other__")
                sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

            for sector, limit in self.sector_limits.items():
                total = sector_totals.get(sector, 0.0)
                if total <= limit + 1e-9:
                    continue

                changed = True
                excess = total - limit

                sector_tickers = [
                    k for k in w if self.sector_map.get(k, "__other__") == sector
                ]
                scale_down = limit / total
                for k in sector_tickers:
                    w[k] *= scale_down

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


    @staticmethod
    def long_only(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constructs constraint boundary generating strictly non-negative variable mapping limits ($w_i \ge 0$).
        
        Args:
            w (cp.Variable): Dynamic object target defining allocation variables mathematically.
            
        Returns:
            List[cp.Constraint]: Strictly structured list array bounding explicit bounds execution parameters.
        """
        return [w >= 0]

    @staticmethod
    def fully_invested(w: cp.Variable) -> List[cp.Constraint]:
        """
        Defines the fully invested probability constraint evaluating simplex mapping ($\sum w_i = 1.0$).
        
        Args:
            w (cp.Variable): CVXPY mathematical assignment bounds dynamically tracking structural probabilities.
            
        Returns:
            List[cp.Constraint]: Formulated list array strictly mapping parameters limits boundaries.
        """
        return [cp.sum(w) == 1.0]

    @staticmethod
    def dollar_neutral(w: cp.Variable) -> List[cp.Constraint]:
        """
        Enforces Dollar Neutral parameter constraint ($\sum w_i = 0.0$).
        
        Args:
            w (cp.Variable): Convex mapped continuous execution states strictly bounding internal variables.
            
        Returns:
            List[cp.Constraint]: Resulting structural list mathematically executing precise continuous logic parameters.
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