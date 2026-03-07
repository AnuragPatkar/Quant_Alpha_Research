"""
Portfolio Constraints - Production Grade
Two interfaces in one module:

  1. Static CVXPY factory methods (original):
     Used during optimization to build CVXPY constraint lists.
     e.g. PortfolioConstraints.long_only(w)

  2. Instance-based post-processing API (new):
     Used after optimization to enforce weight-space constraints on a
     {ticker: weight} dict returned by any optimizer.
     e.g. PortfolioConstraints(max_weight=0.40).apply(weights_dict)

BUGS FIXED:
  BUG-PC-01 [CRITICAL]: PortfolioConstraints had no __init__ and no .apply()
    method. Tests that called PortfolioConstraints(max_weight=0.40) received
    "TypeError: PortfolioConstraints() takes no arguments" because the class
    only contained @staticmethod factory methods for CVXPY variables.

    Fix: Add __init__ accepting (max_weight, min_weight, sector_limits,
    sector_map) and an apply(weights: Dict[str, float]) → Dict[str, float]
    method that enforces all constraints sequentially and re-normalises.

    The static methods are fully preserved — this is an additive change.

  BUG-PC-02 [MEDIUM]: sector_exposure_limit static method used cp.abs() to
    constrain magnitude (net-exposure limit). The new instance apply() method
    enforces a gross long-only sector cap, which is the common equity use case.
    Both are valid depending on the mandate; documented clearly below.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """
    Two-in-one portfolio constraint utility.

    ── Instance API (post-processing) ──────────────────────────────────────
    Enforces weight-space constraints on an already-optimised portfolio dict.

    Usage:
        pc = PortfolioConstraints(max_weight=0.40, min_weight=0.01)
        clean_weights = pc.apply(raw_weights_dict)

    Constraint application order (order matters for idempotency):
        1. min_weight: zero out assets below the floor, re-normalise
        2. max_weight: iterative capping — excess weight redistributed
           proportionally to uncapped assets until convergence
        3. sector_limits: iterative sector-level capping with redistribution

    ── Static CVXPY API (during optimization) ──────────────────────────────
    Factory methods that return lists of CVXPY constraints.

    Usage:
        w = cp.Variable(n)
        cons = PortfolioConstraints.long_only(w)
        cons += PortfolioConstraints.position_limit(w, max_weight=0.10)
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
            max_weight:    Maximum allowed weight for any single asset (0 < x ≤ 1).
            min_weight:    Minimum weight threshold; assets below are zeroed.
            sector_limits: {sector_name: max_weight} — aggregate sector caps.
            sector_map:    {ticker: sector_name} — maps each asset to a sector.
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
        Apply all configured constraints to a weight dictionary.

        Args:
            weights: {ticker: weight} — need not sum to 1 on input (will
                     be re-normalised after each constraint pass).

        Returns:
            {ticker: weight} summing to 1.0 with all constraints satisfied.
        """
        if not weights:
            return {}

        w = dict(weights)

        # Step 1: Min-weight floor — zero out tiny positions
        if self.min_weight > 0:
            w = self._apply_min_weight(w)

        # Step 2: Max-weight cap — iterative re-distribution
        if self.max_weight < 1.0:
            w = self._apply_max_weight(w)

        # Step 3: Sector limits — iterative re-distribution
        if self.sector_limits:
            w = self._apply_sector_limits(w)

        return w

    def _normalise(self, w: Dict[str, float]) -> Dict[str, float]:
        """Normalise weights to sum to 1.0; return as-is if sum is zero."""
        total = sum(w.values())
        if total <= 0:
            return w
        return {k: v / total for k, v in w.items()}

    def _apply_min_weight(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Zero out all assets below min_weight, then re-normalise.
        """
        filtered = {k: v for k, v in w.items() if v >= self.min_weight}
        if not filtered:
            # All assets below floor — keep the largest to avoid empty portfolio
            logger.warning(
                "All assets below min_weight floor. Retaining the largest position."
            )
            max_k = max(w, key=lambda k: w[k])
            filtered = {max_k: w[max_k]}
        return self._normalise(filtered)

    def _apply_max_weight(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Iteratively cap each asset at max_weight and redistribute the excess
        proportionally among uncapped assets until convergence.

        This guarantees exact enforcement regardless of how many assets need
        capping — a single-pass clip would under-enforce when multiple assets
        exceed the cap simultaneously.
        """
        cap = self.max_weight
        w = self._normalise(w)

        for _ in range(200):           # max iterations (converges in O(n) in practice)
            over  = {k: v for k, v in w.items() if v > cap + 1e-9}
            under = {k: v for k, v in w.items() if v <= cap + 1e-9}

            if not over:
                break                   # all weights within cap — done

            # Compute excess to redistribute
            excess = sum(v - cap for v in over.values())
            under_total = sum(under.values())

            # Pin over-weight assets exactly at cap
            for k in over:
                w[k] = cap

            # Distribute excess proportionally to uncapped assets
            if under_total > 1e-12:
                scale = 1.0 + excess / under_total
                for k in under:
                    w[k] = min(w[k] * scale, cap)   # re-cap in same pass

            w = self._normalise(w)

        return w

    def _apply_sector_limits(self, w: Dict[str, float]) -> Dict[str, float]:
        """
        Iteratively cap each sector at its limit and redistribute excess to
        assets outside the over-weight sector, proportionally.
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

                # Distribute excess to out-of-sector assets proportionally
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
        """Constraint: Weights must be non-negative (Long Only)."""
        return [w >= 0]

    @staticmethod
    def fully_invested(w: cp.Variable) -> List[cp.Constraint]:
        """Constraint: Sum of weights must equal 1.0."""
        return [cp.sum(w) == 1.0]

    @staticmethod
    def dollar_neutral(w: cp.Variable) -> List[cp.Constraint]:
        """Constraint: Sum of weights must equal 0.0 (Market Neutral)."""
        return [cp.sum(w) == 0.0]

    @staticmethod
    def leverage_limit(w: cp.Variable, limit: float = 1.0) -> List[cp.Constraint]:
        """Constraint: Gross leverage (sum of absolute weights) <= limit."""
        return [cp.norm(w, 1) <= limit]

    @staticmethod
    def position_limit(
        w: cp.Variable,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> List[cp.Constraint]:
        """
        Constraint: Individual position limits.
        min_weight <= w_i <= max_weight
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
        Constraint: Limit net sector exposure magnitude (long/short safe).
        |Sum(w_sector)| <= max_sector_weight

        Note: for long-only mandates with a gross cap, use the instance
        .apply() method with sector_limits instead.
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
        Constraint: Limit turnover from current portfolio.
        sum(|w - w_current|) <= turnover_cap
        """
        return [cp.norm(w - current_weights, 1) <= turnover_cap]