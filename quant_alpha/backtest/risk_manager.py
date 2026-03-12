"""
Risk Management System
======================
Enforces portfolio construction constraints and risk limits.

Purpose
-------
The `RiskManager` serves as the final gatekeeper in the portfolio construction
process. It applies a hierarchical set of constraints—ranging from hard regulatory
limits (concentration, leverage) to soft liquidity checks (ADV participation).
Unlike optimization-based approaches (e.g., quadratic programming), this module uses
a greedy, heuristic waterfall ($O(N)$) to ensure robust and deterministic behavior
during market stress.

Usage
-----
.. code-block:: python

    risk_manager = RiskManager(
        position_limit=0.05,        # 5% max per ticker
        leverage_limit=1.0,         # 100% gross exposure
        max_adv_participation=0.02  # Max 2% of daily volume
    )

    safe_weights = risk_manager.apply_constraints(
        target_weights=raw_signals,
        portfolio_value=1_000_000,
        adv_map={"AAPL": 50_000_000},
        price_map={"AAPL": 150.0},
        current_volatility=0.25
    )

Importance
----------
- **Tail Risk Mitigation**: Prevents single-name blowups via strict concentration caps.
- **Liquidity Management**: Ensures position sizes are realistic relative to market
  depth ($Position_{\$} \le \alpha \times ADV_{\$}$), preventing high market impact costs.
- **Regime Adaptation**: Dynamically de-leverages the portfolio when realized volatility
  exceeds target thresholds ($\sigma_{realized} > \sigma_{target}$).

Tools & Frameworks
------------------
- **Pandas/NumPy**: Used for efficient vector aggregation and logic masking.
- **Collections (defaultdict)**: Optimized grouping for sector-level aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Orchestrates constraint application for equity portfolios.

    Implements a multi-stage filter pipeline to transform raw signal weights
    into execution-ready, compliant portfolio weights.
    """
    
    def __init__(
        self,
        position_limit: float = 0.05,        # Single-name concentration cap (5%)
        leverage_limit: float = 1.0,         # Gross exposure limit (100%)
        max_positions: Optional[int] = None, # Hard cardinality constraint
        min_position_size: float = 0.001,    # Minimum tradeable threshold (0.1%)
        sector_limit: float = 0.30,          # GICS Sector exposure cap (30%)
        max_adv_participation: float = 0.10, # Liquidity constraint (10% of ADV)
        enable_sector_limits: bool = False,
        target_volatility: float = 0.20,     # Annualized Volatility Target (20%)
        max_drawdown_limit: float = 0.20     # Peak-to-Trough Decline Limit
    ):
        self.position_limit = position_limit
        self.leverage_limit = leverage_limit
        self.max_positions = max_positions
        self.min_position_size = min_position_size
        self.sector_limit = sector_limit
        self.max_adv_participation = max_adv_participation
        self.enable_sector_limits = enable_sector_limits
        self.target_volatility = target_volatility
        self.max_drawdown_limit = max_drawdown_limit
        
        self.violations: List[Tuple] = []
        self._warned_other_sector = False # Prevent log spam
        logger.info(f"RiskManager initialized with {position_limit*100}% pos limit.")

    def apply_constraints(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        adv_map: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        price_map: Optional[Dict[str, float]] = None,
        current_volatility: Optional[float] = None # Realized volatility for regime checks
    ) -> Dict[str, float]:
        """
        Transforms target weights to comply with all active constraints.
        
        Algorithm:
        1. Clip individual weights to `position_limit`.
        2. Cap weights based on Average Daily Volume (ADV) liquidity.
        3. Truncate tail to enforce `max_positions`.
        4. Normalize sector exposures to `sector_limit`.
        5. Scale gross exposure if `current_volatility` > `target_volatility`.
        6. Global leverage normalization.
        """
        self.violations = [] # Reset violations for this run
        
        if not target_weights: return {}
        
        # 1. Single-Name Constraint (Hard Cap)
        # $w_i = \min(w_i, L_{pos})$
        constrained = {t: min(w, self.position_limit) for t, w in target_weights.items()}
        
        # 2. Liquidity Constraint (ADV Participation)
        # Ensures exit capability: $Pos_{\$} \le \gamma \times ADV_{\$}$
        if adv_map and price_map:
            constrained = self._apply_liquidity_limits(constrained, portfolio_value, adv_map, price_map)

        # 3. Cardinality Constraint
        if self.max_positions:
            constrained = self._apply_max_positions(constrained)

        # 4. Sector Risk Control
        if self.enable_sector_limits and sector_map:
            constrained = self._apply_sector_limits(constrained, sector_map)

        # 5. Volatility Targeting (Dynamic De-leverage)
        # Regime-Conditional Scaling: $\sigma_{realized} > \sigma_{target} \implies L_{new} = L_{old} \times \frac{\sigma_{target}}{\sigma_{realized}}$
        if current_volatility and current_volatility > self.target_volatility:
            vol_scalar = self.target_volatility / current_volatility
            # Asymmetric Scaling: Only de-leverage during high vol; do not re-leverage in low vol.
            vol_scalar = min(vol_scalar, 1.0)
            constrained = {t: w * vol_scalar for t, w in constrained.items()}

        # 6. Global Gross Leverage Normalization
        constrained = self._apply_leverage_limit(constrained)
        
        # 6b. Re-verify Sector Constraints (Prevent drift from scaling)
        if self.enable_sector_limits and sector_map:
            constrained = self._apply_sector_limits(constrained, sector_map)
        
        # 7. Herfindahl-Hirschman Index (HHI) Check
        # Monitoring effective breadth ($N_{eff}$) to detect over-concentration.
        if self.check_concentration(constrained):
            self.violations.append(('concentration', 'portfolio', 0.0, 0.0))
            # Note: Violation logged for audit; soft constraint does not block execution.

        # 8. Clean up micro-positions (Dust Pruning)
        return self._remove_tiny_positions(constrained)

    def _apply_liquidity_limits(self, weights, p_value, adv_map, price_map):
        """
        Caps position weight to a fraction of daily dollar volume.
        Constraint: $w_i \times V_{port} \le Limit_{adv} \times P_i \times Vol_i$
        """
        constrained = weights.copy()
        for t, w in weights.items():
            # Zero-weight policy for missing data to ensure conservative execution.
            adv = adv_map.get(t, 0)
            price = price_map.get(t, 0)
            stock_adv_usd = adv * price
            
            if stock_adv_usd <= 0:
                # Asset is effectively illiquid or data is stale; force exit.
                constrained[t] = 0.0
                self.violations.append(('liquidity_missing', t, w, 0.0))
                continue
            
            max_pos_usd = stock_adv_usd * self.max_adv_participation
            max_weight_liq = max_pos_usd / p_value if p_value > 0 else 0
            
            if w > max_weight_liq:
                constrained[t] = max_weight_liq
                self.violations.append(('liquidity', t, w, max_weight_liq))
                
        return constrained

    def _apply_leverage_limit(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Scales weights pro-rata if sum exceeds leverage limit."""
        total = sum(weights.values())
        if total <= self.leverage_limit + 1e-6: return weights
        
        scale = self.leverage_limit / total
        return {t: w * scale for t, w in weights.items()}

    def _apply_max_positions(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Enforces cardinality constraint by retaining Top-N largest weights."""
        if len(weights) <= self.max_positions: return weights
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:self.max_positions])

    def _apply_sector_limits(self, weights, sector_map):
        """Groups exposures by sector and scales down violations pro-rata."""
        sector_totals = defaultdict(float)
        for t, w in weights.items():
            s = sector_map.get(t, 'Other')
            sector_totals[s] += w
            
        # Data Integrity Check: Monitor residual 'Other' exposure.
        if 'Other' in sector_totals and sector_totals['Other'] > 0.10 and not self._warned_other_sector:
            logger.warning(f"⚠️ High unclassified sector exposure: {sector_totals['Other']:.1%}. Check data quality. (Logged once)")
            self._warned_other_sector = True
            
        sector_scales = {}
        for s, total in sector_totals.items():
            if total > self.sector_limit:
                sector_scales[s] = self.sector_limit / total
                self.violations.append(('sector', s, total, self.sector_limit))
        
        if not sector_scales:
            return weights
            
        constrained = weights.copy()
        for t, w in constrained.items():
            s = sector_map.get(t, 'Other')
            if s in sector_scales:
                constrained[t] *= sector_scales[s]
        return constrained
        
    def check_concentration(self, weights: Dict[str, float]) -> bool:
        """
        Returns True if portfolio breadth ($N_{eff}$) is critically low.
        Uses Herfindahl-Hirschman Index (HHI) inverse.
        """
        metrics = self.get_concentration_metrics(weights)
        # Threshold: Effective breadth < 5 stocks indicates extreme idiosyncratic risk.
        if metrics['effective_n'] < 5 and len(weights) > 5:
            return True
        return False

    def _remove_tiny_positions(self, weights):
        """Prunes sub-threshold positions to minimize operational overhead (Dust cleanup)."""
        return {t: w for t, w in weights.items() if w >= self.min_position_size}

    def get_concentration_metrics(self, weights: Dict[str, float]) -> Dict:
        """
        Calculates concentration metrics.
        
        Formula:
        $HHI = \sum_{i} w_i^2$ (Normalized weights)
        $N_{eff} = 1 / HHI$
        """
        if not weights: return {'hhi': 0, 'effective_n': 0}
        total_w = sum(weights.values())
        if total_w == 0: return {'hhi': 0, 'effective_n': 0}
        
        normalized_w = [w/total_w for w in weights.values()]
        hhi = sum(w**2 for w in normalized_w)
        return {
            'hhi': hhi,
            'effective_n': 1/hhi if hhi > 0 else 0
        }