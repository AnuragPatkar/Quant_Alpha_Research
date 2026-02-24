"""
Risk Manager (Finalized Version)
Portfolio risk controls, sector constraints, and liquidity-adjusted limits.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Portfolio risk management system.
    Enforces Position, Leverage, Sector, and Liquidity limits.
    """
    
    def __init__(
        self,
        position_limit: float = 0.05,        # Max 5% per stock
        leverage_limit: float = 1.0,         # Max 100% total exposure
        max_positions: Optional[int] = None, # Max count of holdings
        min_position_size: float = 0.001,    # Minimum tradeable size (0.1%)
        sector_limit: float = 0.30,          # Max 30% per sector
        max_adv_participation: float = 0.10, # Max 10% of Daily Volume
        enable_sector_limits: bool = False,
        target_volatility: float = 0.20      # NEW: Target 20% Annual Volatility
    ):
        self.position_limit = position_limit
        self.leverage_limit = leverage_limit
        self.max_positions = max_positions
        self.min_position_size = min_position_size
        self.sector_limit = sector_limit
        self.max_adv_participation = max_adv_participation
        self.enable_sector_limits = enable_sector_limits
        self.target_volatility = target_volatility
        
        self.violations: List[Tuple] = []
        logger.info(f"RiskManager initialized with {position_limit*100}% pos limit.")

    def apply_constraints(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        adv_map: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        price_map: Optional[Dict[str, float]] = None,
        current_volatility: Optional[float] = None # NEW: Input for Vol Targeting
    ) -> Dict[str, float]:
        """
        Filters and scales weights to meet all risk criteria.
        """
        self.violations = [] # Reset violations for this run
        
        if not target_weights: return {}
        
        # 1. Position Limits (Individual Cap)
        constrained = {t: min(w, self.position_limit) for t, w in target_weights.items()}
        
        # 2. Liquidity Cap (ADV Limit)
        # Prevents taking a position that would take too long to exit
        if adv_map and price_map:
            constrained = self._apply_liquidity_limits(constrained, portfolio_value, adv_map, price_map)

        # 3. Max Positions Count
        if self.max_positions:
            constrained = self._apply_max_positions(constrained)

        # 4. Sector Exposure
        if self.enable_sector_limits and sector_map:
            constrained = self._apply_sector_limits(constrained, sector_map)

        # 5. Volatility Targeting (Dynamic De-leverage)
        # Agar market volatility target se zyada hai, to exposure kam karo
        if current_volatility and current_volatility > self.target_volatility:
            vol_scalar = self.target_volatility / current_volatility
            # Cap scalar at 1.0 (Hum leverage badhayenge nahi, sirf ghatayenge)
            vol_scalar = min(vol_scalar, 1.0)
            constrained = {t: w * vol_scalar for t, w in constrained.items()}

        # 5. Global Leverage Scaling
        constrained = self._apply_leverage_limit(constrained)
        
        # 6. HHI Concentration Monitor
        # Warn if portfolio is effectively holding < 5 stocks
        if self.check_concentration(constrained):
            self.violations.append(('concentration', 'portfolio', 0.0, 0.0))
            # Note: We log violation but don't block trade to avoid stuck positions, 
            # but this flag can be used by Engine to halt new entries.

        # 7. Final cleanup (Remove dust)
        return self._remove_tiny_positions(constrained)

    def _apply_liquidity_limits(self, weights, p_value, adv_map, price_map):
        """Caps position weight based on its dollar-liquidity."""
        constrained = weights.copy()
        for t, w in weights.items():
            # Default to 0 if data missing, which forces weight to 0 (Safe)
            adv = adv_map.get(t, 0)
            price = price_map.get(t, 0)
            stock_adv_usd = adv * price
            
            if stock_adv_usd <= 0:
                # Fix: Treat missing volume as illiquid -> Force exit
                constrained[t] = 0.0
                self.violations.append(('liquidity_missing', t, w, 0.0))
            
            # Max $ position = X% of Daily $ Volume
            max_pos_usd = stock_adv_usd * self.max_adv_participation
            max_weight_liq = max_pos_usd / p_value if p_value > 0 else 0
            
            if w > max_weight_liq:
                constrained[t] = max_weight_liq
                self.violations.append(('liquidity', t, w, max_weight_liq))
                
        return constrained

    def _apply_leverage_limit(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= self.leverage_limit + 1e-6: return weights
        
        scale = self.leverage_limit / total
        return {t: w * scale for t, w in weights.items()}

    def _apply_max_positions(self, weights: Dict[str, float]) -> Dict[str, float]:
        if len(weights) <= self.max_positions: return weights
        # Keep top N
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:self.max_positions])

    def _apply_sector_limits(self, weights, sector_map):
        sector_totals = defaultdict(float)
        for t, w in weights.items():
            sector_totals[sector_map.get(t, 'Other')] += w
            
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
        """Returns True if portfolio is too concentrated (HHI check)."""
        metrics = self.get_concentration_metrics(weights)
        # If effective N < 5, it's too risky
        if metrics['effective_n'] < 5 and len(weights) > 5:
            return True
        return False

    def _remove_tiny_positions(self, weights):
        return {t: w for t, w in weights.items() if w >= self.min_position_size}

    def get_concentration_metrics(self, weights: Dict[str, float]) -> Dict:
        """HHI calculation: Higher = More concentrated (Risky)"""
        if not weights: return {'hhi': 0, 'effective_n': 0}
        total_w = sum(weights.values())
        if total_w == 0: return {'hhi': 0, 'effective_n': 0}
        
        normalized_w = [w/total_w for w in weights.values()]
        hhi = sum(w**2 for w in normalized_w)
        return {
            'hhi': hhi,
            'effective_n': 1/hhi if hhi > 0 else 0 # How many stocks it 'feels' like you own
        }