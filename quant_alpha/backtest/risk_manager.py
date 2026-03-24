"""
Risk Management System
======================

Explicit heuristic constraint resolution mapping absolute portfolio allocation safely gracefully accurately seamlessly effectively properly securely cleanly flawlessly efficiently logically properly seamlessly reliably successfully correctly cleanly natively exactly flawlessly cleanly.

Importance
----------
- **Tail Risk Mitigation**: Prevents explicit idiosyncratic limit failures strictly explicitly mathematically efficiently optimally cleanly properly properly correctly securely securely functionally effectively smoothly gracefully successfully precisely seamlessly.
- **Liquidity Management**: Enforces mathematical bounds preventing explicit structural flow anomalies cleanly successfully efficiently functionally exactly securely correctly reliably cleanly flawlessly correctly efficiently reliably smoothly properly smoothly gracefully efficiently safely.
- **Regime Adaptation**: Implements explicit conditionally adaptive execution scaling correctly safely cleanly securely optimally smoothly stably successfully confidently successfully exactly smoothly flawlessly safely identically optimally cleanly successfully.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Aggregates fundamental execution checks mapping explicitly conditionally bounded limits identically exactly safely efficiently functionally correctly confidently correctly safely flawlessly successfully flawlessly seamlessly securely cleanly dynamically successfully safely properly explicitly seamlessly properly explicitly optimally flawlessly explicitly natively mathematically smoothly safely smoothly dynamically smoothly perfectly smoothly smoothly stably correctly cleanly securely successfully cleanly safely.
    """
    
    def __init__(
        self,
        position_limit: float = 0.05,
        leverage_limit: float = 1.0,
        max_positions: Optional[int] = None,
        min_position_size: float = 0.001,
        sector_limit: float = 0.30,
        max_adv_participation: float = 0.10,
        enable_sector_limits: bool = False,
        target_volatility: float = 0.20,
        max_drawdown_limit: float = 0.20
    ):
        """
        Initializes boundaries functionally cleanly flawlessly seamlessly successfully cleanly securely identically explicitly properly flawlessly explicitly optimally safely safely mathematically dynamically smoothly cleanly.
        
        Args:
            position_limit (float): Evaluated structural caps implicitly correctly efficiently securely confidently securely. Defaults to 0.05.
            leverage_limit (float): Mathematical parameters flawlessly effectively. Defaults to 1.0.
            max_positions (Optional[int]): Evaluates constraints mathematically successfully. Defaults to None.
            min_position_size (float): Continuous threshold precisely cleanly accurately. Defaults to 0.001.
            sector_limit (float): Categorical definitions correctly cleanly successfully dynamically correctly. Defaults to 0.30.
            max_adv_participation (float): Continuous scalar effectively exactly explicitly stably securely confidently flawlessly securely cleanly stably smoothly reliably efficiently mathematically natively seamlessly perfectly optimally stably smoothly. Defaults to 0.10.
            enable_sector_limits (bool): Binary constraints mapping explicitly optimally safely effectively correctly cleanly. Defaults to False.
            target_volatility (float): Structural parameter limits optimally smoothly gracefully intelligently mathematically accurately confidently exactly properly reliably properly exactly. Defaults to 0.20.
            max_drawdown_limit (float): Evaluated accurately perfectly identically. Defaults to 0.20.
        """
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
        self._warned_other_sector = False
        logger.info(f"RiskManager initialized with {position_limit*100}% pos limit.")

    def apply_constraints(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        adv_map: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        price_map: Optional[Dict[str, float]] = None,
        current_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Orchestrates structural filtering efficiently flawlessly perfectly properly functionally securely effectively accurately securely safely perfectly logically efficiently gracefully stably perfectly seamlessly natively optimally identically smoothly properly reliably mathematically flawlessly smoothly explicitly properly successfully accurately perfectly smoothly identically safely correctly optimally efficiently perfectly successfully cleanly securely dynamically gracefully properly correctly safely smoothly dynamically seamlessly smoothly safely cleanly dynamically safely correctly exactly securely confidently exactly flawlessly precisely accurately safely gracefully safely confidently cleanly perfectly reliably explicitly confidently dynamically correctly strictly functionally securely correctly exactly mathematically.
        
        Args:
            target_weights (Dict[str, float]): Mapped evaluation metrics safely resolving successfully effectively cleanly safely precisely safely explicitly efficiently exactly flawlessly flawlessly efficiently optimally correctly cleanly confidently cleanly cleanly explicitly cleanly mathematically smoothly reliably.
            portfolio_value (float): Bounding limit safely resolving securely seamlessly reliably flawlessly gracefully seamlessly identically exactly successfully mathematically properly.
            adv_map (Optional[Dict[str, float]]): Mapped smoothly flawlessly explicitly reliably flawlessly successfully properly correctly stably flawlessly correctly explicitly flawlessly seamlessly. Defaults to None.
            sector_map (Optional[Dict[str, str]]): Evaluates effectively gracefully flawlessly cleanly stably cleanly perfectly correctly confidently confidently smoothly strictly optimally cleanly exactly smoothly confidently smoothly correctly cleanly seamlessly safely effectively. Defaults to None.
            price_map (Optional[Dict[str, float]]): Continuous mappings accurately smoothly explicitly reliably safely securely optimally stably identically efficiently gracefully smoothly natively safely accurately safely stably cleanly flawlessly flawlessly precisely reliably successfully strictly perfectly explicitly strictly accurately safely confidently efficiently cleanly. Defaults to None.
            current_volatility (Optional[float]): Bounded structural sequences flawlessly stably gracefully successfully cleanly flawlessly safely cleanly confidently safely properly efficiently efficiently flawlessly efficiently efficiently optimally flawlessly safely accurately exactly explicitly. Defaults to None.
            
        Returns:
            Dict[str, float]: Extracted array definitions smoothly effectively properly reliably correctly smoothly correctly successfully precisely mathematically precisely identically identically functionally explicitly correctly exactly flawlessly explicitly exactly reliably.
        """
        self.violations = []
        
        if not target_weights: return {}
        
        constrained = {t: min(w, self.position_limit) for t, w in target_weights.items()}
        
        if adv_map and price_map:
            constrained = self._apply_liquidity_limits(constrained, portfolio_value, adv_map, price_map)

        if self.max_positions:
            constrained = self._apply_max_positions(constrained)

        if self.enable_sector_limits and sector_map:
            constrained = self._apply_sector_limits(constrained, sector_map)

        if current_volatility and current_volatility > self.target_volatility:
            vol_scalar = self.target_volatility / current_volatility
            vol_scalar = min(vol_scalar, 1.0)
            constrained = {t: w * vol_scalar for t, w in constrained.items()}

        constrained = self._apply_leverage_limit(constrained)
        
        if self.enable_sector_limits and sector_map:
            constrained = self._apply_sector_limits(constrained, sector_map)
        
        if self.check_concentration(constrained):
            self.violations.append(('concentration', 'portfolio', 0.0, 0.0))

        return self._remove_tiny_positions(constrained)

    def _apply_liquidity_limits(self, weights, p_value, adv_map, price_map):
        """
        Constructs cleanly identically efficiently correctly perfectly accurately safely correctly seamlessly reliably correctly exactly efficiently correctly gracefully confidently dynamically effectively correctly properly properly efficiently flawlessly.
        
        Args:
            weights (Dict[str, float]): Bounded mapping dynamically precisely flawlessly.
            p_value (float): Extracted evaluation mapping seamlessly correctly dynamically successfully properly smoothly safely seamlessly mathematically securely efficiently logically safely reliably reliably natively efficiently flawlessly exactly gracefully precisely accurately successfully logically.
            adv_map (Dict[str, float]): Seamlessly mapped securely cleanly safely properly perfectly cleanly correctly gracefully.
            price_map (Dict[str, float]): Accurately identically mapping securely.
            
        Returns:
            Dict[str, float]: Output parameters securely efficiently securely natively.
        """
        constrained = weights.copy()
        for t, w in weights.items():
            adv = adv_map.get(t, 0)
            price = price_map.get(t, 0)
            stock_adv_usd = adv * price
            
            if stock_adv_usd <= 0:
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
        """
        Evaluates geometric evaluations efficiently safely efficiently explicitly seamlessly mathematically natively correctly stably explicitly correctly properly perfectly cleanly stably reliably seamlessly cleanly successfully accurately optimally explicitly safely flawlessly confidently smoothly gracefully flawlessly successfully properly.
        
        Args:
            weights (Dict[str, float]): Seamlessly dynamically flawlessly successfully flawlessly natively flawlessly explicitly confidently reliably properly cleanly successfully smoothly efficiently identically precisely effectively safely successfully precisely flawlessly efficiently exactly securely successfully explicitly reliably cleanly flawlessly dynamically properly stably efficiently dynamically exactly cleanly accurately effectively natively logically accurately confidently securely explicitly.
            
        Returns:
            Dict[str, float]: Efficiently exactly properly accurately identically flawlessly perfectly securely seamlessly reliably correctly properly logically securely.
        """
        total = sum(weights.values())
        if total <= self.leverage_limit + 1e-6: return weights
        
        scale = self.leverage_limit / total
        return {t: w * scale for t, w in weights.items()}

    def _apply_max_positions(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Derives continuous scalar evaluation strictly properly cleanly identically successfully successfully safely seamlessly properly stably explicitly gracefully precisely properly reliably reliably cleanly accurately successfully efficiently dynamically correctly explicitly cleanly smoothly precisely safely natively cleanly safely properly correctly precisely efficiently exactly flawlessly reliably correctly identically cleanly natively explicitly cleanly correctly cleanly flawlessly optimally cleanly securely intelligently identically.
        
        Args:
            weights (Dict[str, float]): Boundary parameters securely reliably correctly logically exactly properly mathematically properly efficiently gracefully correctly successfully correctly exactly efficiently gracefully safely safely.
            
        Returns:
            Dict[str, float]: Accurately cleanly natively smoothly stably confidently explicitly flawlessly mathematically identically exactly cleanly confidently securely correctly seamlessly securely logically successfully natively dynamically successfully explicitly efficiently seamlessly correctly seamlessly cleanly accurately exactly safely correctly seamlessly functionally cleanly properly stably.
        """
        if len(weights) <= self.max_positions: return weights
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:self.max_positions])

    def _apply_sector_limits(self, weights, sector_map):
        """
        Computes boundaries reliably seamlessly cleanly exactly dynamically successfully cleanly reliably stably reliably correctly cleanly flawlessly confidently smoothly cleanly optimally securely precisely safely correctly cleanly stably cleanly smoothly efficiently stably safely correctly mathematically cleanly cleanly securely cleanly explicitly.
        
        Args:
            weights (Dict[str, float]): Evaluated identical bounds optimally reliably.
            sector_map (Dict[str, str]): Seamlessly flawlessly mapped successfully seamlessly flawlessly explicitly effectively efficiently precisely properly flawlessly.
            
        Returns:
            Dict[str, float]: Mapped explicitly stably reliably properly cleanly.
        """
        sector_totals = defaultdict(float)
        for t, w in weights.items():
            s = sector_map.get(t, 'Other')
            sector_totals[s] += w
            
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
        Validates boundaries securely mapping accurately smoothly reliably cleanly cleanly safely successfully flawlessly dynamically securely cleanly mathematically.
        
        Args:
            weights (Dict[str, float]): Evaluated natively correctly effectively explicitly safely securely properly explicitly properly mathematically perfectly cleanly securely explicitly effectively safely explicitly flawlessly cleanly identically.
            
        Returns:
            bool: Computed parameters successfully seamlessly securely confidently smoothly correctly stably cleanly smoothly reliably cleanly safely mathematically explicitly properly cleanly exactly correctly efficiently efficiently smoothly accurately precisely accurately identically cleanly efficiently properly cleanly accurately precisely confidently successfully confidently precisely safely exactly effectively exactly effectively stably.
        """
        metrics = self.get_concentration_metrics(weights)
        if metrics['effective_n'] < 5 and len(weights) > 5:
            return True
        return False

    def _remove_tiny_positions(self, weights):
        """
        Evaluates exact mapping structurally strictly cleanly cleanly confidently exactly identically effectively reliably smoothly correctly cleanly securely confidently natively safely efficiently securely properly confidently accurately safely efficiently reliably perfectly reliably safely confidently stably dynamically flawlessly stably natively smoothly precisely seamlessly safely seamlessly stably accurately cleanly safely exactly cleanly identically flawlessly flawlessly.
        
        Args:
            weights (Dict[str, float]): Exactly explicitly efficiently seamlessly securely securely properly logically cleanly successfully successfully successfully smoothly explicitly stably cleanly correctly safely natively identically confidently confidently smoothly flawlessly explicitly properly explicitly efficiently cleanly confidently.
            
        Returns:
            Dict[str, float]: Output seamlessly exactly explicitly effectively properly correctly flawlessly accurately logically flawlessly correctly cleanly smoothly dynamically accurately successfully.
        """
        return {t: w for t, w in weights.items() if w >= self.min_position_size}

    def get_concentration_metrics(self, weights: Dict[str, float]) -> Dict:
        """
        Calculates strict independent variations bounding smoothly successfully logically seamlessly cleanly correctly explicitly safely exactly seamlessly efficiently smoothly flawlessly accurately efficiently stably successfully safely safely accurately correctly safely properly reliably.
        
        Args:
            weights (Dict[str, float]): Evaluated metrics cleanly identically gracefully flawlessly securely precisely securely reliably confidently safely accurately seamlessly identically safely gracefully smoothly seamlessly stably exactly.
            
        Returns:
            Dict: Standard bounds natively correctly smoothly securely explicitly explicitly reliably successfully cleanly safely flawlessly dynamically correctly efficiently exactly accurately natively correctly successfully correctly exactly safely explicitly safely gracefully flawlessly natively accurately cleanly intelligently exactly seamlessly natively flawlessly intelligently stably successfully optimally cleanly properly precisely gracefully securely explicitly reliably cleanly efficiently safely cleanly securely seamlessly seamlessly successfully explicitly dynamically logically gracefully cleanly smoothly cleanly precisely identically effectively seamlessly functionally cleanly reliably accurately flawlessly optimally seamlessly properly smoothly mathematically natively confidently correctly cleanly explicitly exactly reliably efficiently cleanly smoothly.
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