"""
Market Impact Models
Realistic price impact from trading

Models:
1. AlmgrenChrissImpact: Institutional standard (Permanent + Temporary components)
2. SimpleImpactModel: Simplified square-root model

Academic Foundation:
- Almgren & Chriss (2000): Optimal execution
- Almgren (2003): Optimal execution with nonlinear impact functions
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class AlmgrenChrissImpact:
    """
    Almgren-Chriss market impact model (Optimized)
    
    Total Impact = Permanent Impact + Temporary Impact
    
    1. Permanent (Information Leakage):
       I = γ * σ * (shares / volume)^α
       Usually linear (α=1.0)
       
    2. Temporary (Liquidity Cost):
       J = η * σ * (shares / volume)^β
       Usually concave (β=0.5 to 0.6)
    
    Where:
    - σ: Daily volatility (normalized by reference vol)
    - shares/volume: Participation rate
    """
    
    def __init__(
        self,
        eta: float = 0.15,          # Temporary impact coefficient (Liquidity)
        gamma: float = 0.1,         # Permanent impact coefficient (Info)
        alpha: float = 1.0,         # Permanent exponent (Linear)
        beta: float = 0.6,          # Temporary exponent (Concave)
        vol_ref: float = 0.02       # Reference volatility (2%)
    ):
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.vol_ref = vol_ref
        
        logger.info(
            f"AlmgrenChriss initialized: eta={eta}, gamma={gamma}, "
            f"alpha={alpha}, beta={beta}"
        )

    def __repr__(self):
        return (f"AlmgrenChrissImpact(eta={self.eta}, gamma={self.gamma}, "
                f"alpha={self.alpha}, beta={self.beta})")
    
    def calculate_impact(
        self,
        shares: float,
        volume: float,
        volatility: float,
        side: str = 'buy'
    ) -> float:
        """
        Calculate total market impact cost as a fraction of price.
        
        Args:
            shares: Trade size (absolute value used internally)
            volume: Daily average volume
            volatility: Daily volatility (decimal, e.g. 0.02)
            side: 'buy' or 'sell' (selling has higher impact in stress)
            
        Returns:
            Impact cost rate (e.g. 0.0010 for 10bps)
        """
        if volume <= 0 or shares == 0:
            return 0.0
        
        # 1. Participation Rate (Always positive)
        participation = abs(shares) / volume
        
        # NEW: Safety Penalty for excessive participation (>10% ADV)
        penalty = 1.0
        if participation > 0.10:
            penalty = 1 + (participation - 0.10) * 5 # Exponential penalty
        
        # 2. Volatility Scaling
        vol_scale = volatility / self.vol_ref if self.vol_ref > 0 else 1.0
        
        # NEW: Asymmetry (Selling in a crisis is costlier)
        side_mult = 1.1 if side == 'sell' else 1.0
        
        # 3. Calculate Components
        # Permanent: Moves the mid-price (Linear)
        perm_impact = self.gamma * (participation ** self.alpha) * vol_scale * side_mult
        
        # Temporary: Cost of liquidity (Concave)
        temp_impact = self.eta * (participation ** self.beta) * vol_scale * penalty
        
        return perm_impact + temp_impact
    
    def estimate_capacity(
        self,
        max_impact_bps: float,
        volume: float,
        volatility: float
    ) -> int:
        """
        Estimate max trade size for a given impact limit (bps).
        Approximation assuming temporary impact dominates execution cost.
        """
        if volume <= 0: return 0
        
        max_impact = max_impact_bps / 10000.0
        vol_scale = volatility / self.vol_ref if self.vol_ref > 0 else 1.0
        
        # Solve J = eta * vol_scale * (shares/volume)^beta for shares
        # shares = volume * (J / (eta * vol_scale))^(1/beta)
        
        denom = self.eta * vol_scale
        if denom == 0: return int(volume)
        
        participation = (max_impact / denom) ** (1.0 / self.beta)
        
        # Cap at 100% ADV to be realistic
        participation = min(participation, 1.0)
        
        return int(volume * participation)


class SimpleImpactModel:
    """
    Simplified square-root impact model
    Impact = k * sqrt(|shares| / volume)
    """
    def __init__(self, k: float = 0.1):
        self.k = k
    
    def __repr__(self):
        return f"SimpleImpactModel(k={self.k})"
    
    def calculate_impact(self, shares: float, volume: float, volatility: float = 0.0, **kwargs) -> float:
        if volume <= 0 or shares == 0: return 0.0
        return self.k * np.sqrt(abs(shares) / volume)

    def estimate_capacity(self, max_impact_bps: float, volume: float, volatility: float = 0.0, **kwargs) -> int:
        """
        Estimate max trade size for a given impact limit (bps).
        """
        if volume <= 0: return 0
        max_impact = max_impact_bps / 10000.0
        
        if self.k == 0: return int(volume)
        
        participation = (max_impact / self.k) ** 2
        participation = min(participation, 1.0)
        
        return int(volume * participation)


def compare_impact_models(shares: float, volume: float, volatility: float = 0.02):
    """
    Helper to compare impact estimates from different models.
    """
    ac_model = AlmgrenChrissImpact()
    simple_model = SimpleImpactModel()
    
    ac_impact = ac_model.calculate_impact(shares, volume, volatility)
    simple_impact = simple_model.calculate_impact(shares, volume)
    
    logger.info(f"--- Impact Comparison ---")
    logger.info(f"Trade: {shares:,.0f} shares | Volume: {volume:,.0f} | Volatility: {volatility:.1%}")
    logger.info(f"Almgren-Chriss: {ac_impact*10000:.2f} bps")
    logger.info(f"Simple Model:   {simple_impact*10000:.2f} bps")
    logger.info(f"Difference:     {abs(ac_impact - simple_impact)*10000:.2f} bps")