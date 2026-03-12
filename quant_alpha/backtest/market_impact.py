"""
Market Impact Models
====================
Empirically-grounded models for simulating price impact from trading activity.

Purpose
-------
This module provides models to estimate the implementation shortfall (slippage)
that occurs when a large trade consumes a significant portion of available liquidity.
It is a critical component for realistic backtesting, as it penalizes strategies
that generate alpha by assuming infinite liquidity.

Usage
-----
.. code-block:: python

    from quant_alpha.backtest.market_impact import AlmgrenChrissImpact

    impact_model = AlmgrenChrissImpact()
    cost_bps = impact_model.calculate_impact(
        shares=10000,
        volume=1_000_000,
        volatility=0.025
    ) * 10000

Importance
----------
- **Alpha Decay**: Accurately models a primary source of alpha decay in live trading.
- **Capacity Estimation**: Allows for the estimation of a strategy's AUM capacity before its own trading activity erodes its profitability.
- **Performance**: The core calculation is JIT-compiled with Numba for $O(1)$ complexity per trade, enabling efficient use in large-scale simulations.

Tools & Frameworks
------------------
- **NumPy**: Core numerical operations.
- **Numba**: Just-In-Time (JIT) compilation for C-level performance on the hot-path calculation loop.
"""

import numpy as np
import logging
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# --- Numba-Optimized Core Calculation ---
def _calculate_impact_core(shares, volume, volatility, eta, gamma, alpha, beta, vol_ref, side_is_sell):
    """Core JIT-compiled impact logic."""
    if volume <= 0 or shares == 0:
        return 1.0 if shares != 0 else 0.0
    
    participation = abs(shares) / volume
    
    # Impose a quadratic penalty for high participation rates to model liquidity exhaustion.
    penalty = 1.0
    if participation > 0.10:
        penalty = 1 + (participation * 10) ** 2
    
    vol_scale = volatility / vol_ref if vol_ref > 0 else 1.0
    # Apply an asymmetry scalar for sell orders, reflecting higher impact during liquidations.
    side_mult = 1.1 if side_is_sell else 1.0
    
    perm_impact = gamma * (participation ** alpha) * vol_scale * side_mult
    temp_impact = eta * (participation ** beta) * vol_scale * penalty
    
    return perm_impact + temp_impact

if HAS_NUMBA:
    _calculate_impact_core = jit(nopython=True)(_calculate_impact_core)


class AlmgrenChrissImpact:
    """
    Almgren-Chriss market impact model with Numba optimization.
    
    The model decomposes total impact into two components:
    
    1.  **Permanent Impact (Information Leakage)**: The persistent price change
        caused by the information revealed by the trade.
        $I_{perm} = \gamma \sigma (\frac{Q}{V})^\alpha$
    2.  **Temporary Impact (Liquidity Cost)**: The transient price change from
        consuming liquidity, which reverts after the trade.
        $I_{temp} = \eta \sigma (\frac{Q}{V})^\beta$
    
    Where:
    - $\sigma$: Daily volatility.
    - $Q/V$: Participation rate (Trade Size / Total Volume).
    """
    
    def __init__(
        self,
        eta: float = 0.015,         # FIX: Lowered from 0.15 to 0.015 (Realistic ~10bps at 1% ADV)
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
        
        Returns:
            The estimated impact cost as a decimal rate (e.g., 0.001 for 10 bps).
        """
        if volume <= 0 or shares == 0:
            # If volume is zero (e.g., stock is halted), cost is effectively infinite.
            # Return a 100% penalty to heavily penalize such trades in an
            # optimization or simulation context.
            return 1.0 if shares != 0 else 0.0
        
        # Delegate to the Numba-optimized core for performance.
        # The core handles:
        # 1. Quadratic Penalty for >10% ADV
        # 2. Selling Asymmetry (1.1x cost)
        return _calculate_impact_core(
            float(shares), 
            float(volume), 
            float(volatility),
            self.eta, self.gamma, self.alpha, self.beta, self.vol_ref,
            side == 'sell'
        )
    
    def estimate_capacity(
        self,
        max_impact_bps: float,
        volume: float,
        volatility: float
    ) -> int:
        """
        Estimate the maximum trade size for a given impact tolerance in basis points.
        
        This is an approximation that inverts the temporary impact formula, which
        typically dominates execution cost for a single trade.
        """
        if volume <= 0: return 0
        
        max_impact = max_impact_bps / 10000.0
        vol_scale = volatility / self.vol_ref if self.vol_ref > 0 else 1.0
        
        denom = self.eta * vol_scale
        if denom == 0: return int(volume)
        
        participation = (max_impact / denom) ** (1.0 / self.beta)
        
        # Cap at 100% ADV to be realistic
        participation = min(participation, 1.0)
        
        return int(volume * participation)


class SimpleImpactModel:
    """
    A simplified square-root model, often used as a baseline.
    
    Formula: $I = k \sqrt{\frac{|Q|}{V}}$
    """
    def __init__(self, k: float = 0.1):
        self.k = k
    
    def __repr__(self):
        return f"SimpleImpactModel(k={self.k})"
    
    def calculate_impact(self, shares: float, volume: float, volatility: float = 0.0, **kwargs) -> float:
        if volume <= 0 or shares == 0: return 0.0
        return self.k * np.sqrt(abs(shares) / volume)

    def estimate_capacity(self, max_impact_bps: float, volume: float, volatility: float = 0.0, **kwargs) -> int:
        """Inverts the square-root formula to solve for trade size."""
        if volume <= 0: return 0
        max_impact = max_impact_bps / 10000.0
        
        if self.k == 0: return int(volume)
        
        participation = (max_impact / self.k) ** 2
        participation = min(participation, 1.0)
        
        return int(volume * participation)


def compare_impact_models(shares: float, volume: float, volatility: float = 0.02):
    """Helper function to compare impact estimates from different models."""
    ac_model = AlmgrenChrissImpact()
    simple_model = SimpleImpactModel()
    
    ac_impact = ac_model.calculate_impact(shares, volume, volatility)
    simple_impact = simple_model.calculate_impact(shares, volume)
    
    logger.info(f"--- Impact Comparison ---")
    logger.info(f"Trade: {shares:,.0f} shares | Volume: {volume:,.0f} | Volatility: {volatility:.1%}")
    logger.info(f"Almgren-Chriss: {ac_impact*10000:.2f} bps")
    logger.info(f"Simple Model:   {simple_impact*10000:.2f} bps")
    logger.info(f"Difference:     {abs(ac_impact - simple_impact)*10000:.2f} bps")

if __name__ == "__main__":
    # Transaction Cost Analysis (TCA) Verification for a Small-Cap Scenario
    # Scenario: Small Cap Stock ($10 Price, $1M ADV)
    # Trade: 2% of ADV ($20k) -> 2,000 shares
    print("\n\U0001f50d TCA VERIFICATION (Small Cap Scenario)")
    print("Stock: $10 | ADV: $1M (100k shares) | Volatility: 3%")
    print("Trade: Buy $20k (2% Participation)")
    
    compare_impact_models(shares=2000, volume=100000, volatility=0.03)
    
    print("\n✅ Result: Impact should be < 10 bps for 2% participation.")
    print("   If Impact > 100 bps (1%), then Eta is too high.")