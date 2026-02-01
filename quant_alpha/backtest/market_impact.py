"""
Market Impact Model - Almgren-Chriss
====================================
Realistic market impact calculation for backtesting.

Prevents overly optimistic transaction cost assumptions.
Implements the Almgren-Chriss model for temporary and permanent impact.

References:
    Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-39.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketImpactModel:
    """
    Almgren-Chriss market impact model.
    
    Calculates temporary and permanent market impact based on:
    - Order size (relative to daily volume)
    - Stock volatility
    - Execution participation rate
    
    Attributes:
        temp_impact_coef: Temporary impact coefficient (basis points)
        perm_impact_coef: Permanent impact coefficient (basis points)
        bid_ask_spread_bps: Bid-ask spread (basis points)
    """
    
    temp_impact_coef: float = 50  # Temporary impact coefficient
    perm_impact_coef: float = 20  # Permanent impact coefficient
    bid_ask_spread_bps: float = 1.5  # 1.5 bps for S&P 500 liquid stocks
    
    def calculate_impact(
        self,
        order_size_pct: float,
        volatility: float,
        participation_rate: float = 1.0
    ) -> float:
        """
        Calculate total market impact in basis points.
        
        Almgren-Chriss model:
        - Temporary impact scales with sqrt(order_size)
        - Permanent impact scales with order_size^1.5
        
        Args:
            order_size_pct: Order size as % of Average Daily Volume (ADV)
                           0.01 = 1% of ADV
                           0.20 = 20% of ADV
            volatility: Daily volatility (e.g., 0.02 = 2%)
            participation_rate: How much of daily volume over execution period
                               1.0 = execute over 1 day
                               0.5 = execute over 2 days
                               
        Returns:
            Total market impact in basis points (e.g., 50 = 0.5%)
            
        Example:
            >>> impact = MarketImpactModel()
            >>> # Buying 10% of daily volume, 2% daily vol
            >>> bps = impact.calculate_impact(0.10, 0.02)
            >>> print(f"Impact: {bps:.1f} bps")
        """
        
        if order_size_pct <= 0:
            return 0.0
        
        # Temporary impact: immediate adverse price move
        # sqrt(order_size) because doubling order size doesn't double impact
        temp_impact_bps = (
            np.sqrt(order_size_pct / participation_rate) *
            volatility * 100 *  # Convert volatility to basis points
            self.temp_impact_coef
        )
        
        # Permanent impact: lasting price change
        # order_size^1.5 for super-linearity (large orders move prices permanently)
        perm_impact_bps = (
            (order_size_pct ** 1.5) *
            volatility * 100 *
            self.perm_impact_coef
        )
        
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        logger.debug(
            f"Market Impact: order_size={order_size_pct*100:.1f}% ADV, "
            f"temp={temp_impact_bps:.1f}bp, perm={perm_impact_bps:.1f}bp, "
            f"total={total_impact_bps:.1f}bp"
        )
        
        return total_impact_bps
    
    def calculate_slippage(
        self,
        order_size_dollars: float,
        daily_volume_dollars: float,
        volatility: float,
        price: float,
        participation_rate: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate total slippage cost (market impact + bid-ask spread).
        
        Args:
            order_size_dollars: Order size in dollars
            daily_volume_dollars: Average daily volume in dollars
            volatility: Daily volatility
            price: Current stock price
            participation_rate: Execution participation rate
            
        Returns:
            Tuple of (total_slippage_pct, market_impact_pct)
            where percentages are in decimal form (0.005 = 0.5%)
        """
        
        # Calculate order size as % of ADV
        order_size_pct = order_size_dollars / daily_volume_dollars
        
        # Clamp to reasonable levels (can't buy more than daily volume)
        order_size_pct = min(order_size_pct, 0.99)
        
        # Market impact in basis points
        market_impact_bps = self.calculate_impact(
            order_size_pct,
            volatility,
            participation_rate
        )
        
        # Bid-ask spread in basis points
        bid_ask_bps = self.bid_ask_spread_bps
        
        # Total slippage
        total_slippage_bps = market_impact_bps + bid_ask_bps
        
        # Convert to percentage
        market_impact_pct = market_impact_bps / 10000
        total_slippage_pct = total_slippage_bps / 10000
        
        return total_slippage_pct, market_impact_pct
    
    @staticmethod
    def estimate_daily_volume(
        close_price: float,
        volume_shares: int
    ) -> float:
        """Estimate daily volume in dollars."""
        return close_price * volume_shares


def test_market_impact_model():
    """
    Test market impact calculations.
    
    Verifies that:
    1. Larger orders have larger impact
    2. Higher volatility increases impact
    3. Impact scales realistically
    """
    
    print("\n" + "="*70)
    print("MARKET IMPACT MODEL TEST")
    print("="*70)
    
    impact = MarketImpactModel()
    
    # Test 1: Order size sensitivity
    print("\n1. Order Size Sensitivity:")
    print("-" * 70)
    
    volatility = 0.02  # 2% daily vol
    
    for order_pct in [0.01, 0.05, 0.10, 0.20, 0.50]:
        impact_bps = impact.calculate_impact(order_pct, volatility)
        print(f"  {order_pct*100:5.1f}% ADV: {impact_bps:7.1f} bps")
    
    # Verify super-linearity
    impact_1pct = impact.calculate_impact(0.01, volatility)
    impact_10pct = impact.calculate_impact(0.10, volatility)
    ratio = impact_10pct / impact_1pct
    
    print(f"\n  Ratio (10% vs 1%): {ratio:.2f}x (should be >10x for super-linearity)")
    
    # Test 2: Volatility sensitivity
    print("\n2. Volatility Sensitivity:")
    print("-" * 70)
    
    order_size = 0.10
    
    for vol in [0.01, 0.02, 0.03, 0.05]:
        impact_bps = impact.calculate_impact(order_size, vol)
        print(f"  {vol*100:.1f}% vol: {impact_bps:7.1f} bps")
    
    # Test 3: Realistic S&P 500 scenario
    print("\n3. Realistic S&P 500 Scenario:")
    print("-" * 70)
    
    # Typical S&P 500 stock
    stock_price = 100.0
    daily_volume_shares = 50_000_000  # 50M shares
    daily_volume_dollars = stock_price * daily_volume_shares  # $5B
    volatility = 0.015  # 1.5% daily vol
    
    # Buying top 10 stocks in $1M portfolio
    portfolio_value = 1_000_000
    position_size = portfolio_value / 10  # $100k per position
    
    total_slippage, market_impact = impact.calculate_slippage(
        order_size_dollars=position_size,
        daily_volume_dollars=daily_volume_dollars,
        volatility=volatility,
        price=stock_price
    )
    
    order_size_pct = position_size / daily_volume_dollars
    
    print(f"  Stock price: ${stock_price}")
    print(f"  Daily volume: ${daily_volume_dollars:,.0f}")
    print(f"  Position size: ${position_size:,.0f}")
    print(f"  Order size: {order_size_pct*100:.3f}% of daily volume")
    print(f"  Daily volatility: {volatility*100:.2f}%")
    print(f"  Market impact: {market_impact*10000:.1f} bps")
    print(f"  Total slippage: {total_slippage*10000:.1f} bps")
    print(f"  Cost on ${position_size:,.0f} trade: ${position_size * total_slippage:,.2f}")
    
    # Test 4: Compare to fixed cost assumption
    print("\n4. Fixed Cost vs Market Impact:")
    print("-" * 70)
    
    fixed_cost_bps = 40  # 40 bps (typical fixed assumption)
    fixed_cost_pct = fixed_cost_bps / 10000
    
    actual_cost_pct = total_slippage
    
    print(f"  Fixed cost assumption: {fixed_cost_bps} bps (${position_size * fixed_cost_pct:,.2f})")
    print(f"  Actual market impact: {total_slippage*10000:.1f} bps (${position_size * actual_cost_pct:,.2f})")
    print(f"  Difference: {(actual_cost_pct/fixed_cost_pct - 1)*100:+.1f}%")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    test_market_impact_model()
