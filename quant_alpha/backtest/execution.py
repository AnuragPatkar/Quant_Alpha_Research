"""
Execution Simulator
===================
Stochastic execution engine for realistic trade simulation.

Purpose
-------
Simulates the microstructure mechanics of trade execution, modeling the
discrepancy between theoretical signal prices and realized fill prices
(Implementation Shortfall). It accounts for explicit costs (commissions)
and implicit costs (spread, volatility-adjusted slippage, and market impact).

Usage
-----
.. code-block:: python

    executor = ExecutionSimulator(commission_rate=0.001, spread_bps=0.0005)
    fill = executor.execute_order(
        ticker="AAPL",
        shares=100,
        side="buy",
        price=150.0,
        volume=50_000_000,
        date="2023-10-25",
        volatility=0.02
    )

Importance
----------
- **Alpha Preservation**: Prevents "paper trading" bias by penalizing high-turnover strategies.
- **Stochasticity**: Uses log-normal slippage distributions to model tail risk in execution.
- **Liquidity Awareness**: Enforces participation limits and probabilistic fill failures.

Tools & Frameworks
------------------
- **NumPy**: Stochastic generation (LogNormal) for slippage variance.
- **Pandas**: Efficient batch alignment of market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Literal, List, Optional
import logging

logger = logging.getLogger(__name__)

class ExecutionSimulator:
    """
    Simulates trade execution with configurable friction models.
    
    Cost Model:
    $P_{fill} = P_{mkt} \pm (Cost_{spread} + Cost_{slippage} + Cost_{impact})$
    
    Where:
    - $Cost_{slippage} \sim \text{LogNormal}(0, \sigma_{vol})$
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,              # 10 bps (0.1%)
        spread_bps: float = 0.0005,                  # 5 bps (Half-Spread)
        slippage_bps: float = 0.0002,                # 2 bps (Base Slippage)
        fill_prob: float = 1.0,                      # 100% Fill Rate (Probability of execution)
        commission_per_share: Optional[float] = None # Override for per-share pricing (e.g., $0.005)
    ):
        self.commission_rate = commission_rate
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.fill_prob = fill_prob
        self.commission_per_share = commission_per_share
        
        logger.info(f"Executor Config: Comm={commission_rate*10000}bps, Spread={spread_bps*10000}bps")

    def execute_order(
        self,
        ticker: str,
        shares: int,
        side: Literal['buy', 'sell'],
        price: float,
        volume: float,
        date: str,
        impact_rate: float = 0.0,  # Explicit naming: Rate, not USD
        volatility: float = 0.02   # Used to scale slippage variance
    ) -> Dict:
        """
        Computes realized fill price and transaction costs.
        
        Commission is calculated separately from price-based friction to facilitate
        accurate accounting in the Portfolio module (Cost Basis vs Expense).
        
        Returns:
            Dict containing execution metadata, costs, and fill status.
        """
        # 0. Liquidity Constraints (Circuit Breaker / Halt Simulation)
        if volume <= 0:
            logger.warning(f"LIQUIDITY_LOCK: {ticker} on {date} (Zero Volume)")
            return self._create_empty_trade(ticker, date, failure_reason="ZERO_VOLUME")

        # Integer constraint: Prevents fractional "dust" from accumulating
        shares = int(shares)

        # 1. Probabilistic Execution (Stochastic Fill Failure)
        if np.random.random() > self.fill_prob:
            logger.warning(f"FILL_FAILURE: {ticker} on {date} (Liquidity/Probability)")
            return self._create_empty_trade(ticker, date, failure_reason="PROBABILITY")

        if shares <= 0 or price <= 0:
            return self._create_empty_trade(ticker, date, failure_reason="INVALID_INPUTS")

        notional = shares * price
        
        # 2. Commission Calculation (Explicit Cost)
        if self.commission_per_share is not None:
            commission_usd = shares * self.commission_per_share
        else:
            commission_usd = self.commission_rate * notional
        
        # 3. Implicit Costs (Spread + Slippage + Impact)
        # Stochastic Slippage: Multiplier follows LogNormal distribution.
        # High volatility -> Fatter tails in execution cost.
        slippage_variance = np.random.lognormal(0, volatility)
        
        total_friction_rate = self.spread_bps + (self.slippage_bps * slippage_variance) + impact_rate
        friction_usd = total_friction_rate * notional
        
        # 4. Fill Price Adjustment
        # $P_{fill} = P_{mid} \pm \frac{Cost_{implicit}}{Shares}$
        # CRITICAL: Commission is excluded here to avoid double-counting in Portfolio basis.
        cost_per_share = friction_usd / shares
        if side == 'buy':
            fill_price = price + cost_per_share
        else:
            fill_price = price - cost_per_share
            
        # Boundary Condition: Prevent negative execution prices
        fill_price = max(0.01, fill_price)

        # 5. Settlement Record (Preserve high-precision floats for accounting)
        return {
            'ticker': ticker,
            'shares': int(shares),
            'side': side,
            'market_price': price,
            'fill_price': fill_price,
            'commission_usd': commission_usd, # Explicit expense
            'friction_usd': friction_usd,
            'impact_rate': impact_rate,
            'total_cost_usd': commission_usd + friction_usd,
            'cost_bps': ((commission_usd + friction_usd) / notional) * 10000,
            'date': date,
            'success': True,
            'failure_reason': None
        }

    def execute_batch(self, orders: List[Dict], prices: pd.DataFrame, date: str) -> List[Dict]:
        """
        Process a list of orders against market data for a specific date.
        
        Optimization:
        Uses dictionary hashing ($O(1)$) for price lookups instead of DataFrame filtering ($O(N)$).
        """
        executed_trades = []
        
        # Optimization: Pre-compute hash map for O(1) access
        if 'ticker' in prices.columns:
            price_map = prices.set_index('ticker').to_dict('index')
        else:
            price_map = prices.to_dict('index')

        for order in orders:
            ticker = order['ticker']
            if ticker not in price_map:
                logger.debug(f"Missing price for {ticker} on {date}")
                continue
                
            mkt = price_map[ticker]
            trade = self.execute_order(
                ticker=ticker,
                shares=order['shares'],
                side=order['side'],
                price=mkt['close'],
                volume=mkt['volume'],
                date=date,
                impact_rate=order.get('impact_rate', 0.0),
                volatility=mkt.get('volatility', 0.02)
            )
            
            if trade['success']:
                executed_trades.append(trade)
                
        return executed_trades

    def _create_empty_trade(self, ticker, date, failure_reason) -> Dict:
        """Generates a null-object trade record for failed executions."""
        return {
            'ticker': ticker, 'shares': 0, 'side': 'none', 
            'market_price': 0.0, 'fill_price': 0.0,
            'commission_usd': 0.0, 'friction_usd': 0.0, 'impact_rate': 0.0,
            'total_cost_usd': 0.0, 'cost_bps': 0.0,
            'date': date, 'success': False, 
            'failure_reason': failure_reason
        }