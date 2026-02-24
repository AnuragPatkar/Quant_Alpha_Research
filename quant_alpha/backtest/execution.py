"""
Execution Simulator (Institutional Grade)
Handles realistic trade execution, transaction costs, and market impact.
"""

import pandas as pd
import numpy as np
from typing import Dict, Literal, List, Optional
import logging

logger = logging.getLogger(__name__)

class ExecutionSimulator:
    def __init__(
        self,
        commission_rate: float = 0.001,    # 10 bps
        spread_bps: float = 0.0005,       # 5 bps (half-spread)
        slippage_bps: float = 0.0002,     # 2 bps
        fill_prob: float = 1.0,           # 100% fill rate
        commission_per_share: Optional[float] = None # If set, overrides rate
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
        Calculates execution details. Commission is kept separate to avoid 
        double-counting in the Portfolio class.
        """
        # 0. Liquidity Check (Circuit Breaker / Halted)
        if volume <= 0:
            logger.warning(f"LIQUIDITY_LOCK: {ticker} on {date} (Zero Volume)")
            return self._create_empty_trade(ticker, date, failure_reason="ZERO_VOLUME")

        # Ensure integer shares to prevent "Dust" disconnects (costs on 0.9 shares, but 0 returned)
        shares = int(shares)

        # 1. Fill Probability (No silent failures - log explicitly)
        if np.random.random() > self.fill_prob:
            logger.warning(f"FILL_FAILURE: {ticker} on {date} (Liquidity/Probability)")
            return self._create_empty_trade(ticker, date, failure_reason="PROBABILITY")

        if shares <= 0 or price <= 0:
            return self._create_empty_trade(ticker, date, failure_reason="INVALID_INPUTS")

        notional = shares * price
        
        # 2. Commission Calculation (Separate from Fill Price)
        if self.commission_per_share is not None:
            commission_usd = shares * self.commission_per_share
        else:
            commission_usd = self.commission_rate * notional
        
        # 3. Market Friction Costs (Spread + Slippage + Impact)
        # Scalable Slippage: Variance scales with volatility
        # Fix: Use LogNormal to ensure positive distribution and realistic tails
        slippage_variance = np.random.lognormal(0, volatility)
        
        total_friction_rate = self.spread_bps + (self.slippage_bps * slippage_variance) + impact_rate
        friction_usd = total_friction_rate * notional
        
        # 4. Fill Price (Market + Friction)
        # NOTE: Commission is NOT included in fill_price to avoid double-counting in Portfolio
        cost_per_share = friction_usd / shares
        if side == 'buy':
            fill_price = price + cost_per_share
        else:
            fill_price = price - cost_per_share
            
        # Protection against negative prices in extreme scenarios
        fill_price = max(0.01, fill_price)

        # 5. Result Construction (NO ROUNDING HERE - Keep Raw Floats)
        return {
            'ticker': ticker,
            'shares': int(shares),
            'side': side,
            'market_price': price,
            'fill_price': fill_price,
            'commission_usd': commission_usd, # Portfolio will use this
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
        Optimized batch execution using O(1) dictionary lookups.
        """
        executed_trades = []
        
        # Fix: O(1) lookup instead of O(N) scanning inside the loop
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
        return {
            'ticker': ticker, 'shares': 0, 'side': 'none', 
            'market_price': 0.0, 'fill_price': 0.0,
            'commission_usd': 0.0, 'friction_usd': 0.0, 'impact_rate': 0.0,
            'total_cost_usd': 0.0, 'cost_bps': 0.0,
            'date': date, 'success': False, 
            'failure_reason': failure_reason
        }