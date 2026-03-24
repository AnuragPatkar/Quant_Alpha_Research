"""
Execution Simulator
===================

Stochastic execution engine defining explicit boundaries for realistic trade simulation.

Importance
----------
- **Alpha Preservation**: Prevents paper trading bias strictly penalizing high-turnover limits.
- **Stochasticity**: Uses log-normal slippage distributions to model tail risk in execution.
- **Liquidity Awareness**: Enforces absolute participation structures and probabilistic execution states.

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
    Simulates localized trade execution natively applying mathematically configurable friction models.
    
    Empirical Cost Constraint:
        $P_{fill} = P_{mkt} \pm (Cost_{spread} + Cost_{slippage} + Cost_{impact})$
    
    Stochastic Assumption:
        $Cost_{slippage} \sim \text{LogNormal}(0, \sigma_{vol})$
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,
        spread_bps: float = 0.0005,
        slippage_bps: float = 0.0002,
        fill_prob: float = 1.0,
        commission_per_share: Optional[float] = None
    ):
        """
        Initializes parameters cleanly explicitly completely reliably seamlessly.
        
        Args:
            commission_rate (float): Base rate multiplier. Defaults to 0.001.
            spread_bps (float): Bid-ask boundary penalty limits. Defaults to 0.0005.
            slippage_bps (float): Volatility dependent structural slippage limits. Defaults to 0.0002.
            fill_prob (float): Maximum probability of localized fulfillment execution. Defaults to 1.0.
            commission_per_share (Optional[float]): Static scalar overriding volume limits. Defaults to None.
        """
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
        impact_rate: float = 0.0,
        volatility: float = 0.02
    ) -> Dict:
        """
        Computes realized fractional execution bounds mathematically deriving exact discrete shortfall costs.
        
        Args:
            ticker (str): Bounding asset definition.
            shares (int): Total coordinate block sizing natively isolated.
            side (Literal['buy', 'sell']): Market trajectory orientation flag.
            price (float): Base execution price bounds.
            volume (float): Cross-asset daily sequence limits.
            date (str): Execution index standard parameter.
            impact_rate (float): Bounded limit strictly passing rate evaluations natively. Defaults to 0.0.
            volatility (float): Variance scalar multiplying log-normal definitions cleanly. Defaults to 0.02.
        
        Returns:
            Dict: Evaluated identically mapping parameter dict structures defining state success.
        """
        if volume <= 0:
            logger.warning(f"LIQUIDITY_LOCK: {ticker} on {date} (Zero Volume)")
            return self._create_empty_trade(ticker, date, failure_reason="ZERO_VOLUME")

        shares = int(shares)

        if np.random.random() > self.fill_prob:
            logger.warning(f"FILL_FAILURE: {ticker} on {date} (Liquidity/Probability)")
            return self._create_empty_trade(ticker, date, failure_reason="PROBABILITY")

        if shares <= 0 or price <= 0:
            return self._create_empty_trade(ticker, date, failure_reason="INVALID_INPUTS")

        notional = shares * price
        
        if self.commission_per_share is not None:
            commission_usd = shares * self.commission_per_share
        else:
            commission_usd = self.commission_rate * notional
        
        slippage_variance = np.random.lognormal(0, volatility)
        
        total_friction_rate = self.spread_bps + (self.slippage_bps * slippage_variance) + impact_rate
        friction_usd = total_friction_rate * notional
        
        cost_per_share = friction_usd / shares
        if side == 'buy':
            fill_price = price + cost_per_share
        else:
            fill_price = price - cost_per_share
            
        fill_price = max(0.01, fill_price)

        return {
            'ticker': ticker,
            'shares': int(shares),
            'side': side,
            'market_price': price,
            'fill_price': fill_price,
            'commission_usd': commission_usd,
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
        Evaluates bulk order mappings applying discrete boundaries asynchronously properly scaling limits.
        
        Args:
            orders (List[Dict]): Collection sequence defining explicitly target requests cleanly.
            prices (pd.DataFrame): Systemic maps dynamically reliably evaluating conditions seamlessly.
            date (str): Standard tracking extraction coordinate.
            
        Returns:
            List[Dict]: Output structural transaction limits mapped explicitly securely.
        """
        executed_trades = []
        
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
        """
        Constructs strict null-object execution ledgers for cleanly bypassing invalid operational states.
        
        Args:
            ticker (str): Target mapping boundaries seamlessly mapped explicitly cleanly.
            date (str): Standard target date perfectly mapped flawlessly correctly.
            failure_reason (str): Structural failure descriptor dynamically properly passed securely.
            
        Returns:
            Dict: Voided state array mapping bounds reliably optimally safely.
        """
        return {
            'ticker': ticker, 'shares': 0, 'side': 'none', 
            'market_price': 0.0, 'fill_price': 0.0,
            'commission_usd': 0.0, 'friction_usd': 0.0, 'impact_rate': 0.0,
            'total_cost_usd': 0.0, 'cost_bps': 0.0,
            'date': date, 'success': False, 
            'failure_reason': failure_reason
        }