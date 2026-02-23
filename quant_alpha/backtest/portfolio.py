import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """
    Enhanced Portfolio Manager (Fixed)
    Tracks state, costs, and daily performance with high precision.
    """
    
    def __init__(self, initial_capital: float, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        self.positions: Dict[str, float] = {}
        self.position_costs: Dict[str, float] = {}
        self.current_prices: Dict[str, float] = {}
        
        self.transaction_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.realized_pnl = 0.0
        self.total_commissions = 0.0
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.0f}")

    # ==================== PROPERTIES ====================
    
    @property
    def positions_value(self) -> float:
        total = 0.0
        for ticker, shares in self.positions.items():
            price = self.current_prices.get(ticker, 0.0)
            
            # --- FIX 1: Crash Prevention ---
            if price <= 0:
                # Fallback to cost basis instead of crashing
                price = self.position_costs.get(ticker, 0.0)
                if price > 0:
                    logger.warning(f"⚠️ Missing price for {ticker}. Using cost basis: {price}")
                else:
                    logger.error(f"❌ Critical: No price or cost for {ticker}. Valued at 0.")
            
            total += shares * price
        return total
    
    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value

    @property
    def cash_pct(self) -> float:
        return self.cash / self.total_value if self.total_value > 0 else 0

    # ==================== TRADING OPERATIONS ====================
    
    def buy(self, ticker: str, shares: float, price: float, commission: Optional[float] = None) -> bool:
        if shares <= 0 or price <= 0: return False

        if commission is not None:
            # External execution: price is already fill_price, commission is explicit
            exec_price = price
            raw_cost = shares * exec_price
        else:
            # Internal execution
            exec_price = price * (1 + self.slippage_pct)
            raw_cost = shares * exec_price
            commission = raw_cost * self.commission_pct
            
        total_cost = raw_cost + commission
        
        # --- FIX 2: Floating Point Precision ---
        # Allow tiny margin of error (epsilon) for full cash usage
        if total_cost > self.cash + 1e-9:
            logger.warning(f"Insufficient cash for {ticker}: Need {total_cost:.2f}, Have {self.cash:.2f}")
            return False
        
        self.cash -= total_cost
        self.total_commissions += commission
        self.current_prices[ticker] = price 
        
        # Average cost basis
        old_shares = self.positions.get(ticker, 0.0)
        old_cost = self.position_costs.get(ticker, 0.0)
        new_shares = old_shares + shares
        
        # Weighted Average Cost
        self.position_costs[ticker] = ((old_shares * old_cost) + total_cost) / new_shares
        self.positions[ticker] = new_shares
        
        self._record_tx('buy', ticker, shares, exec_price, commission, pnl=0.0)
        return True
    
    def sell(self, ticker: str, shares: float, price: float, commission: Optional[float] = None) -> bool:
        # --- FIX 2: Floating Point Precision ---
        current_shares = self.positions.get(ticker, 0.0)
        if ticker not in self.positions or shares > current_shares + 1e-9:
            return False
        
        # Cap shares to held amount (handle rounding errors)
        shares = min(shares, current_shares)

        if commission is not None:
            # External execution
            exec_price = price
            proceeds = shares * exec_price
        else:
            # Internal execution
            exec_price = price * (1 - self.slippage_pct)
            proceeds = shares * exec_price
            commission = proceeds * self.commission_pct
            
        net_proceeds = proceeds - commission
        
        # Realize P&L
        cost_basis = self.position_costs[ticker]
        trade_pnl = net_proceeds - (shares * cost_basis)
        self.realized_pnl += trade_pnl
        
        self.cash += net_proceeds
        self.total_commissions += commission
        self.positions[ticker] -= shares
        self.current_prices[ticker] = price

        # Cleanup
        if self.positions[ticker] < 1e-9:
            del self.positions[ticker]
            del self.position_costs[ticker]
            # Optional: Remove from current_prices to keep state clean
            # if ticker in self.current_prices: del self.current_prices[ticker]
        
        self._record_tx('sell', ticker, shares, exec_price, commission, pnl=trade_pnl)
        return True

    def record_daily_snapshot(self, date: pd.Timestamp):
        self.equity_curve.append({
            'date': date,
            'nav': self.total_value,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl
        })

    def _record_tx(self, type, ticker, shares, price, comm, pnl=0.0):
        self.transaction_history.append({
            'type': type, 'ticker': ticker, 'shares': shares, 
            'price': price, 'commission': comm, 'pnl': pnl, 'nav': self.total_value
        })

    def update_prices(self, prices: Union[pd.DataFrame, Dict[str, float]]):
        """
        Accepts DataFrame with [ticker, close] OR Dictionary {ticker: close}
        Optimized for speed using dictionary mapping.
        """
        if isinstance(prices, dict):
            self.current_prices.update(prices)
            return

        if isinstance(prices, pd.DataFrame):
            if prices.empty: return
            
            # Filter only relevant tickers to save memory
            relevant_prices = prices[prices['ticker'].isin(self.positions.keys())]
            
            # Bulk update
            new_prices = dict(zip(relevant_prices['ticker'], relevant_prices['close']))
            self.current_prices.update(new_prices)
