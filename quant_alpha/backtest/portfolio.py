"""
Portfolio State Manager
=======================
Core component for managing portfolio state within an event-driven backtest.

Purpose
-------
This module provides the `Portfolio` class, which serves as the single source of
truth for the backtesting engine. It meticulously tracks asset holdings, cash
balances, average cost basis, and performance metrics on a point-in-time basis.
Its primary function is to simulate the financial impact of trading decisions
while adhering to standard accounting principles.

Usage
-----
.. code-block:: python

    from quant_alpha.backtest.portfolio import Portfolio

    # Initialize with starting capital
    portfolio = Portfolio(initial_capital=1_000_000)

    # Execute a trade (e.g., from an ExecutionSimulator)
    portfolio.buy(ticker="AAPL", shares=100, price=150.0, commission=1.0)

    # Update market prices for mark-to-market valuation
    portfolio.update_prices({"AAPL": 152.0})

    # Record daily state for performance analysis
    portfolio.record_daily_snapshot(date=pd.Timestamp("2023-10-26"))

    print(f"Current Portfolio Value: {portfolio.total_value:.2f}")

Importance
----------
- **Path Dependence**: Accurately models the compounding effects of returns and
  costs, ensuring the simulation's path-dependent integrity.
- **Accounting Accuracy**: Separates cost basis from transaction expenses (commissions),
  providing a clean ledger for P&L and tax-lot accounting.
- **State Integrity**: Acts as the central, uncorrupted state manager, preventing
  look-ahead bias and ensuring that all calculations are based on point-in-time data.
- **Performance**: Optimized for high-frequency updates, with dictionary lookups
  providing $O(1)$ complexity for core operations like price updates and position queries.

Tools & Frameworks
------------------
- **Pandas**: Used for structuring historical data outputs (equity curve, transactions).
- **NumPy**: Leveraged for high-performance numerical operations and handling of
  potential `NaN`/`inf` values in price data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """
    Manages the state of a trading portfolio through time.

    This class is the core state-tracking object in the backtesting simulation.
    It maintains a precise record of cash, asset positions (shares), and the
    weighted-average cost basis for each holding. It processes buy/sell
    transactions, accounts for costs like commissions and slippage, and generates
    the daily equity curve required for performance analytics.

    Attributes:
        initial_capital (float): The starting capital of the portfolio.
        cash (float): The current cash balance, adjusted for trades and costs.
        positions (Dict[str, float]): A dictionary mapping tickers to the number of shares held.
        position_costs (Dict[str, float]): Maps tickers to their weighted-average entry price.
        current_prices (Dict[str, float]): The most recent market prices for mark-to-market valuation.
        equity_curve (List[Dict]): A time-series log of the portfolio's total value.
        realized_pnl (float): The cumulative profit and loss from closed trades.
        total_commissions (float): The cumulative sum of all commission expenses.
    """
    
    def __init__(self, initial_capital: float, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        """Initializes the portfolio with a starting cash balance and cost parameters."""
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
        """Calculates the total market value of all held positions."""
        total = 0.0
        for ticker, shares in self.positions.items():
            price = self.current_prices.get(ticker, 0.0)
            
            # --- Graceful Handling of Missing Prices ---
            if price <= 0:
                # If a live price is missing (e.g., stale data), fall back to the
                # average cost basis to value the position, preventing a crash.
                price = self.position_costs.get(ticker, 0.0)
                if price > 0:
                    logger.warning(f"⚠️ Missing price for {ticker}. Using cost basis: {price}")
                else:
                    logger.error(f"❌ Critical: No price or cost for {ticker}. Valued at 0.")
            
            total += shares * price
        return total
    
    @property
    def total_value(self) -> float:
        """Calculates the Net Asset Value (NAV) of the portfolio (Cash + Positions)."""
        return self.cash + self.positions_value

    @property
    def cash_pct(self) -> float:
        """Returns the cash balance as a percentage of the total portfolio value."""
        return self.cash / self.total_value if self.total_value > 0 else 0

    def get_holdings(self) -> Dict[str, float]:
        """Returns a copy of the current positions dictionary (ticker: shares)."""
        return self.positions.copy()

    def get_position_value(self, ticker: str) -> float:
        """Returns current market value of a specific position."""
        shares = self.positions.get(ticker, 0.0)
        if shares == 0: return 0.0
        
        price = self.current_prices.get(ticker, 0.0)
        if price <= 0:
            price = self.position_costs.get(ticker, 0.0)
        return shares * price

    # ==================== TRADING OPERATIONS ====================
    
    def buy(self, ticker: str, shares: float, price: float, commission: Optional[float] = None) -> Optional[float]:
        """
        Executes a buy order, updating cash, positions, and cost basis.

        Args:
            ticker: The stock identifier.
            shares: The number of shares to buy.
            price: The execution price per share.
            commission: The explicit commission cost. If None, it's calculated internally.
        Returns:
            0.0 on success, None on failure (e.g., insufficient cash).
        """
        if np.isnan(price) or np.isinf(price):
            logger.error(f"Invalid price received for {ticker}: {price}")
            return None

        if shares <= 0 or price <= 0: return None

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
        
        # --- Cash Constraint with Floating-Point Tolerance ---
        # Use an epsilon (1e-9) to prevent failures from floating-point inaccuracies
        # when an order intends to use the full available cash balance.
        if total_cost > self.cash + 1e-9:
            logger.warning(f"Insufficient cash for {ticker}: Need {total_cost:.2f}, Have {self.cash:.2f}")
            return None
        
        self.cash -= total_cost
        self.total_commissions += commission
        self.current_prices[ticker] = price 
        
        # --- Update Weighted-Average Cost Basis ---
        # Formula: $C_{new} = \frac{(S_{old} \times C_{old}) + (S_{trade} \times P_{trade})}{S_{old} + S_{trade}}$
        # where C is cost basis, S is shares, and P is price.
        old_shares = self.positions.get(ticker, 0.0)
        old_cost = self.position_costs.get(ticker, 0.0)
        new_shares = old_shares + shares
        
        # Per accounting standards, commission is an expense, not part of the asset's cost basis.
        self.position_costs[ticker] = ((old_shares * old_cost) + raw_cost) / new_shares
        self.positions[ticker] = new_shares
        
        self._record_tx('buy', ticker, shares, exec_price, commission, pnl=0.0)
        return 0.0
    
    def sell(self, ticker: str, shares: float, price: float, commission: Optional[float] = None) -> Optional[float]:
        """
        Executes a sell order, updating cash, positions, and realizing P&L.

        Args:
            ticker: The stock identifier.
            shares: The number of shares to sell.
            price: The execution price per share.
            commission: The explicit commission cost. If None, it's calculated internally.
        Returns:
            The realized P&L from the trade, or None on failure.
        """
        if np.isnan(price) or np.isinf(price):
            logger.error(f"Invalid price received for {ticker}: {price}")
            return None

        # --- Share Quantity with Floating-Point Tolerance ---
        current_shares = self.positions.get(ticker, 0.0)
        if ticker not in self.positions or shares > current_shares + 1e-9:
            return None
        
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

        # --- Position Cleanup ---
        if self.positions[ticker] < 1e-9:
            # If all shares are sold, remove the ticker to prevent memory bloat.
            del self.positions[ticker]
            del self.position_costs[ticker]
            # Optional: Remove from current_prices to keep state clean
            # if ticker in self.current_prices: del self.current_prices[ticker]
        
        self._record_tx('sell', ticker, shares, exec_price, commission, pnl=trade_pnl)
        return trade_pnl

    def apply_dividends(self, dividend_map: Dict[str, float]):
        """
        Applies dividends to cash balance without affecting shares or cost basis.
        dividend_map: {ticker: dividend_per_share}
        """
        total_div = 0.0
        for ticker, div_per_share in dividend_map.items():
            if ticker in self.positions:
                shares = self.positions[ticker]
                amount = shares * div_per_share
                if amount > 0:
                    self.cash += amount
                    total_div += amount
                    self._record_tx('dividend', ticker, shares, div_per_share, 0.0, pnl=amount)
        
        if total_div > 0:
            logger.info(f"💰 Dividends received: ${total_div:,.2f}")

    def record_daily_snapshot(self, date: pd.Timestamp):
        """Records the portfolio's total value and cash for a given date."""
        self.equity_curve.append({
            'date': date,
            'nav': self.total_value,
            'total_value': self.total_value,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl
        })

    def _record_tx(self, type, ticker, shares, price, comm, pnl=0.0):
        """Internal helper to log a transaction to the history."""
        self.transaction_history.append({
            'type': type, 'ticker': ticker, 'shares': shares, 
            'price': price, 'commission': comm, 'pnl': pnl, 'total_value': self.total_value
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

    def get_tx_history_df(self) -> pd.DataFrame:
        """Returns the complete transaction history as a DataFrame."""
        return pd.DataFrame(self.transaction_history)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Returns the daily equity curve as a DataFrame with a DatetimeIndex."""
        df = pd.DataFrame(self.equity_curve)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
