"""
Portfolio State Management Engine
=================================
Core ledger system for event-driven backtesting and live trading simulation.

Purpose
-------
The `Portfolio` module serves as the central accounting engine for the backtest.
It enforces double-entry bookkeeping principles to track cash balances,
inventory (share holdings), and cost basis across time. It is responsible for
mark-to-market valuations, transaction lifecycle management (execution,
settlement, commission deduction), and realized/unrealized P&L calculation.

Usage
-----
This class is typically instantiated by an `Engine` or `Strategy` class.

.. code-block:: python

    from quant_alpha.backtest.portfolio import Portfolio

    # 1. Initialize with capital and cost model
    port = Portfolio(initial_capital=1e6, commission_pct=0.001)

    # 2. Process an execution (Buy 100 shares of AAPL @ $150)
    port.buy("AAPL", 100, 150.0)

    # 3. Mark-to-Market update
    port.update_prices({"AAPL": 155.0})

    # 4. Snapshot state
    port.record_daily_snapshot(pd.Timestamp("2023-10-27"))

Importance
----------
- **Accounting Integrity**: Distinguishes between Cost Basis (tax lots) and
  Expense (commissions) to ensure accurate Net vs. Gross return calculations.
- **Path Dependency**: Accurately simulates the compounding of returns and the
  drag of transaction costs over the simulation horizon.
- **Numerical Stability**: Handles floating-point epsilon errors ($\epsilon = 10^{-9}$)
  preventing non-deterministic behavior in zero-balance checks.

Tools & Frameworks
------------------
- **Pandas**: Time-series structuring for equity curves and transaction ledgers.
- **NumPy**: Efficient numerical handling of price updates and `NaN` checks.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """
    Manages the state of a trading portfolio through time.

    Acts as the Source of Truth (SoT) for the simulation. Maintains a precise record 
    of cash, asset positions, and Weighted Average Cost Basis (WACB). It processes
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
        """
        Calculates the Gross Market Value (GMV) of all held positions.
        
        Includes fallback logic for missing price data to prevent simulation halts.
        """
        total = 0.0
        for ticker, shares in self.positions.items():
            price = self.current_prices.get(ticker, 0.0)
            
            # Data Integrity: Fallback mechanism for missing pricing data.
            if price <= 0:
                # Fallback: Use Cost Basis if Mark-to-Market price is unavailable.
                price = self.position_costs.get(ticker, 0.0)
                if price > 0:
                    logger.warning(f"⚠️ Missing price for {ticker}. Using cost basis: {price}")
                else:
                    logger.error(f"❌ Critical: No price or cost for {ticker}. Valued at 0.")
            
            total += shares * price
        return total
    
    @property
    def total_value(self) -> float:
        """Calculates the Net Asset Value (NAV) = Cash + Market Value of Positions."""
        return self.cash + self.positions_value

    @property
    def cash_pct(self) -> float:
        """Liquidity metric: Cash / NAV."""
        return self.cash / self.total_value if self.total_value > 0 else 0

    def get_holdings(self) -> Dict[str, float]:
        """Returns a copy of the holdings ledger to prevent external mutation."""
        return self.positions.copy()

    def get_position_value(self, ticker: str) -> float:
        """Returns current market value (Mark-to-Market) of a specific position."""
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
            ticker: Asset symbol.
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
            # Execution Mode: External (Explicit Commission provided)
            exec_price = price
            raw_cost = shares * exec_price
        else:
            # Execution Mode: Internal Simulation (Implicit Slippage + Commission)
            exec_price = price * (1 + self.slippage_pct)
            raw_cost = shares * exec_price
            commission = raw_cost * self.commission_pct
            
        total_cost = raw_cost + commission
        
        # Liquidity Check: Ensure sufficient purchasing power.
        # Utilizes epsilon (1e-9) to mitigate floating-point rounding errors.
        if total_cost > self.cash + 1e-9:
            logger.warning(f"Insufficient cash for {ticker}: Need {total_cost:.2f}, Have {self.cash:.2f}")
            return None
        
        self.cash -= total_cost
        self.total_commissions += commission
        self.current_prices[ticker] = price 
        
        # Weighted Average Cost Basis (WACB) Calculation:
        # .. math::
        #     \bar{C}_{new} = \frac{(S_{held} \times \bar{C}_{old}) + (S_{buy} \times P_{exec})}{S_{held} + S_{buy}}
        old_shares = self.positions.get(ticker, 0.0)
        old_cost = self.position_costs.get(ticker, 0.0)
        new_shares = old_shares + shares
        
        # Accounting Standard: Commission is treated as a period expense, NOT capitalized into the asset's cost basis.
        self.position_costs[ticker] = ((old_shares * old_cost) + raw_cost) / new_shares
        self.positions[ticker] = new_shares
        
        self._record_tx('buy', ticker, shares, exec_price, commission, pnl=0.0)
        return 0.0
    
    def sell(self, ticker: str, shares: float, price: float, commission: Optional[float] = None) -> Optional[float]:
        """
        Executes a sell order, updating cash, positions, and realizing P&L.

        Args:
            ticker: Asset symbol.
            shares: The number of shares to sell.
            price: The execution price per share.
            commission: The explicit commission cost. If None, it's calculated internally.
        Returns:
            The realized P&L from the trade, or None on failure.
        """
        if np.isnan(price) or np.isinf(price):
            logger.error(f"Invalid price received for {ticker}: {price}")
            return None

        # Validation: Verify holding sufficiency with epsilon tolerance.
        current_shares = self.positions.get(ticker, 0.0)
        if ticker not in self.positions or shares > current_shares + 1e-9:
            return None
        
        # Normalization: Cap shares to held amount to prevent negative inventory.
        shares = min(shares, current_shares)

        if commission is not None:
            # Execution Mode: External
            exec_price = price
            proceeds = shares * exec_price
        else:
            # Execution Mode: Internal Simulation
            exec_price = price * (1 - self.slippage_pct)
            proceeds = shares * exec_price
            commission = proceeds * self.commission_pct
            
        net_proceeds = proceeds - commission
        
        # P&L Realization: Calculate profit relative to Average Cost Basis.
        # .. math::
        #     \text{PnL}_{realized} = \text{Proceeds}_{net} - (S_{sold} \times \bar{C}_{avg})
        cost_basis = self.position_costs[ticker]
        trade_pnl = net_proceeds - (shares * cost_basis)
        self.realized_pnl += trade_pnl
        
        self.cash += net_proceeds
        self.total_commissions += commission
        self.positions[ticker] -= shares
        self.current_prices[ticker] = price

        # Garbage Collection: Remove closed positions to maintain constant memory complexity O(1).
        if self.positions[ticker] < 1e-9:
            del self.positions[ticker]
            del self.position_costs[ticker]
        
        self._record_tx('sell', ticker, shares, exec_price, commission, pnl=trade_pnl)
        return trade_pnl

    def apply_dividends(self, dividend_map: Dict[str, float]):
        """
        Corporate Action: Credits cash balance for dividend payments.
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
        """Appends the current NAV, Cash, and PnL to the time-series equity curve."""
        self.equity_curve.append({
            'date': date,
            'nav': self.total_value,
            'total_value': self.total_value,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl
        })

    def _record_tx(self, type, ticker, shares, price, comm, pnl=0.0):
        """Audit Log: Appends a trade record to the internal transaction ledger."""
        self.transaction_history.append({
            'type': type, 'ticker': ticker, 'shares': shares, 
            'price': price, 'commission': comm, 'pnl': pnl, 'total_value': self.total_value
        })

    def update_prices(self, prices: Union[pd.DataFrame, Dict[str, float]]):
        """
        Performs Mark-to-Market valuation update.
        
        Args:
            prices: Dictionary `{ticker: price}` or DataFrame with columns `['ticker', 'close']`.
        
        Performance:
            Optimized for O(1) bulk updates via dictionary mapping.
        """
        if isinstance(prices, dict):
            self.current_prices.update(prices)
            return

        if isinstance(prices, pd.DataFrame):
            if prices.empty: return
            
            # Optimization: Filter only held tickers to reduce memory overhead.
            relevant_prices = prices[prices['ticker'].isin(self.positions.keys())]
            
            # Vectorized Dictionary Construction
            new_prices = dict(zip(relevant_prices['ticker'], relevant_prices['close']))
            self.current_prices.update(new_prices)

    def get_tx_history_df(self) -> pd.DataFrame:
        """Exports the transaction ledger as a standardized DataFrame."""
        return pd.DataFrame(self.transaction_history)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Exports the NAV time-series with DatetimeIndex for performance analysis."""
        df = pd.DataFrame(self.equity_curve)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
