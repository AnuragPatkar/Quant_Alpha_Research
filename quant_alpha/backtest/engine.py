"""
Backtest Engine
===============
Event-driven simulation core for quantitative strategy evaluation.

Purpose
-------
Orchestrates the entire backtesting lifecycle, simulating the interaction between
alpha signals, portfolio construction, risk management, and market execution.
It implements an event-driven loop that processes market data sequentially to
prevent look-ahead bias and accurately model point-in-time constraints.

Usage
-----
.. code-block:: python

    engine = BacktestEngine(initial_capital=1_000_000, target_volatility=0.20)
    results = engine.run(predictions_df, prices_df, top_n=50)

Importance
----------
- **Path Dependence**: Captures the compounding effects of costs, impact, and constraints over time.
- **Realistic Friction**: Integrates Almgren-Chriss market impact ($I \propto \sigma \sqrt{Q/V}$) and variable transaction costs.
- **Regime Awareness**: Dynamically adjusts exposure based on volatility and drawdown regimes.

Tools & Frameworks
------------------
- **Pandas/NumPy**: High-performance data structures for time-series alignment and vectorized weighting.
- **AlmgrenChrissImpact**: Nonlinear transaction cost modeling.
- **RiskManager**: Portfolio constraint enforcement (Sector, ADV, Leverage).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from .portfolio import Portfolio
from .execution import ExecutionSimulator
from .market_impact import AlmgrenChrissImpact
from .risk_manager import RiskManager
from .metrics import PerformanceMetrics
from .utils import validate_backtest_data

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Event-driven backtesting engine for equity strategies.
    
    Simulates the full trading lifecycle including:
    1. Signal Processing & Ranking
    2. Dynamic Portfolio Construction (Inverse Volatility Weighting)
    3. Risk Management & Constraint Enforcement
    4. Execution Simulation with Market Impact
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        spread: float = 0.0005,
        slippage: float = 0.0002,
        position_limit: float = 0.05,
        leverage_limit: float = 1.0,
        rebalance_freq: str = 'weekly',
        use_market_impact: bool = True,
        target_volatility: float = 0.20,
        max_adv_participation: float = 0.02, # NEW: Default 2% limit
        trailing_stop_pct: float = None,     # NEW: Trailing Stop
        execution_price: str = 'close',      # 'close' or 'open'
        max_turnover: float = 0.20           # Max turnover per rebalance (20%)
    ):
        self.portfolio = Portfolio(initial_capital)
        self.executor = ExecutionSimulator(commission, spread, slippage)
        self.impact_model = AlmgrenChrissImpact() if use_market_impact else None
        self.risk_manager = RiskManager(position_limit, leverage_limit, target_volatility=target_volatility, max_adv_participation=max_adv_participation)
        self.metrics_engine = PerformanceMetrics()
        
        self.trailing_stop_pct = trailing_stop_pct
        self.high_water_marks = {}   # Track peak price for trailing stop
        self.position_entry_map = {} # Track entry for detailed reporting
        
        self.execution_price = execution_price
        self.max_turnover = max_turnover
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.trades = []

    def run(self, predictions: pd.DataFrame, prices: pd.DataFrame, top_n: int = 50):
        """
        Executes the backtest simulation over the provided prediction horizon.
        
        Algorithm:
        For each date $t$ in horizon:
            1. Update Portfolio Mark-to-Market.
            2. Check Trailing Stops (Intraday Risk).
            3. If Rebalance Day:
                a. Generate Target Weights ($w_{target}$).
                b. Apply Risk Constraints ($w_{safe} = f(w_{target}, \Omega)$).
                c. Execute Rebalance (Sell first, then Buy).
            4. Record Daily Snapshot.
        """
        # 0. Data Integrity Verification
        is_valid, errors = validate_backtest_data(predictions, prices)
        if not is_valid:
            raise ValueError(f"Data Validation Failed: {errors}")

        # 1. Data Structure Optimization
        # Ensure MultiIndex is pre-sorted to enable O(1) slicing performance
        predictions = predictions.sort_values(['date', 'prediction'], ascending=[True, False])
        dates = sorted(predictions['date'].unique())
        prices_indexed = prices.set_index(['date', 'ticker']).sort_index()

        for i, date in enumerate(dates):
            # 2. Efficient Data Access (Point-in-Time)
            try:
                day_prices = prices_indexed.loc[date]
            except KeyError:
                logger.warning(f"Market data missing for date: {date}")
                continue
            
            # 3. Mark-to-Market (Sync Portfolio NAV with current prices)
            current_holdings = list(self.portfolio.positions.keys())
            if current_holdings:
                valid_tickers = [t for t in current_holdings if t in day_prices.index]
                if valid_tickers:
                    price_map = day_prices.loc[valid_tickers, 'close'].to_dict()
                    self.portfolio.update_prices(price_map)
            
            # 3.5 Intraday Risk Control (Trailing Stops)
            if self.trailing_stop_pct:
                self._check_trailing_stops(date, day_prices)

            # 4. Portfolio Rebalancing Logic
            if self._is_rebalance_day(date, i):
                # Expand prediction set to support Buffer/Hysteresis logic (e.g. Top 1.6*N)
                day_preds = predictions[predictions['date'] == date].head(top_n * 5)
                
                # Rolling Volatility Estimation (21-day lookback)
                current_vol = 0.0
                if len(self.portfolio.equity_curve) > 22:
                    navs = [x['total_value'] for x in self.portfolio.equity_curve[-22:]]
                    pct_changes = pd.Series(navs).pct_change().dropna()
                    if not pct_changes.empty:
                        current_vol = pct_changes.std() * np.sqrt(252)

                # Optimization: Generate Unconstrained Target Weights
                target_weights = self._generate_smart_weights(
                    day_preds, 
                    top_n, 
                    day_prices, 
                    current_vol
                )
                
                # Data Prep for Constraint Engine
                adv_map = {}
                price_map_risk = {}
                sector_map = {}
                
                if isinstance(day_prices, pd.DataFrame):
                    if 'volume' in day_prices.columns:
                        adv_map = day_prices['volume'].to_dict()
                    if 'close' in day_prices.columns:
                        price_map_risk = day_prices['close'].to_dict()
                    if 'sector' in day_prices.columns:
                        sector_map = day_prices['sector'].to_dict()
                
                # Constraint Enforcement (Sector, ADV, Leverage)
                safe_weights = self.risk_manager.apply_constraints(
                    target_weights, 
                    self.portfolio.total_value,
                    adv_map=adv_map,
                    price_map=price_map_risk,
                    sector_map=sector_map,
                    current_volatility=current_vol
                )
                self._execute_rebalance(date, safe_weights, day_prices)

            # 5. State Persistence (Portfolio is the Single Source of Truth)
            self.portfolio.record_daily_snapshot(date)

        return self._wrap_results()

    def _check_trailing_stops(self, date, day_prices):
        """
        Monitors positions for exit conditions.
        Condition: $P_t < P_{peak} \times (1 - StopPct)$
        """
        current_holdings = list(self.portfolio.positions.keys())
        
        for ticker in current_holdings:
            if ticker not in day_prices.index: continue
            
            current_price = day_prices.loc[ticker, 'close']
            
            # Update High Water Mark (HWM)
            if ticker not in self.high_water_marks:
                self.high_water_marks[ticker] = current_price
            else:
                self.high_water_marks[ticker] = max(self.high_water_marks[ticker], current_price)
            
            # Evaluate Stop Condition
            stop_price = self.high_water_marks[ticker] * (1 - self.trailing_stop_pct)
            
            if current_price < stop_price:
                logger.info(f"🛑 Trailing Stop Hit: {ticker} @ {current_price:.2f} (Peak: {self.high_water_marks[ticker]:.2f})")
                # Immediate Liquidation
                shares = self.portfolio.positions[ticker]
                self._process_order(date, ticker, 0, day_prices, "TRAILING_STOP")
                # Metadata cleanup happens in _process_order via _log_trade logic

    def _execute_rebalance(self, date, target_weights, day_prices):
        """
        Executes portfolio turnover with liquidity preservation logic.
        Phase 1: Liquidations (Sells) to raise capital.
        Phase 2: Acquisitions (Buys) using available cash.
        """
        current_holdings = self.portfolio.get_holdings()
        
        # --- Turnover Constraint Enforcement ---
        # Formula: Turnover = Sum(|w_target - w_current|) / 2
        current_val = self.portfolio.total_value
        if current_val > 0:
            current_weights = {t: self.portfolio.get_position_value(t) / current_val for t in current_holdings}
        else:
            current_weights = {}
        
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        turnover = sum(abs(target_weights.get(t, 0) - current_weights.get(t, 0)) for t in all_tickers) / 2.0
        
        if turnover > self.max_turnover:
            scale = self.max_turnover / turnover
            logger.info(f"⚠️ Turnover {turnover:.1%} > {self.max_turnover:.1%}. Scaling rebalance by {scale:.2f}x")
            
            # Linear shrinkage of target weights towards current weights
            scaled_weights = {}
            for t in all_tickers:
                w_c = current_weights.get(t, 0)
                w_t = target_weights.get(t, 0)
                scaled_weights[t] = w_c + scale * (w_t - w_c)
            target_weights = scaled_weights
        
        # Calculate discrete target shares
        targets = {}
        for ticker, weight in target_weights.items():
            target_dollar = (self.portfolio.total_value * weight) * 0.99
            try:
                if ticker in day_prices.index:
                    px = day_prices.loc[ticker, self.execution_price]
                    if px > 0:
                        targets[ticker] = int(target_dollar // px)
            except Exception:
                continue

        # Identify liquidations (Positions held but not in target set)
        for ticker in current_holdings:
            if ticker not in targets:
                targets[ticker] = 0

        # Segregate Orders for Phased Execution
        sells = []
        buys = []
        
        for ticker, target_shares in targets.items():
            current_shares = current_holdings.get(ticker, 0)
            if target_shares < current_shares:
                sells.append((ticker, target_shares))
            elif target_shares > current_shares:
                buys.append((ticker, target_shares))
        
        # Phase 1: Execution of Sells
        for ticker, target in sells:
            reason = "LIQUIDATE" if target == 0 else "REBALANCE"
            self._process_order(date, ticker, target, day_prices, reason)
            
        # Phase 2: Dynamic Cash Management & Acquisitions
        # Re-assess cash post-sells. Execution failures (e.g. halted stocks) may reduce buying power.
        total_buy_value = 0.0
        buy_orders = []
        for ticker, target in buys:
            if ticker in day_prices.index:
                px = day_prices.loc[ticker, self.execution_price]
                diff = target - self.portfolio.positions.get(ticker, 0)
                if diff > 0:
                    total_buy_value += diff * px
                    buy_orders.append((ticker, target))
        
        # Pro-rata scaling if cash constrained
        # Reserve 1% cash buffer for friction/costs
        available_cash = self.portfolio.cash * 0.99
        scale_factor = 1.0
        if total_buy_value > available_cash and total_buy_value > 0:
            scale_factor = available_cash / total_buy_value
            if scale_factor < 0.9:
                logger.info(f"⚠️ Cash constrained. Scaling buys by {scale_factor:.2f}x (Need ${total_buy_value:,.0f}, Have ${available_cash:,.0f})")
            else:
                logger.debug(f"Cash constrained. Scaling buys by {scale_factor:.2f}x")

        # Execute Buys
        for ticker, target in buy_orders:
            # Apply scaling to the *incremental* shares needed
            current = self.portfolio.positions.get(ticker, 0)
            diff = target - current
            adjusted_diff = int(diff * scale_factor)
            final_target = current + adjusted_diff
            
            if adjusted_diff > 0:
                self._process_order(date, ticker, final_target, day_prices, "REBALANCE")

    def _process_order(self, date, ticker, target_shares, day_prices, reason):
        """Handles order routing, impact estimation, and settlement."""
        current_shares = self.portfolio.positions.get(ticker, 0)
        diff = target_shares - current_shares
        
        if diff == 0: return

        try:
            mkt_data = day_prices.loc[ticker]
            side = 'buy' if diff > 0 else 'sell'
            
            # Market Impact Estimation (Almgren-Chriss)
            impact = 0.0
            vol = mkt_data.get('volatility', 0.02)
            if self.impact_model:
                impact = self.impact_model.calculate_impact(abs(diff), mkt_data['volume'], vol, side=side)

            # Execution Simulation (Fill Price, Comm, Friction)
            fill_result = self.executor.execute_order(
                ticker=ticker,
                shares=abs(diff),
                side=side,
                price=mkt_data[self.execution_price],
                volume=mkt_data['volume'],
                date=date,
                impact_rate=impact,
                volatility=vol
            )

            # Settlement & Portfolio Update
            if fill_result.get('success', False) and fill_result['shares'] > 0:
                comm = fill_result.get('commission_usd', 0.0)
                pnl = 0.0
                success = False
                
                if side == 'buy':
                    res = self.portfolio.buy(ticker, fill_result['shares'], fill_result['fill_price'], commission=comm)
                    if res is not None: success = True
                else:
                    res = self.portfolio.sell(ticker, fill_result['shares'], fill_result['fill_price'], commission=comm)
                    if res is not None: 
                        pnl = res
                        success = True
                
                # Clean up tracking metadata on full exit
                if side == 'sell' and self.portfolio.positions.get(ticker, 0) == 0:
                    if ticker in self.high_water_marks: del self.high_water_marks[ticker]
                
                if success:
                    self._log_trade_detailed(fill_result, reason, pnl)

        except Exception as e:
            logger.error(f"Execution Error: {ticker} | {date} | {str(e)}")

    def _generate_smart_weights(self, day_preds, top_n, day_prices, current_vol):
        """
        Constructs the target portfolio using advanced weighting logic.
        
        Components:
        1. **Regime Filter**: De-risk to cash during high volatility or drawdown.
        2. **Hysteresis (Buffer Zone)**: Reduces turnover by holding existing positions 
           until rank drops below $1.6 \times N$.
        3. **Inverse Volatility Weighting**: $w_i \propto \frac{1}{\max(\sigma_i, \epsilon)}$
        """
        if day_preds.empty: return {}
        
        # --- Regime Detection (Drawdown + Trend) ---
        if len(self.portfolio.equity_curve) > 50:
            navs = [x['total_value'] for x in self.portfolio.equity_curve]
            nav_series = pd.Series(navs)
            current_nav = nav_series.iloc[-1]
            peak_nav = nav_series.cummax().iloc[-1]
            
            current_dd = (current_nav - peak_nav) / peak_nav if peak_nav > 0 else 0.0
            ma_50 = nav_series.rolling(window=50).mean().iloc[-1]
            
            # Logic: Deep Drawdown (-15%) AND Below Trend (MA50) -> Safety (Cash)
            if current_dd < -0.15 and current_nav < ma_50:
                logger.info(f"🛡️ REGIME DEFENSE: DD {current_dd:.1%} & NAV < MA50. Going to Cash.")
                return {}

        # 1. Volatility Regime Filter
        # If current strategy vol > 1.5x Target -> Safety (Cash)
        if current_vol > (self.risk_manager.target_volatility * 1.5):
            logger.info(f"🛡️ DEFENSIVE MODE: Volatility {current_vol:.1%} > Limit. Going to Cash.")
            return {}

        # 2. Selection with Hysteresis
        current_holdings = set(self.portfolio.positions.keys())
        
        # Rank predictions by signal strength
        day_preds = day_preds.copy()
        day_preds['rank'] = range(1, len(day_preds) + 1)
        
        # Buffer Logic: Hold existing if Rank <= 1.6 * N
        buffer_limit = int(top_n * 1.6) # e.g., 25 -> 40
        selected_tickers = []
        
        # Keep existing if within buffer
        for row in day_preds.itertuples():
            if row.ticker in current_holdings and row.rank <= buffer_limit:
                selected_tickers.append(row.ticker)
        
        # Fill remaining slots with Top Alpha picks
        for row in day_preds.itertuples():
            if len(selected_tickers) >= top_n: break
            if row.ticker not in selected_tickers:
                selected_tickers.append(row.ticker)
                
        # 3. Inverse Volatility Weighting
        weights = {}
        inv_vols = {}
        total_inv_vol = 0.0
        
        for ticker in selected_tickers:
            vol = 0.02 # Default fallback
            try:
                if isinstance(day_prices, pd.DataFrame) and ticker in day_prices.index:
                    vol = day_prices.loc[ticker, 'volatility']
            except Exception:
                pass
            
            # Stability: Floor volatility to avoid division by zero or extreme weights
            # Floor at 0.5% daily vol (~8% annualized) to prevent single-stock dominance
            vol = max(vol, 0.005)
            
            inv_vol = 1.0 / vol
            inv_vols[ticker] = inv_vol
            total_inv_vol += inv_vol
            
        if total_inv_vol > 0:
            for ticker in selected_tickers:
                weights[ticker] = inv_vols[ticker] / total_inv_vol
        else:
            # Fallback: Equal Weight
            for ticker in selected_tickers:
                weights[ticker] = 1.0 / len(selected_tickers)

        return weights

    def _log_trade_detailed(self, trade, reason, realized_pnl):
        """
        Logs trade execution with FIFO Entry Price tracking.
        Calculates realized returns and links exits to entry dates.
        """
        ticker = trade['ticker']
        
        if trade['side'] == 'buy':
            # Queue Entry for FIFO tracking
            if ticker not in self.position_entry_map:
                self.position_entry_map[ticker] = []
            
            self.position_entry_map[ticker].append({
                'price': trade['fill_price'],
                'shares': trade['shares'],
                'date': trade['date']
            })
            
            # Initialize High Water Mark for new position
            if ticker not in self.high_water_marks:
                self.high_water_marks[ticker] = trade['fill_price']
                
            record = {
                **trade,
                'reason': reason,
                'pnl': 0.0,
                'entry_price': np.nan,
                'return_pct': 0.0
            }
            
        else: # Sell
            # FIFO Matching: Consume oldest cost basis layers
            shares_to_close = trade['shares']
            total_cost = 0.0
            shares_matched = 0
            entry_date = None
            
            if ticker in self.position_entry_map:
                entries = self.position_entry_map[ticker]
                while shares_to_close > 0 and entries:
                    entry = entries[0]
                    if entry_date is None: entry_date = entry['date']
                    
                    matched = min(shares_to_close, entry['shares'])
                    total_cost += matched * entry['price']
                    shares_matched += matched
                    shares_to_close -= matched
                    
                    entry['shares'] -= matched
                    if entry['shares'] <= 0:
                        entries.pop(0)
            
            avg_entry = total_cost / shares_matched if shares_matched > 0 else 0.0
            
            record = {
                **trade,
                'reason': reason,
                'pnl': realized_pnl,
                'entry_price': avg_entry,
                'entry_date': entry_date,
                'return_pct': (realized_pnl / (avg_entry * trade['shares'])) if avg_entry > 0 else 0.0
            }
            
        self.trades.append(record)

    def _is_rebalance_day(self, date, index):
        if index == 0: return True
        dt = pd.to_datetime(date)
        if self.rebalance_freq == 'weekly': return dt.weekday() == 0 # Monday
        if self.rebalance_freq == 'monthly': return dt.is_month_start
        return True

    def _wrap_results(self):
        # Convert Portfolio's internal equity list to DataFrame
        df_equity = pd.DataFrame(self.portfolio.equity_curve)
        df_trades = pd.DataFrame(self.trades)
        
        metrics = self.metrics_engine.calculate_all(
            df_equity, 
            df_trades, 
            self.initial_capital
        )
        
        return {
            'equity_curve': df_equity, 
            'trades': df_trades, 
            'metrics': metrics
        }