import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from .portfolio import Portfolio
from .execution import ExecutionSimulator
from .market_impact import AlmgrenChrissImpact
from .risk_manager import RiskManager
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        spread: float = 0.0005,
        slippage: float = 0.0002,
        position_limit: float = 0.05,
        leverage_limit: float = 1.0,
        rebalance_freq: str = 'weekly',
        use_market_impact: bool = True
    ):
        self.portfolio = Portfolio(initial_capital)
        self.executor = ExecutionSimulator(commission, spread, slippage)
        self.impact_model = AlmgrenChrissImpact() if use_market_impact else None
        self.risk_manager = RiskManager(position_limit, leverage_limit)
        self.metrics_engine = PerformanceMetrics()
        
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.trades = []

    def run(self, predictions: pd.DataFrame, prices: pd.DataFrame, top_n: int = 50):
        # 1. Setup Data - Ensure MultiIndex is sorted for performance
        predictions = predictions.sort_values(['date', 'prediction'], ascending=[True, False])
        dates = sorted(predictions['date'].unique())
        prices_indexed = prices.set_index(['date', 'ticker']).sort_index()

        for i, date in enumerate(dates):
            # 2. MultiIndex Slicing Safety: Optimized lookup
            try:
                day_prices = prices_indexed.loc[date]
            except KeyError:
                logger.warning(f"Market data missing for date: {date}")
                continue
            
            # 3. Mark-to-Market (Sync Portfolio with current prices)
            current_holdings = list(self.portfolio.positions.keys())
            if current_holdings:
                valid_tickers = [t for t in current_holdings if t in day_prices.index]
                if valid_tickers:
                    price_map = day_prices.loc[valid_tickers, 'close'].to_dict()
                    self.portfolio.update_prices(price_map)

            # 4. Rebalance Logic
            if self._is_rebalance_day(date, i):
                day_preds = predictions[predictions['date'] == date].head(top_n)
                target_weights = self._generate_weights(day_preds, top_n)
                
                # Apply Risk Management
                safe_weights = self.risk_manager.apply_constraints(target_weights, self.portfolio.total_value)
                self._execute_rebalance(date, safe_weights, prices_indexed)

            # 5. Snapshot: Single Source of Truth (Portfolio internally tracks equity_curve)
            self.portfolio.record_daily_snapshot(date)

        return self._wrap_results()

    def _execute_rebalance(self, date, target_weights, prices_indexed):
        """Phase-based execution: Sell first to free up capital, then Buy."""
        current_holdings = self.portfolio.get_holdings()
        
        # Calculate target shares for all desired positions
        targets = {}
        for ticker, weight in target_weights.items():
            target_dollar = (self.portfolio.total_value * weight) * 0.99
            try:
                if (date, ticker) in prices_indexed.index:
                    px = prices_indexed.loc[(date, ticker), 'close']
                    if px > 0:
                        targets[ticker] = int(target_dollar // px)
            except Exception:
                continue

        # Identify liquidations (held but not in target weights)
        for ticker in current_holdings:
            if ticker not in targets:
                targets[ticker] = 0

        # Split into Sells and Buys to prioritize liquidity
        sells = []
        buys = []
        
        for ticker, target_shares in targets.items():
            current_shares = current_holdings.get(ticker, 0)
            if target_shares < current_shares:
                sells.append((ticker, target_shares))
            elif target_shares > current_shares:
                buys.append((ticker, target_shares))
        
        # Execute Sells first
        for ticker, target in sells:
            reason = "LIQUIDATE" if target == 0 else "REBALANCE"
            self._process_order(date, ticker, target, prices_indexed, reason)
            
        # Execute Buys next
        for ticker, target in buys:
            self._process_order(date, ticker, target, prices_indexed, "REBALANCE")

    def _process_order(self, date, ticker, target_shares, prices_indexed, reason):
        current_shares = self.portfolio.positions.get(ticker, 0)
        diff = target_shares - current_shares
        
        if diff == 0: return

        try:
            mkt_data = prices_indexed.loc[(date, ticker)]
            side = 'buy' if diff > 0 else 'sell'
            
            # Market Impact calculation
            impact = 0.0
            vol = mkt_data.get('volatility', 0.02)
            if self.impact_model:
                impact = self.impact_model.calculate_impact(abs(diff), mkt_data['volume'], vol, side=side)

            # Execution Simulator (Architecture Part 5A)
            fill_result = self.executor.execute_order(
                ticker=ticker,
                shares=abs(diff),
                side=side,
                price=mkt_data['close'],
                volume=mkt_data['volume'],
                date=date,
                impact_rate=impact,
                volatility=vol
            )

            # Finalize with Portfolio class logic
            if fill_result.get('success', False) and fill_result['shares'] > 0:
                comm = fill_result.get('commission_usd', 0.0)
                if side == 'buy':
                    self.portfolio.buy(ticker, fill_result['shares'], fill_result['fill_price'], commission=comm)
                else:
                    self.portfolio.sell(ticker, fill_result['shares'], fill_result['fill_price'], commission=comm)
                
                self.trades.append({**fill_result, 'reason': reason})

        except Exception as e:
            logger.error(f"Execution Error: {ticker} | {date} | {str(e)}")

    def _generate_weights(self, day_preds, top_n):
        if day_preds.empty: return {}
        return {row.ticker: 1.0/len(day_preds) for row in day_preds.itertuples()}

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
            'equity': df_equity, 
            'trades': df_trades, 
            'metrics': metrics
        }