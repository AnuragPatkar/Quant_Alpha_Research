"""
quant_alpha/backtest/engine.py
================================
Event-driven Backtest Simulation Engine.

Takes pre-generated alpha predictions and simulates portfolio performance under
realistic market conditions: transaction costs, slippage, market impact,
rebalance frequency, position limits, and trailing stops.

Public API
----------
    engine = BacktestEngine(initial_capital=1_000_000, ...)
    results = engine.run(predictions, prices, top_n=25)

    results keys:
        'equity_curve' : pd.DataFrame  ['date', 'total_value', 'invested_value']
        'trades'       : pd.DataFrame  trade log with pnl column
        'metrics'      : Dict          from backtest.metrics.compute_metrics()
        'positions'    : pd.DataFrame  end-of-day holdings per ticker
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Any

from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Vectorised (date-loop) backtest engine.

    Rebalances the portfolio at the configured frequency, applies round-trip
    transaction costs, linear slippage, and optional Almgren-Chriss market
    impact.  Trailing stops are enforced per position.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.0010,           # one-way, as fraction of trade value
        spread: float = 0.0005,               # half-spread, one-way
        slippage: float = 0.0002,             # linear slippage per unit of ADV fraction
        position_limit: float = 0.10,         # max fraction of NAV per single position
        rebalance_freq: str = "weekly",        # 'daily' | 'weekly' | 'monthly'
        use_market_impact: bool = True,
        target_volatility: float = 0.15,       # annualised; 0 = no vol targeting
        max_adv_participation: float = 0.02,   # max fraction of ADV per trade
        trailing_stop_pct: float = 0.10,       # 0 = disabled
        execution_price: str = "open",         # 'open' | 'close'
        max_turnover: float = 0.20,            # max one-way turnover per rebalance
    ):
        self.initial_capital       = initial_capital
        self.commission            = commission
        self.spread                = spread
        self.slippage              = slippage
        self.position_limit        = position_limit
        self.rebalance_freq        = rebalance_freq
        self.use_market_impact     = use_market_impact
        self.target_volatility     = target_volatility
        self.max_adv_participation = max_adv_participation
        self.trailing_stop_pct     = trailing_stop_pct
        self.execution_price       = execution_price
        self.max_turnover          = max_turnover

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        top_n: int = 25,
        is_weights: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full backtest simulation.

        Parameters
        ----------
        predictions : DataFrame with columns ['date', 'ticker', 'prediction'].
                      Values may be alpha scores (top_n mode) or pre-computed
                      weights (optimiser mode).
        prices      : DataFrame with columns ['date', 'ticker', 'open', 'close',
                      'volume', 'volatility'].  'sector' is optional.
        top_n       : Number of positions when using alpha-score mode.
                      Ignored when predictions already contain portfolio weights.

        Returns
        -------
        Dict with keys: 'equity_curve', 'trades', 'metrics', 'positions'.
        """
        preds  = predictions.copy()
        px_df  = prices.copy()

        preds["date"]  = pd.to_datetime(preds["date"])
        px_df["date"]  = pd.to_datetime(px_df["date"])

        all_dates = sorted(px_df["date"].unique())

        # State
        cash        = float(self.initial_capital)
        holdings: Dict[str, float] = {}          # ticker → shares held
        entry_price: Dict[str, float] = {}       # ticker → entry price for trailing stop
        peak_price:  Dict[str, float] = {}       # ticker → running peak for trailing stop

        equity_rows = []
        trade_rows  = []

        rebal_dates = self._rebalance_dates(all_dates, self.rebalance_freq)

        for date in all_dates:
            day_px = px_df[px_df["date"] == date].set_index("ticker")

            # ---- Mark-to-market ----
            nav = cash
            for ticker, shares in holdings.items():
                if ticker in day_px.index:
                    nav += shares * float(day_px.loc[ticker, "close"])

            # ---- Trailing stop checks ----
            if self.trailing_stop_pct > 0:
                stops_triggered = []
                for ticker, shares in holdings.items():
                    if ticker not in day_px.index:
                        continue
                    close = float(day_px.loc[ticker, "close"])
                    pk    = peak_price.get(ticker, close)
                    peak_price[ticker] = max(pk, close)
                    if close < peak_price[ticker] * (1 - self.trailing_stop_pct):
                        stops_triggered.append(ticker)

                for ticker in stops_triggered:
                    shares = holdings.pop(ticker, 0)
                    ex_px  = float(day_px.loc[ticker, self.execution_price])
                    cost   = abs(shares) * ex_px * (self.commission + self.spread)
                    proceeds = shares * ex_px - cost
                    cash    += proceeds
                    ent_px = entry_price.pop(ticker, ex_px)
                    peak_price.pop(ticker, None)
                    trade_rows.append({
                        "date":        date,
                        "ticker":      ticker,
                        "action":      "sell" if shares > 0 else "buy",
                        "side":        "long" if shares > 0 else "short",
                        "reason":      "trailing_stop_exit",
                        "size":        abs(shares),
                        "entry_price": ent_px,
                        "exit_price":  ex_px,
                        "pnl":         proceeds - (shares * ent_px),
                    })

            # ---- Rebalance ----
            if date in rebal_dates:
                day_preds = preds[preds["date"] == date]

                if not day_preds.empty:
                    target_weights = self._build_target_weights(
                        day_preds, day_px, nav, top_n, is_weights
                    )
                    trades = self._execute_rebalance(
                        target_weights, holdings, day_px, nav, cash
                    )
                    for t in trades:
                        cash      += t["cash_delta"]
                        ticker     = t["ticker"]
                        shares_delta = t["shares_delta"]
                        exec_price   = t["price"]
                        
                        realized_pnl = 0.0
                        old_shares = holdings.get(ticker, 0.0)
                        old_entry  = entry_price.get(ticker, exec_price)
                        
                        # Determine position side (LONG/SHORT)
                        pos_side = "long"
                        if old_shares < 0 or (old_shares == 0 and shares_delta < 0):
                            pos_side = "short"

                        # Action & Reason
                        action = "buy" if shares_delta > 0 else "sell"
                        
                        if old_shares == 0:
                            reason = "entry"
                        elif abs(old_shares + shares_delta) <= 1e-6:
                            reason = "exit"
                        elif (old_shares > 0 and shares_delta > 0) or (old_shares < 0 and shares_delta < 0):
                            reason = "increase"
                        elif abs(shares_delta) > abs(old_shares):
                            reason = "flip"
                        else:
                            reason = "reduce"
                            
                        if shares_delta > 0:
                            # Buying
                            new_shares = old_shares + shares_delta
                            if old_shares >= 0:
                                entry_price[ticker] = ((old_shares * old_entry) + (shares_delta * exec_price)) / new_shares
                            else:
                                covered = min(abs(old_shares), shares_delta)
                                realized_pnl = (old_entry - exec_price) * covered - t["cost"]
                                if new_shares > 0:
                                    entry_price[ticker] = exec_price
                        elif shares_delta < 0:
                            # Selling
                            new_shares = old_shares + shares_delta
                            if old_shares <= 0:
                                entry_price[ticker] = ((abs(old_shares) * old_entry) + (abs(shares_delta) * exec_price)) / abs(new_shares)
                            else:
                                sold = min(old_shares, abs(shares_delta))
                                realized_pnl = (exec_price - old_entry) * sold - t["cost"]
                                if new_shares < 0:
                                    entry_price[ticker] = exec_price

                        # Update holdings
                        holdings[ticker] = old_shares + shares_delta
                        
                        # Cleanup dust / closed positions
                        if abs(holdings[ticker]) <= 1e-6:
                            holdings.pop(ticker, None)
                            entry_price.pop(ticker, None)
                            peak_price.pop(ticker, None)
                        else:
                            peak_price[ticker] = max(peak_price.get(ticker, exec_price), exec_price)
                            
                        trade_rows.append({
                            "date":        date,
                            "ticker":      ticker,
                            "action":      action,
                            "side":        pos_side,
                            "reason":      reason,
                            "size":        abs(t["shares_delta"]),
                            "entry_price": old_entry if reason in ["reduce", "exit", "flip"] else exec_price,
                            "exit_price":  exec_price if reason in ["reduce", "exit", "flip"] else np.nan,
                            "pnl":         realized_pnl,
                        })

            # ---- Recalculate NAV after trades ----
            nav = cash
            invested = 0.0
            for ticker, shares in holdings.items():
                if ticker in day_px.index:
                    val      = shares * float(day_px.loc[ticker, "close"])
                    nav     += val
                    invested += val

            equity_rows.append({
                "date":            date,
                "total_value":     round(nav, 4),
                "invested_value":  round(invested, 4),
                "cash":            round(cash, 4),
            })

        equity_curve = pd.DataFrame(equity_rows)
        trades_df    = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["date", "ticker", "action", "side", "reason", "size", "entry_price", "exit_price", "pnl"]
        )

        if equity_curve.empty:
            equity_curve = pd.DataFrame(columns=["date", "total_value", "invested_value", "cash"])
            metrics = {
                "Total Return": "0.00%", "CAGR": "0.00%", "Ann. Volatility": "0.00%",
                "Sharpe Ratio": 0.0, "Sortino Ratio": 0.0, "Max Drawdown": "0.00%"
            }
        else:
            metrics = compute_metrics(equity_curve)

        last_date = all_dates[-1] if all_dates else None

        return {
            "equity_curve": equity_curve,
            "trades":       trades_df,
            "metrics":      metrics,
            "positions":    self._snapshot_positions(holdings, px_df, last_date),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebalance_dates(self, all_dates, freq: str):
        """Return the set of dates on which rebalancing occurs."""
        dates_s = pd.Series(pd.to_datetime(all_dates))
        if freq == "daily":
            return set(all_dates)
        elif freq == "weekly":
            # Rebalance on the first trading day of each ISO week
            idx = dates_s.groupby(dates_s.dt.isocalendar().week.values * 10000
                                   + dates_s.dt.year.values).first()
            return set(idx)
        elif freq == "monthly":
            idx = dates_s.groupby(dates_s.dt.to_period("M")).first()
            return set(idx)
        else:
            return set(all_dates)   # default: daily

    def _build_target_weights(
        self,
        day_preds: pd.DataFrame,
        day_px: pd.DataFrame,
        nav: float,
        top_n: int,
        is_weights: bool = False,
    ) -> Dict[str, float]:
        """
        Convert predictions to target dollar weights.

        If 'prediction' values sum to ~1 (optimiser mode), treat as explicit
        weights.  Otherwise take the top_n by score and equal-weight them.
        """
        pred_col  = "prediction"
        available = day_preds[day_preds["ticker"].isin(day_px.index)]

        if available.empty:
            return {}

        if is_weights:
            # Predictions are already constrained weights from the optimizer
            weights = available.set_index("ticker")[pred_col]
        else:
            # Predictions are alpha scores. Equal weight top_n.
            top = available.nlargest(top_n, pred_col)
            n   = len(top)
            weights = pd.Series(1.0 / n, index=top["ticker"])

            # Apply position limit and renormalise ONLY for top_n mode
            weights = weights.clip(upper=self.position_limit)
            total   = weights.sum()
            if total > 1e-9:
                weights = weights / total

        return {t: w * nav for t, w in weights.items()}

    def _execution_cost(self, shares_delta: float, price: float, volume: float) -> float:
        """Total one-way cost: commission + spread + slippage + market impact."""
        value = abs(shares_delta) * price
        cost  = value * (self.commission + self.spread)

        if self.use_market_impact and volume > 1e-6:
            adv_frac = abs(shares_delta) / volume
            adv_frac = min(adv_frac, self.max_adv_participation)
            impact   = value * adv_frac * self.slippage
            cost    += impact

        return cost

    def _execute_rebalance(
        self,
        target_weights: Dict[str, float],   # ticker → target dollar value
        holdings: Dict[str, float],
        day_px: pd.DataFrame,
        nav: float,
        cash: float,
    ):
        """
        Compute trades needed to move from current holdings to target weights.
        Respects max_turnover constraint.
        """
        ex_field = self.execution_price
        trades   = []

        # Current dollar holdings
        current = {}
        for ticker, shares in holdings.items():
            if ticker in day_px.index:
                current[ticker] = shares * float(day_px.loc[ticker, "close"])
            else:
                current[ticker] = 0.0

        # All tickers involved
        all_tickers = set(target_weights) | set(current)

        # Compute desired delta
        deltas = {}
        for ticker in all_tickers:
            target  = target_weights.get(ticker, 0.0)
            curr_v  = current.get(ticker, 0.0)
            deltas[ticker] = target - curr_v

        # Turnover cap: scale down all deltas proportionally if needed
        total_turnover = sum(abs(d) for d in deltas.values())
        max_tv_dollars = nav * self.max_turnover
        if total_turnover > max_tv_dollars and total_turnover > 0:
            scale = max_tv_dollars / total_turnover
            deltas = {k: v * scale for k, v in deltas.items()}

        for ticker, delta_value in deltas.items():
            if abs(delta_value) < 1.0:   # skip tiny trades < $1
                continue
            if ticker not in day_px.index:
                continue

            price = float(day_px.loc[ticker, ex_field])
            if price <= 0:
                continue

            volume = float(day_px.loc[ticker, "volume"]) if "volume" in day_px.columns else 1e6

            shares_delta = delta_value / price
            cost         = self._execution_cost(shares_delta, price, volume)
            cash_delta   = -(shares_delta * price) - cost

            trades.append({
                "ticker":       ticker,
                "shares_delta": shares_delta,
                "price":        price,
                "cost":         cost,
                "cash_delta":   cash_delta,
            })

        return trades

    def _snapshot_positions(
        self,
        holdings: Dict[str, float],
        px_df: pd.DataFrame,
        last_date,
    ) -> pd.DataFrame:
        """Return end-of-simulation holdings as a DataFrame."""
        if last_date is None:
            return pd.DataFrame(columns=["ticker", "shares", "price", "value"])
        last_px = px_df[px_df["date"] == last_date].set_index("ticker")
        rows = []
        for ticker, shares in holdings.items():
            price = float(last_px.loc[ticker, "close"]) if ticker in last_px.index else 0.0
            rows.append({
                "ticker": ticker,
                "shares": shares,
                "price":  price,
                "value":  shares * price,
            })
        return pd.DataFrame(rows).sort_values("value", ascending=False)