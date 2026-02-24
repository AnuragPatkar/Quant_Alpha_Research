import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from .utils import calculate_returns, calculate_max_drawdown

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Institutional-Grade Performance Analytics.
    Optimized for efficiency, mathematical accuracy, and frequency flexibility.
    """
    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
            periods_per_year: 252 for Stocks, 365 for Crypto, 52 for Weekly data.
        """
        self.rf_rate = risk_free_rate
        self.ann_factor = periods_per_year
        self.daily_rf = risk_free_rate / periods_per_year

    def calculate_all(self,
                      equity_df:pd.DataFrame,
                      trades_df:pd.DataFrame,
                      initial_capital:float
                      ) -> Dict:
        """
        Main entry point. Calculates metrics in a non-redundant pipeline.
        """

        if equity_df.empty:
            logger.error("Equity DataFrame is empty.")
            return {}
        
        # 1. Data Cleaning & Return Calculation
        df = equity_df.copy()

        # Ensure DatetimeIndex
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError("DataFrame must have a 'date' column or a DatetimeIndex.")
        
        # Calculate returns and drop first row to avoid bias in mean/std
        df['return'] = calculate_returns(df['total_value'])
        returns_series = df['return'].dropna()

        metrics = {}

        # 2. Sequential Calculation Pipeline (No Redundancy)
        ret_metrics = self._calculate_return_metrics(df, initial_capital)
        risk_metrics = self._calculate_risk_metrics(returns_series)
        dd_metrics = self._calculate_drawdown_metrics(df)

        # Risk-Adjusted metrics using previously calculated components
        risk_adj = self._calculate_risk_adjusted_metrics(
            cagr=ret_metrics['cagr'],
            ann_vol=risk_metrics['annual_volatility'],
            downside_vol=risk_metrics['downside_volatility'],
            max_dd=dd_metrics['max_drawdown'],
            daily_mean_ret=returns_series.mean()
        )

        # Merge results
        metrics.update(ret_metrics)
        metrics.update(risk_metrics)
        metrics.update(dd_metrics)
        metrics.update(risk_adj)

        # 3. Trade Metrics (Using Equity Duration for accurate Frequency)
        strategy_duration_days = (df.index[-1] - df.index[0]).days
        metrics.update(self._calculate_trade_metrics(trades_df, strategy_duration_days))
        return metrics
    
    def _calculate_return_metrics(self, df: pd.DataFrame, initial_capital: float) -> Dict:
        final_value = df['total_value'].iloc[-1]
        days = (df.index[-1] - df.index[0]).days
        years = max(days / 365.25, 1 / self.ann_factor)

        cagr = (final_value / initial_capital) ** (1 / years) - 1

        # Periodic analysis
        monthly_ret = df['total_value'].resample('ME').last().pct_change().dropna()

        return {
            'total_return': (final_value / initial_capital) - 1,
            'cagr': cagr,
            'years': years,
            'best_month': monthly_ret.max() if not monthly_ret.empty else 0,
            'worst_month': monthly_ret.min() if not monthly_ret.empty else 0,
            'avg_monthly_return': monthly_ret.mean() if not monthly_ret.empty else 0,
            'monthly_win_rate': (monthly_ret > 0).mean() if not monthly_ret.empty else 0
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(self.ann_factor)

        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.ann_factor) if len(downside_returns) > 1 else 0.0

        # Daily Historical VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if not returns[returns <= var_95].empty else 0.0

        return {
            'annual_volatility': ann_vol,
            'downside_volatility': downside_vol,
            'daily_var_95': var_95,
            'daily_cvar_95': cvar_95,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _calculate_risk_adjusted_metrics(self, cagr, ann_vol, downside_vol, max_dd, daily_mean_ret) -> Dict:
        # Annualized Excess Return for Sharpe/Sortino
        ann_excess_ret = cagr - self.rf_rate

        sharpe = ann_excess_ret / ann_vol if ann_vol > 0 else 0
        sortino = ann_excess_ret / downside_vol if downside_vol > 0 else sharpe
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        return {
            'sharpe_ratio': sharpe, 
            'sortino_ratio': sortino, 
            'calmar_ratio': calmar
        }
    
    def _calculate_drawdown_metrics(self, df: pd.DataFrame) -> Dict:
        equity = df['total_value']
        
        # Use robust calculation from utils for the headline number
        max_dd = calculate_max_drawdown(equity)

        # Re-calculate series for duration/recovery analysis (using same safety logic)
        running_max = equity.cummax()
        safe_max = running_max.replace(0, np.nan) # Safety against division by zero
        drawdowns = (equity - safe_max) / safe_max
        
        trough_date = drawdowns.idxmin()
        peak_date = equity[:trough_date].idxmax()
        peak_val = equity.loc[peak_date]

        # Recovery calculation (Peak-to-Peak)
        post_trough = equity[trough_date:]
        recovery_series = post_trough[post_trough >= peak_val]

        if not recovery_series.empty:
            recovery_date = recovery_series.index[0]
            recovery_days = (recovery_date - peak_date).days
        else:
            recovery_days = (equity.index[-1] - peak_date).days # Still underwater

        return {
            'max_drawdown': max_dd,
            'avg_drawdown': drawdowns[drawdowns < 0].mean() if any(drawdowns < 0) else 0,
            'max_dd_peak_to_trough': (trough_date - peak_date).days,
            'recovery_days': recovery_days,
            'max_drawdown_duration': (trough_date - peak_date).days 
        }
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame, strategy_days: int) -> Dict:
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return {'total_trades': 0, 'trade_win_rate': 0.0, 'profit_factor': 0.0, 'expectancy': 0.0, 'trades_per_day': 0.0}
        
        pnls = trades_df['pnl']
        # Filter for actual PnL events (exclude entries with 0 PnL)
        closed_trades = pnls[pnls != 0]
        
        wins = closed_trades[closed_trades > 0]
        losses = closed_trades[closed_trades < 0]
        
        win_rate = len(wins) / len(closed_trades) if not closed_trades.empty else 0.0
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0

        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            'total_trades': len(trades_df),
            'trade_win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'num_buys': int((trades_df['side'] == 'buy').sum()) if 'side' in trades_df.columns else 0,
            'num_sells': int((trades_df['side'] == 'sell').sum()) if 'side' in trades_df.columns else 0,
            'avg_cost_bps': trades_df['cost_bps'].mean() if 'cost_bps' in trades_df.columns else 0.0,
            'trades_per_day': len(trades_df) / max(strategy_days, 1)
        }

def print_metrics_report(metrics: Dict):
    """Institutional Grade Formatted Metrics Report"""
    print("\n" + "═"*70)
    print(f"{'QUANT ALPHA STRATEGY REPORT':^70}")
    print("═"*70)

    print(f"\n{'[ RETURN METRICS ]':<35} {'[ RISK METRICS ]':<35}")
    print(f"  Total Return: {metrics.get('total_return', 0):>10.2%}      Ann. Volatility: {metrics.get('annual_volatility', 0):>10.2%}")
    print(f"  CAGR:         {metrics.get('cagr', 0):>10.2%}      Downside Vol:    {metrics.get('downside_volatility', 0):>10.2%}")
    print(f"  Best Month:   {metrics.get('best_month', 0):>10.2%}      Daily VaR (95%): {metrics.get('daily_var_95', 0):>10.2%}")
    print(f"  Worst Month:  {metrics.get('worst_month', 0):>10.2%}      Daily CVaR (95%):{metrics.get('daily_cvar_95', 0):>10.2%}")
    
    print(f"\n{'[ RISK-ADJUSTED ]':<35} {'[ DRAWDOWN ]':<35}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):>10.2f}      Max Drawdown:    {metrics.get('max_drawdown', 0):>10.2%}")
    print(f"  Sortino Ratio:{metrics.get('sortino_ratio', 0):>10.2f}      Peak-to-Trough:  {metrics.get('max_dd_peak_to_trough', 0):>7} days")
    print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):>10.2f}      Recovery Time:   {metrics.get('recovery_days', 0):>7} days")
    
    print(f"\n{'[ TRADE STATISTICS ]':<70}")
    print(f"  Total Trades: {metrics.get('total_trades', 0):>10,}      Win Rate:        {metrics.get('trade_win_rate', 0):>10.2%}")
    print(f"  Profit Factor:{metrics.get('profit_factor', 0):>10.2f}      Expectancy:      {metrics.get('expectancy', 0):>10.2f}")
    print(f"  Trades/Day:   {metrics.get('trades_per_day', 0):>10.2f}      Avg Cost (bps):  {metrics.get('avg_cost_bps', 0):>10.1f}")
    
    print("\n" + "═"*70 + "\n")