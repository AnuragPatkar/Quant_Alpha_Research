import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Deque
import logging
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Track live performance metrics
    
    Features:
    - Rolling IC calculation (Spearman)
    - Sharpe ratio tracking (Risk-adjusted)
    - Drawdown monitoring
    - Alert triggers
    - Turnover & Cost tracking
    - Benchmark comparison
    
    Example:
        tracker = PerformanceTracker(window_days=60)
        
        # Update daily
        tracker.update(
            date='2024-01-15',
            predictions={'AAPL': 0.05, 'MSFT': 0.03},
            actual_returns={'AAPL': 0.04, 'MSFT': 0.02},
            portfolio_return=0.015,
            benchmark_return=0.010
        )
    """
    
    def __init__(
        self,
        window_days: int = 60,
        ic_warning_threshold: float = 0.02,
        ic_critical_threshold: float = 0.01,
        dd_warning_threshold: float = 0.10,
        dd_critical_threshold: float = 0.15,
        max_history: int = 252,
        risk_free_rate: float = 0.04  # 4% annual
    ):
        """
        Args:
            window_days: Rolling window for metrics
            ic_warning_threshold: IC warning level
            ic_critical_threshold: IC critical level
            dd_warning_threshold: Drawdown warning level
            dd_critical_threshold: Drawdown critical level
            max_history: Maximum days of history to keep in memory
            risk_free_rate: Annual risk-free rate
        """
        self.window_days = window_days
        self.ic_warning = ic_warning_threshold
        self.ic_critical = ic_critical_threshold
        self.dd_warning = dd_warning_threshold
        self.dd_critical = dd_critical_threshold
        self.max_history = max_history
        self.risk_free_rate = risk_free_rate
        
        # Historical data (using deque for memory efficiency)
        self.history: Deque = deque(maxlen=max_history)
        
        logger.info(f"PerformanceTracker initialized ({window_days}d window, max history {max_history})")
    
    def update(
        self,
        date: str,
        predictions: Dict[str, float],
        actual_returns: Dict[str, float],
        portfolio_return: float,
        benchmark_return: float = 0.0,
        turnover: float = 0.0,
        transaction_costs: float = 0.0,
        long_exposure: float = 0.0,
        short_exposure: float = 0.0,
        sector_exposure: Optional[Dict[str, float]] = None
    ):
        """
        Update tracker with new data
        
        Args:
            date: Date (YYYY-MM-DD)
            predictions: Predicted returns
            actual_returns: Actual returns
            portfolio_return: Portfolio return for the day
            benchmark_return: Benchmark return for the day
            turnover: Daily turnover
            transaction_costs: Estimated transaction costs
            long_exposure: Long exposure
            short_exposure: Short exposure
            sector_exposure: Sector exposure breakdown
        """
        # Calculate IC for this date (Spearman Rank Correlation)
        tickers = set(predictions.keys()) & set(actual_returns.keys())
        
        if len(tickers) < 2:
            daily_ic = 0.0
        else:
            pred_array = np.array([predictions[t] for t in tickers])
            actual_array = np.array([actual_returns[t] for t in tickers])
            
            # Use Spearman Rank Correlation
            daily_ic, _ = spearmanr(pred_array, actual_array)
            
            if np.isnan(daily_ic):
                daily_ic = 0.0
        
        # Store
        record = {
            'date': pd.to_datetime(date),
            'ic': daily_ic,
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'active_return': portfolio_return - benchmark_return,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'num_positions': len(tickers)
        }
        
        if sector_exposure:
            record['sector_exposure'] = sector_exposure
            
        self.history.append(record)
        
        logger.debug(f"Updated: {date}, IC={daily_ic:.4f}, Return={portfolio_return:.4f}")

    def get_status(self) -> Dict:
        """
        Get current performance status
        
        Returns:
            Dictionary with metrics and alerts
        """
        if not self.history:
            return {'status': 'NO_DATA'}
        
        df = pd.DataFrame(list(self.history))
        
        # Rolling window
        recent = df.tail(self.window_days)
        
        # Clean data for stats (Drop NaNs to prevent np.cov errors)
        valid_stats = recent.dropna(subset=['portfolio_return', 'benchmark_return'])
        valid_stats_total = df.dropna(subset=['portfolio_return', 'benchmark_return'])
        
        # Calculate metrics
        ic_recent = recent['ic'].mean()
        ic_std = recent['ic'].std()
        
        returns_recent = valid_stats['portfolio_return']
        benchmark_recent = valid_stats['benchmark_return']
        
        # Rolling Active Return (Simple Outperformance)
        active_ret_daily_rolling = recent['active_return'].mean()
        active_ret_rolling_annual = active_ret_daily_rolling * 252

        # Total Active Return (over full history)
        active_ret_daily_total = df['active_return'].mean()
        active_ret_total_annual = active_ret_daily_total * 252
        
        # Sharpe Ratio (Risk-Adjusted)
        sharpe = self._calculate_sharpe(returns_recent)
        
        # Beta and Alpha
        if len(returns_recent) > 1 and benchmark_recent.std() > 1e-8:
            covariance = np.cov(returns_recent, benchmark_recent)[0, 1]
            variance = benchmark_recent.var()
            beta = covariance / variance
            alpha = (returns_recent.mean() - self.risk_free_rate/252) - beta * (benchmark_recent.mean() - self.risk_free_rate/252)
            # Annualize Alpha
            alpha = alpha * 252
        else:
            beta = 0.0
            alpha = 0.0
            
        # --- TOTAL HISTORY METRICS ---
        # Calculate Alpha/Beta over full history to match Backtest Report
        if len(valid_stats_total) > 1 and valid_stats_total['benchmark_return'].std() > 1e-8:
            cov_total = np.cov(valid_stats_total['portfolio_return'], valid_stats_total['benchmark_return'])[0, 1]
            var_total = valid_stats_total['benchmark_return'].var()
            beta_total = cov_total / var_total
            alpha_total = (valid_stats_total['portfolio_return'].mean() - self.risk_free_rate/252) - beta_total * (valid_stats_total['benchmark_return'].mean() - self.risk_free_rate/252)
            alpha_total = alpha_total * 252
        else:
            beta_total = 0.0
            alpha_total = 0.0
        
        # Cumulative returns
        cum_returns = (1 + df['portfolio_return']).cumprod()
        current_dd = self._calculate_current_drawdown(cum_returns)
        max_dd = self._calculate_max_drawdown(cum_returns)
        
        # Determine status
        if ic_recent < self.ic_critical or current_dd > self.dd_critical:
            status = 'CRITICAL'
        elif ic_recent < self.ic_warning or current_dd > self.dd_warning:
            status = 'WARNING'
        else:
            status = 'HEALTHY'
        
        return {
            'status': status,
            'date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'ic_rolling': ic_recent,
            'ic_std': ic_std,
            'sharpe_rolling': sharpe,
            'beta': beta,
            'alpha_annual': alpha,
            'alpha_total': alpha_total,
            'beta_total': beta_total,
            'active_return_rolling_annual': active_ret_rolling_annual if not np.isnan(active_ret_rolling_annual) else 0.0,
            'active_return_total_annual': active_ret_total_annual if not np.isnan(active_ret_total_annual) else 0.0,
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'avg_turnover': recent['turnover'].mean() if 'turnover' in recent else 0.0,
            'total_costs': recent['transaction_costs'].sum() if 'transaction_costs' in recent else 0.0,
            'num_days': len(recent),
            'avg_positions': recent['num_positions'].mean()
        }
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio with risk-free rate"""
        if len(returns) < 2:
            return 0.0
        
        # Daily risk free rate
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        mean_excess = excess_returns.mean() * 252
        std_excess = excess_returns.std() * np.sqrt(252)
        
        if std_excess > 0:
            return mean_excess / std_excess
        return 0.0
    
    def _calculate_current_drawdown(self, cum_returns: pd.Series) -> float:
        """Calculate current drawdown"""
        cummax = cum_returns.cummax()
        if cummax.iloc[-1] == 0:
            return 0.0
        current_dd = (cum_returns.iloc[-1] - cummax.iloc[-1]) / cummax.iloc[-1]
        return abs(current_dd)
        
    def _calculate_max_drawdown(self, cum_returns: pd.Series) -> float:
        """Calculate maximum drawdown over history"""
        cummax = cum_returns.cummax()
        drawdown = (cum_returns - cummax) / cummax
        return abs(drawdown.min())
    
    def get_history_df(self) -> pd.DataFrame:
        """Get full history as DataFrame"""
        return pd.DataFrame(list(self.history))
    
    def plot_performance(self, save_path: str = None):
        """Plot performance metrics"""
        import matplotlib.pyplot as plt
        
        if not save_path:
            logger.warning("No save_path provided for plot_performance. Skipping plot.")
            return

        df = pd.DataFrame(list(self.history))
        if df.empty:
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(14, 15))
        
        # Rolling IC
        rolling_ic = df['ic'].rolling(window=20).mean()
        axes[0].plot(df['date'], rolling_ic, label='20-day MA')
        axes[0].axhline(y=self.ic_warning, color='orange', linestyle='--', label='Warning')
        axes[0].axhline(y=self.ic_critical, color='red', linestyle='--', label='Critical')
        axes[0].set_title('Rolling IC (Spearman)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative returns vs Benchmark
        cum_returns = (1 + df['portfolio_return']).cumprod() - 1
        cum_bench = (1 + df['benchmark_return']).cumprod() - 1
        
        axes[1].plot(df['date'], cum_returns * 100, label='Portfolio')
        axes[1].plot(df['date'], cum_bench * 100, label='Benchmark', alpha=0.7)
        axes[1].set_title('Cumulative Returns (%)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Turnover & Costs
        if 'turnover' in df.columns:
            ax2 = axes[2].twinx()
            axes[2].bar(df['date'], df['turnover'] * 100, alpha=0.3, color='gray', label='Turnover %')
            ax2.plot(df['date'], df['transaction_costs'].cumsum(), color='red', label='Cum Costs (Impact)')
            axes[2].set_title('Turnover & Transaction Costs', fontweight='bold')
            axes[2].set_ylabel('Turnover (%)')
            ax2.set_ylabel('Cumulative Costs (Impact)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
