"""
Backtesting Engine
==================
Simulates trading strategy with realistic costs.
Tests how the model would have performed in real trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: pd.Series
    cumulative: pd.Series
    metrics: Dict[str, float]
    positions: pd.DataFrame = None


class Backtester:
    """
    Portfolio backtester.
    
    Strategy: Long top N stocks based on predicted returns.
    Includes realistic transaction costs.
    """
    
    def __init__(self):
        """Initialize backtester with settings."""
        self.config = settings.backtest
        self.top_n = self.config.top_n_long
        self.cost = self.config.total_cost_pct
    
    def run(self, predictions: pd.DataFrame) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            predictions: DataFrame with date, ticker, forward_return, prediction
            
        Returns:
            BacktestResult object
        """
        print("\n" + "="*60)
        print("💼 BACKTESTING")
        print("="*60)
        print(f"   Strategy: Long Top {self.top_n} stocks")
        print(f"   Transaction Cost: {self.config.total_cost_bps} bps")
        print(f"   Rebalance: {self.config.rebalance_frequency}")
        
        # Get rebalance dates (monthly)
        dates = sorted(predictions['date'].unique())
        monthly_dates = pd.Series(dates).groupby(
            pd.Series(dates).dt.to_period('M')
        ).first().values
        
        returns_list = []
        positions_list = []
        
        # Loop through each rebalance period
        for i in range(len(monthly_dates) - 1):
            rebal_date = monthly_dates[i]
            next_date = monthly_dates[i + 1]
            
            # Get predictions for rebalance date
            day_preds = predictions[predictions['date'] == rebal_date]
            
            if len(day_preds) < self.top_n:
                continue
            
            # Select top N stocks
            top_stocks = day_preds.nlargest(self.top_n, 'prediction')
            
            # Calculate return (equal weight)
            period_return = top_stocks['forward_return'].mean()
            
            # Apply transaction cost
            period_return -= self.cost
            
            returns_list.append({
                'date': next_date,
                'return': period_return
            })
            
            positions_list.append({
                'date': rebal_date,
                'stocks': top_stocks['ticker'].tolist()
            })
        
        if not returns_list:
            print("\n   ⚠️ No trades executed!")
            return None
        
        # Create return series
        returns_df = pd.DataFrame(returns_list)
        returns = returns_df.set_index('date')['return']
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns)
        
        # Positions
        positions = pd.DataFrame(positions_list)
        
        # Print results
        self._print_results(metrics)
        
        return BacktestResult(
            returns=returns,
            cumulative=cumulative,
            metrics=metrics,
            positions=positions
        )
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = returns.dropna()
        n = len(returns)
        
        if n < 2:
            return {}
        
        # Returns
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (12 / n) - 1  # Assuming monthly
        
        # Risk
        volatility = returns.std() * np.sqrt(12)
        
        # Ratios
        sharpe = annual_return / (volatility + 1e-10)
        
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(12) if len(downside) > 0 else 1e-10
        sortino = annual_return / downside_vol
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar = annual_return / (abs(max_drawdown) + 1e-10)
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'n_periods': n
        }
    
    def _print_results(self, metrics: Dict):
        """Print backtest results."""
        print(f"\n   {'─'*50}")
        print(f"   📈 BACKTEST RESULTS")
        print(f"   {'─'*50}")
        print(f"   Total Return:      {metrics['total_return']:>10.1%}")
        print(f"   Annual Return:     {metrics['annual_return']:>10.1%}")
        print(f"   Volatility:        {metrics['volatility']:>10.1%}")
        print(f"   Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
        print(f"   Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
        print(f"   Max Drawdown:      {metrics['max_drawdown']:>10.1%}")
        print(f"   Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}")
        print(f"   Win Rate:          {metrics['win_rate']:>10.1%}")
        print(f"   Periods:           {metrics['n_periods']:>10}")