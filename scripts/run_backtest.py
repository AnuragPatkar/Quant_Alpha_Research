"""
Run Backtest
============
Portfolio backtesting simulation using ML predictions.

Author: Anurag Patkar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
import json
import argparse

warnings.filterwarnings('ignore')

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.utils import Timer, print_header, print_section, save_results, ensure_dir

try:
    from config import settings, print_welcome
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    from quant_alpha.backtest.engine import Backtester, BacktestConfig
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False

try:
    from quant_alpha.visualization.plots import PerformancePlotter
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False


class SimpleBacktester:
    """Simple portfolio backtesting engine."""
    
    def __init__(self, config=None):
        if config:
            self.config = config
        elif SETTINGS_AVAILABLE:
            self.config = settings.backtest
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(
                initial_capital=1_000_000,
                top_n_long=10,
                total_cost_pct=0.003,
                risk_free_rate=0.05
            )
    
    def run_backtest(self, predictions_df: pd.DataFrame) -> tuple:
        """Run portfolio backtest."""
        print_header("PORTFOLIO BACKTESTING")
        
        # Handle column names
        if 'prediction' in predictions_df.columns and 'predictions' not in predictions_df.columns:
            predictions_df = predictions_df.rename(columns={'prediction': 'predictions'})
        
        required_cols = ['date', 'ticker', 'predictions', 'forward_return']
        missing = set(required_cols) - set(predictions_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        portfolio_value = self.config.initial_capital
        positions = {}
        
        rebalance_dates = self._get_rebalance_dates(predictions_df)
        print(f"📅 Rebalancing dates: {len(rebalance_dates)}")
        print(f"💰 Initial capital: ${self.config.initial_capital:,.0f}")
        
        portfolio_history = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            if i % 10 == 0:
                print(f"   📊 Rebalancing {i+1}/{len(rebalance_dates)}")
            
            date_data = predictions_df[predictions_df['date'] == rebal_date].copy()
            if len(date_data) == 0:
                continue
            
            # Select top stocks
            clean_data = date_data.dropna(subset=['predictions'])
            if len(clean_data) == 0:
                continue
            
            sorted_data = clean_data.sort_values('predictions', ascending=False)
            top_n = min(self.config.top_n_long, len(sorted_data))
            top_stocks = sorted_data.head(top_n)
            
            # Calculate positions
            weight = 1.0 / len(top_stocks)
            new_positions = {row['ticker']: portfolio_value * weight for _, row in top_stocks.iterrows()}
            
            # Transaction costs
            all_tickers = set(positions.keys()) | set(new_positions.keys())
            turnover = sum(
                abs(positions.get(t, 0) - new_positions.get(t, 0))
                for t in all_tickers
            ) / (portfolio_value + 1e-10)
            transaction_cost = turnover * portfolio_value * self.config.total_cost_pct
            portfolio_value -= transaction_cost
            positions = new_positions
            
            # Calculate returns
            if i < len(rebalance_dates) - 1:
                period_return = self._calculate_period_return(positions, predictions_df, rebal_date, rebalance_dates[i + 1])
                portfolio_value *= (1 + period_return)
            
            portfolio_history.append({
                'date': rebal_date,
                'portfolio_value': portfolio_value,
                'n_positions': len(positions),
                'transaction_cost': transaction_cost
            })
        
        if not portfolio_history:
            print("❌ No trades executed!")
            return pd.DataFrame(), {}
        
        results_df = pd.DataFrame(portfolio_history)
        metrics = self._calculate_metrics(results_df)
        self._print_summary(results_df, metrics)
        
        return results_df, metrics
    
    def _get_rebalance_dates(self, df: pd.DataFrame) -> list:
        df['date'] = pd.to_datetime(df['date'])
        monthly = df.groupby([df['date'].dt.year, df['date'].dt.month])['date'].max()
        return sorted(monthly.values)
    
    def _calculate_period_return(self, positions, df, start, end):
        if not positions:
            return 0.0
        
        total_ret = 0.0
        total_wt = 0.0
        
        for ticker, val in positions.items():
            stock = df[(df['ticker'] == ticker) & (df['date'] >= start) & (df['date'] < end)]
            if len(stock) > 0 and 'forward_return' in stock.columns:
                ret = stock['forward_return'].iloc[0]
                if not pd.isna(ret):
                    wt = val / sum(positions.values())
                    total_ret += wt * ret
                    total_wt += wt
        
        return total_ret / total_wt if total_wt > 0 else 0.0
    
    def _calculate_metrics(self, results_df):
        if len(results_df) < 2:
            return {}
        
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        returns = results_df['returns'].dropna()
        
        if len(returns) == 0:
            return {}
        
        total_ret = results_df['portfolio_value'].iloc[-1] / results_df['portfolio_value'].iloc[0] - 1
        n = len(returns)
        ann_ret = (1 + total_ret) ** (12 / n) - 1 if n > 0 else 0
        vol = returns.std() * np.sqrt(12)
        rf = getattr(self.config, 'risk_free_rate', 0.05)
        sharpe = (ann_ret - rf) / vol if vol > 0 else 0
        
        cum = (1 + returns).cumprod()
        max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        
        return {
            'total_return': total_ret,
            'annualized_return': ann_ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': (returns > 0).mean(),
            'n_periods': n,
            'final_value': results_df['portfolio_value'].iloc[-1]
        }
    
    def _print_summary(self, results_df, metrics):
        print_section("BACKTEST RESULTS")
        if not metrics:
            print("❌ No metrics!")
            return
        
        print(f"📅 Period: {results_df['date'].min().date()} to {results_df['date'].max().date()}")
        print(f"💰 Initial: ${self.config.initial_capital:,.0f}")
        print(f"💰 Final: ${metrics['final_value']:,.0f}")
        print(f"📈 Total Return: {metrics['total_return']:.2%}")
        print(f"📈 Annual Return: {metrics['annualized_return']:.2%}")
        print(f"⚡ Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"📉 Max DD: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Portfolio Backtesting')
    parser.add_argument('--predictions', type=str, help='Path to predictions file')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--top-n', type=int, default=10, help='Number of stocks')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if SETTINGS_AVAILABLE:
        print_welcome()
    
    try:
        # Load data
        if args.predictions:
            path = Path(args.predictions)
        elif SETTINGS_AVAILABLE:
            path = settings.data.processed_dir / "features_dataset.pkl"
        else:
            print("❌ No predictions file!")
            return 1
        
        if not path.exists():
            print(f"❌ File not found: {path}")
            return 1
        
        print(f"📂 Loading: {path}")
        
        if path.suffix == '.pkl':
            predictions_df = pd.read_pickle(path)
        elif path.suffix == '.parquet':
            predictions_df = pd.read_parquet(path)
        else:
            predictions_df = pd.read_csv(path, parse_dates=['date'])
        
        print(f"✅ Loaded {len(predictions_df):,} records")
        
        # Create predictions if needed
        if 'predictions' not in predictions_df.columns and 'prediction' not in predictions_df.columns:
            if 'mom_21d' in predictions_df.columns:
                predictions_df['predictions'] = predictions_df['mom_21d']
                print("   Using mom_21d as prediction")
            else:
                print("❌ No prediction column!")
                return 1
        
        # Output dir
        output_dir = Path(args.output) if args.output else (settings.results_dir if SETTINGS_AVAILABLE else ROOT / "output")
        ensure_dir(output_dir)
        
        # Run backtest
        if SETTINGS_AVAILABLE:
            settings.backtest.initial_capital = args.capital
            settings.backtest.top_n_long = args.top_n
        
        backtester = SimpleBacktester()
        results_df, metrics = backtester.run_backtest(predictions_df)
        
        if len(results_df) > 0:
            save_results(results_df, output_dir / "backtest_results.csv")
            save_results(metrics, output_dir / "backtest_metrics.json")
            
            if args.plots and VIZ_AVAILABLE and 'portfolio_value' in results_df.columns:
                plots_dir = output_dir / "plots"
                ensure_dir(plots_dir)
                plotter = PerformancePlotter()
                equity = pd.Series(results_df['portfolio_value'].values, index=pd.to_datetime(results_df['date']))
                plotter.plot_equity_curve(equity, save_path=str(plots_dir / "equity.png"), show=False)
        
        print(f"\n⏱️  Time: {(time.time() - start_time)/60:.1f} min")
        print(f"📁 Output: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())